"""Development-only EEG/cache views bound to one verified protocol role."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType

import numpy as np
import torch
from torch.utils.data import Dataset

from .access import TypedArtifact, verify_typed_artifacts
from .hashing import canonical_json_bytes, ordered_ids_sha256, sha256_json


POSTERIOR_CHANNELS = (
    "P7",
    "P5",
    "P3",
    "P1",
    "Pz",
    "P2",
    "P4",
    "P6",
    "P8",
    "PO7",
    "PO3",
    "POz",
    "PO4",
    "PO8",
    "O1",
    "Oz",
    "O2",
)

_ROLE_PAYLOAD_TYPE = "samga_brain_rw.role_payload"
_ALLOWED_SCOPES = frozenset({"train", "val-dev"})
_ROLE_COUNTS = {"train": (12_540, 1_254), "val-dev": (200, 200)}
_LAYER_IDS = (20, 24, 28, 32, 36)
_PROTOCOL_RE = re.compile(r"^sub-(\d{2})_protocol\.json$")
_SOURCE_RE = re.compile(r"^sub-(\d{2})_train\.json$")
_SUBJECT_RE = re.compile(r"^sub-(\d{2})$")
_SUBJECT_TEST_RE = re.compile(r"^sub-\d{2}_test\.json$", re.IGNORECASE)
_SOURCE_RECORD_KEYS = frozenset(
    {"concept_id", "image_id", "image_path", "row_index", "validation_query"}
)
_SOURCE_KEYS = frozenset(
    {
        "ch_names",
        "eeg_dtype",
        "eeg_shape",
        "records",
        "records_sha256",
        "schema_version",
        "source_pt",
        "split",
        "subject_id",
        "validation_concepts",
        "validation_salt",
    }
)
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_MAX_JSON_BYTES = 64 * 1024 * 1024
_VERIFIED_CACHE_DIGESTS: dict[tuple[object, ...], str] = {}


class ProtocolSubjectDataset(Dataset[dict[str, object]]):
    """Expose exactly one train or val-dev role from a subject protocol."""

    def __init__(
        self,
        manifest_path: Path,
        scope: str,
        seed: int,
        selected_channels: Sequence[str],
        feature_cache: Path | None,
        smooth_probability: float,
    ) -> None:
        if scope not in _ALLOWED_SCOPES:
            raise PermissionError(
                "ProtocolSubjectDataset scope must be train or val-dev; "
                "val-confirm, formal, and test scopes are sealed"
            )
        if type(seed) is not int:
            raise TypeError("seed must be an integer")
        probability = _probability(smooth_probability)
        channels = tuple(selected_channels)
        if channels != POSTERIOR_CHANNELS:
            raise ValueError(
                "selected_channels must be the exact ordered 17 posterior channels"
            )

        protocol_path = _absolute_path(Path(manifest_path))
        _preflight_development_path(protocol_path, "protocol manifest")
        match = _PROTOCOL_RE.fullmatch(protocol_path.name)
        if match is None:
            raise ValueError("protocol filename must be sub-XX_protocol.json")
        filename_subject = int(match.group(1))
        if not 1 <= filename_subject <= 10:
            raise ValueError("protocol subject must be between 1 and 10")

        descriptor = TypedArtifact(
            payload_type=_ROLE_PAYLOAD_TYPE,
            payload_path=protocol_path,
            envelope_path=protocol_path,
            role=scope,
        )
        capability = verify_typed_artifacts(scope, [descriptor])[0]
        with capability.open_verified() as handle:
            protocol = _parse_json_object(handle.read(), "protocol manifest")
        subject = _protocol_subject(protocol, filename_subject)
        _validate_development_disjointness(protocol)
        role = _selected_role(protocol, scope)
        self._bind_role(role, scope)

        declared_source = _string(
            protocol.get("source_manifest_path"),
            "protocol source_manifest_path",
        )
        source_path = _resolve_declared_path(declared_source, protocol_path)
        _preflight_development_path(source_path, "source train manifest")
        source_match = _SOURCE_RE.fullmatch(source_path.name)
        if source_match is None:
            raise ValueError("source manifest must be named sub-XX_train.json")
        if int(source_match.group(1)) != subject:
            raise ValueError("source manifest subject differs from protocol subject")
        source_raw = _read_regular_bytes(source_path, "source train manifest")
        expected_source_sha256 = _sha256(
            protocol.get("source_manifest_sha256"),
            "protocol source_manifest_sha256",
        )
        if hashlib.sha256(source_raw).hexdigest() != expected_source_sha256:
            raise ValueError("source manifest SHA-256 differs from protocol")
        source = _parse_json_object(source_raw, "source train manifest")
        source_records, all_channels, source_pt = _validate_source_manifest(
            source,
            subject=subject,
            protocol_records_sha256=_sha256(
                protocol.get("records_sha256"),
                "protocol records_sha256",
            ),
        )

        selected_records = _bind_source_rows(
            source_records,
            self.row_indices,
            self.concept_ids,
            self.ordered_ids,
            scope=scope,
        )
        self._channel_indices = np.asarray(
            [all_channels.index(channel) for channel in channels],
            dtype=np.int64,
        )
        self.feature_cache_metadata: Mapping[str, object] | None = None
        self._feature_cache: np.ndarray | None = None
        if feature_cache is not None:
            self._feature_cache, metadata = _load_feature_cache(
                Path(feature_cache),
                records_sha256=_sha256(
                    source.get("records_sha256"),
                    "source records_sha256",
                ),
                record_count=len(source_records),
            )
            self.feature_cache_metadata = _deep_freeze(metadata)

        loaded = _load_torch_payload(source_pt)
        eeg = _validate_eeg_payload(
            loaded,
            declared_channels=all_channels,
            source=source,
            record_count=len(source_records),
        )
        self.manifest_path = protocol_path
        self.source_manifest_path = source_path
        self.scope = scope
        self.seed = seed
        self.smooth_probability = probability
        self.subject_id = subject
        self.selected_channels = channels
        self.channels_num = len(channels)
        self.num_sample_points = eeg.shape[-1]
        self._eeg = eeg
        self.records = tuple(
            MappingProxyType(dict(record)) for record in selected_records
        )

    def _bind_role(self, role: Mapping[str, object], scope: str) -> None:
        expected_rows, expected_concepts = _ROLE_COUNTS[scope]
        row_indices = _integer_tuple(role.get("row_indices"), "role row_indices")
        concept_ids = _string_tuple(role.get("concept_ids"), "role concept_ids")
        ordered_ids = _string_tuple(role.get("ordered_ids"), "role ordered_ids")
        query_ids = _string_tuple(role.get("query_ids"), "role query_ids")
        gallery_ids = _string_tuple(role.get("gallery_ids"), "role gallery_ids")
        if len(row_indices) != expected_rows or len(concept_ids) != expected_concepts:
            raise ValueError(
                f"{scope} must contain exactly {expected_rows} rows and "
                f"{expected_concepts} concepts"
            )
        if len(set(row_indices)) != len(row_indices):
            raise ValueError(f"{scope} contains duplicate source rows")
        if len(set(concept_ids)) != len(concept_ids):
            raise ValueError(f"{scope} contains duplicate concept IDs")
        if role.get("row_count") != expected_rows:
            raise ValueError(f"{scope} row_count mismatch")
        if role.get("concept_count") != expected_concepts:
            raise ValueError(f"{scope} concept_count mismatch")
        if role.get("scope") != scope or role.get("payload_type") != _ROLE_PAYLOAD_TYPE:
            raise ValueError(f"selected {scope} role identity mismatch")
        if scope == "train":
            if ordered_ids != concept_ids or query_ids or gallery_ids:
                raise ValueError("train IDs must be concepts with no query/gallery IDs")
            stimuli_per_concept = 10
        else:
            if not (
                len(row_indices)
                == len(concept_ids)
                == len(ordered_ids)
                == len(query_ids)
                == len(gallery_ids)
            ):
                raise ValueError("val-dev role ID and row lengths differ")
            if ordered_ids != query_ids or ordered_ids != gallery_ids:
                raise ValueError("val-dev ordered/query/gallery IDs differ")
            stimuli_per_concept = 1
        self.row_indices = row_indices
        self.concept_ids = concept_ids
        self.ordered_ids = ordered_ids
        self.query_ids = query_ids
        self.gallery_ids = gallery_ids
        self.stimuli_per_concept = stimuli_per_concept

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(self, item: int) -> dict[str, object]:
        record = self.records[item]
        row_index = self.row_indices[item]
        trials = np.asarray(self._eeg[row_index])
        eeg = trials.mean(axis=0, dtype=np.float32)[self._channel_indices]
        if self.scope == "train" and self.smooth_probability > 0.0:
            eeg = _deterministic_smoothing(
                eeg,
                seed=self.seed,
                row_index=row_index,
                probability=self.smooth_probability,
            )
        result: dict[str, object] = {
            "concept_id": record["concept_id"],
            "eeg": torch.from_numpy(np.ascontiguousarray(eeg)).float(),
            "image_id": record["image_id"],
            "image_path": record["image_path"],
            "row_index": row_index,
            "scope": self.scope,
            "subject_id": self.subject_id,
        }
        if self._feature_cache is not None:
            result["layer_features"] = torch.from_numpy(
                np.asarray(self._feature_cache[row_index], dtype=np.float32).copy()
            )
        return result


def _selected_role(
    protocol: Mapping[str, object],
    scope: str,
) -> Mapping[str, object]:
    roles = protocol.get("role_payloads")
    descriptors = protocol.get("role_artifacts")
    if not isinstance(roles, Mapping) or not isinstance(descriptors, Mapping):
        raise ValueError("protocol role containers must be objects")
    role = roles.get(scope)
    descriptor = descriptors.get(scope)
    if not isinstance(role, Mapping) or not isinstance(descriptor, Mapping):
        raise ValueError(f"protocol is missing selected {scope} role")
    if descriptor.get("role_payload_sha256") != sha256_json(role):
        raise ValueError(f"{scope} role payload SHA-256 mismatch")
    if descriptor.get("ordered_ids_sha256") != ordered_ids_sha256(
        _string_tuple(role.get("ordered_ids"), "role ordered_ids")
    ):
        raise ValueError(f"{scope} ordered-ID SHA-256 mismatch")
    return role


def _validate_development_disjointness(protocol: Mapping[str, object]) -> None:
    train = _selected_role(protocol, "train")
    val_dev = _selected_role(protocol, "val-dev")
    train_rows = _integer_tuple(train.get("row_indices"), "train row_indices")
    val_rows = _integer_tuple(val_dev.get("row_indices"), "val-dev row_indices")
    train_concepts = _string_tuple(train.get("concept_ids"), "train concept_ids")
    val_concepts = _string_tuple(
        val_dev.get("concept_ids"),
        "val-dev concept_ids",
    )
    train_count = _ROLE_COUNTS["train"]
    val_count = _ROLE_COUNTS["val-dev"]
    if (len(train_rows), len(train_concepts)) != train_count:
        raise ValueError("train role has incorrect development counts")
    if (len(val_rows), len(val_concepts)) != val_count:
        raise ValueError("val-dev role has incorrect development counts")
    if len(set(train_rows)) != len(train_rows):
        raise ValueError("train role contains duplicate source rows")
    if len(set(val_rows)) != len(val_rows):
        raise ValueError("val-dev role contains duplicate source rows")
    if set(train_rows).intersection(val_rows):
        raise ValueError("train and val-dev source rows overlap")
    if len(set(train_concepts)) != len(train_concepts):
        raise ValueError("train role contains duplicate concepts")
    if len(set(val_concepts)) != len(val_concepts):
        raise ValueError("val-dev role contains duplicate concepts")
    if set(train_concepts).intersection(val_concepts):
        raise ValueError("train and val-dev concepts overlap")


def _protocol_subject(protocol: Mapping[str, object], filename_subject: int) -> int:
    value = protocol.get("subject_id")
    if type(value) is int:
        subject = value
    elif isinstance(value, str) and _SUBJECT_RE.fullmatch(value):
        subject = int(value[-2:])
    else:
        raise ValueError("protocol subject_id is invalid")
    if subject != filename_subject:
        raise ValueError("protocol subject_id differs from protocol filename")
    if not 1 <= subject <= 10:
        raise ValueError("protocol subject must be between 1 and 10")
    return subject


def _validate_source_manifest(
    source: Mapping[str, object],
    *,
    subject: int,
    protocol_records_sha256: str,
) -> tuple[tuple[dict[str, object], ...], tuple[str, ...], Path]:
    if frozenset(source) != _SOURCE_KEYS:
        raise ValueError("source train manifest has an unexpected schema")
    if source.get("schema_version") != 1 or source.get("split") != "train":
        raise ValueError("source manifest must be schema 1 train data")
    source_subject = source.get("subject_id")
    accepted_subjects = {subject, f"sub-{subject:02d}"}
    if source_subject not in accepted_subjects:
        raise ValueError("source manifest subject differs from protocol subject")
    records_value = source.get("records")
    if not isinstance(records_value, list) or len(records_value) != 16_540:
        raise ValueError("source manifest must contain exactly 16540 records")
    records: list[dict[str, object]] = []
    for index, value in enumerate(records_value):
        if not isinstance(value, Mapping) or frozenset(value) != _SOURCE_RECORD_KEYS:
            raise ValueError("source record has an unexpected schema")
        record = dict(value)
        if record.get("row_index") != index:
            raise ValueError("source record rows must be contiguous and ordered")
        for key in ("concept_id", "image_id", "image_path"):
            _string(record.get(key), f"source record {key}")
        if type(record.get("validation_query")) is not bool:
            raise ValueError("source validation_query must be boolean")
        _reject_record_path(str(record["image_path"]))
        records.append(record)
    records_sha256 = hashlib.sha256(canonical_json_bytes(records)).hexdigest()
    if _sha256(source.get("records_sha256"), "source records_sha256") != records_sha256:
        raise ValueError("source record SHA-256 mismatch")
    if records_sha256 != protocol_records_sha256:
        raise ValueError("source records differ from protocol records")
    channels = _string_tuple(source.get("ch_names"), "source ch_names")
    if len(set(channels)) != len(channels):
        raise ValueError("source channels contain duplicates")
    missing = [channel for channel in POSTERIOR_CHANNELS if channel not in channels]
    if missing:
        raise ValueError(f"source is missing posterior channels: {missing}")
    shape = _integer_tuple(source.get("eeg_shape"), "source eeg_shape")
    if len(shape) != 4 or shape[0] != 16_540 or shape[1] != 4 or shape[-1] != 250:
        raise ValueError("source EEG shape must be [16540,4,channels,250]")
    if shape[-2] != len(channels):
        raise ValueError("source EEG channel count differs from ch_names")
    validation_concepts = _string_tuple(
        source.get("validation_concepts"),
        "source validation_concepts",
    )
    if len(set(validation_concepts)) != len(validation_concepts):
        raise ValueError("source validation_concepts contains duplicates")
    source_pt = _absolute_path(Path(_string(source.get("source_pt"), "source_pt")))
    _preflight_development_path(source_pt, "source train EEG")
    if source_pt.name != "train.pt" or source_pt.parent.name != f"sub-{subject:02d}":
        raise ValueError("source EEG must be the selected subject train.pt")
    return tuple(records), channels, source_pt


def _bind_source_rows(
    records: tuple[dict[str, object], ...],
    rows: tuple[int, ...],
    concept_ids: tuple[str, ...],
    ordered_ids: tuple[str, ...],
    *,
    scope: str,
) -> tuple[dict[str, object], ...]:
    if any(row < 0 or row >= len(records) for row in rows):
        raise ValueError(f"{scope} contains an out-of-range source row")
    selected = tuple(records[row] for row in rows)
    if scope == "train":
        for index, concept_id in enumerate(concept_ids):
            group = selected[index * 10 : (index + 1) * 10]
            if len(group) != 10 or any(
                record["concept_id"] != concept_id for record in group
            ):
                raise ValueError("train protocol rows do not match source concepts")
            if len({record["image_id"] for record in group}) != 10:
                raise ValueError("train concept does not expose ten unique stimuli")
    else:
        for record, concept_id, image_id in zip(
            selected,
            concept_ids,
            ordered_ids,
            strict=True,
        ):
            if record["concept_id"] != concept_id or record["image_id"] != image_id:
                raise ValueError("val-dev protocol IDs do not match source rows")
    return selected


def _load_torch_payload(path: Path) -> object:
    descriptor = _open_component_file(path, "source train EEG")
    try:
        before = os.fstat(descriptor)
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            try:
                loaded = torch.load(
                    handle,
                    map_location="cpu",
                    weights_only=False,
                )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                raise ValueError("source train.pt could not be loaded safely") from exc
            after = os.fstat(handle.fileno())
            if _identity(before) != _identity(after):
                raise ValueError("source train.pt changed while it was loaded")
            return loaded
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _validate_eeg_payload(
    loaded: object,
    *,
    declared_channels: tuple[str, ...],
    source: Mapping[str, object],
    record_count: int,
) -> np.ndarray:
    if not isinstance(loaded, Mapping) or "eeg" not in loaded:
        raise ValueError("source train.pt must contain EEG data")
    eeg = np.asarray(loaded["eeg"])
    if eeg.ndim == 5:
        eeg = eeg.reshape(eeg.shape[0] * eeg.shape[1], *eeg.shape[2:])
    if eeg.ndim != 4 or eeg.shape[0] != record_count:
        raise ValueError("source EEG rows do not match source records")
    if eeg.shape[1] != 4:
        raise ValueError("source EEG must contain exactly four trials per row")
    if eeg.shape[2] != len(declared_channels) or eeg.shape[3] != 250:
        raise ValueError("source EEG channel/time layout differs from manifest")
    declared_dtype = _string(source.get("eeg_dtype"), "source eeg_dtype")
    if str(eeg.dtype) != declared_dtype:
        raise ValueError("source EEG dtype differs from manifest")
    payload_channels = loaded.get("ch_names")
    if payload_channels is not None and tuple(payload_channels) != declared_channels:
        raise ValueError("source train.pt channels differ from source manifest")
    if not np.issubdtype(eeg.dtype, np.floating):
        raise ValueError("source EEG must use a floating dtype")
    return eeg


def _load_feature_cache(
    path: Path,
    *,
    records_sha256: str,
    record_count: int,
) -> tuple[np.ndarray, dict[str, object]]:
    cache_path = _absolute_path(path)
    _preflight_development_path(cache_path, "feature cache")
    candidates = (
        cache_path.with_suffix(cache_path.suffix + ".meta.json"),
        cache_path.parent / "metadata.json",
    )
    metadata_paths = [candidate for candidate in candidates if candidate.is_file()]
    if len(metadata_paths) != 1:
        raise ValueError("feature cache must have exactly one metadata sidecar")
    metadata_path = metadata_paths[0]
    _preflight_development_path(metadata_path, "feature cache metadata")
    metadata = _parse_json_object(
        _read_regular_bytes(metadata_path, "feature cache metadata"),
        "feature cache metadata",
    )
    if metadata.get("schema_version") not in (1, 2):
        raise ValueError("feature cache metadata schema must be 1 or 2")
    if metadata.get("complete") is not True or metadata.get("partial_rows") is True:
        raise ValueError("feature cache must be complete and non-partial")
    if metadata.get("records_sha256") != records_sha256:
        raise ValueError("feature cache record binding differs from source records")
    expected_shape = _integer_tuple(metadata.get("shape"), "cache shape")
    if expected_shape != (record_count, len(_LAYER_IDS), 3_200):
        raise ValueError("feature cache shape must be [16540,5,3200]")
    layer_ids = metadata.get("logical_layer_ids", metadata.get("layer_ids"))
    if _integer_tuple(layer_ids, "cache layer IDs") != _LAYER_IDS:
        raise ValueError("feature cache must contain locked InternViT layers")
    if metadata.get("split") != "train":
        raise ValueError("feature cache must be the shared train cache")
    if (
        "feature_filename" in metadata
        and metadata["feature_filename"] != cache_path.name
    ):
        raise ValueError("feature cache filename differs from metadata")
    expected_digest = _sha256(
        metadata.get("feature_sha256", metadata.get("cache_sha256")),
        "feature cache SHA-256",
    )
    descriptor = _open_component_file(cache_path, "feature cache")
    try:
        before = os.fstat(descriptor)
        key = (*_identity(before), expected_digest)
        digest = _VERIFIED_CACHE_DIGESTS.get(key)
        if digest is None:
            digest = _sha256_descriptor(descriptor)
            _VERIFIED_CACHE_DIGESTS[key] = digest
        if digest != expected_digest:
            raise ValueError("feature cache SHA-256 mismatch")
        os.lseek(descriptor, 0, os.SEEK_SET)
        try:
            cache = np.load(
                f"/proc/self/fd/{descriptor}",
                mmap_mode="r",
                allow_pickle=False,
            )
        except (OSError, TypeError, ValueError) as exc:
            raise ValueError("feature cache is not a safe NumPy array") from exc
        after = os.fstat(descriptor)
        if _identity(before) != _identity(after):
            raise ValueError("feature cache changed while it was loaded")
    finally:
        os.close(descriptor)
    if not isinstance(cache, np.ndarray) or tuple(cache.shape) != expected_shape:
        raise ValueError("feature cache array shape differs from metadata")
    try:
        expected_dtype = np.dtype(_string(metadata.get("dtype"), "cache dtype"))
    except TypeError as exc:
        raise ValueError("feature cache dtype is invalid") from exc
    if cache.dtype != expected_dtype or cache.dtype != np.dtype(np.float16):
        raise ValueError("feature cache must use the declared float16 dtype")
    if not cache.flags.c_contiguous:
        raise ValueError("feature cache must be C contiguous")
    return cache, metadata


def _deterministic_smoothing(
    eeg: np.ndarray,
    *,
    seed: int,
    row_index: int,
    probability: float,
) -> np.ndarray:
    material = (
        b"SAMGA-PROTOCOL-SMOOTH-v1\0"
        + str(seed).encode("ascii")
        + b"\0"
        + str(row_index).encode("ascii")
    )
    rng_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    mask = np.random.default_rng(rng_seed).random(eeg.shape[0]) < probability
    if not bool(mask.any()):
        return eeg
    result = eeg.copy()
    result[mask] = _moving_average(result[mask])
    return result


def _moving_average(signal: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    time_len = signal.shape[-1]
    left = np.maximum(0, np.arange(time_len) - kernel_size // 2)
    right = np.minimum(time_len, np.arange(time_len) + kernel_size // 2 + 1)
    cumulative = np.pad(
        np.cumsum(signal, axis=-1, dtype=np.float32),
        ((0, 0), (1, 0)),
    )
    return (cumulative[:, right] - cumulative[:, left]) / (
        right - left
    )[None, :]


def _parse_json_object(raw: bytes, context: str) -> dict[str, object]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def _read_regular_bytes(path: Path, context: str) -> bytes:
    descriptor = _open_component_file(path, context)
    try:
        before = os.fstat(descriptor)
        if before.st_size > _MAX_JSON_BYTES:
            raise ValueError(f"{context} exceeds the size limit")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, _MAX_JSON_BYTES + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > _MAX_JSON_BYTES:
                raise ValueError(f"{context} exceeds the size limit")
        after = os.fstat(descriptor)
        if _identity(before) != _identity(after):
            raise ValueError(f"{context} changed while read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _open_component_file(path: Path, context: str) -> int:
    absolute = _absolute_path(path)
    descriptor = os.open(
        absolute.anchor,
        os.O_RDONLY | _O_DIRECTORY | _O_CLOEXEC,
    )
    try:
        for component in absolute.parts[1:-1]:
            next_descriptor = os.open(
                component,
                os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        result = os.open(
            absolute.name,
            os.O_RDONLY | _O_NOFOLLOW | _O_CLOEXEC,
            dir_fd=descriptor,
        )
    except OSError as exc:
        raise ValueError(f"{context} could not be opened securely") from exc
    finally:
        os.close(descriptor)
    value = os.fstat(result)
    if not stat.S_ISREG(value.st_mode):
        os.close(result)
        raise ValueError(f"{context} must be a regular file")
    return result


def _sha256_descriptor(descriptor: int) -> str:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            return digest.hexdigest()
        digest.update(chunk)


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _preflight_development_path(path: Path, context: str) -> None:
    raw = os.fspath(path)
    if not isinstance(raw, str) or "\x00" in raw:
        raise ValueError(f"{context} must be a safe text path")
    for component in Path(raw).parts:
        normalized = re.sub(r"[^a-z0-9]+", "_", component.lower()).strip("_")
        if normalized in {
            "formal_input",
            "formal_refit",
            "formal_test",
            "test_images",
            "val_confirm",
        } or _SUBJECT_TEST_RE.fullmatch(component):
            raise PermissionError(f"{context} contains a sealed path component")


def _reject_record_path(value: str) -> None:
    for component in re.split(r"[\\/]", value):
        normalized = re.sub(r"[^a-z0-9]+", "_", component.lower()).strip("_")
        if normalized in {
            "formal_input",
            "formal_refit",
            "formal_test",
            "test_images",
            "val_confirm",
        } or _SUBJECT_TEST_RE.fullmatch(component):
            raise PermissionError("source records contain a sealed path")


def _resolve_declared_path(value: str, protocol_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return _absolute_path(path)
    candidates = (
        _absolute_path(path),
        _absolute_path(protocol_path.parent / path),
    )
    existing = tuple(candidate for candidate in candidates if candidate.exists())
    if len(set(existing)) == 1:
        return existing[0]
    if not existing:
        raise ValueError("declared source manifest path does not exist")
    raise ValueError("declared source manifest path is ambiguous")


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(os.path.normpath(os.fspath(path))))


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _sha256(value: object, context: str) -> str:
    text = _string(value, context)
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise ValueError(f"{context} must be lowercase SHA-256")
    return text


def _string_tuple(value: object, context: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{context} must be an array of non-empty strings")
    return tuple(value)


def _integer_tuple(value: object, context: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise ValueError(f"{context} must be an array of integers")
    return tuple(value)


def _probability(value: object) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError("smooth_probability must be finite")
    probability = float(value)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("smooth_probability must be between zero and one")
    return probability


def _deep_freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(child) for key, child in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze(child) for child in value)
    return value
