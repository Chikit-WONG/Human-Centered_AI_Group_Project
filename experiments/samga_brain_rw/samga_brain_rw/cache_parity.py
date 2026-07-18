"""CPU-only Stage 0 cache scoped-view and direct-index verification.

This module verifies exactly one canonical typed feature cache against the
train and val-dev row/ID mappings sealed in the ten Task 2 protocol manifests.
It does not claim equality with an independent cache: no second cache is part
of the public API.  Instead, every selected feature byte is read by direct
index from the verified canonical cache and hashed with a versioned domain.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

from .access import TypedArtifact, VerifiedArtifact, verify_typed_artifacts
from .hashing import canonical_json_bytes, ordered_ids_sha256, sha256_json


EXPECTED_SCOPES = ("train", "val-dev")
EXPECTED_SUBJECTS = 10
EXPECTED_CACHE_SHAPE = (16_540, 5, 3_200)
EXPECTED_CACHE_DTYPE = np.dtype(np.float16)
EXPECTED_ROLE_ROWS = {"train": 12_540, "val-dev": 200}

PINNED_CACHE_SHA256 = (
    "539c7b62ae41c8112e22b3ddc3a6566d997465a10c36d16c8f2378855ba94c71"
)
PINNED_ROW_SHA256 = {
    "train": (
        "a8b6e46076dba46e57658718a30455da1931533c3d274c492f44dcfb87e5c4de"
    ),
    "val-dev": (
        "291074d45ab56837d2c547105e0f96ba11b6d1d24eb95fc545e0bec0a7c7d95b"
    ),
}
PINNED_ORDERED_ID_SHA256 = {
    "train": (
        "ae5aeda4101f8740ebcb63464ca9cf5e126c81b2f124f5caa8f7b57b7a9fad24"
    ),
    "val-dev": (
        "512c222859a31b753ee31c5d6a1ddd1c81bb06e2dd5784d325f4480967162314"
    ),
}
PINNED_CONCEPT_ID_SHA256 = {
    "val-dev": (
        "c8c00ff2b15d98cdcb74d533037d52435bc12e09797151e46b52b86aedba1d15"
    )
}

_ROLE_PAYLOAD_TYPE = "samga_brain_rw.role_payload"
_CACHE_PAYLOAD_TYPES = frozenset(
    {
        "feature-cache",
        "train-cache",
        "samga_brain_rw.feature_cache",
        "samga_brain_rw.train_cache",
    }
)
_FEATURE_HASH_DOMAIN = b"SAMGA-STAGE0-DIRECT-INDEX-FEATURE-BYTES-v1\0"
_FORMAL_TEST_RECORD_SHA256 = (
    "02d7e33b3fe8e5a571f8db232ca5fa86abb0c16981876ec84feae7ba64636f1a"
)
_SUBJECT_TEST_RE = re.compile(r"(?i)^sub-\d{2}_test\.json$")


@dataclass(frozen=True)
class RoleMapping:
    """Selected Task 2 role mapping for one subject."""

    scope: str
    row_indices: tuple[int, ...]
    concept_ids: tuple[str, ...]
    ordered_ids: tuple[str, ...]
    query_ids: tuple[str, ...]
    gallery_ids: tuple[str, ...]
    records_sha256: str
    role_payload_sha256: str
    protocol_payload_sha256: str
    stimuli_per_concept: int

    @property
    def ordered_row_indices_sha256(self) -> str:
        return hash_ordered_rows(self.row_indices)

    @property
    def ordered_ids_sha256(self) -> str:
        return ordered_ids_sha256(self.ordered_ids)

    @property
    def concept_ids_sha256(self) -> str:
        return ordered_ids_sha256(self.concept_ids)

    def semantic_key(self) -> tuple[object, ...]:
        """Return exactly the mapping fields that must match across subjects."""

        return (
            self.scope,
            self.row_indices,
            self.concept_ids,
            self.ordered_ids,
            self.query_ids,
            self.gallery_ids,
            self.records_sha256,
            self.role_payload_sha256,
            self.stimuli_per_concept,
        )


def _validate_scopes(scopes: Sequence[str]) -> tuple[str, str]:
    if isinstance(scopes, (str, bytes, bytearray)):
        raise ValueError(
            "scopes must be exactly the ordered sequence ('train', 'val-dev')"
        )
    values = tuple(scopes)
    if values != EXPECTED_SCOPES:
        raise ValueError(
            "scopes must be exactly ('train', 'val-dev'); empty, duplicate, "
            "reordered, validation-confirmation, formal, and test scopes are "
            "forbidden"
        )
    return EXPECTED_SCOPES


def hash_ordered_rows(rows: Sequence[int]) -> str:
    """Hash ordered decimal row indices with no trailing newline."""

    values = tuple(rows)
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("ordered rows must be non-negative integers")
    return hashlib.sha256(
        "\n".join(str(value) for value in values).encode("utf-8")
    ).hexdigest()


def _feature_hash_prefix(shape: Sequence[int], dtype: np.dtype[object]) -> bytes:
    dimensions = ",".join(str(int(value)) for value in shape)
    return (
        _FEATURE_HASH_DOMAIN
        + np.dtype(dtype).str.encode("ascii")
        + b"\0"
        + dimensions.encode("ascii")
        + b"\0"
    )


def hash_feature_bytes(array: np.ndarray) -> str:
    """Hash exact C-order feature bytes, including dtype and shape."""

    value = np.asarray(array)
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(_feature_hash_prefix(contiguous.shape, contiguous.dtype))
    digest.update(contiguous.view(np.uint8).tobytes(order="C"))
    return digest.hexdigest()


def require_strict_role_count(scope: str, row_count: int) -> None:
    """Require the fixed real Stage 0 row count for one accepted role."""

    if scope not in EXPECTED_ROLE_ROWS:
        raise ValueError(f"unknown Stage 0 role: {scope!r}")
    expected = EXPECTED_ROLE_ROWS[scope]
    if type(row_count) is not int or row_count != expected:
        raise ValueError(
            f"{scope} must contain exactly {expected} ordered rows"
        )


def validate_cache_layout(array: object, *, strict: bool) -> None:
    """Validate the NumPy header without materializing feature contents."""

    shape_value = getattr(array, "shape", None)
    if not isinstance(shape_value, tuple) or len(shape_value) != 3:
        raise ValueError("canonical cache must be a rank-3 array")
    if any(type(value) is not int or value <= 0 for value in shape_value):
        raise ValueError("canonical cache dimensions must be positive integers")
    if np.dtype(getattr(array, "dtype", object)) != EXPECTED_CACHE_DTYPE:
        raise ValueError("canonical cache dtype must be float16")
    flags = getattr(array, "flags", None)
    if flags is None or not bool(getattr(flags, "c_contiguous", False)):
        raise ValueError("canonical cache must be C contiguous")
    if strict and shape_value != EXPECTED_CACHE_SHAPE:
        raise ValueError(
            "canonical cache shape must be exactly [16540, 5, 3200]"
        )


def _string_tuple(value: object, context: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{context} must be an array of non-empty strings")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise ValueError(f"{context} contains duplicate IDs")
    return result


def _row_tuple(value: object, context: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        type(item) is not int for item in value
    ):
        raise ValueError(f"{context} must be an array of integer rows")
    result = tuple(value)
    if any(item < 0 for item in result):
        raise ValueError(f"{context} contains an out of range row")
    if len(set(result)) != len(result):
        raise ValueError(f"{context} contains duplicate rows")
    return result


def _mapping_from_document(
    document: Mapping[str, object],
    capability: VerifiedArtifact,
    *,
    subject: int,
    scope: str,
    strict: bool,
) -> RoleMapping:
    subject_id = document.get("subject_id")
    expected_subject_string = f"sub-{subject:02d}"
    if subject_id not in (subject, expected_subject_string):
        raise ValueError(
            f"subject {subject:02d} protocol subject_id does not match filename"
        )
    records_sha256 = document.get("records_sha256")
    if not isinstance(records_sha256, str):
        raise ValueError("protocol records_sha256 must be a string")
    role_payloads = document.get("role_payloads")
    role_artifacts = document.get("role_artifacts")
    if not isinstance(role_payloads, Mapping) or not isinstance(
        role_artifacts, Mapping
    ):
        raise ValueError("protocol role containers must be objects")
    payload = role_payloads.get(scope)
    descriptor = role_artifacts.get(scope)
    if not isinstance(payload, Mapping) or not isinstance(descriptor, Mapping):
        raise ValueError(f"protocol is missing selected {scope} role")

    rows = _row_tuple(payload.get("row_indices"), f"{scope} row_indices")
    concept_ids = _string_tuple(
        payload.get("concept_ids"), f"{scope} concept_ids"
    )
    selected_ids = _string_tuple(
        payload.get("ordered_ids"), f"{scope} ordered_ids"
    )
    query_ids = _string_tuple(payload.get("query_ids"), f"{scope} query_ids")
    gallery_ids = _string_tuple(
        payload.get("gallery_ids"), f"{scope} gallery_ids"
    )
    if not rows or not concept_ids or not selected_ids:
        raise ValueError(f"{scope} selected role mapping must be non-empty")

    if scope == "train":
        if selected_ids != concept_ids:
            raise ValueError("train ordered IDs must exactly equal concept IDs")
        if query_ids or gallery_ids:
            raise ValueError("train role must not contain query/gallery IDs")
        if len(rows) % len(concept_ids) != 0:
            raise ValueError(
                "train rows must divide evenly across ordered concept IDs"
            )
        stimuli_per_concept = len(rows) // len(concept_ids)
        if stimuli_per_concept <= 0:
            raise ValueError("train must map at least one row per concept")
        if strict and stimuli_per_concept != 10:
            raise ValueError("strict train mapping requires ten rows per concept")
    else:
        if not (
            len(rows)
            == len(concept_ids)
            == len(selected_ids)
            == len(query_ids)
            == len(gallery_ids)
        ):
            raise ValueError(
                "val-dev rows, concepts, ordered IDs, queries, and gallery "
                "IDs must have identical lengths"
            )
        if selected_ids != query_ids or selected_ids != gallery_ids:
            raise ValueError(
                "val-dev ordered IDs must exactly equal query and gallery IDs"
            )
        stimuli_per_concept = 1

    role_payload_sha256 = descriptor.get("role_payload_sha256")
    if (
        not isinstance(role_payload_sha256, str)
        or role_payload_sha256 != sha256_json(payload)
    ):
        raise ValueError(f"{scope} role payload SHA-256 mismatch")
    if descriptor.get("source_records_sha256") != records_sha256:
        raise ValueError(f"{scope} source-record binding mismatch")
    if descriptor.get("ordered_ids_sha256") != ordered_ids_sha256(selected_ids):
        raise ValueError(f"{scope} ordered-ID SHA-256 mismatch")

    mapping = RoleMapping(
        scope=scope,
        row_indices=rows,
        concept_ids=concept_ids,
        ordered_ids=selected_ids,
        query_ids=query_ids,
        gallery_ids=gallery_ids,
        records_sha256=records_sha256,
        role_payload_sha256=role_payload_sha256,
        protocol_payload_sha256=capability.payload_sha256,
        stimuli_per_concept=stimuli_per_concept,
    )
    if strict:
        require_strict_role_count(scope, len(rows))
        if mapping.ordered_row_indices_sha256 != PINNED_ROW_SHA256[scope]:
            raise ValueError(f"{scope} ordered-row SHA-256 oracle mismatch")
        if mapping.ordered_ids_sha256 != PINNED_ORDERED_ID_SHA256[scope]:
            raise ValueError(f"{scope} ordered-ID SHA-256 oracle mismatch")
        if scope == "val-dev" and (
            mapping.concept_ids_sha256 != PINNED_CONCEPT_ID_SHA256["val-dev"]
        ):
            raise ValueError("val-dev concept-ID SHA-256 oracle mismatch")
    return mapping


def _load_selected_mapping(
    capability: VerifiedArtifact,
    *,
    subject: int,
    scope: str,
    strict: bool,
) -> RoleMapping:
    with capability.open_verified() as handle:
        try:
            document = json.load(handle)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("verified protocol is not valid JSON") from exc
    if not isinstance(document, Mapping):
        raise ValueError("verified protocol must be a JSON object")
    return _mapping_from_document(
        document,
        capability,
        subject=subject,
        scope=scope,
        strict=strict,
    )


def _verify_subject_mappings(
    capabilities: Sequence[VerifiedArtifact],
    *,
    strict: bool,
) -> tuple[
    dict[str, RoleMapping],
    list[dict[str, object]],
]:
    expected_capabilities = EXPECTED_SUBJECTS * len(EXPECTED_SCOPES)
    if len(capabilities) != expected_capabilities:
        raise AssertionError("internal selected-role capability count mismatch")
    references: dict[str, RoleMapping] = {}
    subjects: list[dict[str, object]] = []
    position = 0
    common_records_sha256: str | None = None
    for subject in range(1, EXPECTED_SUBJECTS + 1):
        subject_mappings: dict[str, RoleMapping] = {}
        role_hashes: dict[str, str] = {}
        protocol_hash: str | None = None
        for scope in EXPECTED_SCOPES:
            capability = capabilities[position]
            position += 1
            mapping = _load_selected_mapping(
                capability,
                subject=subject,
                scope=scope,
                strict=strict,
            )
            subject_mappings[scope] = mapping
            role_hashes[scope] = mapping.role_payload_sha256
            if protocol_hash is None:
                protocol_hash = mapping.protocol_payload_sha256
            elif protocol_hash != mapping.protocol_payload_sha256:
                raise ValueError(
                    f"subject {subject:02d} selected roles came from "
                    "different protocol bytes"
                )
            if common_records_sha256 is None:
                common_records_sha256 = mapping.records_sha256
            elif common_records_sha256 != mapping.records_sha256:
                raise ValueError("cross-subject records SHA-256 differs")

            reference = references.get(scope)
            if reference is None:
                references[scope] = mapping
            elif reference.semantic_key() != mapping.semantic_key():
                raise ValueError(
                    f"{scope} subject mappings differ from subject 01"
                )
        subjects.append(
            {
                "protocol_manifest": f"sub-{subject:02d}_protocol.json",
                "protocol_payload_sha256": protocol_hash,
                "records_sha256": common_records_sha256,
                "role_payload_sha256": role_hashes,
                "selected_roles": list(EXPECTED_SCOPES),
                "subject_id": subject,
            }
        )
    return references, subjects


@contextmanager
def _open_verified_numpy(
    capability: VerifiedArtifact,
) -> Iterator[np.ndarray]:
    """Memory-map NumPy bytes through the already verified open file."""

    with capability.open_verified() as handle:
        descriptor_path = f"/proc/self/fd/{handle.fileno()}"
        try:
            value = np.load(
                descriptor_path,
                mmap_mode="r",
                allow_pickle=False,
            )
        except (OSError, ValueError) as exc:
            raise ValueError("verified cache is not a safe NumPy array") from exc
        if not isinstance(value, np.ndarray):
            raise ValueError("verified cache payload must be one NumPy array")
        try:
            yield value
        finally:
            mapped = getattr(value, "_mmap", None)
            if mapped is not None:
                mapped.close()


def _new_feature_digest(
    shape: Sequence[int],
    dtype: np.dtype[object],
) -> "hashlib._Hash":
    digest = hashlib.sha256()
    digest.update(_feature_hash_prefix(shape, dtype))
    return digest


def _hash_full_cache_and_require_finite(
    array: np.ndarray,
    *,
    chunk_rows: int,
) -> str:
    digest = _new_feature_digest(array.shape, array.dtype)
    for start in range(0, array.shape[0], chunk_rows):
        chunk = np.ascontiguousarray(array[start : start + chunk_rows])
        if not bool(np.isfinite(chunk).all()):
            raise ValueError("canonical cache contains a non-finite value")
        digest.update(chunk.view(np.uint8).tobytes(order="C"))
    return digest.hexdigest()


def _hash_direct_index_view(
    array: np.ndarray,
    rows: Sequence[int],
    *,
    chunk_rows: int,
) -> str:
    shape = (len(rows), *array.shape[1:])
    digest = _new_feature_digest(shape, array.dtype)
    for start in range(0, len(rows), chunk_rows):
        row_chunk = np.asarray(rows[start : start + chunk_rows], dtype=np.intp)
        values = np.ascontiguousarray(array[row_chunk])
        if not bool(np.isfinite(values).all()):
            raise ValueError("selected cache view contains a non-finite value")
        digest.update(values.view(np.uint8).tobytes(order="C"))
    return digest.hexdigest()


def _scope_report(
    mapping: RoleMapping,
    array: np.ndarray,
    *,
    chunk_rows: int,
) -> dict[str, object]:
    return {
        "concept_count": len(mapping.concept_ids),
        "concept_ids_sha256": mapping.concept_ids_sha256,
        "direct_index_exhaustive": True,
        "direct_index_feature_bytes_sha256": _hash_direct_index_view(
            array,
            mapping.row_indices,
            chunk_rows=chunk_rows,
        ),
        "feature_shape": [
            len(mapping.row_indices),
            *[int(value) for value in array.shape[1:]],
        ],
        "finite": True,
        "ordered_ids_sha256": mapping.ordered_ids_sha256,
        "ordered_row_indices_sha256": (
            mapping.ordered_row_indices_sha256
        ),
        "row_count": len(mapping.row_indices),
        "stimuli_per_concept": mapping.stimuli_per_concept,
        "subject_mappings_identical": True,
    }


def build_stage0_cache_parity(
    manifest_dir: Path,
    canonical_cache: TypedArtifact,
    scopes: Sequence[str] = EXPECTED_SCOPES,
    *,
    strict: bool = True,
    chunk_rows: int = 256,
) -> dict[str, object]:
    """Build an exhaustive scoped-view/direct-index verification report.

    ``canonical_cache`` must be a strict Task 3 typed artifact.  There is no
    raw/legacy fallback and no implicit sidecar discovery.
    """

    accepted_scopes = _validate_scopes(scopes)
    if type(strict) is not bool:
        raise TypeError("strict must be a boolean")
    if type(chunk_rows) is not int or chunk_rows <= 0:
        raise ValueError("chunk_rows must be a positive integer")
    if not isinstance(canonical_cache, TypedArtifact):
        raise TypeError("canonical_cache must be a TypedArtifact")
    if canonical_cache.payload_type not in _CACHE_PAYLOAD_TYPES:
        raise ValueError("canonical_cache must declare a feature-cache type")
    if canonical_cache.role is not None:
        raise ValueError("canonical cache must not declare a protocol role")

    directory = Path(manifest_dir)
    selected_protocols: list[TypedArtifact] = []
    for subject in range(1, EXPECTED_SUBJECTS + 1):
        protocol_path = directory / f"sub-{subject:02d}_protocol.json"
        for scope in accepted_scopes:
            selected_protocols.append(
                TypedArtifact(
                    payload_type=_ROLE_PAYLOAD_TYPE,
                    payload_path=protocol_path,
                    envelope_path=protocol_path,
                    role=scope,
                )
            )

    # A single batch call performs every lexical/symlink/semantic check across
    # the cache and all selected roles before any semantic JSON or NumPy load.
    verified = verify_typed_artifacts(
        "val-dev",
        (canonical_cache, *selected_protocols),
    )
    cache_capability = verified[0]
    protocol_capabilities = verified[1:]
    if strict and cache_capability.payload_sha256 != PINNED_CACHE_SHA256:
        raise ValueError("canonical cache payload SHA-256 oracle mismatch")

    mappings, subjects = _verify_subject_mappings(
        protocol_capabilities,
        strict=strict,
    )
    train_rows = set(mappings["train"].row_indices)
    val_dev_rows = set(mappings["val-dev"].row_indices)
    if train_rows & val_dev_rows:
        raise ValueError("train and val-dev row sets must be disjoint")

    with _open_verified_numpy(cache_capability) as array:
        validate_cache_layout(array, strict=strict)
        cache_row_count = int(array.shape[0])
        for scope, mapping in mappings.items():
            if any(row >= cache_row_count for row in mapping.row_indices):
                raise ValueError(f"{scope} contains an out of range row")
        canonical_feature_sha256 = _hash_full_cache_and_require_finite(
            array,
            chunk_rows=chunk_rows,
        )
        scope_views = {
            scope: _scope_report(
                mappings[scope],
                array,
                chunk_rows=chunk_rows,
            )
            for scope in accepted_scopes
        }
        cache_shape = [int(value) for value in array.shape]
        cache_dtype = str(array.dtype)
        cache_c_contiguous = bool(array.flags.c_contiguous)

    return {
        "bit_identical_comparison_performed": False,
        "canonical_cache": {
            "c_contiguous": cache_c_contiguous,
            "direct_feature_bytes_sha256": canonical_feature_sha256,
            "dtype": cache_dtype,
            "finite": True,
            "path": os.path.abspath(os.fspath(canonical_cache.payload_path)),
            "payload_sha256": cache_capability.payload_sha256,
            "shape": cache_shape,
        },
        "comparison_statement": (
            "No independent cache was supplied. Every selected byte was "
            "exhaustively read from the canonical cache by the sealed direct "
            "row indices."
        ),
        "independent_cache_compared": False,
        "passed": True,
        "payload_type": "samga_brain_rw.stage0_cache_scoped_view_report",
        "schema_version": 1,
        "scope": "val-dev",
        "scope_views": scope_views,
        "scopes": list(accepted_scopes),
        "subjects": subjects,
        "verification_kind": "exhaustive_scoped_view_direct_index",
    }


def validate_stage0_report_path(path: Path) -> None:
    raw = os.fspath(path)
    lowered = raw.lower()
    if _FORMAL_TEST_RECORD_SHA256 in lowered:
        raise ValueError("report path contains the forbidden formal digest")
    parts = Path(raw).parts
    if any(part.lower() == "test_images" for part in parts):
        raise ValueError("report path contains a forbidden test_images component")
    if _SUBJECT_TEST_RE.fullmatch(Path(raw).name):
        raise ValueError("report path is a forbidden subject test manifest")

    absolute = Path(os.path.abspath(raw))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            break
        except OSError as exc:
            raise ValueError("cannot safely inspect report path") from exc
        if stat.S_ISLNK(mode):
            raise ValueError("report path contains a symlink component")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_stage0_cache_parity_report(
    output_path: Path,
    report: Mapping[str, object],
) -> None:
    """Exclusively publish one canonical report without replacement."""

    destination = Path(output_path)
    validate_stage0_report_path(destination)
    payload = canonical_json_bytes(dict(report)) + b"\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary_path = Path(temporary_name)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, destination, follow_symlinks=False)
        published = True
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
    if published:
        _fsync_directory(destination.parent)


__all__ = [
    "EXPECTED_CACHE_SHAPE",
    "EXPECTED_ROLE_ROWS",
    "EXPECTED_SCOPES",
    "PINNED_CACHE_SHA256",
    "RoleMapping",
    "build_stage0_cache_parity",
    "hash_feature_bytes",
    "hash_ordered_rows",
    "require_strict_role_count",
    "validate_cache_layout",
    "validate_stage0_report_path",
    "write_stage0_cache_parity_report",
]
