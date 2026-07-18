"""Deterministic split assignment and strict train-manifest sidecars."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .hashing import (
    SPLIT_SALT,
    STIMULUS_SALT,
    canonical_json_bytes,
    concept_digest,
    ordered_ids_sha256,
    sha256_json,
    stimulus_digest,
)


EXPECTED_CONCEPTS = 1_654
EXPECTED_STIMULI_PER_CONCEPT = 10
EXPECTED_RECORDS = EXPECTED_CONCEPTS * EXPECTED_STIMULI_PER_CONCEPT
VAL_DEV_CONCEPTS = 200
VAL_CONFIRM_CONCEPTS = 200
DEFAULT_PROTOCOL_CONFIG_SHA256 = (
    "0a9bb1dc750145ec94c35aaaddf5a834d303be3e6f69c9740237d9b967fd48bd"
)
SOURCE_TOP_KEYS = {
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
SOURCE_RECORD_KEYS = {
    "concept_id",
    "image_id",
    "image_path",
    "row_index",
    "validation_query",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SUBJECT_FILENAME_RE = re.compile(r"^(sub-\d{2})_train\.json$")


def _strict_keys(
    value: Mapping[str, object],
    expected: set[str],
    context: str,
) -> None:
    actual = set(value)
    missing = expected - actual
    unknown = actual - expected
    if missing:
        raise ValueError(f"{context} is missing keys: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{context} has unknown keys: {sorted(unknown)}")


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _sha256(value: object, context: str) -> str:
    digest = _string(value, context)
    if not _SHA256_RE.fullmatch(digest):
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return digest


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


@dataclass(frozen=True)
class ConceptAssignment:
    concept_id: str
    split_role: Literal["train", "val-dev", "val-confirm"]
    split_rank: int
    split_digest: str
    stimulus_ids: tuple[str, ...]
    row_indices: tuple[int, ...]
    selected_stimulus_id: str | None
    selected_stimulus_digest: str | None
    selected_row_index: int | None

    def to_payload(self) -> dict[str, object]:
        return {
            "concept_id": self.concept_id,
            "row_indices": list(self.row_indices),
            "selected_row_index": self.selected_row_index,
            "selected_stimulus_digest": self.selected_stimulus_digest,
            "selected_stimulus_id": self.selected_stimulus_id,
            "split_digest": self.split_digest,
            "split_rank": self.split_rank,
            "split_role": self.split_role,
            "stimulus_ids": list(self.stimulus_ids),
        }


@dataclass(frozen=True)
class SplitAssignment:
    records_sha256: str
    record_count: int
    concepts: tuple[ConceptAssignment, ...]
    protocol_config_sha256: str = DEFAULT_PROTOCOL_CONFIG_SHA256

    def __post_init__(self) -> None:
        _sha256(self.records_sha256, "records_sha256")
        _sha256(self.protocol_config_sha256, "protocol_config_sha256")
        if self.record_count != EXPECTED_RECORDS:
            raise ValueError(f"record_count must be {EXPECTED_RECORDS}")
        if len(self.concepts) != EXPECTED_CONCEPTS:
            raise ValueError(
                f"split assignment must contain exactly {EXPECTED_CONCEPTS} concepts"
            )
        ranks = [entry.split_rank for entry in self.concepts]
        if ranks != list(range(1, EXPECTED_CONCEPTS + 1)):
            raise ValueError("split ranks must be contiguous from 1 to 1654")
        roles = [entry.split_role for entry in self.concepts]
        if roles[:VAL_DEV_CONCEPTS] != ["val-dev"] * VAL_DEV_CONCEPTS:
            raise ValueError("ranks 1 through 200 must be val-dev")
        if roles[VAL_DEV_CONCEPTS : VAL_DEV_CONCEPTS + VAL_CONFIRM_CONCEPTS] != [
            "val-confirm"
        ] * VAL_CONFIRM_CONCEPTS:
            raise ValueError("ranks 201 through 400 must be val-confirm")
        if roles[VAL_DEV_CONCEPTS + VAL_CONFIRM_CONCEPTS :] != [
            "train"
        ] * (EXPECTED_CONCEPTS - VAL_DEV_CONCEPTS - VAL_CONFIRM_CONCEPTS):
            raise ValueError("a validation concept appears in train")

    def _ordered_ids(self) -> dict[str, list[str]]:
        train = [
            entry.concept_id
            for entry in self.concepts
            if entry.split_role == "train"
        ]
        val_dev = [
            entry for entry in self.concepts if entry.split_role == "val-dev"
        ]
        val_confirm = [
            entry
            for entry in self.concepts
            if entry.split_role == "val-confirm"
        ]
        val_dev_queries = [
            _string(entry.selected_stimulus_id, "val-dev selected stimulus")
            for entry in val_dev
        ]
        val_confirm_queries = [
            _string(entry.selected_stimulus_id, "val-confirm selected stimulus")
            for entry in val_confirm
        ]
        return {
            "train_concept_ids": train,
            "val_confirm_concept_ids": [
                entry.concept_id for entry in val_confirm
            ],
            "val_confirm_gallery_ids": list(val_confirm_queries),
            "val_confirm_query_ids": val_confirm_queries,
            "val_dev_concept_ids": [entry.concept_id for entry in val_dev],
            "val_dev_gallery_ids": list(val_dev_queries),
            "val_dev_query_ids": val_dev_queries,
        }

    def to_payload(self) -> dict[str, object]:
        ordered_ids = self._ordered_ids()
        return {
            "concepts": [entry.to_payload() for entry in self.concepts],
            "ordered_id_sha256": {
                key: ordered_ids_sha256(value)
                for key, value in ordered_ids.items()
            },
            "ordered_ids": ordered_ids,
            "payload_type": "samga_brain_rw.split_assignment",
            "protocol_config_sha256": self.protocol_config_sha256,
            "record_count": self.record_count,
            "records_sha256": self.records_sha256,
            "schema_version": 1,
            "split_salt": SPLIT_SALT,
            "stimulus_salt": STIMULUS_SALT,
        }

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_payload())


@dataclass(frozen=True)
class LoadedSourceManifest:
    path: Path
    raw_sha256: str
    subject_id: int | str
    records_sha256: str
    records: tuple[dict[str, object], ...]


def _normalize_records(
    records: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    if isinstance(records, (str, bytes, bytearray)):
        raise ValueError("records must be a sequence of objects")
    normalized: list[dict[str, object]] = []
    seen_pairs: set[tuple[str, str]] = set()
    concept_stimuli: dict[str, list[tuple[str, int]]] = defaultdict(list)

    for position, raw_record in enumerate(records):
        if not isinstance(raw_record, Mapping):
            raise ValueError(f"records[{position}] must be an object")
        record = dict(raw_record)
        _strict_keys(record, SOURCE_RECORD_KEYS, f"records[{position}]")
        concept_id = _string(
            record["concept_id"], f"records[{position}].concept_id"
        )
        stimulus_id = _string(
            record["image_id"], f"records[{position}].image_id"
        )
        _string(record["image_path"], f"records[{position}].image_path")
        row_index = record["row_index"]
        if type(row_index) is not int:
            raise ValueError(f"records[{position}].row_index must be an integer")
        if row_index != position:
            raise ValueError(
                "row indices must be contiguous in record order from 0 "
                f"through {len(records) - 1}"
            )
        if type(record["validation_query"]) is not bool:
            raise ValueError(
                f"records[{position}].validation_query must be a boolean"
            )
        pair = (concept_id, stimulus_id)
        if pair in seen_pairs:
            raise ValueError(
                "duplicate (concept_id, image_id) pair: "
                f"{concept_id!r}, {stimulus_id!r}"
            )
        seen_pairs.add(pair)
        concept_stimuli[concept_id].append((stimulus_id, row_index))
        normalized.append(record)

    if len(concept_stimuli) != EXPECTED_CONCEPTS:
        raise ValueError(
            f"records must contain exactly {EXPECTED_CONCEPTS} concepts; "
            f"found {len(concept_stimuli)}"
        )
    for concept_id, stimuli in concept_stimuli.items():
        if len(stimuli) != EXPECTED_STIMULI_PER_CONCEPT:
            raise ValueError(
                f"concept {concept_id!r} must contain exactly "
                f"{EXPECTED_STIMULI_PER_CONCEPT} stimuli; found {len(stimuli)}"
            )
    if len(normalized) != EXPECTED_RECORDS:
        raise ValueError(
            f"records must contain exactly {EXPECTED_RECORDS} rows; "
            f"found {len(normalized)}"
        )
    return tuple(normalized)


def partition_concepts(
    records: Sequence[Mapping[str, object]],
    *,
    protocol_config_sha256: str = DEFAULT_PROTOCOL_CONFIG_SHA256,
) -> SplitAssignment:
    """Validate canonical records and derive the sealed split registry."""
    normalized = _normalize_records(records)
    records_sha256 = hashlib.sha256(
        canonical_json_bytes(normalized)
    ).hexdigest()
    grouped: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for record in normalized:
        grouped[str(record["concept_id"])].append(
            (str(record["image_id"]), int(record["row_index"]))
        )

    ranked = sorted(
        grouped,
        key=lambda concept_id: (concept_digest(concept_id), concept_id),
    )
    concepts: list[ConceptAssignment] = []
    for rank, concept_id in enumerate(ranked, start=1):
        if rank <= VAL_DEV_CONCEPTS:
            role: Literal["train", "val-dev", "val-confirm"] = "val-dev"
        elif rank <= VAL_DEV_CONCEPTS + VAL_CONFIRM_CONCEPTS:
            role = "val-confirm"
        else:
            role = "train"
        stimuli = grouped[concept_id]
        stimulus_ids = tuple(stimulus_id for stimulus_id, _ in stimuli)
        row_indices = tuple(row_index for _, row_index in stimuli)
        selected_stimulus_id: str | None = None
        selected_stimulus_digest: str | None = None
        selected_row_index: int | None = None
        if role != "train":
            selected_stimulus_id, selected_row_index = min(
                stimuli,
                key=lambda item: (
                    stimulus_digest(role, concept_id, item[0]),
                    item[0],
                ),
            )
            selected_stimulus_digest = stimulus_digest(
                role, concept_id, selected_stimulus_id
            )
        concepts.append(
            ConceptAssignment(
                concept_id=concept_id,
                split_role=role,
                split_rank=rank,
                split_digest=concept_digest(concept_id),
                stimulus_ids=stimulus_ids,
                row_indices=row_indices,
                selected_stimulus_id=selected_stimulus_id,
                selected_stimulus_digest=selected_stimulus_digest,
                selected_row_index=selected_row_index,
            )
        )
    return SplitAssignment(
        records_sha256=records_sha256,
        record_count=len(normalized),
        concepts=tuple(concepts),
        protocol_config_sha256=_sha256(
            protocol_config_sha256, "protocol_config_sha256"
        ),
    )


def load_source_manifest(path: Path) -> LoadedSourceManifest:
    """Read source bytes once, then hash and strictly parse those same bytes."""
    source_path = Path(path)
    raw = source_path.read_bytes()
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"invalid UTF-8 JSON in {source_path}") from exc
    try:
        value = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {source_path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{source_path}: top-level JSON value must be an object")
    _strict_keys(value, SOURCE_TOP_KEYS, "source manifest")
    if value["schema_version"] != 1 or type(value["schema_version"]) is not int:
        raise ValueError("source manifest schema_version must be 1")
    if value["split"] != "train":
        raise ValueError("source manifest split must be train")

    filename_match = _SUBJECT_FILENAME_RE.fullmatch(source_path.name)
    if filename_match is None:
        raise ValueError(
            "source manifest filename must be sub-XX_train.json"
        )
    filename_subject_id = filename_match.group(1)
    subject_id = value["subject_id"]
    if type(subject_id) is int and 1 <= subject_id <= 99:
        canonical_subject_id = f"sub-{subject_id:02d}"
    elif isinstance(subject_id, str) and re.fullmatch(r"sub-\d{2}", subject_id):
        canonical_subject_id = subject_id
    else:
        raise ValueError(
            "source manifest subject_id must be an integer from 1 to 99 "
            "or a sub-XX string"
        )
    if canonical_subject_id != filename_subject_id:
        raise ValueError(
            "source manifest subject_id does not match filename: "
            f"{subject_id!r} != {filename_subject_id!r}"
        )
    _string(value["source_pt"], "source manifest source_pt")
    _string(value["eeg_dtype"], "source manifest eeg_dtype")
    if not isinstance(value["ch_names"], list) or any(
        not isinstance(channel, str) or not channel
        for channel in value["ch_names"]
    ):
        raise ValueError("source manifest ch_names must be an array of strings")
    if not isinstance(value["eeg_shape"], list) or any(
        type(dimension) is not int or dimension < 0
        for dimension in value["eeg_shape"]
    ):
        raise ValueError(
            "source manifest eeg_shape must be an array of non-negative integers"
        )
    if not isinstance(value["validation_concepts"], list) or any(
        not isinstance(concept_id, str)
        for concept_id in value["validation_concepts"]
    ):
        raise ValueError(
            "source manifest validation_concepts must be an array of strings"
        )
    _string(value["validation_salt"], "source manifest validation_salt")
    if not isinstance(value["records"], list):
        raise ValueError("source manifest records must be an array")
    records = _normalize_records(value["records"])
    records_sha256 = hashlib.sha256(canonical_json_bytes(records)).hexdigest()
    declared_records_sha256 = _sha256(
        value["records_sha256"], "source manifest records_sha256"
    )
    if declared_records_sha256 != records_sha256:
        raise ValueError(
            "source manifest records_sha256 does not match canonical records"
        )
    if value["eeg_shape"] and value["eeg_shape"][0] != len(records):
        raise ValueError(
            "source manifest eeg_shape[0] must equal the record count"
        )
    return LoadedSourceManifest(
        path=source_path,
        raw_sha256=raw_sha256,
        subject_id=subject_id,
        records_sha256=records_sha256,
        records=records,
    )


def _role_payload(
    assignment: SplitAssignment,
    role: Literal["train", "val-dev", "val-confirm"],
) -> dict[str, object]:
    entries = [entry for entry in assignment.concepts if entry.split_role == role]
    concept_ids = [entry.concept_id for entry in entries]
    if role == "train":
        ordered_ids = concept_ids
        row_indices = [
            row_index for entry in entries for row_index in entry.row_indices
        ]
        query_ids: list[str] = []
        gallery_ids: list[str] = []
    else:
        ordered_ids = [
            _string(entry.selected_stimulus_id, f"{role} selected stimulus")
            for entry in entries
        ]
        row_indices = [
            int(entry.selected_row_index)  # guarded by assignment construction
            for entry in entries
            if entry.selected_row_index is not None
        ]
        query_ids = list(ordered_ids)
        gallery_ids = list(ordered_ids)
    return {
        "concept_count": len(concept_ids),
        "concept_ids": concept_ids,
        "gallery_ids": gallery_ids,
        "ordered_ids": ordered_ids,
        "payload_type": "samga_brain_rw.role_payload",
        "query_ids": query_ids,
        "row_count": len(row_indices),
        "row_indices": row_indices,
        "schema_version": 1,
        "scope": role,
    }


def build_subject_protocol_manifest_from_loaded(
    source: LoadedSourceManifest,
    assignment: SplitAssignment,
) -> dict[str, object]:
    if source.records_sha256 != assignment.records_sha256:
        raise ValueError(
            "source records_sha256 or record order differs from split assignment"
        )
    split_assignment = assignment.to_payload()
    role_payloads = {
        role: _role_payload(assignment, role)
        for role in ("train", "val-dev", "val-confirm")
    }
    provenance = {
        "protocol_config_sha256": assignment.protocol_config_sha256,
        "source_manifest_sha256": source.raw_sha256,
    }
    provenance_sha256 = sha256_json(provenance)
    role_artifacts = {
        role: {
            "ordered_ids_sha256": ordered_ids_sha256(
                role_payloads[role]["ordered_ids"]  # type: ignore[arg-type]
            ),
            "payload_type": "samga_brain_rw.role_payload",
            "provenance_sha256": provenance_sha256,
            "role_payload_sha256": sha256_json(role_payloads[role]),
            "schema_version": 1,
            "scope": role,
            "source_records_sha256": source.records_sha256,
        }
        for role in ("train", "val-dev", "val-confirm")
    }
    return {
        "payload_type": "samga_brain_rw.subject_protocol_manifest",
        "protocol_config_sha256": assignment.protocol_config_sha256,
        "records_sha256": source.records_sha256,
        "role_artifacts": role_artifacts,
        "role_payloads": role_payloads,
        "schema_version": 1,
        "source_manifest_path": str(source.path),
        "source_manifest_sha256": source.raw_sha256,
        "split_assignment": split_assignment,
        "split_assignment_payload_sha256": assignment.sha256,
        "subject_id": source.subject_id,
    }


def build_subject_protocol_manifest(
    source_manifest: Path,
    assignment: SplitAssignment,
) -> dict[str, object]:
    return build_subject_protocol_manifest_from_loaded(
        load_source_manifest(Path(source_manifest)),
        assignment,
    )
