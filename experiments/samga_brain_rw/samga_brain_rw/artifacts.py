"""Immutable seals, ledgers, claims, and audits for the SAMGA brain-rw protocol.

The records in this module are control-plane artifacts.  They never open an
EEG, image, model, cache, score, or metric payload.  Every record is canonical
JSON published with a same-directory temporary file and a hard-link
no-replace operation.  In particular, immutable publication never uses
``os.replace``.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Iterator

from .hashing import canonical_json_bytes, sha256_json


SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_DOCUMENT_KEYS = {
    "artifact_type",
    "payload",
    "payload_sha256",
    "schema_version",
}


class ArtifactIntegrityError(ValueError):
    """An immutable artifact is malformed or no longer matches its seal."""


class ArtifactStateError(RuntimeError):
    """An immutable claim/ledger is in the wrong lifecycle state."""


def _require_sha256(value: str, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a 64-character lowercase SHA-256")
    return value


def _require_git_sha(value: str, field: str) -> str:
    if not isinstance(value, str) or _GIT_SHA_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase 40- or 64-character git SHA")
    return value


def _require_nonempty(value: str, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _require_safe_id(value: str, field: str) -> str:
    _require_nonempty(value, field)
    if _SAFE_ID_RE.fullmatch(value) is None or value in {".", ".."}:
        raise ValueError(f"{field} must be a safe portable identifier")
    return value


def _require_exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    context: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ArtifactIntegrityError(
            f"{context} keys mismatch: missing={missing}, extra={extra}"
        )


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ArtifactIntegrityError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _strict_load_bytes(data: bytes, path: Path) -> dict[str, object]:
    try:
        decoded = data.decode("utf-8")
        value = json.loads(decoded, object_pairs_hook=_strict_object)
    except ArtifactIntegrityError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError(f"invalid canonical JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ArtifactIntegrityError(f"artifact root must be an object: {path}")
    return value


def _read_regular_file(path: Path) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ArtifactIntegrityError(f"cannot open immutable artifact: {path}") from exc
    try:
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ArtifactIntegrityError(f"artifact is not a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _exclusive_publish(path: Path, data: bytes) -> None:
    """Publish bytes atomically without ever replacing ``path``.

    The hard-link operation is the single publication point and is atomic with
    respect to competing creators on GPFS and ordinary POSIX filesystems.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb", closefd=True) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except TypeError:
            os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _artifact_document(artifact_type: str, payload: dict[str, object]) -> dict[str, object]:
    return {
        "artifact_type": artifact_type,
        "payload": payload,
        "payload_sha256": sha256_json(payload),
        "schema_version": SCHEMA_VERSION,
    }


def _create_record(
    artifact_type: str,
    payload: dict[str, object],
    output_path: Path,
) -> tuple[Path, dict[str, object], str, str]:
    document = _artifact_document(artifact_type, payload)
    data = canonical_json_bytes(document)
    _exclusive_publish(Path(output_path), data)
    return (
        Path(output_path),
        payload,
        str(document["payload_sha256"]),
        hashlib.sha256(data).hexdigest(),
    )


def _read_record(
    path: Path,
    artifact_type: str,
    *,
    expected_payload_keys: set[str] | None = None,
    expected_payload_sha256: str | None = None,
    expected_artifact_sha256: str | None = None,
) -> tuple[dict[str, object], str, str]:
    data = _read_regular_file(Path(path))
    document = _strict_load_bytes(data, Path(path))
    _require_exact_keys(document, _DOCUMENT_KEYS, artifact_type)
    if document["schema_version"] != SCHEMA_VERSION:
        raise ArtifactIntegrityError(
            f"unsupported schema_version for {artifact_type}: "
            f"{document['schema_version']!r}"
        )
    if document["artifact_type"] != artifact_type:
        raise ArtifactIntegrityError(
            f"expected artifact_type {artifact_type!r}, "
            f"got {document['artifact_type']!r}"
        )
    payload = document["payload"]
    if not isinstance(payload, dict):
        raise ArtifactIntegrityError(f"{artifact_type} payload must be an object")
    if expected_payload_keys is not None:
        _require_exact_keys(payload, expected_payload_keys, f"{artifact_type} payload")
    payload_sha256 = document["payload_sha256"]
    if not isinstance(payload_sha256, str):
        raise ArtifactIntegrityError(f"{artifact_type} payload hash must be a string")
    _require_sha256(payload_sha256, f"{artifact_type}.payload_sha256")
    actual_payload_sha256 = sha256_json(payload)
    if actual_payload_sha256 != payload_sha256:
        raise ArtifactIntegrityError(f"{artifact_type} payload hash mismatch")
    artifact_sha256 = hashlib.sha256(data).hexdigest()
    if (
        expected_payload_sha256 is not None
        and payload_sha256 != expected_payload_sha256
    ):
        raise ArtifactIntegrityError(f"{artifact_type} expected payload hash mismatch")
    if (
        expected_artifact_sha256 is not None
        and artifact_sha256 != expected_artifact_sha256
    ):
        raise ArtifactIntegrityError(f"{artifact_type} artifact hash mismatch")
    return payload, payload_sha256, artifact_sha256


@contextmanager
def _transition_lock(directory: Path) -> Iterator[None]:
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / ".transition.lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


@dataclass(frozen=True)
class _ImmutableRecord:
    path: Path
    payload: dict[str, object]
    payload_sha256: str
    sha256: str

    _ARTIFACT_TYPE: ClassVar[str] = ""
    _PAYLOAD_KEYS: ClassVar[set[str] | None] = None

    def verify_unchanged(self) -> None:
        _read_record(
            self.path,
            self._ARTIFACT_TYPE,
            expected_payload_keys=self._PAYLOAD_KEYS,
            expected_payload_sha256=self.payload_sha256,
            expected_artifact_sha256=self.sha256,
        )

    @classmethod
    def _from_created(
        cls,
        created: tuple[Path, dict[str, object], str, str],
    ) -> "_ImmutableRecord":
        return cls(*created)

    @classmethod
    def _read(
        cls,
        path: Path,
        *,
        expected_payload_sha256: str | None = None,
        expected_artifact_sha256: str | None = None,
    ) -> "_ImmutableRecord":
        payload, payload_sha256, artifact_sha256 = _read_record(
            path,
            cls._ARTIFACT_TYPE,
            expected_payload_keys=cls._PAYLOAD_KEYS,
            expected_payload_sha256=expected_payload_sha256,
            expected_artifact_sha256=expected_artifact_sha256,
        )
        return cls(Path(path), payload, payload_sha256, artifact_sha256)


_REFIT_CELL_KEYS = {
    "adapter_sha256",
    "cell_key",
    "checkpoint_sha256",
    "component_schedule_sha256",
    "config_sha256",
    "dependency_sha256",
    "frozen_base_model_sha256",
    "job_id",
    "manifest_set_sha256",
    "role",
    "seed",
    "subject_set",
    "train_cache_sha256",
}


@dataclass(frozen=True)
class RefitCell:
    """One train-only refit output and all dependencies needed to replay it."""

    cell_key: str
    job_id: str
    subject_set: tuple[int, ...]
    seed: int
    role: str
    component_schedule_sha256: str
    config_sha256: str
    manifest_set_sha256: str
    checkpoint_sha256: str
    frozen_base_model_sha256: str
    adapter_sha256: str | None = None
    train_cache_sha256: str | None = None
    dependency_sha256: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_safe_id(self.cell_key, "cell_key")
        _require_safe_id(self.job_id, "job_id")
        _require_nonempty(self.role, "role")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        subjects = tuple(self.subject_set)
        if not subjects:
            raise ValueError("subject_set must be nonempty")
        if any(
            not isinstance(subject, int)
            or isinstance(subject, bool)
            or subject < 1
            or subject > 10
            for subject in subjects
        ):
            raise ValueError("subject_set values must be integers in 1..10")
        if len(set(subjects)) != len(subjects):
            raise ValueError("subject_set contains duplicate subjects")
        object.__setattr__(self, "subject_set", tuple(sorted(subjects)))
        dependencies = tuple(self.dependency_sha256)
        if len(set(dependencies)) != len(dependencies):
            raise ValueError("dependency_sha256 contains duplicate hashes")
        for index, value in enumerate(dependencies):
            _require_sha256(value, f"dependency_sha256[{index}]")
        object.__setattr__(self, "dependency_sha256", tuple(sorted(dependencies)))
        for field, value in (
            ("component_schedule_sha256", self.component_schedule_sha256),
            ("config_sha256", self.config_sha256),
            ("manifest_set_sha256", self.manifest_set_sha256),
            ("checkpoint_sha256", self.checkpoint_sha256),
            ("frozen_base_model_sha256", self.frozen_base_model_sha256),
        ):
            _require_sha256(value, field)
        if self.adapter_sha256 is not None:
            _require_sha256(self.adapter_sha256, "adapter_sha256")
        if self.train_cache_sha256 is not None:
            _require_sha256(self.train_cache_sha256, "train_cache_sha256")

    def to_payload(self) -> dict[str, object]:
        return {
            "adapter_sha256": self.adapter_sha256,
            "cell_key": self.cell_key,
            "checkpoint_sha256": self.checkpoint_sha256,
            "component_schedule_sha256": self.component_schedule_sha256,
            "config_sha256": self.config_sha256,
            "dependency_sha256": list(self.dependency_sha256),
            "frozen_base_model_sha256": self.frozen_base_model_sha256,
            "job_id": self.job_id,
            "manifest_set_sha256": self.manifest_set_sha256,
            "role": self.role,
            "seed": self.seed,
            "subject_set": list(self.subject_set),
            "train_cache_sha256": self.train_cache_sha256,
        }

    @classmethod
    def from_payload(cls, value: object) -> "RefitCell":
        if not isinstance(value, dict):
            raise ArtifactIntegrityError("refit cell must be an object")
        _require_exact_keys(value, _REFIT_CELL_KEYS, "refit cell")
        try:
            return cls(
                cell_key=value["cell_key"],  # type: ignore[arg-type]
                job_id=value["job_id"],  # type: ignore[arg-type]
                subject_set=tuple(value["subject_set"]),  # type: ignore[arg-type]
                seed=value["seed"],  # type: ignore[arg-type]
                role=value["role"],  # type: ignore[arg-type]
                component_schedule_sha256=value["component_schedule_sha256"],  # type: ignore[arg-type]
                config_sha256=value["config_sha256"],  # type: ignore[arg-type]
                manifest_set_sha256=value["manifest_set_sha256"],  # type: ignore[arg-type]
                checkpoint_sha256=value["checkpoint_sha256"],  # type: ignore[arg-type]
                frozen_base_model_sha256=value["frozen_base_model_sha256"],  # type: ignore[arg-type]
                adapter_sha256=value["adapter_sha256"],  # type: ignore[arg-type]
                train_cache_sha256=value["train_cache_sha256"],  # type: ignore[arg-type]
                dependency_sha256=tuple(value["dependency_sha256"]),  # type: ignore[arg-type]
            )
        except (TypeError, ValueError, KeyError) as exc:
            raise ArtifactIntegrityError("invalid refit cell") from exc


@dataclass(frozen=True)
class RefitArtifactLedger(_ImmutableRecord):
    _ARTIFACT_TYPE: ClassVar[str] = "refit_artifact_ledger"
    _PAYLOAD_KEYS: ClassVar[set[str]] = {"cell_count", "cells"}

    @classmethod
    def create(
        cls,
        cells: Sequence[RefitCell],
        output_path: Path,
    ) -> "RefitArtifactLedger":
        values = list(cells)
        if not values:
            raise ValueError("refit artifact ledger must be nonempty")
        if any(not isinstance(cell, RefitCell) for cell in values):
            raise TypeError("all refit artifact ledger cells must be RefitCell")
        cell_keys = [cell.cell_key for cell in values]
        if len(set(cell_keys)) != len(cell_keys):
            raise ValueError("duplicate cell_key in refit artifact ledger")
        job_ids = [cell.job_id for cell in values]
        if len(set(job_ids)) != len(job_ids):
            raise ValueError("duplicate job_id in refit artifact ledger")
        ordered = sorted(values, key=lambda cell: cell.cell_key)
        payload: dict[str, object] = {
            "cell_count": len(ordered),
            "cells": [cell.to_payload() for cell in ordered],
        }
        return cls._from_created(
            _create_record(cls._ARTIFACT_TYPE, payload, output_path)
        )  # type: ignore[return-value]

    @classmethod
    def verify(
        cls,
        path: Path,
        expected_payload_sha256: str | None = None,
    ) -> "RefitArtifactLedger":
        record = cls._read(path, expected_payload_sha256=expected_payload_sha256)
        cells = record.payload["cells"]
        count = record.payload["cell_count"]
        if not isinstance(cells, list) or not isinstance(count, int):
            raise ArtifactIntegrityError("invalid refit ledger payload types")
        parsed = [RefitCell.from_payload(cell) for cell in cells]
        if count != len(parsed) or not parsed:
            raise ArtifactIntegrityError("invalid refit ledger cell_count")
        keys = [cell.cell_key for cell in parsed]
        jobs = [cell.job_id for cell in parsed]
        if keys != sorted(keys) or len(set(keys)) != len(keys):
            raise ArtifactIntegrityError("refit ledger cells are not unique/canonical")
        if len(set(jobs)) != len(jobs):
            raise ArtifactIntegrityError("refit ledger job IDs are not unique")
        return record  # type: ignore[return-value]


@dataclass(frozen=True)
class ConfirmationSeal(_ImmutableRecord):
    _ARTIFACT_TYPE: ClassVar[str] = "confirmation_seal"
    _PAYLOAD_KEYS: ClassVar[set[str]] = {
        "job_map_sha256",
        "registry_sha256",
        "survivor_config_sha256",
    }

    @classmethod
    def create(
        cls,
        survivor_config_sha256: Sequence[str],
        registry_sha256: str,
        job_map_sha256: str,
        output_path: Path,
    ) -> "ConfirmationSeal":
        survivors = list(survivor_config_sha256)
        if not survivors:
            raise ValueError("survivor_config_sha256 must be nonempty")
        for index, value in enumerate(survivors):
            _require_sha256(value, f"survivor_config_sha256[{index}]")
        if len(set(survivors)) != len(survivors):
            raise ValueError("duplicate survivor config SHA-256")
        payload: dict[str, object] = {
            "job_map_sha256": _require_sha256(job_map_sha256, "job_map_sha256"),
            "registry_sha256": _require_sha256(
                registry_sha256, "registry_sha256"
            ),
            "survivor_config_sha256": sorted(survivors),
        }
        return cls._from_created(
            _create_record(cls._ARTIFACT_TYPE, payload, output_path)
        )  # type: ignore[return-value]

    @classmethod
    def verify(
        cls,
        path: Path,
        expected_payload_sha256: str | None = None,
    ) -> "ConfirmationSeal":
        record = cls._read(path, expected_payload_sha256=expected_payload_sha256)
        survivors = record.payload["survivor_config_sha256"]
        if not isinstance(survivors, list) or not survivors:
            raise ArtifactIntegrityError("confirmation survivors must be nonempty")
        try:
            for index, value in enumerate(survivors):
                _require_sha256(value, f"survivor_config_sha256[{index}]")
        except ValueError as exc:
            raise ArtifactIntegrityError("invalid confirmation survivor hash") from exc
        if survivors != sorted(survivors) or len(set(survivors)) != len(survivors):
            raise ArtifactIntegrityError("confirmation survivors are not unique/canonical")
        try:
            _require_sha256(
                record.payload["registry_sha256"],  # type: ignore[arg-type]
                "registry_sha256",
            )
            _require_sha256(
                record.payload["job_map_sha256"],  # type: ignore[arg-type]
                "job_map_sha256",
            )
        except ValueError as exc:
            raise ArtifactIntegrityError("invalid confirmation dependency hash") from exc
        return record  # type: ignore[return-value]


_PREPARATION_KEYS = {
    "confirmation_registry_sha256",
    "expected_formal_cell_keys_sha256",
    "final_selection_sha256",
    "formal_input_request_sha256",
    "git_sha",
    "refit_artifact_ledger_sha256",
    "refit_plan_sha256",
    "upstream_sha",
}


@dataclass(frozen=True)
class FormalPreparationSeal(_ImmutableRecord):
    _ARTIFACT_TYPE: ClassVar[str] = "formal_preparation_seal"
    _PAYLOAD_KEYS: ClassVar[set[str]] = _PREPARATION_KEYS

    @classmethod
    def create(
        cls,
        final_selection_sha256: str,
        confirmation_registry_sha256: str,
        refit_plan_sha256: str,
        refit_artifact_ledger_sha256: str,
        formal_input_request_sha256: str,
        expected_formal_cell_keys_sha256: str,
        git_sha: str,
        upstream_sha: str,
        output_path: Path,
    ) -> "FormalPreparationSeal":
        payload: dict[str, object] = {
            "confirmation_registry_sha256": _require_sha256(
                confirmation_registry_sha256, "confirmation_registry_sha256"
            ),
            "expected_formal_cell_keys_sha256": _require_sha256(
                expected_formal_cell_keys_sha256,
                "expected_formal_cell_keys_sha256",
            ),
            "final_selection_sha256": _require_sha256(
                final_selection_sha256, "final_selection_sha256"
            ),
            "formal_input_request_sha256": _require_sha256(
                formal_input_request_sha256, "formal_input_request_sha256"
            ),
            "git_sha": _require_git_sha(git_sha, "git_sha"),
            "refit_artifact_ledger_sha256": _require_sha256(
                refit_artifact_ledger_sha256, "refit_artifact_ledger_sha256"
            ),
            "refit_plan_sha256": _require_sha256(
                refit_plan_sha256, "refit_plan_sha256"
            ),
            "upstream_sha": _require_git_sha(upstream_sha, "upstream_sha"),
        }
        return cls._from_created(
            _create_record(cls._ARTIFACT_TYPE, payload, output_path)
        )  # type: ignore[return-value]

    @classmethod
    def verify(
        cls,
        path: Path,
        expected_payload_sha256: str | None = None,
    ) -> "FormalPreparationSeal":
        record = cls._read(path, expected_payload_sha256=expected_payload_sha256)
        _validate_preparation_payload(record.payload)
        return record  # type: ignore[return-value]


def _validate_preparation_payload(payload: Mapping[str, object]) -> None:
    try:
        for field in _PREPARATION_KEYS - {"git_sha", "upstream_sha"}:
            _require_sha256(payload[field], field)  # type: ignore[arg-type]
        _require_git_sha(payload["git_sha"], "git_sha")  # type: ignore[arg-type]
        _require_git_sha(payload["upstream_sha"], "upstream_sha")  # type: ignore[arg-type]
    except (KeyError, ValueError) as exc:
        raise ArtifactIntegrityError("invalid formal preparation seal payload") from exc


_AUDIT_KEYS = {"expected_payload_sha256", "seal_sha256"}


@dataclass(frozen=True)
class FormalPreparationAudit(_ImmutableRecord):
    _ARTIFACT_TYPE: ClassVar[str] = "formal_preparation_audit"
    _PAYLOAD_KEYS: ClassVar[set[str]] = _AUDIT_KEYS

    @classmethod
    def create(
        cls,
        preparation_seal_sha256: str,
        expected_payload_sha256: str,
        output_path: Path,
    ) -> "FormalPreparationAudit":
        payload: dict[str, object] = {
            "expected_payload_sha256": _require_sha256(
                expected_payload_sha256, "expected_payload_sha256"
            ),
            "seal_sha256": _require_sha256(
                preparation_seal_sha256, "preparation_seal_sha256"
            ),
        }
        return cls._from_created(
            _create_record(cls._ARTIFACT_TYPE, payload, output_path)
        )  # type: ignore[return-value]

    @classmethod
    def verify(
        cls,
        path: Path,
        *,
        expected_preparation_seal_sha256: str,
        expected_payload_sha256: str,
    ) -> "FormalPreparationAudit":
        record = cls._read(path)
        _require_sha256(
            expected_preparation_seal_sha256,
            "expected_preparation_seal_sha256",
        )
        _require_sha256(expected_payload_sha256, "expected_payload_sha256")
        if record.payload["seal_sha256"] != expected_preparation_seal_sha256:
            raise ArtifactIntegrityError("preparation audit seal hash mismatch")
        if record.payload["expected_payload_sha256"] != expected_payload_sha256:
            raise ArtifactIntegrityError("preparation audit payload hash mismatch")
        return record  # type: ignore[return-value]


_FINAL_RUN_KEYS = {
    "candidate_config_sha256",
    "confirmation_registry_sha256",
    "control_config_sha256",
    "final_selection_sha256",
    "formal_input_ledger_sha256",
    "formal_job_map_sha256",
    "git_sha",
    "refit_artifact_ledger_sha256",
    "refit_plan_sha256",
    "upstream_sha",
}


@dataclass(frozen=True)
class FinalRunSeal(_ImmutableRecord):
    _ARTIFACT_TYPE: ClassVar[str] = "final_run_seal"
    _PAYLOAD_KEYS: ClassVar[set[str]] = _FINAL_RUN_KEYS

    @classmethod
    def create(
        cls,
        final_selection_sha256: str,
        candidate_config_sha256: str,
        control_config_sha256: str,
        confirmation_registry_sha256: str,
        refit_plan_sha256: str,
        refit_artifact_ledger_sha256: str,
        formal_input_ledger_sha256: str,
        formal_job_map_sha256: str,
        git_sha: str,
        upstream_sha: str,
        output_path: Path,
    ) -> "FinalRunSeal":
        payload: dict[str, object] = {
            "candidate_config_sha256": _require_sha256(
                candidate_config_sha256, "candidate_config_sha256"
            ),
            "confirmation_registry_sha256": _require_sha256(
                confirmation_registry_sha256, "confirmation_registry_sha256"
            ),
            "control_config_sha256": _require_sha256(
                control_config_sha256, "control_config_sha256"
            ),
            "final_selection_sha256": _require_sha256(
                final_selection_sha256, "final_selection_sha256"
            ),
            "formal_input_ledger_sha256": _require_sha256(
                formal_input_ledger_sha256, "formal_input_ledger_sha256"
            ),
            "formal_job_map_sha256": _require_sha256(
                formal_job_map_sha256, "formal_job_map_sha256"
            ),
            "git_sha": _require_git_sha(git_sha, "git_sha"),
            "refit_artifact_ledger_sha256": _require_sha256(
                refit_artifact_ledger_sha256, "refit_artifact_ledger_sha256"
            ),
            "refit_plan_sha256": _require_sha256(
                refit_plan_sha256, "refit_plan_sha256"
            ),
            "upstream_sha": _require_git_sha(upstream_sha, "upstream_sha"),
        }
        return cls._from_created(
            _create_record(cls._ARTIFACT_TYPE, payload, output_path)
        )  # type: ignore[return-value]

    @classmethod
    def verify(
        cls,
        path: Path,
        expected_payload_sha256: str,
    ) -> "FinalRunSeal":
        _require_sha256(expected_payload_sha256, "expected_payload_sha256")
        record = cls._read(path, expected_payload_sha256=expected_payload_sha256)
        try:
            for field in _FINAL_RUN_KEYS - {"git_sha", "upstream_sha"}:
                _require_sha256(record.payload[field], field)  # type: ignore[arg-type]
            _require_git_sha(record.payload["git_sha"], "git_sha")  # type: ignore[arg-type]
            _require_git_sha(
                record.payload["upstream_sha"], "upstream_sha"  # type: ignore[arg-type]
            )
        except ValueError as exc:
            raise ArtifactIntegrityError("invalid final run seal payload") from exc
        return record  # type: ignore[return-value]


@dataclass(frozen=True)
class FinalRunAudit(_ImmutableRecord):
    _ARTIFACT_TYPE: ClassVar[str] = "final_run_audit"
    _PAYLOAD_KEYS: ClassVar[set[str]] = _AUDIT_KEYS

    @classmethod
    def create(
        cls,
        final_run_seal_sha256: str,
        expected_payload_sha256: str,
        output_path: Path,
    ) -> "FinalRunAudit":
        payload: dict[str, object] = {
            "expected_payload_sha256": _require_sha256(
                expected_payload_sha256, "expected_payload_sha256"
            ),
            "seal_sha256": _require_sha256(
                final_run_seal_sha256, "final_run_seal_sha256"
            ),
        }
        return cls._from_created(
            _create_record(cls._ARTIFACT_TYPE, payload, output_path)
        )  # type: ignore[return-value]

    @classmethod
    def verify(
        cls,
        path: Path,
        *,
        expected_final_run_seal_sha256: str,
        expected_payload_sha256: str,
    ) -> "FinalRunAudit":
        record = cls._read(path)
        _require_sha256(
            expected_final_run_seal_sha256,
            "expected_final_run_seal_sha256",
        )
        _require_sha256(expected_payload_sha256, "expected_payload_sha256")
        if record.payload["seal_sha256"] != expected_final_run_seal_sha256:
            raise ArtifactIntegrityError("final run audit seal hash mismatch")
        if record.payload["expected_payload_sha256"] != expected_payload_sha256:
            raise ArtifactIntegrityError("final run audit payload hash mismatch")
        return record  # type: ignore[return-value]


@dataclass(frozen=True)
class ClaimCompletion(_ImmutableRecord):
    """An exclusively published completion record for one immutable claim."""

    artifact_type: str

    _ARTIFACT_TYPE: ClassVar[str] = ""

    def verify_unchanged(self) -> None:
        _read_record(
            self.path,
            self.artifact_type,
            expected_payload_sha256=self.payload_sha256,
            expected_artifact_sha256=self.sha256,
        )


def _completion_from_created(
    artifact_type: str,
    created: tuple[Path, dict[str, object], str, str],
) -> ClaimCompletion:
    return ClaimCompletion(*created, artifact_type=artifact_type)


def _validate_output_hashes(output_hashes: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(output_hashes, Mapping) or not output_hashes:
        raise ValueError("output_hashes must be a nonempty mapping")
    normalized: dict[str, str] = {}
    for key, value in output_hashes.items():
        if not isinstance(key, str) or not key.endswith("_sha256"):
            raise ValueError("output hash keys must end with _sha256")
        if key in normalized:
            raise ValueError(f"duplicate output hash key: {key}")
        normalized[key] = _require_sha256(value, key)
    return dict(sorted(normalized.items()))


@dataclass(frozen=True)
class CellClaim(_ImmutableRecord):
    generation: int
    base_dir: Path
    claim_artifact_type: str
    completion_artifact_type: str
    recovery_artifact_type: str

    _ARTIFACT_TYPE: ClassVar[str] = ""

    @property
    def completion_path(self) -> Path:
        return self.path.parent / "completion.json"

    @property
    def recovery_path(self) -> Path:
        return self.path.parent / "recovery.json"

    def verify_unchanged(self) -> None:
        _read_record(
            self.path,
            self.claim_artifact_type,
            expected_payload_sha256=self.payload_sha256,
            expected_artifact_sha256=self.sha256,
        )

    def _verify_completion(self) -> ClaimCompletion:
        payload, payload_sha256, artifact_sha256 = _read_record(
            self.completion_path,
            self.completion_artifact_type,
            expected_payload_keys={"claim_sha256", "output_hashes"},
        )
        if payload["claim_sha256"] != self.sha256:
            raise ArtifactIntegrityError("completion binds a different claim")
        try:
            outputs = _validate_output_hashes(payload["output_hashes"])  # type: ignore[arg-type]
        except ValueError as exc:
            raise ArtifactIntegrityError("invalid claim completion outputs") from exc
        if outputs != payload["output_hashes"]:
            raise ArtifactIntegrityError("completion outputs are not canonical")
        return ClaimCompletion(
            self.completion_path,
            payload,
            payload_sha256,
            artifact_sha256,
            artifact_type=self.completion_artifact_type,
        )

    def _verify_recovery(self) -> None:
        payload, _, _ = _read_record(
            self.recovery_path,
            self.recovery_artifact_type,
            expected_payload_keys={
                "claim_sha256",
                "next_generation",
                "recovery_audit_sha256",
            },
        )
        if payload["claim_sha256"] != self.sha256:
            raise ArtifactIntegrityError("recovery binds a different claim")
        if payload["next_generation"] != self.generation + 1:
            raise ArtifactIntegrityError("recovery next generation mismatch")
        try:
            _require_sha256(
                payload["recovery_audit_sha256"],  # type: ignore[arg-type]
                "recovery_audit_sha256",
            )
        except ValueError as exc:
            raise ArtifactIntegrityError("invalid recovery audit hash") from exc

    def is_complete(self) -> bool:
        if not self.completion_path.exists():
            return False
        self._verify_completion()
        return True

    def assert_unconsumed(self) -> None:
        self.verify_unchanged()
        if self.completion_path.exists():
            self._verify_completion()
            raise ArtifactStateError("claim is already consumed")
        if self.recovery_path.exists():
            self._verify_recovery()
            raise ArtifactStateError("claim was recovered and is no longer active")

    def complete(self, output_hashes: Mapping[str, str]) -> ClaimCompletion:
        outputs = _validate_output_hashes(output_hashes)
        with _transition_lock(self.base_dir):
            if self.completion_path.exists():
                raise FileExistsError(self.completion_path)
            self.assert_unconsumed()
            payload: dict[str, object] = {
                "claim_sha256": self.sha256,
                "output_hashes": outputs,
            }
            return _completion_from_created(
                self.completion_artifact_type,
                _create_record(
                    self.completion_artifact_type,
                    payload,
                    self.completion_path,
                ),
            )


def _make_cell_claim(
    created: tuple[Path, dict[str, object], str, str],
    *,
    generation: int,
    base_dir: Path,
    claim_artifact_type: str,
    completion_artifact_type: str,
    recovery_artifact_type: str,
) -> CellClaim:
    return CellClaim(
        *created,
        generation=generation,
        base_dir=base_dir,
        claim_artifact_type=claim_artifact_type,
        completion_artifact_type=completion_artifact_type,
        recovery_artifact_type=recovery_artifact_type,
    )


class _CellLedger:
    _CLAIM_TYPE = ""
    _COMPLETION_TYPE = ""
    _RECOVERY_TYPE = ""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _publish_claim(
        self,
        *,
        cell_key: str,
        payload: dict[str, object],
        generation: int = 1,
    ) -> CellClaim:
        base_dir = self.root / cell_key
        generation_dir = base_dir / f"generation-{generation:06d}"
        path = generation_dir / "claim.json"
        with _transition_lock(base_dir):
            created = _create_record(self._CLAIM_TYPE, payload, path)
        return _make_cell_claim(
            created,
            generation=generation,
            base_dir=base_dir,
            claim_artifact_type=self._CLAIM_TYPE,
            completion_artifact_type=self._COMPLETION_TYPE,
            recovery_artifact_type=self._RECOVERY_TYPE,
        )

    def recover(
        self,
        claim: CellClaim,
        recovery_audit_sha256: str,
    ) -> CellClaim:
        _require_sha256(recovery_audit_sha256, "recovery_audit_sha256")
        if claim.claim_artifact_type != self._CLAIM_TYPE:
            raise ValueError("claim belongs to another ledger type")
        if claim.base_dir.parent != self.root:
            raise ValueError("claim belongs to another ledger root")
        with _transition_lock(claim.base_dir):
            if claim.recovery_path.exists():
                raise FileExistsError(claim.recovery_path)
            claim.assert_unconsumed()
            next_generation = claim.generation + 1
            recovery_payload: dict[str, object] = {
                "claim_sha256": claim.sha256,
                "next_generation": next_generation,
                "recovery_audit_sha256": recovery_audit_sha256,
            }
            recovery = _create_record(
                self._RECOVERY_TYPE,
                recovery_payload,
                claim.recovery_path,
            )
            new_payload = dict(claim.payload)
            new_payload["generation"] = next_generation
            new_payload["recovered_from_claim_sha256"] = claim.sha256
            new_payload["recovery_record_sha256"] = recovery[3]
            new_path = (
                claim.base_dir
                / f"generation-{next_generation:06d}"
                / "claim.json"
            )
            created = _create_record(self._CLAIM_TYPE, new_payload, new_path)
        return _make_cell_claim(
            created,
            generation=next_generation,
            base_dir=claim.base_dir,
            claim_artifact_type=self._CLAIM_TYPE,
            completion_artifact_type=self._COMPLETION_TYPE,
            recovery_artifact_type=self._RECOVERY_TYPE,
        )


class ConfirmationCellLedger(_CellLedger):
    _CLAIM_TYPE = "confirmation_cell_claim"
    _COMPLETION_TYPE = "confirmation_cell_completion"
    _RECOVERY_TYPE = "confirmation_cell_recovery"

    def __init__(self, root: Path, *, job_map_sha256: str) -> None:
        super().__init__(root)
        self.job_map_sha256 = _require_sha256(
            job_map_sha256, "job_map_sha256"
        )

    def claim(
        self,
        seal_sha256: str,
        stage: int,
        role: str,
        subject: int,
        seed: int,
    ) -> CellClaim:
        _require_sha256(seal_sha256, "seal_sha256")
        if not isinstance(stage, int) or isinstance(stage, bool) or stage < 1:
            raise ValueError("stage must be a positive integer")
        _require_safe_id(role, "role")
        _require_subject_seed(subject, seed, formal=False)
        cell_key = (
            f"stage-{stage:02d}_role-{role}_sub-{subject:02d}_seed-{seed}"
        )
        payload: dict[str, object] = {
            "cell_key": cell_key,
            "generation": 1,
            "job_map_sha256": self.job_map_sha256,
            "recovered_from_claim_sha256": None,
            "recovery_record_sha256": None,
            "role": role,
            "seal_sha256": seal_sha256,
            "seed": seed,
            "stage": stage,
            "subject": subject,
        }
        return self._publish_claim(cell_key=cell_key, payload=payload)


def _require_subject_seed(subject: int, seed: int, *, formal: bool) -> None:
    if (
        not isinstance(subject, int)
        or isinstance(subject, bool)
        or subject < 1
        or subject > 10
    ):
        raise ValueError("subject must be an integer in 1..10")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if formal and seed not in range(42, 47):
        raise ValueError("formal seed must be one of 42, 43, 44, 45, 46")
    if not formal and seed < 0:
        raise ValueError("seed must be nonnegative")


@dataclass(frozen=True, order=True)
class FormalCellKey:
    role: str
    subject: int
    seed: int


def expected_formal_cell_keys() -> tuple[FormalCellKey, ...]:
    """Return candidate/control × subjects 01..10 × seeds 42..46."""

    return tuple(
        FormalCellKey(role=role, subject=subject, seed=seed)
        for role in ("candidate", "control")
        for subject in range(1, 11)
        for seed in range(42, 47)
    )


class FormalCellLedger(_CellLedger):
    _CLAIM_TYPE = "formal_cell_claim"
    _COMPLETION_TYPE = "formal_cell_completion"
    _RECOVERY_TYPE = "formal_cell_recovery"

    def __init__(self, root: Path, *, formal_job_map_sha256: str) -> None:
        super().__init__(root)
        self.formal_job_map_sha256 = _require_sha256(
            formal_job_map_sha256, "formal_job_map_sha256"
        )

    def claim(
        self,
        final_run_seal_sha256: str,
        final_run_audit_sha256: str,
        role: str,
        subject: int,
        seed: int,
    ) -> CellClaim:
        _require_sha256(final_run_seal_sha256, "final_run_seal_sha256")
        _require_sha256(final_run_audit_sha256, "final_run_audit_sha256")
        if role not in {"candidate", "control"}:
            raise ValueError("formal role must be candidate or control")
        _require_subject_seed(subject, seed, formal=True)
        cell_key = f"role-{role}_sub-{subject:02d}_seed-{seed}"
        payload: dict[str, object] = {
            "cell_key": cell_key,
            "final_run_audit_sha256": final_run_audit_sha256,
            "final_run_seal_sha256": final_run_seal_sha256,
            "formal_job_map_sha256": self.formal_job_map_sha256,
            "generation": 1,
            "recovered_from_claim_sha256": None,
            "recovery_record_sha256": None,
            "role": role,
            "seed": seed,
            "subject": subject,
        }
        return self._publish_claim(cell_key=cell_key, payload=payload)


@dataclass(frozen=True)
class FormalInputClaim(_ImmutableRecord):
    generation: int
    base_dir: Path

    _ARTIFACT_TYPE: ClassVar[str] = "formal_input_claim"

    @property
    def completion_path(self) -> Path:
        return self.path.parent / "completion.json"

    @property
    def recovery_path(self) -> Path:
        return self.path.parent / "recovery.json"

    def _verify_completion(self) -> ClaimCompletion:
        expected_keys = {
            "adapter_sha256",
            "base_model_sha256",
            "claim_sha256",
            "manifest_sha256",
            "ordered_ids_sha256",
            "payload_sha256",
            "preprocessing_sha256",
        }
        payload, payload_hash, artifact_hash = _read_record(
            self.completion_path,
            "formal_input_completion",
            expected_payload_keys=expected_keys,
        )
        if payload["claim_sha256"] != self.sha256:
            raise ArtifactIntegrityError("formal input completion claim mismatch")
        try:
            for field in expected_keys - {"adapter_sha256", "claim_sha256"}:
                _require_sha256(payload[field], field)  # type: ignore[arg-type]
            if payload["adapter_sha256"] is not None:
                _require_sha256(
                    payload["adapter_sha256"],  # type: ignore[arg-type]
                    "adapter_sha256",
                )
        except ValueError as exc:
            raise ArtifactIntegrityError("invalid formal input completion") from exc
        return ClaimCompletion(
            self.completion_path,
            payload,
            payload_hash,
            artifact_hash,
            artifact_type="formal_input_completion",
        )

    def is_complete(self) -> bool:
        if not self.completion_path.exists():
            return False
        self._verify_completion()
        return True

    def assert_unconsumed(self) -> None:
        self.verify_unchanged()
        if self.completion_path.exists():
            self._verify_completion()
            raise ArtifactStateError("formal input claim is already consumed")
        if self.recovery_path.exists():
            _read_record(
                self.recovery_path,
                "formal_input_recovery",
                expected_payload_keys={
                    "claim_sha256",
                    "next_generation",
                    "recovery_audit_sha256",
                },
            )
            raise ArtifactStateError("formal input claim was recovered")

    def complete(
        self,
        *,
        manifest_sha256: str,
        ordered_ids_sha256: str,
        preprocessing_sha256: str,
        base_model_sha256: str,
        payload_sha256: str,
        adapter_sha256: str | None = None,
    ) -> ClaimCompletion:
        payload: dict[str, object] = {
            "adapter_sha256": (
                None
                if adapter_sha256 is None
                else _require_sha256(adapter_sha256, "adapter_sha256")
            ),
            "base_model_sha256": _require_sha256(
                base_model_sha256, "base_model_sha256"
            ),
            "claim_sha256": self.sha256,
            "manifest_sha256": _require_sha256(
                manifest_sha256, "manifest_sha256"
            ),
            "ordered_ids_sha256": _require_sha256(
                ordered_ids_sha256, "ordered_ids_sha256"
            ),
            "payload_sha256": _require_sha256(
                payload_sha256, "payload_sha256"
            ),
            "preprocessing_sha256": _require_sha256(
                preprocessing_sha256, "preprocessing_sha256"
            ),
        }
        with _transition_lock(self.base_dir):
            if self.completion_path.exists():
                raise FileExistsError(self.completion_path)
            self.assert_unconsumed()
            return _completion_from_created(
                "formal_input_completion",
                _create_record(
                    "formal_input_completion",
                    payload,
                    self.completion_path,
                ),
            )


@dataclass(frozen=True)
class FormalInputLedgerSnapshot(_ImmutableRecord):
    _ARTIFACT_TYPE: ClassVar[str] = "formal_input_ledger"
    _PAYLOAD_KEYS: ClassVar[set[str]] = {
        "entries",
        "entry_count",
        "preparation_audit_sha256",
        "preparation_seal_sha256",
    }


class FormalInputLedger:
    """Exclusive formal-input recipe claims and their immutable snapshot."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    @staticmethod
    def _recipe_dir_name(recipe_id: str) -> str:
        return f"recipe-{hashlib.sha256(recipe_id.encode('utf-8')).hexdigest()}"

    def claim(
        self,
        preparation_seal_sha256: str,
        preparation_audit_sha256: str,
        recipe_id: str,
    ) -> FormalInputClaim:
        _require_sha256(
            preparation_seal_sha256, "preparation_seal_sha256"
        )
        _require_sha256(
            preparation_audit_sha256, "preparation_audit_sha256"
        )
        _require_safe_id(recipe_id, "recipe_id")
        base_dir = self.root / self._recipe_dir_name(recipe_id)
        path = base_dir / "generation-000001" / "claim.json"
        payload: dict[str, object] = {
            "generation": 1,
            "preparation_audit_sha256": preparation_audit_sha256,
            "preparation_seal_sha256": preparation_seal_sha256,
            "recipe_id": recipe_id,
            "recovered_from_claim_sha256": None,
            "recovery_record_sha256": None,
        }
        with _transition_lock(base_dir):
            created = _create_record("formal_input_claim", payload, path)
        return FormalInputClaim(*created, generation=1, base_dir=base_dir)

    def recover(
        self,
        claim: FormalInputClaim,
        recovery_audit_sha256: str,
    ) -> FormalInputClaim:
        _require_sha256(recovery_audit_sha256, "recovery_audit_sha256")
        if claim.base_dir.parent != self.root:
            raise ValueError("claim belongs to another formal input ledger")
        with _transition_lock(claim.base_dir):
            if claim.recovery_path.exists():
                raise FileExistsError(claim.recovery_path)
            claim.assert_unconsumed()
            next_generation = claim.generation + 1
            recovery = _create_record(
                "formal_input_recovery",
                {
                    "claim_sha256": claim.sha256,
                    "next_generation": next_generation,
                    "recovery_audit_sha256": recovery_audit_sha256,
                },
                claim.recovery_path,
            )
            new_payload = dict(claim.payload)
            new_payload["generation"] = next_generation
            new_payload["recovered_from_claim_sha256"] = claim.sha256
            new_payload["recovery_record_sha256"] = recovery[3]
            new_path = (
                claim.base_dir
                / f"generation-{next_generation:06d}"
                / "claim.json"
            )
            created = _create_record(
                "formal_input_claim",
                new_payload,
                new_path,
            )
        return FormalInputClaim(
            *created,
            generation=next_generation,
            base_dir=claim.base_dir,
        )

    def _current_claim(self, recipe_dir: Path) -> FormalInputClaim:
        generations = sorted(recipe_dir.glob("generation-*"))
        if not generations:
            raise ArtifactStateError(f"formal input recipe has no claim: {recipe_dir}")
        generation_dir = generations[-1]
        try:
            generation = int(generation_dir.name.removeprefix("generation-"))
        except ValueError as exc:
            raise ArtifactIntegrityError("invalid formal input generation") from exc
        path = generation_dir / "claim.json"
        payload, payload_hash, artifact_hash = _read_record(
            path,
            "formal_input_claim",
            expected_payload_keys={
                "generation",
                "preparation_audit_sha256",
                "preparation_seal_sha256",
                "recipe_id",
                "recovered_from_claim_sha256",
                "recovery_record_sha256",
            },
        )
        if payload["generation"] != generation:
            raise ArtifactIntegrityError("formal input generation mismatch")
        return FormalInputClaim(
            path,
            payload,
            payload_hash,
            artifact_hash,
            generation=generation,
            base_dir=recipe_dir,
        )

    def finalize(self, output_path: Path) -> FormalInputLedgerSnapshot:
        recipe_dirs = sorted(
            path
            for path in self.root.glob("recipe-*")
            if path.is_dir()
        ) if self.root.exists() else []
        if not recipe_dirs:
            raise ArtifactStateError("formal input ledger must be nonempty")
        entries: list[dict[str, object]] = []
        recipe_ids: set[str] = set()
        seal_hashes: set[str] = set()
        audit_hashes: set[str] = set()
        for recipe_dir in recipe_dirs:
            claim = self._current_claim(recipe_dir)
            recipe_id = claim.payload["recipe_id"]
            if not isinstance(recipe_id, str):
                raise ArtifactIntegrityError("formal input recipe_id must be a string")
            if recipe_id in recipe_ids:
                raise ArtifactIntegrityError("duplicate formal input recipe ID")
            recipe_ids.add(recipe_id)
            if not claim.completion_path.exists():
                raise ArtifactStateError(
                    f"formal input recipe is incomplete: {recipe_id}"
                )
            completion = claim._verify_completion()
            seal_hash = claim.payload["preparation_seal_sha256"]
            audit_hash = claim.payload["preparation_audit_sha256"]
            try:
                _require_sha256(seal_hash, "preparation_seal_sha256")  # type: ignore[arg-type]
                _require_sha256(audit_hash, "preparation_audit_sha256")  # type: ignore[arg-type]
            except ValueError as exc:
                raise ArtifactIntegrityError("invalid formal input claim chain") from exc
            seal_hashes.add(seal_hash)  # type: ignore[arg-type]
            audit_hashes.add(audit_hash)  # type: ignore[arg-type]
            entries.append(
                {
                    "claim_sha256": claim.sha256,
                    "completion_payload": completion.payload,
                    "completion_sha256": completion.sha256,
                    "preparation_audit_sha256": audit_hash,
                    "preparation_seal_sha256": seal_hash,
                    "recipe_id": recipe_id,
                }
            )
        if len(seal_hashes) != 1 or len(audit_hashes) != 1:
            raise ArtifactIntegrityError(
                "formal input entries must share one preparation seal/audit"
            )
        entries.sort(key=lambda entry: str(entry["recipe_id"]))
        payload: dict[str, object] = {
            "entries": entries,
            "entry_count": len(entries),
            "preparation_audit_sha256": next(iter(audit_hashes)),
            "preparation_seal_sha256": next(iter(seal_hashes)),
        }
        return FormalInputLedgerSnapshot._from_created(
            _create_record("formal_input_ledger", payload, output_path)
        )  # type: ignore[return-value]


__all__ = [
    "ArtifactIntegrityError",
    "ArtifactStateError",
    "CellClaim",
    "ClaimCompletion",
    "ConfirmationCellLedger",
    "ConfirmationSeal",
    "FinalRunAudit",
    "FinalRunSeal",
    "FormalCellKey",
    "FormalCellLedger",
    "FormalInputClaim",
    "FormalInputLedger",
    "FormalInputLedgerSnapshot",
    "FormalPreparationAudit",
    "FormalPreparationSeal",
    "RefitArtifactLedger",
    "RefitCell",
    "expected_formal_cell_keys",
]
