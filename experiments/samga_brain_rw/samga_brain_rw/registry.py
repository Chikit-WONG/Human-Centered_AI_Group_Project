"""Append-only, hash-chained candidate decisions for development stages."""

from __future__ import annotations

import copy
import fcntl
import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from .hashing import canonical_json_bytes, sha256_json
from .statistics import (
    GateDecision,
    read_development_bytes,
    validate_development_path,
)


CANDIDATE_DECISION_TYPE = "samga_brain_rw.candidate_decision"
REGISTRY_RECORD_TYPE = "samga_brain_rw.candidate_registry_record"
REGISTRY_STATE_TYPE = "samga_brain_rw.candidate_registry_state"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)

_DECISION_KEYS = frozenset(
    {
        "absolute_top1",
        "absolute_top5",
        "candidate_id",
        "candidate_matrix_sha256",
        "component_sha256s",
        "config_sha256",
        "control_config_sha256",
        "control_id",
        "control_matrix_sha256",
        "gate",
        "hyperparameters_sha256",
        "locked",
        "schedule_sha256",
        "scope",
        "stage",
    }
)
_DECISION_DOCUMENT_KEYS = _DECISION_KEYS | frozenset(
    {"artifact_type", "schema_version"}
)
_RECORD_KEYS = frozenset(
    {
        "artifact_type",
        "decision",
        "decision_sha256",
        "previous_record_sha256",
        "previous_state_sha256",
        "record_sha256",
        "schema_version",
        "sequence",
        "state_sha256",
    }
)
_STATE_KEYS = frozenset(
    {
        "artifact_type",
        "head_record_sha256",
        "previous_state_sha256",
        "schema_version",
        "sequence",
        "stages",
        "state_sha256",
    }
)
_STAGE_STATE_KEYS = frozenset({"candidates", "confirmed", "survivor"})
_STATE_REFERENCE_KEYS = frozenset(
    {
        "candidate_id",
        "decision_sha256",
        "frozen_identity_sha256",
    }
)


class RegistryIntegrityError(ValueError):
    """The journal/state pair is malformed, noncanonical, or inconsistent."""


class RegistryStateError(RuntimeError):
    """A requested candidate-registry transition is forbidden."""


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a 64-character lowercase SHA-256")
    return value


def _require_safe_id(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or _SAFE_ID_RE.fullmatch(value) is None
        or value in {".", ".."}
    ):
        raise ValueError(f"{field} must be a safe nonempty identifier")
    return value


def _require_stage(value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= 5
    ):
        raise ValueError("stage must be an integer in 1..5")
    return value


def _require_rate(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be a finite number in [0, 1]")
    return result


def _require_exact_keys(
    value: Mapping[str, object],
    expected: frozenset[str],
    context: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RegistryIntegrityError(
            f"{context} keys mismatch: missing={missing}, extra={extra}"
        )


@dataclass(frozen=True)
class CandidateDecision:
    """One pilot/confirmation evaluation or one locked-survivor event."""

    stage: int
    candidate_id: str
    control_id: str
    scope: Literal["val-dev", "val-confirm"]
    config_sha256: str
    control_config_sha256: str
    hyperparameters_sha256: str
    schedule_sha256: str
    component_sha256s: tuple[str, ...]
    candidate_matrix_sha256: str
    control_matrix_sha256: str
    absolute_top1: float
    absolute_top5: float
    gate: GateDecision
    locked: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", _require_stage(self.stage))
        candidate_id = _require_safe_id(self.candidate_id, "candidate_id")
        control_id = _require_safe_id(self.control_id, "control_id")
        if candidate_id == control_id:
            raise ValueError("candidate_id and control_id must be distinct")
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "control_id", control_id)
        if self.scope not in {"val-dev", "val-confirm"}:
            raise ValueError("scope must be val-dev or val-confirm")
        for field in (
            "config_sha256",
            "control_config_sha256",
            "hyperparameters_sha256",
            "schedule_sha256",
            "candidate_matrix_sha256",
            "control_matrix_sha256",
        ):
            object.__setattr__(
                self,
                field,
                _require_sha256(getattr(self, field), field),
            )
        if not isinstance(self.component_sha256s, tuple):
            object.__setattr__(
                self,
                "component_sha256s",
                tuple(self.component_sha256s),
            )
        components = tuple(
            _require_sha256(value, f"component_sha256s[{index}]")
            for index, value in enumerate(self.component_sha256s)
        )
        if not components:
            raise ValueError("component_sha256s must be nonempty")
        if len(set(components)) != len(components):
            raise ValueError("component_sha256s contains duplicates")
        object.__setattr__(self, "component_sha256s", tuple(sorted(components)))
        object.__setattr__(
            self,
            "absolute_top1",
            _require_rate(self.absolute_top1, "absolute_top1"),
        )
        object.__setattr__(
            self,
            "absolute_top5",
            _require_rate(self.absolute_top5, "absolute_top5"),
        )
        if self.absolute_top5 < self.absolute_top1:
            raise ValueError("absolute_top5 must be >= absolute_top1")
        if not isinstance(self.gate, GateDecision):
            raise ValueError("gate must be a GateDecision")
        expected_kind = "pilot" if self.scope == "val-dev" else "confirmation"
        if self.gate.gate_kind != expected_kind:
            raise ValueError(
                f"{self.scope} requires a {expected_kind} gate decision"
            )
        if self.gate.stage not in {None, self.stage}:
            raise ValueError("gate stage does not match candidate stage")
        if not isinstance(self.locked, bool):
            raise ValueError("locked must be boolean")
        if self.locked and (self.scope != "val-dev" or not self.gate.passed):
            raise ValueError("only a passing val-dev decision can be locked")

    @property
    def decision_sha256(self) -> str:
        return sha256_json(self.to_payload())

    def frozen_identity_payload(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "component_sha256s": list(self.component_sha256s),
            "config_sha256": self.config_sha256,
            "control_config_sha256": self.control_config_sha256,
            "control_id": self.control_id,
            "hyperparameters_sha256": self.hyperparameters_sha256,
            "schedule_sha256": self.schedule_sha256,
            "stage": self.stage,
        }

    @property
    def frozen_identity_sha256(self) -> str:
        return sha256_json(self.frozen_identity_payload())

    def to_payload(self) -> dict[str, object]:
        return {
            "absolute_top1": self.absolute_top1,
            "absolute_top5": self.absolute_top5,
            "candidate_id": self.candidate_id,
            "candidate_matrix_sha256": self.candidate_matrix_sha256,
            "component_sha256s": list(self.component_sha256s),
            "config_sha256": self.config_sha256,
            "control_config_sha256": self.control_config_sha256,
            "control_id": self.control_id,
            "control_matrix_sha256": self.control_matrix_sha256,
            "gate": self.gate.to_payload(),
            "hyperparameters_sha256": self.hyperparameters_sha256,
            "locked": self.locked,
            "schedule_sha256": self.schedule_sha256,
            "scope": self.scope,
            "stage": self.stage,
        }

    def to_document(self) -> dict[str, object]:
        return {
            "artifact_type": CANDIDATE_DECISION_TYPE,
            **self.to_payload(),
            "schema_version": 1,
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
    ) -> "CandidateDecision":
        _require_exact_keys(payload, _DECISION_KEYS, "candidate decision")
        raw_gate = payload["gate"]
        if not isinstance(raw_gate, Mapping):
            raise RegistryIntegrityError("candidate decision gate must be an object")
        raw_components = payload["component_sha256s"]
        if (
            not isinstance(raw_components, Sequence)
            or isinstance(raw_components, (str, bytes, bytearray))
        ):
            raise RegistryIntegrityError(
                "candidate decision component_sha256s must be a sequence"
            )
        try:
            return cls(
                stage=payload["stage"],  # type: ignore[arg-type]
                candidate_id=payload["candidate_id"],  # type: ignore[arg-type]
                control_id=payload["control_id"],  # type: ignore[arg-type]
                scope=payload["scope"],  # type: ignore[arg-type]
                config_sha256=payload["config_sha256"],  # type: ignore[arg-type]
                control_config_sha256=payload[
                    "control_config_sha256"
                ],  # type: ignore[arg-type]
                hyperparameters_sha256=payload[
                    "hyperparameters_sha256"
                ],  # type: ignore[arg-type]
                schedule_sha256=payload["schedule_sha256"],  # type: ignore[arg-type]
                component_sha256s=tuple(raw_components),  # type: ignore[arg-type]
                candidate_matrix_sha256=payload[
                    "candidate_matrix_sha256"
                ],  # type: ignore[arg-type]
                control_matrix_sha256=payload[
                    "control_matrix_sha256"
                ],  # type: ignore[arg-type]
                absolute_top1=payload["absolute_top1"],  # type: ignore[arg-type]
                absolute_top5=payload["absolute_top5"],  # type: ignore[arg-type]
                gate=GateDecision.from_payload(raw_gate),
                locked=payload["locked"],  # type: ignore[arg-type]
            )
        except RegistryIntegrityError:
            raise
        except (TypeError, ValueError) as exc:
            raise RegistryIntegrityError("invalid candidate decision") from exc

    @classmethod
    def from_document(
        cls,
        document: Mapping[str, object],
    ) -> "CandidateDecision":
        _require_exact_keys(
            document,
            _DECISION_DOCUMENT_KEYS,
            "candidate decision document",
        )
        if (
            type(document["schema_version"]) is not int
            or document["schema_version"] != 1
        ):
            raise RegistryIntegrityError(
                "candidate decision schema_version must be integer 1"
            )
        if document["artifact_type"] != CANDIDATE_DECISION_TYPE:
            raise RegistryIntegrityError(
                f"candidate decision artifact_type must be {CANDIDATE_DECISION_TYPE}"
            )
        return cls.from_payload(
            {
                key: value
                for key, value in document.items()
                if key not in {"artifact_type", "schema_version"}
            }
        )


def _blank_state() -> dict[str, object]:
    return {
        "artifact_type": REGISTRY_STATE_TYPE,
        "head_record_sha256": None,
        "previous_state_sha256": None,
        "schema_version": 1,
        "sequence": 0,
        "stages": {},
    }


def _state_with_hash(state_body: Mapping[str, object]) -> dict[str, object]:
    document = dict(state_body)
    document["state_sha256"] = sha256_json(state_body)
    return document


def _require_state_sha256(value: object, field: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise RegistryIntegrityError(
            f"{field} must be a 64-character lowercase SHA-256"
        )
    return value


def _require_state_candidate_id(value: object, field: str) -> str:
    try:
        return _require_safe_id(value, field)
    except ValueError as exc:
        raise RegistryIntegrityError(f"{field} is invalid") from exc


def _validate_state_reference(
    value: object,
    context: str,
) -> dict[str, object] | None:
    if value is None:
        return None
    if type(value) is not dict:
        raise RegistryIntegrityError(f"{context} must be an object or null")
    _require_exact_keys(value, _STATE_REFERENCE_KEYS, context)
    _require_state_candidate_id(
        value["candidate_id"],
        f"{context} candidate_id",
    )
    _require_state_sha256(
        value["decision_sha256"],
        f"{context} decision_sha256",
    )
    _require_state_sha256(
        value["frozen_identity_sha256"],
        f"{context} frozen_identity_sha256",
    )
    return value


def _validate_stored_state_document(state: dict[str, object]) -> int:
    _require_exact_keys(state, _STATE_KEYS, "registry state")
    if (
        type(state["artifact_type"]) is not str
        or state["artifact_type"] != REGISTRY_STATE_TYPE
    ):
        raise RegistryIntegrityError(
            f"registry state artifact_type must be {REGISTRY_STATE_TYPE}"
        )
    if (
        type(state["schema_version"]) is not int
        or state["schema_version"] != 1
    ):
        raise RegistryIntegrityError(
            "registry state schema_version must be integer 1"
        )
    sequence = state["sequence"]
    if type(sequence) is not int or sequence < 0:
        raise RegistryIntegrityError(
            "compact registry state sequence is invalid"
        )

    state_sha256 = _require_state_sha256(
        state["state_sha256"],
        "registry state state_sha256",
    )
    state_body = {
        key: value
        for key, value in state.items()
        if key != "state_sha256"
    }
    if sha256_json(state_body) != state_sha256:
        raise RegistryIntegrityError("registry state SHA-256 mismatch")

    head_record = state["head_record_sha256"]
    previous_state = state["previous_state_sha256"]
    if sequence == 0:
        if head_record is not None or previous_state is not None:
            raise RegistryIntegrityError(
                "empty registry state must have null hash-chain heads"
            )
    else:
        _require_state_sha256(
            head_record,
            "registry state head_record_sha256",
        )
        if sequence == 1:
            if previous_state is not None:
                raise RegistryIntegrityError(
                    "first registry state previous_state_sha256 must be null"
                )
        else:
            _require_state_sha256(
                previous_state,
                "registry state previous_state_sha256",
            )

    stages = state["stages"]
    if type(stages) is not dict:
        raise RegistryIntegrityError("registry state stages must be an object")
    if (sequence == 0) != (not stages):
        raise RegistryIntegrityError(
            "registry state sequence/stages emptiness mismatch"
        )
    for stage_key, raw_stage in stages.items():
        if (
            type(stage_key) is not str
            or stage_key not in {"1", "2", "3", "4", "5"}
        ):
            raise RegistryIntegrityError(
                "registry state stage key must be one of 1..5"
            )
        if type(raw_stage) is not dict:
            raise RegistryIntegrityError(
                f"registry state stage {stage_key} must be an object"
            )
        _require_exact_keys(
            raw_stage,
            _STAGE_STATE_KEYS,
            f"registry state stage {stage_key}",
        )
        candidates = raw_stage["candidates"]
        if type(candidates) is not dict or not candidates:
            raise RegistryIntegrityError(
                f"registry state stage {stage_key} candidates must be "
                "a nonempty object"
            )
        for candidate_id, raw_scopes in candidates.items():
            candidate_id = _require_state_candidate_id(
                candidate_id,
                f"registry state stage {stage_key} candidate_id",
            )
            if type(raw_scopes) is not dict:
                raise RegistryIntegrityError(
                    f"registry state candidate {candidate_id} scopes "
                    "must be an object"
                )
            if (
                "val-dev" not in raw_scopes
                or not set(raw_scopes) <= {"val-dev", "val-confirm"}
            ):
                raise RegistryIntegrityError(
                    f"registry state candidate {candidate_id} scopes are invalid"
                )
            for scope, decision_sha256 in raw_scopes.items():
                _require_state_sha256(
                    decision_sha256,
                    f"registry state candidate {candidate_id} {scope}",
                )

        survivor = _validate_state_reference(
            raw_stage["survivor"],
            f"registry state stage {stage_key} survivor",
        )
        confirmed = _validate_state_reference(
            raw_stage["confirmed"],
            f"registry state stage {stage_key} confirmed",
        )
        if survivor is not None:
            survivor_id = survivor["candidate_id"]
            if survivor_id not in candidates:
                raise RegistryIntegrityError(
                    "registry state survivor is absent from candidates"
                )
        if confirmed is not None:
            confirmed_id = confirmed["candidate_id"]
            if (
                survivor is None
                or confirmed_id != survivor["candidate_id"]
                or confirmed["frozen_identity_sha256"]
                != survivor["frozen_identity_sha256"]
            ):
                raise RegistryIntegrityError(
                    "registry state confirmed/survivor identity mismatch"
                )
            confirmed_scopes = candidates[confirmed_id]
            if (
                type(confirmed_scopes) is not dict
                or confirmed_scopes.get("val-confirm")
                != confirmed["decision_sha256"]
            ):
                raise RegistryIntegrityError(
                    "registry state confirmed decision mismatch"
                )
        elif any(
            "val-confirm" in scopes
            for scopes in candidates.values()
            if type(scopes) is dict
        ):
            raise RegistryIntegrityError(
                "registry state val-confirm lacks confirmed state"
            )
    return sequence


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise RegistryIntegrityError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_nonfinite_json(value: str) -> object:
    raise RegistryIntegrityError(f"non-finite JSON value is forbidden: {value}")


def _decode_object(data: bytes, context: str) -> dict[str, object]:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite_json,
        )
    except RegistryIntegrityError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RegistryIntegrityError(f"invalid UTF-8 JSON {context}") from exc
    if not isinstance(value, dict):
        raise RegistryIntegrityError(f"{context} root must be an object")
    return value


def _open_directory_nofollow(path: Path, *, create: bool) -> int:
    absolute = Path(os.path.abspath(os.fspath(path)))
    flags = os.O_RDONLY | _O_DIRECTORY | _O_CLOEXEC | _O_NOFOLLOW
    descriptor = os.open(absolute.anchor, flags)
    try:
        for component in absolute.parts[1:]:
            try:
                next_descriptor = os.open(
                    component,
                    flags,
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(component, mode=0o700, dir_fd=descriptor)
                os.fsync(descriptor)
                next_descriptor = os.open(
                    component,
                    flags,
                    dir_fd=descriptor,
                )
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _path_exists_nofollow(path: Path) -> bool:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(metadata.st_mode):
        raise RegistryIntegrityError(
            f"registry path is not a regular file: {path}"
        )
    return True


def _read_registry_file(
    path: Path,
    *,
    suffix: str,
    maximum_bytes: int = 64 * 1024 * 1024,
) -> bytes | None:
    if not _path_exists_nofollow(path):
        return None
    try:
        return read_development_bytes(
            path,
            allowed_suffixes=frozenset({suffix}),
            maximum_bytes=maximum_bytes,
        )
    except (OSError, ValueError) as exc:
        raise RegistryIntegrityError(
            f"cannot safely read registry file: {path}"
        ) from exc


def _frozen_identity_difference(
    expected: CandidateDecision,
    actual: CandidateDecision,
) -> str | None:
    comparisons = (
        ("candidate", expected.candidate_id, actual.candidate_id),
        ("control", expected.control_id, actual.control_id),
        ("config", expected.config_sha256, actual.config_sha256),
        (
            "control config",
            expected.control_config_sha256,
            actual.control_config_sha256,
        ),
        (
            "hyperparameter",
            expected.hyperparameters_sha256,
            actual.hyperparameters_sha256,
        ),
        ("schedule", expected.schedule_sha256, actual.schedule_sha256),
        ("component", expected.component_sha256s, actual.component_sha256s),
    )
    for label, left, right in comparisons:
        if left != right:
            return label
    return None


def _apply_decision(
    prior_state: Mapping[str, object],
    decision: CandidateDecision,
    *,
    decision_sha256: str,
    record_sha256: str,
    decisions_by_sha256: Mapping[str, CandidateDecision],
) -> dict[str, object]:
    state = copy.deepcopy(dict(prior_state))
    stages = state["stages"]
    if not isinstance(stages, dict):
        raise RegistryIntegrityError("registry state stages must be an object")
    stage_key = str(decision.stage)
    stage = stages.setdefault(
        stage_key,
        {
            "candidates": {},
            "confirmed": None,
            "survivor": None,
        },
    )
    if not isinstance(stage, dict):
        raise RegistryIntegrityError("registry stage state must be an object")
    if set(stage) != {"candidates", "confirmed", "survivor"}:
        raise RegistryIntegrityError("registry stage state keys mismatch")
    candidates = stage["candidates"]
    if not isinstance(candidates, dict):
        raise RegistryIntegrityError("registry candidates state must be an object")
    survivor = stage["survivor"]
    confirmed = stage["confirmed"]

    if confirmed is not None:
        raise RegistryStateError(
            f"stage {decision.stage} is frozen after val-confirm"
        )

    if decision.locked:
        if survivor is not None:
            raise RegistryStateError(
                f"stage {decision.stage} survivor is already locked"
            )
        candidate_scopes = candidates.get(decision.candidate_id)
        if not isinstance(candidate_scopes, dict):
            raise RegistryStateError(
                "locked survivor lacks a prior val-dev decision"
            )
        development_sha = candidate_scopes.get("val-dev")
        if not isinstance(development_sha, str):
            raise RegistryStateError(
                "locked survivor lacks a prior val-dev decision"
            )
        development = decisions_by_sha256.get(development_sha)
        if development is None:
            raise RegistryIntegrityError(
                "registry state references an unknown val-dev decision"
            )
        if not development.gate.passed:
            raise RegistryStateError("failed candidate cannot be locked")
        difference = _frozen_identity_difference(development, decision)
        if difference is not None:
            raise RegistryStateError(
                f"locked survivor {difference} differs from val-dev"
            )
        stage["survivor"] = {
            "candidate_id": decision.candidate_id,
            "decision_sha256": decision_sha256,
            "frozen_identity_sha256": decision.frozen_identity_sha256,
        }
    elif decision.scope == "val-dev":
        if survivor is not None:
            raise RegistryStateError(
                f"stage {decision.stage} survivor is already locked"
            )
        candidate_scopes = candidates.setdefault(decision.candidate_id, {})
        if not isinstance(candidate_scopes, dict):
            raise RegistryIntegrityError("candidate state must be an object")
        if "val-dev" in candidate_scopes:
            raise RegistryStateError(
                f"duplicate val-dev decision for {decision.candidate_id}"
            )
        candidate_scopes["val-dev"] = decision_sha256
    else:
        if not isinstance(survivor, dict):
            raise RegistryStateError(
                "val-confirm requires a locked stage survivor"
            )
        if survivor.get("candidate_id") != decision.candidate_id:
            raise RegistryStateError(
                "val-confirm candidate is not the locked survivor"
            )
        survivor_sha = survivor.get("decision_sha256")
        if not isinstance(survivor_sha, str):
            raise RegistryIntegrityError(
                "survivor state lacks a decision SHA-256"
            )
        locked = decisions_by_sha256.get(survivor_sha)
        if locked is None:
            raise RegistryIntegrityError(
                "survivor state references an unknown decision"
            )
        difference = _frozen_identity_difference(locked, decision)
        if difference is not None:
            raise RegistryStateError(
                f"val-confirm {difference} change is forbidden"
            )
        candidate_scopes = candidates.get(decision.candidate_id)
        if not isinstance(candidate_scopes, dict):
            raise RegistryIntegrityError(
                "survivor lacks candidate development state"
            )
        if "val-confirm" in candidate_scopes:
            raise RegistryStateError(
                f"duplicate val-confirm decision for {decision.candidate_id}"
            )
        candidate_scopes["val-confirm"] = decision_sha256
        stage["confirmed"] = {
            "candidate_id": decision.candidate_id,
            "decision_sha256": decision_sha256,
            "frozen_identity_sha256": decision.frozen_identity_sha256,
        }

    sequence = prior_state["sequence"]
    if isinstance(sequence, bool) or not isinstance(sequence, int):
        raise RegistryIntegrityError("registry state sequence must be an integer")
    state["previous_state_sha256"] = (
        None if sequence == 0 else sha256_json(prior_state)
    )
    state["head_record_sha256"] = record_sha256
    state["sequence"] = sequence + 1
    return state


class CandidateRegistry:
    """Concurrent-safe journal plus an atomically replaced compact state."""

    def __init__(
        self,
        journal_path: Path,
        state_path: Path | None = None,
    ) -> None:
        journal = validate_development_path(
            Path(journal_path),
            allowed_suffixes=frozenset({".jsonl"}),
        )
        state = validate_development_path(
            (
                Path(state_path)
                if state_path is not None
                else journal.with_name(f"{journal.stem}.state.json")
            ),
            allowed_suffixes=frozenset({".json"}),
        )
        if journal == state:
            raise ValueError("registry journal and state paths must differ")
        if journal.parent != state.parent:
            raise ValueError(
                "registry journal and compact state must share one directory"
            )
        self.journal_path = journal
        self.state_path = state
        self._lock_path = state.with_name(f".{state.name}.lock")

    @contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        parent_fd = _open_directory_nofollow(
            self.journal_path.parent,
            create=True,
        )
        descriptor = -1
        try:
            descriptor = os.open(
                self._lock_path.name,
                os.O_RDWR
                | os.O_CREAT
                | _O_CLOEXEC
                | _O_NOFOLLOW,
                0o600,
                dir_fd=parent_fd,
            )
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise RegistryIntegrityError("registry lock is not a regular file")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            if descriptor >= 0:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                finally:
                    os.close(descriptor)
            os.close(parent_fd)

    def _load_verified_unlocked(
        self,
    ) -> tuple[dict[str, object], list[CandidateDecision]]:
        journal_bytes = _read_registry_file(
            self.journal_path,
            suffix=".jsonl",
        )
        state_bytes = _read_registry_file(
            self.state_path,
            suffix=".json",
        )
        if journal_bytes is None and state_bytes is None:
            return _blank_state(), []
        if journal_bytes is None:
            raise RegistryIntegrityError(
                "registry journal/state existence mismatch"
            )
        if journal_bytes and not journal_bytes.endswith(b"\n"):
            raise RegistryIntegrityError(
                "append-only registry journal lacks a final newline"
            )

        # The journal is the WAL: its fsync precedes compact-state publication.
        # Record a possible stored prefix, but do not recover it until the
        # complete journal and every hash/transition have been validated.
        stored_state: dict[str, object] | None = None
        stored_sequence: int | None = None
        if state_bytes is not None:
            stored_state = _decode_object(
                state_bytes.rstrip(b"\n"),
                "registry state",
            )
            if state_bytes != canonical_json_bytes(stored_state) + b"\n":
                raise RegistryIntegrityError(
                    "compact registry state is not canonical JSON"
                )
            stored_sequence = _validate_stored_state_document(stored_state)

        state: dict[str, object] = _blank_state()
        stored_prefix_state = (
            _state_with_hash(state)
            if stored_sequence == 0
            else None
        )
        decisions: list[CandidateDecision] = []
        decisions_by_sha256: dict[str, CandidateDecision] = {}
        expected_record_sha256: str | None = None
        expected_state_sha256: str | None = None
        for expected_sequence, line in enumerate(
            journal_bytes.splitlines(),
            start=1,
        ):
            if not line:
                raise RegistryIntegrityError(
                    "append-only registry journal contains a blank line"
                )
            record = _decode_object(line, "registry record")
            if canonical_json_bytes(record) != line:
                raise RegistryIntegrityError(
                    "registry record is not canonical compact JSON"
                )
            _require_exact_keys(record, _RECORD_KEYS, "registry record")
            if (
                type(record["schema_version"]) is not int
                or record["schema_version"] != 1
            ):
                raise RegistryIntegrityError(
                    "registry record schema_version must be integer 1"
                )
            if (
                type(record["artifact_type"]) is not str
                or record["artifact_type"] != REGISTRY_RECORD_TYPE
            ):
                raise RegistryIntegrityError(
                    f"registry record artifact_type must be {REGISTRY_RECORD_TYPE}"
                )
            if (
                type(record["sequence"]) is not int
                or record["sequence"] != expected_sequence
            ):
                raise RegistryIntegrityError(
                    "registry record sequence must be a contiguous integer"
                )
            if record["previous_record_sha256"] != expected_record_sha256:
                raise RegistryIntegrityError(
                    "registry previous-record hash chain mismatch"
                )
            if record["previous_state_sha256"] != expected_state_sha256:
                raise RegistryIntegrityError(
                    "registry previous-state hash chain mismatch"
                )
            raw_decision = record["decision"]
            if not isinstance(raw_decision, Mapping):
                raise RegistryIntegrityError(
                    "registry decision payload must be an object"
                )
            decision = CandidateDecision.from_payload(raw_decision)
            decision_sha256 = sha256_json(raw_decision)
            if record["decision_sha256"] != decision_sha256:
                raise RegistryIntegrityError(
                    "registry decision SHA-256 mismatch"
                )
            record_body = {
                key: value
                for key, value in record.items()
                if key not in {"record_sha256", "state_sha256"}
            }
            record_sha256 = sha256_json(record_body)
            if record["record_sha256"] != record_sha256:
                raise RegistryIntegrityError(
                    "registry record SHA-256 mismatch"
                )
            next_state_body = _apply_decision(
                state,
                decision,
                decision_sha256=decision_sha256,
                record_sha256=record_sha256,
                decisions_by_sha256=decisions_by_sha256,
            )
            next_state = _state_with_hash(next_state_body)
            if record["state_sha256"] != next_state["state_sha256"]:
                raise RegistryIntegrityError(
                    "registry state SHA-256 chain mismatch"
                )
            if stored_sequence == expected_sequence:
                stored_prefix_state = next_state
            state = next_state_body
            expected_record_sha256 = record_sha256
            expected_state_sha256 = next_state["state_sha256"]  # type: ignore[assignment]
            decisions.append(decision)
            decisions_by_sha256[decision_sha256] = decision

        expected_state = _state_with_hash(state)
        expected_state_bytes = canonical_json_bytes(expected_state)
        stored_state_bytes = (
            canonical_json_bytes(stored_state)
            if stored_state is not None
            else None
        )
        if stored_state is None:
            self._publish_state_unlocked(expected_state)
        elif stored_state_bytes == expected_state_bytes:
            pass
        elif (
            stored_prefix_state is not None
            and stored_state_bytes
            == canonical_json_bytes(stored_prefix_state)
        ):
            self._publish_state_unlocked(expected_state)
        else:
            raise RegistryIntegrityError(
                "compact registry state differs from the journal reduction"
            )
        return state, decisions

    def verify(self) -> None:
        with self._exclusive_lock():
            self._load_verified_unlocked()

    def load_state(self) -> dict[str, object]:
        with self._exclusive_lock():
            state, _ = self._load_verified_unlocked()
            return _state_with_hash(copy.deepcopy(state))

    def _build_record_and_state(
        self,
        state: Mapping[str, object],
        decisions: Sequence[CandidateDecision],
        decision: CandidateDecision,
    ) -> tuple[dict[str, object], dict[str, object]]:
        if decision.locked and not any(
            item.stage == decision.stage
            and item.candidate_id == decision.candidate_id
            and item.scope == "val-dev"
            and not item.locked
            for item in decisions
        ):
            raise RegistryStateError(
                "locked survivor lacks a prior val-dev decision"
            )
        decision_payload = decision.to_payload()
        decision_sha256 = sha256_json(decision_payload)
        sequence = state["sequence"]
        if isinstance(sequence, bool) or not isinstance(sequence, int):
            raise RegistryIntegrityError("registry state sequence is invalid")
        previous_record = state["head_record_sha256"]
        previous_state = (
            None if sequence == 0 else sha256_json(state)
        )
        record_body = {
            "artifact_type": REGISTRY_RECORD_TYPE,
            "decision": decision_payload,
            "decision_sha256": decision_sha256,
            "previous_record_sha256": previous_record,
            "previous_state_sha256": previous_state,
            "schema_version": 1,
            "sequence": sequence + 1,
        }
        record_sha256 = sha256_json(record_body)
        decisions_by_sha256 = {
            item.decision_sha256: item for item in decisions
        }
        next_state_body = _apply_decision(
            state,
            decision,
            decision_sha256=decision_sha256,
            record_sha256=record_sha256,
            decisions_by_sha256=decisions_by_sha256,
        )
        next_state = _state_with_hash(next_state_body)
        record = {
            **record_body,
            "record_sha256": record_sha256,
            "state_sha256": next_state["state_sha256"],
        }
        return record, next_state

    def _append_journal_unlocked(self, record: Mapping[str, object]) -> None:
        parent_fd = _open_directory_nofollow(
            self.journal_path.parent,
            create=True,
        )
        descriptor = -1
        try:
            exists = _path_exists_nofollow(self.journal_path)
            flags = os.O_WRONLY | _O_CLOEXEC | _O_NOFOLLOW
            flags |= os.O_APPEND if exists else os.O_CREAT | os.O_EXCL
            descriptor = os.open(
                self.journal_path.name,
                flags,
                0o600,
                dir_fd=parent_fd,
            )
            data = canonical_json_bytes(record) + b"\n"
            written = 0
            while written < len(data):
                count = os.write(descriptor, data[written:])
                if count <= 0:
                    raise OSError("short append to candidate registry")
                written += count
            os.fsync(descriptor)
            os.fsync(parent_fd)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            os.close(parent_fd)

    def _publish_state_unlocked(self, state: Mapping[str, object]) -> None:
        parent_fd = _open_directory_nofollow(
            self.state_path.parent,
            create=True,
        )
        descriptor = -1
        temporary_name: str | None = None
        try:
            descriptor, temporary_path = tempfile.mkstemp(
                prefix=f".{self.state_path.name}.tmp-",
                dir=self.state_path.parent,
            )
            temporary_name = Path(temporary_path).name
            data = canonical_json_bytes(state) + b"\n"
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                descriptor = -1
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            if _path_exists_nofollow(self.state_path):
                os.replace(
                    temporary_name,
                    self.state_path.name,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                )
                temporary_name = None
            else:
                os.link(
                    temporary_name,
                    self.state_path.name,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                    follow_symlinks=False,
                )
            os.fsync(parent_fd)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name, dir_fd=parent_fd)
                except FileNotFoundError:
                    pass
                try:
                    os.fsync(parent_fd)
                except OSError:
                    pass
            os.close(parent_fd)

    def _commit_unlocked(
        self,
        state: Mapping[str, object],
        decisions: Sequence[CandidateDecision],
        decision: CandidateDecision,
    ) -> None:
        record, next_state = self._build_record_and_state(
            state,
            decisions,
            decision,
        )
        self._append_journal_unlocked(record)
        self._publish_state_unlocked(next_state)

    def append(self, decision: CandidateDecision) -> None:
        if not isinstance(decision, CandidateDecision):
            raise ValueError("decision must be a CandidateDecision")
        if decision.locked:
            raise RegistryStateError(
                "locked survivor records may only be created by the registry"
            )
        with self._exclusive_lock():
            state, decisions = self._load_verified_unlocked()
            self._commit_unlocked(state, decisions, decision)

    def append_or_reuse_exact(
        self,
        decision: CandidateDecision,
    ) -> CandidateDecision:
        if not isinstance(decision, CandidateDecision):
            raise ValueError("decision must be a CandidateDecision")
        if decision.locked:
            raise RegistryStateError(
                "locked survivor records may only be created by the registry"
            )
        with self._exclusive_lock():
            state, decisions = self._load_verified_unlocked()
            for existing in decisions:
                if (
                    existing.stage == decision.stage
                    and existing.candidate_id == decision.candidate_id
                    and existing.scope == decision.scope
                    and existing.locked == decision.locked
                ):
                    if (
                        canonical_json_bytes(existing.to_payload())
                        == canonical_json_bytes(decision.to_payload())
                        and existing.frozen_identity_sha256
                        == decision.frozen_identity_sha256
                    ):
                        return existing
                    raise RegistryStateError(
                        f"divergent duplicate {decision.scope} decision "
                        f"for {decision.candidate_id}"
                    )
            self._commit_unlocked(state, decisions, decision)
            return decision

    def lock_stage_survivor(self, stage: int) -> CandidateDecision:
        stage = _require_stage(stage)
        with self._exclusive_lock():
            state, decisions = self._load_verified_unlocked()
            stages = state["stages"]
            if not isinstance(stages, dict):
                raise RegistryIntegrityError("registry stages state is invalid")
            stage_state = stages.get(str(stage))
            if isinstance(stage_state, dict) and stage_state.get("survivor") is not None:
                raise RegistryStateError(
                    f"stage {stage} survivor is already locked"
                )
            eligible = [
                decision
                for decision in decisions
                if (
                    decision.stage == stage
                    and decision.scope == "val-dev"
                    and not decision.locked
                    and decision.gate.passed
                )
            ]
            if not eligible:
                raise RegistryStateError(
                    f"stage {stage} has no passing val-dev candidate"
                )
            if len(eligible) != 1:
                raise RegistryStateError(
                    f"stage {stage} has multiple passing val-dev candidates; "
                    "a stage-specific selector must preselect exactly one"
                )
            chosen = eligible[0]
            locked = replace(chosen, locked=True)
            self._commit_unlocked(state, decisions, locked)
            return locked

    def lock_stage_survivor_or_reuse_exact(
        self,
        stage: int,
        expected_development_decision_sha256: str,
    ) -> CandidateDecision:
        stage = _require_stage(stage)
        expected_sha256 = _require_sha256(
            expected_development_decision_sha256,
            "expected_development_decision_sha256",
        )
        with self._exclusive_lock():
            state, decisions = self._load_verified_unlocked()
            stages = state["stages"]
            if not isinstance(stages, dict):
                raise RegistryIntegrityError("registry stages state is invalid")
            stage_state = stages.get(str(stage))
            survivor = (
                stage_state.get("survivor")
                if isinstance(stage_state, dict)
                else None
            )

            if survivor is not None:
                development = next(
                    (
                        decision
                        for decision in decisions
                        if (
                            decision.decision_sha256 == expected_sha256
                            and decision.stage == stage
                            and decision.scope == "val-dev"
                            and not decision.locked
                            and decision.gate.passed
                        )
                    ),
                    None,
                )
                if development is None:
                    raise RegistryStateError(
                        f"stage {stage} does not contain the expected passing "
                        "val-dev decision"
                    )
                if not isinstance(survivor, dict):
                    raise RegistryIntegrityError(
                        "registry survivor state is invalid"
                    )
                survivor_sha256 = survivor.get("decision_sha256")
                locked = next(
                    (
                        decision
                        for decision in decisions
                        if decision.decision_sha256 == survivor_sha256
                    ),
                    None,
                )
                expected_locked = replace(development, locked=True)
                if (
                    locked is None
                    or survivor.get("candidate_id")
                    != development.candidate_id
                    or survivor.get("frozen_identity_sha256")
                    != expected_locked.frozen_identity_sha256
                    or canonical_json_bytes(locked.to_payload())
                    != canonical_json_bytes(expected_locked.to_payload())
                ):
                    raise RegistryStateError(
                        f"stage {stage} survivor is divergent from the "
                        "expected val-dev decision"
                    )
                return locked

            eligible = [
                decision
                for decision in decisions
                if (
                    decision.stage == stage
                    and decision.scope == "val-dev"
                    and not decision.locked
                    and decision.gate.passed
                )
            ]
            if not eligible:
                raise RegistryStateError(
                    f"stage {stage} has no passing val-dev candidate"
                )
            if len(eligible) != 1:
                raise RegistryStateError(
                    f"stage {stage} has multiple passing val-dev candidates; "
                    "a stage-specific selector must preselect exactly one"
                )
            chosen = eligible[0]
            if chosen.decision_sha256 != expected_sha256:
                raise RegistryStateError(
                    f"stage {stage} passing val-dev decision does not match "
                    "the expected decision SHA-256"
                )
            locked = replace(chosen, locked=True)
            self._commit_unlocked(state, decisions, locked)
            return locked


__all__ = [
    "CANDIDATE_DECISION_TYPE",
    "CandidateDecision",
    "CandidateRegistry",
    "REGISTRY_RECORD_TYPE",
    "REGISTRY_STATE_TYPE",
    "RegistryIntegrityError",
    "RegistryStateError",
]
