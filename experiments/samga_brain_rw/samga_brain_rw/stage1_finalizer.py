"""Transactional finalization of the fixed Stage 1 development pilot."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import io
import os
import re
import secrets
import stat
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from .component_proofs import load_stage1_composition_cells
from .config import SemanticConfig
from .cost_capability import load_validated_stage1_cost_capability
from .hashing import canonical_json_bytes, sha256_json
from .registry import (
    REGISTRY_STATE_TYPE,
    CandidateDecision,
    CandidateRegistry,
)
from .stage1 import (
    PILOT_COORDINATES,
    Stage1CompositionCell,
    Stage1CompositionOutcome,
    compose_stage1,
)
from .statistics import (
    _open_directory_nofollow,
    validate_development_path,
)


STAGE1_PILOT_SUMMARY_TYPE = "samga_brain_rw.stage1_pilot_summary"
LOCKED_SURVIVOR_TYPE = "samga_brain_rw.stage1_locked_survivor"
STAGE1_FUSION_CONFIG = Path("experiments/samga_brain_rw/configs/stage1_fusion_v1.json")
FINALIZER_LOCK_RELATIVE_PATH = Path(
    "artifacts/samga_brain_rw/registry/.stage1-finalizer.lock"
)
PILOT_SUMMARY_NAME = "pilot_summary.json"
PILOT_CELLS_NAME = "pilot_cells.csv"
LOCKED_SURVIVOR_NAME = "locked_survivor.json"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REGISTRY_STATE_KEYS = frozenset(
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
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


@dataclass(frozen=True)
class Stage1FinalizationResult:
    """Paths and identities published by one exact Stage 1 finalization."""

    passed: bool
    status: str
    winner_config_id: str
    outcome_sha256: str
    development_decision_sha256: str
    locked_decision_sha256: str | None
    summary_path: Path
    cells_path: Path
    locked_survivor_path: Path | None

    def to_payload(self) -> dict[str, object]:
        return {
            "cells_path": os.fspath(self.cells_path),
            "development_decision_sha256": self.development_decision_sha256,
            "locked_decision_sha256": self.locked_decision_sha256,
            "locked_survivor_path": (
                None
                if self.locked_survivor_path is None
                else os.fspath(self.locked_survivor_path)
            ),
            "outcome_sha256": self.outcome_sha256,
            "passed": self.passed,
            "status": self.status,
            "summary_path": os.fspath(self.summary_path),
            "winner_config_id": self.winner_config_id,
        }


@dataclass(frozen=True)
class _OutputPaths:
    summary: Path
    cells: Path
    locked_survivor: Path


@dataclass(frozen=True)
class _ExpectedArtifacts:
    summary: bytes
    cells: bytes
    locked_survivor: bytes | None
    outcome_sha256: str
    locked_decision: CandidateDecision | None


def _absolute_path(path: Path) -> Path:
    raw = os.fspath(path)
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise ValueError("path is invalid")
    return Path(os.path.abspath(os.path.normpath(raw)))


def _validated_project_root(path: Path) -> Path:
    root = _absolute_path(Path(path))
    descriptor = -1
    try:
        descriptor = _open_directory_nofollow(root, create=False)
    except OSError as exc:
        raise ValueError(
            f"project root must be an existing directory without symlinks: {root}"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return root


@contextmanager
def _exclusive_finalizer_lock(project_root: Path) -> Iterator[None]:
    """Serialize the complete Stage 1 transaction for one project root."""

    lock_path = project_root / FINALIZER_LOCK_RELATIVE_PATH
    try:
        parent_fd = _open_directory_nofollow(lock_path.parent, create=True)
    except OSError as exc:
        raise ValueError(
            "cannot create the fixed Stage 1 finalizer lock directory without symlinks"
        ) from exc
    descriptor = -1
    try:
        try:
            descriptor = os.open(
                lock_path.name,
                os.O_RDWR | os.O_CREAT | _O_CLOEXEC | _O_NOFOLLOW,
                0o600,
                dir_fd=parent_fd,
            )
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError("Stage 1 finalizer lock must be a regular file")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        except OSError as exc:
            raise ValueError(
                "cannot acquire the fixed Stage 1 finalizer lock without symlinks"
            ) from exc
        yield
    finally:
        if descriptor >= 0:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
        os.close(parent_fd)


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a 64-character lowercase SHA-256")
    return value


def _output_paths(output_dir: Path) -> _OutputPaths:
    directory = _absolute_path(Path(output_dir))
    summary = validate_development_path(
        directory / PILOT_SUMMARY_NAME,
        allowed_suffixes=frozenset({".json"}),
    )
    cells = validate_development_path(
        directory / PILOT_CELLS_NAME,
        allowed_suffixes=frozenset({".csv"}),
    )
    locked = validate_development_path(
        directory / LOCKED_SURVIVOR_NAME,
        allowed_suffixes=frozenset({".json"}),
    )
    return _OutputPaths(summary=summary, cells=cells, locked_survivor=locked)


def _inspect_optional_regular(path: Path) -> bool:
    parent_fd = -1
    descriptor = -1
    try:
        try:
            parent_fd = _open_directory_nofollow(path.parent, create=False)
        except FileNotFoundError:
            return False
        try:
            descriptor = os.open(
                path.name,
                os.O_RDONLY | _O_CLOEXEC | _O_NOFOLLOW,
                dir_fd=parent_fd,
            )
        except FileNotFoundError:
            return False
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"preflight path must be a regular file: {path}")
        return True
    except OSError as exc:
        raise ValueError(f"cannot preflight path without symlinks: {path}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if parent_fd >= 0:
            os.close(parent_fd)


def _preflight_registry(
    registry: CandidateRegistry,
) -> dict[str, object]:
    journal_exists = _inspect_optional_regular(registry.journal_path)
    state_exists = _inspect_optional_regular(registry.state_path)
    if state_exists and not journal_exists:
        raise ValueError("registry compact state exists without its journal")
    return _validated_registry_state(registry)


def _preflight_outputs(paths: _OutputPaths) -> None:
    _inspect_optional_regular(paths.summary)
    _inspect_optional_regular(paths.cells)
    _inspect_optional_regular(paths.locked_survivor)


def _read_existing_at(
    parent_fd: int,
    name: str,
    *,
    maximum_bytes: int,
) -> bytes | None:
    descriptor = -1
    try:
        try:
            descriptor = os.open(
                name,
                os.O_RDONLY | _O_CLOEXEC | _O_NOFOLLOW,
                dir_fd=parent_fd,
            )
        except FileNotFoundError:
            return None
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"existing output must be a regular file: {name}")
        if before.st_size > maximum_bytes:
            raise ValueError(f"divergent existing output {name}")
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        after = os.fstat(descriptor)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if before_identity != after_identity or len(data) != before.st_size:
            raise ValueError(f"existing output changed during read: {name}")
        return data
    except OSError as exc:
        raise ValueError(
            f"cannot read existing output without symlinks: {name}"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _read_optional_output(path: Path, maximum_bytes: int) -> bytes | None:
    parent_fd = -1
    try:
        try:
            parent_fd = _open_directory_nofollow(path.parent, create=False)
        except FileNotFoundError:
            return None
        return _read_existing_at(
            parent_fd,
            path.name,
            maximum_bytes=maximum_bytes,
        )
    except OSError as exc:
        raise ValueError(
            f"cannot read output directory without symlinks: {path.parent}"
        ) from exc
    finally:
        if parent_fd >= 0:
            os.close(parent_fd)


def _assert_existing_identical(path: Path, expected: bytes) -> None:
    existing = _read_optional_output(path, len(expected))
    if existing is not None and existing != expected:
        raise ValueError(f"divergent existing output {path.name}")


def _assert_existing_absent(path: Path) -> None:
    if _read_optional_output(path, 1) is not None:
        raise ValueError(f"divergent existing output {path.name}")


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("short write while staging output")
        offset += written


def _publish_create_or_verify_identical(path: Path, payload: bytes) -> None:
    """Atomically create one immutable file, or verify its exact bytes."""

    parent_fd = _open_directory_nofollow(path.parent, create=True)
    temporary_name: str | None = None
    descriptor = -1
    try:
        existing = _read_existing_at(
            parent_fd,
            path.name,
            maximum_bytes=len(payload),
        )
        if existing is not None:
            if existing != payload:
                raise ValueError(f"divergent existing output {path.name}")
            return

        for _ in range(32):
            candidate = (
                f".{path.name}.stage1-finalizer.{os.getpid()}.{secrets.token_hex(8)}"
            )
            try:
                descriptor = os.open(
                    candidate,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_CLOEXEC | _O_NOFOLLOW,
                    0o600,
                    dir_fd=parent_fd,
                )
            except FileExistsError:
                continue
            temporary_name = candidate
            break
        if temporary_name is None or descriptor < 0:
            raise FileExistsError("cannot allocate a unique staging file")

        _write_all(descriptor, payload)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        try:
            os.link(
                temporary_name,
                path.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
            os.fsync(parent_fd)
        except FileExistsError:
            existing = _read_existing_at(
                parent_fd,
                path.name,
                maximum_bytes=len(payload),
            )
            if existing != payload:
                raise ValueError(f"divergent existing output {path.name}")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=parent_fd)
                os.fsync(parent_fd)
            except FileNotFoundError:
                pass
        os.close(parent_fd)


def _decimal_delta(winner: float, control: float) -> str:
    return str(Decimal(str(winner)) - Decimal(str(control)))


def _pilot_cells_csv(outcome: Stage1CompositionOutcome) -> bytes:
    vectors = (
        tuple(outcome.control_top1),
        tuple(outcome.control_top5),
        tuple(outcome.winner_top1),
        tuple(outcome.winner_top5),
    )
    if any(len(vector) != len(PILOT_COORDINATES) for vector in vectors):
        raise ValueError("Stage 1 outcome must contain exactly six metric cells")
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        (
            "subject",
            "seed",
            "control_top1",
            "control_top5",
            "winner_top1",
            "winner_top5",
            "top1_delta",
            "top5_delta",
        )
    )
    for index, (subject, seed) in enumerate(PILOT_COORDINATES):
        control_top1 = vectors[0][index]
        control_top5 = vectors[1][index]
        winner_top1 = vectors[2][index]
        winner_top5 = vectors[3][index]
        writer.writerow(
            (
                subject,
                seed,
                control_top1,
                control_top5,
                winner_top1,
                winner_top5,
                _decimal_delta(winner_top1, control_top1),
                _decimal_delta(winner_top5, control_top5),
            )
        )
    return stream.getvalue().encode("utf-8")


def _locked_decision(decision: CandidateDecision) -> CandidateDecision:
    document = decision.to_document()
    document["locked"] = True
    return CandidateDecision.from_document(document)


def _validated_registry_state(
    registry: CandidateRegistry,
) -> dict[str, object]:
    if not isinstance(registry, CandidateRegistry):
        raise TypeError("registry must be a CandidateRegistry")
    state = registry.load_state()
    if set(state) != _REGISTRY_STATE_KEYS:
        raise ValueError("candidate registry state keys mismatch")
    if (
        state["artifact_type"] != REGISTRY_STATE_TYPE
        or type(state["schema_version"]) is not int
        or state["schema_version"] != 1
        or type(state["sequence"]) is not int
        or not isinstance(state["stages"], Mapping)
    ):
        raise ValueError("candidate registry state type/version is invalid")
    reported_sha256 = _require_sha256(
        state["state_sha256"],
        "registry_state.state_sha256",
    )
    body = {key: value for key, value in state.items() if key != "state_sha256"}
    if sha256_json(body) != reported_sha256:
        raise ValueError("candidate registry state SHA-256 mismatch")
    return state


def _registry_state_contains_locked(
    registry_state: Mapping[str, object],
    locked: CandidateDecision,
) -> bool:
    stages = registry_state.get("stages")
    if not isinstance(stages, Mapping):
        raise ValueError("candidate registry state lacks stages")
    stage = stages.get("1")
    if not isinstance(stage, Mapping):
        return False
    survivor = stage.get("survivor")
    if survivor is None:
        return False
    if not isinstance(survivor, Mapping):
        raise ValueError("candidate registry survivor reference is invalid")
    return (
        survivor.get("candidate_id") == locked.candidate_id
        and survivor.get("decision_sha256") == locked.decision_sha256
        and survivor.get("frozen_identity_sha256") == locked.frozen_identity_sha256
    )


def _registry_survivor_sha256(
    registry_state: Mapping[str, object],
    locked: CandidateDecision,
) -> str:
    if not _registry_state_contains_locked(registry_state, locked):
        raise ValueError(
            "Stage 1 locked survivor is absent from candidate registry state"
        )
    stages = registry_state["stages"]
    assert isinstance(stages, Mapping)
    stage = stages["1"]
    assert isinstance(stage, Mapping)
    survivor = stage["survivor"]
    assert isinstance(survivor, Mapping)
    return sha256_json(dict(survivor))


def validate_stage1_locked_survivor_document(
    document: Mapping[str, object],
    *,
    registry: CandidateRegistry,
) -> CandidateDecision:
    """Validate the Stage 1 survivor against the verified live registry."""

    expected_keys = {
        "artifact_type",
        "decision",
        "decision_sha256",
        "development_decision_sha256",
        "registry_survivor_sha256",
        "schema_version",
        "stage",
    }
    if set(document) != expected_keys:
        raise ValueError("Stage 1 locked-survivor keys mismatch")
    if (
        document["artifact_type"] != LOCKED_SURVIVOR_TYPE
        or type(document["schema_version"]) is not int
        or document["schema_version"] != 1
        or type(document["stage"]) is not int
        or document["stage"] != 1
    ):
        raise ValueError("Stage 1 locked-survivor type/version is invalid")
    raw_decision = document["decision"]
    if not isinstance(raw_decision, Mapping):
        raise ValueError("Stage 1 locked-survivor decision must be an object")
    locked = CandidateDecision.from_payload(raw_decision)
    if not locked.locked or locked.stage != 1 or locked.scope != "val-dev":
        raise ValueError("Stage 1 locked-survivor decision is not locked")
    if document["decision_sha256"] != locked.decision_sha256:
        raise ValueError("Stage 1 locked-survivor decision SHA-256 mismatch")

    development_document = locked.to_document()
    development_document["locked"] = False
    development = CandidateDecision.from_document(development_document)
    if document["development_decision_sha256"] != development.decision_sha256:
        raise ValueError("Stage 1 locked-survivor development SHA-256 mismatch")
    registry_state = _validated_registry_state(registry)
    expected_survivor_sha256 = _registry_survivor_sha256(
        registry_state,
        locked,
    )
    if document["registry_survivor_sha256"] != expected_survivor_sha256:
        raise ValueError("Stage 1 registry-survivor SHA-256 mismatch")
    return locked


def _locked_survivor_document(
    development: CandidateDecision,
    locked: CandidateDecision,
    registry: CandidateRegistry,
) -> dict[str, object]:
    registry_state = _validated_registry_state(registry)
    document = {
        "artifact_type": LOCKED_SURVIVOR_TYPE,
        "decision": locked.to_payload(),
        "decision_sha256": locked.decision_sha256,
        "development_decision_sha256": development.decision_sha256,
        "registry_survivor_sha256": _registry_survivor_sha256(
            registry_state,
            locked,
        ),
        "schema_version": 1,
        "stage": 1,
    }
    validated = validate_stage1_locked_survivor_document(
        document,
        registry=registry,
    )
    if validated.decision_sha256 != locked.decision_sha256:
        raise AssertionError("Stage 1 locked-survivor validation changed")
    return document


def _expected_artifacts(
    *,
    outcome: Stage1CompositionOutcome,
    decision: CandidateDecision,
    semantic_config_sha256: str,
    cost_job_map_sha256: str,
    registry: CandidateRegistry | None,
) -> _ExpectedArtifacts:
    outcome_payload = outcome.to_payload()
    outcome_sha256 = sha256_json(outcome_payload)
    cells_bytes = _pilot_cells_csv(outcome)
    locked_decision = _locked_decision(decision) if outcome.passed else None
    locked_bytes: bytes | None = None
    if locked_decision is not None:
        if registry is None:
            raise ValueError(
                "passing Stage 1 artifacts require the locked registry state"
            )
        locked_document = _locked_survivor_document(
            decision,
            locked_decision,
            registry,
        )
        locked_bytes = canonical_json_bytes(locked_document) + b"\n"
    summary: dict[str, object] = {
        "artifact_type": STAGE1_PILOT_SUMMARY_TYPE,
        "artifacts": {
            PILOT_CELLS_NAME: {
                "rows": len(PILOT_COORDINATES),
                "sha256": hashlib.sha256(cells_bytes).hexdigest(),
            },
            LOCKED_SURVIVOR_NAME: (
                None
                if locked_bytes is None
                else {
                    "decision_sha256": locked_decision.decision_sha256,
                    "registry_survivor_sha256": locked_document[
                        "registry_survivor_sha256"
                    ],
                    "sha256": hashlib.sha256(locked_bytes).hexdigest(),
                }
            ),
        },
        "control_branch_id": outcome.control_branch_id,
        "development_decision_sha256": decision.decision_sha256,
        "inputs": {
            "cost_job_map_sha256": cost_job_map_sha256,
            "semantic_config_sha256": semantic_config_sha256,
        },
        "locked_decision_sha256": (
            None if locked_decision is None else locked_decision.decision_sha256
        ),
        "outcome": outcome_payload,
        "outcome_sha256": outcome_sha256,
        "passed": outcome.passed,
        "schema_version": 1,
        "scope": "val-dev",
        "stage": 1,
        "status": outcome.status,
        "winner_config_id": outcome.winner_config_id,
    }
    return _ExpectedArtifacts(
        summary=canonical_json_bytes(summary) + b"\n",
        cells=cells_bytes,
        locked_survivor=locked_bytes,
        outcome_sha256=outcome_sha256,
        locked_decision=locked_decision,
    )


def _validate_decision_handoff(
    outcome: Stage1CompositionOutcome,
) -> CandidateDecision:
    decision = CandidateDecision.from_document(outcome.candidate_decision_document())
    if (
        decision.stage != 1
        or decision.scope != "val-dev"
        or decision.locked
        or decision.candidate_id != outcome.winner_config_id
        or decision.control_id != outcome.control_branch_id
        or decision.gate.passed != outcome.passed
        or outcome.status != ("passed" if outcome.passed else "failed")
    ):
        raise ValueError("Stage 1 outcome candidate-decision handoff is invalid")
    return decision


def _assert_outputs_match(
    paths: _OutputPaths,
    expected: _ExpectedArtifacts,
) -> None:
    _assert_existing_identical(paths.summary, expected.summary)
    _assert_existing_identical(paths.cells, expected.cells)
    if expected.locked_survivor is None:
        _assert_existing_absent(paths.locked_survivor)
    else:
        _assert_existing_identical(
            paths.locked_survivor,
            expected.locked_survivor,
        )


def _assert_outputs_before_registry_mutation(
    paths: _OutputPaths,
    *,
    outcome: Stage1CompositionOutcome,
    decision: CandidateDecision,
    semantic_config_sha256: str,
    cost_job_map_sha256: str,
    registry_state: Mapping[str, object],
    registry: CandidateRegistry,
) -> None:
    if not outcome.passed:
        expected = _expected_artifacts(
            outcome=outcome,
            decision=decision,
            semantic_config_sha256=semantic_config_sha256,
            cost_job_map_sha256=cost_job_map_sha256,
            registry=None,
        )
        _assert_outputs_match(paths, expected)
        return

    locked = _locked_decision(decision)
    if _registry_state_contains_locked(registry_state, locked):
        expected = _expected_artifacts(
            outcome=outcome,
            decision=decision,
            semantic_config_sha256=semantic_config_sha256,
            cost_job_map_sha256=cost_job_map_sha256,
            registry=registry,
        )
        _assert_outputs_match(paths, expected)
        return

    # Without the exact prior lock, no summary/survivor can belong to this
    # transaction. The deterministic CSV may still be a recoverable prefix.
    _assert_existing_absent(paths.summary)
    _assert_existing_identical(paths.cells, _pilot_cells_csv(outcome))
    _assert_existing_absent(paths.locked_survivor)


def _assert_exact_decision(
    actual: CandidateDecision,
    expected: CandidateDecision,
    context: str,
) -> None:
    if actual.decision_sha256 != expected.decision_sha256 or canonical_json_bytes(
        actual.to_document()
    ) != canonical_json_bytes(expected.to_document()):
        raise RuntimeError(f"registry returned a divergent {context} decision")


def _reject_output_input_aliases(
    paths: _OutputPaths,
    *,
    cost_job_map_path: Path,
    registry: CandidateRegistry,
    semantic_config_path: Path,
) -> None:
    inputs = {
        _absolute_path(cost_job_map_path),
        registry.journal_path,
        registry.state_path,
        _absolute_path(semantic_config_path),
    }
    for output in (paths.summary, paths.cells, paths.locked_survivor):
        if output in inputs:
            raise ValueError("Stage 1 finalizer output aliases an input")


def _revalidate_before_mutation(
    cells: Sequence[Stage1CompositionCell],
    cost_capability: object,
) -> None:
    if len(cells) != len(PILOT_COORDINATES):
        raise ValueError("Stage 1 finalizer requires exactly six cells")
    coordinates = tuple((cell.subject, cell.seed) for cell in cells)
    if coordinates != PILOT_COORDINATES:
        raise ValueError("Stage 1 finalizer cells are not in pilot-grid order")
    for cell in cells:
        cell.revalidate()
    revalidate = getattr(cost_capability, "revalidate", None)
    if not callable(revalidate):
        raise TypeError("Stage 1 cost capability lacks revalidation")
    revalidate()


def finalize_stage1(
    *,
    project_root: Path,
    cost_job_map_path: Path,
    cost_job_map_sha256: str,
    journal_path: Path,
    state_path: Path,
    output_dir: Path,
) -> Stage1FinalizationResult:
    """Finalize only the fixed six-cell Stage 1 val-dev composition."""

    root = _validated_project_root(Path(project_root))
    expected_cost_sha256 = _require_sha256(
        cost_job_map_sha256,
        "cost_job_map_sha256",
    )
    normalized_cost_job_map = _absolute_path(Path(cost_job_map_path))
    registry = CandidateRegistry(Path(journal_path), Path(state_path))
    outputs = _output_paths(Path(output_dir))
    semantic_path = root / STAGE1_FUSION_CONFIG
    _reject_output_input_aliases(
        outputs,
        cost_job_map_path=normalized_cost_job_map,
        registry=registry,
        semantic_config_path=semantic_path,
    )

    with _exclusive_finalizer_lock(root):
        # Full registry and output preflight is serialized with all commits.
        initial_registry_state = _preflight_registry(registry)
        _preflight_outputs(outputs)

        semantic_config = SemanticConfig.from_path(semantic_path)
        cells = load_stage1_composition_cells(root, semantic_config)
        cost_capability = load_validated_stage1_cost_capability(
            normalized_cost_job_map,
            expected_cost_sha256,
        )
        outcome = compose_stage1(
            cells,
            semantic_config=semantic_config,
            cost_capability=cost_capability,
        )
        decision = _validate_decision_handoff(outcome)

        # Reject every incompatible prior public artifact before mutation.
        _assert_outputs_before_registry_mutation(
            outputs,
            outcome=outcome,
            decision=decision,
            semantic_config_sha256=semantic_config.sha256,
            cost_job_map_sha256=expected_cost_sha256,
            registry_state=initial_registry_state,
            registry=registry,
        )

        # This is the last evidence read before the first registry mutation.
        _revalidate_before_mutation(cells, cost_capability)

        development = registry.append_or_reuse_exact(decision)
        _assert_exact_decision(development, decision, "development")
        locked: CandidateDecision | None = None
        if outcome.passed:
            expected_locked = _locked_decision(decision)
            locked = registry.lock_stage_survivor_or_reuse_exact(
                1,
                development.decision_sha256,
            )
            _assert_exact_decision(locked, expected_locked, "locked")

        expected = _expected_artifacts(
            outcome=outcome,
            decision=decision,
            semantic_config_sha256=semantic_config.sha256,
            cost_job_map_sha256=expected_cost_sha256,
            registry=(registry if locked is not None else None),
        )

        # Recheck under the same mutex before every create-or-verify publish.
        _assert_outputs_match(outputs, expected)
        _publish_create_or_verify_identical(outputs.cells, expected.cells)
        if expected.locked_survivor is not None:
            _publish_create_or_verify_identical(
                outputs.locked_survivor,
                expected.locked_survivor,
            )
        # Summary is the completion marker, so publish it last.
        _publish_create_or_verify_identical(outputs.summary, expected.summary)

        return Stage1FinalizationResult(
            passed=outcome.passed,
            status=outcome.status,
            winner_config_id=outcome.winner_config_id,
            outcome_sha256=expected.outcome_sha256,
            development_decision_sha256=development.decision_sha256,
            locked_decision_sha256=(None if locked is None else locked.decision_sha256),
            summary_path=outputs.summary,
            cells_path=outputs.cells,
            locked_survivor_path=(None if locked is None else outputs.locked_survivor),
        )


__all__ = [
    "LOCKED_SURVIVOR_NAME",
    "LOCKED_SURVIVOR_TYPE",
    "FINALIZER_LOCK_RELATIVE_PATH",
    "PILOT_CELLS_NAME",
    "PILOT_SUMMARY_NAME",
    "STAGE1_PILOT_SUMMARY_TYPE",
    "Stage1FinalizationResult",
    "finalize_stage1",
    "validate_stage1_locked_survivor_document",
]
