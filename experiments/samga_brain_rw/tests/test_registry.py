from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from samga_brain_rw.hashing import canonical_json_bytes, sha256_json
from samga_brain_rw.registry import (
    CandidateDecision,
    CandidateRegistry,
    RegistryIntegrityError,
    RegistryStateError,
)
from samga_brain_rw.statistics import GateDecision


def _h(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _gate(*, passed: bool = True, stage: int = 1) -> GateDecision:
    return GateDecision(
        gate_kind="pilot",
        stage=stage,
        passed=passed,
        mean_top1_delta=0.01 if passed else 0.0,
        mean_top5_delta=0.0,
        ci95=None,
        positive_cells=6 if passed else 0,
        positive_subjects=3 if passed else 0,
        worst_subject_top1_delta=0.01 if passed else 0.0,
        subject_mean_top1_deltas=((1, 0.01), (5, 0.01), (8, 0.01)),
        criteria={
            "mean_top1_delta": passed,
            "mean_top5_delta": True,
            "positive_cells": passed,
            "subject_floor": True,
        },
        initialization_evidence=(),
    )


def _decision(
    candidate_id: str,
    *,
    stage: int = 1,
    scope: str = "val-dev",
    passed: bool = True,
    absolute_top1: float = 0.80,
    absolute_top5: float = 0.95,
    config: str | None = None,
    hyperparameters: str | None = None,
    schedule: str | None = None,
    components: tuple[str, ...] | None = None,
) -> CandidateDecision:
    return CandidateDecision(
        stage=stage,
        candidate_id=candidate_id,
        control_id=f"{candidate_id}-control",
        scope=scope,
        config_sha256=_h(config or f"{candidate_id}-config"),
        control_config_sha256=_h(f"{candidate_id}-control-config"),
        hyperparameters_sha256=_h(
            hyperparameters or f"{candidate_id}-hyperparameters"
        ),
        schedule_sha256=_h(schedule or f"{candidate_id}-schedule"),
        component_sha256s=components or (_h(f"{candidate_id}-component"),),
        candidate_matrix_sha256=_h(f"{candidate_id}-{scope}-candidate-matrix"),
        control_matrix_sha256=_h(f"{candidate_id}-{scope}-control-matrix"),
        absolute_top1=absolute_top1,
        absolute_top5=absolute_top5,
        gate=replace(
            _gate(passed=passed, stage=stage),
            gate_kind="confirmation" if scope == "val-confirm" else "pilot",
            ci95=(0.001, 0.02) if scope == "val-confirm" else None,
        ),
    )


def _registry(tmp_path: Path) -> CandidateRegistry:
    return CandidateRegistry(
        tmp_path / "decisions.jsonl",
        tmp_path / "state.json",
    )


def test_registry_writes_canonical_hash_chained_journal_and_compact_state(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    decision = _decision("candidate-a")

    registry.append(decision)

    journal_bytes = registry.journal_path.read_bytes()
    assert journal_bytes.endswith(b"\n")
    assert b" " not in journal_bytes
    lines = journal_bytes.splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["sequence"] == 1
    assert record["previous_record_sha256"] is None
    assert record["decision_sha256"] == sha256_json(decision.to_payload())
    record_body = {
        key: value
        for key, value in record.items()
        if key not in {"record_sha256", "state_sha256"}
    }
    assert record["record_sha256"] == sha256_json(record_body)

    state = json.loads(registry.state_path.read_text("utf-8"))
    state_body = {
        key: value for key, value in state.items() if key != "state_sha256"
    }
    assert state["state_sha256"] == sha256_json(state_body)
    assert state["head_record_sha256"] == record["record_sha256"]
    assert state["sequence"] == 1
    assert state["stages"]["1"]["candidates"]["candidate-a"]["val-dev"] == (
        record["decision_sha256"]
    )
    assert registry.state_path.read_bytes() == canonical_json_bytes(state) + b"\n"
    registry.verify()


def test_registry_chain_links_multiple_records(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.append(_decision("candidate-a"))
    registry.append(_decision("candidate-b"))

    first, second = [
        json.loads(line)
        for line in registry.journal_path.read_text("utf-8").splitlines()
    ]
    assert second["sequence"] == 2
    assert second["previous_record_sha256"] == first["record_sha256"]
    assert second["previous_state_sha256"] == first["state_sha256"]
    registry.verify()


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("schema_version", True),
        ("schema_version", 1.0),
        ("sequence", True),
        ("sequence", 1.0),
    ],
    ids=[
        "schema-boolean-true",
        "schema-float-one",
        "sequence-boolean-true",
        "sequence-float-one",
    ],
)
def test_registry_rejects_type_changed_record_integer_fields(
    tmp_path: Path,
    field: str,
    bad_value: object,
) -> None:
    registry = _registry(tmp_path)
    registry.append(_decision("candidate-a"))
    record = json.loads(registry.journal_path.read_text("utf-8"))
    state = json.loads(registry.state_path.read_text("utf-8"))

    record[field] = bad_value
    record_body = {
        key: value
        for key, value in record.items()
        if key not in {"record_sha256", "state_sha256"}
    }
    record["record_sha256"] = sha256_json(record_body)
    state["head_record_sha256"] = record["record_sha256"]
    state_body = {
        key: value
        for key, value in state.items()
        if key != "state_sha256"
    }
    state["state_sha256"] = sha256_json(state_body)
    record["state_sha256"] = state["state_sha256"]
    registry.journal_path.write_bytes(canonical_json_bytes(record) + b"\n")
    registry.state_path.write_bytes(canonical_json_bytes(state) + b"\n")
    journal_before = registry.journal_path.read_bytes()
    state_before = registry.state_path.read_bytes()

    with pytest.raises(RegistryIntegrityError, match=field):
        registry.verify()

    assert registry.journal_path.read_bytes() == journal_before
    assert registry.state_path.read_bytes() == state_before


@pytest.mark.parametrize(
    "bad_schema_version",
    [True, 1.0],
    ids=["boolean-true", "float-one"],
)
def test_candidate_decision_document_requires_integer_schema_version(
    bad_schema_version: object,
) -> None:
    document = _decision("candidate-a").to_document()
    document["schema_version"] = bad_schema_version

    with pytest.raises(RegistryIntegrityError, match="schema_version"):
        CandidateDecision.from_document(document)


def test_registry_refuses_duplicate_decision_without_mutating_files(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    decision = _decision("candidate-a")
    registry.append(decision)
    journal_before = registry.journal_path.read_bytes()
    state_before = registry.state_path.read_bytes()

    with pytest.raises(RegistryStateError, match="duplicate"):
        registry.append(decision)

    assert registry.journal_path.read_bytes() == journal_before
    assert registry.state_path.read_bytes() == state_before


def test_append_or_reuse_exact_is_idempotent_without_new_record(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    decision = _decision("candidate-a")

    appended = registry.append_or_reuse_exact(decision)
    journal_before = registry.journal_path.read_bytes()
    state_before = registry.state_path.read_bytes()
    reused = registry.append_or_reuse_exact(decision)

    assert appended == decision
    assert reused == decision
    assert registry.journal_path.read_bytes() == journal_before
    assert registry.state_path.read_bytes() == state_before
    assert registry.load_state()["sequence"] == 1


def test_append_or_reuse_exact_rejects_divergent_duplicate_without_mutation(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    decision = _decision("candidate-a")
    registry.append_or_reuse_exact(decision)
    journal_before = registry.journal_path.read_bytes()
    state_before = registry.state_path.read_bytes()
    divergent = replace(
        decision,
        candidate_matrix_sha256=_h("divergent-candidate-matrix"),
    )

    with pytest.raises(RegistryStateError, match="duplicate"):
        registry.append_or_reuse_exact(divergent)

    assert registry.journal_path.read_bytes() == journal_before
    assert registry.state_path.read_bytes() == state_before


def test_append_or_reuse_exact_serializes_concurrent_exact_retries(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "decisions.jsonl"
    state = tmp_path / "state.json"
    decision = _decision("candidate-a")

    def append(_: int) -> CandidateDecision:
        return CandidateRegistry(journal, state).append_or_reuse_exact(decision)

    with ThreadPoolExecutor(max_workers=8) as pool:
        returned = list(pool.map(append, range(16)))

    assert returned == [decision] * 16
    registry = CandidateRegistry(journal, state)
    registry.verify()
    assert len(journal.read_text("utf-8").splitlines()) == 1
    assert registry.load_state()["sequence"] == 1


def test_append_or_reuse_exact_recovers_missing_state_after_publish_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _registry(tmp_path)
    decision = _decision("candidate-a")
    publish = registry._publish_state_unlocked

    def crash_after_journal_fsync(_: object) -> None:
        raise OSError("injected compact-state publish crash")

    monkeypatch.setattr(
        registry,
        "_publish_state_unlocked",
        crash_after_journal_fsync,
    )
    with pytest.raises(OSError, match="injected"):
        registry.append_or_reuse_exact(decision)

    assert len(registry.journal_path.read_text("utf-8").splitlines()) == 1
    assert not registry.state_path.exists()

    monkeypatch.setattr(registry, "_publish_state_unlocked", publish)
    reused = registry.append_or_reuse_exact(decision)

    assert reused == decision
    state = registry.load_state()
    record = json.loads(registry.journal_path.read_text("utf-8"))
    assert state["sequence"] == 1
    assert state["state_sha256"] == record["state_sha256"]
    registry.verify()


def test_append_or_reuse_exact_recovers_prefix_state_after_publish_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _registry(tmp_path)
    registry.append_or_reuse_exact(_decision("candidate-a"))
    prefix_state = registry.state_path.read_bytes()
    decision = _decision("candidate-b")
    publish = registry._publish_state_unlocked

    def crash_after_journal_fsync(_: object) -> None:
        raise OSError("injected compact-state publish crash")

    monkeypatch.setattr(
        registry,
        "_publish_state_unlocked",
        crash_after_journal_fsync,
    )
    with pytest.raises(OSError, match="injected"):
        registry.append_or_reuse_exact(decision)

    assert len(registry.journal_path.read_text("utf-8").splitlines()) == 2
    assert registry.state_path.read_bytes() == prefix_state

    monkeypatch.setattr(registry, "_publish_state_unlocked", publish)
    reused = registry.append_or_reuse_exact(decision)

    assert reused == decision
    state = registry.load_state()
    records = [
        json.loads(line)
        for line in registry.journal_path.read_text("utf-8").splitlines()
    ]
    assert state["sequence"] == 2
    assert state["state_sha256"] == records[-1]["state_sha256"]
    registry.verify()


def test_missing_state_recovery_validates_full_journal_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _registry(tmp_path)

    def crash_after_journal_fsync(_: object) -> None:
        raise OSError("injected compact-state publish crash")

    monkeypatch.setattr(
        registry,
        "_publish_state_unlocked",
        crash_after_journal_fsync,
    )
    with pytest.raises(OSError, match="injected"):
        registry.append_or_reuse_exact(_decision("candidate-a"))

    record = json.loads(registry.journal_path.read_text("utf-8"))
    record["decision"]["absolute_top1"] = 0.79
    registry.journal_path.write_bytes(canonical_json_bytes(record) + b"\n")
    corrupted = registry.journal_path.read_bytes()

    with pytest.raises(RegistryIntegrityError):
        _registry(tmp_path).append_or_reuse_exact(_decision("candidate-a"))

    assert registry.journal_path.read_bytes() == corrupted
    assert not registry.state_path.exists()


def test_recovery_rejects_canonical_state_that_is_not_a_journal_prefix(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path / "target")
    registry.append_or_reuse_exact(_decision("candidate-a"))
    other = _registry(tmp_path / "other")
    other.append_or_reuse_exact(_decision("candidate-b"))
    registry.state_path.write_bytes(other.state_path.read_bytes())
    divergent = registry.state_path.read_bytes()

    with pytest.raises(RegistryIntegrityError, match="state"):
        registry.append_or_reuse_exact(_decision("candidate-a"))

    assert registry.state_path.read_bytes() == divergent
    assert len(registry.journal_path.read_text("utf-8").splitlines()) == 1


@pytest.mark.parametrize(
    "bad_schema_version",
    [True, 1.0],
    ids=["boolean-true", "float-one"],
)
@pytest.mark.parametrize(
    "state_position",
    ["current", "lagging-prefix"],
)
def test_recovery_rejects_type_changed_schema_version_with_stale_state_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_schema_version: object,
    state_position: str,
) -> None:
    registry = _registry(tmp_path)
    registry.append_or_reuse_exact(_decision("candidate-a"))
    retry = _decision("candidate-a")

    if state_position == "lagging-prefix":
        retry = _decision("candidate-b")
        publish = registry._publish_state_unlocked

        def crash_after_journal_fsync(_: object) -> None:
            raise OSError("injected compact-state publish crash")

        monkeypatch.setattr(
            registry,
            "_publish_state_unlocked",
            crash_after_journal_fsync,
        )
        with pytest.raises(OSError, match="injected"):
            registry.append_or_reuse_exact(retry)
        monkeypatch.setattr(registry, "_publish_state_unlocked", publish)

    stored_state = json.loads(registry.state_path.read_text("utf-8"))
    stored_state["schema_version"] = bad_schema_version
    registry.state_path.write_bytes(
        canonical_json_bytes(stored_state) + b"\n"
    )
    state_before = registry.state_path.read_bytes()
    journal_before = registry.journal_path.read_bytes()

    with pytest.raises(RegistryIntegrityError, match="schema|SHA-256"):
        registry.append_or_reuse_exact(retry)

    assert registry.state_path.read_bytes() == state_before
    assert registry.journal_path.read_bytes() == journal_before


@pytest.mark.parametrize("target_name", ["decisions.jsonl", "state.json"])
def test_registry_refuses_corruption_without_overwriting_it(
    tmp_path: Path,
    target_name: str,
) -> None:
    registry = _registry(tmp_path)
    registry.append(_decision("candidate-a"))
    target = tmp_path / target_name
    original = target.read_bytes()
    target.write_bytes(original[:-2] + b"X\n")
    corrupted = target.read_bytes()

    with pytest.raises(RegistryIntegrityError):
        registry.append(_decision("candidate-b"))

    assert target.read_bytes() == corrupted


def test_lock_refuses_multiple_passing_candidates_without_stage_selector(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    registry.append(
        _decision(
            "candidate-z",
            absolute_top1=0.84,
            absolute_top5=0.97,
        )
    )
    registry.append(
        _decision(
            "candidate-a",
            absolute_top1=0.84,
            absolute_top5=0.97,
        )
    )
    registry.append(
        _decision(
            "candidate-failed",
            passed=False,
            absolute_top1=0.99,
            absolute_top5=0.99,
        )
    )

    with pytest.raises(RegistryStateError, match="multiple passing"):
        registry.lock_stage_survivor(1)

    state = registry.load_state()
    assert state["sequence"] == 3
    assert state["stages"]["1"]["survivor"] is None


def test_lock_rejects_stage_without_passing_candidate(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.append(_decision("failed", passed=False))
    with pytest.raises(RegistryStateError, match="passing"):
        registry.lock_stage_survivor(1)


def test_lock_stage_survivor_or_reuse_exact_is_idempotent_without_new_record(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    development = _decision("candidate-a")
    registry.append(development)

    locked = registry.lock_stage_survivor_or_reuse_exact(
        1,
        development.decision_sha256,
    )
    journal_before = registry.journal_path.read_bytes()
    state_before = registry.state_path.read_bytes()
    reused = registry.lock_stage_survivor_or_reuse_exact(
        1,
        development.decision_sha256,
    )

    assert locked == replace(development, locked=True)
    assert reused == locked
    assert registry.journal_path.read_bytes() == journal_before
    assert registry.state_path.read_bytes() == state_before
    assert registry.load_state()["sequence"] == 2


def test_lock_stage_survivor_or_reuse_exact_rejects_wrong_expected_sha(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    registry.append(_decision("candidate-a"))
    journal_before = registry.journal_path.read_bytes()
    state_before = registry.state_path.read_bytes()

    with pytest.raises(RegistryStateError, match="expected"):
        registry.lock_stage_survivor_or_reuse_exact(1, _h("wrong-decision"))

    assert registry.journal_path.read_bytes() == journal_before
    assert registry.state_path.read_bytes() == state_before


def test_lock_stage_survivor_or_reuse_exact_rejects_divergent_existing_lock(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    development = _decision("candidate-a")
    other = _decision("candidate-b", passed=False)
    registry.append(development)
    registry.append(other)
    registry.lock_stage_survivor_or_reuse_exact(
        1,
        development.decision_sha256,
    )
    journal_before = registry.journal_path.read_bytes()
    state_before = registry.state_path.read_bytes()

    with pytest.raises(RegistryStateError, match="expected|divergent"):
        registry.lock_stage_survivor_or_reuse_exact(
            1,
            other.decision_sha256,
        )

    assert registry.journal_path.read_bytes() == journal_before
    assert registry.state_path.read_bytes() == state_before


def test_lock_stage_survivor_or_reuse_exact_preserves_unique_passing_gate(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    selected = _decision("candidate-a")
    registry.append(selected)
    registry.append(_decision("candidate-b"))
    journal_before = registry.journal_path.read_bytes()
    state_before = registry.state_path.read_bytes()

    with pytest.raises(RegistryStateError, match="multiple passing"):
        registry.lock_stage_survivor_or_reuse_exact(
            1,
            selected.decision_sha256,
        )

    assert registry.journal_path.read_bytes() == journal_before
    assert registry.state_path.read_bytes() == state_before


def test_lock_stage_survivor_or_reuse_exact_serializes_concurrent_retries(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "decisions.jsonl"
    state = tmp_path / "state.json"
    development = _decision("candidate-a")
    CandidateRegistry(journal, state).append(development)

    def lock(_: int) -> CandidateDecision:
        return CandidateRegistry(
            journal,
            state,
        ).lock_stage_survivor_or_reuse_exact(
            1,
            development.decision_sha256,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        returned = list(pool.map(lock, range(16)))

    expected = replace(development, locked=True)
    assert returned == [expected] * 16
    registry = CandidateRegistry(journal, state)
    registry.verify()
    assert len(journal.read_text("utf-8").splitlines()) == 2
    assert registry.load_state()["sequence"] == 2


def test_lock_stage_survivor_or_reuse_exact_recovers_prefix_state_after_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _registry(tmp_path)
    development = _decision("candidate-a")
    registry.append_or_reuse_exact(development)
    prefix_state = registry.state_path.read_bytes()
    publish = registry._publish_state_unlocked

    def crash_after_journal_fsync(_: object) -> None:
        raise OSError("injected compact-state publish crash")

    monkeypatch.setattr(
        registry,
        "_publish_state_unlocked",
        crash_after_journal_fsync,
    )
    with pytest.raises(OSError, match="injected"):
        registry.lock_stage_survivor_or_reuse_exact(
            1,
            development.decision_sha256,
        )

    assert len(registry.journal_path.read_text("utf-8").splitlines()) == 2
    assert registry.state_path.read_bytes() == prefix_state

    monkeypatch.setattr(registry, "_publish_state_unlocked", publish)
    reused = registry.lock_stage_survivor_or_reuse_exact(
        1,
        development.decision_sha256,
    )

    assert reused == replace(development, locked=True)
    state = registry.load_state()
    records = [
        json.loads(line)
        for line in registry.journal_path.read_text("utf-8").splitlines()
    ]
    assert state["sequence"] == 2
    assert state["state_sha256"] == records[-1]["state_sha256"]
    registry.verify()


def test_confirmation_requires_locked_survivor_and_identical_frozen_identity(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    development = _decision("candidate-a")

    with pytest.raises(RegistryStateError, match="survivor"):
        registry.append(
            replace(
                development,
                scope="val-confirm",
                gate=replace(
                    development.gate,
                    gate_kind="confirmation",
                    ci95=(0.001, 0.02),
                ),
            )
        )

    registry.append(development)
    registry.lock_stage_survivor(1)
    confirmation = _decision(
        "candidate-a",
        scope="val-confirm",
        config="candidate-a-config",
        hyperparameters="candidate-a-hyperparameters",
        schedule="candidate-a-schedule",
        components=(_h("candidate-a-component"),),
    )
    registry.append(confirmation)
    state = registry.load_state()
    assert state["stages"]["1"]["confirmed"]["candidate_id"] == "candidate-a"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("config_sha256", _h("changed-config"), "config"),
        ("hyperparameters_sha256", _h("changed-hyperparameters"), "hyperparameter"),
        ("schedule_sha256", _h("changed-schedule"), "schedule"),
        ("component_sha256s", (_h("changed-component"),), "component"),
    ],
)
def test_confirmation_refuses_every_frozen_identity_change(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    registry = _registry(tmp_path)
    development = _decision("candidate-a")
    registry.append(development)
    registry.lock_stage_survivor(1)
    confirmation = _decision(
        "candidate-a",
        scope="val-confirm",
        config="candidate-a-config",
        hyperparameters="candidate-a-hyperparameters",
        schedule="candidate-a-schedule",
        components=(_h("candidate-a-component"),),
    )
    confirmation = replace(confirmation, **{field: value})

    with pytest.raises(RegistryStateError, match=message):
        registry.append(confirmation)


def test_registry_serializes_concurrent_appends_without_lost_records(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "decisions.jsonl"
    state = tmp_path / "state.json"

    def append(index: int) -> None:
        CandidateRegistry(journal, state).append(_decision(f"candidate-{index:02d}"))

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(append, range(16)))

    registry = CandidateRegistry(journal, state)
    registry.verify()
    assert len(journal.read_text("utf-8").splitlines()) == 16
    assert registry.load_state()["sequence"] == 16


def _run_lock(
    experiment_root: Path,
    journal: Path,
    state: Path,
    output: Path,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(experiment_root)
    return subprocess.run(
        [
            sys.executable,
            str(experiment_root / "scripts" / "lock_survivor.py"),
            "--journal",
            str(journal),
            "--state",
            str(state),
            "--stage",
            "1",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_lock_cli_uses_explicit_paths_and_exclusive_output(
    experiment_root: Path,
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    registry.append(_decision("candidate-a"))
    output = tmp_path / "survivor.json"

    completed = _run_lock(
        experiment_root,
        registry.journal_path,
        registry.state_path,
        output,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text("utf-8"))
    assert payload["artifact_type"] == "samga_brain_rw.locked_survivor"
    assert payload["decision"]["candidate_id"] == "candidate-a"

    original = output.read_bytes()
    repeated = _run_lock(
        experiment_root,
        registry.journal_path,
        registry.state_path,
        output,
    )
    assert repeated.returncode != 0
    assert output.read_bytes() == original


def test_lock_cli_rejects_symlink_or_formal_registry_paths(
    experiment_root: Path,
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    registry.append(_decision("candidate-a"))
    symlink = tmp_path / "journal-link.jsonl"
    symlink.symlink_to(registry.journal_path)
    output = tmp_path / "survivor.json"

    linked = _run_lock(
        experiment_root,
        symlink,
        registry.state_path,
        output,
    )
    assert linked.returncode != 0
    assert not output.exists()

    formal_output = tmp_path / "formal-test" / "survivor.json"
    rejected = _run_lock(
        experiment_root,
        registry.journal_path,
        registry.state_path,
        formal_output,
    )
    assert rejected.returncode != 0
    assert not formal_output.exists()


def test_lock_cli_rejects_symlinked_output_parent_before_registry_mutation(
    experiment_root: Path,
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    registry.append(_decision("candidate-a"))
    real_output_parent = tmp_path / "real-output"
    real_output_parent.mkdir()
    linked_output_parent = tmp_path / "linked-output"
    linked_output_parent.symlink_to(real_output_parent, target_is_directory=True)
    output = linked_output_parent / "survivor.json"

    rejected = _run_lock(
        experiment_root,
        registry.journal_path,
        registry.state_path,
        output,
    )

    assert rejected.returncode != 0
    assert not output.exists()
    state = registry.load_state()
    assert state["sequence"] == 1
