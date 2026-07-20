from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from samga_brain_rw import stage1_finalizer as finalizer
from samga_brain_rw.hashing import canonical_json_bytes, sha256_json
from samga_brain_rw.registry import CandidateDecision, CandidateRegistry
from samga_brain_rw.stage1 import (
    COMPOSITION_OUTCOME_TYPE,
    PILOT_COORDINATES,
)
from samga_brain_rw.statistics import GateDecision


def _h(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _gate(passed: bool, *, stage: int = 1) -> GateDecision:
    delta = 0.02 if passed else -0.01
    return GateDecision(
        gate_kind="pilot",
        stage=stage,
        passed=passed,
        mean_top1_delta=delta,
        mean_top5_delta=delta / 2,
        ci95=None,
        positive_cells=6 if passed else 0,
        positive_subjects=3 if passed else 0,
        worst_subject_top1_delta=delta,
        subject_mean_top1_deltas=tuple((subject, delta) for subject in (1, 5, 8)),
        criteria={"fixture_gate": passed},
        initialization_evidence=(),
    )


class _FakeOutcome:
    def __init__(self, *, passed: bool, variant: str = "base") -> None:
        self.passed = passed
        self.status = "passed" if passed else "failed"
        self.control_branch_id = "internvit"
        self.winner_config_id = f"fusion-fixture-{variant}-v1"
        self.control_top1 = (0.50, 0.51, 0.52, 0.53, 0.54, 0.55)
        self.control_top5 = (0.70, 0.71, 0.72, 0.73, 0.74, 0.75)
        offset = 0.02 if passed else -0.01
        self.winner_top1 = tuple(value + offset for value in self.control_top1)
        self.winner_top5 = tuple(value + offset / 2 for value in self.control_top5)
        self.gate = _gate(passed)
        self.dependency_sha256 = _h(f"dependency-{variant}")
        self.control_sha256 = _h(f"control-{variant}")
        self.selection_sha256 = _h(f"selection-{variant}")
        self.composite_sha256 = _h(f"composite-{variant}")
        self.composition_spec_sha256 = _h(f"composition-spec-{variant}")
        self.evidence_sha256 = _h(f"evidence-{variant}")
        self._decision = CandidateDecision(
            stage=1,
            candidate_id=self.winner_config_id,
            control_id=self.control_branch_id,
            scope="val-dev",
            config_sha256=_h("semantic-config"),
            control_config_sha256=_h("control-config"),
            hyperparameters_sha256=_h(f"hyperparameters-{variant}"),
            schedule_sha256=self.composition_spec_sha256,
            component_sha256s=tuple(_h(f"component-{index}") for index in range(12)),
            candidate_matrix_sha256=self.composite_sha256,
            control_matrix_sha256=self.control_sha256,
            absolute_top1=sum(self.winner_top1) / len(self.winner_top1),
            absolute_top5=sum(self.winner_top5) / len(self.winner_top5),
            gate=self.gate,
        )

    def candidate_decision_document(self) -> dict[str, object]:
        return self._decision.to_document()

    def to_payload(self) -> dict[str, object]:
        return {
            "artifact_type": COMPOSITION_OUTCOME_TYPE,
            "candidate_decision": self.candidate_decision_document(),
            "composite": {"fixture": "composite"},
            "composite_sha256": self.composite_sha256,
            "composition_spec": {"fixture": "composition-spec"},
            "composition_spec_sha256": self.composition_spec_sha256,
            "control": {"fixture": "control"},
            "control_branch_id": self.control_branch_id,
            "control_sha256": self.control_sha256,
            "control_top1": list(self.control_top1),
            "control_top5": list(self.control_top5),
            "dependency": {"fixture": "dependency"},
            "dependency_sha256": self.dependency_sha256,
            "evidence": {"fixture": "evidence"},
            "evidence_sha256": self.evidence_sha256,
            "gate": self.gate.to_payload(),
            "passed": self.passed,
            "schema_version": 1,
            "scope": "val-dev",
            "selection": {"fixture": "selection"},
            "selection_sha256": self.selection_sha256,
            "stage": 1,
            "status": self.status,
            "winner_config_id": self.winner_config_id,
            "winner_top1": list(self.winner_top1),
            "winner_top5": list(self.winner_top5),
        }


class _FakeCell:
    def __init__(
        self,
        subject: int,
        seed: int,
        events: list[str],
    ) -> None:
        self.subject = subject
        self.seed = seed
        self._events = events

    def revalidate(self) -> None:
        self._events.append(f"cell:{self.subject}:{self.seed}")


class _FakeCost:
    def __init__(self, events: list[str], *, fail_revalidation: bool = False) -> None:
        self._events = events
        self._fail_revalidation = fail_revalidation

    def revalidate(self) -> None:
        self._events.append("cost:revalidate")
        if self._fail_revalidation:
            raise ValueError("injected cost revalidation failure")


def _install_fake_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    *,
    project_root: Path,
    cost_job_map: Path,
    cost_job_map_sha256: str,
    outcome: _FakeOutcome,
    fail_cost_revalidation: bool = False,
) -> tuple[tuple[_FakeCell, ...], _FakeCost, list[str]]:
    events: list[str] = []
    semantic = SimpleNamespace(sha256=_h("semantic-config"))
    cells = tuple(
        _FakeCell(subject, seed, events) for subject, seed in PILOT_COORDINATES
    )
    cost = _FakeCost(events, fail_revalidation=fail_cost_revalidation)

    def from_path(path: Path) -> object:
        assert path == (
            project_root / "experiments/samga_brain_rw/configs/stage1_fusion_v1.json"
        )
        events.append("semantic:load")
        return semantic

    def load_cells(root: Path, loaded_semantic: object) -> tuple[_FakeCell, ...]:
        assert root == project_root
        assert loaded_semantic is semantic
        events.append("cells:load")
        return cells

    def load_cost(path: Path, expected_sha256: str) -> _FakeCost:
        assert path == cost_job_map
        assert expected_sha256 == cost_job_map_sha256
        events.append("cost:load")
        return cost

    def compose(
        loaded_cells: tuple[_FakeCell, ...],
        *,
        semantic_config: object,
        cost_capability: _FakeCost,
    ) -> _FakeOutcome:
        assert loaded_cells is cells
        assert semantic_config is semantic
        assert cost_capability is cost
        events.append("compose")
        return outcome

    monkeypatch.setattr(
        finalizer,
        "SemanticConfig",
        SimpleNamespace(from_path=from_path),
    )
    monkeypatch.setattr(finalizer, "load_stage1_composition_cells", load_cells)
    monkeypatch.setattr(
        finalizer,
        "load_validated_stage1_cost_capability",
        load_cost,
    )
    monkeypatch.setattr(finalizer, "compose_stage1", compose)
    return cells, cost, events


def _paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "project_root": tmp_path / "project",
        "cost_job_map": tmp_path / "inputs" / "cost-job-map.json",
        "journal": tmp_path / "registry" / "candidates.jsonl",
        "state": tmp_path / "registry" / "candidates.state.json",
        "output_dir": tmp_path / "results",
    }


def _finalize(
    paths: dict[str, Path],
    cost_job_map_sha256: str,
) -> finalizer.Stage1FinalizationResult:
    return finalizer.finalize_stage1(
        project_root=paths["project_root"],
        cost_job_map_path=paths["cost_job_map"],
        cost_job_map_sha256=cost_job_map_sha256,
        journal_path=paths["journal"],
        state_path=paths["state"],
        output_dir=paths["output_dir"],
    )


def test_passing_gate_publishes_full_summary_six_rows_and_locked_survivor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    paths["project_root"].mkdir()
    outcome = _FakeOutcome(passed=True)
    expected_cost_sha256 = _h("cost-job-map")
    _, _, events = _install_fake_pipeline(
        monkeypatch,
        project_root=paths["project_root"],
        cost_job_map=paths["cost_job_map"],
        cost_job_map_sha256=expected_cost_sha256,
        outcome=outcome,
    )

    result = _finalize(paths, expected_cost_sha256)

    assert result.passed is True
    assert (paths["project_root"] / finalizer.FINALIZER_LOCK_RELATIVE_PATH).is_file()
    assert result.summary_path == paths["output_dir"] / "pilot_summary.json"
    assert result.cells_path == paths["output_dir"] / "pilot_cells.csv"
    assert result.locked_survivor_path == (paths["output_dir"] / "locked_survivor.json")
    summary_raw = result.summary_path.read_bytes()
    summary = json.loads(summary_raw)
    assert summary_raw == canonical_json_bytes(summary) + b"\n"
    assert summary["artifact_type"] == "samga_brain_rw.stage1_pilot_summary"
    assert summary["outcome"] == outcome.to_payload()
    assert summary["outcome_sha256"] == sha256_json(outcome.to_payload())
    assert summary["development_decision_sha256"] == (outcome._decision.decision_sha256)
    assert summary["locked_decision_sha256"] == result.locked_decision_sha256

    cells_raw = result.cells_path.read_bytes()
    rows = list(csv.DictReader(io.StringIO(cells_raw.decode("utf-8"))))
    assert len(rows) == 6
    assert list(rows[0]) == [
        "subject",
        "seed",
        "control_top1",
        "control_top5",
        "winner_top1",
        "winner_top5",
        "top1_delta",
        "top5_delta",
    ]
    assert [(int(row["subject"]), int(row["seed"])) for row in rows] == list(
        PILOT_COORDINATES
    )
    assert float(rows[0]["top1_delta"]) == pytest.approx(0.02)
    assert float(rows[0]["top5_delta"]) == pytest.approx(0.01)
    assert (
        summary["artifacts"]["pilot_cells.csv"]["sha256"]
        == hashlib.sha256(cells_raw).hexdigest()
    )

    locked_raw = result.locked_survivor_path.read_bytes()
    locked = json.loads(locked_raw)
    assert locked_raw == canonical_json_bytes(locked) + b"\n"
    assert locked["artifact_type"] == "samga_brain_rw.stage1_locked_survivor"
    locked_decision = CandidateDecision.from_payload(locked["decision"])
    assert locked_decision.locked is True
    assert locked["development_decision_sha256"] == (outcome._decision.decision_sha256)
    assert locked["decision_sha256"] == locked_decision.decision_sha256
    assert summary["artifacts"]["locked_survivor.json"]["sha256"] == (
        hashlib.sha256(locked_raw).hexdigest()
    )

    registry = CandidateRegistry(
        paths["journal"],
        paths["state"],
    )
    registry_state = registry.load_state()
    survivor = registry_state["stages"]["1"]["survivor"]
    assert survivor["decision_sha256"] == locked_decision.decision_sha256
    survivor_sha256 = sha256_json(survivor)
    assert locked["registry_survivor_sha256"] == survivor_sha256
    assert (
        summary["artifacts"]["locked_survivor.json"]["registry_survivor_sha256"]
        == survivor_sha256
    )
    validated = finalizer.validate_stage1_locked_survivor_document(
        locked,
        registry=registry,
    )
    assert validated.decision_sha256 == locked_decision.decision_sha256
    boolean_schema = dict(locked)
    boolean_schema["schema_version"] = True
    with pytest.raises(ValueError, match="type/version"):
        finalizer.validate_stage1_locked_survivor_document(
            boolean_schema,
            registry=registry,
        )
    changed_state = dict(registry_state)
    changed_state["state_sha256"] = _h("forged-registry-state")
    monkeypatch.setattr(registry, "load_state", lambda: changed_state)
    with pytest.raises(ValueError, match="state SHA-256"):
        finalizer.validate_stage1_locked_survivor_document(
            locked,
            registry=registry,
        )
    assert events[-7:] == [
        "cell:1:42",
        "cell:1:43",
        "cell:5:42",
        "cell:5:43",
        "cell:8:42",
        "cell:8:43",
        "cost:revalidate",
    ]


def test_failed_gate_publishes_summary_without_locking(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    paths["project_root"].mkdir()
    outcome = _FakeOutcome(passed=False)
    expected_cost_sha256 = _h("cost-job-map")
    _install_fake_pipeline(
        monkeypatch,
        project_root=paths["project_root"],
        cost_job_map=paths["cost_job_map"],
        cost_job_map_sha256=expected_cost_sha256,
        outcome=outcome,
    )

    result = _finalize(paths, expected_cost_sha256)

    assert result.passed is False
    assert result.locked_survivor_path is None
    assert result.locked_decision_sha256 is None
    assert not (paths["output_dir"] / "locked_survivor.json").exists()
    summary = json.loads(result.summary_path.read_text("utf-8"))
    assert summary["status"] == "failed"
    assert summary["locked_decision_sha256"] is None
    assert summary["artifacts"]["locked_survivor.json"] is None

    journal_lines = paths["journal"].read_bytes().splitlines()
    assert len(journal_lines) == 1
    state = CandidateRegistry(paths["journal"], paths["state"]).load_state()
    assert state["stages"]["1"]["survivor"] is None


def test_exact_rerun_reuses_registry_and_outputs_byte_for_byte(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    paths["project_root"].mkdir()
    outcome = _FakeOutcome(passed=True)
    expected_cost_sha256 = _h("cost-job-map")
    _install_fake_pipeline(
        monkeypatch,
        project_root=paths["project_root"],
        cost_job_map=paths["cost_job_map"],
        cost_job_map_sha256=expected_cost_sha256,
        outcome=outcome,
    )
    first = _finalize(paths, expected_cost_sha256)
    expected_bytes = {
        path.name: path.read_bytes()
        for path in (
            first.summary_path,
            first.cells_path,
            first.locked_survivor_path,
        )
        if path is not None
    }
    journal_before = paths["journal"].read_bytes()
    locked_before = json.loads(first.locked_survivor_path.read_text("utf-8"))
    locked_survivor_sha256 = locked_before["registry_survivor_sha256"]

    registry = CandidateRegistry(paths["journal"], paths["state"])
    stage1_state_sha256 = registry.load_state()["state_sha256"]
    registry.append_or_reuse_exact(
        CandidateDecision(
            stage=2,
            candidate_id="later-stage-candidate",
            control_id="later-stage-control",
            scope="val-dev",
            config_sha256=_h("later-config"),
            control_config_sha256=_h("later-control-config"),
            hyperparameters_sha256=_h("later-hyperparameters"),
            schedule_sha256=_h("later-schedule"),
            component_sha256s=(_h("later-component"),),
            candidate_matrix_sha256=_h("later-candidate-matrix"),
            control_matrix_sha256=_h("later-control-matrix"),
            absolute_top1=0.4,
            absolute_top5=0.6,
            gate=_gate(False, stage=2),
        )
    )
    advanced_journal = paths["journal"].read_bytes()
    advanced_state = paths["state"].read_bytes()
    advanced_registry_state = registry.load_state()
    assert advanced_registry_state["state_sha256"] != stage1_state_sha256
    assert (
        sha256_json(advanced_registry_state["stages"]["1"]["survivor"])
        == locked_survivor_sha256
    )

    second = _finalize(paths, expected_cost_sha256)

    assert second == first
    assert paths["journal"].read_bytes() == advanced_journal
    assert paths["state"].read_bytes() == advanced_state
    assert {
        path.name: path.read_bytes()
        for path in (
            second.summary_path,
            second.cells_path,
            second.locked_survivor_path,
        )
        if path is not None
    } == expected_bytes
    assert len(journal_before.splitlines()) == 2
    assert len(advanced_journal.splitlines()) == 3


def test_finalizer_mutex_serializes_divergent_concurrent_invocations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    paths["project_root"].mkdir()
    expected_cost_sha256 = _h("cost-job-map")
    first_outcome = _FakeOutcome(passed=True, variant="first")
    second_outcome = _FakeOutcome(passed=True, variant="second")
    _install_fake_pipeline(
        monkeypatch,
        project_root=paths["project_root"],
        cost_job_map=paths["cost_job_map"],
        cost_job_map_sha256=expected_cost_sha256,
        outcome=first_outcome,
    )
    first_in_compose = threading.Event()
    release_first = threading.Event()
    second_in_compose = threading.Event()
    compose_guard = threading.Lock()
    compose_calls = 0

    def compose(
        loaded_cells: tuple[_FakeCell, ...],
        *,
        semantic_config: object,
        cost_capability: _FakeCost,
    ) -> _FakeOutcome:
        del loaded_cells, semantic_config, cost_capability
        nonlocal compose_calls
        with compose_guard:
            compose_calls += 1
            call = compose_calls
        if call == 1:
            first_in_compose.set()
            assert release_first.wait(timeout=5)
            return first_outcome
        second_in_compose.set()
        return second_outcome

    monkeypatch.setattr(finalizer, "compose_stage1", compose)

    def invoke() -> object:
        try:
            return _finalize(paths, expected_cost_sha256)
        except Exception as exc:  # noqa: BLE001 - thread result is asserted below
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(invoke)
        assert first_in_compose.wait(timeout=5)
        second = executor.submit(invoke)
        second_entered_while_first_held = second_in_compose.wait(timeout=0.5)
        release_first.set()
        first_result = first.result(timeout=10)
        second_result = second.result(timeout=10)

    assert not second_entered_while_first_held
    assert isinstance(first_result, finalizer.Stage1FinalizationResult)
    assert isinstance(second_result, ValueError)
    assert "divergent existing output" in str(second_result)
    assert len(paths["journal"].read_bytes().splitlines()) == 2


def test_retry_recovers_crash_between_development_append_and_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    paths["project_root"].mkdir()
    outcome = _FakeOutcome(passed=True)
    expected_cost_sha256 = _h("cost-job-map")
    _install_fake_pipeline(
        monkeypatch,
        project_root=paths["project_root"],
        cost_job_map=paths["cost_job_map"],
        cost_job_map_sha256=expected_cost_sha256,
        outcome=outcome,
    )
    real_lock = CandidateRegistry.lock_stage_survivor_or_reuse_exact
    injected = False

    def crash_once(
        registry: CandidateRegistry,
        stage: int,
        expected_development_decision_sha256: str,
    ) -> CandidateDecision:
        nonlocal injected
        if not injected:
            injected = True
            raise OSError("injected append-to-lock crash")
        return real_lock(
            registry,
            stage,
            expected_development_decision_sha256,
        )

    monkeypatch.setattr(
        CandidateRegistry,
        "lock_stage_survivor_or_reuse_exact",
        crash_once,
    )
    with pytest.raises(OSError, match="append-to-lock"):
        _finalize(paths, expected_cost_sha256)
    assert len(paths["journal"].read_bytes().splitlines()) == 1
    assert not paths["output_dir"].exists()

    result = _finalize(paths, expected_cost_sha256)

    assert result.passed
    assert len(paths["journal"].read_bytes().splitlines()) == 2
    assert result.summary_path.exists()
    assert result.cells_path.exists()
    assert result.locked_survivor_path is not None
    assert result.locked_survivor_path.exists()


@pytest.mark.parametrize("crash_on_publish", [1, 2, 3])
def test_retry_recovers_each_output_publication_crash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    crash_on_publish: int,
) -> None:
    paths = _paths(tmp_path)
    paths["project_root"].mkdir()
    outcome = _FakeOutcome(passed=True)
    expected_cost_sha256 = _h("cost-job-map")
    _install_fake_pipeline(
        monkeypatch,
        project_root=paths["project_root"],
        cost_job_map=paths["cost_job_map"],
        cost_job_map_sha256=expected_cost_sha256,
        outcome=outcome,
    )
    real_publish = finalizer._publish_create_or_verify_identical
    publish_calls = 0
    injected = False

    def crash_once(path: Path, payload: bytes) -> None:
        nonlocal injected, publish_calls
        publish_calls += 1
        if not injected and publish_calls == crash_on_publish:
            injected = True
            raise OSError(f"injected publish-{crash_on_publish} crash")
        real_publish(path, payload)

    monkeypatch.setattr(
        finalizer,
        "_publish_create_or_verify_identical",
        crash_once,
    )
    with pytest.raises(OSError, match=f"publish-{crash_on_publish}"):
        _finalize(paths, expected_cost_sha256)
    journal_before = paths["journal"].read_bytes()
    state_before = paths["state"].read_bytes()
    assert len(journal_before.splitlines()) == 2

    monkeypatch.setattr(
        finalizer,
        "_publish_create_or_verify_identical",
        real_publish,
    )
    result = _finalize(paths, expected_cost_sha256)

    assert result.passed
    assert paths["journal"].read_bytes() == journal_before
    assert paths["state"].read_bytes() == state_before
    assert result.summary_path.exists()
    assert result.cells_path.exists()
    assert result.locked_survivor_path is not None
    assert result.locked_survivor_path.exists()


def test_divergent_existing_output_fails_without_overwrite_or_registry_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    paths["project_root"].mkdir()
    paths["output_dir"].mkdir()
    summary = paths["output_dir"] / "pilot_summary.json"
    original = b'{"divergent":true}\n'
    summary.write_bytes(original)
    outcome = _FakeOutcome(passed=True)
    expected_cost_sha256 = _h("cost-job-map")
    _install_fake_pipeline(
        monkeypatch,
        project_root=paths["project_root"],
        cost_job_map=paths["cost_job_map"],
        cost_job_map_sha256=expected_cost_sha256,
        outcome=outcome,
    )

    with pytest.raises(ValueError, match="divergent.*pilot_summary"):
        _finalize(paths, expected_cost_sha256)

    assert summary.read_bytes() == original
    assert not paths["journal"].exists()
    assert not paths["state"].exists()
    assert not (paths["output_dir"] / "pilot_cells.csv").exists()
    assert not (paths["output_dir"] / "locked_survivor.json").exists()


def test_cost_revalidation_failure_precedes_every_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    paths["project_root"].mkdir()
    outcome = _FakeOutcome(passed=True)
    expected_cost_sha256 = _h("cost-job-map")
    _, _, events = _install_fake_pipeline(
        monkeypatch,
        project_root=paths["project_root"],
        cost_job_map=paths["cost_job_map"],
        cost_job_map_sha256=expected_cost_sha256,
        outcome=outcome,
        fail_cost_revalidation=True,
    )

    with pytest.raises(ValueError, match="cost revalidation failure"):
        _finalize(paths, expected_cost_sha256)

    assert events[-7:] == [
        "cell:1:42",
        "cell:1:43",
        "cell:5:42",
        "cell:5:43",
        "cell:8:42",
        "cell:8:43",
        "cost:revalidate",
    ]
    assert not paths["journal"].exists()
    assert not paths["state"].exists()
    assert not paths["output_dir"].exists()


def test_finalizer_lock_directory_rejects_symlink_traversal(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    paths["project_root"].mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (paths["project_root"] / "artifacts").symlink_to(
        outside,
        target_is_directory=True,
    )

    with pytest.raises(ValueError, match="lock directory.*symlinks"):
        _finalize(paths, _h("cost-job-map"))

    assert list(outside.iterdir()) == []
    assert not paths["journal"].exists()
    assert not paths["state"].exists()
    assert not paths["output_dir"].exists()


def _load_cli(experiment_root: Path) -> ModuleType:
    path = experiment_root / "scripts/finalize_stage1.py"
    spec = importlib.util.spec_from_file_location("stage1_finalizer_cli_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_exposes_only_fixed_evidence_and_destination_arguments(
    experiment_root: Path,
) -> None:
    cli = _load_cli(experiment_root)
    parser = cli._parser()
    options = {option for action in parser._actions for option in action.option_strings}
    assert options == {
        "-h",
        "--help",
        "--project-root",
        "--cost-job-map",
        "--cost-job-map-sha256",
        "--journal",
        "--state",
        "--output-dir",
    }
    required = {
        action.dest for action in parser._actions if getattr(action, "required", False)
    }
    assert required == {
        "project_root",
        "cost_job_map",
        "cost_job_map_sha256",
        "journal",
        "state",
        "output_dir",
    }
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--project-root",
                "/project",
                "--cost-job-map",
                "/cost.json",
                "--cost-job-map-sha256",
                _h("cost"),
                "--journal",
                "/registry.jsonl",
                "--state",
                "/registry.json",
                "--output-dir",
                "/output",
                "--threshold",
                "0.5",
            ]
        )
