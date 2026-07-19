from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from samga_brain_rw.config import BootstrapConfig
from samga_brain_rw.statistics import (
    Cell,
    CellMatrix,
    confirmation_gate,
    pilot_gate,
    two_way_cluster_bootstrap,
)


PILOT_SUBJECTS = (1, 5, 8)
PILOT_SEEDS = (42, 43)
CONFIRMATION_SUBJECTS = tuple(range(1, 11))
CONFIRMATION_SEEDS = tuple(range(42, 47))


def _h(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _bootstrap() -> BootstrapConfig:
    return BootstrapConfig(
        samples=10_000,
        seed=20260719,
        resampling=(
            "independent_subject_and_seed_indices_with_replacement_cartesian_mean"
        ),
        quantile_method="linear",
    )


def _matrix(
    *,
    role: str,
    top1: np.ndarray,
    top5: np.ndarray,
    scope: str,
    subjects: tuple[int, ...],
    seeds: tuple[int, ...],
    architecture_id: str = "shared-architecture",
    shared_name: str | None = None,
    shared_hash_label: str | None = None,
    specific_prefix: str | None = None,
) -> CellMatrix:
    cells: list[Cell] = []
    for subject_index, subject in enumerate(subjects):
        for seed_index, seed in enumerate(seeds):
            coordinate = f"sub{subject:02d}-seed{seed}"
            cells.append(
                Cell(
                    subject=subject,
                    seed=seed,
                    top1=float(top1[subject_index, seed_index]),
                    top5=float(top5[subject_index, seed_index]),
                    effective_batch_size=32,
                    optimization_steps=120,
                    scope=scope,
                    split_role=scope,
                    data_order_sha256=_h(f"order-{coordinate}"),
                    architecture_id=architecture_id,
                    full_task_initialization_sha256=_h(f"full-{coordinate}"),
                    shared_parameter_intersection_name=shared_name,
                    shared_parameter_intersection_sha256=(
                        _h(f"{shared_hash_label}-{coordinate}")
                        if shared_hash_label is not None
                        else None
                    ),
                    architecture_specific_initialization_sha256=(
                        _h(f"{specific_prefix}-{coordinate}")
                        if specific_prefix is not None
                        else None
                    ),
                )
            )
    config_id = f"{role}-config"
    return CellMatrix(
        role=role,
        config_id=config_id,
        config_sha256=_h(config_id),
        hyperparameters_sha256=_h(f"{config_id}-hyperparameters"),
        schedule_sha256=_h("paired-schedule"),
        component_sha256s=(_h(f"{config_id}-component"),),
        cells=tuple(cells),
    )


def _paired(
    delta_top1: np.ndarray,
    *,
    delta_top5: np.ndarray | None = None,
    scope: str = "val-dev",
    subjects: tuple[int, ...] = PILOT_SUBJECTS,
    seeds: tuple[int, ...] = PILOT_SEEDS,
) -> tuple[CellMatrix, CellMatrix]:
    shape = (len(subjects), len(seeds))
    control_top1 = np.full(shape, 0.50, dtype=np.float64)
    control_top5 = np.full(shape, 0.90, dtype=np.float64)
    top5_delta = (
        np.zeros(shape, dtype=np.float64)
        if delta_top5 is None
        else np.asarray(delta_top5, dtype=np.float64)
    )
    control = _matrix(
        role="control",
        top1=control_top1,
        top5=control_top5,
        scope=scope,
        subjects=subjects,
        seeds=seeds,
    )
    candidate = _matrix(
        role="candidate",
        top1=control_top1 + np.asarray(delta_top1, dtype=np.float64),
        top5=control_top5 + top5_delta,
        scope=scope,
        subjects=subjects,
        seeds=seeds,
    )
    return candidate, control


def _replace_cell(
    matrix: CellMatrix,
    index: int,
    **changes: object,
) -> CellMatrix:
    cells = list(matrix.cells)
    cells[index] = replace(cells[index], **changes)
    return replace(matrix, cells=tuple(cells))


def test_two_way_bootstrap_matches_locked_draw_by_draw_cartesian_algorithm() -> None:
    delta = np.array(
        [[-0.01, 0.02], [0.03, 0.04], [0.05, -0.02]],
        dtype=np.float64,
    )
    rng = np.random.default_rng(20260719)
    draws = np.empty(10_000, dtype=np.float64)
    for index in range(10_000):
        subject_indices = rng.integers(0, 3, size=3)
        seed_indices = rng.integers(0, 2, size=2)
        draws[index] = delta[np.ix_(subject_indices, seed_indices)].mean()
    expected = tuple(
        float(value)
        for value in np.quantile(
            draws,
            (0.025, 0.975),
            method="linear",
        )
    )

    assert two_way_cluster_bootstrap(delta, _bootstrap()) == expected


@pytest.mark.parametrize(
    "config",
    [
        replace(_bootstrap(), samples=9_999),
        replace(_bootstrap(), seed=20260720),
        replace(_bootstrap(), resampling="rows_only"),
        replace(_bootstrap(), quantile_method="nearest"),
    ],
)
def test_two_way_bootstrap_rejects_any_protocol_drift(
    config: BootstrapConfig,
) -> None:
    with pytest.raises(ValueError, match="locked"):
        two_way_cluster_bootstrap(np.ones((3, 2)), config)


@pytest.mark.parametrize(
    "delta",
    [
        np.ones(3),
        np.empty((0, 2)),
        np.array([[np.nan, 0.0]]),
        np.array([[np.inf, 0.0]]),
    ],
)
def test_two_way_bootstrap_rejects_invalid_delta_matrix(
    delta: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        two_way_cluster_bootstrap(delta, _bootstrap())


def test_stage1_pilot_gate_uses_all_inclusive_boundaries() -> None:
    delta_top1 = np.array(
        [[0.010, 0.010], [0.010, 0.010], [-0.020, -0.002]]
    )
    # Exactly +0.003 overall, exactly four positive cells, and exactly -0.011
    # for the worst subject. Top-5 is exactly the inclusive -0.002 boundary.
    candidate, control = _paired(
        delta_top1,
        delta_top5=np.full((3, 2), -0.002),
    )

    decision = pilot_gate(candidate, control, stage=1)

    assert decision.passed is True
    assert decision.mean_top1_delta == pytest.approx(0.003, abs=1e-15)
    assert decision.mean_top5_delta == pytest.approx(-0.002, abs=1e-15)
    assert decision.positive_cells == 4
    assert decision.worst_subject_top1_delta == pytest.approx(-0.011)
    assert all(decision.criteria.values())


def test_other_pilot_requires_inclusive_point_zero_zero_five() -> None:
    candidate, control = _paired(np.full((3, 2), 0.005))
    assert pilot_gate(candidate, control, stage=2).passed is True

    below = np.full((3, 2), 0.005)
    below[0, 0] = 0.004999999999
    candidate, control = _paired(below)
    decision = pilot_gate(candidate, control, stage=2)
    assert decision.passed is False
    assert decision.criteria["mean_top1_delta"] is False


def test_pilot_requires_four_strictly_positive_cells() -> None:
    delta = np.array([[0.012, 0.012], [0.012, 0.0], [-0.002, -0.004]])
    candidate, control = _paired(delta)
    decision = pilot_gate(candidate, control, stage=1)
    assert decision.positive_cells == 3
    assert decision.criteria["positive_cells"] is False
    assert decision.passed is False


def test_pilot_subject_floor_is_inclusive_and_top5_below_boundary_fails() -> None:
    delta = np.array([[0.020, 0.020], [0.020, 0.020], [-0.020, -0.020]])
    candidate, control = _paired(delta)
    assert pilot_gate(candidate, control, stage=1).criteria["subject_floor"] is True

    candidate, control = _paired(
        delta,
        delta_top5=np.full((3, 2), -0.002000000001),
    )
    decision = pilot_gate(candidate, control, stage=1)
    assert decision.criteria["mean_top5_delta"] is False
    assert decision.passed is False


def test_confirmation_uses_all_locked_boundaries_and_exact_grid() -> None:
    delta = np.full((10, 5), 0.006)
    delta[8:, :] = 0.001
    candidate, control = _paired(
        delta,
        delta_top5=np.full((10, 5), -0.002),
        scope="val-confirm",
        subjects=CONFIRMATION_SUBJECTS,
        seeds=CONFIRMATION_SEEDS,
    )

    decision = confirmation_gate(candidate, control, _bootstrap())

    assert decision.passed is True
    assert decision.mean_top1_delta == pytest.approx(0.005)
    assert decision.positive_subjects == 10
    assert decision.ci95 is not None
    assert decision.ci95[0] > 0.0
    assert decision.worst_subject_top1_delta == pytest.approx(0.001)
    assert all(decision.criteria.values())


def test_confirmation_requires_eight_strictly_positive_subject_means() -> None:
    delta = np.full((10, 5), 0.008)
    delta[7:, :] = 0.0
    candidate, control = _paired(
        delta,
        scope="val-confirm",
        subjects=CONFIRMATION_SUBJECTS,
        seeds=CONFIRMATION_SEEDS,
    )
    decision = confirmation_gate(candidate, control, _bootstrap())
    assert decision.positive_subjects == 7
    assert decision.criteria["positive_subjects"] is False
    assert decision.passed is False


def test_confirmation_ci_lower_bound_is_strictly_greater_than_zero() -> None:
    delta = np.zeros((10, 5))
    candidate, control = _paired(
        delta,
        scope="val-confirm",
        subjects=CONFIRMATION_SUBJECTS,
        seeds=CONFIRMATION_SEEDS,
    )
    decision = confirmation_gate(candidate, control, _bootstrap())
    assert decision.ci95 == (0.0, 0.0)
    assert decision.criteria["ci95_lower"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda matrix: replace(matrix, cells=matrix.cells[:-1]), "missing"),
        (
            lambda matrix: replace(
                matrix,
                cells=matrix.cells + (matrix.cells[0],),
            ),
            "duplicate",
        ),
        (
            lambda matrix: _replace_cell(matrix, 0, subject=2),
            "subject",
        ),
        (
            lambda matrix: _replace_cell(matrix, 0, seed=44),
            "seed",
        ),
        (
            lambda matrix: _replace_cell(
                matrix,
                0,
                split_role="val-confirm",
            ),
            "split role",
        ),
    ],
)
def test_pilot_rejects_incomplete_duplicate_or_mixed_grid(
    mutation: object,
    message: str,
) -> None:
    candidate, control = _paired(np.full((3, 2), 0.01))
    with pytest.raises(ValueError, match=message):
        pilot_gate(mutation(candidate), control, stage=1)  # type: ignore[operator]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("effective_batch_size", 16, "batch"),
        ("optimization_steps", 121, "steps"),
        ("scope", "val-confirm", "scope"),
        ("data_order_sha256", "f" * 64, "data-order"),
        ("full_task_initialization_sha256", "e" * 64, "initialization"),
    ],
)
def test_pairing_rejects_mismatched_training_or_same_architecture_initialization(
    field: str,
    value: object,
    message: str,
) -> None:
    candidate, control = _paired(np.full((3, 2), 0.01))
    candidate = _replace_cell(candidate, 0, **{field: value})
    with pytest.raises(ValueError, match=message):
        pilot_gate(candidate, control, stage=1)


def test_pairing_rejects_mismatched_schedule_hashes() -> None:
    candidate, control = _paired(np.full((3, 2), 0.01))
    candidate = replace(
        candidate,
        schedule_sha256=_h("changed-candidate-schedule"),
    )
    with pytest.raises(ValueError, match="schedule"):
        pilot_gate(candidate, control, stage=1)


def test_different_architectures_use_only_named_shared_intersection_hash() -> None:
    shape = (3, 2)
    control = _matrix(
        role="control",
        top1=np.full(shape, 0.50),
        top5=np.full(shape, 0.90),
        scope="val-dev",
        subjects=PILOT_SUBJECTS,
        seeds=PILOT_SEEDS,
        architecture_id="control-architecture",
        shared_name="task_model.backbone",
        shared_hash_label="shared",
        specific_prefix="control-specific",
    )
    candidate = _matrix(
        role="candidate",
        top1=np.full(shape, 0.51),
        top5=np.full(shape, 0.90),
        scope="val-dev",
        subjects=PILOT_SUBJECTS,
        seeds=PILOT_SEEDS,
        architecture_id="candidate-architecture",
        shared_name="task_model.backbone",
        shared_hash_label="shared",
        specific_prefix="candidate-specific",
    )
    # Deliberately unequal whole-state hashes must not be compared across
    # unlike architectures.
    candidate = _replace_cell(
        candidate,
        0,
        full_task_initialization_sha256=_h("candidate-whole-state"),
    )
    control = _replace_cell(
        control,
        0,
        full_task_initialization_sha256=_h("control-whole-state"),
    )

    decision = pilot_gate(candidate, control, stage=1)

    assert decision.passed is True
    evidence = decision.initialization_evidence[0]
    assert evidence.mode == "shared-intersection"
    assert evidence.shared_parameter_intersection_name == "task_model.backbone"
    assert evidence.candidate_specific_initialization_sha256 != (
        evidence.control_specific_initialization_sha256
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("shared_parameter_intersection_name", "other.parameters", "named"),
        ("shared_parameter_intersection_sha256", "e" * 64, "shared"),
        ("architecture_specific_initialization_sha256", None, "specific"),
    ],
)
def test_different_architectures_reject_invalid_shared_pairing(
    field: str,
    value: object,
    message: str,
) -> None:
    shape = (3, 2)
    control = _matrix(
        role="control",
        top1=np.full(shape, 0.50),
        top5=np.full(shape, 0.90),
        scope="val-dev",
        subjects=PILOT_SUBJECTS,
        seeds=PILOT_SEEDS,
        architecture_id="control-architecture",
        shared_name="shared.parameters",
        shared_hash_label="shared",
        specific_prefix="control-specific",
    )
    candidate = _matrix(
        role="candidate",
        top1=np.full(shape, 0.51),
        top5=np.full(shape, 0.90),
        scope="val-dev",
        subjects=PILOT_SUBJECTS,
        seeds=PILOT_SEEDS,
        architecture_id="candidate-architecture",
        shared_name="shared.parameters",
        shared_hash_label="shared",
        specific_prefix="candidate-specific",
    )
    candidate = _replace_cell(candidate, 0, **{field: value})
    with pytest.raises(ValueError, match=message):
        pilot_gate(candidate, control, stage=1)


def _run_aggregate(
    experiment_root: Path,
    candidate_path: Path,
    control_path: Path,
    output: Path,
    *extra: str,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(experiment_root)
    return subprocess.run(
        [
            sys.executable,
            str(experiment_root / "scripts" / "aggregate_stage.py"),
            "--candidate-matrix",
            str(candidate_path),
            "--control-matrix",
            str(control_path),
            "--stage",
            "1",
            "--gate",
            "pilot",
            "--output",
            str(output),
            *extra,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_aggregate_cli_accepts_only_typed_json_and_writes_exclusively(
    experiment_root: Path,
    tmp_path: Path,
) -> None:
    candidate, control = _paired(np.full((3, 2), 0.01))
    candidate_path = tmp_path / "candidate.json"
    control_path = tmp_path / "control.json"
    candidate_path.write_text(
        json.dumps(candidate.to_document()) + "\n",
        encoding="utf-8",
    )
    control_path.write_text(
        json.dumps(control.to_document()) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "decision.json"

    completed = _run_aggregate(
        experiment_root,
        candidate_path,
        control_path,
        output,
    )
    assert completed.returncode == 0, completed.stderr
    document = json.loads(output.read_text("utf-8"))
    assert document["artifact_type"] == "samga_brain_rw.candidate_decision"
    assert document["scope"] == "val-dev"
    assert document["gate"]["passed"] is True

    original = output.read_bytes()
    repeated = _run_aggregate(
        experiment_root,
        candidate_path,
        control_path,
        output,
    )
    assert repeated.returncode != 0
    assert output.read_bytes() == original

    raw = tmp_path / "raw.csv"
    raw.write_text("subject,seed,top1\n1,42,1\n", encoding="utf-8")
    rejected = _run_aggregate(experiment_root, raw, control_path, tmp_path / "x.json")
    assert rejected.returncode != 0
    assert not (tmp_path / "x.json").exists()


def test_aggregate_cli_rejects_symlink_and_formal_paths_before_input_read(
    experiment_root: Path,
    tmp_path: Path,
) -> None:
    candidate, control = _paired(np.full((3, 2), 0.01))
    target = tmp_path / "candidate.json"
    target.write_text(json.dumps(candidate.to_document()), encoding="utf-8")
    symlink = tmp_path / "candidate-link.json"
    symlink.symlink_to(target)
    control_path = tmp_path / "control.json"
    control_path.write_text(json.dumps(control.to_document()), encoding="utf-8")

    linked = _run_aggregate(
        experiment_root,
        symlink,
        control_path,
        tmp_path / "linked-output.json",
    )
    assert linked.returncode != 0
    assert not (tmp_path / "linked-output.json").exists()

    formal = tmp_path / "formal-test" / "output.json"
    rejected = _run_aggregate(
        experiment_root,
        target,
        control_path,
        formal,
    )
    assert rejected.returncode != 0
    assert not formal.exists()
