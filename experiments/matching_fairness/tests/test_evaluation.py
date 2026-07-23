from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pytest

from matching_fairness.artifacts import ScoreArtifact
from matching_fairness.decoders import Assignment
from matching_fairness.evaluation import (
    DecoderConfig,
    EvaluationResult,
    evaluate_artifact,
)


SCRIPT = Path("experiments/matching_fairness/scripts/run_scenarios.py")


def test_run_scenarios_direct_cli_help_resolves_package_without_pythonpath() -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=Path.cwd(),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--artifact-root" in completed.stdout


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_scenarios", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_task7_exact_model_entries_require_isolated_native_audits() -> None:
    runner = _load_runner()

    assert runner._expected_model_entries("nice") == {
        "standard", "eeg_a", "eeg_b", "best_test_audit.json"
    }
    assert runner._expected_model_entries("atm_s") == {
        "standard", "eeg_a", "eeg_b", "best_test_audit.json"
    }
    assert runner._expected_model_entries("our_project") == {
        "standard", "eeg_a", "eeg_b", "runs", "export_manifest.json"
    }


def _artifact(
    similarity: np.ndarray,
    targets: tuple[str, ...],
    gallery: tuple[str, ...],
    *,
    query_ids: tuple[str, ...] | None = None,
    metadata: dict[str, object] | None = None,
) -> ScoreArtifact:
    query_ids = query_ids or tuple(f"query-{index}" for index in range(len(targets)))
    artifact = ScoreArtifact(
        similarity=np.asarray(similarity, dtype=np.float64),
        query_ids=query_ids,
        gallery_entry_ids=tuple(f"entry-{index}" for index in range(len(gallery))),
        gallery_canonical_ids=gallery,
        target_canonical_ids=targets,
        metadata=dict(metadata or {}),
    )
    artifact.validate()
    return artifact


def test_unanswerable_and_unmatched_are_wrong_in_overall_top1() -> None:
    artifact = _artifact(
        np.asarray([[1.0], [0.5]]),
        ("a", "missing"),
        ("a",),
        metadata={"allow_unanswerable_targets": True},
    )

    result = evaluate_artifact(artifact, DecoderConfig("hungarian", seed=42))

    assert result.metrics["correct"] == 1
    assert result.metrics["total"] == 2
    assert result.metrics["top1"] == 50.0
    assert result.metrics["answerable_correct"] == 1
    assert result.metrics["answerable_total"] == 1
    assert result.metrics["answerable_top1"] == 100.0
    assert result.metrics["unanswerable_count"] == 1
    assert result.metrics["unmatched_count"] == 1


def test_duplicate_gallery_canonical_id_counts_as_correct() -> None:
    artifact = _artifact(
        np.asarray([[0.1, 0.8, 0.9]]),
        ("image-a",),
        ("image-b", "image-a", "image-a"),
    )

    result = evaluate_artifact(artifact, DecoderConfig("independent"))

    assert result.metrics["correct"] == 1
    assert result.per_query[0]["predicted_canonical_id"] == "image-a"
    assert result.per_query[0]["correct_top1"] is True


def test_unanswerable_query_is_never_counted_in_independent_top5() -> None:
    artifact = _artifact(
        np.asarray([[1.0], [0.5]]),
        ("a", "missing"),
        ("a",),
        metadata={"allow_unanswerable_targets": True},
    )

    result = evaluate_artifact(artifact, DecoderConfig("independent"))

    assert result.metrics["top5_count"] == 1
    assert result.metrics["top5"] == 50.0
    assert result.per_query[1]["correct_top5"] is False


@pytest.mark.parametrize(
    "decoder",
    ("greedy", "hungarian", "stable_matching", "sinkhorn"),
)
def test_only_independent_emits_top5(decoder: str) -> None:
    artifact = _artifact(np.eye(6), tuple("abcdef"), tuple("abcdef"))

    constrained = evaluate_artifact(artifact, DecoderConfig(decoder, seed=42))
    independent = evaluate_artifact(artifact, DecoderConfig("independent"))

    assert constrained.metrics["top5_count"] is None
    assert constrained.metrics["top5"] is None
    assert all(row["correct_top5"] is None for row in constrained.per_query)
    assert independent.metrics["top5_count"] == 6
    assert independent.metrics["top5"] == 100.0


def _duplicate_artifact(repeat_count: int, *, wrong_b_target: int | None = None) -> ScoreArtifact:
    gallery = tuple(f"image-{index:03d}" for index in range(200))
    repeated = tuple(reversed(gallery[:repeat_count]))
    similarity = np.eye(200, dtype=np.float64)
    b_rows = []
    for repeated_index, target in enumerate(repeated):
        row = np.zeros(200, dtype=np.float64)
        target_index = gallery.index(target)
        prediction = (
            (target_index + 1) % 200
            if wrong_b_target == repeated_index
            else target_index
        )
        row[prediction] = 1.0
        b_rows.append(row)
    if b_rows:
        similarity = np.concatenate((similarity, np.stack(b_rows)), axis=0)
    targets = gallery + repeated
    query_ids = gallery + tuple(f"{target}__eeg_b" for target in repeated)
    return _artifact(
        similarity,
        targets,
        gallery,
        query_ids=query_ids,
        metadata={"query_mode": f"dupq{repeat_count}"},
    )


@pytest.mark.parametrize(
    "repeat_count,total,ceiling",
    ((10, 210, 200), (20, 220, 200)),
)
def test_duplicate_query_strict_one_to_one_ceiling(
    repeat_count: int,
    total: int,
    ceiling: int,
) -> None:
    artifact = _duplicate_artifact(repeat_count)

    result = evaluate_artifact(artifact, DecoderConfig("hungarian", seed=42))

    assert result.metrics["total"] == total
    assert result.metrics["theoretical_ceiling_count"] == ceiling
    assert result.metrics["theoretical_ceiling"] == pytest.approx(100.0 * ceiling / total)
    assert result.metrics["unmatched_repeated_queries"] >= 0
    assert result.metrics["distance_from_ceiling"] == ceiling - result.metrics["correct"]


def test_duplicate_pair_metrics_group_by_canonical_id_not_position() -> None:
    artifact = _duplicate_artifact(10, wrong_b_target=3)

    result = evaluate_artifact(artifact, DecoderConfig("independent"))

    assert result.metrics["base_a_correct"] == 200
    assert result.metrics["base_a_total"] == 200
    assert result.metrics["appended_b_correct"] == 9
    assert result.metrics["appended_b_total"] == 10
    assert result.metrics["repeated_canonical_total"] == 10
    assert result.metrics["at_least_one_correct_count"] == 10
    assert result.metrics["both_correct_count"] == 9
    assert result.metrics["both_correct"] == 90.0


def test_decoder_name_must_be_formal() -> None:
    artifact = _artifact(np.eye(2), ("a", "b"), ("a", "b"))

    with pytest.raises(ValueError, match="decoder"):
        evaluate_artifact(artifact, DecoderConfig("oracle"))


def test_runner_builds_exactly_27_standard_and_3_duplicate_scenarios() -> None:
    runner = _load_runner()
    gallery = tuple(f"image-{index:03d}" for index in range(200))
    standard = _artifact(np.eye(200), gallery, gallery, query_ids=gallery)
    a = _artifact(
        np.eye(200),
        gallery,
        gallery,
        query_ids=gallery,
        metadata={"trial_half": "a"},
    )
    b = _artifact(
        np.eye(200) * 2.0,
        gallery,
        gallery,
        query_ids=gallery,
        metadata={"trial_half": "b"},
    )

    configs = runner._decoder_configs(
        seed=42,
        sinkhorn={
            "temperature": 0.05,
            "max_iterations": 500,
            "tolerance": 1e-8,
        },
    )
    plan = runner._build_scenario_plan(
        gallery,
        seed=42,
        trial_manifest_sha256="a" * 64,
        decoder_configs=configs,
    )
    scenarios = runner._scenario_artifacts(standard, a, b, plan=plan)

    assert len(scenarios) == 30
    assert sum(cell.suite == "standard" for cell in scenarios) == 27
    assert sum(cell.suite == "duplicate_eeg" for cell in scenarios) == 3
    assert [cell.index for cell in scenarios] == list(range(30))
    assert [cell.artifact.similarity.shape for cell in scenarios[-3:]] == [
        (200, 200),
        (210, 200),
        (220, 200),
    ]
    assert scenarios[0].selection == plan.standard_selections[0]
    assert (
        scenarios[-1].selection["duplicate_query_ids"]
        == plan.duplicate_query_ids
    )


def test_scenario_plan_is_shared_byte_stable_and_hash_bound_across_models() -> None:
    runner = _load_runner()
    gallery = tuple(f"image-{index:03d}" for index in range(200))
    configs = runner._decoder_configs(
        seed=42,
        sinkhorn={
            "temperature": 0.05,
            "max_iterations": 500,
            "tolerance": 1e-8,
        },
    )

    first = runner._build_scenario_plan(
        gallery,
        seed=42,
        trial_manifest_sha256="b" * 64,
        decoder_configs=configs,
    )
    repeated = runner._build_scenario_plan(
        gallery,
        seed=42,
        trial_manifest_sha256="b" * 64,
        decoder_configs=configs,
    )

    assert first.manifest_bytes == repeated.manifest_bytes
    assert first.manifest_sha256 == repeated.manifest_sha256
    assert re.fullmatch(r"[0-9a-f]{64}", first.manifest_sha256)
    runner._validate_scenario_manifest(
        first.manifest_bytes,
        first.manifest_sha256,
    )
    with pytest.raises(ValueError, match="SHA-256"):
        runner._validate_scenario_manifest(
            first.manifest_bytes + b" ",
            first.manifest_sha256,
        )
    tampered_payload = json.loads(first.manifest_bytes)
    tampered_payload["gallery_canonical_ids_sha256"] = "0" * 64
    tampered_bytes = runner._json_bytes(tampered_payload)
    with pytest.raises(ValueError, match="gallery"):
        runner._validate_scenario_manifest(
            tampered_bytes,
            hashlib.sha256(tampered_bytes).hexdigest(),
        )
    numeric_hash = int("1" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        runner._build_scenario_plan(
            gallery,
            seed=42,
            trial_manifest_sha256=numeric_hash,
            decoder_configs=configs,
        )
    numeric_payload = json.loads(first.manifest_bytes)
    numeric_payload["trial_manifest_sha256"] = numeric_hash
    numeric_bytes = runner._json_bytes(numeric_payload)
    with pytest.raises(ValueError, match="trial-manifest hash"):
        runner._validate_scenario_manifest(
            numeric_bytes,
            hashlib.sha256(numeric_bytes).hexdigest(),
        )

    selections = []
    selection_objects = []
    for scale in (1.0, 2.0, 3.0):
        standard = _artifact(
            scale * np.eye(200),
            gallery,
            gallery,
            query_ids=gallery,
        )
        a = _artifact(scale * np.eye(200), gallery, gallery, query_ids=gallery)
        b = _artifact((scale + 4.0) * np.eye(200), gallery, gallery, query_ids=gallery)
        cells = runner._scenario_artifacts(standard, a, b, plan=first)
        selections.append(tuple(cell.selection for cell in cells))
        selection_objects.append(cells)
    assert selections[0] == selections[1] == selections[2]
    assert all(
        cells[0].selection is first.standard_selections[0]
        for cells in selection_objects
    )
    assert all(
        cells[27].selection is first.duplicate_selections[0]
        for cells in selection_objects
    )


def test_runner_rejects_any_grid_other_than_exact_450_unique_records() -> None:
    runner = _load_runner()
    records = [
        {
            "model": model,
            "scenario_index": scenario_index,
            "decoder": decoder,
        }
        for model in ("nice", "atm_s", "our_project")
        for scenario_index in range(30)
        for decoder in (
            "independent",
            "greedy",
            "hungarian",
            "stable_matching",
            "sinkhorn",
        )
    ]

    runner._validate_record_grid(records)
    with pytest.raises(ValueError, match="450"):
        runner._validate_record_grid(records[:-1])
    with pytest.raises(ValueError, match="unique"):
        runner._validate_record_grid(records[:-1] + [records[0]])


def test_runner_writes_one_summary_and_ledger_under_formal_subject_path(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    artifact = _artifact(np.eye(2), ("a", "b"), ("a", "b"))
    cell = runner.ScenarioCell(
        "standard",
        0,
        "fixture",
        artifact,
        {"drop_query": [], "drop_gallery": [], "duplicate_gallery": []},
    )
    result = evaluate_artifact(artifact, DecoderConfig("hungarian", seed=42))
    record = runner._result_record("nice", cell, result)

    runner._write_cell(
        tmp_path,
        model="nice",
        subject="sub-08",
        seed=42,
        cell=cell,
        evaluations=(result,),
        records=[record],
        source_hashes={"standard": "0" * 64},
        scenario_manifest_sha256="1" * 64,
    )

    cell_dir = (
        tmp_path
        / "nice"
        / "subj08"
        / "seed42"
        / "standard"
        / "00_fixture"
    )
    assert sorted(path.name for path in cell_dir.iterdir()) == [
        "per_query.csv",
        "summary.json",
    ]
    summary = json.loads((cell_dir / "summary.json").read_text(encoding="utf-8"))
    assignment_metadata = summary["decoder_records"][0]["assignment_metadata"]
    assert re.fullmatch(
        r"[0-9a-f]{64}",
        assignment_metadata["row_permutation_sha256"],
    )
    assert re.fullmatch(
        r"[0-9a-f]{64}",
        assignment_metadata["column_permutation_sha256"],
    )
    with pytest.raises(FileExistsError):
        runner._write_cell(
            tmp_path,
            model="nice",
            subject="sub-08",
            seed=42,
            cell=cell,
            evaluations=(result,),
            records=[record],
            source_hashes={"standard": "0" * 64},
            scenario_manifest_sha256="1" * 64,
        )


@pytest.mark.parametrize("dangling", (False, True))
def test_runner_rejects_artifact_and_output_symlinks_before_resolution(
    tmp_path: Path,
    dangling: bool,
) -> None:
    runner = _load_runner()
    target = tmp_path / "missing" if dangling else tmp_path / "target"
    if not dangling:
        target.mkdir()
    symlink = tmp_path / "link"
    symlink.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        runner._validated_existing_directory(symlink, "formal artifact root")
    with pytest.raises(ValueError, match="symlink"):
        runner._validated_output_directory(symlink)


def _synthetic_formal_artifacts() -> tuple[
    dict[str, dict[str, ScoreArtifact]],
    dict[str, dict[str, str]],
]:
    gallery = tuple(f"image-{index:03d}" for index in range(200))
    artifacts: dict[str, dict[str, ScoreArtifact]] = {}
    hashes: dict[str, dict[str, str]] = {}
    for model_index, model in enumerate(("nice", "atm_s", "our_project")):
        standard = _artifact(
            np.eye(200) + model_index,
            gallery,
            gallery,
            query_ids=gallery,
        )
        eeg_a = _artifact(
            np.eye(200) + model_index + 3.0,
            gallery,
            gallery,
            query_ids=gallery,
        )
        eeg_b = _artifact(
            np.eye(200) + model_index + 7.0,
            gallery,
            gallery,
            query_ids=gallery,
        )
        artifacts[model] = {"standard": standard, "a": eeg_a, "b": eeg_b}
        hashes[model] = {
            "standard": f"{model_index + 1:064x}",
            "a": f"{model_index + 11:064x}",
            "b": f"{model_index + 21:064x}",
        }
    return artifacts, hashes


def _lightweight_evaluation(
    artifact: ScoreArtifact,
    config: DecoderConfig,
) -> EvaluationResult:
    rows = artifact.similarity.shape[0]
    indices = np.zeros(rows, dtype=np.int64)
    unmatched = np.zeros(rows, dtype=bool)
    assignment = Assignment(
        gallery_indices=indices,
        unmatched_mask=unmatched,
        strict_one_to_one=config.name
        in {"greedy", "hungarian", "stable_matching"},
        metadata={"fixture": True},
    )
    per_query = tuple(
        {
            "query_index": row,
            "query_id": artifact.query_ids[row],
            "target_canonical_id": artifact.target_canonical_ids[row],
            "answerable": True,
            "gallery_index": 0,
            "predicted_gallery_entry_id": artifact.gallery_entry_ids[0],
            "predicted_canonical_id": artifact.gallery_canonical_ids[0],
            "assigned_score": float(artifact.similarity[row, 0]),
            "unmatched": False,
            "correct_top1": row == 0,
            "correct_top5": row == 0 if config.name == "independent" else None,
        }
        for row in range(rows)
    )
    return EvaluationResult(
        decoder=config.name,
        assignment=assignment,
        metrics={
            "decoder": config.name,
            "correct": 1,
            "total": rows,
            "top1": 100.0 / rows,
            "top5_count": 1 if config.name == "independent" else None,
            "top5": 100.0 / rows if config.name == "independent" else None,
            "assignment_metadata": dict(assignment.metadata),
        },
        per_query=per_query,
    )


def test_lightweight_runner_publishes_exact_grid_manifest_and_cleans_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    artifacts, hashes = _synthetic_formal_artifacts()
    monkeypatch.setattr(runner, "evaluate_artifact", _lightweight_evaluation)
    results_root = tmp_path / "matching_fairness_v3"
    (results_root / "manifests").mkdir(parents=True)
    (results_root / "matrices").mkdir()
    trial_manifest = results_root / "manifests" / "trial.json"
    trial_manifest.write_text("{}\n", encoding="utf-8")
    verified_trial_hash = hashlib.sha256(trial_manifest.read_bytes()).hexdigest()

    def load_then_replace_manifest(*_args, **_kwargs):
        def verified_result():
            yield artifacts
            yield hashes
            yield verified_trial_hash
            # Three-value unpacking advances once more to confirm exhaustion.
            # This replacement therefore occurs after the loader has returned
            # and before the caller starts constructing its scenario plan.
            trial_manifest.write_text('{"replaced":true}\n', encoding="utf-8")

        return verified_result()

    monkeypatch.setattr(
        runner,
        "_load_formal_artifacts",
        load_then_replace_manifest,
    )
    output = results_root / "runs"

    count = runner.run_scenarios(
        protocol_path=Path(
            "experiments/matching_fairness/configs/protocol_sub08_seed42.json"
        ),
        artifact_root=results_root / "matrices",
        trial_manifest_path=trial_manifest,
        output_dir=output,
    )

    assert count == 450
    assert not (output / "runs").exists()
    summaries = sorted(output.rglob("summary.json"))
    ledgers = sorted(output.rglob("per_query.csv"))
    assert len(summaries) == len(ledgers) == 90
    manifest = output / "scenario_manifest.json"
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["trial_manifest_sha256"] == verified_trial_hash
    assert hashlib.sha256(trial_manifest.read_bytes()).hexdigest() != verified_trial_hash
    records = []
    selections: dict[int, list[object]] = {}
    for path in summaries:
        summary = json.loads(path.read_text(encoding="utf-8"))
        assert summary["scenario_manifest_sha256"] == manifest_hash
        records.extend(summary["decoder_records"])
        selections.setdefault(summary["scenario_index"], []).append(
            summary["scenario_selection"]
        )
    assert len(records) == 450
    runner._validate_record_grid(records)
    assert all(values[0] == values[1] == values[2] for values in selections.values())
    assert (results_root / "manifests").is_dir()
    assert (results_root / "matrices").is_dir()

    with pytest.raises(FileExistsError):
        runner.run_scenarios(
            protocol_path=Path(
                "experiments/matching_fairness/configs/protocol_sub08_seed42.json"
            ),
            artifact_root=results_root / "matrices",
            trial_manifest_path=trial_manifest,
            output_dir=output,
        )

    failed_output = results_root / "failed_runs"
    calls = 0

    def fail_during_evaluation(
        artifact: ScoreArtifact,
        config: DecoderConfig,
    ) -> EvaluationResult:
        nonlocal calls
        calls += 1
        if calls == 8:
            raise RuntimeError("injected runner failure")
        return _lightweight_evaluation(artifact, config)

    monkeypatch.setattr(runner, "evaluate_artifact", fail_during_evaluation)
    with pytest.raises(RuntimeError, match="injected"):
        runner.run_scenarios(
            protocol_path=Path(
                "experiments/matching_fairness/configs/protocol_sub08_seed42.json"
            ),
            artifact_root=results_root / "matrices",
            trial_manifest_path=trial_manifest,
            output_dir=failed_output,
        )
    assert not os.path.lexists(failed_output)
    assert not list(results_root.glob(".failed_runs.tmp-*"))
