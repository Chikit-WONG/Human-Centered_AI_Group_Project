from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from matching_fairness.artifacts import ScoreArtifact
from matching_fairness.evaluation import DecoderConfig, evaluate_artifact


SCRIPT = Path("experiments/matching_fairness/scripts/run_scenarios.py")


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_scenarios", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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

    scenarios = runner._scenario_artifacts(standard, a, b, seed=42)

    assert len(scenarios) == 30
    assert sum(cell.suite == "standard" for cell in scenarios) == 27
    assert sum(cell.suite == "duplicate_eeg" for cell in scenarios) == 3
    assert [cell.index for cell in scenarios] == list(range(30))
    assert [cell.artifact.similarity.shape for cell in scenarios[-3:]] == [
        (200, 200),
        (210, 200),
        (220, 200),
    ]


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
    cell = runner.ScenarioCell("standard", 0, "fixture", artifact)
    result = evaluate_artifact(artifact, DecoderConfig("independent"))
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
    )

    cell_dir = (
        tmp_path
        / "runs"
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
        )
