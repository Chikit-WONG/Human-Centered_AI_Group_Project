from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from samga_brain_rw.fusion import (
    BranchValidationMetrics,
    FusionValidationMetrics,
    assert_aligned,
    convex_fusion,
    enumerate_stage1_configs,
    querywise_zscore,
    ranked_gallery_indices,
    reciprocal_rank_fusion,
    select_best_fusion_config,
    select_stronger_single_branch,
    temperature_fusion,
)
from samga_brain_rw.hashing import sha256_json
from samga_brain_rw.scores import ScoreArtifact


def _metadata(
    *,
    checkpoint: str,
    config: str,
    protocol: str = "d" * 64,
    subject: int = 1,
    seed: int = 17,
    stage: str = "stage-1",
    query_ids: tuple[str, ...] = ("q0", "q1"),
) -> dict[str, object]:
    return {
        "checkpoint_sha256": checkpoint,
        "config_sha256": config,
        "git_sha": "c" * 40,
        "protocol_sha256": protocol,
        "seed": seed,
        "source_records": [{"record_id": query_id} for query_id in query_ids],
        "split_role": "val-dev",
        "stage": stage,
        "subject": subject,
    }


def _save_pair(
    root: Path,
    *,
    right_query_ids: tuple[str, ...] = ("q0", "q1"),
    right_gallery_ids: tuple[str, ...] = ("q0", "q1", "x"),
    right_metadata: dict[str, object] | None = None,
) -> tuple[ScoreArtifact, ScoreArtifact]:
    query_ids = ("q0", "q1")
    gallery_ids = ("q0", "q1", "x")
    internvit = np.array(
        [[0.9, 0.2, 0.1], [0.2, 0.8, 0.3]],
        dtype=np.float32,
    )
    clip = np.array(
        [[0.7, 0.4, 0.2], [0.4, 0.6, 0.5]],
        dtype=np.float32,
    )
    left_directory = root / "internvit"
    right_directory = root / "clip"
    ScoreArtifact.save(
        left_directory,
        internvit,
        query_ids,
        gallery_ids,
        _metadata(checkpoint="a" * 64, config="b" * 64),
    )
    ScoreArtifact.save(
        right_directory,
        clip,
        right_query_ids,
        right_gallery_ids,
        right_metadata
        or _metadata(
            checkpoint="e" * 64,
            config="f" * 64,
            query_ids=right_query_ids,
        ),
    )
    return (
        ScoreArtifact.load(left_directory, allowed_scopes={"val-dev"}),
        ScoreArtifact.load(right_directory, allowed_scopes={"val-dev"}),
    )


def _run_fusion_cli(
    experiment_root: Path,
    left: Path,
    right: Path,
    output: Path,
    *extra: str,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(experiment_root)
    return subprocess.run(
        [
            sys.executable,
            str(experiment_root / "scripts" / "run_score_fusion.py"),
            "--internvit-score-directory",
            str(left),
            "--clip-score-directory",
            str(right),
            "--internvit-branch-id",
            "internvit",
            "--clip-branch-id",
            "clip",
            "--output",
            str(output),
            *extra,
        ],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def test_stage1_grid_has_exact_deterministic_47_configs() -> None:
    first = tuple(enumerate_stage1_configs())
    second = tuple(enumerate_stage1_configs())

    assert first == second
    assert len(first) == 47
    assert len({config.config_id for config in first}) == 47
    assert [config.family for config in first[:11]] == ["zscore_convex"] * 11
    assert [config.alpha for config in first[:11]] == [
        index / 10 for index in range(11)
    ]
    assert [config.config_id for config in first[:11]] == [
        f"s1-z-a{index * 10:03d}" for index in range(11)
    ]
    assert {config.formula for config in first[:11]} == {
        "alpha * zscore_ddof0(S_I) + (1 - alpha) * zscore_ddof0(S_C)"
    }

    temperature = first[11:38]
    assert [config.family for config in temperature] == [
        "temperature_convex"
    ] * 27
    assert [
        (
            config.internvit_temperature,
            config.clip_temperature,
            config.alpha,
        )
        for config in temperature
    ] == [
        (internvit_temperature, clip_temperature, alpha)
        for internvit_temperature in (0.5, 1.0, 2.0)
        for clip_temperature in (0.5, 1.0, 2.0)
        for alpha in (0.25, 0.5, 0.75)
    ]
    assert [config.config_id for config in temperature] == [
        (
            f"s1-temp-ti{int(internvit_temperature * 100):03d}"
            f"-tc{int(clip_temperature * 100):03d}"
            f"-a{int(alpha * 100):03d}"
        )
        for internvit_temperature in (0.5, 1.0, 2.0)
        for clip_temperature in (0.5, 1.0, 2.0)
        for alpha in (0.25, 0.5, 0.75)
    ]
    assert {config.formula for config in temperature} == {
        "alpha * S_I / T_I + (1 - alpha) * S_C / T_C"
    }

    rrf = first[38:]
    assert [config.family for config in rrf] == ["rrf"] * 9
    assert [(config.k, config.internvit_weight) for config in rrf] == [
        (k, weight)
        for k in (10, 30, 60)
        for weight in (0.25, 0.5, 0.75)
    ]
    assert [config.config_id for config in rrf] == [
        f"s1-rrf-k{k:03d}-w{int(weight * 100):03d}"
        for k in (10, 30, 60)
        for weight in (0.25, 0.5, 0.75)
    ]
    assert {config.formula for config in rrf} == {
        "w / (k + rank_I) + (1 - w) / (k + rank_C)"
    }
    assert all(config.rank_origin == 1 for config in rrf)
    assert all(
        config.score_tie_break == "gallery_id_utf8_bytewise"
        and config.final_tie_break == "gallery_id_utf8_bytewise"
        for config in rrf
    )


def test_enumerated_grid_exactly_matches_tracked_stage1_candidates(
    configs_dir: Path,
) -> None:
    tracked = json.loads(
        (configs_dir / "stage1_fusion_v1.json").read_text("utf-8")
    )

    assert [config.to_dict() for config in enumerate_stage1_configs()] == (
        tracked["candidates"]
    )


def test_querywise_zscore_is_float64_population_and_zeroes_constant_rows() -> None:
    scores = np.array([[1.0, 2.0, 3.0], [7.0, 7.0, 7.0]], dtype=np.float32)

    normalized = querywise_zscore(scores)

    assert normalized.dtype == np.float64
    np.testing.assert_allclose(
        normalized[0],
        np.array([-np.sqrt(1.5), 0.0, np.sqrt(1.5)], dtype=np.float64),
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_array_equal(normalized[1], np.zeros(3, dtype=np.float64))
    assert np.isfinite(normalized).all()


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_querywise_zscore_rejects_nonfinite(nonfinite: float) -> None:
    scores = np.array([[0.0, nonfinite]], dtype=np.float64)

    with pytest.raises(ValueError, match="finite"):
        querywise_zscore(scores)


def test_convex_fusion_uses_querywise_zscores_and_float64() -> None:
    internvit = np.array([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0]])
    clip = np.array([[3.0, 1.0, 2.0], [1.0, 2.0, 3.0]])

    fused = convex_fusion(internvit, clip, alpha=0.25)

    expected = (
        0.25 * querywise_zscore(internvit)
        + 0.75 * querywise_zscore(clip)
    )
    assert fused.dtype == np.float64
    np.testing.assert_array_equal(fused, expected)


def test_temperature_fusion_uses_raw_scores_without_softmax() -> None:
    internvit = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    clip = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

    fused = temperature_fusion(
        internvit,
        clip,
        alpha=0.25,
        internvit_temperature=0.5,
        clip_temperature=2.0,
    )

    expected = (
        0.25 * internvit.astype(np.float64) / 0.5
        + 0.75 * clip.astype(np.float64) / 2.0
    )
    assert fused.dtype == np.float64
    np.testing.assert_array_equal(fused, expected)
    assert not np.allclose(fused.sum(axis=1), 1.0)


@pytest.mark.parametrize(
    ("function", "kwargs"),
    [
        (convex_fusion, {"alpha": -0.1}),
        (convex_fusion, {"alpha": 1.1}),
        (
            temperature_fusion,
            {
                "alpha": 0.5,
                "internvit_temperature": 0.0,
                "clip_temperature": 1.0,
            },
        ),
        (
            temperature_fusion,
            {
                "alpha": 0.5,
                "internvit_temperature": 1.0,
                "clip_temperature": np.inf,
            },
        ),
    ],
)
def test_convex_families_reject_invalid_scalars(function, kwargs) -> None:
    scores = np.ones((1, 2), dtype=np.float64)

    with pytest.raises(ValueError):
        function(scores, scores, **kwargs)


def test_all_fusion_families_reject_shape_mismatch_and_nonfinite() -> None:
    finite = np.ones((1, 2), dtype=np.float64)
    wrong_shape = np.ones((2, 1), dtype=np.float64)
    nonfinite = np.array([[0.0, np.nan]], dtype=np.float64)

    with pytest.raises(ValueError, match="shape"):
        convex_fusion(finite, wrong_shape, alpha=0.5)
    with pytest.raises(ValueError, match="finite"):
        temperature_fusion(
            finite,
            nonfinite,
            alpha=0.5,
            internvit_temperature=1.0,
            clip_temperature=1.0,
        )
    with pytest.raises(ValueError, match="finite"):
        reciprocal_rank_fusion(
            finite,
            nonfinite,
            gallery_ids=("a", "b"),
            k=10,
            internvit_weight=0.5,
        )


def test_rrf_uses_one_based_ordinal_ranks_with_utf8_id_ties() -> None:
    gallery_ids = ("é", "a", "z")
    internvit = np.array([[1.0, 1.0, 0.0]], dtype=np.float32)
    clip = np.array([[0.0, 1.0, 1.0]], dtype=np.float32)

    fused = reciprocal_rank_fusion(
        internvit,
        clip,
        gallery_ids=gallery_ids,
        k=10,
        internvit_weight=0.25,
    )

    # InternViT ranks by score then UTF-8 ID: a=1, é=2, z=3.
    # CLIP ranks by score then UTF-8 ID: a=1, z=2, é=3.
    expected = np.array(
        [
            [
                0.25 / (10 + 2) + 0.75 / (10 + 3),
                0.25 / (10 + 1) + 0.75 / (10 + 1),
                0.25 / (10 + 3) + 0.75 / (10 + 2),
            ]
        ],
        dtype=np.float64,
    )
    assert fused.dtype == np.float64
    np.testing.assert_array_equal(fused, expected)


def test_ranked_gallery_indices_breaks_final_ties_by_utf8_id() -> None:
    scores = np.array([[0.4, 0.4, 0.4], [0.1, 0.3, 0.2]])

    ranking = ranked_gallery_indices(scores, ("é", "a", "z"))

    assert ranking.dtype == np.int64
    np.testing.assert_array_equal(
        ranking,
        np.array([[1, 2, 0], [1, 2, 0]], dtype=np.int64),
    )


@pytest.mark.parametrize(
    ("k", "weight"),
    [(0, 0.5), (-1, 0.5), (10.0, 0.5), (10, -0.1), (10, 1.1)],
)
def test_rrf_rejects_invalid_rank_parameters(k: object, weight: float) -> None:
    scores = np.ones((1, 2), dtype=np.float64)

    with pytest.raises((TypeError, ValueError)):
        reciprocal_rank_fusion(
            scores,
            scores,
            gallery_ids=("a", "b"),
            k=k,  # type: ignore[arg-type]
            internvit_weight=weight,
        )


def test_rrf_rejects_missing_duplicate_or_misaligned_gallery_ids() -> None:
    scores = np.ones((1, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="gallery"):
        reciprocal_rank_fusion(
            scores,
            scores,
            gallery_ids=("a",),
            k=10,
            internvit_weight=0.5,
        )
    with pytest.raises(ValueError, match="duplicate"):
        reciprocal_rank_fusion(
            scores,
            scores,
            gallery_ids=("a", "a"),
            k=10,
            internvit_weight=0.5,
        )


def test_assert_aligned_accepts_distinct_branch_specific_provenance(
    tmp_path: Path,
) -> None:
    left, right = _save_pair(tmp_path)

    assert left.verified.payload_sha256 != right.verified.payload_sha256
    assert sha256_json(_thaw(left.provenance)) != sha256_json(
        _thaw(right.provenance)
    )
    assert_aligned(left, right)


@pytest.mark.parametrize(
    "mismatch",
    [
        "query_ids",
        "gallery_ids",
        "protocol",
        "subject",
        "seed",
        "stage",
        "source_records",
    ],
)
def test_assert_aligned_rejects_shared_alignment_mismatch(
    tmp_path: Path,
    mismatch: str,
) -> None:
    kwargs: dict[str, object] = {}
    right_query_ids = ("q0", "q1")
    right_gallery_ids = ("q0", "q1", "x")
    metadata = _metadata(
        checkpoint="e" * 64,
        config="f" * 64,
        query_ids=right_query_ids,
    )
    if mismatch == "query_ids":
        right_query_ids = ("q1", "q0")
        metadata = _metadata(
            checkpoint="e" * 64,
            config="f" * 64,
            query_ids=right_query_ids,
        )
    elif mismatch == "gallery_ids":
        right_gallery_ids = ("q1", "q0", "x")
    elif mismatch == "protocol":
        metadata["protocol_sha256"] = "9" * 64
    elif mismatch == "subject":
        metadata["subject"] = 5
    elif mismatch == "seed":
        metadata["seed"] = 42
    elif mismatch == "stage":
        metadata["stage"] = "stage-2"
    elif mismatch == "source_records":
        metadata["source_records"] = [
            {"record_id": "q0", "source": "different"},
            {"record_id": "q1"},
        ]
    kwargs["right_query_ids"] = right_query_ids
    kwargs["right_gallery_ids"] = right_gallery_ids
    kwargs["right_metadata"] = metadata
    left, right = _save_pair(tmp_path, **kwargs)

    with pytest.raises(ValueError, match="align|mismatch|provenance|ID"):
        assert_aligned(left, right)


def _branch(
    branch_id: str,
    top1: tuple[float, ...],
    top5: tuple[float, ...],
    cost: float,
) -> BranchValidationMetrics:
    return BranchValidationMetrics(
        branch_id=branch_id,
        cell_ids=("01/42", "01/43", "05/42", "05/43", "08/42", "08/43"),
        top1=top1,
        top5=top5,
        measured_inference_cost=cost,
    )


def test_stronger_single_branch_is_one_global_six_cell_choice() -> None:
    internvit = _branch(
        "internvit",
        (0.99, 0.99, 0.99, 0.10, 0.10, 0.10),
        (0.99,) * 6,
        10.0,
    )
    clip = _branch(
        "clip",
        (0.80, 0.80, 0.80, 0.70, 0.70, 0.70),
        (0.95,) * 6,
        5.0,
    )

    selected = select_stronger_single_branch((internvit, clip))

    assert selected == "clip"


def test_stronger_single_branch_ties_top1_then_top5_then_cost_then_id() -> None:
    base_top1 = (0.8,) * 6
    low_top5 = _branch("z-low-top5", base_top1, (0.8,) * 6, 1.0)
    high_top5 = _branch("z-high-top5", base_top1, (0.9,) * 6, 100.0)
    assert (
        select_stronger_single_branch((low_top5, high_top5))
        == "z-high-top5"
    )

    high_cost = _branch("a-high-cost", base_top1, (0.9,) * 6, 2.0)
    low_cost = _branch("z-low-cost", base_top1, (0.9,) * 6, 1.0)
    assert select_stronger_single_branch((high_cost, low_cost)) == "z-low-cost"

    lexical_later = _branch("z-branch", base_top1, (0.9,) * 6, 1.0)
    lexical_first = _branch("a-branch", base_top1, (0.9,) * 6, 1.0)
    assert (
        select_stronger_single_branch((lexical_later, lexical_first))
        == "a-branch"
    )


def test_stronger_single_branch_requires_complete_matching_six_cells() -> None:
    complete = _branch("complete", (0.8,) * 6, (0.9,) * 6, 1.0)
    incomplete = BranchValidationMetrics(
        branch_id="incomplete",
        cell_ids=("01/42",) * 5,
        top1=(0.8,) * 5,
        top5=(0.9,) * 5,
        measured_inference_cost=1.0,
    )

    with pytest.raises(ValueError, match="six|6|cell"):
        select_stronger_single_branch((complete, incomplete))


def _fusion_candidates(
    *,
    top1: float = 0.8,
    top5: float = 0.9,
    cost: float = 2.0,
) -> list[FusionValidationMetrics]:
    cells = ("01/42", "01/43", "05/42", "05/43", "08/42", "08/43")
    return [
        FusionValidationMetrics(
            config_id=config.config_id,
            cell_ids=cells,
            top1=(top1,) * 6,
            top5=(top5,) * 6,
            measured_inference_cost=cost,
        )
        for config in enumerate_stage1_configs()
    ]


def test_fusion_selector_uses_top1_top5_cost_then_config_id() -> None:
    candidates = _fusion_candidates()
    top1_winner = candidates[40]
    candidates[40] = FusionValidationMetrics(
        config_id=top1_winner.config_id,
        cell_ids=top1_winner.cell_ids,
        top1=(0.81,) * 6,
        top5=(0.1,) * 6,
        measured_inference_cost=100.0,
    )
    assert select_best_fusion_config(tuple(reversed(candidates))) == (
        top1_winner.config_id
    )

    candidates = _fusion_candidates()
    top5_winner = candidates[30]
    candidates[30] = FusionValidationMetrics(
        config_id=top5_winner.config_id,
        cell_ids=top5_winner.cell_ids,
        top1=top5_winner.top1,
        top5=(0.91,) * 6,
        measured_inference_cost=100.0,
    )
    assert select_best_fusion_config(candidates) == top5_winner.config_id

    candidates = _fusion_candidates()
    cost_winner = candidates[-1]
    candidates[-1] = FusionValidationMetrics(
        config_id=cost_winner.config_id,
        cell_ids=cost_winner.cell_ids,
        top1=cost_winner.top1,
        top5=cost_winner.top5,
        measured_inference_cost=1.0,
    )
    assert select_best_fusion_config(candidates) == cost_winner.config_id

    candidates = _fusion_candidates()
    assert select_best_fusion_config(tuple(reversed(candidates))) == min(
        config.config_id for config in enumerate_stage1_configs()
    )


def test_fusion_selector_requires_exact_complete_47_config_grid() -> None:
    candidates = _fusion_candidates()

    with pytest.raises(ValueError, match="47|grid|config"):
        select_best_fusion_config(candidates[:-1])

    duplicate = candidates.copy()
    duplicate[-1] = duplicate[0]
    with pytest.raises(ValueError, match="47|grid|config|unique"):
        select_best_fusion_config(duplicate)


def test_score_fusion_cli_consumes_typed_val_dev_and_binds_inputs(
    tmp_path: Path,
    experiment_root: Path,
    configs_dir: Path,
) -> None:
    left, right = _save_pair(tmp_path / "inputs")
    output = tmp_path / "fusion-grid.json"

    result = _run_fusion_cli(
        experiment_root,
        left.directory,
        right.directory,
        output,
    )

    assert result.returncode == 0, result.stderr
    document = json.loads(output.read_text("utf-8"))
    assert document["schema_version"] == 1
    assert document["artifact_type"] == "samga_brain_rw.stage1_fusion_grid"
    assert document["scope"] == "val-dev"
    tracked = json.loads(
        (configs_dir / "stage1_fusion_v1.json").read_text("utf-8")
    )
    assert document["grid"]["config_id"] == "stage1_fusion_v1"
    assert document["grid"]["candidates"] == tracked["candidates"]
    assert document["grid"]["candidates_sha256"] == sha256_json(
        document["grid"]["candidates"]
    )
    assert document["grid"]["candidates_sha256"] == sha256_json(
        [config.to_dict() for config in enumerate_stage1_configs()]
    )
    assert len(document["results"]) == 47
    assert [item["config_id"] for item in document["results"]] == [
        config.config_id for config in enumerate_stage1_configs()
    ]
    assert document["inputs"]["internvit"] == {
        "branch_id": "internvit",
        "provenance_sha256": sha256_json(_thaw(left.provenance)),
        "score_payload_sha256": left.verified.payload_sha256,
    }
    assert document["inputs"]["clip"] == {
        "branch_id": "clip",
        "provenance_sha256": sha256_json(_thaw(right.provenance)),
        "score_payload_sha256": right.verified.payload_sha256,
    }
    assert document["alignment"]["query_ids_sha256"] == left.metadata[
        "query_ids_sha256"
    ]
    assert document["alignment"]["gallery_ids_sha256"] == left.metadata[
        "gallery_ids_sha256"
    ]
    assert document["alignment"]["source_records_sha256"] == left.metadata[
        "source_records_sha256"
    ]


def test_score_fusion_cli_exclusively_preserves_existing_output(
    tmp_path: Path,
    experiment_root: Path,
) -> None:
    left, right = _save_pair(tmp_path / "inputs")
    output = tmp_path / "owned.json"
    output.write_bytes(b"owned")

    result = _run_fusion_cli(
        experiment_root,
        left.directory,
        right.directory,
        output,
    )

    assert result.returncode != 0
    assert output.read_bytes() == b"owned"


@pytest.mark.parametrize(
    "relative_output",
    [
        ("test_images", "fusion.json"),
        ("VAL-CONFIRM", "fusion.json"),
        ("formal-test", "fusion.json"),
        ("SUB-07_TEST.JSON",),
        (
            "02d7e33b3fe8e5a571f8db232ca5fa86abb0c16981876ec84feae7ba64636f1a",
            "fusion.json",
        ),
    ],
    ids=[
        "test-images",
        "val-confirm",
        "formal-test",
        "subject-test-manifest",
        "canonical-formal-digest",
    ],
)
def test_score_fusion_cli_rejects_sensitive_output_paths_without_creation(
    tmp_path: Path,
    experiment_root: Path,
    relative_output: tuple[str, ...],
) -> None:
    left, right = _save_pair(tmp_path / "inputs")
    forbidden_root = tmp_path / "forbidden-output"
    output = forbidden_root.joinpath(*relative_output)

    result = _run_fusion_cli(
        experiment_root,
        left.directory,
        right.directory,
        output,
    )

    assert result.returncode != 0
    assert not output.exists()
    assert not forbidden_root.exists()


def test_score_fusion_cli_rejects_symlink_parent_without_writing_target(
    tmp_path: Path,
    experiment_root: Path,
) -> None:
    left, right = _save_pair(tmp_path / "inputs")
    real_parent = tmp_path / "real-output"
    real_parent.mkdir()
    symlink_parent = tmp_path / "linked-output"
    symlink_parent.symlink_to(real_parent, target_is_directory=True)
    output = symlink_parent / "fusion.json"

    result = _run_fusion_cli(
        experiment_root,
        left.directory,
        right.directory,
        output,
    )

    assert result.returncode != 0
    assert list(real_parent.iterdir()) == []
    assert symlink_parent.is_symlink()


def test_score_fusion_cli_never_writes_inside_an_input_bundle(
    tmp_path: Path,
    experiment_root: Path,
) -> None:
    left, right = _save_pair(tmp_path / "inputs")
    output = left.directory / "fusion-grid.json"

    result = _run_fusion_cli(
        experiment_root,
        left.directory,
        right.directory,
        output,
    )

    assert result.returncode != 0
    assert set(os.listdir(left.directory)) == {
        "similarity.npy",
        "metadata.json",
        "predictions.csv",
    }


def test_score_fusion_cli_rejects_non_val_dev_before_output(
    tmp_path: Path,
    experiment_root: Path,
) -> None:
    left, right = _save_pair(tmp_path / "inputs")
    envelope_path = right.directory / "metadata.json"
    envelope = json.loads(envelope_path.read_text("utf-8"))
    envelope["scope"] = "val-confirm"
    envelope_path.write_text(
        json.dumps(
            envelope,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "forbidden.json"

    result = _run_fusion_cli(
        experiment_root,
        left.directory,
        right.directory,
        output,
    )

    assert result.returncode != 0
    assert not output.exists()


def test_score_fusion_cli_has_no_raw_score_fallback(
    tmp_path: Path,
    experiment_root: Path,
) -> None:
    output = tmp_path / "raw.json"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(experiment_root)
    result = subprocess.run(
        [
            sys.executable,
            str(experiment_root / "scripts" / "run_score_fusion.py"),
            "--input-similarity",
            str(tmp_path / "raw.npy"),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode != 0
    assert not output.exists()
