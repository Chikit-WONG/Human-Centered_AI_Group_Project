from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import train as samga_train


def _h(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _stage2_arguments(
    candidate_id: str,
    **overrides: object,
) -> argparse.Namespace:
    values: dict[str, object] = {
        "stage": 2,
        "candidate_id": candidate_id,
        "layernorm_config_id": "s2-layernorm-off",
        "whitening_config_id": "s2-whitening-off",
        "preprojector_config_id": "s2-preproj-shared",
        "adapter_kind": "identity",
        "adapter_rank": None,
        "adapter_lr_ratio": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _stage2_paths(
    experiment_root: Path,
    tmp_path: Path,
    *,
    whitening_artifact: Path | None = None,
) -> samga_train.TrainingPaths:
    return samga_train.TrainingPaths(
        config=tmp_path / "baseline.json",
        manifest=tmp_path / "sub-08_protocol.json",
        feature_cache=tmp_path / "features.npy",
        output_dir=tmp_path / "run",
        stage2_config=(
            experiment_root
            / "configs"
            / "stage2_candidates_v1.json"
        ),
        whitening_artifact=whitening_artifact,
        resume_checkpoint=None,
    )


def _selection_config() -> object:
    return SimpleNamespace(
        cache_sha256=_h("cache"),
        payload={"config_id": "internvit_baseline_v1"},
    )


@pytest.mark.parametrize(
    ("candidate_id", "expected"),
    [
        (
            "s2-layernorm-off",
            (
                "s2-layernorm-off",
                "s2-whitening-off",
                "s2-preproj-shared",
                "identity",
                None,
                None,
            ),
        ),
        (
            "s2-layernorm-on",
            (
                "s2-layernorm-on",
                "s2-whitening-off",
                "s2-preproj-shared",
                "identity",
                None,
                None,
            ),
        ),
        (
            "s2-preproj-separate",
            (
                "s2-layernorm-off",
                "s2-whitening-off",
                "s2-preproj-separate",
                "identity",
                None,
                None,
            ),
        ),
        (
            "s2-adapter-r16-lr0.10",
            (
                "s2-layernorm-off",
                "s2-whitening-off",
                "s2-preproj-shared",
                "adapter",
                16,
                0.1,
            ),
        ),
    ],
)
def test_stage2_candidate_id_derives_exact_registry_identity(
    experiment_root: Path,
    tmp_path: Path,
    candidate_id: str,
    expected: tuple[object, ...],
) -> None:
    selection = samga_train._resolve_candidate_selection(
        _stage2_arguments(candidate_id),
        _stage2_paths(experiment_root, tmp_path),
        _selection_config(),
    )
    assert (
        selection.layernorm_config_id,
        selection.whitening_config_id,
        selection.preprojector_config_id,
        selection.adapter_kind,
        selection.adapter_rank,
        selection.adapter_lr_ratio,
    ) == expected


@pytest.mark.parametrize(
    ("candidate_id", "overrides"),
    [
        (
            "s2-adapter-identity-control",
            {
                "adapter_kind": "adapter",
                "adapter_rank": 8,
                "adapter_lr_ratio": 0.05,
            },
        ),
        (
            "s2-layernorm-on",
            {"preprojector_config_id": "s2-preproj-separate"},
        ),
        (
            "s2-adapter-r8-lr0.05",
            {"adapter_rank": 16},
        ),
    ],
)
def test_stage2_candidate_rejects_conflicting_cli_overrides(
    experiment_root: Path,
    tmp_path: Path,
    candidate_id: str,
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="registry|conflict"):
        samga_train._resolve_candidate_selection(
            _stage2_arguments(candidate_id, **overrides),
            _stage2_paths(experiment_root, tmp_path),
            _selection_config(),
        )


@pytest.mark.parametrize(
    "candidate_id",
    [
        "s2-raw-epoch60-control",
        "s2-avg-last5",
        "s2-avg-last10",
        "s2-swa-last5",
        "s2-swa-last10",
    ],
)
def test_stage2_post_hoc_candidate_is_not_a_training_cell(
    experiment_root: Path,
    tmp_path: Path,
    candidate_id: str,
) -> None:
    with pytest.raises(ValueError, match="post-hoc"):
        samga_train._resolve_candidate_selection(
            _stage2_arguments(candidate_id),
            _stage2_paths(experiment_root, tmp_path),
            _selection_config(),
        )
