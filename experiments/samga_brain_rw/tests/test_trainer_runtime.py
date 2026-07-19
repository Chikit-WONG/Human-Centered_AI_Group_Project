from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

import train as samga_train
from samga_brain_rw.adapters import (
    DenseBottleneckControl,
    MatchedPerLayerProjectorControl,
    ResidualFeatureAdapter,
)
from samga_brain_rw.trainer import (
    CheckpointPublication,
    SAMGARuntimeModel,
    StatefulEpochSampler,
    TrainingCellSpec,
    evaluate_development_model,
    run_training_cell,
)
from samga_brain_rw.upstream_samga import UpstreamComponents


PINNED_COMMIT = "1a63745b7ff6f98dad34b0f0b8246a9b5260d9c1"


def _h(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class _CheapEEGProject(nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int,
        eeg_sample_points: int,
        channels_num: int,
    ) -> None:
        super().__init__()
        del eeg_sample_points, channels_num
        self.scale = nn.Parameter(torch.ones(feature_dim))

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        summary = eeg.mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        return summary * self.scale.unsqueeze(0)


class _CheapProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        del input_dim
        self.scale = nn.Parameter(torch.ones(output_dim))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values.mean(dim=-1, keepdim=True) * self.scale


class _CheapShareEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        if input_dim != output_dim:
            raise ValueError("the cheap shared encoder requires equal dimensions")
        self.scale = nn.Parameter(torch.ones(output_dim))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.scale


class _CheapRouter(nn.Module):
    def __init__(
        self,
        *,
        layer_ids: tuple[int, ...],
        num_subjects: int,
        prior_center: int,
        prior_strength: float,
        temperature: float,
        subject_dropout: float,
    ) -> None:
        super().__init__()
        del prior_center, prior_strength, temperature, subject_dropout
        self.logits = nn.Parameter(torch.zeros(len(layer_ids)))
        self.subject_bias = nn.Embedding(num_subjects, len(layer_ids))
        nn.init.zeros_(self.subject_bias.weight)
        self.calls: list[bool] = []

    def forward(
        self,
        subject_ids: torch.Tensor,
        *,
        force_global: bool,
    ) -> torch.Tensor:
        self.calls.append(force_global)
        logits = self.logits.unsqueeze(0).expand(subject_ids.shape[0], -1)
        if not force_global:
            logits = logits + self.subject_bias(subject_ids)
        return torch.softmax(logits, dim=-1)


class _CheapContrastiveLoss(nn.Module):
    def __init__(self, *_: object) -> None:
        super().__init__()

    def forward(
        self,
        eeg: torch.Tensor,
        image: torch.Tensor,
        text: torch.Tensor,
    ) -> torch.Tensor:
        del text
        logits = eeg @ F.normalize(image, dim=1).T
        labels = torch.arange(eeg.shape[0], device=eeg.device)
        return 0.5 * (
            F.cross_entropy(logits, labels)
            + F.cross_entropy(logits.T, labels)
        )


def _cheap_mmd(eeg: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    return (eeg.mean(dim=0) - image.mean(dim=0)).square().mean()


@pytest.fixture()
def components(tmp_path: Path) -> UpstreamComponents:
    return UpstreamComponents(
        upstream_root=tmp_path,
        commit=PINNED_COMMIT,
        source_sha256={},
        EEGProject=_CheapEEGProject,
        ProjectorDirect=_CheapProjector,
        ProjectorLinear=_CheapProjector,
        ProjectorMLP=_CheapProjector,
        ShareEncoder=_CheapShareEncoder,
        SubjectAwareLayerMixer=_CheapRouter,
        ContrastiveLoss=_CheapContrastiveLoss,
        mmd_rbf=_cheap_mmd,
    )


class _TinyProtocolDataset(Dataset[dict[str, object]]):
    def __init__(self, scope: str, subject: int, size: int) -> None:
        self.scope = scope
        self.subject_id = subject
        self.row_indices = tuple(
            range(100, 100 + size) if scope == "train" else range(900, 900 + size)
        )
        if scope == "val-dev":
            self.query_ids = tuple(f"image-{index}" for index in range(size))
            self.gallery_ids = self.query_ids

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(self, index: int) -> dict[str, object]:
        value = float(index + 1)
        return {
            "eeg": torch.full((17, 250), value, dtype=torch.float32),
            "layer_features": torch.full(
                (5, 3_200),
                value + 0.25,
                dtype=torch.float32,
            ),
            "subject_id": self.subject_id,
            "row_index": self.row_indices[index],
        }


class _DatasetFactory:
    def __init__(self, *, train_size: int = 6, val_size: int = 3) -> None:
        self.train_size = train_size
        self.val_size = val_size
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> _TinyProtocolDataset:
        self.calls.append(dict(kwargs))
        scope = str(kwargs["scope"])
        return _TinyProtocolDataset(
            scope,
            subject=1,
            size=self.train_size if scope == "train" else self.val_size,
        )


def _spec(
    tmp_path: Path,
    components: UpstreamComponents,
    factory: _DatasetFactory,
    **overrides: Any,
) -> TrainingCellSpec:
    values: dict[str, object] = {
        "components": components,
        "manifest_path": tmp_path / "sub-01_protocol.json",
        "feature_cache": tmp_path / "features.npy",
        "stage": 0,
        "subject": 1,
        "seed": 19,
        "config_sha256": _h("config"),
        "schedule_sha256": samga_train.SCHEDULE_SHA256,
        "trajectory_sha256": _h("trajectory"),
        "data_order_sha256": _h("data-order"),
        "input_hashes": {"manifest_sha256": _h("manifest")},
        "run_manifest": {"run_id": "s0-sub01-seed19"},
        "candidate_spec": {"config_id": "baseline"},
        "checkpoint_builder": samga_train.build_epoch_checkpoint,
        "checkpoint_restorer": samga_train.restore_training_checkpoint,
        "dataset_factory": factory,
        "batch_size": 2,
        "max_train_steps": 1,
        "num_workers": 0,
        "device": "cpu",
    }
    values.update(overrides)
    return TrainingCellSpec(**values)


@pytest.mark.parametrize(
    ("overrides", "active_factor", "module_type"),
    [
        ({}, None, nn.Identity),
        (
            {"layernorm_config_id": "s2-layernorm-on"},
            "layernorm",
            nn.Identity,
        ),
        (
            {"preprojector_config_id": "s2-preproj-separate"},
            "preprojectors",
            nn.Identity,
        ),
        (
            {
                "adapter_kind": "adapter",
                "adapter_rank": 8,
                "adapter_lr_ratio": 0.05,
            },
            "feature_adapter",
            ResidualFeatureAdapter,
        ),
        (
            {
                "adapter_kind": "global_dense",
                "adapter_rank": 8,
                "adapter_lr_ratio": 0.10,
            },
            "feature_adapter",
            DenseBottleneckControl,
        ),
        (
            {
                "adapter_kind": "matched_projector",
                "adapter_rank": 8,
                "adapter_lr_ratio": 0.05,
            },
            "feature_adapter",
            MatchedPerLayerProjectorControl,
        ),
    ],
)
def test_stage2_wrapper_builds_exactly_one_factor_and_controls(
    tmp_path: Path,
    components: UpstreamComponents,
    overrides: dict[str, object],
    active_factor: str | None,
    module_type: type[nn.Module],
) -> None:
    spec = _spec(
        tmp_path,
        components,
        _DatasetFactory(),
        stage=2,
        candidate_spec={"config_id": "candidate"},
        **overrides,
    )
    model = SAMGARuntimeModel(spec)

    assert model.active_factor == active_factor
    assert isinstance(model.feature_adapter, module_type)
    assert model.base._components is components
    assert model.base.config.image_dim == 3_200
    assert (model.separate_image_pre_projectors is not None) is (
        active_factor == "preprojectors"
    )


def test_stage0_and_stage2_reject_factor_combinations_or_unfitted_whitening(
    tmp_path: Path,
    components: UpstreamComponents,
) -> None:
    factory = _DatasetFactory()
    with pytest.raises(ValueError, match="Stage 0|stage 0"):
        _spec(
            tmp_path,
            components,
            factory,
            layernorm_config_id="s2-layernorm-on",
        )
    with pytest.raises(ValueError, match="one.*factor|factor.*one"):
        _spec(
            tmp_path,
            components,
            factory,
            stage=2,
            layernorm_config_id="s2-layernorm-on",
            adapter_kind="adapter",
            adapter_rank=8,
            adapter_lr_ratio=0.05,
        )
    with pytest.raises(ValueError, match="TrainWhitening"):
        _spec(
            tmp_path,
            components,
            factory,
            stage=2,
            whitening_config_id="s2-whitening-on",
        )


def test_stateful_epoch_sampler_restores_the_exact_remaining_order() -> None:
    sampler = StatefulEpochSampler(dataset_size=9, seed=31, epoch=4)
    iterator = iter(sampler)
    prefix = tuple(next(iterator) for _ in range(4))
    state = sampler.state_dict()
    expected_suffix = tuple(iterator)

    restored = StatefulEpochSampler(dataset_size=9, seed=31, epoch=1)
    restored.load_state_dict(state)

    assert len(prefix) == 4
    assert set(prefix).isdisjoint(expected_suffix)
    assert tuple(restored) == expected_suffix
    with pytest.raises(ValueError, match="dataset"):
        StatefulEpochSampler(dataset_size=8, seed=31).load_state_dict(state)


def test_cpu_one_step_smoke_uses_protocol_scopes_and_global_validation(
    tmp_path: Path,
    components: UpstreamComponents,
) -> None:
    factory = _DatasetFactory(train_size=6, val_size=3)
    result = run_training_cell(_spec(tmp_path, components, factory))

    assert result.global_step == 1
    assert result.completed is False
    assert result.final_validation.similarity.shape == (3, 3)
    assert result.final_validation.metrics.query_count == 3
    assert result.final_validation.metrics.gallery_count == 3
    assert result.final_checkpoint["runtime_state"]["epoch_complete"] is False
    assert result.final_checkpoint["sampler_state_dict"]["position"] == 2
    assert torch.equal(
        result.loader_generator_state,
        result.final_checkpoint["loader_generator_state"],
    )
    assert result.trained_row_indices == (result.trained_row_indices[0],)
    assert len(result.trained_row_indices[0]) == 2
    assert [call["scope"] for call in factory.calls] == ["train", "val-dev"]
    assert [call["smooth_probability"] for call in factory.calls] == [0.3, 0.0]
    assert all(len(call["selected_channels"]) == 17 for call in factory.calls)
    assert False in result.model.base.router.calls
    assert True in result.model.base.router.calls


def test_public_development_evaluator_forces_global_cosine_scores(
    tmp_path: Path,
    components: UpstreamComponents,
) -> None:
    spec = _spec(tmp_path, components, _DatasetFactory())
    model = SAMGARuntimeModel(spec)
    dataset = _TinyProtocolDataset("val-dev", subject=1, size=3)
    model.base.router.calls.clear()

    evaluation = evaluate_development_model(
        model,
        dataset,
        batch_size=2,
        device="cpu",
        seed=19,
    )

    assert evaluation.similarity.shape == (3, 3)
    assert np.allclose(np.diag(evaluation.similarity), 1.0)
    assert set(model.base.router.calls) == {True}


def test_training_drops_incomplete_baseline_batches(
    tmp_path: Path,
    components: UpstreamComponents,
) -> None:
    result = run_training_cell(
        _spec(
            tmp_path,
            components,
            _DatasetFactory(train_size=5),
            max_train_steps=3,
        )
    )

    assert result.global_step == 3
    assert tuple(len(rows) for rows in result.trained_row_indices) == (2, 2, 2)
    assert result.snapshots[0].epoch == 1
    assert result.snapshots[0].epoch_complete is True


def _tensor_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def test_mid_epoch_resume_is_identical_to_uninterrupted_two_steps(
    tmp_path: Path,
    components: UpstreamComponents,
) -> None:
    uninterrupted_spec = _spec(
        tmp_path,
        components,
        _DatasetFactory(train_size=6),
        max_train_steps=2,
    )
    uninterrupted = run_training_cell(uninterrupted_spec)

    split_factory = _DatasetFactory(train_size=6)
    first = run_training_cell(
        _spec(tmp_path, components, split_factory, max_train_steps=1)
    )
    resumed = run_training_cell(
        replace(
            _spec(tmp_path, components, split_factory, max_train_steps=1),
            resume_checkpoint=first.final_checkpoint,
        )
    )

    assert resumed.global_step == uninterrupted.global_step == 2
    assert (
        first.trained_row_indices + resumed.trained_row_indices
        == uninterrupted.trained_row_indices
    )
    assert resumed.sampler_state == uninterrupted.sampler_state
    assert torch.equal(
        resumed.loader_generator_state,
        uninterrupted.loader_generator_state,
    )
    assert np.array_equal(
        resumed.final_validation.similarity,
        uninterrupted.final_validation.similarity,
    )
    resumed_state = _tensor_state(resumed.model)
    uninterrupted_state = _tensor_state(uninterrupted.model)
    assert resumed_state.keys() == uninterrupted_state.keys()
    assert all(
        torch.equal(resumed_state[key], uninterrupted_state[key])
        for key in resumed_state
    )


def test_epoch21_freezes_shared_encoder_and_rebuilds_adamw(
    tmp_path: Path,
    components: UpstreamComponents,
) -> None:
    spec = _spec(tmp_path, components, _DatasetFactory(train_size=6))
    first = run_training_cell(spec)
    end_of_stage1 = dict(first.final_checkpoint)
    end_of_stage1["epoch"] = 20
    end_of_stage1["optimizer_stage"] = "stage1"
    end_of_stage1["runtime_state"] = {
        **dict(end_of_stage1["runtime_state"]),
        "epoch_complete": True,
        "next_epoch": 21,
    }

    resumed = run_training_cell(
        replace(spec, resume_checkpoint=end_of_stage1)
    )

    assert resumed.optimizer_rebuild_epochs == (21,)
    assert not any(
        parameter.requires_grad
        for parameter in resumed.model.base.shared_encoder.parameters()
    )
    assert resumed.final_checkpoint["epoch"] == 21
    assert resumed.final_checkpoint["optimizer_stage"] == "stage2"
    assert resumed.final_checkpoint["runtime_state"]["optimizer_base_lr"] == (
        pytest.approx(5e-5)
    )


def test_epochs_51_to_60_are_retained_and_sink_receipts_are_verified(
    tmp_path: Path,
    components: UpstreamComponents,
) -> None:
    factory = _DatasetFactory(train_size=2, val_size=2)
    base_spec = _spec(tmp_path, components, factory)
    first = run_training_cell(base_spec)
    end_of_stage1 = dict(first.final_checkpoint)
    end_of_stage1["epoch"] = 20
    end_of_stage1["optimizer_stage"] = "stage1"
    end_of_stage1["runtime_state"] = {
        **dict(end_of_stage1["runtime_state"]),
        "epoch_complete": True,
        "next_epoch": 21,
    }
    stage2 = run_training_cell(
        replace(base_spec, resume_checkpoint=end_of_stage1)
    )
    end_of_epoch50 = dict(stage2.final_checkpoint)
    end_of_epoch50["epoch"] = 50
    end_of_epoch50["optimizer_stage"] = "stage2"
    end_of_epoch50["runtime_state"] = {
        **dict(end_of_epoch50["runtime_state"]),
        "epoch_complete": True,
        "next_epoch": 51,
    }

    published: list[tuple[int, bool]] = []

    def sink(
        payload: dict[str, object],
        *,
        retain_for_averaging: bool,
    ) -> CheckpointPublication:
        epoch = int(payload["epoch"])
        published.append((epoch, retain_for_averaging))
        return CheckpointPublication(
            reference=f"memory://epoch-{epoch}",
            exclusive_create=True,
            atomic_publish=True,
            verified=True,
        )

    result = run_training_cell(
        replace(
            base_spec,
            resume_checkpoint=end_of_epoch50,
            max_train_steps=10,
            checkpoint_sink=sink,
        )
    )

    assert result.completed is True
    assert tuple(record.epoch for record in result.snapshots) == tuple(
        range(51, 61)
    )
    assert all(record.retain_for_averaging for record in result.snapshots)
    assert all(record.publication is not None for record in result.snapshots)
    assert published == [(epoch, True) for epoch in range(51, 61)]
    assert result.final_checkpoint["retention"]["required_epochs"] == list(
        range(51, 61)
    )


def test_checkpoint_sink_must_attest_exclusive_atomic_verified_publish(
    tmp_path: Path,
    components: UpstreamComponents,
) -> None:
    def unsafe_sink(
        payload: dict[str, object],
        *,
        retain_for_averaging: bool,
    ) -> CheckpointPublication:
        del payload, retain_for_averaging
        return CheckpointPublication(
            reference="unsafe",
            exclusive_create=False,
            atomic_publish=True,
            verified=True,
        )

    with pytest.raises(ValueError, match="exclusive.*atomic.*verified"):
        run_training_cell(
            _spec(
                tmp_path,
                components,
                _DatasetFactory(train_size=2),
                checkpoint_sink=unsafe_sink,
            )
        )
