from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping
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
import samga_brain_rw.trainer as trainer_module
from samga_brain_rw.adapters import (
    DenseBottleneckControl,
    MatchedPerLayerProjectorControl,
    ResidualFeatureAdapter,
)
from samga_brain_rw.feature_transforms import TrainWhitening
from samga_brain_rw.runtime_contract import build_environment_binding
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


def _environment_binding(
    **semantic_overrides: object,
) -> dict[str, object]:
    semantic_environment: dict[str, object] = {
        "schema_version": 1,
        "python": "synthetic-python",
        "torch": "synthetic-torch",
        "transformers": "synthetic-transformers",
        "peft": "synthetic-peft",
        "numpy": "synthetic-numpy",
        "scipy": "synthetic-scipy",
        "cuda": "synthetic-cuda",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    semantic_environment.update(semantic_overrides)
    runtime_contract = {
        "schema_version": 1,
        "device_type": "cpu",
        "device": "cpu",
        "accelerator_name": "synthetic-cpu",
        "compute_capability": [0, 0],
        "compute_dtype": "float32",
        "autocast": "disabled",
        "cudnn_sdp_enabled": False,
        "cuda_matmul_tf32": False,
        "cudnn_tf32": False,
        "attention_evidence_scope": "synthetic_test_contract_only",
        "torch_sdpa_policy": "math_only",
        "torch_sdpa_canary_passed": False,
        "flash_sdp_enabled": False,
        "math_sdp_enabled": True,
        "mem_efficient_sdp_enabled": False,
    }
    return build_environment_binding(
        semantic_environment,
        runtime_contract,
    )


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


def _attested_memory_sink(
    payload: dict[str, object],
    *,
    retain_for_averaging: bool,
) -> CheckpointPublication:
    return CheckpointPublication(
        reference=f"memory://epoch-{payload['epoch']}",
        exclusive_create=True,
        atomic_publish=True,
        verified=True,
        durable_retention=retain_for_averaging,
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
        "input_hashes": {
            "manifest_sha256": _h("manifest"),
            "cache_sha256": _h("feature-cache"),
        },
        "environment": _environment_binding(),
        "run_manifest": {"run_id": "s0-sub01-seed19"},
        "checkpoint_builder": samga_train.build_epoch_checkpoint,
        "checkpoint_restorer": samga_train.restore_training_checkpoint,
        "dataset_factory": factory,
        "batch_size": 2,
        "max_train_steps": 1,
        "num_workers": 0,
        "device": "cpu",
        "checkpoint_sink": _attested_memory_sink,
    }
    values.update(overrides)
    identities = trainer_module.derive_training_identities(
        components=components,
        train_row_indices=tuple(range(100, 100 + factory.train_size)),
        stage=int(values["stage"]),
        subject=int(values["subject"]),
        seed=int(values["seed"]),
        batch_size=int(values["batch_size"]),
        layernorm_config_id=str(
            values.get("layernorm_config_id", "s2-layernorm-off")
        ),
        whitening_config_id=str(
            values.get("whitening_config_id", "s2-whitening-off")
        ),
        preprojector_config_id=str(
            values.get("preprojector_config_id", "s2-preproj-shared")
        ),
        adapter_kind=str(values.get("adapter_kind", "identity")),
        adapter_rank=values.get("adapter_rank"),  # type: ignore[arg-type]
        adapter_lr_ratio=values.get(  # type: ignore[arg-type]
            "adapter_lr_ratio"
        ),
        whitening=values.get("whitening"),  # type: ignore[arg-type]
    )
    values.setdefault("trajectory_sha256", identities.trajectory_sha256)
    values.setdefault("data_order_sha256", identities.data_order_sha256)
    candidate = dict(
        values.get("candidate_spec", {"config_id": "baseline"})
    )
    candidate.setdefault(
        "full_task_initialization_sha256",
        identities.full_task_initialization_sha256,
    )
    candidate.setdefault(
        "shared_parameter_intersection_name",
        identities.shared_parameter_intersection_name,
    )
    candidate.setdefault(
        "shared_parameter_intersection_sha256",
        identities.shared_parameter_intersection_sha256,
    )
    candidate.setdefault(
        "architecture_specific_initialization_sha256",
        identities.architecture_specific_initialization_sha256,
    )
    values["candidate_spec"] = candidate
    return TrainingCellSpec(**values)


def test_checkpoint_sink_is_mandatory(
    tmp_path: Path,
    components: UpstreamComponents,
) -> None:
    with pytest.raises(ValueError, match="checkpoint_sink.*required"):
        _spec(
            tmp_path,
            components,
            _DatasetFactory(),
            checkpoint_sink=None,
        )


def test_checkpoint_publication_has_typed_durable_retention_attestation() -> None:
    receipt = CheckpointPublication(
        reference="memory://ordinary-snapshot",
        exclusive_create=True,
        atomic_publish=True,
        verified=True,
    )

    assert receipt.durable_retention is False
    with pytest.raises(TypeError, match="durable_retention.*boolean"):
        CheckpointPublication(
            reference="memory://retained-snapshot",
            exclusive_create=True,
            atomic_publish=True,
            verified=True,
            durable_retention=1,  # type: ignore[arg-type]
        )


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


def _provenance_only_whitening(
    *,
    rows: tuple[int, ...],
    manifest_sha256: str,
    cache_sha256: str,
) -> TrainWhitening:
    artifact = object.__new__(TrainWhitening)
    nn.Module.__init__(artifact)
    artifact.register_buffer("mean", torch.zeros(5, 3_200))
    artifact.register_buffer("matrix", torch.empty(0))
    artifact.canonical_train_rows = rows
    artifact.canonical_row_count = max(rows) + 1
    artifact.source_scope = "train"
    artifact.eps = 1e-5
    artifact.input_provenance_sha256 = manifest_sha256
    artifact.cache_provenance_sha256 = cache_sha256
    artifact._payload_sha256 = trainer_module.sha256_json(
        artifact._payload_body()
    )
    return artifact


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("rows", "row-list"),
        ("manifest", "manifest.*provenance"),
        ("cache", "cache.*provenance"),
    ],
)
def test_whitening_binds_current_train_rows_manifest_and_cache_provenance(
    tmp_path: Path,
    components: UpstreamComponents,
    tamper: str,
    message: str,
) -> None:
    base = _spec(
        tmp_path,
        components,
        _DatasetFactory(train_size=2, val_size=2),
    )
    rows = (100, 101)
    manifest_sha256 = base.input_hashes["manifest_sha256"]
    cache_sha256 = base.input_hashes["cache_sha256"]
    if tamper == "rows":
        rows = (100, 102)
    elif tamper == "manifest":
        manifest_sha256 = _h("other-manifest")
    else:
        cache_sha256 = _h("other-cache")
    whitening = _provenance_only_whitening(
        rows=rows,
        manifest_sha256=manifest_sha256,
        cache_sha256=cache_sha256,
    )
    spec = replace(
        base,
        stage=2,
        whitening_config_id="s2-whitening-on",
        whitening=whitening,
    )

    with pytest.raises(ValueError, match=message):
        run_training_cell(spec)


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


def test_data_order_derivation_binds_rows_seed_batch_and_all_60_epochs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    derive = getattr(trainer_module, "derive_data_order_sha256")
    rows = tuple(range(100, 106))
    baseline = derive(rows, seed=19, batch_size=2)

    assert baseline != derive(tuple(reversed(rows)), seed=19, batch_size=2)
    assert baseline != derive(rows, seed=20, batch_size=2)
    assert baseline != derive(rows, seed=19, batch_size=3)

    original = StatefulEpochSampler._order_for_epoch

    def changed_epoch_60(
        self: StatefulEpochSampler,
        epoch: int,
    ) -> tuple[int, ...]:
        order = original(self, epoch)
        if epoch != 60:
            return order
        values = list(order)
        values[0], values[1] = values[1], values[0]
        return tuple(values)

    monkeypatch.setattr(
        StatefulEpochSampler,
        "_order_for_epoch",
        changed_epoch_60,
    )
    assert derive(rows, seed=19, batch_size=2) != baseline


def test_public_identity_helper_derives_actual_seeded_initializations(
    components: UpstreamComponents,
) -> None:
    derive = getattr(trainer_module, "derive_training_identities")
    arguments = {
        "components": components,
        "train_row_indices": tuple(range(100, 106)),
        "stage": 2,
        "subject": 1,
        "seed": 19,
        "batch_size": 2,
        "adapter_kind": "adapter",
        "adapter_rank": 8,
        "adapter_lr_ratio": 0.05,
    }

    first = derive(**arguments)
    repeated = derive(**arguments)
    other_seed = derive(**{**arguments, "seed": 20})

    assert first == repeated
    assert first.data_order_sha256 == trainer_module.derive_data_order_sha256(
        arguments["train_row_indices"],
        seed=19,
        batch_size=2,
    )
    assert first.full_task_initialization_sha256 != (
        other_seed.full_task_initialization_sha256
    )
    assert first.shared_parameter_intersection_sha256 == (
        other_seed.shared_parameter_intersection_sha256
    )
    assert first.architecture_specific_initialization_sha256 != (
        other_seed.architecture_specific_initialization_sha256
    )
    assert first.shared_parameter_intersection_name
    for value in (
        first.data_order_sha256,
        first.trajectory_sha256,
        first.full_task_initialization_sha256,
        first.shared_parameter_intersection_sha256,
        first.architecture_specific_initialization_sha256,
    ):
        assert len(value) == 64
        assert set(value) <= set("0123456789abcdef")


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("data_order_sha256", "data.order"),
        ("trajectory_sha256", "trajectory"),
        (
            "architecture_specific_initialization_sha256",
            "architecture.specific",
        ),
    ],
)
def test_runtime_recomputes_and_rejects_identity_declaration_tampering(
    tmp_path: Path,
    components: UpstreamComponents,
    field: str,
    message: str,
) -> None:
    spec = _spec(
        tmp_path,
        components,
        _DatasetFactory(train_size=2, val_size=2),
    )
    if field == "data_order_sha256":
        tampered = replace(spec, data_order_sha256=_h("wrong-data-order"))
    elif field == "trajectory_sha256":
        tampered = replace(spec, trajectory_sha256=_h("wrong-trajectory"))
    else:
        candidate = dict(spec.candidate_spec)
        candidate[field] = _h("wrong-initialization")
        tampered = replace(spec, candidate_spec=candidate)

    with pytest.raises(ValueError, match=message):
        run_training_cell(tampered)


def test_runtime_derives_data_order_from_actual_factory_rows(
    tmp_path: Path,
    components: UpstreamComponents,
) -> None:
    class _ReorderedFactory(_DatasetFactory):
        def __call__(self, **kwargs: object) -> _TinyProtocolDataset:
            dataset = super().__call__(**kwargs)
            if kwargs["scope"] == "train":
                dataset.row_indices = tuple(reversed(dataset.row_indices))
            return dataset

    with pytest.raises(ValueError, match="data.order"):
        run_training_cell(
            _spec(
                tmp_path,
                components,
                _ReorderedFactory(train_size=2, val_size=2),
            )
        )


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


def test_fresh_optimizer_groups_match_complete_locked_adamw_recipe(
    tmp_path: Path,
    components: UpstreamComponents,
) -> None:
    result = run_training_cell(
        _spec(
            tmp_path,
            components,
            _DatasetFactory(train_size=6, val_size=2),
        )
    )
    optimizer_state = result.final_checkpoint["optimizer_state_dict"]
    assert isinstance(optimizer_state, dict)
    groups = optimizer_state["param_groups"]
    assert isinstance(groups, list)
    assert groups
    expected = {
        "lr": samga_train.SCHEDULE["stage1_learning_rate"],
        "initial_lr": samga_train.SCHEDULE["stage1_learning_rate"],
        "betas": tuple(samga_train.SCHEDULE["betas"]),
        "eps": samga_train.SCHEDULE["eps"],
        "weight_decay": samga_train.SCHEDULE["weight_decay"],
        "amsgrad": samga_train.SCHEDULE["amsgrad"],
        "maximize": samga_train.SCHEDULE["maximize"],
        "foreach": samga_train.SCHEDULE["foreach"],
        "capturable": samga_train.SCHEDULE["capturable"],
        "differentiable": samga_train.SCHEDULE["differentiable"],
        "fused": samga_train.SCHEDULE["fused"],
    }
    for group in groups:
        assert {key: group[key] for key in expected} == expected


@pytest.mark.parametrize(
    ("optimizer_stage", "recipe_key", "changed_lr"),
    [
        ("stage1", "stage1_learning_rate", 2e-4),
        ("stage2", "stage2_learning_rate", 3e-5),
    ],
)
def test_optimizer_stage_lr_is_read_from_the_locked_recipe(
    tmp_path: Path,
    components: UpstreamComponents,
    monkeypatch: pytest.MonkeyPatch,
    optimizer_stage: str,
    recipe_key: str,
    changed_lr: float,
) -> None:
    spec = _spec(
        tmp_path,
        components,
        _DatasetFactory(train_size=6, val_size=2),
    )
    model = SAMGARuntimeModel(spec)
    monkeypatch.setitem(trainer_module.SCHEDULE, recipe_key, changed_lr)

    optimizer, _ = trainer_module._build_optimizer(model, optimizer_stage)

    assert optimizer.param_groups[0]["lr"] == changed_lr


@pytest.mark.parametrize(
    ("requested_scope", "metadata_name", "metadata_value", "message"),
    [
        ("train", "scope", "val-dev", "scope"),
        ("val-dev", "subject_id", 2, "subject"),
    ],
)
def test_factory_result_must_match_requested_scope_and_subject(
    tmp_path: Path,
    components: UpstreamComponents,
    requested_scope: str,
    metadata_name: str,
    metadata_value: object,
    message: str,
) -> None:
    class _LyingFactory(_DatasetFactory):
        def __call__(self, **kwargs: object) -> _TinyProtocolDataset:
            dataset = super().__call__(**kwargs)
            if kwargs["scope"] == requested_scope:
                setattr(dataset, metadata_name, metadata_value)
            return dataset

    with pytest.raises(ValueError, match=message):
        run_training_cell(
            _spec(
                tmp_path,
                components,
                _LyingFactory(train_size=2, val_size=2),
            )
        )


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
            resume_source_checkpoint_sha256=_h("first-resume-source"),
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


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("missing_model_hash", "schema|model.*hash"),
        ("wrong_model_hash", "model.*hash"),
        ("missing_rng", "schema|RNG"),
        ("scheduler_step", "scheduler.*step"),
        ("optimizer_lr", "optimizer"),
        ("epoch", "epoch|position"),
    ],
)
def test_checkpoint_builder_result_requires_complete_semantic_schema(
    tmp_path: Path,
    components: UpstreamComponents,
    tamper: str,
    message: str,
) -> None:
    def builder(**kwargs: object) -> dict[str, object]:
        payload = samga_train.build_epoch_checkpoint(**kwargs)
        if tamper == "missing_model_hash":
            payload.pop("model_state_sha256")
        elif tamper == "wrong_model_hash":
            payload["model_state_sha256"] = _h("wrong-model")
        elif tamper == "missing_rng":
            payload.pop("python_rng_state")
        elif tamper == "scheduler_step":
            scheduler_state = copy.deepcopy(payload["scheduler_state_dict"])
            scheduler_state["last_epoch"] = 99
            payload["scheduler_state_dict"] = scheduler_state
        elif tamper == "optimizer_lr":
            optimizer_state = copy.deepcopy(payload["optimizer_state_dict"])
            optimizer_state["param_groups"][0]["lr"] = 9.0
            payload["optimizer_state_dict"] = optimizer_state
        else:
            payload["epoch"] = 2
        return payload

    with pytest.raises(ValueError, match=message):
        run_training_cell(
            _spec(
                tmp_path,
                components,
                _DatasetFactory(train_size=2, val_size=2),
                checkpoint_builder=builder,
            )
        )


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("model_state", "model.*hash"),
        ("global_step", "global.*step|position"),
        ("sampler_epoch", "sampler.*epoch|position"),
        ("epoch_complete", "sampler.*position|epoch.complete"),
        ("scheduler_step", "scheduler.*step"),
        ("snapshot_prefix", "snapshot.*prefix"),
    ],
)
def test_resume_rejects_cross_field_checkpoint_tampering(
    tmp_path: Path,
    components: UpstreamComponents,
    tamper: str,
    message: str,
) -> None:
    spec = _spec(
        tmp_path,
        components,
        _DatasetFactory(train_size=6, val_size=2),
    )
    first = run_training_cell(spec)
    checkpoint = copy.deepcopy(first.final_checkpoint)
    if tamper == "model_state":
        model_state = checkpoint["model_state_dict"]
        assert isinstance(model_state, dict)
        first_key = next(iter(model_state))
        model_state[first_key] = model_state[first_key] + 1
    elif tamper == "global_step":
        checkpoint["global_step"] = 2
    elif tamper == "sampler_epoch":
        sampler_state = checkpoint["sampler_state_dict"]
        assert isinstance(sampler_state, dict)
        epoch_two = StatefulEpochSampler(
            dataset_size=6,
            seed=19,
            epoch=2,
        ).state_dict()
        sampler_state["epoch"] = 2
        sampler_state["order"] = epoch_two["order"]
    elif tamper == "epoch_complete":
        runtime = checkpoint["runtime_state"]
        assert isinstance(runtime, dict)
        runtime["epoch_complete"] = True
        runtime["next_epoch"] = 2
        runtime["snapshot_epochs"] = [1]
    elif tamper == "scheduler_step":
        scheduler = checkpoint["scheduler_state_dict"]
        assert isinstance(scheduler, dict)
        scheduler["last_epoch"] = 2
    else:
        runtime = checkpoint["runtime_state"]
        assert isinstance(runtime, dict)
        runtime["snapshot_epochs"] = [1]

    with pytest.raises(ValueError, match=message):
        run_training_cell(
            replace(
                spec,
                resume_checkpoint=checkpoint,
                resume_source_checkpoint_sha256=_h("tampered-source"),
            )
        )


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("model", "post.restore.*model.*hash"),
        ("optimizer", "post.restore.*optimizer"),
        ("scheduler", "post.restore.*scheduler"),
        ("generator", "post.restore.*generator"),
        ("rng", "post.restore.*RNG"),
    ],
)
def test_resume_verifies_objects_after_restorer_returns(
    tmp_path: Path,
    components: UpstreamComponents,
    corruption: str,
    message: str,
) -> None:
    base = _spec(
        tmp_path,
        components,
        _DatasetFactory(train_size=6, val_size=2),
    )
    first = run_training_cell(base)

    def corrupting_restorer(
        payload: dict[str, object],
        **kwargs: object,
    ) -> tuple[int, int]:
        restored = samga_train.restore_training_checkpoint(
            payload,
            **kwargs,
        )
        if corruption == "model":
            model = kwargs["model"]
            assert isinstance(model, nn.Module)
            with torch.no_grad():
                next(model.parameters()).add_(1)
        elif corruption == "optimizer":
            optimizer = kwargs["optimizer"]
            assert isinstance(optimizer, torch.optim.Optimizer)
            optimizer.param_groups[0]["lr"] = 9.0
        elif corruption == "scheduler":
            scheduler = kwargs["scheduler"]
            scheduler.last_epoch += 1
        elif corruption == "generator":
            generator = kwargs["generator"]
            assert isinstance(generator, torch.Generator)
            generator.manual_seed(999)
        else:
            torch.manual_seed(999)
        return restored

    with pytest.raises(ValueError, match=message):
        run_training_cell(
            replace(
                base,
                resume_checkpoint=first.final_checkpoint,
                resume_source_checkpoint_sha256=_h("corrupt-source"),
                checkpoint_restorer=corrupting_restorer,
            )
        )


def test_epoch21_freezes_shared_encoder_and_rebuilds_adamw(
    tmp_path: Path,
    components: UpstreamComponents,
) -> None:
    spec = _spec(tmp_path, components, _DatasetFactory(train_size=6))
    stage1 = run_training_cell(replace(spec, max_train_steps=60))
    assert stage1.final_checkpoint["epoch"] == 20
    assert stage1.final_checkpoint["runtime_state"]["epoch_complete"] is True

    resumed = run_training_cell(
        replace(
            spec,
            resume_checkpoint=stage1.final_checkpoint,
            resume_source_checkpoint_sha256=_h(
                "stage1-complete-source"
            ),
        )
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
    first_50 = run_training_cell(replace(base_spec, max_train_steps=50))
    assert first_50.final_checkpoint["epoch"] == 50
    assert first_50.final_checkpoint["runtime_state"]["epoch_complete"] is True

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
            durable_retention=True,
        )

    result = run_training_cell(
        replace(
            base_spec,
            resume_checkpoint=first_50.final_checkpoint,
            resume_source_checkpoint_sha256=_h("epoch50-source"),
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


def test_retained_checkpoint_requires_durable_retention_attestation(
    tmp_path: Path,
    components: UpstreamComponents,
) -> None:
    base_spec = _spec(
        tmp_path,
        components,
        _DatasetFactory(train_size=2, val_size=2),
    )
    first_50 = run_training_cell(replace(base_spec, max_train_steps=50))

    def nondurable_sink(
        payload: dict[str, object],
        *,
        retain_for_averaging: bool,
    ) -> CheckpointPublication:
        del payload
        assert retain_for_averaging is True
        return CheckpointPublication(
            reference="memory://volatile",
            exclusive_create=True,
            atomic_publish=True,
            verified=True,
            durable_retention=False,
        )

    with pytest.raises(ValueError, match="durable.*retention"):
        run_training_cell(
            replace(
                base_spec,
                resume_checkpoint=first_50.final_checkpoint,
                resume_source_checkpoint_sha256=_h("retention-source"),
                max_train_steps=1,
                checkpoint_sink=nondurable_sink,
            )
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


@pytest.mark.parametrize(
    "field",
    ["python", "torch", "numpy", "cuda"],
)
def test_resume_environment_mismatch_fails_before_dataset_or_restorer(
    tmp_path: Path,
    components: UpstreamComponents,
    field: str,
) -> None:
    factory = _DatasetFactory(train_size=6, val_size=2)
    spec = _spec(tmp_path, components, factory)
    first = run_training_cell(spec)
    checkpoint = copy.deepcopy(first.final_checkpoint)
    checkpoint["environment"] = _environment_binding(
        **{field: f"mismatched-{field}"}
    )
    factory.calls.clear()
    restorer_calls: list[object] = []

    def recording_restorer(
        payload: Mapping[str, object],
        **kwargs: object,
    ) -> tuple[int, int]:
        restorer_calls.append((payload, kwargs))
        return samga_train.restore_training_checkpoint(
            payload,
            **kwargs,
        )

    with pytest.raises(ValueError, match="environment.*mismatch"):
        run_training_cell(
            replace(
                spec,
                resume_checkpoint=checkpoint,
                resume_source_checkpoint_sha256=_h(
                    f"environment-source-{field}"
                ),
                checkpoint_restorer=recording_restorer,
            )
        )

    assert factory.calls == []
    assert restorer_calls == []


@pytest.mark.parametrize(
    ("group_index", "field"),
    [
        (0, "weight_decay"),
        (0, "betas"),
        (0, "eps"),
        (1, "lr"),
        (1, "initial_lr"),
        (0, "params"),
    ],
)
def test_resume_rejects_every_optimizer_group_recipe_tamper_before_restore(
    tmp_path: Path,
    components: UpstreamComponents,
    group_index: int,
    field: str,
) -> None:
    spec = _spec(
        tmp_path,
        components,
        _DatasetFactory(train_size=6, val_size=2),
        stage=2,
        candidate_spec={"config_id": "adapter-candidate"},
        adapter_kind="adapter",
        adapter_rank=8,
        adapter_lr_ratio=0.05,
    )
    first = run_training_cell(spec)
    checkpoint = copy.deepcopy(first.final_checkpoint)
    optimizer_state = checkpoint["optimizer_state_dict"]
    scheduler_state = checkpoint["scheduler_state_dict"]
    assert isinstance(optimizer_state, dict)
    assert isinstance(scheduler_state, dict)
    groups = optimizer_state["param_groups"]
    assert isinstance(groups, list)
    assert len(groups) == 2
    group = groups[group_index]
    assert isinstance(group, dict)
    if field == "weight_decay":
        group[field] = 0.25
    elif field == "betas":
        group[field] = (0.8, 0.9)
    elif field == "eps":
        group[field] = 1e-7
    elif field == "lr":
        group[field] = float(group[field]) * 2.0
        scheduler_state["base_lrs"][group_index] = group[field]
        scheduler_state["_last_lr"][group_index] = group[field]
    elif field == "initial_lr":
        group[field] = float(group[field]) * 2.0
    else:
        parameters = list(group[field])
        assert len(parameters) >= 2
        group[field] = list(reversed(parameters))
    restorer_calls: list[object] = []

    def recording_restorer(
        payload: Mapping[str, object],
        **kwargs: object,
    ) -> tuple[int, int]:
        restorer_calls.append((payload, kwargs))
        return samga_train.restore_training_checkpoint(
            payload,
            **kwargs,
        )

    with pytest.raises(ValueError, match="optimizer.*recipe"):
        run_training_cell(
            replace(
                spec,
                resume_checkpoint=checkpoint,
                resume_source_checkpoint_sha256=_h(
                    f"optimizer-source-{group_index}-{field}"
                ),
                checkpoint_restorer=recording_restorer,
            )
        )

    assert restorer_calls == []


def test_checkpoint_lineage_records_actual_immediate_resume_source(
    tmp_path: Path,
    components: UpstreamComponents,
) -> None:
    spec = _spec(
        tmp_path,
        components,
        _DatasetFactory(train_size=6, val_size=2),
    )
    fresh = run_training_cell(spec)
    assert (
        fresh.final_checkpoint["runtime_state"][
            "resume_source_checkpoint_sha256"
        ]
        is None
    )

    actual_source = _h("actual-transport-checkpoint")
    resumed = run_training_cell(
        replace(
            spec,
            resume_checkpoint=fresh.final_checkpoint,
            resume_source_checkpoint_sha256=actual_source,
        )
    )

    assert resumed.final_checkpoint["runtime_state"][
        "resume_source_checkpoint_sha256"
    ] == actual_source
    assert (
        resumed.final_checkpoint["input_hashes"]
        == fresh.final_checkpoint["input_hashes"]
    )
    assert (
        resumed.final_checkpoint["run_manifest"]
        == fresh.final_checkpoint["run_manifest"]
    )
    assert (
        resumed.final_checkpoint["config_sha256"]
        == fresh.final_checkpoint["config_sha256"]
    )


def test_resume_payload_and_source_sha_lineage_are_coupled(
    tmp_path: Path,
    components: UpstreamComponents,
) -> None:
    spec = _spec(
        tmp_path,
        components,
        _DatasetFactory(train_size=6, val_size=2),
    )
    with pytest.raises(ValueError, match="resume.*source|lineage"):
        replace(
            spec,
            resume_source_checkpoint_sha256=_h("orphan-source"),
        )

    first = run_training_cell(spec)
    with pytest.raises(ValueError, match="resume.*source|lineage"):
        replace(
            spec,
            resume_checkpoint=first.final_checkpoint,
        )
