"""Deterministic 60-epoch runtime for one sealed SAMGA training cell.

The runtime owns no output path.  Checkpoint publication is an injected
operation whose receipt must attest exclusive creation, atomic publication,
and post-publication verification.
"""

from __future__ import annotations

import copy
import hashlib
import math
import random
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from .adapters import (
    ADAPTER_RANKS,
    LEARNING_RATE_RATIOS,
    DenseBottleneckControl,
    MatchedPerLayerProjectorControl,
    ResidualFeatureAdapter,
)
from .checkpoints import CHECKPOINT_PAYLOAD_TYPE, hash_state_dict
from .data import POSTERIOR_CHANNELS, ProtocolSubjectDataset
from .feature_transforms import LayerNormTransform, TrainWhitening
from .hashing import sha256_json
from .model import SAMGALossOutput, SAMGATaskModel
from .runtime_contract import validate_environment_binding
from .scores import RetrievalMetrics, independent_retrieval_metrics
from .upstream_samga import UpstreamComponents


PINNED_UPSTREAM_SHA = "1a63745b7ff6f98dad34b0f0b8246a9b5260d9c1"
TOTAL_EPOCHS = 60
STAGE1_EPOCHS = 20
LAST10_EPOCHS = tuple(range(51, 61))
SCHEDULE = {
    "epochs": 60,
    "stage1_epochs": 20,
    "stage1_learning_rate": 1e-4,
    "stage2_learning_rate": 5e-5,
    "mmd_start": 0.9,
    "mmd_end": 0.5,
    "optimizer": "AdamW",
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "weight_decay": 1e-4,
    "amsgrad": False,
    "maximize": False,
    "foreach": None,
    "capturable": False,
    "differentiable": False,
    "fused": None,
    "scheduler": "constant_per_optimizer_stage",
}
SCHEDULE_SHA256 = sha256_json(SCHEDULE)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LAYERNORM_IDS = ("s2-layernorm-off", "s2-layernorm-on")
_WHITENING_IDS = ("s2-whitening-off", "s2-whitening-on")
_PREPROJECTOR_IDS = ("s2-preproj-shared", "s2-preproj-separate")
_ADAPTER_KINDS = ("identity", "adapter", "global_dense", "matched_projector")
SHARED_PARAMETER_INTERSECTION_NAME = (
    "SAMGARuntimeModel.state_dict_without_active_factor"
)
_CHECKPOINT_KEYS = {
    "schema_version",
    "payload_type",
    "epoch",
    "global_step",
    "subject",
    "seed",
    "config_sha256",
    "schedule_sha256",
    "optimizer_stage",
    "trajectory_sha256",
    "data_order_sha256",
    "model_state_dict",
    "model_state_sha256",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "python_rng_state",
    "numpy_rng_state",
    "torch_rng_state",
    "cuda_rng_states",
    "loader_generator_state",
    "sampler_state_dict",
    "validation_metrics",
    "input_hashes",
    "effective_batch",
    "environment",
    "run_manifest",
    "candidate_spec",
    "runtime_state",
    "retention",
}
_RUNTIME_STATE_KEYS = {
    "schema_version",
    "epoch_complete",
    "next_epoch",
    "resume_source_checkpoint_sha256",
    "optimizer_base_lr",
    "iterator_generator_state",
    "snapshot_epochs",
    "required_retained_epochs",
}
_RETENTION_KEYS = {
    "policy",
    "required_epochs",
    "retain_for_averaging",
}

DatasetFactory = Callable[..., Dataset[dict[str, object]]]
CheckpointBuilder = Callable[..., Mapping[str, object]]
CheckpointRestorer = Callable[..., tuple[int, int]]


@dataclass(frozen=True)
class CheckpointPublication:
    """A sink's attestation for one externally published checkpoint."""

    reference: object
    exclusive_create: bool
    atomic_publish: bool
    verified: bool
    durable_retention: bool = False

    def __post_init__(self) -> None:
        for name in (
            "exclusive_create",
            "atomic_publish",
            "verified",
            "durable_retention",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be boolean")


CheckpointSink = Callable[..., CheckpointPublication]


@dataclass(frozen=True)
class TrainingIdentities:
    """Runtime-derived ordering and pre-training initialization identities."""

    data_order_sha256: str
    trajectory_sha256: str
    full_task_initialization_sha256: str
    shared_parameter_intersection_name: str
    shared_parameter_intersection_sha256: str
    architecture_specific_initialization_sha256: str


@dataclass(frozen=True)
class _ModelBuildInputs:
    components: UpstreamComponents
    stage: int
    subject: int
    active_factor: str | None
    layernorm_config_id: str
    whitening_config_id: str
    preprojector_config_id: str
    adapter_kind: str
    adapter_rank: int | None
    adapter_lr_ratio: float | None
    whitening: TrainWhitening | None


@dataclass(frozen=True)
class TrainingCellSpec:
    """All identity, data, factor, and persistence inputs for one run cell."""

    components: UpstreamComponents
    manifest_path: Path
    feature_cache: Path
    stage: int
    subject: int
    seed: int
    config_sha256: str
    schedule_sha256: str
    trajectory_sha256: str
    data_order_sha256: str
    input_hashes: Mapping[str, str]
    environment: Mapping[str, object]
    run_manifest: Mapping[str, object]
    candidate_spec: Mapping[str, object]
    checkpoint_builder: CheckpointBuilder
    checkpoint_restorer: CheckpointRestorer
    checkpoint_sink: CheckpointSink
    dataset_factory: DatasetFactory = ProtocolSubjectDataset
    batch_size: int = 512
    max_train_steps: int | None = None
    num_workers: int = 0
    device: str | torch.device = "auto"
    resume_checkpoint: Mapping[str, object] | None = None
    resume_source_checkpoint_sha256: str | None = None
    layernorm_config_id: str = "s2-layernorm-off"
    whitening_config_id: str = "s2-whitening-off"
    preprojector_config_id: str = "s2-preproj-shared"
    adapter_kind: str = "identity"
    adapter_rank: int | None = None
    adapter_lr_ratio: float | None = None
    whitening: TrainWhitening | None = None
    active_factor: str | None = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.components, UpstreamComponents):
            raise TypeError("components must be verified UpstreamComponents")
        if self.components.commit != PINNED_UPSTREAM_SHA:
            raise ValueError(
                f"upstream commit is locked to {PINNED_UPSTREAM_SHA}"
            )
        object.__setattr__(self, "manifest_path", Path(self.manifest_path))
        object.__setattr__(self, "feature_cache", Path(self.feature_cache))
        if type(self.stage) is not int or self.stage not in (0, 2):
            raise ValueError("stage must be 0 or 2")
        if type(self.subject) is not int or not 1 <= self.subject <= 10:
            raise ValueError("subject must be in 1..10")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        for name in (
            "config_sha256",
            "schedule_sha256",
            "trajectory_sha256",
            "data_order_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if self.schedule_sha256 != SCHEDULE_SHA256:
            raise ValueError("schedule_sha256 does not match the locked schedule")
        _validate_hash_mapping(self.input_hashes, "input_hashes")
        object.__setattr__(
            self,
            "environment",
            validate_environment_binding(self.environment),
        )
        _validate_string_mapping(self.run_manifest, "run_manifest")
        _validate_string_mapping(self.candidate_spec, "candidate_spec")
        for value, name in (
            (self.checkpoint_builder, "checkpoint_builder"),
            (self.checkpoint_restorer, "checkpoint_restorer"),
            (self.dataset_factory, "dataset_factory"),
        ):
            if not callable(value):
                raise TypeError(f"{name} must be callable")
        if not callable(self.checkpoint_sink):
            raise ValueError("checkpoint_sink is required and must be callable")
        if type(self.batch_size) is not int or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if self.max_train_steps is not None and (
            type(self.max_train_steps) is not int
            or self.max_train_steps <= 0
        ):
            raise ValueError("max_train_steps must be a positive integer")
        if type(self.num_workers) is not int or self.num_workers != 0:
            raise ValueError(
                "num_workers is locked to zero for exact sampler resume"
            )
        _resolve_device(self.device)
        if self.layernorm_config_id not in _LAYERNORM_IDS:
            raise ValueError("unknown Stage 2 layernorm config_id")
        if self.whitening_config_id not in _WHITENING_IDS:
            raise ValueError("unknown Stage 2 whitening config_id")
        if self.preprojector_config_id not in _PREPROJECTOR_IDS:
            raise ValueError("unknown Stage 2 preprojector config_id")
        if self.adapter_kind not in _ADAPTER_KINDS:
            raise ValueError("unknown Stage 2 adapter kind")

        factor_flags = {
            "layernorm": self.layernorm_config_id == "s2-layernorm-on",
            "whitening": self.whitening_config_id == "s2-whitening-on",
            "preprojectors": (
                self.preprojector_config_id == "s2-preproj-separate"
            ),
            "feature_adapter": self.adapter_kind != "identity",
        }
        active = tuple(
            name for name, enabled in factor_flags.items() if enabled
        )
        if self.stage == 0 and active:
            raise ValueError("Stage 0 cannot enable a Stage 2 factor")
        if len(active) > 1:
            raise ValueError("Stage 2 permits exactly zero or one active factor")

        if self.adapter_kind == "identity":
            if self.adapter_rank is not None or self.adapter_lr_ratio is not None:
                raise ValueError(
                    "identity adapter cannot set adapter rank or LR ratio"
                )
        else:
            if self.adapter_rank not in ADAPTER_RANKS:
                raise ValueError(
                    f"adapter_rank must be one of {ADAPTER_RANKS}"
                )
            if self.adapter_lr_ratio not in LEARNING_RATE_RATIOS:
                raise ValueError(
                    "adapter_lr_ratio must be one of "
                    f"{LEARNING_RATE_RATIOS}"
                )
        if self.whitening_config_id == "s2-whitening-on":
            if not isinstance(self.whitening, TrainWhitening):
                raise ValueError(
                    "s2-whitening-on requires a fitted TrainWhitening"
                )
            if self.whitening.source_scope != "train":
                raise ValueError("TrainWhitening must be fit from train scope")
            if tuple(self.whitening.mean.shape) != (5, 3_200):
                raise ValueError("TrainWhitening must target [5,3200] features")
        elif self.whitening is not None:
            raise ValueError(
                "TrainWhitening is only accepted by s2-whitening-on"
            )
        if self.resume_checkpoint is None:
            if self.resume_source_checkpoint_sha256 is not None:
                raise ValueError(
                    "resume source lineage requires a resume checkpoint"
                )
        else:
            _validate_string_mapping(
                self.resume_checkpoint,
                "resume_checkpoint",
            )
            if self.resume_source_checkpoint_sha256 is None:
                raise ValueError(
                    "resume checkpoint requires its actual source SHA lineage"
                )
            _require_sha256(
                self.resume_source_checkpoint_sha256,
                "resume_source_checkpoint_sha256",
            )
        object.__setattr__(self, "active_factor", active[0] if active else None)


@dataclass(frozen=True)
class ValidationResult:
    """One global-router cosine retrieval evaluation."""

    similarity: np.ndarray
    metrics: RetrievalMetrics


@dataclass(frozen=True)
class SnapshotRecord:
    """One checkpoint candidate and its optional external publication."""

    epoch: int
    optimizer_stage: str
    epoch_complete: bool
    retain_for_averaging: bool
    publication: CheckpointPublication | None


@dataclass(frozen=True)
class TrainingResult:
    """In-memory result of a complete or max-step-limited invocation."""

    model: "SAMGARuntimeModel"
    global_step: int
    completed: bool
    final_validation: ValidationResult
    final_checkpoint: dict[str, object]
    sampler_state: dict[str, object]
    loader_generator_state: torch.Tensor
    snapshots: tuple[SnapshotRecord, ...]
    trained_row_indices: tuple[tuple[int, ...], ...]
    optimizer_rebuild_epochs: tuple[int, ...]


class StatefulEpochSampler(Sampler[int]):
    """Epoch-keyed random permutation with an explicit resumable cursor."""

    def __init__(
        self,
        *,
        dataset_size: int,
        seed: int,
        epoch: int = 1,
    ) -> None:
        if type(dataset_size) is not int or dataset_size <= 0:
            raise ValueError("dataset_size must be a positive integer")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if type(epoch) is not int or epoch <= 0:
            raise ValueError("epoch must be a positive integer")
        self.dataset_size = dataset_size
        self.seed = seed
        self.epoch = epoch
        self.position = 0
        self._order = self._order_for_epoch(epoch)

    def _order_for_epoch(self, epoch: int) -> tuple[int, ...]:
        generator = torch.Generator(device="cpu")
        mixed_seed = (
            self.seed + 0x6A09E667F3BCC909 * epoch
        ) % (2**63 - 1)
        generator.manual_seed(mixed_seed)
        return tuple(
            int(value)
            for value in torch.randperm(
                self.dataset_size,
                generator=generator,
            ).tolist()
        )

    def __iter__(self) -> Iterator[int]:
        while self.position < self.dataset_size:
            value = self._order[self.position]
            self.position += 1
            yield value

    def __len__(self) -> int:
        return self.dataset_size - self.position

    @property
    def exhausted(self) -> bool:
        return self.position == self.dataset_size

    def set_epoch(self, epoch: int) -> None:
        if type(epoch) is not int or epoch <= 0:
            raise ValueError("epoch must be a positive integer")
        if epoch == self.epoch:
            return
        self.epoch = epoch
        self.position = 0
        self._order = self._order_for_epoch(epoch)

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "dataset_size": self.dataset_size,
            "seed": self.seed,
            "epoch": self.epoch,
            "position": self.position,
            "order": list(self._order),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("sampler state must be a mapping")
        if state.get("schema_version") != 1:
            raise ValueError("sampler state schema_version must be 1")
        if state.get("dataset_size") != self.dataset_size:
            raise ValueError("sampler state dataset size mismatch")
        if state.get("seed") != self.seed:
            raise ValueError("sampler state seed mismatch")
        epoch = state.get("epoch")
        position = state.get("position")
        order = state.get("order")
        if type(epoch) is not int or epoch <= 0:
            raise ValueError("sampler state epoch is invalid")
        if (
            type(position) is not int
            or position < 0
            or position > self.dataset_size
        ):
            raise ValueError("sampler state position is invalid")
        if (
            not isinstance(order, Sequence)
            or isinstance(order, (str, bytes, bytearray))
            or any(type(value) is not int for value in order)
        ):
            raise ValueError("sampler state order is invalid")
        normalized = tuple(int(value) for value in order)
        if len(normalized) != self.dataset_size or tuple(
            sorted(normalized)
        ) != tuple(range(self.dataset_size)):
            raise ValueError("sampler state order is not a permutation")
        if normalized != self._order_for_epoch(epoch):
            raise ValueError("sampler state order differs from its seed and epoch")
        self.epoch = epoch
        self.position = position
        self._order = normalized


def derive_data_order_sha256(
    train_row_indices: Sequence[int],
    *,
    seed: int,
    batch_size: int,
) -> str:
    """Hash the complete locked 60-epoch, drop-last training order."""

    if (
        not isinstance(train_row_indices, Sequence)
        or isinstance(train_row_indices, (str, bytes, bytearray))
        or not train_row_indices
        or any(type(row) is not int or row < 0 for row in train_row_indices)
    ):
        raise ValueError(
            "train_row_indices must be a nonempty sequence of non-negative integers"
        )
    rows = tuple(train_row_indices)
    if len(set(rows)) != len(rows):
        raise ValueError("train_row_indices must be unique")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    sampler = StatefulEpochSampler(
        dataset_size=len(rows),
        seed=seed,
    )
    consumed = len(rows) - (len(rows) % batch_size)
    epochs: list[dict[str, object]] = []
    for epoch in range(1, TOTAL_EPOCHS + 1):
        permutation = sampler._order_for_epoch(epoch)
        ordered_rows = [rows[index] for index in permutation]
        epochs.append(
            {
                "epoch": epoch,
                "permutation": list(permutation),
                "ordered_row_indices": ordered_rows,
                "consumed_row_indices": ordered_rows[:consumed],
            }
        )
    return sha256_json(
        {
            "schema_version": 1,
            "payload_type": "samga_brain_rw.training_data_order",
            "ordered_train_row_indices": list(rows),
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "drop_last": True,
        }
    )


def _validated_model_build_inputs(
    *,
    components: UpstreamComponents,
    stage: int,
    subject: int,
    layernorm_config_id: str,
    whitening_config_id: str,
    preprojector_config_id: str,
    adapter_kind: str,
    adapter_rank: int | None,
    adapter_lr_ratio: float | None,
    whitening: TrainWhitening | None,
) -> _ModelBuildInputs:
    if not isinstance(components, UpstreamComponents):
        raise TypeError("components must be verified UpstreamComponents")
    if components.commit != PINNED_UPSTREAM_SHA:
        raise ValueError(f"upstream commit is locked to {PINNED_UPSTREAM_SHA}")
    if type(stage) is not int or stage not in (0, 2):
        raise ValueError("stage must be 0 or 2")
    if type(subject) is not int or not 1 <= subject <= 10:
        raise ValueError("subject must be in 1..10")
    if layernorm_config_id not in _LAYERNORM_IDS:
        raise ValueError("unknown Stage 2 layernorm config_id")
    if whitening_config_id not in _WHITENING_IDS:
        raise ValueError("unknown Stage 2 whitening config_id")
    if preprojector_config_id not in _PREPROJECTOR_IDS:
        raise ValueError("unknown Stage 2 preprojector config_id")
    if adapter_kind not in _ADAPTER_KINDS:
        raise ValueError("unknown Stage 2 adapter kind")

    factor_flags = {
        "layernorm": layernorm_config_id == "s2-layernorm-on",
        "whitening": whitening_config_id == "s2-whitening-on",
        "preprojectors": preprojector_config_id == "s2-preproj-separate",
        "feature_adapter": adapter_kind != "identity",
    }
    active = tuple(name for name, enabled in factor_flags.items() if enabled)
    if stage == 0 and active:
        raise ValueError("Stage 0 cannot enable a Stage 2 factor")
    if len(active) > 1:
        raise ValueError("Stage 2 permits exactly zero or one active factor")
    if adapter_kind == "identity":
        if adapter_rank is not None or adapter_lr_ratio is not None:
            raise ValueError(
                "identity adapter cannot set adapter rank or LR ratio"
            )
    else:
        if adapter_rank not in ADAPTER_RANKS:
            raise ValueError(f"adapter_rank must be one of {ADAPTER_RANKS}")
        if adapter_lr_ratio not in LEARNING_RATE_RATIOS:
            raise ValueError(
                f"adapter_lr_ratio must be one of {LEARNING_RATE_RATIOS}"
            )
    if whitening_config_id == "s2-whitening-on":
        if not isinstance(whitening, TrainWhitening):
            raise ValueError(
                "s2-whitening-on requires a fitted TrainWhitening"
            )
        if whitening.source_scope != "train":
            raise ValueError("TrainWhitening must be fit from train scope")
        if tuple(whitening.mean.shape) != (5, 3_200):
            raise ValueError("TrainWhitening must target [5,3200] features")
    elif whitening is not None:
        raise ValueError("TrainWhitening is only accepted by s2-whitening-on")

    return _ModelBuildInputs(
        components=components,
        stage=stage,
        subject=subject,
        active_factor=active[0] if active else None,
        layernorm_config_id=layernorm_config_id,
        whitening_config_id=whitening_config_id,
        preprojector_config_id=preprojector_config_id,
        adapter_kind=adapter_kind,
        adapter_rank=adapter_rank,
        adapter_lr_ratio=adapter_lr_ratio,
        whitening=whitening,
    )


class SAMGARuntimeModel(nn.Module):
    """Verified SAMGA task plus exactly zero or one preregistered factor."""

    def __init__(
        self,
        spec: TrainingCellSpec | _ModelBuildInputs,
    ) -> None:
        super().__init__()
        if not isinstance(spec, (TrainingCellSpec, _ModelBuildInputs)):
            raise TypeError(
                "spec must be a TrainingCellSpec or validated model inputs"
            )
        self.stage = spec.stage
        self.active_factor = spec.active_factor
        self.subject_id = spec.subject
        self.layernorm_config_id = spec.layernorm_config_id
        self.whitening_config_id = spec.whitening_config_id
        self.preprojector_config_id = spec.preprojector_config_id
        self.adapter_lr_ratio = spec.adapter_lr_ratio
        self.base = SAMGATaskModel(components=spec.components)
        self.layernorm = LayerNormTransform(
            enabled=spec.layernorm_config_id == "s2-layernorm-on"
        )
        self.whitening: nn.Module = (
            spec.whitening
            if spec.whitening is not None
            else nn.Identity()
        )
        self.separate_image_pre_projectors: nn.ModuleList | None
        if spec.preprojector_config_id == "s2-preproj-separate":
            self.separate_image_pre_projectors = nn.ModuleList(
                copy.deepcopy(self.base.image_pre_projector)
                for _ in self.base.config.layer_ids
            )
        else:
            self.separate_image_pre_projectors = None

        target_parameters = (
            len(self.base.config.layer_ids)
            * (2 * self.base.config.image_dim * int(spec.adapter_rank) + 1)
            if spec.adapter_rank is not None
            else None
        )
        if spec.adapter_kind == "identity":
            self.feature_adapter: nn.Module = nn.Identity()
        elif spec.adapter_kind == "adapter":
            self.feature_adapter = ResidualFeatureAdapter(
                hidden_size=self.base.config.image_dim,
                rank=int(spec.adapter_rank),
                layers=len(self.base.config.layer_ids),
            )
        elif spec.adapter_kind == "global_dense":
            self.feature_adapter = DenseBottleneckControl(
                hidden_size=self.base.config.image_dim,
                target_parameters=int(target_parameters),
                layers=len(self.base.config.layer_ids),
            )
        elif spec.adapter_kind == "matched_projector":
            self.feature_adapter = MatchedPerLayerProjectorControl(
                input_dim=self.base.config.image_dim,
                output_dim=self.base.config.feature_dim,
                layers=len(self.base.config.layer_ids),
                target_parameters=int(target_parameters),
            )
        else:  # guarded by TrainingCellSpec
            raise AssertionError("unreachable adapter kind")
        self.adapter_kind = spec.adapter_kind

    def encode_eeg(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.base.encode_eeg(eeg)

    def encode_image(
        self,
        layer_features: torch.Tensor,
        subject_ids: torch.Tensor,
        *,
        force_global: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = (
            len(self.base.config.layer_ids),
            self.base.config.image_dim,
        )
        if (
            layer_features.ndim != 3
            or tuple(layer_features.shape[1:]) != expected
        ):
            raise ValueError(
                "image features must have shape "
                f"[B,{expected[0]},{expected[1]}]"
            )
        self.base._validate_subject_ids(
            subject_ids,
            layer_features.shape[0],
        )
        if not self.training:
            if force_global is False:
                raise ValueError("SAMGA evaluation is locked to the global router")
            effective_force_global = True
        else:
            effective_force_global = bool(force_global)

        transformed = self.layernorm(layer_features)
        transformed = self.whitening(transformed)
        if self.adapter_kind in ("adapter", "global_dense"):
            transformed = self.feature_adapter(transformed)
        if self.separate_image_pre_projectors is None:
            hidden = self.base.image_pre_projector(transformed)
        else:
            hidden = torch.stack(
                [
                    projector(transformed[:, index, :])
                    for index, projector in enumerate(
                        self.separate_image_pre_projectors
                    )
                ],
                dim=1,
            )
        projected = torch.stack(
            [
                projector(hidden[:, index, :])
                for index, projector in enumerate(
                    self.base.image_projectors
                )
            ],
            dim=1,
        )
        if self.adapter_kind == "matched_projector":
            projected = self.feature_adapter(transformed, projected)
        weights = self.base.router(
            subject_ids,
            force_global=effective_force_global,
        )
        mixed = torch.sum(projected * weights.unsqueeze(-1), dim=1)
        return self.base.shared_encoder(mixed), weights

    def forward(
        self,
        eeg: torch.Tensor,
        layer_features: torch.Tensor,
        subject_ids: torch.Tensor,
        *,
        force_global: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eeg_features = self.encode_eeg(eeg)
        image_features, weights = self.encode_image(
            layer_features,
            subject_ids,
            force_global=force_global,
        )
        return eeg_features, image_features, weights

    def loss(
        self,
        eeg_features: torch.Tensor,
        image_features: torch.Tensor,
        *,
        mmd_weight: float,
    ) -> SAMGALossOutput:
        return self.base.loss(
            eeg_features,
            image_features,
            mmd_weight=mmd_weight,
        )

    def freeze_shared_encoder(self) -> None:
        self.base.freeze_shared_encoder()

    def optimizer_parameter_groups(
        self,
        *,
        include_shared: bool,
        base_learning_rate: float,
    ) -> list[dict[str, object]]:
        modules: list[nn.Module] = [
            self.base.eeg_encoder,
            self.base.eeg_projector,
        ]
        if self.separate_image_pre_projectors is None:
            modules.append(self.base.image_pre_projector)
        else:
            modules.append(self.separate_image_pre_projectors)
        modules.append(self.base.text_projector)
        if include_shared:
            modules.append(self.base.shared_encoder)
        modules.extend((self.base.image_projectors, self.base.router))
        base_parameters = [
            parameter
            for module in modules
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        base_parameters.extend(
            parameter
            for parameter in self.base.criterion.parameters()
            if parameter.requires_grad
        )
        factor_parameters = (
            [
                parameter
                for parameter in self.feature_adapter.parameters()
                if parameter.requires_grad
            ]
            if self.adapter_kind != "identity"
            else []
        )
        all_parameters = base_parameters + factor_parameters
        if len({id(parameter) for parameter in all_parameters}) != len(
            all_parameters
        ):
            raise AssertionError("optimizer parameters contain a duplicate")
        groups: list[dict[str, object]] = [
            {
                "params": base_parameters,
                "lr": base_learning_rate,
            }
        ]
        if factor_parameters:
            if self.adapter_lr_ratio is None:
                raise AssertionError("adapter LR ratio was not bound")
            groups.append(
                {
                    "params": factor_parameters,
                    "lr": base_learning_rate * self.adapter_lr_ratio,
                }
            )
        return groups


def _derive_identities_from_model(
    model: SAMGARuntimeModel,
    *,
    train_row_indices: Sequence[int],
    seed: int,
    batch_size: int,
) -> TrainingIdentities:
    data_order = derive_data_order_sha256(
        train_row_indices,
        seed=seed,
        batch_size=batch_size,
    )
    state = {
        key: value.detach().cpu().contiguous()
        for key, value in model.state_dict().items()
    }
    if model.active_factor == "whitening":
        candidate_prefixes = ("whitening.",)
    elif model.active_factor == "preprojectors":
        candidate_prefixes = ("separate_image_pre_projectors.",)
    elif model.active_factor == "feature_adapter":
        candidate_prefixes = ("feature_adapter.",)
    else:
        candidate_prefixes = ()
    candidate_state = {
        key: value
        for key, value in state.items()
        if key.startswith(candidate_prefixes)
    }
    shared_state = {
        key: value
        for key, value in state.items()
        if key not in candidate_state
    }
    full_hash = hash_state_dict(state)
    shared_hash = hash_state_dict(shared_state)
    candidate_state_hash = (
        hash_state_dict(candidate_state)
        if candidate_state
        else sha256_json({"state_dict": {}})
    )
    whitening_payload_sha256 = (
        model.whitening.payload_sha256
        if isinstance(model.whitening, TrainWhitening)
        else None
    )
    specific_hash = sha256_json(
        {
            "schema_version": 1,
            "payload_type": (
                "samga_brain_rw.architecture_specific_initialization"
            ),
            "active_factor": model.active_factor,
            "layernorm_config_id": model.layernorm_config_id,
            "whitening_config_id": model.whitening_config_id,
            "whitening_payload_sha256": whitening_payload_sha256,
            "preprojector_config_id": model.preprojector_config_id,
            "adapter_kind": model.adapter_kind,
            "adapter_lr_ratio": model.adapter_lr_ratio,
            "state_keys": sorted(candidate_state),
            "state_sha256": candidate_state_hash,
        }
    )
    trajectory = sha256_json(
        {
            "schema_version": 1,
            "payload_type": "samga_brain_rw.training_trajectory",
            "stage": model.stage,
            "subject": model.subject_id,
            "seed": seed,
            "schedule_sha256": SCHEDULE_SHA256,
            "data_order_sha256": data_order,
            "effective_batch": batch_size,
            "drop_last": True,
            "full_task_initialization_sha256": full_hash,
            "shared_parameter_intersection_name": (
                SHARED_PARAMETER_INTERSECTION_NAME
            ),
            "shared_parameter_intersection_sha256": shared_hash,
            "architecture_specific_initialization_sha256": specific_hash,
        }
    )
    return TrainingIdentities(
        data_order_sha256=data_order,
        trajectory_sha256=trajectory,
        full_task_initialization_sha256=full_hash,
        shared_parameter_intersection_name=(
            SHARED_PARAMETER_INTERSECTION_NAME
        ),
        shared_parameter_intersection_sha256=shared_hash,
        architecture_specific_initialization_sha256=specific_hash,
    )


def derive_training_identities(
    *,
    components: UpstreamComponents,
    train_row_indices: Sequence[int],
    stage: int,
    subject: int,
    seed: int,
    batch_size: int,
    layernorm_config_id: str = "s2-layernorm-off",
    whitening_config_id: str = "s2-whitening-off",
    preprojector_config_id: str = "s2-preproj-shared",
    adapter_kind: str = "identity",
    adapter_rank: int | None = None,
    adapter_lr_ratio: float | None = None,
    whitening: TrainWhitening | None = None,
) -> TrainingIdentities:
    """Derive CLI-ready identities without trusting identity declarations."""

    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    inputs = _validated_model_build_inputs(
        components=components,
        stage=stage,
        subject=subject,
        layernorm_config_id=layernorm_config_id,
        whitening_config_id=whitening_config_id,
        preprojector_config_id=preprojector_config_id,
        adapter_kind=adapter_kind,
        adapter_rank=adapter_rank,
        adapter_lr_ratio=adapter_lr_ratio,
        whitening=whitening,
    )
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    numpy_state = (
        numpy_state[0],
        numpy_state[1].copy(),
        numpy_state[2],
        numpy_state[3],
        numpy_state[4],
    )
    torch_state = torch.get_rng_state().clone()
    cuda_states = (
        torch.cuda.get_rng_state_all()
        if torch.cuda.is_initialized()
        else None
    )
    try:
        _seed_fresh_process(seed, torch.device("cpu"))
        model = SAMGARuntimeModel(inputs)
        return _derive_identities_from_model(
            model,
            train_row_indices=train_row_indices,
            seed=seed,
            batch_size=batch_size,
        )
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _validate_runtime_identities(
    spec: TrainingCellSpec,
    model: SAMGARuntimeModel,
    train_dataset: Dataset[dict[str, object]],
) -> TrainingIdentities:
    rows = getattr(train_dataset, "row_indices", None)
    if not isinstance(rows, Sequence) or isinstance(
        rows,
        (str, bytes, bytearray),
    ):
        raise ValueError("train dataset must expose ordered row_indices")
    identities = _derive_identities_from_model(
        model,
        train_row_indices=rows,
        seed=spec.seed,
        batch_size=spec.batch_size,
    )
    if spec.data_order_sha256 != identities.data_order_sha256:
        raise ValueError("runtime-derived data-order identity mismatch")
    if spec.trajectory_sha256 != identities.trajectory_sha256:
        raise ValueError("runtime-derived trajectory identity mismatch")
    expected_candidate = {
        "full_task_initialization_sha256": (
            identities.full_task_initialization_sha256
        ),
        "shared_parameter_intersection_name": (
            identities.shared_parameter_intersection_name
        ),
        "shared_parameter_intersection_sha256": (
            identities.shared_parameter_intersection_sha256
        ),
        "architecture_specific_initialization_sha256": (
            identities.architecture_specific_initialization_sha256
        ),
    }
    for name, expected in expected_candidate.items():
        if spec.candidate_spec.get(name) != expected:
            label = name.removesuffix("_sha256").replace("_", "-")
            raise ValueError(
                f"candidate_spec {label} identity mismatch"
            )
    return identities


def _validate_whitening_provenance(
    spec: TrainingCellSpec,
    train_dataset: Dataset[dict[str, object]],
) -> None:
    if spec.whitening_config_id != "s2-whitening-on":
        return
    whitening = spec.whitening
    if not isinstance(whitening, TrainWhitening):
        raise AssertionError("whitening-on spec lost its TrainWhitening")
    whitening.to_payload()
    rows = getattr(train_dataset, "row_indices", None)
    if (
        not isinstance(rows, Sequence)
        or isinstance(rows, (str, bytes, bytearray))
        or any(type(row) is not int for row in rows)
    ):
        raise ValueError("train dataset must expose an integer row-list")
    if tuple(rows) != tuple(whitening.canonical_train_rows):
        raise ValueError(
            "TrainWhitening train row-list differs from the current dataset"
        )
    manifest_sha256 = spec.input_hashes.get("manifest_sha256")
    cache_sha256 = spec.input_hashes.get("cache_sha256")
    _require_sha256(
        manifest_sha256,
        "input_hashes.manifest_sha256",
    )
    _require_sha256(
        cache_sha256,
        "input_hashes.cache_sha256",
    )
    if whitening.input_provenance_sha256 != manifest_sha256:
        raise ValueError(
            "TrainWhitening manifest provenance differs from current input"
        )
    if whitening.cache_provenance_sha256 != cache_sha256:
        raise ValueError(
            "TrainWhitening cache provenance differs from current input"
        )


def run_training_cell(spec: TrainingCellSpec) -> TrainingResult:
    """Run, resume, or smoke-test one locked SAMGA cell in memory."""

    if not isinstance(spec, TrainingCellSpec):
        raise TypeError("spec must be a TrainingCellSpec")
    _validate_resume_environment_before_data(spec)
    device = _resolve_device(spec.device)
    _seed_fresh_process(spec.seed, device)
    train_dataset = _build_dataset(spec, "train", 0.3)
    validation_dataset = _build_dataset(spec, "val-dev", 0.0)
    if len(train_dataset) <= 0 or len(validation_dataset) <= 0:
        raise ValueError("train and val-dev datasets must be non-empty")

    _validate_whitening_provenance(spec, train_dataset)
    model = SAMGARuntimeModel(spec).to(device)
    _validate_runtime_identities(spec, model, train_dataset)
    loader_generator = torch.Generator(device="cpu")
    loader_generator.manual_seed(_loader_seed(spec.seed))
    sampler = StatefulEpochSampler(
        dataset_size=len(train_dataset),
        seed=spec.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=spec.batch_size,
        sampler=sampler,
        num_workers=0,
        generator=loader_generator,
        drop_last=True,
    )

    resume = (
        _validated_resume_checkpoint(spec, dataset_size=len(train_dataset))
        if spec.resume_checkpoint is not None
        else None
    )
    checkpoint_epoch = int(resume["epoch"]) if resume is not None else 0
    checkpoint_stage = (
        str(resume["optimizer_stage"])
        if resume is not None
        else "stage1"
    )
    if checkpoint_stage == "stage2":
        model.freeze_shared_encoder()
    optimizer, scheduler = _build_optimizer(
        model,
        checkpoint_stage,
    )
    if resume is not None:
        _validate_optimizer_state_against_recipe(
            resume["optimizer_state_dict"],
            optimizer.state_dict(),
        )
    global_step = 0
    start_epoch = 1
    resume_mid_epoch = False
    iterator_generator_state: torch.Tensor | None = None
    prior_snapshot_epochs: list[int] = []

    if resume is not None:
        restored_epoch, global_step = spec.checkpoint_restorer(
            resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            generator=loader_generator,
            expected_subject=spec.subject,
            expected_seed=spec.seed,
            expected_config_sha256=spec.config_sha256,
            expected_schedule_sha256=spec.schedule_sha256,
            expected_trajectory_sha256=spec.trajectory_sha256,
            expected_data_order_sha256=spec.data_order_sha256,
        )
        if (restored_epoch, global_step) != (
            checkpoint_epoch,
            int(resume["global_step"]),
        ):
            raise ValueError("checkpoint restorer returned inconsistent position")
        _verify_restored_checkpoint(
            resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            generator=loader_generator,
        )
        sampler.load_state_dict(
            _mapping(
                resume["sampler_state_dict"],
                "checkpoint sampler_state_dict",
            )
        )
        runtime_state = _mapping(
            resume["runtime_state"],
            "checkpoint runtime_state",
        )
        epoch_complete = runtime_state["epoch_complete"]
        if type(epoch_complete) is not bool:
            raise ValueError("checkpoint epoch_complete must be boolean")
        prior_snapshot_epochs = _integer_list(
            runtime_state["snapshot_epochs"],
            "checkpoint snapshot_epochs",
        )
        if epoch_complete:
            start_epoch = checkpoint_epoch + 1
        else:
            start_epoch = checkpoint_epoch
            resume_mid_epoch = True
            iterator_generator_state = _cpu_generator_state(
                runtime_state["iterator_generator_state"],
                "checkpoint iterator_generator_state",
            )

    snapshots: list[SnapshotRecord] = []
    trained_rows: list[tuple[int, ...]] = []
    rebuild_epochs: list[int] = []
    invocation_steps = 0
    final_checkpoint = dict(resume) if resume is not None else {}
    final_validation: ValidationResult | None = None

    if start_epoch > TOTAL_EPOCHS:
        final_validation = _validate_global(
            model,
            validation_dataset,
            batch_size=spec.batch_size,
            device=device,
            seed=spec.seed,
        )
        return TrainingResult(
            model=model,
            global_step=global_step,
            completed=True,
            final_validation=final_validation,
            final_checkpoint=final_checkpoint,
            sampler_state=sampler.state_dict(),
            loader_generator_state=loader_generator.get_state().clone(),
            snapshots=(),
            trained_row_indices=(),
            optimizer_rebuild_epochs=(),
        )

    for epoch in range(start_epoch, TOTAL_EPOCHS + 1):
        if epoch == STAGE1_EPOCHS + 1 and checkpoint_stage == "stage1":
            model.freeze_shared_encoder()
            optimizer, scheduler = _build_optimizer(model, "stage2")
            checkpoint_stage = "stage2"
            rebuild_epochs.append(epoch)
        expected_stage = "stage1" if epoch <= STAGE1_EPOCHS else "stage2"
        if checkpoint_stage != expected_stage:
            raise ValueError("optimizer stage does not match training epoch")

        if not (resume_mid_epoch and epoch == start_epoch):
            sampler.set_epoch(epoch)
        if resume_mid_epoch and epoch == start_epoch:
            if iterator_generator_state is None:
                raise AssertionError("resume iterator state was not loaded")
            loader_generator.set_state(iterator_generator_state)
        iterator_start_state = loader_generator.get_state().clone()

        model.train()
        reached_limit = False
        for batch in train_loader:
            eeg, images, subjects, rows = _training_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            eeg_features, image_features, _ = model(
                eeg,
                images,
                subjects,
                force_global=False,
            )
            loss = model.loss(
                eeg_features,
                image_features,
                mmd_weight=_mmd_weight_for_epoch(epoch),
            )
            if not bool(torch.isfinite(loss.total).item()):
                raise ValueError("training loss is non-finite")
            loss.total.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1
            invocation_steps += 1
            trained_rows.append(rows)
            if (
                spec.max_train_steps is not None
                and invocation_steps >= spec.max_train_steps
            ):
                reached_limit = True
                break

        epoch_complete = sampler.exhausted
        if not epoch_complete and not reached_limit:
            raise AssertionError("DataLoader stopped before sampler exhaustion")
        final_validation = _validate_global(
            model,
            validation_dataset,
            batch_size=spec.batch_size,
            device=device,
            seed=spec.seed,
        )
        retain = epoch_complete and epoch in LAST10_EPOCHS
        if epoch_complete and epoch not in prior_snapshot_epochs:
            prior_snapshot_epochs.append(epoch)
        final_checkpoint = _build_checkpoint(
            spec,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            generator=loader_generator,
            sampler=sampler,
            validation=final_validation,
            epoch=epoch,
            global_step=global_step,
            epoch_complete=epoch_complete,
            iterator_generator_state=iterator_start_state,
            snapshot_epochs=prior_snapshot_epochs,
            optimizer_stage=checkpoint_stage,
            retain_for_averaging=retain,
        )
        publication = _publish_checkpoint(
            spec.checkpoint_sink,
            final_checkpoint,
            retain_for_averaging=retain,
        )
        snapshots.append(
            SnapshotRecord(
                epoch=epoch,
                optimizer_stage=checkpoint_stage,
                epoch_complete=epoch_complete,
                retain_for_averaging=retain,
                publication=publication,
            )
        )
        resume_mid_epoch = False
        if reached_limit:
            break

    if final_validation is None or not final_checkpoint:
        raise AssertionError("training produced no checkpoint")
    completed = (
        int(final_checkpoint["epoch"]) == TOTAL_EPOCHS
        and bool(
            _mapping(
                final_checkpoint["runtime_state"],
                "runtime_state",
            )["epoch_complete"]
        )
    )
    return TrainingResult(
        model=model,
        global_step=global_step,
        completed=completed,
        final_validation=final_validation,
        final_checkpoint=final_checkpoint,
        sampler_state=sampler.state_dict(),
        loader_generator_state=loader_generator.get_state().clone(),
        snapshots=tuple(snapshots),
        trained_row_indices=tuple(trained_rows),
        optimizer_rebuild_epochs=tuple(rebuild_epochs),
    )


def _require_sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _validate_string_mapping(value: object, context: str) -> None:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise TypeError(f"{context} must be a string-keyed mapping")


def _validate_hash_mapping(value: object, context: str) -> None:
    _validate_string_mapping(value, context)
    for key, digest in value.items():
        _require_sha256(digest, f"{context}.{key}")


def _validate_resume_environment_before_data(
    spec: TrainingCellSpec,
) -> None:
    if spec.resume_checkpoint is None:
        return
    resume = _mapping(
        spec.resume_checkpoint,
        "resume checkpoint",
    )
    if "environment" not in resume:
        raise ValueError(
            "resume checkpoint environment mismatch: field is missing"
        )
    resumed_environment = validate_environment_binding(
        resume["environment"]
    )
    if resumed_environment != dict(spec.environment):
        raise ValueError(
            "resume checkpoint environment mismatch"
        )


def _resolve_device(value: str | torch.device) -> torch.device:
    if isinstance(value, str) and value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(value)
    except (RuntimeError, TypeError) as exc:
        raise ValueError("device is invalid") from exc
    if device.type not in ("cpu", "cuda"):
        raise ValueError("device must be CPU or CUDA")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return device


def _seed_fresh_process(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _loader_seed(seed: int) -> int:
    return (seed + 0xBB67AE8584CAA73B) % (2**63 - 1)


def _build_dataset(
    spec: TrainingCellSpec,
    scope: str,
    smooth_probability: float,
) -> Dataset[dict[str, object]]:
    dataset = spec.dataset_factory(
        manifest_path=spec.manifest_path,
        scope=scope,
        seed=spec.seed,
        selected_channels=POSTERIOR_CHANNELS,
        feature_cache=spec.feature_cache,
        smooth_probability=smooth_probability,
    )
    if not isinstance(dataset, Dataset):
        raise TypeError("dataset_factory must return a torch Dataset")
    if getattr(dataset, "scope", None) != scope:
        raise ValueError("dataset_factory result scope mismatch")
    if getattr(dataset, "subject_id", None) != spec.subject:
        raise ValueError(
            "dataset_factory result subject differs from TrainingCellSpec"
        )
    return dataset


def _build_optimizer(
    model: SAMGARuntimeModel,
    optimizer_stage: str,
) -> tuple[torch.optim.AdamW, torch.optim.lr_scheduler.LambdaLR]:
    if optimizer_stage == "stage1":
        include_shared = True
        learning_rate = float(SCHEDULE["stage1_learning_rate"])
    elif optimizer_stage == "stage2":
        include_shared = False
        learning_rate = float(SCHEDULE["stage2_learning_rate"])
    else:
        raise ValueError("optimizer stage must be stage1 or stage2")
    groups = model.optimizer_parameter_groups(
        include_shared=include_shared,
        base_learning_rate=learning_rate,
    )
    betas = SCHEDULE["betas"]
    if not isinstance(betas, list) or len(betas) != 2:
        raise AssertionError("locked AdamW betas are invalid")
    optimizer = torch.optim.AdamW(
        groups,
        lr=learning_rate,
        betas=(float(betas[0]), float(betas[1])),
        eps=float(SCHEDULE["eps"]),
        weight_decay=float(SCHEDULE["weight_decay"]),
        amsgrad=bool(SCHEDULE["amsgrad"]),
        maximize=bool(SCHEDULE["maximize"]),
        foreach=SCHEDULE["foreach"],
        capturable=bool(SCHEDULE["capturable"]),
        differentiable=bool(SCHEDULE["differentiable"]),
        fused=SCHEDULE["fused"],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[lambda _: 1.0 for _ in groups],
    )
    return optimizer, scheduler


def _validate_optimizer_state_against_recipe(
    checkpoint_state_value: object,
    fresh_state_value: object,
) -> None:
    checkpoint_state = _mapping(
        checkpoint_state_value,
        "checkpoint optimizer state",
    )
    fresh_state = _mapping(
        fresh_state_value,
        "fresh optimizer recipe",
    )
    checkpoint_groups = checkpoint_state.get("param_groups")
    fresh_groups = fresh_state.get("param_groups")
    if (
        not isinstance(checkpoint_groups, list)
        or not isinstance(fresh_groups, list)
        or len(checkpoint_groups) != len(fresh_groups)
    ):
        raise ValueError(
            "checkpoint optimizer recipe param-group count mismatch"
        )
    for index, (checkpoint_value, fresh_value) in enumerate(
        zip(checkpoint_groups, fresh_groups, strict=True)
    ):
        checkpoint_group = _mapping(
            checkpoint_value,
            f"checkpoint optimizer recipe group {index}",
        )
        fresh_group = _mapping(
            fresh_value,
            f"fresh optimizer recipe group {index}",
        )
        if set(checkpoint_group) != set(fresh_group):
            raise ValueError(
                f"checkpoint optimizer recipe group {index} schema mismatch"
            )
        for key in fresh_group:
            if _semantic_sha256(
                checkpoint_group[key],
                f"checkpoint optimizer recipe group {index}.{key}",
            ) != _semantic_sha256(
                fresh_group[key],
                f"fresh optimizer recipe group {index}.{key}",
            ):
                raise ValueError(
                    "checkpoint optimizer recipe mismatch at "
                    f"group {index}.{key}"
                )


def _mmd_weight_for_epoch(epoch: int) -> float:
    if epoch > STAGE1_EPOCHS:
        return 0.0
    return 0.9 + (0.5 - 0.9) * ((epoch - 1) / (STAGE1_EPOCHS - 1))


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise ValueError(f"{context} must be a string-keyed mapping")
    return value


def _integer_list(value: object, context: str) -> list[int]:
    if (
        not isinstance(value, list)
        or any(type(item) is not int for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{context} must be a unique integer list")
    return list(value)


def _cpu_generator_state(value: object, context: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.uint8
        or value.ndim != 1
        or value.numel() == 0
    ):
        raise ValueError(f"{context} must be a CPU byte tensor")
    return value.detach().clone()


def _exact_keys(
    value: Mapping[object, object],
    expected: set[str],
    context: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(str(key) for key in actual if key not in expected)
        raise ValueError(
            f"{context} schema mismatch; missing={missing}, unknown={unknown}"
        )


def _semantic_value(value: object, context: str) -> object:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{context} contains a non-finite float")
        return value
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        if tensor.layout != torch.strided:
            raise ValueError(f"{context} tensor must use strided layout")
        if (
            torch.is_floating_point(tensor)
            or torch.is_complex(tensor)
        ) and not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"{context} tensor is non-finite")
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        return {
            "__tensor__": {
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "data_sha256": hashlib.sha256(raw).hexdigest(),
            }
        }
    if isinstance(value, Mapping):
        items = [
            (
                _semantic_value(key, f"{context}.key"),
                _semantic_value(item, f"{context}[{key!r}]"),
            )
            for key, item in value.items()
        ]
        items.sort(key=lambda item: repr(item[0]))
        return {"__mapping__": items}
    if isinstance(value, tuple):
        return {
            "__tuple__": [
                _semantic_value(item, f"{context}[{index}]")
                for index, item in enumerate(value)
            ]
        }
    if isinstance(value, list):
        return {
            "__list__": [
                _semantic_value(item, f"{context}[{index}]")
                for index, item in enumerate(value)
            ]
        }
    raise ValueError(f"{context} contains unsupported state type")


def _semantic_sha256(value: object, context: str) -> str:
    return sha256_json(_semantic_value(value, context))


def _numpy_rng_payload() -> dict[str, object]:
    name, keys, position, has_gauss, cached = np.random.get_state()
    return {
        "bit_generator": name,
        "keys": torch.from_numpy(keys.copy()),
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached),
    }


def _validate_rng_schema(payload: Mapping[str, object]) -> None:
    python_state = payload["python_rng_state"]
    if (
        not isinstance(python_state, tuple)
        or len(python_state) != 3
        or type(python_state[0]) is not int
        or not isinstance(python_state[1], tuple)
        or not python_state[1]
        or any(type(item) is not int for item in python_state[1])
        or (
            python_state[2] is not None
            and (
                type(python_state[2]) is not float
                or not math.isfinite(python_state[2])
            )
        )
    ):
        raise ValueError("checkpoint Python RNG state is invalid")
    numpy_state = _mapping(
        payload["numpy_rng_state"],
        "checkpoint NumPy RNG state",
    )
    _exact_keys(
        numpy_state,
        {
            "bit_generator",
            "keys",
            "position",
            "has_gauss",
            "cached_gaussian",
        },
        "checkpoint NumPy RNG state",
    )
    keys = numpy_state["keys"]
    if (
        numpy_state["bit_generator"] != "MT19937"
        or not isinstance(keys, torch.Tensor)
        or keys.device.type != "cpu"
        or keys.dtype != torch.uint32
        or keys.ndim != 1
        or keys.numel() != 624
        or type(numpy_state["position"]) is not int
        or not 0 <= numpy_state["position"] <= 624
        or type(numpy_state["has_gauss"]) is not int
        or numpy_state["has_gauss"] not in (0, 1)
        or type(numpy_state["cached_gaussian"]) is not float
        or not math.isfinite(numpy_state["cached_gaussian"])
    ):
        raise ValueError("checkpoint NumPy RNG state is invalid")
    _cpu_generator_state(
        payload["torch_rng_state"],
        "checkpoint Torch RNG state",
    )
    _cpu_generator_state(
        payload["loader_generator_state"],
        "checkpoint loader generator state",
    )
    cuda_states = payload["cuda_rng_states"]
    if not isinstance(cuda_states, list):
        raise ValueError("checkpoint CUDA RNG states must be a list")
    for index, state in enumerate(cuda_states):
        _cpu_generator_state(
            state,
            f"checkpoint CUDA RNG state {index}",
        )


def _validate_checkpoint_schema(
    payload: Mapping[str, object],
    context: str,
) -> None:
    _exact_keys(payload, _CHECKPOINT_KEYS, context)
    if payload["schema_version"] != 1:
        raise ValueError(f"{context} schema_version must be 1")
    if payload["payload_type"] != CHECKPOINT_PAYLOAD_TYPE:
        raise ValueError(f"{context} payload_type mismatch")
    model_state = _mapping(
        payload["model_state_dict"],
        f"{context} model state",
    )
    if not model_state or any(
        not isinstance(key, str)
        or not key
        or not isinstance(value, torch.Tensor)
        for key, value in model_state.items()
    ):
        raise ValueError(f"{context} model state schema is invalid")
    claimed_model_hash = _require_sha256(
        payload["model_state_sha256"],
        f"{context} model_state_sha256",
    )
    if hash_state_dict(model_state) != claimed_model_hash:
        raise ValueError(f"{context} model state hash mismatch")
    optimizer_state = _mapping(
        payload["optimizer_state_dict"],
        f"{context} optimizer state",
    )
    _exact_keys(
        optimizer_state,
        {"state", "param_groups"},
        f"{context} optimizer state",
    )
    if (
        not isinstance(optimizer_state["state"], Mapping)
        or not isinstance(optimizer_state["param_groups"], list)
        or not optimizer_state["param_groups"]
    ):
        raise ValueError(f"{context} optimizer state schema is invalid")
    scheduler_state = _mapping(
        payload["scheduler_state_dict"],
        f"{context} scheduler state",
    )
    for name in ("base_lrs", "last_epoch", "_step_count", "_last_lr"):
        if name not in scheduler_state:
            raise ValueError(
                f"{context} scheduler state schema is missing {name}"
            )
    _semantic_sha256(optimizer_state, f"{context} optimizer state")
    _semantic_sha256(scheduler_state, f"{context} scheduler state")
    _validate_rng_schema(payload)
    _mapping(payload["sampler_state_dict"], f"{context} sampler state")
    validation = _mapping(
        payload["validation_metrics"],
        f"{context} validation metrics",
    )
    _exact_keys(
        validation,
        {
            "query_count",
            "gallery_count",
            "top1_count",
            "top5_count",
            "top1_rate",
            "top5_rate",
            "router_mode",
            "similarity",
        },
        f"{context} validation metrics",
    )
    _semantic_sha256(validation, f"{context} validation metrics")
    _validate_hash_mapping(payload["input_hashes"], f"{context} input_hashes")
    _mapping(payload["run_manifest"], f"{context} run_manifest")
    _mapping(payload["candidate_spec"], f"{context} candidate_spec")
    validate_environment_binding(payload["environment"])
    runtime = _mapping(payload["runtime_state"], f"{context} runtime state")
    _exact_keys(runtime, _RUNTIME_STATE_KEYS, f"{context} runtime state")
    retention = _mapping(payload["retention"], f"{context} retention")
    _exact_keys(retention, _RETENTION_KEYS, f"{context} retention")


def _validate_checkpoint_position(
    payload: Mapping[str, object],
    *,
    spec: TrainingCellSpec,
    dataset_size: int,
    context: str,
) -> None:
    _validate_checkpoint_schema(payload, context)
    epoch = payload["epoch"]
    global_step = payload["global_step"]
    if (
        type(epoch) is not int
        or not 1 <= epoch <= TOTAL_EPOCHS
        or type(global_step) is not int
        or global_step < 0
    ):
        raise ValueError(f"{context} epoch/global step is invalid")
    expected_stage = "stage1" if epoch <= STAGE1_EPOCHS else "stage2"
    if payload["optimizer_stage"] != expected_stage:
        raise ValueError(f"{context} optimizer stage mismatch")
    expected_values: tuple[tuple[str, object], ...] = (
        ("subject", spec.subject),
        ("seed", spec.seed),
        ("config_sha256", spec.config_sha256),
        ("schedule_sha256", spec.schedule_sha256),
        ("trajectory_sha256", spec.trajectory_sha256),
        ("data_order_sha256", spec.data_order_sha256),
        ("effective_batch", spec.batch_size),
        ("input_hashes", dict(spec.input_hashes)),
        ("environment", dict(spec.environment)),
        ("run_manifest", dict(spec.run_manifest)),
        ("candidate_spec", dict(spec.candidate_spec)),
    )
    for name, expected in expected_values:
        if payload[name] != expected:
            raise ValueError(f"{context} {name} mismatch")
    runtime = _mapping(payload["runtime_state"], f"{context} runtime state")
    if runtime["schema_version"] != 1:
        raise ValueError(f"{context} runtime state schema mismatch")
    epoch_complete = runtime["epoch_complete"]
    if type(epoch_complete) is not bool:
        raise ValueError(f"{context} epoch_complete must be boolean")
    if runtime["next_epoch"] != epoch + int(epoch_complete):
        raise ValueError(f"{context} next_epoch mismatch")
    resume_source = runtime["resume_source_checkpoint_sha256"]
    if resume_source is not None:
        _require_sha256(
            resume_source,
            f"{context} resume source checkpoint",
        )
    expected_base_lr = float(
        SCHEDULE[
            "stage1_learning_rate"
            if expected_stage == "stage1"
            else "stage2_learning_rate"
        ]
    )
    if runtime["optimizer_base_lr"] != expected_base_lr:
        raise ValueError(f"{context} optimizer base LR mismatch")
    _cpu_generator_state(
        runtime["iterator_generator_state"],
        f"{context} iterator generator state",
    )
    snapshots = _integer_list(
        runtime["snapshot_epochs"],
        f"{context} snapshot epochs",
    )
    expected_snapshots = list(range(1, epoch + int(epoch_complete)))
    if snapshots != expected_snapshots:
        raise ValueError(f"{context} snapshot prefix mismatch")
    if runtime["required_retained_epochs"] != list(LAST10_EPOCHS):
        raise ValueError(f"{context} retention window mismatch")
    sampler_state = _mapping(
        payload["sampler_state_dict"],
        f"{context} sampler state",
    )
    validator = StatefulEpochSampler(
        dataset_size=dataset_size,
        seed=spec.seed,
    )
    validator.load_state_dict(sampler_state)
    if validator.epoch != epoch:
        raise ValueError(f"{context} sampler epoch mismatch")
    if epoch_complete:
        if validator.position != dataset_size:
            raise ValueError(
                f"{context} epoch_complete disagrees with sampler position"
            )
    elif validator.position >= dataset_size:
        raise ValueError(
            f"{context} incomplete epoch disagrees with sampler position"
        )
    if not epoch_complete and validator.position % spec.batch_size != 0:
        raise ValueError(f"{context} sampler position is not a batch boundary")
    steps_per_epoch = dataset_size // spec.batch_size
    expected_global_step = (
        (epoch - 1) * steps_per_epoch
        + min(validator.position // spec.batch_size, steps_per_epoch)
    )
    if global_step != expected_global_step:
        raise ValueError(
            f"{context} global step disagrees with sampler position"
        )
    stage_steps = (
        global_step
        if expected_stage == "stage1"
        else global_step - STAGE1_EPOCHS * steps_per_epoch
    )
    scheduler_state = _mapping(
        payload["scheduler_state_dict"],
        f"{context} scheduler state",
    )
    if (
        scheduler_state["last_epoch"] != stage_steps
        or scheduler_state["_step_count"] != stage_steps + 1
    ):
        raise ValueError(f"{context} scheduler step mismatch")
    optimizer_state = _mapping(
        payload["optimizer_state_dict"],
        f"{context} optimizer state",
    )
    groups = optimizer_state["param_groups"]
    if (
        not isinstance(groups, list)
        or not groups
        or not isinstance(groups[0], Mapping)
        or groups[0].get("lr") != expected_base_lr
    ):
        raise ValueError(f"{context} optimizer stage/LR mismatch")
    base_lrs = scheduler_state["base_lrs"]
    last_lrs = scheduler_state["_last_lr"]
    group_lrs = [group.get("lr") for group in groups]
    if base_lrs != group_lrs or last_lrs != group_lrs:
        raise ValueError(f"{context} optimizer/scheduler LR mismatch")
    retention = _mapping(payload["retention"], f"{context} retention")
    if (
        retention["policy"] != "retain_exact_epochs_51_through_60"
        or retention["required_epochs"] != list(LAST10_EPOCHS)
        or type(retention["retain_for_averaging"]) is not bool
        or retention["retain_for_averaging"]
        != (epoch_complete and epoch in LAST10_EPOCHS)
    ):
        raise ValueError(f"{context} retention policy mismatch")


def _validated_resume_checkpoint(
    spec: TrainingCellSpec,
    *,
    dataset_size: int,
) -> dict[str, object]:
    payload = dict(
        _mapping(spec.resume_checkpoint, "resume_checkpoint")
    )
    _validate_checkpoint_position(
        payload,
        spec=spec,
        dataset_size=dataset_size,
        context="checkpoint",
    )
    return payload


def _verify_restored_checkpoint(
    payload: Mapping[str, object],
    *,
    model: SAMGARuntimeModel,
    optimizer: torch.optim.AdamW,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    generator: torch.Generator,
) -> None:
    claimed_model_hash = str(payload["model_state_sha256"])
    if hash_state_dict(model.state_dict()) != claimed_model_hash:
        raise ValueError("post-restore model state hash mismatch")
    if _semantic_sha256(
        optimizer.state_dict(),
        "post-restore optimizer",
    ) != _semantic_sha256(
        payload["optimizer_state_dict"],
        "checkpoint optimizer",
    ):
        raise ValueError("post-restore optimizer state mismatch")
    if _semantic_sha256(
        scheduler.state_dict(),
        "post-restore scheduler",
    ) != _semantic_sha256(
        payload["scheduler_state_dict"],
        "checkpoint scheduler",
    ):
        raise ValueError("post-restore scheduler state mismatch")
    expected_generator = _cpu_generator_state(
        payload["loader_generator_state"],
        "checkpoint loader generator state",
    )
    if not torch.equal(generator.get_state(), expected_generator):
        raise ValueError("post-restore generator state mismatch")
    if _semantic_sha256(
        random.getstate(),
        "post-restore Python RNG",
    ) != _semantic_sha256(
        payload["python_rng_state"],
        "checkpoint Python RNG",
    ):
        raise ValueError("post-restore RNG state mismatch")
    if _semantic_sha256(
        _numpy_rng_payload(),
        "post-restore NumPy RNG",
    ) != _semantic_sha256(
        payload["numpy_rng_state"],
        "checkpoint NumPy RNG",
    ):
        raise ValueError("post-restore RNG state mismatch")
    torch_state = _cpu_generator_state(
        payload["torch_rng_state"],
        "checkpoint Torch RNG state",
    )
    if not torch.equal(torch.get_rng_state(), torch_state):
        raise ValueError("post-restore RNG state mismatch")
    cuda_states = payload["cuda_rng_states"]
    if (
        torch.cuda.is_available()
        and isinstance(cuda_states, list)
        and cuda_states
        and _semantic_sha256(
            torch.cuda.get_rng_state_all(),
            "post-restore CUDA RNG",
        )
        != _semantic_sha256(cuda_states, "checkpoint CUDA RNG")
    ):
        raise ValueError("post-restore RNG state mismatch")


def _training_batch(
    batch: Mapping[str, object],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...]]:
    values = _mapping(batch, "training batch")
    eeg = values.get("eeg")
    images = values.get("layer_features")
    subjects = values.get("subject_id")
    rows = values.get("row_index")
    if not all(
        isinstance(value, torch.Tensor)
        for value in (eeg, images, subjects, rows)
    ):
        raise ValueError(
            "training batch requires tensor eeg/layer_features/subject_id/row_index"
        )
    row_values = tuple(
        int(value)
        for value in rows.detach().cpu().reshape(-1).tolist()
    )
    return (
        eeg.to(device=device, dtype=torch.float32, non_blocking=False),
        images.to(device=device, dtype=torch.float32, non_blocking=False),
        subjects.to(device=device, dtype=torch.long, non_blocking=False),
        row_values,
    )


def _validation_batch(
    batch: Mapping[str, object],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = _mapping(batch, "validation batch")
    eeg = values.get("eeg")
    images = values.get("layer_features")
    subjects = values.get("subject_id")
    if not all(
        isinstance(value, torch.Tensor)
        for value in (eeg, images, subjects)
    ):
        raise ValueError(
            "validation batch requires tensor eeg/layer_features/subject_id"
        )
    return (
        eeg.to(device=device, dtype=torch.float32, non_blocking=False),
        images.to(device=device, dtype=torch.float32, non_blocking=False),
        subjects.to(device=device, dtype=torch.long, non_blocking=False),
    )


def _validation_ids(
    dataset: Dataset[dict[str, object]],
    name: str,
) -> tuple[str, ...]:
    value = getattr(dataset, name, None)
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) != len(dataset)
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ValueError(
            f"val-dev dataset {name} must contain one non-empty ID per row"
        )
    return tuple(value)


def evaluate_development_model(
    model: SAMGARuntimeModel,
    dataset: Dataset[dict[str, object]],
    *,
    batch_size: int,
    device: str | torch.device,
    seed: int,
) -> ValidationResult:
    """Evaluate one val-dev dataset with global routing and cosine scores."""

    if not isinstance(model, SAMGARuntimeModel):
        raise TypeError("model must be a SAMGARuntimeModel")
    if not isinstance(dataset, Dataset):
        raise TypeError("dataset must be a torch Dataset")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if getattr(dataset, "scope", None) != "val-dev":
        raise PermissionError(
            "development evaluation requires exactly the val-dev scope"
        )
    resolved_device = _resolve_device(device)
    model.to(resolved_device)
    return _validate_global(
        model,
        dataset,
        batch_size=batch_size,
        device=resolved_device,
        seed=seed,
    )


def _validate_global(
    model: SAMGARuntimeModel,
    dataset: Dataset[dict[str, object]],
    *,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> ValidationResult:
    if getattr(dataset, "scope", None) != "val-dev":
        raise PermissionError(
            "development evaluation requires exactly the val-dev scope"
        )
    if getattr(dataset, "subject_id", None) != model.subject_id:
        raise ValueError("val-dev dataset subject differs from model subject")
    query_ids = _validation_ids(dataset, "query_ids")
    gallery_ids = _validation_ids(dataset, "gallery_ids")
    validation_generator = torch.Generator(device="cpu")
    validation_generator.manual_seed(
        (seed + 0x3C6EF372FE94F82B) % (2**63 - 1)
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        generator=validation_generator,
        drop_last=False,
    )
    eeg_parts: list[torch.Tensor] = []
    image_parts: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            eeg, images, subjects = _validation_batch(batch, device)
            eeg_features, image_features, _ = model(
                eeg,
                images,
                subjects,
                force_global=True,
            )
            eeg_parts.append(eeg_features.detach().cpu())
            image_parts.append(image_features.detach().cpu())
    if not eeg_parts:
        raise ValueError("val-dev dataset must be non-empty")
    eeg_features = F.normalize(torch.cat(eeg_parts, dim=0).float(), dim=1)
    image_features = F.normalize(
        torch.cat(image_parts, dim=0).float(),
        dim=1,
    )
    similarity = (
        eeg_features @ image_features.T
    ).numpy().astype(np.float32, copy=False)
    if not bool(np.isfinite(similarity).all()):
        raise ValueError("validation cosine scores are non-finite")
    metrics = independent_retrieval_metrics(
        similarity,
        query_ids,
        gallery_ids,
    )
    return ValidationResult(
        similarity=np.array(similarity, copy=True, order="C"),
        metrics=metrics,
    )


def _validate_built_checkpoint(
    payload: Mapping[str, object],
    *,
    spec: TrainingCellSpec,
    model: SAMGARuntimeModel,
    optimizer: torch.optim.AdamW,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    generator: torch.Generator,
    sampler: StatefulEpochSampler,
    validation_metrics: Mapping[str, object],
) -> None:
    _validate_checkpoint_position(
        payload,
        spec=spec,
        dataset_size=sampler.dataset_size,
        context="checkpoint builder result",
    )
    runtime = _mapping(
        payload["runtime_state"],
        "checkpoint builder runtime state",
    )
    if runtime["resume_source_checkpoint_sha256"] != (
        spec.resume_source_checkpoint_sha256
    ):
        raise ValueError("checkpoint builder resume lineage mismatch")
    if hash_state_dict(model.state_dict()) != payload["model_state_sha256"]:
        raise ValueError("checkpoint builder model state hash mismatch")
    comparisons = (
        (
            optimizer.state_dict(),
            payload["optimizer_state_dict"],
            "optimizer",
        ),
        (
            scheduler.state_dict(),
            payload["scheduler_state_dict"],
            "scheduler",
        ),
        (
            generator.get_state(),
            payload["loader_generator_state"],
            "loader generator",
        ),
        (
            sampler.state_dict(),
            payload["sampler_state_dict"],
            "sampler",
        ),
        (
            dict(validation_metrics),
            payload["validation_metrics"],
            "validation metrics",
        ),
        (
            random.getstate(),
            payload["python_rng_state"],
            "Python RNG",
        ),
        (
            _numpy_rng_payload(),
            payload["numpy_rng_state"],
            "NumPy RNG",
        ),
        (
            torch.get_rng_state(),
            payload["torch_rng_state"],
            "Torch RNG",
        ),
    )
    for actual, recorded, label in comparisons:
        if _semantic_sha256(
            actual,
            f"current {label}",
        ) != _semantic_sha256(
            recorded,
            f"checkpoint {label}",
        ):
            raise ValueError(f"checkpoint builder {label} mismatch")
    if torch.cuda.is_available() and _semantic_sha256(
        torch.cuda.get_rng_state_all(),
        "current CUDA RNG",
    ) != _semantic_sha256(
        payload["cuda_rng_states"],
        "checkpoint CUDA RNG",
    ):
        raise ValueError("checkpoint builder CUDA RNG mismatch")


def _build_checkpoint(
    spec: TrainingCellSpec,
    *,
    model: SAMGARuntimeModel,
    optimizer: torch.optim.AdamW,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    generator: torch.Generator,
    sampler: StatefulEpochSampler,
    validation: ValidationResult,
    epoch: int,
    global_step: int,
    epoch_complete: bool,
    iterator_generator_state: torch.Tensor,
    snapshot_epochs: Sequence[int],
    optimizer_stage: str,
    retain_for_averaging: bool,
) -> dict[str, object]:
    validation_metrics = {
        "query_count": validation.metrics.query_count,
        "gallery_count": validation.metrics.gallery_count,
        "top1_count": validation.metrics.top1_count,
        "top5_count": validation.metrics.top5_count,
        "top1_rate": validation.metrics.top1_rate,
        "top5_rate": validation.metrics.top5_rate,
        "router_mode": "global",
        "similarity": "cosine",
    }
    built = spec.checkpoint_builder(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        global_step=global_step,
        subject=spec.subject,
        seed=spec.seed,
        config_sha256=spec.config_sha256,
        schedule_sha256=spec.schedule_sha256,
        trajectory_sha256=spec.trajectory_sha256,
        data_order_sha256=spec.data_order_sha256,
        generator=generator,
        validation_metrics=validation_metrics,
        input_hashes=dict(spec.input_hashes),
        environment=dict(spec.environment),
        effective_batch=spec.batch_size,
        sampler_state=sampler.state_dict(),
        run_manifest=dict(spec.run_manifest),
        candidate_spec=dict(spec.candidate_spec),
    )
    payload = dict(_mapping(built, "checkpoint builder result"))
    if payload.get("optimizer_stage") != optimizer_stage:
        raise ValueError("checkpoint builder optimizer stage mismatch")
    payload["runtime_state"] = {
        "schema_version": 1,
        "epoch_complete": epoch_complete,
        "next_epoch": epoch + int(epoch_complete),
        "resume_source_checkpoint_sha256": (
            spec.resume_source_checkpoint_sha256
        ),
        "optimizer_base_lr": float(
            SCHEDULE[
                "stage1_learning_rate"
                if optimizer_stage == "stage1"
                else "stage2_learning_rate"
            ]
        ),
        "iterator_generator_state": (
            iterator_generator_state.detach().cpu().clone()
        ),
        "snapshot_epochs": list(snapshot_epochs),
        "required_retained_epochs": list(LAST10_EPOCHS),
    }
    payload["retention"] = {
        "policy": "retain_exact_epochs_51_through_60",
        "required_epochs": list(LAST10_EPOCHS),
        "retain_for_averaging": retain_for_averaging,
    }
    _validate_built_checkpoint(
        payload,
        spec=spec,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        generator=generator,
        sampler=sampler,
        validation_metrics=validation_metrics,
    )
    return payload


def _publish_checkpoint(
    sink: CheckpointSink | None,
    payload: dict[str, object],
    *,
    retain_for_averaging: bool,
) -> CheckpointPublication:
    receipt = sink(
        payload,
        retain_for_averaging=retain_for_averaging,
    )
    if not isinstance(receipt, CheckpointPublication) or not (
        receipt.exclusive_create
        and receipt.atomic_publish
        and receipt.verified
    ):
        raise ValueError(
            "checkpoint sink must attest exclusive, atomic, verified publication"
        )
    if retain_for_averaging and not receipt.durable_retention:
        raise ValueError(
            "checkpoint sink must attest durable retention for epochs 51-60"
        )
    return receipt
