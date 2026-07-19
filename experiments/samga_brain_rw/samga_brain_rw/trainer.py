"""Deterministic 60-epoch runtime for one sealed SAMGA training cell.

The runtime owns no output path.  Checkpoint publication is an injected
operation whose receipt must attest exclusive creation, atomic publication,
and post-publication verification.
"""

from __future__ import annotations

import copy
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
from .checkpoints import CHECKPOINT_PAYLOAD_TYPE
from .data import POSTERIOR_CHANNELS, ProtocolSubjectDataset
from .feature_transforms import LayerNormTransform, TrainWhitening
from .hashing import sha256_json
from .model import SAMGALossOutput, SAMGATaskModel
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
    "weight_decay": 1e-4,
    "scheduler": "constant_per_optimizer_stage",
}
SCHEDULE_SHA256 = sha256_json(SCHEDULE)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LAYERNORM_IDS = ("s2-layernorm-off", "s2-layernorm-on")
_WHITENING_IDS = ("s2-whitening-off", "s2-whitening-on")
_PREPROJECTOR_IDS = ("s2-preproj-shared", "s2-preproj-separate")
_ADAPTER_KINDS = ("identity", "adapter", "global_dense", "matched_projector")

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

    def __post_init__(self) -> None:
        for name in ("exclusive_create", "atomic_publish", "verified"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be boolean")


CheckpointSink = Callable[..., CheckpointPublication]


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
    run_manifest: Mapping[str, object]
    candidate_spec: Mapping[str, object]
    checkpoint_builder: CheckpointBuilder
    checkpoint_restorer: CheckpointRestorer
    dataset_factory: DatasetFactory = ProtocolSubjectDataset
    batch_size: int = 512
    max_train_steps: int | None = None
    num_workers: int = 0
    device: str | torch.device = "auto"
    resume_checkpoint: Mapping[str, object] | None = None
    checkpoint_sink: CheckpointSink | None = None
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
        _validate_string_mapping(self.run_manifest, "run_manifest")
        _validate_string_mapping(self.candidate_spec, "candidate_spec")
        for value, name in (
            (self.checkpoint_builder, "checkpoint_builder"),
            (self.checkpoint_restorer, "checkpoint_restorer"),
            (self.dataset_factory, "dataset_factory"),
        ):
            if not callable(value):
                raise TypeError(f"{name} must be callable")
        if self.checkpoint_sink is not None and not callable(
            self.checkpoint_sink
        ):
            raise TypeError("checkpoint_sink must be callable")
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
        if self.resume_checkpoint is not None:
            _validate_string_mapping(
                self.resume_checkpoint,
                "resume_checkpoint",
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


class SAMGARuntimeModel(nn.Module):
    """Verified SAMGA task plus exactly zero or one preregistered factor."""

    def __init__(self, spec: TrainingCellSpec) -> None:
        super().__init__()
        if not isinstance(spec, TrainingCellSpec):
            raise TypeError("spec must be a TrainingCellSpec")
        self.active_factor = spec.active_factor
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


def run_training_cell(spec: TrainingCellSpec) -> TrainingResult:
    """Run, resume, or smoke-test one locked SAMGA cell in memory."""

    if not isinstance(spec, TrainingCellSpec):
        raise TypeError("spec must be a TrainingCellSpec")
    device = _resolve_device(spec.device)
    _seed_fresh_process(spec.seed, device)
    train_dataset = _build_dataset(spec, "train", 0.3)
    validation_dataset = _build_dataset(spec, "val-dev", 0.0)
    if len(train_dataset) <= 0 or len(validation_dataset) <= 0:
        raise ValueError("train and val-dev datasets must be non-empty")

    model = SAMGARuntimeModel(spec).to(device)
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
        _validated_resume_checkpoint(spec)
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
    return dataset


def _build_optimizer(
    model: SAMGARuntimeModel,
    optimizer_stage: str,
) -> tuple[torch.optim.AdamW, torch.optim.lr_scheduler.LambdaLR]:
    if optimizer_stage == "stage1":
        include_shared = True
        learning_rate = 1e-4
    elif optimizer_stage == "stage2":
        include_shared = False
        learning_rate = 5e-5
    else:
        raise ValueError("optimizer stage must be stage1 or stage2")
    groups = model.optimizer_parameter_groups(
        include_shared=include_shared,
        base_learning_rate=learning_rate,
    )
    optimizer = torch.optim.AdamW(
        groups,
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[lambda _: 1.0 for _ in groups],
    )
    return optimizer, scheduler


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
    ):
        raise ValueError(f"{context} must be a CPU byte tensor")
    return value.detach().clone()


def _validated_resume_checkpoint(
    spec: TrainingCellSpec,
) -> dict[str, object]:
    payload = dict(
        _mapping(spec.resume_checkpoint, "resume_checkpoint")
    )
    if payload.get("schema_version") != 1:
        raise ValueError("checkpoint schema_version must be 1")
    if payload.get("payload_type") != CHECKPOINT_PAYLOAD_TYPE:
        raise ValueError("checkpoint payload_type mismatch")
    epoch = payload.get("epoch")
    global_step = payload.get("global_step")
    if (
        type(epoch) is not int
        or not 1 <= epoch <= TOTAL_EPOCHS
        or type(global_step) is not int
        or global_step < 0
    ):
        raise ValueError("checkpoint epoch/step is invalid")
    expected_stage = "stage1" if epoch <= STAGE1_EPOCHS else "stage2"
    if payload.get("optimizer_stage") != expected_stage:
        raise ValueError("checkpoint optimizer stage mismatch")
    expected_values: tuple[tuple[str, object], ...] = (
        ("subject", spec.subject),
        ("seed", spec.seed),
        ("config_sha256", spec.config_sha256),
        ("schedule_sha256", spec.schedule_sha256),
        ("trajectory_sha256", spec.trajectory_sha256),
        ("data_order_sha256", spec.data_order_sha256),
        ("effective_batch", spec.batch_size),
        ("input_hashes", dict(spec.input_hashes)),
        ("run_manifest", dict(spec.run_manifest)),
        ("candidate_spec", dict(spec.candidate_spec)),
    )
    for name, expected in expected_values:
        if payload.get(name) != expected:
            raise ValueError(f"checkpoint {name} mismatch")
    runtime = _mapping(
        payload.get("runtime_state"),
        "checkpoint runtime_state",
    )
    if runtime.get("schema_version") != 1:
        raise ValueError("checkpoint runtime state schema mismatch")
    next_epoch = runtime.get("next_epoch")
    epoch_complete = runtime.get("epoch_complete")
    if type(epoch_complete) is not bool:
        raise ValueError("checkpoint epoch_complete must be boolean")
    expected_next = epoch + int(epoch_complete)
    if next_epoch != expected_next:
        raise ValueError("checkpoint next_epoch mismatch")
    _integer_list(
        runtime.get("snapshot_epochs"),
        "checkpoint snapshot_epochs",
    )
    if runtime.get("required_retained_epochs") != list(LAST10_EPOCHS):
        raise ValueError("checkpoint retention window mismatch")
    _mapping(
        payload.get("sampler_state_dict"),
        "checkpoint sampler_state_dict",
    )
    retention = _mapping(
        payload.get("retention"),
        "checkpoint retention",
    )
    if retention.get("required_epochs") != list(LAST10_EPOCHS):
        raise ValueError("checkpoint retention policy mismatch")
    return payload


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
        "optimizer_base_lr": (
            1e-4 if optimizer_stage == "stage1" else 5e-5
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
    return payload


def _publish_checkpoint(
    sink: CheckpointSink | None,
    payload: dict[str, object],
    *,
    retain_for_averaging: bool,
) -> CheckpointPublication | None:
    if sink is None:
        return None
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
    return receipt
