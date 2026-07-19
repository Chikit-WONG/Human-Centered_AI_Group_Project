"""Locked SAMGA task assembly built only from verified upstream components."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields

import torch
from torch import nn

from .upstream_samga import UpstreamComponents


_LOCKED_VALUES: dict[str, object] = {
    "layer_ids": (20, 24, 28, 32, 36),
    "image_dim": 3_200,
    "image_mid_dim": 1_024,
    "eeg_dim": 1_024,
    "feature_dim": 512,
    "channels": 17,
    "samples": 250,
    "num_subjects": 11,
    "prior_center": 28,
    "prior_strength": 1.0,
    "router_temperature": 1.0,
    "router_subject_dropout": 0.3,
    "router_eval_mode": "global",
    "initial_temperature": 0.07,
    "alpha": 1.0,
    "beta": 1.0,
    "eeg_l2norm": False,
    "image_l2norm": True,
    "text_l2norm": False,
    "temperature_learnable": False,
    "softplus": True,
}


@dataclass(frozen=True)
class SAMGABaseConfig:
    """The immutable InternViT-SAMGA baseline architecture contract."""

    layer_ids: tuple[int, ...] = (20, 24, 28, 32, 36)
    image_dim: int = 3_200
    image_mid_dim: int = 1_024
    eeg_dim: int = 1_024
    feature_dim: int = 512
    channels: int = 17
    samples: int = 250
    num_subjects: int = 11
    prior_center: int = 28
    prior_strength: float = 1.0
    router_temperature: float = 1.0
    router_subject_dropout: float = 0.3
    router_eval_mode: str = "global"
    initial_temperature: float = 0.07
    alpha: float = 1.0
    beta: float = 1.0
    eeg_l2norm: bool = False
    image_l2norm: bool = True
    text_l2norm: bool = False
    temperature_learnable: bool = False
    softplus: bool = True

    def __post_init__(self) -> None:
        for field in fields(self):
            expected = _LOCKED_VALUES[field.name]
            actual = getattr(self, field.name)
            if actual != expected or type(actual) is not type(expected):
                raise ValueError(
                    f"{field.name} is locked to {expected!r}, got {actual!r}"
                )


@dataclass(frozen=True)
class SAMGALossOutput:
    """The locked two-stage objective decomposition."""

    total: torch.Tensor
    contrastive: torch.Tensor
    mmd: torch.Tensor


class SAMGATaskModel(nn.Module):
    """Exact safe assembly of the released EEG/projector/router primitives."""

    def __init__(
        self,
        *,
        components: UpstreamComponents,
        config: SAMGABaseConfig | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(components, UpstreamComponents):
            raise TypeError("components must be verified UpstreamComponents")
        if config is None:
            config = SAMGABaseConfig()
        if not isinstance(config, SAMGABaseConfig):
            raise TypeError("config must be SAMGABaseConfig")
        self.config = config
        self._components = components

        self.eeg_encoder = components.EEGProject(
            feature_dim=config.eeg_dim,
            eeg_sample_points=config.samples,
            channels_num=config.channels,
        )
        self.eeg_projector = components.ProjectorLinear(
            config.eeg_dim,
            config.feature_dim,
        )
        self.image_pre_projector = components.ProjectorLinear(
            config.image_dim,
            config.image_mid_dim,
        )
        self.text_projector = components.ProjectorLinear(
            config.image_dim,
            config.feature_dim,
        )
        self.image_projectors = nn.ModuleList(
            components.ProjectorLinear(
                config.image_mid_dim,
                config.feature_dim,
            )
            for _ in config.layer_ids
        )
        self.shared_encoder = components.ShareEncoder(
            config.feature_dim,
            config.feature_dim,
        )
        self.router = components.SubjectAwareLayerMixer(
            layer_ids=config.layer_ids,
            num_subjects=config.num_subjects,
            prior_center=config.prior_center,
            prior_strength=config.prior_strength,
            temperature=config.router_temperature,
            subject_dropout=config.router_subject_dropout,
        )
        self.criterion = components.ContrastiveLoss(
            config.initial_temperature,
            config.alpha,
            config.beta,
            config.eeg_l2norm,
            config.image_l2norm,
            config.text_l2norm,
            config.temperature_learnable,
            config.softplus,
        )
        self._shared_frozen = False

    def encode_eeg(self, eeg: torch.Tensor) -> torch.Tensor:
        """Encode exact ``[B, 17, 250]`` trial-averaged EEG tensors."""

        if eeg.ndim != 3 or tuple(eeg.shape[1:]) != (
            self.config.channels,
            self.config.samples,
        ):
            raise ValueError(
                "EEG must have shape "
                f"[B,{self.config.channels},{self.config.samples}]"
            )
        return self.shared_encoder(self.eeg_projector(self.eeg_encoder(eeg)))

    def encode_image(
        self,
        layer_features: torch.Tensor,
        subject_ids: torch.Tensor,
        *,
        force_global: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project five InternViT layers and route globally during evaluation."""

        expected = (
            len(self.config.layer_ids),
            self.config.image_dim,
        )
        if layer_features.ndim != 3 or tuple(layer_features.shape[1:]) != expected:
            raise ValueError(
                "image features must have shape "
                f"[B,{expected[0]},{expected[1]}]"
            )
        self._validate_subject_ids(subject_ids, layer_features.shape[0])
        if not self.training:
            if force_global is False:
                raise ValueError("SAMGA evaluation is locked to the global router")
            effective_force_global = True
        else:
            effective_force_global = bool(force_global)

        common = self.image_pre_projector(layer_features)
        projected = torch.stack(
            [
                projector(common[:, index, :])
                for index, projector in enumerate(self.image_projectors)
            ],
            dim=1,
        )
        weights = self.router(
            subject_ids,
            force_global=effective_force_global,
        )
        mixed = torch.sum(projected * weights.unsqueeze(-1), dim=1)
        return self.shared_encoder(mixed), weights

    def encode_text(self, text_features: torch.Tensor) -> torch.Tensor:
        """Retain the released no-text launch's optimizer-compatible projector."""

        if text_features.ndim != 2 or text_features.shape[1] != self.config.image_dim:
            raise ValueError(
                f"text features must have shape [B,{self.config.image_dim}]"
            )
        return self.text_projector(text_features)

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
        """Compute released image-only-normalized contrastive loss plus MMD."""

        if (
            eeg_features.ndim != 2
            or image_features.ndim != 2
            or eeg_features.shape != image_features.shape
        ):
            raise ValueError("loss expects equal two-dimensional feature tensors")
        if not isinstance(mmd_weight, (int, float)) or not math.isfinite(
            float(mmd_weight)
        ):
            raise ValueError("mmd_weight must be finite")
        weight = float(mmd_weight)
        if not 0.0 <= weight <= 1.0:
            raise ValueError("mmd_weight must be between zero and one")
        contrastive = self.criterion(
            eeg_features,
            image_features,
            image_features,
        )
        mmd = self._components.mmd_rbf(eeg_features, image_features)
        return SAMGALossOutput(
            total=weight * mmd + (1.0 - weight) * contrastive,
            contrastive=contrastive,
            mmd=mmd,
        )

    def optimizer_parameters(self, *, include_shared: bool) -> list[nn.Parameter]:
        """Return parameters in the exact released single-group build order."""

        if type(include_shared) is not bool:
            raise TypeError("include_shared must be boolean")
        modules: list[nn.Module] = [
            self.eeg_encoder,
            self.eeg_projector,
            self.image_pre_projector,
            self.text_projector,
        ]
        if include_shared:
            modules.append(self.shared_encoder)
        modules.extend((self.image_projectors, self.router))
        parameters = [
            parameter
            for module in modules
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        parameters.extend(
            parameter
            for parameter in self.criterion.parameters()
            if parameter.requires_grad
        )
        if len({id(parameter) for parameter in parameters}) != len(parameters):
            raise AssertionError("optimizer parameters contain a duplicate")
        return parameters

    def freeze_shared_encoder(self) -> None:
        for parameter in self.shared_encoder.parameters():
            parameter.requires_grad = False
        self._shared_frozen = True
        self.shared_encoder.eval()

    def train(self, mode: bool = True) -> SAMGATaskModel:
        super().train(mode)
        if self._shared_frozen:
            self.shared_encoder.eval()
        return self

    def checkpoint_state_dicts(self) -> Mapping[str, Mapping[str, torch.Tensor]]:
        """Expose keys matching the released checkpoint component names."""

        return {
            "model_state_dict": self.eeg_encoder.state_dict(),
            "eeg_projector_state_dict": self.eeg_projector.state_dict(),
            "img_pre_projector_state_dict": self.image_pre_projector.state_dict(),
            "text_projector_state_dict": self.text_projector.state_dict(),
            "share_enc_state_dict": self.shared_encoder.state_dict(),
            "img_projectors_state_dict": self.image_projectors.state_dict(),
            "layer_router_state_dict": self.router.state_dict(),
        }

    def _validate_subject_ids(
        self,
        subject_ids: torch.Tensor,
        batch_size: int,
    ) -> None:
        if subject_ids.ndim != 1 or subject_ids.shape[0] != batch_size:
            raise ValueError("subject IDs must have shape [B]")
        if subject_ids.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise ValueError("subject IDs must use an integer dtype")
        if bool(((subject_ids < 1) | (subject_ids > 10)).any()):
            raise ValueError("subject IDs must be between 1 and 10")
