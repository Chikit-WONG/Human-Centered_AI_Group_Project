from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class ResidualMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.input = nn.Linear(input_dim, output_dim)
        self.residual = nn.Sequential(nn.GELU(), nn.Linear(output_dim, output_dim), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.input(values)
        return self.norm(values + self.residual(values))


class EEGProject(nn.Module):
    def __init__(self, channels: int = 17, samples: int = 250, output_dim: int = 1024) -> None:
        super().__init__()
        self.channels = channels
        self.samples = samples
        self.encoder = ResidualMLP(channels * samples, output_dim, dropout=0.3)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        if eeg.shape[-2:] != (self.channels, self.samples):
            raise ValueError(f"Expected EEG [...,{self.channels},{self.samples}], got {tuple(eeg.shape)}")
        return self.encoder(eeg.reshape(eeg.shape[0], -1))


class SubjectAwareRouter(nn.Module):
    def __init__(
        self,
        layer_ids: Sequence[int],
        num_subjects: int = 11,
        prior_center: int = 8,
        prior_strength: float = 1.0,
        temperature: float = 1.0,
        subject_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        ids = torch.tensor(list(layer_ids), dtype=torch.float32)
        if len(ids) > 1:
            step = min(float(value) for value in torch.diff(torch.sort(ids).values) if value > 0)
        else:
            step = 1.0
        self.global_logits = nn.Parameter(-prior_strength * torch.abs(ids - prior_center) / step)
        self.subject_bias = nn.Embedding(num_subjects, len(layer_ids))
        nn.init.zeros_(self.subject_bias.weight)
        self.temperature = float(temperature)
        self.subject_dropout = float(subject_dropout)

    def forward(self, subject_ids: torch.Tensor, *, force_global: bool = False) -> torch.Tensor:
        logits = self.global_logits.unsqueeze(0).expand(subject_ids.shape[0], -1)
        if not force_global:
            bias = self.subject_bias(subject_ids.long())
            if self.training and self.subject_dropout > 0:
                keep = (torch.rand(subject_ids.shape[0], 1, device=subject_ids.device) > self.subject_dropout)
                bias = bias * keep.to(bias.dtype)
            logits = logits + bias
        return torch.softmax(logits / self.temperature, dim=-1)

    def global_weights(self) -> torch.Tensor:
        return torch.softmax(self.global_logits / self.temperature, dim=-1)


class SAMGATaskModel(nn.Module):
    def __init__(
        self,
        *,
        layer_ids: Sequence[int] = (4, 6, 8, 10, 12),
        image_dim: int = 768,
        image_mid_dim: int = 1024,
        eeg_dim: int = 1024,
        feature_dim: int = 512,
        channels: int = 17,
        samples: int = 250,
        prior_center: int = 8,
    ) -> None:
        super().__init__()
        self.layer_ids = tuple(int(value) for value in layer_ids)
        self.eeg_encoder = EEGProject(channels=channels, samples=samples, output_dim=eeg_dim)
        self.eeg_projector = nn.Linear(eeg_dim, feature_dim)
        self.image_pre_projector = nn.Linear(image_dim, image_mid_dim)
        self.image_projectors = nn.ModuleList(
            nn.Linear(image_mid_dim, feature_dim) for _ in self.layer_ids
        )
        self.router = SubjectAwareRouter(self.layer_ids, prior_center=prior_center)
        self.shared_encoder = nn.Linear(feature_dim, feature_dim)
        self.shared_frozen = False

    def encode_eeg(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.shared_encoder(self.eeg_projector(self.eeg_encoder(eeg)))

    def encode_image(
        self,
        layer_features: torch.Tensor,
        subject_ids: torch.Tensor,
        *,
        force_global: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_features.ndim != 3 or layer_features.shape[1] != len(self.layer_ids):
            raise ValueError(
                f"Expected image features [B,{len(self.layer_ids)},D], got {tuple(layer_features.shape)}"
            )
        common = self.image_pre_projector(layer_features)
        projected = torch.stack(
            [projector(common[:, index, :]) for index, projector in enumerate(self.image_projectors)],
            dim=1,
        )
        weights = self.router(subject_ids, force_global=force_global)
        image = self.shared_encoder(torch.sum(projected * weights.unsqueeze(-1), dim=1))
        return image, weights

    def forward(
        self,
        eeg: torch.Tensor,
        layer_features: torch.Tensor,
        subject_ids: torch.Tensor,
        *,
        force_global: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eeg_features = self.encode_eeg(eeg)
        image_features, weights = self.encode_image(
            layer_features, subject_ids, force_global=force_global
        )
        return eeg_features, image_features, weights

    def freeze_shared_encoder(self) -> None:
        for parameter in self.shared_encoder.parameters():
            parameter.requires_grad = False
        self.shared_frozen = True
        self.shared_encoder.eval()

    def task_parameters(self, *, include_shared: bool) -> list[nn.Parameter]:
        modules: list[nn.Module] = [
            self.eeg_encoder,
            self.eeg_projector,
            self.image_pre_projector,
            self.image_projectors,
            self.router,
        ]
        if include_shared:
            modules.append(self.shared_encoder)
        return [parameter for module in modules for parameter in module.parameters() if parameter.requires_grad]


def _pairwise_squared(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_norm = (x * x).sum(dim=1, keepdim=True)
    y_norm = (y * y).sum(dim=1, keepdim=True).T
    return (x_norm + y_norm - 2 * x @ y.T).clamp_min(0)


def multi_kernel_mmd(
    x: torch.Tensor,
    y: torch.Tensor,
    sigmas: Sequence[float] = (0.1, 0.2, 0.5, 1.0, 2.0),
) -> torch.Tensor:
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError(f"MMD expects equal 2D tensors, got {x.shape}/{y.shape}")
    if x.shape[0] < 2:
        return x.new_zeros(())
    distances = (_pairwise_squared(x, x), _pairwise_squared(y, y), _pairwise_squared(x, y))
    kernels = []
    for distance in distances:
        kernels.append(
            torch.stack([torch.exp(-distance / (2 * sigma * sigma)) for sigma in sigmas]).mean(0)
        )
    k_xx, k_yy, k_xy = kernels
    batch = x.shape[0]
    xx = (k_xx.sum() - k_xx.diagonal().sum()) / (batch * (batch - 1))
    yy = (k_yy.sum() - k_yy.diagonal().sum()) / (batch * (batch - 1))
    return (xx + yy - 2 * k_xy.mean()).clamp_min(0)


@dataclass
class LossOutput:
    total: torch.Tensor
    contrastive: torch.Tensor
    mmd: torch.Tensor
    scale: torch.Tensor


class SAMGALoss(nn.Module):
    def __init__(
        self,
        initial_temperature: float = 0.07,
        *,
        eeg_l2norm: bool = False,
        image_l2norm: bool = True,
    ) -> None:
        super().__init__()
        self.register_buffer("raw_scale", torch.tensor(float(np.log(1.0 / initial_temperature))))
        self.eeg_l2norm = bool(eeg_l2norm)
        self.image_l2norm = bool(image_l2norm)

    def forward(
        self,
        eeg_features: torch.Tensor,
        image_features: torch.Tensor,
        *,
        mmd_weight: float,
    ) -> LossOutput:
        eeg = eeg_features.float()
        image = image_features.float()
        if self.eeg_l2norm:
            eeg = F.normalize(eeg, dim=-1)
        if self.image_l2norm:
            image = F.normalize(image, dim=-1)
        scale = F.softplus(self.raw_scale)
        logits = scale * eeg @ image.T
        labels = torch.arange(logits.shape[0], device=logits.device)
        contrastive = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
        mmd = multi_kernel_mmd(eeg_features.float(), image_features.float())
        total = float(mmd_weight) * mmd + (1.0 - float(mmd_weight)) * contrastive
        return LossOutput(total=total, contrastive=contrastive, mmd=mmd, scale=scale)


class MultiLayerCLIPProvider(nn.Module):
    def __init__(self, backbone: nn.Module, layer_ids: Sequence[int], *, trainable: bool) -> None:
        super().__init__()
        self.backbone = backbone
        self.layer_ids = tuple(int(value) for value in layer_ids)
        self.trainable = bool(trainable)

    def _base_model(self) -> nn.Module:
        if hasattr(self.backbone, "get_base_model"):
            return self.backbone.get_base_model()
        return self.backbone

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.trainable:
            outputs = self.backbone(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            )
        else:
            with torch.no_grad():
                outputs = self.backbone(
                    pixel_values=pixel_values,
                    output_hidden_states=True,
                    return_dict=True,
                )
        if outputs.hidden_states is None:
            raise RuntimeError("CLIP did not return hidden states")
        base = self._base_model()
        post_norm = base.vision_model.post_layernorm
        layers = [post_norm(outputs.hidden_states[layer][:, 0, :]) for layer in self.layer_ids]
        return torch.stack(layers, dim=1)


def load_clip_provider(
    *,
    model_path: str,
    layer_ids: Sequence[int],
    vision_mode: str,
    lora_rank: int = 32,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[MultiLayerCLIPProvider, object]:
    from peft import LoraConfig, get_peft_model
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    vision = CLIPVisionModelWithProjection.from_pretrained(model_path, local_files_only=True)
    vision.gradient_checkpointing_enable()
    trainable = vision_mode == "lora"
    if trainable:
        config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
            init_lora_weights="gaussian",
            bias="none",
        )
        vision = get_peft_model(vision, config)
    elif vision_mode == "frozen":
        for parameter in vision.parameters():
            parameter.requires_grad = False
    else:
        raise ValueError(f"Unsupported vision mode {vision_mode}")
    vision.to(device=device, dtype=dtype)
    provider = MultiLayerCLIPProvider(vision, layer_ids, trainable=trainable).to(device)
    processor = CLIPImageProcessor.from_pretrained(model_path, local_files_only=True)
    return provider, processor
