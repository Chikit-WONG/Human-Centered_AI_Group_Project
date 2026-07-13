from typing import Any, Optional
from dataclasses import dataclass

import torch
from torch import nn
from transformers.utils import logging, TransformersKwargs, ModelOutput
from transformers.utils.generic import can_return_tuple
from transformers.configuration_utils import PreTrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.models.clip.modeling_clip import _get_vector_norm, contrastive_loss
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

from .models_brain import BrainEncoder

logger = logging.get_logger(__name__)


@dataclass
class BrainCLIPOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits_per_image: torch.FloatTensor | None = None
    logits_per_brain: torch.FloatTensor | None = None
    brain_embeds: torch.FloatTensor | None = None
    image_embeds: torch.FloatTensor | None = None

    def to_tuple(self) -> tuple[Any]:
        return tuple(
            v.to_tuple() if isinstance(v, ModelOutput) else v for v in self.values()
        )


class BrainCLIPConfig(PreTrainedConfig):
    model_type = "brain_clip"
    model_type = "clip_brain_model"
    base_config_key = "brain_config"

    brain_backbone: str = "brain_mlp"
    num_brain_channels: int = 17
    brain_sequence_length: int = 250

    embed_dim: int = 512
    extra_dim: int = 1440
    dropout: float = 0.0

    logit_scale_init_value: float | int | None = 2.0


class BrainCLIPModel(PreTrainedModel):
    config: BrainCLIPConfig

    base_model_prefix = "brain_clip"
    input_modalities = ("embedding", "brain", "subject_id")

    def __init__(self, config: BrainCLIPConfig):
        super().__init__(config)

        self.brain_model = BrainEncoder(
            backbone=config.brain_backbone,
            brain_channels=config.num_brain_channels,
            num_layers=1,
            brain_sequence_length=config.brain_sequence_length,
            embed_dim=config.embed_dim,
            extra_dim=config.extra_dim,
            dropout=config.dropout,
        )
        self.logit_scale = nn.Parameter(
            torch.tensor(self.config.logit_scale_init_value)
        )

        self.post_init()

    def get_brain_features(
        self, brain_signals: torch.FloatTensor, subject_ids: torch.LongTensor = None
    ):
        return self.brain_model(brain_signals=brain_signals, subject_ids=subject_ids)

    @can_return_tuple
    def forward(
        self,
        brain_signals: torch.FloatTensor | None = None,
        image_embeds: torch.FloatTensor | None = None,
        subject_ids: torch.LongTensor | None = None,
        return_loss: bool = True,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BrainCLIPOutput:
        brain_embeds: torch.FloatTensor = self.brain_model(
            brain_signals=brain_signals, subject_ids=subject_ids
        )
        # normalized features
        image_embeds = image_embeds / _get_vector_norm(image_embeds)
        brain_embeds = brain_embeds / _get_vector_norm(brain_embeds)

        # cosine similarity as logits
        logits_per_brain = torch.matmul(
            brain_embeds, image_embeds.t().to(brain_embeds.device)
        )
        logits_per_brain = logits_per_brain * self.logit_scale.exp().to(
            brain_embeds.device
        )

        logits_per_image = logits_per_brain.t()

        loss = None
        if return_loss:
            loss = contrastive_loss(logits_per_brain)

        return BrainCLIPOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_brain=logits_per_brain,
            brain_embeds=brain_embeds,
            image_embeds=image_embeds,
        )


class BrainModel(PreTrainedModel):
    config: BrainCLIPConfig

    def __init__(self, config: BrainCLIPConfig):
        super().__init__(config)

        self.brain_model = BrainEncoder(
            backbone=config.brain_backbone,
            brain_channels=config.num_brain_channels,
            num_layers=1,
            brain_sequence_length=config.brain_sequence_length,
            embed_dim=config.embed_dim,
            extra_dim=config.extra_dim,
            dropout=config.dropout,
        )

        self.post_init()

    def forward(
        self,
        brain_signals: torch.FloatTensor | None = None,
        subject_ids: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        return self.brain_model(brain_signals, subject_ids=subject_ids)
