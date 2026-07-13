from typing import Dict, Tuple, Union, Optional


import torch
import safetensors
import torch.nn as nn
import torch.nn.functional as F
from diffusers.utils import logging
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.attention_processor import Attention, AttnProcessor2_0

logger = logging.get_logger(__name__)


class IPAttnProcessor2_0(nn.Module):
    r"""
    Attention processor for IP-Adapater for PyTorch 2.0.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        sequence_len (`int`, defaults to 4 when do ip_adapter_plus it should be 16):
            The context length of the image features.
    """

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: int,
        scale: Optional[float] = None,
    ):
        super().__init__()

        if not hasattr(nn.functional, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale if scale is not None else 1.0

        self.to_k_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        temb: torch.Tensor = None,
        ip_hidden_states: Tuple[torch.Tensor, torch.Tensor] = None,
        *args,
        **kwargs,
    ):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # for ip-adapter
        if ip_hidden_states is not None:
            ip_key = self.to_k_ip(ip_hidden_states)
            ip_value = self.to_v_ip(ip_hidden_states)

            ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(
                1, 2
            )

            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            ip_hidden_states = F.scaled_dot_product_attention(
                query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
            )

            with torch.no_grad():
                self.attn_map = query @ ip_key.transpose(-2, -1).softmax(dim=-1)

            ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(
                batch_size, -1, attn.heads * head_dim
            )
            ip_hidden_states = ip_hidden_states.to(query.dtype)

            hidden_states = hidden_states + self.scale * ip_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class IPAdapterModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        embed_dim: int = 1024,
        cross_attention_dim: int = 2048,
        sequence_length: int = 4,
        processor_hidden_size_map: Optional[Dict[str, int]] = None,
        scale: Optional[Union[float, int]] = 1.0,
    ):
        super().__init__()

        self.proj = nn.Linear(embed_dim, cross_attention_dim * sequence_length)
        self.norm = nn.LayerNorm(cross_attention_dim)

        self.embed_dim = embed_dim
        self.cross_attention_dim = cross_attention_dim
        self.sequence_length = sequence_length
        self.scale = float(scale)

        self.attn_processors = nn.ModuleDict()
        processor_hidden_size_map = processor_hidden_size_map or {}

        for index, hidden_size in sorted(
            processor_hidden_size_map.items(), key=lambda x: int(x[0])
        ):
            self.attn_processors[str(index)] = IPAttnProcessor2_0(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                scale=self.scale,
            )

    @classmethod
    def from_ip_adapter(
        cls,
        ip_adapter_model_path: str,
        scale: Optional[Union[float, int]] = 1.0,
    ):
        weights = safetensors.torch.load_file(ip_adapter_model_path)

        proj_weight = weights["image_proj.proj.weight"]
        proj_bias = weights["image_proj.proj.bias"]
        norm_weight = weights["image_proj.norm.weight"]
        norm_bias = weights["image_proj.norm.bias"]

        cross_attention_dim = norm_weight.shape[0]
        embed_dim = proj_weight.shape[1]

        if proj_bias.numel() % cross_attention_dim != 0:
            raise ValueError(
                "Invalid IP-Adapter projection shape: "
                f"proj_bias={tuple(proj_bias.shape)}, "
                f"cross_attention_dim={cross_attention_dim}."
            )

        sequence_length = proj_bias.numel() // cross_attention_dim

        processor_hidden_size_map = {}
        for key, value in weights.items():
            if key.startswith("ip_adapter.") and key.endswith(".to_k_ip.weight"):
                index = key.split(".")[1]
                processor_hidden_size_map[index] = value.shape[0]

        model = cls(
            embed_dim=embed_dim,
            cross_attention_dim=cross_attention_dim,
            sequence_length=sequence_length,
            processor_hidden_size_map=processor_hidden_size_map,
            scale=scale,
        )

        model.proj.load_state_dict({"weight": proj_weight, "bias": proj_bias})
        model.norm.load_state_dict({"weight": norm_weight, "bias": norm_bias})

        for index in processor_hidden_size_map:
            processor = model.attn_processors[index]

            processor.to_k_ip.load_state_dict(
                {"weight": weights[f"ip_adapter.{index}.to_k_ip.weight"]}
            )
            processor.to_v_ip.load_state_dict(
                {"weight": weights[f"ip_adapter.{index}.to_v_ip.weight"]}
            )

        model.eval()
        return model

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        batch_size = image_embeds.shape[0]
        image_embeds = self.proj(image_embeds)
        image_embeds = image_embeds.reshape(
            batch_size, self.sequence_length, self.cross_attention_dim
        )
        return self.norm(image_embeds)

    def bind_unet(self, unet: UNet2DConditionModel) -> Dict:
        attn_processors = dict(unet.attn_processors)
        bound_indices = set()

        for index, name in enumerate(attn_processors.keys()):
            index = str(index)
            if index in self.attn_processors:
                attn_processors[name] = self.attn_processors[index]
                bound_indices.add(index)

        missing = set(self.attn_processors.keys()) - bound_indices
        if missing:
            raise ValueError(
                "Some IP-Adapter processors were not bound to the UNet: "
                f"{sorted(missing, key=int)}"
            )

        unet.set_attn_processor(attn_processors)
        return attn_processors

    def unbind_unet(self, unet: UNet2DConditionModel) -> Dict:
        unet.set_attn_processor(AttnProcessor2_0())
        return unet.attn_processors

    def set_scale(self, scale: Union[float, int]):
        self.scale = float(scale)
        for processor in self.attn_processors.values():
            processor.scale = self.scale
