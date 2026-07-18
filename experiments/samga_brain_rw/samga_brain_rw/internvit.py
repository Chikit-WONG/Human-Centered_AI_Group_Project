"""Semantic InternViT component and LoRA-target resolution.

The pinned InternViT V2.5 model exposes exactly ``model.encoder.layers``.
Resolution is deliberately structural and fail-closed: this module has no
CLIP-style or name-suffix fallbacks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch.nn as nn

from .hashing import sha256_json


@dataclass(frozen=True)
class LoraTarget:
    """One concrete, semantically classified InternViT LoRA target."""

    module_path: str
    module: nn.Linear = field(compare=False, repr=False)
    semantic_roles: tuple[str, ...]
    block_one_based: int
    block_zero_based: int
    in_features: int
    out_features: int
    has_bias: bool


def resolve_internvit_components(
    model: nn.Module,
) -> tuple[nn.Module, Sequence[nn.Module]]:
    """Return the pinned InternViT encoder and its concrete layer list."""
    if not isinstance(model, nn.Module):
        raise ValueError("model must be an nn.Module with model.encoder")
    encoder = getattr(model, "encoder", None)
    if not isinstance(encoder, nn.Module):
        raise ValueError("pinned InternViT requires model.encoder")
    layers = getattr(encoder, "layers", None)
    if not isinstance(layers, nn.ModuleList):
        raise ValueError(
            "pinned InternViT requires encoder.layers to be an nn.ModuleList"
        )
    return encoder, layers


def _linear(
    parent: nn.Module,
    attribute: str,
    module_path: str,
) -> nn.Linear:
    module = getattr(parent, attribute, None)
    if module is None:
        raise ValueError(f"missing required InternViT target {module_path}")
    if not isinstance(module, nn.Linear):
        raise ValueError(f"{module_path} must be an nn.Linear")
    return module


def _block_linears(
    block: nn.Module,
    block_zero_based: int,
) -> tuple[tuple[str, nn.Linear, tuple[str, ...]], ...]:
    prefix = f"encoder.layers.{block_zero_based}"
    attention = getattr(block, "attn", None)
    if not isinstance(attention, nn.Module):
        raise ValueError(f"missing required InternViT component {prefix}.attn")
    mlp = getattr(block, "mlp", None)
    if not isinstance(mlp, nn.Module):
        raise ValueError(f"missing required InternViT component {prefix}.mlp")

    qkv_path = f"{prefix}.attn.qkv"
    projection_path = f"{prefix}.attn.proj"
    fc1_path = f"{prefix}.mlp.fc1"
    fc2_path = f"{prefix}.mlp.fc2"
    qkv = _linear(attention, "qkv", qkv_path)
    projection = _linear(attention, "proj", projection_path)
    fc1 = _linear(mlp, "fc1", fc1_path)
    fc2 = _linear(mlp, "fc2", fc2_path)

    if qkv.out_features != 3 * qkv.in_features:
        raise ValueError(f"{qkv_path} has an incompatible qkv shape")
    if projection.in_features != projection.out_features:
        raise ValueError(f"{projection_path} must have a square shape")
    hidden_size = qkv.in_features
    if (
        projection.in_features != hidden_size
        or fc1.in_features != hidden_size
        or fc2.out_features != hidden_size
    ):
        raise ValueError(f"{prefix} components have incompatible hidden shapes")
    if fc1.out_features != fc2.in_features:
        raise ValueError(f"{prefix}.mlp components have incompatible shapes")

    return (
        (qkv_path, qkv, ("query", "key", "value")),
        (projection_path, projection, ("attention_output",)),
        (fc1_path, fc1, ("mlp_fc1",)),
        (fc2_path, fc2, ("mlp_fc2",)),
    )


def resolve_lora_targets(
    model: nn.Module,
    first_block: int = 28,
    last_block: int = 36,
) -> Sequence[LoraTarget]:
    """Resolve an inclusive one-based InternViT block range."""
    _, layers = resolve_internvit_components(model)
    if (
        type(first_block) is not int
        or type(last_block) is not int
        or first_block < 1
        or first_block > last_block
        or last_block > len(layers)
    ):
        raise ValueError(
            "block range must use integers satisfying "
            "1 <= first_block <= last_block <= len(encoder.layers)"
        )

    targets: list[LoraTarget] = []
    seen_paths: set[str] = set()
    seen_modules: set[int] = set()
    for block_one_based in range(first_block, last_block + 1):
        block_zero_based = block_one_based - 1
        block = layers[block_zero_based]
        for module_path, module, semantic_roles in _block_linears(
            block,
            block_zero_based,
        ):
            if module_path in seen_paths:
                raise ValueError(f"duplicate InternViT target path: {module_path}")
            module_identity = id(module)
            if module_identity in seen_modules:
                raise ValueError(
                    f"aliased InternViT target module: {module_path}"
                )
            seen_paths.add(module_path)
            seen_modules.add(module_identity)
            targets.append(
                LoraTarget(
                    module_path=module_path,
                    module=module,
                    semantic_roles=semantic_roles,
                    block_one_based=block_one_based,
                    block_zero_based=block_zero_based,
                    in_features=module.in_features,
                    out_features=module.out_features,
                    has_bias=module.bias is not None,
                )
            )

    expected_count = (last_block - first_block + 1) * 4
    if len(targets) != expected_count:
        raise ValueError(
            f"expected {expected_count} InternViT targets; found {len(targets)}"
        )
    return tuple(targets)


def build_lora_target_manifest(
    model: nn.Module,
    first_block: int = 28,
    last_block: int = 36,
) -> dict[str, object]:
    """Build the canonical JSON-compatible target manifest."""
    targets = resolve_lora_targets(model, first_block, last_block)
    payload: dict[str, object] = {
        "block_numbering": {
            "configuration": "one_based",
            "python_module_list": "zero_based",
            "zero_based_offset": -1,
        },
        "first_block_one_based": first_block,
        "last_block_one_based": last_block,
        "payload_type": "samga_brain_rw.internvit_lora_targets",
        "schema_version": 1,
        "scope": "train",
        "target_count": len(targets),
        "targets": [
            {
                "block_one_based": target.block_one_based,
                "block_zero_based": target.block_zero_based,
                "has_bias": target.has_bias,
                "in_features": target.in_features,
                "module_path": target.module_path,
                "out_features": target.out_features,
                "semantic_roles": list(target.semantic_roles),
                "weight_shape": [target.out_features, target.in_features],
            }
            for target in targets
        ],
    }
    return {**payload, "payload_sha256": sha256_json(payload)}


__all__ = [
    "LoraTarget",
    "build_lora_target_manifest",
    "resolve_internvit_components",
    "resolve_lora_targets",
]
