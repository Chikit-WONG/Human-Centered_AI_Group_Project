"""Development-only Brain-RW/CLIP-LoRA primitives.

There is deliberately no formal-test loader in this module. Protocol metadata
and typed checkpoint sidecars are verified before EEG, image, NumPy, or
PyTorch payload loading.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import re
import stat
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, MethodType

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as torch_checkpoint
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Sampler

from .access import TypedArtifact, verify_typed_artifacts
from .config import SemanticConfig, make_run_key
from .data import (
    POSTERIOR_CHANNELS,
    ProtocolSubjectDataset,
    inspect_source_payload_identity,
)
from .hashing import canonical_json_bytes, ordered_ids_sha256, sha256_json
from .runtime_contract import (
    capture_semantic_environment,
    require_pinned_semantic_environment,
)


CLIP_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "fc1",
    "fc2",
    "visual_projection",
)
BRAINRW_CHECKPOINT_TYPE = "samga_brain_rw.checkpoint"
_ROLE_PAYLOAD_TYPE = "samga_brain_rw.role_payload"
_PROTOCOL_RE = re.compile(r"^sub-(\d{2})_protocol\.json$")
_SUBJECT_TEST_RE = re.compile(
    r"^sub-\d{2}_test(?:\.[A-Za-z0-9._-]+)?$",
    re.IGNORECASE,
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_DEVELOPMENT_SCOPES = frozenset({"train", "val-dev"})
_SEALED_COMPONENTS = frozenset(
    {
        "formal",
        "formal_input",
        "formal_refit",
        "formal_test",
        "test",
        "test_images",
        "val_confirm",
    }
)
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)

_BRAINRW_CHECKPOINT_KEYS = frozenset(
    {
        "candidate_initialization_sha256",
        "candidate_state",
        "clip_config_sha256",
        "clip_path",
        "clip_preprocessor_sha256",
        "clip_weights_sha256",
        "complete",
        "config_path",
        "config_payload",
        "config_sha256",
        "data_order_sha256",
        "dataloader_generator_state",
        "effective_batch_size",
        "environment",
        "epoch",
        "git_provenance",
        "git_sha",
        "global_step",
        "input_bundle_sha256",
        "input_hashes",
        "manifest_path",
        "manifest_sha256",
        "model_manifest",
        "model_manifest_sha256",
        "observed_scopes",
        "optimizer_state",
        "payload_type",
        "planned_steps",
        "protocol_sha256",
        "resumed_from_sha256",
        "rng_state",
        "run_key",
        "runtime_contract",
        "runtime_contract_sha256",
        "runtime_dtype",
        "runtime_evidence",
        "runtime_evidence_sha256",
        "semantic_environment",
        "semantic_environment_sha256",
        "sampler_state",
        "scheduler_state",
        "schema_version",
        "scope",
        "seed",
        "steps",
        "subject",
        "target_manifest_sha256",
        "training_complete",
        "task_initialization_sha256",
        "task_state",
        "validation_metrics",
        "validation_scope",
    }
)
_BRAINRW_INPUT_HASH_KEYS = frozenset(
    {
        "clip_config",
        "clip_preprocessor",
        "clip_weights",
        "config",
        "manifest",
        "protocol",
        "records",
        "semantic_environment",
        "source_manifest",
        "source_payload",
        "train_role",
        "val_dev_role",
    }
)
_BRAINRW_TRAIN_ONLY_INPUT_HASH_KEYS = (
    _BRAINRW_INPUT_HASH_KEYS | {"validation_policy"}
)
_BRAINRW_RUNTIME_CONTRACT = MappingProxyType(
    {
        "accelerator": "NVIDIA A40",
        "device_type": "cuda",
        "dtype": "bfloat16",
        "schema_version": 1,
    }
)
_BRAINRW_RUNTIME_EVIDENCE_KEYS = frozenset(
    {
        "accelerator_name",
        "bf16_supported",
        "cuda_available",
        "cuda_capability",
        "cuda_device_count",
        "cuda_device_index",
        "cuda_version",
        "device_type",
        "dtype",
        "schema_version",
        "torch_version",
        "total_memory_bytes",
    }
)


@dataclass(frozen=True)
class BrainRWProductionRuntime:
    device: torch.device
    dtype: torch.dtype
    contract: Mapping[str, object]
    contract_sha256: str
    semantic_environment: Mapping[str, object]
    semantic_environment_sha256: str
    evidence: Mapping[str, object]
    evidence_sha256: str


def probe_brainrw_production_runtime() -> BrainRWProductionRuntime:
    semantic_environment = require_pinned_semantic_environment(
        capture_semantic_environment()
    )
    if not torch.cuda.is_available():
        raise RuntimeError("Brain-RW production runtime requires CUDA")
    device_count = int(torch.cuda.device_count())
    device_index = int(torch.cuda.current_device())
    if device_count <= 0 or not 0 <= device_index < device_count:
        raise RuntimeError("Brain-RW CUDA device identity is invalid")
    properties = torch.cuda.get_device_properties(device_index)
    accelerator_name = str(properties.name)
    if accelerator_name != _BRAINRW_RUNTIME_CONTRACT["accelerator"]:
        raise RuntimeError("Brain-RW production runtime requires NVIDIA A40")
    if torch.cuda.is_bf16_supported() is not True:
        raise RuntimeError("Brain-RW production runtime requires bfloat16")
    cuda_version = torch.version.cuda
    if not isinstance(cuda_version, str) or not cuda_version:
        raise RuntimeError("Brain-RW CUDA version evidence is unavailable")
    evidence = {
        "accelerator_name": accelerator_name,
        "bf16_supported": True,
        "cuda_available": True,
        "cuda_capability": [
            int(properties.major),
            int(properties.minor),
        ],
        "cuda_device_count": device_count,
        "cuda_device_index": device_index,
        "cuda_version": cuda_version,
        "device_type": "cuda",
        "dtype": "bfloat16",
        "schema_version": 1,
        "torch_version": str(torch.__version__),
        "total_memory_bytes": int(properties.total_memory),
    }
    contract = dict(_BRAINRW_RUNTIME_CONTRACT)
    return BrainRWProductionRuntime(
        device=torch.device("cuda", device_index),
        dtype=torch.bfloat16,
        contract=MappingProxyType(contract),
        contract_sha256=sha256_json(contract),
        semantic_environment=MappingProxyType(
            semantic_environment
        ),
        semantic_environment_sha256=sha256_json(semantic_environment),
        evidence=MappingProxyType(evidence),
        evidence_sha256=sha256_json(evidence),
    )


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _probability(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result < 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1)")
    return result


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def state_dict_sha256(state: Mapping[str, Tensor]) -> str:
    if not isinstance(state, Mapping):
        raise TypeError("state must be a mapping")
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(name, str) or not isinstance(value, Tensor):
            raise TypeError("state must map strings to tensors")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_tensor_sha256(value).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def reject_development_path(path: Path, context: str) -> Path:
    value = Path(path)
    raw = os.fspath(value)
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise ValueError(f"{context} must be a non-empty text path")
    normalized = Path(os.path.abspath(os.path.normpath(raw)))
    for component in normalized.parts:
        semantic = re.sub(
            r"[^a-z0-9]+",
            "_",
            component.lower(),
        ).strip("_")
        if (
            semantic in _SEALED_COMPONENTS
            or _SUBJECT_TEST_RE.fullmatch(component)
        ):
            raise PermissionError(
                f"{context} contains a sealed-scope component"
            )
    current = Path(normalized.anchor)
    for component in normalized.parts[1:]:
        current /= component
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            break
        except OSError as exc:
            raise ValueError(
                f"{context} cannot be inspected safely"
            ) from exc
        if stat.S_ISLNK(mode):
            raise PermissionError(
                f"{context} contains a symlink component"
            )
    return normalized


class _BrainResidualLayer(nn.Module):
    def __init__(self, dimension: int, dropout: float) -> None:
        super().__init__()
        self.w1 = nn.Linear(dimension, dimension, bias=False)
        self.w2 = nn.Linear(dimension, dimension, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dimension, eps=1e-6)

    def forward(self, values: Tensor) -> Tensor:
        update = self.w2(self.dropout(F.silu(self.w1(values))))
        return self.norm(values + update)


class BrainMLP(nn.Module):
    """Audited one-residual-layer brain-rw MLP."""

    def __init__(
        self,
        channels: int,
        samples: int,
        hidden_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.channels = _positive_int(channels, "channels")
        self.samples = _positive_int(samples, "samples")
        self.hidden_size = _positive_int(hidden_size, "hidden_size")
        probability = _probability(dropout, "dropout")
        self.proj_in = nn.Linear(self.channels * self.samples, self.hidden_size)
        self.layers = nn.ModuleList(
            [_BrainResidualLayer(self.hidden_size, probability)]
        )

    def forward(self, brain_signals: Tensor) -> Tensor:
        if (
            not isinstance(brain_signals, Tensor)
            or not brain_signals.is_floating_point()
        ):
            raise TypeError("brain_signals must be a floating torch.Tensor")
        if brain_signals.ndim != 3 or tuple(brain_signals.shape[1:]) != (
            self.channels,
            self.samples,
        ):
            raise ValueError(
                f"brain_signals must have shape [B,{self.channels},{self.samples}]"
            )
        if not bool(torch.isfinite(brain_signals).all()):
            raise ValueError("brain_signals contain non-finite values")
        values = self.proj_in(
            brain_signals.reshape(brain_signals.shape[0], -1)
        )
        for layer in self.layers:
            values = layer(values)
        return values


def _initialize_legacy_brain_module(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class LoRALinear(nn.Module):
    """Frozen Linear plus a zero-output LoRA branch."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear base must be nn.Linear")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.rank = _positive_int(rank, "rank")
        self.alpha = _positive_int(alpha, "alpha")
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(_probability(dropout, "lora_dropout"))
        factory = {
            "device": base.weight.device,
            "dtype": base.weight.dtype,
        }
        self.lora_A = nn.Parameter(
            torch.empty(self.rank, base.in_features, **factory)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(base.out_features, self.rank, **factory)
        )
        nn.init.normal_(self.lora_A, mean=0.0, std=1.0 / self.rank)

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, values: Tensor) -> Tensor:
        baseline = self.base(values)
        update = F.linear(
            F.linear(self.dropout(values), self.lora_A),
            self.lora_B,
        )
        return baseline + update * self.scaling


def _replace_module(root: nn.Module, path: str, replacement: nn.Module) -> None:
    parent_name, _, leaf = path.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    if leaf.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
        parent[int(leaf)] = replacement
    else:
        setattr(parent, leaf, replacement)


def _inject_exact_clip_lora(
    model: nn.Module,
    targets: Sequence[str],
    rank: int,
    alpha: int,
    dropout: float,
) -> tuple[tuple[str, ...], tuple[dict[str, object], ...]]:
    requested = tuple(targets)
    if requested != CLIP_LORA_TARGETS:
        raise ValueError(
            "LoRA targets must equal the exact locked CLIP target order"
        )
    named = {
        name: module
        for name, module in model.named_modules()
        if name and isinstance(module, nn.Linear)
    }
    grouped = {
        target: sorted(
            name
            for name in named
            if name.rsplit(".", 1)[-1] == target
        )
        for target in requested
    }
    for target, names in grouped.items():
        if not names:
            raise ValueError(f"CLIP is missing locked LoRA target {target}")
    if len(grouped["visual_projection"]) != 1:
        raise ValueError("CLIP must expose exactly one visual_projection")
    if len({len(grouped[target]) for target in requested[:-1]}) != 1:
        raise ValueError("CLIP transformer LoRA target counts are unbalanced")
    resolved = tuple(
        sorted(name for names in grouped.values() for name in names)
    )
    manifest: list[dict[str, object]] = []
    for name in resolved:
        layer = named[name]
        manifest.append(
            {
                "module_path": name,
                "in_features": layer.in_features,
                "out_features": layer.out_features,
                "weight_sha256": _tensor_sha256(layer.weight),
                "bias_sha256": (
                    None
                    if layer.bias is None
                    else _tensor_sha256(layer.bias)
                ),
            }
        )
    for name in resolved:
        _replace_module(
            model,
            name,
            LoRALinear(
                named[name],
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            ),
        )
    return resolved, tuple(manifest)


def _explicit_checkpointed_clip_layer_forward(
    layer: nn.Module,
    *args: object,
    **kwargs: object,
) -> object:
    original_forward = object.__getattribute__(
        layer,
        "_brainrw_original_forward",
    )
    if layer.training and torch.is_grad_enabled():
        return torch_checkpoint.checkpoint(
            original_forward,
            *args,
            use_reentrant=False,
            **kwargs,
        )
    return original_forward(*args, **kwargs)


def _enable_vision_gradient_checkpointing(
    vision_model: nn.Module,
) -> dict[str, object]:
    try:
        encoder = vision_model.get_submodule(
            "vision_model.encoder"
        )
    except (AttributeError, TypeError) as exc:
        raise RuntimeError(
            "CLIP vision model lacks the locked encoder target"
        ) from exc
    layers = getattr(encoder, "layers", None)
    configured_layers = getattr(
        getattr(vision_model, "config", None),
        "num_hidden_layers",
        None,
    )
    if (
        not isinstance(layers, nn.ModuleList)
        or not layers
        or type(configured_layers) is not int
        or configured_layers != len(layers)
    ):
        raise RuntimeError(
            "CLIP vision encoder layers differ from the locked target"
        )
    for layer in layers:
        if (
            not isinstance(layer, nn.Module)
            or not callable(getattr(layer, "forward", None))
            or hasattr(layer, "_brainrw_original_forward")
        ):
            raise RuntimeError(
                "CLIP vision layer cannot be checkpoint-wrapped exactly"
            )
    for layer in layers:
        original_forward = layer.forward
        object.__setattr__(
            layer,
            "_brainrw_original_forward",
            original_forward,
        )
        object.__setattr__(
            layer,
            "forward",
            MethodType(
                _explicit_checkpointed_clip_layer_forward,
                layer,
            ),
        )
    return {
        "enabled": True,
        "method": "explicit_per_layer_torch_utils_checkpoint",
        "requested": True,
        "target": "vision_model.encoder.layers",
        "use_reentrant": False,
        "wrapped_layer_count": len(layers),
    }


@dataclass(frozen=True)
class BrainRWOutput:
    loss: Tensor | None
    similarity: Tensor
    brain_embeds: Tensor
    image_embeds: Tensor


class BrainRWCLIPLoRAModel(nn.Module):
    """Audited BrainMLP aligned to an exact frozen-base CLIP LoRA model."""

    def __init__(
        self,
        vision_model: nn.Module,
        *,
        channels: int = 17,
        samples: int = 250,
        projection_dim: int = 512,
        dropout: float = 0.1,
        lora_targets: Sequence[str] = CLIP_LORA_TARGETS,
        lora_rank: int = 32,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(vision_model, nn.Module):
            raise TypeError("vision_model must be an nn.Module")
        gradient_checkpointing = (
            _enable_vision_gradient_checkpointing(vision_model)
        )
        self.projection_dim = _positive_int(
            projection_dim, "projection_dim"
        )
        self.brain_mlp = BrainMLP(
            channels=channels,
            samples=samples,
            hidden_size=self.projection_dim,
            dropout=dropout,
        )
        self.brain_mlp.apply(
            _initialize_legacy_brain_module
        )
        self.logit_scale = nn.Parameter(
            torch.tensor(2.0, dtype=torch.float32)
        )
        self.vision_model = vision_model
        for parameter in self.vision_model.parameters():
            parameter.requires_grad = False
        resolved, target_manifest = _inject_exact_clip_lora(
            self.vision_model,
            lora_targets,
            lora_rank,
            lora_alpha,
            lora_dropout,
        )
        self.resolved_lora_targets = resolved
        self.target_manifest = target_manifest
        self.target_manifest_sha256 = sha256_json(list(target_manifest))
        manifest = {
            "schema_version": 1,
            "brain_mlp": {
                "channels": channels,
                "samples": samples,
                "projection_dim": self.projection_dim,
                "layers": 1,
                "dropout": float(dropout),
                "layer_norm_eps": 1e-6,
            },
            "gradient_checkpointing": gradient_checkpointing,
            "lora": {
                "semantic_targets": list(lora_targets),
                "resolved_targets": list(resolved),
                "target_manifest_sha256": self.target_manifest_sha256,
                "rank": int(lora_rank),
                "alpha": int(lora_alpha),
                "dropout": float(lora_dropout),
            },
        }
        self.model_manifest = MappingProxyType(manifest)
        self.model_manifest_sha256 = sha256_json(manifest)

    def encode_brain(self, brain_signals: Tensor) -> Tensor:
        return F.normalize(
            self.brain_mlp(brain_signals).float(), dim=-1
        )

    def encode_image(self, pixel_values: Tensor) -> Tensor:
        if (
            not isinstance(pixel_values, Tensor)
            or not pixel_values.is_floating_point()
        ):
            raise TypeError("pixel_values must be a floating torch.Tensor")
        outputs = self.vision_model(
            pixel_values=pixel_values,
            return_dict=True,
        )
        if hasattr(outputs, "image_embeds"):
            embeds = outputs.image_embeds
        elif isinstance(outputs, Mapping) and "image_embeds" in outputs:
            embeds = outputs["image_embeds"]
        elif isinstance(outputs, (tuple, list)) and outputs:
            embeds = outputs[0]
        else:
            raise ValueError("CLIP output does not expose image_embeds")
        if not isinstance(embeds, Tensor) or embeds.ndim != 2:
            raise ValueError(
                "CLIP image_embeds must be a two-dimensional tensor"
            )
        if embeds.shape[-1] != self.projection_dim:
            raise ValueError(
                "CLIP projection dimension differs from BrainMLP"
            )
        if not bool(torch.isfinite(embeds).all()):
            raise ValueError(
                "CLIP image embeddings contain non-finite values"
            )
        return F.normalize(embeds.float(), dim=-1)

    def forward(
        self,
        *,
        brain_signals: Tensor,
        pixel_values: Tensor,
        return_loss: bool = True,
    ) -> BrainRWOutput:
        brain = self.encode_brain(brain_signals)
        image = self.encode_image(pixel_values)
        if brain.shape[0] != image.shape[0]:
            raise ValueError("brain and image batch sizes differ")
        similarity = brain @ image.T
        loss = None
        if return_loss:
            logits = self.logit_scale.exp() * similarity
            labels = torch.arange(
                logits.shape[0], device=logits.device
            )
            loss = F.cross_entropy(logits, labels)
        return BrainRWOutput(loss, similarity, brain, image)

    def task_state_dict(self) -> dict[str, Tensor]:
        state = {
            f"brain_mlp.{name}": value.detach().cpu().clone()
            for name, value in self.brain_mlp.state_dict().items()
        }
        state["logit_scale"] = (
            self.logit_scale.detach().cpu().clone()
        )
        return state

    def candidate_state_dict(self) -> dict[str, Tensor]:
        return {
            name: value.detach().cpu().clone()
            for name, value in self.vision_model.state_dict().items()
            if ".lora_A" in name or ".lora_B" in name
        }

    def load_checkpoint_states(
        self,
        task_state: Mapping[str, Tensor],
        candidate_state: Mapping[str, Tensor],
    ) -> None:
        expected_task = set(self.task_state_dict())
        if set(task_state) != expected_task:
            raise ValueError(
                "checkpoint task-state keys differ from BrainMLP"
            )
        brain_state = {
            name.removeprefix("brain_mlp."): value
            for name, value in task_state.items()
            if name.startswith("brain_mlp.")
        }
        self.brain_mlp.load_state_dict(brain_state, strict=True)
        with torch.no_grad():
            self.logit_scale.copy_(
                task_state["logit_scale"].to(self.logit_scale)
            )
        current = self.candidate_state_dict()
        if set(candidate_state) != set(current):
            raise ValueError(
                "checkpoint candidate-state keys differ from resolved LoRA"
            )
        modules = dict(self.vision_model.named_parameters())
        with torch.no_grad():
            for name, value in candidate_state.items():
                parameter = modules.get(name)
                if parameter is None or parameter.shape != value.shape:
                    raise ValueError(
                        "checkpoint LoRA tensor shape mismatch"
                    )
                parameter.copy_(value.to(parameter))

    def brain_parameters(self) -> list[nn.Parameter]:
        return [*self.brain_mlp.parameters(), self.logit_scale]

    def lora_parameters(self) -> list[nn.Parameter]:
        return [
            parameter
            for name, parameter in self.vision_model.named_parameters()
            if (
                ".lora_A" in name or ".lora_B" in name
            )
            and parameter.requires_grad
        ]


def _load_rgb_image(path: Path) -> object:
    image_path = reject_development_path(path, "image path")
    if not image_path.is_file():
        raise ValueError("image path must identify a regular file")
    from PIL import Image

    with image_path.open("rb") as handle:
        image = Image.open(handle)
        image.load()
        return image.convert("RGB")


class BrainRWDevelopmentDataset(Dataset[dict[str, object]]):
    """Raw-image view over one verified train or val-dev protocol role."""

    def __init__(
        self,
        manifest_path: Path,
        scope: str,
        seed: int,
        *,
        image_loader: Callable[[Path], object] | None = None,
        expected_source_payload_sha256: str | None = None,
    ) -> None:
        if scope not in _DEVELOPMENT_SCOPES:
            raise PermissionError(
                "BrainRWDevelopmentDataset scope must be train or val-dev"
            )
        self._base = ProtocolSubjectDataset(
            Path(manifest_path),
            scope,
            seed,
            POSTERIOR_CHANNELS,
            None,
            0.3 if scope == "train" else 0.0,
            expected_source_payload_sha256=(
                expected_source_payload_sha256
            ),
        )
        self._image_loader = (
            _load_rgb_image if image_loader is None else image_loader
        )
        self.scope = scope
        for name in (
            "subject_id",
            "ordered_ids",
            "query_ids",
            "gallery_ids",
            "row_indices",
        ):
            setattr(self, name, getattr(self._base, name))

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, index: int) -> dict[str, object]:
        result = dict(self._base[index])
        if result.get("scope") != self.scope:
            raise PermissionError(
                "dataset item scope differs from verified role"
            )
        image_path = reject_development_path(
            Path(str(result["image_path"])), "image path"
        )
        result["image"] = self._image_loader(image_path)
        return result


class BrainRWCollator:
    def __init__(self, processor: object) -> None:
        if not callable(processor):
            raise TypeError("processor must be callable")
        self.processor = processor

    def __call__(
        self,
        examples: Sequence[Mapping[str, object]],
    ) -> dict[str, object]:
        if (
            isinstance(examples, (str, bytes, bytearray))
            or not examples
        ):
            raise ValueError("examples must be a non-empty sequence")
        scopes = {item.get("scope") for item in examples}
        if (
            len(scopes) != 1
            or next(iter(scopes)) not in _DEVELOPMENT_SCOPES
        ):
            raise PermissionError(
                "a batch must contain one development scope"
            )
        image_ids = tuple(str(item["image_id"]) for item in examples)
        if len(set(image_ids)) != len(image_ids):
            raise ValueError(
                "contrastive batches require unique image IDs"
            )
        eeg = (
            torch.stack([item["eeg"] for item in examples])
            .float()
            .contiguous()
        )
        if eeg.ndim != 3 or tuple(eeg.shape[1:]) != (17, 250):
            raise ValueError(
                "Brain-RW EEG batches must have shape [B,17,250]"
            )
        if any(
            tuple(item["eeg"].shape) != tuple(eeg.shape[1:])
            for item in examples
        ):
            raise ValueError("EEG batch shapes differ")
        processed = self.processor(
            images=[item["image"] for item in examples],
            return_tensors="pt",
        )
        pixel_values = (
            processed["pixel_values"]
            if isinstance(processed, Mapping)
            else processed.pixel_values
        )
        if not isinstance(pixel_values, Tensor):
            raise TypeError(
                "processor pixel_values must be a torch.Tensor"
            )
        return {
            "brain_signals": eeg,
            "pixel_values": pixel_values.float().contiguous(),
            "subject_ids": torch.tensor(
                [int(item["subject_id"]) for item in examples],
                dtype=torch.long,
            ),
            "image_ids": image_ids,
            "row_indices": tuple(
                int(item["row_index"]) for item in examples
            ),
            "scope": next(iter(scopes)),
        }


@dataclass(frozen=True)
class ManifestIdentity:
    path: Path
    subject: int
    manifest_sha256: str
    protocol_sha256: str
    records_sha256: str
    source_manifest_sha256: str
    source_payload_path: Path
    source_payload_sha256: str
    source_payload_byte_count: int
    train_role_sha256: str
    val_dev_role_sha256: str
    train_ordered_ids: tuple[str, ...]
    val_dev_ordered_ids: tuple[str, ...]
    train_ordered_ids_sha256: str
    val_dev_ordered_ids_sha256: str
    train_row_count: int = 12_540
    val_dev_row_count: int = 200


def load_development_manifest_identity(
    manifest_path: Path,
    *,
    expected_subject: int,
) -> ManifestIdentity:
    subject = _positive_int(expected_subject, "expected_subject")
    if subject > 10:
        raise ValueError(
            "expected_subject must be between 1 and 10"
        )
    path = reject_development_path(
        manifest_path, "protocol manifest"
    )
    match = _PROTOCOL_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(
            "protocol filename must be sub-XX_protocol.json"
        )
    if int(match.group(1)) != subject:
        raise ValueError(
            "protocol filename subject differs from CLI subject"
        )
    descriptors = [
        TypedArtifact(
            _ROLE_PAYLOAD_TYPE,
            path,
            path,
            role=role,
        )
        for role in ("train", "val-dev")
    ]
    train_capability = verify_typed_artifacts(
        "train", [descriptors[0]]
    )[0]
    val_capability = verify_typed_artifacts(
        "val-dev", [descriptors[1]]
    )[0]
    if (
        train_capability.payload_sha256
        != val_capability.payload_sha256
    ):
        raise ValueError(
            "train and val-dev capabilities bind different manifests"
        )
    with train_capability.open_verified() as handle:
        document = json.loads(handle.read().decode("utf-8"))
    if not isinstance(document, dict):
        raise ValueError("protocol manifest must be an object")
    if document.get("subject_id") not in {
        subject,
        f"sub-{subject:02d}",
    }:
        raise ValueError(
            "protocol manifest subject differs from CLI subject"
        )
    roles = document.get("role_payloads")
    role_descriptors = document.get("role_artifacts")
    if not isinstance(roles, Mapping) or not isinstance(
        role_descriptors, Mapping
    ):
        raise ValueError(
            "protocol manifest is missing role containers"
        )
    train = roles.get("train")
    val_dev = roles.get("val-dev")
    if not isinstance(train, Mapping) or not isinstance(
        val_dev, Mapping
    ):
        raise ValueError(
            "protocol manifest lacks development roles"
        )
    if (
        train.get("row_count"),
        train.get("concept_count"),
    ) != (12_540, 1_254):
        raise ValueError(
            "protocol train role has incorrect locked counts"
        )
    if (
        val_dev.get("row_count"),
        val_dev.get("concept_count"),
    ) != (200, 200):
        raise ValueError(
            "protocol val-dev role has incorrect locked counts"
        )
    train_ids = tuple(train.get("ordered_ids", ()))
    val_ids = tuple(val_dev.get("ordered_ids", ()))
    if any(
        not isinstance(item, str)
        for item in (*train_ids, *val_ids)
    ):
        raise ValueError(
            "protocol ordered IDs must be strings"
        )
    train_descriptor = role_descriptors.get("train")
    val_descriptor = role_descriptors.get("val-dev")
    if not isinstance(train_descriptor, Mapping) or not isinstance(
        val_descriptor, Mapping
    ):
        raise ValueError(
            "protocol role descriptors are missing"
        )
    source_manifest_sha256 = _sha256(
        document.get("source_manifest_sha256"),
        "source manifest hash",
    )
    declared_source = document.get("source_manifest_path")
    if not isinstance(declared_source, str) or not declared_source:
        raise ValueError(
            "protocol source_manifest_path must be a non-empty string"
        )
    declared_path = Path(declared_source)
    candidates = (
        reject_development_path(
            declared_path,
            "source train manifest",
        ),
        reject_development_path(
            path.parent / declared_path,
            "source train manifest",
        ),
    )
    existing_sources = tuple(
        dict.fromkeys(
            candidate
            for candidate in candidates
            if candidate.is_file()
        )
    )
    if len(existing_sources) != 1:
        raise ValueError(
            "protocol source manifest path is missing or ambiguous"
        )
    source_payload = inspect_source_payload_identity(
        existing_sources[0],
        expected_manifest_sha256=source_manifest_sha256,
        subject=subject,
    )
    from .provenance import DEFAULT_ORACLES

    source_oracle = DEFAULT_ORACLES.source_files[subject - 1]
    if (
        source_oracle.manifest_sha256 != source_manifest_sha256
        or source_oracle.byte_count != source_payload.byte_count
        or source_oracle.sha256 != source_payload.sha256
    ):
        raise ValueError(
            "source manifest/train.pt identity differs from pinned provenance"
        )
    return ManifestIdentity(
        path=path,
        subject=subject,
        manifest_sha256=train_capability.payload_sha256,
        protocol_sha256=_sha256(
            document.get("protocol_config_sha256"),
            "protocol hash",
        ),
        records_sha256=_sha256(
            document.get("records_sha256"), "records hash"
        ),
        source_manifest_sha256=source_manifest_sha256,
        source_payload_path=source_payload.path,
        source_payload_sha256=source_payload.sha256,
        source_payload_byte_count=source_payload.byte_count,
        train_role_sha256=_sha256(
            train_descriptor.get("role_payload_sha256"),
            "train role hash",
        ),
        val_dev_role_sha256=_sha256(
            val_descriptor.get("role_payload_sha256"),
            "val-dev role hash",
        ),
        train_ordered_ids=train_ids,
        val_dev_ordered_ids=val_ids,
        train_ordered_ids_sha256=ordered_ids_sha256(train_ids),
        val_dev_ordered_ids_sha256=ordered_ids_sha256(val_ids),
        train_row_count=int(train["row_count"]),
        val_dev_row_count=int(val_dev["row_count"]),
    )


@dataclass(frozen=True)
class BrainRWConfigIdentity:
    path: Path
    payload: Mapping[str, object]
    sha256: str
    clip_path: Path
    clip_config_sha256: str
    clip_preprocessor_sha256: str
    clip_weights_sha256: str


def verify_brainrw_config(
    config_path: Path,
    clip_path: Path,
) -> BrainRWConfigIdentity:
    path = reject_development_path(
        config_path, "brain-rw config"
    )
    semantic = SemanticConfig.from_path(path)
    payload = semantic.canonical_payload()
    if payload.get("config_type") != "brainrw_clip_lora":
        raise ValueError(
            "config must be a brainrw_clip_lora semantic config"
        )
    clip = payload["clip"]
    brain = payload["brain_mlp"]
    lora = payload["lora"]
    optimizer = payload["optimizer"]
    training = payload["training"]
    assert isinstance(clip, dict) and isinstance(brain, dict)
    assert isinstance(lora, dict) and isinstance(optimizer, dict)
    assert isinstance(training, dict)
    if lora != {
        "targets": list(CLIP_LORA_TARGETS),
        "rank": 32,
        "alpha": 32,
        "dropout": 0.0,
    }:
        raise ValueError(
            "brain-rw LoRA section differs from the locked recipe"
        )
    if brain != {"dropout": 0.1}:
        raise ValueError(
            "brain-rw BrainMLP section differs from the locked recipe"
        )
    if optimizer != {
        "name": "AdamW",
        "brain_learning_rate": 0.0005,
        "visual_learning_rate": 0.00005,
        "weight_decay": 0.05,
        "schedule": "cosine",
    }:
        raise ValueError(
            "brain-rw optimizer section differs from the locked recipe"
        )
    if training != {
        "epochs": 25,
        "epoch_policy": "fixed",
        "gradient_checkpointing": True,
        "precision": "bf16",
        "batch_size": 512,
        "trial_averaging": 4,
        "channels": list(POSTERIOR_CHANNELS),
    }:
        raise ValueError(
            "brain-rw training section differs from the locked recipe"
        )
    resolved_clip = reject_development_path(
        clip_path, "CLIP path"
    )
    declared = Path(
        os.path.abspath(os.path.normpath(str(clip["path"])))
    )
    if resolved_clip != declared:
        raise ValueError(
            "CLI clip-path differs from semantic config"
        )
    config_file = resolved_clip / "config.json"
    weights_file = resolved_clip / "model.safetensors"
    preprocessor_file = resolved_clip / "preprocessor_config.json"
    if (
        not config_file.is_file()
        or not weights_file.is_file()
        or not preprocessor_file.is_file()
    ):
        raise ValueError(
            "CLIP path lacks required local model files"
        )
    config_hash = file_sha256(config_file)
    weights_hash = file_sha256(weights_file)
    preprocessor_hash = file_sha256(preprocessor_file)
    if config_hash != _sha256(
        clip["config_sha256"], "CLIP config hash"
    ):
        raise ValueError("CLIP config SHA-256 mismatch")
    if weights_hash != _sha256(
        clip["weights_sha256"], "CLIP weights hash"
    ):
        raise ValueError("CLIP weights SHA-256 mismatch")
    return BrainRWConfigIdentity(
        path=path,
        payload=MappingProxyType(payload),
        sha256=semantic.sha256,
        clip_path=resolved_clip,
        clip_config_sha256=config_hash,
        clip_preprocessor_sha256=preprocessor_hash,
        clip_weights_sha256=weights_hash,
    )


def input_hashes(
    config: BrainRWConfigIdentity,
    manifest: ManifestIdentity,
    semantic_environment_sha256: str,
    validation_scope: str = "val-dev",
) -> dict[str, str]:
    if validation_scope not in {"val-dev", "none"}:
        raise ValueError("validation scope must be val-dev or none")
    hashes = {
        "clip_config": config.clip_config_sha256,
        "clip_preprocessor": config.clip_preprocessor_sha256,
        "clip_weights": config.clip_weights_sha256,
        "config": config.sha256,
        "manifest": manifest.manifest_sha256,
        "protocol": manifest.protocol_sha256,
        "records": manifest.records_sha256,
        "semantic_environment": _sha256(
            semantic_environment_sha256,
            "semantic environment hash",
        ),
        "source_manifest": manifest.source_manifest_sha256,
        "source_payload": manifest.source_payload_sha256,
        "train_role": manifest.train_role_sha256,
        "val_dev_role": manifest.val_dev_role_sha256,
    }
    if validation_scope == "none":
        hashes["validation_policy"] = sha256_json(
            {"validation_scope": "none"}
        )
    return hashes


def brainrw_run_key(
    config: BrainRWConfigIdentity,
    manifest: ManifestIdentity,
    subject: int,
    seed: int,
    semantic_environment_sha256: str,
    validation_scope: str = "val-dev",
) -> tuple[str, str, dict[str, str]]:
    hashes = input_hashes(
        config,
        manifest,
        semantic_environment_sha256,
        validation_scope,
    )
    bundle = sha256_json(hashes)
    key = make_run_key(
        "brainrw-clip-lora",
        str(config.payload["config_id"]),
        subject,
        seed,
        config.sha256,
        bundle,
    )
    return key, bundle, hashes


def load_clip_components(
    clip_path: Path,
    *,
    expected_config_sha256: str,
    expected_weights_sha256: str,
    expected_preprocessor_sha256: str,
) -> tuple[nn.Module, object]:
    path = reject_development_path(clip_path, "CLIP path")
    expected = {
        "config": _sha256(
            expected_config_sha256,
            "CLIP config hash",
        ),
        "weights": _sha256(
            expected_weights_sha256,
            "CLIP weights hash",
        ),
        "preprocessor": _sha256(
            expected_preprocessor_sha256,
            "CLIP preprocessor hash",
        ),
    }
    components = {
        "config": path / "config.json",
        "weights": path / "model.safetensors",
        "preprocessor": path / "preprocessor_config.json",
    }
    for name, component in components.items():
        if file_sha256(component) != expected[name]:
            raise ValueError(
                f"CLIP {name} SHA-256 mismatch before load"
            )
    from transformers import (
        CLIPImageProcessor,
        CLIPVisionModelWithProjection,
    )

    model = CLIPVisionModelWithProjection.from_pretrained(
        str(path),
        local_files_only=True,
        use_safetensors=True,
    )
    processor = CLIPImageProcessor.from_pretrained(
        str(path),
        local_files_only=True,
    )
    for name, component in components.items():
        if file_sha256(component) != expected[name]:
            raise ValueError(
                f"CLIP {name} SHA-256 mismatch after load"
            )
    return model, processor


def build_brainrw_model(
    config_payload: Mapping[str, object],
    clip_path: Path,
    *,
    expected_preprocessor_sha256: str,
) -> tuple[BrainRWCLIPLoRAModel, object]:
    training = config_payload.get("training")
    if (
        not isinstance(training, Mapping)
        or training.get("gradient_checkpointing") is not True
    ):
        raise ValueError(
            "Brain-RW requires locked vision gradient checkpointing"
        )
    clip = config_payload["clip"]
    if not isinstance(clip, Mapping):
        raise ValueError("CLIP config must be a mapping")
    preprocessor_sha256 = _sha256(
        expected_preprocessor_sha256,
        "CLIP preprocessor hash",
    )
    vision, processor = load_clip_components(
        clip_path,
        expected_config_sha256=_sha256(
            clip.get("config_sha256"), "CLIP config hash"
        ),
        expected_weights_sha256=_sha256(
            clip.get("weights_sha256"), "CLIP weights hash"
        ),
        expected_preprocessor_sha256=preprocessor_sha256,
    )
    lora = config_payload["lora"]
    brain = config_payload["brain_mlp"]
    assert isinstance(lora, Mapping) and isinstance(brain, Mapping)
    projection_dim = int(
        getattr(vision.config, "projection_dim")
    )
    model = BrainRWCLIPLoRAModel(
        vision,
        channels=17,
        samples=250,
        projection_dim=projection_dim,
        dropout=float(brain["dropout"]),
        lora_targets=tuple(lora["targets"]),
        lora_rank=int(lora["rank"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
    )
    return model, processor


class StatefulIndexSampler(Sampler[int]):
    def __init__(self, size: int, seed: int) -> None:
        self.size = _positive_int(size, "sampler size")
        self.seed = _nonnegative_int(seed, "sampler seed")
        self.epoch = 0
        self.position = 0
        self.order = self._order_for_epoch(0)

    def _order_for_epoch(self, epoch: int) -> tuple[int, ...]:
        generator = torch.Generator().manual_seed(
            self.seed + epoch
        )
        return tuple(
            torch.randperm(
                self.size, generator=generator
            ).tolist()
        )

    def __iter__(self) -> Iterator[int]:
        while self.position < self.size:
            value = self.order[self.position]
            self.position += 1
            yield value

    def __len__(self) -> int:
        return self.size - self.position

    def advance_epoch(self) -> None:
        if self.position != self.size:
            raise ValueError(
                "cannot advance a partially consumed sampler epoch"
            )
        self.epoch += 1
        self.position = 0
        self.order = self._order_for_epoch(self.epoch)

    def state_dict(self) -> dict[str, object]:
        return {
            "size": self.size,
            "seed": self.seed,
            "epoch": self.epoch,
            "position": self.position,
            "order": list(self.order),
        }

    def load_state_dict(
        self, state: Mapping[str, object]
    ) -> None:
        if set(state) != {
            "size",
            "seed",
            "epoch",
            "position",
            "order",
        }:
            raise ValueError(
                "sampler state has an unexpected schema"
            )
        if (
            state["size"] != self.size
            or state["seed"] != self.seed
        ):
            raise ValueError(
                "sampler size/seed differs from checkpoint"
            )
        epoch = _nonnegative_int(
            state["epoch"], "sampler epoch"
        )
        position = _nonnegative_int(
            state["position"], "sampler position"
        )
        order = tuple(state["order"])
        if (
            order != self._order_for_epoch(epoch)
            or position > self.size
        ):
            raise ValueError(
                "sampler order/position differs from deterministic contract"
            )
        self.epoch = epoch
        self.position = position
        self.order = order


def data_order_sha256(
    dataset: object,
    sampler: StatefulIndexSampler,
) -> str:
    rows = tuple(
        int(value) for value in getattr(dataset, "row_indices")
    )
    if len(rows) != sampler.size:
        raise ValueError(
            "dataset row count differs from sampler"
        )
    return ordered_ids_sha256(
        [str(rows[index]) for index in sampler.order]
    )


def capture_rng_state() -> dict[str, object]:
    python_version, python_values, python_gauss = (
        random.getstate()
    )
    (
        numpy_generator,
        numpy_keys,
        numpy_position,
        numpy_has_gauss,
        numpy_cached_gaussian,
    ) = np.random.get_state()
    return {
        "python": {
            "version": int(python_version),
            "state": torch.tensor(
                python_values,
                dtype=torch.int64,
            ),
            "gauss": (
                None
                if python_gauss is None
                else float(python_gauss)
            ),
        },
        "numpy": {
            "bit_generator": str(numpy_generator),
            "keys": torch.tensor(
                numpy_keys.astype(np.int64, copy=False),
                dtype=torch.int64,
            ),
            "position": int(numpy_position),
            "has_gauss": int(numpy_has_gauss),
            "cached_gaussian": float(
                numpy_cached_gaussian
            ),
        },
        "torch": torch.get_rng_state().cpu(),
        "cuda": (
            [
                value.detach().cpu().contiguous()
                for value in torch.cuda.get_rng_state_all()
            ]
            if torch.cuda.is_available()
            else []
        ),
    }


def restore_rng_state(state: Mapping[str, object]) -> None:
    if (
        not isinstance(state, Mapping)
        or set(state) != {"python", "numpy", "torch", "cuda"}
    ):
        raise ValueError(
            "RNG state has an unexpected schema"
        )
    python_state = state["python"]
    numpy_state = state["numpy"]
    if (
        not isinstance(python_state, Mapping)
        or set(python_state)
        != {"version", "state", "gauss"}
        or not isinstance(numpy_state, Mapping)
        or set(numpy_state)
        != {
            "bit_generator",
            "keys",
            "position",
            "has_gauss",
            "cached_gaussian",
        }
    ):
        raise ValueError("RNG state has an invalid nested schema")
    python_version = python_state["version"]
    python_values = python_state["state"]
    python_gauss = python_state["gauss"]
    if (
        type(python_version) is not int
        or python_version != 3
        or not isinstance(python_values, Tensor)
        or python_values.dtype != torch.int64
        or python_values.ndim != 1
        or python_values.numel() == 0
        or bool((python_values < 0).any().item())
        or bool((python_values > 2**32 - 1).any().item())
        or (
            python_gauss is not None
            and (
                isinstance(python_gauss, bool)
                or not isinstance(python_gauss, (int, float))
                or not math.isfinite(float(python_gauss))
            )
        )
    ):
        raise ValueError("Python RNG state is invalid")
    numpy_generator = numpy_state["bit_generator"]
    numpy_keys = numpy_state["keys"]
    numpy_position = numpy_state["position"]
    numpy_has_gauss = numpy_state["has_gauss"]
    numpy_cached = numpy_state["cached_gaussian"]
    if (
        numpy_generator != "MT19937"
        or not isinstance(numpy_keys, Tensor)
        or numpy_keys.dtype != torch.int64
        or numpy_keys.ndim != 1
        or numpy_keys.numel() != 624
        or bool((numpy_keys < 0).any().item())
        or bool((numpy_keys > 2**32 - 1).any().item())
        or type(numpy_position) is not int
        or not 0 <= numpy_position <= 624
        or type(numpy_has_gauss) is not int
        or numpy_has_gauss not in {0, 1}
        or isinstance(numpy_cached, bool)
        or not isinstance(numpy_cached, (int, float))
        or not math.isfinite(float(numpy_cached))
    ):
        raise ValueError("NumPy RNG state is invalid")
    torch_state = state["torch"]
    cuda_state = state["cuda"]
    if (
        not isinstance(torch_state, Tensor)
        or torch_state.dtype != torch.uint8
        or torch_state.ndim != 1
        or torch_state.numel() == 0
        or not isinstance(cuda_state, (list, tuple))
        or any(
            not isinstance(value, Tensor)
            or value.dtype != torch.uint8
            or value.ndim != 1
            or value.numel() == 0
            for value in cuda_state
        )
    ):
        raise ValueError("Torch RNG state is invalid")

    random.setstate(
        (
            python_version,
            tuple(int(value) for value in python_values.tolist()),
            (
                None
                if python_gauss is None
                else float(python_gauss)
            ),
        )
    )
    np.random.set_state(
        (
            numpy_generator,
            np.asarray(
                numpy_keys.tolist(),
                dtype=np.uint32,
            ),
            numpy_position,
            numpy_has_gauss,
            float(numpy_cached),
        )
    )
    torch.set_rng_state(
        torch_state.detach().cpu().contiguous()
    )
    if cuda_state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [
                value.detach().cpu().contiguous()
                for value in cuda_state
            ]
        )


@dataclass(frozen=True)
class LoadedBrainRWCheckpoint:
    payload: Mapping[str, object]
    sha256: str


def checkpoint_sidecar(path: Path) -> Path:
    value = Path(path)
    return value.with_suffix(value.suffix + ".meta.json")


def _node_identity(
    value: os.stat_result,
) -> tuple[int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        stat.S_IFMT(value.st_mode),
    )


@dataclass
class _SecureParent:
    path: Path
    leaf: str
    parent_fd: int
    descriptors: list[int]
    edges: list[tuple[int, str, tuple[int, int, int]]]

    def verify(self) -> None:
        for parent_fd, component, expected in self.edges:
            try:
                current = os.stat(
                    component,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise ValueError(
                    "development path changed during secure traversal"
                ) from exc
            if (
                stat.S_ISLNK(current.st_mode)
                or _node_identity(current) != expected
            ):
                raise ValueError(
                    "development path changed during secure traversal"
                )


def _open_directory_at(
    parent_fd: int,
    component: str,
) -> int:
    try:
        descriptor = os.open(
            component,
            os.O_RDONLY
            | _O_CLOEXEC
            | _O_NOFOLLOW
            | _O_DIRECTORY,
            dir_fd=parent_fd,
        )
    except OSError as exc:
        raise ValueError(
            "development path contains an unsafe directory"
        ) from exc
    value = os.fstat(descriptor)
    if not stat.S_ISDIR(value.st_mode):
        os.close(descriptor)
        raise ValueError(
            "development path component must be a directory"
        )
    return descriptor


@contextmanager
def _secure_parent_directory(
    path: Path,
    *,
    context: str,
) -> Iterator[_SecureParent]:
    normalized = reject_development_path(path, context)
    parts = normalized.parts
    if len(parts) < 2 or not normalized.name:
        raise ValueError(f"{context} must name a path leaf")
    descriptors: list[int] = []
    edges: list[
        tuple[int, str, tuple[int, int, int]]
    ] = []
    try:
        root_fd = os.open(
            normalized.anchor,
            os.O_RDONLY | _O_CLOEXEC | _O_DIRECTORY,
        )
        descriptors.append(root_fd)
        current_fd = root_fd
        for component in parts[1:-1]:
            child_fd = _open_directory_at(
                current_fd,
                component,
            )
            child_stat = os.fstat(child_fd)
            edges.append(
                (
                    current_fd,
                    component,
                    _node_identity(child_stat),
                )
            )
            descriptors.append(child_fd)
            current_fd = child_fd
        secured = _SecureParent(
            path=normalized,
            leaf=parts[-1],
            parent_fd=current_fd,
            descriptors=descriptors,
            edges=edges,
        )
        secured.verify()
        yield secured
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _unlink_created_file(
    parent: _SecureParent,
    leaf: str,
    identity: tuple[int, int, int],
) -> None:
    try:
        current = os.stat(
            leaf,
            dir_fd=parent.parent_fd,
            follow_symlinks=False,
        )
    except OSError:
        return
    if _node_identity(current) != identity:
        return
    try:
        os.unlink(leaf, dir_fd=parent.parent_fd)
    except OSError:
        pass


def _write_relative_exclusive(
    parent: _SecureParent,
    leaf: str,
    payload: bytes,
    *,
    context: str,
) -> tuple[int, int, int]:
    if (
        not isinstance(leaf, str)
        or not leaf
        or leaf in {".", ".."}
        or "/" in leaf
        or "\x00" in leaf
    ):
        raise ValueError(f"{context} has an invalid file name")
    if not isinstance(payload, bytes):
        raise TypeError(f"{context} payload must be bytes")
    parent.verify()
    descriptor = -1
    created_identity: tuple[int, int, int] | None = None
    try:
        descriptor = os.open(
            leaf,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | _O_CLOEXEC
            | _O_NOFOLLOW,
            0o600,
            dir_fd=parent.parent_fd,
        )
        created = os.fstat(descriptor)
        created_identity = _node_identity(created)
        if not stat.S_ISREG(created.st_mode):
            raise ValueError(
                f"{context} must be a regular file"
            )
        parent.verify()
        offset = 0
        while offset < len(payload):
            written = os.write(
                descriptor,
                payload[offset:],
            )
            if written <= 0:
                raise OSError(f"short {context} write")
            offset += written
        os.fsync(descriptor)
        named = os.stat(
            leaf,
            dir_fd=parent.parent_fd,
            follow_symlinks=False,
        )
        if _node_identity(named) != created_identity:
            raise ValueError(
                f"{context} path changed during write"
            )
        parent.verify()
        os.fsync(parent.parent_fd)
        return created_identity
    except FileExistsError as exc:
        raise FileExistsError(
            f"{context} already exists"
        ) from exc
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
            descriptor = -1
        if created_identity is not None:
            _unlink_created_file(
                parent,
                leaf,
                created_identity,
            )
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def create_development_directory_exclusive(
    path: Path,
    *,
    context: str,
) -> Path:
    output = reject_development_path(path, context)
    with _secure_parent_directory(
        output,
        context=context,
    ) as parent:
        parent.verify()
        try:
            os.mkdir(
                parent.leaf,
                mode=0o700,
                dir_fd=parent.parent_fd,
            )
        except FileExistsError as exc:
            raise FileExistsError(
                f"{context} already exists"
            ) from exc
        except OSError as exc:
            raise ValueError(
                f"{context} cannot be created safely"
            ) from exc
        descriptor = -1
        created_identity: tuple[int, int, int] | None = None
        try:
            descriptor = _open_directory_at(
                parent.parent_fd,
                parent.leaf,
            )
            created = os.fstat(descriptor)
            created_identity = _node_identity(created)
            named = os.stat(
                parent.leaf,
                dir_fd=parent.parent_fd,
                follow_symlinks=False,
            )
            if _node_identity(named) != created_identity:
                raise ValueError(
                    f"{context} path changed during creation"
                )
            parent.verify()
            os.fsync(descriptor)
            os.fsync(parent.parent_fd)
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
                descriptor = -1
            if created_identity is not None:
                try:
                    current = os.stat(
                        parent.leaf,
                        dir_fd=parent.parent_fd,
                        follow_symlinks=False,
                    )
                    if (
                        _node_identity(current)
                        == created_identity
                    ):
                        os.rmdir(
                            parent.leaf,
                            dir_fd=parent.parent_fd,
                        )
                except OSError:
                    pass
            raise
        finally:
            if descriptor >= 0:
                os.close(descriptor)
    return output


def write_development_file_exclusive(
    path: Path,
    payload: bytes,
    *,
    context: str,
) -> Path:
    output = reject_development_path(path, context)
    with _secure_parent_directory(
        output,
        context=context,
    ) as parent:
        _write_relative_exclusive(
            parent,
            parent.leaf,
            payload,
            context=context,
        )
    return output


def _write_exclusive(path: Path, payload: bytes) -> None:
    write_development_file_exclusive(
        path,
        payload,
        context="development file output",
    )


def save_brainrw_checkpoint(
    path: Path,
    payload: Mapping[str, object],
    manifest: ManifestIdentity,
) -> str:
    payload = _validate_loaded_checkpoint(payload)
    checkpoint_path = reject_development_path(
        path, "checkpoint output"
    )
    sidecar = reject_development_path(
        checkpoint_sidecar(checkpoint_path),
        "checkpoint sidecar output",
    )
    if sidecar.parent != checkpoint_path.parent:
        raise ValueError(
            "checkpoint and sidecar must share one secure parent"
        )
    if payload.get("payload_type") != BRAINRW_CHECKPOINT_TYPE:
        raise ValueError(
            "checkpoint payload_type mismatch"
        )
    if payload.get("scope") != "train":
        raise PermissionError(
            "Brain-RW checkpoints must bind the train scope"
        )
    observation_policy = (
        payload.get("validation_scope"),
        payload.get("observed_scopes"),
    )
    if observation_policy not in (
        ("val-dev", ["train", "val-dev"]),
        ("none", ["train"]),
    ):
        raise PermissionError(
            "checkpoint validation and observed scopes are inconsistent"
        )
    buffer = io.BytesIO()
    torch.save(dict(payload), buffer)
    raw = buffer.getvalue()
    payload_hash = hashlib.sha256(raw).hexdigest()
    source_records = [
        {
            "manifest_sha256": manifest.manifest_sha256,
            "records_sha256": manifest.records_sha256,
            "role": "train",
            "role_payload_sha256": manifest.train_role_sha256,
            "source_manifest_sha256": (
                manifest.source_manifest_sha256
            ),
            "source_payload_byte_count": manifest.source_payload_byte_count,
            "source_payload_path": str(manifest.source_payload_path),
            "source_payload_sha256": manifest.source_payload_sha256,
        }
    ]
    provenance = {
        "config_sha256": payload["config_sha256"],
        "git_sha": payload["git_sha"],
        "manifest_sha256": manifest.manifest_sha256,
        "protocol_sha256": manifest.protocol_sha256,
        "run_key": payload["run_key"],
        "seed": payload["seed"],
        "subject": payload["subject"],
    }
    metadata = {
        "complete": True,
        "global_step": payload["global_step"],
        "planned_steps": payload["planned_steps"],
        "training_complete": payload["training_complete"],
        "observed_scopes": list(payload["observed_scopes"]),
        "ordered_ids": list(manifest.train_ordered_ids),
        "run_key": payload["run_key"],
        "source_records": source_records,
    }
    envelope = {
        "schema_version": 1,
        "payload_type": BRAINRW_CHECKPOINT_TYPE,
        "scope": "train",
        "source_records_sha256": sha256_json(
            source_records
        ),
        "ordered_ids_sha256": ordered_ids_sha256(
            manifest.train_ordered_ids
        ),
        "payload_sha256": payload_hash,
        "provenance": provenance,
        "provenance_sha256": sha256_json(provenance),
        "metadata": metadata,
        "metadata_sha256": sha256_json(metadata),
    }
    with _secure_parent_directory(
        checkpoint_path,
        context="checkpoint output",
    ) as parent:
        if sidecar.name == parent.leaf:
            raise ValueError(
                "checkpoint and sidecar names must differ"
            )
        parent.verify()
        for leaf in (parent.leaf, sidecar.name):
            try:
                os.stat(
                    leaf,
                    dir_fd=parent.parent_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise ValueError(
                    "checkpoint output cannot be inspected safely"
                ) from exc
            raise FileExistsError(
                "checkpoint or checkpoint sidecar already exists"
            )
        checkpoint_identity = _write_relative_exclusive(
            parent,
            parent.leaf,
            raw,
            context="checkpoint output",
        )
        try:
            _write_relative_exclusive(
                parent,
                sidecar.name,
                canonical_json_bytes(envelope) + b"\n",
                context="checkpoint sidecar output",
            )
        except BaseException:
            _unlink_created_file(
                parent,
                parent.leaf,
                checkpoint_identity,
            )
            raise
    return payload_hash


def _checkpoint_json_sha256(value: object, name: str) -> str:
    try:
        return sha256_json(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"checkpoint {name} is not canonical JSON"
        ) from exc


def _checkpoint_mapping(
    value: object,
    name: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(
            f"checkpoint {name} must be a mapping"
        )
    result = dict(value)
    if any(not isinstance(key, str) for key in result):
        raise ValueError(
            f"checkpoint {name} keys must be strings"
        )
    return result


def _validate_checkpoint_model_manifest(
    value: object,
    declared_sha256: object,
    target_sha256: object,
) -> dict[str, object]:
    manifest = _checkpoint_mapping(value, "model_manifest")
    if set(manifest) != {
        "schema_version",
        "brain_mlp",
        "gradient_checkpointing",
        "lora",
    }:
        raise ValueError(
            "checkpoint model_manifest has an unexpected schema"
        )
    if manifest["schema_version"] != 1:
        raise ValueError(
            "checkpoint model_manifest schema mismatch"
        )
    if (
        _checkpoint_json_sha256(
            manifest,
            "model_manifest",
        )
        != _sha256(
            declared_sha256,
            "checkpoint model_manifest_sha256",
        )
    ):
        raise ValueError(
            "checkpoint model_manifest hash mismatch"
        )
    brain = _checkpoint_mapping(
        manifest["brain_mlp"],
        "model_manifest.brain_mlp",
    )
    gradient_checkpointing = _checkpoint_mapping(
        manifest["gradient_checkpointing"],
        "model_manifest.gradient_checkpointing",
    )
    lora = _checkpoint_mapping(
        manifest["lora"],
        "model_manifest.lora",
    )
    if (
        set(gradient_checkpointing)
        != {
            "enabled",
            "method",
            "requested",
            "target",
            "use_reentrant",
            "wrapped_layer_count",
        }
        or gradient_checkpointing["enabled"] is not True
        or gradient_checkpointing["method"]
        != "explicit_per_layer_torch_utils_checkpoint"
        or gradient_checkpointing["requested"] is not True
        or gradient_checkpointing["target"]
        != "vision_model.encoder.layers"
        or gradient_checkpointing["use_reentrant"] is not False
    ):
        raise ValueError(
            "checkpoint gradient-checkpointing evidence is invalid"
        )
    wrapped_layer_count = _positive_int(
        gradient_checkpointing["wrapped_layer_count"],
        "checkpoint wrapped CLIP layer count",
    )
    if set(brain) != {
        "channels",
        "samples",
        "projection_dim",
        "layers",
        "dropout",
        "layer_norm_eps",
    } or set(lora) != {
        "semantic_targets",
        "resolved_targets",
        "target_manifest_sha256",
        "rank",
        "alpha",
        "dropout",
    }:
        raise ValueError(
            "checkpoint model_manifest has an unexpected nested schema"
        )
    target = _sha256(
        target_sha256,
        "checkpoint target_manifest_sha256",
    )
    if (
        _sha256(
            lora["target_manifest_sha256"],
            "checkpoint model manifest target hash",
        )
        != target
    ):
        raise ValueError(
            "checkpoint target manifest hash mismatch"
        )
    _positive_int(
        brain["channels"],
        "checkpoint model channels",
    )
    _positive_int(
        brain["samples"],
        "checkpoint model samples",
    )
    _positive_int(
        brain["projection_dim"],
        "checkpoint model projection dimension",
    )
    _positive_int(
        brain["layers"],
        "checkpoint model layer count",
    )
    _probability(
        brain["dropout"],
        "checkpoint model dropout",
    )
    if (
        isinstance(brain["layer_norm_eps"], bool)
        or not isinstance(
            brain["layer_norm_eps"],
            (int, float),
        )
        or not math.isfinite(float(brain["layer_norm_eps"]))
        or float(brain["layer_norm_eps"]) <= 0.0
    ):
        raise ValueError(
            "checkpoint model layer_norm_eps is invalid"
        )
    if not isinstance(lora["semantic_targets"], list) or not isinstance(
        lora["resolved_targets"],
        list,
    ) or any(
        not isinstance(item, str)
        for item in (
            list(lora["semantic_targets"])
            + list(lora["resolved_targets"])
        )
    ):
        raise ValueError(
            "checkpoint model LoRA targets are invalid"
        )
    resolved_targets = list(lora["resolved_targets"])
    target_counts = {
        target: sum(
            name.rsplit(".", 1)[-1] == target
            for name in resolved_targets
        )
        for target in CLIP_LORA_TARGETS
    }
    if (
        target_counts["visual_projection"] != 1
        or any(
            target_counts[target] != wrapped_layer_count
            for target in CLIP_LORA_TARGETS[:-1]
        )
        or len(resolved_targets) != wrapped_layer_count * 6 + 1
    ):
        raise ValueError(
            "checkpoint gradient-checkpoint layer count differs from LoRA targets"
        )
    _positive_int(lora["rank"], "checkpoint LoRA rank")
    _positive_int(lora["alpha"], "checkpoint LoRA alpha")
    _probability(lora["dropout"], "checkpoint LoRA dropout")
    return manifest


def _validate_checkpoint_resume_state(
    value: dict[str, object],
) -> None:
    for name in ("task_state", "candidate_state"):
        state = _checkpoint_mapping(value[name], name)
        if not state:
            raise ValueError(
                f"checkpoint {name} must not be empty"
            )
        try:
            state_dict_sha256(state)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"checkpoint {name} is invalid"
            ) from exc
    _checkpoint_mapping(
        value["optimizer_state"],
        "optimizer_state",
    )
    _checkpoint_mapping(
        value["scheduler_state"],
        "scheduler_state",
    )
    epoch = _nonnegative_int(
        value["epoch"],
        "checkpoint epoch",
    )
    global_step = _positive_int(
        value["global_step"],
        "checkpoint global_step",
    )
    planned_steps = _positive_int(
        value["planned_steps"],
        "checkpoint planned_steps",
    )
    if (
        value["steps"] != global_step
        or global_step > planned_steps
    ):
        raise ValueError(
            "checkpoint optimization step identity mismatch"
        )
    effective_batch_size = _positive_int(
        value["effective_batch_size"],
        "checkpoint effective_batch_size",
    )
    rng_state = _checkpoint_mapping(
        value["rng_state"],
        "rng_state",
    )
    if set(rng_state) != {
        "python",
        "numpy",
        "torch",
        "cuda",
    }:
        raise ValueError(
            "checkpoint RNG state has an unexpected schema"
        )
    sampler = _checkpoint_mapping(
        value["sampler_state"],
        "sampler_state",
    )
    if set(sampler) != {
        "size",
        "seed",
        "epoch",
        "position",
        "order",
    }:
        raise ValueError(
            "checkpoint sampler state has an unexpected schema"
        )
    sampler_size = _positive_int(
        sampler["size"],
        "checkpoint sampler size",
    )
    sampler_epoch = _nonnegative_int(
        sampler["epoch"],
        "checkpoint sampler epoch",
    )
    sampler_position = _nonnegative_int(
        sampler["position"],
        "checkpoint sampler position",
    )
    if (
        sampler["seed"] != value["seed"]
        or sampler_epoch != epoch
        or sampler_position > sampler_size
        or not isinstance(sampler["order"], list)
        or sorted(sampler["order"]) != list(range(sampler_size))
    ):
        raise ValueError(
            "checkpoint sampler state identity mismatch"
        )
    config_payload = _checkpoint_mapping(
        value["config_payload"],
        "config_payload",
    )
    training = _checkpoint_mapping(
        config_payload.get("training"),
        "config_payload.training",
    )
    epochs = _positive_int(
        training.get("epochs"),
        "checkpoint configured epochs",
    )
    configured_batch_size = _positive_int(
        training.get("batch_size"),
        "checkpoint configured batch_size",
    )
    if effective_batch_size != configured_batch_size:
        raise ValueError(
            "checkpoint effective batch size differs from config"
        )
    batches_per_epoch = math.ceil(
        sampler_size / effective_batch_size
    )
    if planned_steps != epochs * batches_per_epoch:
        raise ValueError(
            "checkpoint planned steps differ from config and sampler"
        )
    if (
        sampler_epoch >= epochs
        or (
            sampler_position != sampler_size
            and sampler_position % effective_batch_size != 0
        )
    ):
        raise ValueError(
            "checkpoint sampler progress is not on a batch boundary"
        )
    expected_global_step = (
        sampler_epoch * batches_per_epoch
        + math.ceil(
            sampler_position / effective_batch_size
        )
    )
    if global_step != expected_global_step:
        raise ValueError(
            "checkpoint global step differs from sampler progress"
        )
    training_complete = value["training_complete"]
    expected_complete = global_step == planned_steps
    if (
        type(training_complete) is not bool
        or training_complete is not expected_complete
    ):
        raise ValueError(
            "checkpoint training_complete differs from step progress"
        )
    if training_complete and (
        sampler_epoch != epochs - 1
        or sampler_position != sampler_size
    ):
        raise ValueError(
            "checkpoint terminal sampler state is inconsistent"
        )
    generator_state = value["dataloader_generator_state"]
    if (
        not isinstance(generator_state, Tensor)
        or generator_state.dtype != torch.uint8
        or generator_state.ndim != 1
        or generator_state.numel() == 0
    ):
        raise ValueError(
            "checkpoint dataloader generator state is invalid"
        )


def _validate_checkpoint_metrics(value: object) -> None:
    metrics = _checkpoint_mapping(
        value,
        "validation_metrics",
    )
    if metrics == {
        "performed": False,
        "validation_scope": "none",
    }:
        return
    if set(metrics) != {
        "gallery_count",
        "query_count",
        "top1_count",
        "top1_rate",
        "top5_count",
        "top5_rate",
    }:
        raise ValueError(
            "checkpoint validation metrics have an unexpected schema"
        )
    query_count = _positive_int(
        metrics["query_count"],
        "checkpoint validation query_count",
    )
    _positive_int(
        metrics["gallery_count"],
        "checkpoint validation gallery_count",
    )
    top1_count = _nonnegative_int(
        metrics["top1_count"],
        "checkpoint validation top1_count",
    )
    top5_count = _nonnegative_int(
        metrics["top5_count"],
        "checkpoint validation top5_count",
    )
    if (
        top1_count > top5_count
        or top5_count > query_count
    ):
        raise ValueError(
            "checkpoint validation count identity mismatch"
        )
    for name, count in (
        ("top1_rate", top1_count),
        ("top5_rate", top5_count),
    ):
        rate = metrics[name]
        if (
            isinstance(rate, bool)
            or not isinstance(rate, (int, float))
            or not math.isfinite(float(rate))
            or not math.isclose(
                float(rate),
                count / query_count,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                f"checkpoint validation {name} mismatch"
            )


def _validate_checkpoint_runtime_attestation(
    value: Mapping[str, object],
) -> None:
    semantic_environment = require_pinned_semantic_environment(
        value["semantic_environment"]
    )
    semantic_environment_sha256 = _sha256(
        value["semantic_environment_sha256"],
        "checkpoint semantic_environment_sha256",
    )
    if (
        sha256_json(semantic_environment)
        != semantic_environment_sha256
    ):
        raise ValueError(
            "checkpoint semantic environment hash mismatch"
        )
    environment = _checkpoint_mapping(
        value["environment"],
        "environment",
    )
    if environment != semantic_environment:
        raise ValueError(
            "checkpoint environment differs from semantic environment"
        )
    contract = _checkpoint_mapping(
        value["runtime_contract"],
        "runtime_contract",
    )
    if contract != dict(_BRAINRW_RUNTIME_CONTRACT):
        raise ValueError(
            "checkpoint runtime contract is not CUDA+bfloat16+A40"
        )
    if _checkpoint_json_sha256(
        contract,
        "runtime_contract",
    ) != _sha256(
        value["runtime_contract_sha256"],
        "checkpoint runtime_contract_sha256",
    ):
        raise ValueError("checkpoint runtime contract hash mismatch")

    evidence = _checkpoint_mapping(
        value["runtime_evidence"],
        "runtime_evidence",
    )
    if set(evidence) != _BRAINRW_RUNTIME_EVIDENCE_KEYS:
        raise ValueError(
            "checkpoint runtime evidence has an unexpected schema"
        )
    if _checkpoint_json_sha256(
        evidence,
        "runtime_evidence",
    ) != _sha256(
        value["runtime_evidence_sha256"],
        "checkpoint runtime_evidence_sha256",
    ):
        raise ValueError("checkpoint runtime evidence hash mismatch")
    if (
        evidence["schema_version"] != 1
        or evidence["cuda_available"] is not True
        or evidence["bf16_supported"] is not True
        or evidence["accelerator_name"] != contract["accelerator"]
        or evidence["device_type"] != contract["device_type"]
        or evidence["dtype"] != contract["dtype"]
    ):
        raise ValueError(
            "checkpoint runtime evidence differs from the contract"
        )
    capability = evidence["cuda_capability"]
    if (
        not isinstance(capability, list)
        or capability != [8, 6]
    ):
        raise ValueError(
            "checkpoint runtime evidence is not an A40 capability"
        )
    device_count = _positive_int(
        evidence["cuda_device_count"],
        "checkpoint CUDA device count",
    )
    device_index = _nonnegative_int(
        evidence["cuda_device_index"],
        "checkpoint CUDA device index",
    )
    if device_index >= device_count:
        raise ValueError(
            "checkpoint CUDA device index is out of range"
        )
    _positive_int(
        evidence["total_memory_bytes"],
        "checkpoint CUDA total memory",
    )
    for key in ("cuda_version", "torch_version"):
        if not isinstance(evidence[key], str) or not evidence[key]:
            raise ValueError(
                f"checkpoint runtime evidence {key} is invalid"
            )
    if value["runtime_dtype"] != contract["dtype"]:
        raise ValueError(
            "checkpoint runtime dtype differs from runtime contract"
        )


def _validate_loaded_checkpoint(
    payload: object,
) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise ValueError(
            "checkpoint payload must be a mapping"
        )
    value = dict(payload)
    missing = sorted(
        _BRAINRW_CHECKPOINT_KEYS - set(value)
    )
    if missing:
        raise ValueError(
            f"checkpoint is missing {missing[0]}"
        )
    unknown = sorted(
        set(value) - _BRAINRW_CHECKPOINT_KEYS
    )
    if unknown:
        raise ValueError(
            f"checkpoint has unknown field {unknown[0]}"
        )
    if (
        value["schema_version"] != 1
        or value["payload_type"] != BRAINRW_CHECKPOINT_TYPE
    ):
        raise ValueError(
            "checkpoint identity/schema mismatch"
        )
    if value["complete"] is not True:
        raise ValueError("checkpoint is incomplete")
    if value["scope"] != "train":
        raise PermissionError(
            "checkpoint contains a non-development scope"
        )
    observation_policy = (
        value["validation_scope"],
        value["observed_scopes"],
    )
    if observation_policy not in (
        ("val-dev", ["train", "val-dev"]),
        ("none", ["train"]),
    ):
        raise PermissionError(
            "checkpoint validation and observed scopes are inconsistent"
        )
    subject = _positive_int(
        value["subject"], "checkpoint subject"
    )
    if subject > 10:
        raise ValueError(
            "checkpoint subject must be between 1 and 10"
        )
    _nonnegative_int(value["seed"], "checkpoint seed")
    for key in (
        "candidate_initialization_sha256",
        "clip_config_sha256",
        "clip_preprocessor_sha256",
        "clip_weights_sha256",
        "config_sha256",
        "data_order_sha256",
        "input_bundle_sha256",
        "manifest_sha256",
        "model_manifest_sha256",
        "protocol_sha256",
        "target_manifest_sha256",
        "task_initialization_sha256",
    ):
        _sha256(value[key], f"checkpoint {key}")
    for key in (
        "config_path",
        "manifest_path",
        "clip_path",
    ):
        path = value[key]
        if not isinstance(path, str) or not path:
            raise ValueError(
                f"checkpoint {key} must be a non-empty path"
            )
        reject_development_path(
            Path(path),
            f"checkpoint {key}",
        )
    if (
        not isinstance(value["git_sha"], str)
        or _GIT_RE.fullmatch(value["git_sha"]) is None
    ):
        raise ValueError("checkpoint git_sha is invalid")
    config_payload = _checkpoint_mapping(
        value["config_payload"],
        "config_payload",
    )
    if (
        _checkpoint_json_sha256(
            config_payload,
            "config_payload",
        )
        != value["config_sha256"]
    ):
        raise ValueError(
            "checkpoint config payload hash mismatch"
        )
    config_id = config_payload.get("config_id")
    if not isinstance(config_id, str):
        raise ValueError(
            "checkpoint config payload lacks config_id"
        )
    hashes = _checkpoint_mapping(
        value["input_hashes"],
        "input_hashes",
    )
    expected_hash_keys = (
        _BRAINRW_INPUT_HASH_KEYS
        if value["validation_scope"] == "val-dev"
        else _BRAINRW_TRAIN_ONLY_INPUT_HASH_KEYS
    )
    if set(hashes) != expected_hash_keys:
        raise ValueError(
            "checkpoint input hashes have an unexpected schema"
        )
    for name, digest in hashes.items():
        _sha256(
            digest,
            f"checkpoint input hash {name}",
        )
    expected_bundle = _checkpoint_json_sha256(
        hashes,
        "input_hashes",
    )
    if value["input_bundle_sha256"] != expected_bundle:
        raise ValueError(
            "checkpoint input bundle hash mismatch"
        )
    if (
        value["validation_scope"] == "none"
        and hashes["validation_policy"]
        != sha256_json({"validation_scope": "none"})
    ):
        raise ValueError(
            "checkpoint validation policy hash mismatch"
        )
    expected_run_key = make_run_key(
        "brainrw-clip-lora",
        config_id,
        subject,
        int(value["seed"]),
        str(value["config_sha256"]),
        expected_bundle,
    )
    if value["run_key"] != expected_run_key:
        raise ValueError(
            "checkpoint run key identity mismatch"
        )
    if (
        hashes["config"] != value["config_sha256"]
        or hashes["manifest"] != value["manifest_sha256"]
        or hashes["protocol"] != value["protocol_sha256"]
        or hashes["clip_config"] != value["clip_config_sha256"]
        or hashes["clip_preprocessor"]
        != value["clip_preprocessor_sha256"]
        or hashes["clip_weights"] != value["clip_weights_sha256"]
        or hashes["semantic_environment"]
        != value["semantic_environment_sha256"]
    ):
        raise ValueError(
            "checkpoint duplicated input identity mismatch"
        )
    _validate_checkpoint_model_manifest(
        value["model_manifest"],
        value["model_manifest_sha256"],
        value["target_manifest_sha256"],
    )
    _validate_checkpoint_resume_state(value)
    _validate_checkpoint_runtime_attestation(value)
    provenance = _checkpoint_mapping(
        value["git_provenance"],
        "git_provenance",
    )
    if (
        set(provenance)
        != {
            "clean",
            "git_sha",
            "repository_root",
        }
        or provenance["clean"] is not True
        or provenance["git_sha"] != value["git_sha"]
        or not isinstance(
            provenance["repository_root"],
            str,
        )
        or not Path(
            provenance["repository_root"]
        ).is_absolute()
    ):
        raise ValueError(
            "checkpoint Git provenance is invalid"
        )
    _validate_checkpoint_metrics(
        value["validation_metrics"]
    )
    resumed = value["resumed_from_sha256"]
    if resumed is not None:
        _sha256(
            resumed,
            "checkpoint resumed_from_sha256",
        )
    return value


def validate_brainrw_checkpoint_identity(
    payload: Mapping[str, object],
    *,
    config: BrainRWConfigIdentity,
    manifest: ManifestIdentity,
    subject: int,
    seed: int,
) -> Mapping[str, object]:
    if not isinstance(config, BrainRWConfigIdentity):
        raise TypeError(
            "checkpoint config identity is invalid"
        )
    if not isinstance(manifest, ManifestIdentity):
        raise TypeError(
            "checkpoint manifest identity is invalid"
        )
    subject = _positive_int(
        subject,
        "checkpoint expected subject",
    )
    seed = _nonnegative_int(
        seed,
        "checkpoint expected seed",
    )
    value = _validate_loaded_checkpoint(payload)
    expected_run_key, expected_bundle, expected_hashes = (
        brainrw_run_key(
            config,
            manifest,
            subject,
            seed,
            str(value["semantic_environment_sha256"]),
            str(value["validation_scope"]),
        )
    )
    comparisons = {
        "subject": subject,
        "seed": seed,
        "config_path": str(config.path),
        "config_sha256": config.sha256,
        "manifest_path": str(manifest.path),
        "manifest_sha256": manifest.manifest_sha256,
        "protocol_sha256": manifest.protocol_sha256,
        "input_bundle_sha256": expected_bundle,
        "run_key": expected_run_key,
        "clip_path": str(config.clip_path),
        "clip_config_sha256": config.clip_config_sha256,
        "clip_preprocessor_sha256": (
            config.clip_preprocessor_sha256
        ),
        "clip_weights_sha256": config.clip_weights_sha256,
    }
    for name, expected in comparisons.items():
        if value[name] != expected:
            raise ValueError(
                f"checkpoint {name} mismatch"
            )
    if value["input_hashes"] != expected_hashes:
        raise ValueError(
            "checkpoint input hashes mismatch"
        )
    expected_config_payload = dict(config.payload)
    if (
        _checkpoint_json_sha256(
            value["config_payload"],
            "config_payload",
        )
        != _checkpoint_json_sha256(
            expected_config_payload,
            "expected config payload",
        )
    ):
        raise ValueError(
            "checkpoint config payload mismatch"
        )
    model_manifest = _checkpoint_mapping(
        value["model_manifest"],
        "model_manifest",
    )
    brain = _checkpoint_mapping(
        model_manifest["brain_mlp"],
        "model_manifest.brain_mlp",
    )
    gradient_checkpointing = _checkpoint_mapping(
        model_manifest["gradient_checkpointing"],
        "model_manifest.gradient_checkpointing",
    )
    lora = _checkpoint_mapping(
        model_manifest["lora"],
        "model_manifest.lora",
    )
    config_brain = _checkpoint_mapping(
        config.payload["brain_mlp"],
        "config brain_mlp",
    )
    config_lora = _checkpoint_mapping(
        config.payload["lora"],
        "config lora",
    )
    config_training = _checkpoint_mapping(
        config.payload["training"],
        "config training",
    )
    if (
        brain["channels"] != 17
        or brain["samples"] != 250
        or brain["layers"] != 1
        or brain["dropout"] != config_brain["dropout"]
        or float(brain["layer_norm_eps"]) != 1e-6
        or lora["semantic_targets"] != config_lora["targets"]
        or lora["rank"] != config_lora["rank"]
        or lora["alpha"] != config_lora["alpha"]
        or lora["dropout"] != config_lora["dropout"]
        or config_training.get("gradient_checkpointing") is not True
        or gradient_checkpointing.get("requested") is not True
        or gradient_checkpointing.get("enabled") is not True
        or gradient_checkpointing.get("method")
        != "explicit_per_layer_torch_utils_checkpoint"
        or gradient_checkpointing.get("target")
        != "vision_model.encoder.layers"
        or gradient_checkpointing.get("use_reentrant") is not False
    ):
        raise ValueError(
            "checkpoint model manifest differs from semantic config"
        )
    return MappingProxyType(value)


def load_brainrw_checkpoint(
    path: Path,
    *,
    requested_scope: str,
) -> LoadedBrainRWCheckpoint:
    if requested_scope not in _DEVELOPMENT_SCOPES:
        raise PermissionError(
            "Brain-RW checkpoint access is development-only"
        )
    checkpoint_path = reject_development_path(
        path, "checkpoint"
    )
    descriptor = TypedArtifact(
        BRAINRW_CHECKPOINT_TYPE,
        checkpoint_path,
        checkpoint_sidecar(checkpoint_path),
    )
    capability = verify_typed_artifacts(
        requested_scope, [descriptor]
    )[0]
    with capability.open_verified() as handle:
        try:
            payload = torch.load(
                handle,
                map_location="cpu",
                weights_only=True,
            )
        except (
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                "checkpoint payload could not be loaded"
            ) from exc
    return LoadedBrainRWCheckpoint(
        payload=MappingProxyType(
            _validate_loaded_checkpoint(payload)
        ),
        sha256=capability.payload_sha256,
    )


def build_model_from_checkpoint(
    payload: Mapping[str, object],
) -> tuple[BrainRWCLIPLoRAModel, object]:
    config_payload = payload.get("config_payload")
    clip_path = payload.get("clip_path")
    if not isinstance(config_payload, Mapping) or not isinstance(
        clip_path, str
    ):
        raise ValueError(
            "checkpoint lacks model reconstruction metadata"
        )
    path = reject_development_path(
        Path(clip_path), "checkpoint CLIP path"
    )
    if (
        file_sha256(path / "config.json")
        != payload.get("clip_config_sha256")
    ):
        raise ValueError(
            "checkpoint CLIP config hash no longer matches"
        )
    if (
        file_sha256(path / "model.safetensors")
        != payload.get("clip_weights_sha256")
    ):
        raise ValueError(
            "checkpoint CLIP weights hash no longer matches"
        )
    if (
        file_sha256(path / "preprocessor_config.json")
        != payload.get("clip_preprocessor_sha256")
    ):
        raise ValueError(
            "checkpoint CLIP preprocessor hash no longer matches"
        )
    model, processor = build_brainrw_model(
        config_payload,
        path,
        expected_preprocessor_sha256=str(
            payload["clip_preprocessor_sha256"]
        ),
    )
    if (
        model.model_manifest_sha256
        != payload.get("model_manifest_sha256")
    ):
        raise ValueError(
            "resolved BrainMLP/CLIP-LoRA manifest differs from checkpoint"
        )
    model.load_checkpoint_states(
        payload["task_state"],
        payload["candidate_state"],
    )
    return model, processor




def _model_floating_dtype(model: nn.Module) -> torch.dtype:
    for parameter in model.parameters():
        if parameter.is_floating_point():
            return parameter.dtype
    raise ValueError("model has no floating parameters")


def evaluate_brainrw_similarity(
    model: BrainRWCLIPLoRAModel,
    dataset: Dataset[dict[str, object]],
    processor: object,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    numerical_dtype = (
        _model_floating_dtype(model)
        if dtype is None
        else dtype
    )
    if numerical_dtype not in {torch.float32, torch.bfloat16}:
        raise ValueError("evaluation dtype must be float32 or bfloat16")
    loader = DataLoader(
        dataset,
        batch_size=_positive_int(
            batch_size, "batch_size"
        ),
        shuffle=False,
        num_workers=0,
        collate_fn=BrainRWCollator(processor),
    )
    model.to(device=device, dtype=numerical_dtype)
    model.eval()
    brain_values: list[Tensor] = []
    image_values: list[Tensor] = []
    identifiers: list[str] = []
    with torch.inference_mode():
        for batch in loader:
            brain_values.append(
                model.encode_brain(
                    batch["brain_signals"].to(
                        device=device,
                        dtype=numerical_dtype,
                    )
                ).cpu()
            )
            image_values.append(
                model.encode_image(
                    batch["pixel_values"].to(
                        device=device,
                        dtype=numerical_dtype,
                    )
                ).cpu()
            )
            identifiers.extend(batch["image_ids"])
    expected = tuple(getattr(dataset, "ordered_ids"))
    if tuple(identifiers) != expected:
        raise ValueError(
            "evaluation dataset order differs from protocol ordered IDs"
        )
    brain = torch.cat(brain_values).float()
    image = torch.cat(image_values).float()
    similarity = (
        (brain @ image.T)
        .numpy()
        .astype(np.float32, copy=False)
    )
    return similarity, expected
