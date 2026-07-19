"""Development-only Brain-RW/CLIP-LoRA primitives.

There is deliberately no formal-test loader in this module. Protocol metadata
and typed checkpoint sidecars are verified before EEG, image, NumPy, or
PyTorch payload loading.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import random
import re
import socket
import stat
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Sampler

from .access import TypedArtifact, verify_typed_artifacts
from .config import SemanticConfig, make_run_key
from .data import POSTERIOR_CHANNELS, ProtocolSubjectDataset
from .hashing import canonical_json_bytes, ordered_ids_sha256, sha256_json


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
        self.projection_dim = _positive_int(
            projection_dim, "projection_dim"
        )
        self.brain_mlp = BrainMLP(
            channels=channels,
            samples=samples,
            hidden_size=self.projection_dim,
            dropout=dropout,
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
            logits = (
                self.logit_scale.exp().clamp(max=100.0) * similarity
            )
            labels = torch.arange(
                logits.shape[0], device=logits.device
            )
            loss = 0.5 * (
                F.cross_entropy(logits, labels)
                + F.cross_entropy(logits.T, labels)
            )
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
    train_role_sha256: str
    val_dev_role_sha256: str
    train_ordered_ids: tuple[str, ...]
    val_dev_ordered_ids: tuple[str, ...]
    train_ordered_ids_sha256: str
    val_dev_ordered_ids_sha256: str


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
        source_manifest_sha256=_sha256(
            document.get("source_manifest_sha256"),
            "source manifest hash",
        ),
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
    )


@dataclass(frozen=True)
class BrainRWConfigIdentity:
    path: Path
    payload: Mapping[str, object]
    sha256: str
    clip_path: Path
    clip_config_sha256: str
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
    if not config_file.is_file() or not weights_file.is_file():
        raise ValueError(
            "CLIP path lacks required local model files"
        )
    config_hash = file_sha256(config_file)
    weights_hash = file_sha256(weights_file)
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
        clip_weights_sha256=weights_hash,
    )


def input_hashes(
    config: BrainRWConfigIdentity,
    manifest: ManifestIdentity,
) -> dict[str, str]:
    return {
        "clip_config": config.clip_config_sha256,
        "clip_weights": config.clip_weights_sha256,
        "config": config.sha256,
        "manifest": manifest.manifest_sha256,
        "protocol": manifest.protocol_sha256,
        "records": manifest.records_sha256,
        "source_manifest": manifest.source_manifest_sha256,
        "train_role": manifest.train_role_sha256,
        "val_dev_role": manifest.val_dev_role_sha256,
    }


def brainrw_run_key(
    config: BrainRWConfigIdentity,
    manifest: ManifestIdentity,
    subject: int,
    seed: int,
) -> tuple[str, str, dict[str, str]]:
    hashes = input_hashes(config, manifest)
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
) -> tuple[nn.Module, object]:
    from transformers import (
        CLIPImageProcessor,
        CLIPVisionModelWithProjection,
    )

    model = CLIPVisionModelWithProjection.from_pretrained(
        str(clip_path),
        local_files_only=True,
    )
    processor = CLIPImageProcessor.from_pretrained(
        str(clip_path),
        local_files_only=True,
    )
    return model, processor


def build_brainrw_model(
    config_payload: Mapping[str, object],
    clip_path: Path,
) -> tuple[BrainRWCLIPLoRAModel, object]:
    vision, processor = load_clip_components(clip_path)
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


def capture_environment() -> dict[str, object]:
    packages = {}
    for name in ("numpy", "torch", "transformers", "peft"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "packages": packages,
    }


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
    if (
        payload.get("scope") != "train"
        or payload.get("validation_scope") != "val-dev"
    ):
        raise PermissionError(
            "Brain-RW checkpoints must bind train and val-dev only"
        )
    if payload.get("observed_scopes") != [
        "train",
        "val-dev",
    ]:
        raise PermissionError(
            "checkpoint observed_scopes must be train and val-dev"
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
        "observed_scopes": ["train", "val-dev"],
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


def _validate_loaded_checkpoint(
    payload: object,
) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise ValueError(
            "checkpoint payload must be a mapping"
        )
    value = dict(payload)
    required = (
        "schema_version",
        "payload_type",
        "complete",
        "scope",
        "validation_scope",
        "subject",
        "seed",
        "config_sha256",
        "manifest_sha256",
        "protocol_sha256",
        "task_state",
        "candidate_state",
        "git_sha",
        "run_key",
        "observed_scopes",
    )
    missing = [key for key in required if key not in value]
    if missing:
        raise ValueError(
            f"checkpoint is missing {missing[0]}"
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
    if (
        value["scope"] != "train"
        or value["validation_scope"] != "val-dev"
    ):
        raise PermissionError(
            "checkpoint contains a non-development scope"
        )
    if value["observed_scopes"] != [
        "train",
        "val-dev",
    ]:
        raise PermissionError(
            "checkpoint observed scopes are not development-only"
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
        "config_sha256",
        "manifest_sha256",
        "protocol_sha256",
    ):
        _sha256(value[key], f"checkpoint {key}")
    if (
        not isinstance(value["git_sha"], str)
        or _GIT_RE.fullmatch(value["git_sha"]) is None
    ):
        raise ValueError("checkpoint git_sha is invalid")
    if not isinstance(value["task_state"], Mapping) or not isinstance(
        value["candidate_state"], Mapping
    ):
        raise ValueError(
            "checkpoint model states must be mappings"
        )
    return value


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
    model, processor = build_brainrw_model(
        config_payload, path
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


def evaluate_brainrw_similarity(
    model: BrainRWCLIPLoRAModel,
    dataset: Dataset[dict[str, object]],
    processor: object,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, tuple[str, ...]]:
    loader = DataLoader(
        dataset,
        batch_size=_positive_int(
            batch_size, "batch_size"
        ),
        shuffle=False,
        num_workers=0,
        collate_fn=BrainRWCollator(processor),
    )
    model.to(device)
    model.eval()
    brain_values: list[Tensor] = []
    image_values: list[Tensor] = []
    identifiers: list[str] = []
    with torch.inference_mode():
        for batch in loader:
            brain_values.append(
                model.encode_brain(
                    batch["brain_signals"].to(device)
                ).cpu()
            )
            image_values.append(
                model.encode_image(
                    batch["pixel_values"].to(device)
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
