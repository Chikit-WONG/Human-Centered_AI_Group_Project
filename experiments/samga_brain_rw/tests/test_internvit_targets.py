from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch.nn as nn

from samga_brain_rw.hashing import sha256_json
from samga_brain_rw.internvit import (
    LoraTarget,
    build_lora_target_manifest,
    resolve_internvit_components,
    resolve_lora_targets,
)


class FakeAttention(nn.Module):
    def __init__(self, hidden_size: int = 8) -> None:
        super().__init__()
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.proj = nn.Linear(hidden_size, hidden_size)


class FakeMlp(nn.Module):
    def __init__(self, hidden_size: int = 8, intermediate_size: int = 32) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)


class FakeBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = FakeAttention()
        self.mlp = FakeMlp()


class FakeEncoder(nn.Module):
    def __init__(self, count: int = 45) -> None:
        super().__init__()
        self.layers = nn.ModuleList(FakeBlock() for _ in range(count))


class FakeInternViT(nn.Module):
    def __init__(self, count: int = 45) -> None:
        super().__init__()
        self.encoder = FakeEncoder(count)


def test_resolve_internvit_components_accepts_only_concrete_encoder_layers() -> None:
    model = FakeInternViT()

    encoder, layers = resolve_internvit_components(model)

    assert encoder is model.encoder
    assert layers is model.encoder.layers
    assert len(layers) == 45


@pytest.mark.parametrize(
    "model",
    [
        nn.Identity(),
        type("WrongRoot", (nn.Module,), {"__init__": lambda self: nn.Module.__init__(self)})(),
    ],
)
def test_resolve_internvit_components_rejects_missing_encoder(model: nn.Module) -> None:
    with pytest.raises(ValueError, match=r"model\.encoder"):
        resolve_internvit_components(model)


def test_resolve_internvit_components_rejects_missing_or_non_modulelist_layers() -> None:
    missing = FakeInternViT()
    del missing.encoder.layers
    with pytest.raises(ValueError, match=r"encoder\.layers"):
        resolve_internvit_components(missing)

    wrong = FakeInternViT()
    del wrong.encoder.layers
    object.__setattr__(wrong.encoder, "layers", [])
    with pytest.raises(ValueError, match="ModuleList"):
        resolve_internvit_components(wrong)


def test_resolve_internvit_components_has_no_clip_or_suffix_fallback() -> None:
    class ClipLike(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vision_model = FakeInternViT()

    with pytest.raises(ValueError, match=r"model\.encoder"):
        resolve_internvit_components(ClipLike())


def test_default_targets_map_one_based_28_36_to_zero_based_27_35() -> None:
    model = FakeInternViT()

    targets = resolve_lora_targets(model)

    assert isinstance(targets, tuple)
    assert len(targets) == 36
    assert targets[0].module_path == "encoder.layers.27.attn.qkv"
    assert targets[-1].module_path == "encoder.layers.35.mlp.fc2"
    assert [target.module_path.rsplit(".", 2)[-2:] for target in targets[:4]] == [
        ["attn", "qkv"],
        ["attn", "proj"],
        ["mlp", "fc1"],
        ["mlp", "fc2"],
    ]
    assert [target.block_one_based for target in targets[:4]] == [28] * 4
    assert [target.block_zero_based for target in targets[:4]] == [27] * 4
    assert [target.block_one_based for target in targets[-4:]] == [36] * 4
    assert [target.block_zero_based for target in targets[-4:]] == [35] * 4
    assert len({target.module_path for target in targets}) == 36
    assert len({id(target.module) for target in targets}) == 36


def test_targets_bind_roles_shapes_and_bias() -> None:
    targets = resolve_lora_targets(FakeInternViT(), 28, 28)

    assert [
        (
            target.semantic_roles,
            target.in_features,
            target.out_features,
            target.has_bias,
        )
        for target in targets
    ] == [
        (("query", "key", "value"), 8, 24, False),
        (("attention_output",), 8, 8, True),
        (("mlp_fc1",), 8, 32, True),
        (("mlp_fc2",), 32, 8, True),
    ]
    assert all(isinstance(target, LoraTarget) for target in targets)
    with pytest.raises(FrozenInstanceError):
        targets[0].block_one_based = 29  # type: ignore[misc]


@pytest.mark.parametrize(
    ("first_block", "last_block"),
    [
        (0, 1),
        (28, 46),
        (36, 28),
        (True, 36),
        (28, False),
        (28.0, 36),
        (28, "36"),
    ],
)
def test_resolver_rejects_invalid_block_ranges(
    first_block: object,
    last_block: object,
) -> None:
    with pytest.raises(ValueError, match="block range"):
        resolve_lora_targets(  # type: ignore[arg-type]
            FakeInternViT(),
            first_block,
            last_block,
        )


@pytest.mark.parametrize(
    ("parent_name", "component_name"),
    [
        ("attn", "qkv"),
        ("attn", "proj"),
        ("mlp", "fc1"),
        ("mlp", "fc2"),
    ],
)
def test_resolver_rejects_missing_components(
    parent_name: str,
    component_name: str,
) -> None:
    model = FakeInternViT()
    parent = getattr(model.encoder.layers[27], parent_name)
    delattr(parent, component_name)

    with pytest.raises(ValueError, match=component_name):
        resolve_lora_targets(model, 28, 28)


def test_resolver_rejects_clip_style_split_attention_projections() -> None:
    class ClipAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(8, 8)
            self.k_proj = nn.Linear(8, 8)
            self.v_proj = nn.Linear(8, 8)
            self.out_proj = nn.Linear(8, 8)

    model = FakeInternViT()
    model.encoder.layers[27].attn = ClipAttention()

    with pytest.raises(ValueError, match=r"attn\.qkv"):
        resolve_lora_targets(model, 28, 28)


@pytest.mark.parametrize(
    ("parent_name", "component_name"),
    [
        ("attn", "qkv"),
        ("attn", "proj"),
        ("mlp", "fc1"),
        ("mlp", "fc2"),
    ],
)
def test_resolver_rejects_non_linear_components(
    parent_name: str,
    component_name: str,
) -> None:
    model = FakeInternViT()
    parent = getattr(model.encoder.layers[27], parent_name)
    setattr(parent, component_name, nn.Identity())

    with pytest.raises(ValueError, match="nn.Linear"):
        resolve_lora_targets(model, 28, 28)


def test_resolver_rejects_repeated_blocks_and_component_aliases() -> None:
    repeated = FakeInternViT()
    repeated.encoder.layers[28] = repeated.encoder.layers[27]
    with pytest.raises(ValueError, match="aliased"):
        resolve_lora_targets(repeated, 28, 29)

    aliased = FakeInternViT()
    aliased.encoder.layers[28].attn.qkv = aliased.encoder.layers[27].attn.qkv
    with pytest.raises(ValueError, match="aliased"):
        resolve_lora_targets(aliased, 28, 29)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda block: setattr(block.attn, "qkv", nn.Linear(8, 23)),
        lambda block: setattr(block.attn, "proj", nn.Linear(8, 7)),
        lambda block: setattr(block.mlp, "fc1", nn.Linear(7, 32)),
        lambda block: setattr(block.mlp, "fc2", nn.Linear(31, 8)),
        lambda block: setattr(block.mlp, "fc2", nn.Linear(32, 7)),
    ],
)
def test_resolver_rejects_incompatible_component_shapes(mutate: object) -> None:
    model = FakeInternViT()
    mutate(model.encoder.layers[27])  # type: ignore[operator]

    with pytest.raises(ValueError, match="shape"):
        resolve_lora_targets(model, 28, 28)


def test_manifest_serializes_semantics_without_live_modules_and_binds_hash() -> None:
    manifest = build_lora_target_manifest(FakeInternViT())

    assert set(manifest) == {
        "block_numbering",
        "first_block_one_based",
        "last_block_one_based",
        "payload_sha256",
        "payload_type",
        "schema_version",
        "scope",
        "target_count",
        "targets",
    }
    assert manifest["schema_version"] == 1
    assert manifest["payload_type"] == "samga_brain_rw.internvit_lora_targets"
    assert manifest["scope"] == "train"
    assert manifest["target_count"] == 36
    assert manifest["block_numbering"] == {
        "configuration": "one_based",
        "python_module_list": "zero_based",
        "zero_based_offset": -1,
    }
    targets = manifest["targets"]
    assert isinstance(targets, list)
    assert targets[0] == {
        "block_one_based": 28,
        "block_zero_based": 27,
        "has_bias": False,
        "in_features": 8,
        "module_path": "encoder.layers.27.attn.qkv",
        "out_features": 24,
        "semantic_roles": ["query", "key", "value"],
        "weight_shape": [24, 8],
    }
    payload = dict(manifest)
    payload_sha256 = payload.pop("payload_sha256")
    assert payload_sha256 == sha256_json(payload)
    assert not any(
        isinstance(value, nn.Module)
        for target in targets
        for value in target.values()
    )
