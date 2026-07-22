from __future__ import annotations

import copy
import hashlib
import io
import importlib.util
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from samga_brain_rw import brainrw as br
from samga_brain_rw.data import POSTERIOR_CHANNELS
from samga_brain_rw.hashing import canonical_json_bytes, ordered_ids_sha256
from samga_brain_rw.runtime_contract import PINNED_SEMANTIC_ENVIRONMENT
from samga_brain_rw.scores import ScoreArtifact


class _FakeBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.k_proj = nn.Linear(8, 8)
        self.v_proj = nn.Linear(8, 8)
        self.out_proj = nn.Linear(8, 8)
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 8)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        mixed = self.q_proj(values) + self.k_proj(values) + self.v_proj(values)
        return self.fc2(torch.nn.functional.gelu(self.fc1(self.out_proj(mixed))))


class _FakeVision(nn.Module):
    def __init__(self, *, include_visual_projection: bool = True) -> None:
        super().__init__()
        self.gradient_checkpointing_calls: list[dict[str, object]] = []
        self.is_gradient_checkpointing = False
        self.vision_model = nn.Module()
        self.vision_model.encoder = nn.Module()
        self.vision_model.encoder.layers = nn.ModuleList([_FakeBlock()])
        if include_visual_projection:
            self.visual_projection = nn.Linear(8, 8, bias=False)
        self.config = SimpleNamespace(
            num_hidden_layers=1,
            projection_dim=8,
        )

    def forward(self, pixel_values: torch.Tensor, **_: object) -> SimpleNamespace:
        values = pixel_values.reshape(pixel_values.shape[0], 8)
        values = self.vision_model.encoder.layers[0](values)
        return SimpleNamespace(image_embeds=self.visual_projection(values))

    def gradient_checkpointing_enable(
        self,
        *,
        gradient_checkpointing_kwargs: dict[str, object],
    ) -> None:
        self.gradient_checkpointing_calls.append(
            dict(gradient_checkpointing_kwargs)
        )
        self.is_gradient_checkpointing = True


class _FakeProcessor:
    def __call__(self, *, images: list[torch.Tensor], return_tensors: str):
        assert return_tensors == "pt"
        return {"pixel_values": torch.stack(images).float()}


def _model(
    *,
    channels: int = 2,
    samples: int = 4,
    lora_rank: int = 2,
    lora_alpha: int = 2,
) -> br.BrainRWCLIPLoRAModel:
    torch.manual_seed(5)
    return br.BrainRWCLIPLoRAModel(
        _FakeVision(),
        channels=channels,
        samples=samples,
        projection_dim=8,
        dropout=0.1,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
    )


def test_model_locks_audited_brain_mlp_and_exact_clip_lora_targets() -> None:
    torch.manual_seed(5)
    raw = _FakeVision()
    reference = copy.deepcopy(raw)
    pixels = torch.randn(3, 8)
    reference_output = reference(pixel_values=pixels).image_embeds
    model = br.BrainRWCLIPLoRAModel(
        raw,
        channels=2,
        samples=4,
        projection_dim=8,
        dropout=0.1,
        lora_rank=2,
        lora_alpha=2,
        lora_dropout=0.0,
    )

    expected = (
        "vision_model.encoder.layers.0.fc1",
        "vision_model.encoder.layers.0.fc2",
        "vision_model.encoder.layers.0.k_proj",
        "vision_model.encoder.layers.0.out_proj",
        "vision_model.encoder.layers.0.q_proj",
        "vision_model.encoder.layers.0.v_proj",
        "visual_projection",
    )
    assert model.resolved_lora_targets == expected
    assert len(model.target_manifest_sha256) == 64
    assert model.brain_mlp.proj_in.in_features == 8
    assert model.brain_mlp.proj_in.out_features == 8
    assert len(model.brain_mlp.layers) == 1
    assert model.brain_mlp.layers[0].dropout.p == 0.1
    assert model.brain_mlp.layers[0].norm.eps == 1e-6
    assert torch.equal(
        model.vision_model(pixel_values=pixels).image_embeds,
        reference_output,
    )
    assert all(
        not parameter.requires_grad
        for name, parameter in model.vision_model.named_parameters()
        if ".lora_" not in name
    )
    assert all(
        parameter.requires_grad
        for name, parameter in model.vision_model.named_parameters()
        if ".lora_" in name
    )


def test_model_forward_gradient_isolation_and_state_roundtrip() -> None:
    model = _model()
    model.train()
    output = model(
        brain_signals=torch.randn(3, 2, 4),
        pixel_values=torch.randn(3, 8),
    )
    assert output.loss is not None and output.loss.ndim == 0
    assert output.similarity.shape == (3, 3)
    assert output.brain_embeds.shape == output.image_embeds.shape == (3, 8)
    output.loss.backward()
    assert model.brain_mlp.proj_in.weight.grad is not None
    candidate_gradients = {
        name: parameter.grad
        for name, parameter in model.vision_model.named_parameters()
        if ".lora_" in name
    }
    assert candidate_gradients
    assert any(
        gradient is not None and torch.count_nonzero(gradient) > 0
        for name, gradient in candidate_gradients.items()
        if name.endswith("lora_B")
    )
    assert all(
        parameter.grad is None
        for name, parameter in model.vision_model.named_parameters()
        if ".lora_" not in name
    )

    task_state = copy.deepcopy(model.task_state_dict())
    candidate_state = copy.deepcopy(model.candidate_state_dict())
    restored = _model()
    restored.load_checkpoint_states(task_state, candidate_state)
    model.eval()
    restored.eval()
    brain = torch.randn(2, 2, 4)
    pixels = torch.randn(2, 8)
    with torch.no_grad():
        assert torch.equal(
            model(brain_signals=brain, pixel_values=pixels).similarity,
            restored(brain_signals=brain, pixel_values=pixels).similarity,
        )


def test_brain_mlp_uses_legacy_pretrained_model_initialization() -> None:
    model = _model(channels=17, samples=250)
    linears = [
        module
        for module in model.brain_mlp.modules()
        if isinstance(module, nn.Linear)
    ]
    weights = torch.cat(
        [module.weight.detach().reshape(-1) for module in linears]
    )
    assert abs(float(weights.mean())) < 0.002
    assert float(weights.std(unbiased=False)) == pytest.approx(
        0.02,
        abs=0.002,
    )
    assert all(
        module.bias is None
        or torch.count_nonzero(module.bias.detach()).item() == 0
        for module in linears
    )
    norms = [
        module
        for module in model.brain_mlp.modules()
        if isinstance(module, nn.LayerNorm)
    ]
    assert norms
    assert all(
        torch.equal(module.weight, torch.ones_like(module.weight))
        for module in norms
    )
    assert all(
        torch.equal(module.bias, torch.zeros_like(module.bias))
        for module in norms
    )


def test_brainrw_loss_is_unclamped_one_way_brain_to_image_ce() -> None:
    model = _model()
    model.eval()
    with torch.no_grad():
        model.logit_scale.fill_(math.log(101.0))
    output = model(
        brain_signals=torch.randn(3, 2, 4),
        pixel_values=torch.randn(3, 8),
    )
    labels = torch.arange(3)
    logits_per_brain = model.logit_scale.exp() * output.similarity
    expected = torch.nn.functional.cross_entropy(
        logits_per_brain,
        labels,
    )
    symmetric = 0.5 * (
        expected
        + torch.nn.functional.cross_entropy(
            logits_per_brain.T,
            labels,
        )
    )
    assert torch.allclose(output.loss, expected)
    assert not torch.allclose(output.loss, symmetric)


def test_model_rejects_incomplete_or_unbalanced_target_sets() -> None:
    with pytest.raises(ValueError, match="visual_projection"):
        br.BrainRWCLIPLoRAModel(
            _FakeVision(include_visual_projection=False),
            channels=2,
            samples=4,
            projection_dim=8,
            dropout=0.1,
            lora_rank=2,
            lora_alpha=2,
            lora_dropout=0.0,
        )


class _BaseDataset:
    calls: list[tuple[object, ...]] = []

    def __init__(
        self,
        manifest_path: Path,
        scope: str,
        seed: int,
        selected_channels: tuple[str, ...],
        feature_cache: None,
        smooth_probability: float,
        *,
        expected_source_payload_sha256: str | None = None,
    ) -> None:
        self.calls.append(
            (
                manifest_path,
                scope,
                seed,
                selected_channels,
                feature_cache,
                smooth_probability,
                expected_source_payload_sha256,
            )
        )
        self.scope = scope
        self.subject_id = 1
        self.ordered_ids = ("image-a",)
        self.query_ids = self.ordered_ids if scope == "val-dev" else ()
        self.gallery_ids = self.query_ids
        self.row_indices = (9,)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, _: int) -> dict[str, object]:
        return {
            "concept_id": "concept-a",
            "eeg": torch.ones(17, 250),
            "image_id": "image-a",
            "image_path": "/safe/training_images/concept-a/image-a.jpg",
            "row_index": 9,
            "scope": self.scope,
            "subject_id": 1,
        }


def test_development_dataset_and_collator_never_expose_test_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _BaseDataset.calls.clear()
    monkeypatch.setattr(br, "ProtocolSubjectDataset", _BaseDataset)
    loaded: list[Path] = []
    dataset = br.BrainRWDevelopmentDataset(
        Path("sub-01_protocol.json"),
        "train",
        42,
        image_loader=lambda path: loaded.append(path) or torch.arange(8).float(),
        expected_source_payload_sha256="a" * 64,
    )
    item = dataset[0]
    assert loaded == [Path("/safe/training_images/concept-a/image-a.jpg")]
    assert item["scope"] == "train"
    assert _BaseDataset.calls == [
        (
            Path("sub-01_protocol.json"),
            "train",
            42,
            POSTERIOR_CHANNELS,
            None,
            0.3,
            "a" * 64,
        )
    ]
    collator = br.BrainRWCollator(_FakeProcessor())
    batch = collator([item, {**item, "image_id": "image-b", "row_index": 10}])
    assert batch["brain_signals"].shape == (2, 17, 250)
    assert batch["pixel_values"].shape == (2, 8)
    assert batch["subject_ids"].tolist() == [1, 1]
    assert batch["image_ids"] == ("image-a", "image-b")
    with pytest.raises(ValueError, match="17,250"):
        collator(
            [{**item, "eeg": torch.ones(2, 4)}]
        )
    with pytest.raises(PermissionError, match="train or val-dev"):
        br.BrainRWDevelopmentDataset(
            Path("sub-01_protocol.json"), "test", 42
        )
    with pytest.raises(PermissionError, match="one development scope"):
        collator([{**item, "scope": "train"}, {**item, "scope": "val-dev"}])


def test_manifest_preflight_rejects_renamed_and_metadata_free_inputs(
    tmp_path: Path,
) -> None:
    renamed = tmp_path / "sub-01_test.json"
    renamed.write_text("{}\n", encoding="utf-8")
    with pytest.raises((PermissionError, ValueError), match="test|protocol"):
        br.load_development_manifest_identity(renamed, expected_subject=1)
    metadata_free = tmp_path / "sub-01_protocol.json"
    metadata_free.write_text('{"split":"test"}\n', encoding="utf-8")
    with pytest.raises((PermissionError, ValueError)):
        br.load_development_manifest_identity(metadata_free, expected_subject=1)


@pytest.mark.parametrize(
    "component",
    ("test", "formal", "formal-test", "formal_test", "val-confirm", "val_confirm"),
)
def test_all_sealed_scope_path_components_are_rejected(
    tmp_path: Path,
    component: str,
) -> None:
    with pytest.raises((PermissionError, ValueError), match="sealed|scope"):
        br.reject_development_path(
            tmp_path / component / "artifact.pt",
            "artifact",
        )


def test_symlink_components_and_unsafe_absolute_creates_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises((PermissionError, ValueError), match="symlink"):
        br.reject_development_path(linked / "artifact.pt", "artifact")

    create_opens: list[tuple[object, int, object]] = []
    relative_mkdirs: list[tuple[object, object]] = []
    real_open = br.os.open
    real_mkdir = br.os.mkdir

    def open_spy(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if flags & br.os.O_CREAT:
            create_opens.append((path, flags, dir_fd))
        return real_open(path, flags, mode, dir_fd=dir_fd)

    def mkdir_spy(
        path: object,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> None:
        relative_mkdirs.append((path, dir_fd))
        real_mkdir(path, mode=mode, dir_fd=dir_fd)

    monkeypatch.setattr(br.os, "open", open_spy)
    monkeypatch.setattr(br.os, "mkdir", mkdir_spy)
    output = tmp_path / "run"
    br.create_development_directory_exclusive(
        output,
        context="run output",
    )
    br.write_development_file_exclusive(
        output / "run_manifest.json",
        b"{}\n",
        context="run manifest",
    )
    assert relative_mkdirs
    assert all(
        isinstance(path, str)
        and "/" not in path
        and isinstance(dir_fd, int)
        for path, dir_fd in relative_mkdirs
    )
    assert create_opens
    assert all(
        isinstance(path, str)
        and "/" not in path
        and isinstance(dir_fd, int)
        and flags & br.os.O_EXCL
        and flags & getattr(br.os, "O_NOFOLLOW", 0)
        for path, flags, dir_fd in create_opens
    )


def test_rng_state_is_basic_tensor_schema_safe_for_weights_only_load() -> None:
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    state = br.capture_rng_state()
    assert isinstance(state["python"], dict)
    assert isinstance(state["numpy"], dict)
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    loaded = torch.load(
        buffer,
        map_location="cpu",
        weights_only=True,
    )
    expected = (
        random.random(),
        float(np.random.random()),
        torch.rand(3),
    )
    br.restore_rng_state(loaded)
    actual = (
        random.random(),
        float(np.random.random()),
        torch.rand(3),
    )
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def _write_config(configs_dir: Path, root: Path) -> Path:
    clip = root / "clip"
    clip.mkdir()
    (clip / "config.json").write_text("{}\n", encoding="utf-8")
    (clip / "model.safetensors").write_bytes(b"fake-weights")
    (clip / "preprocessor_config.json").write_text(
        '{"image_processor_type":"CLIPImageProcessor"}\n',
        encoding="utf-8",
    )
    payload = json.loads(
        (configs_dir / "brainrw_clip_lora_v1.json").read_text(encoding="utf-8")
    )
    payload["clip"]["path"] = str(clip)
    payload["clip"]["config_sha256"] = hashlib.sha256(
        (clip / "config.json").read_bytes()
    ).hexdigest()
    payload["clip"]["weights_sha256"] = hashlib.sha256(
        (clip / "model.safetensors").read_bytes()
    ).hexdigest()
    path = root / "brainrw.json"
    path.write_bytes(canonical_json_bytes(payload) + b"\n")
    return path


def _identity() -> br.ManifestIdentity:
    train_ids = ("concept-a", "concept-b")
    val_ids = ("image-a", "image-b")
    return br.ManifestIdentity(
        path=Path("sub-01_protocol.json"),
        subject=1,
        manifest_sha256="1" * 64,
        protocol_sha256="2" * 64,
        records_sha256="3" * 64,
        source_manifest_sha256="4" * 64,
        source_payload_path=Path("/safe/sub-01/train.pt"),
        source_payload_sha256="a" * 64,
        source_payload_byte_count=123,
        train_role_sha256="5" * 64,
        val_dev_role_sha256="6" * 64,
        train_ordered_ids=train_ids,
        val_dev_ordered_ids=val_ids,
        train_ordered_ids_sha256=ordered_ids_sha256(train_ids),
        val_dev_ordered_ids_sha256=ordered_ids_sha256(val_ids),
        train_row_count=2,
        val_dev_row_count=2,
    )


def test_config_and_run_identity_bind_preprocessor_and_source_payload(
    configs_dir: Path,
    tmp_path: Path,
) -> None:
    config_path = _write_config(configs_dir, tmp_path)
    config = br.verify_brainrw_config(
        config_path,
        tmp_path / "clip",
    )
    preprocessor = tmp_path / "clip" / "preprocessor_config.json"
    assert config.clip_preprocessor_sha256 == hashlib.sha256(
        preprocessor.read_bytes()
    ).hexdigest()
    semantic_environment_sha256 = hashlib.sha256(
        canonical_json_bytes(PINNED_SEMANTIC_ENVIRONMENT)
    ).hexdigest()
    hashes = br.input_hashes(
        config,
        _identity(),
        semantic_environment_sha256,
    )
    assert hashes["clip_preprocessor"] == (
        config.clip_preprocessor_sha256
    )
    assert hashes["source_payload"] == "a" * 64
    assert hashes["semantic_environment"] == (
        semantic_environment_sha256
    )
    run_key, _, _ = br.brainrw_run_key(
        config,
        _identity(),
        1,
        42,
        semantic_environment_sha256,
    )
    assert "validation_policy" not in hashes
    train_only_run_key, _, train_only_hashes = br.brainrw_run_key(
        config,
        _identity(),
        1,
        42,
        semantic_environment_sha256,
        "none",
    )
    assert "validation_policy" in train_only_hashes
    assert train_only_run_key != run_key
    changed_environment = dict(PINNED_SEMANTIC_ENVIRONMENT)
    changed_environment["transformers"] = "0.0.0"
    changed_environment_sha256 = hashlib.sha256(
        canonical_json_bytes(changed_environment)
    ).hexdigest()
    changed_run_key, _, _ = br.brainrw_run_key(
        config,
        _identity(),
        1,
        42,
        changed_environment_sha256,
    )
    assert changed_run_key != run_key


def test_model_build_uses_verified_preprocessor_digest(
    configs_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = br.verify_brainrw_config(
        _write_config(configs_dir, tmp_path),
        tmp_path / "clip",
    )
    observed: list[str] = []

    def fake_load(
        _clip_path: Path,
        *,
        expected_config_sha256: str,
        expected_weights_sha256: str,
        expected_preprocessor_sha256: str,
    ) -> tuple[nn.Module, object]:
        assert expected_config_sha256 == config.clip_config_sha256
        assert expected_weights_sha256 == config.clip_weights_sha256
        observed.append(expected_preprocessor_sha256)
        return _FakeVision(), _FakeProcessor()

    monkeypatch.setattr(br, "load_clip_components", fake_load)
    br.build_brainrw_model(
        config.payload,
        config.clip_path,
        expected_preprocessor_sha256=(
            config.clip_preprocessor_sha256
        ),
    )
    assert observed == [config.clip_preprocessor_sha256]


def test_model_build_enables_and_manifests_locked_gradient_checkpointing(
    configs_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = br.verify_brainrw_config(
        _write_config(configs_dir, tmp_path),
        tmp_path / "clip",
    )
    vision = _FakeVision()
    monkeypatch.setattr(
        br,
        "load_clip_components",
        lambda *_a, **_k: (vision, _FakeProcessor()),
    )

    model, _ = br.build_brainrw_model(
        config.payload,
        config.clip_path,
        expected_preprocessor_sha256=(
            config.clip_preprocessor_sha256
        ),
    )

    assert vision.gradient_checkpointing_calls == []
    assert model.model_manifest["gradient_checkpointing"] == {
        "enabled": True,
        "method": "explicit_per_layer_torch_utils_checkpoint",
        "requested": True,
        "target": "vision_model.encoder.layers",
        "use_reentrant": False,
        "wrapped_layer_count": 1,
    }


def test_model_build_does_not_depend_on_transformers_checkpointing_flag(
    configs_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = br.verify_brainrw_config(
        _write_config(configs_dir, tmp_path),
        tmp_path / "clip",
    )
    unsupported = _FakeVision()
    unsupported.gradient_checkpointing_enable = None  # type: ignore[method-assign]
    monkeypatch.setattr(
        br,
        "load_clip_components",
        lambda *_a, **_k: (unsupported, _FakeProcessor()),
    )

    model, _ = br.build_brainrw_model(
        config.payload,
        config.clip_path,
        expected_preprocessor_sha256=(
            config.clip_preprocessor_sha256
        ),
    )
    assert model.model_manifest["gradient_checkpointing"]["enabled"] is True


def test_real_clip_executes_explicit_train_only_per_layer_checkpointing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch.utils.checkpoint as torch_checkpoint
    from transformers import (
        CLIPVisionConfig,
        CLIPVisionModelWithProjection,
    )

    vision = CLIPVisionModelWithProjection(
        CLIPVisionConfig(
            attention_dropout=0.0,
            hidden_act="gelu",
            hidden_size=8,
            image_size=8,
            intermediate_size=16,
            num_attention_heads=2,
            num_channels=3,
            num_hidden_layers=2,
            patch_size=4,
            projection_dim=8,
        )
    )
    model = br.BrainRWCLIPLoRAModel(
        vision,
        channels=2,
        samples=4,
        projection_dim=8,
        dropout=0.0,
        lora_rank=2,
        lora_alpha=2,
    )
    calls: list[dict[str, object]] = []
    real_checkpoint = torch_checkpoint.checkpoint

    def checkpoint_spy(
        function: object,
        *args: object,
        **kwargs: object,
    ) -> object:
        calls.append(dict(kwargs))
        return real_checkpoint(function, *args, **kwargs)

    monkeypatch.setattr(
        torch_checkpoint,
        "checkpoint",
        checkpoint_spy,
    )
    model.train()
    output = model(
        brain_signals=torch.randn(2, 2, 4),
        pixel_values=torch.randn(2, 3, 8, 8),
    )
    assert output.loss is not None
    output.loss.backward()

    assert calls == [
        {"use_reentrant": False},
        {"use_reentrant": False},
    ]
    lora_b_gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if ".lora_B" in name
    ]
    assert lora_b_gradients
    assert all(value is not None for value in lora_b_gradients)
    assert any(
        bool(torch.count_nonzero(value).item())
        for value in lora_b_gradients
        if value is not None
    )
    assert model.model_manifest["gradient_checkpointing"] == {
        "enabled": True,
        "method": "explicit_per_layer_torch_utils_checkpoint",
        "requested": True,
        "target": "vision_model.encoder.layers",
        "use_reentrant": False,
        "wrapped_layer_count": 2,
    }

    calls.clear()
    model.eval()
    with torch.inference_mode():
        model.encode_image(torch.randn(2, 3, 8, 8))
    assert calls == []


def test_clip_loader_pins_components_and_forces_safetensors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clip = tmp_path / "clip"
    clip.mkdir()
    files = {
        "config": clip / "config.json",
        "weights": clip / "model.safetensors",
        "preprocessor": clip / "preprocessor_config.json",
    }
    for name, path in files.items():
        path.write_bytes(name.encode("ascii"))
    digests = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in files.items()
    }
    calls: list[tuple[str, str, dict[str, object]]] = []

    class Vision:
        @classmethod
        def from_pretrained(
            cls,
            path: str,
            **kwargs: object,
        ) -> str:
            calls.append(("vision", path, kwargs))
            return "vision"

    class Processor:
        @classmethod
        def from_pretrained(
            cls,
            path: str,
            **kwargs: object,
        ) -> str:
            calls.append(("processor", path, kwargs))
            return "processor"

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            CLIPImageProcessor=Processor,
            CLIPVisionModelWithProjection=Vision,
        ),
    )
    assert br.load_clip_components(
        clip,
        expected_config_sha256=digests["config"],
        expected_weights_sha256=digests["weights"],
        expected_preprocessor_sha256=digests["preprocessor"],
    ) == ("vision", "processor")
    assert calls == [
        (
            "vision",
            str(clip),
            {
                "local_files_only": True,
                "use_safetensors": True,
            },
        ),
        (
            "processor",
            str(clip),
            {"local_files_only": True},
        ),
    ]


def _runtime_attestation() -> dict[str, object]:
    semantic_environment = dict(PINNED_SEMANTIC_ENVIRONMENT)
    contract = {
        "accelerator": "NVIDIA A40",
        "device_type": "cuda",
        "dtype": "bfloat16",
        "schema_version": 1,
    }
    evidence = {
        "accelerator_name": "NVIDIA A40",
        "bf16_supported": True,
        "cuda_available": True,
        "cuda_capability": [8, 6],
        "cuda_device_count": 1,
        "cuda_device_index": 0,
        "cuda_version": "12.8",
        "device_type": "cuda",
        "dtype": "bfloat16",
        "schema_version": 1,
        "torch_version": "2.11.0+cu128",
        "total_memory_bytes": 48 * 1024**3,
    }
    return {
        "semantic_environment": semantic_environment,
        "semantic_environment_sha256": hashlib.sha256(
            canonical_json_bytes(semantic_environment)
        ).hexdigest(),
        "runtime_contract": contract,
        "runtime_contract_sha256": hashlib.sha256(
            canonical_json_bytes(contract)
        ).hexdigest(),
        "runtime_evidence": evidence,
        "runtime_evidence_sha256": hashlib.sha256(
            canonical_json_bytes(evidence)
        ).hexdigest(),
    }


def _fake_cpu_runtime() -> br.BrainRWProductionRuntime:
    attestation = _runtime_attestation()
    return br.BrainRWProductionRuntime(
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        contract=attestation["runtime_contract"],
        semantic_environment=attestation[
            "semantic_environment"
        ],
        semantic_environment_sha256=str(
            attestation["semantic_environment_sha256"]
        ),
        contract_sha256=str(
            attestation["runtime_contract_sha256"]
        ),
        evidence=attestation["runtime_evidence"],
        evidence_sha256=str(
            attestation["runtime_evidence_sha256"]
        ),
    )


def _complete_checkpoint_payload(
    config: br.BrainRWConfigIdentity,
    manifest: br.ManifestIdentity,
) -> dict[str, object]:
    model = _model(
        channels=17,
        samples=250,
        lora_rank=32,
        lora_alpha=32,
    )
    attestation = _runtime_attestation()
    run_key, input_bundle, hashes = br.brainrw_run_key(
        config,
        manifest,
        1,
        42,
        str(attestation["semantic_environment_sha256"]),
    )
    sampler = br.StatefulIndexSampler(2, 42)
    tuple(iter(sampler))
    return {
        **attestation,
        "schema_version": 1,
        "payload_type": br.BRAINRW_CHECKPOINT_TYPE,
        "complete": True,
        "training_complete": False,
        "scope": "train",
        "validation_scope": "val-dev",
        "observed_scopes": ["train", "val-dev"],
        "subject": 1,
        "seed": 42,
        "config_path": str(config.path),
        "config_payload": json.loads(
            canonical_json_bytes(dict(config.payload))
        ),
        "config_sha256": config.sha256,
        "manifest_path": str(manifest.path),
        "manifest_sha256": manifest.manifest_sha256,
        "protocol_sha256": manifest.protocol_sha256,
        "input_hashes": hashes,
        "input_bundle_sha256": input_bundle,
        "run_key": run_key,
        "clip_path": str(config.clip_path),
        "clip_config_sha256": config.clip_config_sha256,
        "clip_preprocessor_sha256": (
            config.clip_preprocessor_sha256
        ),
        "clip_weights_sha256": config.clip_weights_sha256,
        "model_manifest": json.loads(
            canonical_json_bytes(dict(model.model_manifest))
        ),
        "model_manifest_sha256": model.model_manifest_sha256,
        "target_manifest_sha256": model.target_manifest_sha256,
        "task_state": model.task_state_dict(),
        "candidate_state": model.candidate_state_dict(),
        "task_initialization_sha256": br.state_dict_sha256(
            model.task_state_dict()
        ),
        "candidate_initialization_sha256": br.state_dict_sha256(
            model.candidate_state_dict()
        ),
        "optimizer_state": {},
        "scheduler_state": {},
        "epoch": 0,
        "global_step": 1,
        "planned_steps": 25,
        "rng_state": br.capture_rng_state(),
        "sampler_state": sampler.state_dict(),
        "dataloader_generator_state": (
            torch.Generator().manual_seed(1).get_state()
        ),
        "data_order_sha256": "b" * 64,
        "effective_batch_size": 512,
        "steps": 1,
        "runtime_dtype": "bfloat16",
        "environment": json.loads(
            canonical_json_bytes(attestation["semantic_environment"])
        ),
        "git_sha": "c" * 40,
        "git_provenance": {
            "clean": True,
            "git_sha": "c" * 40,
            "repository_root": "/safe/repository",
        },
        "validation_metrics": {
            "gallery_count": 2,
            "query_count": 2,
            "top1_count": 1,
            "top1_rate": 0.5,
            "top5_count": 2,
            "top5_rate": 1.0,
        },
        "resumed_from_sha256": None,
    }


def test_checkpoint_distinguishes_resumable_partial_and_terminal_training(
    configs_dir: Path,
    tmp_path: Path,
) -> None:
    config = br.verify_brainrw_config(
        _write_config(configs_dir, tmp_path),
        tmp_path / "clip",
    )
    manifest = _identity()
    partial = _complete_checkpoint_payload(config, manifest)
    validated_partial = br.validate_brainrw_checkpoint_identity(
        partial,
        config=config,
        manifest=manifest,
        subject=1,
        seed=42,
    )
    assert validated_partial["training_complete"] is False

    terminal = copy.deepcopy(partial)
    sampler = br.StatefulIndexSampler(2, 42)
    for _ in range(24):
        tuple(iter(sampler))
        sampler.advance_epoch()
    tuple(iter(sampler))
    terminal.update(
        {
            "epoch": 24,
            "global_step": 25,
            "steps": 25,
            "sampler_state": sampler.state_dict(),
            "training_complete": True,
        }
    )
    validated_terminal = br.validate_brainrw_checkpoint_identity(
        terminal,
        config=config,
        manifest=manifest,
        subject=1,
        seed=42,
    )
    assert validated_terminal["training_complete"] is True

    terminal["training_complete"] = False
    with pytest.raises(ValueError, match="training_complete"):
        br.validate_brainrw_checkpoint_identity(
            terminal,
            config=config,
            manifest=manifest,
            subject=1,
            seed=42,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "missing-optimizer",
        "config-payload",
        "input-bundle",
        "input-hashes",
        "run-key",
    ),
)
def test_complete_checkpoint_identity_rejects_every_semantic_mismatch(
    configs_dir: Path,
    tmp_path: Path,
    mutation: str,
) -> None:
    config = br.verify_brainrw_config(
        _write_config(configs_dir, tmp_path),
        tmp_path / "clip",
    )
    manifest = _identity()
    payload = _complete_checkpoint_payload(config, manifest)
    validated = br.validate_brainrw_checkpoint_identity(
        payload,
        config=config,
        manifest=manifest,
        subject=1,
        seed=42,
    )
    assert validated["run_key"] == payload["run_key"]
    payload = copy.deepcopy(payload)
    if mutation == "missing-optimizer":
        del payload["optimizer_state"]
    elif mutation == "config-payload":
        payload["config_payload"]["config_id"] = "tampered"
    elif mutation == "input-bundle":
        payload["input_bundle_sha256"] = "0" * 64
    elif mutation == "input-hashes":
        payload["input_hashes"]["source_payload"] = "0" * 64
    else:
        payload["run_key"] = "tampered"
    with pytest.raises(ValueError, match="checkpoint"):
        br.validate_brainrw_checkpoint_identity(
            payload,
            config=config,
            manifest=manifest,
            subject=1,
            seed=42,
        )


def test_checkpoint_rejects_wrapped_layer_count_inconsistent_with_lora_targets(
    configs_dir: Path,
    tmp_path: Path,
) -> None:
    config = br.verify_brainrw_config(
        _write_config(configs_dir, tmp_path),
        tmp_path / "clip",
    )
    manifest = _identity()
    payload = copy.deepcopy(
        _complete_checkpoint_payload(config, manifest)
    )
    model_manifest = payload["model_manifest"]
    assert isinstance(model_manifest, dict)
    checkpointing = model_manifest["gradient_checkpointing"]
    assert isinstance(checkpointing, dict)
    checkpointing["wrapped_layer_count"] = 2
    payload["model_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(model_manifest)
    ).hexdigest()

    with pytest.raises(
        ValueError,
        match="gradient-checkpoint.*LoRA targets",
    ):
        br.validate_brainrw_checkpoint_identity(
            payload,
            config=config,
            manifest=manifest,
            subject=1,

            seed=42,
        )
def test_checkpoint_save_validates_complete_payload_before_writing(
    configs_dir: Path,
    tmp_path: Path,
) -> None:
    config = br.verify_brainrw_config(
        _write_config(configs_dir, tmp_path),
        tmp_path / "clip",
    )
    manifest = _identity()
    payload = _complete_checkpoint_payload(config, manifest)
    del payload["optimizer_state"]
    checkpoint = tmp_path / "checkpoint.pt"
    with pytest.raises(ValueError, match="checkpoint"):
        br.save_brainrw_checkpoint(
            checkpoint,
            payload,
            manifest,
        )
    assert not checkpoint.exists()
    assert not br.checkpoint_sidecar(checkpoint).exists()


def test_checkpoint_binds_exact_production_runtime_attestation(
    configs_dir: Path,
    tmp_path: Path,
) -> None:
    config = br.verify_brainrw_config(
        _write_config(configs_dir, tmp_path),
        tmp_path / "clip",
    )
    manifest = _identity()
    payload = _complete_checkpoint_payload(config, manifest)

    validated = br.validate_brainrw_checkpoint_identity(
        payload,
        config=config,
        manifest=manifest,
        subject=1,
        seed=42,
    )
    assert validated["runtime_contract"]["accelerator"] == "NVIDIA A40"
    assert validated["environment"] == validated["semantic_environment"]

    tampered = copy.deepcopy(payload)
    tampered["runtime_evidence"]["accelerator_name"] = "NVIDIA A800"
    tampered["runtime_evidence_sha256"] = hashlib.sha256(
        canonical_json_bytes(tampered["runtime_evidence"])
    ).hexdigest()
    with pytest.raises(ValueError, match="runtime"):
        br.validate_brainrw_checkpoint_identity(
            tampered,
            config=config,
            manifest=manifest,
            subject=1,
            seed=42,
        )

    semantic_tamper = copy.deepcopy(payload)
    semantic_tamper["semantic_environment"]["transformers"] = "0.0.0"
    with pytest.raises(ValueError, match="semantic environment"):
        br.validate_brainrw_checkpoint_identity(
            semantic_tamper,
            config=config,
            manifest=manifest,
            subject=1,
            seed=42,
        )


def test_checkpoint_reconstruction_binds_clip_preprocessor_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clip = tmp_path / "clip"
    clip.mkdir()
    config_file = clip / "config.json"
    weights_file = clip / "model.safetensors"
    preprocessor_file = clip / "preprocessor_config.json"
    config_file.write_bytes(b"config")
    weights_file.write_bytes(b"weights")
    preprocessor_file.write_bytes(b"preprocessor")
    model = _model()
    payload = {
        "clip_path": str(clip),
        "clip_config_sha256": hashlib.sha256(
            config_file.read_bytes()
        ).hexdigest(),
        "clip_preprocessor_sha256": "0" * 64,
        "clip_weights_sha256": hashlib.sha256(
            weights_file.read_bytes()
        ).hexdigest(),
        "config_payload": {},
        "model_manifest_sha256": model.model_manifest_sha256,
        "task_state": model.task_state_dict(),
        "candidate_state": model.candidate_state_dict(),
    }
    monkeypatch.setattr(
        br,
        "build_brainrw_model",
        lambda *_a, **_k: (model, _FakeProcessor()),
    )
    with pytest.raises(ValueError, match="preprocessor"):
        br.build_model_from_checkpoint(payload)


class _TinyDataset:
    def __init__(
        self,
        _: Path,
        scope: str,
        seed: int,
        *,
        expected_source_payload_sha256: str | None = None,
    ) -> None:
        assert scope in {"train", "val-dev"}
        assert seed in {42, 43}
        if expected_source_payload_sha256 is not None:
            assert expected_source_payload_sha256 == "a" * 64
        self.scope = scope
        self.subject_id = 1
        self.row_indices = (10, 11)
        self.ordered_ids = (
            ("concept-a", "concept-b") if scope == "train" else ("image-a", "image-b")
        )
        self.query_ids = self.ordered_ids if scope == "val-dev" else ()
        self.gallery_ids = self.query_ids

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict[str, object]:
        identifier = (
            f"train-image-{index}" if self.scope == "train" else f"image-{chr(97 + index)}"
        )
        return {
            "concept_id": f"concept-{index}",
            "eeg": torch.arange(4_250).reshape(17, 250).float() + index,
            "image": torch.arange(8).float() + index,
            "image_id": identifier,
            "image_path": f"/safe/training_images/{identifier}.jpg",
            "row_index": 10 + index,
            "scope": self.scope,
            "subject_id": 1,
        }


def test_reload_evaluation_uses_explicit_checkpoint_numerical_dtype() -> None:
    model = _model(channels=17, samples=250)
    similarity, identifiers = br.evaluate_brainrw_similarity(
        model,
        _TinyDataset(Path("ignored"), "val-dev", 42),
        _FakeProcessor(),
        batch_size=2,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    assert similarity.shape == (2, 2)
    assert identifiers == ("image-a", "image-b")
    assert {
        parameter.dtype
        for parameter in model.parameters()
        if parameter.is_floating_point()
    } == {torch.bfloat16}


def test_checkpoint_runtime_has_no_public_cpu_or_float32_fallback() -> None:
    assert not hasattr(br, "checkpoint_runtime_dtype")


def test_training_smoke_score_artifact_binds_nonterminal_progress(
    tmp_path: Path,
) -> None:
    similarity = np.eye(2, dtype=np.float32)
    metadata = {
        "checkpoint_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "git_sha": "c" * 40,
        "global_step": 1,
        "planned_steps": 25,
        "protocol_sha256": "d" * 64,
        "seed": 42,
        "source_records": [
            {
                "manifest_sha256": "1" * 64,
                "records_sha256": "2" * 64,
                "role": "val-dev",
                "role_payload_sha256": "3" * 64,
                "source_manifest_sha256": "4" * 64,
                "source_payload_byte_count": 123,
                "source_payload_path": "/safe/sub-01/train.pt",
                "source_payload_sha256": "5" * 64,
            }
        ],
        "split_role": "val-dev",
        "stage": "training_smoke/in_loop",
        "subject": 1,
        "training_complete": False,
    }
    output = tmp_path / "training_smoke" / "in_loop"

    ScoreArtifact.save(
        output,
        similarity,
        ("image-a", "image-b"),
        ("image-a", "image-b"),
        metadata,
    )
    loaded = ScoreArtifact.load(output, {"val-dev"})

    for values in (loaded.metadata, loaded.provenance):
        assert values["stage"] == "training_smoke/in_loop"
        assert values["training_complete"] is False
        assert values["global_step"] == 1
        assert values["planned_steps"] == 25
        assert values["checkpoint_sha256"] == "a" * 64


def _load_train_script(experiment_root: Path):
    path = experiment_root / "train_brainrw.py"
    spec = importlib.util.spec_from_file_location("task12_train_brainrw", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_production_runtime_probe_locks_cuda_bfloat16_and_a40(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        br,
        "capture_semantic_environment",
        lambda: dict(PINNED_SEMANTIC_ENVIRONMENT),
        raising=False,
    )
    properties = SimpleNamespace(
        name="NVIDIA A40",
        major=8,
        minor=6,
        total_memory=48 * 1024**3,
    )
    monkeypatch.setattr(br.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(br.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(br.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        br.torch.cuda,
        "get_device_properties",
        lambda _index: properties,
    )
    monkeypatch.setattr(
        br.torch.cuda,
        "is_bf16_supported",
        lambda: True,
    )

    runtime = br.probe_brainrw_production_runtime()

    assert runtime.device == torch.device("cuda", 0)
    assert runtime.dtype is torch.bfloat16
    assert dict(runtime.semantic_environment) == (
        PINNED_SEMANTIC_ENVIRONMENT
    )
    assert runtime.semantic_environment_sha256 == hashlib.sha256(
        canonical_json_bytes(PINNED_SEMANTIC_ENVIRONMENT)
    ).hexdigest()
    assert dict(runtime.contract) == {
        "accelerator": "NVIDIA A40",
        "device_type": "cuda",
        "dtype": "bfloat16",
        "schema_version": 1,
    }
    assert runtime.evidence["accelerator_name"] == "NVIDIA A40"
    assert runtime.evidence["bf16_supported"] is True
    assert runtime.contract_sha256 == hashlib.sha256(
        canonical_json_bytes(dict(runtime.contract))
    ).hexdigest()
    assert runtime.evidence_sha256 == hashlib.sha256(
        canonical_json_bytes(dict(runtime.evidence))
    ).hexdigest()

    monkeypatch.setattr(br.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        br.probe_brainrw_production_runtime()

    monkeypatch.setattr(br.torch.cuda, "is_available", lambda: True)
    properties.name = "NVIDIA A800-SXM4-80GB"
    with pytest.raises(RuntimeError, match="A40"):
        br.probe_brainrw_production_runtime()


def test_runtime_probe_rejects_semantic_mismatch_before_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = dict(PINNED_SEMANTIC_ENVIRONMENT)
    environment["transformers"] = "0.0.0"
    monkeypatch.setattr(
        br,
        "capture_semantic_environment",
        lambda: environment,
        raising=False,
    )
    cuda_touched = False

    def forbidden_cuda() -> bool:
        nonlocal cuda_touched
        cuda_touched = True
        raise AssertionError("CUDA must follow semantic preflight")

    monkeypatch.setattr(br.torch.cuda, "is_available", forbidden_cuda)
    with pytest.raises(ValueError, match="transformers"):
        br.probe_brainrw_production_runtime()

def test_runtime_preflight_fails_before_config_model_or_dataset(
    experiment_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = _load_train_script(experiment_root)
    touched: list[str] = []

    def forbidden(name: str):
        def fail(*_args: object, **_kwargs: object) -> None:
            touched.append(name)
            raise AssertionError(f"{name} must follow runtime preflight")

        return fail

    monkeypatch.setattr(
        br,
        "probe_brainrw_production_runtime",
        lambda: (_ for _ in ()).throw(RuntimeError("CUDA unavailable")),
        raising=False,
    )
    monkeypatch.setattr(br, "verify_brainrw_config", forbidden("config"))
    monkeypatch.setattr(br, "build_brainrw_model", forbidden("model"))
    monkeypatch.setattr(
        br,
        "BrainRWDevelopmentDataset",
        forbidden("dataset"),
    )
    monkeypatch.setattr(
        train,
        "_git_provenance",
        lambda: {
            "clean": True,
            "git_sha": "c" * 40,
            "repository_root": str(experiment_root.parents[1]),
        },
    )
    args = train.parse_args(
        [
            "--scope", "train",
            "--validation-scope", "val-dev",
            "--subject", "1",
            "--seed", "42",
            "--resume", "none",
            "--config", str(tmp_path / "config.json"),
            "--manifest", str(tmp_path / "sub-01_protocol.json"),
            "--clip-path", str(tmp_path / "clip"),
            "--output-dir", str(tmp_path / "run"),
            "--max-train-steps", "1",
        ]
    )

    with pytest.raises(RuntimeError, match="CUDA"):
        train.run(args)
    assert touched == []


def test_git_sha_lookup_fails_closed(
    experiment_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = _load_train_script(experiment_root)

    def unavailable(*_args: object, **_kwargs: object) -> bytes:
        raise OSError("git unavailable")

    monkeypatch.setattr(train.subprocess, "check_output", unavailable)
    with pytest.raises(RuntimeError, match="Git SHA"):
        train._git_sha()

    monkeypatch.setattr(
        train.subprocess,
        "check_output",
        lambda *_args, **_kwargs: "not-a-commit\n",
    )
    with pytest.raises(ValueError, match="Git SHA"):
        train._git_sha()


def test_git_provenance_is_repo_anchored_and_requires_clean_tree(
    experiment_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = _load_train_script(experiment_root)
    repository_root = experiment_root.parents[1]
    revision = "a" * 40
    calls: list[tuple[str, ...]] = []

    def clean_output(
        args: list[str],
        **_kwargs: object,
    ) -> str:
        command = tuple(args)
        calls.append(command)
        prefix = ("git", "-C", str(repository_root))
        assert command[:3] == prefix
        suffix = command[3:]
        if suffix == ("rev-parse", "--show-toplevel"):
            return str(repository_root) + "\n"
        if suffix == ("rev-parse", "HEAD"):
            return revision + "\n"
        if suffix == (
            "status",
            "--porcelain",
            "--untracked-files=all",
        ):
            return ""
        raise AssertionError(f"unexpected git command: {command}")

    monkeypatch.setattr(
        train.subprocess,
        "check_output",
        clean_output,
    )
    assert train._git_provenance() == {
        "clean": True,
        "git_sha": revision,
        "repository_root": str(repository_root),
    }
    assert len(calls) == 3

    def dirty_output(
        args: list[str],
        **kwargs: object,
    ) -> str:
        if args[3] == "status":
            return " M tracked.py\n"
        return clean_output(args, **kwargs)

    monkeypatch.setattr(
        train.subprocess,
        "check_output",
        dirty_output,
    )
    with pytest.raises(RuntimeError, match="clean"):
        train._git_provenance()


def test_resume_runtime_state_fails_before_eeg_dataset_construction(
    configs_dir: Path,
    experiment_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_config(configs_dir, tmp_path)
    config = br.verify_brainrw_config(
        config_path,
        tmp_path / "clip",
    )
    manifest = _identity()
    payload = _complete_checkpoint_payload(config, manifest)
    payload["task_state"] = {
        "wrong.task.key": torch.zeros(1),
    }
    loaded = br.LoadedBrainRWCheckpoint(
        payload=payload,
        sha256="d" * 64,
    )
    monkeypatch.setattr(
        br,
        "load_development_manifest_identity",
        lambda *_a, **_k: manifest,
    )
    monkeypatch.setattr(
        br,
        "load_brainrw_checkpoint",
        lambda *_a, **_k: loaded,
    )
    monkeypatch.setattr(
        br,
        "build_brainrw_model",
        lambda *_a, **_k: (
            _model(
                channels=17,
                samples=250,
                lora_rank=32,
                lora_alpha=32,
            ),
            _FakeProcessor(),
        ),
    )
    touched = False

    class ForbiddenDataset:
        def __init__(self, *_a: object, **_k: object) -> None:
            nonlocal touched
            touched = True
            raise AssertionError(
                "malformed resume state must fail before EEG loading"
            )

    monkeypatch.setattr(
        br,
        "BrainRWDevelopmentDataset",
        ForbiddenDataset,
    )
    train = _load_train_script(experiment_root)
    monkeypatch.setattr(
        br,
        "probe_brainrw_production_runtime",
        _fake_cpu_runtime,
    )
    monkeypatch.setattr(
        train,
        "_git_provenance",
        lambda: dict(payload["git_provenance"]),
    )
    args = train.parse_args(
        [
            "--scope", "train",
            "--validation-scope", "val-dev",
            "--subject", "1",
            "--seed", "42",
            "--resume", str(tmp_path / "resume.pt"),
            "--config", str(config_path),
            "--manifest", str(tmp_path / "sub-01_protocol.json"),
            "--clip-path", str(tmp_path / "clip"),
            "--output-dir", str(tmp_path / "never-created"),
            "--max-train-steps", "2",
        ]
    )
    with pytest.raises(ValueError, match="task-state"):
        train.run(args)
    assert touched is False


def test_resume_rejects_empty_adamw_state_for_positive_step(
    experiment_root: Path,
) -> None:
    train = _load_train_script(experiment_root)
    model = _model(
        channels=17,
        samples=250,
        lora_rank=32,
        lora_alpha=32,
    )
    optimizer = train._build_optimizer(
        model,
        {
            "brain_learning_rate": 0.0005,
            "visual_learning_rate": 0.00005,
            "weight_decay": 0.05,
        },
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=25,
    )
    sampler = br.StatefulIndexSampler(2, 42)
    payload = {
        "task_state": model.task_state_dict(),
        "candidate_state": model.candidate_state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "sampler_state": sampler.state_dict(),
        "dataloader_generator_state": (
            torch.Generator().manual_seed(1).get_state()
        ),
        "rng_state": br.capture_rng_state(),
    }
    with pytest.raises(
        ValueError,
        match="optimizer.*state",
    ):
        train._load_resume_runtime_before_data(
            payload,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            loader_generator=torch.Generator(),
            device=torch.device("cpu"),
            planned_steps=25,
            global_step=1,
        )


def test_cpu_one_step_smoke_persists_complete_resume_state_and_hashes(
    configs_dir: Path,
    experiment_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _write_config(configs_dir, tmp_path)
    manifest = tmp_path / "sub-01_protocol.json"
    manifest.write_text("preflight is monkeypatched\n", encoding="utf-8")
    real_torch_load = torch.load
    weights_only_modes: list[object] = []

    def torch_load_spy(*args: object, **kwargs: object) -> object:
        weights_only_modes.append(kwargs.get("weights_only"))
        return real_torch_load(*args, **kwargs)

    monkeypatch.setattr(
        br.torch,
        "load",
        torch_load_spy,
    )
    monkeypatch.setattr(br, "load_development_manifest_identity", lambda *_a, **_k: _identity())
    monkeypatch.setattr(br, "BrainRWDevelopmentDataset", _TinyDataset)
    monkeypatch.setattr(
        br,
        "build_brainrw_model",
        lambda *_a, **_k: (
            _model(
                channels=17,
                samples=250,
                lora_rank=32,
                lora_alpha=32,
            ),
            _FakeProcessor(),
        ),
    )
    train = _load_train_script(experiment_root)
    monkeypatch.setattr(
        br,
        "probe_brainrw_production_runtime",
        _fake_cpu_runtime,
    )
    provenance = {
        "clean": True,
        "git_sha": "c" * 40,
        "repository_root": str(
            experiment_root.parents[1]
        ),
    }
    git_checks: list[dict[str, object]] = []

    def stable_git() -> dict[str, object]:
        git_checks.append(provenance)
        return provenance.copy()

    monkeypatch.setattr(train, "_git_provenance", stable_git)
    first = tmp_path / "run-first"
    arguments = [
        "--scope", "train",
        "--validation-scope", "val-dev",
        "--subject", "1",
        "--seed", "42",
        "--resume", "none",
        "--config", str(config),
        "--manifest", str(manifest),
        "--clip-path", str(tmp_path / "clip"),
        "--output-dir", str(first),
        "--max-train-steps", "1",
    ]
    assert train.main(arguments) == 0
    checkpoint_path = first / "checkpoint.pt"
    assert checkpoint_path.is_file()
    checkpoint_sidecar = checkpoint_path.with_suffix(".pt.meta.json")
    assert checkpoint_sidecar.is_file()
    loaded = br.load_brainrw_checkpoint(checkpoint_path, requested_scope="train")
    envelope = json.loads(
        checkpoint_sidecar.read_text(encoding="utf-8")
    )
    assert envelope["metadata"]["training_complete"] is False
    assert envelope["metadata"]["global_step"] == 1
    assert envelope["metadata"]["planned_steps"] == 25
    source_record = envelope["metadata"]["source_records"][0]
    assert source_record["source_manifest_sha256"] == "4" * 64
    assert source_record["source_payload_sha256"] == "a" * 64
    assert source_record["source_payload_path"] == (
        "/safe/sub-01/train.pt"
    )
    assert source_record["source_payload_byte_count"] == 123
    payload = loaded.payload
    assert payload["global_step"] == 1
    assert payload["training_complete"] is False
    assert payload["subject"] == 1 and payload["seed"] == 42
    assert payload["scope"] == "train"
    assert payload["validation_scope"] == "val-dev"
    for key in (
        "task_state",
        "candidate_state",
        "optimizer_state",
        "scheduler_state",
        "rng_state",
        "sampler_state",
        "dataloader_generator_state",
        "data_order_sha256",
        "effective_batch_size",
        "environment",
        "git_provenance",
        "git_sha",
        "input_hashes",
        "model_manifest",
        "runtime_dtype",
        "validation_metrics",
        "run_key",
        "clip_preprocessor_sha256",
    ):
        assert key in payload
    runtime_attestation = _runtime_attestation()
    assert payload["runtime_dtype"] == "bfloat16"
    assert payload["runtime_contract"] == runtime_attestation["runtime_contract"]
    assert payload["runtime_contract_sha256"] == (
        runtime_attestation["runtime_contract_sha256"]
    )
    assert payload["runtime_evidence"] == runtime_attestation["runtime_evidence"]
    assert payload["runtime_evidence_sha256"] == (
        runtime_attestation["runtime_evidence_sha256"]
    )
    assert payload["git_provenance"] == provenance
    assert payload["input_hashes"]["source_payload"] == "a" * 64
    assert set(payload["observed_scopes"]) == {"train", "val-dev"}
    run_manifest = json.loads((first / "run_manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["run_key"] == payload["run_key"]
    assert run_manifest["training_complete"] is False
    assert run_manifest["subject"] == 1 and run_manifest["seed"] == 42
    assert run_manifest["runtime_contract_sha256"] == (
        runtime_attestation["runtime_contract_sha256"]
    )
    assert run_manifest["runtime_evidence_sha256"] == (
        runtime_attestation["runtime_evidence_sha256"]
    )
    assert run_manifest["training_smoke_score_directory"] == (
        "training_smoke/in_loop"
    )
    smoke_score = ScoreArtifact.load(
        first / "training_smoke" / "in_loop",
        {"val-dev"},
    )
    assert smoke_score.provenance["checkpoint_sha256"] == loaded.sha256
    assert smoke_score.provenance["stage"] == "training_smoke/in_loop"
    assert smoke_score.provenance["training_complete"] is False
    assert smoke_score.provenance["global_step"] == 1
    assert smoke_score.provenance["planned_steps"] == 25

    resumed = tmp_path / "run-resumed"
    resumed_args = arguments.copy()
    resumed_args[resumed_args.index("none")] = str(checkpoint_path)
    resumed_args[resumed_args.index(str(first))] = str(resumed)
    resumed_args[resumed_args.index("1", resumed_args.index("--max-train-steps"))] = "2"
    assert train.main(resumed_args) == 0
    resumed_payload = br.load_brainrw_checkpoint(
        resumed / "checkpoint.pt", requested_scope="train"
    ).payload
    assert resumed_payload["global_step"] == 2
    assert resumed_payload["training_complete"] is False
    assert len(git_checks) == 4
    assert weights_only_modes
    assert all(mode is True for mode in weights_only_modes)

    observed_scopes: list[str] = []

    class _TrackingDataset(_TinyDataset):
        def __init__(self, *args: object, **kwargs: object) -> None:
            observed_scopes.append(str(args[1]))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(br, "BrainRWDevelopmentDataset", _TrackingDataset)
    train_only_output = tmp_path / "run-train-only"
    train_only_args = arguments.copy()
    train_only_args[train_only_args.index("val-dev")] = "none"
    train_only_args[train_only_args.index(str(first))] = str(train_only_output)
    assert train.main(train_only_args) == 0
    train_only_payload = br.load_brainrw_checkpoint(
        train_only_output / "checkpoint.pt", requested_scope="train"
    ).payload
    assert train_only_payload["validation_scope"] == "none"
    assert train_only_payload["observed_scopes"] == ["train"]
    assert train_only_payload["validation_metrics"] == {
        "performed": False,
        "validation_scope": "none",
    }
    train_only_manifest = json.loads(
        (train_only_output / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert train_only_manifest["training_smoke_score_directory"] is None
    assert not (train_only_output / "training_smoke").exists()
    assert observed_scopes == ["train"]
