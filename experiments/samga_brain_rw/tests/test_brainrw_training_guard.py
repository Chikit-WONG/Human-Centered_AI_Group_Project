from __future__ import annotations

import copy
import hashlib
import io
import importlib.util
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from samga_brain_rw import brainrw as br
from samga_brain_rw.data import POSTERIOR_CHANNELS
from samga_brain_rw.hashing import canonical_json_bytes, ordered_ids_sha256


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
        self.vision_model = nn.Module()
        self.vision_model.encoder = nn.Module()
        self.vision_model.encoder.layers = nn.ModuleList([_FakeBlock()])
        if include_visual_projection:
            self.visual_projection = nn.Linear(8, 8, bias=False)
        self.config = SimpleNamespace(projection_dim=8)

    def forward(self, pixel_values: torch.Tensor, **_: object) -> SimpleNamespace:
        values = pixel_values.reshape(pixel_values.shape[0], 8)
        values = self.vision_model.encoder.layers[0](values)
        return SimpleNamespace(image_embeds=self.visual_projection(values))


class _FakeProcessor:
    def __call__(self, *, images: list[torch.Tensor], return_tensors: str):
        assert return_tensors == "pt"
        return {"pixel_values": torch.stack(images).float()}


def _model(
    *, channels: int = 2, samples: int = 4
) -> br.BrainRWCLIPLoRAModel:
    torch.manual_seed(5)
    return br.BrainRWCLIPLoRAModel(
        _FakeVision(),
        channels=channels,
        samples=samples,
        projection_dim=8,
        dropout=0.1,
        lora_rank=2,
        lora_alpha=2,
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
    ) -> None:
        self.calls.append(
            (
                manifest_path,
                scope,
                seed,
                selected_channels,
                feature_cache,
                smooth_probability,
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
        train_role_sha256="5" * 64,
        val_dev_role_sha256="6" * 64,
        train_ordered_ids=train_ids,
        val_dev_ordered_ids=val_ids,
        train_ordered_ids_sha256=ordered_ids_sha256(train_ids),
        val_dev_ordered_ids_sha256=ordered_ids_sha256(val_ids),
    )


class _TinyDataset:
    def __init__(self, _: Path, scope: str, seed: int) -> None:
        assert scope in {"train", "val-dev"}
        assert seed in {42, 43}
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


def _load_train_script(experiment_root: Path):
    path = experiment_root / "train_brainrw.py"
    spec = importlib.util.spec_from_file_location("task12_train_brainrw", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
            _model(channels=17, samples=250),
            _FakeProcessor(),
        ),
    )
    train = _load_train_script(experiment_root)
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
    assert checkpoint_path.with_suffix(".pt.meta.json").is_file()
    loaded = br.load_brainrw_checkpoint(checkpoint_path, requested_scope="train")
    payload = loaded.payload
    assert payload["global_step"] == 1
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
        "git_sha",
        "input_hashes",
        "model_manifest",
        "validation_metrics",
        "run_key",
    ):
        assert key in payload
    assert set(payload["observed_scopes"]) == {"train", "val-dev"}
    run_manifest = json.loads((first / "run_manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["run_key"] == payload["run_key"]
    assert run_manifest["subject"] == 1 and run_manifest["seed"] == 42

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
    assert weights_only_modes
    assert all(mode is True for mode in weights_only_modes)
