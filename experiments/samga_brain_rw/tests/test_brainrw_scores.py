from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from samga_brain_rw import brainrw as br
from samga_brain_rw.hashing import ordered_ids_sha256
from samga_brain_rw.scores import ScoreArtifact


class _FakeBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(8, 8)
        self.k_proj = torch.nn.Linear(8, 8)
        self.v_proj = torch.nn.Linear(8, 8)
        self.out_proj = torch.nn.Linear(8, 8)
        self.fc1 = torch.nn.Linear(8, 16)
        self.fc2 = torch.nn.Linear(16, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.q_proj(x) + self.k_proj(x) + self.v_proj(x)
        return self.fc2(torch.nn.functional.gelu(self.fc1(self.out_proj(x))))


class _FakeVision(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision_model = torch.nn.Module()
        self.vision_model.encoder = torch.nn.Module()
        self.vision_model.encoder.layers = torch.nn.ModuleList([_FakeBlock()])
        self.visual_projection = torch.nn.Linear(8, 8, bias=False)
        self.config = type("Config", (), {"projection_dim": 8})()

    def forward(self, pixel_values: torch.Tensor, **_: object):
        x = pixel_values.reshape(pixel_values.shape[0], 8)
        x = self.vision_model.encoder.layers[0](x)
        return type("Output", (), {"image_embeds": self.visual_projection(x)})()


class _Processor:
    def __call__(self, *, images: list[torch.Tensor], return_tensors: str):
        assert return_tensors == "pt"
        return {"pixel_values": torch.stack(images).float()}


def _model(
    *, channels: int = 2, samples: int = 4
) -> br.BrainRWCLIPLoRAModel:
    torch.manual_seed(9)
    model = br.BrainRWCLIPLoRAModel(
        _FakeVision(),
        channels=channels,
        samples=samples,
        projection_dim=8,
        dropout=0.1,
        lora_rank=2,
        lora_alpha=2,
        lora_dropout=0.0,
    )
    model.eval()
    return model


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


class _ValDataset:
    calls = 0

    def __init__(self, _: Path, scope: str, seed: int) -> None:
        type(self).calls += 1
        assert scope == "val-dev" and seed == 42
        self.scope = scope
        self.subject_id = 1
        self.ordered_ids = ("image-a", "image-b")
        self.query_ids = self.gallery_ids = self.ordered_ids
        self.row_indices = (20, 21)

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "concept_id": f"concept-{index}",
            "eeg": torch.arange(4_250).reshape(17, 250).float() + index,
            "image": torch.arange(8).float() + index,
            "image_id": self.ordered_ids[index],
            "image_path": f"/safe/training_images/{self.ordered_ids[index]}.jpg",
            "row_index": self.row_indices[index],
            "scope": "val-dev",
            "subject_id": 1,
        }


def _checkpoint(model: br.BrainRWCLIPLoRAModel, *, subject: int = 1) -> dict[str, object]:
    return {
        "schema_version": 1,
        "payload_type": br.BRAINRW_CHECKPOINT_TYPE,
        "complete": True,
        "scope": "train",
        "validation_scope": "val-dev",
        "subject": subject,
        "seed": 42,
        "config_sha256": "7" * 64,
        "manifest_sha256": "1" * 64,
        "protocol_sha256": "2" * 64,
        "clip_path": "/fake/clip",
        "clip_config_sha256": "8" * 64,
        "clip_weights_sha256": "9" * 64,
        "task_state": model.task_state_dict(),
        "candidate_state": model.candidate_state_dict(),
        "model_manifest": model.model_manifest,
        "model_manifest_sha256": model.model_manifest_sha256,
        "git_sha": "a" * 40,
        "run_key": "brainrw-run",
        "observed_scopes": ["train", "val-dev"],
    }


def _load_script(experiment_root: Path):
    path = experiment_root / "scripts" / "emit_brainrw_scores.py"
    spec = importlib.util.spec_from_file_location("task12_emit_brainrw_scores", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_emitter_writes_only_typed_val_dev_independent_scores(
    experiment_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emit = _load_script(experiment_root)
    manifest = tmp_path / "sub-01_protocol.json"
    manifest.write_text("verified by monkeypatch\n", encoding="utf-8")
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"verified by monkeypatch")
    model = _model(channels=17, samples=250)
    loaded = br.LoadedBrainRWCheckpoint(
        payload=_checkpoint(model),
        sha256="b" * 64,
    )
    monkeypatch.setattr(br, "load_development_manifest_identity", lambda *_a, **_k: _identity())
    monkeypatch.setattr(br, "load_brainrw_checkpoint", lambda *_a, **_k: loaded)
    monkeypatch.setattr(br, "BrainRWDevelopmentDataset", _ValDataset)
    monkeypatch.setattr(br, "build_model_from_checkpoint", lambda *_a, **_k: (model, _Processor()))
    output = tmp_path / "scores"
    assert emit.main([
        "--scope", "val-dev", "--subject", "1", "--seed", "42",
        "--checkpoint", str(checkpoint_path),
        "--manifest", str(manifest), "--output-dir", str(output),
    ]) == 0
    artifact = ScoreArtifact.load(output, allowed_scopes={"val-dev"})
    assert artifact.scope == "val-dev"
    assert artifact.similarity.shape == (2, 2)
    assert artifact.query_ids == artifact.gallery_ids == ("image-a", "image-b")
    assert artifact.metadata["subject"] == 1
    assert artifact.metadata["seed"] == 42
    assert artifact.metadata["split_role"] == "val-dev"
    assert artifact.provenance["checkpoint_sha256"] == "b" * 64


@pytest.mark.parametrize("scope", ["test", "formal-test", "val-confirm"])
def test_emitter_cli_has_no_non_development_scope(
    experiment_root: Path,
    scope: str,
) -> None:
    emit = _load_script(experiment_root)
    with pytest.raises(SystemExit):
        emit.main([
            "--scope", scope, "--subject", "1", "--seed", "42",
            "--checkpoint", "checkpoint.pt", "--manifest", "manifest.json",
            "--output-dir", "scores",
        ])


def test_renamed_or_metadata_free_artifacts_fail_before_eeg_or_image_loading(
    experiment_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emit = _load_script(experiment_root)
    touched = False

    class ForbiddenDataset:
        def __init__(self, *_a: object, **_k: object) -> None:
            nonlocal touched
            touched = True
            raise AssertionError("EEG/image loading must not start")

    monkeypatch.setattr(br, "BrainRWDevelopmentDataset", ForbiddenDataset)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"subject": 1, "seed": 42}, checkpoint)
    renamed = tmp_path / "sub-01_test.json"
    renamed.write_text('{"split":"test"}\n', encoding="utf-8")
    base = [
        "--scope", "val-dev", "--subject", "1", "--seed", "42",
        "--checkpoint", str(checkpoint), "--manifest", str(renamed),
        "--output-dir", str(tmp_path / "scores-a"),
    ]
    with pytest.raises(SystemExit):
        emit.main(base)
    assert touched is False

    manifest = tmp_path / "sub-01_protocol.json"
    manifest.write_text('{"split":"test"}\n', encoding="utf-8")
    base[base.index(str(renamed))] = str(manifest)
    base[base.index(str(tmp_path / "scores-a"))] = str(tmp_path / "scores-b")
    with pytest.raises(SystemExit):
        emit.main(base)
    assert touched is False

    monkeypatch.setattr(br, "load_development_manifest_identity", lambda *_a, **_k: _identity())
    base[base.index(str(tmp_path / "scores-b"))] = str(tmp_path / "scores-c")
    with pytest.raises(SystemExit):
        emit.main(base)
    assert touched is False


def test_checkpoint_identity_and_existing_output_fail_before_dataset(
    experiment_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emit = _load_script(experiment_root)
    manifest = tmp_path / "sub-01_protocol.json"
    manifest.write_text("verified by monkeypatch\n", encoding="utf-8")
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"verified by monkeypatch")
    model = _model()
    loaded = br.LoadedBrainRWCheckpoint(_checkpoint(model, subject=2), "b" * 64)
    monkeypatch.setattr(br, "load_development_manifest_identity", lambda *_a, **_k: _identity())
    monkeypatch.setattr(br, "load_brainrw_checkpoint", lambda *_a, **_k: loaded)
    touched = False

    class ForbiddenDataset:
        def __init__(self, *_a: object, **_k: object) -> None:
            nonlocal touched
            touched = True
