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
        source_payload_path=Path("/safe/sub-01/train.pt"),
        source_payload_sha256="a" * 64,
        source_payload_byte_count=123,
        train_role_sha256="5" * 64,
        val_dev_role_sha256="6" * 64,
        train_ordered_ids=train_ids,
        val_dev_ordered_ids=val_ids,
        train_ordered_ids_sha256=ordered_ids_sha256(train_ids),
        val_dev_ordered_ids_sha256=ordered_ids_sha256(val_ids),
    )


class _ValDataset:
    calls: list[str | None] = []

    def __init__(
        self,
        _: Path,
        scope: str,
        seed: int,
        *,
        expected_source_payload_sha256: str | None = None,
    ) -> None:
        type(self).calls.append(
            expected_source_payload_sha256
        )
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
        "training_complete": True,
        "global_step": 25,
        "planned_steps": 25,
        "scope": "train",
        "validation_scope": "val-dev",
        "subject": subject,
        "seed": 42,
        "config_path": "/fake/brainrw.json",
        "config_payload": {
            "training": {"batch_size": 2},
        },
        "config_sha256": "7" * 64,
        "manifest_sha256": "1" * 64,
        "protocol_sha256": "2" * 64,
        "clip_path": "/fake/clip",
        "clip_config_sha256": "8" * 64,
        "clip_preprocessor_sha256": "0" * 64,
        "clip_weights_sha256": "9" * 64,
        "task_state": model.task_state_dict(),
        "candidate_state": model.candidate_state_dict(),
        "model_manifest": model.model_manifest,
        "model_manifest_sha256": model.model_manifest_sha256,
        "git_sha": "a" * 40,
        "git_provenance": {
            "clean": True,
            "git_sha": "a" * 40,
            "repository_root": str(
                Path(__file__).resolve().parents[3]
            ),
        },
        "runtime_dtype": "float32",
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


def test_emitter_git_provenance_is_anchored_and_clean(
    experiment_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emit = _load_script(experiment_root)
    repository_root = experiment_root.parents[1]
    revision = "a" * 40
    calls: list[tuple[str, ...]] = []

    def clean_output(
        args: list[str],
        **_kwargs: object,
    ) -> str:
        command = tuple(args)
        calls.append(command)
        assert command[:3] == (
            "git",
            "-C",
            str(repository_root),
        )
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
        raise AssertionError(
            f"unexpected Git command: {command}"
        )

    monkeypatch.setattr(
        emit.subprocess,
        "check_output",
        clean_output,
    )
    assert emit._git_provenance() == {
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
            return "?? untracked.py\n"
        return clean_output(args, **kwargs)

    monkeypatch.setattr(
        emit.subprocess,
        "check_output",
        dirty_output,

    )
    with pytest.raises(RuntimeError, match="clean"):
        emit._git_provenance()


def test_emitter_rejects_partial_checkpoint_before_dataset(
    experiment_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emit = _load_script(experiment_root)
    model = _model()
    payload = _checkpoint(model)
    payload["training_complete"] = False
    payload["global_step"] = 1
    loaded = br.LoadedBrainRWCheckpoint(
        payload=payload,
        sha256="b" * 64,
    )
    monkeypatch.setattr(
        br,
        "validate_brainrw_checkpoint_identity",
        lambda *_a, **_k: payload,
    )
    with pytest.raises(
        ValueError,
        match="terminal|complete",
    ):
        emit._validate_identities(
            loaded,
            _identity(),
            object(),
            subject=1,
            seed=42,
            git_provenance=payload["git_provenance"],
        )


def test_emitter_rejects_malformed_model_state_before_dataset(
    experiment_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emit = _load_script(experiment_root)
    manifest_path = tmp_path / "sub-01_protocol.json"
    manifest_path.write_text(
        "verified by monkeypatch\n",
        encoding="utf-8",
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"verified by monkeypatch")
    model = _model()
    payload = _checkpoint(model)
    loaded = br.LoadedBrainRWCheckpoint(
        payload=payload,
        sha256="b" * 64,
    )
    monkeypatch.setattr(
        br,
        "load_development_manifest_identity",
        lambda *_a, **_k: _identity(),
    )
    monkeypatch.setattr(
        br,
        "load_brainrw_checkpoint",
        lambda *_a, **_k: loaded,
    )
    monkeypatch.setattr(
        br,
        "verify_brainrw_config",
        lambda *_a, **_k: object(),
    )
    monkeypatch.setattr(
        br,
        "validate_brainrw_checkpoint_identity",
        lambda *_a, **_k: payload,
    )
    touched = False

    class ForbiddenDataset:
        def __init__(self, *_a: object, **_k: object) -> None:
            nonlocal touched
            touched = True
            raise AssertionError(
                "malformed model state must fail before EEG loading"
            )

    monkeypatch.setattr(
        br,
        "BrainRWDevelopmentDataset",
        ForbiddenDataset,
    )
    monkeypatch.setattr(
        br,
        "build_model_from_checkpoint",
        lambda *_a, **_k: (_ for _ in ()).throw(
            ValueError("checkpoint task-state keys differ")
        ),
    )
    monkeypatch.setattr(
        emit,
        "_git_provenance",
        lambda: dict(payload["git_provenance"]),
    )
    args = emit.parse_args(
        [
            "--scope", "val-dev",
            "--subject", "1",
            "--seed", "42",
            "--checkpoint", str(checkpoint_path),
            "--manifest", str(manifest_path),
            "--output-dir", str(tmp_path / "never-created"),
        ]
    )
    with pytest.raises(ValueError, match="task-state"):
        emit.run(args)
    assert touched is False


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
    _ValDataset.calls.clear()
    loaded = br.LoadedBrainRWCheckpoint(
        payload=_checkpoint(model),
        sha256="b" * 64,
    )
    monkeypatch.setattr(br, "load_development_manifest_identity", lambda *_a, **_k: _identity())
    monkeypatch.setattr(br, "load_brainrw_checkpoint", lambda *_a, **_k: loaded)
    config_identity = object()
    monkeypatch.setattr(
        br,
        "verify_brainrw_config",
        lambda *_a, **_k: config_identity,
    )
    validations: list[tuple[object, ...]] = []

    def validate_identity(
        payload: object,
        *,
        config: object,
        manifest: object,
        subject: int,
        seed: int,
    ) -> object:
        assert _ValDataset.calls == []
        validations.append(
            (payload, config, manifest, subject, seed)
        )
        return payload

    monkeypatch.setattr(
        br,
        "validate_brainrw_checkpoint_identity",
        validate_identity,
    )
    monkeypatch.setattr(br, "BrainRWDevelopmentDataset", _ValDataset)
    monkeypatch.setattr(br, "build_model_from_checkpoint", lambda *_a, **_k: (model, _Processor()))
    provenance = {
        "clean": True,
        "git_sha": "a" * 40,
        "repository_root": str(
            experiment_root.parents[1]
        ),
    }
    monkeypatch.setattr(
        emit,
        "_git_provenance",
        lambda: provenance.copy(),
        raising=False,
    )
    real_evaluate = br.evaluate_brainrw_similarity
    evaluation_dtypes: list[object] = []

    def evaluate_spy(*args: object, **kwargs: object):
        evaluation_dtypes.append(kwargs.get("dtype"))
        return real_evaluate(*args, **kwargs)

    monkeypatch.setattr(
        br,
        "evaluate_brainrw_similarity",
        evaluate_spy,
    )
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
    source_record = artifact.metadata["source_records"][0]
    assert source_record["source_manifest_sha256"] == "4" * 64
    assert source_record["source_payload_sha256"] == "a" * 64
    assert source_record["source_payload_path"] == (
        "/safe/sub-01/train.pt"
    )
    assert source_record["source_payload_byte_count"] == 123
    assert len(validations) == 1
    assert validations[0][1] is config_identity
    assert validations[0][3:] == (1, 42)
    assert _ValDataset.calls == ["a" * 64]
    assert evaluation_dtypes == [torch.float32]


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
            raise AssertionError("dataset must not be constructed")

    monkeypatch.setattr(br, "BrainRWDevelopmentDataset", ForbiddenDataset)
    monkeypatch.setattr(
        br,
        "verify_brainrw_config",
        lambda *_a, **_k: object(),
    )
    monkeypatch.setattr(
        emit,
        "_git_provenance",
        lambda: {
            "clean": True,
            "git_sha": "a" * 40,
            "repository_root": str(
                experiment_root.parents[1]
            ),
        },
        raising=False,
    )
    arguments = [
        "--scope", "val-dev", "--subject", "1", "--seed", "42",
        "--checkpoint", str(checkpoint_path),
        "--manifest", str(manifest),
        "--output-dir", str(tmp_path / "scores"),
    ]
    with pytest.raises(SystemExit):
        emit.main(arguments)
    assert touched is False

    existing = tmp_path / "existing-scores"
    existing.mkdir()
    arguments[arguments.index(str(tmp_path / "scores"))] = str(existing)
    with pytest.raises(SystemExit):
        emit.main(arguments)
