from __future__ import annotations

import ctypes
import errno
import io
import json
import math
import os
import subprocess
import textwrap
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

import matching_fairness.artifacts as artifact_module
import matching_fairness.native_export as native_export_module
from matching_fairness.artifacts import (
    ScoreArtifact,
    independent_ranks,
    read_score_artifact,
    write_score_artifact,
)
from matching_fairness.native_export import (
    NativeExportConfig,
    audit_native_checkpoints,
    build_native_score_artifact,
    evaluate_native_checkpoint,
    export_native_scores,
    native_scores,
)
from matching_fairness.provenance import (
    OFFICIAL_SOURCE_URL,
    inspect_checkout,
    sha256_file,
    sha256_path,
)
from matching_fairness.trial_splits import build_trial_manifest


FAKE_URL = "https://example.invalid/native-export.git"


def test_atms_uses_normalized_native_scores() -> None:
    eeg = np.array([[3.0, 4.0], [0.0, 2.0]])
    image = np.array([[4.0, 3.0], [2.0, 0.0]])

    scores = native_scores("atm_s", eeg, image, logit_scale=2.0)

    eeg_unit = eeg / np.linalg.norm(eeg, axis=1, keepdims=True)
    image_unit = image / np.linalg.norm(image, axis=1, keepdims=True)
    expected = 2.0 * eeg_unit @ image_unit.T
    np.testing.assert_allclose(scores, expected)


def test_nice_uses_official_raw_logit_scores() -> None:
    eeg = np.array([[3.0, 4.0], [0.0, 2.0]])
    image = np.array([[4.0, 3.0], [2.0, 0.0]])

    scores = native_scores("nice", eeg, image, logit_scale=2.0)

    np.testing.assert_allclose(scores, 2.0 * eeg @ image.T)


@pytest.mark.parametrize(
    "eeg,image,message",
    (
        (
            np.array([[0.0, 0.0]]),
            np.array([[1.0, 0.0]]),
            "nonzero EEG row norms",
        ),
        (
            np.array([[1.0, 0.0]]),
            np.array([[0.0, 0.0]]),
            "nonzero image row norms",
        ),
        (
            np.array([[np.nan, 0.0]]),
            np.array([[1.0, 0.0]]),
            "finite",
        ),
    ),
)
def test_atms_rejects_invalid_normalization_inputs(
    eeg: np.ndarray,
    image: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        native_scores("atm_s", eeg, image, logit_scale=1.0)


def test_native_artifact_ranking_uses_canonical_ids_not_diagonal() -> None:
    eeg = np.array([[1.0, 0.0], [0.0, 1.0]])
    image = np.array([[0.0, 1.0], [1.0, 0.0]])
    scores = native_scores("nice", eeg, image, logit_scale=1.0)

    artifact = build_native_score_artifact(
        model="nice",
        similarity=scores,
        query_ids=("query-a", "query-b"),
        gallery_ids=("image-b", "image-a"),
        target_ids=("image-a", "image-b"),
        trial_half="standard",
        checkpoint=Path("best_val.pth"),
        checkpoint_sha256="a" * 64,
        logit_scale=1.0,
        query_embeddings=eeg,
    )

    assert independent_ranks(artifact).tolist() == [1, 1]
    assert artifact.metadata["native_metrics"] == {
        "top1_count": 2,
        "top5_count": 2,
        "sample_count": 2,
    }


class _TinyEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(2))
        self.logit_scale = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        return eeg @ self.weight


def _checkpoint(path: Path, raw_logit_scale: float) -> Path:
    model = _TinyEncoder()
    model.logit_scale.data.fill_(raw_logit_scale)
    torch.save(model.state_dict(), path)
    return path


def test_checkpoint_raw_logit_scale_drives_effective_export_scale(
    tmp_path: Path,
) -> None:
    eeg = np.eye(2, dtype=np.float32)
    image = np.eye(2, dtype=np.float32)
    low = _checkpoint(tmp_path / "low.pth", math.log(2.0))
    high = _checkpoint(tmp_path / "high.pth", math.log(5.0))

    low_result = evaluate_native_checkpoint(
        model=_TinyEncoder(),
        checkpoint=low,
        model_slug="nice",
        eeg=eeg,
        image_features=image,
        subject_index=7,
        device=torch.device("cpu"),
    )
    high_result = evaluate_native_checkpoint(
        model=_TinyEncoder(),
        checkpoint=high,
        model_slug="nice",
        eeg=eeg,
        image_features=image,
        subject_index=7,
        device=torch.device("cpu"),
    )

    assert low_result.logit_scale == pytest.approx(2.0)
    assert high_result.logit_scale == pytest.approx(5.0)
    np.testing.assert_allclose(low_result.similarity, 2.0 * np.eye(2))
    np.testing.assert_allclose(high_result.similarity, 5.0 * np.eye(2))



def _git(checkout: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-C", str(checkout), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _fake_locked_checkout(root: Path) -> tuple[Path, Path]:
    checkout = root / "official"
    subprocess.run(
        ["git", "init", str(checkout)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    _git(checkout, "config", "user.email", "test@example.com")
    _git(checkout, "config", "user.name", "Test")
    _git(checkout, "remote", "add", "origin", FAKE_URL)
    sources = {
        "Retrieval/train_unified.py": "raise RuntimeError(\"sealed\")\n",
        "Retrieval/retrieval_engine.py": "raise RuntimeError(\"sealed\")\n",
        "eegdatasets.py": "raise RuntimeError(\"sealed\")\n",
        "encoder_utils.py": "raise RuntimeError(\"sealed\")\n",
        "models/atms.py": "ORIGIN = \"locked\"\n",
        "Retrieval/eeg_encoders.py": r"""
            import torch

            class TinyOfficialEncoder(torch.nn.Module):
                def __init__(self, encoder_type):
                    super().__init__()
                    self.encoder_type = encoder_type
                    self.weight = torch.nn.Parameter(torch.eye(2))
                    self.logit_scale = torch.nn.Parameter(torch.tensor(0.0))

                def forward(self, eeg, subject_ids=None):
                    if self.encoder_type == "ATMS":
                        assert subject_ids is not None
                        assert subject_ids.tolist() == [7] * len(eeg)
                    else:
                        assert subject_ids is None
                    return eeg.squeeze(1) @ self.weight

            def build_encoder(
                encoder_type,
                n_chans=63,
                n_times=250,
                joint_train=False,
                **kwargs,
            ):
                assert encoder_type in {"NICE", "ATMS"}
                assert n_chans == 1
                assert n_times == 2
                assert joint_train is False
                assert kwargs == {}
                return TinyOfficialEncoder(encoder_type)
        """,
    }
    for relative, source in sources.items():
        path = checkout / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source).lstrip(), encoding="utf-8")
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-m", "fake official source")
    _git(checkout, "checkout", "--detach", "HEAD")
    lock = inspect_checkout(
        checkout,
        expected_url=FAKE_URL,
        expected_branch="develop",
    )
    lock_path = root / "source_lock.json"
    lock_path.write_text(
        json.dumps(lock.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return checkout, lock_path


def _native_export_config(root: Path) -> NativeExportConfig:
    checkout, source_lock = _fake_locked_checkout(root)
    image_ids = ("image-a", "image-b")
    sessions = np.tile(np.repeat(np.arange(4), 20), (2, 1))
    manifest = build_trial_manifest(image_ids, sessions, seed=42)
    manifest_path = root / "trial_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    eeg = np.empty((2, 80, 1, 2), dtype=np.float32)
    for image_index, image_id in enumerate(image_ids):
        a_indices = [
            index
            for split in manifest["images"][image_id].values()
            for index in split["a"]
        ]
        b_indices = [
            index
            for split in manifest["images"][image_id].values()
            for index in split["b"]
        ]
        if image_index == 0:
            eeg[image_index, a_indices] = np.array([[[1.0, 0.0]]])
            eeg[image_index, b_indices] = np.array([[[0.8, 0.2]]])
        else:
            eeg[image_index, a_indices] = np.array([[[0.0, 1.0]]])
            eeg[image_index, b_indices] = np.array([[[0.2, 0.8]]])
    asset_root = root / "official_assets"
    test_eeg = (
        asset_root
        / "Preprocessed_data_250Hz/sub-08/preprocessed_eeg_test.npy"
    )
    test_eeg.parent.mkdir(parents=True)
    np.save(test_eeg, {"preprocessed_eeg_data": eeg}, allow_pickle=True)
    test_features = asset_root / "ViT-H-14_features_test.pt"
    torch.save(
        {
            "img_features": torch.eye(2),
            "text_features": torch.eye(2),
        },
        test_features,
    )
    training_eeg = (
        asset_root
        / "Preprocessed_data_250Hz/sub-08/preprocessed_eeg_training.npy"
    )
    training_eeg.write_bytes(b"sealed training eeg")
    training_features = asset_root / "ViT-H-14_features_train.pt"
    training_features.write_bytes(b"sealed training features")
    asset_files = {
        str(path.relative_to(asset_root)): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in (
            training_eeg,
            test_eeg,
            training_features,
            test_features,
        )
    }
    asset_lock = root / "assets_lock.json"
    asset_lock.write_text(
        json.dumps(
            {
                "repo_id": "LidongYang/EEG_Image_decode",
                "repo_type": "dataset",
                "asset_root": str(asset_root),
                "files": asset_files,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    test_images = root / "test_images"
    (test_images / "class-a").mkdir(parents=True)
    (test_images / "class-b").mkdir()
    (test_images / "class-a/image-a.jpg").write_bytes(b"a")
    (test_images / "class-b/image-b.jpg").write_bytes(b"b")

    checkpoint_dir = root / "checkpoints/nice"
    checkpoint_dir.mkdir(parents=True)
    checkpoints = {}
    for name, raw_scale, weight in (
        ("epoch_0001.pth", math.log(2.0), torch.eye(2)),
        ("epoch_0002.pth", math.log(3.0), torch.tensor([[0.0, 1.0], [1.0, 0.0]])),
        ("best_val.pth", math.log(2.0), torch.eye(2)),
    ):
        encoder = _TinyEncoder()
        encoder.logit_scale.data.fill_(raw_scale)
        encoder.weight.data.copy_(weight)
        checkpoint = checkpoint_dir / name
        torch.save(encoder.state_dict(), checkpoint)
        checkpoints[name] = checkpoint
    checkpoints["best_val.pth"].write_bytes(
        checkpoints["epoch_0001.pth"].read_bytes()
    )
    checkpoint_manifest = {
        "schema_version": 1,
        "model": "nice",
        "encoder_type": "NICE",
        "subject": "sub-08",
        "seed": 42,
        "source": json.loads(source_lock.read_text(encoding="utf-8")),
        "inputs": {
            "training_eeg": {
                "name": "preprocessed_eeg_training.npy",
                "sha256": sha256_file(training_eeg),
            },
            "training_features": {
                "name": "ViT-H-14_features_train.pt",
                "sha256": sha256_file(training_features),
            },
        },
        "hyperparameters": {
            "epochs": 500,
            "batch_size": 1024,
            "learning_rate": 0.0003,
            "val_ratio": 0.1,
            "early_stopping_patience": 10,
            "ema_decay": 0.999,
            "logit_scale_type": "exp",
            "avg_trials": True,
            "n_chans": 1,
            "n_times": 2,
        },
        "encoder_behavior": {
            "use_subject_id": False,
            "normalize_feats": False,
        },
        "checkpoints": [
            {
                "epoch": epoch,
                "val_loss": value,
                "checkpoint": f"epoch_{epoch:04d}.pth",
                "sha256": sha256_file(checkpoints[f"epoch_{epoch:04d}.pth"]),
            }
            for epoch, value in ((1, 0.2), (2, 0.4))
        ],
        "selection": {
            "epoch": 1,
            "val_loss": 0.2,
            "checkpoint": "epoch_0001.pth",
        },
        "best_checkpoint": {
            "name": "best_val.pth",
            "sha256": sha256_file(checkpoints["best_val.pth"]),
        },
        "history": {"name": "history.csv", "sha256": ""},
        "stopped_early": True,
    }
    history = checkpoint_dir / "history.csv"
    history.write_text("epoch,val_loss\n1,0.2\n2,0.4\n", encoding="utf-8")
    checkpoint_manifest["history"]["sha256"] = sha256_file(history)
    (checkpoint_dir / "checkpoint_manifest.json").write_text(
        json.dumps(checkpoint_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return NativeExportConfig(
        source_checkout=checkout,
        source_lock=source_lock,
        asset_root=asset_root,
        asset_lock=asset_lock,
        test_eeg=test_eeg,
        test_features=test_features,
        test_images=test_images,
        trial_manifest=manifest_path,
        checkpoint_dir=checkpoint_dir,
        output_dir=root / "scores/nice",
        model="nice",
        subject="sub-08",
        device="cpu",
        expected_image_count=2,
        n_chans=1,
        n_times=2,
    )


def test_native_export_rejects_tampered_pickled_eeg_before_numpy_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _native_export_config(tmp_path)
    tampered = bytearray(config.test_eeg.read_bytes())
    tampered[-1] ^= 1
    config.test_eeg.write_bytes(tampered)
    called = False

    def forbidden_load(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("untrusted pickle deserialization was reached")

    monkeypatch.setattr(native_export_module.np, "load", forbidden_load)

    with pytest.raises(ValueError, match="asset.*SHA-256 mismatch"):
        export_native_scores(config)
    assert called is False


def test_native_export_deserializes_the_verified_eeg_byte_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _native_export_config(tmp_path)
    expected_hash = sha256_file(config.test_eeg)
    real_load = native_export_module.np.load
    observed_pickle_load = False

    def replace_path_then_load(source: object, *args: object, **kwargs: object) -> object:
        nonlocal observed_pickle_load
        if kwargs.get("allow_pickle") is True:
            observed_pickle_load = True
            assert isinstance(source, io.BytesIO)
            config.test_eeg.write_bytes(b"replacement after verified open")
        return real_load(source, *args, **kwargs)

    monkeypatch.setattr(native_export_module.np, "load", replace_path_then_load)

    result = export_native_scores(config)

    assert observed_pickle_load is True
    artifact = read_score_artifact(result.artifact_paths["standard"])
    assert artifact.metadata["input_sha256"]["test_eeg"] == expected_hash


def test_native_export_opens_pickled_eeg_with_no_follow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _native_export_config(tmp_path)
    real_open = native_export_module.os.open
    observed_no_follow = False

    def recording_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
        nonlocal observed_no_follow
        if os.fspath(path) == os.fspath(config.test_eeg):
            observed_no_follow = bool(flags & getattr(os, "O_NOFOLLOW", 0))
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(native_export_module.os, "open", recording_open)

    export_native_scores(config)

    assert observed_no_follow is True


@pytest.mark.parametrize(
    "field_path,value,message",
    (
        (("source", "branch"), "wrong", "source provenance"),
        (("hyperparameters", "batch_size"), 512, "hyperparameters"),
        (("encoder_behavior", "normalize_feats"), True, "encoder behavior"),
        (("selection", "val_loss"), 0.3, "selection"),
        (("inputs", "training_eeg", "sha256"), "0" * 64, "training inputs"),
    ),
)
def test_native_export_rejects_checkpoint_manifest_provenance_tampering(
    tmp_path: Path,
    field_path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    config = _native_export_config(tmp_path)
    path = config.checkpoint_dir / "checkpoint_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    target = manifest
    for field in field_path[:-1]:
        target = target[field]
    target[field_path[-1]] = value
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        export_native_scores(config)


def test_native_export_rejects_checkpoint_manifest_schema_extension(
    tmp_path: Path,
) -> None:
    config = _native_export_config(tmp_path)
    path = config.checkpoint_dir / "checkpoint_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["unreviewed"] = True
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exact Task 5 schema"):
        export_native_scores(config)


def _complete_formal_inventory(
    root: Path,
    native_paths: dict[str, Path],
) -> tuple[Path, ...]:
    paths = list(native_paths.values())
    template = read_score_artifact(native_paths["standard"])
    for model in ("atm_s", "our_project"):
        for half in ("standard", "a", "b"):
            path = root / "inventory" / model / half
            metadata = dict(template.metadata)
            metadata.update(
                {
                    "model_slug": model,
                    "trial_half": half,
                    "subject": "sub-08",
                    "seed": 42,
                }
            )
            if model == "our_project":
                metadata.update(
                    {
                        "checkpoint_role": "fixed_formal",
                        "similarity": "cosine",
                        "checkpoint_content_sha256": "1" * 64,
                        "trial_manifest_sha256": template.metadata["input_sha256"][
                            "trial_manifest"
                        ],
                        "brain_test_sha256": "2" * 64,
                        "protocol_sha256": "6" * 64,
                        "model_content_sha256": {
                            "brain_model": "1" * 64,
                            "vision_adapter": "3" * 64,
                            "pretrained_vision_base": "4" * 64,
                        },
                        "evaluator_version": "AIAA3800-BRAINRW-FORMAL-v1",
                        "evaluator_sha256": "5" * 64,
                        "runtime_inputs": {
                            "test_image_tree_sha256": "7" * 64,
                            "selected_channel_indices": [30],
                            "time_slice": [0, 250],
                            "dataset_name": "things",
                            "expected_sample_count": 2,
                        },
                    }
                )
            write_score_artifact(
                path,
                ScoreArtifact(
                    similarity=template.similarity,
                    query_ids=template.query_ids,
                    gallery_entry_ids=template.gallery_entry_ids,
                    gallery_canonical_ids=template.gallery_canonical_ids,
                    target_canonical_ids=template.target_canonical_ids,
                    metadata=metadata,
                ),
            )
            paths.append(path)
    return tuple(paths)


def _replace_inventory_artifact(
    root: Path,
    inventory: tuple[Path, ...],
    *,
    model: str,
    half: str,
    transform: object,
) -> tuple[Path, ...]:
    paths = list(inventory)
    index = next(
        position
        for position, path in enumerate(paths)
        if read_score_artifact(path).metadata.get("model_slug") == model
        and read_score_artifact(path).metadata.get("trial_half") == half
    )
    original = read_score_artifact(paths[index])
    replacement = transform(original)
    replacement_path = root / "tampered" / model / half
    write_score_artifact(replacement_path, replacement)
    paths[index] = replacement_path
    return tuple(paths)


def test_native_audit_rejects_formal_metric_mismatch(tmp_path: Path) -> None:
    config = _native_export_config(tmp_path)
    result = export_native_scores(config)
    inventory = _complete_formal_inventory(tmp_path, result.artifact_paths)

    def corrupt(artifact: ScoreArtifact) -> ScoreArtifact:
        metadata = dict(artifact.metadata)
        metrics = dict(metadata["native_metrics"])
        metrics["top1_count"] += 1
        metadata["native_metrics"] = metrics
        return ScoreArtifact(
            artifact.similarity,
            artifact.query_ids,
            artifact.gallery_entry_ids,
            artifact.gallery_canonical_ids,
            artifact.target_canonical_ids,
            metadata,
        )

    inventory = _replace_inventory_artifact(
        tmp_path, inventory, model="our_project", half="standard", transform=corrupt
    )
    with pytest.raises(ValueError, match="metric parity"):
        audit_native_checkpoints(config, inventory)


def test_native_audit_rejects_query_target_order_mismatch(tmp_path: Path) -> None:
    config = _native_export_config(tmp_path)
    result = export_native_scores(config)
    inventory = _complete_formal_inventory(tmp_path, result.artifact_paths)

    def corrupt(artifact: ScoreArtifact) -> ScoreArtifact:
        return ScoreArtifact(
            artifact.similarity,
            artifact.query_ids,
            artifact.gallery_entry_ids,
            artifact.gallery_canonical_ids,
            tuple(reversed(artifact.target_canonical_ids)),
            artifact.metadata,
        )

    inventory = _replace_inventory_artifact(
        tmp_path, inventory, model="our_project", half="standard", transform=corrupt
    )
    with pytest.raises(ValueError, match="canonical query/target/gallery order"):
        audit_native_checkpoints(config, inventory)


def test_native_audit_rejects_one_half_input_identity_mismatch(
    tmp_path: Path,
) -> None:
    config = _native_export_config(tmp_path)
    result = export_native_scores(config)
    inventory = _complete_formal_inventory(tmp_path, result.artifact_paths)

    def corrupt(artifact: ScoreArtifact) -> ScoreArtifact:
        metadata = dict(artifact.metadata)
        inputs = dict(metadata["input_sha256"])
        inputs["test_features"] = "8" * 64
        metadata["input_sha256"] = inputs
        return ScoreArtifact(
            artifact.similarity,
            artifact.query_ids,
            artifact.gallery_entry_ids,
            artifact.gallery_canonical_ids,
            artifact.target_canonical_ids,
            metadata,
        )

    inventory = _replace_inventory_artifact(
        tmp_path, inventory, model="atm_s", half="a", transform=corrupt
    )
    with pytest.raises(ValueError, match="native provenance must be identical"):
        audit_native_checkpoints(config, inventory)


@pytest.mark.parametrize(
    "field,value",
    (
        ("selected_channel_indices", [31]),
        ("test_image_tree_sha256", "9" * 64),
    ),
)
def test_native_audit_rejects_one_half_brainrw_runtime_mismatch(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    config = _native_export_config(tmp_path)
    result = export_native_scores(config)
    inventory = _complete_formal_inventory(tmp_path, result.artifact_paths)

    def corrupt(artifact: ScoreArtifact) -> ScoreArtifact:
        metadata = dict(artifact.metadata)
        runtime = dict(metadata["runtime_inputs"])
        runtime[field] = value
        metadata["runtime_inputs"] = runtime
        return ScoreArtifact(
            artifact.similarity,
            artifact.query_ids,
            artifact.gallery_entry_ids,
            artifact.gallery_canonical_ids,
            artifact.target_canonical_ids,
            metadata,
        )

    inventory = _replace_inventory_artifact(
        tmp_path, inventory, model="our_project", half="b", transform=corrupt
    )
    with pytest.raises(ValueError, match="BrainRW provenance must be identical"):
        audit_native_checkpoints(config, inventory)


def test_native_audit_binds_current_config_to_supplied_artifacts(
    tmp_path: Path,
) -> None:
    config = _native_export_config(tmp_path)
    result = export_native_scores(config)
    inventory = _complete_formal_inventory(tmp_path, result.artifact_paths)
    alternate_manifest = tmp_path / "alternate_trial_manifest.json"
    alternate_manifest.write_text(
        json.dumps(json.loads(config.trial_manifest.read_text(encoding="utf-8"))),
        encoding="utf-8",
    )
    mismatched = replace(config, trial_manifest=alternate_manifest)

    with pytest.raises(ValueError, match="current native audit config"):
        audit_native_checkpoints(mismatched, inventory)


def test_native_export_publishes_main_artifacts_before_separate_audit(
    tmp_path: Path,
) -> None:
    config = _native_export_config(tmp_path)

    result = export_native_scores(config)

    assert set(result.artifact_paths) == {"standard", "eeg_a", "eeg_b"}
    assert set(result.artifact_hashes) == {"standard", "eeg_a", "eeg_b"}
    artifacts = {
        half: read_score_artifact(path)
        for half, path in result.artifact_paths.items()
    }
    assert artifacts["standard"].similarity.shape == (2, 2)
    assert artifacts["eeg_a"].gallery_canonical_ids == artifacts["eeg_b"].gallery_canonical_ids
    assert (
        artifacts["eeg_a"].metadata["query_embeddings_sha256"]
        != artifacts["eeg_b"].metadata["query_embeddings_sha256"]
    )
    assert artifacts["standard"].metadata["checkpoint"].endswith("best_val.pth")
    standard_ranks = independent_ranks(artifacts["standard"])
    assert artifacts["standard"].metadata["native_metrics"] == {
        "top1_count": int(np.count_nonzero(standard_ranks <= 1)),
        "top5_count": int(np.count_nonzero(standard_ranks <= 5)),
        "sample_count": 2,
    }
    assert not (config.output_dir / "best_test_audit.json").exists()

    inventory = _complete_formal_inventory(tmp_path, result.artifact_paths)
    audit_path = audit_native_checkpoints(config, inventory)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))

    assert audit["scope"] == "best_test_audit_only"
    assert len(audit["formal_artifact_inventory"]) == 9
    assert [row["epoch"] for row in audit["runs"]] == [1, 2]
    assert audit["best_test"]["epoch"] == 1
    assert {
        path.name for path in config.output_dir.iterdir()
    } == {"standard", "eeg_a", "eeg_b", "best_test_audit.json"}



def test_native_export_cli_maps_only_fixed_protocol_fields(tmp_path: Path) -> None:
    from matching_fairness.config import Protocol
    from scripts.export_native_scores import native_export_config_from_protocol

    protocol = Protocol.load(
        Path(__file__).resolve().parents[1] / "configs/protocol_sub08_seed42.json"
    )
    config = native_export_config_from_protocol(
        protocol=protocol,
        source_checkout=tmp_path / "source",
        source_lock=tmp_path / "source.json",
        asset_root=tmp_path / "assets",
        asset_lock=tmp_path / "assets.json",
        test_eeg=tmp_path / "preprocessed_eeg_test.npy",
        test_features=tmp_path / "ViT-H-14_features_test.pt",
        test_images=tmp_path / "test_images",
        trial_manifest=tmp_path / "trials.json",
        checkpoint_dir=tmp_path / "checkpoints/nice",
        output_dir=tmp_path / "scores/nice",
        model="nice",
        device="cpu",
    )

    assert config.subject == "sub-08"
    assert config.expected_image_count == 200
    assert config.n_chans == 63
    assert config.n_times == 250
    assert config.logit_scale_type == "exp"


def test_brainrw_cli_builds_three_separate_evaluator_commands(tmp_path: Path) -> None:
    from scripts.export_brainrw_scores import build_evaluator_commands, build_parser

    manifest = tmp_path / "trials.json"
    arguments = build_parser().parse_args(
        [
            "--brain-model-path",
            "brain.pt",
            "--vision-adapter-path",
            "adapter",
            "--pretrained-model-name-or-path",
            "clip",
            "--brain-directory",
            "brain",
            "--image-directory",
            "images",
            "--selected-channels",
            "Cz",
            "--trial-split-manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "scores"),
        ]
    )

    commands = build_evaluator_commands(arguments)

    assert set(commands) == {"standard", "eeg_a", "eeg_b"}
    standard = commands["standard"]
    assert "--trial-split-manifest" not in standard
    assert standard[standard.index("--score-provenance-manifest") + 1] == str(
        manifest
    )
    assert standard[standard.index("--trial-half") + 1] == "standard"
    for name, half in (("eeg_a", "a"), ("eeg_b", "b")):
        command = commands[name]
        assert command[command.index("--trial-half") + 1] == half
        assert command[command.index("--trial-split-manifest") + 1] == str(manifest)
        artifact = command[command.index("--score-artifact-output") + 1]
        assert artifact.endswith(f"/{name}")



def test_native_export_rejects_manifest_with_extra_canonical_id(
    tmp_path: Path,
) -> None:
    config = _native_export_config(tmp_path)
    manifest = json.loads(config.trial_manifest.read_text(encoding="utf-8"))
    manifest["image_ids"].append("extra-image")
    manifest["images"]["extra-image"] = manifest["images"]["image-a"]
    config.trial_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="canonical image IDs"):
        export_native_scores(config)


def test_native_export_failure_leaves_no_partial_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _native_export_config(tmp_path)
    real_write = native_export_module.write_score_artifact
    calls = 0

    def fail_second(path: Path, artifact: ScoreArtifact) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected second-artifact failure")
        real_write(path, artifact)

    monkeypatch.setattr(native_export_module, "write_score_artifact", fail_second)

    with pytest.raises(RuntimeError, match="injected"):
        export_native_scores(config)
    assert not config.output_dir.exists()
    assert not list(config.output_dir.parent.glob(f".{config.output_dir.name}.tmp-*"))


def test_brainrw_export_failure_leaves_no_partial_publication(
    tmp_path: Path,
) -> None:
    from scripts.export_brainrw_scores import (
        build_parser,
        export_brainrw_scores,
    )

    output = tmp_path / "brainrw-scores"
    manifest = tmp_path / "trials.json"
    manifest.write_text("{}\n", encoding="utf-8")
    arguments = build_parser().parse_args(
        [
            "--brain-model-path", "brain",
            "--vision-adapter-path", "adapter",
            "--pretrained-model-name-or-path", "base",
            "--brain-directory", "eeg",
            "--image-directory", "images",
            "--selected-channels", "Cz",
            "--trial-split-manifest", str(manifest),
            "--output-dir", str(output),
        ]
    )
    calls = 0

    def runner(command: list[str], *, check: bool) -> object:
        nonlocal calls
        calls += 1
        artifact = Path(command[command.index("--score-artifact-output") + 1])
        artifact.mkdir(parents=True)
        (artifact / "partial").write_bytes(b"partial")
        if calls == 2:
            raise subprocess.CalledProcessError(1, command)
        return object()

    with pytest.raises(subprocess.CalledProcessError):
        export_brainrw_scores(arguments, runner=runner)
    assert not output.exists()
    assert not list(output.parent.glob(f".{output.name}.tmp-*"))


def test_brainrw_real_publisher_emits_exact_relative_hash_bound_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.export_brainrw_scores import build_parser, export_brainrw_scores
    import scripts.run_scenarios as scenario_runner_module

    class UnsupportedRenameAt2:
        argtypes: object | None = None
        restype: object | None = None

        def __call__(self, *args: object) -> int:
            ctypes.set_errno(errno.EINVAL)
            return -1

    class UnsupportedLibc:
        renameat2 = UnsupportedRenameAt2()

    monkeypatch.setattr(
        artifact_module.ctypes,
        "CDLL",
        lambda *args, **kwargs: UnsupportedLibc(),
    )

    output = tmp_path / "matrices" / "our_project"
    protocol = tmp_path / "protocol.json"
    protocol.write_text('{"formal":true}\n', encoding="utf-8")
    trial_manifest = tmp_path / "trials.json"
    trial_manifest.write_text('{"seed":42}\n', encoding="utf-8")
    brain_directory = tmp_path / "brain"
    brain_test = brain_directory / "sub-08/test.pt"
    brain_test.parent.mkdir(parents=True)
    brain_test.write_bytes(b"brain-test")
    image_directory = tmp_path / "images"
    test_images = image_directory / "test_images"
    test_images.mkdir(parents=True)
    (test_images / "image-000.jpg").write_bytes(b"image")
    model_paths = {
        "brain": tmp_path / "brain-model",
        "adapter": tmp_path / "vision-adapter",
        "base": tmp_path / "vision-base",
    }
    for name, path in model_paths.items():
        path.mkdir()
        (path / "weights.bin").write_bytes(name.encode("ascii"))
    model_content_sha256 = {
        "brain_model": sha256_path(model_paths["brain"]),
        "vision_adapter": sha256_path(model_paths["adapter"]),
        "pretrained_vision_base": sha256_path(model_paths["base"]),
    }
    evaluator_sha256 = sha256_file(
        Path(__file__).resolve().parents[3] / "scripts/evaluate_retrieval.py"
    )
    protocol_sha256 = sha256_file(protocol)
    trial_manifest_sha256 = sha256_file(trial_manifest)
    brain_test_sha256 = sha256_file(brain_test)
    test_image_tree_sha256 = sha256_path(test_images)
    arguments = build_parser().parse_args(
        [
            "--brain-model-path", str(model_paths["brain"]),
            "--vision-adapter-path", str(model_paths["adapter"]),
            "--pretrained-model-name-or-path", str(model_paths["base"]),
            "--brain-directory", str(brain_directory),
            "--image-directory", str(image_directory),
            "--selected-channels", "Cz",
            "--trial-split-manifest", str(trial_manifest),
            "--protocol", str(protocol),
            "--output-dir", str(output),
        ]
    )
    ids = tuple(f"image-{index:03d}" for index in range(200))

    def runner(command: list[str], *, check: bool) -> object:
        assert check is True
        half = command[command.index("--trial-half") + 1]
        artifact = Path(command[command.index("--score-artifact-output") + 1])
        write_score_artifact(
            artifact,
            ScoreArtifact(
                similarity=np.eye(200, dtype=np.float32),
                query_ids=ids,
                gallery_entry_ids=ids,
                gallery_canonical_ids=ids,
                target_canonical_ids=ids,
                metadata={
                    "model_slug": "our_project",
                    "trial_half": half,
                    "checkpoint_role": "fixed_formal",
                    "checkpoint": str(model_paths["brain"]),
                    "checkpoint_content_sha256": model_content_sha256[
                        "brain_model"
                    ],
                    "similarity": "cosine",
                    "query_embeddings_sha256": {
                        "standard": "1" * 64,
                        "a": "2" * 64,
                        "b": "3" * 64,
                    }[half],
                    "subject": "sub-08",
                    "seed": 42,
                    "trial_manifest_sha256": trial_manifest_sha256,
                    "protocol_sha256": protocol_sha256,
                    "brain_test_sha256": brain_test_sha256,
                    "model_content_sha256": model_content_sha256,
                    "evaluator_version": "AIAA3800-BRAINRW-FORMAL-v1",
                    "evaluator_sha256": evaluator_sha256,
                    "runtime_inputs": {
                        "test_image_tree_sha256": test_image_tree_sha256,
                        "selected_channel_indices": [30],
                        "time_slice": [0, 250],
                        "dataset_name": "things",
                        "expected_sample_count": 200,
                    },
                    "native_metrics": {
                        "top1_count": 200,
                        "top5_count": 200,
                        "sample_count": 200,
                    },
                },
            ),
        )
        metrics = Path(command[command.index("--metrics-output") + 1])
        predictions = Path(command[command.index("--predictions-output") + 1])
        metrics.parent.mkdir(parents=True, exist_ok=True)
        metrics.write_text('{"top1_count":200}\n', encoding="utf-8")
        predictions.write_text("query,prediction\n", encoding="utf-8")
        return object()

    export_brainrw_scores(arguments, runner=runner)

    assert {path.name for path in output.iterdir()} == {
        "standard", "eeg_a", "eeg_b", "runs", "export_manifest.json"
    }
    manifest = json.loads((output / "export_manifest.json").read_text(encoding="utf-8"))
    assert set(manifest) == {
        "schema_version", "scope", "checkpoint_role", "model_slug",
        "subject", "seed", "artifacts", "runs", "inputs",
    }
    assert manifest["scope"] == "fixed_formal_export"
    assert manifest["checkpoint_role"] == "fixed_formal"
    assert manifest["model_slug"] == "our_project"
    assert manifest["subject"] == "sub-08"
    assert manifest["seed"] == 42
    assert set(manifest["artifacts"]) == {"standard", "eeg_a", "eeg_b"}
    assert set(manifest["runs"]) == {"standard", "eeg_a", "eeg_b"}
    for name in ("standard", "eeg_a", "eeg_b"):
        assert manifest["artifacts"][name]["path"] == name
        assert manifest["runs"][name] == {
            "path": f"runs/{name}",
            "sha256": sha256_path(output / "runs" / name),
        }
    assert set(manifest["inputs"]) == {
        "protocol_sha256", "trial_manifest_sha256", "brain_test_sha256",
        "evaluator_sha256", "test_image_tree_sha256", "model_content_sha256",
    }
    assert manifest["inputs"]["protocol_sha256"] == sha256_file(protocol)
    assert manifest["inputs"]["trial_manifest_sha256"] == sha256_file(trial_manifest)
    assert manifest["inputs"]["brain_test_sha256"] == sha256_file(brain_test)
    assert manifest["inputs"]["test_image_tree_sha256"] == sha256_path(test_images)
    assert manifest["inputs"]["model_content_sha256"] == {
        "brain_model": sha256_path(model_paths["brain"]),
        "vision_adapter": sha256_path(model_paths["adapter"]),
        "pretrained_vision_base": sha256_path(model_paths["base"]),
    }
    assert "/hpc" not in (output / "export_manifest.json").read_text(encoding="utf-8")

    # Exercise the real Task 7 directory loader against the tree emitted by
    # the BrainRW publisher.  The strict Task 6 inventory itself is covered by
    # its dedicated tests; this integration isolates the shared tree contract.
    for model in ("nice", "atm_s"):
        for directory, half in (("standard", "standard"), ("eeg_a", "a"), ("eeg_b", "b")):
            source = read_score_artifact(output / directory)
            metadata = dict(source.metadata)
            metadata.update({"model_slug": model, "trial_half": half})
            write_score_artifact(
                output.parent / model / directory,
                ScoreArtifact(
                    similarity=source.similarity,
                    query_ids=source.query_ids,
                    gallery_entry_ids=source.gallery_entry_ids,
                    gallery_canonical_ids=source.gallery_canonical_ids,
                    target_canonical_ids=source.target_canonical_ids,
                    metadata=metadata,
                ),
            )

    def formal_inventory(
        directories: tuple[Path, ...] | list[Path],
        *,
        expected_image_count: int,
    ) -> list[dict[str, object]]:
        assert expected_image_count == 200
        entries = []
        for path in directories:
            artifact = read_score_artifact(path)
            entries.append(
                {
                    "model_slug": artifact.metadata["model_slug"],
                    "trial_half": artifact.metadata["trial_half"],
                    "path": str(path),
                    "sha256": native_export_module._score_artifact_sha256(path),
                }
            )
        return entries

    inventory = formal_inventory(
        [
            output.parent / model / directory
            for model in ("nice", "atm_s", "our_project")
            for directory in ("standard", "eeg_a", "eeg_b")
        ],
        expected_image_count=200,
    )
    checkpoint_sha256 = "e" * 64
    audit_run = {
        "epoch": 1,
        "checkpoint": "/sealed/epoch_0001.pth",
        "checkpoint_sha256": checkpoint_sha256,
        "effective_logit_scale": 1.0,
        "top1_count": 200,
        "top5_count": 200,
        "sample_count": 200,
    }
    for model in ("nice", "atm_s"):
        checkpoint_manifest = {
            "schema_version": 1,
            "model": model,
            "encoder_type": "NICE" if model == "nice" else "ATMS",
            "subject": "sub-08",
            "seed": 42,
            "source": {
                "url": OFFICIAL_SOURCE_URL,
                "branch": "develop",
                "commit": "f" * 40,
                "checkout_sha256": "a" * 64,
            },
            "inputs": {
                "training_eeg": {
                    "name": "preprocessed_eeg_training.npy",
                    "sha256": "b" * 64,
                },
                "training_features": {
                    "name": "ViT-H-14_features_train.pt",
                    "sha256": "c" * 64,
                },
            },
            "hyperparameters": {
                "epochs": 500,
                "batch_size": 1024,
                "learning_rate": 3e-4,
                "val_ratio": 0.1,
                "early_stopping_patience": 10,
                "ema_decay": 0.999,
                "logit_scale_type": "exp",
                "avg_trials": True,
                "n_chans": 63,
                "n_times": 250,
            },
            "encoder_behavior": {
                "use_subject_id": model == "atm_s",
                "normalize_feats": model == "atm_s",
            },
            "checkpoints": [
                {
                    "epoch": 1,
                    "val_loss": 0.1,
                    "checkpoint": "epoch_0001.pth",
                    "sha256": checkpoint_sha256,
                }
            ],
            "selection": {
                "epoch": 1,
                "val_loss": 0.1,
                "checkpoint": "epoch_0001.pth",
            },
            "best_checkpoint": {
                "name": "best_val.pth",
                "sha256": checkpoint_sha256,
            },
            "history": {"name": "history.csv", "sha256": "d" * 64},
            "stopped_early": False,
        }
        checkpoint_path = (
            output.parent.parent
            / "checkpoints"
            / model
            / "checkpoint_manifest.json"
        )
        checkpoint_path.parent.mkdir(parents=True)
        checkpoint_path.write_text(
            json.dumps(
                checkpoint_manifest,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        audit = {
            "schema_version": 1,
            "scope": "best_test_audit_only",
            "model_slug": model,
            "checkpoint_policy": "every_epoch_checkpoint",
            "fairness_artifact_created": False,
            "formal_artifact_inventory": inventory,
            "runs": [audit_run],
            "best_test": audit_run,
        }
        (output.parent / model / "best_test_audit.json").write_text(
            json.dumps(audit, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        scenario_runner_module,
        "_formal_artifact_inventory",
        formal_inventory,
    )
    monkeypatch.setattr(
        scenario_runner_module,
        "validate_trial_manifest",
        lambda _manifest, _ids: {},
    )
    loaded, hashes, trial_hash = scenario_runner_module._load_formal_artifacts(
        output.parent,
        trial_manifest_path=trial_manifest,
        expected_image_count=200,
        protocol_sha256=protocol_sha256,
    )

    assert set(loaded) == set(hashes) == {"nice", "atm_s", "our_project"}
    assert all(set(loaded[model]) == {"standard", "a", "b"} for model in loaded)
    assert trial_hash == sha256_file(trial_manifest)
