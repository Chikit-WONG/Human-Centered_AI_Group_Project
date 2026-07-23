from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from matching_fairness.artifacts import (
    ScoreArtifact,
    read_score_artifact,
    write_score_artifact,
)
from matching_fairness.native_export import _score_artifact_sha256
from matching_fairness.provenance import OFFICIAL_SOURCE_URL, sha256_path


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _brainrw_tree(
    root: Path,
    *,
    protocol_sha256: str = "1" * 64,
    trial_manifest_sha256: str = "2" * 64,
) -> tuple[Path, dict[str, object]]:
    ids = ("image-000", "image-001")
    inputs: dict[str, object] = {
        "protocol_sha256": protocol_sha256,
        "trial_manifest_sha256": trial_manifest_sha256,
        "brain_test_sha256": "3" * 64,
        "evaluator_sha256": "4" * 64,
        "test_image_tree_sha256": "5" * 64,
        "model_content_sha256": {
            "brain_model": "6" * 64,
            "vision_adapter": "7" * 64,
            "pretrained_vision_base": "8" * 64,
        },
    }
    halves = {"standard": "standard", "eeg_a": "a", "eeg_b": "b"}
    for index, (directory, half) in enumerate(halves.items()):
        write_score_artifact(
            root / directory,
            ScoreArtifact(
                similarity=np.eye(2, dtype=np.float32),
                query_ids=ids,
                gallery_entry_ids=ids,
                gallery_canonical_ids=ids,
                target_canonical_ids=ids,
                metadata={
                    "model_slug": "our_project",
                    "trial_half": half,
                    "checkpoint_role": "fixed_formal",
                    "checkpoint": "/sealed/brain_model",
                    "checkpoint_content_sha256": "6" * 64,
                    "similarity": "cosine",
                    "query_embeddings_sha256": f"{index + 9:064x}",
                    "subject": "sub-08",
                    "seed": 42,
                    "trial_manifest_sha256": trial_manifest_sha256,
                    "protocol_sha256": protocol_sha256,
                    "brain_test_sha256": "3" * 64,
                    "model_content_sha256": inputs["model_content_sha256"],
                    "evaluator_version": "AIAA3800-BRAINRW-FORMAL-v1",
                    "evaluator_sha256": "4" * 64,
                    "runtime_inputs": {
                        "test_image_tree_sha256": "5" * 64,
                        "selected_channel_indices": [30],
                        "time_slice": [0, 250],
                        "dataset_name": "things",
                        "expected_sample_count": 2,
                    },
                    "native_metrics": {
                        "top1_count": 2,
                        "top5_count": 2,
                        "sample_count": 2,
                    },
                },
            ),
        )
        run = root / "runs" / directory
        run.mkdir(parents=True)
        _write_json(run / "metrics.json", {"top1_count": 2})
        (run / "predictions.csv").write_text(
            "query,prediction\nimage-000,image-000\n",
            encoding="utf-8",
        )
    manifest = {
        "schema_version": 1,
        "scope": "fixed_formal_export",
        "checkpoint_role": "fixed_formal",
        "model_slug": "our_project",
        "subject": "sub-08",
        "seed": 42,
        "artifacts": {
            name: {
                "path": name,
                "sha256": _score_artifact_sha256(root / name),
            }
            for name in halves
        },
        "runs": {
            name: {
                "path": f"runs/{name}",
                "sha256": sha256_path(root / "runs" / name),
            }
            for name in halves
        },
        "inputs": inputs,
    }
    _write_json(root / "export_manifest.json", manifest)
    return root, inputs


def test_shared_brainrw_tree_validator_returns_hash_bound_artifacts(
    tmp_path: Path,
) -> None:
    from matching_fairness.formal_artifacts import validate_brainrw_export_tree

    root, inputs = _brainrw_tree(tmp_path / "our_project")

    validated = validate_brainrw_export_tree(
        root,
        expected_image_count=2,
        expected_inputs=inputs,
    )

    assert set(validated.artifacts) == {"standard", "eeg_a", "eeg_b"}
    assert validated.inputs == inputs
    assert validated.artifact_sha256 == {
        name: _score_artifact_sha256(root / name)
        for name in ("standard", "eeg_a", "eeg_b")
    }


@pytest.mark.parametrize(
    "case",
    (
        "artifact_hash",
        "run_hash",
        "absolute_path",
        "traversal_path",
        "missing_entry",
        "extra_entry",
        "symlink",
        "dangling_symlink",
    ),
)
def test_shared_brainrw_tree_validator_rejects_tamper(
    tmp_path: Path,
    case: str,
) -> None:
    from matching_fairness.formal_artifacts import validate_brainrw_export_tree

    root, inputs = _brainrw_tree(tmp_path / "our_project")
    manifest_path = root / "export_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if case == "artifact_hash":
        manifest["artifacts"]["standard"]["sha256"] = "f" * 64
        _write_json(manifest_path, manifest)
    elif case == "run_hash":
        (root / "runs/standard/metrics.json").write_text(
            '{"top1_count":0}\n', encoding="utf-8"
        )
    elif case == "absolute_path":
        manifest["artifacts"]["standard"]["path"] = "/tmp/standard"
        _write_json(manifest_path, manifest)
    elif case == "traversal_path":
        manifest["runs"]["standard"]["path"] = "../runs/standard"
        _write_json(manifest_path, manifest)
    elif case == "missing_entry":
        (root / "runs/standard/predictions.csv").unlink()
    elif case == "extra_entry":
        (root / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
    elif case == "symlink":
        (root / "runs/standard/link").symlink_to(root / "runs/eeg_a")
    else:
        (root / "runs/standard/dangling").symlink_to(root / "missing")

    with pytest.raises(ValueError):
        validate_brainrw_export_tree(
            root,
            expected_image_count=2,
            expected_inputs=inputs,
        )


def _checkpoint_manifest(model: str) -> dict[str, object]:
    checkpoint = {
        "epoch": 1,
        "val_loss": 0.1,
        "checkpoint": "epoch_0001.pth",
        "sha256": "e" * 64,
    }
    return {
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
        "checkpoints": [checkpoint],
        "selection": {
            "epoch": 1,
            "val_loss": 0.1,
            "checkpoint": "epoch_0001.pth",
        },
        "best_checkpoint": {"name": "best_val.pth", "sha256": "e" * 64},
        "history": {"name": "history.csv", "sha256": "d" * 64},
        "stopped_early": False,
    }


def _task7_tree(
    root: Path,
) -> tuple[Path, Path, str, list[dict[str, object]]]:
    artifact_root = root / "matrices"
    trial_manifest = root / "trial_manifest.json"
    _write_json(trial_manifest, {})
    trial_hash = hashlib.sha256(trial_manifest.read_bytes()).hexdigest()
    protocol_hash = "1" * 64
    _brainrw_tree(
        artifact_root / "our_project",
        protocol_sha256=protocol_hash,
        trial_manifest_sha256=trial_hash,
    )
    ids = ("image-000", "image-001")
    for model in ("nice", "atm_s"):
        for directory, half in (
            ("standard", "standard"),
            ("eeg_a", "a"),
            ("eeg_b", "b"),
        ):
            write_score_artifact(
                artifact_root / model / directory,
                ScoreArtifact(
                    similarity=np.eye(2, dtype=np.float32),
                    query_ids=ids,
                    gallery_entry_ids=ids,
                    gallery_canonical_ids=ids,
                    target_canonical_ids=ids,
                    metadata={
                        "model_slug": model,
                        "trial_half": half,
                        "trial_manifest_sha256": trial_hash,
                    },
                ),
            )
    inventory = []
    for model in ("nice", "atm_s", "our_project"):
        for directory, half in (
            ("standard", "standard"),
            ("eeg_a", "a"),
            ("eeg_b", "b"),
        ):
            path = artifact_root / model / directory
            inventory.append(
                {
                    "model_slug": model,
                    "trial_half": half,
                    "path": str(path),
                    "sha256": _score_artifact_sha256(path),
                }
            )
    audit_run = {
        "epoch": 1,
        "checkpoint": "/sealed/epoch_0001.pth",
        "checkpoint_sha256": "e" * 64,
        "effective_logit_scale": 1.0,
        "top1_count": 2,
        "top5_count": 2,
        "sample_count": 200,
    }
    for model in ("nice", "atm_s"):
        _write_json(
            root / "checkpoints" / model / "checkpoint_manifest.json",
            _checkpoint_manifest(model),
        )
        _write_json(
            artifact_root / model / "best_test_audit.json",
            {
                "schema_version": 1,
                "scope": "best_test_audit_only",
                "model_slug": model,
                "checkpoint_policy": "every_epoch_checkpoint",
                "fairness_artifact_created": False,
                "formal_artifact_inventory": inventory,
                "runs": [audit_run],
                "best_test": audit_run,
            },
        )
    return artifact_root, trial_manifest, protocol_hash, inventory


def _patch_task7_inventory(monkeypatch: pytest.MonkeyPatch, runner) -> None:
    def inventory(
        directories: tuple[Path, ...] | list[Path],
        *,
        expected_image_count: int,
    ) -> list[dict[str, object]]:
        assert expected_image_count == 2
        entries = []
        for path in directories:
            artifact = read_score_artifact(path)
            entries.append(
                {
                    "model_slug": artifact.metadata["model_slug"],
                    "trial_half": artifact.metadata["trial_half"],
                    "path": str(path),
                    "sha256": _score_artifact_sha256(path),
                }
            )
        return entries

    monkeypatch.setattr(runner, "_formal_artifact_inventory", inventory)
    monkeypatch.setattr(runner, "validate_trial_manifest", lambda *_args: {})


def test_task7_loader_accepts_exact_native_audits_and_brainrw_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.run_scenarios as runner

    artifact_root, trial_manifest, protocol_hash, _inventory = _task7_tree(tmp_path)
    _patch_task7_inventory(monkeypatch, runner)

    artifacts, hashes, _trial_hash = runner._load_formal_artifacts(
        artifact_root,
        trial_manifest_path=trial_manifest,
        expected_image_count=2,
        protocol_sha256=protocol_hash,
    )

    assert set(artifacts) == set(hashes) == {"nice", "atm_s", "our_project"}
    assert all(
        set(artifacts[model]) == {"standard", "a", "b"}
        for model in artifacts
    )


@pytest.mark.parametrize("case", ("manifest_hash", "run_hash", "path", "symlink"))
def test_task7_loader_uses_shared_brainrw_tamper_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    import scripts.run_scenarios as runner

    artifact_root, trial_manifest, protocol_hash, _inventory = _task7_tree(tmp_path)
    _patch_task7_inventory(monkeypatch, runner)
    brainrw = artifact_root / "our_project"
    manifest_path = brainrw / "export_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if case == "manifest_hash":
        manifest["artifacts"]["standard"]["sha256"] = "f" * 64
        _write_json(manifest_path, manifest)
    elif case == "run_hash":
        (brainrw / "runs/standard/metrics.json").write_text(
            "{}\n", encoding="utf-8"
        )
    elif case == "path":
        manifest["runs"]["standard"]["path"] = "../standard"
        _write_json(manifest_path, manifest)
    else:
        (brainrw / "runs/standard/link").symlink_to(brainrw / "runs/eeg_a")

    with pytest.raises(ValueError, match="BrainRW"):
        runner._load_formal_artifacts(
            artifact_root,
            trial_manifest_path=trial_manifest,
            expected_image_count=2,
            protocol_sha256=protocol_hash,
        )


def test_task7_run_gate_rejects_missing_native_audit_before_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.run_scenarios as runner

    artifact_root, trial_manifest, _protocol_hash, _inventory = _task7_tree(tmp_path)
    (artifact_root / "nice/best_test_audit.json").unlink()
    _patch_task7_inventory(monkeypatch, runner)
    monkeypatch.setattr(
        runner,
        "evaluate_artifact",
        lambda *_args: pytest.fail("evaluation ran before native audit gate"),
    )

    with pytest.raises(ValueError, match="audit|entry"):
        runner.run_scenarios(
            protocol_path=Path(
                "experiments/matching_fairness/configs/protocol_sub08_seed42.json"
            ),
            artifact_root=artifact_root,
            trial_manifest_path=trial_manifest,
            output_dir=tmp_path / "runs",
        )
