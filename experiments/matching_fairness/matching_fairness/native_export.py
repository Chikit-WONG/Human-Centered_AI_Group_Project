"""Formal native NICE/ATM-S score export with explicit identity semantics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np
import torch

from .artifacts import (
    ScoreArtifact,
    independent_ranks,
    publish_staged_directory,
    read_score_artifact,
    write_score_artifact,
)
from .native_training import (
    ENCODERS,
    _load_official_module,
    _official_source_context,
    _required_symbol,
    _validate_source_lock,
)
from .provenance import SourceLock, sha256_file
from .trial_splits import average_trial_half, validate_trial_manifest


_ARTIFACT_HALVES = {
    "standard": "standard",
    "eeg_a": "a",
    "eeg_b": "b",
}
_FORMAL_MODELS = frozenset({"nice", "atm_s", "our_project"})
_ASSET_REPO_ID = "LidongYang/EEG_Image_decode"
_ASSET_RELATIVE_PATHS = (
    "Preprocessed_data_250Hz/sub-08/preprocessed_eeg_training.npy",
    "Preprocessed_data_250Hz/sub-08/preprocessed_eeg_test.npy",
    "ViT-H-14_features_train.pt",
    "ViT-H-14_features_test.pt",
)


@dataclass(frozen=True)
class NativeCheckpointEvaluation:
    eeg_embeddings: np.ndarray
    similarity: np.ndarray
    logit_scale: float


@dataclass(frozen=True)
class NativeExportConfig:
    source_checkout: Path
    source_lock: Path
    asset_root: Path
    asset_lock: Path
    test_eeg: Path
    test_features: Path
    test_images: Path
    trial_manifest: Path
    checkpoint_dir: Path
    output_dir: Path
    model: str
    subject: str = "sub-08"
    device: str = "cuda"
    expected_image_count: int = 200
    n_chans: int = 63
    n_times: int = 250
    logit_scale_type: str = "exp"


@dataclass(frozen=True)
class NativeExportResult:
    artifact_paths: Mapping[str, Path]
    artifact_hashes: Mapping[str, str]


@dataclass(frozen=True)
class _NativeInputs:
    image_ids: tuple[str, ...]
    image_features: np.ndarray
    averaged_eeg: Mapping[str, np.ndarray]
    input_hashes: Mapping[str, str]


@dataclass(frozen=True)
class _CheckpointRecord:
    epoch: int
    val_loss: float
    path: Path
    sha256: str


def native_scores(
    model: str,
    eeg: np.ndarray,
    image_features: np.ndarray,
    *,
    logit_scale: float,
) -> np.ndarray:
    """Apply the official model-specific full-gallery score rule."""
    if model not in {"nice", "atm_s"}:
        raise ValueError("model must be nice or atm_s")
    eeg = np.asarray(eeg)
    image_features = np.asarray(image_features)
    if eeg.ndim != 2 or image_features.ndim != 2:
        raise ValueError("EEG and image features must be 2-D")
    if eeg.shape[1] != image_features.shape[1]:
        raise ValueError("EEG and image feature dimensions must match")
    if not np.isfinite(eeg).all() or not np.isfinite(image_features).all():
        raise ValueError("EEG and image features must be finite")
    scale = float(logit_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("logit_scale must be finite and positive")

    if model == "atm_s":
        eeg_norms = np.linalg.norm(eeg, axis=1, keepdims=True)
        image_norms = np.linalg.norm(image_features, axis=1, keepdims=True)
        if np.any(eeg_norms <= 0):
            raise ValueError("ATM-S requires finite nonzero EEG row norms")
        if np.any(image_norms <= 0):
            raise ValueError("ATM-S requires finite nonzero image row norms")
        eeg = eeg / eeg_norms
        image_features = image_features / image_norms
    scores = scale * (eeg @ image_features.T)
    if not np.isfinite(scores).all():
        raise ValueError("native scores contain NaN or Inf")
    return np.ascontiguousarray(scores)


def build_native_score_artifact(
    *,
    model: str,
    similarity: np.ndarray,
    query_ids: Sequence[str],
    gallery_ids: Sequence[str],
    target_ids: Sequence[str],
    trial_half: str,
    checkpoint: Path,
    checkpoint_sha256: str,
    logit_scale: float,
    query_embeddings: np.ndarray,
    metadata: Mapping[str, object] | None = None,
) -> ScoreArtifact:
    """Build and parity-check one native ScoreArtifact."""
    if trial_half not in {"standard", "a", "b"}:
        raise ValueError("trial_half must be standard, a, or b")
    query_ids = tuple(query_ids)
    gallery_ids = tuple(gallery_ids)
    target_ids = tuple(target_ids)
    artifact_metadata: dict[str, object] = {
        "model_slug": model,
        "trial_half": trial_half,
        "checkpoint_role": "val_selected_formal",
        "checkpoint": str(Path(checkpoint)),
        "checkpoint_sha256": checkpoint_sha256,
        "logit_scale_type": "exp",
        "effective_logit_scale": float(logit_scale),
        "query_embeddings_sha256": _sha256_array(query_embeddings),
    }
    if metadata is not None:
        overlap = sorted(set(artifact_metadata).intersection(metadata))
        if overlap:
            raise ValueError(f"native artifact metadata collision: {overlap}")
        artifact_metadata.update(metadata)
    artifact = ScoreArtifact(
        similarity=np.ascontiguousarray(similarity),
        query_ids=query_ids,
        gallery_entry_ids=gallery_ids,
        gallery_canonical_ids=gallery_ids,
        target_canonical_ids=target_ids,
        metadata=artifact_metadata,
    )
    ranks = independent_ranks(artifact)
    artifact_metadata["native_metrics"] = {
        "top1_count": int(np.count_nonzero(ranks <= 1)),
        "top5_count": int(np.count_nonzero(ranks <= 5)),
        "sample_count": len(ranks),
    }
    artifact = ScoreArtifact(
        similarity=artifact.similarity,
        query_ids=artifact.query_ids,
        gallery_entry_ids=artifact.gallery_entry_ids,
        gallery_canonical_ids=artifact.gallery_canonical_ids,
        target_canonical_ids=artifact.target_canonical_ids,
        metadata=artifact_metadata,
    )
    artifact.validate()
    recomputed = independent_ranks(artifact)
    recomputed_metrics = {
        "top1_count": int(np.count_nonzero(recomputed <= 1)),
        "top5_count": int(np.count_nonzero(recomputed <= 5)),
        "sample_count": len(recomputed),
    }
    if artifact.metadata["native_metrics"] != recomputed_metrics:
        raise RuntimeError("native metric parity check failed")
    return artifact


def evaluate_native_checkpoint(
    *,
    model: torch.nn.Module,
    checkpoint: Path,
    model_slug: str,
    eeg: np.ndarray,
    image_features: np.ndarray,
    subject_index: int,
    device: torch.device,
) -> NativeCheckpointEvaluation:
    """Load one full model state and evaluate all EEG rows exactly once."""
    checkpoint = Path(checkpoint)
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise ValueError(f"checkpoint must be a regular file: {checkpoint}")
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"could not safely load checkpoint: {checkpoint}") from error
    if not isinstance(state, Mapping) or any(
        not isinstance(name, str) or not isinstance(value, torch.Tensor)
        for name, value in state.items()
    ):
        raise ValueError("checkpoint must contain a tensor state_dict")
    model.load_state_dict(dict(state), strict=True)
    model.to(device).eval().requires_grad_(False)

    eeg_array = np.asarray(eeg)
    if eeg_array.ndim < 2 or not np.isfinite(eeg_array).all():
        raise ValueError("averaged EEG must be a finite array with a batch axis")
    if (
        isinstance(subject_index, bool)
        or not isinstance(subject_index, int)
        or subject_index < 0
    ):
        raise ValueError("subject_index must be a non-negative integer")
    eeg_tensor = torch.as_tensor(eeg_array, dtype=torch.float32, device=device)
    with torch.inference_mode():
        if model_slug == "atm_s":
            subject_ids = torch.full(
                (len(eeg_tensor),),
                subject_index,
                dtype=torch.long,
                device=device,
            )
            output = model(eeg_tensor, subject_ids)
        elif model_slug == "nice":
            output = model(eeg_tensor)
        else:
            raise ValueError("model_slug must be nice or atm_s")
        if isinstance(output, tuple):
            output = output[0]
        if not isinstance(output, torch.Tensor) or output.ndim != 2:
            raise ValueError("official encoder must return a 2-D feature tensor")
        raw_scale = getattr(model, "logit_scale", None)
        if not isinstance(raw_scale, torch.Tensor) or raw_scale.numel() != 1:
            raise ValueError("official encoder lacks scalar logit_scale")
        effective_scale = raw_scale.detach().float().exp().clamp(max=100.0)
        logit_scale = float(effective_scale.cpu().item())
        embeddings = np.ascontiguousarray(output.detach().float().cpu().numpy())
    similarity = native_scores(
        model_slug,
        embeddings,
        image_features,
        logit_scale=logit_scale,
    )
    return NativeCheckpointEvaluation(
        eeg_embeddings=embeddings,
        similarity=similarity,
        logit_scale=logit_scale,
    )


def export_native_scores(config: NativeExportConfig) -> NativeExportResult:
    """Publish standard/EEG-A/EEG-B artifacts from only ``best_val.pth``."""
    source_lock = _validate_export_config(config, require_new_output=True)
    asset_lock = _validate_asset_lock(config)
    inputs = _load_native_inputs(config)
    records, best_checkpoint, checkpoint_manifest_hash = _load_checkpoint_manifest(
        config, source_lock, asset_lock
    )
    del records
    device = _resolve_device(config.device)
    subject_index = _subject_index(config.subject)

    evaluations: dict[str, NativeCheckpointEvaluation] = {}
    with _official_source_context(config.source_checkout):
        model = _build_official_encoder(config, source_lock)
        for artifact_name, trial_half in _ARTIFACT_HALVES.items():
            evaluations[artifact_name] = evaluate_native_checkpoint(
                model=model,
                checkpoint=best_checkpoint,
                model_slug=config.model,
                eeg=inputs.averaged_eeg[trial_half],
                image_features=inputs.image_features,
                subject_index=subject_index,
                device=device,
            )

    if np.array_equal(
        evaluations["eeg_a"].eeg_embeddings,
        evaluations["eeg_b"].eeg_embeddings,
    ):
        raise ValueError("EEG-A and EEG-B query embeddings are byte-identical")

    checkpoint_hash = sha256_file(best_checkpoint)
    artifacts: dict[str, ScoreArtifact] = {}
    for artifact_name, trial_half in _ARTIFACT_HALVES.items():
        evaluation = evaluations[artifact_name]
        artifacts[artifact_name] = build_native_score_artifact(
            model=config.model,
            similarity=evaluation.similarity,
            query_ids=inputs.image_ids,
            gallery_ids=inputs.image_ids,
            target_ids=inputs.image_ids,
            trial_half=trial_half,
            checkpoint=best_checkpoint,
            checkpoint_sha256=checkpoint_hash,
            logit_scale=evaluation.logit_scale,
            query_embeddings=evaluation.eeg_embeddings,
            metadata={
                "source_lock": source_lock.to_dict(),
                "asset_lock_manifest_sha256": asset_lock["manifest_sha256"],
                "asset_lock": asset_lock["provenance"],
                "checkpoint_manifest_sha256": checkpoint_manifest_hash,
                "input_sha256": dict(inputs.input_hashes),
                "trial_manifest_sha256": inputs.input_hashes["trial_manifest"],
                "subject": config.subject,
                "seed": 42,
            },
        )
    _validate_native_artifacts(artifacts, config.expected_image_count)

    config.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{config.output_dir.name}.tmp-",
            dir=config.output_dir.parent,
        )
    )
    artifact_paths: dict[str, Path] = {}
    artifact_hashes: dict[str, str] = {}
    try:
        for artifact_name in _ARTIFACT_HALVES:
            staged_path = staging / artifact_name
            write_score_artifact(staged_path, artifacts[artifact_name])
            read_score_artifact(staged_path)
            artifact_paths[artifact_name] = config.output_dir / artifact_name
            artifact_hashes[artifact_name] = _score_artifact_sha256(staged_path)
        publish_staged_directory(staging, config.output_dir)
    finally:
        if os.path.lexists(staging):
            shutil.rmtree(staging)
    return NativeExportResult(
        artifact_paths=artifact_paths,
        artifact_hashes=artifact_hashes,
    )


def audit_native_checkpoints(
    config: NativeExportConfig,
    formal_artifact_directories: Sequence[Path],
) -> Path:
    """Audit every epoch only after hashing the complete nine-artifact grid."""
    source_lock = _validate_export_config(config, require_new_output=False)
    asset_lock = _validate_asset_lock(config)
    inventory = _formal_artifact_inventory(
        formal_artifact_directories,
        expected_image_count=config.expected_image_count,
    )
    inputs = _load_native_inputs(config)
    records, _best_checkpoint, _manifest_hash = _load_checkpoint_manifest(
        config, source_lock, asset_lock
    )
    device = _resolve_device(config.device)
    subject_index = _subject_index(config.subject)

    runs: list[dict[str, object]] = []
    with _official_source_context(config.source_checkout):
        model = _build_official_encoder(config, source_lock)
        for record in records:
            evaluation = evaluate_native_checkpoint(
                model=model,
                checkpoint=record.path,
                model_slug=config.model,
                eeg=inputs.averaged_eeg["standard"],
                image_features=inputs.image_features,
                subject_index=subject_index,
                device=device,
            )
            artifact = build_native_score_artifact(
                model=config.model,
                similarity=evaluation.similarity,
                query_ids=inputs.image_ids,
                gallery_ids=inputs.image_ids,
                target_ids=inputs.image_ids,
                trial_half="standard",
                checkpoint=record.path,
                checkpoint_sha256=record.sha256,
                logit_scale=evaluation.logit_scale,
                query_embeddings=evaluation.eeg_embeddings,
            )
            metrics = artifact.metadata["native_metrics"]
            if not isinstance(metrics, Mapping):
                raise RuntimeError("native audit metrics are malformed")
            runs.append(
                {
                    "epoch": record.epoch,
                    "checkpoint": str(record.path),
                    "checkpoint_sha256": record.sha256,
                    "effective_logit_scale": evaluation.logit_scale,
                    "top1_count": int(metrics["top1_count"]),
                    "top5_count": int(metrics["top5_count"]),
                    "sample_count": int(metrics["sample_count"]),
                }
            )
    if not runs:
        raise ValueError("no epoch checkpoints available for audit")
    best = max(
        runs,
        key=lambda row: (
            int(row["top1_count"]),
            int(row["top5_count"]),
            -int(row["epoch"]),
        ),
    )
    payload = {
        "schema_version": 1,
        "scope": "best_test_audit_only",
        "model_slug": config.model,
        "checkpoint_policy": "every_epoch_checkpoint",
        "fairness_artifact_created": False,
        "formal_artifact_inventory": inventory,
        "runs": runs,
        "best_test": best,
    }
    audit_path = config.output_dir / "best_test_audit.json"
    if audit_path.exists() or audit_path.is_symlink():
        raise FileExistsError(f"audit output already exists: {audit_path}")
    _atomic_write_json(audit_path, payload)
    return audit_path


def _validate_export_config(
    config: NativeExportConfig,
    *,
    require_new_output: bool,
) -> SourceLock:
    if config.model not in {"nice", "atm_s"}:
        raise ValueError("model must be nice or atm_s")
    if not re.fullmatch(r"sub-\d{2}", config.subject):
        raise ValueError("subject must use sub-NN form")
    if config.logit_scale_type != "exp":
        raise ValueError("formal native export requires logit_scale_type='exp'")
    for name in ("expected_image_count", "n_chans", "n_times"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    paths = {
        "source checkout": config.source_checkout,
        "source lock": config.source_lock,
        "asset root": config.asset_root,
        "asset lock": config.asset_lock,
        "test EEG": config.test_eeg,
        "test features": config.test_features,
        "test images": config.test_images,
        "trial manifest": config.trial_manifest,
        "checkpoint directory": config.checkpoint_dir,
    }
    for label, path in paths.items():
        path = Path(path)
        if path.is_symlink():
            raise ValueError(f"{label} must not be a symbolic link: {path}")
    for label in (
        "source checkout",
        "asset root",
        "test images",
        "checkpoint directory",
    ):
        if not Path(paths[label]).is_dir():
            raise ValueError(f"{label} must be a directory: {paths[label]}")
    for label in (
        "source lock",
        "asset lock",
        "test EEG",
        "test features",
        "trial manifest",
    ):
        if not Path(paths[label]).is_file():
            raise ValueError(f"{label} must be a file: {paths[label]}")
    if config.test_eeg.name != "preprocessed_eeg_test.npy":
        raise ValueError("test EEG must be named preprocessed_eeg_test.npy")
    if config.test_features.name != "ViT-H-14_features_test.pt":
        raise ValueError("test features must be named ViT-H-14_features_test.pt")
    if config.checkpoint_dir.name != config.model:
        raise ValueError("checkpoint directory name must match model")
    if require_new_output and (
        config.output_dir.exists() or os.path.lexists(config.output_dir)
    ):
        raise FileExistsError(f"output directory already exists: {config.output_dir}")
    if not require_new_output and not config.output_dir.is_dir():
        raise ValueError("native main artifacts must exist before audit")
    return _validate_source_lock(config.source_checkout, config.source_lock)


def _validate_asset_lock(config: NativeExportConfig) -> dict[str, object]:
    """Verify all four official assets before any pickle-capable load."""
    lock_path = Path(config.asset_lock)
    root = Path(config.asset_root)
    if lock_path.is_symlink() or not lock_path.is_file():
        raise ValueError("asset lock must be a regular file")
    if root.is_symlink() or not root.is_dir():
        raise ValueError("asset root must be a regular directory")
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid asset lock manifest") from error
    if not isinstance(payload, Mapping) or set(payload) != {
        "repo_id",
        "repo_type",
        "asset_root",
        "files",
    }:
        raise ValueError("asset lock must contain exactly the Task 2 schema")
    if payload["repo_id"] != _ASSET_REPO_ID or payload["repo_type"] != "dataset":
        raise ValueError("asset lock repository identity mismatch")
    if payload["asset_root"] != str(root):
        raise ValueError("asset lock root path mismatch")
    files = payload["files"]
    if not isinstance(files, Mapping) or set(files) != set(_ASSET_RELATIVE_PATHS):
        raise ValueError("asset lock file paths do not match the four official assets")
    root_resolved = root.resolve(strict=True)
    expected_test_paths = {
        "Preprocessed_data_250Hz/sub-08/preprocessed_eeg_test.npy": Path(config.test_eeg),
        "ViT-H-14_features_test.pt": Path(config.test_features),
    }
    for relative in _ASSET_RELATIVE_PATHS:
        path = root / relative
        if relative in expected_test_paths and path != expected_test_paths[relative]:
            raise ValueError(f"configured official asset path mismatch: {relative}")
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"official asset must be a regular file: {relative}")
        resolved = path.resolve(strict=True)
        try:
            resolved.relative_to(root_resolved)
        except ValueError as error:
            raise ValueError(f"official asset escapes asset root: {relative}") from error
        entry = files[relative]
        if not isinstance(entry, Mapping) or set(entry) != {"bytes", "sha256"}:
            raise ValueError(f"asset lock entry schema mismatch: {relative}")
        expected_size = entry["bytes"]
        expected_hash = entry["sha256"]
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size < 0
        ):
            raise ValueError(f"asset lock byte count is invalid: {relative}")
        if not isinstance(expected_hash, str) or re.fullmatch(
            r"[0-9a-f]{64}", expected_hash
        ) is None:
            raise ValueError(f"asset lock SHA-256 is invalid: {relative}")
        if path.stat().st_size != expected_size:
            raise ValueError(f"asset byte count mismatch: {relative}")
        if sha256_file(path) != expected_hash:
            raise ValueError(f"asset SHA-256 mismatch: {relative}")
    return {
        "manifest_sha256": sha256_file(lock_path),
        "provenance": {
            "repo_id": payload["repo_id"],
            "repo_type": payload["repo_type"],
            "asset_root": payload["asset_root"],
            "files": {key: dict(value) for key, value in files.items()},
        },
    }


def _load_native_inputs(config: NativeExportConfig) -> _NativeInputs:
    suffixes = {".jpg", ".jpeg", ".png"}
    image_paths = sorted(
        path
        for path in config.test_images.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )
    image_ids = tuple(path.stem for path in image_paths)
    if len(image_ids) != config.expected_image_count:
        raise ValueError(
            f"expected {config.expected_image_count} test images, found {len(image_ids)}"
        )
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("official test images have duplicate canonical stems")

    loaded = np.load(config.test_eeg, allow_pickle=True)
    try:
        payload = loaded.item() if getattr(loaded, "shape", None) == () else loaded
        if not isinstance(payload, Mapping) or "preprocessed_eeg_data" not in payload:
            raise ValueError("official test EEG lacks preprocessed_eeg_data")
        eeg = np.asarray(payload["preprocessed_eeg_data"])
    finally:
        close = getattr(loaded, "close", None)
        if close is not None:
            close()
    expected_shape = (
        config.expected_image_count,
        80,
        config.n_chans,
        config.n_times,
    )
    if eeg.shape != expected_shape or not np.isfinite(eeg).all():
        raise ValueError(
            f"official test EEG must be finite with shape {expected_shape}, "
            f"found {eeg.shape}"
        )

    try:
        features = torch.load(
            config.test_features,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        raise ValueError("could not safely load official test features") from error
    if not isinstance(features, Mapping) or "img_features" not in features:
        raise ValueError("official test features lack img_features")
    image_tensor = features["img_features"]
    if not isinstance(image_tensor, torch.Tensor) or image_tensor.ndim != 2:
        raise ValueError("official img_features must be a 2-D tensor")
    image_features = np.ascontiguousarray(image_tensor.float().cpu().numpy())
    if (
        len(image_features) != config.expected_image_count
        or not np.isfinite(image_features).all()
    ):
        raise ValueError("official img_features row count or values are invalid")

    try:
        manifest = json.loads(config.trial_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("invalid trial manifest") from error
    if not isinstance(manifest, Mapping):
        raise ValueError("trial manifest must be a mapping")
    validate_trial_manifest(manifest, image_ids)
    averaged = {
        "standard": np.ascontiguousarray(eeg.mean(axis=1)),
        "a": np.ascontiguousarray(average_trial_half(eeg, image_ids, manifest, "a")),
        "b": np.ascontiguousarray(average_trial_half(eeg, image_ids, manifest, "b")),
    }
    return _NativeInputs(
        image_ids=image_ids,
        image_features=image_features,
        averaged_eeg=averaged,
        input_hashes={
            "test_eeg": sha256_file(config.test_eeg),
            "test_features": sha256_file(config.test_features),
            "trial_manifest": sha256_file(config.trial_manifest),
        },
    )


def _load_checkpoint_manifest(
    config: NativeExportConfig,
    source_lock: SourceLock,
    asset_lock: Mapping[str, object],
) -> tuple[tuple[_CheckpointRecord, ...], Path, str]:
    manifest_path = config.checkpoint_dir / "checkpoint_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("checkpoint_manifest.json must be a regular file")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("invalid checkpoint manifest") from error
    expected_top_keys = {
        "schema_version", "model", "encoder_type", "subject", "seed",
        "source", "inputs", "hyperparameters", "encoder_behavior",
        "checkpoints", "selection", "best_checkpoint", "history",
        "stopped_early",
    }
    if not isinstance(manifest, Mapping) or set(manifest) != expected_top_keys:
        raise ValueError("checkpoint manifest must use the exact Task 5 schema")
    encoder = ENCODERS[config.model]
    if (
        manifest["schema_version"] != 1
        or manifest["model"] != config.model
        or manifest["encoder_type"] != encoder["encoder_type"]
        or manifest["subject"] != config.subject
        or manifest["seed"] != 42
    ):
        raise ValueError("checkpoint manifest formal identity mismatch")
    if manifest["source"] != source_lock.to_dict():
        raise ValueError("checkpoint manifest source provenance mismatch")

    provenance = asset_lock.get("provenance")
    if not isinstance(provenance, Mapping):
        raise RuntimeError("validated asset provenance is malformed")
    asset_files = provenance.get("files")
    if not isinstance(asset_files, Mapping):
        raise RuntimeError("validated asset file ledger is malformed")
    training_eeg_entry = asset_files.get(
        "Preprocessed_data_250Hz/sub-08/preprocessed_eeg_training.npy"
    )
    training_features_entry = asset_files.get("ViT-H-14_features_train.pt")
    if not isinstance(training_eeg_entry, Mapping) or not isinstance(
        training_features_entry, Mapping
    ):
        raise RuntimeError("validated training asset entries are malformed")
    expected_inputs = {
        "training_eeg": {
            "name": "preprocessed_eeg_training.npy",
            "sha256": training_eeg_entry["sha256"],
        },
        "training_features": {
            "name": "ViT-H-14_features_train.pt",
            "sha256": training_features_entry["sha256"],
        },
    }
    if manifest["inputs"] != expected_inputs:
        raise ValueError("checkpoint manifest training inputs mismatch asset lock")
    expected_hyperparameters = {
        "epochs": 500,
        "batch_size": 1024,
        "learning_rate": 3e-4,
        "val_ratio": 0.1,
        "early_stopping_patience": 10,
        "ema_decay": 0.999,
        "logit_scale_type": "exp",
        "avg_trials": True,
        "n_chans": config.n_chans,
        "n_times": config.n_times,
    }
    if manifest["hyperparameters"] != expected_hyperparameters:
        raise ValueError("checkpoint manifest formal hyperparameters mismatch")
    expected_behavior = {
        "use_subject_id": encoder["use_subject_id"],
        "normalize_feats": encoder["normalize_feats"],
    }
    if manifest["encoder_behavior"] != expected_behavior:
        raise ValueError("checkpoint manifest encoder behavior mismatch")
    if not isinstance(manifest["stopped_early"], bool):
        raise ValueError("checkpoint manifest stopped_early must be boolean")

    rows = manifest["checkpoints"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("checkpoint manifest has no epoch checkpoints")
    records: list[_CheckpointRecord] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "epoch", "val_loss", "checkpoint", "sha256"
        }:
            raise ValueError("checkpoint manifest row schema is invalid")
        epoch = row.get("epoch")
        val_loss = row.get("val_loss")
        name = row.get("checkpoint")
        expected_hash = row.get("sha256")
        if (
            isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or epoch <= 0
            or isinstance(val_loss, bool)
            or not isinstance(val_loss, (int, float))
            or not math.isfinite(float(val_loss))
            or name != f"epoch_{epoch:04d}.pth"
            or not isinstance(expected_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None
        ):
            raise ValueError("checkpoint manifest epoch row is invalid")
        path = config.checkpoint_dir / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"epoch checkpoint is not a regular file: {path}")
        if sha256_file(path) != expected_hash:
            raise ValueError(f"epoch checkpoint SHA-256 mismatch: {name}")
        records.append(_CheckpointRecord(epoch, float(val_loss), path, expected_hash))
    if len({record.epoch for record in records}) != len(records):
        raise ValueError("checkpoint manifest epochs must be unique")
    if [record.epoch for record in records] != sorted(
        record.epoch for record in records
    ):
        raise ValueError("checkpoint manifest rows must be ordered by epoch")
    actual_epoch_files = {
        path.name for path in config.checkpoint_dir.glob("epoch_*.pth")
    }
    if actual_epoch_files != {record.path.name for record in records}:
        raise ValueError("checkpoint manifest does not cover every epoch checkpoint")

    best = manifest["best_checkpoint"]
    selection = manifest["selection"]
    if (
        not isinstance(best, Mapping)
        or set(best) != {"name", "sha256"}
        or best.get("name") != "best_val.pth"
        or not isinstance(best.get("sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", best["sha256"]) is None
        or not isinstance(selection, Mapping)
        or set(selection) != {"epoch", "val_loss", "checkpoint"}
    ):
        raise ValueError("checkpoint manifest best validation entry is invalid")
    best_path = config.checkpoint_dir / "best_val.pth"
    if best_path.is_symlink() or not best_path.is_file():
        raise ValueError("best_val.pth must be a regular file")
    best_hash = sha256_file(best_path)
    if best_hash != best["sha256"]:
        raise ValueError("best_val.pth SHA-256 mismatch")
    selected = min(records, key=lambda record: (record.val_loss, record.epoch))
    if selection != {
        "epoch": selected.epoch,
        "val_loss": selected.val_loss,
        "checkpoint": selected.path.name,
    }:
        raise ValueError("checkpoint manifest validation selection mismatch")
    if selected.sha256 != best_hash:
        raise ValueError("best_val.pth does not match selected validation epoch")
    history = manifest["history"]
    if (
        not isinstance(history, Mapping)
        or set(history) != {"name", "sha256"}
        or history.get("name") != "history.csv"
        or not isinstance(history.get("sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", history["sha256"]) is None
    ):
        raise ValueError("checkpoint manifest history entry is invalid")
    history_path = config.checkpoint_dir / "history.csv"
    if history_path.is_symlink() or not history_path.is_file():
        raise ValueError("history.csv must be a regular file")
    if sha256_file(history_path) != history["sha256"]:
        raise ValueError("history.csv SHA-256 mismatch")
    return tuple(records), best_path, sha256_file(manifest_path)


def _build_official_encoder(
    config: NativeExportConfig,
    _source_lock: SourceLock,
) -> torch.nn.Module:
    module = _load_official_module(
        config.source_checkout / "Retrieval/eeg_encoders.py",
        "Retrieval.eeg_encoders",
    )
    builder = _required_symbol(module, "build_encoder")
    return builder(
        ENCODERS[config.model]["encoder_type"],
        n_chans=config.n_chans,
        n_times=config.n_times,
        joint_train=False,
    )


def _validate_native_artifacts(
    artifacts: Mapping[str, ScoreArtifact],
    expected_image_count: int,
) -> None:
    if set(artifacts) != set(_ARTIFACT_HALVES):
        raise ValueError("native export must contain standard, eeg_a, and eeg_b")
    gallery_ids = artifacts["standard"].gallery_canonical_ids
    for name, artifact in artifacts.items():
        artifact.validate()
        if artifact.similarity.shape != (
            expected_image_count,
            expected_image_count,
        ):
            raise ValueError(f"{name} native score matrix has invalid shape")
        if artifact.gallery_canonical_ids != gallery_ids:
            raise ValueError("native artifacts do not share canonical gallery order")
        ranks = independent_ranks(artifact)
        metrics = artifact.metadata.get("native_metrics")
        expected = {
            "top1_count": int(np.count_nonzero(ranks <= 1)),
            "top5_count": int(np.count_nonzero(ranks <= 5)),
            "sample_count": len(ranks),
        }
        if metrics != expected:
            raise ValueError("native artifact metric counts failed parity")


def _formal_artifact_inventory(
    directories: Sequence[Path],
    *,
    expected_image_count: int,
) -> list[dict[str, object]]:
    paths = tuple(Path(path) for path in directories)
    if len(paths) != 9 or len(set(paths)) != 9:
        raise ValueError("audit requires exactly nine distinct formal artifacts")
    expected_grid = {
        (model, half)
        for model in _FORMAL_MODELS
        for half in ("standard", "a", "b")
    }
    entries: list[dict[str, object]] = []
    seen: set[tuple[object, object]] = set()
    common_gallery: tuple[str, ...] | None = None
    for path in paths:
        artifact = read_score_artifact(path)
        model = artifact.metadata.get("model_slug")
        half = artifact.metadata.get("trial_half")
        key = (model, half)
        if key in seen:
            raise ValueError(f"duplicate formal artifact role: {key}")
        seen.add(key)
        if artifact.similarity.shape != (
            expected_image_count,
            expected_image_count,
        ):
            raise ValueError("formal artifact matrix must match canonical image count")
        expected_role = (
            "fixed_formal" if model == "our_project" else "val_selected_formal"
        )
        if artifact.metadata.get("checkpoint_role") != expected_role:
            raise ValueError(f"formal artifact checkpoint role mismatch: {key}")
        if (
            artifact.metadata.get("subject") != "sub-08"
            or artifact.metadata.get("seed") != 42
        ):
            raise ValueError(f"formal artifact subject/seed mismatch: {key}")
        if not (
            artifact.query_ids
            == artifact.target_canonical_ids
            == artifact.gallery_entry_ids
            == artifact.gallery_canonical_ids
        ):
            raise ValueError("formal artifacts require canonical query/target/gallery order")
        if common_gallery is None:
            common_gallery = artifact.gallery_canonical_ids
        elif artifact.gallery_canonical_ids != common_gallery:
            raise ValueError("formal artifacts must share canonical gallery order")
        ranks = independent_ranks(artifact)
        expected_metrics = {
            "top1_count": int(np.count_nonzero(ranks <= 1)),
            "top5_count": int(np.count_nonzero(ranks <= 5)),
            "sample_count": len(ranks),
        }
        if artifact.metadata.get("native_metrics") != expected_metrics:
            raise ValueError(f"formal artifact metric parity failed: {key}")
        _validate_formal_artifact_provenance(artifact, str(model))
        entries.append(
            {
                "model_slug": model,
                "trial_half": half,
                "path": str(path),
                "sha256": _score_artifact_sha256(path),
            }
        )
    if seen != expected_grid:
        raise ValueError("formal artifact inventory must cover the 3 x 3 grid")
    return sorted(
        entries,
        key=lambda entry: (str(entry["model_slug"]), str(entry["trial_half"])),
    )


def _validate_formal_artifact_provenance(
    artifact: ScoreArtifact,
    model: str,
) -> None:
    metadata = artifact.metadata
    for field in ("query_embeddings_sha256", "trial_manifest_sha256"):
        if not _is_sha256(metadata.get(field)):
            raise ValueError(f"formal artifact lacks immutable {field}")
    if model in {"nice", "atm_s"}:
        for field in (
            "checkpoint_sha256",
            "checkpoint_manifest_sha256",
            "asset_lock_manifest_sha256",
        ):
            if not _is_sha256(metadata.get(field)):
                raise ValueError(f"native formal artifact lacks immutable {field}")
        source = metadata.get("source_lock")
        if not isinstance(source, Mapping) or set(source) != {
            "url", "branch", "commit", "checkout_sha256"
        }:
            raise ValueError("native formal artifact source lock is incomplete")
        if not _is_sha256(source.get("checkout_sha256")):
            raise ValueError("native formal artifact source checkout hash is invalid")
        if (
            not isinstance(source.get("url"), str)
            or not source["url"]
            or source.get("branch") != "develop"
            or not isinstance(source.get("commit"), str)
            or re.fullmatch(r"[0-9a-f]{40}", source["commit"]) is None
        ):
            raise ValueError("native formal artifact source identity is invalid")
        inputs = metadata.get("input_sha256")
        if not isinstance(inputs, Mapping) or set(inputs) != {
            "test_eeg", "test_features", "trial_manifest"
        } or not all(_is_sha256(value) for value in inputs.values()):
            raise ValueError("native formal artifact input provenance is incomplete")
        if inputs["trial_manifest"] != metadata["trial_manifest_sha256"]:
            raise ValueError("native formal artifact trial manifest hashes disagree")
        _validate_embedded_asset_lock(metadata.get("asset_lock"))
        if (
            metadata.get("logit_scale_type") != "exp"
            or not isinstance(metadata.get("checkpoint"), str)
        ):
            raise ValueError("native formal artifact checkpoint semantics are incomplete")
        return

    if model != "our_project":
        raise ValueError(f"unsupported formal artifact model: {model}")
    for field in (
        "checkpoint_content_sha256",
        "brain_test_sha256",
        "evaluator_sha256",
        "protocol_sha256",
    ):
        if not _is_sha256(metadata.get(field)):
            raise ValueError(f"BrainRW formal artifact lacks immutable {field}")
    content = metadata.get("model_content_sha256")
    if not isinstance(content, Mapping) or set(content) != {
        "brain_model", "vision_adapter", "pretrained_vision_base"
    } or not all(_is_sha256(value) for value in content.values()):
        raise ValueError("BrainRW model content provenance is incomplete")
    if content["brain_model"] != metadata["checkpoint_content_sha256"]:
        raise ValueError("BrainRW checkpoint content hashes disagree")
    if metadata.get("evaluator_version") != "AIAA3800-BRAINRW-FORMAL-v1":
        raise ValueError("BrainRW evaluator identity is invalid")
    if (
        metadata.get("similarity") != "cosine"
        or not isinstance(metadata.get("checkpoint"), str)
    ):
        raise ValueError("BrainRW formal scoring semantics are incomplete")


def _validate_embedded_asset_lock(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "repo_id", "repo_type", "asset_root", "files"
    }:
        raise ValueError("native formal artifact asset lock provenance is incomplete")
    if (
        value["repo_id"] != _ASSET_REPO_ID
        or value["repo_type"] != "dataset"
        or not isinstance(value["asset_root"], str)
        or not value["asset_root"]
    ):
        raise ValueError("native formal artifact asset lock identity is invalid")
    files = value["files"]
    if not isinstance(files, Mapping) or set(files) != set(_ASSET_RELATIVE_PATHS):
        raise ValueError("native formal artifact asset file ledger is incomplete")
    for entry in files.values():
        if (
            not isinstance(entry, Mapping)
            or set(entry) != {"bytes", "sha256"}
            or isinstance(entry["bytes"], bool)
            or not isinstance(entry["bytes"], int)
            or entry["bytes"] < 0
            or not _is_sha256(entry["sha256"])
        ):
            raise ValueError("native formal artifact asset file entry is invalid")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("native export device must be cpu or cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _subject_index(subject: str) -> int:
    match = re.search(r"\d+", subject)
    if match is None:
        raise ValueError("subject lacks numeric identity")
    return int(match.group()) - 1


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _score_artifact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for name in ("metadata.json", "similarity.npy"):
        digest.update(name.encode("ascii"))
        digest.update(bytes.fromhex(sha256_file(path / name)))
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
