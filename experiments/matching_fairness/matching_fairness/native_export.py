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
import tempfile

import numpy as np
import torch

from .artifacts import (
    ScoreArtifact,
    independent_ranks,
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
from .trial_splits import average_trial_half


_ARTIFACT_HALVES = {
    "standard": "standard",
    "eeg_a": "a",
    "eeg_b": "b",
}
_FORMAL_MODELS = frozenset({"nice", "atm_s", "our_project"})


@dataclass(frozen=True)
class NativeCheckpointEvaluation:
    eeg_embeddings: np.ndarray
    similarity: np.ndarray
    logit_scale: float


@dataclass(frozen=True)
class NativeExportConfig:
    source_checkout: Path
    source_lock: Path
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
    inputs = _load_native_inputs(config)
    records, best_checkpoint = _load_checkpoint_manifest(config)
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
                "input_sha256": dict(inputs.input_hashes),
            },
        )
    _validate_native_artifacts(artifacts, config.expected_image_count)

    config.output_dir.parent.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir()
    artifact_paths: dict[str, Path] = {}
    artifact_hashes: dict[str, str] = {}
    for artifact_name in _ARTIFACT_HALVES:
        path = config.output_dir / artifact_name
        write_score_artifact(path, artifacts[artifact_name])
        read_score_artifact(path)
        artifact_paths[artifact_name] = path
        artifact_hashes[artifact_name] = _score_artifact_sha256(path)
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
    inventory = _formal_artifact_inventory(
        formal_artifact_directories,
        expected_image_count=config.expected_image_count,
    )
    inputs = _load_native_inputs(config)
    records, _best_checkpoint = _load_checkpoint_manifest(config)
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
    for label in ("source checkout", "test images", "checkpoint directory"):
        if not Path(paths[label]).is_dir():
            raise ValueError(f"{label} must be a directory: {paths[label]}")
    for label in ("source lock", "test EEG", "test features", "trial manifest"):
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
    manifest_ids = manifest.get("image_ids")
    manifest_images = manifest.get("images")
    if (
        not isinstance(manifest_ids, list)
        or any(not isinstance(value, str) or not value for value in manifest_ids)
        or len(set(manifest_ids)) != len(manifest_ids)
        or set(manifest_ids) != set(image_ids)
        or not isinstance(manifest_images, Mapping)
        or set(manifest_images) != set(image_ids)
    ):
        raise ValueError(
            "trial manifest canonical image IDs must exactly match test images"
        )
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
) -> tuple[tuple[_CheckpointRecord, ...], Path]:
    manifest_path = config.checkpoint_dir / "checkpoint_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("checkpoint_manifest.json must be a regular file")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("invalid checkpoint manifest") from error
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("model") != config.model
        or manifest.get("subject") != config.subject
    ):
        raise ValueError("checkpoint manifest model or subject mismatch")
    rows = manifest.get("checkpoints")
    if not isinstance(rows, list) or not rows:
        raise ValueError("checkpoint manifest has no epoch checkpoints")
    records: list[_CheckpointRecord] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("checkpoint manifest row must be a mapping")
        epoch = row.get("epoch")
        name = row.get("checkpoint")
        expected_hash = row.get("sha256")
        if (
            isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or epoch <= 0
            or name != f"epoch_{epoch:04d}.pth"
            or not isinstance(expected_hash, str)
        ):
            raise ValueError("checkpoint manifest epoch row is invalid")
        path = config.checkpoint_dir / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"epoch checkpoint is not a regular file: {path}")
        if sha256_file(path) != expected_hash:
            raise ValueError(f"epoch checkpoint SHA-256 mismatch: {name}")
        records.append(_CheckpointRecord(epoch, path, expected_hash))
    if len({record.epoch for record in records}) != len(records):
        raise ValueError("checkpoint manifest epochs must be unique")
    records.sort(key=lambda record: record.epoch)
    actual_epoch_files = {
        path.name for path in config.checkpoint_dir.glob("epoch_*.pth")
    }
    if actual_epoch_files != {record.path.name for record in records}:
        raise ValueError("checkpoint manifest does not cover every epoch checkpoint")

    best = manifest.get("best_checkpoint")
    selection = manifest.get("selection")
    if (
        not isinstance(best, Mapping)
        or best.get("name") != "best_val.pth"
        or not isinstance(best.get("sha256"), str)
        or not isinstance(selection, Mapping)
    ):
        raise ValueError("checkpoint manifest best validation entry is invalid")
    best_path = config.checkpoint_dir / "best_val.pth"
    if best_path.is_symlink() or not best_path.is_file():
        raise ValueError("best_val.pth must be a regular file")
    best_hash = sha256_file(best_path)
    if best_hash != best["sha256"]:
        raise ValueError("best_val.pth SHA-256 mismatch")
    selected_epoch = selection.get("epoch")
    selected = next(
        (record for record in records if record.epoch == selected_epoch),
        None,
    )
    if selected is None or selected.sha256 != best_hash:
        raise ValueError("best_val.pth does not match selected validation epoch")
    return tuple(records), best_path


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
        if common_gallery is None:
            common_gallery = artifact.gallery_canonical_ids
        elif artifact.gallery_canonical_ids != common_gallery:
            raise ValueError("formal artifacts must share canonical gallery order")
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
