"""Shared strict validators for fixed formal score-export trees."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat

from .artifacts import ScoreArtifact, independent_ranks, read_score_artifact
from .provenance import sha256_file, sha256_path


_ARTIFACT_HALVES = {"standard": "standard", "eeg_a": "a", "eeg_b": "b"}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_TOP_LEVEL_ENTRIES = {
    *_ARTIFACT_HALVES,
    "runs",
    "export_manifest.json",
}
_INPUT_FIELDS = {
    "protocol_sha256",
    "trial_manifest_sha256",
    "brain_test_sha256",
    "evaluator_sha256",
    "test_image_tree_sha256",
    "model_content_sha256",
}
_MODEL_CONTENT_FIELDS = {
    "brain_model",
    "vision_adapter",
    "pretrained_vision_base",
}
_ARTIFACT_METADATA_FIELDS = {
    "model_slug",
    "trial_half",
    "checkpoint_role",
    "checkpoint",
    "checkpoint_content_sha256",
    "similarity",
    "query_embeddings_sha256",
    "subject",
    "seed",
    "trial_manifest_sha256",
    "protocol_sha256",
    "brain_test_sha256",
    "model_content_sha256",
    "evaluator_version",
    "evaluator_sha256",
    "runtime_inputs",
    "native_metrics",
}


@dataclass(frozen=True)
class BrainRWExportTree:
    """One fully verified fixed-formal BrainRW export snapshot."""

    artifacts: Mapping[str, ScoreArtifact]
    artifact_sha256: Mapping[str, str]
    inputs: Mapping[str, object]


def validate_brainrw_export_tree(
    root: Path,
    *,
    expected_image_count: int,
    expected_inputs: Mapping[str, object] | None = None,
) -> BrainRWExportTree:
    """Validate and load the canonical five-entry BrainRW export tree."""

    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("BrainRW export root must be a regular directory")
    _reject_symlinks_and_special_files(root)
    if {entry.name for entry in root.iterdir()} != _TOP_LEVEL_ENTRIES:
        raise ValueError("BrainRW export root has an invalid exact entry set")

    manifest = _canonical_json(
        _read_regular_file(root / "export_manifest.json", "BrainRW export manifest"),
        "BrainRW export manifest",
    )
    expected_manifest_fields = {
        "schema_version",
        "scope",
        "checkpoint_role",
        "model_slug",
        "subject",
        "seed",
        "artifacts",
        "runs",
        "inputs",
    }
    if set(manifest) != expected_manifest_fields or (
        manifest.get("schema_version") != 1
        or manifest.get("scope") != "fixed_formal_export"
        or manifest.get("checkpoint_role") != "fixed_formal"
        or manifest.get("model_slug") != "our_project"
        or manifest.get("subject") != "sub-08"
        or manifest.get("seed") != 42
    ):
        raise ValueError("BrainRW export manifest identity/schema is invalid")

    inputs = manifest.get("inputs")
    _validate_inputs(inputs)
    assert isinstance(inputs, Mapping)
    if expected_inputs is not None and dict(inputs) != dict(expected_inputs):
        raise ValueError("BrainRW export manifest input identities mismatch")

    artifact_inventory = manifest.get("artifacts")
    run_inventory = manifest.get("runs")
    if (
        not isinstance(artifact_inventory, Mapping)
        or set(artifact_inventory) != set(_ARTIFACT_HALVES)
        or not isinstance(run_inventory, Mapping)
        or set(run_inventory) != set(_ARTIFACT_HALVES)
    ):
        raise ValueError("BrainRW export inventories are incomplete")

    artifacts: dict[str, ScoreArtifact] = {}
    artifact_hashes: dict[str, str] = {}
    common_gallery: tuple[str, ...] | None = None
    for directory, half in _ARTIFACT_HALVES.items():
        artifact_path = root / directory
        artifact_hash = _score_artifact_sha256(artifact_path)
        _validate_inventory_entry(
            artifact_inventory[directory],
            expected_path=directory,
            expected_sha256=artifact_hash,
            label="artifact",
        )
        run_path = root / "runs" / directory
        _validate_run_directory(run_path)
        _validate_inventory_entry(
            run_inventory[directory],
            expected_path=f"runs/{directory}",
            expected_sha256=sha256_path(run_path),
            label="run",
        )
        artifact = read_score_artifact(artifact_path)
        _validate_brainrw_artifact(
            artifact,
            half=half,
            expected_image_count=expected_image_count,
            inputs=inputs,
        )
        if common_gallery is None:
            common_gallery = artifact.gallery_canonical_ids
        elif artifact.gallery_canonical_ids != common_gallery:
            raise ValueError("BrainRW artifacts do not share canonical gallery order")
        artifacts[directory] = artifact
        artifact_hashes[directory] = artifact_hash

    if (
        artifacts["eeg_a"].metadata["query_embeddings_sha256"]
        == artifacts["eeg_b"].metadata["query_embeddings_sha256"]
    ):
        raise ValueError("BrainRW EEG-A and EEG-B query embeddings are identical")
    return BrainRWExportTree(
        artifacts=artifacts,
        artifact_sha256=artifact_hashes,
        inputs=dict(inputs),
    )


def _validate_inputs(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != _INPUT_FIELDS:
        raise ValueError("BrainRW export manifest input schema is invalid")
    for field in _INPUT_FIELDS - {"model_content_sha256"}:
        if _SHA256.fullmatch(str(value.get(field, ""))) is None:
            raise ValueError("BrainRW export manifest input hash is invalid")
    content = value.get("model_content_sha256")
    if (
        not isinstance(content, Mapping)
        or set(content) != _MODEL_CONTENT_FIELDS
        or any(_SHA256.fullmatch(str(content.get(field, ""))) is None for field in content)
    ):
        raise ValueError("BrainRW export manifest model identity is invalid")


def _validate_inventory_entry(
    value: object,
    *,
    expected_path: str,
    expected_sha256: str,
    label: str,
) -> None:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"path", "sha256"}
        or value.get("path") != expected_path
        or value.get("sha256") != expected_sha256
    ):
        raise ValueError(f"BrainRW {label} inventory path/hash is invalid")
    path = Path(expected_path)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != expected_path:
        raise ValueError(f"BrainRW {label} inventory path is not contained")


def _validate_run_directory(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ValueError("BrainRW run output must be a regular directory")
    if {entry.name for entry in path.iterdir()} != {"metrics.json", "predictions.csv"}:
        raise ValueError("BrainRW run output has an invalid exact entry set")
    for entry in path.iterdir():
        if entry.is_symlink() or not entry.is_file():
            raise ValueError("BrainRW run output member must be a regular file")


def _validate_brainrw_artifact(
    artifact: ScoreArtifact,
    *,
    half: str,
    expected_image_count: int,
    inputs: Mapping[str, object],
) -> None:
    metadata = artifact.metadata
    content = inputs["model_content_sha256"]
    assert isinstance(content, Mapping)
    runtime = metadata.get("runtime_inputs")
    metrics = metadata.get("native_metrics")
    ranks = independent_ranks(artifact)
    expected_metrics = {
        "top1_count": int((ranks <= 1).sum()),
        "top5_count": int((ranks <= 5).sum()),
        "sample_count": len(ranks),
    }
    channels = runtime.get("selected_channel_indices") if isinstance(runtime, Mapping) else None
    if (
        set(metadata) != _ARTIFACT_METADATA_FIELDS
        or artifact.similarity.shape != (expected_image_count, expected_image_count)
        or not (
            artifact.query_ids
            == artifact.target_canonical_ids
            == artifact.gallery_entry_ids
            == artifact.gallery_canonical_ids
        )
        or metadata.get("model_slug") != "our_project"
        or metadata.get("trial_half") != half
        or metadata.get("checkpoint_role") != "fixed_formal"
        or not isinstance(metadata.get("checkpoint"), str)
        or not metadata["checkpoint"]
        or metadata.get("checkpoint_content_sha256") != content["brain_model"]
        or metadata.get("similarity") != "cosine"
        or _SHA256.fullmatch(str(metadata.get("query_embeddings_sha256", ""))) is None
        or metadata.get("subject") != "sub-08"
        or metadata.get("seed") != 42
        or metadata.get("trial_manifest_sha256") != inputs["trial_manifest_sha256"]
        or metadata.get("protocol_sha256") != inputs["protocol_sha256"]
        or metadata.get("brain_test_sha256") != inputs["brain_test_sha256"]
        or metadata.get("model_content_sha256") != content
        or metadata.get("evaluator_version") != "AIAA3800-BRAINRW-FORMAL-v1"
        or metadata.get("evaluator_sha256") != inputs["evaluator_sha256"]
        or not isinstance(runtime, Mapping)
        or set(runtime) != {
            "test_image_tree_sha256",
            "selected_channel_indices",
            "time_slice",
            "dataset_name",
            "expected_sample_count",
        }
        or runtime.get("test_image_tree_sha256") != inputs["test_image_tree_sha256"]
        or not isinstance(channels, list)
        or not channels
        or any(isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 63 for index in channels)
        or len(set(channels)) != len(channels)
        or runtime.get("time_slice") != [0, 250]
        or runtime.get("dataset_name") != "things"
        or runtime.get("expected_sample_count") != expected_image_count
        or metrics != expected_metrics
    ):
        raise ValueError("BrainRW artifact formal identity/schema is invalid")


def _score_artifact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for name in ("metadata.json", "similarity.npy"):
        digest.update(name.encode("ascii"))
        digest.update(bytes.fromhex(sha256_file(Path(path) / name)))
    return digest.hexdigest()


def _reject_symlinks_and_special_files(root: Path) -> None:
    for entry in root.rglob("*"):
        if entry.is_symlink():
            raise ValueError(f"BrainRW export tree contains a symlink: {entry}")
        if not entry.is_dir() and not entry.is_file():
            raise ValueError(f"BrainRW export tree contains a special file: {entry}")


def _read_regular_file(path: Path, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{label} must be a non-symlink regular file") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    encoded = b"".join(chunks)
    if before_identity != after_identity or len(encoded) != before.st_size:
        raise ValueError(f"{label} changed while it was being read")
    return encoded


def _canonical_json(encoded: bytes, label: str) -> Mapping[str, object]:
    try:
        payload = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be valid UTF-8 JSON") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must contain a JSON object")
    try:
        canonical = (
            json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} contains non-canonical values") from error
    if encoded != canonical:
        raise ValueError(f"{label} must use canonical sorted JSON")
    return payload
