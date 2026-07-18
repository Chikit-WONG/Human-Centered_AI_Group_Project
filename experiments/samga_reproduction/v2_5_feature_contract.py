#!/usr/bin/env python3
"""Shared, auditable contract for InternViT-6B-448px-V2_5 feature artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 2
MODEL_REPO = "OpenGVLab/InternViT-6B-448px-V2_5"
MODEL_REVISION = "9d1a4344077479c93d42584b6941c64d795d508d"
MODEL_WEIGHT_SHA256 = {
    "model-00001-of-00003.safetensors": (
        "9818659d13d932da8bc0c3b8ee15f5b5d68d8c94d66eb525be566066630111da"
    ),
    "model-00002-of-00003.safetensors": (
        "4f0c10e72d6f6513f421baa6ec843d5508657435059c1d18b6b5fd7789f9d5b7"
    ),
    "model-00003-of-00003.safetensors": (
        "d21c4fe0bc4af1425cfae1a59a8f5fbb00fde9d8e2888325a60913ac61b0494d"
    ),
}
MODEL_WEIGHT_BYTES = {
    "model-00001-of-00003.safetensors": 4_988_565_944,
    "model-00002-of-00003.safetensors": 4_937_250_176,
    "model-00003-of-00003.safetensors": 1_147_238_088,
}
MODEL_SMALL_FILE_SHA256 = {
    "config.json": "4fc4a1187b20575c0da8d27df2ad17f5ad6e8ac1c8b2af707bc8b263bd40c0a2",
    "configuration_intern_vit.py": "e620864fe9f2ef0104b39ea496cb844e1b363caaf8208e6f0bef1a72f31f00a3",
    "flash_attention.py": "d84f36949763545b58039d28669f9dc46fcace6c94b796e3f91a92553f5f5cad",
    "model.safetensors.index.json": "94d376c898c00585a38a588df9ff354fa965eafa9a1d56f69c1c8bad7ad08502",
    "modeling_intern_vit.py": "56220ba82cb511d51d5f2fa71eebd728b330fbccad9dfb128088a8fcc8f7d260",
    "preprocessor_config.json": "0658115064c561026539aeeead9ed3b1a8e0cc90967df8c142849199f955d2b4",
}
CANONICAL_RECORDS_SHA256 = {
    "train": "f59500f36e273f66fce5c2019670b076d75d538feccf296c7d7ed75f19ae3fac",
    "test": "02d7e33b3fe8e5a571f8db232ca5fa86abb0c16981876ec84feae7ba64636f1a",
}
EXPECTED_SHARD_COUNTS = {"train": 8, "test": 1}
HIDDEN_SIZE = 3200
LOGICAL_LAYER_IDS = (20, 24, 28, 32, 36)
CAPTURED_BLOCK_OUTPUTS = (20, 21, 24, 25, 28, 29, 32, 33, 36, 37)
POOLING_FILENAMES = {
    "raw_cls": "raw_cls.npy",
    "patch_mean": "patch_mean.npy",
}
SHARD_ARTIFACT_KIND = "internvit_v2_5_feature_shard"
MERGED_ARTIFACT_KIND = "internvit_v2_5_feature_merged"


def hash_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_jsonable(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, destination)


def git_revision(root: str | Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def resolve_without_symlinks(path: str | Path, description: str) -> Path:
    """Return an absolute path only after rejecting every symlink component."""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate = Path(os.path.abspath(candidate))
    current = Path(candidate.anchor)
    for component in candidate.parts[1:]:
        current = current / component
        if current.is_symlink():
            raise ValueError(f"{description} contains a symlink component: {current}")
    return candidate.resolve(strict=False)


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = resolve_without_symlinks(path, "Manifest path")
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported manifest schema in {manifest_path}")
    if manifest.get("subject_id") != 1:
        raise ValueError(f"Only the canonical sub-01 manifest is allowed: {manifest_path}")
    split = manifest.get("split")
    if split not in CANONICAL_RECORDS_SHA256:
        raise ValueError(f"Unsupported manifest split in {manifest_path}")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"Manifest has no records: {manifest_path}")
    if manifest.get("records_sha256") != hash_jsonable(records):
        raise ValueError(f"Manifest record hash mismatch in {manifest_path}")
    if manifest["records_sha256"] != CANONICAL_RECORDS_SHA256[split]:
        raise ValueError(f"Manifest is not the canonical sub-01 {split} manifest")
    rows = [int(record.get("row_index", -1)) for record in records]
    if rows != list(range(len(records))):
        raise ValueError(f"Manifest row_index values are not contiguous in {manifest_path}")
    manifest["_manifest_path"] = str(manifest_path)
    manifest["_manifest_sha256"] = hash_file(manifest_path)
    return manifest


def logical_layer_routes() -> dict[str, list[dict[str, int]]]:
    """Map router-facing logical IDs onto actual captured block outputs."""
    return {
        "idx0": [
            {
                "logical_layer_id": logical,
                "captured_block_output": logical,
                "source_axis_index": CAPTURED_BLOCK_OUTPUTS.index(logical),
            }
            for logical in LOGICAL_LAYER_IDS
        ],
        "idx_plus_1": [
            {
                "logical_layer_id": logical,
                "captured_block_output": logical + 1,
                "source_axis_index": CAPTURED_BLOCK_OUTPUTS.index(logical + 1),
            }
            for logical in LOGICAL_LAYER_IDS
        ],
    }


def pooling_file_metadata(path: Path, shape: Sequence[int]) -> dict[str, Any]:
    return {
        "filename": path.name,
        "shape": [int(value) for value in shape],
        "dtype": "float16",
        "sha256": hash_file(path),
    }


def _require_regular_file(path: Path, description: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{description} is not a regular file: {path}")


def _validate_layer_contract(metadata: Mapping[str, Any], path: Path) -> None:
    if metadata.get("logical_layer_ids") != list(LOGICAL_LAYER_IDS):
        raise ValueError(f"Logical layer IDs do not match the V2.5 contract: {path}")
    if metadata.get("captured_block_outputs") != list(CAPTURED_BLOCK_OUTPUTS):
        raise ValueError(f"Captured block outputs do not match the V2.5 contract: {path}")
    if metadata.get("logical_layer_routes") != logical_layer_routes():
        raise ValueError(f"Logical-to-actual layer routes are invalid: {path}")


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _validate_merged_source_shards(
    metadata: Mapping[str, Any],
    manifest: Mapping[str, Any],
    metadata_path: Path,
) -> None:
    source_shards = metadata.get("source_shards")
    expected_count = EXPECTED_SHARD_COUNTS[manifest["split"]]
    if (
        not isinstance(source_shards, list)
        or len(source_shards) != expected_count
        or metadata.get("shard_count") != expected_count
    ):
        raise ValueError(
            f"Merged source-shard count is invalid: {metadata_path}"
        )
    total = len(manifest["records"])
    expected_keys = {
        "directory",
        "shard_index",
        "row_start",
        "row_end",
        "metadata_sha256",
        "pooling_sha256",
    }
    for index, entry in enumerate(source_shards):
        expected_start = total * index // expected_count
        expected_end = total * (index + 1) // expected_count
        if (
            not isinstance(entry, dict)
            or set(entry) != expected_keys
            or not isinstance(entry.get("directory"), str)
            or not Path(entry["directory"]).is_absolute()
            or entry.get("shard_index") != index
            or entry.get("row_start") != expected_start
            or entry.get("row_end") != expected_end
            or not _is_sha256(entry.get("metadata_sha256"))
            or not isinstance(entry.get("pooling_sha256"), dict)
            or set(entry["pooling_sha256"]) != set(POOLING_FILENAMES)
            or not all(
                _is_sha256(entry["pooling_sha256"][pooling])
                for pooling in POOLING_FILENAMES
            )
        ):
            raise ValueError(
                f"Merged source-shard entry {index} is invalid: {metadata_path}"
            )
    if metadata.get("source_shards_sha256") != hash_jsonable(source_shards):
        raise ValueError(
            f"Merged source-shard summary hash mismatch: {metadata_path}"
        )


@dataclass(frozen=True)
class VerifiedFeatureDirectory:
    directory: Path
    metadata: dict[str, Any]
    arrays: Mapping[str, np.ndarray]


def validate_feature_directory(
    manifest: Mapping[str, Any],
    feature_directory: str | Path,
    *,
    expected_artifact_kind: str | None = None,
) -> VerifiedFeatureDirectory:
    directory = resolve_without_symlinks(feature_directory, "Feature directory")
    if not directory.is_dir():
        raise ValueError(f"Feature directory is not a regular directory: {directory}")
    metadata_path = directory / "metadata.json"
    _require_regular_file(metadata_path, "Feature metadata")
    metadata = read_json(metadata_path)
    kind = metadata.get("artifact_kind")
    if (
        metadata.get("schema_version") != SCHEMA_VERSION
        or metadata.get("complete") is not True
        or kind not in {SHARD_ARTIFACT_KIND, MERGED_ARTIFACT_KIND}
    ):
        raise ValueError(f"Incomplete or unsupported feature metadata: {metadata_path}")
    if expected_artifact_kind is not None and kind != expected_artifact_kind:
        raise ValueError(
            f"Feature artifact kind {kind!r} does not match "
            f"{expected_artifact_kind!r}: {directory}"
        )
    if (
        metadata.get("model_repo") != MODEL_REPO
        or metadata.get("model_revision") != MODEL_REVISION
        or metadata.get("model_weight_sha256") != MODEL_WEIGHT_SHA256
        or metadata.get("model_small_file_sha256") != MODEL_SMALL_FILE_SHA256
    ):
        raise ValueError(f"Model identity/hash contract mismatch: {metadata_path}")
    if (
        metadata.get("subject_id") != 1 or
        metadata.get("records_sha256") != manifest.get("records_sha256")
        or metadata.get("manifest_sha256") != manifest.get("_manifest_sha256")
        or metadata.get("split") != manifest.get("split")
    ):
        raise ValueError(f"Feature rows/order do not match the manifest: {directory}")
    if metadata.get("hidden_size") != HIDDEN_SIZE:
        raise ValueError(f"Feature hidden_size must be {HIDDEN_SIZE}: {metadata_path}")
    _validate_layer_contract(metadata, metadata_path)

    row_start = int(metadata.get("row_start", -1))
    row_end = int(metadata.get("row_end", -1))
    total = len(manifest["records"])
    if not 0 <= row_start < row_end <= total:
        raise ValueError(f"Invalid feature row interval {row_start}:{row_end}: {directory}")
    if kind == MERGED_ARTIFACT_KIND and (row_start != 0 or row_end != total):
        raise ValueError(f"Merged features do not cover the full manifest: {directory}")

    entries = metadata.get("pooling_files")
    if not isinstance(entries, dict) or set(entries) != set(POOLING_FILENAMES):
        raise ValueError(f"Pooling file metadata is incomplete: {metadata_path}")
    arrays: dict[str, np.ndarray] = {}
    reference_shape: tuple[int, ...] | None = None
    for pooling, expected_filename in POOLING_FILENAMES.items():
        entry = entries[pooling]
        if entry.get("filename") != expected_filename:
            raise ValueError(f"Unexpected {pooling} filename in {metadata_path}")
        path = directory / expected_filename
        _require_regular_file(path, f"{pooling} feature array")
        expected_hash = entry.get("sha256")
        actual_hash = hash_file(path)
        if not expected_hash or actual_hash != expected_hash:
            raise ValueError(
                f"{pooling} hash mismatch for {path}: {actual_hash} != {expected_hash}"
            )
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        shape = tuple(int(value) for value in entry.get("shape", []))
        if tuple(array.shape) != shape:
            raise ValueError(
                f"{pooling} shape mismatch for {path}: {array.shape} != {shape}"
            )
        if array.dtype != np.float16 or entry.get("dtype") != "float16":
            raise ValueError(f"{pooling} must be float16: {path}")
        if (
            len(shape) != 3
            or shape[0] != row_end - row_start
            or shape[1] != len(CAPTURED_BLOCK_OUTPUTS)
            or shape[2] != HIDDEN_SIZE
        ):
            raise ValueError(f"Invalid {pooling} feature shape {shape}: {path}")
        if reference_shape is None:
            reference_shape = shape
        elif shape != reference_shape:
            raise ValueError(f"Pooling arrays have different shapes: {directory}")
        arrays[pooling] = array
    payload_hash = hash_jsonable(
        {
            pooling: entries[pooling]["sha256"]
            for pooling in POOLING_FILENAMES
        }
    )
    payload_key = (
        "shard_payload_sha256"
        if kind == SHARD_ARTIFACT_KIND
        else "merged_payload_sha256"
    )
    if metadata.get(payload_key) != payload_hash:
        raise ValueError(f"Feature payload summary hash mismatch: {metadata_path}")
    if kind == SHARD_ARTIFACT_KIND:
        shard_count = int(metadata.get("shard_count", -1))
        shard_index = int(metadata.get("shard_index", -1))
        full_end = int(metadata.get("full_shard_row_end", -1))
        debug_partial = metadata.get("debug_partial_shard")
        if (
            shard_count <= 0
            or not 0 <= shard_index < shard_count
            or not row_end <= full_end <= total
            or debug_partial != (row_end != full_end)
            or not isinstance(metadata.get("seed"), int)
        ):
            raise ValueError(f"Invalid shard identity/interval metadata: {metadata_path}")
    else:
        _validate_merged_source_shards(metadata, manifest, metadata_path)
    return VerifiedFeatureDirectory(directory, metadata, arrays)


SHARD_INVARIANT_KEYS = (
    "model_repo",
    "model_revision",
    "model_config_sha256",
    "preprocessor_config_sha256",
    "model_weight_sha256",
    "model_small_file_sha256",
    "subject_id",
    "records_sha256",
    "manifest_sha256",
    "split",
    "hidden_size",
    "logical_layer_ids",
    "captured_block_outputs",
    "logical_layer_routes",
)


def validate_complete_shard_set(
    manifest: Mapping[str, Any], shard_directories: Iterable[str | Path]
) -> list[VerifiedFeatureDirectory]:
    verified = [
        validate_feature_directory(
            manifest, path, expected_artifact_kind=SHARD_ARTIFACT_KIND
        )
        for path in shard_directories
    ]
    if not verified:
        raise ValueError("No feature shards were supplied")
    reference = verified[0].metadata
    shard_count = int(reference.get("shard_count", -1))
    if shard_count <= 0 or len(verified) != shard_count:
        raise ValueError(
            f"Feature shard set has {len(verified)} shards, expected {shard_count}"
        )
    expected_for_split = EXPECTED_SHARD_COUNTS[manifest["split"]]
    if shard_count != expected_for_split:
        raise ValueError(
            f"{manifest['split']} requires {expected_for_split} shards, got {shard_count}"
        )
    for item in verified[1:]:
        if any(
            item.metadata.get(key) != reference.get(key)
            for key in SHARD_INVARIANT_KEYS
        ):
            raise ValueError(f"Incompatible feature shard: {item.directory}")
    verified.sort(key=lambda item: int(item.metadata.get("shard_index", -1)))
    indices = [int(item.metadata.get("shard_index", -1)) for item in verified]
    if indices != list(range(shard_count)):
        raise ValueError(f"Feature shard indices are incomplete: {indices}")
    total = len(manifest["records"])
    trailing_shape = tuple(verified[0].arrays["raw_cls"].shape[1:])
    for index, item in enumerate(verified):
        start = int(item.metadata["row_start"])
        end = int(item.metadata["row_end"])
        expected_start = total * index // shard_count
        expected_end = total * (index + 1) // shard_count
        if (
            start != expected_start
            or end != expected_end
            or int(item.metadata["full_shard_row_end"]) != expected_end
            or item.metadata["debug_partial_shard"] is not False
        ):
            raise ValueError(
                f"Feature shard interval is not canonical at {item.directory}: "
                f"{start}:{end}, expected {expected_start}:{expected_end}"
            )
        if tuple(item.arrays["raw_cls"].shape[1:]) != trailing_shape:
            raise ValueError(f"Feature shards have different trailing shapes: {item.directory}")
    return verified
