#!/usr/bin/env python3
"""Validate and atomically merge row-contiguous InternViT V2.5 shards."""

from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from v2_5_feature_contract import (
    CAPTURED_BLOCK_OUTPUTS,
    LOGICAL_LAYER_IDS,
    MERGED_ARTIFACT_KIND,
    POOLING_FILENAMES,
    SCHEMA_VERSION,
    VerifiedFeatureDirectory,
    atomic_write_json,
    git_revision,
    hash_file,
    hash_jsonable,
    load_manifest,
    logical_layer_routes,
    pooling_file_metadata,
    resolve_without_symlinks,
    validate_complete_shard_set,
    validate_feature_directory,
)


REPRODUCTION_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = REPRODUCTION_ROOT.parents[1]


def _validated_shards_for_merge(
    manifest: dict[str, Any], shard_directories: Sequence[str | Path]
) -> list[VerifiedFeatureDirectory]:
    return validate_complete_shard_set(manifest, shard_directories)


def _source_shard_metadata(
    shards: Sequence[VerifiedFeatureDirectory],
) -> list[dict[str, Any]]:
    return [
        {
            "directory": str(shard.directory),
            "shard_index": int(shard.metadata["shard_index"]),
            "row_start": int(shard.metadata["row_start"]),
            "row_end": int(shard.metadata["row_end"]),
            "metadata_sha256": hash_file(shard.directory / "metadata.json"),
            "pooling_sha256": {
                pooling: shard.metadata["pooling_files"][pooling]["sha256"]
                for pooling in POOLING_FILENAMES
            },
        }
        for shard in shards
    ]


def _validate_existing_merged(
    manifest: dict[str, Any],
    output_directory: Path,
    expected_source_shards: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if output_directory.is_symlink():
        raise FileExistsError(f"Refusing symlink output directory: {output_directory}")
    if not output_directory.exists():
        return None
    try:
        verified = validate_feature_directory(
            manifest,
            output_directory,
            expected_artifact_kind=MERGED_ARTIFACT_KIND,
        )
        if verified.metadata.get("source_shards") != expected_source_shards:
            raise ValueError(
                "Existing merge was produced from a different shard set"
            )
    except Exception as error:
        raise FileExistsError(
            f"Refusing to replace an invalid merged directory: {output_directory}"
        ) from error
    print(f"verified-existing {output_directory}", flush=True)
    return verified.metadata


def merge_feature_shards(
    *,
    manifest_path: str | Path,
    shard_directories: Sequence[str | Path],
    output_directory: str | Path,
    chunk_rows: int = 128,
) -> dict[str, Any]:
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    manifest = load_manifest(manifest_path)
    manifest_path = Path(manifest["_manifest_path"])
    output_directory = resolve_without_symlinks(
        output_directory, "Merged output directory"
    )
    shards = _validated_shards_for_merge(manifest, shard_directories)
    source_shards = _source_shard_metadata(shards)
    existing = _validate_existing_merged(manifest, output_directory, source_shards)
    if existing is not None:
        return existing
    reference = shards[0].metadata
    total = len(manifest["records"])
    trailing_shape = tuple(shards[0].arrays["raw_cls"].shape[1:])

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = output_directory.with_name(
        f".{output_directory.name}.partial-{os.getpid()}"
    )
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(f"Refusing existing staging path: {staging}")
    started = time.time()
    staging.mkdir()
    try:
        merged_arrays = {
            pooling: np.lib.format.open_memmap(
                staging / filename,
                mode="w+",
                dtype=np.float16,
                shape=(total, *trailing_shape),
            )
            for pooling, filename in POOLING_FILENAMES.items()
        }
        for shard in shards:
            start = int(shard.metadata["row_start"])
            end = int(shard.metadata["row_end"])
            for local_start in range(0, end - start, chunk_rows):
                local_end = min(end - start, local_start + chunk_rows)
                target_slice = slice(start + local_start, start + local_end)
                source_slice = slice(local_start, local_end)
                for pooling in POOLING_FILENAMES:
                    merged_arrays[pooling][target_slice] = shard.arrays[pooling][
                        source_slice
                    ]
        for array in merged_arrays.values():
            array.flush()
        del merged_arrays

        merged_shape = (total, *trailing_shape)
        pooling_files = {
            pooling: pooling_file_metadata(staging / filename, merged_shape)
            for pooling, filename in POOLING_FILENAMES.items()
        }
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": MERGED_ARTIFACT_KIND,
            "complete": True,
            "model_repo": reference["model_repo"],
            "model_revision": reference["model_revision"],
            "model_path": reference["model_path"],
            "model_config_sha256": reference["model_config_sha256"],
            "preprocessor_config_sha256": reference["preprocessor_config_sha256"],
            "model_weight_sha256": reference["model_weight_sha256"],
            "model_small_file_sha256": reference["model_small_file_sha256"],
            "model_weight_verification": reference.get(
                "model_weight_verification", "not_recorded_by_source_fixture"
            ),
            "manifest": str(manifest_path),
            "manifest_sha256": manifest["_manifest_sha256"],
            "subject_id": 1,
            "records_sha256": manifest["records_sha256"],
            "split": manifest["split"],
            "row_start": 0,
            "row_end": total,
            "shard_count": len(shards),
            "logical_layer_ids": list(LOGICAL_LAYER_IDS),
            "captured_block_outputs": list(CAPTURED_BLOCK_OUTPUTS),
            "captured_block_semantics": (
                "actual_encoder_block_output_embedding_not_counted"
            ),
            "logical_layer_routes": logical_layer_routes(),
            "pooling_semantics": {
                "raw_cls": "hidden[:,0,:]",
                "patch_mean": "mean(hidden[:,1:,:],axis=1)_excluding_cls",
            },
            "pooling_files": pooling_files,
            "merged_payload_sha256": hash_jsonable(
                {
                    pooling: entry["sha256"]
                    for pooling, entry in pooling_files.items()
                }
            ),
            "source_shards": source_shards,
            "source_shards_sha256": hash_jsonable(source_shards),
            "dtype": "float16",
            "hidden_size": int(trailing_shape[-1]),
            "git_revision": git_revision(PROJECT_ROOT),
            "merge_elapsed_seconds": time.time() - started,
        }
        atomic_write_json(staging / "metadata.json", metadata)
        if output_directory.exists() or output_directory.is_symlink():
            raise FileExistsError(
                f"Output appeared while merging; refusing replacement: "
                f"{output_directory}"
            )
        os.rename(staging, output_directory)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    verified = validate_feature_directory(
        manifest,
        output_directory,
        expected_artifact_kind=MERGED_ARTIFACT_KIND,
    )
    print(f"complete {output_directory}", flush=True)
    return verified.metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and merge V2.5 feature shard directories"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--shard-directories", nargs="+", required=True)
    parser.add_argument("--chunk-rows", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge_feature_shards(
        manifest_path=args.manifest,
        shard_directories=args.shard_directories,
        output_directory=args.output_directory,
        chunk_rows=args.chunk_rows,
    )


if __name__ == "__main__":
    main()
