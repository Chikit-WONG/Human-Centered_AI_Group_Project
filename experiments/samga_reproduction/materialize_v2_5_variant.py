#!/usr/bin/env python3
"""Materialize one logical-layer V2.5 pooling/indexing variant for SAMGA."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from v2_5_feature_contract import (
    HIDDEN_SIZE,
    LOGICAL_LAYER_IDS,
    MERGED_ARTIFACT_KIND,
    SCHEMA_VERSION,
    atomic_write_json,
    hash_file,
    load_manifest,
    logical_layer_routes,
    read_json,
    resolve_without_symlinks,
    validate_feature_directory,
)


VARIANT_ARTIFACT_KIND = "internvit_v2_5_selected_variant"
FEATURE_FILENAME = "features.npy"


def selection_metadata(indexing_variant: str) -> dict[str, list[int]]:
    routes = logical_layer_routes()
    if indexing_variant not in routes:
        raise ValueError(
            f"Unsupported indexing variant {indexing_variant!r}; "
            f"expected one of {sorted(routes)}"
        )
    selected = routes[indexing_variant]
    return {
        "logical_layer_ids": [
            int(route["logical_layer_id"]) for route in selected
        ],
        "selected_captured_block_outputs": [
            int(route["captured_block_output"]) for route in selected
        ],
        "source_axis_indices": [
            int(route["source_axis_index"]) for route in selected
        ],
    }


def select_variant_values(
    source: np.ndarray, *, indexing_variant: str
) -> tuple[np.ndarray, dict[str, list[int]]]:
    if source.ndim != 3 or source.shape[1] != 10:
        raise ValueError(
            f"Expected [rows,10,hidden] captured feature array, got {source.shape}"
        )
    selection = selection_metadata(indexing_variant)
    values = np.asarray(source[:, selection["source_axis_indices"], :])
    return values, selection


def _validate_existing_variant(
    output_directory: Path,
    *,
    manifest: dict[str, Any],
    pooling: str,
    indexing_variant: str,
    source: Any,
    selection: dict[str, list[int]],
) -> dict[str, Any] | None:
    if not output_directory.exists():
        return None
    try:
        metadata_path = output_directory / "metadata.json"
        if metadata_path.is_symlink() or not metadata_path.is_file():
            raise ValueError("Selected variant metadata is not a regular file")
        metadata = read_json(metadata_path)
        feature_path = output_directory / FEATURE_FILENAME
        if feature_path.is_symlink() or not feature_path.is_file():
            raise ValueError("Selected variant feature is not a regular file")
        array = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        expected_shape = (
            len(manifest["records"]), len(LOGICAL_LAYER_IDS), HIDDEN_SIZE
        )
        if (
            metadata.get("schema_version") != SCHEMA_VERSION
            or metadata.get("artifact_kind") != VARIANT_ARTIFACT_KIND
            or metadata.get("complete") is not True
            or metadata.get("subject_id") != 1
            or metadata.get("records_sha256") != manifest["records_sha256"]
            or metadata.get("manifest_sha256") != manifest["_manifest_sha256"]
            or metadata.get("split") != manifest["split"]
            or metadata.get("pooling") != pooling
            or metadata.get("indexing_variant") != indexing_variant
            or metadata.get("source_merged_directory") != str(source.directory)
            or metadata.get("source_pooling_sha256")
            != source.metadata["pooling_files"][pooling]["sha256"]
            or any(
                metadata.get(key) != source.metadata.get(key)
                for key in (
                    "model_repo",
                    "model_revision",
                    "model_config_sha256",
                    "preprocessor_config_sha256",
                    "model_weight_sha256",
                    "model_small_file_sha256",
                )
            )
            or metadata.get("logical_layer_ids") != selection["logical_layer_ids"]
            or metadata.get("selected_captured_block_outputs")
            != selection["selected_captured_block_outputs"]
            or metadata.get("source_axis_indices") != selection["source_axis_indices"]
            or tuple(metadata.get("shape", [])) != expected_shape
            or tuple(array.shape) != expected_shape
            or metadata.get("feature_filename") != FEATURE_FILENAME
            or metadata.get("dtype") != "float16"
            or array.dtype != np.float16
            or metadata.get("feature_sha256") != hash_file(feature_path)
        ):
            raise ValueError("Existing selected variant is invalid")
    except Exception as error:
        raise FileExistsError(
            f"Refusing to replace an invalid selected variant: {output_directory}"
        ) from error
    print(f"verified-existing {output_directory}", flush=True)
    return metadata


def materialize_variant(
    *,
    manifest_path: str | Path,
    merged_directory: str | Path,
    output_directory: str | Path,
    pooling: str,
    indexing_variant: str,
    chunk_rows: int = 256,
) -> dict[str, Any]:
    if pooling not in {"raw_cls", "patch_mean"}:
        raise ValueError("pooling must be raw_cls or patch_mean")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    selection = selection_metadata(indexing_variant)
    if selection["logical_layer_ids"] != list(LOGICAL_LAYER_IDS):
        raise RuntimeError("Router logical layer IDs changed unexpectedly")
    manifest = load_manifest(manifest_path)
    manifest_path = Path(manifest["_manifest_path"])
    source = validate_feature_directory(
        manifest,
        merged_directory,
        expected_artifact_kind=MERGED_ARTIFACT_KIND,
    )
    output_directory = resolve_without_symlinks(
        output_directory, "Variant output directory"
    )
    existing = _validate_existing_variant(
        output_directory,
        manifest=manifest,
        pooling=pooling,
        indexing_variant=indexing_variant,
        source=source,
        selection=selection,
    )
    if existing is not None:
        return existing

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = output_directory.with_name(
        f".{output_directory.name}.partial-{os.getpid()}"
    )
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(f"Refusing existing staging path: {staging}")
    staging.mkdir()
    try:
        source_array = source.arrays[pooling]
        shape = (
            source_array.shape[0],
            len(selection["source_axis_indices"]),
            source_array.shape[2],
        )
        target = np.lib.format.open_memmap(
            staging / FEATURE_FILENAME,
            mode="w+",
            dtype=np.float16,
            shape=shape,
        )
        for start in range(0, shape[0], chunk_rows):
            end = min(shape[0], start + chunk_rows)
            target[start:end] = source_array[
                start:end, selection["source_axis_indices"], :
            ]
        target.flush()
        del target
        feature_path = staging / FEATURE_FILENAME
        feature_sha256 = hash_file(feature_path)
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": VARIANT_ARTIFACT_KIND,
            "complete": True,
            "manifest": str(manifest_path),
            "manifest_sha256": manifest["_manifest_sha256"],
            "subject_id": 1,
            "records_sha256": manifest["records_sha256"],
            "split": manifest["split"],
            "source_merged_directory": str(source.directory),
            "source_pooling_sha256": source.metadata["pooling_files"][pooling][
                "sha256"
            ],
            "model_repo": source.metadata["model_repo"],
            "model_revision": source.metadata["model_revision"],
            "model_config_sha256": source.metadata["model_config_sha256"],
            "preprocessor_config_sha256": source.metadata[
                "preprocessor_config_sha256"
            ],
            "model_weight_sha256": source.metadata["model_weight_sha256"],
            "model_small_file_sha256": source.metadata["model_small_file_sha256"],
            "pooling": pooling,
            "indexing_variant": indexing_variant,
            **selection,
            "router_layer_semantics": (
                "router_uses_logical_layer_ids_20_24_28_32_36"
            ),
            "feature_filename": FEATURE_FILENAME,
            "shape": list(shape),
            "dtype": "float16",
            "feature_sha256": feature_sha256,
        }
        atomic_write_json(staging / "metadata.json", metadata)
        if output_directory.exists() or output_directory.is_symlink():
            raise FileExistsError(
                f"Output appeared while materializing; refusing replacement: "
                f"{output_directory}"
            )
        os.rename(staging, output_directory)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"complete {output_directory}", flush=True)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select an auditable [rows,5,3200] V2.5 SAMGA variant"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--merged-directory", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--pooling", choices=("raw_cls", "patch_mean"), required=True)
    parser.add_argument(
        "--indexing-variant",
        choices=("idx0", "idx_plus_1"),
        required=True,
    )
    parser.add_argument("--chunk-rows", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    materialize_variant(
        manifest_path=args.manifest,
        merged_directory=args.merged_directory,
        output_directory=args.output_directory,
        pooling=args.pooling,
        indexing_variant=args.indexing_variant,
        chunk_rows=args.chunk_rows,
    )


if __name__ == "__main__":
    main()
