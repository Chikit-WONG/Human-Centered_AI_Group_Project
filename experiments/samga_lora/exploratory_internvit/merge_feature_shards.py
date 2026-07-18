#!/usr/bin/env python3
"""Merge validated row-contiguous InternViT feature shards."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from samga_lora.data import feature_cache_metadata_path, load_manifest  # noqa: E402
from samga_lora.utils import atomic_write_json, git_revision, hash_file, read_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge frozen InternViT feature shards")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output).resolve()
    metadata_output = feature_cache_metadata_path(output)
    for path in (output, metadata_output):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.manifest)
    entries: list[tuple[int, int, Path, np.ndarray, dict[str, Any]]] = []
    reference: dict[str, Any] | None = None
    invariant_keys = (
        "model_repo",
        "model_revision",
        "model_config_sha256",
        "model_weight_sha256",
        "layer_semantics",
        "records_sha256",
        "split",
        "dtype",
        "layer_ids",
    )
    for raw_path in args.shards:
        path = Path(raw_path).resolve()
        metadata = read_json(feature_cache_metadata_path(path))
        array = np.load(path, mmap_mode="r")
        start = int(metadata.get("row_start", -1))
        end = int(metadata.get("row_end", -1))
        if (
            metadata.get("schema_version") != 1
            or metadata.get("complete") is not True
            or metadata.get("partial_rows") is not True
            or metadata.get("records_sha256") != manifest["records_sha256"]
            or metadata.get("cache_sha256") != hash_file(path)
            or tuple(metadata.get("shape", [])) != tuple(array.shape)
            or end - start != array.shape[0]
            or array.dtype != np.float16
        ):
            raise ValueError(f"Invalid feature shard or provenance: {path}")
        if reference is None:
            reference = metadata
        elif any(metadata.get(key) != reference.get(key) for key in invariant_keys):
            raise ValueError(f"Incompatible feature shard: {path}")
        entries.append((start, end, path, array, metadata))
    entries.sort(key=lambda item: item[0])
    expected_start = 0
    for start, end, path, _, _ in entries:
        if start != expected_start or end <= start:
            raise ValueError(f"Non-contiguous/overlapping shard interval at {path}: {start}:{end}")
        expected_start = end
    total = len(manifest["records"])
    if expected_start != total or reference is None:
        raise ValueError(f"Shards cover 0:{expected_start}, expected 0:{total}")
    trailing_shape = tuple(entries[0][3].shape[1:])
    if any(tuple(entry[3].shape[1:]) != trailing_shape for entry in entries):
        raise ValueError("Feature shards have different trailing shapes")
    temporary = output.with_name(output.name + f".partial-{os.getpid()}.npy")
    merged = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float16,
        shape=(total, *trailing_shape),
    )
    started = time.time()
    source_hashes: dict[str, str] = {}
    for start, end, path, array, metadata in entries:
        merged[start:end] = array
        source_hashes[str(path)] = metadata["cache_sha256"]
    merged.flush()
    del merged
    os.replace(temporary, output)
    metadata = {
        key: reference[key]
        for key in (
            "schema_version",
            "exploratory",
            "inferred_model",
            "model_repo",
            "model_revision",
            "model_path",
            "model_config_sha256",
            "model_weight_sha256",
            "layer_semantics",
            "records_sha256",
            "split",
            "dtype",
            "layer_ids",
        )
    }
    metadata.update(
        {
            "manifest": str(Path(args.manifest).resolve()),
            "row_start": 0,
            "row_end": total,
            "partial_rows": False,
            "shape": [total, *trailing_shape],
            "source_shard_sha256": source_hashes,
            "cache_sha256": hash_file(output),
            "git_revision": git_revision(PROJECT_ROOT),
            "merge_elapsed_seconds": time.time() - started,
            "complete": True,
        }
    )
    atomic_write_json(metadata_output, metadata)


if __name__ == "__main__":
    main()
