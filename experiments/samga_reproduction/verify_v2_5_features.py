#!/usr/bin/env python3
"""Verify InternViT V2.5 shard or merged feature directories and provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from v2_5_feature_contract import (
    MERGED_ARTIFACT_KIND,
    SHARD_ARTIFACT_KIND,
    load_manifest,
    validate_complete_shard_set,
    validate_feature_directory,
)


def verify_feature_directory(
    *,
    manifest_path: str | Path,
    feature_directory: str | Path,
    expected_artifact_kind: str | None = None,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    return validate_feature_directory(
        manifest,
        feature_directory,
        expected_artifact_kind=expected_artifact_kind,
    ).metadata


def verify_feature_set(
    *,
    manifest_path: str | Path,
    feature_directories: list[str | Path],
    expected_artifact_kind: str,
) -> tuple[dict[str, Any], list[Any]]:
    manifest = load_manifest(manifest_path)
    if expected_artifact_kind == MERGED_ARTIFACT_KIND:
        if len(feature_directories) != 1:
            raise ValueError("Exactly one merged feature directory is required")
        verified = [
            validate_feature_directory(
                manifest,
                feature_directories[0],
                expected_artifact_kind=MERGED_ARTIFACT_KIND,
            )
        ]
    elif expected_artifact_kind == SHARD_ARTIFACT_KIND:
        verified = validate_complete_shard_set(manifest, feature_directories)
    else:
        raise ValueError(f"Unsupported artifact kind: {expected_artifact_kind}")
    return manifest, verified


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify hashes, shapes, coverage, model identity, and layer routes"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--feature-directories", nargs="+", required=True)
    parser.add_argument(
        "--expected-artifact-kind",
        choices=(SHARD_ARTIFACT_KIND, MERGED_ARTIFACT_KIND),
        required=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest, verified = verify_feature_set(
        manifest_path=args.manifest,
        feature_directories=args.feature_directories,
        expected_artifact_kind=args.expected_artifact_kind,
    )
    summary = {
        "complete": True,
        "manifest": manifest["_manifest_path"],
        "manifest_sha256": manifest["_manifest_sha256"],
        "subject_id": 1,
        "artifact_kind": args.expected_artifact_kind,
        "directories": [str(item.directory) for item in verified],
        "row_intervals": [
            [int(item.metadata["row_start"]), int(item.metadata["row_end"])]
            for item in verified
        ],
        "records_sha256": manifest["records_sha256"],
        "model_revision": verified[0].metadata["model_revision"],
        "model_weight_sha256": verified[0].metadata["model_weight_sha256"],
        "logical_layer_ids": verified[0].metadata["logical_layer_ids"],
        "captured_block_outputs": verified[0].metadata[
            "captured_block_outputs"
        ],
        "logical_layer_routes": verified[0].metadata["logical_layer_routes"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
