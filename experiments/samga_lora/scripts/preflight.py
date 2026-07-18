#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import sys
from pathlib import Path
from typing import Any

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from samga_lora.data import load_manifest  # noqa: E402
from samga_lora.utils import POSTERIOR_CHANNELS, atomic_write_json, hash_file  # noqa: E402


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the local SAMGA experiment inputs")
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--clip-path", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_dir = Path(args.manifest_dir).resolve()
    clip_path = Path(args.clip_path).resolve()
    problems: list[str] = []
    manifests: dict[tuple[int, str], dict[str, Any]] = {}
    split_hashes: dict[str, set[str]] = {"train": set(), "test": set()}
    for subject in range(1, 11):
        for split, expected_rows in (("train", 16540), ("test", 200)):
            path = manifest_dir / f"sub-{subject:02d}_{split}.json"
            manifest = load_manifest(path)
            manifests[(subject, split)] = manifest
            split_hashes[split].add(manifest["records_sha256"])
            if int(manifest["subject_id"]) != subject or manifest["split"] != split:
                problems.append(f"subject/split metadata mismatch: {path}")
            if len(manifest["records"]) != expected_rows:
                problems.append(f"row count mismatch: {path}")
            if not Path(manifest["source_pt"]).is_file():
                problems.append(f"missing source EEG: {manifest['source_pt']}")
            if any(channel not in manifest["ch_names"] for channel in POSTERIOR_CHANNELS):
                problems.append(f"posterior channel missing: {path}")
            if any(not Path(record["image_path"]).is_file() for record in manifest["records"]):
                problems.append(f"image path missing below: {path}")
            if split == "train":
                validation = set(manifest["validation_concepts"])
                validation_rows = [record for record in manifest["records"] if record["validation_query"]]
                if len(validation) != 200 or len(validation_rows) != 200:
                    problems.append(f"validation split count mismatch: {path}")
                if any(record["concept_id"] not in validation for record in validation_rows):
                    problems.append(f"validation marker/concept mismatch: {path}")
            else:
                ids = [record["image_id"] for record in manifest["records"]]
                if len(set(ids)) != 200 or not all(record["validation_query"] for record in manifest["records"]):
                    problems.append(f"test gallery is not 200 unique images: {path}")
    if any(len(values) != 1 for values in split_hashes.values()):
        problems.append("image row order is not shared across all ten subjects")
    train_concepts = {record["concept_id"] for record in manifests[(1, "train")]["records"]}
    test_concepts = {record["concept_id"] for record in manifests[(1, "test")]["records"]}
    overlap = sorted(train_concepts & test_concepts)
    if overlap:
        problems.append(f"train/test concept overlap: {overlap[:5]}")
    required_clip_files = ("config.json", "model.safetensors", "preprocessor_config.json")
    for filename in required_clip_files:
        if not (clip_path / filename).is_file():
            problems.append(f"missing CLIP file: {clip_path / filename}")
    report = {
        "schema_version": 1,
        "passed": not problems,
        "problems": problems,
        "manifest_dir": str(manifest_dir),
        "shared_records_sha256": {split: next(iter(values)) for split, values in split_hashes.items()},
        "train_concepts": len(train_concepts),
        "test_concepts": len(test_concepts),
        "validation_concepts": len(manifests[(1, "train")]["validation_concepts"]),
        "clip_path": str(clip_path),
        "clip_config_sha256": hash_file(clip_path / "config.json"),
        "clip_weights_sha256": hash_file(clip_path / "model.safetensors"),
        "environment": {
            name: package_version(name)
            for name in ("torch", "torchvision", "transformers", "peft", "accelerate", "numpy", "scipy", "scikit-learn")
        },
        "python_version": sys.version.split()[0],
    }
    atomic_write_json(args.output, report)
    if problems:
        raise RuntimeError("Preflight failed:\n- " + "\n- ".join(problems))


if __name__ == "__main__":
    main()
