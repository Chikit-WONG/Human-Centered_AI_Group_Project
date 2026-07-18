#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from samga_lora.data import feature_cache_metadata_path, load_manifest  # noqa: E402
from samga_lora.model import load_clip_provider  # noqa: E402
from samga_lora.utils import atomic_write_json, git_revision, hash_file, seed_everything  # noqa: E402


class ManifestImageDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[int, Image.Image]:
        record = self.records[index]
        with Image.open(record["image_path"]) as source:
            image = source.convert("RGB")
        return int(record["row_index"]), image


class ImageCollator:
    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def __call__(self, examples: list[tuple[int, Image.Image]]) -> dict[str, Any]:
        rows, images = zip(*examples)
        pixels = self.processor(images=list(images), return_tensors="pt").pixel_values
        return {"row_indices": torch.tensor(rows, dtype=torch.long), "pixel_values": pixels}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache frozen multi-layer CLIP image features")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--clip-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layer-ids", type=int, nargs="+", default=[4, 6, 8, 10, 12])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
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
    records = manifest["records"]
    expected_rows = list(range(len(records)))
    actual_rows = [int(record["row_index"]) for record in records]
    if actual_rows != expected_rows:
        raise ValueError("Feature extraction requires contiguous manifest row order")
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and args.device.startswith("cuda"):
        raise RuntimeError("CUDA was requested but is unavailable")
    provider, processor = load_clip_provider(
        model_path=args.clip_path,
        layer_ids=args.layer_ids,
        vision_mode="frozen",
        device=device,
        dtype=torch.float32,
    )
    provider.eval()
    loader = DataLoader(
        ManifestImageDataset(records),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=ImageCollator(processor),
    )
    started = time.time()
    temporary = output.with_name(output.name + f".partial-{os.getpid()}.npy")
    cache = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float16,
        shape=(len(records), len(args.layer_ids), 768),
    )
    seen = 0
    with torch.no_grad():
        for batch in loader:
            rows = batch["row_indices"].numpy()
            if rows.tolist() != list(range(seen, seen + len(rows))):
                raise RuntimeError("DataLoader changed manifest row order")
            pixels = batch["pixel_values"].to(device=device, dtype=torch.float32, non_blocking=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                features = provider(pixels)
            cache[rows] = features.float().cpu().numpy().astype(np.float16)
            seen += len(rows)
    cache.flush()
    del cache
    if seen != len(records):
        raise RuntimeError(f"Only extracted {seen}/{len(records)} feature rows")
    os.replace(temporary, output)
    metadata = {
        "schema_version": 1,
        "manifest": str(Path(args.manifest).resolve()),
        "records_sha256": manifest["records_sha256"],
        "split": manifest["split"],
        "shape": [len(records), len(args.layer_ids), 768],
        "dtype": "float16",
        "layer_ids": [int(value) for value in args.layer_ids],
        "clip_path": str(Path(args.clip_path).resolve()),
        "clip_config_sha256": hash_file(Path(args.clip_path) / "config.json"),
        "cache_sha256": hash_file(output),
        "git_revision": git_revision(PROJECT_ROOT),
        "elapsed_seconds": time.time() - started,
        "complete": True,
    }
    atomic_write_json(metadata_output, metadata)


if __name__ == "__main__":
    main()
