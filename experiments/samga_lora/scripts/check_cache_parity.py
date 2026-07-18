#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from samga_lora.data import load_manifest, validate_feature_cache  # noqa: E402
from samga_lora.model import load_clip_provider  # noqa: E402
from samga_lora.utils import atomic_write_json, seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare cached CLIP features with a fresh forward pass")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--clip-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layer-ids", type=int, nargs="+", default=[4, 6, 8, 10, 12])
    parser.add_argument("--rows", type=int, nargs="+", default=[0, 8270, 16539])
    parser.add_argument("--maximum-mean-absolute-error", type=float, default=0.01)
    parser.add_argument("--maximum-relative-l2-error", type=float, default=0.02)
    parser.add_argument("--minimum-cosine", type=float, default=0.9999)
    parser.add_argument("--lora-initial-atol", type=float, default=0.002)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--check-lora-init", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(0)
    manifest = load_manifest(args.manifest)
    cache, metadata = validate_feature_cache(
        args.cache, manifest, expected_layer_ids=args.layer_ids
    )
    if any(row < 0 or row >= len(manifest["records"]) for row in args.rows):
        raise IndexError("Parity row outside manifest")
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
    images = []
    for row in args.rows:
        with Image.open(manifest["records"][row]["image_path"]) as source:
            images.append(source.convert("RGB"))
    pixels = processor(images=images, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        online = provider(pixels).float().cpu().numpy()
    cached = np.asarray(cache[args.rows], dtype=np.float32)
    absolute = np.abs(online - cached)
    flat_online = online.reshape(-1, online.shape[-1])
    flat_cached = cached.reshape(-1, cached.shape[-1])
    dot = np.sum(flat_online * flat_cached, axis=-1)
    online_norm = np.linalg.norm(flat_online, axis=-1)
    cached_norm = np.linalg.norm(flat_cached, axis=-1)
    cosine = dot / np.maximum(online_norm * cached_norm, 1e-12)
    relative_l2 = np.linalg.norm(flat_online - flat_cached, axis=-1) / np.maximum(
        online_norm, 1e-12
    )
    lora_initial_error = None
    if args.check_lora_init:
        del provider
        if device.type == "cuda":
            torch.cuda.empty_cache()
        lora_provider, _ = load_clip_provider(
            model_path=args.clip_path,
            layer_ids=args.layer_ids,
            vision_mode="lora",
            lora_rank=32,
            device=device,
            dtype=torch.float32,
        )
        lora_provider.eval()
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            lora_initial = lora_provider(pixels).float().cpu().numpy()
        lora_initial_error = float(np.abs(online - lora_initial).max())
    report = {
        "schema_version": 1,
        "rows": args.rows,
        "max_absolute_error": float(absolute.max()),
        "mean_absolute_error": float(absolute.mean()),
        "minimum_vector_cosine": float(cosine.min()),
        "mean_vector_cosine": float(cosine.mean()),
        "maximum_relative_l2_error": float(relative_l2.max()),
        "thresholds": {
            "maximum_mean_absolute_error": args.maximum_mean_absolute_error,
            "maximum_relative_l2_error": args.maximum_relative_l2_error,
            "minimum_cosine": args.minimum_cosine,
            "lora_initial_atol": args.lora_initial_atol,
        },
        "lora_initial_max_absolute_error": lora_initial_error,
        "passed": bool(
            absolute.mean() <= args.maximum_mean_absolute_error
            and relative_l2.max() <= args.maximum_relative_l2_error
            and cosine.min() >= args.minimum_cosine
            and (
                lora_initial_error is None
                or lora_initial_error <= args.lora_initial_atol
            )
        ),
        "cache_sha256": metadata["cache_sha256"],
    }
    atomic_write_json(args.output, report)
    if not report["passed"]:
        raise RuntimeError(f"Cache parity failed: {report}")


if __name__ == "__main__":
    main()
