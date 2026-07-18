#!/usr/bin/env python3
"""Extract frozen multi-level InternViT CLS features in manifest row order."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from samga_lora.data import feature_cache_metadata_path, load_manifest  # noqa: E402
from samga_lora.utils import atomic_write_json, git_revision, hash_file, seed_everything  # noqa: E402


MODEL_REPO = "OpenGVLab/InternViT-6B-448px-V1-5"
MODEL_REVISION = "03e138c81d3fd538c77439fd43a42c067d827427"
EXPECTED_SHARDS = {
    "model-00001-of-00003.safetensors": "331fc0e79147081bb4260491b4db121aaf4252e0f29ed3509ae5df11bd8ae41e",
    "model-00002-of-00003.safetensors": "4785be9bec8771f0b25a2f33c52fe9e53623068eb0e7d72aa01e410c43a91cbc",
    "model-00003-of-00003.safetensors": "95fe64ed513580d1fbd4257823adfd5b7b1a283c70cdd2771453443dd1f0b6b6",
}


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


def internvit_components(model: nn.Module) -> tuple[nn.Module, Sequence[nn.Module]]:
    vision = getattr(model, "vision_model", model)
    embeddings = getattr(vision, "embeddings", None)
    encoder = getattr(vision, "encoder", None)
    layers = getattr(encoder, "layers", None)
    if embeddings is None or layers is None:
        raise TypeError("The inferred InternViT model does not expose embeddings/encoder.layers")
    return embeddings, layers


def collect_hidden_state_cls(
    model: nn.Module, pixel_values: torch.Tensor, layer_ids: Sequence[int]
) -> torch.Tensor:
    """Match Transformers hidden-state indices (embedding is state zero)."""
    requested = tuple(int(value) for value in layer_ids)
    if not requested or min(requested) < 1 or len(set(requested)) != len(requested):
        raise ValueError(f"Invalid layer IDs: {requested}")
    embeddings, layers = internvit_components(model)
    if max(requested) > len(layers):
        raise ValueError(f"Requested layer {max(requested)} from a {len(layers)}-block model")
    hidden = embeddings(pixel_values)
    captured: dict[int, torch.Tensor] = {}
    for state_index, layer in enumerate(layers, start=1):
        hidden = layer(hidden)
        if state_index in requested:
            captured[state_index] = hidden[:, 0, :]
        if state_index >= max(requested):
            break
    return torch.stack([captured[index] for index in requested], dim=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache frozen InternViT multi-layer features")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layer-ids", type=int, nargs="+", default=[20, 24, 28, 32, 36])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--end-row", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def verify_model_files(model_path: Path) -> dict[str, str]:
    actual: dict[str, str] = {}
    for filename, expected in EXPECTED_SHARDS.items():
        path = model_path / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing InternViT weight shard: {path}")
        digest = hash_file(path)
        if digest != expected:
            raise ValueError(f"Weight hash mismatch for {path}: {digest} != {expected}")
        actual[filename] = digest
    return actual


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative")
    output = Path(args.output).resolve()
    metadata_output = feature_cache_metadata_path(output)
    for path in (output, metadata_output):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model_path).resolve()
    shard_hashes = verify_model_files(model_path)
    manifest = load_manifest(args.manifest)
    all_records = manifest["records"]
    start_row = int(args.start_row)
    end_row = len(all_records) if args.end_row is None else int(args.end_row)
    if args.max_rows is not None:
        end_row = min(end_row, start_row + int(args.max_rows))
    if not 0 <= start_row < end_row <= len(all_records):
        raise ValueError(
            f"Invalid row interval [{start_row},{end_row}) for {len(all_records)} records"
        )
    row_count = end_row - start_row
    if row_count <= 0:
        raise ValueError("No image rows selected")
    records = list(all_records[start_row:end_row])
    actual_rows = [int(record["row_index"]) for record in records]
    if actual_rows != list(range(start_row, end_row)):
        raise ValueError("Feature extraction requires contiguous manifest row order")
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and args.device.startswith("cuda"):
        raise RuntimeError("CUDA was requested but is unavailable")

    from transformers import AutoModel, CLIPImageProcessor

    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    processor = CLIPImageProcessor.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True
    )
    hidden_size = int(model.config.hidden_size)
    if hidden_size != 3200 or int(model.config.num_hidden_layers) < max(args.layer_ids):
        raise ValueError(
            f"Unexpected InternViT config hidden/layers={hidden_size}/{model.config.num_hidden_layers}"
        )
    loader = DataLoader(
        ManifestImageDataset(records),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=ImageCollator(processor),
    )
    temporary = output.with_name(output.name + f".partial-{os.getpid()}.npy")
    cache = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float16,
        shape=(row_count, len(args.layer_ids), hidden_size),
    )
    started = time.time()
    seen = 0
    with torch.inference_mode():
        for batch in loader:
            rows = batch["row_indices"].numpy()
            if rows.tolist() != list(
                range(start_row + seen, start_row + seen + len(rows))
            ):
                raise RuntimeError("DataLoader changed manifest row order")
            pixels = batch["pixel_values"].to(
                device=device, dtype=torch.bfloat16, non_blocking=True
            )
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                features = collect_hidden_state_cls(model, pixels, args.layer_ids)
            cache[rows - start_row] = features.float().cpu().numpy().astype(np.float16)
            seen += len(rows)
    cache.flush()
    del cache
    if seen != row_count:
        raise RuntimeError(f"Only extracted {seen}/{row_count} feature rows")
    os.replace(temporary, output)
    metadata = {
        "schema_version": 1,
        "exploratory": True,
        "inferred_model": True,
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "model_path": str(model_path),
        "model_config_sha256": hash_file(model_path / "config.json"),
        "model_weight_sha256": shard_hashes,
        "layer_semantics": "transformers_hidden_state_index_embedding_is_zero",
        "manifest": str(Path(args.manifest).resolve()),
        "records_sha256": manifest["records_sha256"],
        "split": manifest["split"],
        "row_start": start_row,
        "row_end": end_row,
        "partial_rows": start_row != 0 or end_row != len(all_records),
        "shape": [row_count, len(args.layer_ids), hidden_size],
        "dtype": "float16",
        "layer_ids": [int(value) for value in args.layer_ids],
        "cache_sha256": hash_file(output),
        "git_revision": git_revision(PROJECT_ROOT),
        "elapsed_seconds": time.time() - started,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "complete": True,
    }
    atomic_write_json(metadata_output, metadata)


if __name__ == "__main__":
    main()
