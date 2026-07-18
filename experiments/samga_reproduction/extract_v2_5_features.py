#!/usr/bin/env python3
"""Extract auditable V2.5 CLS and patch-mean features in manifest row order."""

from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from v2_5_feature_contract import (
    CAPTURED_BLOCK_OUTPUTS,
    LOGICAL_LAYER_IDS,
    MODEL_REPO,
    MODEL_REVISION,
    MODEL_SMALL_FILE_SHA256,
    MODEL_WEIGHT_BYTES,
    MODEL_WEIGHT_SHA256,
    POOLING_FILENAMES,
    SCHEMA_VERSION,
    SHARD_ARTIFACT_KIND,
    atomic_write_json,
    git_revision,
    hash_file,
    hash_jsonable,
    load_manifest,
    logical_layer_routes,
    pooling_file_metadata,
    read_json,
    resolve_without_symlinks,
    validate_feature_directory,
)


REPRODUCTION_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = REPRODUCTION_ROOT.parents[1]
DEFAULT_MODEL_ENV = "INTERNVIT_V2_5_MODEL_PATH"


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
        return {
            "row_indices": torch.tensor(rows, dtype=torch.long),
            "pixel_values": pixels,
        }


def internvit_components(model: nn.Module) -> tuple[nn.Module, Sequence[nn.Module]]:
    vision = getattr(model, "vision_model", model)
    embeddings = getattr(vision, "embeddings", None)
    encoder = getattr(vision, "encoder", None)
    layers = getattr(encoder, "layers", None)
    if embeddings is None or layers is None:
        raise TypeError("InternViT does not expose embeddings/encoder.layers")
    return embeddings, layers


def collect_block_poolings(
    model: nn.Module,
    pixel_values: torch.Tensor,
    captured_block_outputs: Sequence[int] = CAPTURED_BLOCK_OUTPUTS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run embeddings and blocks once, pooling selected actual block outputs."""
    requested = tuple(int(value) for value in captured_block_outputs)
    if (
        not requested
        or min(requested) < 1
        or len(set(requested)) != len(requested)
    ):
        raise ValueError(f"Invalid actual block outputs: {requested}")
    embeddings, layers = internvit_components(model)
    if max(requested) > len(layers):
        raise ValueError(
            f"Requested block output {max(requested)} from a {len(layers)}-block model"
        )
    hidden = embeddings(pixel_values)
    if hidden.ndim != 3 or hidden.shape[1] < 2:
        raise ValueError(
            "InternViT embeddings must have [batch, CLS+patches, hidden] shape"
        )
    cls_by_block: dict[int, torch.Tensor] = {}
    patch_mean_by_block: dict[int, torch.Tensor] = {}
    requested_set = set(requested)
    for actual_block_output, layer in enumerate(layers, start=1):
        hidden = layer(hidden)
        if actual_block_output in requested_set:
            cls_by_block[actual_block_output] = hidden[:, 0, :].clone()
            patch_mean_by_block[actual_block_output] = hidden[:, 1:, :].mean(dim=1)
        if actual_block_output >= max(requested):
            break
    return (
        torch.stack([cls_by_block[index] for index in requested], dim=1),
        torch.stack([patch_mean_by_block[index] for index in requested], dim=1),
    )


def shard_interval(
    total_rows: int,
    shard_index: int,
    shard_count: int,
    max_rows: int | None = None,
) -> tuple[int, int, int]:
    if total_rows <= 0 or shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError(
            f"Invalid shard selection total={total_rows}, "
            f"index={shard_index}, count={shard_count}"
        )
    full_start = total_rows * shard_index // shard_count
    full_end = total_rows * (shard_index + 1) // shard_count
    end = full_end
    if max_rows is not None:
        if max_rows <= 0:
            raise ValueError("max-rows must be positive")
        end = min(end, full_start + max_rows)
    if full_start >= end:
        raise ValueError(f"Shard {shard_index}/{shard_count} selects no rows")
    return full_start, end, full_end


def _regular_model_file(model_path: Path, filename: str) -> Path:
    path = model_path / filename
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"Missing regular V2.5 model file: {path}")
    return path


def verify_model_directory(model_path: Path) -> dict[str, Any]:
    """Verify the pinned checkpoint without any network access."""
    model_path = resolve_without_symlinks(model_path, "Model path")
    config_path = _regular_model_file(model_path, "config.json")
    processor_path = _regular_model_file(model_path, "preprocessor_config.json")
    actual_small_hashes = {}
    for filename, expected_hash in MODEL_SMALL_FILE_SHA256.items():
        path = _regular_model_file(model_path, filename)
        actual_hash = hash_file(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"V2.5 small-file hash mismatch for {path}: "
                f"{actual_hash} != {expected_hash}"
            )
        actual_small_hashes[filename] = actual_hash
    config = read_json(config_path)
    processor = read_json(processor_path)
    if (
        config.get("hidden_size") != 3200
        or config.get("num_hidden_layers") != 45
        or config.get("image_size") != 448
        or config.get("patch_size") != 14
    ):
        raise ValueError(f"Unexpected V2.5 model config: {config_path}")
    if (
        processor.get("crop_size") != 448
        or processor.get("size") != 448
        or processor.get("image_mean") != [0.485, 0.456, 0.406]
        or processor.get("image_std") != [0.229, 0.224, 0.225]
    ):
        raise ValueError(f"Unexpected V2.5 preprocessing config: {processor_path}")

    for filename, expected_bytes in MODEL_WEIGHT_BYTES.items():
        path = _regular_model_file(model_path, filename)
        if path.stat().st_size != expected_bytes:
            raise ValueError(
                f"V2.5 weight size mismatch for {path}: "
                f"{path.stat().st_size} != {expected_bytes}"
            )
    provenance_path = model_path / "model_provenance.json"
    if provenance_path.is_file() and not provenance_path.is_symlink():
        provenance = read_json(provenance_path)
        if (
            provenance.get("complete") is not True
            or provenance.get("model_repo") != MODEL_REPO
            or provenance.get("model_revision") != MODEL_REVISION
            or provenance.get("model_weight_sha256") != MODEL_WEIGHT_SHA256
            or provenance.get("small_file_sha256") != MODEL_SMALL_FILE_SHA256
        ):
            raise ValueError(f"Invalid V2.5 model provenance: {provenance_path}")
    actual_weights = {
        filename: hash_file(model_path / filename)
        for filename in MODEL_WEIGHT_SHA256
    }
    if actual_weights != MODEL_WEIGHT_SHA256:
        raise ValueError(
            f"V2.5 weight hash mismatch in {model_path}: {actual_weights}"
        )
    return {
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "model_path": str(model_path),
        "model_config_sha256": hash_file(config_path),
        "preprocessor_config_sha256": hash_file(processor_path),
        "model_weight_sha256": actual_weights,
        "model_small_file_sha256": actual_small_hashes,
        "model_weight_verification": "direct_sha256_current_files",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract V2.5 actual block outputs into one atomic feature shard"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-path", default=os.environ.get(DEFAULT_MODEL_ENV))
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _validate_existing_output(
    manifest: dict[str, Any],
    output_directory: Path,
    *,
    shard_index: int,
    shard_count: int,
    row_start: int,
    row_end: int,
    full_shard_row_end: int,
    seed: int,
    model_identity: dict[str, Any],
) -> bool:
    if not output_directory.exists():
        return False
    try:
        verified = validate_feature_directory(
            manifest,
            output_directory,
            expected_artifact_kind=SHARD_ARTIFACT_KIND,
        )
        if (
            int(verified.metadata.get("shard_index", -1)) != shard_index
            or int(verified.metadata.get("shard_count", -1)) != shard_count
            or int(verified.metadata.get("row_start", -1)) != row_start
            or int(verified.metadata.get("row_end", -1)) != row_end
            or int(verified.metadata.get("full_shard_row_end", -1))
            != full_shard_row_end
            or verified.metadata.get("debug_partial_shard")
            != (row_end != full_shard_row_end)
            or int(verified.metadata.get("seed", -1)) != seed
            or any(
                verified.metadata.get(key) != model_identity.get(key)
                for key in (
                    "model_path",
                    "model_config_sha256",
                    "preprocessor_config_sha256",
                    "model_weight_sha256",
                    "model_small_file_sha256",
                )
            )
        ):
            raise ValueError("Existing shard identity does not match the request")
    except Exception as error:
        raise FileExistsError(
            f"Refusing to replace an existing invalid/incompatible shard: "
            f"{output_directory}"
        ) from error
    print(f"verified-existing {output_directory}", flush=True)
    return True


def main() -> None:
    args = parse_args()
    if not args.model_path:
        raise ValueError(
            f"--model-path or ${DEFAULT_MODEL_ENV} is required; no model is downloaded"
        )
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative")
    manifest = load_manifest(args.manifest)
    manifest_path = Path(manifest["_manifest_path"])
    expected_shards = 8 if manifest["split"] == "train" else 1
    if args.shard_count != expected_shards:
        raise ValueError(
            f"{manifest['split']} extraction requires {expected_shards} shards, "
            f"got {args.shard_count}"
        )
    row_start, row_end, full_row_end = shard_interval(
        len(manifest["records"]),
        args.shard_index,
        args.shard_count,
        args.max_rows,
    )
    output_directory = resolve_without_symlinks(
        args.output_directory, "Output directory"
    )
    model_path = resolve_without_symlinks(args.model_path, "Model path")
    model_identity = verify_model_directory(model_path)
    if _validate_existing_output(
        manifest,
        output_directory,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        row_start=row_start,
        row_end=row_end,
        full_shard_row_end=full_row_end,
        seed=args.seed,
        model_identity=model_identity,
    ):
        return
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = output_directory.with_name(
        f".{output_directory.name}.partial-{os.getpid()}"
    )
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(f"Refusing existing staging path: {staging}")

    _seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
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
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    hidden_size = int(model.config.hidden_size)
    if (
        hidden_size != 3200
        or int(model.config.num_hidden_layers) < max(CAPTURED_BLOCK_OUTPUTS)
    ):
        raise ValueError(
            f"Unexpected InternViT hidden/layer config: "
            f"{hidden_size}/{model.config.num_hidden_layers}"
        )

    records = list(manifest["records"][row_start:row_end])
    rows = [int(record["row_index"]) for record in records]
    if rows != list(range(row_start, row_end)):
        raise ValueError("Feature extraction requires contiguous manifest row order")
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
    staging.mkdir()
    try:
        shape = (row_end - row_start, len(CAPTURED_BLOCK_OUTPUTS), hidden_size)
        raw_cls_cache = np.lib.format.open_memmap(
            staging / POOLING_FILENAMES["raw_cls"],
            mode="w+",
            dtype=np.float16,
            shape=shape,
        )
        patch_mean_cache = np.lib.format.open_memmap(
            staging / POOLING_FILENAMES["patch_mean"],
            mode="w+",
            dtype=np.float16,
            shape=shape,
        )
        seen = 0
        with torch.inference_mode():
            for batch in loader:
                batch_rows = batch["row_indices"].numpy()
                expected_rows = list(
                    range(row_start + seen, row_start + seen + len(batch_rows))
                )
                if batch_rows.tolist() != expected_rows:
                    raise RuntimeError("DataLoader changed manifest row order")
                pixels = batch["pixel_values"].to(
                    device=device,
                    dtype=torch.bfloat16,
                    non_blocking=True,
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    raw_cls, patch_mean = collect_block_poolings(model, pixels)
                local_rows = batch_rows - row_start
                raw_cls_cache[local_rows] = (
                    raw_cls.float().cpu().numpy().astype(np.float16)
                )
                patch_mean_cache[local_rows] = (
                    patch_mean.float().cpu().numpy().astype(np.float16)
                )
                seen += len(batch_rows)
                print(
                    f"rows-complete {seen}/{len(records)} "
                    f"split={manifest['split']} shard={args.shard_index}",
                    flush=True,
                )
        if seen != len(records):
            raise RuntimeError(f"Only extracted {seen}/{len(records)} rows")
        raw_cls_cache.flush()
        patch_mean_cache.flush()
        del raw_cls_cache, patch_mean_cache

        pooling_files = {
            pooling: pooling_file_metadata(staging / filename, shape)
            for pooling, filename in POOLING_FILENAMES.items()
        }
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": SHARD_ARTIFACT_KIND,
            "complete": True,
            **model_identity,
            "manifest": str(manifest_path),
            "manifest_sha256": manifest["_manifest_sha256"],
            "subject_id": 1,
            "records_sha256": manifest["records_sha256"],
            "split": manifest["split"],
            "row_start": row_start,
            "row_end": row_end,
            "full_shard_row_end": full_row_end,
            "debug_partial_shard": row_end != full_row_end,
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
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
            "shard_payload_sha256": hash_jsonable(
                {
                    pooling: entry["sha256"]
                    for pooling, entry in pooling_files.items()
                }
            ),
            "dtype": "float16",
            "hidden_size": hidden_size,
            "preprocessing": "CLIPImageProcessor.from_pretrained(local_model_path)",
            "online_forward": "embeddings_once_then_encoder_blocks_1_through_37_once",
            "seed": args.seed,
            "git_revision": git_revision(PROJECT_ROOT),
            "elapsed_seconds": time.time() - started,
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
        }
        atomic_write_json(staging / "metadata.json", metadata)
        if output_directory.exists() or output_directory.is_symlink():
            raise FileExistsError(
                f"Output appeared while extracting; refusing replacement: "
                f"{output_directory}"
            )
        os.rename(staging, output_directory)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    validate_feature_directory(
        manifest,
        output_directory,
        expected_artifact_kind=SHARD_ARTIFACT_KIND,
    )
    print(f"complete {output_directory}", flush=True)


if __name__ == "__main__":
    main()
