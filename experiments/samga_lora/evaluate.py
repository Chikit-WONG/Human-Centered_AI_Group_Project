#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
from peft import set_peft_model_state_dict
from torch.utils.data import DataLoader

EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from samga_lora.data import SAMGACollator, ThingsEEGSubjectDataset, load_manifest  # noqa: E402
from samga_lora.model import SAMGATaskModel, load_clip_provider  # noqa: E402
from samga_lora.utils import atomic_write_json, git_revision, hash_file, retrieval_metrics, seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one SAMGA checkpoint once")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--test-feature-cache", default=None)
    parser.add_argument("--metrics-output", required=True)
    parser.add_argument("--predictions-output", required=True)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics_output)
    predictions_path = Path(args.predictions_output)
    for output in (metrics_path, predictions_path):
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    test_manifest = load_manifest(args.test_manifest)
    if (
        int(test_manifest["subject_id"]) != int(config["subject_id"])
        or test_manifest["split"] != "test"
    ):
        raise ValueError("Test manifest subject/split does not match the checkpoint")
    seed_everything(int(config["seed"]))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and args.device.startswith("cuda"):
        raise RuntimeError("CUDA was requested but is unavailable")
    vision_provider = None
    processor = None
    if config["vision_mode"] == "lora":
        vision_provider, processor = load_clip_provider(
            model_path=config["clip_path"],
            layer_ids=config["layer_ids"],
            vision_mode="lora",
            lora_rank=int(config["lora_rank"]),
            device=device,
            dtype=torch.float32,
        )
        load_result = set_peft_model_state_dict(
            vision_provider.backbone, checkpoint["vision_adapter_state_dict"]
        )
        if getattr(load_result, "unexpected_keys", []):
            raise RuntimeError(f"Unexpected LoRA keys: {load_result.unexpected_keys}")
        vision_provider.eval()
    elif not args.test_feature_cache:
        raise ValueError("Frozen checkpoint evaluation requires --test-feature-cache")
    dataset = ThingsEEGSubjectDataset(
        manifest_path=args.test_manifest,
        subset="test",
        seed=int(config["seed"]),
        feature_cache=args.test_feature_cache,
        expected_layer_ids=config["layer_ids"],
        smooth_probability=0.0,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=SAMGACollator(processor),
    )
    task = SAMGATaskModel(
        layer_ids=config["layer_ids"], prior_center=int(config["prior_center"])
    ).to(device)
    task.load_state_dict(checkpoint["task_state_dict"], strict=True)
    task.eval()
    all_eeg: list[torch.Tensor] = []
    all_image: list[torch.Tensor] = []
    image_ids: list[str] = []
    with torch.no_grad():
        for batch in loader:
            eeg = batch["eeg"].to(device=device, dtype=torch.float32)
            subject_ids = batch["subject_ids"].to(device)
            if "layer_features" in batch:
                layers = batch["layer_features"].to(device=device, dtype=torch.float32)
            else:
                pixels = batch["pixel_values"].to(device=device, dtype=torch.float32)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    layers = vision_provider(pixels).float()
            eeg_features, image_features, _ = task(eeg, layers, subject_ids, force_global=True)
            all_eeg.append(eeg_features.cpu())
            all_image.append(image_features.cpu())
            image_ids.extend(batch["image_ids"])
    metrics, predictions = retrieval_metrics(
        torch.cat(all_eeg), torch.cat(all_image), image_ids, image_ids
    )
    metrics.update(
        {
            "schema_version": 1,
            "subject_id": int(config["subject_id"]),
            "seed": int(config["seed"]),
            "vision_mode": config["vision_mode"],
            "vision_lr_ratio": float(config["vision_lr_ratio"]),
            "task_initial_state_sha256": config["task_initial_state_sha256"],
            "contrastive_eeg_l2norm": bool(config["eeg_l2norm"]),
            "contrastive_image_l2norm": bool(config["image_l2norm"]),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "checkpoint_sha256": hash_file(args.checkpoint),
            "manifest_sha256": hash_file(args.test_manifest),
            "git_revision": git_revision(PROJECT_ROOT),
        }
    )
    fieldnames = list(predictions[0])
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=predictions_path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in predictions:
            serializable = dict(row)
            serializable["top5_image_ids"] = "|".join(row["top5_image_ids"])
            serializable["top5_scores"] = "|".join(f"{value:.9g}" for value in row["top5_scores"])
            writer.writerow(serializable)
        temporary_predictions = Path(handle.name)
    os.replace(temporary_predictions, predictions_path)
    metrics["predictions_sha256"] = hash_file(predictions_path)
    metrics["test_records_sha256"] = test_manifest["records_sha256"]
    atomic_write_json(metrics_path, metrics)


if __name__ == "__main__":
    main()
