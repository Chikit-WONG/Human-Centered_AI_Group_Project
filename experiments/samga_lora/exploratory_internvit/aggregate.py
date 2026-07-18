#!/usr/bin/env python3
"""Strictly aggregate the ten-subject, one-seed inferred-InternViT run."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from samga_lora.data import feature_cache_metadata_path  # noqa: E402
from samga_lora.utils import atomic_write_json, hash_file, read_json  # noqa: E402


SUBJECTS = tuple(range(1, 11))
EXPECTED_SEED = 2025
EXPECTED_EPOCH = 60
EXPECTED_LAYERS = [20, 24, 28, 32, 36]
EXPECTED_REVISION = "03e138c81d3fd538c77439fd43a42c067d827427"


def validate_predictions(path: Path, metrics: dict[str, Any]) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 200 or [int(row["query_index"]) for row in rows] != list(range(200)):
        raise ValueError(f"Prediction cardinality/order mismatch in {path}")
    if len({row["query_image_id"] for row in rows}) != 200:
        raise ValueError(f"Duplicate query IDs in {path}")
    ranks = [int(row["target_rank"]) for row in rows]
    correct1 = sum(rank == 1 for rank in ranks)
    correct5 = sum(rank <= 5 for rank in ranks)
    for row, rank in zip(rows, ranks):
        top5 = row["top5_image_ids"].split("|")
        if row["predicted_image_id"] != top5[0]:
            raise ValueError(f"Prediction/top-5 mismatch in {path}")
        if (row["query_image_id"] in top5) != (rank <= 5):
            raise ValueError(f"Target-rank mismatch in {path}")
    if (
        correct1 != int(metrics["top1_correct"])
        or correct5 != int(metrics["top5_correct"])
        or abs(correct1 / 200 - float(metrics["top1"])) > 1e-12
        or abs(correct5 / 200 - float(metrics["top5"])) > 1e-12
        or metrics.get("predictions_sha256") != hash_file(path)
    ):
        raise ValueError(f"Prediction/metric disagreement in {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate exploratory InternViT SAMGA results")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--test-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_cache = Path(args.train_cache).resolve()
    test_cache = Path(args.test_cache).resolve()
    cache_meta = []
    for cache_path, expected_split, expected_shape in (
        (train_cache, "train", [16540, 5, 3200]),
        (test_cache, "test", [200, 5, 3200]),
    ):
        metadata = read_json(feature_cache_metadata_path(cache_path))
        if (
            metadata.get("exploratory") is not True
            or metadata.get("inferred_model") is not True
            or metadata.get("model_revision") != EXPECTED_REVISION
            or metadata.get("layer_ids") != EXPECTED_LAYERS
            or metadata.get("partial_rows") is not False
            or metadata.get("split") != expected_split
            or metadata.get("shape") != expected_shape
            or metadata.get("dtype") != "float16"
            or metadata.get("cache_sha256") != hash_file(cache_path)
        ):
            raise ValueError(f"InternViT cache provenance mismatch for {cache_path}")
        cache_meta.append(metadata)
    rows: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        directory = run_root / f"sub-{subject:02d}"
        metrics = read_json(directory / "test_metrics.json")
        checkpoint = directory / f"checkpoint_epoch{EXPECTED_EPOCH:03d}.pt"
        verification = read_json(directory / "checkpoint_verification.json")
        config = read_json(directory / "run_config.json")
        completion = read_json(directory / "completion.json")
        with (directory / "training_history.jsonl").open("r", encoding="utf-8") as handle:
            history = [json.loads(line) for line in handle if line.strip()]
        manifest = read_json(Path(config["manifest"]))
        checkpoint_sha256 = hash_file(checkpoint)
        if (
            int(metrics["subject_id"]) != subject
            or int(metrics["seed"]) != EXPECTED_SEED
            or int(metrics["checkpoint_epoch"]) != EXPECTED_EPOCH
            or metrics["vision_mode"] != "frozen"
            or metrics["protocol"] != "standard_independent_exact_image"
            or int(metrics["num_queries"]) != 200
            or int(metrics["num_gallery"]) != 200
            or metrics.get("contrastive_eeg_l2norm") is not False
            or metrics.get("contrastive_image_l2norm") is not True
            or int(config.get("subject_id")) != subject
            or int(config.get("seed")) != EXPECTED_SEED
            or int(config.get("image_dim")) != 3200
            or config.get("vision_mode") != "frozen"
            or int(config.get("vision_trainable_parameters")) != 0
            or int(config.get("num_epochs")) != EXPECTED_EPOCH
            or int(config.get("stage1_epochs")) != 20
            or config.get("candidate_epochs") != str(EXPECTED_EPOCH)
            or Path(config.get("feature_cache", "")).resolve() != train_cache
            or config.get("layer_ids") != EXPECTED_LAYERS
            or int(config.get("prior_center")) != 28
            or config.get("clip_config_sha256") != cache_meta[0]["model_config_sha256"]
            or manifest.get("subject_id") != subject
            or manifest.get("split") != "train"
            or manifest.get("records_sha256") != cache_meta[0]["records_sha256"]
            or metrics.get("test_records_sha256") != cache_meta[1]["records_sha256"]
            or completion.get("completed") is not True
            or int(completion.get("final_epoch")) != EXPECTED_EPOCH
            or int(completion.get("global_step")) != 1920
            or completion.get("first_step_gradient_norms", {}).get("vision") != 0.0
            or verification.get("passed") is not True
            or int(verification.get("epoch")) != EXPECTED_EPOCH
            or verification.get("vision_mode") != "frozen"
            or verification.get("checkpoint_sha256") != checkpoint_sha256
            or metrics.get("checkpoint_sha256") != checkpoint_sha256
            or len(history) != EXPECTED_EPOCH
            or [int(row["epoch"]) for row in history] != list(range(1, EXPECTED_EPOCH + 1))
            or int(history[-1]["global_step"]) != 1920
            or not np.isfinite([float(row["train_loss"]) for row in history]).all()
            or sorted(path.name for path in directory.glob("checkpoint_epoch*.pt"))
            != [f"checkpoint_epoch{EXPECTED_EPOCH:03d}.pt"]
            or (directory / "validation_metrics.jsonl").exists()
        ):
            raise ValueError(f"Run provenance mismatch in {directory}")
        validate_predictions(directory / "test_predictions.csv", metrics)
        rows.append(
            {
                "subject_id": subject,
                "seed": EXPECTED_SEED,
                "top1": float(metrics["top1"]),
                "top5": float(metrics["top5"]),
                "top1_correct": int(metrics["top1_correct"]),
                "top5_correct": int(metrics["top5_correct"]),
            }
        )
    top1 = np.asarray([row["top1"] for row in rows], dtype=np.float64)
    top5 = np.asarray([row["top5"] for row in rows], dtype=np.float64)
    summary = {
        "schema_version": 1,
        "status": "exploratory_inferred_model_not_exact_paper_reproduction",
        "protocol": "standard_independent_exact_image",
        "model_repo": cache_meta[0]["model_repo"],
        "model_revision": EXPECTED_REVISION,
        "layer_ids": EXPECTED_LAYERS,
        "layer_semantics": cache_meta[0]["layer_semantics"],
        "feature_cache_sha256": {
            "train": cache_meta[0]["cache_sha256"],
            "test": cache_meta[1]["cache_sha256"],
        },
        "model_weight_sha256": cache_meta[0]["model_weight_sha256"],
        "subjects": list(SUBJECTS),
        "seed": EXPECTED_SEED,
        "epoch": EXPECTED_EPOCH,
        "top1": {
            "mean": float(top1.mean()),
            "subject_sample_sd": float(top1.std(ddof=1)),
            "correct": int(sum(row["top1_correct"] for row in rows)),
            "total": 2000,
        },
        "top5": {
            "mean": float(top5.mean()),
            "subject_sample_sd": float(top5.std(ddof=1)),
            "correct": int(sum(row["top5_correct"] for row in rows)),
            "total": 2000,
        },
        "limitations": [
            "SAMGA does not disclose the exact InternViT checkpoint identifier.",
            "The feature-extraction code and precise hidden-state indexing are not released.",
            "This run uses one fixed seed (2025), matching the released intra.sh launcher, not five seeds.",
            "The fixed epoch-60 checkpoint is evaluated once; no test-set early stopping is used.",
        ],
    }
    atomic_write_json(output_dir / "summary.json", summary)
    with (output_dir / "per_subject_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    top1_pct = 100 * summary["top1"]["mean"]
    top5_pct = 100 * summary["top5"]["mean"]
    en = f"""# Exploratory SAMGA InternViT Result

- Status: inferred-model exploratory reproduction; **not** an exact reproduction of the paper.
- Scope: ten THINGS-EEG2 subjects, released-launcher seed 2025, fixed epoch 60.
- Protocol: standard independent 200-way exact-image retrieval; no Hungarian decoding and no test-set checkpoint selection.
- Model: `{summary['model_repo']}` at revision `{EXPECTED_REVISION}`; hidden-state indices 20/24/28/32/36.
- Top-1: **{top1_pct:.2f}%** ({summary['top1']['correct']}/2000).
- Top-5: **{top5_pct:.2f}%** ({summary['top5']['correct']}/2000).

The SAMGA paper does not disclose the exact InternViT checkpoint, extraction code, or hidden-state indexing, so this result must not be substituted for the paper-reported 91.3%/98.8% five-seed result.
"""
    zh = f"""# SAMGA InternViT 探索性结果

- 状态：推断模型的探索性复现；**不是**论文的精确复现。
- 范围：THINGS-EEG2 十名受试者、发布脚本使用的随机种子 2025、固定 epoch 60。
- 协议：标准独立 200-way 精确图片检索；不使用匈牙利解码，也不按测试集选择检查点。
- 模型：`{summary['model_repo']}`，revision `{EXPECTED_REVISION}`；hidden-state 索引 20/24/28/32/36。
- Top-1：**{top1_pct:.2f}%**（{summary['top1']['correct']}/2000）。
- Top-5：**{top5_pct:.2f}%**（{summary['top5']['correct']}/2000）。

SAMGA 论文没有披露精确的 InternViT checkpoint、特征提取代码或 hidden-state 索引语义，因此该结果不能替代论文报告的五随机种子 91.3%/98.8%。
"""
    (output_dir / "RESULTS_EN.md").write_text(en, encoding="utf-8")
    (output_dir / "RESULTS_ZH.md").write_text(zh, encoding="utf-8")


if __name__ == "__main__":
    main()
