#!/usr/bin/env python3
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

from samga_lora.utils import atomic_write_json, hash_file, read_json  # noqa: E402


SUBJECTS = tuple(range(1, 11))
SEEDS = tuple(range(42, 47))


def run_dir(root: Path, mode: str, subject: int, seed: int) -> Path:
    return root / mode / f"sub-{subject:02d}" / f"seed-{seed}"


def two_way_bootstrap(
    matrix: np.ndarray, *, samples: int, seed: int
) -> tuple[float, float, np.ndarray]:
    rng = np.random.default_rng(seed)
    values = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        subjects = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
        seeds = rng.integers(0, matrix.shape[1], size=matrix.shape[1])
        values[index] = matrix[np.ix_(subjects, seeds)].mean()
    lower, upper = np.quantile(values, [0.025, 0.975])
    return float(lower), float(upper), values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate paired 10-subject x 5-seed SAMGA results")
    parser.add_argument("--formal-root", required=True)
    parser.add_argument("--locked-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260716)
    return parser.parse_args()


def mean_sd(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "sample_sd": float(values.std(ddof=1)),
    }


def validate_predictions(path: Path, metrics: dict[str, Any]) -> None:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 200 or [int(row["query_index"]) for row in rows] != list(range(200)):
        raise ValueError(f"Prediction cardinality/order mismatch in {path}")
    query_ids = [row["query_image_id"] for row in rows]
    if len(set(query_ids)) != 200:
        raise ValueError(f"Duplicate query image IDs in {path}")
    correct1 = 0
    correct5 = 0
    for row in rows:
        top5 = row["top5_image_ids"].split("|")
        rank = int(row["target_rank"])
        if not 1 <= rank <= 200 or row["predicted_image_id"] != top5[0]:
            raise ValueError(f"Invalid prediction row in {path}: {row}")
        is_top1 = row["predicted_image_id"] == row["query_image_id"]
        is_top5 = row["query_image_id"] in top5
        if is_top1 != (rank == 1) or is_top5 != (rank <= 5):
            raise ValueError(f"Rank/ID disagreement in {path}: {row}")
        correct1 += int(is_top1)
        correct5 += int(is_top5)
    if (
        correct1 != int(metrics["top1_correct"])
        or correct5 != int(metrics["top5_correct"])
        or abs(correct1 / 200 - float(metrics["top1"])) > 1e-12
        or abs(correct5 / 200 - float(metrics["top5"])) > 1e-12
        or metrics.get("predictions_sha256") != hash_file(path)
    ):
        raise ValueError(f"Prediction/metric disagreement in {path}")


def main() -> None:
    args = parse_args()
    root = Path(args.formal_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    locked = read_json(args.locked_config)
    if not locked.get("gate_passed"):
        raise RuntimeError("Formal aggregation is forbidden because the pilot gate did not pass")
    selected = locked["selected"]
    epoch = int(selected["epoch"])
    ratio = float(selected["vision_lr_ratio"])
    matrices = {
        "frozen_top1": np.zeros((10, 5), dtype=np.float64),
        "frozen_top5": np.zeros((10, 5), dtype=np.float64),
        "lora_top1": np.zeros((10, 5), dtype=np.float64),
        "lora_top5": np.zeros((10, 5), dtype=np.float64),
    }
    paired_rows: list[dict[str, Any]] = []
    for subject_index, subject in enumerate(SUBJECTS):
        for seed_index, seed in enumerate(SEEDS):
            cell: dict[str, dict[str, Any]] = {}
            for mode in ("frozen", "lora"):
                path = run_dir(root, mode, subject, seed) / "test_metrics.json"
                metrics = read_json(path)
                prediction_path = run_dir(root, mode, subject, seed) / "test_predictions.csv"
                if (
                    int(metrics["subject_id"]) != subject
                    or int(metrics["seed"]) != seed
                    or int(metrics["checkpoint_epoch"]) != epoch
                    or metrics["protocol"] != "standard_independent_exact_image"
                    or int(metrics["num_queries"]) != 200
                    or int(metrics["num_gallery"]) != 200
                    or metrics.get("contrastive_eeg_l2norm") is not False
                    or metrics.get("contrastive_image_l2norm") is not True
                ):
                    raise ValueError(f"Metric provenance mismatch in {path}")
                validate_predictions(prediction_path, metrics)
                checkpoint_path = run_dir(root, mode, subject, seed) / f"checkpoint_epoch{epoch:03d}.pt"
                if metrics.get("checkpoint_sha256") != hash_file(checkpoint_path):
                    raise ValueError(f"Checkpoint hash mismatch in {path}")
                expected_mode = mode
                if metrics["vision_mode"] != expected_mode:
                    raise ValueError(f"Vision-mode mismatch in {path}")
                if mode == "lora" and abs(float(metrics["vision_lr_ratio"]) - ratio) > 1e-12:
                    raise ValueError(f"LoRA ratio mismatch in {path}")
                cell[mode] = metrics
                matrices[f"{mode}_top1"][subject_index, seed_index] = float(metrics["top1"])
                matrices[f"{mode}_top5"][subject_index, seed_index] = float(metrics["top5"])
            if cell["frozen"].get("task_initial_state_sha256") != cell["lora"].get(
                "task_initial_state_sha256"
            ):
                raise ValueError(
                    f"Paired task initialization mismatch for subject {subject}, seed {seed}"
                )
            paired_rows.append(
                {
                    "subject_id": subject,
                    "seed": seed,
                    "frozen_top1": cell["frozen"]["top1"],
                    "lora_top1": cell["lora"]["top1"],
                    "delta_top1": float(cell["lora"]["top1"]) - float(cell["frozen"]["top1"]),
                    "frozen_top5": cell["frozen"]["top5"],
                    "lora_top5": cell["lora"]["top5"],
                    "delta_top5": float(cell["lora"]["top5"]) - float(cell["frozen"]["top5"]),
                }
            )
    delta_top1 = matrices["lora_top1"] - matrices["frozen_top1"]
    delta_top5 = matrices["lora_top5"] - matrices["frozen_top5"]
    top1_ci_low, top1_ci_high, _ = two_way_bootstrap(
        delta_top1, samples=args.bootstrap_samples, seed=args.bootstrap_seed
    )
    top5_ci_low, top5_ci_high, _ = two_way_bootstrap(
        delta_top5, samples=args.bootstrap_samples, seed=args.bootstrap_seed + 1
    )
    subject_rows: list[dict[str, Any]] = []
    for index, subject in enumerate(SUBJECTS):
        subject_rows.append(
            {
                "subject_id": subject,
                "frozen_top1_mean": matrices["frozen_top1"][index].mean(),
                "frozen_top1_sd": matrices["frozen_top1"][index].std(ddof=1),
                "lora_top1_mean": matrices["lora_top1"][index].mean(),
                "lora_top1_sd": matrices["lora_top1"][index].std(ddof=1),
                "delta_top1_mean": delta_top1[index].mean(),
                "frozen_top5_mean": matrices["frozen_top5"][index].mean(),
                "lora_top5_mean": matrices["lora_top5"][index].mean(),
                "delta_top5_mean": delta_top5[index].mean(),
            }
        )
    seed_rows: list[dict[str, Any]] = []
    for index, seed in enumerate(SEEDS):
        seed_rows.append(
            {
                "seed": seed,
                "frozen_top1_subject_macro": matrices["frozen_top1"][:, index].mean(),
                "lora_top1_subject_macro": matrices["lora_top1"][:, index].mean(),
                "delta_top1_subject_macro": delta_top1[:, index].mean(),
                "frozen_top5_subject_macro": matrices["frozen_top5"][:, index].mean(),
                "lora_top5_subject_macro": matrices["lora_top5"][:, index].mean(),
                "delta_top5_subject_macro": delta_top5[:, index].mean(),
            }
        )
    success = bool(delta_top1.mean() >= 0.005 and top1_ci_low > 0 and delta_top5.mean() >= -0.002)
    summary = {
        "schema_version": 1,
        "protocol": "standard_independent_exact_image",
        "subjects": list(SUBJECTS),
        "seeds": list(SEEDS),
        "selected_epoch": epoch,
        "vision_lr_ratio": ratio,
        "bootstrap": {
            "method": "independent_subject_and_seed_cluster_resampling",
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
        },
        "frozen": {
            "top1": {
                "mean": float(matrices["frozen_top1"].mean()),
                "seed_macro_sample_sd": float(matrices["frozen_top1"].mean(0).std(ddof=1)),
                "cell_sample_sd": float(matrices["frozen_top1"].std(ddof=1)),
            },
            "top5": {
                "mean": float(matrices["frozen_top5"].mean()),
                "seed_macro_sample_sd": float(matrices["frozen_top5"].mean(0).std(ddof=1)),
                "cell_sample_sd": float(matrices["frozen_top5"].std(ddof=1)),
            },
        },
        "lora": {
            "top1": {
                "mean": float(matrices["lora_top1"].mean()),
                "seed_macro_sample_sd": float(matrices["lora_top1"].mean(0).std(ddof=1)),
                "cell_sample_sd": float(matrices["lora_top1"].std(ddof=1)),
            },
            "top5": {
                "mean": float(matrices["lora_top5"].mean()),
                "seed_macro_sample_sd": float(matrices["lora_top5"].mean(0).std(ddof=1)),
                "cell_sample_sd": float(matrices["lora_top5"].std(ddof=1)),
            },
        },
        "paired_delta": {
            "top1": {
                **mean_sd(delta_top1),
                "ci95": [top1_ci_low, top1_ci_high],
                "positive_cells": int((delta_top1 > 0).sum()),
            },
            "top5": {
                **mean_sd(delta_top5),
                "ci95": [top5_ci_low, top5_ci_high],
                "positive_cells": int((delta_top5 > 0).sum()),
            },
        },
        "success_criterion_passed": success,
    }
    atomic_write_json(output_dir / "summary.json", summary)
    for filename, rows in (
        ("paired_metrics.csv", paired_rows),
        ("per_subject_metrics.csv", subject_rows),
        ("per_seed_metrics.csv", seed_rows),
    ):
        with open(output_dir / filename, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    frozen_1 = 100 * summary["frozen"]["top1"]["mean"]
    frozen_5 = 100 * summary["frozen"]["top5"]["mean"]
    lora_1 = 100 * summary["lora"]["top1"]["mean"]
    lora_5 = 100 * summary["lora"]["top5"]["mean"]
    delta_1 = 100 * summary["paired_delta"]["top1"]["mean"]
    delta_5 = 100 * summary["paired_delta"]["top5"]["mean"]
    ci_1 = [100 * value for value in summary["paired_delta"]["top1"]["ci95"]]
    frozen_1_sd = 100 * summary["frozen"]["top1"]["seed_macro_sample_sd"]
    frozen_5_sd = 100 * summary["frozen"]["top5"]["seed_macro_sample_sd"]
    lora_1_sd = 100 * summary["lora"]["top1"]["seed_macro_sample_sd"]
    lora_5_sd = 100 * summary["lora"]["top5"]["seed_macro_sample_sd"]
    en = f"""# SAMGA + Visual LoRA Results\n\n- Protocol: standard independent 200-way exact-image retrieval (no Hungarian decoding).\n- Scope: 10 subjects x 5 seeds (42--46); global epoch {epoch}, selected on sealed concept-disjoint validation data.\n- Frozen CLIP SAMGA: Top-1 {frozen_1:.2f}% +/- {frozen_1_sd:.2f}, Top-5 {frozen_5:.2f}% +/- {frozen_5_sd:.2f} (sample SD across five seed-level subject macro scores).\n- SAMGA + LoRA (vision LR ratio {ratio:.2f}): Top-1 {lora_1:.2f}% +/- {lora_1_sd:.2f}, Top-5 {lora_5:.2f}% +/- {lora_5_sd:.2f}.\n- Paired change: Top-1 {delta_1:+.2f} points (two-way cluster-bootstrap 95% CI [{ci_1[0]:+.2f}, {ci_1[1]:+.2f}]); Top-5 {delta_5:+.2f} points.\n- Pre-registered success criterion: **{'passed' if success else 'not passed'}**.\n"""
    zh = f"""# SAMGA + 视觉 LoRA 实验结果\n\n- 协议：标准独立 200-way 精确图片检索（不使用匈牙利解码）。\n- 范围：10 名受试者 x 5 个种子（42--46）；统一 epoch {epoch}，仅由 concept-disjoint 验证集锁定。\n- 冻结 CLIP 的 SAMGA：Top-1 {frozen_1:.2f}% +/- {frozen_1_sd:.2f}，Top-5 {frozen_5:.2f}% +/- {frozen_5_sd:.2f}（五个“单 seed 十人宏平均”的样本标准差）。\n- SAMGA + LoRA（视觉学习率比例 {ratio:.2f}）：Top-1 {lora_1:.2f}% +/- {lora_1_sd:.2f}，Top-5 {lora_5:.2f}% +/- {lora_5_sd:.2f}。\n- 配对变化：Top-1 {delta_1:+.2f} 个百分点（subject/seed 双向 cluster bootstrap 95% CI [{ci_1[0]:+.2f}, {ci_1[1]:+.2f}]）；Top-5 {delta_5:+.2f} 个百分点。\n- 预注册成功标准：**{'通过' if success else '未通过'}**。\n"""
    (output_dir / "RESULTS_EN.md").write_text(en, encoding="utf-8")
    (output_dir / "RESULTS_ZH.md").write_text(zh, encoding="utf-8")


if __name__ == "__main__":
    main()
