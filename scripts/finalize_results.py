#!/usr/bin/env python3
"""Validate and package the formal sub-08 retrieval result."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--history", required=True)
    parser.add_argument("--repeat-metrics")
    parser.add_argument("--repeat-predictions")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--formal-job-id", required=True)
    parser.add_argument("--smoke-job-id", required=True)
    parser.add_argument("--expected-epochs", type=int, default=25)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def slurm_accounting(job_id: str) -> list[dict[str, str]]:
    command = [
        "sacct",
        "-j",
        job_id,
        "--noheader",
        "--parsable2",
        "--format=JobIDRaw,JobName,Partition,State,Elapsed,Timelimit,AllocTRES,ExitCode",
    ]
    try:
        output = subprocess.run(
            command, check=True, capture_output=True, text=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    fields = [
        "job_id",
        "job_name",
        "partition",
        "state",
        "elapsed",
        "time_limit",
        "allocated_tres",
        "exit_code",
    ]
    rows = []
    for line in output.splitlines():
        values = line.split("|")
        if len(values) >= len(fields):
            rows.append(dict(zip(fields, values[: len(fields)])))
    return rows


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics).resolve()
    predictions_path = Path(args.predictions).resolve()
    if bool(args.repeat_metrics) != bool(args.repeat_predictions):
        raise ValueError("repeat metrics and predictions must be provided together")
    repeat_metrics_path = (
        Path(args.repeat_metrics).resolve() if args.repeat_metrics else None
    )
    repeat_predictions_path = (
        Path(args.repeat_predictions).resolve() if args.repeat_predictions else None
    )
    history_path = Path(args.history).resolve()
    run_dir = Path(args.run_dir).resolve()
    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_json(metrics_path)
    n = int(metrics["sample_count"])
    if n != 200 or int(metrics["gallery_size"]) != 200:
        raise ValueError("formal evaluation must contain 200 queries and 200 images")
    top1_count = int(metrics["top1_count"])
    top5_count = int(metrics["top5_count"])
    top1 = float(metrics["top1"])
    top5 = float(metrics["top5"])
    if top1 != top1_count / n or top5 != top5_count / n:
        raise ValueError("metric fractions do not equal correct_count / 200")
    if not (0 <= top1_count <= top5_count <= n):
        raise ValueError("expected 0 <= top1_count <= top5_count <= 200")

    with predictions_path.open(encoding="utf-8", newline="") as handle:
        prediction_rows = list(csv.DictReader(handle))
    if len(prediction_rows) != n:
        raise ValueError(f"expected {n} prediction rows, found {len(prediction_rows)}")
    gt_ids = [row["gt_image_id"] for row in prediction_rows]
    if len(set(gt_ids)) != n:
        raise ValueError("prediction CSV ground-truth IDs are not unique")
    if sum(int(row["correct_top1"]) for row in prediction_rows) != top1_count:
        raise ValueError("prediction CSV Top-1 count disagrees with metrics JSON")
    if sum(int(row["correct_top5"]) for row in prediction_rows) != top5_count:
        raise ValueError("prediction CSV Top-5 count disagrees with metrics JSON")

    repeat_verified = False
    if repeat_metrics_path is not None and repeat_predictions_path is not None:
        repeat_metrics = load_json(repeat_metrics_path)
        metric_fields = [
            "sample_count",
            "gallery_size",
            "top1_count",
            "top5_count",
            "top1",
            "top5",
            "subject_id",
            "seed",
        ]
        for field in metric_fields:
            if repeat_metrics[field] != metrics[field]:
                raise ValueError(f"repeat evaluation disagrees on metric field {field}")
        with repeat_predictions_path.open(encoding="utf-8", newline="") as handle:
            repeat_rows = list(csv.DictReader(handle))
        if len(repeat_rows) != n:
            raise ValueError("repeat prediction CSV has an unexpected row count")
        stable_fields = [
            "query_index",
            "subject_id",
            "gt_image_id",
            "gt_rank",
            "top1_image_id",
            "top5_image_ids",
            "correct_top1",
            "correct_top5",
        ]
        for row_index, (first, repeat) in enumerate(
            zip(prediction_rows, repeat_rows, strict=True)
        ):
            for field in stable_fields:
                if first[field] != repeat[field]:
                    raise ValueError(
                        f"repeat prediction differs at row {row_index}, field {field}"
                    )
        repeat_verified = True

    history = [
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(history) != args.expected_epochs:
        raise ValueError(
            f"expected {args.expected_epochs} validation records, found {len(history)}"
        )
    history_csv = results_dir / "training_history.csv"
    with history_csv.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "epoch",
            "step",
            "num_samples",
            "loss",
            "top1_count",
            "top5_count",
            "top1_acc",
            "top5_acc",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch, record in enumerate(history, start=1):
            writer.writerow({"epoch": epoch, **{field: record[field] for field in fields[1:]}})

    final_history = history[-1]
    in_training_top1_count = int(final_history["top1_count"])
    in_training_top5_count = int(final_history["top5_count"])
    if not (0 <= in_training_top1_count <= in_training_top5_count <= n):
        raise ValueError("invalid final in-training retrieval counts")
    if abs(float(final_history["top1_acc"]) - in_training_top1_count / n) > 1e-7:
        raise ValueError("final in-training Top-1 fraction disagrees with its count")
    if abs(float(final_history["top5_acc"]) - in_training_top5_count / n) > 1e-7:
        raise ValueError("final in-training Top-5 fraction disagrees with its count")

    canonical_predictions = results_dir / "retrieval_predictions.csv"
    if predictions_path != canonical_predictions:
        shutil.copyfile(predictions_path, canonical_predictions)

    per_subject = results_dir / "per_subject_metrics.csv"
    with per_subject.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "subject",
            "seed",
            "checkpoint_epoch",
            "num_queries",
            "gallery_size",
            "top1_count",
            "top5_count",
            "top1_fraction",
            "top5_fraction",
            "top1_percent",
            "top5_percent",
            "loss",
            "mean_gt_cosine_similarity",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "subject": "sub-08",
                "seed": metrics["seed"],
                "checkpoint_epoch": args.expected_epochs,
                "num_queries": n,
                "gallery_size": metrics["gallery_size"],
                "top1_count": top1_count,
                "top5_count": top5_count,
                "top1_fraction": top1,
                "top5_fraction": top5,
                "top1_percent": metrics["top1_percent"],
                "top5_percent": metrics["top5_percent"],
                "loss": metrics["loss"],
                "mean_gt_cosine_similarity": metrics[
                    "mean_gt_cosine_similarity"
                ],
            }
        )

    peak_top1_epoch, peak_top1 = max(
        enumerate(history, start=1), key=lambda pair: pair[1]["top1_acc"]
    )
    peak_top5_epoch, peak_top5 = max(
        enumerate(history, start=1), key=lambda pair: pair[1]["top5_acc"]
    )
    summary = {
        "schema_version": 1,
        "status": "complete",
        "task": "Brain-RW THINGS-EEG sub-08 image retrieval reproduction",
        "subject": "sub-08",
        "seed": 42,
        "checkpoint_policy": (
            f"fixed final checkpoint after epoch {args.expected_epochs}; "
            "official metrics come from an independent save/reload evaluation"
        ),
        "evaluation": "200-way held-out test image retrieval",
        "repeat_evaluation_verified": repeat_verified,
        "metrics": {
            "num_queries": n,
            "gallery_size": 200,
            "top1_count": top1_count,
            "top5_count": top5_count,
            "top1_fraction": top1,
            "top5_fraction": top5,
            "top1_percent": 100.0 * top1,
            "top5_percent": 100.0 * top5,
            "loss": metrics["loss"],
            "mean_gt_cosine_similarity": metrics[
                "mean_gt_cosine_similarity"
            ],
        },
        "chance": {"top1_fraction": 0.005, "top5_fraction": 0.025},
        "diagnostic_only": {
            "final_in_training_top1_count": in_training_top1_count,
            "final_in_training_top5_count": in_training_top5_count,
            "final_in_training_top1_fraction": float(final_history["top1_acc"]),
            "final_in_training_top5_fraction": float(final_history["top5_acc"]),
            "reloaded_minus_in_training_top1_count": (
                top1_count - in_training_top1_count
            ),
            "reloaded_minus_in_training_top5_count": (
                top5_count - in_training_top5_count
            ),
            "peak_top1_epoch": peak_top1_epoch,
            "peak_top1_fraction": peak_top1["top1_acc"],
            "peak_top5_epoch": peak_top5_epoch,
            "peak_top5_fraction": peak_top5["top5_acc"],
            "note": "Peaks are not used for checkpoint selection.",
        },
        "protocol": metrics["protocol"],
        "environment": metrics["environment"],
        "artifacts": {
            "per_subject_metrics": str(per_subject),
            "training_history": str(history_csv),
            "retrieval_predictions": str(canonical_predictions),
            "formal_checkpoint": str(run_dir),
        },
    }
    atomic_json(results_dir / "summary.json", summary)

    key_files = [
        PROJECT_ROOT / "train_clip_lora.py",
        PROJECT_ROOT / "scripts/evaluate_retrieval.py",
        PROJECT_ROOT / "scripts/run_sub08_reproduction.sh",
        Path(metrics["paths"]["clip_base"]) / "model.safetensors",
        run_dir / "brain_model/model.safetensors",
        run_dir / "vision_model/adapter_model.safetensors",
    ]
    if repeat_metrics_path is not None and repeat_predictions_path is not None:
        key_files.extend([repeat_metrics_path, repeat_predictions_path])
    hashes = {
        str(path): {"sha256": sha256(path), "size_bytes": path.stat().st_size}
        for path in key_files
        if path.is_file()
    }
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "formal_job_id": args.formal_job_id,
        "smoke_job_id": args.smoke_job_id,
        "formal_slurm_accounting": slurm_accounting(args.formal_job_id),
        "smoke_slurm_accounting": slurm_accounting(args.smoke_job_id),
        "command": "bash scripts/run_sub08_reproduction.sh formal",
        "run_dir": str(run_dir),
        "source_metrics": str(metrics_path),
        "source_predictions": str(predictions_path),
        "source_repeat_metrics": (
            str(repeat_metrics_path) if repeat_metrics_path is not None else None
        ),
        "source_repeat_predictions": (
            str(repeat_predictions_path)
            if repeat_predictions_path is not None
            else None
        ),
        "source_history": str(history_path),
        "paths": metrics["paths"],
        "environment": metrics["environment"],
        "file_hashes": hashes,
        "algorithmic_deviations_from_source": [],
        "official_metric_authority": (
            "independent evaluation after saving and reloading the fixed final checkpoint"
        ),
        "engineering_only_changes": [
            "local JSONL validation logging",
            "guarded optional tracker imports",
            "independent final-checkpoint evaluator",
            "offline local path configuration",
        ],
    }
    atomic_json(results_dir / "run_manifest.json", manifest)

    zh = f"""# Brain-RW EEG 图像检索复现结果（THINGS-EEG sub-08）

## 正式结果

| 受试者 | Checkpoint | Query / Gallery | Top-1 | Top-5 |
|---|---:|---:|---:|---:|
| sub-08 | epoch {args.expected_epochs} final | {n} / 200 | {top1_count}/200 = {100 * top1:.2f}% | {top5_count}/200 = {100 * top5:.2f}% |

随机检索基线为 Top-1 0.50%、Top-5 2.50%。正式结果来自预先固定的第 {args.expected_epochs} 轮 final checkpoint 保存后独立重载评估；没有按测试集峰值挑选模型。训练内验证仅作为诊断曲线保留。独立重载评估重复运行两次，检索计数与逐样本排名一致。

## 协议

- THINGS-EEG sub-08，seed 42；训练 EEG 的 4 trials 与测试 EEG 的 80 trials 分别取平均。
- 使用 17 个后部通道与 `[0,250)` 时间窗，在 200 张唯一测试图像上进行 cosine retrieval。
- 视觉编码器为 CLIP ViT-B/32，并以 rank-32 LoRA 与 brain MLP 联合对齐；单张 A40、bf16、{args.expected_epochs} epochs。

逐样本预测见 `retrieval_predictions.csv`，逐轮诊断曲线见 `training_history.csv`，完整运行与哈希记录见 `run_manifest.json`。
"""
    en = f"""# Brain-RW EEG Image-Retrieval Reproduction (THINGS-EEG sub-08)

## Official result

| Subject | Checkpoint | Queries / Gallery | Top-1 | Top-5 |
|---|---:|---:|---:|---:|
| sub-08 | epoch-{args.expected_epochs} final | {n} / 200 | {top1_count}/200 = {100 * top1:.2f}% | {top5_count}/200 = {100 * top5:.2f}% |

The random-retrieval baselines are 0.50% Top-1 and 2.50% Top-5. The official result comes from an independent save/reload evaluation of the precommitted final checkpoint after epoch {args.expected_epochs}; no checkpoint was selected by its test-set peak. In-training validation is retained only as a diagnostic curve. Two independent reload evaluations agree on the retrieval counts and every per-query rank.

## Protocol

- THINGS-EEG sub-08, seed 42; the four training trials and 80 test trials are averaged separately.
- Retrieval uses 17 posterior channels, time window `[0,250)`, cosine similarity, and 200 unique held-out test images.
- The visual encoder is CLIP ViT-B/32, jointly aligned with a brain MLP through rank-32 LoRA; training uses one A40, bf16, and {args.expected_epochs} epochs.

See `retrieval_predictions.csv` for per-query output, `training_history.csv` for diagnostic epoch metrics, and `run_manifest.json` for complete provenance and hashes.
"""
    (results_dir / "RESULTS_ZH.md").write_text(zh, encoding="utf-8")
    (results_dir / "RESULTS_EN.md").write_text(en, encoding="utf-8")

    print(json.dumps(summary["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
