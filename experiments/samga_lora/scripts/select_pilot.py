#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from samga_lora.utils import atomic_write_json, hash_file  # noqa: E402


SUBJECTS = (1, 5, 8)
SEEDS = (42, 43)
RATIOS = (0.05, 0.10, 0.20)
EXPECTED_EPOCHS = (20, 25, 30, 35, 40, 45, 50, 55, 60)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def metrics_by_epoch(path: Path) -> dict[int, dict[str, Any]]:
    rows = read_jsonl(path)
    result = {int(row["epoch"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate validation epochs in {path}")
    if tuple(sorted(result)) != EXPECTED_EPOCHS:
        raise ValueError(f"Incomplete/unexpected validation epochs in {path}: {sorted(result)}")
    for row in rows:
        if (
            row.get("protocol") != "standard_independent_exact_image"
            or int(row.get("num_queries", -1)) != 200
            or int(row.get("num_gallery", -1)) != 200
            or not 0 <= float(row["top1"]) <= float(row["top5"]) <= 1
        ):
            raise ValueError(f"Invalid validation metric row in {path}: {row}")
    return result


def load_run_config(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    return config


def task_initialization_hash(config: dict[str, Any], path: Path) -> str:
    value = config.get("task_initial_state_sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"Missing task initialization hash in {path}")
    return value


def run_dir(root: Path, mode: str, subject: int, seed: int, ratio: float | None = None) -> Path:
    family = "frozen" if mode == "frozen" else f"lora-ratio-{ratio:.2f}"
    return root / family / f"sub-{subject:02d}" / f"seed-{seed}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select the global SAMGA-LoRA pilot setting")
    parser.add_argument("--pilot-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--table-output", required=True)
    parser.add_argument("--report-en", default=None)
    parser.add_argument("--report-zh", default=None)
    parser.add_argument("--minimum-mean-delta", type=float, default=0.005)
    parser.add_argument("--minimum-positive-cells", type=int, default=4)
    parser.add_argument("--subject-floor", type=float, default=-0.02)
    parser.add_argument("--preflight-report", default=None)
    parser.add_argument("--smoke-report", default=None)
    parser.add_argument("--fail-on-gate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.pilot_root).resolve()
    frozen: dict[tuple[int, int], dict[int, dict[str, Any]]] = {}
    lora: dict[tuple[float, int, int], dict[int, dict[str, Any]]] = {}
    for subject in SUBJECTS:
        for seed in SEEDS:
            frozen_dir = run_dir(root, "frozen", subject, seed)
            frozen[(subject, seed)] = metrics_by_epoch(frozen_dir / "validation_metrics.jsonl")
            frozen_config_path = frozen_dir / "run_config.json"
            frozen_config = load_run_config(frozen_config_path)
            frozen_hash = task_initialization_hash(frozen_config, frozen_config_path)
            frozen_completion = load_run_config(frozen_dir / "completion.json")
            if (
                frozen_config.get("vision_mode") != "frozen"
                or int(frozen_config.get("subject_id", -1)) != subject
                or int(frozen_config.get("seed", -1)) != seed
                or int(frozen_config.get("batch_size", -1)) != 512
                or int(frozen_config.get("train_rows", -1)) != 14540
                or frozen_config.get("eeg_l2norm") is not False
                or frozen_config.get("image_l2norm") is not True
                or not frozen_completion.get("completed")
                or int(frozen_completion.get("global_step", -1)) != 1680
            ):
                raise ValueError(f"Frozen run provenance mismatch in {frozen_dir}")
            for ratio in RATIOS:
                lora_dir = run_dir(root, "lora", subject, seed, ratio)
                lora[(ratio, subject, seed)] = metrics_by_epoch(
                    lora_dir / "validation_metrics.jsonl"
                )
                lora_config_path = lora_dir / "run_config.json"
                lora_config = load_run_config(lora_config_path)
                lora_hash = task_initialization_hash(lora_config, lora_config_path)
                lora_completion = load_run_config(lora_dir / "completion.json")
                if (
                    lora_config.get("vision_mode") != "lora"
                    or int(lora_config.get("subject_id", -1)) != subject
                    or int(lora_config.get("seed", -1)) != seed
                    or abs(float(lora_config.get("vision_lr_ratio", -1)) - ratio) > 1e-12
                    or int(lora_config.get("batch_size", -1)) != 512
                    or int(lora_config.get("train_rows", -1)) != 14540
                    or lora_config.get("eeg_l2norm") is not False
                    or lora_config.get("image_l2norm") is not True
                    or lora_config.get("manifest_sha256") != frozen_config.get("manifest_sha256")
                    or not lora_completion.get("completed")
                    or int(lora_completion.get("global_step", -1)) != 1680
                ):
                    raise ValueError(f"LoRA run provenance mismatch in {lora_dir}")
                if lora_hash != frozen_hash:
                    raise RuntimeError(
                        f"Paired task initialization mismatch for subject {subject}, "
                        f"seed {seed}, ratio {ratio}"
                    )
    common_epochs = set.intersection(
        *[set(rows) for rows in [*frozen.values(), *lora.values()]]
    )
    if not common_epochs:
        raise RuntimeError("Pilot runs have no common validation epoch")
    candidates: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []
    for ratio in RATIOS:
        for epoch in sorted(common_epochs):
            deltas: list[float] = []
            subject_deltas: dict[int, list[float]] = defaultdict(list)
            for subject in SUBJECTS:
                for seed in SEEDS:
                    baseline = float(frozen[(subject, seed)][epoch]["top1"])
                    treatment = float(lora[(ratio, subject, seed)][epoch]["top1"])
                    delta = treatment - baseline
                    deltas.append(delta)
                    subject_deltas[subject].append(delta)
                    table_rows.append(
                        {
                            "ratio": ratio,
                            "epoch": epoch,
                            "subject_id": subject,
                            "seed": seed,
                            "frozen_top1": baseline,
                            "lora_top1": treatment,
                            "delta_top1": delta,
                        }
                    )
            per_subject = {
                str(subject): sum(values) / len(values) for subject, values in subject_deltas.items()
            }
            candidates.append(
                {
                    "vision_lr_ratio": ratio,
                    "epoch": epoch,
                    "mean_top1_delta": sum(deltas) / len(deltas),
                    "positive_cells": sum(value > 0 for value in deltas),
                    "per_subject_mean_delta": per_subject,
                    "minimum_subject_mean_delta": min(per_subject.values()),
                }
            )
    candidates.sort(
        key=lambda row: (
            -row["mean_top1_delta"],
            -row["positive_cells"],
            row["vision_lr_ratio"],
            row["epoch"],
        )
    )
    selected = candidates[0]
    gate_passed = bool(
        selected["mean_top1_delta"] >= args.minimum_mean_delta
        and selected["positive_cells"] >= args.minimum_positive_cells
        and selected["minimum_subject_mean_delta"] >= args.subject_floor
    )
    upstream_gates: dict[str, Any] = {}
    for label, report_path in (
        ("preflight", args.preflight_report),
        ("smoke", args.smoke_report),
    ):
        if report_path:
            path = Path(report_path).resolve()
            with open(path, "r", encoding="utf-8") as handle:
                report = json.load(handle)
            if not report.get("passed"):
                raise RuntimeError(f"Upstream {label} gate did not pass: {path}")
            upstream_gates[label] = {
                "path": str(path),
                "sha256": hash_file(path),
            }
    output = {
        "schema_version": 1,
        "gate_passed": gate_passed,
        "selected": selected,
        "gate": {
            "minimum_mean_top1_delta": args.minimum_mean_delta,
            "minimum_positive_cells": args.minimum_positive_cells,
            "minimum_subject_mean_delta": args.subject_floor,
        },
        "subjects": list(SUBJECTS),
        "seeds": list(SEEDS),
        "candidates": candidates,
        "protocol": "concept_disjoint_validation_standard_independent_exact_image",
        "upstream_gates": upstream_gates,
        "source_sha256": {
            str(path.relative_to(EXPERIMENT_ROOT)): hash_file(path)
            for path in (
                EXPERIMENT_ROOT / "train.py",
                EXPERIMENT_ROOT / "evaluate.py",
                EXPERIMENT_ROOT / "samga_lora/data.py",
                EXPERIMENT_ROOT / "samga_lora/model.py",
                EXPERIMENT_ROOT / "samga_lora/utils.py",
                EXPERIMENT_ROOT / "scripts/aggregate_formal.py",
                EXPERIMENT_ROOT / "scripts/run_formal_cell.sh",
            )
        },
    }
    atomic_write_json(args.output, output)
    table_path = Path(args.table_output)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with open(table_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)
    selected_cells = [
        row for row in table_rows
        if row["ratio"] == selected["vision_lr_ratio"] and row["epoch"] == selected["epoch"]
    ]
    candidate_lines = "\n".join(
        f"| {row['vision_lr_ratio']:.2f} | {row['epoch']} | "
        f"{100 * row['mean_top1_delta']:+.2f} | {row['positive_cells']}/6 | "
        f"{100 * row['minimum_subject_mean_delta']:+.2f} |"
        for row in candidates
    )
    cell_lines = "\n".join(
        f"| {row['subject_id']:02d} | {row['seed']} | {100 * row['frozen_top1']:.2f} | "
        f"{100 * row['lora_top1']:.2f} | {100 * row['delta_top1']:+.2f} |"
        for row in selected_cells
    )
    status_en = "PASSED" if gate_passed else "NOT PASSED"
    status_zh = "通过" if gate_passed else "未通过"
    en = f"""# SAMGA + Visual LoRA Pilot\n\n- Protocol: concept-disjoint validation; standard independent 200-way retrieval.\n- Selected setting: vision LR ratio **{selected['vision_lr_ratio']:.2f}**, epoch **{selected['epoch']}**.\n- Selected mean paired Top-1 change: **{100 * selected['mean_top1_delta']:+.2f} percentage points**.\n- Positive cells: **{selected['positive_cells']}/6**; minimum subject-level mean change: **{100 * selected['minimum_subject_mean_delta']:+.2f} points**.\n- Pre-registered pilot gate: **{status_en}**.\n\n## Selected paired cells\n\n| Subject | Seed | Frozen Top-1 (%) | LoRA Top-1 (%) | Change (points) |\n|---:|---:|---:|---:|---:|\n{cell_lines}\n\n## All candidates in selection order\n\n| Vision LR ratio | Epoch | Mean Top-1 change (points) | Positive cells | Minimum subject mean (points) |\n|---:|---:|---:|---:|---:|\n{candidate_lines}\n"""
    zh = f"""# SAMGA + 视觉 LoRA Pilot 结果\n\n- 协议：concept-disjoint 验证集；标准独立 200-way 检索。\n- 锁定配置：视觉学习率比例 **{selected['vision_lr_ratio']:.2f}**，epoch **{selected['epoch']}**。\n- 所选配置的平均配对 Top-1 变化：**{100 * selected['mean_top1_delta']:+.2f} 个百分点**。\n- 正向单元：**{selected['positive_cells']}/6**；最小受试者平均变化：**{100 * selected['minimum_subject_mean_delta']:+.2f} 个百分点**。\n- 预注册 pilot 门控：**{status_zh}**。\n\n## 所选配置的配对单元\n\n| 受试者 | Seed | Frozen Top-1 (%) | LoRA Top-1 (%) | 变化（百分点） |\n|---:|---:|---:|---:|---:|\n{cell_lines}\n\n## 全部候选（按选择顺序）\n\n| 视觉学习率比例 | Epoch | Top-1 平均变化（百分点） | 正向单元 | 最小受试者平均变化（百分点） |\n|---:|---:|---:|---:|---:|\n{candidate_lines}\n"""
    if args.report_en:
        report_path = Path(args.report_en)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(en, encoding="utf-8")
    if args.report_zh:
        report_path = Path(args.report_zh)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(zh, encoding="utf-8")
    print(json.dumps(output["selected"], indent=2, sort_keys=True))
    print(f"gate_passed={gate_passed}")
    if args.fail_on_gate and not gate_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
