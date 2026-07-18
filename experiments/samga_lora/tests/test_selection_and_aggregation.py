from __future__ import annotations

import json
import csv
import hashlib
import subprocess
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_validation(path: Path, top1: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "epoch": epoch,
            "top1": top1,
            "top5": min(1.0, top1 + 0.2),
            "num_queries": 200,
            "num_gallery": 200,
            "protocol": "standard_independent_exact_image",
        }
        for epoch in (20, 25, 30, 35, 40, 45, 50, 55, 60)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_formal_run(path: Path, *, mode: str, subject: int, seed: int, top1_correct: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    checkpoint = path / "checkpoint_epoch025.pt"
    checkpoint.write_bytes(f"{mode}-{subject}-{seed}".encode())
    predictions = path / "test_predictions.csv"
    fieldnames = [
        "query_index", "query_image_id", "target_gallery_index", "predicted_image_id",
        "target_rank", "top5_image_ids", "top5_scores",
    ]
    with open(predictions, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(200):
            query = f"img{index:03d}"
            other = f"img{(index + 1) % 200:03d}"
            if index < top1_correct:
                rank = 1
                top5 = [query, other, "x2", "x3", "x4"]
            elif index < 196:
                rank = 2
                top5 = [other, query, "x2", "x3", "x4"]
            else:
                rank = 6
                top5 = [other, "x1", "x2", "x3", "x4"]
            writer.writerow(
                {
                    "query_index": index,
                    "query_image_id": query,
                    "target_gallery_index": index,
                    "predicted_image_id": top5[0],
                    "target_rank": rank,
                    "top5_image_ids": "|".join(top5),
                    "top5_scores": "1|0.9|0.8|0.7|0.6",
                }
            )
    _write_json(
        path / "test_metrics.json",
        {
            "subject_id": subject,
            "seed": seed,
            "checkpoint_epoch": 25,
            "protocol": "standard_independent_exact_image",
            "num_queries": 200,
            "num_gallery": 200,
            "vision_mode": mode,
            "vision_lr_ratio": 0.1,
            "task_initial_state_sha256": f"{subject * 100 + seed:064x}",
            "contrastive_eeg_l2norm": False,
            "contrastive_image_l2norm": True,
            "top1": top1_correct / 200,
            "top5": 0.98,
            "top1_correct": top1_correct,
            "top5_correct": 196,
            "checkpoint_sha256": _sha256(checkpoint),
            "predictions_sha256": _sha256(predictions),
        },
    )


def test_pilot_selector_locks_one_global_setting(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    for subject in (1, 5, 8):
        for seed in (42, 43):
            initial_hash = f"{seed:064x}"
            _write_validation(
                root / "frozen" / f"sub-{subject:02d}" / f"seed-{seed}" / "validation_metrics.jsonl",
                0.50,
            )
            _write_json(
                root / "frozen" / f"sub-{subject:02d}" / f"seed-{seed}" / "run_config.json",
                {
                    "task_initial_state_sha256": initial_hash,
                    "vision_mode": "frozen",
                    "subject_id": subject,
                    "seed": seed,
                    "batch_size": 512,
                    "train_rows": 14540,
                    "manifest_sha256": "manifest",
                    "eeg_l2norm": False,
                    "image_l2norm": True,
                },
            )
            _write_json(
                root / "frozen" / f"sub-{subject:02d}" / f"seed-{seed}" / "completion.json",
                {"completed": True, "global_step": 1680},
            )
            for ratio, score in ((0.05, 0.49), (0.10, 0.52), (0.20, 0.51)):
                _write_validation(
                    root / f"lora-ratio-{ratio:.2f}" / f"sub-{subject:02d}" / f"seed-{seed}" / "validation_metrics.jsonl",
                    score,
                )
                _write_json(
                    root / f"lora-ratio-{ratio:.2f}" / f"sub-{subject:02d}" / f"seed-{seed}" / "run_config.json",
                    {
                        "task_initial_state_sha256": initial_hash,
                        "vision_mode": "lora",
                        "vision_lr_ratio": ratio,
                        "subject_id": subject,
                        "seed": seed,
                        "batch_size": 512,
                        "train_rows": 14540,
                        "manifest_sha256": "manifest",
                        "eeg_l2norm": False,
                        "image_l2norm": True,
                    },
                )
                _write_json(
                    root / f"lora-ratio-{ratio:.2f}" / f"sub-{subject:02d}" / f"seed-{seed}" / "completion.json",
                    {"completed": True, "global_step": 1680},
                )
    output = tmp_path / "selection.json"
    subprocess.run(
        [
            sys.executable,
            str(EXPERIMENT_ROOT / "scripts/select_pilot.py"),
            "--pilot-root", str(root),
            "--output", str(output),
            "--table-output", str(tmp_path / "table.csv"),
        ],
        check=True,
    )
    selection = json.loads(output.read_text(encoding="utf-8"))
    assert selection["gate_passed"]
    assert selection["selected"]["vision_lr_ratio"] == 0.10
    assert selection["selected"]["epoch"] == 20


def test_formal_aggregation_is_paired_and_gated(tmp_path: Path) -> None:
    root = tmp_path / "formal"
    locked = tmp_path / "locked.json"
    locked.write_text(
        json.dumps({"gate_passed": True, "selected": {"epoch": 25, "vision_lr_ratio": 0.1}}),
        encoding="utf-8",
    )
    for mode, top1 in (("frozen", 0.80), ("lora", 0.81)):
        for subject in range(1, 11):
            for seed in range(42, 47):
                _write_formal_run(
                    root / mode / f"sub-{subject:02d}" / f"seed-{seed}",
                    mode=mode,
                    subject=subject,
                    seed=seed,
                    top1_correct=round(top1 * 200),
                )
    output = tmp_path / "results"
    subprocess.run(
        [
            sys.executable,
            str(EXPERIMENT_ROOT / "scripts/aggregate_formal.py"),
            "--formal-root", str(root),
            "--locked-config", str(locked),
            "--output-dir", str(output),
            "--bootstrap-samples", "200",
        ],
        check=True,
    )
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert abs(summary["paired_delta"]["top1"]["mean"] - 0.01) < 1e-12
    assert summary["success_criterion_passed"]
