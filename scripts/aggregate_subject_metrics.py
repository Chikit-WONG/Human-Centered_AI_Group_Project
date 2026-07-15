#!/usr/bin/env python3
"""Validate and aggregate independent retrieval metrics across THINGS-EEG subjects."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TASK = "THINGS-EEG brain-to-image retrieval"
EXPECTED_PROTOCOL = {
    "selected_channels": [
        "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
        "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2",
    ],
    "similarity": "cosine",
    "target_match": "unique image_id",
    "time_slice": [0, 250],
    "trial_averaging": True,
}


def parse_subjects(value: str) -> list[int]:
    subjects: set[int] = set()
    for item in value.split(","):
        token = item.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise argparse.ArgumentTypeError(f"invalid subject range: {token}")
            subjects.update(range(start, end + 1))
        else:
            subjects.add(int(token))
    if not subjects or min(subjects) < 1 or max(subjects) > 10:
        raise argparse.ArgumentTypeError("subjects must be in the range 1..10")
    return sorted(subjects)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        default=str(PROJECT_ROOT / "results" / "all_subjects" / "seed42"),
    )
    parser.add_argument(
        "--output-dir",
        help="Defaults to --results-root.",
    )
    parser.add_argument(
        "--legacy-sub08-metrics",
        default=str(PROJECT_ROOT / "results" / "sub08_seed42_formal_metrics.json"),
        help="Fallback for the previously verified sub-08 run.",
    )
    parser.add_argument("--subjects", type=parse_subjects, default=parse_subjects("1-10"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-epochs", type=int, default=25)
    parser.add_argument(
        "--allow-missing-repeat",
        action="store_true",
        help="Allow a subject without the independent repeat evaluation.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(path: Path) -> str:
    value = load_json(path)
    if isinstance(value.get("target_modules"), list):
        value["target_modules"] = sorted(value["target_modules"])
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def prediction_path_for(metrics_path: Path) -> Path:
    suffix = "_metrics.json"
    if not metrics_path.name.endswith(suffix):
        raise ValueError(f"unexpected metrics filename: {metrics_path.name}")
    return metrics_path.with_name(metrics_path.name[: -len(suffix)] + "_predictions.csv")


def repeat_metrics_path_for(metrics_path: Path) -> Path:
    suffix = "_formal_metrics.json"
    if not metrics_path.name.endswith(suffix):
        raise ValueError(f"unexpected formal metrics filename: {metrics_path.name}")
    return metrics_path.with_name(
        metrics_path.name[: -len(suffix)] + "_formal_repeat_metrics.json"
    )


def validate_predictions(
    path: Path,
    *,
    subject_id: int,
    sample_count: int,
    gallery_size: int,
    top1_count: int,
    top5_count: int,
) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"prediction CSV is missing: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != sample_count:
        raise ValueError(
            f"sub-{subject_id:02d}: expected {sample_count} prediction rows, "
            f"found {len(rows)}"
        )
    if len({row["gt_image_id"] for row in rows}) != sample_count:
        raise ValueError(f"sub-{subject_id:02d}: ground-truth image IDs are not unique")
    if any(int(row["subject_id"]) != subject_id for row in rows):
        raise ValueError(f"sub-{subject_id:02d}: prediction subject IDs disagree")

    query_indices = [int(row["query_index"]) for row in rows]
    if query_indices != list(range(sample_count)):
        raise ValueError(
            f"sub-{subject_id:02d}: query indices are not the ordered range "
            f"0..{sample_count - 1}"
        )

    derived_top1_count = 0
    derived_top5_count = 0
    for query_index, row in enumerate(rows):
        try:
            top5_image_ids = json.loads(row["top5_image_ids"])
            top5_similarities = json.loads(row["top5_cosine_similarities"])
        except (json.JSONDecodeError, TypeError) as error:
            raise ValueError(
                f"sub-{subject_id:02d}: invalid JSON list at query {query_index}"
            ) from error
        if not isinstance(top5_image_ids, list) or len(top5_image_ids) != 5:
            raise ValueError(
                f"sub-{subject_id:02d}: Top-5 ID list has invalid length at query "
                f"{query_index}"
            )
        if len(set(top5_image_ids)) != 5:
            raise ValueError(
                f"sub-{subject_id:02d}: duplicate Top-5 image ID at query "
                f"{query_index}"
            )
        if not isinstance(top5_similarities, list) or len(top5_similarities) != 5:
            raise ValueError(
                f"sub-{subject_id:02d}: Top-5 similarity list has invalid length "
                f"at query {query_index}"
            )
        top5_similarities = [float(value) for value in top5_similarities]
        if not all(math.isfinite(value) for value in top5_similarities):
            raise ValueError(
                f"sub-{subject_id:02d}: non-finite Top-5 similarity at query "
                f"{query_index}"
            )
        if any(
            left < right
            for left, right in zip(top5_similarities, top5_similarities[1:])
        ):
            raise ValueError(
                f"sub-{subject_id:02d}: Top-5 similarities are not descending at "
                f"query {query_index}"
            )
        if row["top1_image_id"] != top5_image_ids[0]:
            raise ValueError(
                f"sub-{subject_id:02d}: Top-1 is not the first Top-5 item at query "
                f"{query_index}"
            )
        top1_similarity = float(row["top1_cosine_similarity"])
        gt_similarity = float(row["gt_cosine_similarity"])
        if not math.isfinite(top1_similarity) or not math.isfinite(gt_similarity):
            raise ValueError(
                f"sub-{subject_id:02d}: non-finite recorded similarity at query "
                f"{query_index}"
            )
        if not math.isclose(
            top1_similarity,
            top5_similarities[0],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"sub-{subject_id:02d}: Top-1 similarity disagrees with Top-5 at "
                f"query {query_index}"
            )

        gt_image_id = row["gt_image_id"]
        gt_rank = int(row["gt_rank"])
        if not 1 <= gt_rank <= gallery_size:
            raise ValueError(
                f"sub-{subject_id:02d}: invalid ground-truth rank at query "
                f"{query_index}"
            )
        derived_top1 = int(row["top1_image_id"] == gt_image_id)
        derived_top5 = int(gt_image_id in top5_image_ids)
        if int(row["correct_top1"]) != derived_top1 or (gt_rank == 1) != bool(
            derived_top1
        ):
            raise ValueError(
                f"sub-{subject_id:02d}: Top-1 fields disagree at query {query_index}"
            )
        if int(row["correct_top5"]) != derived_top5 or (gt_rank <= 5) != bool(
            derived_top5
        ):
            raise ValueError(
                f"sub-{subject_id:02d}: Top-5 fields disagree at query {query_index}"
            )
        if derived_top5:
            top5_rank = top5_image_ids.index(gt_image_id) + 1
            if gt_rank != top5_rank:
                raise ValueError(
                    f"sub-{subject_id:02d}: ground-truth rank disagrees with Top-5 "
                    f"order at query {query_index}"
                )
            if not math.isclose(
                gt_similarity,
                top5_similarities[top5_rank - 1],
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"sub-{subject_id:02d}: ground-truth similarity disagrees with "
                    f"Top-5 at query {query_index}"
                )
        derived_top1_count += derived_top1
        derived_top5_count += derived_top5

    if derived_top1_count != top1_count:
        raise ValueError(f"sub-{subject_id:02d}: prediction Top-1 count disagrees")
    if derived_top5_count != top5_count:
        raise ValueError(f"sub-{subject_id:02d}: prediction Top-5 count disagrees")
    return rows


def validate_repeat(
    first_metrics: dict[str, Any],
    first_rows: list[dict[str, str]],
    repeat_metrics_path: Path,
    *,
    allow_missing: bool,
) -> bool:
    if not repeat_metrics_path.is_file():
        if allow_missing:
            return False
        raise FileNotFoundError(f"repeat metrics are missing: {repeat_metrics_path}")
    repeat_metrics = load_json(repeat_metrics_path)
    stable_metric_fields = [
        "sample_count",
        "gallery_size",
        "top1_count",
        "top5_count",
        "top1",
        "top5",
        "loss",
        "mean_gt_cosine_similarity",
        "subject_id",
        "seed",
    ]
    for field in stable_metric_fields:
        if repeat_metrics[field] != first_metrics[field]:
            raise ValueError(
                f"sub-{first_metrics['subject_id']:02d}: repeat metric differs on {field}"
            )

    repeat_rows = validate_predictions(
        prediction_path_for(repeat_metrics_path),
        subject_id=int(first_metrics["subject_id"]),
        sample_count=int(first_metrics["sample_count"]),
        gallery_size=int(first_metrics["gallery_size"]),
        top1_count=int(first_metrics["top1_count"]),
        top5_count=int(first_metrics["top5_count"]),
    )
    stable_prediction_fields = [
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
        zip(first_rows, repeat_rows, strict=True)
    ):
        for field in stable_prediction_fields:
            if first[field] != repeat[field]:
                raise ValueError(
                    f"sub-{first_metrics['subject_id']:02d}: repeat prediction differs "
                    f"at row {row_index}, field {field}"
                )
    return True


def locate_metrics(
    results_root: Path,
    legacy_sub08_metrics: Path,
    subject_id: int,
    seed: int,
) -> Path:
    padded = f"{subject_id:02d}"
    candidate = (
        results_root
        / f"subj{padded}"
        / f"sub{padded}_seed{seed}_formal_metrics.json"
    )
    if candidate.is_file():
        return candidate.resolve()
    if subject_id == 8 and legacy_sub08_metrics.is_file():
        return legacy_sub08_metrics.resolve()
    raise FileNotFoundError(f"formal metrics are missing for sub-{padded}: {candidate}")


def validate_subject(
    metrics_path: Path,
    *,
    subject_id: int,
    seed: int,
    expected_epochs: int,
    allow_missing_repeat: bool,
) -> dict[str, Any]:
    metrics = load_json(metrics_path)
    if metrics.get("task") != EXPECTED_TASK:
        raise ValueError(f"sub-{subject_id:02d}: unexpected task")
    if metrics.get("split") != "test":
        raise ValueError(f"sub-{subject_id:02d}: split is not test")
    if int(metrics.get("embedding_dim", -1)) != 512:
        raise ValueError(f"sub-{subject_id:02d}: embedding dimension is not 512")
    if metrics.get("protocol") != EXPECTED_PROTOCOL:
        raise ValueError(f"sub-{subject_id:02d}: retrieval protocol differs")
    if int(metrics["subject_id"]) != subject_id:
        raise ValueError(f"subject mismatch in {metrics_path}")
    if int(metrics["seed"]) != seed:
        raise ValueError(f"seed mismatch in {metrics_path}")
    if metrics.get("checkpoint_policy") != "final_epoch":
        raise ValueError(f"sub-{subject_id:02d}: checkpoint policy is not final_epoch")

    sample_count = int(metrics["sample_count"])
    gallery_size = int(metrics["gallery_size"])
    if sample_count != 200 or gallery_size != 200:
        raise ValueError(f"sub-{subject_id:02d}: expected a 200-query/200-image test")
    top1_count = int(metrics["top1_count"])
    top5_count = int(metrics["top5_count"])
    if not (0 <= top1_count <= top5_count <= sample_count):
        raise ValueError(f"sub-{subject_id:02d}: invalid retrieval counts")
    top1 = float(metrics["top1"])
    top5 = float(metrics["top5"])
    if not math.isclose(top1, top1_count / sample_count, abs_tol=1e-12):
        raise ValueError(f"sub-{subject_id:02d}: Top-1 fraction disagrees with count")
    if not math.isclose(top5, top5_count / sample_count, abs_tol=1e-12):
        raise ValueError(f"sub-{subject_id:02d}: Top-5 fraction disagrees with count")

    brain_model_dir = Path(metrics["paths"]["brain_model"])
    vision_model_dir = Path(metrics["paths"]["vision_adapter"])
    for model_path in (brain_model_dir, vision_model_dir):
        if not model_path.is_dir():
            raise FileNotFoundError(
                f"sub-{subject_id:02d}: saved model directory is missing: {model_path}"
            )
    brain_config_path = brain_model_dir / "config.json"
    vision_config_path = vision_model_dir / "adapter_config.json"
    for config_path in (brain_config_path, vision_config_path):
        if not config_path.is_file():
            raise FileNotFoundError(
                f"sub-{subject_id:02d}: saved model config is missing: {config_path}"
            )
    history_path = brain_model_dir.parent / "validation_metrics.jsonl"
    if not history_path.is_file():
        raise FileNotFoundError(
            f"sub-{subject_id:02d}: validation history is missing: {history_path}"
        )
    history = [
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(history) != expected_epochs:
        raise ValueError(
            f"sub-{subject_id:02d}: expected {expected_epochs} validation records, "
            f"found {len(history)}"
        )
    history_steps = [int(record["step"]) for record in history]
    if any(
        current <= previous
        for previous, current in zip(history_steps, history_steps[1:])
    ):
        raise ValueError(
            f"sub-{subject_id:02d}: validation-history steps are not strictly "
            "increasing"
        )
    for record_index, record in enumerate(history):
        history_samples = int(record["num_samples"])
        history_top1_count = int(record["top1_count"])
        history_top5_count = int(record["top5_count"])
        if history_samples != sample_count:
            raise ValueError(
                f"sub-{subject_id:02d}: validation-history sample count differs "
                f"at record {record_index}"
            )
        if not 0 <= history_top1_count <= history_top5_count <= history_samples:
            raise ValueError(
                f"sub-{subject_id:02d}: invalid validation-history counts at "
                f"record {record_index}"
            )
        if not math.isclose(
            float(record["top1_acc"]),
            history_top1_count / history_samples,
            rel_tol=0.0,
            abs_tol=1e-6,
        ) or not math.isclose(
            float(record["top5_acc"]),
            history_top5_count / history_samples,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(
                f"sub-{subject_id:02d}: validation-history accuracy/count "
                f"mismatch at record {record_index}"
            )

    brain_weights_path = brain_model_dir / "model.safetensors"
    vision_weights_path = vision_model_dir / "adapter_model.safetensors"
    for weights_path in (brain_weights_path, vision_weights_path):
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"sub-{subject_id:02d}: saved model weights are missing: "
                f"{weights_path}"
            )

    predictions_path = prediction_path_for(metrics_path)
    prediction_rows = validate_predictions(
        predictions_path,
        subject_id=subject_id,
        sample_count=sample_count,
        gallery_size=gallery_size,
        top1_count=top1_count,
        top5_count=top5_count,
    )
    repeat_metrics_path = repeat_metrics_path_for(metrics_path)
    repeat_verified = validate_repeat(
        metrics,
        prediction_rows,
        repeat_metrics_path,
        allow_missing=allow_missing_repeat,
    )

    return {
        "subject_id": subject_id,
        "subject": f"sub-{subject_id:02d}",
        "seed": seed,
        "checkpoint_epoch": expected_epochs,
        "sample_count": sample_count,
        "gallery_size": gallery_size,
        "top1_count": top1_count,
        "top5_count": top5_count,
        "top1_fraction": top1,
        "top5_fraction": top5,
        "top1_percent": 100.0 * top1,
        "top5_percent": 100.0 * top5,
        "loss": float(metrics["loss"]),
        "mean_gt_cosine_similarity": float(metrics["mean_gt_cosine_similarity"]),
        "repeat_verified": repeat_verified,
        "environment": metrics["environment"],
        "clip_base_path": str(Path(metrics["paths"]["clip_base"]).resolve()),
        "brain_config_sha256": canonical_json_sha256(brain_config_path),
        "vision_adapter_config_sha256": canonical_json_sha256(vision_config_path),
        "brain_weights_path": str(brain_weights_path),
        "brain_weights_sha256": sha256(brain_weights_path),
        "vision_adapter_weights_path": str(vision_weights_path),
        "vision_adapter_weights_sha256": sha256(vision_weights_path),
        "metrics_path": str(metrics_path),
        "metrics_sha256": sha256(metrics_path),
        "predictions_path": str(predictions_path),
        "predictions_sha256": sha256(predictions_path),
        "history_path": str(history_path),
        "history_sha256": sha256(history_path),
    }


def render_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Subject | Top-1 | Top-5 | Correct@1 | Correct@5 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['subject']} | {row['top1_percent']:.1f}% | "
            f"{row['top5_percent']:.1f}% | {row['top1_count']}/200 | "
            f"{row['top5_count']}/200 |"
        )
    return lines


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else results_root
    legacy_sub08_metrics = Path(args.legacy_sub08_metrics).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        validate_subject(
            locate_metrics(results_root, legacy_sub08_metrics, subject_id, args.seed),
            subject_id=subject_id,
            seed=args.seed,
            expected_epochs=args.expected_epochs,
            allow_missing_repeat=args.allow_missing_repeat,
        )
        for subject_id in args.subjects
    ]
    legacy_subjects_reused = [
        row["subject"]
        for row in rows
        if Path(row["metrics_path"]) == legacy_sub08_metrics
    ]

    environment_keys = ["python", "torch", "transformers", "datasets", "peft"]
    reference_environment = rows[0]["environment"]
    for row in rows[1:]:
        for key in environment_keys:
            if row["environment"].get(key) != reference_environment.get(key):
                raise ValueError(
                    f"environment mismatch for {row['subject']} on {key}: "
                    f"{row['environment'].get(key)} != {reference_environment.get(key)}"
                )
        for key in [
            "clip_base_path",
            "brain_config_sha256",
            "vision_adapter_config_sha256",
        ]:
            if row[key] != rows[0][key]:
                raise ValueError(
                    f"model/config mismatch for {row['subject']} on {key}"
                )

    total_queries = sum(row["sample_count"] for row in rows)
    total_top1 = sum(row["top1_count"] for row in rows)
    total_top5 = sum(row["top5_count"] for row in rows)
    macro_top1 = statistics.fmean(row["top1_fraction"] for row in rows)
    macro_top5 = statistics.fmean(row["top5_fraction"] for row in rows)
    micro_top1 = total_top1 / total_queries
    micro_top5 = total_top5 / total_queries
    if not math.isclose(macro_top1, micro_top1, abs_tol=1e-12):
        raise ValueError("macro and pooled Top-1 should match for equal subject sizes")
    if not math.isclose(macro_top5, micro_top5, abs_tol=1e-12):
        raise ValueError("macro and pooled Top-5 should match for equal subject sizes")

    aggregate = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": "THINGS-EEG brain-to-image independent retrieval",
        "protocol": "standard independent per-query retrieval",
        "checkpoint_policy": "fixed final checkpoint",
        "checkpoint_epoch": args.expected_epochs,
        "seed": args.seed,
        "subject_count": len(rows),
        "subjects": [row["subject"] for row in rows],
        "legacy_subjects_reused": legacy_subjects_reused,
        "queries_per_subject": 200,
        "total_queries": total_queries,
        "top1_count": total_top1,
        "top5_count": total_top5,
        "macro_top1_fraction": macro_top1,
        "macro_top5_fraction": macro_top5,
        "macro_top1_percent": 100.0 * macro_top1,
        "macro_top5_percent": 100.0 * macro_top5,
        "population_std_top1_percent": statistics.pstdev(
            row["top1_percent"] for row in rows
        ),
        "population_std_top5_percent": statistics.pstdev(
            row["top5_percent"] for row in rows
        ),
        "pooled_top1_fraction": micro_top1,
        "pooled_top5_fraction": micro_top5,
        "repeat_evaluation_verified_for_all": all(
            row["repeat_verified"] for row in rows
        ),
        "environment": reference_environment,
        "per_subject": rows,
    }
    atomic_json(output_dir / "summary.json", aggregate)

    csv_path = output_dir / "per_subject_metrics.csv"
    temporary_csv = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "subject",
            "seed",
            "checkpoint_epoch",
            "sample_count",
            "gallery_size",
            "top1_count",
            "top5_count",
            "top1_fraction",
            "top5_fraction",
            "top1_percent",
            "top5_percent",
            "loss",
            "mean_gt_cosine_similarity",
            "repeat_verified",
            "clip_base_path",
            "brain_config_sha256",
            "vision_adapter_config_sha256",
            "brain_weights_path",
            "brain_weights_sha256",
            "vision_adapter_weights_path",
            "vision_adapter_weights_sha256",
            "metrics_path",
            "metrics_sha256",
            "predictions_path",
            "predictions_sha256",
            "history_path",
            "history_sha256",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})
    temporary_csv.replace(csv_path)

    table = render_table(rows)
    complete_ten_subject_study = args.subjects == list(range(1, 11))
    en_study_title = (
        "Ten-Subject" if complete_ten_subject_study else f"{len(rows)}-Subject"
    )
    zh_study_title = "十名受试者" if complete_ten_subject_study else f"{len(rows)} 名受试者"
    en_lines = [
        f"# {en_study_title} THINGS-EEG Retrieval Results",
        "",
        f"Seed: `{args.seed}`. Checkpoint: fixed final epoch `{args.expected_epochs}`.",
        "Protocol: standard independent per-query retrieval over 200 images.",
        "",
        *table,
        "",
        "## Aggregate",
        "",
        f"- Mean Top-1: **{100.0 * macro_top1:.2f}%** "
        f"({total_top1}/{total_queries} pooled correct).",
        f"- Mean Top-5: **{100.0 * macro_top5:.2f}%** "
        f"({total_top5}/{total_queries} pooled correct).",
        f"- Between-subject population SD: Top-1 "
        f"{aggregate['population_std_top1_percent']:.2f} points; Top-5 "
        f"{aggregate['population_std_top5_percent']:.2f} points.",
        f"- Independent repeat evaluation verified for every subject: "
        f"**{aggregate['repeat_evaluation_verified_for_all']}**.",
        f"- Previously completed same-protocol subjects reused: "
        f"**{', '.join(legacy_subjects_reused) if legacy_subjects_reused else 'none'}**.",
        "",
        "These are standard Top-1/Top-5 results. Hungarian assignment is not used.",
        "",
    ]
    atomic_text(output_dir / "RESULTS_EN.md", "\n".join(en_lines))

    zh_table = [
        line.replace("Subject", "受试者")
        .replace("Correct@1", "Top-1 正确数")
        .replace("Correct@5", "Top-5 正确数")
        for line in table
    ]
    zh_lines = [
        f"# THINGS-EEG {zh_study_title}检索结果",
        "",
        f"随机种子：`{args.seed}`。检查点：固定的最终第 `{args.expected_epochs}` 个 epoch。",
        "协议：在 200 张图像上进行标准逐查询独立检索。",
        "",
        *zh_table,
        "",
        "## 汇总",
        "",
        f"- 平均 Top-1：**{100.0 * macro_top1:.2f}%**"
        f"（合并计数 {total_top1}/{total_queries}）。",
        f"- 平均 Top-5：**{100.0 * macro_top5:.2f}%**"
        f"（合并计数 {total_top5}/{total_queries}）。",
        f"- 被试间总体标准差：Top-1 为 "
        f"{aggregate['population_std_top1_percent']:.2f} 个百分点，Top-5 为 "
        f"{aggregate['population_std_top5_percent']:.2f} 个百分点。",
        f"- 所有受试者均通过独立重复评估："
        f"**{'是' if aggregate['repeat_evaluation_verified_for_all'] else '否'}**。",
        f"- 复用此前已完成的同协议受试者："
        f"**{', '.join(legacy_subjects_reused) if legacy_subjects_reused else '无'}**。",
        "",
        "以上均为标准 Top-1/Top-5 指标，没有使用匈牙利分配。",
        "",
    ]
    atomic_text(output_dir / "RESULTS_ZH.md", "\n".join(zh_lines))

    print(f"subjects={len(rows)}")
    print(f"mean_top1_percent={100.0 * macro_top1:.6f}")
    print(f"mean_top5_percent={100.0 * macro_top5:.6f}")
    print(f"summary={output_dir / 'summary.json'}")
    print(f"csv={csv_path}")


if __name__ == "__main__":
    main()
