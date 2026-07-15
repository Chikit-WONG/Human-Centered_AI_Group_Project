#!/usr/bin/env python3
"""Strictly validate and aggregate a complete subject-by-seed retrieval study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from .aggregate_subject_metrics import parse_subjects, validate_subject
except ImportError:  # Direct execution: python scripts/aggregate_multiseed_metrics.py
    from aggregate_subject_metrics import parse_subjects, validate_subject


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TASK = "THINGS-EEG brain-to-image independent retrieval"
EXPECTED_PROTOCOL = "standard independent per-query retrieval"


def parse_seeds(value: str) -> list[int]:
    seeds: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            seed = int(token)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"seed must be a non-negative integer: {token!r}"
            ) from error
        if seed < 0:
            raise argparse.ArgumentTypeError(
                f"seed must be non-negative: {seed}"
            )
        if seed in seeds:
            raise argparse.ArgumentTypeError(f"duplicate seed: {seed}")
        seeds.append(seed)
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def default_output_name(seeds: list[int]) -> str:
    """Return a compact, unambiguous directory name for an ordered seed list."""
    if len(seeds) == 1:
        # Avoid colliding with the source RESULTS_ROOT/seedN directory.
        return f"seeds{seeds[0]}"
    if all(current == previous + 1 for previous, current in zip(seeds, seeds[1:])):
        return f"seeds{seeds[0]}-{seeds[-1]}"
    return "seeds" + "_".join(map(str, seeds))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        default=str(PROJECT_ROOT / "results" / "all_subjects"),
        help="Directory containing seedN/summary.json for every requested seed.",
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Defaults to RESULTS_ROOT/seeds<first>-<last> for an ascending "
            "contiguous range, or an explicit ordered seed-list name otherwise."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=parse_seeds,
        default=parse_seeds("42,43,44,45,46"),
    )
    parser.add_argument(
        "--subjects",
        type=parse_subjects,
        default=parse_subjects("1-10"),
    )
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


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def atomic_csv(
    path: Path,
    fieldnames: list[str],
    rows: Iterable[dict[str, Any]],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})
    temporary.replace(path)


def sample_stdev(values: Iterable[float]) -> float:
    materialized = list(values)
    return statistics.stdev(materialized) if len(materialized) > 1 else 0.0


def build_cross_seed_metric_fields(
    *,
    seed_count: int,
    mean_top1: float,
    sd_top1: float,
    mean_top5: float,
    sd_top5: float,
    between_subject_sd_top1: float,
    between_subject_sd_top5: float,
) -> dict[str, float]:
    """Build generic primary fields plus five-seed compatibility aliases."""
    fields = {
        "cross_seed_mean_top1_percent": mean_top1,
        "cross_seed_sample_sd_top1_points": sd_top1,
        "cross_seed_mean_top5_percent": mean_top5,
        "cross_seed_sample_sd_top5_points": sd_top5,
        "between_subject_sample_sd_of_cross_seed_means_top1_points": (
            between_subject_sd_top1
        ),
        "between_subject_sample_sd_of_cross_seed_means_top5_points": (
            between_subject_sd_top5
        ),
    }
    if seed_count == 5:
        fields.update(
            {
                "five_seed_mean_top1_percent": mean_top1,
                "five_seed_sample_sd_top1_points": sd_top1,
                "five_seed_mean_top5_percent": mean_top5,
                "five_seed_sample_sd_top5_points": sd_top5,
                "between_subject_sample_sd_of_five_seed_means_top1_points": (
                    between_subject_sd_top1
                ),
                "between_subject_sample_sd_of_five_seed_means_top5_points": (
                    between_subject_sd_top5
                ),
            }
        )
    return fields


def assert_close(actual: float, expected: float, message: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{message}: {actual} != {expected}")


def validate_seed_summary(
    path: Path,
    *,
    seed: int,
    subjects: list[int],
    expected_epochs: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(f"seed {seed} summary is missing: {path}")
    summary = load_json(path)
    expected_subject_names = [f"sub-{subject_id:02d}" for subject_id in subjects]

    checks = {
        "task": EXPECTED_TASK,
        "protocol": EXPECTED_PROTOCOL,
        "checkpoint_policy": "fixed final checkpoint",
        "checkpoint_epoch": expected_epochs,
        "seed": seed,
        "subject_count": len(subjects),
        "subjects": expected_subject_names,
        "queries_per_subject": 200,
        "total_queries": 200 * len(subjects),
        "repeat_evaluation_verified_for_all": True,
    }
    for field, expected in checks.items():
        if summary.get(field) != expected:
            raise ValueError(
                f"seed {seed} summary field {field!r} differs: "
                f"{summary.get(field)!r} != {expected!r}"
            )

    source_rows = summary.get("per_subject")
    if not isinstance(source_rows, list) or len(source_rows) != len(subjects):
        raise ValueError(f"seed {seed} does not contain exactly {len(subjects)} rows")
    rows_by_subject = {int(row["subject_id"]): row for row in source_rows}
    if sorted(rows_by_subject) != subjects or len(rows_by_subject) != len(source_rows):
        raise ValueError(f"seed {seed} has duplicate or unexpected subjects")

    validated_rows: list[dict[str, Any]] = []
    for subject_id in subjects:
        source_row = rows_by_subject[subject_id]
        metrics_path = Path(source_row["metrics_path"])
        validated = validate_subject(
            metrics_path,
            subject_id=subject_id,
            seed=seed,
            expected_epochs=expected_epochs,
            allow_missing_repeat=False,
        )
        if validated != source_row:
            differing = sorted(
                key
                for key in set(validated) | set(source_row)
                if validated.get(key) != source_row.get(key)
            )
            raise ValueError(
                f"seed {seed}, sub-{subject_id:02d}: source summary is stale or "
                f"inconsistent on fields {differing}"
            )
        validated_rows.append(validated)

    total_top1 = sum(int(row["top1_count"]) for row in validated_rows)
    total_top5 = sum(int(row["top5_count"]) for row in validated_rows)
    total_queries = 200 * len(validated_rows)
    macro_top1 = statistics.fmean(
        float(row["top1_fraction"]) for row in validated_rows
    )
    macro_top5 = statistics.fmean(
        float(row["top5_fraction"]) for row in validated_rows
    )
    if int(summary.get("top1_count", -1)) != total_top1:
        raise ValueError(f"seed {seed}: summary Top-1 count is inconsistent")
    if int(summary.get("top5_count", -1)) != total_top5:
        raise ValueError(f"seed {seed}: summary Top-5 count is inconsistent")
    assert_close(macro_top1, total_top1 / total_queries, f"seed {seed} Top-1")
    assert_close(macro_top5, total_top5 / total_queries, f"seed {seed} Top-5")
    assert_close(
        float(summary["macro_top1_fraction"]), macro_top1, f"seed {seed} macro Top-1"
    )
    assert_close(
        float(summary["macro_top5_fraction"]), macro_top5, f"seed {seed} macro Top-5"
    )
    return summary, validated_rows


def check_cross_run_consistency(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reference = rows[0]
    required_environment_keys = [
        "python",
        "torch",
        "transformers",
        "datasets",
        "peft",
        "device",
        "dtype",
        "cuda_device",
    ]
    optional_environment_keys = ["conda_environment", "scipy"]
    metadata_exceptions: list[dict[str, Any]] = []
    for row in rows[1:]:
        label = f"seed {row['seed']}, {row['subject']}"
        for key in required_environment_keys:
            if row["environment"].get(key) != reference["environment"].get(key):
                raise ValueError(f"{label}: environment mismatch on {key}")
        for key in optional_environment_keys:
            reference_value = reference["environment"].get(key)
            row_value = row["environment"].get(key)
            if reference_value is None or row_value is None:
                metadata_exceptions.append(
                    {
                        "seed": int(row["seed"]),
                        "subject": row["subject"],
                        "field": key,
                        "reason": "missing from legacy metrics metadata",
                    }
                )
            elif row_value != reference_value:
                raise ValueError(f"{label}: environment mismatch on {key}")
        for key in [
            "checkpoint_epoch",
            "sample_count",
            "gallery_size",
            "clip_base_path",
            "brain_config_sha256",
            "vision_adapter_config_sha256",
        ]:
            if row[key] != reference[key]:
                raise ValueError(f"{label}: protocol/model mismatch on {key}")
    return metadata_exceptions


def build_seed_rows(
    rows: list[dict[str, Any]], seeds: list[int]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for seed in seeds:
        seed_rows = [row for row in rows if int(row["seed"]) == seed]
        total_queries = sum(int(row["sample_count"]) for row in seed_rows)
        top1_count = sum(int(row["top1_count"]) for row in seed_rows)
        top5_count = sum(int(row["top5_count"]) for row in seed_rows)
        result.append(
            {
                "seed": seed,
                "subject_count": len(seed_rows),
                "total_queries": total_queries,
                "top1_count": top1_count,
                "top5_count": top5_count,
                "top1_percent": 100.0 * top1_count / total_queries,
                "top5_percent": 100.0 * top5_count / total_queries,
                "between_subject_sample_sd_top1_points": sample_stdev(
                    float(row["top1_percent"]) for row in seed_rows
                ),
                "between_subject_sample_sd_top5_points": sample_stdev(
                    float(row["top5_percent"]) for row in seed_rows
                ),
                "repeat_verified_for_all": all(
                    bool(row["repeat_verified"]) for row in seed_rows
                ),
            }
        )
    return result


def build_subject_rows(
    rows: list[dict[str, Any]], subjects: list[int]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for subject_id in subjects:
        subject_rows = [
            row for row in rows if int(row["subject_id"]) == subject_id
        ]
        top1_values = [float(row["top1_percent"]) for row in subject_rows]
        top5_values = [float(row["top5_percent"]) for row in subject_rows]
        total_queries = sum(int(row["sample_count"]) for row in subject_rows)
        top1_count = sum(int(row["top1_count"]) for row in subject_rows)
        top5_count = sum(int(row["top5_count"]) for row in subject_rows)
        result.append(
            {
                "subject_id": subject_id,
                "subject": f"sub-{subject_id:02d}",
                "seed_count": len(subject_rows),
                "total_query_evaluations": total_queries,
                "top1_count": top1_count,
                "top5_count": top5_count,
                "mean_top1_percent": statistics.fmean(top1_values),
                "sample_sd_top1_points": sample_stdev(top1_values),
                "min_top1_percent": min(top1_values),
                "max_top1_percent": max(top1_values),
                "mean_top5_percent": statistics.fmean(top5_values),
                "sample_sd_top5_points": sample_stdev(top5_values),
                "min_top5_percent": min(top5_values),
                "max_top5_percent": max(top5_values),
                "repeat_verified_for_all": all(
                    bool(row["repeat_verified"]) for row in subject_rows
                ),
            }
        )
    return result


def collect_dataset_sources(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        metrics = load_json(Path(row["metrics_path"]))
        brain_directory = str(Path(metrics["paths"]["brain_directory"]))
        image_directory = str(Path(metrics["paths"]["image_directory"]))
        key = (brain_directory, image_directory)
        grouped.setdefault(key, []).append(f"seed{row['seed']}/{row['subject']}")
    return [
        {
            "brain_directory": brain_directory,
            "image_directory": image_directory,
            "brain_directory_exists_at_aggregation": Path(brain_directory).is_dir(),
            "image_directory_exists_at_aggregation": Path(image_directory).is_dir(),
            "run_count": len(run_labels),
            "runs": sorted(run_labels),
        }
        for (brain_directory, image_directory), run_labels in sorted(grouped.items())
    ]


def render_seed_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Seed | Top-1 | Top-5 | Correct@1 | Correct@5 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['top1_percent']:.2f}% | "
            f"{row['top5_percent']:.2f}% | {row['top1_count']}/"
            f"{row['total_queries']} | {row['top5_count']}/"
            f"{row['total_queries']} |"
        )
    return lines


def render_subject_table(
    rows: list[dict[str, Any]], *, chinese: bool, seed_count: int
) -> list[str]:
    english_seed_label = "five-seed" if seed_count == 5 else f"{seed_count}-seed"
    chinese_seed_label = "五-seed" if seed_count == 5 else f"{seed_count}-seed"
    if chinese:
        lines = [
            f"| 受试者 | Top-1 {chinese_seed_label} 均值 ± SD | 范围 | "
            f"Top-5 {chinese_seed_label} 均值 ± SD | 范围 |",
            "|---|---:|---:|---:|---:|",
        ]
    else:
        lines = [
            f"| Subject | Top-1 {english_seed_label} mean ± SD | Range | "
            f"Top-5 {english_seed_label} mean ± SD | Range |",
            "|---|---:|---:|---:|---:|",
        ]
    for row in rows:
        lines.append(
            f"| {row['subject']} | {row['mean_top1_percent']:.2f}% ± "
            f"{row['sample_sd_top1_points']:.2f} | "
            f"{row['min_top1_percent']:.1f}–{row['max_top1_percent']:.1f}% | "
            f"{row['mean_top5_percent']:.2f}% ± "
            f"{row['sample_sd_top5_points']:.2f} | "
            f"{row['min_top5_percent']:.1f}–{row['max_top5_percent']:.1f}% |"
        )
    return lines


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root).resolve()
    default_name = default_output_name(args.seeds)
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else results_root / default_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []
    for seed in args.seeds:
        path = (results_root / f"seed{seed}" / "summary.json").resolve()
        summary, rows = validate_seed_summary(
            path,
            seed=seed,
            subjects=args.subjects,
            expected_epochs=args.expected_epochs,
        )
        source_summaries.append(
            {
                "seed": seed,
                "path": str(path),
                "sha256": sha256(path),
                "legacy_subjects_reused": summary["legacy_subjects_reused"],
            }
        )
        all_rows.extend(rows)

    expected_run_count = len(args.seeds) * len(args.subjects)
    cell_keys = {(int(row["seed"]), int(row["subject_id"])) for row in all_rows}
    if len(all_rows) != expected_run_count or len(cell_keys) != expected_run_count:
        raise ValueError("subject-by-seed grid is incomplete or contains duplicates")
    environment_metadata_exceptions = check_cross_run_consistency(all_rows)

    seed_rows = build_seed_rows(all_rows, args.seeds)
    subject_rows = build_subject_rows(all_rows, args.subjects)
    dataset_sources = collect_dataset_sources(all_rows)
    unavailable_dataset_sources = [
        source
        for source in dataset_sources
        if not source["brain_directory_exists_at_aggregation"]
        or not source["image_directory_exists_at_aggregation"]
    ]
    seed_top1 = [float(row["top1_percent"]) for row in seed_rows]
    seed_top5 = [float(row["top5_percent"]) for row in seed_rows]
    grand_top1 = statistics.fmean(seed_top1)
    grand_top5 = statistics.fmean(seed_top5)
    seed_sd_top1 = sample_stdev(seed_top1)
    seed_sd_top5 = sample_stdev(seed_top5)
    total_queries = sum(int(row["sample_count"]) for row in all_rows)
    total_top1 = sum(int(row["top1_count"]) for row in all_rows)
    total_top5 = sum(int(row["top5_count"]) for row in all_rows)
    pooled_top1 = 100.0 * total_top1 / total_queries
    pooled_top5 = 100.0 * total_top5 / total_queries
    assert_close(grand_top1, pooled_top1, "grand/pooled Top-1")
    assert_close(grand_top5, pooled_top5, "grand/pooled Top-5")
    assert_close(
        grand_top1,
        statistics.fmean(float(row["mean_top1_percent"]) for row in subject_rows),
        "seed-marginal/subject-marginal Top-1",
    )
    assert_close(
        grand_top5,
        statistics.fmean(float(row["mean_top5_percent"]) for row in subject_rows),
        "seed-marginal/subject-marginal Top-5",
    )

    seed_count = len(args.seeds)
    subject_count = len(args.subjects)
    run_count = len(all_rows)
    subject_count_en = "ten" if subject_count == 10 else str(subject_count)
    between_subject_sd_top1 = sample_stdev(
        float(row["mean_top1_percent"]) for row in subject_rows
    )
    between_subject_sd_top5 = sample_stdev(
        float(row["mean_top5_percent"]) for row in subject_rows
    )
    aggregate = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": EXPECTED_TASK,
        "protocol": EXPECTED_PROTOCOL,
        "checkpoint_policy": "fixed final checkpoint",
        "checkpoint_epoch": args.expected_epochs,
        "seeds": args.seeds,
        "seed_count": seed_count,
        "subjects": [f"sub-{subject_id:02d}" for subject_id in args.subjects],
        "subject_count": subject_count,
        "run_count": run_count,
        "queries_per_run": 200,
        "total_query_evaluations": total_queries,
        "top1_count": total_top1,
        "top5_count": total_top5,
        **build_cross_seed_metric_fields(
            seed_count=seed_count,
            mean_top1=grand_top1,
            sd_top1=seed_sd_top1,
            mean_top5=grand_top5,
            sd_top5=seed_sd_top5,
            between_subject_sd_top1=between_subject_sd_top1,
            between_subject_sd_top5=between_subject_sd_top5,
        ),
        "pooled_top1_percent": pooled_top1,
        "pooled_top5_percent": pooled_top5,
        "cell_level_descriptive_sample_sd_top1_points": sample_stdev(
            float(row["top1_percent"]) for row in all_rows
        ),
        "cell_level_descriptive_sample_sd_top5_points": sample_stdev(
            float(row["top5_percent"]) for row in all_rows
        ),
        "primary_sd_definition": (
            "sample SD (ddof=1) across seed-level "
            f"{subject_count_en}-subject macro accuracies"
        ),
        "repeat_evaluation_verified_for_all_runs": all(
            bool(row["repeat_verified"]) for row in all_rows
        ),
        "source_file_hashes_revalidated": True,
        "environment": all_rows[0]["environment"],
        "environment_metadata_exceptions": environment_metadata_exceptions,
        "dataset_sources": dataset_sources,
        "dataset_provenance_limitations": [
            {
                "affected_runs": source["runs"],
                "reason": (
                    "The metrics record a legacy dataset path that no longer exists; "
                    "its byte identity with the current dataset root cannot be "
                    "retrospectively verified. Model/result artifacts and the retrieval "
                    "protocol remain strictly validated."
                ),
            }
            for source in unavailable_dataset_sources
        ],
        "source_summaries": source_summaries,
        "per_seed": seed_rows,
        "per_subject_across_seeds": subject_rows,
    }
    atomic_json(output_dir / "summary.json", aggregate)

    run_fields = [
        "subject",
        "subject_id",
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
    atomic_csv(output_dir / "per_run_metrics.csv", run_fields, all_rows)
    seed_fields = list(seed_rows[0])
    atomic_csv(output_dir / "per_seed_metrics.csv", seed_fields, seed_rows)
    subject_fields = list(subject_rows[0])
    atomic_csv(output_dir / "per_subject_metrics.csv", subject_fields, subject_rows)

    seed_table = render_seed_table(seed_rows)
    subject_table_en = render_subject_table(
        subject_rows, chinese=False, seed_count=seed_count
    )
    seed_word = "seed" if seed_count == 1 else "seeds"
    run_word = "run" if run_count == 1 else "runs"
    seed_count_en = "five" if seed_count == 5 else str(seed_count)
    title_en = (
        "# Five-Seed, Ten-Subject THINGS-EEG Retrieval Results"
        if seed_count == 5 and subject_count == 10
        else f"# {seed_count}-Seed, {subject_count}-Subject THINGS-EEG Retrieval Results"
    )
    en_lines = [
        title_en,
        "",
        f"Seeds: `{', '.join(map(str, args.seeds))}`. Fixed final checkpoint: "
        f"epoch `{args.expected_epochs}`. Protocol: standard independent 200-way "
        "retrieval; Hungarian assignment is not used.",
        "",
        "## Primary result",
        "",
        f"- Top-1: **{grand_top1:.2f}% ± {seed_sd_top1:.2f} percentage points**.",
        f"- Top-5: **{grand_top5:.2f}% ± {seed_sd_top5:.2f} percentage points**.",
        f"- Pooled counts: Top-1 **{total_top1}/{total_queries}**; Top-5 "
        f"**{total_top5}/{total_queries}**.",
        "",
        f"The ± term is the sample SD (ddof=1) across the {seed_count_en} seed-level "
        f"{subject_count_en}-subject macro accuracies. The {total_queries:,} query "
        "evaluations are repeated subject × seed evaluations on the same held-out "
        f"stimulus set, not {total_queries:,} independent test examples.",
        "",
        f"## Per-seed {subject_count_en}-subject means",
        "",
        *seed_table,
        "",
        f"## Per-subject results across {seed_count_en} {seed_word}",
        "",
        *subject_table_en,
        "",
        f"Every one of the {run_count} subject–seed {run_word} passed strict "
        "artifact validation and an independent checkpoint-reload repeat check.",
        *(
            [
                "",
                "Provenance note: some reused legacy metrics omit the Conda-"
                "environment or SciPy metadata fields. Their recorded "
                "Python, PyTorch, Transformers, Datasets, PEFT, CUDA device, and "
                "dtype values match the other runs.",
            ]
            if environment_metadata_exceptions
            else []
        ),
        *(
            [
                "",
                "Dataset-provenance limitation: one or more reused legacy metrics "
                "record an earlier dataset root that is no longer available. Its "
                "historical byte identity with the current dataset root "
                "cannot be verified retrospectively; this does not invalidate the "
                "saved model/result and repeat-reload checks, but it limits the data-"
                "source claim for that one run.",
            ]
            if unavailable_dataset_sources
            else []
        ),
        "",
    ]
    atomic_text(output_dir / "RESULTS_EN.md", "\n".join(en_lines))

    zh_seed_table = [
        line.replace("Seed", "随机种子")
        .replace("Correct@1", "Top-1 正确数")
        .replace("Correct@5", "Top-5 正确数")
        for line in seed_table
    ]
    subject_table_zh = render_subject_table(
        subject_rows, chinese=True, seed_count=seed_count
    )
    seed_count_zh = "五" if seed_count == 5 else str(seed_count)
    seed_quantity_zh = "五个" if seed_count == 5 else f"{seed_count} 个"
    subject_quantity_zh = "十名" if subject_count == 10 else f"{subject_count} 名"
    title_zh = (
        "# THINGS-EEG 十名受试者五随机种子检索结果"
        if seed_count == 5 and subject_count == 10
        else f"# THINGS-EEG {subject_count} 名受试者 {seed_count} 个随机种子检索结果"
    )
    zh_lines = [
        title_zh,
        "",
        f"随机种子：`{', '.join(map(str, args.seeds))}`。固定最终检查点：第 "
        f"`{args.expected_epochs}` 个 epoch。协议：标准 200-way 逐查询独立检索；"
        "不使用匈牙利分配。",
        "",
        "## 主结果",
        "",
        f"- Top-1：**{grand_top1:.2f}% ± {seed_sd_top1:.2f} 个百分点**。",
        f"- Top-5：**{grand_top5:.2f}% ± {seed_sd_top5:.2f} 个百分点**。",
        f"- 合并计数：Top-1 **{total_top1}/{total_queries}**；Top-5 "
        f"**{total_top5}/{total_queries}**。",
        "",
        f"± 项是{seed_quantity_zh}“单 seed {subject_quantity_zh}受试者宏平均”之间的"
        f"样本标准差（ddof=1）。这里的 {total_queries:,} 次查询评估是同一留出"
        f"刺激集上的受试者 × seed 重复评估，并非 {total_queries:,} 个相互独立的"
        "测试样本。",
        "",
        f"## 各 seed 的{subject_quantity_zh}受试者平均值",
        "",
        *zh_seed_table,
        "",
        f"## 各受试者的{seed_count_zh}-seed 结果",
        "",
        *subject_table_zh,
        "",
        f"全部 {run_count} 个 subject–seed 运行均通过严格产物验证及独立检查点"
        "重载重复验证。",
        *(
            [
                "",
                "来源说明：部分复用的旧版指标没有记录 Conda 环境名或 SciPy "
                "版本；其已记录的 Python、PyTorch、Transformers、Datasets、"
                "PEFT、CUDA 设备和 dtype 均与其余运行一致。",
            ]
            if environment_metadata_exceptions
            else []
        ),
        *(
            [
                "",
                "数据来源局限：一个或多个复用的旧版指标记录了现已不可用的早期"
                "数据根，因而无法事后验证它与当前数据根逐字节相同。"
                "这不影响已保存模型、结果与重复重载检查的有效性，但限制了对这一次运行"
                "的数据来源声明。",
            ]
            if unavailable_dataset_sources
            else []
        ),
        "",
    ]
    atomic_text(output_dir / "RESULTS_ZH.md", "\n".join(zh_lines))

    print(f"runs={len(all_rows)}")
    print(f"top1_mean_percent={grand_top1:.6f}")
    print(f"top1_seed_sample_sd_points={seed_sd_top1:.6f}")
    print(f"top5_mean_percent={grand_top5:.6f}")
    print(f"top5_seed_sample_sd_points={seed_sd_top5:.6f}")
    print(f"summary={output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
