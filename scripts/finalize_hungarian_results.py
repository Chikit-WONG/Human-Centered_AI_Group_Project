#!/usr/bin/env python3
"""Verify both Hungarian evaluation passes and write standalone result reports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSITION_NAMES = (
    "correct_to_correct",
    "correct_to_wrong",
    "wrong_to_correct",
    "wrong_to_wrong",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=PROJECT_ROOT / "results"
    )
    parser.add_argument("--expected-sample-count", type=int, default=200)
    parser.add_argument("--expected-top1-count", type=int, default=182)
    parser.add_argument("--expected-top5-count", type=int, default=199)
    parser.add_argument(
        "--slurm-job-id", default=os.environ.get("SLURM_JOB_ID")
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    require(isinstance(payload, dict), f"expected JSON object: {path}")
    return payload


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, f"missing CSV header: {path}")
        return list(reader.fieldnames), list(reader)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        return {name: bundle[name] for name in bundle.files}


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def sha256_strings(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="little"))
        digest.update(encoded)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
    )


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    paths = {
        "metrics": results_dir
        / "sub08_seed42_formal_hungarian_assignment_metrics.json",
        "repeat_metrics": results_dir
        / "sub08_seed42_formal_hungarian_assignment_repeat_metrics.json",
        "assignment_predictions": results_dir
        / "sub08_seed42_formal_hungarian_assignment_predictions.csv",
        "repeat_assignment_predictions": results_dir
        / "sub08_seed42_formal_hungarian_assignment_repeat_predictions.csv",
        "similarity_matrix": results_dir
        / "sub08_seed42_formal_cosine_similarity.npz",
        "repeat_similarity_matrix": results_dir
        / "sub08_seed42_formal_cosine_similarity_repeat.npz",
        "standard_predictions": results_dir
        / "sub08_seed42_formal_hungarian_standard_predictions.csv",
        "repeat_standard_predictions": results_dir
        / "sub08_seed42_formal_hungarian_standard_repeat_predictions.csv",
        "canonical_standard_predictions": results_dir
        / "sub08_seed42_formal_predictions.csv",
    }
    for label, path in paths.items():
        require(path.is_file(), f"missing {label}: {path}")

    metrics = load_json(paths["metrics"])
    repeat_metrics = load_json(paths["repeat_metrics"])
    for label, payload in (("primary", metrics), ("repeat", repeat_metrics)):
        require(
            payload["sample_count"] == args.expected_sample_count,
            f"{label}: unexpected sample count",
        )
        require(
            payload["top1_count"] == args.expected_top1_count,
            f"{label}: standard Top-1 regression failed",
        )
        require(
            payload["top5_count"] == args.expected_top5_count,
            f"{label}: standard Top-5 regression failed",
        )
        require(
            "hungarian_assignment" in payload,
            f"{label}: missing Hungarian metrics",
        )

    semantic_metrics = {
        key: value for key, value in metrics.items() if key != "paths"
    }
    semantic_repeat_metrics = {
        key: value for key, value in repeat_metrics.items() if key != "paths"
    }
    require(
        semantic_metrics == semantic_repeat_metrics,
        "independent reload metrics differ outside output paths",
    )

    stable_standard_fields = (
        "subject_id",
        "seed",
        "sample_count",
        "gallery_size",
        "embedding_dim",
        "top1_count",
        "top5_count",
        "top1",
        "top5",
        "loss",
        "mean_gt_cosine_similarity",
        "logit_scale",
        "protocol",
        "environment",
    )
    for field in stable_standard_fields:
        require(
            metrics[field] == repeat_metrics[field],
            f"independent reload mismatch in standard field: {field}",
        )

    assignment = metrics["hungarian_assignment"]
    repeat_assignment = repeat_metrics["hungarian_assignment"]
    stable_assignment_fields = (
        "metric_name",
        "protocol",
        "evaluation_scope",
        "method",
        "solver_version",
        "objective",
        "matrix_shape",
        "matrix_dtype",
        "matrix_sha256",
        "query_ids_sha256",
        "gallery_ids_sha256",
        "targets_sha256",
        "assignment_sha256",
        "assignment_correct_count",
        "assignment_accuracy",
        "delta_count_vs_independent_top1",
        "delta_percentage_points_vs_independent_top1",
        "changed_assignment_count",
        "total_assigned_cosine_similarity",
        "mean_assigned_cosine_similarity",
        "independent_rowwise_max_total_cosine_similarity",
        "independent_top1_unique_gallery_count",
        "independent_top1_duplicate_excess",
        "transitions",
        "top5_status",
    )
    for field in stable_assignment_fields:
        require(
            assignment[field] == repeat_assignment[field],
            f"independent reload mismatch in assignment field: {field}",
        )

    primary_bundle = load_npz(paths["similarity_matrix"])
    repeat_bundle = load_npz(paths["repeat_similarity_matrix"])
    expected_bundle_fields = {
        "cosine_similarity",
        "query_ids",
        "gallery_ids",
        "target_gallery_indices",
        "solver_row_permutation",
        "solver_column_permutation",
    }
    require(
        set(primary_bundle) == expected_bundle_fields,
        "primary similarity bundle has unexpected fields",
    )
    require(
        set(repeat_bundle) == expected_bundle_fields,
        "repeat similarity bundle has unexpected fields",
    )
    for field in sorted(expected_bundle_fields):
        require(
            np.array_equal(primary_bundle[field], repeat_bundle[field]),
            f"independent reload matrix bundle mismatch: {field}",
        )

    similarity = primary_bundle["cosine_similarity"]
    query_ids = primary_bundle["query_ids"].tolist()
    gallery_ids = primary_bundle["gallery_ids"].tolist()
    targets = primary_bundle["target_gallery_indices"]
    solver_row_permutation = primary_bundle["solver_row_permutation"]
    solver_column_permutation = primary_bundle["solver_column_permutation"]
    n = args.expected_sample_count
    require(similarity.shape == (n, n), "similarity matrix is not 200x200")
    require(np.isfinite(similarity).all(), "similarity matrix contains NaN/Inf")
    require(targets.shape == (n,), "target index array has the wrong shape")
    require(
        np.issubdtype(targets.dtype, np.integer),
        "target index array is not integer-valued",
    )
    require(
        bool(np.all((targets >= 0) & (targets < n))),
        "target index array contains out-of-range values",
    )
    for label, permutation in (
        ("solver row", solver_row_permutation),
        ("solver column", solver_column_permutation),
    ):
        require(permutation.shape == (n,), f"{label} permutation has wrong shape")
        require(
            np.issubdtype(permutation.dtype, np.integer),
            f"{label} permutation is not integer-valued",
        )
        require(
            np.array_equal(np.sort(permutation), np.arange(n)),
            f"{label} ordering is not a complete permutation",
        )
    require(len(set(query_ids)) == n, "query IDs are not unique")
    require(len(set(gallery_ids)) == n, "gallery IDs are not unique")
    require(set(query_ids) == set(gallery_ids), "query/gallery ID sets differ")
    for row_index, query_id in enumerate(query_ids):
        require(
            gallery_ids[int(targets[row_index])] == query_id,
            "saved target mapping does not follow image IDs",
        )
    require(
        assignment["matrix_sha256"] == sha256_array(similarity),
        "matrix SHA-256 does not match saved matrix",
    )
    require(
        assignment["query_ids_sha256"] == sha256_strings(query_ids),
        "query ID SHA-256 mismatch",
    )
    require(
        assignment["gallery_ids_sha256"] == sha256_strings(gallery_ids),
        "gallery ID SHA-256 mismatch",
    )
    require(
        assignment["targets_sha256"] == sha256_array(targets),
        "target index SHA-256 mismatch",
    )
    require(
        assignment["solver_row_permutation_sha256"]
        == sha256_array(solver_row_permutation),
        "solver row permutation SHA-256 mismatch",
    )
    require(
        assignment["solver_column_permutation_sha256"]
        == sha256_array(solver_column_permutation),
        "solver column permutation SHA-256 mismatch",
    )
    primary_rng = np.random.default_rng(assignment["primary_order_seed"])
    require(
        np.array_equal(primary_rng.permutation(n), solver_row_permutation),
        "saved primary solver row permutation is not seed-reproducible",
    )
    require(
        np.array_equal(primary_rng.permutation(n), solver_column_permutation),
        "saved primary solver column permutation is not seed-reproducible",
    )

    similarity_float64 = np.ascontiguousarray(similarity, dtype=np.float64)
    permuted_similarity = similarity_float64[solver_row_permutation][
        :, solver_column_permutation
    ]
    solved_rows, solved_columns = linear_sum_assignment(
        permuted_similarity, maximize=True
    )
    require(
        np.array_equal(np.sort(solved_rows), np.arange(n)),
        "independent solver did not cover every query",
    )
    require(
        np.array_equal(np.sort(solved_columns), np.arange(n)),
        "independent solver did not cover every gallery image",
    )
    independently_solved_assignment = np.full(n, -1, dtype=np.int64)
    independently_solved_assignment[solver_row_permutation[solved_rows]] = (
        solver_column_permutation[solved_columns]
    )
    independently_solved_objective = float(
        similarity_float64[
            np.arange(n, dtype=np.int64), independently_solved_assignment
        ].sum(dtype=np.float64)
    )
    require(
        math.isclose(
            independently_solved_objective,
            assignment["total_assigned_cosine_similarity"],
            rel_tol=0.0,
            abs_tol=1e-10,
        ),
        "saved assignment objective is not independently optimal",
    )

    # Negating the matrix preserves the evaluator's stable descending tie rule.
    independently_ranked = np.argsort(
        -similarity, axis=1, kind="stable"
    )
    independently_ranked_top1 = independently_ranked[:, 0]

    ordering_audit = assignment["ordering_sensitivity_audit"]
    audit_assignment_hashes: list[str] = []
    audit_objectives: list[float] = []
    audit_correct_counts: list[int] = []
    for record in ordering_audit["runs"]:
        audit_seed = int(record["order_seed"])
        audit_rng = np.random.default_rng(audit_seed)
        audit_rows = np.ascontiguousarray(
            audit_rng.permutation(n), dtype=np.int64
        )
        audit_columns = np.ascontiguousarray(
            audit_rng.permutation(n), dtype=np.int64
        )
        audit_matrix = similarity_float64[audit_rows][:, audit_columns]
        audit_solver_rows, audit_solver_columns = linear_sum_assignment(
            audit_matrix, maximize=True
        )
        audit_assignment = np.full(n, -1, dtype=np.int64)
        audit_assignment[audit_rows[audit_solver_rows]] = audit_columns[
            audit_solver_columns
        ]
        audit_hash = sha256_array(audit_assignment)
        audit_objective = float(
            similarity_float64[np.arange(n), audit_assignment].sum(
                dtype=np.float64
            )
        )
        audit_correct_count = int(
            np.count_nonzero(audit_assignment == targets)
        )
        require(
            record["assignment_sha256"] == audit_hash,
            f"ordering audit assignment mismatch for seed {audit_seed}",
        )
        require(
            math.isclose(
                record["objective_total_cosine_similarity"],
                audit_objective,
                rel_tol=0.0,
                abs_tol=1e-10,
            ),
            f"ordering audit objective mismatch for seed {audit_seed}",
        )
        require(
            record["assignment_correct_count"] == audit_correct_count,
            f"ordering audit accuracy mismatch for seed {audit_seed}",
        )
        audit_assignment_hashes.append(audit_hash)
        audit_objectives.append(audit_objective)
        audit_correct_counts.append(audit_correct_count)
    require(
        ordering_audit["run_count"] == len(ordering_audit["runs"]),
        "ordering audit run count mismatch",
    )
    require(
        ordering_audit["order_seeds"]
        == [record["order_seed"] for record in ordering_audit["runs"]],
        "ordering audit seed list mismatch",
    )
    require(
        ordering_audit["unique_assignment_count"]
        == len(set(audit_assignment_hashes)),
        "ordering audit unique-assignment count mismatch",
    )
    require(
        ordering_audit["all_mapped_assignments_equal"]
        == (len(set(audit_assignment_hashes)) == 1),
        "ordering audit equality flag mismatch",
    )
    require(
        math.isclose(
            ordering_audit["objective_min"],
            min(audit_objectives),
            rel_tol=0.0,
            abs_tol=1e-10,
        )
        and math.isclose(
            ordering_audit["objective_max"],
            max(audit_objectives),
            rel_tol=0.0,
            abs_tol=1e-10,
        ),
        "ordering audit objective range mismatch",
    )
    require(
        ordering_audit["assignment_correct_count_min"]
        == min(audit_correct_counts)
        and ordering_audit["assignment_correct_count_max"]
        == max(audit_correct_counts),
        "ordering audit accuracy range mismatch",
    )

    fields, rows = read_csv(paths["assignment_predictions"])
    repeat_fields, repeat_rows = read_csv(paths["repeat_assignment_predictions"])
    require(fields == repeat_fields, "assignment CSV headers differ across reloads")
    require(rows == repeat_rows, "assignment CSV rows differ across reloads")
    require(len(rows) == n, "assignment CSV does not have 200 rows")

    assigned_indices = np.asarray(
        [int(row["assigned_gallery_index"]) for row in rows], dtype=np.int64
    )
    raw_indices = np.asarray(
        [int(row["independent_top1_gallery_index"]) for row in rows],
        dtype=np.int64,
    )
    require(
        bool(np.all((assigned_indices >= 0) & (assigned_indices < n))),
        "assigned gallery indices contain out-of-range values",
    )
    require(
        bool(np.all((raw_indices >= 0) & (raw_indices < n))),
        "raw Top-1 indices contain out-of-range values",
    )
    require(
        sorted(assigned_indices.tolist()) == list(range(n)),
        "assigned gallery indices are not a bijection",
    )
    require(
        len({row["assigned_image_id"] for row in rows}) == n,
        "assigned image IDs are not unique",
    )
    require(
        assignment["assignment_sha256"] == sha256_array(assigned_indices),
        "assignment SHA-256 mismatch",
    )
    require(
        np.array_equal(assigned_indices, independently_solved_assignment),
        "saved assignment differs from the independently repeated solver",
    )
    require(
        np.array_equal(raw_indices, independently_ranked_top1),
        "saved raw Top-1 is not the stable per-row argmax",
    )

    derived_transitions: dict[str, list[str]] = {
        name: [] for name in TRANSITION_NAMES
    }
    raw_correct_count = 0
    assignment_correct_count = 0
    changed_count = 0
    assigned_score_values: list[float] = []
    for expected_row_index, row in enumerate(rows):
        row_index = int(row["query_index"])
        require(row_index == expected_row_index, "query indices are not ordered")
        query_id = row["query_image_id"]
        target_index = int(row["gt_gallery_index"])
        assigned_index = int(row["assigned_gallery_index"])
        raw_index = int(row["independent_top1_gallery_index"])
        require(query_id == query_ids[row_index], "query ID/order mismatch")
        require(
            int(row["subject_id"]) == metrics["subject_id"],
            "CSV subject ID mismatch",
        )
        require(row["gt_image_id"] == query_id, "GT ID differs from query ID")
        require(target_index == int(targets[row_index]), "target index mismatch")
        require(
            gallery_ids[target_index] == row["gt_image_id"],
            "target index does not map to GT image ID",
        )
        require(
            gallery_ids[assigned_index] == row["assigned_image_id"],
            "assigned index/image ID mismatch",
        )
        require(
            gallery_ids[raw_index] == row["independent_top1_image_id"],
            "raw Top-1 index/image ID mismatch",
        )

        expected_rank = int(
            np.flatnonzero(independently_ranked[row_index] == target_index)[0]
            + 1
        )
        require(int(row["gt_rank"]) == expected_rank, "GT rank mismatch")
        require(
            math.isclose(
                float(row["gt_cosine_similarity"]),
                float(similarity[row_index, target_index]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            "GT similarity mismatch",
        )

        assigned_score = float(row["assigned_similarity"])
        raw_score = float(row["independent_top1_similarity"])
        assigned_score_values.append(assigned_score)
        require(
            math.isclose(
                assigned_score,
                float(similarity[row_index, assigned_index]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            "assigned CSV similarity does not match saved matrix",
        )
        require(
            math.isclose(
                raw_score,
                float(similarity[row_index, raw_index]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            "raw CSV similarity does not match saved matrix",
        )
        require(
            assigned_score <= float(similarity[row_index].max()) + 1e-7,
            "assigned score exceeds row maximum",
        )
        require(
            int(row["assignment_changed_from_independent_top1"])
            == int(assigned_index != raw_index),
            "changed-assignment flag mismatch",
        )
        require(
            math.isclose(
                float(row["similarity_drop_from_independent_row_max"]),
                float(
                    similarity[row_index].max()
                    - similarity[row_index, assigned_index]
                ),
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            "similarity-drop field mismatch",
        )

        raw_correct = int(row["independent_top1_correct"])
        assignment_correct_row = int(row["assignment_correct"])
        require(raw_correct in (0, 1), "invalid raw correctness value")
        require(
            assignment_correct_row in (0, 1),
            "invalid assignment correctness value",
        )
        require(
            raw_correct == int(raw_index == target_index),
            "raw correctness does not follow ID mapping",
        )
        require(
            assignment_correct_row == int(assigned_index == target_index),
            "assignment correctness does not follow ID mapping",
        )
        transition = row["transition"]
        require(transition in derived_transitions, "unknown transition label")
        derived_transitions[transition].append(query_id)
        raw_correct_count += raw_correct
        assignment_correct_count += assignment_correct_row
        changed_count += int(assigned_index != raw_index)

    require(raw_correct_count == args.expected_top1_count, "raw CSV Top-1 mismatch")
    require(
        assignment_correct_count == assignment["assignment_correct_count"],
        "assignment CSV accuracy mismatch",
    )
    require(
        changed_count == assignment["changed_assignment_count"],
        "changed assignment count mismatch",
    )
    for name in TRANSITION_NAMES:
        require(
            assignment["transitions"][name]["count"]
            == len(derived_transitions[name]),
            f"transition count mismatch: {name}",
        )
        require(
            assignment["transitions"][name]["query_image_ids"]
            == derived_transitions[name],
            f"transition query list mismatch: {name}",
        )

    wrong_to_correct = len(derived_transitions["wrong_to_correct"])
    correct_to_wrong = len(derived_transitions["correct_to_wrong"])
    require(
        assignment_correct_count
        == args.expected_top1_count + wrong_to_correct - correct_to_wrong,
        "transition gain identity failed",
    )
    require(
        math.isclose(
            math.fsum(assigned_score_values),
            assignment["total_assigned_cosine_similarity"],
            rel_tol=0.0,
            abs_tol=1e-10,
        ),
        "assignment objective does not equal CSV score sum",
    )

    assignment_fraction = assignment_correct_count / n
    raw_prediction_counts = np.bincount(raw_indices, minlength=n)
    raw_total = float(
        similarity[np.arange(n, dtype=np.int64), raw_indices].sum(
            dtype=np.float64
        )
    )
    require(
        math.isclose(
            assignment["assignment_accuracy"],
            assignment_fraction,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        and math.isclose(
            assignment["assignment_fraction"],
            assignment_fraction,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        and math.isclose(
            assignment["assignment_percent"],
            100.0 * assignment_fraction,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "assignment fraction/percent fields are inconsistent",
    )
    require(
        assignment["delta_count_vs_independent_top1"]
        == assignment_correct_count - raw_correct_count,
        "assignment delta count mismatch",
    )
    require(
        math.isclose(
            assignment["delta_percentage_points_vs_independent_top1"],
            100.0 * (assignment_correct_count - raw_correct_count) / n,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "assignment percentage-point delta mismatch",
    )
    require(
        math.isclose(
            assignment["mean_assigned_cosine_similarity"],
            assignment["total_assigned_cosine_similarity"] / n,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "mean assignment similarity mismatch",
    )
    require(
        math.isclose(
            assignment["independent_rowwise_max_total_cosine_similarity"],
            raw_total,
            rel_tol=0.0,
            abs_tol=1e-10,
        ),
        "row-wise maximum objective mismatch",
    )
    require(
        math.isclose(
            assignment["one_to_one_constraint_similarity_cost"],
            raw_total - assignment["total_assigned_cosine_similarity"],
            rel_tol=0.0,
            abs_tol=1e-10,
        ),
        "one-to-one similarity cost mismatch",
    )
    require(
        assignment["independent_top1_unique_gallery_count"]
        == int(np.count_nonzero(raw_prediction_counts)),
        "raw unique-prediction count mismatch",
    )
    require(
        assignment["independent_top1_duplicate_excess"]
        == int(n - np.count_nonzero(raw_prediction_counts)),
        "raw duplicate-excess count mismatch",
    )
    require(
        assignment["independent_top1_collided_gallery_count"]
        == int(np.count_nonzero(raw_prediction_counts > 1)),
        "raw collision count mismatch",
    )
    require(
        assignment["independent_top1_max_collision_multiplicity"]
        == int(raw_prediction_counts.max()),
        "raw maximum collision multiplicity mismatch",
    )
    require(
        assignment["standard_reference"]["independent_top1_count"]
        == raw_correct_count
        and assignment["standard_reference"]["independent_top5_count"]
        == metrics["top5_count"],
        "assignment standard-reference counts mismatch",
    )

    standard_primary_path = Path(metrics["paths"]["predictions"])
    standard_repeat_path = Path(repeat_metrics["paths"]["predictions"])
    require(standard_primary_path.is_file(), "missing primary standard CSV")
    require(standard_repeat_path.is_file(), "missing repeat standard CSV")
    require(
        standard_primary_path.resolve() == paths["standard_predictions"],
        "primary metrics point to an unexpected standard CSV",
    )
    require(
        standard_repeat_path.resolve() == paths["repeat_standard_predictions"],
        "repeat metrics point to an unexpected standard CSV",
    )
    require(
        sha256_file(standard_primary_path) == sha256_file(standard_repeat_path),
        "standard prediction CSVs differ across reloads",
    )
    require(
        sha256_file(standard_primary_path)
        == sha256_file(paths["canonical_standard_predictions"]),
        "new standard predictions differ from the canonical formal evaluator",
    )

    top1_percent = 100.0 * metrics["top1"]
    top5_percent = 100.0 * metrics["top5"]
    assignment_percent = assignment["assignment_percent"]
    delta_pp = assignment["delta_percentage_points_vs_independent_top1"]
    transition_counts = {
        name: len(derived_transitions[name]) for name in TRANSITION_NAMES
    }
    summary_path = results_dir / "hungarian_summary.json"
    zh_report_path = results_dir / "HUNGARIAN_RESULTS_ZH.md"
    en_report_path = results_dir / "HUNGARIAN_RESULTS_EN.md"

    summary = {
        "schema_version": 1,
        "task": metrics["task"],
        "subject_id": metrics["subject_id"],
        "seed": metrics["seed"],
        "checkpoint_policy": metrics["checkpoint_policy"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": args.slurm_job_id,
        "standard_independent_retrieval": {
            "protocol": "independent_per_query_ranking",
            "top1_count": metrics["top1_count"],
            "top1_fraction": metrics["top1"],
            "top1_percent": top1_percent,
            "top5_count": metrics["top5_count"],
            "top5_fraction": metrics["top5"],
            "top5_percent": top5_percent,
        },
        "global_one_to_one_hungarian_assignment": assignment,
        "verification": {
            "passed": True,
            "standard_metric_regression_passed": True,
            "independent_checkpoint_reload_metrics_equal": True,
            "independent_checkpoint_reload_matrix_equal": True,
            "independent_checkpoint_reload_assignment_equal": True,
            "independent_solver_optimality_passed": True,
            "stable_rowwise_argmax_recomputed": True,
            "canonical_standard_predictions_equal": True,
            "assignment_is_bijection": True,
            "transition_ledger_valid": True,
            "objective_matches_prediction_csv": True,
            "ordering_sensitivity_audit_recomputed": True,
            "ordering_sensitivity_all_assignments_equal": ordering_audit[
                "all_mapped_assignments_equal"
            ],
        },
        "paths": {
            **{name: str(path) for name, path in paths.items()},
            "summary": str(summary_path),
            "report_zh": str(zh_report_path),
            "report_en": str(en_report_path),
        },
        "sha256": {
            name: sha256_file(path) for name, path in paths.items()
        },
    }
    write_json(summary_path, summary)

    zh_report = f"""# THINGS-EEG Sub-08：Hungarian 一对一检索结果

## 结论

在同一个 Sub-08 final checkpoint 和同一个 `200 × 200` 余弦相似度矩阵上，标准逐查询 Top-1 为 **{metrics['top1_count']}/200（{top1_percent:.1f}%）**，Top-5 为 **{metrics['top5_count']}/200（{top5_percent:.1f}%）**。加入 Hungarian 全局一对一约束后，assignment accuracy 为 **{assignment['assignment_correct_count']}/200（{assignment_percent:.1f}%）**，相对标准 Top-1 变化 **{assignment['delta_count_vs_independent_top1']:+d} 个样本 / {delta_pp:+.1f} 个百分点**。

这个 Hungarian 数字不是标准 Top-1：它同时看到完整测试 query batch，并使用“200 个 query 与 200 张 gallery 图片严格一一对应”的额外先验。单次 assignment 每个 query 只有一个结果，因此没有可比的 Hungarian Top-5。

## 指标

| 评估协议 | Top-1 / assignment accuracy | Top-5 |
|---|---:|---:|
| 标准逐 query 独立排序 | {metrics['top1_count']}/200（{top1_percent:.1f}%） | {metrics['top5_count']}/200（{top5_percent:.1f}%） |
| Hungarian 全局一对一 assignment | {assignment['assignment_correct_count']}/200（{assignment_percent:.1f}%） | N/A |

## 样本迁移与冲突

- correct → correct：{transition_counts['correct_to_correct']}
- wrong → correct：{transition_counts['wrong_to_correct']}
- correct → wrong：{transition_counts['correct_to_wrong']}
- wrong → wrong：{transition_counts['wrong_to_wrong']}
- assignment 改变了 {assignment['changed_assignment_count']} 个 query 的独立 Top-1 选择。
- 原始 200 个独立 Top-1 预测只覆盖 {assignment['independent_top1_unique_gallery_count']} 张唯一图片，重复占位为 {assignment['independent_top1_duplicate_excess']} 个；Hungarian 强制 200 张图片各使用一次。

提分恒等式已核验：`Hungarian correct = standard Top-1 correct + wrong→correct − correct→wrong`。

## 协议与验收

- 模型：既有 Sub-08、seed 42、25 epoch final checkpoint；未重训、未挑 checkpoint。
- 输入：trial-averaged EEG、17 个通道、时间窗 `[0, 250)`、normalized brain/image embeddings。
- 求解器只接收余弦相似度矩阵并最大化总相似度；真值只在 assignment 完成后用于评分。
- 矩阵：`{assignment['matrix_shape'][0]} × {assignment['matrix_shape'][1]}`，SHA-256 `{assignment['matrix_sha256']}`。
- 为避免同序 query/gallery 在平局时偏向对角线，主结果预先固定 seed `{assignment['primary_order_seed']}` 的独立行列排列；另用 {ordering_audit['run_count'] - 1} 组预定排列审计。映回原 ID 后共有 {ordering_audit['unique_assignment_count']} 个不同 assignment，正确数范围为 {ordering_audit['assignment_correct_count_min']}–{ordering_audit['assignment_correct_count_max']}；这些准确率没有参与主结果选择。
- 最优 assignment 总余弦相似度：{assignment['total_assigned_cosine_similarity']:.9f}；逐 query 独立行最大值之和：{assignment['independent_rowwise_max_total_cosine_similarity']:.9f}。
- 两次独立 checkpoint 重载的标准指标、矩阵、assignment、逐样本 CSV 与迁移清单完全一致。
- 环境：conda `{metrics['environment']['conda_environment']}`，PyTorch `{metrics['environment']['torch']}`，SciPy `{metrics['environment']['scipy']}`，GPU `{metrics['environment']['cuda_device']}`。

## 使用建议

课程报告应把 **91.0% Top-1 / 99.5% Top-5** 保留为标准主结果，把 Hungarian 结果作为“利用 closed-set 一对一先验的 transductive inference ablation”。这样既能展示算法带来的变化，也不会把不同协议的数字混为标准 retrieval 成绩。
"""
    write_text(zh_report_path, zh_report)

    en_report = f"""# THINGS-EEG Subject 08: Hungarian One-to-One Retrieval Result

## Result

Using the same Subject 08 final checkpoint and the same `200 × 200` cosine-similarity matrix, standard independent retrieval obtains **{metrics['top1_count']}/200 ({top1_percent:.1f}%) Top-1** and **{metrics['top5_count']}/200 ({top5_percent:.1f}%) Top-5**. Global Hungarian one-to-one decoding obtains **{assignment['assignment_correct_count']}/200 ({assignment_percent:.1f}%) assignment accuracy**, a change of **{assignment['delta_count_vs_independent_top1']:+d} samples / {delta_pp:+.1f} percentage points** from standard Top-1.

The Hungarian number is not standard Top-1. It jointly observes the complete test query batch and assumes a known bijection between all 200 queries and all 200 gallery images. A single assignment returns only one image per query, so Hungarian Top-5 is not defined here.

## Metrics

| Evaluation protocol | Top-1 / assignment accuracy | Top-5 |
|---|---:|---:|
| Standard independent per-query ranking | {metrics['top1_count']}/200 ({top1_percent:.1f}%) | {metrics['top5_count']}/200 ({top5_percent:.1f}%) |
| Global Hungarian one-to-one assignment | {assignment['assignment_correct_count']}/200 ({assignment_percent:.1f}%) | N/A |

## Transitions and collisions

- correct → correct: {transition_counts['correct_to_correct']}
- wrong → correct: {transition_counts['wrong_to_correct']}
- correct → wrong: {transition_counts['correct_to_wrong']}
- wrong → wrong: {transition_counts['wrong_to_wrong']}
- The assignment changes {assignment['changed_assignment_count']} independent Top-1 selections.
- The 200 independent Top-1 predictions cover only {assignment['independent_top1_unique_gallery_count']} unique images, leaving {assignment['independent_top1_duplicate_excess']} duplicate slots; Hungarian uses every gallery image exactly once.

The identity `Hungarian correct = standard Top-1 correct + wrong→correct − correct→wrong` was verified.

## Protocol and verification

- Model: existing Subject 08, seed 42, 25-epoch final checkpoint; no retraining or checkpoint selection.
- Inputs: trial-averaged EEG, 17 channels, `[0, 250)` time slice, and normalized brain/image embeddings.
- The solver receives only the cosine-similarity matrix and maximizes total similarity. Ground truth is consulted only after assignment for scoring.
- Matrix: `{assignment['matrix_shape'][0]} × {assignment['matrix_shape'][1]}`, SHA-256 `{assignment['matrix_sha256']}`.
- To prevent aligned query/gallery order from favoring the diagonal under exact ties, the primary result uses predeclared independent row/column permutations with seed `{assignment['primary_order_seed']}`. Another {ordering_audit['run_count'] - 1} predeclared orderings were audited. After mapping back to original IDs, they produced {ordering_audit['unique_assignment_count']} distinct assignment(s), with correct-count range {ordering_audit['assignment_correct_count_min']}–{ordering_audit['assignment_correct_count_max']}; accuracy was never used to select the primary result.
- Total cosine similarity of the optimal assignment: {assignment['total_assigned_cosine_similarity']:.9f}; sum of independent row maxima: {assignment['independent_rowwise_max_total_cosine_similarity']:.9f}.
- Two independent checkpoint reloads produced identical standard metrics, matrices, assignments, per-query CSVs, and transition ledgers.
- Environment: conda `{metrics['environment']['conda_environment']}`, PyTorch `{metrics['environment']['torch']}`, SciPy `{metrics['environment']['scipy']}`, GPU `{metrics['environment']['cuda_device']}`.

## Reporting recommendation

Keep **91.0% Top-1 / 99.5% Top-5** as the primary standard result. Present the Hungarian result as a transductive inference ablation that exploits a closed-set one-to-one prior, not as a directly comparable standard retrieval score.
"""
    write_text(en_report_path, en_report)

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
