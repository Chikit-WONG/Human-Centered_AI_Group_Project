from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from matching_fairness.artifacts import ScoreArtifact, write_score_artifact
from matching_fairness.native_export import _score_artifact_sha256
from matching_fairness.provenance import OFFICIAL_SOURCE_URL
from matching_fairness.reporting import (
    _validate_checkpoint_manifest,
    aggregate_records,
    aggregate_results,
    load_reproduction_audits,
    load_run_records,
    publish_aggregate,
    render_chinese_report,
    render_english_report,
)
from matching_fairness.scenarios import standard_scenarios


MODELS = ("nice", "atm_s", "our_project")
DECODERS = (
    "independent",
    "greedy",
    "hungarian",
    "stable_matching",
    "sinkhorn",
)
MANIFEST_HASH = "a" * 64


def _percent(count: int, total: int) -> float | None:
    return None if total == 0 else 100.0 * count / total


def _record(model: str, scenario_index: int, decoder: str) -> dict[str, object]:
    duplicate_count = (0, 10, 20)[scenario_index - 27] if scenario_index >= 27 else None
    suite = "duplicate_eeg" if duplicate_count is not None else "standard"
    scenario = (
        f"dupq{duplicate_count}"
        if duplicate_count is not None
        else standard_scenarios()[scenario_index].slug
    )
    total = 200 + duplicate_count if duplicate_count is not None else 200
    strict = decoder in {"greedy", "hungarian", "stable_matching"}
    unmatched = duplicate_count if duplicate_count is not None and strict else 0
    correct = total - unmatched
    independent_correct = total
    if decoder == "independent":
        assignment_metadata: dict[str, object] = {}
    elif decoder == "greedy":
        assignment_metadata = {
            "matched_count": total - unmatched,
            "unmatched_count": unmatched,
        }
    elif decoder == "hungarian":
        assignment_metadata = {
            "seed": 42,
            "row_permutation_sha256": "1" * 64,
            "column_permutation_sha256": "2" * 64,
            "matched_count": total - unmatched,
            "unmatched_count": unmatched,
            "assigned_sum_similarity": float(total - unmatched),
        }
    elif decoder == "stable_matching":
        assignment_metadata = {
            "matched_count": total - unmatched,
            "unmatched_count": unmatched,
            "proposal_count": total,
        }
    else:
        assignment_metadata = {
            "temperature": 0.05,
            "max_iterations": 500,
            "iterations": 10,
            "tolerance": 1e-8,
            "marginal_error": 1e-9,
            "converged": True,
            "plan_min": 0.0,
            "plan_max": 1.0,
            "plan_sum": 1.0,
            "plan_sha256": "3" * 64,
        }
    source = (
        {"standard": f"{MODELS.index(model) + 1:064x}"}
        if suite == "standard"
        else {
            "eeg_a": f"{MODELS.index(model) + 11:064x}",
            "eeg_b": f"{MODELS.index(model) + 21:064x}",
        }
    )
    record: dict[str, object] = {
        "model": model,
        "subject": "sub-08",
        "seed": 42,
        "suite": suite,
        "scenario_index": scenario_index,
        "scenario": scenario,
        "scenario_manifest_sha256": MANIFEST_HASH,
        "source_artifact_sha256": source,
        "decoder": decoder,
        "correct": correct,
        "total": total,
        "top1": _percent(correct, total),
        "answerable_correct": correct,
        "answerable_total": total,
        "answerable_top1": _percent(correct, total),
        "unanswerable_count": 0,
        "assigned_count": total - unmatched,
        "unmatched_count": unmatched,
        "unique_gallery_entry_predictions": total - unmatched,
        "unique_canonical_predictions": total - unmatched,
        "strict_one_to_one": strict,
        "top5_count": total if decoder == "independent" else None,
        "top5": 100.0 if decoder == "independent" else None,
        "assignment_changes_from_independent": unmatched,
        "delta_correct_vs_independent": correct - independent_correct,
        "assignment_metadata": assignment_metadata,
        "correct_to_correct": correct,
        "correct_to_wrong": independent_correct - correct,
        "wrong_to_correct": 0,
        "wrong_to_wrong": 0,
    }
    if duplicate_count is not None:
        both = 0 if strict else duplicate_count
        ceiling = total - duplicate_count if strict else total
        record.update(
            {
                "base_a_correct": 200,
                "base_a_total": 200,
                "base_a_top1": 100.0,
                "appended_b_correct": both,
                "appended_b_total": duplicate_count,
                "appended_b_top1": _percent(both, duplicate_count),
                "repeated_canonical_total": duplicate_count,
                "at_least_one_correct_count": duplicate_count,
                "at_least_one_coverage": _percent(duplicate_count, duplicate_count),
                "both_correct_count": both,
                "both_correct": _percent(both, duplicate_count),
                "theoretical_ceiling_count": ceiling,
                "theoretical_ceiling": _percent(ceiling, total),
                "distance_from_ceiling": ceiling - correct,
                "unmatched_repeated_queries": unmatched,
            }
        )
    return record


def valid_records() -> list[dict[str, object]]:
    return [
        _record(model, scenario_index, decoder)
        for model in MODELS
        for scenario_index in range(30)
        for decoder in DECODERS
    ]


def audit_rows() -> tuple[dict[str, object], ...]:
    return (
        {
            "model": "nice",
            "formal_epoch": 4,
            "formal_val_loss": 0.125,
            "formal_top1_count": 150,
            "formal_top5_count": 190,
            "sample_count": 200,
            "best_test_epoch": 7,
            "best_test_top1_count": 155,
            "best_test_top5_count": 192,
            "source_commit": "f" * 40,
            "checkpoint_manifest_sha256": "1" * 64,
            "audit_manifest_sha256": "2" * 64,
        },
        {
            "model": "atm_s",
            "formal_epoch": 3,
            "formal_val_loss": 0.25,
            "formal_top1_count": 140,
            "formal_top5_count": 188,
            "sample_count": 200,
            "best_test_epoch": 6,
            "best_test_top1_count": 146,
            "best_test_top5_count": 189,
            "source_commit": "e" * 40,
            "checkpoint_manifest_sha256": "3" * 64,
            "audit_manifest_sha256": "4" * 64,
        },
    )


def test_report_rejects_incomplete_extra_and_duplicate_grid() -> None:
    records = valid_records()
    with pytest.raises(ValueError, match="450"):
        aggregate_records(records[:-1])
    with pytest.raises(ValueError, match="450"):
        aggregate_records(records + [records[0]])
    with pytest.raises(ValueError, match="unique"):
        aggregate_records(records[:-1] + [records[0]])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("model", "samga", "model"),
        ("suite", "audit", "suite"),
        ("scenario_index", 30, "scenario"),
        ("decoder", "beam", "decoder"),
    ),
)
def test_report_rejects_bad_grid_identity(
    field: str,
    value: object,
    message: str,
) -> None:
    records = valid_records()
    records[0][field] = value
    with pytest.raises(ValueError, match=message):
        aggregate_records(records)


def test_report_rejects_metric_tampering_and_assignment_top5() -> None:
    records = valid_records()
    records[0]["top1"] = 12.5
    with pytest.raises(ValueError, match="top1"):
        aggregate_records(records)

    records = valid_records()
    assignment = next(row for row in records if row["decoder"] == "hungarian")
    assignment["top5_count"] = 1
    assignment["top5"] = 0.5
    with pytest.raises(ValueError, match="Top-5"):
        aggregate_records(records)


def test_report_rejects_mixed_manifest_hashes_and_audit_contamination() -> None:
    records = valid_records()
    records[-1]["scenario_manifest_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="manifest"):
        aggregate_records(records)

    records = valid_records()
    records[0]["scope"] = "best_test_audit_only"
    with pytest.raises(ValueError, match="audit"):
        aggregate_records(records)


def test_report_rejects_arbitrary_extra_fields_and_formula_cells() -> None:
    records = valid_records()
    records[0]["note"] = "=HYPERLINK(\"https://invalid\")"
    with pytest.raises(ValueError, match="schema|extra|formula"):
        aggregate_records(records)


def test_sinkhorn_metadata_is_strict_and_nonconvergence_is_bilingual() -> None:
    records = valid_records()
    sinkhorn = next(row for row in records if row["decoder"] == "sinkhorn")
    sinkhorn["assignment_metadata"] = {
        **sinkhorn["assignment_metadata"],
        "converged": False,
        "iterations": 500,
        "marginal_error": 1e-5,
    }
    aggregate = aggregate_records(records)
    english = render_english_report(aggregate, audit_rows())
    chinese = render_chinese_report(aggregate, audit_rows())
    assert "WARNING" in english and "1/90" in english
    assert "警告" in chinese and "1/90" in chinese

    records = valid_records()
    sinkhorn = next(row for row in records if row["decoder"] == "sinkhorn")
    sinkhorn["assignment_metadata"]["temperature"] = 0.2
    with pytest.raises(ValueError, match="Sinkhorn"):
        aggregate_records(records)

    adversarial = (
        {"converged": True, "iterations": 1, "marginal_error": 1e-3},
        {"converged": False, "iterations": 1, "marginal_error": 1e-12},
    )
    for mutation in adversarial:
        records = valid_records()
        sinkhorn = next(row for row in records if row["decoder"] == "sinkhorn")
        sinkhorn["assignment_metadata"].update(mutation)
        with pytest.raises(ValueError, match="Sinkhorn"):
            aggregate_records(records)


def test_publish_rejects_extra_audit_field_with_hpc_path(tmp_path: Path) -> None:
    rows = [dict(row) for row in audit_rows()]
    rows[0]["debug_path"] = "/hpc2hdd/home/private/checkpoint.pth"
    destination = tmp_path / "aggregate"

    with pytest.raises(ValueError, match="audit.*schema|schema.*audit"):
        publish_aggregate(aggregate_records(valid_records()), destination, rows)
    assert not destination.exists()


def test_standard_presentation_represents_all_27_scenarios() -> None:
    report = render_english_report(aggregate_records(valid_records()), audit_rows())
    for index, scenario in enumerate(standard_scenarios()):
        assert f"{index:02d} {scenario.slug}" in report


def test_reports_contain_bilingual_single_cell_limitations_and_audit_warning() -> None:
    aggregate = aggregate_records(valid_records())
    english = render_english_report(aggregate, audit_rows())
    chinese = render_chinese_report(aggregate, audit_rows())

    assert "sub-08 / seed-42" in english
    assert "does not establish cross-subject significance" in english
    assert "test-set-selected" in english
    assert "epoch 4" in english and "0.125" in english
    assert "sub-08 / seed-42" in chinese
    assert "不能建立跨被试显著性" in chinese
    assert "测试集选模" in chinese
    assert "第 4 轮" in chinese and "0.125" in chinese


def test_assignment_top5_is_never_rendered() -> None:
    aggregate = aggregate_records(valid_records())
    report = render_english_report(aggregate, audit_rows())
    assert "Hungarian Top-5" not in report
    assert "Stable Matching Top-5" not in report
    assert "Greedy Top-5" not in report


def test_best_highlighting_is_comparable_and_marks_all_ties() -> None:
    records = valid_records()
    for record in records:
        if record["scenario_index"] == 0 and record["decoder"] == "independent":
            if record["model"] == "our_project":
                record["correct"] = 199
                record["top1"] = 99.5
                record["answerable_correct"] = 199
                record["answerable_top1"] = 99.5
                record["correct_to_correct"] = 199
                record["wrong_to_wrong"] = 1
    report = render_english_report(aggregate_records(records), audit_rows())
    assert report.count("**200/200 (100.00%)**") >= 2
    assert "Standard (80-trial averages)" in report
    assert "Duplicate EEG (40-trial disjoint averages)" in report
    assert "not compared across these two suites" in report


def test_duplicate_metrics_and_denominators_are_preserved() -> None:
    aggregate = aggregate_records(valid_records())
    row = next(
        row
        for row in aggregate.records
        if row["model"] == "nice"
        and row["scenario_index"] == 29
        and row["decoder"] == "hungarian"
    )
    assert row["theoretical_ceiling_count"] == 200
    assert row["distance_from_ceiling"] == 0
    assert row["unmatched_count"] == 20
    assert row["at_least_one_correct_count"] == 20
    assert row["repeated_canonical_total"] == 20
    assert row["both_correct_count"] == 0


def test_publication_is_byte_stable_atomic_and_no_clobber(tmp_path: Path) -> None:
    aggregate = aggregate_records(valid_records())
    first = tmp_path / "aggregate-a"
    second = tmp_path / "aggregate-b"
    publish_aggregate(aggregate, first, audit_rows())
    publish_aggregate(aggregate, second, audit_rows())

    expected = {
        "RESULTS.md",
        "RESULTS_ZH.md",
        "aggregate_metrics.csv",
        "aggregate_summary.json",
        "presentation_duplicate_eeg.md",
        "presentation_standard.md",
    }
    assert {path.name for path in first.iterdir()} == expected
    assert {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in first.iterdir()
    } == {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in second.iterdir()
    }
    with pytest.raises(FileExistsError):
        publish_aggregate(aggregate, first, audit_rows())
    assert not list(tmp_path.glob(".aggregate-a.tmp-*"))


def _runner_module():
    path = Path("experiments/matching_fairness/scripts/run_scenarios.py")
    spec = importlib.util.spec_from_file_location("task7_reporting_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PER_QUERY_FIELDS = (
    "model",
    "subject",
    "seed",
    "suite",
    "scenario_index",
    "scenario",
    "decoder",
    "query_index",
    "query_id",
    "target_canonical_id",
    "answerable",
    "gallery_index",
    "predicted_gallery_entry_id",
    "predicted_canonical_id",
    "assigned_score",
    "unmatched",
    "correct_top1",
    "correct_top5",
)


def _fixture_cell(
    model: str,
    index: int,
    selection: dict[str, object],
    gallery: tuple[str, ...],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    suite = "standard" if index < 27 else "duplicate_eeg"
    scenario = (
        standard_scenarios()[index].slug
        if index < 27
        else f"dupq{(0, 10, 20)[index - 27]}"
    )
    if suite == "standard":
        dropped_queries = set(selection["drop_query"]) | set(selection["drop_pair"])
        dropped_gallery = set(selection["drop_gallery"]) | set(selection["drop_pair"])
        query_ids = tuple(value for value in gallery if value not in dropped_queries)
        targets = query_ids
        base_gallery = tuple(value for value in gallery if value not in dropped_gallery)
        duplicates = tuple(selection["duplicate_gallery"])
        gallery_canonical = base_gallery + duplicates
        gallery_entries = base_gallery + tuple(
            f"{value}__duplicate_entry_{position:04d}"
            for position, value in enumerate(duplicates)
        )
    else:
        repeated = tuple(selection["duplicate_query_ids"])
        query_ids = gallery + tuple(f"{value}__eeg_b" for value in repeated)
        targets = gallery + repeated
        gallery_canonical = gallery
        gallery_entries = gallery

    rows_by_decoder: dict[str, list[dict[str, object]]] = {}
    columns_by_target: dict[str, list[int]] = {}
    for column, canonical_id in enumerate(gallery_canonical):
        columns_by_target.setdefault(canonical_id, []).append(column)
    for decoder in DECODERS:
        strict = decoder in {"greedy", "hungarian", "stable_matching"}
        used: set[int] = set()
        rows: list[dict[str, object]] = []
        for query_index, (query_id, target) in enumerate(zip(query_ids, targets)):
            answerable = target in columns_by_target
            candidates = columns_by_target.get(target, [])
            if strict:
                correct_candidates = [value for value in candidates if value not in used]
                available = [
                    value for value in range(len(gallery_entries)) if value not in used
                ]
                gallery_index = (
                    correct_candidates[0]
                    if correct_candidates
                    else (available[0] if available else -1)
                )
                if gallery_index >= 0:
                    used.add(gallery_index)
            else:
                gallery_index = candidates[0] if candidates else 0
            unmatched = gallery_index < 0
            predicted_entry = None if unmatched else gallery_entries[gallery_index]
            predicted_canonical = (
                None if unmatched else gallery_canonical[gallery_index]
            )
            correct_top1 = predicted_canonical == target
            correct_top5 = answerable if decoder == "independent" else None
            rows.append(
                {
                    "model": model,
                    "subject": "sub-08",
                    "seed": 42,
                    "suite": suite,
                    "scenario_index": index,
                    "scenario": scenario,
                    "decoder": decoder,
                    "query_index": query_index,
                    "query_id": query_id,
                    "target_canonical_id": target,
                    "answerable": answerable,
                    "gallery_index": gallery_index,
                    "predicted_gallery_entry_id": predicted_entry,
                    "predicted_canonical_id": predicted_canonical,
                    "assigned_score": None if unmatched else (1.0 if correct_top1 else 0.0),
                    "unmatched": unmatched,
                    "correct_top1": correct_top1,
                    "correct_top5": correct_top5,
                }
            )
        rows_by_decoder[decoder] = rows

    records: list[dict[str, object]] = []
    independent = rows_by_decoder["independent"]
    for decoder in DECODERS:
        rows = rows_by_decoder[decoder]
        correct = sum(row["correct_top1"] is True for row in rows)
        answerable = sum(row["answerable"] is True for row in rows)
        answerable_correct = sum(
            row["answerable"] is True and row["correct_top1"] is True
            for row in rows
        )
        matched = sum(row["unmatched"] is False for row in rows)
        transitions = {
            "correct_to_correct": sum(
                old["correct_top1"] is True and new["correct_top1"] is True
                for old, new in zip(independent, rows)
            ),
            "correct_to_wrong": sum(
                old["correct_top1"] is True and new["correct_top1"] is False
                for old, new in zip(independent, rows)
            ),
            "wrong_to_correct": sum(
                old["correct_top1"] is False and new["correct_top1"] is True
                for old, new in zip(independent, rows)
            ),
            "wrong_to_wrong": sum(
                old["correct_top1"] is False and new["correct_top1"] is False
                for old, new in zip(independent, rows)
            ),
        }
        metadata = _fixture_assignment_metadata(
            decoder, matched, len(rows) - matched, len(rows)
        )
        if decoder == "hungarian":
            metadata["assigned_sum_similarity"] = sum(
                float(row["assigned_score"])
                for row in rows
                if row["assigned_score"] is not None
            )
        record = {
            "model": model,
            "suite": suite,
            "scenario_index": index,
            "scenario": scenario,
            "decoder": decoder,
            "correct": correct,
            "total": len(rows),
            "top1": _percent(correct, len(rows)),
            "answerable_correct": answerable_correct,
            "answerable_total": answerable,
            "answerable_top1": _percent(answerable_correct, answerable),
            "unanswerable_count": len(rows) - answerable,
            "assigned_count": matched,
            "unmatched_count": len(rows) - matched,
            "unique_gallery_entry_predictions": len(
                {row["predicted_gallery_entry_id"] for row in rows if not row["unmatched"]}
            ),
            "unique_canonical_predictions": len(
                {row["predicted_canonical_id"] for row in rows if not row["unmatched"]}
            ),
            "strict_one_to_one": decoder
            in {"greedy", "hungarian", "stable_matching"},
            "top5_count": sum(row["correct_top5"] is True for row in rows)
            if decoder == "independent"
            else None,
            "top5": _percent(
                sum(row["correct_top5"] is True for row in rows), len(rows)
            )
            if decoder == "independent"
            else None,
            "assignment_changes_from_independent": sum(
                old["gallery_index"] != new["gallery_index"]
                for old, new in zip(independent, rows)
            ),
            "delta_correct_vs_independent": correct
            - sum(row["correct_top1"] is True for row in independent),
            "assignment_metadata": metadata,
            **transitions,
        }
        if suite == "duplicate_eeg":
            repeated_targets = {
                row["target_canonical_id"]
                for row in rows
                if str(row["query_id"]).endswith("__eeg_b")
            }
            by_target = {
                target: [
                    row for row in rows if row["target_canonical_id"] == target
                ]
                for target in repeated_targets
            }
            base = [row for row in rows if not str(row["query_id"]).endswith("__eeg_b")]
            appended = [row for row in rows if str(row["query_id"]).endswith("__eeg_b")]
            strict = decoder in {"greedy", "hungarian", "stable_matching"}
            ceiling = min(len(rows), len(gallery_entries)) if strict else answerable
            at_least_one = sum(
                any(row["correct_top1"] for row in values)
                for values in by_target.values()
            )
            both = sum(
                all(row["correct_top1"] for row in values)
                for values in by_target.values()
            )
            record.update(
                {
                    "base_a_correct": sum(row["correct_top1"] for row in base),
                    "base_a_total": len(base),
                    "base_a_top1": _percent(
                        sum(row["correct_top1"] for row in base), len(base)
                    ),
                    "appended_b_correct": sum(row["correct_top1"] for row in appended),
                    "appended_b_total": len(appended),
                    "appended_b_top1": _percent(
                        sum(row["correct_top1"] for row in appended), len(appended)
                    ),
                    "repeated_canonical_total": len(repeated_targets),
                    "at_least_one_correct_count": at_least_one,
                    "at_least_one_coverage": _percent(at_least_one, len(repeated_targets)),
                    "both_correct_count": both,
                    "both_correct": _percent(both, len(repeated_targets)),
                    "theoretical_ceiling_count": ceiling,
                    "theoretical_ceiling": _percent(ceiling, len(rows)),
                    "distance_from_ceiling": ceiling - correct,
                    "unmatched_repeated_queries": sum(
                        row["unmatched"]
                        and row["target_canonical_id"] in repeated_targets
                        for row in rows
                    ),
                }
            )
        records.append(record)
    return records, [row for decoder in DECODERS for row in rows_by_decoder[decoder]]


def _fixture_assignment_metadata(
    decoder: str, matched: int, unmatched: int, total: int
) -> dict[str, object]:
    record = _record("nice", 29 if total > 200 else 0, decoder)
    metadata = dict(record["assignment_metadata"])
    if decoder in {"greedy", "hungarian", "stable_matching"}:
        metadata["matched_count"] = matched
        metadata["unmatched_count"] = unmatched
    if decoder == "hungarian":
        metadata["assigned_sum_similarity"] = float(matched)
    if decoder == "stable_matching":
        metadata["proposal_count"] = matched
    return metadata


def _csv_text(rows: list[dict[str, object]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=PER_QUERY_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(_csv_text(rows), encoding="utf-8")


def _write_task7_fixture(
    root: Path,
    source_hashes: dict[tuple[str, str], str] | None = None,
) -> None:
    runner = _runner_module()
    gallery = tuple(f"image-{index:03d}" for index in range(200))
    configs = runner._decoder_configs(
        seed=42,
        sinkhorn={
            "temperature": 0.05,
            "max_iterations": 500,
            "tolerance": 1e-8,
        },
    )
    plan = runner._build_scenario_plan(
        gallery,
        seed=42,
        trial_manifest_sha256="c" * 64,
        decoder_configs=configs,
    )
    root.mkdir()
    (root / "scenario_manifest.json").write_bytes(plan.manifest_bytes)
    for model in MODELS:
        for index in range(30):
            suite = "standard" if index < 27 else "duplicate_eeg"
            scenario = (
                standard_scenarios()[index].slug
                if index < 27
                else f"dupq{(0, 10, 20)[index - 27]}"
            )
            directory = root / model / "subj08" / "seed42" / suite / f"{index:02d}_{scenario}"
            directory.mkdir(parents=True)
            defaults = {
                (name, role): f"{value:064x}"
                for name_index, name in enumerate(MODELS)
                for role, value in (
                    ("standard", name_index + 1),
                    ("eeg_a", name_index + 11),
                    ("eeg_b", name_index + 21),
                )
            }
            resolved_hashes = {**defaults, **(source_hashes or {})}
            source = (
                {"standard": resolved_hashes[(model, "standard")]}
                if suite == "standard"
                else {
                    "eeg_a": resolved_hashes[(model, "eeg_a")],
                    "eeg_b": resolved_hashes[(model, "eeg_b")],
                }
            )
            if index < 27:
                spec = standard_scenarios()[index]
                matrix_shape = [
                    200 - spec.drop_query - spec.drop_pair,
                    200
                    - spec.drop_gallery
                    - spec.drop_pair
                    + spec.duplicate_gallery,
                ]
            else:
                matrix_shape = [200 + (0, 10, 20)[index - 27], 200]
            selection = runner._jsonable_selection(
                plan.standard_selections[index]
                if index < 27
                else plan.duplicate_selections[index - 27]
            )
            decoder_records, ledger_rows = _fixture_cell(
                model, index, selection, gallery
            )
            payload = {
                "schema_version": 1,
                "model": model,
                "subject": "sub-08",
                "seed": 42,
                "suite": suite,
                "scenario_index": index,
                "scenario": scenario,
                "matrix_shape": matrix_shape,
                "scenario_selection": selection,
                "scenario_manifest_sha256": plan.manifest_sha256,
                "source_artifact_sha256": source,
                "decoder_records": decoder_records,
            }
            (directory / "summary.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (directory / "per_query.csv").write_text(
                _csv_text(ledger_rows), encoding="utf-8"
            )


def test_loads_lightweight_real_task7_schema_and_rejects_partial_or_audit_input(
    tmp_path: Path,
) -> None:
    runs = tmp_path / "runs"
    _write_task7_fixture(runs)
    aggregate = aggregate_records(load_run_records(runs))
    assert len(aggregate.records) == 450
    assert aggregate.scenario_manifest_sha256 == hashlib.sha256(
        (runs / "scenario_manifest.json").read_bytes()
    ).hexdigest()

    summary_path = next(runs.rglob("summary.json"))
    original_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    tampered_summary = dict(original_summary)
    tampered_summary["scenario_selection"] = {}
    _write_canonical_json(summary_path, tampered_summary)
    with pytest.raises(ValueError, match="selection"):
        load_run_records(runs)
    _write_canonical_json(summary_path, original_summary)

    missing = next(runs.rglob("per_query.csv"))
    original_ledger = missing.read_bytes()
    missing.unlink()
    with pytest.raises(ValueError, match="partial"):
        load_run_records(runs)

    missing.write_bytes(original_ledger)
    (runs / "best_test_audit.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="audit"):
        load_run_records(runs)


def test_ledger_rejects_header_rows_indexes_ids_formulas_and_types(
    tmp_path: Path,
) -> None:
    runs = tmp_path / "runs"
    _write_task7_fixture(runs)
    ledger = next(runs.rglob("per_query.csv"))
    original = ledger.read_bytes()

    ledger.write_text("garbage\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ledger|header"):
        load_run_records(runs)
    ledger.write_bytes(original)

    rows = _read_csv_rows(ledger)
    header_tamper = original.replace(b"query_id", b"query_name", 1)
    ledger.write_bytes(header_tamper)
    with pytest.raises(ValueError, match="header"):
        load_run_records(runs)
    ledger.write_bytes(original)

    for mutate, message in (
        (lambda values: values.__setitem__(0, {**values[0], "query_index": "1"}), "index"),
        (lambda values: values.__setitem__(0, {**values[0], "query_id": "=CMD()"}), "formula"),
        (lambda values: values.__setitem__(0, {**values[0], "assigned_score": "NaN"}), "finite|score"),
        (lambda values: values.__setitem__(0, {**values[0], "answerable": "1"}), "boolean"),
        (lambda values: values.pop(), "row"),
        (lambda values: values.append(dict(values[-1])), "row|duplicate"),
    ):
        changed = [dict(row) for row in rows]
        mutate(changed)
        _write_csv_rows(ledger, changed)
        with pytest.raises(ValueError, match=message):
            load_run_records(runs)
        ledger.write_bytes(original)


def test_ledger_rejects_coordinated_impossible_summary_metrics(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _write_task7_fixture(runs)
    summary_path = next(runs.rglob("00_dropq0_dropg0_dropp0_dupg0/summary.json"))
    original = json.loads(summary_path.read_text(encoding="utf-8"))

    tampered = json.loads(json.dumps(original))
    independent = next(
        row for row in tampered["decoder_records"] if row["decoder"] == "independent"
    )
    for field in (
        "correct",
        "total",
        "answerable_correct",
        "answerable_total",
        "assigned_count",
        "unique_gallery_entry_predictions",
        "unique_canonical_predictions",
        "top5_count",
        "correct_to_correct",
    ):
        independent[field] = 270
    independent.update(
        {
            "top1": 100.0,
            "answerable_top1": 100.0,
            "top5": 100.0,
            "unanswerable_count": 0,
            "unmatched_count": 0,
            "correct_to_wrong": 0,
            "wrong_to_correct": 0,
            "wrong_to_wrong": 0,
        }
    )
    _write_canonical_json(summary_path, tampered)
    with pytest.raises(ValueError, match="ledger|row|matrix|total"):
        load_run_records(runs)

    tampered = json.loads(json.dumps(original))
    independent = next(
        row for row in tampered["decoder_records"] if row["decoder"] == "independent"
    )
    independent["top5_count"] = 0
    independent["top5"] = 0.0
    _write_canonical_json(summary_path, tampered)
    with pytest.raises(ValueError, match="Top-5|ledger"):
        load_run_records(runs)

    duplicate_path = next(runs.rglob("29_dupq20/summary.json"))
    duplicate = json.loads(duplicate_path.read_text(encoding="utf-8"))
    hungarian = next(
        row for row in duplicate["decoder_records"] if row["decoder"] == "hungarian"
    )
    hungarian.update(
        {
            "correct": 220,
            "top1": 100.0,
            "answerable_correct": 220,
            "answerable_top1": 100.0,
            "assigned_count": 220,
            "unmatched_count": 0,
            "unique_gallery_entry_predictions": 220,
            "unique_canonical_predictions": 200,
            "assignment_changes_from_independent": 0,
            "delta_correct_vs_independent": 0,
            "correct_to_correct": 220,
            "correct_to_wrong": 0,
            "wrong_to_correct": 0,
            "wrong_to_wrong": 0,
            "appended_b_correct": 20,
            "appended_b_top1": 100.0,
            "both_correct_count": 20,
            "both_correct": 100.0,
            "theoretical_ceiling_count": 220,
            "theoretical_ceiling": 100.0,
            "distance_from_ceiling": 0,
            "unmatched_repeated_queries": 0,
        }
    )
    hungarian["assignment_metadata"].update(
        {"matched_count": 220, "unmatched_count": 0, "assigned_sum_similarity": 220.0}
    )
    _write_canonical_json(duplicate_path, duplicate)
    with pytest.raises(ValueError, match="ceiling|ledger|unmatched"):
        load_run_records(runs)


def test_load_rejects_symlinked_runs(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "runs"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        load_run_records(link)


def _write_canonical_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_audit_inputs(root: Path) -> dict[tuple[str, str], str]:
    source_hashes = {
        (model, role): f"{value:064x}"
        for model_index, model in enumerate(MODELS)
        for role, value in (
            ("standard", model_index + 1),
            ("eeg_a", model_index + 11),
            ("eeg_b", model_index + 21),
        )
    }
    native: dict[str, tuple[dict[str, object], bytes]] = {}
    for model_index, model in enumerate(("nice", "atm_s")):
        source = {
            "url": OFFICIAL_SOURCE_URL,
            "branch": "develop",
            "commit": "f" * 40,
            "checkout_sha256": "a" * 64,
        }
        inputs = {
            "training_eeg": {
                "name": "preprocessed_eeg_training.npy",
                "sha256": "4" * 64,
            },
            "training_features": {
                "name": "ViT-H-14_features_train.pt",
                "sha256": "5" * 64,
            },
        }
        checkpoints = [
            {
                "epoch": 1,
                "val_loss": 0.2 + model_index,
                "checkpoint": "epoch_0001.pth",
                "sha256": "d" * 64,
            },
            {
                "epoch": 2,
                "val_loss": 0.1 + model_index,
                "checkpoint": "epoch_0002.pth",
                "sha256": "e" * 64,
            },
        ]
        manifest = {
            "schema_version": 1,
            "model": model,
            "encoder_type": "NICE" if model == "nice" else "ATMS",
            "subject": "sub-08",
            "seed": 42,
            "source": source,
            "inputs": inputs,
            "hyperparameters": {
                "epochs": 500,
                "batch_size": 1024,
                "learning_rate": 3e-4,
                "val_ratio": 0.1,
                "early_stopping_patience": 10,
                "ema_decay": 0.999,
                "logit_scale_type": "exp",
                "avg_trials": True,
                "n_chans": 63,
                "n_times": 250,
            },
            "encoder_behavior": {
                "use_subject_id": model == "atm_s",
                "normalize_feats": model == "atm_s",
            },
            "checkpoints": checkpoints,
            "selection": {
                "epoch": 2,
                "val_loss": 0.1 + model_index,
                "checkpoint": "epoch_0002.pth",
            },
            "best_checkpoint": {"name": "best_val.pth", "sha256": "e" * 64},
            "history": {"name": "history.csv", "sha256": "6" * 64},
            "stopped_early": False,
        }
        manifest_path = root / "checkpoints" / model / "checkpoint_manifest.json"
        _write_canonical_json(manifest_path, manifest)
        manifest_bytes = manifest_path.read_bytes()
        native[model] = (manifest, manifest_bytes)

        asset_files = {
            "Preprocessed_data_250Hz/sub-08/preprocessed_eeg_training.npy": {
                "bytes": 1,
                "sha256": inputs["training_eeg"]["sha256"],
            },
            "Preprocessed_data_250Hz/sub-08/preprocessed_eeg_test.npy": {
                "bytes": 1,
                "sha256": "7" * 64,
            },
            "ViT-H-14_features_train.pt": {
                "bytes": 1,
                "sha256": inputs["training_features"]["sha256"],
            },
            "ViT-H-14_features_test.pt": {
                "bytes": 1,
                "sha256": "8" * 64,
            },
        }
        ids = tuple(f"image-{index:03d}" for index in range(200))
        artifact = ScoreArtifact(
            similarity=np.eye(200, dtype=np.float32),
            query_ids=ids,
            gallery_entry_ids=ids,
            gallery_canonical_ids=ids,
            target_canonical_ids=ids,
            metadata={
                "model_slug": model,
                "trial_half": "standard",
                "checkpoint_role": "val_selected_formal",
                "checkpoint": "/not-rendered/best_val.pth",
                "checkpoint_sha256": "e" * 64,
                "checkpoint_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "source_lock": source,
                "asset_lock_manifest_sha256": "9" * 64,
                "asset_lock": {
                    "repo_id": "LidongYang/EEG_Image_decode",
                    "repo_type": "dataset",
                    "asset_root": "/not-rendered/assets",
                    "files": asset_files,
                },
                "input_sha256": {
                    "test_eeg": asset_files["Preprocessed_data_250Hz/sub-08/preprocessed_eeg_test.npy"]["sha256"],
                    "test_features": asset_files["ViT-H-14_features_test.pt"]["sha256"],
                    "trial_manifest": "c" * 64,
                },
                "trial_manifest_sha256": "c" * 64,
                "subject": "sub-08",
                "seed": 42,
                "logit_scale_type": "exp",
                "effective_logit_scale": 1.0,
                "query_embeddings_sha256": "b" * 64,
                "native_metrics": {
                    "top1_count": 200,
                    "top5_count": 200,
                    "sample_count": 200,
                },
            },
        )
        artifact_path = root / "matrices" / model / "standard"
        write_score_artifact(artifact_path, artifact)
        source_hashes[(model, "standard")] = _score_artifact_sha256(artifact_path)

    inventory = []
    for model in MODELS:
        for half, role in (("standard", "standard"), ("a", "eeg_a"), ("b", "eeg_b")):
            inventory.append(
                {
                    "model_slug": model,
                    "trial_half": half,
                    "path": str(root / "matrices" / model / role),
                    "sha256": source_hashes[(model, role)],
                }
            )
    for model_index, model in enumerate(("nice", "atm_s")):
        runs = [
            {
                "epoch": 1,
                "checkpoint": "/not-rendered/epoch_0001.pth",
                "checkpoint_sha256": "d" * 64,
                "effective_logit_scale": 1.0,
                "top1_count": 150 - model_index,
                "top5_count": 190,
                "sample_count": 200,
            },
            {
                "epoch": 2,
                "checkpoint": "/not-rendered/epoch_0002.pth",
                "checkpoint_sha256": "e" * 64,
                "effective_logit_scale": 1.0,
                "top1_count": 155 - model_index,
                "top5_count": 192,
                "sample_count": 200,
            },
        ]
        _write_canonical_json(
            root / "matrices" / model / "best_test_audit.json",
            {
                "schema_version": 1,
                "scope": "best_test_audit_only",
                "model_slug": model,
                "checkpoint_policy": "every_epoch_checkpoint",
                "fairness_artifact_created": False,
                "formal_artifact_inventory": inventory,
                "runs": runs,
                "best_test": runs[1],
            },
        )
    return source_hashes


def test_full_results_root_loads_audit_separately_and_publishes(tmp_path: Path) -> None:
    root = tmp_path / "matching_fairness_v3"
    root.mkdir()
    source_hashes = _write_audit_inputs(root)
    _write_task7_fixture(root / "runs", source_hashes)

    destination = aggregate_results(root)

    assert destination == root / "aggregate"
    summary = json.loads(
        (destination / "aggregate_summary.json").read_text(encoding="utf-8")
    )
    assert summary["record_count"] == 450
    assert summary["standard_record_count"] == 405
    assert summary["duplicate_eeg_record_count"] == 45
    assert [row["formal_epoch"] for row in summary["reproduction_audit"]] == [2, 2]
    assert all("scope" not in row for row in summary["reproduction_audit"])
    assert "/not-rendered/" not in (destination / "RESULTS.md").read_text(
        encoding="utf-8"
    )
    assert "f" * 40 in (destination / "RESULTS.md").read_text(encoding="utf-8")


def test_audit_checkpoint_hashes_must_bind_training_manifest(tmp_path: Path) -> None:
    root = tmp_path / "matching_fairness_v3"
    root.mkdir()
    source_hashes = _write_audit_inputs(root)
    _write_task7_fixture(root / "runs", source_hashes)
    aggregate = aggregate_records(load_run_records(root / "runs"))
    audit_path = root / "matrices" / "nice" / "best_test_audit.json"
    original = json.loads(audit_path.read_text(encoding="utf-8"))
    mutations = (
        lambda audit: audit.update({"unexpected": "field"}),
        lambda audit: audit["runs"][0].update({"unexpected": "field"}),
        lambda audit: audit["runs"][0].update(
            {"checkpoint_sha256": "0" * 64}
        ),
    )
    for mutation in mutations:
        audit = json.loads(json.dumps(original))
        mutation(audit)
        _write_canonical_json(audit_path, audit)
        with pytest.raises(ValueError, match="audit|checkpoint"):
            load_reproduction_audits(root, aggregate)


def test_audit_rejects_task5_manifest_provenance_tampering(
    tmp_path: Path,
) -> None:
    root = tmp_path / "matching_fairness_v3"
    root.mkdir()
    source_hashes = _write_audit_inputs(root)
    _write_task7_fixture(root / "runs", source_hashes)
    aggregate = aggregate_records(load_run_records(root / "runs"))
    manifest_path = root / "checkpoints" / "nice" / "checkpoint_manifest.json"
    original = json.loads(manifest_path.read_text(encoding="utf-8"))
    wrong_source = json.loads(json.dumps(original))
    wrong_source["source"]["url"] = "https://github.com/eeyhsong/NICE-EEG.git"
    with pytest.raises(
        ValueError,
        match="source|official|provenance",
    ):
        _validate_checkpoint_manifest(wrong_source, "nice")
    mutations = (
        lambda manifest: manifest["source"].update(
            {"url": "https://github.com/eeyhsong/NICE-EEG.git"}
        ),
        lambda manifest: manifest["source"].update(
            {"commit": "0" * 40, "checkout_sha256": "1" * 64}
        ),
        lambda manifest: manifest["inputs"]["training_eeg"].update(
            {"sha256": "2" * 64}
        ),
        lambda manifest: manifest["hyperparameters"].update({"batch_size": 512}),
        lambda manifest: manifest["encoder_behavior"].update(
            {"normalize_feats": True}
        ),
        lambda manifest: manifest["best_checkpoint"].update(
            {"sha256": "3" * 64}
        ),
        lambda manifest: manifest["history"].update({"sha256": "4" * 64}),
        lambda manifest: manifest.update({"unexpected": "field"}),
    )
    for mutation in mutations:
        manifest = json.loads(json.dumps(original))
        mutation(manifest)
        _write_canonical_json(manifest_path, manifest)
        with pytest.raises(
            ValueError,
            match="checkpoint|manifest|provenance|artifact",
        ):
            load_reproduction_audits(root, aggregate)


def test_audit_rejects_self_consistent_checkpoint_and_audit_tampering(
    tmp_path: Path,
) -> None:
    root = tmp_path / "matching_fairness_v3"
    root.mkdir()
    source_hashes = _write_audit_inputs(root)
    _write_task7_fixture(root / "runs", source_hashes)
    aggregate = aggregate_records(load_run_records(root / "runs"))

    manifest_path = root / "checkpoints" / "nice" / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checkpoints"][1]["sha256"] = "0" * 64
    manifest["best_checkpoint"]["sha256"] = "0" * 64
    _write_canonical_json(manifest_path, manifest)

    audit_path = root / "matrices" / "nice" / "best_test_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["runs"][1]["checkpoint_sha256"] = "0" * 64
    audit["best_test"]["checkpoint_sha256"] = "0" * 64
    _write_canonical_json(audit_path, audit)

    with pytest.raises(ValueError, match="checkpoint|manifest|artifact"):
        load_reproduction_audits(root, aggregate)


def test_audit_requires_the_native_standard_artifact(tmp_path: Path) -> None:
    root = tmp_path / "matching_fairness_v3"
    root.mkdir()
    source_hashes = _write_audit_inputs(root)
    _write_task7_fixture(root / "runs", source_hashes)
    aggregate = aggregate_records(load_run_records(root / "runs"))
    (root / "matrices" / "nice" / "standard" / "metadata.json").unlink()

    with pytest.raises(ValueError, match="artifact|formal input"):
        load_reproduction_audits(root, aggregate)


def test_audit_recomputes_native_metrics_from_the_standard_artifact(
    tmp_path: Path,
) -> None:
    root = tmp_path / "matching_fairness_v3"
    root.mkdir()
    source_hashes = _write_audit_inputs(root)
    metadata_path = root / "matrices" / "nice" / "standard" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["native_metrics"]["top1_count"] = 199
    metadata_path.write_text(
        json.dumps(
            metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        + "\n",
        encoding="utf-8",
    )
    new_hash = _score_artifact_sha256(metadata_path.parent)
    source_hashes[("nice", "standard")] = new_hash
    for model in ("nice", "atm_s"):
        audit_path = root / "matrices" / model / "best_test_audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        inventory_entry = next(
            entry
            for entry in audit["formal_artifact_inventory"]
            if entry["model_slug"] == "nice" and entry["trial_half"] == "standard"
        )
        inventory_entry["sha256"] = new_hash
        _write_canonical_json(audit_path, audit)
    _write_task7_fixture(root / "runs", source_hashes)
    aggregate = aggregate_records(load_run_records(root / "runs"))

    with pytest.raises(ValueError, match="metric|artifact"):
        load_reproduction_audits(root, aggregate)


def test_cli_help_exposes_results_root_only() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "experiments/matching_fairness/scripts/aggregate_results.py",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--results-root" in completed.stdout
    assert "--subject" not in completed.stdout
    assert "--seed" not in completed.stdout
