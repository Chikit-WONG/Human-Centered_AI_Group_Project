from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from matching_fairness.reporting import (
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
    record: dict[str, object] = {
        "model": model,
        "subject": "sub-08",
        "seed": 42,
        "suite": suite,
        "scenario_index": scenario_index,
        "scenario": scenario,
        "scenario_manifest_sha256": MANIFEST_HASH,
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
        "assignment_metadata": {},
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


def _write_task7_fixture(root: Path) -> None:
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
            source = (
                {"standard": f"{MODELS.index(model) + 1:064x}"}
                if suite == "standard"
                else {
                    "eeg_a": f"{MODELS.index(model) + 11:064x}",
                    "eeg_b": f"{MODELS.index(model) + 21:064x}",
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
            payload = {
                "schema_version": 1,
                "model": model,
                "subject": "sub-08",
                "seed": 42,
                "suite": suite,
                "scenario_index": index,
                "scenario": scenario,
                "matrix_shape": matrix_shape,
                "scenario_selection": runner._jsonable_selection(
                    plan.standard_selections[index]
                    if index < 27
                    else plan.duplicate_selections[index - 27]
                ),
                "scenario_manifest_sha256": plan.manifest_sha256,
                "source_artifact_sha256": source,
                "decoder_records": [_record(model, index, decoder) for decoder in DECODERS],
            }
            for record in payload["decoder_records"]:
                record.pop("scenario_manifest_sha256")
            (directory / "summary.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (directory / "per_query.csv").write_text(
                "query_index,correct_top1\n0,1\n",
                encoding="utf-8",
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
    missing.unlink()
    with pytest.raises(ValueError, match="partial"):
        load_run_records(runs)

    missing.write_text("query_index,correct_top1\n0,1\n", encoding="utf-8")
    (runs / "best_test_audit.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="audit"):
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


def _write_audit_inputs(root: Path) -> None:
    inventory = []
    for model_index, model in enumerate(MODELS):
        for half, value in (("standard", model_index + 1), ("a", model_index + 11), ("b", model_index + 21)):
            inventory.append(
                {
                    "model_slug": model,
                    "trial_half": half,
                    "path": f"/not-rendered/{model}/{half}",
                    "sha256": f"{value:064x}",
                }
            )
    for model_index, model in enumerate(("nice", "atm_s")):
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
        _write_canonical_json(
            root / "checkpoints" / model / "checkpoint_manifest.json",
            {
                "schema_version": 1,
                "model": model,
                "subject": "sub-08",
                "seed": 42,
                "source": {"commit": "f" * 40},
                "checkpoints": checkpoints,
                "selection": {
                    "epoch": 2,
                    "val_loss": 0.1 + model_index,
                    "checkpoint": "epoch_0002.pth",
                },
            },
        )
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


def test_full_results_root_loads_audit_separately_and_publishes(tmp_path: Path) -> None:
    root = tmp_path / "matching_fairness_v3"
    root.mkdir()
    _write_task7_fixture(root / "runs")
    _write_audit_inputs(root)

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
    _write_task7_fixture(root / "runs")
    _write_audit_inputs(root)
    aggregate = aggregate_records(load_run_records(root / "runs"))
    audit_path = root / "matrices" / "nice" / "best_test_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["runs"][0]["checkpoint_sha256"] = "0" * 64
    _write_canonical_json(audit_path, audit)

    with pytest.raises(ValueError, match="checkpoint"):
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
