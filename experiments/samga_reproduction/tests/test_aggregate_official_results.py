from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import pytest


REPRODUCTION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRODUCTION_ROOT))

import aggregate_official_results as aggregation  # noqa: E402
from aggregate_official_results import (  # noqa: E402
    AggregationError,
    OUTPUT_FILENAMES,
    build_group_summaries,
    main,
    scan_official_runs,
    write_reports,
)


RESULT_FIELDS = (
    "top1 acc",
    "top5 acc",
    "best top1 acc",
    "best top5 acc",
    "best test loss",
    "best epoch",
)
DEFAULT_VALUES = {
    "top1 acc": "75.00",
    "top5 acc": "96.50",
    "best top1 acc": "79.50",
    "best top5 acc": "96.00",
    "best test loss": "0.7053",
    "best epoch": "49",
}


def write_result(
    root: Path,
    *,
    variant: str = "raw",
    seed: int = 2025,
    subject: int = 1,
    batch_size: int = 512,
    patience: int = 0,
    timestamp: str = "20260718-120000-run",
    values: dict[str, str] | None = None,
    fields: tuple[str, ...] = RESULT_FIELDS,
    data_rows: list[dict[str, str]] | None = None,
) -> Path:
    path = (
        root
        / variant
        / f"seed{seed}"
        / f"sub-{subject:02d}-b{batch_size}-p{patience}"
        / timestamp
        / "result.csv"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {**DEFAULT_VALUES, **(values or {})}
    rows = data_rows if data_rows is not None else [row]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(fields)
        for item in rows:
            writer.writerow([item.get(field, "") for field in fields])
    return path


def tree_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_scan_parses_cell_and_lists_incomplete_expected_matrix(
    tmp_path: Path,
) -> None:
    runs = tmp_path / "official_runs"
    result_path = write_result(runs)
    (runs / "raw" / "seed2025" / "sub-02-b512-p0").mkdir(
        parents=True
    )

    scan = scan_official_runs(
        runs,
        expected_subjects=(1, 2),
        expected_seeds=(2025, 2026),
    )

    assert len(scan.rows) == 1
    row = scan.rows[0]
    assert row.variant == "raw"
    assert row.seed == 2025
    assert row.subject == 1
    assert row.batch_size == 512
    assert row.early_stop_patience == 0
    assert row.final_stopping_epoch_top1_percent == 75.0
    assert row.final_stopping_epoch_top5_percent == 96.5
    assert row.test_selected_top1_percent == 79.5
    assert row.test_selected_top5_percent == 96.0
    assert row.test_loss_at_selected_epoch == 0.7053
    assert row.test_selected_epoch == 49
    assert row.result_path == result_path.relative_to(runs).as_posix()
    assert not scan.is_complete
    assert {
        (
            missing.seed,
            missing.subject,
            missing.reason,
        )
        for missing in scan.missing_cells
    } == {
        (2025, 2, "no_result"),
        (2026, 1, "expected_cell_absent"),
        (2026, 2, "expected_cell_absent"),
    }


def test_scan_rejects_multiple_results_for_one_logical_cell(
    tmp_path: Path,
) -> None:
    runs = tmp_path / "official_runs"
    write_result(runs, timestamp="20260718-120000-first")
    write_result(runs, timestamp="20260718-130000-second")

    with pytest.raises(AggregationError, match=r"multiple result\.csv"):
        scan_official_runs(runs)


def test_scan_rejects_missing_duplicate_and_unexpected_headers(
    tmp_path: Path,
) -> None:
    cases = (
        (
            RESULT_FIELDS[:-1],
            "missing fields",
        ),
        (
            (*RESULT_FIELDS[:-1], "top1 acc"),
            "duplicate header",
        ),
        (
            (*RESULT_FIELDS, "extra metric"),
            "unexpected fields",
        ),
    )
    for index, (fields, message) in enumerate(cases):
        runs = tmp_path / f"case-{index}"
        write_result(runs, fields=fields)
        with pytest.raises(AggregationError, match=message):
            scan_official_runs(runs)


def test_scan_rejects_duplicate_or_multiple_data_rows(tmp_path: Path) -> None:
    runs = tmp_path / "official_runs"
    write_result(
        runs,
        data_rows=[DEFAULT_VALUES.copy(), DEFAULT_VALUES.copy()],
    )

    with pytest.raises(
        AggregationError,
        match="expected exactly one data row",
    ):
        scan_official_runs(runs)


@pytest.mark.parametrize(
    ("values", "message"),
    (
        ({"top1 acc": "not-a-number"}, "numeric"),
        ({"top1 acc": "nan"}, "finite"),
        ({"top1 acc": "-0.1"}, r"\[0, 100\]"),
        ({"top5 acc": "100.1"}, r"\[0, 100\]"),
        ({"best test loss": "-0.1"}, "non-negative"),
        ({"best epoch": "2.5"}, "integer"),
        ({"best epoch": "0"}, "at least 1"),
        (
            {"top1 acc": "80", "top5 acc": "79"},
            "final/stopping Top-5",
        ),
        (
            {"best top1 acc": "80", "best top5 acc": "79"},
            "test-selected Top-5",
        ),
    ),
)
def test_scan_rejects_non_numeric_non_finite_and_out_of_range_values(
    tmp_path: Path,
    values: dict[str, str],
    message: str,
) -> None:
    runs = tmp_path / "official_runs"
    write_result(runs, values=values)

    with pytest.raises(AggregationError, match=message):
        scan_official_runs(runs)


def test_scan_rejects_result_outside_exact_layout(tmp_path: Path) -> None:
    runs = tmp_path / "official_runs"
    malformed = runs / "raw" / "not-a-seed" / "result.csv"
    malformed.parent.mkdir(parents=True)
    malformed.write_text(
        ",".join(RESULT_FIELDS)
        + "\n"
        + ",".join(DEFAULT_VALUES[field] for field in RESULT_FIELDS)
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(AggregationError, match="exact official-run layout"):
        scan_official_runs(runs)


def test_scan_rejects_result_symlink_that_escapes_input_root(
    tmp_path: Path,
) -> None:
    outside_result = write_result(tmp_path / "outside")
    runs = tmp_path / "official_runs"
    linked_result = (
        runs
        / "raw"
        / "seed2025"
        / "sub-01-b512-p0"
        / "20260718-120000-run"
        / "result.csv"
    )
    linked_result.parent.mkdir(parents=True)
    linked_result.symlink_to(outside_result)

    with pytest.raises(AggregationError, match="symlink|outside input root"):
        scan_official_runs(runs)


def test_scan_rejects_symlinked_structural_directory(tmp_path: Path) -> None:
    source_result = write_result(tmp_path / "outside")
    source_cell = source_result.parents[1]
    runs = tmp_path / "official_runs"
    linked_cell = runs / "raw" / "seed2025" / "sub-01-b512-p0"
    linked_cell.parent.mkdir(parents=True)
    linked_cell.symlink_to(source_cell, target_is_directory=True)

    with pytest.raises(AggregationError, match="symlink"):
        scan_official_runs(runs)


@pytest.mark.parametrize(
    ("observed", "expectations"),
    (
        ({"subject": 3}, {"expected_subjects": (1, 2)}),
        ({"seed": 2027}, {"expected_seeds": (2025, 2026)}),
    ),
)
def test_expected_axes_reject_observed_cells_outside_exact_matrix(
    tmp_path: Path,
    observed: dict[str, int],
    expectations: dict[str, tuple[int, ...]],
) -> None:
    runs = tmp_path / "official_runs"
    write_result(runs, **observed)

    with pytest.raises(AggregationError, match="outside expected matrix"):
        scan_official_runs(runs, **expectations)


def test_scan_requires_timestamp_prefixed_run_directory(
    tmp_path: Path,
) -> None:
    runs = tmp_path / "official_runs"
    write_result(runs, timestamp="not-a-timestamp")

    with pytest.raises(AggregationError, match="exact official-run layout"):
        scan_official_runs(runs)


def test_scan_rejects_blank_csv_record_instead_of_discarding_it(
    tmp_path: Path,
) -> None:
    runs = tmp_path / "official_runs"
    result = write_result(runs)
    with result.open("a", encoding="utf-8", newline="") as handle:
        handle.write("\n")

    with pytest.raises(
        AggregationError,
        match="expected exactly one data row",
    ):
        scan_official_runs(runs)


def test_statistics_keep_batch_protocols_separate_and_use_sample_sd(
    tmp_path: Path,
) -> None:
    runs = tmp_path / "official_runs"
    write_result(
        runs,
        subject=2,
        values={
            "top1 acc": "90",
            "top5 acc": "100",
            "best top1 acc": "94",
            "best top5 acc": "99",
            "best test loss": "0.4",
            "best epoch": "20",
        },
    )
    write_result(
        runs,
        subject=1,
        values={
            "top1 acc": "80",
            "top5 acc": "90",
            "best top1 acc": "82",
            "best top5 acc": "92",
            "best test loss": "0.8",
            "best epoch": "10",
        },
    )
    write_result(
        runs,
        subject=1,
        batch_size=1024,
        values={
            "top1 acc": "70",
            "top5 acc": "95",
            "best top1 acc": "75",
            "best top5 acc": "96",
            "best test loss": "1.0",
            "best epoch": "5",
        },
    )

    summaries = build_group_summaries(scan_official_runs(runs).rows)

    assert [(item.batch_size, item.n) for item in summaries] == [
        (512, 2),
        (1024, 1),
    ]
    group = summaries[0]
    assert group.final_stopping_epoch_top1_mean_percent == 85.0
    assert group.final_stopping_epoch_top1_sample_sd_points == pytest.approx(
        math.sqrt(50.0)
    )
    assert group.final_stopping_epoch_top1_gap_to_paper_points == pytest.approx(
        -6.3
    )
    assert group.final_stopping_epoch_top5_mean_percent == 95.0
    assert group.final_stopping_epoch_top5_gap_to_paper_points == pytest.approx(
        -3.8
    )
    assert group.test_selected_top1_mean_percent == 88.0
    assert group.test_selected_top1_gap_to_paper_points == pytest.approx(-3.3)
    assert group.test_selected_top5_mean_percent == 95.5
    assert group.test_selected_top5_gap_to_paper_points == pytest.approx(-3.3)
    assert group.test_loss_at_selected_epoch_mean == pytest.approx(0.6)
    assert group.test_loss_at_selected_epoch_sample_sd == pytest.approx(
        math.sqrt(0.08)
    )
    assert group.test_selected_epoch_mean == 15.0
    assert group.test_selected_epoch_sample_sd == pytest.approx(math.sqrt(50))
    assert summaries[1].final_stopping_epoch_top1_sample_sd_points == 0.0


def test_reports_are_byte_deterministic_and_disclose_metric_semantics(
    tmp_path: Path,
) -> None:
    runs = tmp_path / "official_runs"
    write_result(runs, variant="z_variant", subject=2)
    write_result(runs, variant="a_variant", subject=1)
    scan = scan_official_runs(
        runs,
        expected_subjects=(1, 2),
        expected_seeds=(2025,),
    )
    before = tree_snapshot(runs)

    first_output = tmp_path / "output-a"
    second_output = tmp_path / "output-b"
    write_reports(scan, first_output)
    write_reports(scan, second_output)

    for filename in OUTPUT_FILENAMES:
        assert (first_output / filename).read_bytes() == (
            second_output / filename
        ).read_bytes()
    assert tree_snapshot(runs) == before

    with (first_output / "official_results.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        csv_rows = list(csv.DictReader(handle))
    assert [row["variant"] for row in csv_rows] == [
        "a_variant",
        "z_variant",
    ]
    assert "final_stopping_epoch_top1_percent" in csv_rows[0]
    assert "test_selected_top1_percent" in csv_rows[0]

    payload = json.loads(
        (first_output / "official_results.json").read_text(encoding="utf-8")
    )
    assert payload["schema_version"] == 1
    assert payload["paper_reference_percent"] == {
        "top1": 91.3,
        "top5": 98.8,
    }
    assert payload["completeness"]["is_complete"] is False
    assert payload["completeness"]["missing_cell_count"] == 2
    assert len(payload["protocol_groups"]) == 2
    assert "test leakage" in payload["metric_semantics"]["test_selected"]["en"]
    assert (
        "not Hungarian assignment"
        in payload["metric_semantics"]["retrieval_protocol"]["en"]
    )

    english = (first_output / "official_results.md").read_text(
        encoding="utf-8"
    )
    chinese = (first_output / "official_results_zh.md").read_text(
        encoding="utf-8"
    )
    assert "Final/stopping-epoch metrics" in english
    assert "Per-epoch test-selected metrics (test leakage)" in english
    assert "not Hungarian assignment" in english
    assert "companion values at the Top-1-selected epoch" in english
    assert "最终/停止轮次指标" in chinese
    assert "逐轮测试集选模（测试泄漏）" in chinese
    assert "不是匈牙利匹配" in chinese
    assert "Top-1 规则所选轮次的伴随值" in chinese


def test_reports_show_per_protocol_matrix_completeness(
    tmp_path: Path,
) -> None:
    runs = tmp_path / "official_runs"
    write_result(runs, variant="main", subject=1)
    write_result(runs, variant="main", subject=2)
    write_result(runs, variant="exploratory", subject=1)
    scan = scan_official_runs(
        runs,
        expected_subjects=(1, 2),
        expected_seeds=(2025,),
    )
    output = tmp_path / "output"

    write_reports(scan, output)

    payload = json.loads(
        (output / "official_results.json").read_text(encoding="utf-8")
    )
    assert payload["completeness"]["is_complete"] is False
    by_variant = {
        item["variant"]: item
        for item in payload["protocol_completeness"]
    }
    assert by_variant["main"] == {
        "variant": "main",
        "batch_size": 512,
        "early_stop_patience": 0,
        "completed_cell_count": 2,
        "missing_cell_count": 0,
        "expected_cell_count": 2,
        "is_complete": True,
    }
    assert by_variant["exploratory"]["completed_cell_count"] == 1
    assert by_variant["exploratory"]["missing_cell_count"] == 1
    assert by_variant["exploratory"]["is_complete"] is False
    english = (output / "official_results.md").read_text(encoding="utf-8")
    chinese = (output / "official_results_zh.md").read_text(encoding="utf-8")
    assert "| main | 512 | 0 | 2 | 2 | 0 | yes |" in english
    assert "| exploratory | 512 | 0 | 1 | 2 | 1 | no |" in english
    assert "| main | 512 | 0 | 2 | 2 | 0 | 是 |" in chinese
    assert "| exploratory | 512 | 0 | 1 | 2 | 1 | 否 |" in chinese


def test_reports_refuse_to_overwrite_any_existing_target(
    tmp_path: Path,
) -> None:
    runs = tmp_path / "official_runs"
    write_result(runs)
    scan = scan_official_runs(runs)
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "official_results.csv"
    sentinel.write_text("keep me\n", encoding="utf-8")

    with pytest.raises(AggregationError, match="already exists"):
        write_reports(scan, output)

    assert sentinel.read_text(encoding="utf-8") == "keep me\n"
    assert sorted(path.name for path in output.iterdir()) == [
        "official_results.csv"
    ]


def test_reports_roll_back_bundle_when_a_later_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = tmp_path / "official_runs"
    write_result(runs)
    scan = scan_official_runs(runs)
    output = tmp_path / "output"
    real_write = getattr(aggregation, "_write_report_file", None)
    calls = 0

    def fail_second_write(path: Path, content: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected write failure")
        if real_write is None:
            raise AssertionError("production write helper is missing")
        real_write(path, content)

    monkeypatch.setattr(
        aggregation,
        "_write_report_file",
        fail_second_write,
        raising=False,
    )

    with pytest.raises(
        AggregationError,
        match="could not write report bundle",
    ):
        write_reports(scan, output)

    assert not output.exists() or list(output.iterdir()) == []


def test_reports_do_not_remove_concurrently_created_output_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = tmp_path / "official_runs"
    write_result(runs)
    scan = scan_official_runs(runs)
    output = tmp_path / "output"
    real_mkdir = Path.mkdir
    injected = False

    def concurrent_mkdir(
        path: Path,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal injected
        if path == output and not injected:
            injected = True
            real_mkdir(path, parents=True, exist_ok=True)
            if kwargs.get("exist_ok") is False:
                raise FileExistsError(path)
            return
        real_mkdir(path, *args, **kwargs)

    def fail_write(path: Path, content: str) -> None:
        raise OSError("injected write failure")

    monkeypatch.setattr(Path, "mkdir", concurrent_mkdir)
    monkeypatch.setattr(
        aggregation,
        "_write_report_file",
        fail_write,
    )

    with pytest.raises(
        AggregationError,
        match="could not write report bundle",
    ):
        write_reports(scan, output)

    assert output.is_dir()
    assert list(output.iterdir()) == []


def test_cli_accepts_ranges_and_custom_output_directory(
    tmp_path: Path,
) -> None:
    runs = tmp_path / "official_runs"
    write_result(runs, subject=1, seed=2025)
    output = tmp_path / "custom-results"

    return_code = main(
        [
            "--input-root",
            str(runs),
            "--output-dir",
            str(output),
            "--expected-subjects",
            "1-2",
            "--expected-seeds",
            "2025-2026",
        ]
    )

    assert return_code == 0
    assert sorted(path.name for path in output.iterdir()) == sorted(
        OUTPUT_FILENAMES
    )
    payload = json.loads(
        (output / "official_results.json").read_text(encoding="utf-8")
    )
    assert payload["expectations"] == {
        "subjects": [1, 2],
        "seeds": [2025, 2026],
    }
    assert payload["completeness"]["missing_cell_count"] == 3


def test_scan_rejects_missing_root(tmp_path: Path) -> None:
    with pytest.raises(AggregationError, match="does not exist"):
        scan_official_runs(tmp_path / "missing")
