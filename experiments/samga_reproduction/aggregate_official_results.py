#!/usr/bin/env python3
"""Aggregate released-SAMGA reproduction results without mutating run data.

The input contract is::

    <variant>/seed<seed>/sub-XX-b<B>-p<P>/<run-name>/result.csv

The CSV's ``top1 acc`` and ``top5 acc`` fields are the online metrics at the
actual final/stopping epoch.  Its ``best *`` fields are test-set-selected
diagnostics, not leakage-controlled validation estimates.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


PAPER_TOP1_PERCENT = 91.3
PAPER_TOP5_PERCENT = 98.8
RESULT_FIELDS = (
    "top1 acc",
    "top5 acc",
    "best top1 acc",
    "best top5 acc",
    "best test loss",
    "best epoch",
)
OUTPUT_FILENAMES = (
    "official_results.csv",
    "official_results.json",
    "official_results.md",
    "official_results_zh.md",
)
CSV_OUTPUT_FIELDS = (
    "variant",
    "seed",
    "subject",
    "batch_size",
    "early_stop_patience",
    "final_stopping_epoch_top1_percent",
    "final_stopping_epoch_top5_percent",
    "test_selected_top1_percent",
    "test_selected_top5_percent",
    "test_loss_at_selected_epoch",
    "test_selected_epoch",
    "result_path",
)

_VARIANT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SEED_RE = re.compile(r"^seed(0|[1-9][0-9]*)$")
_CELL_RE = re.compile(
    r"^sub-([0-9]{2})-b([1-9][0-9]*)-p(0|[1-9][0-9]*)$"
)
_RUN_DIR_RE = re.compile(
    r"^[0-9]{8}-[0-9]{6}(?:-[A-Za-z0-9][A-Za-z0-9._-]*)?$"
)


class AggregationError(ValueError):
    """Raised when an input or output would make aggregation unsafe."""


@dataclass(frozen=True)
class CellKey:
    variant: str
    seed: int
    subject: int
    batch_size: int
    early_stop_patience: int


@dataclass(frozen=True, order=True)
class ProtocolKey:
    variant: str
    batch_size: int
    early_stop_patience: int


@dataclass(frozen=True)
class ResultRow:
    variant: str
    seed: int
    subject: int
    batch_size: int
    early_stop_patience: int
    final_stopping_epoch_top1_percent: float
    final_stopping_epoch_top5_percent: float
    test_selected_top1_percent: float
    test_selected_top5_percent: float
    test_loss_at_selected_epoch: float
    test_selected_epoch: int
    result_path: str

    @property
    def cell_key(self) -> CellKey:
        return CellKey(
            variant=self.variant,
            seed=self.seed,
            subject=self.subject,
            batch_size=self.batch_size,
            early_stop_patience=self.early_stop_patience,
        )

    @property
    def protocol_key(self) -> ProtocolKey:
        return ProtocolKey(
            variant=self.variant,
            batch_size=self.batch_size,
            early_stop_patience=self.early_stop_patience,
        )


@dataclass(frozen=True)
class MissingCell:
    variant: str
    seed: int
    subject: int
    batch_size: int
    early_stop_patience: int
    reason: str
    cell_path: str

    @property
    def cell_key(self) -> CellKey:
        return CellKey(
            variant=self.variant,
            seed=self.seed,
            subject=self.subject,
            batch_size=self.batch_size,
            early_stop_patience=self.early_stop_patience,
        )


@dataclass(frozen=True)
class ScanResult:
    input_root: Path
    rows: tuple[ResultRow, ...]
    missing_cells: tuple[MissingCell, ...]
    protocol_keys: tuple[ProtocolKey, ...]
    expected_subjects: tuple[int, ...] | None
    expected_seeds: tuple[int, ...] | None

    @property
    def is_complete(self) -> bool:
        return bool(self.protocol_keys) and not self.missing_cells


@dataclass(frozen=True)
class GroupSummary:
    variant: str
    batch_size: int
    early_stop_patience: int
    n: int
    final_stopping_epoch_top1_mean_percent: float
    final_stopping_epoch_top1_sample_sd_points: float
    final_stopping_epoch_top1_gap_to_paper_points: float
    final_stopping_epoch_top5_mean_percent: float
    final_stopping_epoch_top5_sample_sd_points: float
    final_stopping_epoch_top5_gap_to_paper_points: float
    test_selected_top1_mean_percent: float
    test_selected_top1_sample_sd_points: float
    test_selected_top1_gap_to_paper_points: float
    test_selected_top5_mean_percent: float
    test_selected_top5_sample_sd_points: float
    test_selected_top5_gap_to_paper_points: float
    test_loss_at_selected_epoch_mean: float
    test_loss_at_selected_epoch_sample_sd: float
    test_selected_epoch_mean: float
    test_selected_epoch_sample_sd: float


def _cell_sort_key(key: CellKey) -> tuple[object, ...]:
    return (
        key.variant,
        key.batch_size,
        key.early_stop_patience,
        key.seed,
        key.subject,
    )


def _row_sort_key(row: ResultRow) -> tuple[object, ...]:
    return _cell_sort_key(row.cell_key)


def _missing_sort_key(cell: MissingCell) -> tuple[object, ...]:
    return (*_cell_sort_key(cell.cell_key), cell.reason)


def _cell_path(key: CellKey) -> str:
    return (
        f"{key.variant}/seed{key.seed}/"
        f"sub-{key.subject:02d}-b{key.batch_size}"
        f"-p{key.early_stop_patience}"
    )


def _validate_result_location(
    input_root: Path,
    path: Path,
    relative: Path,
) -> None:
    current = input_root
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise AggregationError(
                f"{relative.as_posix()} contains a symlink component"
            )
    try:
        resolved_root = input_root.resolve(strict=True)
        resolved_path = path.resolve(strict=True)
    except OSError as exc:
        raise AggregationError(f"could not resolve {path}: {exc}") from exc
    if not resolved_path.is_relative_to(resolved_root):
        raise AggregationError(
            f"{relative.as_posix()} resolves outside input root"
        )


def _parse_result_path(input_root: Path, path: Path) -> CellKey:
    try:
        relative = path.relative_to(input_root)
    except ValueError as exc:
        raise AggregationError(f"{path} is outside input root {input_root}") from exc
    _validate_result_location(input_root, path, relative)
    parts = relative.parts
    if len(parts) != 5 or parts[-1] != "result.csv":
        raise AggregationError(
            f"{relative.as_posix()} does not match the exact official-run layout "
            "<variant>/seed<seed>/sub-XX-b<B>-p<P>/<timestamp-run>/result.csv"
        )
    variant, seed_part, cell_part, run_part, _ = parts
    seed_match = _SEED_RE.fullmatch(seed_part)
    cell_match = _CELL_RE.fullmatch(cell_part)
    if (
        not _VARIANT_RE.fullmatch(variant)
        or seed_match is None
        or cell_match is None
        or not _RUN_DIR_RE.fullmatch(run_part)
    ):
        raise AggregationError(
            f"{relative.as_posix()} does not match the exact official-run layout "
            "<variant>/seed<seed>/sub-XX-b<B>-p<P>/<timestamp-run>/result.csv"
        )
    subject = int(cell_match.group(1))
    if subject < 1:
        raise AggregationError(
            f"{relative.as_posix()} has subject {subject}; subject must be 1..99"
        )
    return CellKey(
        variant=variant,
        seed=int(seed_match.group(1)),
        subject=subject,
        batch_size=int(cell_match.group(2)),
        early_stop_patience=int(cell_match.group(3)),
    )


def _discover_cell_directories(input_root: Path) -> dict[CellKey, Path]:
    discovered: dict[CellKey, Path] = {}
    for variant_dir in sorted(input_root.iterdir(), key=lambda path: path.name):
        if not _VARIANT_RE.fullmatch(variant_dir.name):
            continue
        if variant_dir.is_symlink():
            raise AggregationError(
                f"official-run structure contains symlink: {variant_dir}"
            )
        if not variant_dir.is_dir():
            continue
        for seed_dir in sorted(variant_dir.iterdir(), key=lambda path: path.name):
            seed_match = _SEED_RE.fullmatch(seed_dir.name)
            if seed_match is None:
                continue
            if seed_dir.is_symlink():
                raise AggregationError(
                    f"official-run structure contains symlink: {seed_dir}"
                )
            if not seed_dir.is_dir():
                continue
            for cell_dir in sorted(seed_dir.iterdir(), key=lambda path: path.name):
                cell_match = _CELL_RE.fullmatch(cell_dir.name)
                if cell_match is None:
                    continue
                if cell_dir.is_symlink():
                    raise AggregationError(
                        f"official-run structure contains symlink: {cell_dir}"
                    )
                if not cell_dir.is_dir():
                    continue
                subject = int(cell_match.group(1))
                if subject < 1:
                    continue
                key = CellKey(
                    variant=variant_dir.name,
                    seed=int(seed_match.group(1)),
                    subject=subject,
                    batch_size=int(cell_match.group(2)),
                    early_stop_patience=int(cell_match.group(3)),
                )
                discovered[key] = cell_dir
                for run_dir in cell_dir.iterdir():
                    if (
                        _RUN_DIR_RE.fullmatch(run_dir.name)
                        and run_dir.is_symlink()
                    ):
                        raise AggregationError(
                            f"official-run structure contains symlink: {run_dir}"
                        )
    return discovered


def _parse_finite_float(raw: str, field: str, path: Path) -> float:
    try:
        value = float(raw.strip())
    except ValueError as exc:
        raise AggregationError(
            f"{path}: field {field!r} must be numeric, got {raw!r}"
        ) from exc
    if not math.isfinite(value):
        raise AggregationError(
            f"{path}: field {field!r} must be finite, got {raw!r}"
        )
    return value


def _parse_percentage(raw: str, field: str, path: Path) -> float:
    value = _parse_finite_float(raw, field, path)
    if not 0.0 <= value <= 100.0:
        raise AggregationError(
            f"{path}: field {field!r} must be in [0, 100], got {value}"
        )
    return value


def _read_result_csv(input_root: Path, path: Path, key: CellKey) -> ResultRow:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle, strict=True))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise AggregationError(f"could not read {path}: {exc}") from exc
    if not rows:
        raise AggregationError(f"{path}: empty result.csv")

    header = tuple(rows[0])
    duplicates = sorted(
        {field for field in header if header.count(field) > 1}
    )
    if duplicates:
        raise AggregationError(
            f"{path}: duplicate header fields: {', '.join(duplicates)}"
        )
    missing = sorted(set(RESULT_FIELDS) - set(header))
    if missing:
        raise AggregationError(
            f"{path}: missing fields: {', '.join(missing)}"
        )
    unexpected = sorted(set(header) - set(RESULT_FIELDS))
    if unexpected:
        raise AggregationError(
            f"{path}: unexpected fields: {', '.join(unexpected)}"
        )
    if header != RESULT_FIELDS:
        raise AggregationError(
            f"{path}: header must exactly match the official field order: "
            + ", ".join(RESULT_FIELDS)
        )
    data_rows = rows[1:]
    if len(data_rows) != 1:
        raise AggregationError(
            f"{path}: expected exactly one data row; found {len(data_rows)}"
        )
    if len(data_rows[0]) != len(header):
        raise AggregationError(
            f"{path}: data row has {len(data_rows[0])} columns, "
            f"expected {len(header)}"
        )
    values = dict(zip(header, data_rows[0]))

    final_top1 = _parse_percentage(values["top1 acc"], "top1 acc", path)
    final_top5 = _parse_percentage(values["top5 acc"], "top5 acc", path)
    selected_top1 = _parse_percentage(
        values["best top1 acc"], "best top1 acc", path
    )
    selected_top5 = _parse_percentage(
        values["best top5 acc"], "best top5 acc", path
    )
    selected_loss = _parse_finite_float(
        values["best test loss"], "best test loss", path
    )
    if selected_loss < 0.0:
        raise AggregationError(
            f"{path}: field 'best test loss' must be non-negative, "
            f"got {selected_loss}"
        )
    selected_epoch_value = _parse_finite_float(
        values["best epoch"], "best epoch", path
    )
    if not selected_epoch_value.is_integer():
        raise AggregationError(
            f"{path}: field 'best epoch' must be an integer, "
            f"got {selected_epoch_value}"
        )
    selected_epoch = int(selected_epoch_value)
    if selected_epoch < 1:
        raise AggregationError(
            f"{path}: field 'best epoch' must be at least 1, "
            f"got {selected_epoch}"
        )
    if final_top5 < final_top1:
        raise AggregationError(
            f"{path}: final/stopping Top-5 ({final_top5}) cannot be lower "
            f"than Top-1 ({final_top1})"
        )
    if selected_top5 < selected_top1:
        raise AggregationError(
            f"{path}: test-selected Top-5 ({selected_top5}) cannot be lower "
            f"than Top-1 ({selected_top1})"
        )

    return ResultRow(
        variant=key.variant,
        seed=key.seed,
        subject=key.subject,
        batch_size=key.batch_size,
        early_stop_patience=key.early_stop_patience,
        final_stopping_epoch_top1_percent=final_top1,
        final_stopping_epoch_top5_percent=final_top5,
        test_selected_top1_percent=selected_top1,
        test_selected_top5_percent=selected_top5,
        test_loss_at_selected_epoch=selected_loss,
        test_selected_epoch=selected_epoch,
        result_path=path.relative_to(input_root).as_posix(),
    )


def _normalise_expected(
    values: Sequence[int] | None,
    *,
    name: str,
    minimum: int,
    maximum: int | None = None,
) -> tuple[int, ...] | None:
    if values is None:
        return None
    if not values:
        raise AggregationError(f"{name} cannot be empty")
    normalised: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise AggregationError(f"{name} must contain integers")
        if value < minimum or (maximum is not None and value > maximum):
            allowed = (
                f"{minimum}..{maximum}"
                if maximum is not None
                else f">= {minimum}"
            )
            raise AggregationError(
                f"{name} value {value} is outside the allowed range {allowed}"
            )
        normalised.append(value)
    if len(set(normalised)) != len(normalised):
        raise AggregationError(f"{name} contains duplicate values")
    return tuple(sorted(normalised))


def scan_official_runs(
    input_root: Path,
    expected_subjects: Sequence[int] | None = None,
    expected_seeds: Sequence[int] | None = None,
) -> ScanResult:
    """Scan and strictly validate completed cells without writing anything."""

    input_root = Path(input_root)
    if not input_root.exists():
        raise AggregationError(f"input root does not exist: {input_root}")
    if not input_root.is_dir():
        raise AggregationError(f"input root is not a directory: {input_root}")
    subjects = _normalise_expected(
        expected_subjects,
        name="expected subjects",
        minimum=1,
        maximum=99,
    )
    seeds = _normalise_expected(
        expected_seeds,
        name="expected seeds",
        minimum=0,
    )

    discovered = _discover_cell_directories(input_root)
    paths_by_cell: dict[CellKey, list[Path]] = {}
    result_paths = sorted(
        (
            path
            for path in input_root.rglob("result.csv")
            if path.is_file() or path.is_symlink()
        ),
        key=lambda path: path.relative_to(input_root).as_posix(),
    )
    for path in result_paths:
        key = _parse_result_path(input_root, path)
        paths_by_cell.setdefault(key, []).append(path)
        discovered.setdefault(key, path.parents[1])

    outside_expected = [
        key
        for key in sorted(discovered, key=_cell_sort_key)
        if (
            subjects is not None
            and key.subject not in subjects
        )
        or (
            seeds is not None
            and key.seed not in seeds
        )
    ]
    if outside_expected:
        raise AggregationError(
            "observed cells outside expected matrix: "
            + ", ".join(_cell_path(key) for key in outside_expected)
        )

    for key in sorted(paths_by_cell, key=_cell_sort_key):
        paths = paths_by_cell[key]
        if len(paths) > 1:
            relative_paths = ", ".join(
                path.relative_to(input_root).as_posix() for path in paths
            )
            raise AggregationError(
                f"cell {_cell_path(key)} has multiple result.csv files: "
                f"{relative_paths}"
            )

    rows = tuple(
        sorted(
            (
                _read_result_csv(input_root, paths[0], key)
                for key, paths in paths_by_cell.items()
            ),
            key=_row_sort_key,
        )
    )
    completed_keys = set(paths_by_cell)
    protocols = tuple(
        sorted(
            {
                ProtocolKey(
                    key.variant,
                    key.batch_size,
                    key.early_stop_patience,
                )
                for key in discovered
            }
        )
    )

    missing_by_key: dict[CellKey, MissingCell] = {}
    for key, cell_dir in discovered.items():
        if key in completed_keys:
            continue
        missing_by_key[key] = MissingCell(
            variant=key.variant,
            seed=key.seed,
            subject=key.subject,
            batch_size=key.batch_size,
            early_stop_patience=key.early_stop_patience,
            reason="no_result",
            cell_path=cell_dir.relative_to(input_root).as_posix(),
        )

    if subjects is not None or seeds is not None:
        for protocol in protocols:
            observed = [
                key
                for key in discovered
                if ProtocolKey(
                    key.variant,
                    key.batch_size,
                    key.early_stop_patience,
                )
                == protocol
            ]
            group_subjects = subjects or tuple(
                sorted({key.subject for key in observed})
            )
            group_seeds = seeds or tuple(sorted({key.seed for key in observed}))
            for seed in group_seeds:
                for subject in group_subjects:
                    key = CellKey(
                        variant=protocol.variant,
                        seed=seed,
                        subject=subject,
                        batch_size=protocol.batch_size,
                        early_stop_patience=protocol.early_stop_patience,
                    )
                    if key in completed_keys or key in missing_by_key:
                        continue
                    missing_by_key[key] = MissingCell(
                        variant=key.variant,
                        seed=key.seed,
                        subject=key.subject,
                        batch_size=key.batch_size,
                        early_stop_patience=key.early_stop_patience,
                        reason="expected_cell_absent",
                        cell_path=_cell_path(key),
                    )

    return ScanResult(
        input_root=input_root,
        rows=rows,
        missing_cells=tuple(
            sorted(missing_by_key.values(), key=_missing_sort_key)
        ),
        protocol_keys=protocols,
        expected_subjects=subjects,
        expected_seeds=seeds,
    )


def _mean_and_sample_sd(values: Sequence[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    sample_sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, sample_sd


def build_group_summaries(
    rows: Sequence[ResultRow],
) -> tuple[GroupSummary, ...]:
    """Aggregate equal-weight completed cells by variant/batch/patience."""

    grouped: dict[ProtocolKey, list[ResultRow]] = {}
    for row in rows:
        grouped.setdefault(row.protocol_key, []).append(row)

    summaries: list[GroupSummary] = []
    for key in sorted(grouped):
        group = sorted(grouped[key], key=_row_sort_key)
        final_top1_mean, final_top1_sd = _mean_and_sample_sd(
            [row.final_stopping_epoch_top1_percent for row in group]
        )
        final_top5_mean, final_top5_sd = _mean_and_sample_sd(
            [row.final_stopping_epoch_top5_percent for row in group]
        )
        selected_top1_mean, selected_top1_sd = _mean_and_sample_sd(
            [row.test_selected_top1_percent for row in group]
        )
        selected_top5_mean, selected_top5_sd = _mean_and_sample_sd(
            [row.test_selected_top5_percent for row in group]
        )
        selected_loss_mean, selected_loss_sd = _mean_and_sample_sd(
            [row.test_loss_at_selected_epoch for row in group]
        )
        selected_epoch_mean, selected_epoch_sd = _mean_and_sample_sd(
            [float(row.test_selected_epoch) for row in group]
        )
        summaries.append(
            GroupSummary(
                variant=key.variant,
                batch_size=key.batch_size,
                early_stop_patience=key.early_stop_patience,
                n=len(group),
                final_stopping_epoch_top1_mean_percent=final_top1_mean,
                final_stopping_epoch_top1_sample_sd_points=final_top1_sd,
                final_stopping_epoch_top1_gap_to_paper_points=(
                    final_top1_mean - PAPER_TOP1_PERCENT
                ),
                final_stopping_epoch_top5_mean_percent=final_top5_mean,
                final_stopping_epoch_top5_sample_sd_points=final_top5_sd,
                final_stopping_epoch_top5_gap_to_paper_points=(
                    final_top5_mean - PAPER_TOP5_PERCENT
                ),
                test_selected_top1_mean_percent=selected_top1_mean,
                test_selected_top1_sample_sd_points=selected_top1_sd,
                test_selected_top1_gap_to_paper_points=(
                    selected_top1_mean - PAPER_TOP1_PERCENT
                ),
                test_selected_top5_mean_percent=selected_top5_mean,
                test_selected_top5_sample_sd_points=selected_top5_sd,
                test_selected_top5_gap_to_paper_points=(
                    selected_top5_mean - PAPER_TOP5_PERCENT
                ),
                test_loss_at_selected_epoch_mean=selected_loss_mean,
                test_loss_at_selected_epoch_sample_sd=selected_loss_sd,
                test_selected_epoch_mean=selected_epoch_mean,
                test_selected_epoch_sample_sd=selected_epoch_sd,
            )
        )
    return tuple(summaries)


def build_protocol_completeness(
    scan: ScanResult,
) -> tuple[dict[str, object], ...]:
    """Report completeness separately for every inferred protocol group."""

    statuses: list[dict[str, object]] = []
    for protocol in scan.protocol_keys:
        completed_count = sum(
            row.protocol_key == protocol
            for row in scan.rows
        )
        missing_count = sum(
            ProtocolKey(
                cell.variant,
                cell.batch_size,
                cell.early_stop_patience,
            )
            == protocol
            for cell in scan.missing_cells
        )
        expected_count = completed_count + missing_count
        statuses.append(
            {
                "variant": protocol.variant,
                "batch_size": protocol.batch_size,
                "early_stop_patience": protocol.early_stop_patience,
                "completed_cell_count": completed_count,
                "missing_cell_count": missing_count,
                "expected_cell_count": expected_count,
                "is_complete": expected_count > 0 and missing_count == 0,
            }
        )
    return tuple(statuses)


def _row_dict(row: ResultRow) -> dict[str, object]:
    return {field: getattr(row, field) for field in CSV_OUTPUT_FIELDS}


def _render_csv(rows: Sequence[ResultRow]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=CSV_OUTPUT_FIELDS,
        lineterminator="\n",
    )
    writer.writeheader()
    for row in sorted(rows, key=_row_sort_key):
        values = _row_dict(row)
        writer.writerow(
            {
                field: (
                    format(value, ".12g")
                    if isinstance(value, float)
                    else value
                )
                for field, value in values.items()
            }
        )
    return output.getvalue()


def _json_payload(
    scan: ScanResult,
    summaries: Sequence[GroupSummary],
    protocol_status: Sequence[dict[str, object]],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "paper_reference_percent": {
            "top1": PAPER_TOP1_PERCENT,
            "top5": PAPER_TOP5_PERCENT,
        },
        "expectations": {
            "subjects": (
                list(scan.expected_subjects)
                if scan.expected_subjects is not None
                else None
            ),
            "seeds": (
                list(scan.expected_seeds)
                if scan.expected_seeds is not None
                else None
            ),
        },
        "metric_semantics": {
            "final_stopping_epoch": {
                "source_fields": ["top1 acc", "top5 acc"],
                "en": (
                    "Online test metrics at the actual final/stopping epoch; "
                    "they are fixed-epoch-60 only when the run completed all "
                    "60 epochs."
                ),
                "zh": (
                    "实际最终/停止轮次的在线测试指标；仅当运行完整完成 60 "
                    "轮时才是固定第 60 轮指标。"
                ),
            },
            "test_selected": {
                "source_fields": [
                    "best top1 acc",
                    "best top5 acc",
                    "best test loss",
                    "best epoch",
                ],
                "en": (
                    "Per-epoch test-selected diagnostic with test leakage. "
                    "Top-1 selects the epoch and lower test loss breaks exact "
                    "Top-1 ties; Top-5 and loss are companion values at that "
                    "epoch, not independently optimized extrema."
                ),
                "zh": (
                    "逐轮使用测试集选模的诊断值，存在测试泄漏。Top-1 决定"
                    "轮次，Top-1 精确同分时以更低测试 loss 破同分；Top-5 "
                    "和 loss 是该轮伴随值，并非各自独立最优值。"
                ),
            },
            "retrieval_protocol": {
                "en": (
                    "Standard independent per-query retrieval, not Hungarian "
                    "assignment."
                ),
                "zh": "标准的逐查询独立检索，不是匈牙利匹配。",
            },
            "standard_deviation": {
                "en": (
                    "Sample standard deviation (ddof=1); reported as 0 for "
                    "singleton groups."
                ),
                "zh": "样本标准差（ddof=1）；单样本协议组记为 0。",
            },
        },
        "completeness": {
            "is_complete": scan.is_complete,
            "has_discoverable_protocol": bool(scan.protocol_keys),
            "completed_cell_count": len(scan.rows),
            "missing_cell_count": len(scan.missing_cells),
            "missing_cells": [
                asdict(cell)
                for cell in sorted(
                    scan.missing_cells,
                    key=_missing_sort_key,
                )
            ],
        },
        "rows": [
            _row_dict(row) for row in sorted(scan.rows, key=_row_sort_key)
        ],
        "protocol_completeness": [
            dict(item) for item in protocol_status
        ],
        "protocol_groups": [asdict(summary) for summary in summaries],
    }


def _render_json(
    scan: ScanResult,
    summaries: Sequence[GroupSummary],
    protocol_status: Sequence[dict[str, object]],
) -> str:
    return (
        json.dumps(
            _json_payload(scan, summaries, protocol_status),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def _md_escape(value: object) -> str:
    return str(value).replace("|", r"\|")


def _format_metric(value: float) -> str:
    return f"{value:.2f}"


def _format_loss(value: float) -> str:
    return f"{value:.4f}"


def _render_english(
    scan: ScanResult,
    summaries: Sequence[GroupSummary],
    protocol_status: Sequence[dict[str, object]],
) -> str:
    lines = [
        "# Official SAMGA reproduction aggregation",
        "",
        "## Interpretation guardrails",
        "",
        "- **Final/stopping-epoch metrics:** CSV `top1 acc` and `top5 acc` "
        "are online test metrics at the actual final/stopping epoch. They "
        "are fixed-epoch-60 metrics only when all 60 epochs completed.",
        "- **Per-epoch test-selected metrics (test leakage):** CSV `best "
        "top1 acc` selects an epoch after inspecting the test set every "
        "epoch. Lower test loss only breaks exact Top-1 ties.",
        "- `test_selected_top5_percent` and "
        "`test_loss_at_selected_epoch` are companion values at the "
        "Top-1-selected epoch; they are not independently optimized.",
        "- Retrieval is standard independent per-query retrieval, not "
        "Hungarian assignment.",
        "- Each completed subject × seed cell is one equal-weight "
        "observation. SD is sample SD (ddof=1), or 0 for a singleton.",
        "",
        "Paper reference: Top-1 91.30%, Top-5 98.80%. Gaps below are signed "
        "aggregate-minus-paper percentage points.",
        "",
        "## Protocol-group summary",
        "",
    ]
    if summaries:
        lines.extend(
            [
                "| Variant | Batch | Patience | n | Final Top-1 mean ± SD "
                "| Gap | Final Top-5 mean ± SD | Gap | Test-selected Top-1 "
                "mean ± SD | Gap | Test-selected Top-5 mean ± SD | Gap |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for summary in summaries:
            lines.append(
                f"| {_md_escape(summary.variant)} "
                f"| {summary.batch_size} "
                f"| {summary.early_stop_patience} "
                f"| {summary.n} "
                f"| {_format_metric(summary.final_stopping_epoch_top1_mean_percent)} "
                f"± {_format_metric(summary.final_stopping_epoch_top1_sample_sd_points)} "
                f"| {summary.final_stopping_epoch_top1_gap_to_paper_points:+.2f} "
                f"| {_format_metric(summary.final_stopping_epoch_top5_mean_percent)} "
                f"± {_format_metric(summary.final_stopping_epoch_top5_sample_sd_points)} "
                f"| {summary.final_stopping_epoch_top5_gap_to_paper_points:+.2f} "
                f"| {_format_metric(summary.test_selected_top1_mean_percent)} "
                f"± {_format_metric(summary.test_selected_top1_sample_sd_points)} "
                f"| {summary.test_selected_top1_gap_to_paper_points:+.2f} "
                f"| {_format_metric(summary.test_selected_top5_mean_percent)} "
                f"± {_format_metric(summary.test_selected_top5_sample_sd_points)} "
                f"| {summary.test_selected_top5_gap_to_paper_points:+.2f} |"
            )
    else:
        lines.append("No completed protocol group was found.")

    lines.extend(["", "## Completed cells", ""])
    if scan.rows:
        lines.extend(
            [
                "| Variant | Seed | Subject | Batch | Patience | Final Top-1 "
                "| Final Top-5 | Test-selected Top-1 | Test-selected Top-5 "
                "| Test loss at selected epoch | Selected epoch | Result path |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in scan.rows:
            lines.append(
                f"| {_md_escape(row.variant)} "
                f"| {row.seed} | {row.subject} | {row.batch_size} "
                f"| {row.early_stop_patience} "
                f"| {_format_metric(row.final_stopping_epoch_top1_percent)} "
                f"| {_format_metric(row.final_stopping_epoch_top5_percent)} "
                f"| {_format_metric(row.test_selected_top1_percent)} "
                f"| {_format_metric(row.test_selected_top5_percent)} "
                f"| {_format_loss(row.test_loss_at_selected_epoch)} "
                f"| {row.test_selected_epoch} "
                f"| `{_md_escape(row.result_path)}` |"
            )
    else:
        lines.append("No completed result.csv was found.")

    lines.extend(["", "## Completeness", ""])
    state = "complete" if scan.is_complete else "incomplete"
    lines.append(
        f"Scan state: **{state}**; {len(scan.rows)} completed cell(s), "
        f"{len(scan.missing_cells)} missing cell(s)."
    )
    lines.append("")
    lines.extend(
        [
            "### Protocol matrix status",
            "",
            "| Variant | Batch | Patience | Completed | Expected | Missing | Complete |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for item in protocol_status:
        complete = "yes" if item["is_complete"] else "no"
        lines.append(
            f"| {_md_escape(item['variant'])} "
            f"| {item['batch_size']} "
            f"| {item['early_stop_patience']} "
            f"| {item['completed_cell_count']} "
            f"| {item['expected_cell_count']} "
            f"| {item['missing_cell_count']} "
            f"| {complete} |"
        )
    lines.append("")
    if scan.missing_cells:
        lines.extend(
            [
                "| Variant | Seed | Subject | Batch | Patience | Reason | Cell path |",
                "|---|---:|---:|---:|---:|---|---|",
            ]
        )
        for cell in scan.missing_cells:
            lines.append(
                f"| {_md_escape(cell.variant)} | {cell.seed} "
                f"| {cell.subject} | {cell.batch_size} "
                f"| {cell.early_stop_patience} | `{cell.reason}` "
                f"| `{_md_escape(cell.cell_path)}` |"
            )
    else:
        lines.append("No missing cell was detected.")
    return "\n".join(lines) + "\n"


def _render_chinese(
    scan: ScanResult,
    summaries: Sequence[GroupSummary],
    protocol_status: Sequence[dict[str, object]],
) -> str:
    lines = [
        "# SAMGA 官方实现复现结果聚合",
        "",
        "## 解读边界",
        "",
        "- **最终/停止轮次指标：** CSV 的 `top1 acc` 与 `top5 acc` 是实际"
        "最终/停止轮次的在线测试指标；仅当完整跑完 60 轮时才可称为固定第 "
        "60 轮指标。",
        "- **逐轮测试集选模（测试泄漏）：** CSV 的 `best top1 acc` 来自"
        "每轮查看测试集后的轮次选择；仅当 Top-1 精确同分时才以更低测试 "
        "loss 破同分。",
        "- `test_selected_top5_percent` 和 "
        "`test_loss_at_selected_epoch` 是 Top-1 规则所选轮次的伴随值，"
        "并非各自独立优化所得。",
        "- 检索协议是标准的逐查询独立检索，不是匈牙利匹配。",
        "- 每个已完成的被试 × seed cell 等权计为一个观测。SD 是样本标准差"
        "（ddof=1），单样本组记为 0。",
        "",
        "论文参考值：Top-1 91.30%，Top-5 98.80%。下表差距为“聚合值减论文值”"
        "的有符号百分点。",
        "",
        "## 协议组汇总",
        "",
    ]
    if summaries:
        lines.extend(
            [
                "| 变体 | Batch | Patience | n | 最终 Top-1 均值 ± SD "
                "| 差距 | 最终 Top-5 均值 ± SD | 差距 | 测试集选模 Top-1 "
                "均值 ± SD | 差距 | 测试集选模 Top-5 均值 ± SD | 差距 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for summary in summaries:
            lines.append(
                f"| {_md_escape(summary.variant)} "
                f"| {summary.batch_size} "
                f"| {summary.early_stop_patience} "
                f"| {summary.n} "
                f"| {_format_metric(summary.final_stopping_epoch_top1_mean_percent)} "
                f"± {_format_metric(summary.final_stopping_epoch_top1_sample_sd_points)} "
                f"| {summary.final_stopping_epoch_top1_gap_to_paper_points:+.2f} "
                f"| {_format_metric(summary.final_stopping_epoch_top5_mean_percent)} "
                f"± {_format_metric(summary.final_stopping_epoch_top5_sample_sd_points)} "
                f"| {summary.final_stopping_epoch_top5_gap_to_paper_points:+.2f} "
                f"| {_format_metric(summary.test_selected_top1_mean_percent)} "
                f"± {_format_metric(summary.test_selected_top1_sample_sd_points)} "
                f"| {summary.test_selected_top1_gap_to_paper_points:+.2f} "
                f"| {_format_metric(summary.test_selected_top5_mean_percent)} "
                f"± {_format_metric(summary.test_selected_top5_sample_sd_points)} "
                f"| {summary.test_selected_top5_gap_to_paper_points:+.2f} |"
            )
    else:
        lines.append("未发现已完成的协议组。")

    lines.extend(["", "## 已完成 cells", ""])
    if scan.rows:
        lines.extend(
            [
                "| 变体 | Seed | 被试 | Batch | Patience | 最终 Top-1 "
                "| 最终 Top-5 | 测试集选模 Top-1 | 测试集选模 Top-5 "
                "| 所选轮测试 loss | 所选轮次 | 结果路径 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in scan.rows:
            lines.append(
                f"| {_md_escape(row.variant)} "
                f"| {row.seed} | {row.subject} | {row.batch_size} "
                f"| {row.early_stop_patience} "
                f"| {_format_metric(row.final_stopping_epoch_top1_percent)} "
                f"| {_format_metric(row.final_stopping_epoch_top5_percent)} "
                f"| {_format_metric(row.test_selected_top1_percent)} "
                f"| {_format_metric(row.test_selected_top5_percent)} "
                f"| {_format_loss(row.test_loss_at_selected_epoch)} "
                f"| {row.test_selected_epoch} "
                f"| `{_md_escape(row.result_path)}` |"
            )
    else:
        lines.append("未发现已完成的 result.csv。")

    lines.extend(["", "## 完整性", ""])
    state = "完整" if scan.is_complete else "不完整"
    lines.append(
        f"扫描状态：**{state}**；已完成 {len(scan.rows)} 个 cell，"
        f"缺失 {len(scan.missing_cells)} 个 cell。"
    )
    lines.append("")
    lines.extend(
        [
            "### 各协议矩阵状态",
            "",
            "| 变体 | Batch | Patience | 已完成 | 期望 | 缺失 | 完整 |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for item in protocol_status:
        complete = "是" if item["is_complete"] else "否"
        lines.append(
            f"| {_md_escape(item['variant'])} "
            f"| {item['batch_size']} "
            f"| {item['early_stop_patience']} "
            f"| {item['completed_cell_count']} "
            f"| {item['expected_cell_count']} "
            f"| {item['missing_cell_count']} "
            f"| {complete} |"
        )
    lines.append("")
    if scan.missing_cells:
        lines.extend(
            [
                "| 变体 | Seed | 被试 | Batch | Patience | 原因 | Cell 路径 |",
                "|---|---:|---:|---:|---:|---|---|",
            ]
        )
        for cell in scan.missing_cells:
            reason = (
                "已有 cell 目录但无 result.csv"
                if cell.reason == "no_result"
                else "期望矩阵中的 cell 不存在"
            )
            lines.append(
                f"| {_md_escape(cell.variant)} | {cell.seed} "
                f"| {cell.subject} | {cell.batch_size} "
                f"| {cell.early_stop_patience} | {reason} "
                f"| `{_md_escape(cell.cell_path)}` |"
            )
    else:
        lines.append("未检测到缺失 cell。")
    return "\n".join(lines) + "\n"


def _paths_overlap(first: Path, second: Path) -> bool:
    first_resolved = first.resolve()
    second_resolved = second.resolve()
    return (
        first_resolved == second_resolved
        or first_resolved.is_relative_to(second_resolved)
        or second_resolved.is_relative_to(first_resolved)
    )


def _write_report_file(path: Path, content: str) -> None:
    created = False
    try:
        with path.open("x", encoding="utf-8", newline="") as handle:
            created = True
            handle.write(content)
    except Exception:
        if created:
            try:
                path.unlink()
            except OSError:
                pass
        raise


def write_reports(scan: ScanResult, output_dir: Path) -> tuple[Path, ...]:
    """Write four deterministic reports after all validation has succeeded."""

    output_dir = Path(output_dir)
    if _paths_overlap(scan.input_root, output_dir):
        raise AggregationError(
            "output directory must be separate from the official-runs input tree"
        )
    if output_dir.exists() and not output_dir.is_dir():
        raise AggregationError(f"output path is not a directory: {output_dir}")
    targets = tuple(output_dir / filename for filename in OUTPUT_FILENAMES)
    existing = [target for target in targets if target.exists()]
    if existing:
        raise AggregationError(
            "refusing to overwrite output that already exists: "
            + ", ".join(str(path) for path in existing)
        )

    summaries = build_group_summaries(scan.rows)
    protocol_status = build_protocol_completeness(scan)
    rendered = {
        "official_results.csv": _render_csv(scan.rows),
        "official_results.json": _render_json(scan, summaries, protocol_status),
        "official_results.md": _render_english(scan, summaries, protocol_status),
        "official_results_zh.md": _render_chinese(scan, summaries, protocol_status),
    }
    created_output_dir = False
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        if not output_dir.is_dir():
            raise AggregationError(f"output path is not a directory: {output_dir}")
    else:
        created_output_dir = True
    created_targets: list[Path] = []
    try:
        for target in targets:
            _write_report_file(target, rendered[target.name])
            created_targets.append(target)
    except Exception as exc:
        cleanup_errors: list[str] = []
        for target in reversed(created_targets):
            try:
                target.unlink()
            except FileNotFoundError:
                pass
            except OSError as cleanup_exc:
                cleanup_errors.append(f"{target}: {cleanup_exc}")
        if created_output_dir:
            try:
                output_dir.rmdir()
            except OSError as cleanup_exc:
                cleanup_errors.append(f"{output_dir}: {cleanup_exc}")
        if isinstance(exc, FileExistsError):
            raise AggregationError(
                "refusing to overwrite an output created concurrently"
            ) from exc
        detail = str(exc)
        if cleanup_errors:
            detail += "; rollback incomplete: " + "; ".join(cleanup_errors)
        raise AggregationError(
            f"could not write report bundle: {detail}"
        ) from exc
    return targets


def _parse_integer_set(
    raw: str,
    *,
    option_name: str,
    minimum: int,
    maximum: int | None = None,
) -> tuple[int, ...]:
    values: list[int] = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            raise argparse.ArgumentTypeError(
                f"{option_name} contains an empty item"
            )
        range_match = re.fullmatch(r"([0-9]+)-([0-9]+)", token)
        if range_match:
            start = int(range_match.group(1))
            stop = int(range_match.group(2))
            if stop < start:
                raise argparse.ArgumentTypeError(
                    f"{option_name} range {token!r} is descending"
                )
            if stop - start > 100_000:
                raise argparse.ArgumentTypeError(
                    f"{option_name} range {token!r} is too large"
                )
            values.extend(range(start, stop + 1))
        elif re.fullmatch(r"[0-9]+", token):
            values.append(int(token))
        else:
            raise argparse.ArgumentTypeError(
                f"{option_name} item {token!r} is not an integer or range"
            )
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(
            f"{option_name} contains duplicate values"
        )
    for value in values:
        if value < minimum or (maximum is not None and value > maximum):
            allowed = (
                f"{minimum}..{maximum}"
                if maximum is not None
                else f">= {minimum}"
            )
            raise argparse.ArgumentTypeError(
                f"{option_name} value {value} is outside {allowed}"
            )
    return tuple(sorted(values))


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Strictly aggregate official SAMGA reproduction result.csv cells "
            "without modifying the run tree."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=(
            project_root
            / "artifacts"
            / "samga_reproduction"
            / "official_runs"
        ),
        help="official_runs root (default: project artifacts tree)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "results" / "samga_reproduction",
        help="new report directory (default: project results tree)",
    )
    parser.add_argument(
        "--expected-subjects",
        type=lambda value: _parse_integer_set(
            value,
            option_name="--expected-subjects",
            minimum=1,
            maximum=99,
        ),
        help="comma/range list, for example 1-10 or 1,5,8",
    )
    parser.add_argument(
        "--expected-seeds",
        type=lambda value: _parse_integer_set(
            value,
            option_name="--expected-seeds",
            minimum=0,
        ),
        help="comma/range list, for example 2025-2029 or 42,43",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        scan = scan_official_runs(
            args.input_root,
            expected_subjects=args.expected_subjects,
            expected_seeds=args.expected_seeds,
        )
        targets = write_reports(scan, args.output_dir)
    except AggregationError as exc:
        parser.error(str(exc))
    print(
        f"Wrote {len(targets)} reports for {len(scan.rows)} completed cell(s); "
        f"{len(scan.missing_cells)} missing cell(s).",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
