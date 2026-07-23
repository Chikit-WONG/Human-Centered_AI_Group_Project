"""Strict, deterministic aggregation for the sealed matching-fairness run."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile

from .artifacts import (
    ScoreArtifact,
    independent_ranks,
    publish_staged_directory,
    read_score_artifact,
)
from .provenance import OFFICIAL_SOURCE_URL, sha256_file
from .scenarios import ScenarioSpec, build_standard_manifest, standard_scenarios
from .trial_splits import select_duplicate_image_ids


MODEL_ORDER = ("nice", "atm_s", "our_project")
SUITE_ORDER = ("standard", "duplicate_eeg")
DECODER_ORDER = (
    "independent",
    "greedy",
    "hungarian",
    "stable_matching",
    "sinkhorn",
)
STRICT_DECODERS = frozenset({"greedy", "hungarian", "stable_matching"})
EXPECTED_RECORDS = 3 * 30 * 5
_RUN_MANIFEST_ALGORITHM = "AIAA3800-MATCHING-FAIRNESS-RUN-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PER_QUERY_FIELDS = (
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
_AUDIT_FIELDS = frozenset(
    {"scope", "best_test", "best_test_audit_only", "checkpoint_policy"}
)
_DUPLICATE_FIELDS = frozenset(
    {
        "base_a_correct",
        "base_a_total",
        "base_a_top1",
        "appended_b_correct",
        "appended_b_total",
        "appended_b_top1",
        "repeated_canonical_total",
        "at_least_one_correct_count",
        "at_least_one_coverage",
        "both_correct_count",
        "both_correct",
        "theoretical_ceiling_count",
        "theoretical_ceiling",
        "distance_from_ceiling",
        "unmatched_repeated_queries",
    }
)
_CSV_FIELDS = (
    "model",
    "subject",
    "seed",
    "suite",
    "scenario_index",
    "scenario",
    "scenario_manifest_sha256",
    "source_artifact_sha256",
    "decoder",
    "correct",
    "total",
    "top1",
    "answerable_correct",
    "answerable_total",
    "answerable_top1",
    "unanswerable_count",
    "assigned_count",
    "unmatched_count",
    "unique_gallery_entry_predictions",
    "unique_canonical_predictions",
    "strict_one_to_one",
    "top5_count",
    "top5",
    "assignment_changes_from_independent",
    "delta_correct_vs_independent",
    "assignment_metadata",
    "correct_to_correct",
    "correct_to_wrong",
    "wrong_to_correct",
    "wrong_to_wrong",
    "base_a_correct",
    "base_a_total",
    "base_a_top1",
    "appended_b_correct",
    "appended_b_total",
    "appended_b_top1",
    "repeated_canonical_total",
    "at_least_one_correct_count",
    "at_least_one_coverage",
    "both_correct_count",
    "both_correct",
    "theoretical_ceiling_count",
    "theoretical_ceiling",
    "distance_from_ceiling",
    "unmatched_repeated_queries",
)
_BASE_FIELDS = frozenset(
    {
        "model",
        "subject",
        "seed",
        "suite",
        "scenario_index",
        "scenario",
        "scenario_manifest_sha256",
        "source_artifact_sha256",
        "decoder",
        "correct",
        "total",
        "top1",
        "answerable_correct",
        "answerable_total",
        "answerable_top1",
        "unanswerable_count",
        "assigned_count",
        "unmatched_count",
        "unique_gallery_entry_predictions",
        "unique_canonical_predictions",
        "strict_one_to_one",
        "top5_count",
        "top5",
        "assignment_changes_from_independent",
        "delta_correct_vs_independent",
        "assignment_metadata",
        "correct_to_correct",
        "correct_to_wrong",
        "wrong_to_correct",
        "wrong_to_wrong",
    }
)


@dataclass(frozen=True)
class AggregateBundle:
    """Validated records plus their single immutable run identity."""

    records: tuple[Mapping[str, object], ...]
    scenario_manifest_sha256: str
    source_artifact_sha256: Mapping[str, Mapping[str, str]]


def aggregate_records(records: Sequence[Mapping[str, object]]) -> AggregateBundle:
    """Validate, recompute percentages, and sort the exact 450-cell grid."""

    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise ValueError("formal records must be a sequence")
    if len(records) != EXPECTED_RECORDS:
        raise ValueError(
            f"formal aggregation requires exactly 450 decoder records; got {len(records)}"
        )

    normalized: list[dict[str, object]] = []
    keys: list[tuple[str, int, str]] = []
    manifest_hashes: set[str] = set()
    source_by_role: dict[str, dict[str, str]] = {model: {} for model in MODEL_ORDER}
    for raw in records:
        if not isinstance(raw, Mapping):
            raise ValueError("every formal record must be a mapping")
        if _AUDIT_FIELDS.intersection(raw):
            raise ValueError("audit data must not contaminate the 450 formal records")
        record = dict(raw)
        model = record.get("model")
        decoder = record.get("decoder")
        index = record.get("scenario_index")
        if model not in MODEL_ORDER:
            raise ValueError(f"formal record has invalid model: {model}")
        if decoder not in DECODER_ORDER:
            raise ValueError(f"formal record has invalid decoder: {decoder}")
        if isinstance(index, bool) or not isinstance(index, int) or index not in range(30):
            raise ValueError(f"formal record has invalid scenario index: {index}")
        suite, scenario = _scenario_identity(index)
        if record.get("suite") != suite:
            raise ValueError(f"formal record has invalid suite for scenario {index}")
        if record.get("scenario") != scenario:
            raise ValueError(f"formal record has invalid scenario name for index {index}")
        if record.get("subject") != "sub-08" or record.get("seed") != 42:
            raise ValueError("formal report is locked to sub-08 / seed-42")
        manifest_hash = record.get("scenario_manifest_sha256")
        if not _is_sha256(manifest_hash):
            raise ValueError("formal record has invalid scenario manifest hash")
        manifest_hashes.add(str(manifest_hash))
        _validate_metric_record(record, duplicate=suite == "duplicate_eeg")
        _record_source_hashes(record, source_by_role)
        normalized.append(_recomputed_record(record))
        keys.append((str(model), index, str(decoder)))

    if len(set(keys)) != EXPECTED_RECORDS:
        raise ValueError("formal decoder records must use 450 unique grid keys")
    expected = {
        (model, index, decoder)
        for model in MODEL_ORDER
        for index in range(30)
        for decoder in DECODER_ORDER
    }
    if set(keys) != expected:
        raise ValueError("formal decoder records do not cover the exact model/scenario/decoder grid")
    if len(manifest_hashes) != 1:
        raise ValueError("formal records bind mixed scenario manifest hashes")

    normalized.sort(
        key=lambda row: (
            MODEL_ORDER.index(str(row["model"])),
            SUITE_ORDER.index(str(row["suite"])),
            int(row["scenario_index"]),
            DECODER_ORDER.index(str(row["decoder"])),
        )
    )
    immutable_sources = {
        model: dict(sorted(values.items()))
        for model, values in source_by_role.items()
        if values
    }
    return AggregateBundle(
        records=tuple(normalized),
        scenario_manifest_sha256=next(iter(manifest_hashes)),
        source_artifact_sha256=immutable_sources,
    )


def load_run_records(runs_dir: Path) -> list[dict[str, object]]:
    """Securely consume the exact directory tree published by Task 7."""

    runs_dir = _regular_directory(Path(runs_dir), "runs directory")
    _reject_symlinks_below(runs_dir, "runs tree")
    manifest_path = runs_dir / "scenario_manifest.json"
    manifest_bytes = _read_regular_file(manifest_path, "scenario manifest")
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = _validate_run_manifest(manifest_bytes)

    expected_files = {manifest_path}
    records: list[dict[str, object]] = []
    standard_specs = standard_scenarios()
    for model in MODEL_ORDER:
        for index in range(30):
            suite, scenario = _scenario_identity(index)
            cell = runs_dir / model / "subj08" / "seed42" / suite / f"{index:02d}_{scenario}"
            summary_path = cell / "summary.json"
            ledger_path = cell / "per_query.csv"
            expected_files.update({summary_path, ledger_path})
            summary_bytes = _read_regular_file(summary_path, "cell summary")
            ledger_bytes = _read_regular_file(ledger_path, "per-query ledger")
            summary = _canonical_json(summary_bytes, "cell summary")
            expected_summary_keys = {
                "schema_version",
                "model",
                "subject",
                "seed",
                "suite",
                "scenario_index",
                "scenario",
                "matrix_shape",
                "scenario_selection",
                "scenario_manifest_sha256",
                "source_artifact_sha256",
                "decoder_records",
            }
            if set(summary) != expected_summary_keys or summary.get("schema_version") != 1:
                raise ValueError("cell summary does not use the exact Task 7 schema")
            identity = (
                summary.get("model"),
                summary.get("subject"),
                summary.get("seed"),
                summary.get("suite"),
                summary.get("scenario_index"),
                summary.get("scenario"),
            )
            if identity != (model, "sub-08", 42, suite, index, scenario):
                raise ValueError("cell summary path and identity disagree")
            if summary.get("scenario_manifest_sha256") != manifest_hash:
                raise ValueError("cell summary does not bind the canonical scenario manifest")
            expected_selection = (
                manifest["standard_scenarios"][index]["selected_canonical_ids"]
                if index < 27
                else {
                    "duplicate_query_ids": manifest["duplicate_query"][
                        "selected_by_count"
                    ][str((0, 10, 20)[index - 27])],
                    "duplicate_query_count": (0, 10, 20)[index - 27],
                }
            )
            if summary.get("scenario_selection") != expected_selection:
                raise ValueError("cell scenario selection differs from canonical manifest")
            _validate_matrix_shape(summary.get("matrix_shape"), index, standard_specs)
            ledger = _parse_per_query_ledger(
                ledger_bytes,
                model=model,
                suite=suite,
                scenario_index=index,
                scenario=scenario,
                matrix_shape=summary["matrix_shape"],
                selection=expected_selection,
                canonical_ids=manifest["gallery_canonical_ids"],
            )
            sources = _validate_source_hashes(summary.get("source_artifact_sha256"), suite)
            decoder_records = summary.get("decoder_records")
            if not isinstance(decoder_records, list) or len(decoder_records) != 5:
                raise ValueError("cell summary must contain exactly five decoder records")
            for decoder_record in decoder_records:
                if not isinstance(decoder_record, dict):
                    raise ValueError("decoder record must be a JSON object")
                enriched = dict(decoder_record)
                enriched.update(
                    {
                        "subject": "sub-08",
                        "seed": 42,
                        "scenario_manifest_sha256": manifest_hash,
                        "source_artifact_sha256": sources,
                    }
                )
                expected_record = _record_from_ledger(
                    ledger,
                    decoder=str(enriched.get("decoder")),
                    model=model,
                    suite=suite,
                    scenario_index=index,
                    scenario=scenario,
                    matrix_shape=summary["matrix_shape"],
                    assignment_metadata=enriched.get("assignment_metadata"),
                )
                summary_record = {
                    key: value
                    for key, value in enriched.items()
                    if key
                    not in {
                        "subject",
                        "seed",
                        "scenario_manifest_sha256",
                        "source_artifact_sha256",
                    }
                }
                if summary_record != expected_record:
                    raise ValueError(
                        "decoder summary metrics do not match the per-query ledger"
                    )
                records.append(enriched)

    actual_files: set[Path] = set()
    for path in runs_dir.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"runs tree must not contain symlinks: {path}")
        if path.is_file():
            actual_files.add(path)
    extras = actual_files.difference(expected_files)
    missing = expected_files.difference(actual_files)
    if any("audit" in path.name.lower() for path in extras):
        raise ValueError("audit artifact must not be placed in the formal runs input")
    if missing or extras:
        raise ValueError(
            "formal runs input is partial or contains extra files: "
            f"missing={len(missing)}, extra={len(extras)}"
        )
    if manifest.get("seed") != 42:
        raise ValueError("scenario manifest is not seed-42")
    return records


def load_reproduction_audits(
    results_root: Path,
    aggregate: AggregateBundle,
) -> tuple[dict[str, object], ...]:
    """Load validation-selected manifests and isolated best-test audits."""

    root = _regular_directory(Path(results_root), "results root")
    for container in (root / "checkpoints", root / "matrices"):
        if container.is_symlink():
            raise ValueError("audit input containers must not be symlinks")
        if container.exists():
            _reject_symlinks_below(container, "audit input tree")
    expected_inventory = {
        (model, role): digest
        for model, roles in aggregate.source_artifact_sha256.items()
        for role, digest in roles.items()
    }
    rows: list[dict[str, object]] = []
    common_inventory: dict[tuple[str, str], str] | None = None
    for model in ("nice", "atm_s"):
        checkpoint_path = root / "checkpoints" / model / "checkpoint_manifest.json"
        audit_path = root / "matrices" / model / "best_test_audit.json"
        checkpoint_bytes = _read_regular_file(checkpoint_path, "checkpoint manifest")
        audit_bytes = _read_regular_file(audit_path, "best-test audit")
        checkpoint = _canonical_json(checkpoint_bytes, "checkpoint manifest")
        audit = _canonical_json(audit_bytes, "best-test audit")
        selection, checkpoint_identities = _validate_checkpoint_manifest(
            checkpoint, model
        )
        audit_data, inventory, audit_identities = _validate_audit_manifest(
            audit, model
        )
        if checkpoint_identities != audit_identities:
            raise ValueError(
                "best-test audit checkpoint hashes do not bind the training manifest"
            )
        if common_inventory is None:
            common_inventory = inventory
        elif inventory != common_inventory:
            raise ValueError("native audit manifests bind different formal inventories")
        formal = _lookup_record(aggregate, model, 0, "independent")
        artifact_path = root / "matrices" / model / "standard"
        artifact = read_score_artifact(artifact_path)
        artifact_digest = _score_artifact_sha256(artifact_path)
        if artifact_digest != expected_inventory[(model, "standard")]:
            raise ValueError(
                "native standard artifact hash does not match the formal run ledger"
            )
        _validate_native_standard_artifact(
            artifact,
            model=model,
            checkpoint=checkpoint,
            checkpoint_manifest_sha256=hashlib.sha256(checkpoint_bytes).hexdigest(),
            formal=formal,
        )
        source = checkpoint.get("source")
        rows.append(
            {
                "model": model,
                "formal_epoch": selection["epoch"],
                "formal_val_loss": selection["val_loss"],
                "formal_top1_count": formal["correct"],
                "formal_top5_count": formal["top5_count"],
                "sample_count": formal["total"],
                "best_test_epoch": audit_data["epoch"],
                "best_test_top1_count": audit_data["top1_count"],
                "best_test_top5_count": audit_data["top5_count"],
                "source_commit": source.get("commit") if isinstance(source, Mapping) else None,
                "checkpoint_manifest_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
                "audit_manifest_sha256": hashlib.sha256(audit_bytes).hexdigest(),
            }
        )
    if common_inventory != expected_inventory:
        raise ValueError("audit parity inventory does not match Task 7 source artifact hashes")
    return tuple(rows)


def render_english_report(
    aggregate: AggregateBundle,
    audits: Sequence[Mapping[str, object]] = (),
) -> str:
    """Render a compact English report without cross-cell overclaiming."""

    audit_rows = _validated_audit_rows(audits)
    standard = _standard_table(aggregate, language="en")
    duplicate = _duplicate_table(aggregate, language="en")
    sinkhorn = _sinkhorn_status(aggregate, language="en")
    provenance = _provenance_table(aggregate, language="en")
    audit = _audit_table(audit_rows, language="en")
    return (
        "# Matching Fairness Results\n\n"
        "This formal result covers **sub-08 / seed-42** only and does not establish "
        "cross-subject significance. Counts are reported before percentages.\n\n"
        f"{sinkhorn}\n\n"
        "## Standard (80-trial averages)\n\n"
        f"{standard}\n\n"
        "## Duplicate EEG (40-trial disjoint averages)\n\n"
        "The standard and duplicate-EEG absolute values are not compared across these "
        "two suites. Duplicate columns compare only the same scenario and metric.\n\n"
        f"{duplicate}\n\n"
        "Assignment decoders produce one global assignment, so assignment Top-5 is "
        "undefined and is not reported.\n\n"
        "## Reproduction audit\n\n"
        "Formal values use the validation-selected checkpoint. The best-test rows are "
        "**test-set-selected audit-only** values and must not be used as formal fairness "
        "inputs or evidence of generalization.\n\n"
        f"{audit}\n\n"
        "## Provenance\n\n"
        f"Scenario manifest SHA-256: `{aggregate.scenario_manifest_sha256}`.\n\n"
        f"{provenance}\n"
    )


def render_chinese_report(
    aggregate: AggregateBundle,
    audits: Sequence[Mapping[str, object]] = (),
) -> str:
    """Render the semantically aligned Chinese report."""

    audit_rows = _validated_audit_rows(audits)
    standard = _standard_table(aggregate, language="zh")
    duplicate = _duplicate_table(aggregate, language="zh")
    sinkhorn = _sinkhorn_status(aggregate, language="zh")
    provenance = _provenance_table(aggregate, language="zh")
    audit = _audit_table(audit_rows, language="zh")
    return (
        "# 匹配公平性实验结果\n\n"
        "本正式结果仅覆盖 **sub-08 / seed-42**，不能建立跨被试显著性。所有百分比前均报告精确计数。\n\n"
        f"{sinkhorn}\n\n"
        "## 标准套件（80-trial 平均）\n\n"
        f"{standard}\n\n"
        "## 重复 EEG 套件（互不重叠的 40-trial 平均）\n\n"
        "标准套件与重复 EEG 套件的绝对分数不作跨套件比较；重复套件只在同一场景、同一指标内比较。\n\n"
        f"{duplicate}\n\n"
        "分配式 decoder 只产生一个全局分配，因此其 Top-5 未定义，也不报告。\n\n"
        "## 复现审计\n\n"
        "正式数值使用验证集选择的 checkpoint。best-test 行属于**测试集选模、仅供审计**的数值，"
        "不得进入正式公平性输入，也不能作为泛化证据。\n\n"
        f"{audit}\n\n"
        "## 来源记录\n\n"
        f"场景 manifest SHA-256：`{aggregate.scenario_manifest_sha256}`。\n\n"
        f"{provenance}\n"
    )


def publish_aggregate(
    aggregate: AggregateBundle,
    output_dir: Path,
    audits: Sequence[Mapping[str, object]] = (),
) -> Path:
    """Atomically publish all deterministic aggregate deliverables."""

    audit_rows = _validated_audit_rows(audits)
    output_dir = _new_output_directory(Path(output_dir))
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        english = render_english_report(aggregate, audit_rows)
        chinese = render_chinese_report(aggregate, audit_rows)
        _write_exclusive(staging / "RESULTS.md", english.encode("utf-8"))
        _write_exclusive(staging / "RESULTS_ZH.md", chinese.encode("utf-8"))
        _write_exclusive(staging / "aggregate_metrics.csv", _csv_bytes(aggregate.records))
        summary = {
            "schema_version": 1,
            "subject": "sub-08",
            "seed": 42,
            "record_count": len(aggregate.records),
            "standard_record_count": sum(
                row["suite"] == "standard" for row in aggregate.records
            ),
            "duplicate_eeg_record_count": sum(
                row["suite"] == "duplicate_eeg" for row in aggregate.records
            ),
            "scenario_manifest_sha256": aggregate.scenario_manifest_sha256,
            "source_artifact_sha256": aggregate.source_artifact_sha256,
            "reproduction_audit": list(audit_rows),
            "sinkhorn": _sinkhorn_summary(aggregate),
            "limitation": "sub-08 / seed-42; no cross-subject significance",
        }
        _write_exclusive(
            staging / "aggregate_summary.json",
            _json_bytes(summary),
        )
        _write_exclusive(
            staging / "presentation_standard.md",
            (_standard_table(aggregate, language="en") + "\n").encode("utf-8"),
        )
        _write_exclusive(
            staging / "presentation_duplicate_eeg.md",
            (_duplicate_table(aggregate, language="en") + "\n").encode("utf-8"),
        )
        publish_staged_directory(staging, output_dir)
    finally:
        if os.path.lexists(staging):
            shutil.rmtree(staging)
    return output_dir


def aggregate_results(results_root: Path) -> Path:
    """Validate one results root and publish its aggregate directory."""

    root = _regular_directory(Path(results_root), "results root")
    aggregate = aggregate_records(load_run_records(root / "runs"))
    audits = load_reproduction_audits(root, aggregate)
    return publish_aggregate(aggregate, root / "aggregate", audits)


def _parse_per_query_ledger(
    encoded: bytes,
    *,
    model: str,
    suite: str,
    scenario_index: int,
    scenario: str,
    matrix_shape: Sequence[int],
    selection: Mapping[str, object],
    canonical_ids: Sequence[str],
) -> dict[str, tuple[dict[str, object], ...]]:
    try:
        text = encoded.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("per-query ledger must be valid UTF-8") from error
    if "\x00" in text or "\r" in text or not text.endswith("\n"):
        raise ValueError("per-query ledger has invalid control characters/newlines")
    try:
        raw_rows = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except csv.Error as error:
        raise ValueError("per-query ledger is malformed CSV") from error
    if not raw_rows or tuple(raw_rows[0]) != _PER_QUERY_FIELDS:
        raise ValueError("per-query ledger header does not match Task 7 schema")
    query_count, gallery_count = (int(value) for value in matrix_shape)
    expected_data_rows = query_count * len(DECODER_ORDER)
    if len(raw_rows) != expected_data_rows + 1:
        raise ValueError(
            "per-query ledger row count does not match matrix rows x five decoders"
        )

    expected_query_ids, expected_targets, gallery_entries, gallery_canonical = (
        _scenario_identities(
            suite=suite,
            selection=selection,
            canonical_ids=canonical_ids,
        )
    )
    if (
        len(expected_query_ids) != query_count
        or len(expected_targets) != query_count
        or len(gallery_entries) != gallery_count
        or len(gallery_canonical) != gallery_count
    ):
        raise ValueError("scenario identity cardinality does not match matrix shape")

    groups: dict[str, list[dict[str, object]]] = {
        decoder: [] for decoder in DECODER_ORDER
    }
    gallery_identity: dict[int, tuple[str, str]] = {}
    for flat_index, cells in enumerate(raw_rows[1:]):
        if len(cells) != len(_PER_QUERY_FIELDS):
            raise ValueError("per-query ledger row has missing or extra cells")
        raw = dict(zip(_PER_QUERY_FIELDS, cells))
        decoder = DECODER_ORDER[flat_index // query_count]
        query_index = flat_index % query_count
        expected_literals = {
            "model": model,
            "subject": "sub-08",
            "seed": "42",
            "suite": suite,
            "scenario_index": str(scenario_index),
            "scenario": scenario,
            "decoder": decoder,
            "query_index": str(query_index),
        }
        if any(raw[field] != value for field, value in expected_literals.items()):
            raise ValueError("per-query ledger identity/order/index is invalid")
        query_id = _safe_text(raw["query_id"], "query ID")
        target = _safe_text(raw["target_canonical_id"], "target canonical ID")
        if query_id != expected_query_ids[query_index] or target != expected_targets[query_index]:
            raise ValueError("per-query ledger query/target ID is invalid")
        answerable = _parse_bool(raw["answerable"], "answerable")
        expected_answerable = target in set(gallery_canonical)
        if answerable is not expected_answerable:
            raise ValueError("per-query ledger answerable flag is invalid")
        gallery_index = _parse_int(raw["gallery_index"], "gallery index", allow_negative=True)
        unmatched = _parse_bool(raw["unmatched"], "unmatched")
        if unmatched is not (gallery_index == -1):
            raise ValueError("per-query ledger unmatched/gallery index disagree")
        if gallery_index < -1 or gallery_index >= gallery_count:
            raise ValueError("per-query ledger gallery index is out of range")

        predicted_entry = _optional_safe_text(
            raw["predicted_gallery_entry_id"], "predicted gallery entry ID"
        )
        predicted_canonical = _optional_safe_text(
            raw["predicted_canonical_id"], "predicted canonical ID"
        )
        assigned_score = _parse_optional_float(raw["assigned_score"], "assigned score")
        if unmatched:
            if predicted_entry is not None or predicted_canonical is not None or assigned_score is not None:
                raise ValueError("unmatched ledger row must have empty prediction/score cells")
        else:
            if predicted_entry != gallery_entries[gallery_index] or predicted_canonical != gallery_canonical[gallery_index] or assigned_score is None:
                raise ValueError("ledger prediction does not match canonical gallery index")
            previous = gallery_identity.setdefault(
                gallery_index, (predicted_entry, predicted_canonical)
            )
            if previous != (predicted_entry, predicted_canonical):
                raise ValueError("gallery index has inconsistent ledger identity")

        correct_top1 = _parse_bool(raw["correct_top1"], "correct_top1")
        expected_correct = not unmatched and predicted_canonical == target
        if correct_top1 is not expected_correct:
            raise ValueError("per-query ledger Top-1 flag is invalid")
        if decoder == "independent":
            correct_top5 = _parse_bool(raw["correct_top5"], "correct_top5")
            if correct_top1 and not correct_top5:
                raise ValueError("Independent Top-5 cannot be false when Top-1 is true")
            if correct_top5 and not answerable:
                raise ValueError("unanswerable query cannot be Independent Top-5 correct")
        else:
            if raw["correct_top5"] != "":
                raise ValueError("assignment decoder per-query Top-5 must be empty")
            correct_top5 = None
        groups[decoder].append(
            {
                "query_index": query_index,
                "query_id": query_id,
                "target_canonical_id": target,
                "answerable": answerable,
                "gallery_index": gallery_index,
                "predicted_gallery_entry_id": predicted_entry,
                "predicted_canonical_id": predicted_canonical,
                "assigned_score": assigned_score,
                "unmatched": unmatched,
                "correct_top1": correct_top1,
                "correct_top5": correct_top5,
            }
        )

    baseline_identity: tuple[tuple[object, ...], ...] | None = None
    for decoder in DECODER_ORDER:
        rows = groups[decoder]
        if len({row["query_index"] for row in rows}) != query_count or len({row["query_id"] for row in rows}) != query_count:
            raise ValueError("per-query ledger contains duplicate or missing query indexes/IDs")
        identity = tuple(
            (
                row["query_index"],
                row["query_id"],
                row["target_canonical_id"],
                row["answerable"],
            )
            for row in rows
        )
        if baseline_identity is None:
            baseline_identity = identity
        elif identity != baseline_identity:
            raise ValueError("decoder ledger blocks do not share exact query identity")
        strict = decoder in STRICT_DECODERS
        unmatched_count = sum(row["unmatched"] is True for row in rows)
        expected_unmatched = max(query_count - gallery_count, 0) if strict else 0
        if unmatched_count != expected_unmatched:
            raise ValueError("decoder ledger unmatched count violates matrix cardinality")
        matched_entries = [
            row["predicted_gallery_entry_id"] for row in rows if not row["unmatched"]
        ]
        if strict and len(set(matched_entries)) != len(matched_entries):
            raise ValueError("strict one-to-one ledger reuses a gallery entry")
    return {decoder: tuple(rows) for decoder, rows in groups.items()}


def _scenario_identities(
    *,
    suite: str,
    selection: Mapping[str, object],
    canonical_ids: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    canonical = tuple(str(value) for value in canonical_ids)
    if suite == "standard":
        dropped_query = set(selection["drop_query"]) | set(selection["drop_pair"])
        dropped_gallery = set(selection["drop_gallery"]) | set(selection["drop_pair"])
        queries = tuple(value for value in canonical if value not in dropped_query)
        base_gallery = tuple(value for value in canonical if value not in dropped_gallery)
        duplicates = tuple(str(value) for value in selection["duplicate_gallery"])
        duplicate_entries = tuple(
            f"{value}__duplicate_entry_{position:04d}"
            for position, value in enumerate(duplicates)
        )
        return (
            queries,
            queries,
            base_gallery + duplicate_entries,
            base_gallery + duplicates,
        )
    repeated = tuple(str(value) for value in selection["duplicate_query_ids"])
    return (
        canonical + tuple(f"{value}__eeg_b" for value in repeated),
        canonical + repeated,
        canonical,
        canonical,
    )


def _record_from_ledger(
    ledger: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    decoder: str,
    model: str,
    suite: str,
    scenario_index: int,
    scenario: str,
    matrix_shape: Sequence[int],
    assignment_metadata: object,
) -> dict[str, object]:
    if decoder not in DECODER_ORDER:
        raise ValueError("decoder summary has invalid decoder identity")
    rows = ledger[decoder]
    independent = ledger["independent"]
    total = len(rows)
    correct = sum(row["correct_top1"] is True for row in rows)
    answerable = sum(row["answerable"] is True for row in rows)
    answerable_correct = sum(
        row["answerable"] is True and row["correct_top1"] is True for row in rows
    )
    assigned = sum(row["unmatched"] is False for row in rows)
    unmatched = total - assigned
    _validate_assignment_metadata(
        decoder,
        assignment_metadata,
        matched=assigned,
        unmatched=unmatched,
        total=total,
    )
    if decoder == "hungarian":
        assigned_sum = sum(
            float(row["assigned_score"])
            for row in rows
            if row["assigned_score"] is not None
        )
        if not math.isclose(
            float(assignment_metadata["assigned_sum_similarity"]),
            assigned_sum,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("Hungarian assigned similarity sum disagrees with ledger")
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
    result: dict[str, object] = {
        "model": model,
        "suite": suite,
        "scenario_index": scenario_index,
        "scenario": scenario,
        "decoder": decoder,
        "correct": correct,
        "total": total,
        "top1": _percent(correct, total),
        "answerable_correct": answerable_correct,
        "answerable_total": answerable,
        "answerable_top1": _percent(answerable_correct, answerable),
        "unanswerable_count": total - answerable,
        "assigned_count": assigned,
        "unmatched_count": unmatched,
        "unique_gallery_entry_predictions": len(
            {
                row["predicted_gallery_entry_id"]
                for row in rows
                if not row["unmatched"]
            }
        ),
        "unique_canonical_predictions": len(
            {
                row["predicted_canonical_id"]
                for row in rows
                if not row["unmatched"]
            }
        ),
        "strict_one_to_one": decoder in STRICT_DECODERS,
        "top5_count": sum(row["correct_top5"] is True for row in rows)
        if decoder == "independent"
        else None,
        "top5": _percent(
            sum(row["correct_top5"] is True for row in rows), total
        )
        if decoder == "independent"
        else None,
        "assignment_changes_from_independent": sum(
            old["gallery_index"] != new["gallery_index"]
            for old, new in zip(independent, rows)
        ),
        "delta_correct_vs_independent": correct
        - sum(row["correct_top1"] is True for row in independent),
        "assignment_metadata": dict(assignment_metadata),
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
        base = [
            row for row in rows if not str(row["query_id"]).endswith("__eeg_b")
        ]
        appended = [
            row for row in rows if str(row["query_id"]).endswith("__eeg_b")
        ]
        at_least_one = sum(
            any(row["correct_top1"] for row in values)
            for values in by_target.values()
        )
        both = sum(
            all(row["correct_top1"] for row in values)
            for values in by_target.values()
        )
        ceiling = (
            min(int(matrix_shape[0]), int(matrix_shape[1]))
            if decoder in STRICT_DECODERS
            else answerable
        )
        if correct > ceiling:
            raise ValueError("duplicate EEG correct count exceeds theoretical ceiling")
        result.update(
            {
                "base_a_correct": sum(row["correct_top1"] for row in base),
                "base_a_total": len(base),
                "base_a_top1": _percent(
                    sum(row["correct_top1"] for row in base), len(base)
                ),
                "appended_b_correct": sum(
                    row["correct_top1"] for row in appended
                ),
                "appended_b_total": len(appended),
                "appended_b_top1": _percent(
                    sum(row["correct_top1"] for row in appended), len(appended)
                ),
                "repeated_canonical_total": len(repeated_targets),
                "at_least_one_correct_count": at_least_one,
                "at_least_one_coverage": _percent(
                    at_least_one, len(repeated_targets)
                ),
                "both_correct_count": both,
                "both_correct": _percent(both, len(repeated_targets)),
                "theoretical_ceiling_count": ceiling,
                "theoretical_ceiling": _percent(ceiling, total),
                "distance_from_ceiling": ceiling - correct,
                "unmatched_repeated_queries": sum(
                    row["unmatched"]
                    and row["target_canonical_id"] in repeated_targets
                    for row in rows
                ),
            }
        )
    return result


def _parse_bool(value: str, label: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"per-query ledger {label} must be an exact boolean")


def _parse_int(value: str, label: str, *, allow_negative: bool = False) -> int:
    if re.fullmatch(r"-?(0|[1-9][0-9]*)", value) is None:
        raise ValueError(f"per-query ledger {label} must be an exact integer")
    parsed = int(value)
    if not allow_negative and parsed < 0:
        raise ValueError(f"per-query ledger {label} must be non-negative")
    return parsed


def _parse_optional_float(value: str, label: str) -> float | None:
    if value == "":
        return None
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"per-query ledger {label} must be numeric") from error
    if not math.isfinite(parsed):
        raise ValueError(f"per-query ledger {label} must be finite")
    return parsed


def _safe_text(value: str, label: str) -> str:
    if not value:
        raise ValueError(f"per-query ledger {label} must be non-empty")
    if value[0] in "=+-@" or any(
        ord(character) < 32 or ord(character) == 127 for character in value
    ):
        raise ValueError(f"per-query ledger {label} contains formula/control text")
    return value


def _optional_safe_text(value: str, label: str) -> str | None:
    return None if value == "" else _safe_text(value, label)


def _scenario_identity(index: int) -> tuple[str, str]:
    if index < 27:
        return "standard", standard_scenarios()[index].slug
    return "duplicate_eeg", f"dupq{(0, 10, 20)[index - 27]}"


def _validate_metric_record(record: Mapping[str, object], *, duplicate: bool) -> None:
    expected = _BASE_FIELDS | (_DUPLICATE_FIELDS if duplicate else frozenset())
    if set(record) != expected:
        missing = sorted(expected.difference(record))
        extra = sorted(set(record).difference(expected))
        raise ValueError(
            "formal metric record schema mismatch: "
            f"missing={missing}, extra={extra}"
        )

    decoder = str(record["decoder"])
    strict = decoder in STRICT_DECODERS
    if record.get("strict_one_to_one") is not strict:
        raise ValueError("decoder strict-one-to-one label is invalid")
    integer_fields = (
        "correct",
        "total",
        "answerable_correct",
        "answerable_total",
        "unanswerable_count",
        "assigned_count",
        "unmatched_count",
        "unique_gallery_entry_predictions",
        "unique_canonical_predictions",
        "assignment_changes_from_independent",
        "delta_correct_vs_independent",
        "correct_to_correct",
        "correct_to_wrong",
        "wrong_to_correct",
        "wrong_to_wrong",
    )
    values = {field: _integer(record.get(field), field, allow_negative=field == "delta_correct_vs_independent") for field in integer_fields}
    total = values["total"]
    if total <= 0:
        raise ValueError("total must be positive")
    if not 0 <= values["correct"] <= total:
        raise ValueError("correct count is outside total")
    if values["assigned_count"] + values["unmatched_count"] != total:
        raise ValueError("assigned and unmatched counts do not sum to total")
    if values["answerable_total"] + values["unanswerable_count"] != total:
        raise ValueError("answerable and unanswerable counts do not sum to total")
    if not 0 <= values["answerable_correct"] <= values["answerable_total"]:
        raise ValueError("answerable correct count is invalid")
    transitions = sum(
        values[field]
        for field in (
            "correct_to_correct",
            "correct_to_wrong",
            "wrong_to_correct",
            "wrong_to_wrong",
        )
    )
    if transitions != total:
        raise ValueError("transition counts do not sum to total")
    current_correct = values["correct_to_correct"] + values["wrong_to_correct"]
    independent_correct = values["correct_to_correct"] + values["correct_to_wrong"]
    if current_correct != values["correct"]:
        raise ValueError("transition counts disagree with correct count")
    if values["delta_correct_vs_independent"] != values["correct"] - independent_correct:
        raise ValueError("delta_correct_vs_independent is inconsistent")
    _check_percent(record.get("top1"), values["correct"], total, "top1")
    _check_percent(
        record.get("answerable_top1"),
        values["answerable_correct"],
        values["answerable_total"],
        "answerable_top1",
    )
    top5_count = record.get("top5_count")
    top5 = record.get("top5")
    if decoder == "independent":
        count = _integer(top5_count, "top5_count")
        if not 0 <= count <= total:
            raise ValueError("independent Top-5 count is invalid")
        _check_percent(top5, count, total, "top5")
    elif top5_count is not None or top5 is not None:
        raise ValueError("assignment decoder Top-5 must be absent/null")

    _validate_assignment_metadata(
        decoder,
        record.get("assignment_metadata"),
        matched=values["assigned_count"],
        unmatched=values["unmatched_count"],
        total=total,
    )
    if duplicate:
        _validate_duplicate_metrics(record, total, values["correct"])


def _validate_duplicate_metrics(
    record: Mapping[str, object],
    total: int,
    correct: int,
) -> None:
    count_fields = (
        "base_a_correct",
        "base_a_total",
        "appended_b_correct",
        "appended_b_total",
        "repeated_canonical_total",
        "at_least_one_correct_count",
        "both_correct_count",
        "theoretical_ceiling_count",
        "distance_from_ceiling",
        "unmatched_repeated_queries",
    )
    counts = {field: _integer(record.get(field), field) for field in count_fields}
    if counts["base_a_total"] + counts["appended_b_total"] != total:
        raise ValueError("duplicate EEG base/appended denominators do not sum to total")
    if counts["base_a_correct"] + counts["appended_b_correct"] != correct:
        raise ValueError("duplicate EEG subgroup counts disagree with overall correct")
    repeated = counts["repeated_canonical_total"]
    if counts["appended_b_total"] != repeated:
        raise ValueError("duplicate EEG repeated/appended denominators disagree")
    for numerator in ("at_least_one_correct_count", "both_correct_count"):
        if counts[numerator] > repeated:
            raise ValueError(f"{numerator} exceeds its denominator")
    ceiling = counts["theoretical_ceiling_count"]
    if ceiling > total or counts["distance_from_ceiling"] != ceiling - correct:
        raise ValueError("duplicate EEG theoretical ceiling/distance is inconsistent")
    if counts["unmatched_repeated_queries"] > record["unmatched_count"]:
        raise ValueError("unmatched repeated queries exceed all unmatched queries")
    _check_percent(record.get("base_a_top1"), counts["base_a_correct"], counts["base_a_total"], "base_a_top1")
    _check_percent(record.get("appended_b_top1"), counts["appended_b_correct"], counts["appended_b_total"], "appended_b_top1")
    _check_percent(record.get("at_least_one_coverage"), counts["at_least_one_correct_count"], repeated, "at_least_one_coverage")
    _check_percent(record.get("both_correct"), counts["both_correct_count"], repeated, "both_correct")
    _check_percent(record.get("theoretical_ceiling"), ceiling, total, "theoretical_ceiling")


def _recomputed_record(record: Mapping[str, object]) -> dict[str, object]:
    value = dict(record)
    value["top1"] = _percent(int(value["correct"]), int(value["total"]))
    value["answerable_top1"] = _percent(
        int(value["answerable_correct"]), int(value["answerable_total"])
    )
    if value["decoder"] == "independent":
        value["top5"] = _percent(int(value["top5_count"]), int(value["total"]))
    if value["suite"] == "duplicate_eeg":
        for output, numerator, denominator in (
            ("base_a_top1", "base_a_correct", "base_a_total"),
            ("appended_b_top1", "appended_b_correct", "appended_b_total"),
            ("at_least_one_coverage", "at_least_one_correct_count", "repeated_canonical_total"),
            ("both_correct", "both_correct_count", "repeated_canonical_total"),
            ("theoretical_ceiling", "theoretical_ceiling_count", "total"),
        ):
            value[output] = _percent(int(value[numerator]), int(value[denominator]))
    return value


def _validate_assignment_metadata(
    decoder: str,
    value: object,
    *,
    matched: int,
    unmatched: int,
    total: int,
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("assignment_metadata must be a mapping")
    metadata = dict(value)
    if decoder == "independent":
        if metadata:
            raise ValueError("Independent assignment metadata must be empty")
        return
    if decoder == "greedy":
        if metadata != {"matched_count": matched, "unmatched_count": unmatched}:
            raise ValueError("Greedy assignment metadata is invalid")
        return
    if decoder == "hungarian":
        expected = {
            "seed",
            "row_permutation_sha256",
            "column_permutation_sha256",
            "matched_count",
            "unmatched_count",
            "assigned_sum_similarity",
        }
        if (
            set(metadata) != expected
            or metadata.get("seed") != 42
            or metadata.get("matched_count") != matched
            or metadata.get("unmatched_count") != unmatched
            or not _is_sha256(metadata.get("row_permutation_sha256"))
            or not _is_sha256(metadata.get("column_permutation_sha256"))
            or not _finite_number(metadata.get("assigned_sum_similarity"))
        ):
            raise ValueError("Hungarian assignment metadata is invalid")
        return
    if decoder == "stable_matching":
        expected = {"matched_count", "unmatched_count", "proposal_count"}
        proposal_count = metadata.get("proposal_count")
        if (
            set(metadata) != expected
            or metadata.get("matched_count") != matched
            or metadata.get("unmatched_count") != unmatched
            or isinstance(proposal_count, bool)
            or not isinstance(proposal_count, int)
            or not 0 <= proposal_count <= total * max(total, 1)
        ):
            raise ValueError("Stable Matching assignment metadata is invalid")
        return
    expected = {
        "temperature",
        "max_iterations",
        "iterations",
        "tolerance",
        "marginal_error",
        "converged",
        "plan_min",
        "plan_max",
        "plan_sum",
        "plan_sha256",
    }
    iterations = metadata.get("iterations")
    if (
        set(metadata) != expected
        or metadata.get("temperature") != 0.05
        or metadata.get("max_iterations") != 500
        or metadata.get("tolerance") != 1e-8
        or isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or not 1 <= iterations <= 500
        or not isinstance(metadata.get("converged"), bool)
        or not _finite_number(metadata.get("marginal_error"))
        or float(metadata["marginal_error"]) < 0
        or not _finite_number(metadata.get("plan_min"))
        or not _finite_number(metadata.get("plan_max"))
        or not _finite_number(metadata.get("plan_sum"))
        or float(metadata["plan_min"]) < 0
        or float(metadata["plan_max"]) < float(metadata["plan_min"])
        or not math.isclose(float(metadata["plan_sum"]), 1.0, abs_tol=1e-6)
        or not _is_sha256(metadata.get("plan_sha256"))
    ):
        raise ValueError("Sinkhorn assignment metadata is invalid")

    tolerance = float(metadata["tolerance"])
    marginal_error = float(metadata["marginal_error"])
    converged = metadata["converged"]
    if converged != (marginal_error <= tolerance):
        raise ValueError("Sinkhorn convergence flag contradicts marginal error")
    if not converged and iterations != metadata["max_iterations"]:
        raise ValueError(
            "Sinkhorn non-convergence must exhaust the configured iterations"
        )


def _record_source_hashes(
    record: Mapping[str, object],
    source_by_role: dict[str, dict[str, str]],
) -> None:
    source = record.get("source_artifact_sha256")
    if source is None:
        return
    suite = str(record["suite"])
    validated = _validate_source_hashes(source, suite)
    model_sources = source_by_role[str(record["model"])]
    for role, digest in validated.items():
        previous = model_sources.setdefault(role, digest)
        if previous != digest:
            raise ValueError("model scenarios bind mixed source artifact hashes")


def _validate_source_hashes(value: object, suite: str) -> dict[str, str]:
    expected = {"standard"} if suite == "standard" else {"eeg_a", "eeg_b"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("cell source artifact hash roles are invalid")
    result = {str(key): str(digest) for key, digest in value.items()}
    if not all(_is_sha256(digest) for digest in result.values()):
        raise ValueError("cell source artifact SHA-256 is invalid")
    return result


def _chinese_standard_scenario_label(spec: ScenarioSpec) -> str:
    operations = []
    if spec.drop_query:
        operations.append(f"删除 {spec.drop_query} 条 EEG")
    if spec.drop_gallery:
        operations.append(f"删除 {spec.drop_gallery} 张图片")
    if spec.duplicate_gallery:
        operations.append(f"重复 {spec.duplicate_gallery} 张图片")
    description = "、".join(operations) if operations else "标准一一匹配"
    query_count = 200 - spec.drop_query - spec.drop_pair
    gallery_count = 200 - spec.drop_gallery - spec.drop_pair + spec.duplicate_gallery
    return f"{description}（{query_count} 条 EEG × {gallery_count} 张图片）"


def _chinese_duplicate_scenario_label(index: int) -> str:
    labels = {
        27: "真实重复 EEG 基准（200 条 EEG-A × 200 张图片）",
        28: "加入 10 条真实重复 EEG-B（210 条 EEG × 200 张图片）",
        29: "加入 20 条真实重复 EEG-B（220 条 EEG × 200 张图片）",
    }
    try:
        return labels[index]
    except KeyError:
        raise ValueError(f"invalid duplicate-EEG scenario index: {index}") from None


def _standard_table(aggregate: AggregateBundle, *, language: str) -> str:
    decoders = DECODER_ORDER
    records = {
        (int(row["scenario_index"]), str(row["model"]), str(row["decoder"])): row
        for row in aggregate.records
        if row["suite"] == "standard"
    }
    best = {
        (index, decoder): max(
            int(records[(index, model, decoder)]["correct"])
            for model in MODEL_ORDER
        )
        for index in range(27)
        for decoder in decoders
    }
    top5_best = {
        index: max(
            int(records[(index, model, "independent")]["top5_count"])
            for model in MODEL_ORDER
        )
        for index in range(27)
    }
    if language == "zh":
        headers = ("场景", "模型", "Independent Top-1", "Independent Top-5", "Greedy Top-1", "Hungarian Top-1", "Stable Matching Top-1", "Sinkhorn Top-1", "Sinkhorn 收敛")
    else:
        headers = ("Scenario", "Model", "Independent Top-1", "Independent Top-5", "Greedy Top-1", "Hungarian Top-1", "Stable Matching Top-1", "Sinkhorn Top-1", "Sinkhorn converged")
    rows = []
    for index, spec in enumerate(standard_scenarios()):
        for model in MODEL_ORDER:
            scenario_label = (
                _chinese_standard_scenario_label(spec)
                if language == "zh"
                else f"{index:02d} {spec.slug}"
            )
            cells = [scenario_label, model]
            independent = records[(index, model, "independent")]
            cells.append(_score_cell(independent, bold=int(independent["correct"]) == best[(index, "independent")]))
            cells.append(_count_cell(int(independent["top5_count"]), int(independent["total"]), bold=int(independent["top5_count"]) == top5_best[index]))
            for decoder in decoders[1:]:
                row = records[(index, model, decoder)]
                cells.append(_score_cell(row, bold=int(row["correct"]) == best[(index, decoder)]))
            sinkhorn = records[(index, model, "sinkhorn")]["assignment_metadata"]
            cells.append("yes" if sinkhorn["converged"] else "NO")
            rows.append(cells)
    return _markdown_table(headers, rows)


def _sinkhorn_summary(aggregate: AggregateBundle) -> dict[str, int]:
    rows = [row for row in aggregate.records if row["decoder"] == "sinkhorn"]
    nonconverged = sum(
        row["assignment_metadata"]["converged"] is False for row in rows
    )
    return {
        "total": len(rows),
        "converged": len(rows) - nonconverged,
        "nonconverged": nonconverged,
    }


def _sinkhorn_status(aggregate: AggregateBundle, *, language: str) -> str:
    summary = _sinkhorn_summary(aggregate)
    if language == "zh":
        if summary["nonconverged"]:
            return (
                f"**警告：Sinkhorn 有 {summary['nonconverged']}/{summary['total']} 个单元未收敛；"
                "这些单元保留作透明报告，不应解释为可靠的最优传输解。**"
            )
        return f"Sinkhorn 收敛检查：{summary['converged']}/{summary['total']} 个单元收敛。"
    if summary["nonconverged"]:
        return (
            f"**WARNING: {summary['nonconverged']}/{summary['total']} Sinkhorn cells did not converge; "
            "they are retained for transparency and must not be interpreted as reliable transport optima.**"
        )
    return f"Sinkhorn convergence check: {summary['converged']}/{summary['total']} cells converged."


def _duplicate_table(aggregate: AggregateBundle, *, language: str) -> str:
    duplicate = [row for row in aggregate.records if row["suite"] == "duplicate_eeg"]
    best = {
        (int(row["scenario_index"]), str(row["decoder"])): max(
            int(candidate["correct"])
            for candidate in duplicate
            if candidate["scenario_index"] == row["scenario_index"]
            and candidate["decoder"] == row["decoder"]
        )
        for row in duplicate
    }
    headers = (
        ("模型", "场景", "Decoder", "Top-1", "理论上限", "距上限", "未匹配", "至少一条正确", "两条均正确")
        if language == "zh"
        else ("Model", "Scenario", "Decoder", "Top-1", "Ceiling", "Distance", "Unmatched", "At-least-one", "Both-correct")
    )
    rows = []
    for row in duplicate:
        key = (int(row["scenario_index"]), str(row["decoder"]))
        rows.append(
            [
                str(row["model"]),
                (
                    _chinese_duplicate_scenario_label(int(row["scenario_index"]))
                    if language == "zh"
                    else str(row["scenario"])
                ),
                str(row["decoder"]),
                _score_cell(row, bold=int(row["correct"]) == best[key]),
                _count_cell(int(row["theoretical_ceiling_count"]), int(row["total"])),
                str(row["distance_from_ceiling"]),
                f"{row['unmatched_count']}/{row['total']}",
                _count_cell(int(row["at_least_one_correct_count"]), int(row["repeated_canonical_total"])),
                _count_cell(int(row["both_correct_count"]), int(row["repeated_canonical_total"])),
            ]
        )
    return _markdown_table(headers, rows)


def _audit_table(rows: Sequence[Mapping[str, object]], *, language: str) -> str:
    if not rows:
        return "_Not supplied._" if language == "en" else "_尚未提供。_"
    headers = (
        ("模型", "正式 checkpoint", "验证损失", "正式 Top-1", "正式 Top-5", "best-test（仅审计）", "上游 commit")
        if language == "zh"
        else ("Model", "Formal checkpoint", "Validation loss", "Formal Top-1", "Formal Top-5", "Best-test (audit only)", "Upstream commit")
    )
    values = []
    for row in rows:
        epoch = int(row["formal_epoch"])
        epoch_text = f"第 {epoch} 轮" if language == "zh" else f"epoch {epoch}"
        values.append(
            [
                str(row["model"]),
                epoch_text,
                f"{float(row['formal_val_loss']):.6g}",
                _count_cell(int(row["formal_top1_count"]), int(row["sample_count"])),
                _count_cell(int(row["formal_top5_count"]), int(row["sample_count"])),
                f"epoch {row['best_test_epoch']}: Top-1 "
                + _count_cell(int(row["best_test_top1_count"]), int(row["sample_count"]))
                + ("；Top-5 " if language == "zh" else "; Top-5 ")
                + _count_cell(int(row["best_test_top5_count"]), int(row["sample_count"])),
                f"`{row['source_commit']}`" if row.get("source_commit") else "n/a",
            ]
        )
    return _markdown_table(headers, values)


def _provenance_table(aggregate: AggregateBundle, *, language: str) -> str:
    headers = ("模型", "产物角色", "SHA-256") if language == "zh" else ("Model", "Artifact role", "SHA-256")
    rows = [
        [model, role, f"`{digest}`"]
        for model in MODEL_ORDER
        for role, digest in aggregate.source_artifact_sha256.get(model, {}).items()
    ]
    return _markdown_table(headers, rows) if rows else ("_Not supplied._" if language == "en" else "_尚未提供。_")


def _validated_audit_rows(audits: Sequence[Mapping[str, object]]) -> tuple[dict[str, object], ...]:
    if not audits:
        return ()
    if len(audits) != 2:
        raise ValueError("reproduction audit requires NICE and ATM-S rows")
    fields = (
        "model",
        "formal_epoch",
        "formal_val_loss",
        "formal_top1_count",
        "formal_top5_count",
        "sample_count",
        "best_test_epoch",
        "best_test_top1_count",
        "best_test_top5_count",
        "source_commit",
        "checkpoint_manifest_sha256",
        "audit_manifest_sha256",
    )
    integer_fields = (
        "formal_epoch",
        "formal_top1_count",
        "formal_top5_count",
        "sample_count",
        "best_test_epoch",
        "best_test_top1_count",
        "best_test_top5_count",
    )
    result: list[dict[str, object]] = []
    expected_models = ("nice", "atm_s")
    for expected, raw in zip(expected_models, audits):
        if not isinstance(raw, Mapping) or set(raw) != set(fields):
            raise ValueError("reproduction audit row schema is invalid")
        row = {field: raw[field] for field in fields}
        if row["model"] != expected:
            raise ValueError("reproduction audit model order is invalid")
        for field in integer_fields:
            _integer(row[field], field)
        loss = row["formal_val_loss"]
        if (
            isinstance(loss, bool)
            or not isinstance(loss, (int, float))
            or not math.isfinite(float(loss))
        ):
            raise ValueError("formal validation loss must be finite")
        sample = int(row["sample_count"])
        if (
            int(row["formal_epoch"]) <= 0
            or int(row["best_test_epoch"]) <= 0
            or sample <= 0
            or not 0
            <= int(row["formal_top1_count"])
            <= int(row["formal_top5_count"])
            <= sample
            or not 0
            <= int(row["best_test_top1_count"])
            <= int(row["best_test_top5_count"])
            <= sample
        ):
            raise ValueError("reproduction audit metric counts are invalid")
        source_commit = row["source_commit"]
        if (
            not isinstance(source_commit, str)
            or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
            or not _is_sha256(row["checkpoint_manifest_sha256"])
            or not _is_sha256(row["audit_manifest_sha256"])
        ):
            raise ValueError("reproduction audit immutable provenance is invalid")
        result.append(row)
    return tuple(result)


def _validate_checkpoint_manifest(
    payload: Mapping[str, object], model: str
) -> tuple[Mapping[str, object], dict[int, str]]:
    expected_fields = {
        "schema_version", "model", "encoder_type", "subject", "seed", "source",
        "inputs", "hyperparameters", "encoder_behavior", "checkpoints",
        "selection", "best_checkpoint", "history", "stopped_early",
    }
    if set(payload) != expected_fields:
        raise ValueError("checkpoint manifest must use the exact Task 5 schema")
    expected_encoder = "NICE" if model == "nice" else "ATMS"
    if (
        payload.get("schema_version") != 1
        or payload.get("model") != model
        or payload.get("encoder_type") != expected_encoder
        or payload.get("subject") != "sub-08"
        or payload.get("seed") != 42
    ):
        raise ValueError("checkpoint manifest formal identity is invalid")
    source = payload.get("source")
    if (
        not isinstance(source, Mapping)
        or set(source) != {"url", "branch", "commit", "checkout_sha256"}
        or source.get("url") != OFFICIAL_SOURCE_URL
        or source.get("branch") != "develop"
        or re.fullmatch(r"[0-9a-f]{40}", str(source.get("commit", ""))) is None
        or not _is_sha256(source.get("checkout_sha256"))
    ):
        raise ValueError("checkpoint manifest source provenance is invalid")
    inputs = payload.get("inputs")
    expected_names = {
        "training_eeg": "preprocessed_eeg_training.npy",
        "training_features": "ViT-H-14_features_train.pt",
    }
    if not isinstance(inputs, Mapping) or set(inputs) != set(expected_names):
        raise ValueError("checkpoint manifest training inputs are incomplete")
    for role, name in expected_names.items():
        entry = inputs[role]
        if (
            not isinstance(entry, Mapping)
            or set(entry) != {"name", "sha256"}
            or entry.get("name") != name
            or not _is_sha256(entry.get("sha256"))
        ):
            raise ValueError("checkpoint manifest training input provenance is invalid")
    expected_hyperparameters = {
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
    }
    if payload.get("hyperparameters") != expected_hyperparameters:
        raise ValueError("checkpoint manifest formal hyperparameters are invalid")
    expected_behavior = {
        "use_subject_id": model == "atm_s",
        "normalize_feats": model == "atm_s",
    }
    if payload.get("encoder_behavior") != expected_behavior:
        raise ValueError("checkpoint manifest encoder behavior is invalid")
    if not isinstance(payload.get("stopped_early"), bool):
        raise ValueError("checkpoint manifest stopped_early must be boolean")
    selection = payload.get("selection")
    checkpoints = payload.get("checkpoints")
    if not isinstance(selection, Mapping) or set(selection) != {"epoch", "val_loss", "checkpoint"} or not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError("checkpoint manifest selection is incomplete")
    epoch = _integer(selection.get("epoch"), "selected epoch")
    if epoch <= 0:
        raise ValueError("selected epoch must be positive")
    loss = selection.get("val_loss")
    if isinstance(loss, bool) or not isinstance(loss, (int, float)) or not math.isfinite(float(loss)):
        raise ValueError("selected validation loss is invalid")
    candidates = [row for row in checkpoints if isinstance(row, Mapping)]
    if len(candidates) != len(checkpoints):
        raise ValueError("checkpoint manifest contains a malformed checkpoint row")
    identities: dict[int, str] = {}
    for row in candidates:
        if set(row) != {"epoch", "val_loss", "checkpoint", "sha256"}:
            raise ValueError("checkpoint manifest checkpoint row is incomplete")
        row_epoch = _integer(row.get("epoch"), "checkpoint epoch")
        row_loss = row.get("val_loss")
        if (
            isinstance(row_loss, bool)
            or not isinstance(row_loss, (int, float))
            or not math.isfinite(float(row_loss))
            or not isinstance(row.get("checkpoint"), str)
            or row["checkpoint"] != f"epoch_{row_epoch:04d}.pth"
            or not _is_sha256(row.get("sha256"))
            or row_epoch <= 0
            or row_epoch in identities
        ):
            raise ValueError("checkpoint manifest checkpoint row is invalid")
        identities[row_epoch] = str(row["sha256"])
    if list(identities) != sorted(identities):
        raise ValueError("checkpoint manifest rows must be ordered by epoch")
    selected = min(candidates, key=lambda row: (float(row["val_loss"]), int(row["epoch"])))
    if selection != {"epoch": selected["epoch"], "val_loss": selected["val_loss"], "checkpoint": selected["checkpoint"]} or epoch != int(selected["epoch"]):
        raise ValueError("checkpoint manifest does not select minimum validation loss")
    best = payload.get("best_checkpoint")
    if (
        not isinstance(best, Mapping)
        or set(best) != {"name", "sha256"}
        or best.get("name") != "best_val.pth"
        or not _is_sha256(best.get("sha256"))
        or best.get("sha256") != selected.get("sha256")
    ):
        raise ValueError("checkpoint manifest best checkpoint is invalid")
    history = payload.get("history")
    if (
        not isinstance(history, Mapping)
        or set(history) != {"name", "sha256"}
        or history.get("name") != "history.csv"
        or not _is_sha256(history.get("sha256"))
    ):
        raise ValueError("checkpoint manifest history provenance is invalid")
    return selection, identities


def _score_artifact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for name in ("metadata.json", "similarity.npy"):
        digest.update(name.encode("ascii"))
        digest.update(bytes.fromhex(sha256_file(Path(path) / name)))
    return digest.hexdigest()


def _validate_native_standard_artifact(
    artifact: ScoreArtifact,
    *,
    model: str,
    checkpoint: Mapping[str, object],
    checkpoint_manifest_sha256: str,
    formal: Mapping[str, object],
) -> None:
    if artifact.similarity.shape != (200, 200):
        raise ValueError("native standard artifact has invalid matrix shape")
    if not (
        artifact.query_ids
        == artifact.target_canonical_ids
        == artifact.gallery_entry_ids
        == artifact.gallery_canonical_ids
    ):
        raise ValueError("native standard artifact canonical identities disagree")
    metadata = artifact.metadata
    expected_fields = {
        "model_slug", "trial_half", "checkpoint_role", "checkpoint",
        "checkpoint_sha256", "checkpoint_manifest_sha256", "source_lock",
        "asset_lock_manifest_sha256", "asset_lock", "input_sha256",
        "trial_manifest_sha256", "subject", "seed", "logit_scale_type",
        "effective_logit_scale", "query_embeddings_sha256", "native_metrics",
    }
    if set(metadata) != expected_fields:
        raise ValueError("native standard artifact metadata schema is invalid")
    best = checkpoint["best_checkpoint"]
    if (
        metadata.get("model_slug") != model
        or metadata.get("trial_half") != "standard"
        or metadata.get("checkpoint_role") != "val_selected_formal"
        or not isinstance(metadata.get("checkpoint"), str)
        or not metadata["checkpoint"]
        or metadata.get("checkpoint_sha256") != best["sha256"]
        or metadata.get("checkpoint_manifest_sha256")
        != checkpoint_manifest_sha256
        or metadata.get("source_lock") != checkpoint["source"]
        or metadata.get("subject") != "sub-08"
        or metadata.get("seed") != 42
        or metadata.get("logit_scale_type") != "exp"
        or not _is_sha256(metadata.get("query_embeddings_sha256"))
        or not _is_sha256(metadata.get("asset_lock_manifest_sha256"))
    ):
        raise ValueError("native standard artifact checkpoint provenance is invalid")
    scale = metadata.get("effective_logit_scale")
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not math.isfinite(float(scale))
        or float(scale) <= 0
    ):
        raise ValueError("native standard artifact logit scale is invalid")
    _validate_native_asset_lock(metadata.get("asset_lock"))
    asset_files = metadata["asset_lock"]["files"]
    training_inputs = checkpoint["inputs"]
    if (
        asset_files[
            "Preprocessed_data_250Hz/sub-08/preprocessed_eeg_training.npy"
        ]["sha256"]
        != training_inputs["training_eeg"]["sha256"]
        or asset_files["ViT-H-14_features_train.pt"]["sha256"]
        != training_inputs["training_features"]["sha256"]
    ):
        raise ValueError("native standard artifact training inputs disagree")
    input_hashes = metadata.get("input_sha256")
    if (
        not isinstance(input_hashes, Mapping)
        or set(input_hashes) != {"test_eeg", "test_features", "trial_manifest"}
        or not all(_is_sha256(value) for value in input_hashes.values())
        or input_hashes["test_eeg"]
        != asset_files[
            "Preprocessed_data_250Hz/sub-08/preprocessed_eeg_test.npy"
        ]["sha256"]
        or input_hashes["test_features"]
        != asset_files["ViT-H-14_features_test.pt"]["sha256"]
        or input_hashes["trial_manifest"]
        != metadata.get("trial_manifest_sha256")
    ):
        raise ValueError("native standard artifact test input provenance is invalid")
    ranks = independent_ranks(artifact)
    expected_metrics = {
        "top1_count": int((ranks <= 1).sum()),
        "top5_count": int((ranks <= 5).sum()),
        "sample_count": len(ranks),
    }
    if metadata.get("native_metrics") != expected_metrics:
        raise ValueError("native standard artifact metric parity failed")
    if expected_metrics != {
        "top1_count": formal["correct"],
        "top5_count": formal["top5_count"],
        "sample_count": formal["total"],
    }:
        raise ValueError("native standard artifact metrics disagree with formal ledger")


def _validate_native_asset_lock(value: object) -> None:
    paths = {
        "Preprocessed_data_250Hz/sub-08/preprocessed_eeg_training.npy",
        "Preprocessed_data_250Hz/sub-08/preprocessed_eeg_test.npy",
        "ViT-H-14_features_train.pt",
        "ViT-H-14_features_test.pt",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != {"repo_id", "repo_type", "asset_root", "files"}
        or value.get("repo_id") != "LidongYang/EEG_Image_decode"
        or value.get("repo_type") != "dataset"
        or not isinstance(value.get("asset_root"), str)
        or not value["asset_root"]
        or not isinstance(value.get("files"), Mapping)
        or set(value["files"]) != paths
    ):
        raise ValueError("native standard artifact asset lock is incomplete")
    for entry in value["files"].values():
        if (
            not isinstance(entry, Mapping)
            or set(entry) != {"bytes", "sha256"}
            or isinstance(entry.get("bytes"), bool)
            or not isinstance(entry.get("bytes"), int)
            or entry["bytes"] < 0
            or not _is_sha256(entry.get("sha256"))
        ):
            raise ValueError("native standard artifact asset entry is invalid")


def _validate_audit_manifest(
    payload: Mapping[str, object], model: str
) -> tuple[
    Mapping[str, object],
    dict[tuple[str, str], str],
    dict[int, str],
]:
    expected_fields = {
        "schema_version", "scope", "model_slug", "checkpoint_policy",
        "fairness_artifact_created", "formal_artifact_inventory", "runs",
        "best_test",
    }
    if set(payload) != expected_fields:
        raise ValueError("best-test audit must use the exact Task 6 schema")
    if payload.get("schema_version") != 1 or payload.get("scope") != "best_test_audit_only" or payload.get("model_slug") != model or payload.get("checkpoint_policy") != "every_epoch_checkpoint" or payload.get("fairness_artifact_created") is not False:
        raise ValueError("best-test audit scope/identity is invalid")
    runs = payload.get("runs")
    best = payload.get("best_test")
    if not isinstance(runs, list) or not runs or not isinstance(best, Mapping):
        raise ValueError("best-test audit runs are incomplete")
    identities: dict[int, str] = {}
    for row in runs:
        if (
            not isinstance(row, Mapping)
            or set(row) != {
                "epoch", "checkpoint", "checkpoint_sha256",
                "effective_logit_scale", "top1_count", "top5_count",
                "sample_count",
            }
        ):
            raise ValueError("best-test audit run row is invalid")
        epoch = _integer(row.get("epoch"), "audit checkpoint epoch")
        top1 = _integer(row.get("top1_count"), "audit Top-1 count")
        top5 = _integer(row.get("top5_count"), "audit Top-5 count")
        sample = _integer(row.get("sample_count"), "audit sample count")
        scale = row.get("effective_logit_scale")
        if (
            epoch in identities
            or epoch <= 0
            or not isinstance(row.get("checkpoint"), str)
            or not row["checkpoint"]
            or not _is_sha256(row.get("checkpoint_sha256"))
            or isinstance(scale, bool)
            or not isinstance(scale, (int, float))
            or not math.isfinite(float(scale))
            or float(scale) <= 0
            or sample != 200
            or not 0 <= top1 <= top5 <= sample
        ):
            raise ValueError("best-test audit run metrics/checkpoint identity is invalid")
        identities[epoch] = str(row["checkpoint_sha256"])
    if list(identities) != sorted(identities):
        raise ValueError("best-test audit runs must be ordered by epoch")
    expected_best = max(runs, key=lambda row: (int(row["top1_count"]), int(row["top5_count"]), -int(row["epoch"])))
    if dict(best) != dict(expected_best):
        raise ValueError("best-test audit selector is invalid")
    inventory_payload = payload.get("formal_artifact_inventory")
    if not isinstance(inventory_payload, list) or len(inventory_payload) != 9:
        raise ValueError("best-test audit must bind the nine formal artifacts")
    inventory: dict[tuple[str, str], str] = {}
    role_map = {"standard": "standard", "a": "eeg_a", "b": "eeg_b"}
    for entry in inventory_payload:
        if (
            not isinstance(entry, Mapping)
            or set(entry) != {"model_slug", "trial_half", "path", "sha256"}
            or not isinstance(entry.get("path"), str)
            or not entry["path"]
        ):
            raise ValueError("audit inventory entry is invalid")
        model_slug = entry.get("model_slug")
        half = entry.get("trial_half")
        digest = entry.get("sha256")
        if model_slug not in MODEL_ORDER or half not in role_map or not _is_sha256(digest):
            raise ValueError("audit inventory identity/hash is invalid")
        key = (str(model_slug), role_map[str(half)])
        if key in inventory:
            raise ValueError("audit inventory contains a duplicate artifact role")
        inventory[key] = str(digest)
    return best, inventory, identities


def _lookup_record(aggregate: AggregateBundle, model: str, index: int, decoder: str) -> Mapping[str, object]:
    return next(row for row in aggregate.records if row["model"] == model and row["scenario_index"] == index and row["decoder"] == decoder)


def _validate_run_manifest(encoded: bytes) -> Mapping[str, object]:
    payload = _canonical_json(encoded, "scenario manifest")
    required = {
        "schema_version", "algorithm_version", "seed", "gallery_canonical_ids",
        "gallery_canonical_ids_sha256", "standard_master", "standard_scenarios",
        "duplicate_query", "trial_manifest_sha256", "decoder_configs",
    }
    if set(payload) != required or payload.get("schema_version") != 1 or payload.get("algorithm_version") != _RUN_MANIFEST_ALGORITHM or payload.get("seed") != 42:
        raise ValueError("scenario manifest does not use the formal Task 7 schema")
    gallery = payload.get("gallery_canonical_ids")
    if not isinstance(gallery, list) or len(gallery) != 200 or len(set(gallery)) != 200 or any(not isinstance(value, str) or not value for value in gallery):
        raise ValueError("scenario manifest gallery identity is invalid")
    if payload.get("gallery_canonical_ids_sha256") != _ordered_ids_sha256(gallery):
        raise ValueError("scenario manifest gallery hash is invalid")
    expected_master = build_standard_manifest(tuple(gallery), seed=42)
    if payload.get("standard_master") != expected_master:
        raise ValueError("scenario manifest standard master is invalid")
    scenarios = payload.get("standard_scenarios")
    expected_standard = []
    for index, spec in enumerate(standard_scenarios()):
        expected_standard.append(
            {
                "scenario_index": index,
                "scenario": spec.slug,
                "parameters": {
                    "drop_query": spec.drop_query,
                    "drop_gallery": spec.drop_gallery,
                    "drop_pair": spec.drop_pair,
                    "duplicate_gallery": spec.duplicate_gallery,
                },
                "selected_canonical_ids": _standard_selection(
                    expected_master,
                    spec.drop_query,
                    spec.drop_gallery,
                    spec.drop_pair,
                    spec.duplicate_gallery,
                ),
            }
        )
    if scenarios != expected_standard:
        raise ValueError("scenario manifest standard suite identity is invalid")
    duplicate = payload.get("duplicate_query")
    duplicate_ids = select_duplicate_image_ids(tuple(gallery), seed=42)
    expected_duplicate = {
        "ordered_ids": list(duplicate_ids),
        "counts": [0, 10, 20],
        "selected_by_count": {
            str(count): list(duplicate_ids[:count]) for count in (0, 10, 20)
        },
    }
    if duplicate != expected_duplicate:
        raise ValueError("scenario manifest duplicate EEG suite identity is invalid")
    decoders = payload.get("decoder_configs")
    expected_decoders = [
        {"name": "independent"},
        {"name": "greedy"},
        {"name": "hungarian", "seed": 42},
        {"name": "stable_matching"},
        {
            "name": "sinkhorn",
            "temperature": 0.05,
            "max_iterations": 500,
            "tolerance": 1e-8,
        },
    ]
    if decoders != expected_decoders:
        raise ValueError("scenario manifest decoder identity is invalid")
    if not _is_sha256(payload.get("trial_manifest_sha256")):
        raise ValueError("scenario manifest trial manifest hash is invalid")
    return payload


def _standard_selection(
    master: Mapping[str, object],
    drop_query: int,
    drop_gallery: int,
    drop_pair: int,
    duplicate_gallery: int,
) -> dict[str, object]:
    dropped_gallery = set(master["drop_gallery"][:drop_gallery])
    duplicates = [
        canonical_id
        for canonical_id in master["duplicate_gallery"]
        if canonical_id not in dropped_gallery
    ][:duplicate_gallery]
    return {
        "drop_query": list(master["drop_query"][:drop_query]),
        "drop_gallery": list(master["drop_gallery"][:drop_gallery]),
        "drop_pair": list(master["drop_pair"][:drop_pair]),
        "duplicate_gallery": duplicates,
    }


def _validate_matrix_shape(value: object, index: int, specs: Sequence[object]) -> None:
    if not isinstance(value, list) or len(value) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in value):
        raise ValueError("cell matrix shape is invalid")
    if index >= 27:
        expected = [200 + (0, 10, 20)[index - 27], 200]
    else:
        spec = specs[index]
        expected = [200 - spec.drop_query - spec.drop_pair, 200 - spec.drop_gallery - spec.drop_pair + spec.duplicate_gallery]
    if value != expected:
        raise ValueError(f"cell matrix shape does not match scenario {index}")


def _canonical_json(encoded: bytes, label: str) -> dict[str, object]:
    try:
        payload = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be valid UTF-8 JSON") from error
    if not isinstance(payload, dict) or _json_bytes(payload) != encoded:
        raise ValueError(f"{label} must be canonical sorted JSON")
    return payload


def _read_regular_file(path: Path, label: str) -> bytes:
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"formal input is partial; missing {label}: {path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        encoded = b"".join(chunks)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) or len(encoded) != before.st_size:
            raise ValueError(f"{label} changed while being read")
        if not encoded:
            raise ValueError(f"formal input is partial; empty {label}")
        return encoded
    finally:
        os.close(descriptor)


def _regular_directory(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    if not os.path.lexists(path) or not path.is_dir():
        raise ValueError(f"{label} must be an existing regular directory")
    return path.resolve(strict=True)


def _reject_symlinks_below(directory: Path, label: str) -> None:
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"{label} must not contain symlinks: {path}")


def _new_output_directory(path: Path) -> Path:
    if path.is_symlink():
        raise ValueError("aggregate output must not be a symlink")
    if os.path.lexists(path):
        raise FileExistsError(f"aggregate output already exists: {path}")
    if path.name in {"", ".", ".."}:
        raise ValueError("aggregate output name is invalid")
    parent = _regular_directory(path.parent, "aggregate output parent")
    destination = parent / path.name
    if os.path.lexists(destination):
        raise FileExistsError(f"aggregate output already exists: {destination}")
    return destination


def _write_exclusive(path: Path, encoded: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _csv_bytes(records: Sequence[Mapping[str, object]]) -> bytes:
    if not records:
        raise ValueError("aggregate CSV requires records")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=_CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for record in records:
        if not set(record).issubset(_CSV_FIELDS):
            raise ValueError("aggregate record contains a field outside fixed CSV schema")
        writer.writerow(
            {field: _csv_value(record.get(field)) for field in _CSV_FIELDS}
        )
    return stream.getvalue().encode("utf-8")


def _csv_value(value: object) -> object:
    if isinstance(value, Mapping) or isinstance(value, (list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if isinstance(value, str):
        if value.startswith(("=", "+", "-", "@")) or any(
            ord(character) < 32 or ord(character) == 127 for character in value
        ):
            raise ValueError("aggregate CSV contains formula/control text")
    return value


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _ordered_ids_sha256(values: Sequence[str]) -> str:
    encoded = json.dumps(list(values), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _integer(value: object, label: str, *, allow_negative: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if not allow_negative and value < 0:
        raise ValueError(f"{label} must be non-negative")
    return value


def _finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _check_percent(value: object, numerator: int, denominator: int, label: str) -> None:
    expected = _percent(numerator, denominator)
    if expected is None:
        if value is not None:
            raise ValueError(f"{label} must be null for a zero denominator")
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{label} percentage does not match exact counts")


def _percent(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else 100.0 * numerator / denominator


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _count_cell(count: int, total: int, *, bold: bool = False) -> str:
    percent = _percent(count, total)
    value = f"{count}/{total} (n/a)" if percent is None else f"{count}/{total} ({percent:.2f}%)"
    return f"**{value}**" if bold else value


def _score_cell(row: Mapping[str, object], *, bold: bool = False) -> str:
    return _count_cell(int(row["correct"]), int(row["total"]), bold=bold)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join((head, divider, *body))
