#!/usr/bin/env python3
"""Run the sealed 3-model x 30-scenario x 5-decoder fairness matrix."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
from types import MappingProxyType
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = PROJECT_ROOT / "experiments" / "matching_fairness"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from matching_fairness.artifacts import (  # noqa: E402
    ScoreArtifact,
    independent_ranks,
    publish_staged_directory,
    read_score_artifact,
)
from matching_fairness.config import Protocol  # noqa: E402
from matching_fairness.evaluation import (  # noqa: E402
    DECODER_NAMES,
    DecoderConfig,
    EvaluationResult,
    evaluate_artifact,
)
from matching_fairness.formal_artifacts import (  # noqa: E402
    validate_brainrw_export_tree,
)
from matching_fairness.native_export import (  # noqa: E402
    _formal_model_identity,
    _formal_artifact_inventory,
    _score_artifact_sha256,
    _validate_formal_artifact_provenance,
)
from matching_fairness.scenarios import (  # noqa: E402
    ScenarioSpec,
    apply_standard_scenario,
    build_duplicate_query_artifact,
    build_standard_manifest,
    standard_scenarios,
)
from matching_fairness.trial_splits import (  # noqa: E402
    select_duplicate_image_ids,
    validate_trial_manifest,
)


_MODEL_ORDER = ("nice", "atm_s", "our_project")
_HALF_DIRECTORIES = {
    "standard": "standard",
    "a": "eeg_a",
    "b": "eeg_b",
}
_RUN_MANIFEST_ALGORITHM = "AIAA3800-MATCHING-FAIRNESS-RUN-v1"


def _expected_model_entries(model: str) -> set[str]:
    if model not in _MODEL_ORDER:
        raise ValueError(f"unsupported formal artifact model: {model}")
    entries = set(_HALF_DIRECTORIES.values())
    if model in {"nice", "atm_s"}:
        entries.add("best_test_audit.json")
    else:
        entries.update({"runs", "export_manifest.json"})
    return entries


@dataclass(frozen=True)
class ScenarioCell:
    suite: str
    index: int
    slug: str
    artifact: ScoreArtifact
    selection: Mapping[str, object]


@dataclass(frozen=True)
class ScenarioPlan:
    """One immutable model-independent selection and provenance plan."""

    standard_manifest: Mapping[str, object]
    standard_selections: tuple[Mapping[str, object], ...]
    duplicate_query_ids: tuple[str, ...]
    duplicate_selections: tuple[Mapping[str, object], ...]
    manifest_bytes: bytes
    manifest_sha256: str


def _decoder_configs(
    *,
    seed: int,
    sinkhorn: Mapping[str, object],
) -> tuple[DecoderConfig, ...]:
    return tuple(
        DecoderConfig(
            name=name,
            seed=seed,
            temperature=float(sinkhorn["temperature"]),
            max_iterations=int(sinkhorn["max_iterations"]),
            tolerance=float(sinkhorn["tolerance"]),
        )
        for name in DECODER_NAMES
    )


def _build_scenario_plan(
    gallery_canonical_ids: Sequence[str],
    *,
    seed: int,
    trial_manifest_sha256: str,
    decoder_configs: Sequence[DecoderConfig],
) -> ScenarioPlan:
    """Build and hash the single canonical selection used by every model."""

    gallery_ids = tuple(gallery_canonical_ids)
    if (
        not isinstance(trial_manifest_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", trial_manifest_sha256) is None
    ):
        raise ValueError("trial manifest SHA-256 is invalid")
    if tuple(config.name for config in decoder_configs) != DECODER_NAMES:
        raise ValueError("scenario plan requires the exact five decoder configs")
    mutable_master = build_standard_manifest(gallery_ids, seed=seed)
    standard_manifest = MappingProxyType(
        {
            key: tuple(value) if isinstance(value, list) else value
            for key, value in mutable_master.items()
        }
    )
    selections = tuple(
        _standard_selection(standard_manifest, scenario)
        for scenario in standard_scenarios()
    )
    duplicate_query_ids = select_duplicate_image_ids(gallery_ids, seed=seed)
    duplicate_selections = tuple(
        MappingProxyType(
            {
                "duplicate_query_ids": duplicate_query_ids[:count],
                "duplicate_query_count": count,
            }
        )
        for count in (0, 10, 20)
    )
    decoder_payload = []
    for config in decoder_configs:
        entry: dict[str, object] = {"name": config.name}
        if config.name == "hungarian":
            entry["seed"] = config.seed
        elif config.name == "sinkhorn":
            entry.update(
                {
                    "temperature": config.temperature,
                    "max_iterations": config.max_iterations,
                    "tolerance": config.tolerance,
                }
            )
        decoder_payload.append(entry)
    payload = {
        "schema_version": 1,
        "algorithm_version": _RUN_MANIFEST_ALGORITHM,
        "seed": seed,
        "gallery_canonical_ids": list(gallery_ids),
        "gallery_canonical_ids_sha256": _ordered_ids_sha256(gallery_ids),
        "standard_master": {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in standard_manifest.items()
        },
        "standard_scenarios": [
            {
                "scenario_index": index,
                "scenario": scenario.slug,
                "parameters": {
                    "drop_query": scenario.drop_query,
                    "drop_gallery": scenario.drop_gallery,
                    "drop_pair": scenario.drop_pair,
                    "duplicate_gallery": scenario.duplicate_gallery,
                },
                "selected_canonical_ids": _jsonable_selection(selections[index]),
            }
            for index, scenario in enumerate(standard_scenarios())
        ],
        "duplicate_query": {
            "ordered_ids": list(duplicate_query_ids),
            "counts": [0, 10, 20],
            "selected_by_count": {
                str(count): list(duplicate_query_ids[:count])
                for count in (0, 10, 20)
            },
        },
        "trial_manifest_sha256": trial_manifest_sha256,
        "decoder_configs": decoder_payload,
    }
    manifest_bytes = _json_bytes(payload)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    _validate_scenario_manifest(manifest_bytes, manifest_sha256)
    return ScenarioPlan(
        standard_manifest=standard_manifest,
        standard_selections=selections,
        duplicate_query_ids=duplicate_query_ids,
        duplicate_selections=duplicate_selections,
        manifest_bytes=manifest_bytes,
        manifest_sha256=manifest_sha256,
    )


def _standard_selection(
    manifest: Mapping[str, object],
    scenario: ScenarioSpec,
) -> Mapping[str, object]:
    dropped_gallery = set(manifest["drop_gallery"][: scenario.drop_gallery])
    duplicate_gallery = tuple(
        canonical_id
        for canonical_id in manifest["duplicate_gallery"]
        if canonical_id not in dropped_gallery
    )[: scenario.duplicate_gallery]
    return MappingProxyType(
        {
            "drop_query": tuple(manifest["drop_query"][: scenario.drop_query]),
            "drop_gallery": tuple(
                manifest["drop_gallery"][: scenario.drop_gallery]
            ),
            "drop_pair": tuple(manifest["drop_pair"][: scenario.drop_pair]),
            "duplicate_gallery": duplicate_gallery,
        }
    )


def _jsonable_selection(selection: Mapping[str, object]) -> dict[str, object]:
    return {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in selection.items()
    }


def _ordered_ids_sha256(values: Sequence[str]) -> str:
    encoded = json.dumps(
        list(values),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_scenario_manifest(
    manifest_bytes: bytes,
    expected_sha256: str,
) -> Mapping[str, object]:
    if (
        not isinstance(expected_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
        or hashlib.sha256(manifest_bytes).hexdigest() != expected_sha256
    ):
        raise ValueError("scenario manifest SHA-256 mismatch")
    try:
        payload = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("scenario manifest must be valid UTF-8 JSON") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("algorithm_version") != _RUN_MANIFEST_ALGORITHM
        or _json_bytes(payload) != manifest_bytes
    ):
        raise ValueError("scenario manifest is not the canonical formal schema")
    expected_keys = {
        "schema_version",
        "algorithm_version",
        "seed",
        "gallery_canonical_ids",
        "gallery_canonical_ids_sha256",
        "standard_master",
        "standard_scenarios",
        "duplicate_query",
        "trial_manifest_sha256",
        "decoder_configs",
    }
    gallery_ids = payload.get("gallery_canonical_ids")
    if (
        set(payload) != expected_keys
        or payload.get("seed") != 42
        or not isinstance(gallery_ids, list)
        or len(gallery_ids) != 200
        or any(not isinstance(value, str) or not value for value in gallery_ids)
        or len(set(gallery_ids)) != 200
        or payload.get("gallery_canonical_ids_sha256")
        != _ordered_ids_sha256(gallery_ids)
    ):
        raise ValueError("scenario manifest gallery identity is invalid")
    expected_master = build_standard_manifest(tuple(gallery_ids), seed=42)
    if payload.get("standard_master") != expected_master:
        raise ValueError("scenario manifest standard master is invalid")
    frozen_master = MappingProxyType(
        {
            key: tuple(value) if isinstance(value, list) else value
            for key, value in expected_master.items()
        }
    )
    expected_standard = []
    for index, scenario in enumerate(standard_scenarios()):
        expected_standard.append(
            {
                "scenario_index": index,
                "scenario": scenario.slug,
                "parameters": {
                    "drop_query": scenario.drop_query,
                    "drop_gallery": scenario.drop_gallery,
                    "drop_pair": scenario.drop_pair,
                    "duplicate_gallery": scenario.duplicate_gallery,
                },
                "selected_canonical_ids": _jsonable_selection(
                    _standard_selection(frozen_master, scenario)
                ),
            }
        )
    if payload.get("standard_scenarios") != expected_standard:
        raise ValueError("scenario manifest standard selections are invalid")
    duplicate_ids = select_duplicate_image_ids(tuple(gallery_ids), seed=42)
    expected_duplicate = {
        "ordered_ids": list(duplicate_ids),
        "counts": [0, 10, 20],
        "selected_by_count": {
            str(count): list(duplicate_ids[:count]) for count in (0, 10, 20)
        },
    }
    if payload.get("duplicate_query") != expected_duplicate:
        raise ValueError("scenario manifest duplicate-query selection is invalid")
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
    if payload.get("decoder_configs") != expected_decoders:
        raise ValueError("scenario manifest decoder configuration is invalid")
    trial_manifest_sha256 = payload.get("trial_manifest_sha256")
    if (
        not isinstance(trial_manifest_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", trial_manifest_sha256) is None
    ):
        raise ValueError("scenario manifest trial-manifest hash is invalid")
    return payload


def _scenario_artifacts(
    standard: ScoreArtifact,
    eeg_a: ScoreArtifact,
    eeg_b: ScoreArtifact,
    *,
    plan: ScenarioPlan,
) -> tuple[ScenarioCell, ...]:
    """Build the exact shared 27 standard plus 3 real-repeat cells."""

    if tuple(plan.standard_manifest["canonical_image_ids"]) != (
        standard.gallery_canonical_ids
    ):
        raise ValueError("scenario plan gallery does not match model artifact")
    cells: list[ScenarioCell] = []
    for index, scenario in enumerate(standard_scenarios()):
        artifact = apply_standard_scenario(
            standard,
            plan.standard_manifest,
            scenario,
        )
        selection = plan.standard_selections[index]
        if artifact.metadata.get("selected_canonical_ids") != dict(selection):
            raise ValueError("Task 4 scenario selection differs from sealed plan")
        cells.append(
            ScenarioCell(
                suite="standard",
                index=index,
                slug=scenario.slug,
                artifact=artifact,
                selection=selection,
            )
        )
    for selection_index, count in enumerate((0, 10, 20)):
        offset = 27 + selection_index
        selection = plan.duplicate_selections[selection_index]
        cells.append(
            ScenarioCell(
                suite="duplicate_eeg",
                index=offset,
                slug=f"dupq{count}",
                artifact=build_duplicate_query_artifact(
                    eeg_a,
                    eeg_b,
                    plan.duplicate_query_ids,
                    count,
                ),
                selection=selection,
            )
        )
    if (
        len(cells) != 30
        or sum(cell.suite == "standard" for cell in cells) != 27
        or sum(cell.suite == "duplicate_eeg" for cell in cells) != 3
        or [cell.index for cell in cells] != list(range(30))
    ):
        raise RuntimeError("scenario builder did not produce the formal 27+3 grid")
    return tuple(cells)


def _validate_record_grid(
    records: list[dict[str, object]],
    *,
    models: Sequence[str] = _MODEL_ORDER,
) -> None:
    """Fail closed unless every formal model/scenario/decoder cell is unique."""

    selected_models = tuple(models)
    if (
        not selected_models
        or len(set(selected_models)) != len(selected_models)
        or any(model not in _MODEL_ORDER for model in selected_models)
    ):
        raise ValueError("record grid models must be unique formal model slugs")
    expected_records = len(selected_models) * 30 * len(DECODER_NAMES)
    if len(records) != expected_records:
        raise ValueError(
            f"formal fairness run requires exactly {expected_records} decoder records; "
            f"got {len(records)}"
        )
    keys = [
        (record.get("model"), record.get("scenario_index"), record.get("decoder"))
        for record in records
    ]
    if len(set(keys)) != expected_records:
        raise ValueError(
            f"formal decoder records must use {expected_records} unique grid keys"
        )
    expected = {
        (model, scenario_index, decoder)
        for model in selected_models
        for scenario_index in range(30)
        for decoder in DECODER_NAMES
    }
    if set(keys) != expected:
        raise ValueError("formal decoder records do not cover the exact 450-cell grid")


def run_scenarios(
    *,
    protocol_path: Path,
    artifact_root: Path,
    trial_manifest_path: Path,
    output_dir: Path,
    model: str | None = None,
) -> int:
    """Validate sealed inputs, evaluate all cells, then atomically publish."""

    output_dir = _validated_output_directory(Path(output_dir))
    protocol_path = _validated_existing_file(Path(protocol_path), "formal protocol")
    artifact_root = _validated_existing_directory(
        Path(artifact_root),
        "formal artifact root",
    )
    trial_manifest_path = _validated_existing_file(
        Path(trial_manifest_path),
        "trial manifest",
    )
    protocol = Protocol.load(protocol_path)
    protocol.assert_formal_scope()
    if model is not None and model not in _MODEL_ORDER:
        raise ValueError(f"unsupported formal artifact model: {model}")
    selected_models = _MODEL_ORDER if model is None else (model,)
    protocol_sha256 = hashlib.sha256(
        _read_regular_file_nofollow(protocol_path, "formal protocol")
    ).hexdigest()
    if model is None:
        artifacts, artifact_hashes, trial_manifest_sha256 = _load_formal_artifacts(
            artifact_root,
            trial_manifest_path=trial_manifest_path,
            expected_image_count=200,
            protocol_sha256=protocol_sha256,
        )
    else:
        model_artifacts, model_hashes, trial_manifest_sha256 = (
            _load_single_model_artifacts(
                artifact_root,
                model=model,
                trial_manifest_path=trial_manifest_path,
                expected_image_count=200,
                protocol_sha256=protocol_sha256,
            )
        )
        artifacts = {model: model_artifacts}
        artifact_hashes = {model: model_hashes}
    common_gallery = artifacts[selected_models[0]]["standard"].gallery_canonical_ids
    for selected_model in selected_models:
        if artifacts[selected_model]["standard"].gallery_canonical_ids != common_gallery:
            raise ValueError("formal models must share exact canonical gallery order")

    configs = _decoder_configs(
        seed=protocol.seed,
        sinkhorn=protocol.sinkhorn,
    )
    plan = _build_scenario_plan(
        common_gallery,
        seed=protocol.seed,
        trial_manifest_sha256=trial_manifest_sha256,
        decoder_configs=configs,
    )
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.tmp-",
            dir=output_dir.parent,
        )
    )
    records: list[dict[str, object]] = []
    try:
        _write_bytes_exclusive(
            staging / "scenario_manifest.json",
            plan.manifest_bytes,
        )
        for selected_model in selected_models:
            cells = _scenario_artifacts(
                artifacts[selected_model]["standard"],
                artifacts[selected_model]["a"],
                artifacts[selected_model]["b"],
                plan=plan,
            )
            for cell in cells:
                evaluations = tuple(
                    evaluate_artifact(cell.artifact, config) for config in configs
                )
                cell_records = [
                    _result_record(selected_model, cell, result)
                    for result in evaluations
                ]
                records.extend(cell_records)
                _write_cell(
                    staging,
                    model=selected_model,
                    subject=protocol.subject,
                    seed=protocol.seed,
                    cell=cell,
                    evaluations=evaluations,
                    records=cell_records,
                    source_hashes=_cell_source_hashes(
                        cell,
                        artifact_hashes[selected_model],
                    ),
                    scenario_manifest_sha256=plan.manifest_sha256,
                )
        _validate_record_grid(records, models=selected_models)
        manifest_path = staging / "scenario_manifest.json"
        _validate_scenario_manifest(
            manifest_path.read_bytes(),
            plan.manifest_sha256,
        )
        summaries = tuple(staging.rglob("summary.json"))
        ledgers = tuple(staging.rglob("per_query.csv"))
        expected_cells = len(selected_models) * 30
        if len(summaries) != expected_cells or len(ledgers) != expected_cells:
            raise RuntimeError(
                f"formal run must contain exactly {expected_cells} JSON/CSV result pairs"
            )
        publish_staged_directory(staging, output_dir)
    finally:
        if os.path.lexists(staging):
            shutil.rmtree(staging)
    return len(records)


def _validated_existing_directory(path: Path, label: str) -> Path:
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    if not os.path.lexists(path) or not path.is_dir():
        raise ValueError(f"{label} must be an existing regular directory")
    return path.resolve(strict=True)


def _validated_existing_file(path: Path, label: str) -> Path:
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    if not os.path.lexists(path) or not path.is_file():
        raise ValueError(f"{label} must be an existing regular file")
    return path.resolve(strict=True)


def _read_regular_file_nofollow(path: Path, label: str) -> bytes:
    """Read one immutable byte snapshot through a no-follow descriptor."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"could not securely open {label}: {path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        value = b"".join(chunks)
        after = os.fstat(descriptor)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if len(value) != before.st_size or before_identity != after_identity:
            raise ValueError(f"{label} changed while being read")
        return value
    except OSError as error:
        raise ValueError(f"could not securely read {label}: {path}") from error
    finally:
        os.close(descriptor)


def _validated_output_directory(path: Path) -> Path:
    path = Path(path)
    if path.is_symlink():
        raise ValueError("scenario output directory must not be a symlink")
    if os.path.lexists(path):
        raise FileExistsError(f"scenario output already exists: {path}")
    parent = _validated_existing_directory(
        path.parent,
        "scenario output parent",
    )
    if not path.name or path.name in {".", ".."}:
        raise ValueError("scenario output directory name is invalid")
    destination = parent / path.name
    if os.path.lexists(destination):
        raise FileExistsError(f"scenario output already exists: {destination}")
    return destination


def _validate_native_audits(
    artifact_root: Path,
    inventory: Sequence[Mapping[str, object]],
) -> None:
    from matching_fairness.reporting import (
        _canonical_json,
        _read_regular_file,
        _validate_audit_manifest,
        _validate_checkpoint_manifest,
    )

    role_map = {"standard": "standard", "a": "eeg_a", "b": "eeg_b"}
    expected_inventory = {
        (str(entry["model_slug"]), role_map[str(entry["trial_half"])]): str(
            entry["sha256"]
        )
        for entry in inventory
    }
    if len(expected_inventory) != 9:
        raise ValueError("native audits require the exact nine-artifact inventory")
    for model in ("nice", "atm_s"):
        checkpoint_path = (
            artifact_root.parent / "checkpoints" / model / "checkpoint_manifest.json"
        )
        audit_path = artifact_root / model / "best_test_audit.json"
        checkpoint = _canonical_json(
            _read_regular_file(checkpoint_path, "checkpoint manifest"),
            "checkpoint manifest",
        )
        audit = _canonical_json(
            _read_regular_file(audit_path, "best-test audit"),
            "best-test audit",
        )
        _selection, checkpoint_identities = _validate_checkpoint_manifest(
            checkpoint, model
        )
        _best, audit_inventory, audit_identities = _validate_audit_manifest(
            audit, model
        )
        if checkpoint_identities != audit_identities:
            raise ValueError("native audit does not bind its checkpoint manifest")
        if audit_inventory != expected_inventory:
            raise ValueError("native audit does not bind the current nine artifacts")
        if audit.get("formal_artifact_inventory") != list(inventory):
            raise ValueError("native audit inventory ordering/content is not canonical")


def _load_formal_artifacts(
    artifact_root: Path,
    *,
    trial_manifest_path: Path,
    expected_image_count: int,
    protocol_sha256: str,
) -> tuple[
    dict[str, dict[str, ScoreArtifact]],
    dict[str, dict[str, str]],
    str,
]:
    artifact_root = _validated_existing_directory(
        artifact_root,
        "formal artifact root",
    )
    trial_manifest_path = _validated_existing_file(
        trial_manifest_path,
        "trial manifest",
    )
    if re.fullmatch(r"[0-9a-f]{64}", protocol_sha256) is None:
        raise ValueError("formal protocol SHA-256 is invalid")
    if set(path.name for path in artifact_root.iterdir()) != set(_MODEL_ORDER):
        raise ValueError("formal artifact root must contain exactly three model dirs")
    directories: list[Path] = []
    for model in _MODEL_ORDER:
        model_dir = artifact_root / model
        if model_dir.is_symlink() or not model_dir.is_dir():
            raise ValueError(f"formal model directory is invalid: {model}")
        if set(path.name for path in model_dir.iterdir()) != _expected_model_entries(
            model
        ):
            raise ValueError(f"formal model {model} has an invalid exact entry set")
        directories.extend(
            model_dir / directory for directory in _HALF_DIRECTORIES.values()
        )

    trial_bytes = _read_regular_file_nofollow(trial_manifest_path, "trial manifest")
    trial_hash = hashlib.sha256(trial_bytes).hexdigest()
    brainrw = validate_brainrw_export_tree(
        artifact_root / "our_project",
        expected_image_count=expected_image_count,
    )
    if (
        brainrw.inputs.get("protocol_sha256") != protocol_sha256
        or brainrw.inputs.get("trial_manifest_sha256") != trial_hash
    ):
        raise ValueError("BrainRW export does not bind the supplied protocol/trials")

    inventory = _formal_artifact_inventory(
        directories,
        expected_image_count=expected_image_count,
    )
    if len(inventory) != 9:
        raise ValueError("strict Task 6 inventory did not return exactly nine artifacts")
    _validate_native_audits(artifact_root, inventory)
    artifacts: dict[str, dict[str, ScoreArtifact]] = {
        model: {} for model in _MODEL_ORDER
    }
    hashes: dict[str, dict[str, str]] = {model: {} for model in _MODEL_ORDER}
    for entry in inventory:
        model = str(entry["model_slug"])
        half = str(entry["trial_half"])
        artifact = read_score_artifact(Path(str(entry["path"])))
        artifacts[model][half] = artifact
        actual_hash = _score_artifact_sha256(Path(str(entry["path"])))
        if entry["sha256"] != actual_hash:
            raise ValueError("formal artifact inventory hash changed after validation")
        hashes[model][half] = actual_hash
    if any(set(artifacts[model]) != {"standard", "a", "b"} for model in _MODEL_ORDER):
        raise ValueError("strict Task 6 inventory roles are incomplete")

    try:
        trial_manifest = json.loads(trial_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("trial manifest must be valid UTF-8 JSON") from error
    common_gallery = artifacts["nice"]["standard"].gallery_canonical_ids
    validate_trial_manifest(trial_manifest, common_gallery)
    for model in _MODEL_ORDER:
        for artifact in artifacts[model].values():
            if artifact.metadata.get("trial_manifest_sha256") != trial_hash:
                raise ValueError("formal artifact does not bind the supplied trial manifest")
            if artifact.gallery_canonical_ids != common_gallery:
                raise ValueError("all nine artifacts must share canonical gallery order")
    return artifacts, hashes, trial_hash


def _load_single_model_artifacts(
    artifact_root: Path,
    *,
    model: str,
    trial_manifest_path: Path,
    expected_image_count: int,
    protocol_sha256: str,
) -> tuple[dict[str, ScoreArtifact], dict[str, str], str]:
    """Strictly load one baseline without requiring unrelated baseline outputs."""

    if model not in _MODEL_ORDER:
        raise ValueError(f"unsupported formal artifact model: {model}")
    artifact_root = _validated_existing_directory(artifact_root, "formal artifact root")
    trial_manifest_path = _validated_existing_file(trial_manifest_path, "trial manifest")
    if re.fullmatch(r"[0-9a-f]{64}", protocol_sha256) is None:
        raise ValueError("formal protocol SHA-256 is invalid")
    model_dir = artifact_root / model
    if model_dir.is_symlink() or not model_dir.is_dir():
        raise ValueError(f"formal model directory is invalid: {model}")
    trial_bytes = _read_regular_file_nofollow(trial_manifest_path, "trial manifest")
    trial_hash = hashlib.sha256(trial_bytes).hexdigest()
    artifacts: dict[str, ScoreArtifact] = {}
    hashes: dict[str, str] = {}
    common_gallery: tuple[str, ...] | None = None
    model_identity: dict[str, object] | None = None
    expected_role = "fixed_formal" if model == "our_project" else "val_selected_formal"
    for half, directory in _HALF_DIRECTORIES.items():
        path = model_dir / directory
        artifact = read_score_artifact(path)
        metadata = artifact.metadata
        if (
            artifact.similarity.shape != (expected_image_count, expected_image_count)
            or metadata.get("model_slug") != model
            or metadata.get("trial_half") != half
            or metadata.get("checkpoint_role") != expected_role
            or metadata.get("subject") != "sub-08"
            or metadata.get("seed") != 42
            or metadata.get("trial_manifest_sha256") != trial_hash
            or not (
                artifact.query_ids
                == artifact.target_canonical_ids
                == artifact.gallery_entry_ids
                == artifact.gallery_canonical_ids
            )
        ):
            raise ValueError(f"formal artifact identity mismatch: {(model, half)}")
        ranks = independent_ranks(artifact)
        expected_metrics = {
            "top1_count": int((ranks <= 1).sum()),
            "top5_count": int((ranks <= 5).sum()),
            "sample_count": len(ranks),
        }
        if metadata.get("native_metrics") != expected_metrics:
            raise ValueError(f"formal artifact metric parity failed: {(model, half)}")
        _validate_formal_artifact_provenance(
            artifact, model, expected_image_count=expected_image_count
        )
        if model == "our_project" and metadata.get("protocol_sha256") != protocol_sha256:
            raise ValueError("BrainRW artifact does not bind the supplied protocol")
        identity = _formal_model_identity(metadata, model)
        if model_identity is None:
            model_identity = identity
        elif identity != model_identity:
            raise ValueError("model provenance must be identical across halves")
        if common_gallery is None:
            common_gallery = artifact.gallery_canonical_ids
        elif artifact.gallery_canonical_ids != common_gallery:
            raise ValueError("model artifacts do not share canonical gallery order")
        artifacts[half] = artifact
        hashes[half] = _score_artifact_sha256(path)
    if (
        artifacts["a"].metadata.get("query_embeddings_sha256")
        == artifacts["b"].metadata.get("query_embeddings_sha256")
    ):
        raise ValueError("EEG-A and EEG-B query embeddings are identical")
    try:
        trial_manifest = json.loads(trial_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("trial manifest must be valid UTF-8 JSON") from error
    assert common_gallery is not None
    validate_trial_manifest(trial_manifest, common_gallery)
    return artifacts, hashes, trial_hash


def _result_record(
    model: str,
    cell: ScenarioCell,
    result: EvaluationResult,
) -> dict[str, object]:
    return {
        "model": model,
        "suite": cell.suite,
        "scenario_index": cell.index,
        "scenario": cell.slug,
        **dict(result.metrics),
    }


def _cell_source_hashes(
    cell: ScenarioCell,
    hashes: dict[str, str],
) -> dict[str, str]:
    if cell.suite == "standard":
        return {"standard": hashes["standard"]}
    return {"eeg_a": hashes["a"], "eeg_b": hashes["b"]}


def _write_cell(
    staging: Path,
    *,
    model: str,
    subject: str,
    seed: int,
    cell: ScenarioCell,
    evaluations: tuple[EvaluationResult, ...],
    records: list[dict[str, object]],
    source_hashes: dict[str, str],
    scenario_manifest_sha256: str,
) -> None:
    if subject != "sub-08":
        raise ValueError("formal scenario output requires subject sub-08")
    cell_dir = (
        staging
        / model
        / "subj08"
        / f"seed{seed}"
        / cell.suite
        / f"{cell.index:02d}_{cell.slug}"
    )
    cell_dir.mkdir(parents=True, exist_ok=False)
    payload = {
        "schema_version": 1,
        "model": model,
        "subject": subject,
        "seed": seed,
        "suite": cell.suite,
        "scenario_index": cell.index,
        "scenario": cell.slug,
        "matrix_shape": [int(value) for value in cell.artifact.similarity.shape],
        "scenario_selection": _jsonable_selection(cell.selection),
        "scenario_manifest_sha256": scenario_manifest_sha256,
        "source_artifact_sha256": source_hashes,
        "decoder_records": records,
    }
    _write_json_exclusive(cell_dir / "summary.json", payload)
    rows: list[dict[str, Any]] = []
    for result in evaluations:
        for query in result.per_query:
            rows.append(
                {
                    "model": model,
                    "subject": subject,
                    "seed": seed,
                    "suite": cell.suite,
                    "scenario_index": cell.index,
                    "scenario": cell.slug,
                    "decoder": result.decoder,
                    **dict(query),
                }
            )
    _write_csv_exclusive(cell_dir / "per_query.csv", rows)


def _write_json_exclusive(path: Path, payload: object) -> None:
    encoded = _json_bytes(payload)
    _write_bytes_exclusive(path, encoded)


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_bytes_exclusive(path: Path, encoded: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _write_csv_exclusive(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("per-query CSV requires at least one row")
    fieldnames = list(rows[0])
    if any(list(row) != fieldnames for row in rows):
        raise ValueError("per-query CSV rows do not share a stable schema")
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PACKAGE_ROOT / "configs" / "protocol_sub08_seed42.json",
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--trial-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=_MODEL_ORDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = run_scenarios(
        protocol_path=args.protocol,
        artifact_root=args.artifact_root,
        trial_manifest_path=args.trial_manifest,
        output_dir=args.output_dir,
        model=args.model,
    )
    print(f"published {count} decoder records to {args.output_dir}")


if __name__ == "__main__":
    main()
