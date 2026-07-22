#!/usr/bin/env python3
"""Run the sealed 3-model x 30-scenario x 5-decoder fairness matrix."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "experiments" / "matching_fairness"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from matching_fairness.artifacts import (  # noqa: E402
    ScoreArtifact,
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
from matching_fairness.native_export import (  # noqa: E402
    _formal_artifact_inventory,
    _score_artifact_sha256,
)
from matching_fairness.provenance import sha256_file  # noqa: E402
from matching_fairness.scenarios import (  # noqa: E402
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
_EXPECTED_RECORDS = 3 * 30 * 5


@dataclass(frozen=True)
class ScenarioCell:
    suite: str
    index: int
    slug: str
    artifact: ScoreArtifact


def _scenario_artifacts(
    standard: ScoreArtifact,
    eeg_a: ScoreArtifact,
    eeg_b: ScoreArtifact,
    *,
    seed: int,
) -> tuple[ScenarioCell, ...]:
    """Build the exact shared 27 standard plus 3 real-repeat cells."""

    manifest = build_standard_manifest(standard.gallery_canonical_ids, seed=seed)
    cells = [
        ScenarioCell(
            suite="standard",
            index=index,
            slug=scenario.slug,
            artifact=apply_standard_scenario(standard, manifest, scenario),
        )
        for index, scenario in enumerate(standard_scenarios())
    ]
    repeated = select_duplicate_image_ids(
        standard.gallery_canonical_ids,
        seed=seed,
    )
    for offset, count in enumerate((0, 10, 20), start=27):
        cells.append(
            ScenarioCell(
                suite="duplicate_eeg",
                index=offset,
                slug=f"dupq{count}",
                artifact=build_duplicate_query_artifact(
                    eeg_a,
                    eeg_b,
                    repeated,
                    count,
                ),
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


def _validate_record_grid(records: list[dict[str, object]]) -> None:
    """Fail closed unless every formal model/scenario/decoder cell is unique."""

    if len(records) != _EXPECTED_RECORDS:
        raise ValueError(
            f"formal fairness run requires exactly 450 decoder records; got {len(records)}"
        )
    keys = [
        (record.get("model"), record.get("scenario_index"), record.get("decoder"))
        for record in records
    ]
    if len(set(keys)) != _EXPECTED_RECORDS:
        raise ValueError("formal decoder records must use 450 unique grid keys")
    expected = {
        (model, scenario_index, decoder)
        for model in _MODEL_ORDER
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
) -> int:
    """Validate sealed inputs, evaluate all cells, then atomically publish."""

    protocol = Protocol.load(Path(protocol_path))
    protocol.assert_formal_scope()
    artifacts, artifact_hashes = _load_formal_artifacts(
        Path(artifact_root),
        trial_manifest_path=Path(trial_manifest_path),
        expected_image_count=200,
    )
    common_gallery = artifacts["nice"]["standard"].gallery_canonical_ids
    for model in _MODEL_ORDER:
        if artifacts[model]["standard"].gallery_canonical_ids != common_gallery:
            raise ValueError("formal models must share exact canonical gallery order")

    configs = tuple(
        DecoderConfig(
            name=name,
            seed=protocol.seed,
            temperature=float(protocol.sinkhorn["temperature"]),
            max_iterations=int(protocol.sinkhorn["max_iterations"]),
            tolerance=float(protocol.sinkhorn["tolerance"]),
        )
        for name in DECODER_NAMES
    )
    output_dir = Path(output_dir)
    if os.path.lexists(output_dir):
        raise FileExistsError(f"scenario output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.tmp-",
            dir=output_dir.parent,
        )
    )
    records: list[dict[str, object]] = []
    try:
        for model in _MODEL_ORDER:
            cells = _scenario_artifacts(
                artifacts[model]["standard"],
                artifacts[model]["a"],
                artifacts[model]["b"],
                seed=protocol.seed,
            )
            for cell in cells:
                evaluations = tuple(
                    evaluate_artifact(cell.artifact, config) for config in configs
                )
                cell_records = [
                    _result_record(model, cell, result) for result in evaluations
                ]
                records.extend(cell_records)
                _write_cell(
                    staging,
                    model=model,
                    subject=protocol.subject,
                    seed=protocol.seed,
                    cell=cell,
                    evaluations=evaluations,
                    records=cell_records,
                    source_hashes=_cell_source_hashes(
                        cell,
                        artifact_hashes[model],
                    ),
                )
        _validate_record_grid(records)
        summaries = tuple(staging.rglob("summary.json"))
        ledgers = tuple(staging.rglob("per_query.csv"))
        if len(summaries) != 90 or len(ledgers) != 90:
            raise RuntimeError("formal run must contain exactly 90 JSON/CSV result pairs")
        publish_staged_directory(staging, output_dir)
    finally:
        if os.path.lexists(staging):
            shutil.rmtree(staging)
    return len(records)


def _load_formal_artifacts(
    artifact_root: Path,
    *,
    trial_manifest_path: Path,
    expected_image_count: int,
) -> tuple[
    dict[str, dict[str, ScoreArtifact]],
    dict[str, dict[str, str]],
]:
    if artifact_root.is_symlink() or not artifact_root.is_dir():
        raise ValueError("formal artifact root must be a regular directory")
    if set(path.name for path in artifact_root.iterdir()) != set(_MODEL_ORDER):
        raise ValueError("formal artifact root must contain exactly three model dirs")
    directories: list[Path] = []
    for model in _MODEL_ORDER:
        model_dir = artifact_root / model
        if model_dir.is_symlink() or not model_dir.is_dir():
            raise ValueError(f"formal model directory is invalid: {model}")
        if set(path.name for path in model_dir.iterdir()) != set(
            _HALF_DIRECTORIES.values()
        ):
            raise ValueError(f"formal model {model} must contain exactly three artifacts")
        directories.extend(
            model_dir / directory for directory in _HALF_DIRECTORIES.values()
        )

    inventory = _formal_artifact_inventory(
        directories,
        expected_image_count=expected_image_count,
    )
    if len(inventory) != 9:
        raise ValueError("strict Task 6 inventory did not return exactly nine artifacts")
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
        trial_text = trial_manifest_path.read_text(encoding="utf-8")
        trial_manifest = json.loads(trial_text)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("trial manifest must be valid UTF-8 JSON") from error
    common_gallery = artifacts["nice"]["standard"].gallery_canonical_ids
    validate_trial_manifest(trial_manifest, common_gallery)
    trial_hash = sha256_file(trial_manifest_path)
    for model in _MODEL_ORDER:
        for artifact in artifacts[model].values():
            if artifact.metadata.get("trial_manifest_sha256") != trial_hash:
                raise ValueError("formal artifact does not bind the supplied trial manifest")
            if artifact.gallery_canonical_ids != common_gallery:
                raise ValueError("all nine artifacts must share canonical gallery order")
    return artifacts, hashes


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
) -> None:
    if subject != "sub-08":
        raise ValueError("formal scenario output requires subject sub-08")
    cell_dir = (
        staging
        / "runs"
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
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = run_scenarios(
        protocol_path=args.protocol.resolve(),
        artifact_root=args.artifact_root.resolve(),
        trial_manifest_path=args.trial_manifest.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(f"published {count} decoder records to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
