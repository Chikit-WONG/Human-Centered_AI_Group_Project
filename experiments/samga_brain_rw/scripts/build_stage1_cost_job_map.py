#!/usr/bin/env python3
"""Build the single sealed A40 Stage 1 branch-cost benchmark job map."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

import build_job_map as job_maps
from samga_brain_rw.config import make_run_key
from samga_brain_rw.cost_capability import (
    load_stage1_cost_execution_plan,
    load_stage1_cost_model_manifest,
    load_stage1_cost_score_input_manifest,
    stable_regular_file_sha256,
)
from samga_brain_rw.hashing import canonical_json_bytes, sha256_json
from samga_brain_rw.inference_cost import load_cost_protocol


_CONFIG_RELATIVE = Path(
    "experiments/samga_brain_rw/configs/stage1_cost_v1.json"
)
_EXECUTION_CONFIG_RELATIVE = Path(
    "experiments/samga_brain_rw/configs/stage1_cost_execution_v1.json"
)
_RUNNER_RELATIVE = Path(
    "experiments/samga_brain_rw/scripts/run_stage1_cost.py"
)
_INPUT_ROOT_RELATIVE = Path(
    "artifacts/samga_brain_rw/stage-1-cost-inputs"
)
_OUTPUT_ROOT_RELATIVE = Path(
    "artifacts/samga_brain_rw/stage-1-cost-benchmark"
)
_STAGE = "stage-1-cost-benchmark"
_CONFIG_ID = "stage1_cost_v1"
_SUBJECT = 1
_SEED = 20260720


def _absolute_normalized(path: Path, context: str) -> Path:
    raw = os.fspath(path)
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise ValueError(f"{context} must be a non-empty path")
    absolute = Path(os.path.abspath(os.path.normpath(raw)))
    if not Path(raw).is_absolute() or Path(raw) != absolute:
        raise ValueError(f"{context} must be absolute and normalized")
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{context} cannot be resolved") from exc
    if resolved != absolute:
        raise ValueError(f"{context} contains a symlink component")
    return absolute


def _canonical_job_map_output(project_root: Path, path: Path) -> Path:
    raw = os.fspath(path)
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise ValueError("cost job map must be a non-empty path")
    output = Path(os.path.abspath(os.path.normpath(raw)))
    if not Path(raw).is_absolute() or Path(raw) != output:
        raise ValueError("cost job map must be absolute and normalized")
    fixed_parent = project_root / "artifacts/samga_brain_rw/job_maps"
    if output.parent != fixed_parent:
        raise ValueError(
            f"cost job map must be a direct child of {fixed_parent}"
        )
    return output


def _publish_stage1_cost_job_map(
    rows: Sequence[dict[str, object]],
    *,
    project_root: Path,
    output: Path,
) -> dict[str, object]:
    payload = job_maps.build_job_map(rows)
    fixed_parent = project_root / "artifacts/samga_brain_rw/job_maps"
    try:
        directory_fd = job_maps._open_directory_path_nofollow(
            fixed_parent,
            create=True,
        )
    except (OSError, ValueError) as exc:
        raise ValueError(
            "cost job-map parent is symbolic or unavailable"
        ) from exc
    try:
        job_maps._exclusive_publish_at(
            directory_fd,
            output.name,
            canonical_json_bytes(payload),
        )
        try:
            observed_fd = job_maps._open_directory_path_nofollow(
                fixed_parent,
                create=False,
            )
        except (OSError, ValueError) as exc:
            raise ValueError(
                "cost job-map parent became symbolic or changed"
            ) from exc
        try:
            expected = os.fstat(directory_fd)
            observed = os.fstat(observed_fd)
            if (expected.st_dev, expected.st_ino) != (
                observed.st_dev,
                observed.st_ino,
            ):
                raise ValueError("cost job-map parent identity changed")
        finally:
            os.close(observed_fd)
    finally:
        os.close(directory_fd)
    return payload


def _validate_locked_workload(protocol: object) -> None:
    workload = getattr(protocol, "synthetic_workload", None)
    if not isinstance(workload, dict):
        raise ValueError("cost protocol lacks the synthetic workload")
    if workload.get("seed") != _SEED:
        raise ValueError("cost synthetic workload seed mismatch")
    if (
        workload.get("query_count") != 200
        or workload.get("gallery_count") != 200
    ):
        raise ValueError("cost synthetic workload must be exactly 200x200")
    if workload.get("labels_present") is not False:
        raise ValueError("cost synthetic workload must not contain labels")
    if workload.get("metrics_computed") is not False:
        raise ValueError("cost synthetic workload must not compute metrics")
    if (
        getattr(protocol, "warmup_runs", None) != 10
        or getattr(protocol, "measured_runs", None) != 50
        or getattr(protocol, "mad_ratio_max", None) != 0.05
    ):
        raise ValueError("cost benchmark must use 10 warmups, 50 runs, MAD 0.05")


def build_stage1_cost_rows(
    *,
    project_root: Path,
) -> list[dict[str, object]]:
    """Return the one immutable raw row for the exact cost benchmark."""

    root = _absolute_normalized(project_root, "project root")
    if not root.is_dir() or not (root / ".git").exists():
        raise ValueError("project root must be the repository root")
    config_path = root / _CONFIG_RELATIVE
    execution_path = root / _EXECUTION_CONFIG_RELATIVE
    protocol = load_cost_protocol(config_path)
    execution = load_stage1_cost_execution_plan(execution_path)
    if getattr(protocol, "config_id", None) != _CONFIG_ID:
        raise ValueError("cost protocol config_id mismatch")
    if getattr(execution, "config_id", None) != "stage1_cost_execution_v1":
        raise ValueError("cost execution config_id mismatch")
    _validate_locked_workload(protocol)
    if (
        execution.seed != 20260720
        or execution.query_count != 200
        or execution.gallery_count != 200
        or execution.labels_present is not False
        or execution.metrics_computed is not False
        or execution.branch_order != ("internvit", "brainrw")
    ):
        raise ValueError("cost execution plan differs from the fixed workload")

    score_path = _absolute_normalized(
        root / _INPUT_ROOT_RELATIVE / "score-inputs.json",
        "score-input manifest",
    )
    model_path = _absolute_normalized(
        root / _INPUT_ROOT_RELATIVE / "model-manifest.json",
        "model manifest",
    )
    score_document, score_file_sha256 = (
        load_stage1_cost_score_input_manifest(score_path)
    )
    model_document, model_file_sha256 = load_stage1_cost_model_manifest(
        model_path
    )
    if (
        not isinstance(score_document.get("score_inputs"), list)
        or len(score_document["score_inputs"]) != 6
    ):
        raise ValueError("cost score-input manifest must bind six cells")
    branches = model_document.get("branches")
    if not isinstance(branches, dict) or set(branches) != {
        "internvit",
        "brainrw",
    }:
        raise ValueError("cost model manifest must bind two branches")

    runner_path = root / _RUNNER_RELATIVE
    runner_file_sha256 = stable_regular_file_sha256(runner_path)
    input_bundle_sha256 = sha256_json(
        {
            "execution_config_sha256": execution.sha256,
            "model_manifest_file_sha256": model_file_sha256,
            "runner_file_sha256": runner_file_sha256,
            "score_inputs_file_sha256": score_file_sha256,
        }
    )
    config_sha256 = str(getattr(protocol, "sha256", ""))
    run_key = make_run_key(
        "stage1-cost",
        _CONFIG_ID,
        _SUBJECT,
        _SEED,
        config_sha256,
        input_bundle_sha256,
    )
    output_dir = root / _OUTPUT_ROOT_RELATIVE / run_key
    argv = [
        "python",
        str(runner_path),
        "--subject",
        str(_SUBJECT),
        "--seed",
        str(_SEED),
        "--config",
        str(config_path),
        "--execution-config",
        str(execution_path),
        "--score-inputs",
        str(score_path),
        "--model-manifest",
        str(model_path),
        "--output-dir",
        str(output_dir),
        "--project-root",
        str(root),
        "--config-id",
        _CONFIG_ID,
        "--expected-config-sha256",
        config_sha256,
        "--expected-execution-config-sha256",
        execution.sha256,
        "--expected-input-bundle-sha256",
        input_bundle_sha256,
        "--run-key",
        run_key,
        "--device",
        "cuda",
    ]
    return [
        {
            "argv": argv,
            "completion_path": str(output_dir / "completion.json"),
            "config_id": _CONFIG_ID,
            "config_sha256": config_sha256,
            "cpus": 16,
            "expected_completion_schema": {
                "payload_type": "samga_brain_rw.stage1_cost_completion",
                "required_output_hashes": [
                    "raw_record_file_sha256",
                    "run_manifest_file_sha256",
                    "runtime_manifest_file_sha256",
                ],
                "schema_version": 1,
            },
            "gres": "gpu:a40:1",
            "input_bundle_sha256": input_bundle_sha256,
            "memory": "64G",
            "partition": "i64m1tga40u",
            "role": "cost-benchmark",
            "run_key": run_key,
            "seed": _SEED,
            "stage": _STAGE,
            "stderr_path": "logs/samga_brain_rw/stage1_cost_%A_%a.err",
            "stdout_path": "logs/samga_brain_rw/stage1_cost_%A_%a.out",
            "subject": _SUBJECT,
            "time": "12:00:00",
        }
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    root = _absolute_normalized(arguments.project_root, "project root")
    rows = build_stage1_cost_rows(
        project_root=root,
    )
    output = _canonical_job_map_output(
        root,
        arguments.output,
    )
    payload = _publish_stage1_cost_job_map(
        rows,
        project_root=root,
        output=output,
    )
    print(payload["payload_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
