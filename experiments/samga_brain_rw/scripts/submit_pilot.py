#!/usr/bin/env python3
"""Inspect the A40 queues and submit only the smoke-gated current pilot."""

from __future__ import annotations

import argparse
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from build_job_map import (
    completion_is_valid,
    completion_output_hashes,
    load_job_map,
    should_submit_row,
)
from run_training_cell import validate_training_command_outputs


QUEUE_COMMAND = [
    "squeue",
    "-h",
    "-p",
    "debug,i64m1tga40u,i64m1tga40ue,emergency_gpua40",
    "-o",
    "%.10i %.14P %.10u %.2t %.10M %.6D %R",
]


def _stage(payload: Mapping[str, object]) -> str:
    stage = payload["stage"]
    if not isinstance(stage, str):
        raise ValueError("job map has an invalid stage")
    return stage


def _refuse_out_of_scope_submission(
    smoke: Mapping[str, object],
    pilot: Mapping[str, object],
    slurm_script: Path,
) -> None:
    smoke_stage = _stage(smoke).lower()
    pilot_stage = _stage(pilot).lower()
    if "confirm" in smoke_stage or "confirm" in pilot_stage:
        raise ValueError("confirmation submission is forbidden at the current stage")
    if "confirmation" in slurm_script.name.lower():
        raise ValueError("confirmation launcher is not a current-stage pilot")
    if "smoke" not in smoke_stage:
        raise ValueError("the prerequisite job map must be a smoke stage")
    if "pilot" not in pilot_stage:
        raise ValueError("the submitted job map must be a current pilot stage")
    match = re.search(r"stage[-_]?(\d+)", pilot_stage)
    if match is not None and int(match.group(1)) >= 3:
        raise ValueError("Stage 3-5 submission is forbidden at the current stage")


def _rows(payload: Mapping[str, object]) -> list[dict[str, object]]:
    rows = payload["rows"]
    if not isinstance(rows, list):
        raise ValueError("job map rows must be a list")
    return [dict(row) for row in rows]


def _incomplete_rows(
    payload: Mapping[str, object],
) -> list[dict[str, object]]:
    return [
        row
        for row in _rows(payload)
        if not completion_is_valid(payload, row)
    ]


def _revalidate_completed_smoke_outputs(
    payload: Mapping[str, object],
) -> None:
    expected_names = {
        "final_checkpoint_sha256",
        "in_loop_metadata_sha256",
        "run_manifest_sha256",
    }
    for row in _rows(payload):
        declared = completion_output_hashes(payload, row)
        if declared is None:
            raise ValueError(
                "smoke output revalidation requires every completion"
            )
        if set(declared) != expected_names:
            raise ValueError(
                "smoke completion hashes differ from the release gate schema"
            )
        outputs = validate_training_command_outputs(
            row["argv"],
            expected_mode="smoke",
        )
        actual = {
            "final_checkpoint_sha256": outputs.final_checkpoint_sha256,
            "in_loop_metadata_sha256": outputs.in_loop_metadata_sha256,
            "run_manifest_sha256": outputs.run_manifest_sha256,
        }
        if actual != declared:
            raise ValueError(
                "smoke artifact hashes differ from the sealed completion"
            )


def _compress_indices(indices: Sequence[int]) -> str:
    if not indices:
        raise ValueError("cannot submit an empty array")
    ordered = sorted(set(indices))
    if ordered != list(indices):
        raise ValueError("submitted array indices must be unique and sorted")
    ranges: list[str] = []
    start = previous = ordered[0]
    for index in ordered[1:]:
        if index == previous + 1:
            previous = index
            continue
        ranges.append(f"{start}-{previous}")
        start = previous = index
    ranges.append(f"{start}-{previous}")
    return ",".join(ranges)


def _single_argv_value(
    row: Mapping[str, object],
    flag: str,
) -> str:
    argv = row.get("argv")
    if not isinstance(argv, list) or any(
        not isinstance(value, str) for value in argv
    ):
        raise ValueError("job row argv must be a list of strings")
    positions = [
        index for index, value in enumerate(argv) if value == flag
    ]
    if len(positions) != 1:
        raise ValueError(f"job row must contain {flag} exactly once")
    position = positions[0]
    if position + 1 >= len(argv):
        raise ValueError(f"job row {flag} has no value")
    return argv[position + 1]


def _verified_project_root(
    payload: Mapping[str, object],
    slurm_script: Path,
) -> tuple[Path, Path]:
    roots = {
        _single_argv_value(row, "--project-root")
        for row in _rows(payload)
    }
    if len(roots) != 1:
        raise ValueError("job-map rows must share one project root")
    raw_root = Path(roots.pop())
    if not raw_root.is_absolute() or ".." in raw_root.parts:
        raise ValueError("project root must be absolute and normalized")
    try:
        root = raw_root.resolve(strict=True)
    except OSError as exc:
        raise ValueError("project root cannot be verified") from exc
    if root != raw_root or not root.is_dir() or not (root / ".git").exists():
        raise ValueError("project root is not a verified repository root")
    try:
        script = Path(slurm_script).resolve(strict=True)
    except OSError as exc:
        raise ValueError("SLURM script cannot be verified") from exc
    try:
        script.relative_to(root)
    except ValueError as exc:
        raise ValueError("SLURM script is outside the project root") from exc
    return root, script


def _verified_log_directory(
    root: Path,
    row: Mapping[str, object],
    log_dir: Path,
) -> Path:
    relative = Path(str(row["stdout_path"])).parent
    expected = root / relative
    declared = Path(log_dir)
    if declared.is_absolute():
        if declared != expected:
            raise ValueError("absolute log_dir differs from sealed row logs")
    elif declared != relative:
        raise ValueError("log_dir differs from the sealed row log paths")
    return expected


def _resource_command(
    payload: Mapping[str, object],
    *,
    job_map_path: Path,
    job_map_sha256: str,
    slurm_script: Path,
    log_dir: Path,
    indices: Sequence[int],
) -> list[str]:
    rows = _rows(payload)
    if not rows:
        raise ValueError("cannot submit a job map without rows")
    row = rows[0]
    project_root, verified_script = _verified_project_root(
        payload, slurm_script
    )
    bounds = payload.get("array_bounds")
    if (
        not isinstance(bounds, list)
        or len(bounds) != 2
        or type(bounds[0]) is not int
        or type(bounds[1]) is not int
    ):
        raise ValueError("job map has invalid immutable array bounds")
    stdout_path = Path(str(row["stdout_path"]))
    stderr_path = Path(str(row["stderr_path"]))
    sealed_log_dir = _verified_log_directory(
        project_root,
        row,
        log_dir,
    )
    if stderr_path.parent != stdout_path.parent:
        raise ValueError("log_dir differs from the sealed row log paths")
    try:
        sealed_job_map = Path(job_map_path).resolve(strict=True)
    except OSError as exc:
        raise ValueError("job-map path cannot be verified") from exc
    for value in (str(sealed_job_map), str(project_root)):
        if "," in value or "\n" in value:
            raise ValueError(
                "sealed path cannot be encoded in SLURM --export"
            )
    if not sealed_log_dir.is_absolute():
        raise ValueError("job-map path cannot be encoded in SLURM --export")
    return [
        "sbatch",
        "--parsable",
        f"--chdir={project_root}",
        f"--partition={row['partition']}",
        f"--gres={row['gres']}",
        f"--cpus-per-task={row['cpus']}",
        f"--mem={row['memory']}",
        f"--time={row['time']}",
        f"--array={_compress_indices(indices)}",
        f"--output={project_root / stdout_path}",
        f"--error={project_root / stderr_path}",
        (
            "--export=ALL,"
            f"PROJECT_ROOT={project_root},"
            f"JOB_MAP={sealed_job_map},"
            f"JOB_MAP_SHA256={job_map_sha256},"
            f"JOB_MAP_ARRAY_MIN={bounds[0]},"
            f"JOB_MAP_ARRAY_MAX={bounds[1]}"
        ),
        str(verified_script),
    ]


def _submit(
    payload: Mapping[str, object],
    *,
    job_map_path: Path,
    job_map_sha256: str,
    slurm_script: Path,
    log_dir: Path,
    rows: Sequence[Mapping[str, object]],
    runner: Callable[..., Any],
) -> None:
    eligible: list[int] = []
    blocked: list[int] = []
    for row in rows:
        index = int(row["array_index"])
        if should_submit_row(payload, row):
            eligible.append(index)
        else:
            blocked.append(index)
    if blocked:
        raise RuntimeError(
            "incomplete rows have active stale claims; audited recovery is "
            f"required before submission: {blocked}"
        )
    if not eligible:
        raise ValueError("no incomplete rows are eligible for submission")
    project_root, _ = _verified_project_root(payload, slurm_script)
    first_row = _rows(payload)[0]
    sealed_log_dir = _verified_log_directory(
        project_root,
        first_row,
        log_dir,
    )
    sealed_log_dir.mkdir(parents=True, exist_ok=True)
    runner(
        QUEUE_COMMAND,
        check=True,
        capture_output=True,
        text=True,
    )
    runner(
        _resource_command(
            payload,
            job_map_path=job_map_path,
            job_map_sha256=job_map_sha256,
            slurm_script=slurm_script,
            log_dir=log_dir,
            indices=eligible,
        ),
        check=True,
        capture_output=True,
        text=True,
    )


def submit_available_pilot(
    *,
    smoke_job_map: Path,
    smoke_sha256: str,
    pilot_job_map: Path,
    pilot_sha256: str,
    slurm_script: Path,
    log_dir: Path,
    runner: Callable[..., Any] = subprocess.run,
) -> str:
    """Submit debug smoke first, full pilot only after every smoke completion."""

    smoke_path = Path(smoke_job_map)
    pilot_path = Path(pilot_job_map)
    script_path = Path(slurm_script)
    smoke = load_job_map(smoke_path, expected_sha256=smoke_sha256)
    pilot = load_job_map(pilot_path, expected_sha256=pilot_sha256)
    _refuse_out_of_scope_submission(smoke, pilot, script_path)

    smoke_incomplete = _incomplete_rows(smoke)
    if smoke_incomplete:
        _submit(
            smoke,
            job_map_path=smoke_path,
            job_map_sha256=smoke_sha256,
            slurm_script=script_path,
            log_dir=Path(log_dir),
            rows=smoke_incomplete,
            runner=runner,
        )
        return "smoke-submitted"

    pilot_incomplete = _incomplete_rows(pilot)
    if not pilot_incomplete:
        return "already-complete"
    _revalidate_completed_smoke_outputs(smoke)
    _submit(
        pilot,
        job_map_path=pilot_path,
        job_map_sha256=pilot_sha256,
        slurm_script=script_path,
        log_dir=Path(log_dir),
        rows=pilot_incomplete,
        runner=runner,
    )
    return "pilot-submitted"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-job-map", required=True, type=Path)
    parser.add_argument("--smoke-sha256", required=True)
    parser.add_argument("--pilot-job-map", required=True, type=Path)
    parser.add_argument("--pilot-sha256", required=True)
    parser.add_argument(
        "--slurm-script",
        type=Path,
        default=Path("experiments/samga_brain_rw/slurm/pilot_array.slurm"),
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs/samga_brain_rw"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    phase = submit_available_pilot(
        smoke_job_map=args.smoke_job_map,
        smoke_sha256=args.smoke_sha256,
        pilot_job_map=args.pilot_job_map,
        pilot_sha256=args.pilot_sha256,
        slurm_script=args.slurm_script,
        log_dir=args.log_dir,
    )
    print(phase)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
