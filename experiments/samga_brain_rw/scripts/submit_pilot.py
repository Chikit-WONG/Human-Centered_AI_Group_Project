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
    load_job_map,
    should_submit_row,
)


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
    expected_log_dir = stdout_path.parent
    if (
        stderr_path.parent != expected_log_dir
        or Path(log_dir) != expected_log_dir
    ):
        raise ValueError("log_dir differs from the sealed row log paths")
    if "," in str(job_map_path) or "\n" in str(job_map_path):
        raise ValueError("job-map path cannot be encoded in SLURM --export")
    return [
        "sbatch",
        "--parsable",
        f"--partition={row['partition']}",
        f"--gres={row['gres']}",
        f"--cpus-per-task={row['cpus']}",
        f"--mem={row['memory']}",
        f"--time={row['time']}",
        f"--array={_compress_indices(indices)}",
        f"--output={row['stdout_path']}",
        f"--error={row['stderr_path']}",
        (
            "--export=ALL,"
            f"JOB_MAP={job_map_path},"
            f"JOB_MAP_SHA256={job_map_sha256},"
            f"JOB_MAP_ARRAY_MIN={bounds[0]},"
            f"JOB_MAP_ARRAY_MAX={bounds[1]}"
        ),
        str(slurm_script),
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
    log_dir.mkdir(parents=True, exist_ok=True)
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
