#!/usr/bin/env python3
"""Finalize the fixed six-cell Stage 1 development composition."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from samga_brain_rw.hashing import canonical_json_bytes
from samga_brain_rw.stage1_finalizer import finalize_stage1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        required=True,
        help="SAMGA reproduction experiment root",
    )
    parser.add_argument(
        "--cost-job-map",
        type=Path,
        required=True,
        help="sealed completed Stage 1 cost job map",
    )
    parser.add_argument(
        "--cost-job-map-sha256",
        required=True,
        help="expected canonical Stage 1 cost job-map SHA-256",
    )
    parser.add_argument(
        "--journal",
        type=Path,
        required=True,
        help="append-only candidate registry JSONL",
    )
    parser.add_argument(
        "--state",
        type=Path,
        required=True,
        help="compact candidate registry JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory for the fixed Stage 1 final artifacts",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = finalize_stage1(
        project_root=arguments.project_root,
        cost_job_map_path=arguments.cost_job_map,
        cost_job_map_sha256=arguments.cost_job_map_sha256,
        journal_path=arguments.journal,
        state_path=arguments.state,
        output_dir=arguments.output_dir,
    )
    print(canonical_json_bytes(result.to_payload()).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
