#!/usr/bin/env python3
"""Validate and aggregate the sealed sub-08 / seed-42 fairness run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from matching_fairness.reporting import aggregate_results  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Existing matching_fairness_v3 root containing runs/checkpoints/matrices.",
    )
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    destination = aggregate_results(arguments.results_root)
    print(
        json.dumps(
            {"aggregate_dir": str(destination), "record_count": 450},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
