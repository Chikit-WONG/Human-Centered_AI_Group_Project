#!/usr/bin/env python3
"""Verify sealed Stage 0 train/val-dev views of one canonical cache."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from samga_brain_rw.access import TypedArtifact
from samga_brain_rw.cache_parity import (
    EXPECTED_SCOPES,
    build_stage0_cache_parity,
    validate_stage0_report_path,
    write_stage0_cache_parity_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--canonical-cache", type=Path, required=True)
    parser.add_argument(
        "--canonical-cache-envelope",
        type=Path,
        required=True,
        help="strict Task 3 envelope; no legacy sidecar discovery is allowed",
    )
    parser.add_argument(
        "--scopes",
        nargs="+",
        default=list(EXPECTED_SCOPES),
        help="must be exactly: train val-dev",
    )
    parser.add_argument("--chunk-rows", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    validate_stage0_report_path(arguments.output)
    cache = TypedArtifact(
        payload_type="samga_brain_rw.train_cache",
        payload_path=arguments.canonical_cache,
        envelope_path=arguments.canonical_cache_envelope,
    )
    report = build_stage0_cache_parity(
        arguments.manifest_dir,
        cache,
        scopes=arguments.scopes,
        strict=True,
        chunk_rows=arguments.chunk_rows,
    )
    write_stage0_cache_parity_report(arguments.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
