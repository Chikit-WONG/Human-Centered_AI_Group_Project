#!/usr/bin/env python3
"""Lock at most one passing val-dev survivor for one development stage."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from samga_brain_rw.registry import CandidateRegistry
from samga_brain_rw.statistics import (
    _open_directory_nofollow,
    validate_development_path,
    write_development_json_exclusive,
)

_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--journal",
        type=Path,
        required=True,
        help="explicit append-only candidate decisions JSONL",
    )
    parser.add_argument(
        "--state",
        type=Path,
        required=True,
        help="explicit compact candidate registry state JSON",
    )
    parser.add_argument(
        "--stage",
        type=int,
        required=True,
        choices=range(1, 6),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new canonical locked-survivor JSON (never overwritten)",
    )
    return parser


def _preflight_output(path: Path, inputs: Sequence[Path]) -> Path:
    output = validate_development_path(
        path,
        allowed_suffixes=frozenset({".json"}),
    )
    normalized_inputs = {
        validate_development_path(
            inputs[0],
            allowed_suffixes=frozenset({".jsonl"}),
        ),
        validate_development_path(
            inputs[1],
            allowed_suffixes=frozenset({".json"}),
        ),
    }
    if output in normalized_inputs:
        raise ValueError("locked-survivor output must differ from registry inputs")
    parent_fd = -1
    descriptor = -1
    try:
        parent_fd = _open_directory_nofollow(output.parent, create=False)
    except FileNotFoundError:
        return output
    try:
        try:
            descriptor = os.open(
                output.name,
                os.O_RDONLY | _O_CLOEXEC | _O_NOFOLLOW,
                dir_fd=parent_fd,
            )
        except FileNotFoundError:
            return output
        raise FileExistsError(
            f"locked-survivor output already exists: {output}"
        )
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if parent_fd >= 0:
            os.close(parent_fd)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    output = _preflight_output(
        arguments.output,
        (arguments.journal, arguments.state),
    )
    registry = CandidateRegistry(arguments.journal, arguments.state)
    survivor = registry.lock_stage_survivor(arguments.stage)
    state = registry.load_state()
    document = {
        "artifact_type": "samga_brain_rw.locked_survivor",
        "decision": survivor.to_payload(),
        "decision_sha256": survivor.decision_sha256,
        "registry_state_sha256": state["state_sha256"],
        "schema_version": 1,
        "stage": arguments.stage,
    }
    write_development_json_exclusive(output, document)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
