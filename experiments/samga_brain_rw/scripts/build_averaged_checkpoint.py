#!/usr/bin/env python3
"""Build one exclusive locked Stage-2 averaged checkpoint."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from samga_brain_rw.checkpoints import (
    AVERAGING_CANDIDATES,
    build_averaged_checkpoint,
    write_averaged_checkpoint_exclusive,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-id",
        choices=tuple(AVERAGING_CANDIDATES),
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=Path,
        required=True,
        help="ordered epoch checkpoint; repeat exactly 5 or 10 times",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _write_exclusive(path: Path, payload: object) -> None:
    write_averaged_checkpoint_exclusive(Path(path), payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        output = Path(os.path.abspath(os.path.normpath(os.fspath(arguments.output))))
        inputs = [
            Path(os.path.abspath(os.path.normpath(os.fspath(path))))
            for path in arguments.checkpoint
        ]
        if output in inputs:
            raise ValueError("output must differ from every source checkpoint")
        payload = build_averaged_checkpoint(
            inputs,
            candidate_id=arguments.candidate_id,
        )
        _write_exclusive(output, payload)
    except (FileExistsError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
