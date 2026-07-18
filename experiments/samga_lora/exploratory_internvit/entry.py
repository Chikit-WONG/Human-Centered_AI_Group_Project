#!/usr/bin/env python3
"""Run the locked SAMGA train/evaluate utilities with a different image width.

The confirmatory CLIP experiment is source-locked with a 768-dimensional visual
input.  This exploratory entry point leaves those files byte-identical and only
rebinds their task-model constructor to the explicitly requested frozen-feature
width.  It supports no online vision model and therefore cannot train LoRA.
"""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path
from typing import Callable


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from samga_lora.model import SAMGATaskModel as _SAMGATaskModel  # noqa: E402


def add_image_dim_to_parsed_args(
    parse_args: Callable[[], argparse.Namespace], image_dim: int
) -> argparse.Namespace:
    """Persist the wrapper-only width in train configs and checkpoints."""
    parsed = parse_args()
    parsed.image_dim = int(image_dim)
    return parsed


def parse_wrapper_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run a SAMGA utility with an explicit frozen visual-feature width"
    )
    parser.add_argument("mode", choices=("train", "evaluate", "verify"))
    parser.add_argument("--image-dim", type=int, required=True)
    return parser.parse_known_args()


def main() -> None:
    args, remaining = parse_wrapper_args()
    if args.image_dim <= 0:
        raise ValueError("image-dim must be positive")
    constructor = partial(_SAMGATaskModel, image_dim=args.image_dim)
    if args.mode == "train":
        import train as target
        target.parse_args = partial(
            add_image_dim_to_parsed_args, target.parse_args, args.image_dim
        )
    elif args.mode == "evaluate":
        import evaluate as target
    else:
        from scripts import verify_checkpoint as target
    target.SAMGATaskModel = constructor
    sys.argv = [str(Path(target.__file__).resolve()), *remaining]
    target.main()


if __name__ == "__main__":
    main()
