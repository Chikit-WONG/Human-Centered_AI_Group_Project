#!/usr/bin/env python3
"""Re-emit one complete typed val-dev score bundle on CPU."""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from samga_brain_rw.scores import ScoreArtifact


_SOURCE_METADATA_KEYS = (
    "checkpoint_sha256",
    "config_sha256",
    "git_sha",
    "protocol_sha256",
    "seed",
    "source_records",
    "split_role",
    "stage",
    "subject",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-similarity",
        type=Path,
        required=True,
        help="similarity.npy from a complete typed val-dev score bundle",
    )
    parser.add_argument(
        "--input-envelope",
        type=Path,
        required=True,
        help="metadata.json from the same complete typed score bundle",
    )
    parser.add_argument(
        "--input-predictions",
        type=Path,
        required=True,
        help="predictions.csv from the same complete typed score bundle",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
        help="new exclusive score-bundle directory",
    )
    return parser


def _normalized(path: Path) -> Path:
    return Path(os.path.abspath(os.path.normpath(os.fspath(path))))


def _source_directory(
    similarity: Path,
    envelope: Path,
    predictions: Path,
) -> Path:
    descriptors = (
        (_normalized(similarity), "similarity.npy"),
        (_normalized(envelope), "metadata.json"),
        (_normalized(predictions), "predictions.csv"),
    )
    if any(path.name != expected for path, expected in descriptors):
        raise ValueError(
            "the three input paths must identify the same complete typed bundle"
        )
    parents = {path.parent for path, _ in descriptors}
    if len(parents) != 1:
        raise ValueError(
            "the three input paths must identify the same complete typed bundle"
        )
    return next(iter(parents))


def _require_output_outside_source(
    source_directory: Path,
    output_directory: Path,
) -> None:
    source = _normalized(source_directory)
    output = _normalized(output_directory)
    if output == source or source in output.parents:
        raise ValueError(
            "output directory must be outside the source bundle"
        )


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(child) for child in value]
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        source_directory = _source_directory(
            arguments.input_similarity,
            arguments.input_envelope,
            arguments.input_predictions,
        )
        _require_output_outside_source(
            source_directory,
            arguments.output_directory,
        )
        source = ScoreArtifact.load(
            source_directory,
            allowed_scopes={"val-dev"},
        )
        source_metadata = {
            key: _thaw_json(source.metadata[key])
            for key in _SOURCE_METADATA_KEYS
        }
        ScoreArtifact.save(
            arguments.output_directory,
            source.similarity,
            source.query_ids,
            source.gallery_ids,
            source_metadata,
        )
    except (FileExistsError, OSError, PermissionError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
