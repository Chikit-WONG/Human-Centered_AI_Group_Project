#!/usr/bin/env python3
"""Build strict train sidecars and the fixed 49-entry Stage-0 capability map."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from samga_brain_rw.capability_map import build_stage0_capability_map
from samga_brain_rw.provenance import DEFAULT_ORACLES, ProvenanceInputs
from scripts.preflight import capture_environment


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate all fixed SAMGA Stage-0 train inputs, then exclusively "
            "write 39 generic sidecars and one canonical capability map"
        )
    )
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--internvit-config", type=Path, required=True)
    parser.add_argument("--brainrw-config", type=Path, required=True)
    parser.add_argument("--source-manifest-dir", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--feature-directory", type=Path, required=True)
    parser.add_argument("--variant-directory", type=Path, required=True)
    parser.add_argument("--canonical-cache", type=Path, required=True)
    parser.add_argument("--clip-train-cache", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--clip-model-path", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--experiment-revision", required=True)
    parser.add_argument("--upstream-revision", required=True)
    parser.add_argument("--cache-generator-revision", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    inputs = ProvenanceInputs(
        repository_root=args.repository_root,
        protocol_path=args.protocol,
        internvit_config_path=args.internvit_config,
        brainrw_config_path=args.brainrw_config,
        source_manifest_dir=args.source_manifest_dir,
        protocol_manifest_dir=args.manifest_dir,
        feature_directory=args.feature_directory,
        variant_directory=args.variant_directory,
        canonical_cache=args.canonical_cache,
        clip_train_cache=args.clip_train_cache,
        data_root=args.data_root,
        model_path=args.model_path,
        clip_model_path=args.clip_model_path,
        upstream_root=args.upstream_root,
        experiment_revision=args.experiment_revision,
        upstream_revision=args.upstream_revision,
        cache_generator_revision=args.cache_generator_revision,
        verified_artifacts={},
        environment=capture_environment(),
        oracles=DEFAULT_ORACLES,
    )
    build_stage0_capability_map(inputs, args.output_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
