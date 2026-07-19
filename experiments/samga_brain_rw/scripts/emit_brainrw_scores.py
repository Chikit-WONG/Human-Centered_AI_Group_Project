#!/usr/bin/env python3
"""Emit one typed development-only Brain-RW val-dev score bundle."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from samga_brain_rw import brainrw as br
from samga_brain_rw.scores import ScoreArtifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope",
        required=True,
        choices=["val-dev"],
    )
    parser.add_argument("--subject", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> Path:
    if not 1 <= args.subject <= 10:
        raise ValueError("subject must be between 1 and 10")
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    output = br.reject_development_path(
        args.output_dir,
        "score output",
    )
    if output.exists():
        raise FileExistsError("score output already exists")
    if not output.parent.is_dir():
        raise ValueError(
            "score output parent must already exist"
        )
    return output


def _validate_identities(
    checkpoint: br.LoadedBrainRWCheckpoint,
    manifest: br.ManifestIdentity,
    *,
    subject: int,
    seed: int,
) -> None:
    payload = checkpoint.payload
    expected = {
        "subject": subject,
        "seed": seed,
        "manifest_sha256": manifest.manifest_sha256,
        "protocol_sha256": manifest.protocol_sha256,
        "scope": "train",
        "validation_scope": "val-dev",
        "observed_scopes": ["train", "val-dev"],
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise ValueError(
                f"checkpoint {name} differs from evaluator"
            )


def run(args: argparse.Namespace) -> ScoreArtifact:
    output = _validate_args(args)
    # Both metadata-bearing artifacts are verified before constructing a
    # dataset; therefore no EEG or image read can precede these guards.
    manifest = br.load_development_manifest_identity(
        args.manifest,
        expected_subject=args.subject,
    )
    checkpoint = br.load_brainrw_checkpoint(
        args.checkpoint,
        requested_scope="val-dev",
    )
    _validate_identities(
        checkpoint,
        manifest,
        subject=args.subject,
        seed=args.seed,
    )

    dataset = br.BrainRWDevelopmentDataset(
        manifest.path,
        "val-dev",
        args.seed,
    )
    if dataset.subject_id != args.subject:
        raise ValueError(
            "validation dataset subject differs from CLI"
        )
    if tuple(dataset.query_ids) != tuple(
        manifest.val_dev_ordered_ids
    ) or tuple(dataset.gallery_ids) != tuple(
        manifest.val_dev_ordered_ids
    ):
        raise ValueError(
            "validation dataset IDs differ from protocol identity"
        )
    model, processor = br.build_model_from_checkpoint(
        checkpoint.payload
    )
    config_payload = checkpoint.payload.get(
        "config_payload", {}
    )
    batch_size = 512
    if isinstance(config_payload, Mapping):
        training = config_payload.get("training")
        if isinstance(training, Mapping):
            batch_size = int(training.get("batch_size", 512))
    similarity, identifiers = br.evaluate_brainrw_similarity(
        model,
        dataset,
        processor,
        batch_size=batch_size,
        device=torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    )
    source_records = [
        {
            "manifest_sha256": manifest.manifest_sha256,
            "records_sha256": manifest.records_sha256,
            "role": "val-dev",
            "role_payload_sha256": (
                manifest.val_dev_role_sha256
            ),
        }
    ]
    ScoreArtifact.save(
        output,
        similarity,
        identifiers,
        identifiers,
        {
            "checkpoint_sha256": checkpoint.sha256,
            "config_sha256": checkpoint.payload[
                "config_sha256"
            ],
            "git_sha": checkpoint.payload["git_sha"],
            "protocol_sha256": manifest.protocol_sha256,
            "seed": args.seed,
            "source_records": source_records,
            "split_role": "val-dev",
            "stage": "brainrw-clip-lora",
            "subject": args.subject,
        },
    )
    return ScoreArtifact.load(
        output,
        allowed_scopes={"val-dev"},
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    try:
        args = parse_args(argv)
        run(args)
    except SystemExit:
        raise
    except (
        FileExistsError,
        OSError,
        PermissionError,
        TypeError,
        ValueError,
    ) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
