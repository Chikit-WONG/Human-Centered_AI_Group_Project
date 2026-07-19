#!/usr/bin/env python3
"""Emit one typed development-only Brain-RW val-dev score bundle."""

from __future__ import annotations

import argparse
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from samga_brain_rw import brainrw as br
from samga_brain_rw.scores import ScoreArtifact


_GIT_SHA_RE = re.compile(
    r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$"
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _git_provenance() -> dict[str, object]:
    root = _REPOSITORY_ROOT

    def output(*arguments: str) -> str:
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), *arguments],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (
            OSError,
            subprocess.CalledProcessError,
        ) as exc:
            raise RuntimeError(
                "Git provenance cannot be resolved"
            ) from exc

    actual_root = output(
        "rev-parse",
        "--show-toplevel",
    )
    try:
        resolved_actual = Path(actual_root).resolve(
            strict=True
        )
        resolved_expected = root.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(
            "Git repository root cannot be resolved"
        ) from exc
    if resolved_actual != resolved_expected:
        raise RuntimeError(
            "Git repository root differs from the anchored project"
        )
    revision = output("rev-parse", "HEAD")
    if _GIT_SHA_RE.fullmatch(revision) is None:
        raise ValueError("Git SHA is invalid")
    status = output(
        "status",
        "--porcelain",
        "--untracked-files=all",
    )
    if status:
        raise RuntimeError(
            "Git worktree must be clean before BrainRW scoring"
        )
    return {
        "clean": True,
        "git_sha": revision,
        "repository_root": str(resolved_expected),
    }


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
    config: object,
    *,
    subject: int,
    seed: int,
    git_provenance: Mapping[str, object],
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
    br.validate_brainrw_checkpoint_identity(
        payload,
        config=config,
        manifest=manifest,
        subject=subject,
        seed=seed,
    )
    if (
        payload.get("training_complete") is not True
        or payload.get("global_step")
        != payload.get("planned_steps")
    ):
        raise ValueError(
            "score emission requires a terminal complete training checkpoint"
        )
    checkpoint_git = payload.get("git_provenance")
    if (
        not isinstance(checkpoint_git, Mapping)
        or dict(checkpoint_git) != dict(git_provenance)
    ):
        raise ValueError(
            "checkpoint Git provenance differs from evaluator"
        )


def run(args: argparse.Namespace) -> ScoreArtifact:
    output = _validate_args(args)
    initial_git_provenance = _git_provenance()
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
    config_path = checkpoint.payload.get("config_path")
    clip_path = checkpoint.payload.get("clip_path")
    if (
        not isinstance(config_path, str)
        or not config_path
        or not isinstance(clip_path, str)
        or not clip_path
    ):
        raise ValueError(
            "checkpoint lacks config reconstruction paths"
        )
    config = br.verify_brainrw_config(
        Path(config_path),
        Path(clip_path),
    )
    _validate_identities(
        checkpoint,
        manifest,
        config,
        subject=args.subject,
        seed=args.seed,
        git_provenance=initial_git_provenance,
    )
    model, processor = br.build_model_from_checkpoint(
        checkpoint.payload
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    dtype = br.checkpoint_runtime_dtype(checkpoint.payload, device)
    config_payload = checkpoint.payload.get(
        "config_payload", {}
    )
    batch_size = 512
    if isinstance(config_payload, Mapping):
        training = config_payload.get("training")
        if isinstance(training, Mapping):
            batch_size = int(training.get("batch_size", 512))

    dataset = br.BrainRWDevelopmentDataset(
        manifest.path,
        "val-dev",
        args.seed,
        expected_source_payload_sha256=(
            manifest.source_payload_sha256
        ),
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
    similarity, identifiers = br.evaluate_brainrw_similarity(
        model,
        dataset,
        processor,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
    )
    if _git_provenance() != initial_git_provenance:
        raise RuntimeError(
            "Git provenance changed during BrainRW scoring"
        )
    source_records = [
        {
            "manifest_sha256": manifest.manifest_sha256,
            "records_sha256": manifest.records_sha256,
            "role": "val-dev",
            "role_payload_sha256": (
                manifest.val_dev_role_sha256
            ),
            "source_manifest_sha256": (
                manifest.source_manifest_sha256
            ),
            "source_payload_byte_count": manifest.source_payload_byte_count,
            "source_payload_path": str(manifest.source_payload_path),
            "source_payload_sha256": manifest.source_payload_sha256,
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
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
