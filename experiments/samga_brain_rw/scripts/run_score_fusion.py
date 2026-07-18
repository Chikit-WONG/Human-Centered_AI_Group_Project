#!/usr/bin/env python3
"""Evaluate the sealed 47-config Stage 1 grid on typed val-dev scores."""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from samga_brain_rw.fusion import (
    assert_aligned,
    enumerate_stage1_configs,
)
from samga_brain_rw.hashing import canonical_json_bytes, sha256_json
from samga_brain_rw.scores import (
    ScoreArtifact,
    independent_retrieval_metrics,
)


_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--internvit-score-directory",
        type=Path,
        required=True,
        help="complete typed val-dev ScoreArtifact directory for InternViT",
    )
    parser.add_argument(
        "--clip-score-directory",
        type=Path,
        required=True,
        help="complete typed val-dev ScoreArtifact directory for CLIP",
    )
    parser.add_argument("--internvit-branch-id", required=True)
    parser.add_argument("--clip-branch-id", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new canonical JSON result; an existing path is never replaced",
    )
    return parser


def _deep_thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    return value


def _branch_id(value: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    if "\n" in value or "\r" in value:
        raise ValueError(f"{name} must not contain line breaks")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must be valid UTF-8") from exc
    return value


def _reject_existing_output(path: Path) -> None:
    try:
        os.lstat(path)
    except FileNotFoundError:
        return
    raise FileExistsError(f"fusion output already exists: {path}")


def _reject_output_inside_inputs(
    output: Path,
    input_directories: Sequence[Path],
) -> None:
    destination = Path(os.path.abspath(os.fspath(output)))
    for input_directory in input_directories:
        bundle = Path(os.path.abspath(os.fspath(input_directory)))
        try:
            destination.relative_to(bundle)
        except ValueError:
            continue
        raise ValueError(
            "fusion output must not be located inside an input "
            f"ScoreArtifact bundle: {bundle}"
        )


def _write_exclusive(path: Path, payload: bytes) -> None:
    destination = Path(os.path.abspath(os.fspath(path)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = -1
    created = False
    try:
        descriptor = os.open(
            destination,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | _O_NOFOLLOW
            | _O_CLOEXEC,
            0o600,
        )
        created = True
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        parent_fd = os.open(
            destination.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | _O_CLOEXEC,
        )
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        if created:
            try:
                os.unlink(destination)
            except FileNotFoundError:
                pass
        raise


def _metrics_payload(metrics: object) -> dict[str, object]:
    return {
        "query_count": metrics.query_count,
        "gallery_count": metrics.gallery_count,
        "top1_count": metrics.top1_count,
        "top5_count": metrics.top5_count,
        "top1_rate": metrics.top1_rate,
        "top5_rate": metrics.top5_rate,
    }


def _input_binding(
    artifact: ScoreArtifact,
    branch_id: str,
) -> dict[str, str]:
    return {
        "branch_id": branch_id,
        "provenance_sha256": sha256_json(_deep_thaw(artifact.provenance)),
        "score_payload_sha256": artifact.verified.payload_sha256,
    }


def _alignment_payload(artifact: ScoreArtifact) -> dict[str, object]:
    return {
        "scope": artifact.scope,
        "protocol_sha256": artifact.metadata["protocol_sha256"],
        "subject": artifact.metadata["subject"],
        "seed": artifact.metadata["seed"],
        "stage": artifact.metadata["stage"],
        "source_records_sha256": artifact.metadata["source_records_sha256"],
        "query_ids": list(artifact.query_ids),
        "query_ids_sha256": artifact.metadata["query_ids_sha256"],
        "gallery_ids": list(artifact.gallery_ids),
        "gallery_ids_sha256": artifact.metadata["gallery_ids_sha256"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    output = Path(arguments.output)
    # Refuse conflicts before opening either input.
    _reject_output_inside_inputs(
        output,
        (
            arguments.internvit_score_directory,
            arguments.clip_score_directory,
        ),
    )
    _reject_existing_output(output)
    internvit_branch_id = _branch_id(
        arguments.internvit_branch_id, "internvit_branch_id"
    )
    clip_branch_id = _branch_id(arguments.clip_branch_id, "clip_branch_id")
    if internvit_branch_id == clip_branch_id:
        raise ValueError("branch IDs must be distinct")

    internvit = ScoreArtifact.load(
        arguments.internvit_score_directory,
        allowed_scopes={"val-dev"},
    )
    clip = ScoreArtifact.load(
        arguments.clip_score_directory,
        allowed_scopes={"val-dev"},
    )
    assert_aligned(internvit, clip)

    results: list[dict[str, object]] = []
    for config in enumerate_stage1_configs():
        fused = config.apply(
            internvit.similarity,
            clip.similarity,
            gallery_ids=internvit.gallery_ids,
        )
        metrics = independent_retrieval_metrics(
            fused,
            internvit.query_ids,
            internvit.gallery_ids,
        )
        result = config.to_dict()
        result["metrics"] = _metrics_payload(metrics)
        results.append(result)

    document = {
        "schema_version": 1,
        "artifact_type": "samga_brain_rw.stage1_fusion_grid",
        "scope": "val-dev",
        "inputs": {
            "internvit": _input_binding(internvit, internvit_branch_id),
            "clip": _input_binding(clip, clip_branch_id),
        },
        "alignment": _alignment_payload(internvit),
        "results": results,
    }
    _write_exclusive(output, canonical_json_bytes(document) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
