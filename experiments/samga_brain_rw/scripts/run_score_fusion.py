#!/usr/bin/env python3
"""Evaluate the sealed 47-config Stage 1 grid on typed val-dev scores."""

from __future__ import annotations

import argparse
import os
import re
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
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_FORMAL_TEST_RECORD_SHA256 = (
    "02d7e33b3fe8e5a571f8db232ca5fa86abb0c16981876ec84feae7ba64636f1a"
)
_SUBJECT_TEST_RE = re.compile(r"^sub-\d{2}_test\.json$", re.IGNORECASE)
_FORBIDDEN_OUTPUT_COMPONENTS = frozenset(
    {"test_images", "val-confirm", "formal-test"}
)
_GRID_CONFIG_ID = "stage1_fusion_v1"


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


def _validated_output_path(path: Path) -> Path:
    raw = os.fspath(path)
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise ValueError("fusion output path is forbidden")
    destination = Path(os.path.abspath(os.path.normpath(raw)))
    lowered = raw.lower()
    if _FORMAL_TEST_RECORD_SHA256 in lowered:
        raise ValueError("fusion output path contains the formal-test digest")
    components: list[str] = []
    for component in (*Path(raw).parts, *destination.parts):
        components.extend(
            part for part in re.split(r"[\\/]+", component) if part
        )
    if any(_SUBJECT_TEST_RE.fullmatch(component) for component in components):
        raise ValueError("fusion output path names a formal-test subject manifest")
    if any(
        component.lower() in _FORBIDDEN_OUTPUT_COMPONENTS
        for component in components
    ):
        raise ValueError("fusion output path contains a sealed-scope component")
    if not destination.name or destination == destination.parent:
        raise ValueError("fusion output must name a file")
    return destination


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


def _open_output_parent_nofollow(path: Path) -> int:
    absolute = Path(os.path.abspath(os.fspath(path)))
    flags = os.O_RDONLY | _O_DIRECTORY | _O_CLOEXEC | _O_NOFOLLOW
    directory_fd = -1
    try:
        directory_fd = os.open(absolute.anchor, flags)
        for component in absolute.parts[1:]:
            try:
                next_fd = os.open(component, flags, dir_fd=directory_fd)
            except FileNotFoundError:
                try:
                    os.mkdir(component, mode=0o700, dir_fd=directory_fd)
                    os.fsync(directory_fd)
                except OSError as exc:
                    raise ValueError(
                        f"cannot safely create fusion output parent: {absolute}"
                    ) from exc
                try:
                    next_fd = os.open(component, flags, dir_fd=directory_fd)
                except OSError as exc:
                    raise ValueError(
                        f"cannot safely open fusion output parent: {absolute}"
                    ) from exc
            except OSError as exc:
                raise ValueError(
                    f"symlink or invalid fusion output parent: {absolute}"
                ) from exc
            os.close(directory_fd)
            directory_fd = next_fd
        return directory_fd
    except BaseException:
        if directory_fd >= 0:
            os.close(directory_fd)
        raise


def _write_exclusive(path: Path, payload: bytes) -> None:
    destination = _validated_output_path(path)
    parent_fd = _open_output_parent_nofollow(destination.parent)
    descriptor = -1
    created = False
    try:
        descriptor = os.open(
            destination.name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | _O_NOFOLLOW
            | _O_CLOEXEC,
            0o600,
            dir_fd=parent_fd,
        )
        created = True
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.fsync(parent_fd)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        if created:
            try:
                os.unlink(destination.name, dir_fd=parent_fd)
            except OSError:
                pass
            try:
                os.fsync(parent_fd)
            except OSError:
                pass
        raise
    finally:
        os.close(parent_fd)


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
    output = _validated_output_path(arguments.output)
    # Refuse conflicts before opening either input.
    _reject_output_inside_inputs(
        output,
        (
            arguments.internvit_score_directory,
            arguments.clip_score_directory,
        ),
    )
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

    configs = enumerate_stage1_configs()
    candidate_payload = [config.to_dict() for config in configs]
    results: list[dict[str, object]] = []
    for config in configs:
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
        "grid": {
            "config_id": _GRID_CONFIG_ID,
            "candidates": candidate_payload,
            "candidates_sha256": sha256_json(candidate_payload),
        },
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
