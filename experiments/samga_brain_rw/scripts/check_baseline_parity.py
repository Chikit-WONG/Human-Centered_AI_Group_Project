#!/usr/bin/env python3
"""Verify four sealed val-dev emissions from one fresh baseline run."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import stat
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path

import numpy as np

from samga_brain_rw.hashing import canonical_json_bytes, ordered_ids_sha256
from samga_brain_rw.scores import RetrievalMetrics, ScoreArtifact


REPORT_TYPE = "samga_brain_rw.baseline_parity"
SCOPE = "val-dev"
TOLERANCE = 1e-6
ARTIFACT_DIRECTORIES = {
    "in_loop": "in_loop",
    "saved_checkpoint": "saved_checkpoint",
    "repeat_emission": "repeat_emission",
    "reload_evaluation": "reload_evaluation",
}
_BUNDLE_FILES = ("metadata.json", "predictions.csv", "similarity.npy")
_FORBIDDEN_COMPONENTS = frozenset(
    {
        "formal",
        "formal_input",
        "formal_refit",
        "formal_test",
        "test",
        "test_images",
        "val_confirm",
    }
)
_SUBJECT_TEST_RE = re.compile(r"^sub-\d{2}_test\.json$", re.IGNORECASE)
_FORMAL_TEST_RECORD_SHA256 = (
    "02d7e33b3fe8e5a571f8db232ca5fa86abb0c16981876ec84feae7ba64636f1a"
)
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=(SCOPE,), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.path.normpath(os.fspath(path))))


def _semantic_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _preflight_path(path: Path, context: str) -> Path:
    raw = os.fspath(path)
    if not isinstance(raw, str):
        raise TypeError(f"{context} must be a text path")
    if "\x00" in raw:
        raise ValueError(f"{context} contains a NUL byte")
    if _FORMAL_TEST_RECORD_SHA256 in raw.lower():
        raise PermissionError(f"{context} contains the formal-test record hash")
    for component in Path(raw).parts:
        if (
            _semantic_name(component) in _FORBIDDEN_COMPONENTS
            or _SUBJECT_TEST_RE.fullmatch(component)
        ):
            raise PermissionError(f"{context} contains a sealed-scope component")
    absolute = _absolute(path)
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            value = os.lstat(current)
        except FileNotFoundError:
            break
        except OSError as exc:
            raise ValueError(f"{context} cannot be inspected safely") from exc
        if stat.S_ISLNK(value.st_mode):
            raise ValueError(f"{context} contains a symlink component")
    return absolute


def _open_directory_components(
    path: Path,
    *,
    create: bool,
    context: str,
) -> int:
    absolute = _absolute(path)
    descriptor = os.open(
        absolute.anchor,
        os.O_RDONLY | _O_DIRECTORY | _O_CLOEXEC,
    )
    try:
        for component in absolute.parts[1:]:
            try:
                next_descriptor = os.open(
                    component,
                    os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC,
                    dir_fd=descriptor,
                )
            except FileNotFoundError as exc:
                if not create:
                    raise ValueError(
                        f"{context} could not be opened securely"
                    ) from exc
                try:
                    os.mkdir(component, mode=0o700, dir_fd=descriptor)
                    os.fsync(descriptor)
                    next_descriptor = os.open(
                        component,
                        os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC,
                        dir_fd=descriptor,
                    )
                except OSError as create_exc:
                    raise ValueError(
                        f"{context} contains an unsafe component"
                    ) from create_exc
            except OSError as exc:
                raise ValueError(
                    f"{context} contains a symlink or unsafe component"
                ) from exc
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _directory_identity_payload(value: os.stat_result) -> dict[str, int]:
    return {
        "device": value.st_dev,
        "inode": value.st_ino,
    }


def _open_relative_directory(parent_fd: int, name: str, context: str) -> int:
    if not name or Path(name).name != name:
        raise ValueError(f"{context} must be a single directory name")
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC,
            dir_fd=parent_fd,
        )
    except OSError as exc:
        raise ValueError(f"{context} could not be opened securely") from exc
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ValueError(f"{context} must be a directory")
    return descriptor


def _hash_relative_file(
    directory_fd: int,
    name: str,
    context: str,
) -> tuple[dict[str, object], os.stat_result]:
    if not name or Path(name).name != name:
        raise ValueError(f"{context} must be a single filename")
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | _O_NOFOLLOW | _O_CLOEXEC,
            dir_fd=directory_fd,
        )
    except OSError as exc:
        raise ValueError(f"{context} could not be opened securely") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{context} must be a regular file")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        if _identity(before) != _identity(after):
            raise ValueError(f"{context} changed while it was hashed")
        return {
            "sha256": digest.hexdigest(),
            "size": after.st_size,
        }, after
    finally:
        os.close(descriptor)


def _expected_payload_identity(artifact: ScoreArtifact) -> tuple[int, ...]:
    verified = artifact.verified
    return (
        verified.device,
        verified.inode,
        verified.size,
        verified.mtime_ns,
        verified.ctime_ns,
    )


def _expected_envelope_identity(artifact: ScoreArtifact) -> tuple[int, ...]:
    verified = artifact.verified
    return (
        verified.envelope_device,
        verified.envelope_inode,
        verified.envelope_size,
        verified.envelope_mtime_ns,
        verified.envelope_ctime_ns,
    )


def _load_pinned_artifact(
    run_directory: Path,
    run_fd: int,
    *,
    role: str,
    directory_name: str,
) -> tuple[ScoreArtifact, dict[str, object], tuple[int, ...]]:
    bundle_fd = _open_relative_directory(
        run_fd,
        directory_name,
        f"{role} score bundle",
    )
    try:
        initial = os.fstat(bundle_fd)
        actual_files = frozenset(os.listdir(bundle_fd))
        if actual_files != frozenset(_BUNDLE_FILES):
            raise ValueError(f"{role} score bundle has an unexpected file set")
        artifact = ScoreArtifact.load(
            run_directory / directory_name,
            allowed_scopes={SCOPE},
        )
        files: dict[str, object] = {}
        stats: dict[str, os.stat_result] = {}
        for name in _BUNDLE_FILES:
            files[name], stats[name] = _hash_relative_file(
                bundle_fd,
                name,
                f"{role} {name}",
            )
        if _identity(stats["similarity.npy"]) != _expected_payload_identity(
            artifact
        ):
            raise ValueError(f"{role} verified score path identity changed")
        if files["similarity.npy"]["sha256"] != artifact.verified.payload_sha256:
            raise ValueError(f"{role} score payload hash binding mismatch")
        if _identity(stats["metadata.json"]) != _expected_envelope_identity(
            artifact
        ):
            raise ValueError(f"{role} verified sidecar path identity changed")
        if files["metadata.json"]["sha256"] != artifact.verified.envelope_sha256:
            raise ValueError(f"{role} metadata sidecar hash binding mismatch")
        if files["predictions.csv"]["sha256"] != artifact.metadata[
            "predictions_sha256"
        ]:
            raise ValueError(f"{role} predictions hash binding mismatch")
        final = os.fstat(bundle_fd)
        if _identity(initial) != _identity(final):
            raise ValueError(f"{role} score bundle changed during parity load")
        return artifact, files, _identity(final)
    finally:
        os.close(bundle_fd)


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw(child) for child in value]
    return value


def _metric_payload(metrics: RetrievalMetrics) -> dict[str, object]:
    return {
        "gallery_count": metrics.gallery_count,
        "query_count": metrics.query_count,
        "top1_count": metrics.top1_count,
        "top1_rate": metrics.top1_rate,
        "top5_count": metrics.top5_count,
        "top5_rate": metrics.top5_rate,
    }


def _prediction_payload(metrics: RetrievalMetrics) -> list[dict[str, object]]:
    return [
        {
            "predicted_gallery_id": prediction.predicted_gallery_id,
            "query_id": prediction.query_id,
            "query_index": prediction.query_index,
            "target_gallery_id": prediction.target_gallery_id,
            "target_rank": prediction.target_rank,
            "top1": prediction.top1,
            "top5": prediction.top5,
        }
        for prediction in metrics.predictions
    ]


def _json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _artifact_report(
    artifact: ScoreArtifact,
    *,
    directory_name: str,
    files: dict[str, object],
) -> dict[str, object]:
    ordered_ids = [*artifact.query_ids, *artifact.gallery_ids]
    prediction_payload = _prediction_payload(artifact.metrics)
    return {
        "directory": directory_name,
        "files": files,
        "gallery_ids_sha256": artifact.gallery_ids_sha256,
        "metrics": _metric_payload(artifact.metrics),
        "ordered_ids_sha256": ordered_ids_sha256(ordered_ids),
        "prediction_semantics_sha256": _json_sha256(prediction_payload),
        "provenance": _thaw(artifact.provenance),
        "query_ids_sha256": artifact.query_ids_sha256,
        "similarity_dtype": str(artifact.similarity.dtype),
        "similarity_shape": [int(value) for value in artifact.similarity.shape],
    }


def _maximum_absolute_difference(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError("score tensor shapes differ")
    difference = np.abs(
        left.astype(np.longdouble, copy=False)
        - right.astype(np.longdouble, copy=False)
    )
    maximum = np.max(difference)
    if not bool(np.isfinite(maximum)):
        raise ValueError("score tensor difference is non-finite")
    return float(maximum)


def _compare_pair(
    left_role: str,
    left: ScoreArtifact,
    right_role: str,
    right: ScoreArtifact,
) -> dict[str, object]:
    if (
        left.query_ids != right.query_ids
        or left.gallery_ids != right.gallery_ids
    ):
        raise ValueError(
            f"ordered IDs differ between {left_role} and {right_role}"
        )
    if left.metrics.predictions != right.metrics.predictions:
        raise ValueError(
            f"Top-1/Top-5 predictions differ between {left_role} and {right_role}"
        )
    if _metric_payload(left.metrics) != _metric_payload(right.metrics):
        raise ValueError(
            f"Top-1/Top-5 metrics differ between {left_role} and {right_role}"
        )
    maximum = _maximum_absolute_difference(left.similarity, right.similarity)
    if maximum > TOLERANCE:
        raise ValueError(
            f"score tensor tolerance exceeded between {left_role} and {right_role}: "
            f"{maximum!r} > {TOLERANCE!r}"
        )
    if left.provenance != right.provenance:
        raise ValueError(
            f"score provenance differs between {left_role} and {right_role}"
        )
    return {
        "left": left_role,
        "max_absolute_score_difference": maximum,
        "metrics_identical": True,
        "ordered_ids_identical": True,
        "predictions_identical": True,
        "provenance_identical": True,
        "right": right_role,
        "within_tolerance": True,
    }


def _require_run_path_identity(
    run_directory: Path,
    expected: os.stat_result,
) -> None:
    descriptor = _open_directory_components(
        run_directory,
        create=False,
        context="baseline run directory",
    )
    try:
        if _identity(os.fstat(descriptor)) != _identity(expected):
            raise ValueError("baseline run directory identity changed")
    finally:
        os.close(descriptor)


def build_baseline_parity_report(
    run_directory: Path,
    *,
    scope: str,
) -> dict[str, object]:
    """Load and pairwise-compare the four locked val-dev score bundles."""

    if scope != SCOPE:
        raise PermissionError("baseline parity scope must be val-dev")
    run = _preflight_path(Path(run_directory), "baseline run directory")
    run_fd = _open_directory_components(
        run,
        create=False,
        context="baseline run directory",
    )
    try:
        initial_run = os.fstat(run_fd)
        if not stat.S_ISDIR(initial_run.st_mode):
            raise ValueError("baseline run must be a directory")
        artifacts: dict[str, ScoreArtifact] = {}
        artifact_reports: dict[str, object] = {}
        directory_identities: set[tuple[int, ...]] = set()
        for role, directory_name in ARTIFACT_DIRECTORIES.items():
            artifact, files, directory_identity = _load_pinned_artifact(
                run,
                run_fd,
                role=role,
                directory_name=directory_name,
            )
            if directory_identity in directory_identities:
                raise ValueError("parity roles must use four distinct score bundles")
            directory_identities.add(directory_identity)
            artifacts[role] = artifact
            artifact_reports[role] = _artifact_report(
                artifact,
                directory_name=directory_name,
                files=files,
            )
        final_run = os.fstat(run_fd)
        if _identity(initial_run) != _identity(final_run):
            raise ValueError("baseline run directory changed during parity check")
        _require_run_path_identity(run, initial_run)
    finally:
        os.close(run_fd)

    comparisons = [
        _compare_pair(
            left_role,
            artifacts[left_role],
            right_role,
            artifacts[right_role],
        )
        for left_role, right_role in combinations(ARTIFACT_DIRECTORIES, 2)
    ]
    maximum = max(
        float(comparison["max_absolute_score_difference"])
        for comparison in comparisons
    )
    shared_provenance = _thaw(artifacts["in_loop"].provenance)
    return {
        "artifacts": artifact_reports,
        "comparisons": comparisons,
        "passed": True,
        "report_type": REPORT_TYPE,
        "run_directory": str(run),
        "run_directory_identity": _directory_identity_payload(initial_run),
        "schema_version": 1,
        "scope": SCOPE,
        "summary": {
            "artifact_count": len(artifacts),
            "comparison_count": len(comparisons),
            "maximum_absolute_score_difference": maximum,
            "shared_provenance_sha256": _json_sha256(shared_provenance),
        },
        "tolerance": TOLERANCE,
    }


def _validate_output_path(path: Path, run_directory: Path) -> Path:
    output = _preflight_path(path, "baseline parity output")
    if output.suffix != ".json" or not output.name:
        raise ValueError("baseline parity output must be a .json file")
    for directory_name in ARTIFACT_DIRECTORIES.values():
        bundle = run_directory / directory_name
        if output == bundle or bundle in output.parents:
            raise ValueError("baseline parity output must be outside score bundles")
    try:
        os.lstat(output)
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise ValueError("baseline parity output cannot be inspected safely") from exc
    else:
        raise FileExistsError("baseline parity output already exists")
    return output


def _write_exclusive_report(path: Path, report: Mapping[str, object]) -> None:
    payload = canonical_json_bytes(report) + b"\n"
    parent_fd = _open_directory_components(
        path.parent,
        create=True,
        context="baseline parity output parent",
    )
    descriptor = -1
    created = False
    try:
        descriptor = os.open(
            path.name,
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
            published = os.fstat(handle.fileno())
        os.fsync(parent_fd)
        current_parent = _open_directory_components(
            path.parent,
            create=False,
            context="baseline parity output parent",
        )
        try:
            if _identity(os.fstat(current_parent)) != _identity(os.fstat(parent_fd)):
                raise ValueError("baseline parity output parent identity changed")
            descriptor_check = os.open(
                path.name,
                os.O_RDONLY | _O_NOFOLLOW | _O_CLOEXEC,
                dir_fd=current_parent,
            )
            try:
                if _identity(os.fstat(descriptor_check)) != _identity(published):
                    raise ValueError("baseline parity output identity changed")
            finally:
                os.close(descriptor_check)
        finally:
            os.close(current_parent)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
            descriptor = -1
        if created:
            try:
                os.unlink(path.name, dir_fd=parent_fd)
                os.fsync(parent_fd)
            except OSError:
                pass
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        run_directory = _preflight_path(
            arguments.run_dir,
            "baseline run directory",
        )
        output = _validate_output_path(arguments.output, run_directory)
        report = build_baseline_parity_report(
            run_directory,
            scope=arguments.scope,
        )
        _write_exclusive_report(output, report)
    except (
        FileExistsError,
        OSError,
        OverflowError,
        PermissionError,
        TypeError,
        ValueError,
    ) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
