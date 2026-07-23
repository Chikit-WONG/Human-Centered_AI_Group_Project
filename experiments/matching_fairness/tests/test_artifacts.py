from pathlib import Path
import ctypes
import errno
import fcntl
import hashlib
import json
import multiprocessing
import os

import numpy as np
import pytest

import matching_fairness.artifacts as artifact_module
from matching_fairness.artifacts import (
    ScoreArtifact,
    independent_ranks,
    read_score_artifact,
    write_score_artifact,
)


class _RenameAt2Double:
    def __init__(
        self,
        error_number: int | None,
        *,
        before_result: object | None = None,
    ) -> None:
        self.error_number = error_number
        self.before_result = before_result
        self.argtypes: object | None = None
        self.restype: object | None = None

    def __call__(
        self,
        source_fd: int,
        source: bytes,
        destination_fd: int,
        destination: bytes,
        flags: int,
    ) -> int:
        assert source_fd == artifact_module._AT_FDCWD
        assert destination_fd == artifact_module._AT_FDCWD
        assert flags == artifact_module._RENAME_NOREPLACE
        if callable(self.before_result):
            self.before_result(source, destination)
        if self.error_number is None:
            return 0
        ctypes.set_errno(self.error_number)
        return -1


class _LibcDouble:
    def __init__(self, renameat2: _RenameAt2Double) -> None:
        self.renameat2 = renameat2


def _force_renameat2_result(
    monkeypatch: pytest.MonkeyPatch,
    error_number: int | None,
    *,
    before_result: object | None = None,
) -> None:
    renameat2 = _RenameAt2Double(
        error_number,
        before_result=before_result,
    )
    monkeypatch.setattr(
        artifact_module.ctypes,
        "CDLL",
        lambda *args, **kwargs: _LibcDouble(renameat2),
    )


def _concurrent_fallback_publisher(
    source: str,
    destination: str,
    start: object,
    outcomes: object,
) -> None:
    start.wait()
    try:
        artifact_module._rename_directory_noreplace(
            Path(source),
            Path(destination),
        )
    except FileExistsError:
        outcomes.put("exists")
    except BaseException as error:  # pragma: no cover - diagnostic payload
        outcomes.put(f"error:{type(error).__name__}:{error}")
    else:
        outcomes.put("published")


def _artifact(
    *,
    similarity: np.ndarray | None = None,
    query_ids: tuple[str, ...] = ("q0", "q1"),
    gallery_entry_ids: tuple[str, ...] = ("entry0", "entry1"),
    gallery_canonical_ids: tuple[str, ...] = ("image0", "image1"),
    target_canonical_ids: tuple[str, ...] = ("image0", "image1"),
    metadata: dict[str, object] | None = None,
) -> ScoreArtifact:
    return ScoreArtifact(
        similarity=(
            np.eye(2, dtype=np.float32)
            if similarity is None
            else similarity
        ),
        query_ids=query_ids,
        gallery_entry_ids=gallery_entry_ids,
        gallery_canonical_ids=gallery_canonical_ids,
        target_canonical_ids=target_canonical_ids,
        metadata={"model_slug": "fixture"} if metadata is None else metadata,
    )


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n"


def _ordered_ids_sha256(values: list[str]) -> str:
    encoded = json.dumps(
        values,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_round_trip_preserves_ids_and_matrix(tmp_path: Path) -> None:
    artifact = ScoreArtifact(
        similarity=np.eye(3),
        query_ids=("q0", "q1", "q2"),
        gallery_entry_ids=("e0", "e1", "e2"),
        gallery_canonical_ids=("i0", "i1", "i2"),
        target_canonical_ids=("i0", "i1", "i2"),
        metadata={"model_slug": "fixture"},
    )
    write_score_artifact(tmp_path / "score", artifact)
    loaded = read_score_artifact(tmp_path / "score")
    np.testing.assert_array_equal(loaded.similarity, artifact.similarity)
    assert loaded.query_ids == artifact.query_ids
    assert loaded.target_canonical_ids == artifact.target_canonical_ids


def test_targets_are_resolved_by_canonical_id_not_diagonal() -> None:
    artifact = ScoreArtifact(
        similarity=np.array([[0.1, 0.9], [0.8, 0.2]]),
        query_ids=("q-a", "q-b"),
        gallery_entry_ids=("entry-b", "entry-a"),
        gallery_canonical_ids=("image-b", "image-a"),
        target_canonical_ids=("image-a", "image-b"),
        metadata={"model_slug": "fixture"},
    )
    assert independent_ranks(artifact).tolist() == [1, 1]


def test_highest_scoring_duplicate_canonical_entry_determines_rank() -> None:
    artifact = ScoreArtifact(
        similarity=np.array([[0.95, 0.2, 0.9, 0.1]]),
        query_ids=("q-target",),
        gallery_entry_ids=("distractor-0", "target-a", "target-b", "distractor-1"),
        gallery_canonical_ids=("other-0", "target", "target", "other-1"),
        target_canonical_ids=("target",),
        metadata={"model_slug": "fixture"},
    )

    assert independent_ranks(artifact).tolist() == [2]


def test_write_creates_exact_canonical_bundle_with_audit_hashes(
    tmp_path: Path,
) -> None:
    artifact = _artifact(
        metadata={
            "checkpoint_role": "val_selected_formal",
            "checkpoint_sha256": "a" * 64,
            "data_hashes": {"test": "b" * 64},
            "model_slug": "fixture",
            "query_mode": "standard",
            "score_semantics": "cosine_similarity_higher_is_better",
            "seed": 42,
            "source_commit": "c" * 40,
            "subject": "sub-08",
        }
    )
    directory = tmp_path / "score"

    write_score_artifact(directory, artifact)

    assert {path.name for path in directory.iterdir()} == {
        "metadata.json",
        "similarity.npy",
    }
    metadata_text = (directory / "metadata.json").read_text(encoding="utf-8")
    payload = json.loads(metadata_text)
    assert metadata_text == _canonical_json(payload)
    assert payload["similarity_sha256"] == hashlib.sha256(
        (directory / "similarity.npy").read_bytes()
    ).hexdigest()
    for field in (
        "query_ids",
        "gallery_entry_ids",
        "gallery_canonical_ids",
        "target_canonical_ids",
    ):
        assert payload[f"{field}_sha256"] == _ordered_ids_sha256(payload[field])
    for field, value in artifact.metadata.items():
        assert payload[field] == value


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_validation_rejects_non_finite_similarity(bad_value: float) -> None:
    similarity = np.eye(2)
    similarity[0, 0] = bad_value

    with pytest.raises(ValueError, match="NaN or Inf"):
        _artifact(similarity=similarity).validate()


def test_validation_rejects_duplicate_gallery_entry_ids() -> None:
    with pytest.raises(ValueError, match="gallery entry IDs must be unique"):
        _artifact(gallery_entry_ids=("entry0", "entry0")).validate()


def test_validation_rejects_duplicate_query_ids() -> None:
    with pytest.raises(ValueError, match="query IDs must be unique"):
        _artifact(query_ids=("q0", "q0")).validate()


def test_validation_rejects_row_id_mismatch() -> None:
    with pytest.raises(ValueError, match="query metadata does not match rows"):
        _artifact(query_ids=("q0",)).validate()


def test_validation_rejects_missing_target_by_default() -> None:
    with pytest.raises(ValueError, match="target canonical IDs missing from gallery"):
        _artifact(target_canonical_ids=("image0", "absent")).validate()


def test_explicitly_allowed_unanswerable_target_gets_gallery_size_plus_one_rank(
    tmp_path: Path,
) -> None:
    artifact = _artifact(
        target_canonical_ids=("image0", "absent"),
        metadata={
            "allow_unanswerable_targets": True,
            "model_slug": "fixture",
        },
    )
    directory = tmp_path / "score"

    write_score_artifact(directory, artifact)
    loaded = read_score_artifact(directory)

    assert loaded.target_canonical_ids == ("image0", "absent")
    assert loaded.metadata["allow_unanswerable_targets"] is True
    assert independent_ranks(loaded).tolist() == [1, 3]


def test_read_rejects_tampered_matrix_hash(tmp_path: Path) -> None:
    directory = tmp_path / "score"
    write_score_artifact(directory, _artifact())
    with (directory / "similarity.npy").open("ab") as stream:
        stream.write(b"tampered")

    with pytest.raises(ValueError, match="similarity SHA-256 mismatch"):
        read_score_artifact(directory)


@pytest.mark.parametrize(
    "field",
    (
        "query_ids",
        "gallery_entry_ids",
        "gallery_canonical_ids",
        "target_canonical_ids",
    ),
)
def test_read_verifies_every_stored_ordered_id_hash(
    tmp_path: Path,
    field: str,
) -> None:
    directory = tmp_path / "score"
    write_score_artifact(directory, _artifact())
    path = directory / "metadata.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = list(reversed(payload[field]))
    path.write_text(_canonical_json(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=f"{field} SHA-256 mismatch"):
        read_score_artifact(directory)


@pytest.mark.parametrize("present_file", ("similarity.npy", "metadata.json"))
def test_read_rejects_partial_artifact(
    tmp_path: Path,
    present_file: str,
) -> None:
    directory = tmp_path / "partial"
    directory.mkdir()
    (directory / present_file).write_bytes(b"partial")

    with pytest.raises(ValueError, match="exactly similarity.npy and metadata.json"):
        read_score_artifact(directory)


def test_write_is_exclusive_and_preserves_existing_artifact(tmp_path: Path) -> None:
    directory = tmp_path / "score"
    write_score_artifact(directory, _artifact())
    before = {path.name: path.read_bytes() for path in directory.iterdir()}

    with pytest.raises(FileExistsError):
        write_score_artifact(directory, _artifact(similarity=2 * np.eye(2)))

    assert {path.name: path.read_bytes() for path in directory.iterdir()} == before


def test_write_atomically_preserves_destination_created_during_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "score"
    marker = tmp_path / "existing-destination.marker"
    real_lexists = artifact_module._lexists
    destination_checks = 0

    def inject_racing_destination(path: Path) -> bool:
        nonlocal destination_checks
        if Path(path) == directory:
            destination_checks += 1
            if destination_checks == 2:
                directory.mkdir()
                marker.write_text(str(directory.stat().st_ino), encoding="utf-8")
                return False
        return real_lexists(path)

    monkeypatch.setattr(artifact_module, "_lexists", inject_racing_destination)

    with pytest.raises(FileExistsError):
        write_score_artifact(directory, _artifact())

    assert destination_checks == 2
    assert directory.is_dir()
    assert list(directory.iterdir()) == []
    assert marker.read_text(encoding="utf-8") == str(directory.stat().st_ino)
    assert {path.name for path in tmp_path.iterdir()} == {
        "existing-destination.marker",
        "score",
    }


def test_failed_write_does_not_publish_partial_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def interrupted_save(*args: object, **kwargs: object) -> None:
        raise OSError("interrupted")

    monkeypatch.setattr(artifact_module.np, "save", interrupted_save)
    directory = tmp_path / "score"

    with pytest.raises(OSError, match="interrupted"):
        write_score_artifact(directory, _artifact())

    assert not directory.exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "error_number",
    tuple(
        dict.fromkeys(
            (
                errno.EINVAL,
                errno.ENOSYS,
                getattr(errno, "ENOTSUP", errno.EINVAL),
                getattr(errno, "EOPNOTSUPP", errno.EINVAL),
            )
        )
    ),
)
def test_capability_errors_publish_complete_directory_through_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_number: int,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    (source / "payload").write_bytes(b"complete")
    _force_renameat2_result(monkeypatch, error_number)

    artifact_module._rename_directory_noreplace(source, destination)

    assert not source.exists()
    assert (destination / "payload").read_bytes() == b"complete"


@pytest.mark.parametrize(
    "destination_kind",
    ("empty-directory", "nonempty-directory", "file", "symlink", "broken-symlink"),
)
def test_fallback_preserves_every_existing_destination_kind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    destination_kind: str,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    (source / "payload").write_bytes(b"candidate")
    if destination_kind in {"empty-directory", "nonempty-directory"}:
        destination.mkdir()
        if destination_kind == "nonempty-directory":
            (destination / "marker").write_bytes(b"existing")
    elif destination_kind == "file":
        destination.write_bytes(b"existing")
    elif destination_kind == "symlink":
        target = tmp_path / "symlink-target"
        target.write_bytes(b"existing")
        destination.symlink_to(target.name)
    else:
        destination.symlink_to("missing-target")
    before = os.lstat(destination)
    _force_renameat2_result(monkeypatch, errno.EINVAL)

    with pytest.raises(FileExistsError):
        artifact_module._rename_directory_noreplace(source, destination)

    after = os.lstat(destination)
    assert (after.st_dev, after.st_ino, after.st_mode) == (
        before.st_dev,
        before.st_ino,
        before.st_mode,
    )
    assert (source / "payload").read_bytes() == b"candidate"
    if destination_kind == "nonempty-directory":
        assert (destination / "marker").read_bytes() == b"existing"
    elif destination_kind == "file":
        assert destination.read_bytes() == b"existing"


def test_two_cooperative_fallback_publishers_have_exactly_one_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = (tmp_path / "source-a", tmp_path / "source-b")
    artifacts = (_artifact(), _artifact(similarity=2 * np.eye(2, dtype=np.float32)))
    for source, artifact in zip(sources, artifacts, strict=True):
        write_score_artifact(source, artifact)
    destination = tmp_path / "destination"
    _force_renameat2_result(monkeypatch, errno.EINVAL)
    context = multiprocessing.get_context("fork")
    start = context.Event()
    outcomes = context.Queue()
    processes = [
        context.Process(
            target=_concurrent_fallback_publisher,
            args=(str(source), str(destination), start, outcomes),
        )
        for source in sources
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(timeout=10)

    assert [process.exitcode for process in processes] == [0, 0]
    assert sorted(outcomes.get(timeout=1) for _ in processes) == [
        "exists",
        "published",
    ]
    published = read_score_artifact(destination)
    assert any(
        np.array_equal(published.similarity, artifact.similarity)
        for artifact in artifacts
    )


@pytest.mark.parametrize("source_kind", ("symlink", "file"))
def test_fallback_rejects_non_directory_source_without_publishing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_kind: str,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    if source_kind == "symlink":
        target = tmp_path / "target"
        target.mkdir()
        source.symlink_to(target.name, target_is_directory=True)
    else:
        source.write_bytes(b"not-a-directory")
    _force_renameat2_result(monkeypatch, errno.EINVAL)

    with pytest.raises(ValueError, match="source must be a non-symlink directory"):
        artifact_module._rename_directory_noreplace(source, destination)

    assert not os.path.lexists(destination)


def test_fallback_rejects_source_replaced_after_capability_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    displaced = tmp_path / "displaced"
    source.mkdir()
    (source / "original").write_bytes(b"original")

    def replace_source(_source: bytes, _destination: bytes) -> None:
        source.rename(displaced)
        source.mkdir()
        (source / "replacement").write_bytes(b"replacement")

    _force_renameat2_result(
        monkeypatch,
        errno.EINVAL,
        before_result=replace_source,
    )

    with pytest.raises(RuntimeError, match="source changed during publication"):
        artifact_module._rename_directory_noreplace(source, destination)

    assert not destination.exists()
    assert (displaced / "original").read_bytes() == b"original"
    assert (source / "replacement").read_bytes() == b"replacement"


def test_fallback_rejects_non_sibling_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source-parent" / "source"
    destination = tmp_path / "destination-parent" / "destination"
    source.mkdir(parents=True)
    destination.parent.mkdir()
    _force_renameat2_result(monkeypatch, errno.EINVAL)

    with pytest.raises(ValueError, match="source and destination must be siblings"):
        artifact_module._rename_directory_noreplace(source, destination)

    assert source.is_dir()
    assert not destination.exists()


@pytest.mark.parametrize("traversal_operand", ("source", "destination"))
def test_publication_rejects_dotdot_instead_of_treating_it_as_a_basename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    traversal_operand: str,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    if traversal_operand == "source":
        source = parent / ".."
        destination = parent / "destination"
    else:
        source = parent / "source"
        source.mkdir()
        destination = parent / ".."
    _force_renameat2_result(monkeypatch, errno.EINVAL)

    with pytest.raises(ValueError, match="ordinary basenames"):
        artifact_module._rename_directory_noreplace(source, destination)

    assert not (parent / "destination").exists()


def test_fallback_rejects_symlink_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    parent = tmp_path / "parent-link"
    parent.symlink_to(real_parent.name, target_is_directory=True)
    source = parent / "source"
    source.mkdir()
    destination = parent / "destination"
    _force_renameat2_result(monkeypatch, errno.EINVAL)

    with pytest.raises(OSError):
        artifact_module._rename_directory_noreplace(source, destination)

    assert source.is_dir()
    assert not destination.exists()


@pytest.mark.parametrize("error_number", (errno.EACCES, errno.EPERM, errno.EIO))
def test_non_capability_errors_never_attempt_fallback_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_number: int,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    _force_renameat2_result(monkeypatch, error_number)

    def forbidden_rename(*args: object, **kwargs: object) -> None:
        raise AssertionError("ordinary rename must not be attempted")

    monkeypatch.setattr(artifact_module.os, "rename", forbidden_rename)
    with pytest.raises(OSError) as captured:
        artifact_module._rename_directory_noreplace(source, destination)

    assert captured.value.errno == error_number
    assert source.is_dir()
    assert not destination.exists()


def test_kernel_success_path_remains_preferred(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()

    def kernel_rename(source_bytes: bytes, destination_bytes: bytes) -> None:
        os.rename(os.fsdecode(source_bytes), os.fsdecode(destination_bytes))

    _force_renameat2_result(
        monkeypatch,
        None,
        before_result=kernel_rename,
    )

    artifact_module._rename_directory_noreplace(source, destination)

    assert not source.exists()
    assert destination.is_dir()


def test_kernel_eexist_path_remains_no_clobber(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    inode = destination.stat().st_ino
    _force_renameat2_result(monkeypatch, errno.EEXIST)

    with pytest.raises(FileExistsError):
        artifact_module._rename_directory_noreplace(source, destination)

    assert source.is_dir()
    assert destination.stat().st_ino == inode


def test_fallback_rename_failure_leaves_no_destination_or_temporary_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "score"
    _force_renameat2_result(monkeypatch, errno.EINVAL)

    def failed_rename(*args: object, **kwargs: object) -> None:
        raise OSError(errno.EIO, os.strerror(errno.EIO))

    monkeypatch.setattr(artifact_module.os, "rename", failed_rename)

    with pytest.raises(OSError) as captured:
        write_score_artifact(destination, _artifact())

    assert captured.value.errno == errno.EIO
    assert not os.path.lexists(destination)
    assert not list(tmp_path.glob(".score.tmp-*"))


def test_fallback_orders_directory_fsync_and_parent_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / "staging"
    destination = tmp_path / "destination"
    staging.mkdir()
    (staging / "complete").write_bytes(b"complete")
    _force_renameat2_result(monkeypatch, errno.EINVAL)
    events: list[object] = []
    real_open = os.open
    real_rename = os.rename

    def recording_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
        descriptor = real_open(path, flags, *args, **kwargs)
        events.append(("open", os.fspath(path), flags, descriptor))
        return descriptor

    def recording_fsync(descriptor: int) -> None:
        events.append(("fsync", descriptor))

    def recording_flock(descriptor: int, operation: int) -> None:
        events.append(("flock", descriptor, operation))

    def recording_rename(
        source: object,
        target: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        events.append(("rename", source, target, dict(kwargs)))
        real_rename(source, target, *args, **kwargs)

    monkeypatch.setattr(artifact_module.os, "open", recording_open)
    monkeypatch.setattr(artifact_module.os, "fsync", recording_fsync)
    monkeypatch.setattr(fcntl, "flock", recording_flock)
    monkeypatch.setattr(artifact_module.os, "rename", recording_rename)

    artifact_module.publish_staged_directory(staging, destination)

    assert [event[0] for event in events] == [
        "open",
        "fsync",
        "open",
        "flock",
        "rename",
        "fsync",
        "flock",
        "open",
        "fsync",
    ]
    lock_fd = events[3][1]
    assert events[3] == ("flock", lock_fd, fcntl.LOCK_EX)
    assert events[4][3] == {"src_dir_fd": lock_fd, "dst_dir_fd": lock_fd}
    assert events[5] == ("fsync", lock_fd)
    assert events[6] == ("flock", lock_fd, fcntl.LOCK_UN)
    fallback_parent_open = events[2]
    assert fallback_parent_open[1] == os.fspath(tmp_path)
    required_flags = (
        getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    assert fallback_parent_open[2] & required_flags == required_flags
