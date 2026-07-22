"""Auditable score matrices with explicit query-to-gallery identity semantics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import ctypes
from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .provenance import sha256_file


_ARTIFACT_FILES = frozenset({"metadata.json", "similarity.npy"})
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_ID_FIELDS = (
    "query_ids",
    "gallery_entry_ids",
    "gallery_canonical_ids",
    "target_canonical_ids",
)
_ENVELOPE_FIELDS = frozenset(
    {
        "schema_version",
        "similarity_dtype",
        "similarity_shape",
        "similarity_sha256",
        *(_ID_FIELDS),
        *(f"{field}_sha256" for field in _ID_FIELDS),
    }
)


@dataclass(frozen=True)
class ScoreArtifact:
    """One model's score matrix and its explicit ordered identity relations."""

    similarity: np.ndarray
    query_ids: tuple[str, ...]
    gallery_entry_ids: tuple[str, ...]
    gallery_canonical_ids: tuple[str, ...]
    target_canonical_ids: tuple[str, ...]
    metadata: Mapping[str, object]

    def validate(self) -> None:
        """Reject malformed or nominally unanswerable score matrices."""

        if not isinstance(self.similarity, np.ndarray):
            raise ValueError("similarity must be a NumPy array")
        if self.similarity.ndim != 2:
            raise ValueError("similarity must be a non-empty 2-D matrix")
        rows, cols = self.similarity.shape
        if rows < 1 or cols < 1:
            raise ValueError("similarity must be a non-empty 2-D matrix")
        if not np.issubdtype(self.similarity.dtype, np.floating):
            raise ValueError("similarity must use a floating-point dtype")
        if not np.isfinite(self.similarity).all():
            raise ValueError("similarity contains NaN or Inf")
        if len(self.query_ids) != rows or len(self.target_canonical_ids) != rows:
            raise ValueError("query metadata does not match rows")
        if (
            len(self.gallery_entry_ids) != cols
            or len(self.gallery_canonical_ids) != cols
        ):
            raise ValueError("gallery metadata does not match columns")

        for field in _ID_FIELDS:
            values = getattr(self, field)
            if any(not isinstance(value, str) or not value for value in values):
                raise ValueError(f"{field} must contain non-empty strings")
        if len(set(self.query_ids)) != rows:
            raise ValueError("query IDs must be unique")
        if len(set(self.gallery_entry_ids)) != cols:
            raise ValueError("gallery entry IDs must be unique")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")

        allow_unanswerable = self.metadata.get(
            "allow_unanswerable_targets",
            False,
        )
        if not isinstance(allow_unanswerable, bool):
            raise ValueError("allow_unanswerable_targets must be a boolean")
        missing_targets = sorted(
            set(self.target_canonical_ids).difference(self.gallery_canonical_ids)
        )
        if missing_targets and not allow_unanswerable:
            raise ValueError(
                "target canonical IDs missing from gallery: "
                f"{missing_targets}"
            )


def write_score_artifact(directory: Path, artifact: ScoreArtifact) -> None:
    """Exclusively and atomically publish a validated two-file score bundle."""

    artifact.validate()
    directory = Path(directory)
    if _lexists(directory):
        raise FileExistsError(f"score artifact already exists: {directory}")

    metadata = dict(artifact.metadata)
    if any(not isinstance(key, str) for key in metadata):
        raise ValueError("metadata keys must be strings")
    collisions = sorted(_ENVELOPE_FIELDS.intersection(metadata))
    if collisions:
        raise ValueError(f"metadata uses reserved artifact fields: {collisions}")

    parent = directory.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{directory.name}.tmp-", dir=parent)
    )
    try:
        similarity_path = temporary / "similarity.npy"
        with similarity_path.open("xb") as stream:
            np.save(stream, artifact.similarity, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())

        envelope = {
            **metadata,
            "schema_version": 1,
            "similarity_dtype": str(artifact.similarity.dtype),
            "similarity_shape": [int(value) for value in artifact.similarity.shape],
            "similarity_sha256": sha256_file(similarity_path),
        }
        for field in _ID_FIELDS:
            values = list(getattr(artifact, field))
            envelope[field] = values
            envelope[f"{field}_sha256"] = _ordered_ids_sha256(values)

        metadata_bytes = _canonical_json_bytes(envelope) + b"\n"
        metadata_path = temporary / "metadata.json"
        with metadata_path.open("xb") as stream:
            stream.write(metadata_bytes)
            stream.flush()
            os.fsync(stream.fileno())

        _fsync_directory(temporary)
        if _lexists(directory):
            raise FileExistsError(f"score artifact already exists: {directory}")
        _rename_directory_noreplace(temporary, directory)
        _fsync_directory(parent)
    finally:
        if _lexists(temporary):
            shutil.rmtree(temporary)


def publish_staged_directory(staging: Path, destination: Path) -> None:
    """Atomically publish a complete sibling directory without replacement."""
    staging = Path(staging)
    destination = Path(destination)
    if staging.parent.resolve() != destination.parent.resolve():
        raise ValueError("staging and destination must be siblings")
    if staging.is_symlink() or not staging.is_dir():
        raise ValueError("staging must be a regular directory")
    if _lexists(destination):
        raise FileExistsError(f"publication destination exists: {destination}")
    _fsync_directory(staging)
    _rename_directory_noreplace(staging, destination)
    _fsync_directory(destination.parent)


def read_score_artifact(directory: Path) -> ScoreArtifact:
    """Load a complete bundle after verifying its matrix and ordered-ID hashes."""

    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"score artifact is not a regular directory: {directory}")
    actual_files = frozenset(path.name for path in directory.iterdir())
    if actual_files != _ARTIFACT_FILES:
        raise ValueError(
            "score artifact must contain exactly similarity.npy and metadata.json"
        )

    similarity_path = directory / "similarity.npy"
    metadata_path = directory / "metadata.json"
    for path in (similarity_path, metadata_path):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"score artifact member must be a regular file: {path.name}")

    try:
        metadata_text = metadata_path.read_text(encoding="utf-8")
        envelope = json.loads(metadata_text)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("metadata.json is not valid UTF-8 JSON") from error
    if not isinstance(envelope, dict):
        raise ValueError("metadata.json must contain a JSON object")
    try:
        canonical_text = (_canonical_json_bytes(envelope) + b"\n").decode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("metadata.json contains non-canonical values") from error
    if metadata_text != canonical_text:
        raise ValueError("metadata.json is not canonical sorted JSON")
    if envelope.get("schema_version") != 1:
        raise ValueError("unsupported score artifact schema_version")

    ids: dict[str, tuple[str, ...]] = {}
    for field in _ID_FIELDS:
        values = envelope.get(field)
        if not isinstance(values, list) or any(
            not isinstance(value, str) for value in values
        ):
            raise ValueError(f"{field} must be a JSON array of strings")
        expected_hash = envelope.get(f"{field}_sha256")
        if expected_hash != _ordered_ids_sha256(values):
            raise ValueError(f"{field} SHA-256 mismatch")
        ids[field] = tuple(values)

    expected_similarity_hash = envelope.get("similarity_sha256")
    if expected_similarity_hash != sha256_file(similarity_path):
        raise ValueError("similarity SHA-256 mismatch")

    with similarity_path.open("rb") as stream:
        similarity = np.load(stream, allow_pickle=False)
    if not isinstance(similarity, np.ndarray):
        raise ValueError("similarity.npy must contain one NumPy array")
    expected_shape = envelope.get("similarity_shape")
    if expected_shape != [int(value) for value in similarity.shape]:
        raise ValueError("similarity shape does not match metadata")
    if envelope.get("similarity_dtype") != str(similarity.dtype):
        raise ValueError("similarity dtype does not match metadata")

    metadata = {
        key: value
        for key, value in envelope.items()
        if key not in _ENVELOPE_FIELDS
    }
    artifact = ScoreArtifact(
        similarity=similarity,
        query_ids=ids["query_ids"],
        gallery_entry_ids=ids["gallery_entry_ids"],
        gallery_canonical_ids=ids["gallery_canonical_ids"],
        target_canonical_ids=ids["target_canonical_ids"],
        metadata=metadata,
    )
    artifact.validate()
    return artifact


def independent_ranks(artifact: ScoreArtifact) -> np.ndarray:
    """Return deterministic one-based target ranks for independent retrieval."""

    artifact.validate()
    columns_by_canonical_id: dict[str, list[int]] = {}
    for column, canonical_id in enumerate(artifact.gallery_canonical_ids):
        columns_by_canonical_id.setdefault(canonical_id, []).append(column)

    gallery_size = artifact.similarity.shape[1]
    ranks = np.empty(artifact.similarity.shape[0], dtype=np.int64)
    for row, target_id in enumerate(artifact.target_canonical_ids):
        target_columns = columns_by_canonical_id.get(target_id)
        if target_columns is None:
            ranks[row] = gallery_size + 1
            continue
        descending_columns = np.argsort(
            -artifact.similarity[row],
            kind="stable",
        )
        positions = np.empty(gallery_size, dtype=np.int64)
        positions[descending_columns] = np.arange(1, gallery_size + 1)
        ranks[row] = min(int(positions[column]) for column in target_columns)
    return ranks


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _ordered_ids_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256(_canonical_json_bytes(list(values))).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish ``source`` only when ``destination`` is absent."""

    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as error:
        raise RuntimeError(
            "atomic score publication requires Linux renameat2(RENAME_NOREPLACE)"
        ) from error
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int

    ctypes.set_errno(0)
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return

    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            os.strerror(error_number),
            os.fspath(destination),
        )
    unavailable_errors = {
        errno.EINVAL,
        errno.ENOSYS,
        getattr(errno, "ENOTSUP", errno.EINVAL),
        getattr(errno, "EOPNOTSUPP", errno.EINVAL),
    }
    if error_number in unavailable_errors:
        raise RuntimeError(
            "atomic score publication requires Linux renameat2(RENAME_NOREPLACE)"
        ) from OSError(error_number, os.strerror(error_number))
    raise OSError(error_number, os.strerror(error_number), os.fspath(destination))


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))
