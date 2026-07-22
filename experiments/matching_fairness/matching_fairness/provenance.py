from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import subprocess
from typing import Any


OFFICIAL_SOURCE_URL = "https://github.com/dongyangli-del/EEG_Image_decode.git"
OFFICIAL_SOURCE_BRANCH = "develop"
REQUIRED_SOURCE_FILES = (
    "Retrieval/train_unified.py",
    "Retrieval/retrieval_engine.py",
    "Retrieval/eeg_encoders.py",
    "eegdatasets.py",
    "encoder_utils.py",
    "models/atms.py",
)


@dataclass(frozen=True)
class SourceLock:
    url: str
    branch: str
    commit: str
    checkout_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    """Hash a regular file or a complete symlink-free directory tree."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"provenance path must not be a symbolic link: {path}")
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise ValueError(f"provenance path must be a file or directory: {path}")

    digest = hashlib.sha256()
    entries = sorted(
        path.rglob("*"),
        key=lambda entry: entry.relative_to(path).as_posix(),
    )
    for entry in entries:
        relative = entry.relative_to(path).as_posix().encode("utf-8")
        if entry.is_symlink():
            raise ValueError(
                f"provenance tree member must not be a symbolic link: {entry}"
            )
        if entry.is_dir():
            digest.update(b"D\0")
            digest.update(relative)
            digest.update(b"\0")
        elif entry.is_file():
            digest.update(b"F\0")
            digest.update(relative)
            digest.update(b"\0")
            digest.update(bytes.fromhex(sha256_file(entry)))
        else:
            raise ValueError(f"unsupported provenance tree member: {entry}")
    return digest.hexdigest()


def _git_output(path: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), *arguments],
        text=True,
        stderr=subprocess.PIPE,
    ).strip()


def _checkout_digest(path: Path) -> str:
    tracked = subprocess.check_output(
        ["git", "-C", str(path), "ls-files", "-z"]
    ).split(b"\0")
    digest = hashlib.sha256()
    for encoded_relative in sorted(item for item in tracked if item):
        relative = encoded_relative.decode("utf-8", errors="surrogateescape")
        tracked_path = path / relative
        if tracked_path.is_symlink():
            raise ValueError(f"tracked source file must not be a symbolic link: {relative}")
        digest.update(encoded_relative)
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(tracked_path)))
    return digest.hexdigest()


def inspect_checkout(
    path: Path,
    *,
    expected_url: str = OFFICIAL_SOURCE_URL,
    expected_branch: str = OFFICIAL_SOURCE_BRANCH,
) -> SourceLock:
    path = Path(path)
    if not path.is_dir():
        raise ValueError(f"source checkout is not a directory: {path}")
    try:
        inside = _git_output(path, "rev-parse", "--is-inside-work-tree")
    except subprocess.CalledProcessError as error:
        raise ValueError(f"source checkout is not a Git worktree: {path}") from error
    if inside != "true":
        raise ValueError(f"source checkout is not a Git worktree: {path}")

    symbolic = subprocess.run(
        ["git", "-C", str(path), "symbolic-ref", "-q", "HEAD"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if symbolic.returncode == 0:
        raise ValueError(
            "source checkout must use detached HEAD, found "
            f"{symbolic.stdout.strip()}"
        )
    if symbolic.returncode != 1:
        raise RuntimeError(
            f"could not determine detached HEAD state for {path}: "
            f"{symbolic.stderr.strip()}"
        )

    status = _git_output(path, "status", "--porcelain")
    if status:
        raise ValueError("source checkout must have a clean worktree")
    try:
        actual_url = _git_output(path, "remote", "get-url", "origin")
    except subprocess.CalledProcessError as error:
        raise ValueError("source checkout must define the exact origin remote URL") from error
    if actual_url != expected_url:
        raise ValueError(
            f"source checkout remote URL mismatch: expected {expected_url}, "
            f"found {actual_url}"
        )

    missing = [
        relative
        for relative in REQUIRED_SOURCE_FILES
        if not (path / relative).is_file() or (path / relative).is_symlink()
    ]
    if missing:
        raise ValueError(f"source checkout is missing required files: {missing}")
    commit = _git_output(path, "rev-parse", "HEAD")
    return SourceLock(
        url=actual_url,
        branch=expected_branch,
        commit=commit,
        checkout_sha256=_checkout_digest(path),
    )
