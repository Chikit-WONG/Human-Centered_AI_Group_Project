"""Verified, component-only loading for the pinned read-only SAMGA checkout.

The upstream CLI module is deliberately outside the import surface.  Source
bytes for the three allowlisted component modules are read from the pinned Git
object, compared with the clean working-tree bytes, then executed under
private module names.  No upstream file is modified and ``train.py`` is never
imported.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any


_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_PINNED_COMMIT = "1a63745b7ff6f98dad34b0f0b8246a9b5260d9c1"
_COMPONENT_FILES = (
    "module/eeg_encoder/model.py",
    "module/projector.py",
    "module/loss.py",
)


@dataclass(frozen=True)
class UpstreamComponents:
    """Classes and functions loaded from one verified upstream commit."""

    upstream_root: Path
    commit: str
    source_sha256: Mapping[str, str]
    EEGProject: type
    ProjectorDirect: type
    ProjectorLinear: type
    ProjectorMLP: type
    ShareEncoder: type
    SubjectAwareLayerMixer: type
    ContrastiveLoss: type
    mmd_rbf: Callable[..., Any]


def load_locked_upstream_components(
    upstream_root: Path,
    expected_commit: str,
) -> UpstreamComponents:
    """Load only allowlisted component modules from a clean pinned checkout."""

    if not isinstance(expected_commit, str) or not _COMMIT_RE.fullmatch(
        expected_commit
    ):
        raise ValueError("expected upstream commit must be 40 lowercase hex")
    if expected_commit != _PINNED_COMMIT:
        raise ValueError(f"upstream commit is locked to {_PINNED_COMMIT}")
    root = _verified_root(Path(upstream_root))
    _require_git_snapshot(root, expected_commit)

    modules: dict[str, ModuleType] = {}
    source_sha256: dict[str, str] = {}
    for relative in _COMPONENT_FILES:
        working_path = root / relative
        working_bytes = _read_regular_nofollow(working_path)
        committed_bytes = _git_bytes(
            root,
            "show",
            f"{expected_commit}:{relative}",
        )
        if working_bytes != committed_bytes:
            raise ValueError(
                f"upstream component differs from pinned commit: {relative}"
            )
        source_sha256[relative] = hashlib.sha256(committed_bytes).hexdigest()
        module_name = (
            f"_samga_locked_{expected_commit[:12]}_"
            f"{relative.replace('/', '_').removesuffix('.py')}"
        )
        modules[relative] = _execute_source(
            module_name,
            working_path,
            committed_bytes,
        )

    _require_git_snapshot(root, expected_commit)
    eeg_module = modules["module/eeg_encoder/model.py"]
    projector_module = modules["module/projector.py"]
    loss_module = modules["module/loss.py"]
    return UpstreamComponents(
        upstream_root=root,
        commit=expected_commit,
        source_sha256=MappingProxyType(source_sha256),
        EEGProject=_require_class(eeg_module, "EEGProject"),
        ProjectorDirect=_require_class(projector_module, "ProjectorDirect"),
        ProjectorLinear=_require_class(projector_module, "ProjectorLinear"),
        ProjectorMLP=_require_class(projector_module, "ProjectorMLP"),
        ShareEncoder=_require_class(projector_module, "ShareEncoder"),
        SubjectAwareLayerMixer=_require_class(
            eeg_module,
            "SubjectAwareLayerMixer",
        ),
        ContrastiveLoss=_require_class(loss_module, "ContrastiveLoss"),
        mmd_rbf=_require_callable(loss_module, "mmd_rbf"),
    )


def _verified_root(path: Path) -> Path:
    raw = os.fspath(path)
    if not isinstance(raw, str) or "\x00" in raw:
        raise ValueError("upstream root must be a safe text path")
    absolute = Path(os.path.abspath(os.path.normpath(raw)))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            value = os.lstat(current)
        except OSError as exc:
            raise ValueError("upstream root cannot be inspected safely") from exc
        if stat.S_ISLNK(value.st_mode):
            raise ValueError("upstream root contains a symlink component")
    if not absolute.is_dir():
        raise ValueError("upstream root must be an existing directory")
    root = absolute.resolve(strict=True)
    top_level = Path(_git_text(root, "rev-parse", "--show-toplevel"))
    if top_level.resolve(strict=True) != root:
        raise ValueError("upstream root must be the Git checkout top level")
    return root


def _require_git_snapshot(root: Path, expected_commit: str) -> None:
    head = _git_text(root, "rev-parse", "HEAD")
    if head != expected_commit:
        raise ValueError(
            f"upstream commit mismatch: expected {expected_commit}, found {head}"
        )
    status = _git_bytes(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if status:
        raise ValueError("upstream checkout must be clean before component load")


def _git_text(root: Path, *arguments: str) -> str:
    return _git_bytes(root, *arguments).decode("utf-8").strip()


def _git_bytes(root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ("git", "-C", os.fspath(root), *arguments),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"cannot verify upstream Git checkout: {message}")
    return completed.stdout


def _read_regular_nofollow(path: Path) -> bytes:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"upstream component is not a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if _identity(before) != _identity(after):
            raise ValueError(f"upstream component changed while read: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _execute_source(name: str, path: Path, source: bytes) -> ModuleType:
    try:
        code = compile(source, os.fspath(path), "exec", dont_inherit=True)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"invalid pinned upstream source: {path}") from exc
    module = ModuleType(name)
    module.__file__ = os.fspath(path)
    module.__package__ = ""
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        exec(code, module.__dict__)
    except BaseException:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
        raise
    return module


def _require_class(module: ModuleType, name: str) -> type:
    value = getattr(module, name, None)
    if not isinstance(value, type):
        raise ValueError(f"pinned upstream component is missing class {name}")
    return value


def _require_callable(module: ModuleType, name: str) -> Callable[..., Any]:
    value = getattr(module, name, None)
    if not callable(value):
        raise ValueError(f"pinned upstream component is missing function {name}")
    return value
