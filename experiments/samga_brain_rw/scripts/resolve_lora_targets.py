#!/usr/bin/env python3
"""Resolve the pinned InternViT V2.5 LoRA targets without network access."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import stat
import tempfile
from collections.abc import Sequence
from pathlib import Path

import torch.nn as nn

from samga_brain_rw.hashing import canonical_json_bytes
from samga_brain_rw.internvit import build_lora_target_manifest


PINNED_MODEL_REVISION = "9d1a4344077479c93d42584b6941c64d795d508d"
PINNED_CONFIG_SHA256 = (
    "4fc4a1187b20575c0da8d27df2ad17f5ad6e8ac1c8b2af707bc8b263bd40c0a2"
)
PINNED_MODELING_SHA256 = (
    "56220ba82cb511d51d5f2fa71eebd728b330fbccad9dfb128088a8fcc8f7d260"
)
FORMAL_TEST_RECORD_SHA256 = (
    "02d7e33b3fe8e5a571f8db232ca5fa86abb0c16981876ec84feae7ba64636f1a"
)
_FORMAL_MANIFEST_RE = re.compile(r"(?i)^sub-\d{2}_test\.json$")


def _reject_forbidden_lexical_path(path: Path) -> None:
    for component in Path(path).parts:
        lowered = component.lower()
        if (
            lowered == "test_images"
            or FORMAL_TEST_RECORD_SHA256 in lowered
            or _FORMAL_MANIFEST_RE.fullmatch(component) is not None
        ):
            raise ValueError(f"forbidden path component: {component}")


def _reject_symlink_components(path: Path) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            break
        if stat.S_ISLNK(mode):
            raise ValueError(f"forbidden symlink path component: {current}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_pinned_model_directory(model_path: Path) -> Path:
    path = Path(model_path)
    _reject_forbidden_lexical_path(path)
    _reject_symlink_components(path)
    if path.name != PINNED_MODEL_REVISION:
        raise ValueError(
            "model path must end in the pinned InternViT revision "
            f"{PINNED_MODEL_REVISION}"
        )
    if not path.is_dir():
        raise FileNotFoundError(f"pinned model directory does not exist: {path}")
    expected_files = (
        ("config.json", PINNED_CONFIG_SHA256),
        ("modeling_intern_vit.py", PINNED_MODELING_SHA256),
    )
    for filename, expected_sha256 in expected_files:
        candidate = path / filename
        _reject_symlink_components(candidate)
        if not candidate.is_file():
            raise FileNotFoundError(f"missing pinned model file: {candidate}")
        actual_sha256 = _sha256_file(candidate)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"pinned model file hash mismatch for {filename}: "
                f"{actual_sha256}"
            )
    return path


def load_pinned_model_offline(model_path: Path) -> nn.Module:
    """Load only the hash-checked local pinned revision.

    This is intentionally not exercised by Task 4. The first real model load
    remains gated on Task 17's dedicated safe launcher.
    """
    path = _validate_pinned_model_directory(Path(model_path))
    try:
        from transformers import AutoModel
    except ImportError as exc:  # pragma: no cover - environment failure
        raise RuntimeError("transformers is required for offline model load") from exc

    model = AutoModel.from_pretrained(
        os.fspath(path),
        local_files_only=True,
        trust_remote_code=True,
    )
    if not isinstance(model, nn.Module):
        raise ValueError("offline loader did not return an nn.Module")
    return model


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_exclusive(path: Path, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary_path = Path(temporary_name)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, destination, follow_symlinks=False)
        published = True
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
    if published:
        _fsync_directory(destination.parent)


def publish_lora_target_manifest(
    model: nn.Module,
    output_path: Path,
    first_block: int = 28,
    last_block: int = 36,
) -> dict[str, object]:
    """Resolve and exclusively publish the canonical target manifest."""
    destination = Path(output_path)
    _reject_forbidden_lexical_path(destination)
    _reject_symlink_components(destination)
    manifest = build_lora_target_manifest(model, first_block, last_block)
    _publish_exclusive(
        destination,
        canonical_json_bytes(manifest) + b"\n",
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-block", type=int, default=28)
    parser.add_argument("--last-block", type=int, default=36)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)

    # Classify every path before exists(), hashing, importing Transformers, or
    # loading a model. This local guard remains narrower than the pending Task
    # 3 typed-artifact integration.
    _reject_forbidden_lexical_path(arguments.model_path)
    _reject_forbidden_lexical_path(arguments.output)
    _reject_symlink_components(arguments.model_path)
    _reject_symlink_components(arguments.output)
    if arguments.output.exists():
        raise FileExistsError(
            f"target manifest already exists: {arguments.output}"
        )

    model = load_pinned_model_offline(arguments.model_path)
    publish_lora_target_manifest(
        model,
        arguments.output,
        arguments.first_block,
        arguments.last_block,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
