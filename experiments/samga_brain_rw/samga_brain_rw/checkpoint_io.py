"""Shared fail-closed IO for typed PyTorch checkpoint bundles.

This module owns transport verification only.  A consumer remains responsible
for validating the checkpoint's payload-specific semantic schema.
"""

from __future__ import annotations

import json
import pickle
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import torch

from .access import AccessScope, TypedArtifact, verify_typed_artifacts


@dataclass(frozen=True)
class LoadedTypedTorchCheckpoint:
    """A weights-only payload bound to its verified typed envelope."""

    payload: Mapping[str, object]
    envelope: Mapping[str, object]
    sha256: str


def checkpoint_sidecar(path: Path) -> Path:
    """Return the canonical same-directory envelope path for ``path``."""

    value = Path(path)
    return value.with_suffix(value.suffix + ".meta.json")


def load_typed_torch_checkpoint(
    path: Path,
    *,
    payload_type: str,
    requested_scope: AccessScope,
) -> LoadedTypedTorchCheckpoint:
    """Verify a typed bundle and weights-only load its exact payload bytes.

    ``verify_typed_artifacts`` validates the strict JSON envelope, its semantic
    hashes, scope, payload hash, no-symlink traversal, and stable file
    identities.  Both capabilities are re-opened and revalidated here so the
    semantic PyTorch load cannot race a sidecar or payload replacement.
    """

    checkpoint_path = Path(path)
    if not isinstance(payload_type, str) or not payload_type:
        raise ValueError("checkpoint payload_type must be nonempty")
    capability = verify_typed_artifacts(
        requested_scope,
        [
            TypedArtifact(
                payload_type,
                checkpoint_path,
                checkpoint_sidecar(checkpoint_path),
            )
        ],
    )[0]
    with capability.open_envelope_verified() as handle:
        try:
            envelope = json.load(handle)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
            raise ValueError(
                "typed checkpoint envelope could not be loaded"
            ) from exc
    if not isinstance(envelope, Mapping):
        raise ValueError("typed checkpoint envelope must be a mapping")
    with capability.open_verified() as handle:
        try:
            payload = torch.load(
                handle,
                map_location="cpu",
                weights_only=True,
            )
        except (
            AssertionError,
            EOFError,
            IndexError,
            KeyError,
            OSError,
            RuntimeError,
            struct.error,
            TypeError,
            ValueError,
            pickle.UnpicklingError,
        ) as exc:
            raise ValueError(
                "typed checkpoint payload could not be loaded safely"
            ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError("typed checkpoint payload must be a mapping")
    if any(not isinstance(key, str) for key in payload):
        raise ValueError("typed checkpoint payload keys must be strings")

    capability.revalidate_envelope()
    return LoadedTypedTorchCheckpoint(
        payload=MappingProxyType(dict(payload)),
        envelope=MappingProxyType(dict(envelope)),
        sha256=capability.payload_sha256,
    )


__all__ = [
    "LoadedTypedTorchCheckpoint",
    "checkpoint_sidecar",
    "load_typed_torch_checkpoint",
]
