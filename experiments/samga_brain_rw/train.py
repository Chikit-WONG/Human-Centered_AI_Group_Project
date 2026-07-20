#!/usr/bin/env python3
"""Development-only frozen-cache SAMGA trainer."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import re
import secrets
import stat
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from types import MappingProxyType

import numpy as np
import torch
from torch import nn

from samga_brain_rw.brainrw import (
    ManifestIdentity,
    _node_identity,
    _secure_parent_directory,
    _unlink_created_file,
    _write_relative_exclusive,
    checkpoint_sidecar,
    create_development_directory_exclusive,
    load_development_manifest_identity,
    reject_development_path,
    write_development_file_exclusive,
)
from samga_brain_rw.candidate_registry import (
    select_stage2_factor_identity,
    stage2_registry_identities,
)
from samga_brain_rw.checkpoint_identity import (
    validate_epoch_checkpoint_identity,
)
from samga_brain_rw.checkpoint_io import load_typed_torch_checkpoint
from samga_brain_rw.checkpoints import CHECKPOINT_PAYLOAD_TYPE, hash_state_dict
from samga_brain_rw.config import (
    ProtocolConfig,
    SemanticConfig,
    resolve_run_config,
)
from samga_brain_rw.data import POSTERIOR_CHANNELS, ProtocolSubjectDataset
from samga_brain_rw.feature_transforms import (
    TrainWhitening,
)
from samga_brain_rw.hashing import (
    canonical_json_bytes,
    ordered_ids_sha256,
    sha256_json,
)
from samga_brain_rw.score_provenance import (
    development_score_source_records,
)
from samga_brain_rw.runtime_contract import (
    require_production_runtime,
    validate_environment_binding,
)
from samga_brain_rw.scores import ScoreArtifact
from samga_brain_rw.trainer import (
    CheckpointPublication,
    SCHEDULE as _TRAINING_SCHEDULE,
    SCHEDULE_SHA256,
    TrainingCellSpec,
    TrainingIdentities,
    derive_training_identities,
    run_training_cell,
)
from samga_brain_rw.upstream_samga import (
    UpstreamComponents,
    load_locked_upstream_components,
)


PINNED_UPSTREAM_SHA = "1a63745b7ff6f98dad34b0f0b8246a9b5260d9c1"
RUN_PAYLOAD_TYPE = "samga_brain_rw.development_run"
SCHEDULE = _TRAINING_SCHEDULE
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SENSITIVE = {
    "formal",
    "formal_input",
    "formal_refit",
    "formal_test",
    "test",
    "test_images",
    "val_confirm",
}
_CHECKPOINT_SCOPE_KEYS = frozenset({"scope", "validation_scope", "observed_scopes"})
_DURABLE_CHECKPOINT_NAMES = tuple(
    f"checkpoint_epoch{epoch:03d}.pt"
    for epoch in range(51, 61)
)
_CHECKPOINT_NAME_RE = re.compile(
    r"^checkpoint_epoch(?P<epoch>\d{3})(?:_step\d{8})?\.pt$"
)


@dataclass(frozen=True)
class LoadedSAMGACheckpoint:
    payload: Mapping[str, object]
    sha256: str


def samga_checkpoint_sidecar(path: Path) -> Path:
    return checkpoint_sidecar(Path(path))


def _reject_sealed_checkpoint_metadata(
    value: object,
    *,
    context: str = "checkpoint",
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if (
                context == "checkpoint.optimizer_state_dict.state"
                and type(key) is int
                and key >= 0
            ):
                child_context = f"{context}[{key}]"
            elif isinstance(key, str):
                token = _normalized_token(key)
                if token in _SENSITIVE or re.fullmatch(
                    r"sub_?\d+_test(?:_.*)?",
                    token,
                ):
                    raise PermissionError(
                        f"{context} contains sealed test/formal/confirm metadata"
                    )
                child_context = f"{context}.{key}"
            else:
                raise ValueError(f"{context} keys must be strings")
            _reject_sealed_checkpoint_metadata(
                child,
                context=child_context,
            )
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_sealed_checkpoint_metadata(
                child,
                context=f"{context}[{index}]",
            )
        return
    if isinstance(value, str):
        lowered = value.lower()
        if (
            "02d7e33b3fe8e5a571f8db232ca5fa86abb0c16981876ec84feae7ba64636f1a"
        ) in lowered:
            raise PermissionError(f"{context} contains the formal-test record hash")
        for component in Path(value).parts:
            token = _normalized_token(component)
            if token in _SENSITIVE or re.fullmatch(
                r"sub_?\d+_test(?:_.*)?",
                token,
            ):
                raise PermissionError(f"{context} contains a sealed path component")


def _validated_samga_checkpoint_payload(
    payload: object,
    *,
    transport: bool = False,
) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise ValueError("SAMGA checkpoint payload must be a mapping")
    value = dict(payload)
    required = {
        "schema_version",
        "payload_type",
        "epoch",
        "global_step",
        "subject",
        "seed",
        "config_sha256",
        "schedule_sha256",
        "optimizer_stage",
        "trajectory_sha256",
        "data_order_sha256",
        "model_state_dict",
        "model_state_sha256",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "python_rng_state",
        "numpy_rng_state",
        "torch_rng_state",
        "cuda_rng_states",
        "loader_generator_state",
        "sampler_state_dict",
        "validation_metrics",
        "input_hashes",
        "effective_batch",
        "environment",
        "run_manifest",
        "candidate_spec",
        "runtime_state",
        "retention",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"SAMGA checkpoint is missing required field {missing[0]}")
    extra = set(value) - required
    expected_extra = set(_CHECKPOINT_SCOPE_KEYS) if transport else set()
    if extra != expected_extra:
        raise ValueError("SAMGA checkpoint contains unexpected fields")
    if value["schema_version"] != 1 or value["payload_type"] != CHECKPOINT_PAYLOAD_TYPE:
        raise ValueError("SAMGA checkpoint identity/schema mismatch")
    if type(value["epoch"]) is not int or not 1 <= value["epoch"] <= 60:
        raise ValueError("SAMGA checkpoint epoch is invalid")
    if type(value["global_step"]) is not int or value["global_step"] < 0:
        raise ValueError("SAMGA checkpoint global_step is invalid")
    if (
        type(value["subject"]) is not int
        or not 1 <= value["subject"] <= 10
        or type(value["seed"]) is not int
        or value["seed"] < 0
    ):
        raise ValueError("SAMGA checkpoint subject/seed is invalid")
    for key in (
        "config_sha256",
        "schedule_sha256",
        "trajectory_sha256",
        "data_order_sha256",
        "model_state_sha256",
    ):
        _require_sha(value[key], f"SAMGA checkpoint {key}")
    model_state = value["model_state_dict"]
    if not isinstance(model_state, Mapping) or any(
        not isinstance(key, str) or not isinstance(tensor, torch.Tensor)
        for key, tensor in model_state.items()
    ):
        raise ValueError("SAMGA checkpoint model state is invalid")
    if hash_state_dict(model_state) != value["model_state_sha256"]:
        raise ValueError("SAMGA checkpoint model-state hash mismatch")
    for key in (
        "optimizer_state_dict",
        "scheduler_state_dict",
        "numpy_rng_state",
        "sampler_state_dict",
        "validation_metrics",
        "input_hashes",
        "environment",
        "run_manifest",
        "candidate_spec",
        "runtime_state",
        "retention",
    ):
        if not isinstance(value[key], Mapping):
            raise ValueError(f"SAMGA checkpoint {key} must be a mapping")
    _reject_sealed_checkpoint_metadata(value)
    return value


def _read_relative_regular(
    directory_fd: int,
    name: str,
    *,
    context: str,
) -> bytes:
    descriptor = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=directory_fd,
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{context} must be a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError(f"{context} changed while read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _regular_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        stat.S_IFMT(value.st_mode),
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _rename_stable_identity(value: tuple[int, ...]) -> tuple[int, ...]:
    # Some filesystems update inode ctime when a directory entry is renamed.
    # Device, inode, type, size, and mtime remain stable across that operation.
    return value[:-1]


def _read_prunable_regular(
    parent: object,
    leaf: str,
    *,
    context: str,
) -> tuple[bytes, tuple[int, ...]]:
    parent.verify()
    try:
        before = os.stat(
            leaf,
            dir_fd=parent.parent_fd,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise ValueError(f"{context} cannot be inspected safely") from exc
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{context} must be a regular file")
    try:
        raw = _read_relative_regular(
            parent.parent_fd,
            leaf,
            context=context,
        )
        after = os.stat(
            leaf,
            dir_fd=parent.parent_fd,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise ValueError(f"{context} changed while read") from exc
    identity = _regular_identity(before)
    if _regular_identity(after) != identity:
        raise ValueError(f"{context} changed while read")
    parent.verify()
    return raw, identity


def _sha256_prunable_regular(
    parent: object,
    leaf: str,
    *,
    context: str,
) -> tuple[str, tuple[int, ...]]:
    parent.verify()
    try:
        descriptor = os.open(
            leaf,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent.parent_fd,
        )
    except OSError as exc:
        raise ValueError(f"{context} cannot be opened safely") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{context} must be a regular file")
        identity = _regular_identity(before)
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        if _regular_identity(after) != identity:
            raise ValueError(f"{context} changed while hashed")
        try:
            named = os.stat(
                leaf,
                dir_fd=parent.parent_fd,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise ValueError(f"{context} changed while hashed") from exc
        if (
            not stat.S_ISREG(named.st_mode)
            or _regular_identity(named) != identity
        ):
            raise ValueError(f"{context} changed while hashed")
        parent.verify()
        return digest.hexdigest(), identity
    finally:
        os.close(descriptor)


def _tombstone_prunable_bundle(
    parent: object,
    *,
    entries: Sequence[tuple[str, tuple[int, ...], str]],
) -> None:
    renamed: list[tuple[str, str, tuple[int, ...], str]] = []
    try:
        for leaf, expected_identity, context in entries:
            parent.verify()
            tombstone = (
                f".{leaf}.prune-{os.getpid()}-{secrets.token_hex(16)}"
            )
            try:
                os.stat(
                    tombstone,
                    dir_fd=parent.parent_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            except OSError as exc:
                raise ValueError(
                    f"{context} tombstone cannot be inspected"
                ) from exc
            else:
                raise ValueError(f"{context} tombstone collision")
            try:
                os.rename(
                    leaf,
                    tombstone,
                    src_dir_fd=parent.parent_fd,
                    dst_dir_fd=parent.parent_fd,
                )
            except OSError as exc:
                raise ValueError(
                    f"{context} could not be tombstoned safely"
                ) from exc
            renamed.append(
                (leaf, tombstone, expected_identity, context)
            )
            try:
                moved = os.stat(
                    tombstone,
                    dir_fd=parent.parent_fd,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise ValueError(
                    f"{context} tombstone disappeared after rename"
                ) from exc
            if (
                not stat.S_ISREG(moved.st_mode)
                or _rename_stable_identity(_regular_identity(moved))
                != _rename_stable_identity(expected_identity)
            ):
                raise ValueError(
                    f"{context} identity changed after tombstone rename"
                )
        parent.verify()
    except BaseException as original:
        restoration_error: BaseException | None = None
        for leaf, tombstone, _expected, context in reversed(renamed):
            try:
                try:
                    os.stat(
                        leaf,
                        dir_fd=parent.parent_fd,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    pass
                else:
                    raise ValueError(
                        f"{context} original name reappeared during restore"
                    )
                moved = os.stat(
                    tombstone,
                    dir_fd=parent.parent_fd,
                    follow_symlinks=False,
                )
                moved_identity = _regular_identity(moved)
                os.rename(
                    tombstone,
                    leaf,
                    src_dir_fd=parent.parent_fd,
                    dst_dir_fd=parent.parent_fd,
                )
                restored = os.stat(
                    leaf,
                    dir_fd=parent.parent_fd,
                    follow_symlinks=False,
                )
                if _rename_stable_identity(
                    _regular_identity(restored)
                ) != _rename_stable_identity(moved_identity):
                    raise ValueError(
                        f"{context} restoration identity mismatch"
                    )
            except BaseException as exc:
                restoration_error = exc
        try:
            os.fsync(parent.parent_fd)
        except OSError as exc:
            restoration_error = exc
        parent.verify()
        if restoration_error is not None:
            raise ValueError(
                "checkpoint tombstone restoration failed"
            ) from restoration_error
        raise original

    for _leaf, tombstone, expected_identity, context in renamed:
        parent.verify()
        try:
            current = os.stat(
                tombstone,
                dir_fd=parent.parent_fd,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise ValueError(
                f"{context} tombstone disappeared before pruning"
            ) from exc
        if (
            not stat.S_ISREG(current.st_mode)
            or _rename_stable_identity(_regular_identity(current))
            != _rename_stable_identity(expected_identity)
        ):
            raise ValueError(
                f"{context} tombstone identity changed before pruning"
            )
    for _leaf, tombstone, _expected_identity, context in renamed:
        try:
            os.unlink(tombstone, dir_fd=parent.parent_fd)
        except OSError as exc:
            raise ValueError(
                f"{context} tombstone could not be pruned safely"
            ) from exc
    try:
        os.fsync(parent.parent_fd)
    except OSError as exc:
        raise ValueError(
            "checkpoint pruning directory fsync failed"
        ) from exc
    parent.verify()
    for _leaf, tombstone, _expected_identity, context in renamed:
        try:
            os.stat(
                tombstone,
                dir_fd=parent.parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ValueError(
                f"{context} tombstone removal cannot be verified"
            ) from exc
        raise ValueError(f"{context} tombstone still exists after pruning")


def _prune_checkpoint_bundle(
    checkpoint: Path,
    *,
    expected_sha256: str,
) -> None:
    checkpoint_path = reject_development_path(
        checkpoint,
        "transient checkpoint pruning",
    )
    expected_digest = _require_sha(
        expected_sha256,
        "transient checkpoint hash",
    )
    sidecar = reject_development_path(
        samga_checkpoint_sidecar(checkpoint_path),
        "transient checkpoint sidecar pruning",
    )
    if sidecar.parent != checkpoint_path.parent:
        raise ValueError("checkpoint pruning pair must share one parent")
    with _secure_parent_directory(
        checkpoint_path,
        context="transient checkpoint pruning",
    ) as parent:
        checkpoint_digest, checkpoint_identity = _sha256_prunable_regular(
            parent,
            parent.leaf,
            context="transient checkpoint",
        )
        if checkpoint_digest != expected_digest:
            raise ValueError("transient checkpoint hash mismatch")
        sidecar_raw, sidecar_identity = _read_prunable_regular(
            parent,
            sidecar.name,
            context="transient checkpoint sidecar",
        )
        try:
            sidecar_document = json.loads(sidecar_raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                "transient checkpoint sidecar is invalid JSON"
            ) from exc
        if (
            not isinstance(sidecar_document, dict)
            or canonical_json_bytes(sidecar_document) + b"\n"
            != sidecar_raw
            or sidecar_document.get("schema_version") != 1
            or sidecar_document.get("payload_type")
            != CHECKPOINT_PAYLOAD_TYPE
            or sidecar_document.get("scope") != "train"
            or sidecar_document.get("payload_sha256") != expected_digest
        ):
            raise ValueError(
                "transient checkpoint sidecar binding mismatch"
            )
        _tombstone_prunable_bundle(
            parent,
            entries=(
                (
                    sidecar.name,
                    sidecar_identity,
                    "transient checkpoint sidecar",
                ),
                (
                    parent.leaf,
                    checkpoint_identity,
                    "transient checkpoint",
                ),
            ),
        )


@dataclass
class _CheckpointRetention:
    hashes: dict[str, str] = dataclass_field(default_factory=dict)
    _transient: tuple[Path, str] | None = None

    def record_published(
        self,
        checkpoint: Path,
        digest: str,
        *,
        retain_for_averaging: bool,
    ) -> None:
        path = reject_development_path(
            checkpoint,
            "published checkpoint retention",
        )
        sha256 = _require_sha(digest, "published checkpoint hash")
        if type(retain_for_averaging) is not bool:
            raise TypeError("retain_for_averaging must be boolean")
        if path.name in self.hashes:
            raise ValueError("checkpoint was recorded more than once")
        durable_names = tuple(
            name
            for name in _DURABLE_CHECKPOINT_NAMES
            if name in self.hashes
        )
        if retain_for_averaging:
            if path.name not in _DURABLE_CHECKPOINT_NAMES:
                raise ValueError(
                    "durable checkpoint is outside epochs 51 through 60"
                )
            expected_name = _DURABLE_CHECKPOINT_NAMES[len(durable_names)]
            if path.name != expected_name:
                raise ValueError(
                    "durable checkpoint retention must be a contiguous "
                    "prefix beginning at epoch 51"
                )
            if self._transient is not None:
                transient_path, transient_sha256 = self._transient
                _prune_checkpoint_bundle(
                    transient_path,
                    expected_sha256=transient_sha256,
                )
                del self.hashes[transient_path.name]
                self._transient = None
            self.hashes[path.name] = sha256
            return
        name_match = _CHECKPOINT_NAME_RE.fullmatch(path.name)
        if name_match is None:
            raise ValueError("transient checkpoint name is invalid")
        if durable_names:
            if len(durable_names) == len(_DURABLE_CHECKPOINT_NAMES):
                raise ValueError(
                    "transient checkpoint cannot follow completed durable "
                    "retention"
                )
            expected_epoch = 51 + len(durable_names)
            if int(name_match.group("epoch")) != expected_epoch:
                raise ValueError(
                    "late transient checkpoint must immediately follow the "
                    "durable retention prefix"
                )
        if self._transient is not None:
            transient_path, transient_sha256 = self._transient
            _prune_checkpoint_bundle(
                transient_path,
                expected_sha256=transient_sha256,
            )
            del self.hashes[transient_path.name]
        self.hashes[path.name] = sha256
        self._transient = (path, sha256)

    def validate_final(
        self,
        *,
        completed: bool,
        final_checkpoint: Path,
    ) -> None:
        if type(completed) is not bool:
            raise TypeError("completed must be boolean")
        final_path = reject_development_path(
            final_checkpoint,
            "final checkpoint retention",
        )
        if completed:
            if (
                self._transient is not None
                or tuple(sorted(self.hashes))
                != _DURABLE_CHECKPOINT_NAMES
                or final_path.name != _DURABLE_CHECKPOINT_NAMES[-1]
            ):
                raise ValueError(
                    "completed run must retain exact epochs 51 through 60"
                )
            return
        durable_names = tuple(
            name
            for name in _DURABLE_CHECKPOINT_NAMES
            if name in self.hashes
        )
        if durable_names != _DURABLE_CHECKPOINT_NAMES[: len(durable_names)]:
            raise ValueError(
                "partial durable checkpoint retention must be contiguous"
            )
        expected_hashes = {
            name: self.hashes[name]
            for name in durable_names
        }
        if self._transient is not None:
            transient_path, transient_sha256 = self._transient
            expected_hashes[transient_path.name] = transient_sha256
            expected_final = transient_path
        elif durable_names:
            expected_final = final_path.with_name(durable_names[-1])
        else:
            raise ValueError(
                "partial run must retain a checkpoint"
            )
        if self.hashes != expected_hashes or expected_final != final_path:
            raise ValueError(
                "partial run checkpoint retention is inconsistent"
            )


def save_samga_checkpoint(
    path: Path,
    payload: Mapping[str, object],
    manifest: ManifestIdentity,
) -> str:
    checkpoint_path = reject_development_path(path, "SAMGA checkpoint output")
    sidecar = reject_development_path(
        samga_checkpoint_sidecar(checkpoint_path),
        "SAMGA checkpoint sidecar output",
    )
    if sidecar.parent != checkpoint_path.parent:
        raise ValueError("SAMGA checkpoint and sidecar must share a parent")
    if not isinstance(manifest, ManifestIdentity):
        raise TypeError("manifest must be a verified ManifestIdentity")
    value = _validated_samga_checkpoint_payload(payload)
    if value["subject"] != manifest.subject:
        raise ValueError("SAMGA checkpoint subject differs from manifest")
    serialized = dict(value)
    serialized.update(
        {
            "scope": "train",
            "validation_scope": "val-dev",
            "observed_scopes": ["train", "val-dev"],
        }
    )
    _reject_sealed_checkpoint_metadata(serialized)
    buffer = io.BytesIO()
    torch.save(serialized, buffer)
    raw = buffer.getvalue()
    try:
        checked = torch.load(
            io.BytesIO(raw),
            map_location="cpu",
            weights_only=True,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("SAMGA checkpoint is not weights-only reloadable") from exc
    _validated_samga_checkpoint_payload(checked, transport=True)
    payload_sha256 = hashlib.sha256(raw).hexdigest()
    source_records = [
        {
            "manifest_sha256": manifest.manifest_sha256,
            "records_sha256": manifest.records_sha256,
            "role": "train",
            "role_payload_sha256": manifest.train_role_sha256,
            "source_manifest_sha256": manifest.source_manifest_sha256,
            "source_payload_sha256": manifest.source_payload_sha256,
        },
        {
            "manifest_sha256": manifest.manifest_sha256,
            "records_sha256": manifest.records_sha256,
            "role": "val-dev",
            "role_payload_sha256": manifest.val_dev_role_sha256,
            "source_manifest_sha256": manifest.source_manifest_sha256,
            "source_payload_sha256": manifest.source_payload_sha256,
        },
    ]
    provenance = {
        "config_sha256": value["config_sha256"],
        "manifest_sha256": manifest.manifest_sha256,
        "protocol_sha256": manifest.protocol_sha256,
        "seed": value["seed"],
        "subject": value["subject"],
    }
    metadata = {
        "complete": True,
        "observed_scopes": ["train", "val-dev"],
        "ordered_ids": [
            *manifest.train_ordered_ids,
            *manifest.val_dev_ordered_ids,
        ],
        "train_ordered_ids": list(manifest.train_ordered_ids),
        "val_dev_ordered_ids": list(manifest.val_dev_ordered_ids),
        "retention": value["retention"],
        "source_records": source_records,
    }
    envelope = {
        "schema_version": 1,
        "payload_type": CHECKPOINT_PAYLOAD_TYPE,
        "scope": "train",
        "source_records_sha256": sha256_json(source_records),
        "ordered_ids_sha256": ordered_ids_sha256(
            [
                *manifest.train_ordered_ids,
                *manifest.val_dev_ordered_ids,
            ]
        ),
        "payload_sha256": payload_sha256,
        "provenance": provenance,
        "provenance_sha256": sha256_json(provenance),
        "metadata": metadata,
        "metadata_sha256": sha256_json(metadata),
    }
    validate_epoch_checkpoint_identity(serialized, envelope)
    sidecar_bytes = canonical_json_bytes(envelope) + b"\n"

    with _secure_parent_directory(
        checkpoint_path,
        context="SAMGA checkpoint output",
    ) as parent:
        for leaf in (parent.leaf, sidecar.name):
            try:
                os.stat(
                    leaf,
                    dir_fd=parent.parent_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                continue
            raise FileExistsError("SAMGA checkpoint or sidecar already exists")
        checkpoint_identity: tuple[int, int, int] | None = None
        sidecar_identity: tuple[int, int, int] | None = None
        checkpoint_temporary_identity: tuple[int, int, int] | None = None
        sidecar_temporary_identity: tuple[int, int, int] | None = None
        nonce = f"{os.getpid()}-{secrets.token_hex(8)}"
        checkpoint_temporary_leaf = f".{parent.leaf}.tmp-{nonce}"
        sidecar_temporary_leaf = f".{sidecar.name}.tmp-{nonce}"
        try:
            checkpoint_temporary_identity = _write_relative_exclusive(
                parent,
                checkpoint_temporary_leaf,
                raw,
                context="SAMGA checkpoint temporary output",
            )
            if (
                _read_relative_regular(
                    parent.parent_fd,
                    checkpoint_temporary_leaf,
                    context="SAMGA checkpoint temporary output",
                )
                != raw
            ):
                raise ValueError("SAMGA checkpoint temporary verification mismatch")
            os.link(
                checkpoint_temporary_leaf,
                parent.leaf,
                src_dir_fd=parent.parent_fd,
                dst_dir_fd=parent.parent_fd,
                follow_symlinks=False,
            )
            published_checkpoint = os.stat(
                parent.leaf,
                dir_fd=parent.parent_fd,
                follow_symlinks=False,
            )
            checkpoint_identity = _node_identity(published_checkpoint)
            if checkpoint_identity != checkpoint_temporary_identity:
                raise ValueError("SAMGA checkpoint publication changed")
            if (
                _read_relative_regular(
                    parent.parent_fd,
                    parent.leaf,
                    context="SAMGA checkpoint output",
                )
                != raw
            ):
                raise ValueError("SAMGA checkpoint verification mismatch")
            sidecar_temporary_identity = _write_relative_exclusive(
                parent,
                sidecar_temporary_leaf,
                sidecar_bytes,
                context="SAMGA checkpoint sidecar temporary output",
            )
            os.link(
                sidecar_temporary_leaf,
                sidecar.name,
                src_dir_fd=parent.parent_fd,
                dst_dir_fd=parent.parent_fd,
                follow_symlinks=False,
            )
            published = os.stat(
                sidecar.name,
                dir_fd=parent.parent_fd,
                follow_symlinks=False,
            )
            sidecar_identity = _node_identity(published)
            if sidecar_identity != sidecar_temporary_identity:
                raise ValueError("SAMGA checkpoint sidecar publication changed")
            if (
                _read_relative_regular(
                    parent.parent_fd,
                    sidecar.name,
                    context="SAMGA checkpoint sidecar output",
                )
                != sidecar_bytes
            ):
                raise ValueError("SAMGA checkpoint sidecar verification mismatch")
            os.fsync(parent.parent_fd)
        except BaseException:
            if sidecar_identity is not None:
                _unlink_created_file(
                    parent,
                    sidecar.name,
                    sidecar_identity,
                )
            if checkpoint_identity is not None:
                _unlink_created_file(
                    parent,
                    parent.leaf,
                    checkpoint_identity,
                )
            raise
        finally:
            if sidecar_temporary_identity is not None:
                _unlink_created_file(
                    parent,
                    sidecar_temporary_leaf,
                    sidecar_temporary_identity,
                )
            if checkpoint_temporary_identity is not None:
                _unlink_created_file(
                    parent,
                    checkpoint_temporary_leaf,
                    checkpoint_temporary_identity,
                )
    return payload_sha256


def load_samga_checkpoint(
    path: Path,
    *,
    requested_scope: str,
) -> LoadedSAMGACheckpoint:
    if requested_scope != "train":
        raise PermissionError("SAMGA checkpoint payload access requires train scope")
    checkpoint_path = reject_development_path(path, "SAMGA checkpoint")
    loaded = load_typed_torch_checkpoint(
        checkpoint_path,
        payload_type=CHECKPOINT_PAYLOAD_TYPE,
        requested_scope="train",
    )
    value = _validated_samga_checkpoint_payload(loaded.payload, transport=True)
    validate_epoch_checkpoint_identity(value, loaded.envelope)
    if (
        value.get("scope") != "train"
        or value.get("validation_scope") != "val-dev"
        or value.get("observed_scopes") != ["train", "val-dev"]
    ):
        raise PermissionError("SAMGA checkpoint is not development-only")
    resume_value = {
        key: child for key, child in value.items() if key not in _CHECKPOINT_SCOPE_KEYS
    }
    return LoadedSAMGACheckpoint(
        payload=MappingProxyType(resume_value),
        sha256=loaded.sha256,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", required=True, choices=("train",))
    parser.add_argument(
        "--validation-scope",
        required=True,
        choices=("val-dev",),
    )
    parser.add_argument("--stage", required=True, type=int, choices=(0, 2))
    parser.add_argument("--subject", required=True, type=int, choices=range(1, 11))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--resume", required=True)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--feature-cache", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stage2-config", type=Path)
    parser.add_argument("--candidate-id")
    parser.add_argument("--layernorm-config-id", default="s2-layernorm-off")
    parser.add_argument("--whitening-config-id", default="s2-whitening-off")
    parser.add_argument("--preprojector-config-id", default="s2-preproj-shared")
    parser.add_argument(
        "--adapter-kind",
        choices=("identity", "adapter", "global_dense", "matched_projector"),
        default="identity",
    )
    parser.add_argument("--adapter-rank", type=int)
    parser.add_argument("--adapter-lr-ratio", type=float)
    parser.add_argument("--whitening-artifact", type=Path)
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("cuda",),
        default="cuda",
    )
    return parser


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.seed < 0:
        parser.error("--seed must be non-negative")
    if not arguments.resume:
        parser.error("--resume must be explicit: none or checkpoint.pt")
    if arguments.max_train_steps is not None and arguments.max_train_steps <= 0:
        parser.error("--max-train-steps must be positive")
    if arguments.num_workers != 0:
        parser.error("--num-workers is locked to zero")
    return arguments


def _normalized_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def require_development_path(path: Path, context: str) -> Path:
    """Reject sealed/test names and every existing symlink component."""

    absolute = Path(os.path.abspath(os.path.normpath(os.fspath(path))))
    for component in absolute.parts:
        token = _normalized_token(component)
        if token in _SENSITIVE or re.fullmatch(
            r"sub_?\d+_test(?:_.*)?",
            token,
        ):
            raise ValueError(f"{context} contains a sealed test/formal/confirm name")
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            break
        if stat.S_ISLNK(mode):
            raise ValueError(f"{context} contains a symlink component")
    return absolute


def _require_sha(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be SHA-256")
    return value


def mmd_weight_for_epoch(epoch: int) -> float:
    if type(epoch) is not int or epoch <= 0:
        raise ValueError("epoch must be positive")
    if epoch > 20:
        return 0.0
    return 0.9 + (0.5 - 0.9) * ((epoch - 1) / 19)


def learning_rate_for_epoch(epoch: int) -> float:
    if type(epoch) is not int or not 1 <= epoch <= 60:
        raise ValueError("epoch must be in 1..60")
    return float(
        SCHEDULE["stage1_learning_rate"]
        if epoch <= 20
        else SCHEDULE["stage2_learning_rate"]
    )


def _numpy_rng_payload() -> dict[str, object]:
    name, keys, position, has_gauss, cached = np.random.get_state()
    return {
        "bit_generator": name,
        "keys": torch.from_numpy(keys.copy()),
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached),
    }


def _restore_numpy_rng(value: Mapping[str, object]) -> None:
    keys = value.get("keys")
    if not isinstance(keys, torch.Tensor):
        raise ValueError("checkpoint NumPy RNG keys are invalid")
    np.random.set_state(
        (
            str(value["bit_generator"]),
            keys.detach().cpu().numpy().astype(np.uint32, copy=True),
            int(value["position"]),
            int(value["has_gauss"]),
            float(value["cached_gaussian"]),
        )
    )


def build_epoch_checkpoint(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    epoch: int,
    global_step: int,
    subject: int,
    seed: int,
    config_sha256: str,
    schedule_sha256: str,
    trajectory_sha256: str,
    data_order_sha256: str,
    generator: torch.Generator,
    validation_metrics: Mapping[str, object],
    input_hashes: Mapping[str, str],
    environment: Mapping[str, object],
    effective_batch: int,
    sampler_state: Mapping[str, object] | None = None,
    run_manifest: Mapping[str, object] | None = None,
    candidate_spec: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if not 1 <= epoch <= 60 or global_step < 0:
        raise ValueError("checkpoint epoch/step is invalid")
    for label, digest in (
        ("config", config_sha256),
        ("schedule", schedule_sha256),
        ("trajectory", trajectory_sha256),
        ("data order", data_order_sha256),
    ):
        _require_sha(digest, label)
    if not hasattr(scheduler, "state_dict"):
        raise TypeError("scheduler must expose state_dict")
    model_state = {
        key: value.detach().cpu().contiguous()
        for key, value in model.state_dict().items()
    }
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return {
        "schema_version": 1,
        "payload_type": CHECKPOINT_PAYLOAD_TYPE,
        "epoch": epoch,
        "global_step": int(global_step),
        "subject": int(subject),
        "seed": int(seed),
        "config_sha256": config_sha256,
        "schedule_sha256": schedule_sha256,
        "optimizer_stage": "stage1" if epoch <= 20 else "stage2",
        "trajectory_sha256": trajectory_sha256,
        "data_order_sha256": data_order_sha256,
        "model_state_dict": model_state,
        "model_state_sha256": hash_state_dict(model_state),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": _numpy_rng_payload(),
        "torch_rng_state": torch.get_rng_state().clone(),
        "cuda_rng_states": [state.detach().cpu() for state in cuda_state],
        "loader_generator_state": generator.get_state().clone(),
        "sampler_state_dict": dict(sampler_state or {}),
        "validation_metrics": dict(validation_metrics),
        "input_hashes": dict(input_hashes),
        "environment": validate_environment_binding(environment),
        "effective_batch": int(effective_batch),
        "run_manifest": dict(run_manifest or {}),
        "candidate_spec": dict(candidate_spec or {}),
    }


def restore_training_checkpoint(
    payload: Mapping[str, object],
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    generator: torch.Generator,
    expected_subject: int,
    expected_seed: int,
    expected_config_sha256: str,
    expected_schedule_sha256: str,
    expected_trajectory_sha256: str,
    expected_data_order_sha256: str,
) -> tuple[int, int]:
    identities = {
        "subject": expected_subject,
        "seed": expected_seed,
        "config_sha256": expected_config_sha256,
        "schedule_sha256": expected_schedule_sha256,
        "trajectory_sha256": expected_trajectory_sha256,
        "data_order_sha256": expected_data_order_sha256,
    }
    for field, expected in identities.items():
        if payload.get(field) != expected:
            raise ValueError(f"checkpoint {field} mismatch")
    state = payload.get("model_state_dict")
    optimizer_state = payload.get("optimizer_state_dict")
    scheduler_state = payload.get("scheduler_state_dict")
    if (
        not isinstance(state, Mapping)
        or not isinstance(optimizer_state, Mapping)
        or not isinstance(scheduler_state, Mapping)
    ):
        raise ValueError("checkpoint task/optimizer/scheduler state is invalid")
    if not hasattr(scheduler, "load_state_dict"):
        raise TypeError("scheduler must expose load_state_dict")
    model.load_state_dict(state, strict=True)
    optimizer.load_state_dict(optimizer_state)
    scheduler.load_state_dict(scheduler_state)
    python_state = payload.get("python_rng_state")
    numpy_state = payload.get("numpy_rng_state")
    torch_state = payload.get("torch_rng_state")
    loader_state = payload.get("loader_generator_state")
    if (
        not isinstance(python_state, tuple)
        or not isinstance(numpy_state, Mapping)
        or not isinstance(torch_state, torch.Tensor)
        or not isinstance(loader_state, torch.Tensor)
    ):
        raise ValueError("checkpoint RNG state is invalid")
    random.setstate(python_state)
    _restore_numpy_rng(numpy_state)
    torch.set_rng_state(torch_state.detach().cpu())
    cuda_states = payload.get("cuda_rng_states")
    if torch.cuda.is_available() and isinstance(cuda_states, list) and cuda_states:
        torch.cuda.set_rng_state_all(cuda_states)
    generator.set_state(loader_state.detach().cpu())
    epoch = payload.get("epoch")
    step = payload.get("global_step")
    if type(epoch) is not int or type(step) is not int:
        raise ValueError("checkpoint epoch/step is invalid")
    return epoch, step


@dataclass(frozen=True)
class TrainingPaths:
    config: Path
    manifest: Path
    feature_cache: Path
    output_dir: Path
    stage2_config: Path | None
    whitening_artifact: Path | None
    resume_checkpoint: Path | None


@dataclass(frozen=True)
class TrainingConfig:
    semantic: SemanticConfig
    payload: Mapping[str, object]
    protocol: ProtocolConfig
    upstream_root: Path
    upstream_commit: str
    cache_sha256: str
    model_sha256: str
    batch_size: int


@dataclass(frozen=True)
class UpstreamPreflight:
    semantic_sha256: str
    semantic_payload: Mapping[str, object]
    upstream_root: Path
    upstream_commit: str
    components: UpstreamComponents


@dataclass(frozen=True)
class CandidateSelection:
    config_id: str
    stage2_config_sha256: str | None
    layernorm_config_id: str
    whitening_config_id: str
    preprojector_config_id: str
    adapter_kind: str
    adapter_rank: int | None
    adapter_lr_ratio: float | None
    whitening: TrainWhitening | None


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PROTOCOL_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "protocol_v1.json"
_STAGE2_CONFIG_PATH = (
    Path(__file__).resolve().parent / "configs" / "stage2_candidates_v1.json"
)
_STAGE2_CONFIG_SHA256 = (
    "f7a34ae1ed66c0ac574669ad393645052c8c90f8c8bfc78a747094429415f263"
)
_NO_INITIAL_CHECKPOINT_SHA256 = sha256_json(
    {
        "schema_version": 1,
        "payload_type": "samga_brain_rw.no_initial_checkpoint",
    }
)


def _mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{context} must be a string-keyed mapping")
    return dict(value)


def _declared_path(value: object, context: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{context} must be a non-empty text path")
    path = Path(value)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return reject_development_path(path, context)


def _guard_training_paths(arguments: argparse.Namespace) -> TrainingPaths:
    stage2_config = (
        None
        if arguments.stage2_config is None
        else reject_development_path(
            arguments.stage2_config,
            "Stage 2 semantic config",
        )
    )
    whitening_artifact = (
        None
        if arguments.whitening_artifact is None
        else reject_development_path(
            arguments.whitening_artifact,
            "train-only whitening artifact",
        )
    )
    resume_checkpoint = (
        None
        if arguments.resume == "none"
        else reject_development_path(
            Path(arguments.resume),
            "SAMGA resume checkpoint",
        )
    )
    paths = TrainingPaths(
        config=reject_development_path(
            arguments.config,
            "SAMGA baseline config",
        ),
        manifest=reject_development_path(
            arguments.manifest,
            "SAMGA protocol manifest",
        ),
        feature_cache=reject_development_path(
            arguments.feature_cache,
            "SAMGA feature cache",
        ),
        output_dir=reject_development_path(
            arguments.output_dir,
            "SAMGA training output",
        ),
        stage2_config=stage2_config,
        whitening_artifact=whitening_artifact,
        resume_checkpoint=resume_checkpoint,
    )
    if paths.output_dir.exists():
        raise FileExistsError(
            f"SAMGA training output already exists: {paths.output_dir}"
        )
    return paths


def preflight_upstream_config(path: Path) -> UpstreamPreflight:
    """Verify pinned config/upstream code before manifest or data access."""

    semantic = SemanticConfig.from_path(path)
    payload = semantic.canonical_payload()
    if payload.get("config_type") != "internvit_baseline":
        raise ValueError("SAMGA requires an internvit_baseline config")
    upstream = _mapping(payload.get("upstream"), "config upstream")
    upstream_root = _declared_path(
        upstream.get("path"),
        "configured upstream SAMGA root",
    )
    upstream_commit = upstream.get("git_commit")
    if upstream_commit != PINNED_UPSTREAM_SHA:
        raise ValueError("configured upstream SAMGA revision mismatch")
    components = load_locked_upstream_components(
        upstream_root,
        upstream_commit,
    )
    return UpstreamPreflight(
        semantic_sha256=semantic.sha256,
        semantic_payload=MappingProxyType(payload),
        upstream_root=upstream_root,
        upstream_commit=upstream_commit,
        components=components,
    )


def _load_training_config(
    path: Path,
    feature_cache: Path,
    *,
    manifest: ManifestIdentity,
    preflight: UpstreamPreflight,
) -> TrainingConfig:
    semantic = SemanticConfig.from_path(path)
    payload = semantic.canonical_payload()
    if semantic.sha256 != preflight.semantic_sha256 or payload != dict(
        preflight.semantic_payload
    ):
        raise ValueError("full config identity differs from upstream preflight")
    if payload.get("config_type") != "internvit_baseline":
        raise ValueError("SAMGA training requires an internvit_baseline config")
    upstream = _mapping(payload.get("upstream"), "config upstream")
    model = _mapping(payload.get("model"), "config model")
    cache = _mapping(payload.get("cache"), "config cache")
    task = _mapping(payload.get("task"), "config task")
    upstream_root = _declared_path(
        upstream.get("path"),
        "configured upstream SAMGA root",
    )
    upstream_commit = upstream.get("git_commit")
    if upstream_commit != PINNED_UPSTREAM_SHA:
        raise ValueError("configured upstream SAMGA revision mismatch")
    if (
        upstream_root != preflight.upstream_root
        or upstream_commit != preflight.upstream_commit
    ):
        raise ValueError("full upstream identity differs from upstream preflight")
    _declared_path(model.get("path"), "configured InternViT model")
    configured_cache = _declared_path(
        cache.get("path"),
        "configured feature cache",
    )
    if configured_cache != feature_cache:
        raise ValueError("CLI feature cache path differs from the semantic config")
    cache_sha256 = _require_sha(
        cache.get("sha256"),
        "config cache SHA-256",
    )
    if tuple(task.get("channels", ())) != POSTERIOR_CHANNELS:
        raise ValueError(
            "config channels differ from the locked posterior-channel order"
        )
    if task.get("force_global") is not True:
        raise ValueError("config must lock force_global=true")
    batch_size = task.get("batch_size")
    if type(batch_size) is not int or batch_size != 512:
        raise ValueError("config batch_size must be locked to 512")
    protocol = ProtocolConfig.from_path(_PROTOCOL_CONFIG_PATH)
    if protocol.sha256 != manifest.protocol_sha256:
        raise ValueError("protocol config SHA-256 differs from the verified manifest")
    return TrainingConfig(
        semantic=semantic,
        payload=payload,
        protocol=protocol,
        upstream_root=upstream_root,
        upstream_commit=upstream_commit,
        cache_sha256=cache_sha256,
        model_sha256=sha256_json(model),
        batch_size=batch_size,
    )


def _reject_duplicate_pairs(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> object:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def _load_whitening(path: Path) -> TrainWhitening:
    with _secure_parent_directory(
        path,
        context="train-only whitening artifact",
    ) as parent:
        raw = _read_relative_regular(
            parent.parent_fd,
            parent.leaf,
            context="train-only whitening artifact",
        )
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("whitening artifact is not strict JSON") from exc
    return TrainWhitening.from_payload(payload)


def _resolve_candidate_selection(
    arguments: argparse.Namespace,
    paths: TrainingPaths,
    config: TrainingConfig,
) -> CandidateSelection:
    if arguments.stage == 0:
        defaults = (
            paths.stage2_config is None
            and arguments.candidate_id is None
            and arguments.layernorm_config_id == "s2-layernorm-off"
            and arguments.whitening_config_id == "s2-whitening-off"
            and arguments.preprojector_config_id == "s2-preproj-shared"
            and arguments.adapter_kind == "identity"
            and arguments.adapter_rank is None
            and arguments.adapter_lr_ratio is None
            and paths.whitening_artifact is None
        )
        if not defaults:
            raise ValueError("Stage 0 forbids every Stage 2 candidate option")
        config_id = config.payload.get("config_id")
        if not isinstance(config_id, str) or not config_id:
            raise ValueError("baseline config_id is invalid")
        return CandidateSelection(
            config_id=config_id,
            stage2_config_sha256=None,
            layernorm_config_id="s2-layernorm-off",
            whitening_config_id="s2-whitening-off",
            preprojector_config_id="s2-preproj-shared",
            adapter_kind="identity",
            adapter_rank=None,
            adapter_lr_ratio=None,
            whitening=None,
        )

    if paths.stage2_config is None or not arguments.candidate_id:
        raise ValueError("Stage 2 requires --stage2-config and --candidate-id")
    if paths.stage2_config.resolve(strict=True) != _STAGE2_CONFIG_PATH.resolve(
        strict=True
    ):
        raise ValueError("Stage 2 config must be the canonical registry path")
    stage2_semantic = SemanticConfig.from_path(paths.stage2_config)
    stage2_payload = stage2_semantic.canonical_payload()
    if (
        stage2_semantic.sha256 != _STAGE2_CONFIG_SHA256
        or stage2_payload.get("config_type") != "stage2_candidates"
        or stage2_payload.get("config_id") != "stage2_candidates_v1"
        or stage2_payload.get("combination_policy")
        != "one_factor_only_no_post_hoc_combinations"
    ):
        raise ValueError("Stage 2 config differs from the canonical sealed registry")
    candidate_id = arguments.candidate_id
    if candidate_id in {
        "s2-raw-epoch60-control",
        "s2-avg-last5",
        "s2-avg-last10",
        "s2-swa-last5",
        "s2-swa-last10",
    }:
        raise ValueError(
            "checkpoint raw/averaging/SWA is a post-hoc Stage 2 artifact, "
            "not a training cell"
        )
    identities = stage2_registry_identities(stage2_payload)
    allowed = identities.get(candidate_id)
    if allowed is None:
        raise ValueError("candidate_id is not a trainable Stage 2 registry entry")
    factor = select_stage2_factor_identity(
        allowed,
        (
            arguments.layernorm_config_id,
            arguments.whitening_config_id,
            arguments.preprojector_config_id,
            arguments.adapter_kind,
            arguments.adapter_rank,
            arguments.adapter_lr_ratio,
        ),
    )
    (
        layernorm_id,
        whitening_id,
        preprojector_id,
        adapter_kind,
        adapter_rank,
        adapter_lr_ratio,
    ) = factor
    if whitening_id == "s2-whitening-on":
        if paths.whitening_artifact is None:
            raise ValueError("whitening-on requires --whitening-artifact")
        whitening = _load_whitening(paths.whitening_artifact)
        if whitening.cache_provenance_sha256 != config.cache_sha256:
            raise ValueError(
                "whitening cache provenance differs from the baseline cache"
            )
    else:
        if paths.whitening_artifact is not None:
            raise ValueError("whitening artifact is only valid for whitening-on")
        whitening = None
    return CandidateSelection(
        config_id=candidate_id,
        stage2_config_sha256=stage2_semantic.sha256,
        layernorm_config_id=layernorm_id,
        whitening_config_id=whitening_id,
        preprojector_config_id=preprojector_id,
        adapter_kind=adapter_kind,
        adapter_rank=adapter_rank,
        adapter_lr_ratio=adapter_lr_ratio,
        whitening=whitening,
    )


def build_resolved_candidate_payload(
    *,
    stage: int,
    config_id: str,
    subject: int,
    seed: int,
    baseline_config_sha256: str,
    stage2_config_sha256: str | None,
    layernorm_config_id: str,
    whitening_config_id: str,
    preprojector_config_id: str,
    adapter_kind: str,
    adapter_rank: int | None,
    adapter_lr_ratio: float | None,
    whitening_payload_sha256: str | None,
    environment_binding: Mapping[str, object],
) -> dict[str, object]:
    if stage not in {0, 2} or not 1 <= subject <= 10 or seed < 0:
        raise ValueError("resolved candidate stage/subject/seed is invalid")
    if not isinstance(config_id, str) or not config_id:
        raise ValueError("resolved candidate config_id must be non-empty")
    _require_sha(baseline_config_sha256, "baseline config")
    if stage == 0:
        if stage2_config_sha256 is not None:
            raise ValueError("Stage 0 cannot bind a Stage 2 config")
    else:
        _require_sha(stage2_config_sha256, "Stage 2 config")
    if whitening_config_id == "s2-whitening-on":
        _require_sha(
            whitening_payload_sha256,
            "whitening payload",
        )
    elif whitening_payload_sha256 is not None:
        raise ValueError("whitening-off cannot bind whitening payload provenance")
    normalized_environment = validate_environment_binding(environment_binding)
    return {
        "schema_version": 1,
        "stage": f"stage{stage}",
        "config_id": config_id,
        "subject": subject,
        "seed": seed,
        "semantics": {
            "baseline_config_sha256": baseline_config_sha256,
            "stage2_config_sha256": stage2_config_sha256,
            "layernorm_config_id": layernorm_config_id,
            "whitening_config_id": whitening_config_id,
            "preprojector_config_id": preprojector_config_id,
            "adapter_kind": adapter_kind,
            "adapter_rank": adapter_rank,
            "adapter_lr_ratio": adapter_lr_ratio,
            "whitening_payload_sha256": whitening_payload_sha256,
        },
        "runtime": {
            "schedule_sha256": SCHEDULE_SHA256,
            "batch_size": 512,
            "epochs": 60,
            "force_global_validation": True,
            "num_workers": 0,
            "environment": normalized_environment,
        },
    }


def build_candidate_spec(
    *,
    stage: int,
    config_id: str,
    subject: int,
    seed: int,
    baseline_config_sha256: str,
    stage2_config_sha256: str | None,
    semantic_config_sha256: str,
    input_bundle_sha256: str,
    run_key: str,
    layernorm_config_id: str,
    whitening_config_id: str,
    preprojector_config_id: str,
    adapter_kind: str,
    adapter_rank: int | None,
    adapter_lr_ratio: float | None,
    whitening: TrainWhitening | None,
    identities: TrainingIdentities,
) -> dict[str, object]:
    if not isinstance(identities, TrainingIdentities):
        raise TypeError("identities must be derived TrainingIdentities")
    if stage not in {0, 2} or not 1 <= subject <= 10 or seed < 0:
        raise ValueError("candidate stage/subject/seed is invalid")
    if not isinstance(config_id, str) or not config_id:
        raise ValueError("candidate config_id must be non-empty")
    expected_prefix = f"stage{stage}__{config_id}__sub-{subject:02d}__seed-{seed}"
    if not isinstance(run_key, str) or not run_key.startswith(expected_prefix):
        raise ValueError("candidate run_key identity mismatch")
    for name, digest in (
        ("baseline config", baseline_config_sha256),
        ("semantic config", semantic_config_sha256),
        ("input bundle", input_bundle_sha256),
        ("data order", identities.data_order_sha256),
        ("trajectory", identities.trajectory_sha256),
        ("full initialization", identities.full_task_initialization_sha256),
        (
            "shared parameter intersection",
            identities.shared_parameter_intersection_sha256,
        ),
        (
            "architecture-specific initialization",
            identities.architecture_specific_initialization_sha256,
        ),
    ):
        _require_sha(digest, name)
    if (
        not isinstance(identities.shared_parameter_intersection_name, str)
        or not identities.shared_parameter_intersection_name
    ):
        raise ValueError("shared parameter intersection name must be non-empty")
    if stage == 0:
        if stage2_config_sha256 is not None:
            raise ValueError("Stage 0 cannot bind a Stage 2 config")
    else:
        _require_sha(stage2_config_sha256, "Stage 2 config")
    allowed_ids = (
        layernorm_config_id in {"s2-layernorm-off", "s2-layernorm-on"}
        and whitening_config_id in {"s2-whitening-off", "s2-whitening-on"}
        and preprojector_config_id in {"s2-preproj-shared", "s2-preproj-separate"}
        and adapter_kind in {"identity", "adapter", "global_dense", "matched_projector"}
    )
    if not allowed_ids:
        raise ValueError("candidate contains an unknown Stage 2 factor")
    active_factors = sum(
        (
            layernorm_config_id == "s2-layernorm-on",
            whitening_config_id == "s2-whitening-on",
            preprojector_config_id == "s2-preproj-separate",
            adapter_kind != "identity",
        )
    )
    if stage == 0 and active_factors:
        raise ValueError("Stage 0 cannot enable a Stage 2 factor")
    if active_factors > 1:
        raise ValueError("candidate may enable at most one Stage 2 factor")
    if adapter_kind == "identity":
        if adapter_rank is not None or adapter_lr_ratio is not None:
            raise ValueError("identity adapter cannot set rank/LR ratio")
    elif (
        adapter_rank not in {8, 16, 32}
        or type(adapter_lr_ratio) is not float
        or adapter_lr_ratio not in {0.05, 0.1}
    ):
        raise ValueError("adapter rank/LR ratio is outside the locked grid")
    if whitening_config_id == "s2-whitening-on":
        if not isinstance(whitening, TrainWhitening):
            raise ValueError("whitening-on requires TrainWhitening")
        whitening_payload: object = whitening.to_payload()
    else:
        if whitening is not None:
            raise ValueError("whitening-off forbids whitening statistics")
        whitening_payload = None
    body: dict[str, object] = {
        "schema_version": 1,
        "config_id": config_id,
        "stage": f"stage{stage}",
        "subject": subject,
        "seed": seed,
        "baseline_config_sha256": baseline_config_sha256,
        "stage2_config_sha256": stage2_config_sha256,
        "semantic_config_sha256": semantic_config_sha256,
        "input_bundle_sha256": input_bundle_sha256,
        "run_key": run_key,
        "layernorm_config_id": layernorm_config_id,
        "whitening_config_id": whitening_config_id,
        "preprojector_config_id": preprojector_config_id,
        "adapter_kind": adapter_kind,
        "adapter_rank": adapter_rank,
        "adapter_lr_ratio": adapter_lr_ratio,
        "whitening_payload": whitening_payload,
        "full_task_initialization_sha256": (identities.full_task_initialization_sha256),
        "shared_parameter_intersection_name": (
            identities.shared_parameter_intersection_name
        ),
        "shared_parameter_intersection_sha256": (
            identities.shared_parameter_intersection_sha256
        ),
        "architecture_specific_initialization_sha256": (
            identities.architecture_specific_initialization_sha256
        ),
        "data_order_sha256": identities.data_order_sha256,
        "trajectory_sha256": identities.trajectory_sha256,
    }
    return {**body, "candidate_spec_sha256": sha256_json(body)}


def build_run_manifest(
    *,
    stage: int,
    subject: int,
    seed: int,
    config_id: str,
    config_sha256: str,
    protocol_sha256: str,
    cache_sha256: str,
    git_sha: str,
    upstream_sha: str,
    data_order_sha256: str,
    candidate_spec_sha256: str,
    run_key: str,
    schema_version: int = 1,
    payload_type: str = RUN_PAYLOAD_TYPE,
) -> dict[str, object]:
    body = {
        "schema_version": schema_version,
        "payload_type": payload_type,
        "stage": stage,
        "subject": subject,
        "seed": seed,
        "config_id": config_id,
        "config_sha256": _require_sha(config_sha256, "config"),
        "protocol_sha256": _require_sha(protocol_sha256, "protocol"),
        "cache_sha256": _require_sha(cache_sha256, "cache"),
        "git_sha": git_sha,
        "upstream_sha": upstream_sha,
        "data_order_sha256": _require_sha(data_order_sha256, "data order"),
        "candidate_spec_sha256": _require_sha(
            candidate_spec_sha256,
            "candidate spec",
        ),
        "run_key": run_key,
    }
    if schema_version != 1 or payload_type != RUN_PAYLOAD_TYPE:
        raise ValueError("run manifest identity mismatch")
    if stage not in {0, 2} or not 1 <= subject <= 10 or seed < 0:
        raise ValueError("run manifest stage/subject/seed is invalid")
    if not config_id or _GIT_RE.fullmatch(git_sha) is None:
        raise ValueError("run manifest config/git identity is invalid")
    if upstream_sha != PINNED_UPSTREAM_SHA:
        raise ValueError("run manifest upstream revision mismatch")
    expected_prefix = f"stage{stage}__{config_id}__sub-{subject:02d}__seed-{seed}"
    if not isinstance(run_key, str) or not run_key.startswith(expected_prefix):
        raise ValueError("run manifest run_key identity mismatch")
    return {**body, "run_manifest_sha256": sha256_json(body)}


def _build_input_hashes(
    manifest: ManifestIdentity,
    config: TrainingConfig,
) -> dict[str, str]:
    return {
        "cache_sha256": config.cache_sha256,
        "checkpoint_sha256": _NO_INITIAL_CHECKPOINT_SHA256,
        "manifest_sha256": manifest.manifest_sha256,
        "model_sha256": config.model_sha256,
        "ordered_ids_sha256": ordered_ids_sha256(
            [*manifest.train_ordered_ids, *manifest.val_dev_ordered_ids]
        ),
        "protocol_sha256": manifest.protocol_sha256,
        "records_sha256": manifest.records_sha256,
        "source_manifest_sha256": manifest.source_manifest_sha256,
        "source_payload_sha256": manifest.source_payload_sha256,
        "source_payload_path_sha256": sha256_json(str(manifest.source_payload_path)),
        "source_payload_byte_count_sha256": sha256_json(
            manifest.source_payload_byte_count
        ),
        "train_ordered_ids_sha256": manifest.train_ordered_ids_sha256,
        "train_role_sha256": manifest.train_role_sha256,
        "val_dev_ordered_ids_sha256": manifest.val_dev_ordered_ids_sha256,
        "val_dev_role_sha256": manifest.val_dev_role_sha256,
    }


def _cached_dataset_factory(
    paths: TrainingPaths,
    manifest: ManifestIdentity,
    *,
    seed: int,
    cache_sha256: str,
    train_dataset: ProtocolSubjectDataset,
) -> tuple[
    Callable[..., ProtocolSubjectDataset],
    dict[str, ProtocolSubjectDataset],
]:
    _require_sha(cache_sha256, "feature cache")
    datasets = {"train": train_dataset}
    required_arguments = {
        "manifest_path",
        "scope",
        "seed",
        "selected_channels",
        "feature_cache",
        "smooth_probability",
    }

    def build(**kwargs: object) -> ProtocolSubjectDataset:
        if set(kwargs) != required_arguments:
            raise ValueError("dataset factory arguments differ from the locked runtime")
        scope = kwargs["scope"]
        if scope not in {"train", "val-dev"}:
            raise PermissionError("dataset factory scope must be train or val-dev")
        expected = {
            "manifest_path": manifest.path,
            "scope": scope,
            "seed": seed,
            "selected_channels": POSTERIOR_CHANNELS,
            "feature_cache": paths.feature_cache,
            "smooth_probability": (0.3 if scope == "train" else 0.0),
        }
        if kwargs != expected or type(kwargs["smooth_probability"]) is not float:
            raise ValueError("dataset factory arguments differ from the locked runtime")
        dataset = datasets.get(scope)
        if dataset is None:
            dataset = _development_dataset(
                paths,
                manifest,
                scope=scope,
                seed=seed,
            )
            _verify_development_dataset(
                dataset,
                manifest=manifest,
                scope=scope,
                cache_sha256=cache_sha256,
            )
            datasets[scope] = dataset
        return dataset

    return build, datasets


def _development_dataset(
    paths: TrainingPaths,
    manifest: ManifestIdentity,
    *,
    scope: str,
    seed: int,
) -> ProtocolSubjectDataset:
    return ProtocolSubjectDataset(
        manifest_path=manifest.path,
        scope=scope,
        seed=seed,
        selected_channels=POSTERIOR_CHANNELS,
        feature_cache=paths.feature_cache,
        smooth_probability=0.3 if scope == "train" else 0.0,
        expected_source_payload_sha256=manifest.source_payload_sha256,
    )


def _verify_development_dataset(
    dataset: ProtocolSubjectDataset,
    *,
    manifest: ManifestIdentity,
    scope: str,
    cache_sha256: str,
) -> None:
    if dataset.scope != scope or dataset.subject_id != manifest.subject:
        raise ValueError("development dataset scope/subject mismatch")
    if dataset.manifest_path != manifest.path:
        raise ValueError("development dataset manifest path mismatch")
    expected_ids = (
        manifest.train_ordered_ids if scope == "train" else manifest.val_dev_ordered_ids
    )
    if tuple(dataset.ordered_ids) != expected_ids:
        raise ValueError("development dataset ordered IDs mismatch")
    if scope == "val-dev" and (
        tuple(dataset.query_ids) != expected_ids
        or tuple(dataset.gallery_ids) != expected_ids
    ):
        raise ValueError("val-dev query/gallery IDs mismatch")
    metadata = _mapping(
        dataset.feature_cache_metadata,
        "feature cache metadata",
    )
    actual_cache_sha256 = metadata.get(
        "feature_sha256",
        metadata.get("cache_sha256"),
    )
    if actual_cache_sha256 != cache_sha256:
        raise ValueError("verified feature cache SHA-256 mismatch")


def _git_text(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(_PROJECT_ROOT), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "git command failed"
        raise ValueError(f"git provenance failed: {detail}")
    return completed.stdout.strip()


def clean_repository_git_sha() -> str:
    top_level = Path(_git_text("rev-parse", "--show-toplevel"))
    if top_level.resolve(strict=True) != _PROJECT_ROOT.resolve(strict=True):
        raise ValueError("git provenance resolved a different repository")
    git_sha = _git_text("rev-parse", "HEAD")
    if _GIT_RE.fullmatch(git_sha) is None:
        raise ValueError("git provenance HEAD is not lowercase 40-hex")
    if _git_text("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("training requires a clean repository worktree")
    return git_sha


def _checkpoint_path(
    output_dir: Path,
    payload: Mapping[str, object],
) -> Path:
    epoch = payload.get("epoch")
    step = payload.get("global_step")
    runtime = payload.get("runtime_state")
    if (
        type(epoch) is not int
        or type(step) is not int
        or not isinstance(runtime, Mapping)
        or type(runtime.get("epoch_complete")) is not bool
    ):
        raise ValueError("checkpoint publication position is invalid")
    if runtime["epoch_complete"]:
        leaf = f"checkpoint_epoch{epoch:03d}.pt"
    else:
        leaf = f"checkpoint_epoch{epoch:03d}_step{step:08d}.pt"
    return output_dir / leaf


def _score_source_records(
    manifest: ManifestIdentity,
    *,
    run_key: str,
) -> list[dict[str, object]]:
    return development_score_source_records(
        manifest,
        run_key=run_key,
    )


def _runtime_manifest_metadata(runtime: object) -> dict[str, object]:
    environment = validate_environment_binding(
        getattr(runtime, "environment_binding", None)
    )
    contract = _mapping(
        getattr(runtime, "contract", None),
        "production runtime contract",
    )
    if dict(contract) != environment["runtime_contract"]:
        raise ValueError("production runtime contract differs from environment binding")
    evidence = _mapping(
        getattr(runtime, "evidence", None),
        "production runtime evidence",
    )
    return {
        "environment": environment,
        "runtime_contract": dict(contract),
        "runtime_contract_sha256": environment["runtime_contract_sha256"],
        "semantic_environment_sha256": environment["semantic_environment_sha256"],
        "runtime_evidence": dict(evidence),
    }


def _build_in_loop_score_metadata(
    *,
    completed: bool,
    global_step: int,
    planned_steps: int,
    checkpoint_sha256: str,
    config_sha256: str,
    git_sha: str,
    protocol_sha256: str,
    seed: int,
    source_records: Sequence[Mapping[str, object]],
    stage: int,
    subject: int,
) -> dict[str, object]:
    if type(completed) is not bool:
        raise TypeError("training completion marker must be boolean")
    if type(stage) is not int or stage not in (0, 2):
        raise ValueError("training score stage must be 0 or 2")
    metadata: dict[str, object] = {
        "checkpoint_sha256": _require_sha(
            checkpoint_sha256,
            "training score checkpoint",
        ),
        "config_sha256": _require_sha(config_sha256, "training score config"),
        "git_sha": git_sha,
        "protocol_sha256": _require_sha(
            protocol_sha256,
            "training score protocol",
        ),
        "seed": seed,
        "source_records": [dict(record) for record in source_records],
        "split_role": "val-dev",
        "stage": f"stage{stage}",
        "subject": subject,
    }
    if not completed:
        if (
            type(global_step) is not int
            or type(planned_steps) is not int
            or global_step <= 0
            or global_step >= planned_steps
        ):
            raise ValueError("partial training progress is invalid")
        metadata.update(
            {
                "global_step": global_step,
                "planned_steps": planned_steps,
                "stage": "training_smoke/in_loop",
                "training_complete": False,
            }
        )
    return metadata


def run_training(arguments: argparse.Namespace) -> int:
    runtime = require_production_runtime(arguments.device)
    git_sha = clean_repository_git_sha()
    paths = _guard_training_paths(arguments)
    upstream_preflight = preflight_upstream_config(paths.config)
    resume_payload: Mapping[str, object] | None = None
    resume_source_checkpoint_sha256: str | None = None
    if paths.resume_checkpoint is not None:
        loaded_resume = load_samga_checkpoint(
            paths.resume_checkpoint,
            requested_scope="train",
        )
        resume_payload = loaded_resume.payload
        resume_epoch = resume_payload.get("epoch")
        if type(resume_epoch) is not int or resume_epoch < 1:
            raise ValueError("resume checkpoint epoch is invalid")
        if resume_epoch >= 51:
            raise ValueError(
                f"resume checkpoint epoch {resume_epoch} requires fresh "
                "recovery because earlier retained epochs cannot be "
                "reconstructed"
            )
        resume_source_checkpoint_sha256 = _require_sha(
            loaded_resume.sha256,
            "resume source checkpoint",
        )
        resume_environment = validate_environment_binding(
            resume_payload.get("environment")
        )
        runtime_environment = validate_environment_binding(runtime.environment_binding)
        if resume_environment != runtime_environment:
            raise ValueError(
                "resume checkpoint environment differs from production runtime"
            )
    manifest = load_development_manifest_identity(
        paths.manifest,
        expected_subject=arguments.subject,
    )
    if manifest.path != paths.manifest:
        raise ValueError("verified manifest path differs from CLI path")
    config = _load_training_config(
        paths.config,
        paths.feature_cache,
        manifest=manifest,
        preflight=upstream_preflight,
    )
    selection = _resolve_candidate_selection(
        arguments,
        paths,
        config,
    )
    components = upstream_preflight.components
    train_dataset = _development_dataset(
        paths,
        manifest,
        scope="train",
        seed=arguments.seed,
    )
    _verify_development_dataset(
        train_dataset,
        manifest=manifest,
        scope="train",
        cache_sha256=config.cache_sha256,
    )
    identities = derive_training_identities(
        components=components,
        train_row_indices=train_dataset.row_indices,
        stage=arguments.stage,
        subject=arguments.subject,
        seed=arguments.seed,
        batch_size=config.batch_size,
        layernorm_config_id=selection.layernorm_config_id,
        whitening_config_id=selection.whitening_config_id,
        preprojector_config_id=selection.preprojector_config_id,
        adapter_kind=selection.adapter_kind,
        adapter_rank=selection.adapter_rank,
        adapter_lr_ratio=selection.adapter_lr_ratio,
        whitening=selection.whitening,
    )
    input_hashes = _build_input_hashes(manifest, config)
    whitening_payload_sha256 = (
        None if selection.whitening is None else selection.whitening.payload_sha256
    )
    resolved_candidate = build_resolved_candidate_payload(
        stage=arguments.stage,
        config_id=selection.config_id,
        subject=arguments.subject,
        seed=arguments.seed,
        baseline_config_sha256=config.semantic.sha256,
        stage2_config_sha256=selection.stage2_config_sha256,
        layernorm_config_id=selection.layernorm_config_id,
        whitening_config_id=selection.whitening_config_id,
        preprojector_config_id=selection.preprojector_config_id,
        adapter_kind=selection.adapter_kind,
        adapter_rank=selection.adapter_rank,
        adapter_lr_ratio=selection.adapter_lr_ratio,
        whitening_payload_sha256=whitening_payload_sha256,
        environment_binding=runtime.environment_binding,
    )
    resolved = resolve_run_config(
        config.protocol,
        resolved_candidate,
        input_hashes,
    )
    if paths.output_dir.name != resolved.run_key:
        raise ValueError("output directory name must equal the resolved run_key")
    candidate_spec = build_candidate_spec(
        stage=arguments.stage,
        config_id=selection.config_id,
        subject=arguments.subject,
        seed=arguments.seed,
        baseline_config_sha256=config.semantic.sha256,
        stage2_config_sha256=selection.stage2_config_sha256,
        semantic_config_sha256=resolved.semantic_config_sha256,
        input_bundle_sha256=resolved.input_bundle_sha256,
        run_key=resolved.run_key,
        layernorm_config_id=selection.layernorm_config_id,
        whitening_config_id=selection.whitening_config_id,
        preprojector_config_id=selection.preprojector_config_id,
        adapter_kind=selection.adapter_kind,
        adapter_rank=selection.adapter_rank,
        adapter_lr_ratio=selection.adapter_lr_ratio,
        whitening=selection.whitening,
        identities=identities,
    )
    run_manifest = build_run_manifest(
        stage=arguments.stage,
        subject=arguments.subject,
        seed=arguments.seed,
        config_id=selection.config_id,
        config_sha256=resolved.semantic_config_sha256,
        protocol_sha256=manifest.protocol_sha256,
        cache_sha256=config.cache_sha256,
        git_sha=git_sha,
        upstream_sha=config.upstream_commit,
        data_order_sha256=identities.data_order_sha256,
        candidate_spec_sha256=candidate_spec["candidate_spec_sha256"],
        run_key=resolved.run_key,
    )
    checkpoint_retention = _CheckpointRetention()

    def checkpoint_sink(
        payload: dict[str, object],
        *,
        retain_for_averaging: bool,
    ) -> CheckpointPublication:
        checkpoint = _checkpoint_path(paths.output_dir, payload)
        digest = save_samga_checkpoint(
            checkpoint,
            payload,
            manifest,
        )
        verified = load_samga_checkpoint(
            checkpoint,
            requested_scope="train",
        )
        if verified.sha256 != digest:
            raise ValueError("published checkpoint SHA-256 verification mismatch")
        checkpoint_retention.record_published(
            checkpoint,
            digest,
            retain_for_averaging=retain_for_averaging,
        )
        return CheckpointPublication(
            reference=str(checkpoint),
            exclusive_create=True,
            atomic_publish=True,
            verified=True,
            durable_retention=retain_for_averaging,
        )

    dataset_factory, development_datasets = _cached_dataset_factory(
        paths,
        manifest,
        seed=arguments.seed,
        cache_sha256=config.cache_sha256,
        train_dataset=train_dataset,
    )
    spec = TrainingCellSpec(
        components=components,
        manifest_path=manifest.path,
        feature_cache=paths.feature_cache,
        stage=arguments.stage,
        subject=arguments.subject,
        seed=arguments.seed,
        config_sha256=resolved.semantic_config_sha256,
        schedule_sha256=SCHEDULE_SHA256,
        trajectory_sha256=identities.trajectory_sha256,
        data_order_sha256=identities.data_order_sha256,
        input_hashes=input_hashes,
        environment=runtime.environment_binding,
        run_manifest=run_manifest,
        candidate_spec=candidate_spec,
        checkpoint_builder=build_epoch_checkpoint,
        checkpoint_restorer=restore_training_checkpoint,
        checkpoint_sink=checkpoint_sink,
        dataset_factory=dataset_factory,
        batch_size=config.batch_size,
        max_train_steps=arguments.max_train_steps,
        num_workers=arguments.num_workers,
        device=runtime.device,
        resume_checkpoint=resume_payload,
        resume_source_checkpoint_sha256=resume_source_checkpoint_sha256,
        layernorm_config_id=selection.layernorm_config_id,
        whitening_config_id=selection.whitening_config_id,
        preprojector_config_id=selection.preprojector_config_id,
        adapter_kind=selection.adapter_kind,
        adapter_rank=selection.adapter_rank,
        adapter_lr_ratio=selection.adapter_lr_ratio,
        whitening=selection.whitening,
    )
    create_development_directory_exclusive(
        paths.output_dir,
        context="SAMGA training output",
    )
    result = run_training_cell(spec)
    final_checkpoint = _checkpoint_path(
        paths.output_dir,
        result.final_checkpoint,
    )
    checkpoint_retention.validate_final(
        completed=result.completed,
        final_checkpoint=final_checkpoint,
    )
    try:
        final_checkpoint_sha256 = checkpoint_retention.hashes[
            final_checkpoint.name
        ]
    except KeyError as exc:
        raise ValueError("final checkpoint was not durably published") from exc

    try:
        validation_dataset = development_datasets["val-dev"]
    except KeyError as exc:
        raise ValueError("training did not construct the val-dev dataset") from exc
    planned_steps = int(SCHEDULE["epochs"]) * len(train_dataset) // config.batch_size
    ScoreArtifact.save(
        paths.output_dir / "in_loop",
        result.final_validation.similarity,
        tuple(validation_dataset.query_ids),
        tuple(validation_dataset.gallery_ids),
        _build_in_loop_score_metadata(
            completed=result.completed,
            global_step=result.global_step,
            planned_steps=planned_steps,
            checkpoint_sha256=final_checkpoint_sha256,
            config_sha256=resolved.semantic_config_sha256,
            git_sha=git_sha,
            protocol_sha256=manifest.protocol_sha256,
            seed=arguments.seed,
            source_records=_score_source_records(manifest, run_key=resolved.run_key),
            stage=arguments.stage,
            subject=arguments.subject,
        ),
    )
    metrics = result.final_validation.metrics
    run_summary = {
        **run_manifest,
        "completed": result.completed,
        "global_step": result.global_step,
        "final_checkpoint": final_checkpoint.name,
        "final_checkpoint_sha256": final_checkpoint_sha256,
        "checkpoint_hashes": dict(
            sorted(checkpoint_retention.hashes.items())
        ),
        "in_loop_score_directory": "in_loop",
        "max_train_steps": arguments.max_train_steps,
        "resume_source_checkpoint_sha256": (resume_source_checkpoint_sha256),
        **_runtime_manifest_metadata(runtime),
        "top1_rate": metrics.top1_rate,
        "top5_rate": metrics.top5_rate,
    }
    write_development_file_exclusive(
        paths.output_dir / "run_manifest.json",
        canonical_json_bytes(run_summary) + b"\n",
        context="SAMGA run manifest output",
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    try:
        arguments = parse_arguments(argv)
        return run_training(arguments)
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
    return 2  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
