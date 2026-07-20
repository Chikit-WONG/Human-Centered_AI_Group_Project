"""Locked checkpoint averaging and SWA for Stage 2 development runs."""

from __future__ import annotations

import hashlib
import io
import math
import os
import pickle
import re
import stat
import struct
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel

from .checkpoint_identity import (
    validate_epoch_checkpoint_identity,
)
from .checkpoint_io import load_typed_torch_checkpoint
from .hashing import canonical_json_bytes


CHECKPOINT_PAYLOAD_TYPE = "samga_brain_rw.epoch_checkpoint"
AVERAGED_CHECKPOINT_PAYLOAD_TYPE = "samga_brain_rw.averaged_checkpoint"
LAST5_EPOCHS = (56, 57, 58, 59, 60)
LAST10_EPOCHS = (51, 52, 53, 54, 55, 56, 57, 58, 59, 60)
AVERAGING_CANDIDATES = {
    "s2-avg-last5": ("arithmetic", LAST5_EPOCHS),
    "s2-avg-last10": ("arithmetic", LAST10_EPOCHS),
    "s2-swa-last5": ("swa", LAST5_EPOCHS),
    "s2-swa-last10": ("swa", LAST10_EPOCHS),
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SUBJECT_TEST_RE = re.compile(
    r"^sub[-_]?\d{2}[-_]?test(?:\.[A-Za-z0-9._-]+)?$",
    re.IGNORECASE,
)
_SEALED_COMPONENTS = {
    "formal",
    "formal_input",
    "formal_refit",
    "formal_test",
    "test",
    "test_images",
    "val_confirm",
}
_CANONICAL_FORMAL_TEST_RECORD_SHA256 = (
    "02d7e33b3fe8e5a571f8db232ca5fa86"
    "abb0c16981876ec84feae7ba64636f1a"
)
_CHECKPOINT_BODY_KEYS = frozenset(
    {
        "candidate_spec",
        "config_sha256",
        "cuda_rng_states",
        "data_order_sha256",
        "effective_batch",
        "environment",
        "epoch",
        "global_step",
        "input_hashes",
        "loader_generator_state",
        "model_state_dict",
        "model_state_sha256",
        "numpy_rng_state",
        "optimizer_stage",
        "optimizer_state_dict",
        "payload_type",
        "python_rng_state",
        "retention",
        "run_manifest",
        "runtime_state",
        "sampler_state_dict",
        "schedule_sha256",
        "scheduler_state_dict",
        "schema_version",
        "seed",
        "subject",
        "torch_rng_state",
        "trajectory_sha256",
        "validation_metrics",
    }
)
_CHECKPOINT_SCOPE_KEYS = frozenset(
    {"observed_scopes", "scope", "validation_scope"}
)
_SIDECAR_PROVENANCE_KEYS = frozenset(
    {"config_sha256", "manifest_sha256", "protocol_sha256", "seed", "subject"}
)
_SIDECAR_METADATA_KEYS = frozenset(
    {
        "complete",
        "observed_scopes",
        "ordered_ids",
        "retention",
        "source_records",
        "train_ordered_ids",
        "val_dev_ordered_ids",
    }
)
_SOURCE_RECORD_KEYS = frozenset(
    {
        "manifest_sha256",
        "records_sha256",
        "role",
        "role_payload_sha256",
        "source_manifest_sha256",
        "source_payload_sha256",
    }
)
_AVERAGED_BODY_KEYS = frozenset(
    {
        "schema_version",
        "payload_type",
        "candidate_id",
        "method",
        "epochs",
        "subject",
        "seed",
        "config_sha256",
        "data_order_sha256",
        "candidate_spec_sha256",
        "input_bundle_sha256",
        "run_key",
        "schedule_sha256",
        "optimizer_stage",
        "trajectory_sha256",
        "model_state_dict",
        "model_state_sha256",
        "arithmetic_model_state_sha256",
        "source_checkpoints",
        "strict_control_epoch",
        "strict_control_checkpoint_sha256",
        "alias_of",
    }
)
_AVERAGED_KEYS = _AVERAGED_BODY_KEYS | {"payload_sha256"}
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)


@dataclass(frozen=True)
class VerifiedEpochCheckpoint:
    """A transport-, schema-, state-, and identity-verified epoch checkpoint."""

    path: Path
    sha256: str
    epoch: int
    global_step: int
    subject: int
    seed: int
    config_id: str
    config_sha256: str
    schedule_sha256: str
    optimizer_stage: str
    trajectory_sha256: str
    data_order_sha256: str
    candidate_spec_sha256: str
    input_bundle_sha256: str
    run_key: str
    payload: Mapping[str, object]
    input_hashes: Mapping[str, object]
    environment: Mapping[str, object]
    run_manifest: Mapping[str, object]
    candidate_spec: Mapping[str, object]
    runtime_state: Mapping[str, object]
    retention: Mapping[str, object]
    model_state_dict: dict[str, torch.Tensor]


@dataclass
class _SecureParent:
    path: Path
    leaf: str
    parent_fd: int
    descriptors: list[int]
    edges: list[tuple[int, str, tuple[int, int, int]]]

    def verify(self) -> None:
        for parent_fd, component, expected in self.edges:
            try:
                current = os.stat(
                    component,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise ValueError(
                    "checkpoint path changed during secure traversal"
                ) from exc
            if stat.S_ISLNK(current.st_mode) or _node_identity(current) != expected:
                raise ValueError("checkpoint path changed during secure traversal")


def _require_sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _require_positive_integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context} must be a positive integer")
    return value


def _require_nonnegative_integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context} must be a non-negative integer")
    return value


def _normalized(path: Path) -> Path:
    raw = os.fspath(path)
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise ValueError("checkpoint path must be nonempty safe text")
    return Path(os.path.abspath(os.path.normpath(raw)))


def _reject_sensitive_or_symlink_path(path: Path, context: str) -> Path:
    normalized = _normalized(path)
    for component in normalized.parts:
        semantic = re.sub(r"[^a-z0-9]+", "_", component.lower()).strip("_")
        if semantic in _SEALED_COMPONENTS or _SUBJECT_TEST_RE.fullmatch(component):
            raise ValueError(f"{context} contains a sealed-scope component")
    current = Path(normalized.anchor)
    for component in normalized.parts[1:]:
        current /= component
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            break
        except OSError as exc:
            raise ValueError(f"{context} cannot be inspected safely") from exc
        if stat.S_ISLNK(mode):
            raise ValueError(f"{context} contains a symlink component")
    return normalized


def validate_development_checkpoint_path(path: Path, context: str) -> Path:
    """Reject sealed-scope and currently symlinked path components."""

    return _reject_sensitive_or_symlink_path(Path(path), context)


def _node_identity(value: os.stat_result) -> tuple[int, int, int]:
    return value.st_dev, value.st_ino, stat.S_IFMT(value.st_mode)


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _open_directory_at(parent_fd: int, component: str) -> int:
    try:
        descriptor = os.open(
            component,
            os.O_RDONLY | _O_CLOEXEC | _O_NOFOLLOW | _O_DIRECTORY,
            dir_fd=parent_fd,
        )
    except OSError as exc:
        raise ValueError("checkpoint path contains an unsafe directory") from exc
    value = os.fstat(descriptor)
    if not stat.S_ISDIR(value.st_mode):
        os.close(descriptor)
        raise ValueError("checkpoint path component must be a directory")
    return descriptor


@contextmanager
def _secure_parent_directory(
    path: Path,
    *,
    create: bool,
    context: str,
) -> Iterator[_SecureParent]:
    normalized = validate_development_checkpoint_path(path, context)
    parts = normalized.parts
    if len(parts) < 2 or not normalized.name:
        raise ValueError(f"{context} must name a file")
    descriptors: list[int] = []
    edges: list[tuple[int, str, tuple[int, int, int]]] = []
    try:
        root_fd = os.open(
            normalized.anchor,
            os.O_RDONLY | _O_CLOEXEC | _O_DIRECTORY,
        )
        descriptors.append(root_fd)
        current_fd = root_fd
        for component in parts[1:-1]:
            try:
                child_fd = _open_directory_at(current_fd, component)
            except ValueError:
                if not create:
                    raise
                try:
                    os.mkdir(component, mode=0o700, dir_fd=current_fd)
                except FileExistsError:
                    pass
                except OSError as exc:
                    raise ValueError(
                        "checkpoint parent directory cannot be created safely"
                    ) from exc
                child_fd = _open_directory_at(current_fd, component)
            child_stat = os.fstat(child_fd)
            edges.append((current_fd, component, _node_identity(child_stat)))
            descriptors.append(child_fd)
            current_fd = child_fd
        secured = _SecureParent(
            path=normalized,
            leaf=parts[-1],
            parent_fd=current_fd,
            descriptors=descriptors,
            edges=edges,
        )
        secured.verify()
        yield secured
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _read_checkpoint_bytes(path: Path) -> tuple[bytes, str]:
    with _secure_parent_directory(
        path,
        create=False,
        context="checkpoint path",
    ) as parent:
        parent.verify()
        try:
            descriptor = os.open(
                parent.leaf,
                os.O_RDONLY | _O_CLOEXEC | _O_NOFOLLOW,
                dir_fd=parent.parent_fd,
            )
        except OSError as exc:
            raise ValueError("checkpoint could not be opened securely") from exc
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("checkpoint must be a regular file")
            try:
                named = os.stat(
                    parent.leaf,
                    dir_fd=parent.parent_fd,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise ValueError("checkpoint path changed before read") from exc
            if _node_identity(named) != _node_identity(before):
                raise ValueError("checkpoint path changed before read")
            parent.verify()
            chunks: list[bytes] = []
            digest = hashlib.sha256()
            while True:
                chunk = os.read(descriptor, 4 * 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
                digest.update(chunk)
            after = os.fstat(descriptor)
            if _file_identity(before) != _file_identity(after):
                raise ValueError("checkpoint changed while it was read")
            try:
                named_after = os.stat(
                    parent.leaf,
                    dir_fd=parent.parent_fd,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise ValueError("checkpoint path changed during read") from exc
            if _node_identity(named_after) != _node_identity(after):
                raise ValueError("checkpoint path changed during read")
            parent.verify()
            return b"".join(chunks), digest.hexdigest()
        finally:
            os.close(descriptor)


def _load_torch_mapping(path: Path, context: str) -> tuple[dict[str, object], str]:
    normalized = _normalized(path)
    raw, digest = _read_checkpoint_bytes(normalized)
    try:
        payload = torch.load(
            io.BytesIO(raw),
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
        raise ValueError(f"invalid {context}: {normalized}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{context} payload must be a mapping")
    if any(not isinstance(key, str) for key in payload):
        raise ValueError(f"{context} payload keys must be strings")
    return dict(payload), digest


def _validate_model_state(
    raw_state: object,
    *,
    context: str,
) -> dict[str, torch.Tensor]:
    if not isinstance(raw_state, Mapping) or not raw_state:
        raise ValueError(f"{context} must be a nonempty mapping")
    state: dict[str, torch.Tensor] = {}
    for key, value in raw_state.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{context} keys must be nonempty strings")
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{context} {key!r} must be a tensor")
        tensor = value.detach().cpu().contiguous()
        if tensor.layout != torch.strided:
            raise ValueError(f"{context} {key!r} must be a strided tensor")
        if (torch.is_floating_point(tensor) or torch.is_complex(tensor)) and not bool(
            torch.isfinite(tensor).all().item()
        ):
            raise ValueError(f"{context} {key!r} is non-finite")
        state[key] = tensor
    return state


def _require_mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{context} keys must be strings")
    return value


def _require_exact_keys(
    value: Mapping[str, object],
    expected: frozenset[str],
    context: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{context} schema mismatch: missing={missing}, extra={extra}"
        )


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
                token = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
                if token in _SEALED_COMPONENTS or re.fullmatch(
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
        if _CANONICAL_FORMAL_TEST_RECORD_SHA256 in value.lower():
            raise PermissionError(
                f"{context} contains the formal-test record hash"
            )
        for component in Path(value).parts:
            token = re.sub(
                r"[^a-z0-9]+",
                "_",
                component.lower(),
            ).strip("_")
            if token in _SEALED_COMPONENTS or re.fullmatch(
                r"sub_?\d+_test(?:_.*)?",
                token,
            ):
                raise PermissionError(
                    f"{context} contains a sealed path component"
                )


def _validate_source_records(
    value: object,
    *,
    provenance: Mapping[str, object],
) -> None:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(
            "checkpoint sidecar source_records must contain train and val-dev"
        )
    expected_roles = ("train", "val-dev")
    shared: tuple[object, ...] | None = None
    for index, (raw, role) in enumerate(
        zip(value, expected_roles, strict=True)
    ):
        record = _require_mapping(
            raw,
            f"checkpoint sidecar source record {index}",
        )
        _require_exact_keys(
            record,
            _SOURCE_RECORD_KEYS,
            f"checkpoint sidecar source record {index}",
        )
        if record["role"] != role:
            raise ValueError("checkpoint sidecar source role mismatch")
        for key in _SOURCE_RECORD_KEYS - {"role"}:
            _require_sha256(
                record[key],
                f"checkpoint sidecar source record {index} {key}",
            )
        if record["manifest_sha256"] != provenance["manifest_sha256"]:
            raise ValueError(
                "checkpoint sidecar source manifest binding mismatch"
            )
        current_shared = (
            record["manifest_sha256"],
            record["records_sha256"],
            record["source_manifest_sha256"],
            record["source_payload_sha256"],
        )
        if shared is None:
            shared = current_shared
        elif current_shared != shared:
            raise ValueError(
                "checkpoint sidecar source-record identity mismatch"
            )


def _validate_checkpoint_bundle(
    payload: Mapping[str, object],
    envelope: Mapping[str, object],
) -> tuple[
    int,
    int,
    int,
    str,
    str,
    str,
    str,
    dict[str, torch.Tensor],
]:
    _require_exact_keys(
        payload,
        _CHECKPOINT_BODY_KEYS | _CHECKPOINT_SCOPE_KEYS,
        "checkpoint payload",
    )
    validate_epoch_checkpoint_identity(payload, envelope)
    if (
        payload["schema_version"] != 1
        or payload["payload_type"] != CHECKPOINT_PAYLOAD_TYPE
    ):
        raise ValueError("checkpoint payload identity/schema mismatch")
    epoch = _require_positive_integer(payload["epoch"], "checkpoint epoch")
    if epoch > 60:
        raise ValueError("checkpoint epoch must be in 1..60")
    _require_nonnegative_integer(
        payload["global_step"],
        "checkpoint global_step",
    )
    subject = _require_positive_integer(payload["subject"], "checkpoint subject")
    if subject > 10:
        raise ValueError("checkpoint subject must be in 1..10")
    seed = _require_nonnegative_integer(payload["seed"], "checkpoint seed")
    config = _require_sha256(payload["config_sha256"], "checkpoint config")
    schedule = _require_sha256(payload["schedule_sha256"], "checkpoint schedule")
    trajectory = _require_sha256(
        payload["trajectory_sha256"],
        "checkpoint trajectory",
    )
    _require_sha256(
        payload["data_order_sha256"],
        "checkpoint data order",
    )
    optimizer_stage = payload["optimizer_stage"]
    if not isinstance(optimizer_stage, str) or not optimizer_stage:
        raise ValueError("checkpoint optimizer stage must be a nonempty string")
    state = _validate_model_state(
        payload["model_state_dict"],
        context="checkpoint model state",
    )
    claimed_state_hash = _require_sha256(
        payload["model_state_sha256"],
        "checkpoint model state",
    )
    if hash_state_dict(state) != claimed_state_hash:
        raise ValueError("checkpoint model-state hash mismatch")
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
        _require_mapping(payload[key], f"checkpoint {key}")
    if (
        payload["scope"] != "train"
        or payload["validation_scope"] != "val-dev"
        or payload["observed_scopes"] != ["train", "val-dev"]
    ):
        raise PermissionError("checkpoint is not development-only")

    provenance = _require_mapping(
        envelope.get("provenance"),
        "checkpoint sidecar provenance",
    )
    _require_exact_keys(
        provenance,
        _SIDECAR_PROVENANCE_KEYS,
        "checkpoint sidecar provenance",
    )
    for key in ("manifest_sha256", "protocol_sha256"):
        _require_sha256(provenance[key], f"checkpoint sidecar {key}")
    expected_provenance = {
        "config_sha256": config,
        "seed": seed,
        "subject": subject,
    }
    for key, expected in expected_provenance.items():
        if provenance[key] != expected:
            raise ValueError(
                f"checkpoint sidecar provenance {key} binding mismatch"
            )

    metadata = _require_mapping(
        envelope.get("metadata"),
        "checkpoint sidecar metadata",
    )
    _require_exact_keys(
        metadata,
        _SIDECAR_METADATA_KEYS,
        "checkpoint sidecar metadata",
    )
    if (
        metadata["complete"] is not True
        or metadata["observed_scopes"] != ["train", "val-dev"]
        or metadata["retention"] != payload["retention"]
    ):
        raise ValueError("checkpoint sidecar metadata binding mismatch")
    _validate_source_records(
        metadata["source_records"],
        provenance=provenance,
    )
    _reject_sealed_checkpoint_metadata(payload)
    return (
        epoch,
        subject,
        seed,
        config,
        schedule,
        optimizer_stage,
        trajectory,
        state,
    )


def verify_epoch_checkpoint(path: Path) -> VerifiedEpochCheckpoint:
    """Fully verify one development epoch checkpoint bundle."""

    normalized = validate_development_checkpoint_path(
        Path(path),
        "checkpoint path",
    )
    loaded = load_typed_torch_checkpoint(
        normalized,
        payload_type=CHECKPOINT_PAYLOAD_TYPE,
        requested_scope="train",
    )
    (
        epoch,
        subject,
        seed,
        config,
        schedule,
        optimizer_stage,
        trajectory,
        state,
    ) = _validate_checkpoint_bundle(loaded.payload, loaded.envelope)
    input_hashes = _require_mapping(
        loaded.payload["input_hashes"],
        "checkpoint input_hashes",
    )
    environment = _require_mapping(
        loaded.payload["environment"],
        "checkpoint environment",
    )
    run_manifest = _require_mapping(
        loaded.payload["run_manifest"],
        "checkpoint run manifest",
    )
    candidate_spec = _require_mapping(
        loaded.payload["candidate_spec"],
        "checkpoint candidate_spec",
    )
    runtime_state = _require_mapping(
        loaded.payload["runtime_state"],
        "checkpoint runtime state",
    )
    retention = _require_mapping(
        loaded.payload["retention"],
        "checkpoint retention",
    )
    config_id = run_manifest["config_id"]
    if not isinstance(config_id, str) or not config_id:
        raise ValueError("checkpoint config_id must be nonempty")
    run_key = run_manifest["run_key"]
    if not isinstance(run_key, str) or not run_key:
        raise ValueError("checkpoint run_key must be nonempty")
    return VerifiedEpochCheckpoint(
        path=normalized,
        sha256=loaded.sha256,
        epoch=epoch,
        global_step=_require_nonnegative_integer(
            loaded.payload["global_step"],
            "checkpoint global_step",
        ),
        subject=subject,
        seed=seed,
        config_id=config_id,
        config_sha256=config,
        schedule_sha256=schedule,
        optimizer_stage=optimizer_stage,
        trajectory_sha256=trajectory,
        data_order_sha256=_require_sha256(
            loaded.payload["data_order_sha256"],
            "checkpoint data order",
        ),
        candidate_spec_sha256=_require_sha256(
            candidate_spec["candidate_spec_sha256"],
            "checkpoint candidate spec",
        ),
        input_bundle_sha256=_require_sha256(
            candidate_spec["input_bundle_sha256"],
            "checkpoint input bundle",
        ),
        run_key=run_key,
        payload=loaded.payload,
        input_hashes=input_hashes,
        environment=environment,
        run_manifest=run_manifest,
        candidate_spec=candidate_spec,
        runtime_state=runtime_state,
        retention=retention,
        model_state_dict=state,
    )


def _load_checkpoint(path: Path) -> VerifiedEpochCheckpoint:
    return verify_epoch_checkpoint(path)


def _validate_window(
    paths: Sequence[Path],
) -> tuple[VerifiedEpochCheckpoint, ...]:
    if isinstance(paths, (str, bytes, bytearray)):
        raise TypeError("checkpoint paths must be a sequence")
    normalized = tuple(Path(path) for path in paths)
    if len(normalized) not in {5, 10}:
        raise ValueError("checkpoint window must be exactly last-5 or last-10")
    if len({_normalized(path) for path in normalized}) != len(normalized):
        raise ValueError("checkpoint window contains duplicate paths")
    checkpoints = tuple(_load_checkpoint(path) for path in normalized)
    epochs = tuple(item.epoch for item in checkpoints)
    if epochs not in {LAST5_EPOCHS, LAST10_EPOCHS}:
        raise ValueError("checkpoint window must be exact, complete, and ordered")

    first = checkpoints[0]
    if first.optimizer_stage != "stage2":
        raise ValueError("averaging checkpoints must use optimizer stage2")
    identity_fields = (
        ("config", "config_sha256"),
        ("subject", "subject"),
        ("seed", "seed"),
        ("schedule", "schedule_sha256"),
        ("optimizer stage", "optimizer_stage"),
        ("trajectory", "trajectory_sha256"),
        ("data order", "data_order_sha256"),
        ("candidate", "candidate_spec_sha256"),
        ("input bundle", "input_bundle_sha256"),
        ("run key", "run_key"),
    )
    for item in checkpoints[1:]:
        for label, field in identity_fields:
            if getattr(item, field) != getattr(first, field):
                raise ValueError(f"checkpoint {label} mismatch")

    reference = first.model_state_dict
    reference_keys = set(reference)
    for item in checkpoints[1:]:
        state = item.model_state_dict
        if set(state) != reference_keys:
            raise ValueError("checkpoint model state key mismatch")
        for key, original in reference.items():
            candidate = state[key]
            if candidate.shape != original.shape:
                raise ValueError(f"checkpoint state shape mismatch for {key}")
            if candidate.dtype != original.dtype:
                raise ValueError(f"checkpoint state dtype mismatch for {key}")
            if not torch.is_floating_point(original) and not torch.equal(
                candidate, original
            ):
                raise ValueError(f"checkpoint non-floating state mismatch for {key}")
    return checkpoints


def _arithmetic(
    checkpoints: Sequence[VerifiedEpochCheckpoint],
) -> dict[str, torch.Tensor]:
    reference = checkpoints[0].model_state_dict
    result: dict[str, torch.Tensor] = {}
    for key, original in reference.items():
        if torch.is_floating_point(original):
            total = torch.zeros_like(original, dtype=torch.float64)
            for item in checkpoints:
                total.add_(item.model_state_dict[key].to(torch.float64))
            result[key] = (total / len(checkpoints)).to(original.dtype)
        else:
            result[key] = original.clone()
    return result


class _SwaCarrier(nn.Module):
    def __init__(self, tensors: Sequence[torch.Tensor]) -> None:
        super().__init__()
        self.values = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros_like(tensor),
                    requires_grad=False,
                )
                for tensor in tensors
            ]
        )

    def load_values(self, tensors: Sequence[torch.Tensor]) -> None:
        with torch.no_grad():
            for destination, source in zip(self.values, tensors, strict=True):
                destination.copy_(source)


def _swa(
    checkpoints: Sequence[VerifiedEpochCheckpoint],
) -> dict[str, torch.Tensor]:
    reference = checkpoints[0].model_state_dict
    floating_keys = tuple(
        key for key, value in reference.items() if torch.is_floating_point(value)
    )
    carrier = _SwaCarrier([reference[key] for key in floating_keys])
    averaged = AveragedModel(carrier, device=torch.device("cpu"), use_buffers=False)
    for item in checkpoints:
        carrier.load_values([item.model_state_dict[key] for key in floating_keys])
        averaged.update_parameters(carrier)
    floating_indices = {key: index for index, key in enumerate(floating_keys)}
    result: dict[str, torch.Tensor] = {}
    for key, original in reference.items():
        if torch.is_floating_point(original):
            result[key] = averaged.module.values[floating_indices[key]].detach().clone()
        else:
            result[key] = original.clone()
    return result


def average_state_dicts(paths: Sequence[Path]) -> dict[str, torch.Tensor]:
    """Equal-weight arithmetic average over exactly epochs 56–60 or 51–60."""

    return _arithmetic(_validate_window(paths))


def swa_state_dicts(paths: Sequence[Path]) -> dict[str, torch.Tensor]:
    """Update a real-dtype PyTorch AveragedModel once at each locked epoch."""

    return _swa(_validate_window(paths))


def _tensor_semantics(tensor: torch.Tensor) -> dict[str, object]:
    value = tensor.detach().cpu().contiguous()
    if value.layout != torch.strided:
        raise ValueError("payload tensors must use strided layout")
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "data_sha256": hashlib.sha256(raw).hexdigest(),
    }


def hash_state_dict(state_dict: Mapping[str, torch.Tensor]) -> str:
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("state_dict must be a nonempty mapping")
    normalized: dict[str, object] = {}
    for key, tensor in state_dict.items():
        if not isinstance(key, str) or not key:
            raise ValueError("state_dict keys must be nonempty strings")
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("state_dict values must be tensors")
        normalized[key] = _tensor_semantics(tensor)
    return hashlib.sha256(canonical_json_bytes(normalized)).hexdigest()


def _payload_semantics(value: object, context: str = "payload") -> object:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{context} contains a non-finite float")
        return value
    if isinstance(value, torch.Tensor):
        return {"__tensor__": _tensor_semantics(value)}
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{context} keys must be nonempty strings")
            result[key] = _payload_semantics(item, f"{context}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _payload_semantics(item, f"{context}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ValueError(f"{context} contains an unsupported value")


def hash_averaged_checkpoint_payload(payload: Mapping[str, object]) -> str:
    """Hash every averaged-checkpoint semantic except the hash field itself."""

    if not isinstance(payload, Mapping):
        raise ValueError("averaged checkpoint payload must be a mapping")
    body = {key: value for key, value in payload.items() if key != "payload_sha256"}
    normalized = _payload_semantics(body, "averaged payload")
    return hashlib.sha256(canonical_json_bytes(normalized)).hexdigest()


def _verify_source_checkpoints(
    value: object,
    expected_epochs: tuple[int, ...],
) -> list[dict[str, object]]:
    if not isinstance(value, list) or len(value) != len(expected_epochs):
        raise ValueError("source checkpoints must match the exact epoch window")
    result: list[dict[str, object]] = []
    for index, (entry_value, epoch) in enumerate(
        zip(value, expected_epochs, strict=True)
    ):
        if not isinstance(entry_value, Mapping):
            raise ValueError(f"source checkpoint {index} must be a mapping")
        entry = dict(entry_value)
        if set(entry) != {"epoch", "path", "sha256"}:
            raise ValueError(f"source checkpoint {index} schema mismatch")
        if entry["epoch"] != epoch:
            raise ValueError("source checkpoint epoch mismatch")
        source_path = entry["path"]
        if not isinstance(source_path, str) or not Path(source_path).is_absolute():
            raise ValueError("source checkpoint path must be absolute")
        validate_development_checkpoint_path(
            Path(source_path),
            "source checkpoint path",
        )
        _require_sha256(entry["sha256"], "source checkpoint sha256")
        result.append(entry)
    normalized_paths = {
        _normalized(Path(str(entry["path"]))) for entry in result
    }
    if len(normalized_paths) != len(result):
        raise ValueError("source checkpoint paths must be unique")
    return result


def verify_averaged_checkpoint_payload(value: object) -> dict[str, object]:
    """Verify the sealed averaged-checkpoint schema and every semantic binding."""

    if not isinstance(value, Mapping):
        raise ValueError("averaged checkpoint payload must be a mapping")
    payload = dict(value)
    if set(payload) != _AVERAGED_KEYS:
        raise ValueError("averaged checkpoint payload schema mismatch")
    claimed_hash = _require_sha256(
        payload["payload_sha256"],
        "averaged payload SHA-256",
    )
    calculated_hash = hash_averaged_checkpoint_payload(payload)
    if claimed_hash != calculated_hash:
        raise ValueError("averaged payload SHA-256 hash does not match")
    if payload["schema_version"] != 1:
        raise ValueError("averaged checkpoint schema_version must be 1")
    if payload["payload_type"] != AVERAGED_CHECKPOINT_PAYLOAD_TYPE:
        raise ValueError("averaged checkpoint payload_type mismatch")
    candidate_id = payload["candidate_id"]
    if not isinstance(candidate_id, str) or candidate_id not in AVERAGING_CANDIDATES:
        raise ValueError("unknown averaging candidate ID")
    expected_method, expected_epochs = AVERAGING_CANDIDATES[candidate_id]
    if payload["method"] != expected_method:
        raise ValueError("averaging candidate method mismatch")
    if payload["epochs"] != list(expected_epochs):
        raise ValueError("averaging candidate epoch window mismatch")
    subject = _require_positive_integer(payload["subject"], "averaged subject")
    if subject > 10:
        raise ValueError("averaged subject must be in 1..10")
    _require_nonnegative_integer(payload["seed"], "averaged seed")
    _require_sha256(payload["config_sha256"], "averaged config")
    _require_sha256(payload["data_order_sha256"], "averaged data order")
    _require_sha256(
        payload["candidate_spec_sha256"],
        "averaged candidate spec",
    )
    _require_sha256(
        payload["input_bundle_sha256"],
        "averaged input bundle",
    )
    if not isinstance(payload["run_key"], str) or not payload["run_key"]:
        raise ValueError("averaged run key must be nonempty")
    _require_sha256(payload["schedule_sha256"], "averaged schedule")
    _require_sha256(payload["trajectory_sha256"], "averaged trajectory")
    if payload["optimizer_stage"] != "stage2":
        raise ValueError("averaged optimizer stage must be stage2")
    state = _validate_model_state(
        payload["model_state_dict"],
        context="averaged model state",
    )
    state_hash = _require_sha256(
        payload["model_state_sha256"],
        "averaged model state",
    )
    if hash_state_dict(state) != state_hash:
        raise ValueError("averaged model state hash does not match")
    arithmetic_hash = _require_sha256(
        payload["arithmetic_model_state_sha256"],
        "arithmetic model state",
    )
    sources = _verify_source_checkpoints(payload["source_checkpoints"], expected_epochs)
    if payload["strict_control_epoch"] != 60:
        raise ValueError("strict paired control must be raw epoch 60")
    strict_hash = _require_sha256(
        payload["strict_control_checkpoint_sha256"],
        "strict control checkpoint",
    )
    if sources[-1]["epoch"] != 60 or sources[-1]["sha256"] != strict_hash:
        raise ValueError("strict paired control must match the source epoch 60")
    expected_alias: str | None = None
    if expected_method == "arithmetic":
        if arithmetic_hash != state_hash:
            raise ValueError("arithmetic candidate state hash mismatch")
    elif arithmetic_hash == state_hash:
        expected_alias = (
            "s2-avg-last5" if expected_epochs == LAST5_EPOCHS else "s2-avg-last10"
        )
    if payload["alias_of"] != expected_alias:
        raise ValueError("averaging alias binding does not match state hashes")
    return payload


def load_averaged_checkpoint(path: Path) -> dict[str, object]:
    """Securely load and verify one sealed averaged checkpoint."""

    payload, _ = _load_torch_mapping(Path(path), "averaged checkpoint")
    return verify_averaged_checkpoint_payload(payload)


def build_averaged_checkpoint(
    paths: Sequence[Path],
    *,
    candidate_id: str,
) -> dict[str, object]:
    if candidate_id not in AVERAGING_CANDIDATES:
        raise ValueError("unknown averaging candidate ID")
    method, expected_epochs = AVERAGING_CANDIDATES[candidate_id]
    checkpoints = _validate_window(paths)
    epochs = tuple(item.epoch for item in checkpoints)
    if epochs != expected_epochs:
        raise ValueError("candidate ID does not match checkpoint window")
    arithmetic = _arithmetic(checkpoints)
    state = arithmetic if method == "arithmetic" else _swa(checkpoints)
    state_hash = hash_state_dict(state)
    arithmetic_hash = hash_state_dict(arithmetic)
    alias_of: str | None = None
    if method == "swa" and state_hash == arithmetic_hash:
        alias_of = "s2-avg-last5" if epochs == LAST5_EPOCHS else "s2-avg-last10"
    source_checkpoints = [
        {
            "epoch": item.epoch,
            "path": str(item.path),
            "sha256": item.sha256,
        }
        for item in checkpoints
    ]
    first = checkpoints[0]
    result: dict[str, object] = {
        "schema_version": 1,
        "payload_type": AVERAGED_CHECKPOINT_PAYLOAD_TYPE,
        "candidate_id": candidate_id,
        "method": method,
        "epochs": list(epochs),
        "subject": first.subject,
        "seed": first.seed,
        "config_sha256": first.config_sha256,
        "data_order_sha256": first.data_order_sha256,
        "candidate_spec_sha256": first.candidate_spec_sha256,
        "input_bundle_sha256": first.input_bundle_sha256,
        "run_key": first.run_key,
        "schedule_sha256": first.schedule_sha256,
        "optimizer_stage": first.optimizer_stage,
        "trajectory_sha256": first.trajectory_sha256,
        "model_state_dict": state,
        "model_state_sha256": state_hash,
        "arithmetic_model_state_sha256": arithmetic_hash,
        "source_checkpoints": source_checkpoints,
        "strict_control_epoch": 60,
        "strict_control_checkpoint_sha256": checkpoints[-1].sha256,
        "alias_of": alias_of,
    }
    result["payload_sha256"] = hash_averaged_checkpoint_payload(result)
    return verify_averaged_checkpoint_payload(result)


def _unlink_created_file(parent: _SecureParent, identity: tuple[int, int, int]) -> None:
    try:
        current = os.stat(
            parent.leaf,
            dir_fd=parent.parent_fd,
            follow_symlinks=False,
        )
    except OSError:
        return
    if _node_identity(current) != identity:
        return
    try:
        os.unlink(parent.leaf, dir_fd=parent.parent_fd)
    except OSError:
        pass


def write_averaged_checkpoint_exclusive(
    path: Path,
    payload: object,
) -> Path:
    """Verify and exclusively publish one averaged checkpoint through dirfds."""

    output = _normalized(Path(path))
    if output.suffix != ".pt":
        raise ValueError("averaged checkpoint output must use .pt")
    verified = verify_averaged_checkpoint_payload(payload)
    buffer = io.BytesIO()
    torch.save(verified, buffer)
    raw = buffer.getvalue()
    with _secure_parent_directory(
        output,
        create=True,
        context="averaged checkpoint output",
    ) as parent:
        parent.verify()
        descriptor = -1
        created_identity: tuple[int, int, int] | None = None
        try:
            descriptor = os.open(
                parent.leaf,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_CLOEXEC | _O_NOFOLLOW,
                0o600,
                dir_fd=parent.parent_fd,
            )
            created = os.fstat(descriptor)
            if not stat.S_ISREG(created.st_mode):
                raise ValueError("averaged checkpoint output must be a regular file")
            created_identity = _node_identity(created)
            parent.verify()
            offset = 0
            while offset < len(raw):
                written = os.write(descriptor, raw[offset:])
                if written <= 0:
                    raise OSError("short averaged-checkpoint write")
                offset += written
            os.fsync(descriptor)
            named = os.stat(
                parent.leaf,
                dir_fd=parent.parent_fd,
                follow_symlinks=False,
            )
            if _node_identity(named) != created_identity:
                raise ValueError("averaged checkpoint output path changed")
            parent.verify()
            os.fsync(parent.parent_fd)
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
                descriptor = -1
            if created_identity is not None:
                _unlink_created_file(parent, created_identity)
            raise
        finally:
            if descriptor >= 0:
                os.close(descriptor)
    return output


__all__ = [
    "AVERAGING_CANDIDATES",
    "VerifiedEpochCheckpoint",
    "average_state_dicts",
    "build_averaged_checkpoint",
    "hash_averaged_checkpoint_payload",
    "hash_state_dict",
    "load_averaged_checkpoint",
    "swa_state_dicts",
    "validate_development_checkpoint_path",
    "verify_epoch_checkpoint",
    "verify_averaged_checkpoint_payload",
    "write_averaged_checkpoint_exclusive",
]
