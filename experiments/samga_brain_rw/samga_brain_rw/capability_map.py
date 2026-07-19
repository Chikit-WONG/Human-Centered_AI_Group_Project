"""Build the fixed Stage-0 train capability map without directory scans.

All 49 explicit inputs are denied, opened, hashed, and semantically validated
before the output directory is created.  The 39 regular generic artifacts get
strict Task-3 sidecars; the 10 Task-2 protocol manifests remain their own
role-aware envelopes.
"""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from . import provenance as _provenance
from .access import TypedArtifact, VerifiedArtifact, verify_typed_artifacts
from .hashing import canonical_json_bytes, ordered_ids_sha256, sha256_json
from .provenance import (
    CAPABILITY_PAYLOAD_TYPES,
    ProvenanceInputs,
    preflight_provenance_inputs,
)


CAPABILITY_MAP_FILENAME = "capability_map.json"
_READ_FLAGS = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_WRITE_FLAGS = (
    os.O_WRONLY
    | os.O_CREAT
    | os.O_EXCL
    | getattr(os, "O_CLOEXEC", 0)
    | _O_NOFOLLOW
)
_HASH_CHUNK_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class _FileSnapshot:
    path: Path
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int
    sha256: str

    @classmethod
    def from_stat(
        cls,
        path: Path,
        value: os.stat_result,
        digest: str,
    ) -> "_FileSnapshot":
        return cls(
            path=path,
            device=value.st_dev,
            inode=value.st_ino,
            size=value.st_size,
            mtime_ns=value.st_mtime_ns,
            ctime_ns=value.st_ctime_ns,
            sha256=digest,
        )

    @property
    def identity(self) -> tuple[int, int, int, int, int]:
        return (
            self.device,
            self.inode,
            self.size,
            self.mtime_ns,
            self.ctime_ns,
        )


@dataclass
class _OutputDirectory:
    path: Path
    name: str
    parent_fd: int
    directory_fd: int


def build_stage0_capability_map(
    inputs: ProvenanceInputs,
    output_directory: Path,
) -> Path:
    """Validate every fixed train input, then exclusively publish sidecars."""

    if not isinstance(inputs, ProvenanceInputs):
        raise TypeError("inputs must be ProvenanceInputs")
    output = _normalized(Path(output_directory))
    _validate_output_destination(output)

    paths = preflight_provenance_inputs(inputs)
    if tuple(paths) != tuple(CAPABILITY_PAYLOAD_TYPES):
        raise ValueError("capability path registry order mismatch")

    snapshots = {
        key: _hash_regular_file(_normalized(path))
        for key, path in paths.items()
    }
    if len(snapshots) != 49:
        raise AssertionError("Stage-0 requires exactly 49 input snapshots")

    _validate_snapshots_semantically(inputs, snapshots)
    for snapshot in snapshots.values():
        _recheck_snapshot_identity(snapshot)

    sidecars, map_payload = _build_output_payloads(
        inputs,
        output,
        snapshots,
    )
    handle = _create_output_directory(output)
    created_names: list[str] = []
    try:
        for name, payload in sidecars:
            created_names.append(name)
            _write_exclusive(
                handle.directory_fd,
                name,
                canonical_json_bytes(payload),
            )
        created_names.append(CAPABILITY_MAP_FILENAME)
        _write_exclusive(
            handle.directory_fd,
            CAPABILITY_MAP_FILENAME,
            canonical_json_bytes(map_payload),
        )
        os.fsync(handle.directory_fd)
        os.fsync(handle.parent_fd)
    except BaseException:
        _cleanup_failed_output(handle, created_names)
        raise
    else:
        os.close(handle.directory_fd)
        handle.directory_fd = -1
        os.close(handle.parent_fd)
        handle.parent_fd = -1
    return output / CAPABILITY_MAP_FILENAME


def _build_output_payloads(
    inputs: ProvenanceInputs,
    output: Path,
    snapshots: Mapping[str, _FileSnapshot],
) -> tuple[list[tuple[str, dict[str, object]]], dict[str, object]]:
    sidecars: list[tuple[str, dict[str, object]]] = []
    entries: list[dict[str, object]] = []
    for index, key in enumerate(CAPABILITY_PAYLOAD_TYPES):
        snapshot = snapshots[key]
        payload_type = CAPABILITY_PAYLOAD_TYPES[key]
        is_protocol_role = key.startswith("protocol_manifest.")
        if is_protocol_role:
            envelope_path = snapshot.path
            role: str | None = "train"
        else:
            filename = f"{index:02d}-{key}.envelope.json"
            envelope_path = output / filename
            role = None
            sidecars.append(
                (
                    filename,
                    _build_generic_envelope(
                        inputs,
                        key,
                        payload_type,
                        snapshot,
                    ),
                )
            )
        entries.append(
            {
                "envelope_path": str(envelope_path),
                "key": key,
                "payload_path": str(snapshot.path),
                "payload_type": payload_type,
                "role": role,
            }
        )
    if len(sidecars) != 39 or len(entries) != 49:
        raise AssertionError("Stage-0 output cardinality invariant failed")
    return sidecars, {
        "artifacts": entries,
        "payload_type": "samga_brain_rw.capability_map",
        "schema_version": 1,
        "scope": "train",
    }


def _build_generic_envelope(
    inputs: ProvenanceInputs,
    key: str,
    payload_type: str,
    snapshot: _FileSnapshot,
) -> dict[str, object]:
    ordered_ids = [key]
    source_records: list[object] = []
    metadata = {
        "absolute_payload_path": str(snapshot.path),
        "byte_count": snapshot.size,
        "capability_key": key,
        "ordered_ids": ordered_ids,
        "source_records": source_records,
    }
    provenance = {
        "experiment_revision": inputs.experiment_revision,
        "generator": _provenance.CAPABILITY_ENVELOPE_GENERATOR,
        "protocol_config_sha256": inputs.oracles.protocol_config_sha256,
    }
    return {
        "metadata": metadata,
        "metadata_sha256": sha256_json(metadata),
        "ordered_ids_sha256": ordered_ids_sha256(ordered_ids),
        "payload_sha256": snapshot.sha256,
        "payload_type": payload_type,
        "provenance": provenance,
        "provenance_sha256": sha256_json(provenance),
        "schema_version": 1,
        "scope": "train",
        "source_records_sha256": sha256_json(source_records),
    }


def _validate_snapshots_semantically(
    inputs: ProvenanceInputs,
    snapshots: Mapping[str, _FileSnapshot],
) -> None:
    capabilities = _snapshot_capabilities(snapshots)
    oracles = inputs.oracles

    protocol = _provenance._read_json(
        capabilities["protocol"],
        "protocol config",
    )
    internvit = _provenance._read_json(
        capabilities["internvit_config"],
        "InternViT semantic config",
    )
    brainrw = _provenance._read_json(
        capabilities["brainrw_config"],
        "brain-rw semantic config",
    )
    _provenance._validate_protocol(protocol, oracles)
    _provenance._validate_semantic_configs(
        internvit,
        brainrw,
        inputs,
        oracles,
    )
    _provenance._validate_protocol_registry(
        capabilities,
        inputs,
        oracles,
    )
    _provenance._validate_sources(
        inputs,
        capabilities,
        oracles,
    )
    _provenance._validate_protocol_capabilities(capabilities)
    _provenance._validate_model_files(capabilities, oracles)
    _provenance._validate_caches(
        capabilities,
        inputs,
        internvit,
        oracles,
    )

    model_config = _provenance._read_json(
        capabilities["internvit.config"],
        "InternViT model config",
    )
    transformers_version = model_config.get("transformers_version")
    if transformers_version is not None and not isinstance(
        transformers_version,
        str,
    ):
        raise ValueError("model config transformers_version must be a string")


def _snapshot_capabilities(
    snapshots: Mapping[str, _FileSnapshot],
) -> dict[str, VerifiedArtifact]:
    capabilities: dict[str, VerifiedArtifact] = {}
    protocol_keys: list[str] = []
    protocol_descriptors: list[TypedArtifact] = []
    for key in CAPABILITY_PAYLOAD_TYPES:
        snapshot = snapshots[key]
        role = "train" if key.startswith("protocol_manifest.") else None
        descriptor = TypedArtifact(
            payload_type=CAPABILITY_PAYLOAD_TYPES[key],
            payload_path=snapshot.path,
            envelope_path=snapshot.path,
            role=role,
        )
        if role is None:
            capabilities[key] = VerifiedArtifact(
                artifact=descriptor,
                scope="train",
                device=snapshot.device,
                inode=snapshot.inode,
                size=snapshot.size,
                mtime_ns=snapshot.mtime_ns,
                ctime_ns=snapshot.ctime_ns,
                payload_sha256=snapshot.sha256,
                envelope_device=snapshot.device,
                envelope_inode=snapshot.inode,
                envelope_size=snapshot.size,
                envelope_mtime_ns=snapshot.mtime_ns,
                envelope_ctime_ns=snapshot.ctime_ns,
                envelope_sha256=snapshot.sha256,
            )
        else:
            protocol_keys.append(key)
            protocol_descriptors.append(descriptor)

    verified_protocols = verify_typed_artifacts(
        "train",
        protocol_descriptors,
    )
    if len(verified_protocols) != 10:
        raise ValueError("exactly 10 train protocol manifests are required")
    for key, capability in zip(
        protocol_keys,
        verified_protocols,
        strict=True,
    ):
        snapshot = snapshots[key]
        identity = (
            capability.device,
            capability.inode,
            capability.size,
            capability.mtime_ns,
            capability.ctime_ns,
        )
        if identity != snapshot.identity or (
            capability.payload_sha256 != snapshot.sha256
        ):
            raise ValueError(f"{key} changed after its initial hash")
        capabilities[key] = capability
    return capabilities


def _hash_regular_file(path: Path) -> _FileSnapshot:
    descriptor = _open_readonly_nofollow(path, "Stage-0 input")
    try:
        before = os.fstat(descriptor)
        _require_regular_file(before, "Stage-0 input")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, _HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        _require_stable_identity(before, after, "Stage-0 input")
    finally:
        os.close(descriptor)
    return _FileSnapshot.from_stat(path, after, digest.hexdigest())


def _recheck_snapshot_identity(snapshot: _FileSnapshot) -> None:
    descriptor = _open_readonly_nofollow(
        snapshot.path,
        "Stage-0 identity recheck",
    )
    try:
        current = os.fstat(descriptor)
        _require_regular_file(current, "Stage-0 identity recheck")
    finally:
        os.close(descriptor)
    identity = (
        current.st_dev,
        current.st_ino,
        current.st_size,
        current.st_mtime_ns,
        current.st_ctime_ns,
    )
    if identity != snapshot.identity:
        raise ValueError("Stage-0 input identity changed after validation")


def _validate_output_destination(output: Path) -> None:
    _provenance._reject_forbidden_path(output)
    if output == Path(output.anchor):
        raise ValueError("output directory cannot be a filesystem root")
    parent_fd = _open_directory_nofollow(
        output.parent,
        "output parent directory",
    )
    try:
        try:
            os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        raise FileExistsError(
            f"output directory already exists: {output}"
        )
    finally:
        os.close(parent_fd)


def _create_output_directory(output: Path) -> _OutputDirectory:
    parent_fd = _open_directory_nofollow(
        output.parent,
        "output parent directory",
    )
    directory_fd = -1
    directory_identity: tuple[int, int] | None = None
    created = False
    try:
        os.mkdir(output.name, mode=0o700, dir_fd=parent_fd)
        created = True
        directory_fd = os.open(
            output.name,
            _READ_FLAGS | _O_DIRECTORY | _O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        current = os.fstat(directory_fd)
        directory_identity = (current.st_dev, current.st_ino)
        os.fsync(parent_fd)
    except BaseException:
        if directory_fd >= 0:
            os.close(directory_fd)
        if created and directory_identity is not None:
            try:
                _rmdir_if_owned(
                    parent_fd,
                    output.name,
                    directory_identity,
                )
                os.fsync(parent_fd)
            except OSError:
                pass
        os.close(parent_fd)
        raise
    return _OutputDirectory(
        path=output,
        name=output.name,
        parent_fd=parent_fd,
        directory_fd=directory_fd,
    )


def _write_exclusive(
    directory_fd: int,
    name: str,
    data: bytes,
) -> None:
    descriptor = os.open(
        name,
        _WRITE_FLAGS,
        0o600,
        dir_fd=directory_fd,
    )
    try:
        offset = 0
        while offset < len(data):
            written = os.write(descriptor, data[offset:])
            if written <= 0:
                raise OSError("exclusive output write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _cleanup_failed_output(
    handle: _OutputDirectory,
    created_names: list[str],
) -> None:
    directory_identity: tuple[int, int] | None = None
    if handle.directory_fd >= 0:
        current = os.fstat(handle.directory_fd)
        directory_identity = (current.st_dev, current.st_ino)
        for name in reversed(created_names):
            try:
                os.unlink(name, dir_fd=handle.directory_fd)
            except FileNotFoundError:
                pass
            except OSError:
                pass
        try:
            os.fsync(handle.directory_fd)
        except OSError:
            pass
        os.close(handle.directory_fd)
        handle.directory_fd = -1
    if handle.parent_fd >= 0:
        try:
            if directory_identity is not None:
                _rmdir_if_owned(
                    handle.parent_fd,
                    handle.name,
                    directory_identity,
                )
        except OSError:
            pass
        try:
            os.fsync(handle.parent_fd)
        except OSError:
            pass
        os.close(handle.parent_fd)
        handle.parent_fd = -1


def _rmdir_if_owned(
    parent_fd: int,
    name: str,
    expected_identity: tuple[int, int],
) -> None:
    try:
        current = os.stat(
            name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(current.st_mode):
        return
    if (current.st_dev, current.st_ino) != expected_identity:
        return
    os.rmdir(name, dir_fd=parent_fd)


def _open_readonly_nofollow(path: Path, context: str) -> int:
    normalized = _normalized(path)
    parts = normalized.parts
    if len(parts) <= 1:
        raise ValueError(f"{context} must name a regular file")
    directory_fd = os.open(
        normalized.anchor,
        _READ_FLAGS | _O_DIRECTORY,
    )
    try:
        for component in parts[1:-1]:
            next_fd = os.open(
                component,
                _READ_FLAGS | _O_DIRECTORY | _O_NOFOLLOW,
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = next_fd
        return os.open(
            parts[-1],
            _READ_FLAGS | _O_NOFOLLOW,
            dir_fd=directory_fd,
        )
    except OSError as exc:
        raise ValueError(f"{context} could not be opened securely") from exc
    finally:
        os.close(directory_fd)


def _open_directory_nofollow(path: Path, context: str) -> int:
    normalized = _normalized(path)
    directory_fd = os.open(
        normalized.anchor,
        _READ_FLAGS | _O_DIRECTORY,
    )
    try:
        for component in normalized.parts[1:]:
            next_fd = os.open(
                component,
                _READ_FLAGS | _O_DIRECTORY | _O_NOFOLLOW,
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = next_fd
        result = directory_fd
        directory_fd = -1
        return result
    except OSError as exc:
        raise ValueError(f"{context} could not be opened securely") from exc
    finally:
        if directory_fd >= 0:
            os.close(directory_fd)


def _require_regular_file(value: os.stat_result, context: str) -> None:
    if not stat.S_ISREG(value.st_mode):
        raise ValueError(f"{context} must be a regular file")


def _require_stable_identity(
    before: os.stat_result,
    after: os.stat_result,
    context: str,
) -> None:
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity:
        raise ValueError(f"{context} changed while hashing")


def _normalized(path: Path) -> Path:
    return Path(os.path.abspath(os.path.normpath(os.fspath(path))))
