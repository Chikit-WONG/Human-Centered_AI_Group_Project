#!/usr/bin/env python3
"""CPU-only train provenance preflight.

The capability map supplies exactly 49 explicit typed descriptors.  It never
grants trust by itself: every descriptor is passed through the Task 3 access
verifier before any experiment payload is semantically loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import secrets
import socket
import stat
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from samga_brain_rw.access import (
    TypedArtifact,
    VerifiedArtifact,
    verify_typed_artifacts,
)
from samga_brain_rw.hashing import canonical_json_bytes
from samga_brain_rw.provenance import (
    CAPABILITY_PAYLOAD_TYPES,
    DEFAULT_ORACLES,
    ENVIRONMENT_VARIABLE_ALLOWLIST,
    PACKAGE_VERSION_ALLOWLIST,
    EnvironmentSnapshot,
    ProvenanceInputs,
    build_provenance_manifest,
    expected_capability_paths,
    preflight_provenance_inputs,
)


_MAP_TOP_KEYS = {
    "artifacts",
    "payload_type",
    "schema_version",
    "scope",
}
_MAP_ENTRY_KEYS = {
    "envelope_path",
    "key",
    "payload_path",
    "payload_type",
    "role",
}
_SUBJECT_TEST_RE = re.compile(r"^sub-\d{2}_test\.json$", re.IGNORECASE)
_FORMAL_TEST_RECORD_SHA256 = (
    "02d7e33b3fe8e5a571f8db232ca5fa86abb0c16981876ec84feae7ba64636f1a"
)
_MAX_MAP_BYTES = 8 * 1024 * 1024


_READ_FLAGS = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_READ_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class _CapabilityMapDocument:
    path: Path
    payload: dict[str, object]
    raw_sha256: str


def capture_environment(
    *,
    environ: Mapping[str, str] | None = None,
    version_lookup: Callable[[str], str | None] | None = None,
) -> EnvironmentSnapshot:
    """Capture only the preregistered package and environment allowlists."""

    source_environment = os.environ if environ is None else environ

    def default_version_lookup(package: str) -> str | None:
        try:
            return importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            return None

    lookup = default_version_lookup if version_lookup is None else version_lookup
    return EnvironmentSnapshot(
        python_version=sys.version,
        python_executable=sys.executable,
        sys_prefix=sys.prefix,
        platform=platform.platform(),
        machine=platform.machine(),
        hostname=socket.gethostname(),
        package_versions={
            package: lookup(package) for package in PACKAGE_VERSION_ALLOWLIST
        },
        selected_environment={
            name: source_environment.get(name)
            for name in ENVIRONMENT_VARIABLE_ALLOWLIST
        },
    )


def load_and_verify_capability_map(
    path: Path,
    expected_paths: Mapping[str, Path],
) -> dict[str, VerifiedArtifact]:
    """Strictly parse and jointly verify all 49 train-only descriptors."""

    capability_map_path = Path(path)
    _reject_path(capability_map_path, "capability-map path")
    document = _read_canonical_map(capability_map_path)
    return _verify_capability_map_document(document, expected_paths)


def _verify_capability_map_document(
    document: _CapabilityMapDocument,
    expected_paths: Mapping[str, Path],
) -> dict[str, VerifiedArtifact]:
    payload = document.payload
    if set(payload) != _MAP_TOP_KEYS:
        raise ValueError("capability map top-level keys do not match schema")
    if payload["schema_version"] != 1 or type(payload["schema_version"]) is not int:
        raise ValueError("capability map schema_version must be 1")
    if payload["payload_type"] != "samga_brain_rw.capability_map":
        raise ValueError("capability map payload_type mismatch")
    if payload["scope"] != "train":
        raise ValueError("capability map scope must be train")
    artifacts = payload["artifacts"]
    if not isinstance(artifacts, list):
        raise ValueError("capability map artifacts must be an array")
    expected_keys = tuple(CAPABILITY_PAYLOAD_TYPES)
    if len(expected_keys) != 49:
        raise AssertionError("the sealed capability registry must contain 49 keys")
    if len(artifacts) != len(expected_keys):
        raise ValueError("capability map must contain exactly 49 artifacts")

    normalized_expected = {
        key: _normalized(value) for key, value in expected_paths.items()
    }
    if tuple(normalized_expected) != expected_keys:
        raise ValueError("expected capability path keys do not match registry")

    descriptors: list[TypedArtifact] = []
    for index, (expected_key, raw_entry) in enumerate(
        zip(expected_keys, artifacts, strict=True)
    ):
        if not isinstance(raw_entry, dict) or set(raw_entry) != _MAP_ENTRY_KEYS:
            raise ValueError(f"capability map artifacts[{index}] schema mismatch")
        key = raw_entry["key"]
        if key != expected_key:
            raise ValueError(
                "capability map keys must be complete and in canonical registry order"
            )
        payload_type = raw_entry["payload_type"]
        if payload_type != CAPABILITY_PAYLOAD_TYPES[expected_key]:
            raise ValueError(f"{expected_key} payload_type mismatch")
        payload_path = _absolute_path(
            raw_entry["payload_path"], f"{expected_key} payload_path"
        )
        envelope_path = _absolute_path(
            raw_entry["envelope_path"], f"{expected_key} envelope_path"
        )
        _reject_path(payload_path, f"{expected_key} payload_path")
        _reject_path(envelope_path, f"{expected_key} envelope_path")
        if _normalized(payload_path) != normalized_expected[expected_key]:
            raise ValueError(f"{expected_key} payload path mismatch")
        expected_role = (
            "train" if expected_key.startswith("protocol_manifest.") else None
        )
        role = raw_entry["role"]
        if role != expected_role:
            raise ValueError(f"{expected_key} role mismatch")
        descriptors.append(
            TypedArtifact(
                payload_type=payload_type,
                payload_path=payload_path,
                envelope_path=envelope_path,
                role=role,
            )
        )

    verified = verify_typed_artifacts("train", descriptors)
    if len(verified) != 49:
        raise ValueError("typed verifier returned the wrong capability count")
    return {
        key: capability
        for key, capability in zip(expected_keys, verified, strict=True)
    }


def publish_canonical_exclusive(path: Path, payload: object) -> None:
    """Publish canonical JSON atomically and never replace an existing file."""

    output = Path(path)
    _reject_path(output, "output path")
    data = canonical_json_bytes(payload)
    parent = output.parent
    if not parent.is_dir():
        raise ValueError("output parent directory must already exist")
    temporary = parent / f".{output.name}.{secrets.token_hex(16)}.tmp"
    descriptor = -1
    linked = False
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.link(temporary, output, follow_symlinks=False)
        linked = True
        directory_fd = os.open(
            parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        if not linked and output.exists():
            # An existing output belongs to the winner of an exclusive race.
            # It is deliberately never removed here.
            pass


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the sealed train-only SAMGA brain-rw preflight"
    )
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--internvit-config", type=Path, required=True)
    parser.add_argument("--brainrw-config", type=Path, required=True)
    parser.add_argument("--source-manifest-dir", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--feature-directory", type=Path, required=True)
    parser.add_argument("--variant-directory", type=Path, required=True)
    parser.add_argument("--canonical-cache", type=Path, required=True)
    parser.add_argument("--clip-train-cache", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--clip-model-path", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--experiment-revision", required=True)
    parser.add_argument("--upstream-revision", required=True)
    parser.add_argument("--cache-generator-revision", required=True)
    parser.add_argument("--capability-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    environment = capture_environment()
    inputs = ProvenanceInputs(
        repository_root=args.repository_root,
        protocol_path=args.protocol,
        internvit_config_path=args.internvit_config,
        brainrw_config_path=args.brainrw_config,
        source_manifest_dir=args.source_manifest_dir,
        protocol_manifest_dir=args.manifest_dir,
        feature_directory=args.feature_directory,
        variant_directory=args.variant_directory,
        canonical_cache=args.canonical_cache,
        clip_train_cache=args.clip_train_cache,
        data_root=args.data_root,
        model_path=args.model_path,
        clip_model_path=args.clip_model_path,
        upstream_root=args.upstream_root,
        experiment_revision=args.experiment_revision,
        upstream_revision=args.upstream_revision,
        cache_generator_revision=args.cache_generator_revision,
        verified_artifacts={},
        environment=environment,
        oracles=DEFAULT_ORACLES,
    )
    expected_paths = preflight_provenance_inputs(inputs)
    capability_map_path = Path(args.capability_map)
    _reject_path(capability_map_path, "capability-map path")
    capability_map_document = _read_canonical_map(capability_map_path)
    capabilities = _verify_capability_map_document(
        capability_map_document,
        expected_paths,
    )
    manifest = build_provenance_manifest(
        replace(inputs, verified_artifacts=capabilities)
    )
    manifest = _bind_capability_map(
        manifest,
        capability_map_document,
    )
    publish_canonical_exclusive(args.output, manifest)
    return 0


def _read_canonical_map(path: Path) -> _CapabilityMapDocument:
    normalized = _normalized(path)
    descriptor = _open_readonly_nofollow(
        normalized,
        "capability map",
    )
    try:
        before = os.fstat(descriptor)
        _require_regular_file(before, "capability map")
        chunks: list[bytes] = []
        byte_count = 0
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, _READ_CHUNK_BYTES)
            if not chunk:
                break
            byte_count += len(chunk)
            if byte_count > _MAX_MAP_BYTES:
                raise ValueError("capability map exceeds the size limit")
            chunks.append(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
        _require_stable_identity(before, after, "capability map")
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("capability map is malformed canonical JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("capability map must be a JSON object")
    if raw != canonical_json_bytes(value):
        raise ValueError("capability map must use exact canonical JSON bytes")
    return _CapabilityMapDocument(
        path=normalized,
        payload=value,
        raw_sha256=digest.hexdigest(),
    )


def _bind_capability_map(
    manifest: Mapping[str, object],
    document: _CapabilityMapDocument,
) -> dict[str, object]:
    inventory = manifest.get("capability_inventory")
    if not isinstance(inventory, dict) or set(inventory) != {
        "artifact_count",
        "artifacts",
        "inventory_sha256",
    }:
        raise ValueError("provenance capability inventory schema mismatch")
    artifacts = inventory["artifacts"]
    if not isinstance(artifacts, list) or len(artifacts) != 49:
        raise ValueError("provenance capability inventory cardinality mismatch")
    if inventory["artifact_count"] != len(artifacts):
        raise ValueError("provenance capability inventory count mismatch")
    if inventory["inventory_sha256"] != hashlib.sha256(
        canonical_json_bytes(artifacts)
    ).hexdigest():
        raise ValueError("provenance capability inventory digest mismatch")

    projection: list[dict[str, object]] = []
    for index, item in enumerate(artifacts):
        if not isinstance(item, dict):
            raise ValueError(
                f"provenance capability inventory artifacts[{index}] invalid"
            )
        try:
            projection.append(
                {
                    "envelope_path": item["envelope_path"],
                    "key": item["key"],
                    "payload_path": item["payload_path"],
                    "payload_type": item["payload_type"],
                    "role": item["role"],
                }
            )
        except KeyError as exc:
            raise ValueError(
                f"provenance capability inventory artifacts[{index}] invalid"
            ) from exc
    if projection != document.payload["artifacts"]:
        raise ValueError(
            "capability map artifacts do not match provenance inventory"
        )

    bound = dict(manifest)
    bound["capability_map"] = {
        "artifact_count": len(artifacts),
        "inventory_sha256": inventory["inventory_sha256"],
        "path": str(document.path),
        "raw_sha256": document.raw_sha256,
        "schema_version": document.payload["schema_version"],
    }
    return bound


def _open_readonly_nofollow(path: Path, context: str) -> int:
    normalized = _normalized(path)
    components = normalized.parts
    if len(components) <= 1:
        raise ValueError(f"{context} must name a regular file")
    directory_fd = os.open(
        normalized.anchor,
        _READ_FLAGS | _O_DIRECTORY,
    )
    try:
        for component in components[1:-1]:
            next_fd = os.open(
                component,
                _READ_FLAGS | _O_DIRECTORY | _O_NOFOLLOW,
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = next_fd
        return os.open(
            components[-1],
            _READ_FLAGS | _O_NOFOLLOW,
            dir_fd=directory_fd,
        )
    finally:
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
        raise ValueError(f"{context} changed while it was read")


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate capability-map JSON key: {key}")
        result[key] = value
    return result


def _reject_non_finite(value: str) -> object:
    raise ValueError(f"non-finite capability-map JSON value: {value}")


def _absolute_path(value: object, context: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{context} must be absolute")
    return path


def _reject_path(path: Path, context: str) -> None:
    raw = os.fspath(path)
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise ValueError(f"{context} is forbidden")
    normalized = _normalized(path)
    lowered = str(normalized).lower()
    if _FORMAL_TEST_RECORD_SHA256 in lowered:
        raise ValueError(f"{context} is forbidden")
    if _SUBJECT_TEST_RE.fullmatch(normalized.name):
        raise ValueError(f"{context} is forbidden")
    if any(
        component.lower() in {"test_images", "val-confirm", "formal-test"}
        for component in normalized.parts
    ):
        raise ValueError(f"{context} is forbidden")
    current = Path(normalized.anchor)
    for component in normalized.parts[1:]:
        current = current / component
        try:
            current_stat = os.lstat(current)
        except FileNotFoundError:
            break
        if stat.S_ISLNK(current_stat.st_mode):
            raise ValueError(f"{context} is forbidden: symlink component")


def _normalized(path: Path) -> Path:
    return Path(os.path.abspath(os.path.normpath(os.fspath(path))))


if __name__ == "__main__":
    raise SystemExit(main())
