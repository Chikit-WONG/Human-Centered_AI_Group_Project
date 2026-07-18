"""Fail-closed access guards for typed SAMGA brain-rw artifacts.

The verifier in this module deliberately stops at producing a verified file
capability.  It never performs semantic NumPy, PyTorch, or JSON payload loads.
Consumers must use :meth:`VerifiedArtifact.open_verified` so that the bytes
they load come from the same no-follow file descriptor whose identity and
digest were rechecked.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal

from .hashing import ordered_ids_sha256, sha256_json


AccessScope = Literal[
    "train",
    "val-dev",
    "val-confirm",
    "formal-refit",
    "formal-input",
    "formal-test",
]

_ACCESS_SCOPES = frozenset(
    {
        "train",
        "val-dev",
        "val-confirm",
        "formal-refit",
        "formal-input",
        "formal-test",
    }
)
_SCOPE_INPUTS = {
    "train": frozenset({"train"}),
    "val-dev": frozenset({"train", "val-dev"}),
    "val-confirm": frozenset({"train", "val-confirm"}),
    "formal-refit": frozenset({"formal-refit"}),
    "formal-input": frozenset({"formal-refit", "formal-input"}),
    "formal-test": frozenset(
        {"formal-refit", "formal-input", "formal-test"}
    ),
}
_AUTHORIZATION_SCOPES = frozenset(
    {"val-confirm", "formal-refit", "formal-input", "formal-test"}
)

_CANONICAL_FORMAL_TEST_RECORD_SHA256 = (
    "02d7e33b3fe8e5a571f8db232ca5fa86abb0c16981876ec84feae7ba64636f1a"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SUBJECT_TEST_FILENAME_RE = re.compile(
    r"^sub-\d{2}_test\.json$", re.IGNORECASE
)
_MAX_ENVELOPE_BYTES = 16 * 1024 * 1024

_GENERIC_ENVELOPE_KEYS = frozenset(
    {
        "schema_version",
        "payload_type",
        "scope",
        "source_records_sha256",
        "ordered_ids_sha256",
        "payload_sha256",
        "provenance",
        "provenance_sha256",
        "metadata",
        "metadata_sha256",
    }
)
_GENERIC_PAYLOAD_TYPES = frozenset(
    {
        "feature-cache",
        "checkpoint",
        "score-matrix",
        "refit-artifact",
        "train-cache",
        "adapter",
        "formal-image",
        "direct-input",
        "eeg",
        "similarities",
        "predictions",
        "metrics",
        "samga_brain_rw.feature_cache",
        "samga_brain_rw.model_config",
        "samga_brain_rw.model_preprocessor",
        "samga_brain_rw.model_source",
        "samga_brain_rw.model_weights",
        "samga_brain_rw.protocol_config",
        "samga_brain_rw.semantic_config",
        "samga_brain_rw.checkpoint",
        "samga_brain_rw.score_matrix",
        "samga_brain_rw.refit_artifact",
        "samga_brain_rw.refit_manifest",
        "samga_brain_rw.refit_checkpoint",
        "samga_brain_rw.source_manifest",
        "samga_brain_rw.source_train_pt",
        "samga_brain_rw.adapter",
        "samga_brain_rw.train_cache",
        "samga_brain_rw.train_cache_metadata",
        "samga_brain_rw.formal_image",
        "samga_brain_rw.formal_direct_input",
        "samga_brain_rw.eeg",
        "samga_brain_rw.similarities",
        "samga_brain_rw.predictions",
        "samga_brain_rw.metrics",
    }
)
_FORMAL_INPUT_FORBIDDEN_TYPE_TERMS = (
    "eeg",
    "similarit",
    "score",
    "prediction",
    "metric",
    "rank",
)

_SUBJECT_PROTOCOL_TYPE = "samga_brain_rw.subject_protocol_manifest"
_ROLE_PAYLOAD_TYPE = "samga_brain_rw.role_payload"
_PROTOCOL_ROLES = frozenset({"train", "val-dev", "val-confirm"})
_PROTOCOL_TOP_KEYS = frozenset(
    {
        "payload_type",
        "protocol_config_sha256",
        "records_sha256",
        "role_artifacts",
        "role_payloads",
        "schema_version",
        "source_manifest_path",
        "source_manifest_sha256",
        "split_assignment",
        "split_assignment_payload_sha256",
        "subject_id",
    }
)
_ROLE_DESCRIPTOR_KEYS = frozenset(
    {
        "ordered_ids_sha256",
        "payload_type",
        "provenance_sha256",
        "role_payload_sha256",
        "schema_version",
        "scope",
        "source_records_sha256",
    }
)
_ROLE_PAYLOAD_KEYS = frozenset(
    {
        "concept_count",
        "concept_ids",
        "gallery_ids",
        "ordered_ids",
        "payload_type",
        "query_ids",
        "row_count",
        "row_indices",
        "schema_version",
        "scope",
    }
)

_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_READ_FLAGS = os.O_RDONLY | _O_CLOEXEC


@dataclass(frozen=True)
class TypedArtifact:
    """A payload and the strict envelope that grants it a declared type."""

    payload_type: str
    payload_path: Path
    envelope_path: Path
    role: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.payload_type, str) or not self.payload_type:
            raise ValueError("payload_type must be a non-empty string")
        object.__setattr__(self, "payload_path", Path(self.payload_path))
        object.__setattr__(self, "envelope_path", Path(self.envelope_path))
        if self.role is not None and (
            not isinstance(self.role, str) or not self.role
        ):
            raise ValueError("role must be a non-empty string or None")


@dataclass(frozen=True, init=False)
class AccessAuthorization:
    """Reserved opaque proof for scope-specific verified chain validators.

    No generic issuer exists. Until scope-specific validators are wired to
    immutable job maps and cross-process claim loaders, every sensitive scope
    remains deliberately unavailable.
    """

    scope: AccessScope
    seal_sha256: str
    job_map_sha256: str
    claim_sha256: str
    audit_sha256: str | None

    def __init__(
        self,
        scope: AccessScope,
        seal_sha256: str,
        job_map_sha256: str,
        claim_sha256: str,
        audit_sha256: str | None = None,
        *,
        _issuer_token: object | None = None,
    ) -> None:
        raise PermissionError(
            "AccessAuthorization is an opaque verified capability"
        )


@dataclass(frozen=True)
class VerifiedArtifact:
    """Identity- and digest-bound capability for one verified payload."""

    artifact: TypedArtifact
    scope: AccessScope
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int
    payload_sha256: str

    @property
    def dev(self) -> int:
        """Alias matching POSIX ``st_dev`` terminology."""

        return self.device

    @property
    def sha256(self) -> str:
        """Alias for the verified payload digest."""

        return self.payload_sha256

    @contextmanager
    def open_verified(self) -> Iterator[BinaryIO]:
        """Yield the same no-follow fd whose identity and hash were checked."""

        fd = _open_readonly_nofollow(
            self.artifact.payload_path, "verified payload"
        )
        try:
            before = os.fstat(fd)
            _require_regular_file(before, "verified payload")
            self._check_identity(before)
            digest = _sha256_fd(fd)
            after = os.fstat(fd)
            _require_stable_identity(before, after, "verified payload")
            self._check_identity(after)
            if digest != self.payload_sha256:
                raise ValueError(
                    "verified payload changed: SHA-256 digest mismatch"
                )
            os.lseek(fd, 0, os.SEEK_SET)
            with os.fdopen(fd, "rb", closefd=True) as payload_file:
                fd = -1
                yield payload_file

                live_fd = payload_file.fileno()
                post_load = os.fstat(live_fd)
                self._check_identity(post_load)
                post_digest = _sha256_fd(live_fd)
                final = os.fstat(live_fd)
                _require_stable_identity(
                    post_load, final, "verified payload"
                )
                self._check_identity(final)
                if post_digest != self.payload_sha256:
                    raise ValueError(
                        "verified payload changed while it was in use"
                    )
        finally:
            if fd >= 0:
                os.close(fd)

    def revalidate(self) -> None:
        """Recheck identity and digest without semantically loading bytes."""

        with self.open_verified():
            pass

    def _check_identity(self, current: os.stat_result) -> None:
        identity = (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
        )
        expected = (self.device, self.inode, self.size, self.mtime_ns, self.ctime_ns)
        if identity != expected:
            raise ValueError("verified payload file identity changed")


def verify_typed_artifacts(
    scope: AccessScope,
    artifacts: Sequence[TypedArtifact],
    *,
    authorization: AccessAuthorization | None = None,
) -> tuple[VerifiedArtifact, ...]:
    """Verify strict envelopes and return payload capabilities.

    Every lexical, explicit-deny, symlink-component, scope, and authorization
    check is completed before any envelope is opened.  Each generic envelope is
    then validated in full before its payload is opened and hashed.
    """

    requested_scope = _require_access_scope(scope)
    if isinstance(artifacts, (str, bytes, bytearray)):
        raise TypeError("artifacts must be a sequence of TypedArtifact values")
    descriptors = tuple(artifacts)
    if any(not isinstance(item, TypedArtifact) for item in descriptors):
        raise TypeError("artifacts must contain only TypedArtifact values")

    # This loop is intentionally separate from envelope processing.  A denied
    # later descriptor must not allow an earlier envelope to be opened first.
    for artifact in descriptors:
        _preflight_artifact_paths(artifact)
    _validate_authorization(requested_scope, authorization)

    verified: list[VerifiedArtifact] = []
    for artifact in descriptors:
        envelope, envelope_digest, envelope_stat = _read_strict_envelope(
            artifact.envelope_path
        )
        if envelope.get("payload_type") == _SUBJECT_PROTOCOL_TYPE:
            _verify_protocol_role_envelope(
                artifact,
                envelope,
                requested_scope,
            )
            expected_payload_sha256 = envelope_digest
            if _normalized_path(artifact.payload_path) != _normalized_path(
                artifact.envelope_path
            ):
                raise ValueError(
                    "Task 2 protocol payload and envelope paths must match"
                )
            expected_identity = envelope_stat
        else:
            expected_payload_sha256 = _verify_generic_envelope(
                artifact,
                envelope,
                requested_scope,
            )
            expected_identity = None

        payload_stat, payload_digest = _verify_payload_file(
            artifact.payload_path,
            expected_payload_sha256,
            expected_identity=expected_identity,
        )
        verified.append(
            VerifiedArtifact(
                artifact=artifact,
                scope=requested_scope,
                device=payload_stat.st_dev,
                inode=payload_stat.st_ino,
                size=payload_stat.st_size,
                mtime_ns=payload_stat.st_mtime_ns,
                ctime_ns=payload_stat.st_ctime_ns,
                payload_sha256=payload_digest,
            )
        )
    return tuple(verified)


def require_typed_artifacts(
    scope: AccessScope,
    artifacts: Sequence[TypedArtifact],
    *,
    authorization: AccessAuthorization | None = None,
) -> None:
    """Compatibility guard that discards the verified capabilities."""

    verify_typed_artifacts(
        scope,
        artifacts,
        authorization=authorization,
    )
    return None


def _require_access_scope(value: object) -> AccessScope:
    if not isinstance(value, str) or value not in _ACCESS_SCOPES:
        raise ValueError(f"unknown access scope: {value!r}")
    return value  # type: ignore[return-value]


def _validate_authorization(
    scope: AccessScope,
    authorization: AccessAuthorization | None,
) -> None:
    if scope not in _AUTHORIZATION_SCOPES:
        if authorization is not None:
            raise ValueError(
                f"{scope} does not accept an authorization capability"
            )
        return
    raise PermissionError(
        f"{scope} requires its verified seal/job-map/audit/claim chain; "
        "generic authorization issuance is disabled"
    )


def _preflight_artifact_paths(artifact: TypedArtifact) -> None:
    _preflight_path(artifact.payload_path, "payload path")
    _preflight_path(artifact.envelope_path, "envelope path")


def _preflight_path(path: Path, context: str) -> None:
    raw = os.fspath(path)
    if not isinstance(raw, str):
        raise TypeError(f"{context} must be a text path")
    if "\x00" in raw:
        raise ValueError(f"{context} denied: NUL byte")
    normalized = _normalized_path(path)
    _reject_denied_path_text(raw, context)
    _reject_denied_path_text(normalized, context)
    _reject_symlink_components(normalized, context)
    try:
        resolved = str(Path(normalized).resolve(strict=False))
    except OSError as exc:
        raise ValueError(f"{context} denied: cannot resolve safely") from exc
    _reject_denied_path_text(resolved, context)


def _normalized_path(path: Path) -> str:
    return os.path.abspath(os.path.normpath(os.fspath(path)))


def _reject_denied_path_text(value: str, context: str) -> None:
    lowered = value.lower()
    if _CANONICAL_FORMAL_TEST_RECORD_SHA256 in lowered:
        raise ValueError(f"{context} denied: canonical formal-test digest")
    path = Path(value)
    if _SUBJECT_TEST_FILENAME_RE.fullmatch(path.name):
        raise ValueError(f"{context} denied: formal-test subject filename")
    if any(part.lower() == "test_images" for part in path.parts):
        raise ValueError(f"{context} denied: test_images path component")


def _reject_symlink_components(path: str, context: str) -> None:
    current = Path(path).anchor
    for component in Path(path).parts:
        if component == Path(path).anchor:
            continue
        current = os.path.join(current, component)
        try:
            component_stat = os.lstat(current)
        except FileNotFoundError:
            return
        except OSError as exc:
            raise ValueError(
                f"{context} denied: cannot inspect path component"
            ) from exc
        if stat.S_ISLNK(component_stat.st_mode):
            raise ValueError(f"{context} denied: symlink component")


def _open_readonly_nofollow(path: Path, context: str) -> int:
    normalized = _normalized_path(path)
    components = Path(normalized).parts
    if len(components) <= 1:
        raise ValueError(f"{context} must name a regular file")
    directory_fd = os.open(
        Path(normalized).anchor,
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
    except OSError as exc:
        raise ValueError(f"{context} could not be opened securely") from exc
    finally:
        os.close(directory_fd)


def _read_strict_envelope(
    path: Path,
) -> tuple[dict[str, object], str, os.stat_result]:
    fd = _open_readonly_nofollow(path, "artifact envelope")
    try:
        before = os.fstat(fd)
        _require_regular_file(before, "artifact envelope")
        raw = _read_limited(fd, _MAX_ENVELOPE_BYTES)
        after = os.fstat(fd)
        _require_stable_identity(before, after, "artifact envelope")
    finally:
        os.close(fd)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("artifact envelope is not valid UTF-8") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite_json,
        )
    except json.JSONDecodeError as exc:
        raise ValueError("artifact envelope is malformed JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("artifact envelope must be a JSON object")
    return value, hashlib.sha256(raw).hexdigest(), after


def _read_limited(fd: int, limit: int) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = os.read(fd, min(1024 * 1024, limit + 1 - size))
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
        size += len(chunk)
        if size > limit:
            raise ValueError("artifact envelope exceeds the size limit")


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_non_finite_json(value: str) -> object:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def _verify_generic_envelope(
    artifact: TypedArtifact,
    envelope: Mapping[str, object],
    requested_scope: AccessScope,
) -> str:
    _strict_keys(envelope, _GENERIC_ENVELOPE_KEYS, "artifact envelope")
    _require_schema_one(envelope["schema_version"], "artifact envelope")
    payload_type = _require_string(
        envelope["payload_type"], "artifact envelope payload_type"
    )
    if payload_type not in _GENERIC_PAYLOAD_TYPES:
        raise ValueError(f"unrecognized payload type: {payload_type!r}")
    if payload_type != artifact.payload_type:
        raise ValueError("descriptor and envelope payload types differ")
    if artifact.role is not None:
        raise ValueError("generic typed artifacts must not declare a role")

    artifact_scope = _require_access_scope(envelope["scope"])
    _validate_scope_dependency(requested_scope, artifact_scope)
    if requested_scope == "formal-input" and any(
        term in payload_type.lower()
        for term in _FORMAL_INPUT_FORBIDDEN_TYPE_TERMS
    ):
        raise PermissionError(
            f"formal-input forbids payload type {payload_type!r}"
        )

    source_records_sha256 = _require_sha256(
        envelope["source_records_sha256"], "source_records_sha256"
    )
    declared_ordered_ids_sha256 = _require_sha256(
        envelope["ordered_ids_sha256"], "ordered_ids_sha256"
    )
    payload_sha256 = _require_sha256(
        envelope["payload_sha256"], "payload_sha256"
    )
    provenance_sha256 = _require_sha256(
        envelope["provenance_sha256"], "provenance_sha256"
    )
    metadata_sha256 = _require_sha256(
        envelope["metadata_sha256"], "metadata_sha256"
    )

    provenance = _require_mapping(envelope["provenance"], "provenance")
    metadata = _require_mapping(envelope["metadata"], "metadata")
    if sha256_json(provenance) != provenance_sha256:
        raise ValueError("provenance SHA-256 mismatch")
    if sha256_json(metadata) != metadata_sha256:
        raise ValueError("metadata SHA-256 mismatch")

    if "source_records" not in metadata:
        raise ValueError("metadata is missing source_records")
    source_records = metadata["source_records"]
    if not isinstance(source_records, list):
        raise ValueError("metadata source_records must be an array")
    if sha256_json(source_records) != source_records_sha256:
        raise ValueError("source-record SHA-256 mismatch")

    if "ordered_ids" not in metadata:
        raise ValueError("metadata is missing ordered_ids")
    ordered_ids = metadata["ordered_ids"]
    if not isinstance(ordered_ids, list) or any(
        not isinstance(value, str) for value in ordered_ids
    ):
        raise ValueError("metadata ordered_ids must be an array of strings")
    if ordered_ids_sha256(ordered_ids) != declared_ordered_ids_sha256:
        raise ValueError("ordered-ID SHA-256 mismatch")

    _reject_denied_metadata(
        provenance,
        requested_scope=requested_scope,
        artifact_scope=artifact_scope,
        context="provenance",
    )
    _reject_denied_metadata(
        metadata,
        requested_scope=requested_scope,
        artifact_scope=artifact_scope,
        context="metadata",
    )
    return payload_sha256


def _verify_protocol_role_envelope(
    artifact: TypedArtifact,
    envelope: Mapping[str, object],
    requested_scope: AccessScope,
) -> None:
    _strict_keys(envelope, _PROTOCOL_TOP_KEYS, "subject protocol manifest")
    _require_schema_one(
        envelope["schema_version"], "subject protocol manifest"
    )
    if envelope["payload_type"] != _SUBJECT_PROTOCOL_TYPE:
        raise ValueError("unexpected subject protocol payload_type")
    if artifact.payload_type != _ROLE_PAYLOAD_TYPE:
        raise ValueError(
            "Task 2 protocol descriptors must request the role payload type"
        )
    role = artifact.role
    if role not in _PROTOCOL_ROLES:
        raise ValueError("Task 2 protocol descriptors require a known role")
    _validate_scope_dependency(requested_scope, role)

    records_sha256 = _require_sha256(
        envelope["records_sha256"], "protocol records_sha256"
    )
    protocol_config_sha256 = _require_sha256(
        envelope["protocol_config_sha256"],
        "protocol protocol_config_sha256",
    )
    source_manifest_sha256 = _require_sha256(
        envelope["source_manifest_sha256"],
        "protocol source_manifest_sha256",
    )
    split_assignment_sha256 = _require_sha256(
        envelope["split_assignment_payload_sha256"],
        "protocol split_assignment_payload_sha256",
    )
    source_manifest_path = _require_string(
        envelope["source_manifest_path"], "protocol source_manifest_path"
    )
    _reject_denied_path_text(source_manifest_path, "source manifest path")

    split_assignment = _require_mapping(
        envelope["split_assignment"], "protocol split_assignment"
    )
    if sha256_json(split_assignment) != split_assignment_sha256:
        raise ValueError("split-assignment SHA-256 mismatch")
    if split_assignment.get("records_sha256") != records_sha256:
        raise ValueError("split assignment records binding mismatch")
    if (
        split_assignment.get("protocol_config_sha256")
        != protocol_config_sha256
    ):
        raise ValueError("split assignment protocol-config binding mismatch")

    role_artifacts = _require_mapping(
        envelope["role_artifacts"], "protocol role_artifacts"
    )
    role_payloads = _require_mapping(
        envelope["role_payloads"], "protocol role_payloads"
    )
    _strict_keys(role_artifacts, _PROTOCOL_ROLES, "protocol role_artifacts")
    _strict_keys(role_payloads, _PROTOCOL_ROLES, "protocol role_payloads")

    descriptor = _require_mapping(
        role_artifacts[role], f"protocol {role} role descriptor"
    )
    payload = _require_mapping(
        role_payloads[role], f"protocol {role} role payload"
    )
    _strict_keys(
        descriptor,
        _ROLE_DESCRIPTOR_KEYS,
        f"protocol {role} role descriptor",
    )
    _strict_keys(
        payload,
        _ROLE_PAYLOAD_KEYS,
        f"protocol {role} role payload",
    )
    _require_schema_one(
        descriptor["schema_version"], f"protocol {role} role descriptor"
    )
    _require_schema_one(
        payload["schema_version"], f"protocol {role} role payload"
    )
    if descriptor["payload_type"] != _ROLE_PAYLOAD_TYPE:
        raise ValueError("role descriptor payload_type mismatch")
    if payload["payload_type"] != _ROLE_PAYLOAD_TYPE:
        raise ValueError("role payload payload_type mismatch")
    if descriptor["scope"] != role or payload["scope"] != role:
        raise ValueError("selected Task 2 role scope mismatch")

    if (
        _require_sha256(
            descriptor["source_records_sha256"],
            "role source_records_sha256",
        )
        != records_sha256
    ):
        raise ValueError("role source-record binding mismatch")
    provenance = {
        "protocol_config_sha256": protocol_config_sha256,
        "source_manifest_sha256": source_manifest_sha256,
    }
    if _require_sha256(
        descriptor["provenance_sha256"], "role provenance_sha256"
    ) != sha256_json(provenance):
        raise ValueError("role provenance SHA-256 mismatch")

    ordered_ids = payload["ordered_ids"]
    if not isinstance(ordered_ids, list) or any(
        not isinstance(value, str) for value in ordered_ids
    ):
        raise ValueError("role ordered_ids must be an array of strings")
    if _require_sha256(
        descriptor["ordered_ids_sha256"], "role ordered_ids_sha256"
    ) != ordered_ids_sha256(ordered_ids):
        raise ValueError("role ordered-ID SHA-256 mismatch")
    if _require_sha256(
        descriptor["role_payload_sha256"], "role payload SHA-256"
    ) != sha256_json(payload):
        raise ValueError("role payload SHA-256 mismatch")

    _validate_role_payload_shape(payload, role)
    _reject_denied_metadata(
        payload,
        requested_scope=requested_scope,
        artifact_scope=role,  # type: ignore[arg-type]
        context=f"selected {role} role payload",
    )
    return None


def _validate_role_payload_shape(
    payload: Mapping[str, object],
    role: str,
) -> None:
    for key in ("concept_ids", "gallery_ids", "query_ids"):
        value = payload[key]
        if not isinstance(value, list) or any(
            not isinstance(item, str) for item in value
        ):
            raise ValueError(f"{role} role {key} must be an array of strings")
    row_indices = payload["row_indices"]
    if not isinstance(row_indices, list) or any(
        type(item) is not int or item < 0 for item in row_indices
    ):
        raise ValueError(f"{role} role row_indices must be non-negative integers")
    concept_count = payload["concept_count"]
    row_count = payload["row_count"]
    if type(concept_count) is not int or concept_count < 0:
        raise ValueError(f"{role} role concept_count must be non-negative")
    if type(row_count) is not int or row_count < 0:
        raise ValueError(f"{role} role row_count must be non-negative")
    if concept_count != len(payload["concept_ids"]):  # type: ignore[arg-type]
        raise ValueError(f"{role} role concept_count mismatch")
    if row_count != len(row_indices):
        raise ValueError(f"{role} role row_count mismatch")


def _validate_scope_dependency(
    requested_scope: AccessScope,
    artifact_scope: AccessScope | str,
) -> None:
    if artifact_scope not in _SCOPE_INPUTS[requested_scope]:
        raise PermissionError(
            f"{requested_scope} cannot consume {artifact_scope!r} artifacts"
        )


def _reject_denied_metadata(
    value: object,
    *,
    requested_scope: AccessScope,
    artifact_scope: AccessScope,
    context: str,
    path: tuple[str, ...] = (),
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{context} contains a non-string key")
            lowered_key = _normalize_field_name(key)
            child_path = path + (lowered_key,)
            _reject_denied_scalar(
                key,
                requested_scope=requested_scope,
                artifact_scope=artifact_scope,
                context=context,
                field_path=child_path,
                is_key=True,
            )
            _reject_denied_metadata(
                child,
                requested_scope=requested_scope,
                artifact_scope=artifact_scope,
                context=context,
                path=child_path,
            )
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _reject_denied_metadata(
                child,
                requested_scope=requested_scope,
                artifact_scope=artifact_scope,
                context=context,
                path=path + (str(index),),
            )
        return
    _reject_denied_scalar(
        value,
        requested_scope=requested_scope,
        artifact_scope=artifact_scope,
        context=context,
        field_path=path,
        is_key=False,
    )


def _reject_denied_scalar(
    value: object,
    *,
    requested_scope: AccessScope,
    artifact_scope: AccessScope,
    context: str,
    field_path: tuple[str, ...],
    is_key: bool,
) -> None:
    if not isinstance(value, str):
        return
    lowered = value.lower()
    if _CANONICAL_FORMAL_TEST_RECORD_SHA256 in lowered:
        raise ValueError(f"{context} denied: canonical formal-test digest")
    if _has_named_path_component(value, "test_images"):
        raise ValueError(f"{context} denied: test_images provenance")

    if (
        not is_key
        and field_path
        and field_path[-1] in {"scope", "split", "subset"}
        and _normalize_field_name(value) in {"test", "formal_test"}
        and not (
            requested_scope == "formal-test"
            and artifact_scope == "formal-test"
        )
    ):
        raise PermissionError(
            f"{context} denied: unauthorized test split declaration"
        )

    normalized_value = _normalize_field_name(value)
    semantic_path = field_path + (() if is_key else (normalized_value,))
    if requested_scope == "formal-input":
        forbidden_formal_input_terms = (
            "eeg",
            "similarity",
            "similarities",
            "score",
            "scores",
            "prediction",
            "predictions",
            "metric",
            "metrics",
        )
        if any(
            any(term in component for term in forbidden_formal_input_terms)
            for component in semantic_path
        ):
            raise PermissionError(
                f"{context} denied: formal-input output semantics"
            )
    has_test_context = any(
        component == "test"
        or component == "formal_test"
        or component.startswith("test_")
        or component.endswith("_test")
        for component in semantic_path
    )
    sensitive_terms = (
        "metric",
        "score",
        "prediction",
        "rank",
        "top1",
        "top_1",
        "top5",
        "top_5",
        "error_analysis",
        "best_test",
    )
    has_sensitive_output = any(
        any(term in component for term in sensitive_terms)
        for component in semantic_path
    )
    if has_test_context and has_sensitive_output:
        raise PermissionError(
            f"{context} denied: checkpoint contains test-derived outputs"
        )


def _normalize_field_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _has_named_path_component(value: str, component: str) -> bool:
    parts = re.split(r"[\\/]+", value)
    return any(part.lower() == component for part in parts)


def _verify_payload_file(
    path: Path,
    expected_sha256: str,
    *,
    expected_identity: os.stat_result | None,
) -> tuple[os.stat_result, str]:
    fd = _open_readonly_nofollow(path, "artifact payload")
    try:
        before = os.fstat(fd)
        _require_regular_file(before, "artifact payload")
        if expected_identity is not None:
            _require_same_identity(
                expected_identity,
                before,
                "protocol envelope and payload",
            )
        digest = _sha256_fd(fd)
        after = os.fstat(fd)
        _require_stable_identity(before, after, "artifact payload")
        if expected_identity is not None:
            _require_same_identity(
                expected_identity,
                after,
                "protocol envelope and payload",
            )
    finally:
        os.close(fd)
    if digest != expected_sha256:
        raise ValueError("payload SHA-256 mismatch")
    return after, digest


def _sha256_fd(fd: int) -> str:
    os.lseek(fd, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            return digest.hexdigest()
        digest.update(chunk)


def _require_regular_file(value: os.stat_result, context: str) -> None:
    if not stat.S_ISREG(value.st_mode):
        raise ValueError(f"{context} must be a regular file")


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )

def _require_stable_identity(
    before: os.stat_result,
    after: os.stat_result,
    context: str,
) -> None:
    if _identity(before) != _identity(after):
        raise ValueError(f"{context} changed while it was being read")


def _require_same_identity(
    expected: os.stat_result,
    actual: os.stat_result,
    context: str,
) -> None:
    if _identity(expected) != _identity(actual):
        raise ValueError(f"{context} file identity mismatch")


def _strict_keys(
    value: Mapping[str, object],
    expected: frozenset[str],
    context: str,
) -> None:
    actual = set(value)
    missing = expected - actual
    unknown = actual - expected
    if missing:
        raise ValueError(f"{context} is missing keys: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{context} has unknown keys: {sorted(unknown)}")


def _require_schema_one(value: object, context: str) -> None:
    if type(value) is not int or value != 1:
        raise ValueError(f"{context} schema_version must be 1")


def _require_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _require_sha256(value: object, context: str) -> str:
    digest = _require_string(value, context)
    if _SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return digest


def _require_mapping(
    value: object,
    context: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a JSON object")
    return value


__all__ = [
    "AccessAuthorization",
    "AccessScope",
    "TypedArtifact",
    "VerifiedArtifact",
    "require_typed_artifacts",
    "verify_typed_artifacts",
]
