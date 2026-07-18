from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import samga_brain_rw.access as access_module

from samga_brain_rw.access import (
    AccessAuthorization,
    TypedArtifact,
    VerifiedArtifact,
    require_typed_artifacts,
    verify_typed_artifacts,
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_canonical_json_bytes(value))


def _generic_artifact(
    root: Path,
    *,
    scope: str = "train",
    payload_type: str = "feature-cache",
    payload_name: str = "features.bin",
    envelope_name: str = "features.envelope.json",
) -> tuple[TypedArtifact, dict[str, object]]:
    root.mkdir(parents=True, exist_ok=True)
    payload_path = root / payload_name
    payload_path.write_bytes(b"synthetic feature bytes")
    provenance = {
        "producer": "synthetic-unit-test",
        "source_components": ["tests", "fixtures"],
    }
    metadata = {
        "ordered_ids": ["stimulus-001", "stimulus-002"],
        "source_records": [
            {"record_id": "record-001"},
            {"record_id": "record-002"},
        ],
    }
    envelope: dict[str, object] = {
        "schema_version": 1,
        "payload_type": payload_type,
        "scope": scope,
        "source_records_sha256": _sha256_json(metadata["source_records"]),
        "ordered_ids_sha256": hashlib.sha256(
            b"stimulus-001\nstimulus-002"
        ).hexdigest(),
        "payload_sha256": hashlib.sha256(payload_path.read_bytes()).hexdigest(),
        "provenance": provenance,
        "provenance_sha256": _sha256_json(provenance),
        "metadata": metadata,
        "metadata_sha256": _sha256_json(metadata),
    }
    envelope_path = root / envelope_name
    _write_json(envelope_path, envelope)
    return (
        TypedArtifact(
            payload_type=payload_type,
            payload_path=payload_path,
            envelope_path=envelope_path,
        ),
        envelope,
    )


def test_train_artifact_returns_frozen_verified_capability(tmp_path: Path) -> None:
    artifact, _ = _generic_artifact(tmp_path)

    verified = verify_typed_artifacts("train", [artifact])

    assert isinstance(verified, tuple)
    assert len(verified) == 1
    capability = verified[0]
    assert isinstance(capability, VerifiedArtifact)
    assert capability.artifact == artifact
    stat = artifact.payload_path.stat()
    assert (
        capability.device,
        capability.inode,
        capability.size,
        capability.mtime_ns,
    ) == (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    assert capability.payload_sha256 == hashlib.sha256(
        artifact.payload_path.read_bytes()
    ).hexdigest()
    assert capability.revalidate() is None
    with pytest.raises(FrozenInstanceError):
        capability.size = 0  # type: ignore[misc]


def test_required_wrapper_delegates_and_returns_none(tmp_path: Path) -> None:
    artifact, _ = _generic_artifact(tmp_path)

    assert require_typed_artifacts("train", [artifact]) is None


@pytest.mark.parametrize(
    "payload_name",
    [
        "sub-01_test.json",
        (
            "features-"
            "02d7e33b3fe8e5a571f8db232ca5fa86abb0c16981876ec84feae7ba64636f1a"
            ".bin"
        ),
    ],
)
def test_lexical_denies_happen_before_envelope_open(
    tmp_path: Path,
    payload_name: str,
) -> None:
    artifact = TypedArtifact(
        payload_type="feature-cache",
        payload_path=tmp_path / payload_name,
        envelope_path=tmp_path / "missing-envelope.json",
    )

    with pytest.raises(ValueError, match="denied"):
        verify_typed_artifacts("train", [artifact])


def _rewrite_envelope(
    artifact: TypedArtifact,
    envelope: dict[str, object],
) -> None:
    _write_json(artifact.envelope_path, envelope)


@pytest.mark.parametrize("scope", ["val-confirm", "formal-test"])
def test_sensitive_scopes_fail_closed_without_issued_authorization(
    tmp_path: Path,
    scope: str,
) -> None:
    artifact, _ = _generic_artifact(tmp_path)

    with pytest.raises(PermissionError, match="authorization"):
        verify_typed_artifacts(scope, [artifact])  # type: ignore[arg-type]


def test_access_authorization_cannot_be_publicly_forged() -> None:
    digest = "0" * 64

    with pytest.raises(PermissionError, match="opaque"):
        AccessAuthorization(
            scope="val-confirm",
            seal_sha256=digest,
            job_map_sha256=digest,
            claim_sha256=digest,
        )


def test_sensitive_scope_rejects_even_object_new_forgery(
    tmp_path: Path,
) -> None:
    artifact, _ = _generic_artifact(tmp_path)
    forged = object.__new__(AccessAuthorization)
    object.__setattr__(forged, "scope", "val-confirm")
    object.__setattr__(forged, "seal_sha256", "0" * 64)
    object.__setattr__(forged, "job_map_sha256", "0" * 64)
    object.__setattr__(forged, "claim_sha256", "0" * 64)
    object.__setattr__(forged, "audit_sha256", None)

    with pytest.raises(PermissionError, match="generic authorization issuance"):
        verify_typed_artifacts(
            "val-confirm", [artifact], authorization=forged
        )


@pytest.mark.parametrize(
    "payload_type",
    [
        "samga_brain_rw.protocol_config",
        "samga_brain_rw.semantic_config",
        "samga_brain_rw.split_assignment",
        "samga_brain_rw.manifest_summary",
        "samga_brain_rw.source_manifest",
        "samga_brain_rw.source_train_pt",
        "samga_brain_rw.model_config",
        "samga_brain_rw.model_preprocessor",
        "samga_brain_rw.model_source",
        "samga_brain_rw.model_weights",
        "samga_brain_rw.train_cache",
        "samga_brain_rw.train_cache_metadata",
    ],
)
def test_task4_train_provenance_types_are_explicitly_allowed(
    tmp_path: Path,
    payload_type: str,
) -> None:
    artifact, _ = _generic_artifact(
        tmp_path / payload_type.rsplit(".", 1)[-1],
        payload_type=payload_type,
    )

    assert len(verify_typed_artifacts("train", [artifact])) == 1


def test_scope_allowlist_rejects_val_dev_from_train(tmp_path: Path) -> None:
    artifact, envelope = _generic_artifact(tmp_path)
    envelope["scope"] = "val-dev"
    _rewrite_envelope(artifact, envelope)

    with pytest.raises(PermissionError, match="cannot consume"):
        verify_typed_artifacts("train", [artifact])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda envelope: envelope.pop("metadata"), "missing keys"),
        (lambda envelope: envelope.update({"unbound": True}), "unknown keys"),
        (lambda envelope: envelope.update({"schema_version": 2}), "schema"),
        (
            lambda envelope: envelope.update({"payload_type": "unknown-cache"}),
            "unrecognized payload type",
        ),
    ],
)
def test_generic_envelope_is_exact_and_known(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    artifact, envelope = _generic_artifact(tmp_path)
    mutation(envelope)
    _rewrite_envelope(artifact, envelope)

    with pytest.raises(ValueError, match=message):
        verify_typed_artifacts("train", [artifact])


def test_generic_envelope_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    artifact, envelope = _generic_artifact(tmp_path)
    raw = _canonical_json_bytes(envelope).decode("utf-8")
    duplicated = raw.replace(
        '"scope":"train"',
        '"scope":"train","scope":"train"',
        1,
    )
    artifact.envelope_path.write_text(duplicated, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON key"):
        verify_typed_artifacts("train", [artifact])


@pytest.mark.parametrize(
    "mismatch",
    ["payload", "provenance", "metadata", "ordered_ids", "source_records"],
)
def test_generic_envelope_recomputes_every_bound_hash(
    tmp_path: Path,
    mismatch: str,
) -> None:
    artifact, envelope = _generic_artifact(tmp_path)
    if mismatch == "payload":
        artifact.payload_path.write_bytes(b"changed feature payload")
    elif mismatch == "provenance":
        envelope["provenance"]["producer"] = "tampered"  # type: ignore[index]
    elif mismatch == "metadata":
        envelope["metadata"]["bound_note"] = "tampered"  # type: ignore[index]
    elif mismatch == "ordered_ids":
        envelope["metadata"]["ordered_ids"].reverse()  # type: ignore[index,union-attr]
        envelope["metadata_sha256"] = _sha256_json(envelope["metadata"])
    else:
        envelope["metadata"]["source_records"].reverse()  # type: ignore[index,union-attr]
        envelope["metadata_sha256"] = _sha256_json(envelope["metadata"])
    _rewrite_envelope(artifact, envelope)

    with pytest.raises(ValueError, match="mismatch"):
        verify_typed_artifacts("train", [artifact])


@pytest.mark.parametrize(
    "forbidden_metadata",
    [
        {"nested": {"split": "test"}},
        {"nested": {"source_manifest": "/sealed/sub-01_test.json"}},
        {
            "nested": {
                "digest": (
                    "02d7e33b3fe8e5a571f8db232ca5fa86abb0c16981876ec84"
                    "feae7ba64636f1a"
                )
            }
        },
        {"nested": {"image_path": "/sealed/test_images/image-001.jpg"}},
        {"checkpoint": {"metrics": {"test": {"top1": 0.8}}}},
    ],
)
def test_nested_metadata_denies_test_material(
    tmp_path: Path,
    forbidden_metadata: dict[str, object],
) -> None:
    artifact, envelope = _generic_artifact(tmp_path)
    envelope["metadata"].update(forbidden_metadata)  # type: ignore[union-attr]
    envelope["metadata_sha256"] = _sha256_json(envelope["metadata"])
    _rewrite_envelope(artifact, envelope)

    with pytest.raises((ValueError, PermissionError), match="denied"):
        verify_typed_artifacts("train", [artifact])


def test_symlinked_path_component_is_rejected(tmp_path: Path) -> None:
    real_root = tmp_path / "real"
    artifact, _ = _generic_artifact(real_root)
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(real_root, target_is_directory=True)
    linked = TypedArtifact(
        payload_type=artifact.payload_type,
        payload_path=linked_root / artifact.payload_path.name,
        envelope_path=linked_root / artifact.envelope_path.name,
    )

    with pytest.raises(ValueError, match="symlink component"):
        verify_typed_artifacts("train", [linked])


def test_open_verified_rechecks_hash_on_the_same_identity(
    tmp_path: Path,
) -> None:
    artifact, _ = _generic_artifact(tmp_path)
    capability = verify_typed_artifacts("train", [artifact])[0]
    original = artifact.payload_path.read_bytes()
    original_stat = artifact.payload_path.stat()
    artifact.payload_path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    os.utime(
        artifact.payload_path,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )

    with pytest.raises(ValueError, match="identity changed|digest mismatch"):
        with capability.open_verified():
            pytest.fail("mutated payload must not be yielded")


def test_open_verified_rejects_path_replacement(tmp_path: Path) -> None:
    artifact, _ = _generic_artifact(tmp_path)
    capability = verify_typed_artifacts("train", [artifact])[0]
    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"synthetic feature bytes")
    artifact.payload_path.unlink()
    os.link(replacement, artifact.payload_path)

    with pytest.raises(ValueError, match="identity changed"):
        with capability.open_verified():
            pytest.fail("replacement inode must not be yielded")


def test_rejection_never_invokes_semantic_loader(tmp_path: Path) -> None:
    artifact, _ = _generic_artifact(tmp_path)
    artifact.payload_path.write_bytes(b"tampered before verification")
    load_count = 0

    def loader(payload_file) -> bytes:
        nonlocal load_count
        load_count += 1
        return payload_file.read()

    def guarded_load() -> list[bytes]:
        loaded: list[bytes] = []
        for capability in verify_typed_artifacts("train", [artifact]):
            with capability.open_verified() as payload_file:
                loaded.append(loader(payload_file))
        return loaded

    with pytest.raises(ValueError, match="payload SHA-256 mismatch"):
        guarded_load()
    assert load_count == 0


def _role_payload(role: str, token: str) -> dict[str, object]:
    validation = role != "train"
    return {
        "concept_count": 1,
        "concept_ids": [f"concept-{token}"],
        "gallery_ids": [f"stimulus-{token}"] if validation else [],
        "ordered_ids": [
            f"stimulus-{token}" if validation else f"concept-{token}"
        ],
        "payload_type": "samga_brain_rw.role_payload",
        "query_ids": [f"stimulus-{token}"] if validation else [],
        "row_count": 1,
        "row_indices": [0],
        "schema_version": 1,
        "scope": role,
    }


def _task2_protocol_artifact(
    root: Path,
) -> tuple[TypedArtifact, dict[str, object]]:
    records_sha256 = "1" * 64
    protocol_config_sha256 = "2" * 64
    source_manifest_sha256 = "3" * 64
    provenance = {
        "protocol_config_sha256": protocol_config_sha256,
        "source_manifest_sha256": source_manifest_sha256,
    }
    role_payloads = {
        role: _role_payload(role, token)
        for role, token in (
            ("train", "train"),
            ("val-dev", "dev"),
            ("val-confirm", "confirm"),
        )
    }
    role_artifacts = {
        role: {
            "ordered_ids_sha256": hashlib.sha256(
                "\n".join(payload["ordered_ids"]).encode("utf-8")  # type: ignore[arg-type]
            ).hexdigest(),
            "payload_type": "samga_brain_rw.role_payload",
            "provenance_sha256": _sha256_json(provenance),
            "role_payload_sha256": _sha256_json(payload),
            "schema_version": 1,
            "scope": role,
            "source_records_sha256": records_sha256,
        }
        for role, payload in role_payloads.items()
    }
    split_assignment = {
        "payload_type": "samga_brain_rw.split_assignment",
        "protocol_config_sha256": protocol_config_sha256,
        "records_sha256": records_sha256,
        "schema_version": 1,
    }
    manifest: dict[str, object] = {
        "payload_type": "samga_brain_rw.subject_protocol_manifest",
        "protocol_config_sha256": protocol_config_sha256,
        "records_sha256": records_sha256,
        "role_artifacts": role_artifacts,
        "role_payloads": role_payloads,
        "schema_version": 1,
        "source_manifest_path": "sub-01_train.json",
        "source_manifest_sha256": source_manifest_sha256,
        "split_assignment": split_assignment,
        "split_assignment_payload_sha256": _sha256_json(split_assignment),
        "subject_id": 1,
    }
    path = root / "sub-01_protocol.json"
    _write_json(path, manifest)
    return (
        TypedArtifact(
            payload_type="samga_brain_rw.role_payload",
            payload_path=path,
            envelope_path=path,
            role="train",
        ),
        manifest,
    )


def test_task2_validates_only_selected_role_and_recomputes_it(
    tmp_path: Path,
) -> None:
    artifact, manifest = _task2_protocol_artifact(tmp_path)
    sibling_payload = manifest["role_payloads"]["val-confirm"]  # type: ignore[index]
    sibling_payload["scope"] = "formal-test"  # type: ignore[index]
    sibling_payload["test_metrics"] = {"top1": 1.0}  # type: ignore[index]
    manifest["role_artifacts"]["val-confirm"]["role_payload_sha256"] = "invalid"  # type: ignore[index]
    artifact.envelope_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    verified = verify_typed_artifacts("train", [artifact])

    assert len(verified) == 1
    tampered = deepcopy(manifest)
    tampered["role_payloads"]["train"]["ordered_ids"] = ["changed"]  # type: ignore[index]
    _write_json(artifact.envelope_path, tampered)
    with pytest.raises(ValueError, match="ordered-ID|payload SHA-256"):
        verify_typed_artifacts("train", [artifact])


@pytest.mark.parametrize(
    "forbidden_metadata",
    [
        {"eeg_shape": [1, 2, 3]},
        {"similarity_cache": "cache.bin"},
        {"score_matrix": "scores.npy"},
        {"predictions": ["concept-001"]},
        {"metrics": {"top1": 0.5}},
    ],
)
def test_formal_input_rejects_output_semantics_in_bound_metadata(
    forbidden_metadata: dict[str, object],
) -> None:
    with pytest.raises(PermissionError, match="formal-input output semantics"):
        access_module._reject_denied_metadata(
            forbidden_metadata,
            requested_scope="formal-input",
            artifact_scope="formal-input",
            context="metadata",
        )
