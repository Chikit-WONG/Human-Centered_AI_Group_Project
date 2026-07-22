"""Exact semantic identity binding for development epoch checkpoints."""

from __future__ import annotations

import re
from collections.abc import Mapping

from .hashing import ordered_ids_sha256, sha256_json


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_RE = re.compile(r"^[0-9a-f]{40}$")
_UPSTREAM_SHA = "1a63745b7ff6f98dad34b0f0b8246a9b5260d9c1"
_INPUT_HASH_KEYS = frozenset(
    {
        "cache_sha256",
        "checkpoint_sha256",
        "manifest_sha256",
        "model_sha256",
        "ordered_ids_sha256",
        "protocol_sha256",
        "records_sha256",
        "source_manifest_sha256",
        "source_payload_byte_count_sha256",
        "source_payload_path_sha256",
        "source_payload_sha256",
        "train_ordered_ids_sha256",
        "train_role_sha256",
        "val_dev_ordered_ids_sha256",
        "val_dev_role_sha256",
    }
)
_RUN_MANIFEST_KEYS = frozenset(
    {
        "cache_sha256",
        "candidate_spec_sha256",
        "config_id",
        "config_sha256",
        "data_order_sha256",
        "git_sha",
        "payload_type",
        "protocol_sha256",
        "run_key",
        "run_manifest_sha256",
        "schema_version",
        "seed",
        "stage",
        "subject",
        "upstream_sha",
    }
)
_CANDIDATE_SPEC_KEYS = frozenset(
    {
        "adapter_kind",
        "adapter_lr_ratio",
        "adapter_rank",
        "architecture_specific_initialization_sha256",
        "baseline_config_sha256",
        "candidate_spec_sha256",
        "config_id",
        "data_order_sha256",
        "full_task_initialization_sha256",
        "input_bundle_sha256",
        "layernorm_config_id",
        "preprojector_config_id",
        "run_key",
        "schema_version",
        "seed",
        "semantic_config_sha256",
        "shared_parameter_intersection_name",
        "shared_parameter_intersection_sha256",
        "stage",
        "stage2_config_sha256",
        "subject",
        "trajectory_sha256",
        "whitening_config_id",
        "whitening_payload",
    }
)
_PROVENANCE_KEYS = frozenset(
    {"config_sha256", "manifest_sha256", "protocol_sha256", "seed", "subject"}
)
_METADATA_KEYS = frozenset(
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


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise ValueError(f"{context} must be a string-keyed mapping")
    return value


def _exact_keys(
    value: Mapping[str, object],
    expected: frozenset[str],
    context: str,
) -> None:
    if set(value) != expected:
        raise ValueError(f"{context} schema mismatch")


def _sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _input_hashes(payload: Mapping[str, object]) -> Mapping[str, object]:
    value = _mapping(payload.get("input_hashes"), "checkpoint input_hashes")
    _exact_keys(value, _INPUT_HASH_KEYS, "checkpoint input_hashes")
    for key, digest in value.items():
        _sha256(digest, f"checkpoint input_hashes.{key}")
    return value


def _run_manifest(
    payload: Mapping[str, object],
    input_hashes: Mapping[str, object],
) -> Mapping[str, object]:
    value = _mapping(payload.get("run_manifest"), "checkpoint run_manifest")
    _exact_keys(value, _RUN_MANIFEST_KEYS, "checkpoint run_manifest")
    claimed_hash = _sha256(
        value["run_manifest_sha256"],
        "checkpoint run_manifest hash",
    )
    body = {
        key: child
        for key, child in value.items()
        if key != "run_manifest_sha256"
    }
    if sha256_json(body) != claimed_hash:
        raise ValueError("checkpoint run_manifest hash mismatch")
    if type(value["schema_version"]) is not int:
        raise ValueError(
            "checkpoint run_manifest schema_version must be an integer"
        )
    if type(value["stage"]) is not int:
        raise ValueError("checkpoint run_manifest stage must be an integer")
    if (
        type(value["subject"]) is not int
        or not 1 <= value["subject"] <= 10
    ):
        raise ValueError("checkpoint run_manifest subject is invalid")
    if type(value["seed"]) is not int or value["seed"] < 0:
        raise ValueError("checkpoint run_manifest seed is invalid")
    if (
        value["schema_version"] != 1
        or value["payload_type"] != "samga_brain_rw.development_run"
        or value["stage"] not in {0, 2}
        or not isinstance(value["config_id"], str)
        or not value["config_id"]
        or not isinstance(value["run_key"], str)
        or not value["run_key"]
        or not isinstance(value["git_sha"], str)
        or _GIT_RE.fullmatch(value["git_sha"]) is None
        or value["upstream_sha"] != _UPSTREAM_SHA
    ):
        raise ValueError("checkpoint run_manifest identity mismatch")
    for key in (
        "cache_sha256",
        "candidate_spec_sha256",
        "config_sha256",
        "data_order_sha256",
        "protocol_sha256",
    ):
        _sha256(value[key], f"checkpoint run_manifest.{key}")
    expected = {
        "cache_sha256": input_hashes["cache_sha256"],
        "config_sha256": payload["config_sha256"],
        "data_order_sha256": payload["data_order_sha256"],
        "protocol_sha256": input_hashes["protocol_sha256"],
        "seed": payload["seed"],
        "subject": payload["subject"],
    }
    for key, expected_value in expected.items():
        if value[key] != expected_value:
            raise ValueError(f"checkpoint run_manifest {key} binding mismatch")
    return value


def _candidate_spec(
    payload: Mapping[str, object],
    input_hashes: Mapping[str, object],
    run_manifest: Mapping[str, object],
) -> Mapping[str, object]:
    value = _mapping(
        payload.get("candidate_spec"),
        "checkpoint candidate_spec",
    )
    _exact_keys(value, _CANDIDATE_SPEC_KEYS, "checkpoint candidate_spec")
    claimed_hash = _sha256(
        value["candidate_spec_sha256"],
        "checkpoint candidate_spec hash",
    )
    body = {
        key: child
        for key, child in value.items()
        if key != "candidate_spec_sha256"
    }
    if sha256_json(body) != claimed_hash:
        raise ValueError("checkpoint candidate_spec hash mismatch")
    if type(value["schema_version"]) is not int:
        raise ValueError(
            "checkpoint candidate_spec schema_version must be an integer"
        )
    if type(value["subject"]) is not int:
        raise ValueError("checkpoint candidate_spec subject must be an integer")
    if type(value["seed"]) is not int:
        raise ValueError("checkpoint candidate_spec seed must be an integer")
    expected = {
        "schema_version": 1,
        "config_id": run_manifest["config_id"],
        "stage": f"stage{run_manifest['stage']}",
        "subject": payload["subject"],
        "seed": payload["seed"],
        "semantic_config_sha256": payload["config_sha256"],
        "input_bundle_sha256": sha256_json(
            dict(sorted(input_hashes.items()))
        ),
        "run_key": run_manifest["run_key"],
        "data_order_sha256": payload["data_order_sha256"],
        "trajectory_sha256": payload["trajectory_sha256"],
    }
    for key, expected_value in expected.items():
        if value[key] != expected_value:
            raise ValueError(
                f"checkpoint candidate_spec {key} binding mismatch"
            )
    expected_run_key = (
        f"stage{run_manifest['stage']}__{run_manifest['config_id']}__"
        f"sub-{payload['subject']:02d}__seed-{payload['seed']}__"
        f"config-{payload['config_sha256']}__"
        f"inputs-{value['input_bundle_sha256']}"
    )
    if value["run_key"] != expected_run_key:
        raise ValueError(
            "checkpoint candidate_spec run_key is not canonical"
        )
    if run_manifest["candidate_spec_sha256"] != claimed_hash:
        raise ValueError(
            "checkpoint run_manifest candidate_spec_sha256 binding mismatch"
        )
    for key in (
        "baseline_config_sha256",
        "full_task_initialization_sha256",
        "semantic_config_sha256",
        "shared_parameter_intersection_sha256",
        "architecture_specific_initialization_sha256",
        "data_order_sha256",
        "input_bundle_sha256",
        "trajectory_sha256",
    ):
        _sha256(value[key], f"checkpoint candidate_spec.{key}")
    stage2_config = value["stage2_config_sha256"]
    if run_manifest["stage"] == 0:
        if stage2_config is not None:
            raise ValueError(
                "checkpoint candidate_spec Stage 0 stage2 config mismatch"
            )
    else:
        _sha256(
            stage2_config,
            "checkpoint candidate_spec.stage2_config_sha256",
        )
    shared_name = value["shared_parameter_intersection_name"]
    if not isinstance(shared_name, str) or not shared_name:
        raise ValueError(
            "checkpoint candidate_spec shared parameter name is invalid"
        )
    return value


def validate_epoch_checkpoint_identity(
    payload_value: object,
    envelope_value: object,
) -> None:
    """Reject a transport-valid pair whose scientific identities conflict."""

    payload = _mapping(payload_value, "checkpoint payload")
    envelope = _mapping(envelope_value, "checkpoint sidecar")
    input_hashes = _input_hashes(payload)
    run_manifest = _run_manifest(payload, input_hashes)
    _candidate_spec(payload, input_hashes, run_manifest)
    observation_keys = {
        "scope",
        "validation_scope",
        "observed_scopes",
    }
    present_observation_keys = observation_keys.intersection(payload)
    if not present_observation_keys:
        train_only = False
    else:
        if present_observation_keys != observation_keys:
            raise PermissionError(
                "checkpoint observation policy is incomplete"
            )
        if payload["scope"] != "train":
            raise PermissionError("checkpoint scope must be train")
        observation = (
            payload["validation_scope"],
            tuple(payload["observed_scopes"]),
        )
        if observation == ("val-dev", ("train", "val-dev")):
            train_only = False
        elif observation == ("none", ("train",)):
            train_only = True
        else:
            raise PermissionError(
                "checkpoint observation policy is invalid"
            )

    provenance = _mapping(
        envelope.get("provenance"),
        "checkpoint sidecar provenance",
    )
    _exact_keys(provenance, _PROVENANCE_KEYS, "checkpoint sidecar provenance")
    expected_provenance = {
        "config_sha256": payload["config_sha256"],
        "manifest_sha256": input_hashes["manifest_sha256"],
        "protocol_sha256": input_hashes["protocol_sha256"],
        "seed": payload["seed"],
        "subject": payload["subject"],
    }
    for key, expected in expected_provenance.items():
        if provenance[key] != expected:
            raise ValueError(
                f"checkpoint sidecar provenance {key} binding mismatch"
            )

    metadata = _mapping(
        envelope.get("metadata"),
        "checkpoint sidecar metadata",
    )
    _exact_keys(metadata, _METADATA_KEYS, "checkpoint sidecar metadata")
    expected_scopes = ["train"] if train_only else ["train", "val-dev"]
    if metadata["observed_scopes"] != expected_scopes:
        raise ValueError(
            "checkpoint sidecar observed-scopes binding mismatch"
        )
    ordered_ids = metadata["ordered_ids"]
    if not isinstance(ordered_ids, list) or any(
        not isinstance(value, str) for value in ordered_ids
    ):
        raise ValueError("checkpoint sidecar ordered_ids are invalid")
    train_ordered_ids = metadata["train_ordered_ids"]
    val_dev_ordered_ids = metadata["val_dev_ordered_ids"]
    for values, role in (
        (train_ordered_ids, "train"),
        (val_dev_ordered_ids, "val-dev"),
    ):
        if not isinstance(values, list) or any(
            not isinstance(value, str) for value in values
        ):
            raise ValueError(
                f"checkpoint sidecar {role} ordered IDs are invalid"
            )
        if not values and (role == "train" or not train_only):
            raise ValueError(
                f"checkpoint sidecar {role} ordered IDs must be nonempty"
            )
        if len(set(values)) != len(values):
            raise ValueError(
                f"checkpoint sidecar {role} ordered IDs must be unique"
            )
    if set(train_ordered_ids) & set(val_dev_ordered_ids):
        raise ValueError(
            "checkpoint sidecar ordered-ID partitions must be disjoint"
        )
    if ordered_ids != [*train_ordered_ids, *val_dev_ordered_ids]:
        raise ValueError(
            "checkpoint sidecar ordered-ID partition binding mismatch"
        )
    if (
        ordered_ids_sha256(ordered_ids)
        != input_hashes["ordered_ids_sha256"]
    ):
        raise ValueError("checkpoint sidecar ordered-ID binding mismatch")
    if (
        ordered_ids_sha256(train_ordered_ids)
        != input_hashes["train_ordered_ids_sha256"]
    ):
        raise ValueError(
            "checkpoint sidecar train ordered-ID binding mismatch"
        )
    if not train_only and (
        ordered_ids_sha256(val_dev_ordered_ids)
        != input_hashes["val_dev_ordered_ids_sha256"]
    ):
        raise ValueError(
            "checkpoint sidecar val-dev ordered-ID binding mismatch"
        )

    records = metadata["source_records"]
    expected_roles = ("train",) if train_only else ("train", "val-dev")
    if not isinstance(records, list) or len(records) != len(expected_roles):
        raise ValueError(
            "checkpoint sidecar source_records do not match observed scopes"
        )
    for index, role in enumerate(expected_roles):
        record = _mapping(
            records[index],
            f"checkpoint sidecar source record {index}",
        )
        _exact_keys(
            record,
            _SOURCE_RECORD_KEYS,
            f"checkpoint sidecar source record {index}",
        )
        expected_record = {
            "manifest_sha256": input_hashes["manifest_sha256"],
            "records_sha256": input_hashes["records_sha256"],
            "role": role,
            "role_payload_sha256": input_hashes[
                "train_role_sha256" if role == "train" else "val_dev_role_sha256"
            ],
            "source_manifest_sha256": input_hashes[
                "source_manifest_sha256"
            ],
            "source_payload_sha256": input_hashes["source_payload_sha256"],
        }
        for key, expected in expected_record.items():
            if record[key] != expected:
                raise ValueError(
                    f"checkpoint sidecar source record {key} binding mismatch"
                )


__all__ = ["validate_epoch_checkpoint_identity"]
