from __future__ import annotations

import hashlib

import pytest

from samga_brain_rw.checkpoint_identity import (
    validate_epoch_checkpoint_identity,
)
from samga_brain_rw.hashing import ordered_ids_sha256, sha256_json


def _h(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _valid_pair() -> tuple[dict[str, object], dict[str, object]]:
    train_ids = ["train:0", "train:1"]
    val_dev_ids = ["val-dev:0"]
    ordered_ids = [*train_ids, *val_dev_ids]
    input_hashes = {
        "cache_sha256": _h("cache"),
        "checkpoint_sha256": _h("initial-checkpoint"),
        "manifest_sha256": _h("manifest"),
        "model_sha256": _h("model"),
        "ordered_ids_sha256": ordered_ids_sha256(ordered_ids),
        "protocol_sha256": _h("protocol"),
        "records_sha256": _h("records"),
        "source_manifest_sha256": _h("source-manifest"),
        "source_payload_byte_count_sha256": _h("source-payload-bytes"),
        "source_payload_path_sha256": _h("source-payload-path"),
        "source_payload_sha256": _h("source-payload"),
        "train_ordered_ids_sha256": ordered_ids_sha256(train_ids),
        "train_role_sha256": _h("train-role"),
        "val_dev_ordered_ids_sha256": ordered_ids_sha256(val_dev_ids),
        "val_dev_role_sha256": _h("val-dev-role"),
    }
    input_bundle_sha256 = sha256_json(dict(sorted(input_hashes.items())))
    config_sha256 = _h("resolved-config")
    run_key = (
        "stage2__fixture__sub-01__seed-42__"
        f"config-{config_sha256}__inputs-{input_bundle_sha256}"
    )
    candidate_body = {
        "schema_version": 1,
        "config_id": "fixture",
        "stage": "stage2",
        "subject": 1,
        "seed": 42,
        "baseline_config_sha256": _h("baseline-config"),
        "stage2_config_sha256": _h("stage2-config"),
        "semantic_config_sha256": config_sha256,
        "input_bundle_sha256": input_bundle_sha256,
        "run_key": run_key,
        "layernorm_config_id": "s2-layernorm-off",
        "whitening_config_id": "s2-whitening-off",
        "preprojector_config_id": "s2-preproj-shared",
        "adapter_kind": "identity",
        "adapter_rank": None,
        "adapter_lr_ratio": None,
        "whitening_payload": None,
        "full_task_initialization_sha256": _h("full-init"),
        "shared_parameter_intersection_name": "fixture-shared",
        "shared_parameter_intersection_sha256": _h("shared-init"),
        "architecture_specific_initialization_sha256": _h("specific-init"),
        "data_order_sha256": _h("data-order"),
        "trajectory_sha256": _h("trajectory"),
    }
    candidate = {
        **candidate_body,
        "candidate_spec_sha256": sha256_json(candidate_body),
    }
    run_body = {
        "schema_version": 1,
        "payload_type": "samga_brain_rw.development_run",
        "stage": 2,
        "subject": 1,
        "seed": 42,
        "config_id": "fixture",
        "config_sha256": config_sha256,
        "protocol_sha256": input_hashes["protocol_sha256"],
        "cache_sha256": input_hashes["cache_sha256"],
        "git_sha": "1" * 40,
        "upstream_sha": "1a63745b7ff6f98dad34b0f0b8246a9b5260d9c1",
        "data_order_sha256": _h("data-order"),
        "candidate_spec_sha256": candidate["candidate_spec_sha256"],
        "run_key": run_key,
    }
    payload = {
        "subject": 1,
        "seed": 42,
        "config_sha256": config_sha256,
        "data_order_sha256": _h("data-order"),
        "trajectory_sha256": _h("trajectory"),
        "input_hashes": input_hashes,
        "candidate_spec": candidate,
        "run_manifest": {
            **run_body,
            "run_manifest_sha256": sha256_json(run_body),
        },
    }
    source_records = [
        {
            "manifest_sha256": input_hashes["manifest_sha256"],
            "records_sha256": input_hashes["records_sha256"],
            "role": role,
            "role_payload_sha256": input_hashes[
                "train_role_sha256"
                if role == "train"
                else "val_dev_role_sha256"
            ],
            "source_manifest_sha256": input_hashes[
                "source_manifest_sha256"
            ],
            "source_payload_sha256": input_hashes[
                "source_payload_sha256"
            ],
        }
        for role in ("train", "val-dev")
    ]
    provenance = {
        "config_sha256": config_sha256,
        "manifest_sha256": input_hashes["manifest_sha256"],
        "protocol_sha256": input_hashes["protocol_sha256"],
        "seed": 42,
        "subject": 1,
    }
    metadata = {
        "complete": True,
        "observed_scopes": ["train", "val-dev"],
        "ordered_ids": ordered_ids,
        "train_ordered_ids": train_ids,
        "val_dev_ordered_ids": val_dev_ids,
        "retention": {"retain_for_averaging": True},
        "source_records": source_records,
    }
    envelope = {
        "provenance": provenance,
        "metadata": metadata,
    }
    return payload, envelope


def _reseal_candidate(payload: dict[str, object]) -> None:
    candidate = payload["candidate_spec"]
    assert isinstance(candidate, dict)
    body = {
        key: value
        for key, value in candidate.items()
        if key != "candidate_spec_sha256"
    }
    candidate["candidate_spec_sha256"] = sha256_json(body)


def _reseal_run(payload: dict[str, object]) -> None:
    run = payload["run_manifest"]
    assert isinstance(run, dict)
    body = {
        key: value
        for key, value in run.items()
        if key != "run_manifest_sha256"
    }
    run["run_manifest_sha256"] = sha256_json(body)


def test_valid_checkpoint_crossbindings_are_accepted() -> None:
    payload, envelope = _valid_pair()
    validate_epoch_checkpoint_identity(payload, envelope)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("config_id", "other", "config_id"),
        ("stage", "stage0", "stage"),
        ("run_key", "other-run", "run_key"),
        ("input_bundle_sha256", _h("other-input"), "input.bundle"),
    ],
)
def test_resigned_candidate_conflict_is_rejected(
    field: str,
    value: object,
    message: str,
) -> None:
    payload, envelope = _valid_pair()
    candidate = payload["candidate_spec"]
    assert isinstance(candidate, dict)
    candidate[field] = value
    _reseal_candidate(payload)
    run = payload["run_manifest"]
    assert isinstance(run, dict)
    run["candidate_spec_sha256"] = candidate["candidate_spec_sha256"]
    _reseal_run(payload)

    with pytest.raises(ValueError, match=message):
        validate_epoch_checkpoint_identity(payload, envelope)


def test_resigned_candidate_and_run_reject_noncanonical_run_key() -> None:
    payload, envelope = _valid_pair()
    candidate = payload["candidate_spec"]
    run = payload["run_manifest"]
    assert isinstance(candidate, dict)
    assert isinstance(run, dict)
    candidate["run_key"] = "stage2__fixture__self-consistent-but-incomplete"
    _reseal_candidate(payload)
    run["run_key"] = candidate["run_key"]
    run["candidate_spec_sha256"] = candidate["candidate_spec_sha256"]
    _reseal_run(payload)

    with pytest.raises(ValueError, match="run_key"):
        validate_epoch_checkpoint_identity(payload, envelope)


def test_resigned_partition_swap_is_rejected() -> None:
    payload, envelope = _valid_pair()
    metadata = envelope["metadata"]
    assert isinstance(metadata, dict)
    train_ids = metadata["train_ordered_ids"]
    val_dev_ids = metadata["val_dev_ordered_ids"]
    metadata["train_ordered_ids"] = val_dev_ids
    metadata["val_dev_ordered_ids"] = train_ids

    with pytest.raises(ValueError, match="train.*ordered|partition"):
        validate_epoch_checkpoint_identity(payload, envelope)


@pytest.mark.parametrize(
    ("train_ids", "val_dev_ids", "message"),
    [
        ([], ["val-dev:0"], "nonempty"),
        (["train:0"], [], "nonempty"),
        (["train:0", "train:0"], ["val-dev:0"], "unique"),
        (["shared:0"], ["shared:0"], "disjoint"),
    ],
)
def test_ordered_id_partitions_require_nonempty_unique_disjoint_sets(
    train_ids: list[str],
    val_dev_ids: list[str],
    message: str,
) -> None:
    payload, envelope = _valid_pair()
    metadata = envelope["metadata"]
    assert isinstance(metadata, dict)
    metadata["train_ordered_ids"] = train_ids
    metadata["val_dev_ordered_ids"] = val_dev_ids
    metadata["ordered_ids"] = [*train_ids, *val_dev_ids]

    with pytest.raises(ValueError, match=message):
        validate_epoch_checkpoint_identity(payload, envelope)


@pytest.mark.parametrize(("field", "value"), [("schema_version", True), ("stage", False)])
def test_run_manifest_rejects_boolean_integer_identities(
    field: str,
    value: object,
) -> None:
    payload, envelope = _valid_pair()
    run = payload["run_manifest"]
    assert isinstance(run, dict)
    run[field] = value
    _reseal_run(payload)

    with pytest.raises(ValueError, match=field):
        validate_epoch_checkpoint_identity(payload, envelope)
