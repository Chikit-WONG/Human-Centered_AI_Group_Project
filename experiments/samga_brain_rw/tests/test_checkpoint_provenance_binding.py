from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from samga_brain_rw.checkpoints import average_state_dicts, hash_state_dict
from samga_brain_rw.hashing import (
    canonical_json_bytes,
    ordered_ids_sha256,
    sha256_json,
)


def _h(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_checkpoint(path: Path, *, epoch: int) -> Path:
    ordered_ids = ["train:0", "val-dev:0"]
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
        "train_ordered_ids_sha256": ordered_ids_sha256(ordered_ids[:1]),
        "train_role_sha256": _h("train"),
        "val_dev_ordered_ids_sha256": ordered_ids_sha256(ordered_ids[1:]),
        "val_dev_role_sha256": _h("val-dev"),
    }
    input_bundle_sha256 = sha256_json(dict(sorted(input_hashes.items())))
    config_sha256 = _h("config")
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
    candidate_spec = {
        **candidate_body,
        "candidate_spec_sha256": sha256_json(candidate_body),
    }
    run_manifest_body = {
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
        "candidate_spec_sha256": candidate_spec["candidate_spec_sha256"],
        "run_key": run_key,
    }
    model_state = {
        "weight": torch.tensor([float(epoch), float(epoch + 2)]),
        "counter": torch.tensor([7], dtype=torch.int64),
    }
    retention = {"retain_for_averaging": True}
    payload = {
        "schema_version": 1,
        "payload_type": "samga_brain_rw.epoch_checkpoint",
        "epoch": epoch,
        "global_step": epoch * 10,
        "subject": 1,
        "seed": 42,
        "config_sha256": config_sha256,
        "schedule_sha256": _h("schedule"),
        "optimizer_stage": "stage2",
        "trajectory_sha256": _h("trajectory"),
        "data_order_sha256": _h("data-order"),
        "model_state_dict": model_state,
        "model_state_sha256": hash_state_dict(model_state),
        "optimizer_state_dict": {},
        "scheduler_state_dict": {},
        "python_rng_state": [1, 2, 3],
        "numpy_rng_state": {},
        "torch_rng_state": torch.tensor([1, 2, 3], dtype=torch.uint8),
        "cuda_rng_states": [],
        "loader_generator_state": torch.tensor([4, 5], dtype=torch.uint8),
        "sampler_state_dict": {},
        "validation_metrics": {},
        "input_hashes": input_hashes,
        "effective_batch": {},
        "environment": {},
        "run_manifest": {
            **run_manifest_body,
            "run_manifest_sha256": sha256_json(run_manifest_body),
        },
        "candidate_spec": candidate_spec,
        "runtime_state": {},
        "retention": retention,
        "scope": "train",
        "validation_scope": "val-dev",
        "observed_scopes": ["train", "val-dev"],
    }
    torch.save(payload, path)
    source_records = [
        {
            "manifest_sha256": _h("manifest"),
            "records_sha256": _h("records"),
            "role": role,
            "role_payload_sha256": _h(role),
            "source_manifest_sha256": _h("source-manifest"),
            "source_payload_sha256": _h("source-payload"),
        }
        for role in ("train", "val-dev")
    ]
    provenance = {
        "config_sha256": payload["config_sha256"],
        "manifest_sha256": _h("manifest"),
        "protocol_sha256": _h("protocol"),
        "seed": payload["seed"],
        "subject": payload["subject"],
    }
    metadata = {
        "complete": True,
        "observed_scopes": ["train", "val-dev"],
        "ordered_ids": ordered_ids,
        "train_ordered_ids": ordered_ids[:1],
        "val_dev_ordered_ids": ordered_ids[1:],
        "retention": retention,
        "source_records": source_records,
    }
    envelope = {
        "schema_version": 1,
        "payload_type": "samga_brain_rw.epoch_checkpoint",
        "scope": "train",
        "source_records_sha256": sha256_json(source_records),
        "ordered_ids_sha256": ordered_ids_sha256(ordered_ids),
        "payload_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "provenance": provenance,
        "provenance_sha256": sha256_json(provenance),
        "metadata": metadata,
        "metadata_sha256": sha256_json(metadata),
    }
    path.with_suffix(path.suffix + ".meta.json").write_bytes(
        canonical_json_bytes(envelope) + b"\n"
    )
    return path


def _window(tmp_path: Path) -> list[Path]:
    return [
        _write_checkpoint(
            tmp_path / f"checkpoint_epoch{epoch:03d}.pt",
            epoch=epoch,
        )
        for epoch in range(56, 61)
    ]


def _resign_sidecar(path: Path, binding: str) -> None:
    sidecar = path.with_suffix(path.suffix + ".meta.json")
    document = json.loads(sidecar.read_text(encoding="utf-8"))
    provenance = document["provenance"]
    metadata = document["metadata"]
    source_records = metadata["source_records"]
    replacement = _h(f"mismatched-{binding}")
    if binding == "protocol":
        provenance["protocol_sha256"] = replacement
    elif binding == "manifest":
        provenance["manifest_sha256"] = replacement
        for record in source_records:
            record["manifest_sha256"] = replacement
    elif binding == "records":
        for record in source_records:
            record["records_sha256"] = replacement
    elif binding == "source_manifest":
        for record in source_records:
            record["source_manifest_sha256"] = replacement
    elif binding == "source_payload":
        for record in source_records:
            record["source_payload_sha256"] = replacement
    elif binding == "train_role":
        source_records[0]["role_payload_sha256"] = replacement
    elif binding == "val_dev_role":
        source_records[1]["role_payload_sha256"] = replacement
    elif binding == "ordered_ids":
        metadata["ordered_ids"] = ["other-train:0", "other-val-dev:0"]
        document["ordered_ids_sha256"] = ordered_ids_sha256(
            metadata["ordered_ids"]
        )
    else:  # pragma: no cover - parameter list is exhaustive
        raise AssertionError(binding)
    document["source_records_sha256"] = sha256_json(source_records)
    document["provenance_sha256"] = sha256_json(provenance)
    document["metadata_sha256"] = sha256_json(metadata)
    sidecar.write_bytes(canonical_json_bytes(document) + b"\n")


@pytest.mark.parametrize(
    "binding",
    (
        "protocol",
        "manifest",
        "records",
        "source_manifest",
        "source_payload",
        "train_role",
        "val_dev_role",
        "ordered_ids",
    ),
)
def test_averaging_rejects_resigned_sidecar_payload_identity_conflict(
    tmp_path: Path,
    binding: str,
) -> None:
    paths = _window(tmp_path)
    _resign_sidecar(paths[-1], binding)

    with pytest.raises(ValueError, match="binding|mismatch|ordered"):
        average_state_dicts(paths)


def test_typed_checkpoint_normalizes_truncated_pickle_error(
    tmp_path: Path,
) -> None:
    paths = _window(tmp_path)
    target = paths[0]
    target.write_bytes(b"\x80")
    sidecar = target.with_suffix(target.suffix + ".meta.json")
    document = json.loads(sidecar.read_text(encoding="utf-8"))
    document["payload_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    sidecar.write_bytes(canonical_json_bytes(document) + b"\n")

    with pytest.raises(ValueError, match="loaded safely|payload"):
        average_state_dicts(paths)
