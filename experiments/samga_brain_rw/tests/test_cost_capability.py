from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from samga_brain_rw.fusion import enumerate_stage1_configs
from samga_brain_rw.hashing import canonical_json_bytes, sha256_json
from samga_brain_rw.inference_cost import (
    benchmark_alternating_branches,
    build_raw_stage1_cost_record,
    load_cost_protocol,
)
from samga_brain_rw.stage1 import ValidatedStage1CostCapability


_INTERNVIT_FILE_ROLES = {
    "internvit_config",
    "internvit_configuration_code",
    "internvit_flash_attention_code",
    "internvit_feature_contract_code",
    "internvit_feature_extractor_code",
    "internvit_modeling_code",
    "internvit_preprocessor_config",
    "internvit_weight_index",
    "internvit_weight_shard_1",
    "internvit_weight_shard_2",
    "internvit_weight_shard_3",
    "samga_adapters_code",
    "samga_checkpoint",
    "samga_checkpoint_identity_code",
    "samga_checkpoint_io_code",
    "samga_checkpoint_loader_code",
    "samga_checkpoint_sidecar",
    "samga_checkpoints_code",
    "samga_feature_transforms_code",
    "samga_model_code",
    "samga_trainer_code",
    "samga_upstream_loader_code",
    "semantic_config",
    "upstream_eeg_encoder_code",
    "upstream_loss_code",
    "upstream_projector_code",
}
_BRAINRW_FILE_ROLES = {
    "brainrw_access_code",
    "brainrw_artifacts_code",
    "brainrw_checkpoint",
    "brainrw_checkpoint_sidecar",
    "brainrw_config_code",
    "brainrw_data_code",
    "brainrw_factory_code",
    "brainrw_hashing_code",
    "brainrw_runtime_contract_code",
    "clip_config",
    "clip_preprocessor_config",
    "clip_weights",
}


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


class _DurationClock:
    def __init__(self, durations_ns: list[int]) -> None:
        self._durations = iter(durations_ns)
        self._now_ns = 0
        self._start = True

    def __call__(self) -> int:
        if self._start:
            self._start = False
            return self._now_ns
        self._now_ns += next(self._durations)
        self._start = True
        return self._now_ns


def _runtime_reference() -> dict[str, object]:
    environment = {
        "cuda_version": "12.6",
        "python_version": "3.10.18",
        "torch_version": "2.10.0+cu126",
    }
    contract = {
        "accelerator": "NVIDIA A40",
        "branch_device_binding": "same_cuda_device",
        "device_index": 0,
        "device_type": "cuda",
        "process_mode": "single_process",
        "schema_version": 1,
    }
    evidence = {
        "accelerator_name": "NVIDIA A40",
        "bf16_supported": True,
        "cuda_available": True,
        "cuda_capability": [8, 6],
        "cuda_device_count": 1,
        "cuda_device_index": 0,
        "cuda_version": "12.6",
        "schema_version": 1,
        "torch_version": "2.10.0+cu126",
        "total_memory_bytes": 48 * 1024**3,
    }
    return {
        "declared_runtime_contract": contract,
        "declared_runtime_contract_sha256": sha256_json(contract),
        "declared_runtime_observation": evidence,
        "declared_runtime_observation_sha256": sha256_json(evidence),
        "declared_semantic_environment": environment,
        "declared_semantic_environment_sha256": sha256_json(environment),
    }


def _raw_model_reference() -> dict[str, object]:
    return {
        "branches": {
            branch_id: {
                "checkpoint_sha256": _digest(
                    f"01/42-{branch_id}-checkpoint"
                ),
                "model_code_sha256": _digest(f"{branch_id}-code"),
                "model_config_sha256": _digest(f"{branch_id}-config"),
                "model_id": f"{branch_id}_stage1_cost_model",
                "parameter_dtypes": (
                    {"foundation": "bfloat16", "task": "float32"}
                    if branch_id == "internvit"
                    else {"model": "bfloat16"}
                ),
                "weights_sha256": _digest(f"{branch_id}-weights"),
            }
            for branch_id in ("internvit", "brainrw")
        },
        "schema_version": 1,
    }


def _raw_input_reference() -> dict[str, object]:
    cells: list[dict[str, object]] = []
    for subject in (1, 5, 8):
        for seed in (42, 43):
            cell_id = f"{subject:02d}/{seed}"
            cells.append(
                {
                    "alignment_sha256": _digest(f"{cell_id}-alignment"),
                    "branches": {
                        branch_id: {
                            "checkpoint_sha256": _digest(
                                f"{cell_id}-{branch_id}-checkpoint"
                            ),
                            "input_bundle_sha256": _digest(
                                f"{cell_id}-{branch_id}-input"
                            ),
                            "resolved_config_sha256": _digest(
                                f"{cell_id}-{branch_id}-config"
                            ),
                            "run_key": f"{branch_id}-{subject}-{seed}",
                            "run_manifest_sha256": _digest(
                                f"{cell_id}-{branch_id}-run"
                            ),
                            "score_envelope_sha256": _digest(
                                f"{cell_id}-{branch_id}-envelope"
                            ),
                            "score_payload_sha256": _digest(
                                f"{cell_id}-{branch_id}-score"
                            ),
                            "source_payload_sha256": _digest(
                                f"{cell_id}-{branch_id}-source"
                            ),
                        }
                        for branch_id in ("internvit", "brainrw")
                    },
                    "cell_id": cell_id,
                    "gallery_ids_sha256": _digest(f"{cell_id}-gallery"),
                    "query_ids_sha256": _digest(f"{cell_id}-query"),
                    "seed": seed,
                    "subject": subject,
                }
            )
    return {
        "cells": cells,
        "provenance_scope": "val-dev-identities-only",
        "schema_version": 1,
    }


def _score_inputs(raw_inputs: dict[str, object]) -> list[dict[str, object]]:
    values: list[dict[str, object]] = []
    for raw_cell in raw_inputs["cells"]:  # type: ignore[index]
        cell = dict(raw_cell)  # type: ignore[arg-type]
        raw_branches = cell["branches"]
        branches: dict[str, object] = {}
        for branch_id in ("internvit", "brainrw"):
            branch = dict(raw_branches[branch_id])  # type: ignore[index]
            branches[branch_id] = {
                "binding_sha256": _digest(
                    f"{cell['cell_id']}-{branch_id}-binding"
                ),
                "checkpoint_sha256": branch["checkpoint_sha256"],
                "resolved_config_sha256": branch["resolved_config_sha256"],
                "run_proof_sha256": _digest(
                    f"{cell['cell_id']}-{branch_id}-proof"
                ),
                "score_envelope_sha256": branch["score_envelope_sha256"],
                "score_payload_sha256": branch["score_payload_sha256"],
            }
        values.append(
            {
                "alignment_sha256": cell["alignment_sha256"],
                "brainrw": branches["brainrw"],
                "cell_id": cell["cell_id"],
                "gallery_count": 200,
                "gallery_ids_sha256": cell["gallery_ids_sha256"],
                "internvit": branches["internvit"],
                "query_count": 200,
                "query_ids_sha256": cell["query_ids_sha256"],
                "seed": cell["seed"],
                "subject": cell["subject"],
            }
        )
    return values


class _Completion:
    def __init__(self, output_hashes: dict[str, str]) -> None:
        self.output_hashes = output_hashes
        self.document: dict[str, object] = {}
        self.revalidation_count = 0

    def bind_current_claim(
        self,
        *,
        generation: int,
        job_map_sha256: str,
        row_sha256: str,
        claim_sha256: str,
    ) -> None:
        payload = {
            "array_index": 0,
            "claim_sha256": claim_sha256,
            "generation": generation,
            "job_map_sha256": job_map_sha256,
            "output_hashes": self.output_hashes,
            "row_sha256": row_sha256,
        }
        self.document = {
            "payload": payload,
            "payload_sha256": sha256_json(payload),
            "payload_type": "samga_brain_rw.stage1_cost_completion",
            "schema_version": 1,
        }

    def revalidate(self) -> None:
        self.revalidation_count += 1


def _sealed_fixture(
    experiment_root: Path,
    tmp_path: Path,
    *,
    generation: int = 1,
) -> tuple[Path, str, object, Path]:
    from samga_brain_rw import cost_capability

    project_root = (tmp_path / "project").resolve()
    (project_root / ".git").mkdir(parents=True)
    config_root = project_root / "experiments/samga_brain_rw/configs"
    config_root.mkdir(parents=True)
    protocol_path = config_root / "stage1_cost_v1.json"
    protocol_path.write_bytes(
        (experiment_root / "configs/stage1_cost_v1.json").read_bytes()
    )
    protocol = load_cost_protocol(protocol_path)
    execution_path = config_root / "stage1_cost_execution_v1.json"
    execution_path.write_bytes(
        (
            experiment_root / "configs/stage1_cost_execution_v1.json"
        ).read_bytes()
    )
    execution = cost_capability.load_stage1_cost_execution_plan(execution_path)
    runner_path = (
        project_root
        / "experiments/samga_brain_rw/scripts/run_stage1_cost.py"
    )
    runner_path.parent.mkdir(parents=True)
    runner_path.write_bytes(
        (experiment_root / "scripts/run_stage1_cost.py").read_bytes()
    )
    raw_inputs = _raw_input_reference()
    scores = _score_inputs(raw_inputs)
    score_document = {
        "artifact_type": "samga_brain_rw.stage1_cost_score_inputs",
        "raw_input_reference": raw_inputs,
        "raw_input_reference_sha256": sha256_json(raw_inputs),
        "schema_version": 1,
        "scope": "val-dev",
        "score_inputs": scores,
        "score_inputs_sha256": sha256_json(scores),
    }
    score_path = (tmp_path / "score-inputs.json").resolve()
    _write_json(score_path, score_document)

    foundation_dir = (tmp_path / "foundation").resolve()
    foundation_dir.mkdir()
    foundation_names = {
        "internvit_config": "config.json",
        "internvit_configuration_code": "configuration_intern_vit.py",
        "internvit_flash_attention_code": "flash_attention.py",
        "internvit_modeling_code": "modeling_intern_vit.py",
        "internvit_preprocessor_config": "preprocessor_config.json",
        "internvit_weight_index": "model.safetensors.index.json",
        "internvit_weight_shard_1": "model-00001-of-00003.safetensors",
        "internvit_weight_shard_2": "model-00002-of-00003.safetensors",
        "internvit_weight_shard_3": "model-00003-of-00003.safetensors",
    }
    role_paths: dict[str, dict[str, Path]] = {
        "internvit": {},
        "brainrw": {},
    }
    for role in sorted(_INTERNVIT_FILE_ROLES):
        if role in foundation_names:
            path = foundation_dir / foundation_names[role]
        elif role == "samga_checkpoint":
            path = (tmp_path / "internvit-checkpoint.pt").resolve()
        elif role == "samga_checkpoint_sidecar":
            path = (tmp_path / "internvit-checkpoint.pt.meta.json").resolve()
        elif role == "semantic_config":
            path = (tmp_path / "internvit-semantic.json").resolve()
        else:
            path = (tmp_path / "bound-files/internvit" / f"{role}.py").resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{role}-bytes".encode())
        role_paths["internvit"][role] = path
    _write_json(
        role_paths["internvit"]["semantic_config"],
        {
            "config_id": "internvit_baseline_v1",
            "config_type": "internvit_baseline",
            "model": {
                "config_sha256": _file_sha256(
                    role_paths["internvit"]["internvit_config"]
                ),
                "path": str(foundation_dir),
                "preprocessor_sha256": _file_sha256(
                    role_paths["internvit"][
                        "internvit_preprocessor_config"
                    ]
                ),
                "repo": "OpenGVLab/InternViT-6B-448px-V2_5",
                "revision": "9d1a4344077479c93d42584b6941c64d795d508d",
                "weight_sha256": {
                    filename: _file_sha256(
                        foundation_dir / filename
                    )
                    for filename in (
                        "model-00001-of-00003.safetensors",
                        "model-00002-of-00003.safetensors",
                        "model-00003-of-00003.safetensors",
                    )
                },
            },
            "schema_version": 1,
        },
    )
    for role in sorted(_BRAINRW_FILE_ROLES):
        if role == "brainrw_checkpoint":
            path = (tmp_path / "brainrw-checkpoint.pt").resolve()
        elif role == "brainrw_checkpoint_sidecar":
            path = (tmp_path / "brainrw-checkpoint.pt.meta.json").resolve()
        else:
            path = (tmp_path / "bound-files/brainrw" / f"{role}.bin").resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{role}-bytes".encode())
        role_paths["brainrw"][role] = path
    representative_checkpoint_sha256s = {
        "internvit": _file_sha256(
            role_paths["internvit"]["samga_checkpoint"]
        ),
        "brainrw": _file_sha256(
            role_paths["brainrw"]["brainrw_checkpoint"]
        ),
    }
    for branch_id, checkpoint_sha256 in (
        representative_checkpoint_sha256s.items()
    ):
        raw_inputs["cells"][0]["branches"][branch_id][  # type: ignore[index]
            "checkpoint_sha256"
        ] = checkpoint_sha256
    scores = _score_inputs(raw_inputs)
    score_document = {
        "artifact_type": "samga_brain_rw.stage1_cost_score_inputs",
        "raw_input_reference": raw_inputs,
        "raw_input_reference_sha256": sha256_json(raw_inputs),
        "schema_version": 1,
        "scope": "val-dev",
        "score_inputs": scores,
        "score_inputs_sha256": sha256_json(scores),
    }
    _write_json(score_path, score_document)
    raw_models = _raw_model_reference()
    for branch_id, checkpoint_sha256 in (
        representative_checkpoint_sha256s.items()
    ):
        raw_models["branches"][branch_id][  # type: ignore[index]
            "checkpoint_sha256"
        ] = checkpoint_sha256
    model_branches = {
        branch_id: {
                "factory": (
                    "internvit_v2_5_plus_samga"
                    if branch_id == "internvit"
                    else "brainrw_clip_lora"
                ),
                "files": [
                    {
                        "path": str(role_paths[branch_id][role]),
                        "role": role,
                        "sha256": _file_sha256(role_paths[branch_id][role]),
                    }
                    for role in sorted(
                        _INTERNVIT_FILE_ROLES
                        if branch_id == "internvit"
                        else _BRAINRW_FILE_ROLES
                    )
                ],
                "parameters": (
                    {
                        "checkpoint_path": str(
                            role_paths[branch_id]["samga_checkpoint"]
                        ),
                        "foundation_model_path": str(foundation_dir),
                        "representative_seed": 42,
                        "representative_subject": 1,
                        "semantic_config_path": str(
                            role_paths[branch_id]["semantic_config"]
                        ),
                    }
                    if branch_id == "internvit"
                    else {
                        "checkpoint_path": str(
                            role_paths[branch_id]["brainrw_checkpoint"]
                        ),
                        "representative_seed": 42,
                        "representative_subject": 1,
                    }
                ),
            }
        for branch_id in ("internvit", "brainrw")
    }
    for branch_id in ("internvit", "brainrw"):
        for field_name, roles in (
            cost_capability._MODEL_AGGREGATE_ROLES[branch_id].items()
        ):
            raw_models["branches"][branch_id][field_name] = (  # type: ignore[index]
                cost_capability._model_file_aggregate_sha256(
                    model_branches[branch_id]["files"],
                    roles,
                )
            )
    model_document = {
        "artifact_type": "samga_brain_rw.stage1_cost_model_manifest",
        "branches": model_branches,
        "raw_model_reference": raw_models,
        "raw_model_reference_sha256": sha256_json(raw_models),
        "schema_version": 1,
        "scope": "stage1-cost",
    }
    model_path = (tmp_path / "model-manifest.json").resolve()
    _write_json(model_path, model_document)

    runtime_reference = _runtime_reference()
    runtime_evidence = runtime_reference["declared_runtime_observation"]
    runtime_document = {
        "artifact_type": "samga_brain_rw.stage1_cost_runtime_manifest",
        "execution_config_file_sha256": _file_sha256(execution_path),
        "execution_config_sha256": execution.sha256,
        "runtime_evidence": runtime_evidence,
        "runtime_evidence_sha256": sha256_json(runtime_evidence),
        "runtime_reference": runtime_reference,
        "runtime_reference_sha256": sha256_json(runtime_reference),
        "schema_version": 1,
        "scope": "stage1-cost",
    }
    output_dir = (tmp_path / "cost-output").resolve()
    output_dir.mkdir()
    runtime_path = output_dir / "runtime-manifest.json"
    _write_json(runtime_path, runtime_document)

    measured_order = [
        branch_id
        for round_index in range(10, 60)
        for branch_id in (
            ("internvit", "brainrw")
            if round_index % 2 == 0
            else ("brainrw", "internvit")
        )
    ]
    benchmark = benchmark_alternating_branches(
        protocol,
        {"internvit": lambda: None, "brainrw": lambda: None},
        clock_ns=_DurationClock(
            [
                2_000_000 if branch_id == "internvit" else 1_000_000
                for branch_id in measured_order
            ]
        ),
        synchronize=lambda: None,
    )
    raw_record = build_raw_stage1_cost_record(
        protocol,
        benchmark,
        runtime_reference=runtime_reference,
        model_reference=raw_models,
        job_claim_reference={
            "authority_execution_file_sha256": _digest(
                "authority-execution"
            ),
            "authority_execution_payload_sha256": _digest(
                "authority-execution-payload"
            ),
            "attempt_id": f"attempt-{generation - 1:04d}",
            "attempt_index": generation - 1,
            "claim_id": "stage1-cost-test",
            "schema_version": 1,
            "slurm_job_id": "123456_0",
            "slurm_partition": "i64m1tga40u",
            "unverified_claim_sha256": _digest("claim"),
            "unverified_previous_record_sha256": None,
        },
        input_reference=raw_inputs,
    )
    raw_path = output_dir / f"stage1-cost-attempt-{generation - 1:04d}.json"
    _write_json(raw_path, raw_record.to_document())

    assert runner_path.is_file()
    input_bundle_sha256 = sha256_json(
        {
            "execution_config_sha256": execution.sha256,
            "model_manifest_file_sha256": _file_sha256(model_path),
            "runner_file_sha256": _file_sha256(runner_path),
            "score_inputs_file_sha256": _file_sha256(score_path),
        }
    )
    run_manifest = {
        "artifact_type": "samga_brain_rw.stage1_cost_run_manifest",
        "authority_execution_file_sha256": _digest(
            "authority-execution"
        ),
        "authority_execution_path": str(
            tmp_path
            / ".job-claims"
            / f"generation-{generation:06d}"
            / "execution.json"
        ),
        "authority_execution_payload_sha256": _digest(
            "authority-execution-payload"
        ),
        "execution_config_file_sha256": _file_sha256(execution_path),
        "execution_config_path": str(execution_path.resolve()),
        "execution_config_sha256": execution.sha256,
        "input_bundle_sha256": input_bundle_sha256,
        "model_manifest_file_sha256": _file_sha256(model_path),
        "model_manifest_path": str(model_path),
        "protocol_file_sha256": _file_sha256(protocol_path),
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": protocol.sha256,
        "raw_record_file_sha256": _file_sha256(raw_path),
        "raw_record_path": str(raw_path),
        "raw_record_sha256": raw_record.record_sha256,
        "runner_file_sha256": _file_sha256(runner_path),
        "runner_path": str(runner_path),
        "runtime_evidence_sha256": sha256_json(runtime_evidence),
        "runtime_manifest_file_sha256": _file_sha256(runtime_path),
        "runtime_manifest_path": str(runtime_path),
        "schema_version": 1,
        "scope": "stage1-cost",
        "score_inputs_file_sha256": _file_sha256(score_path),
        "score_inputs_path": str(score_path),
    }
    run_path = output_dir / "run-manifest.json"
    _write_json(run_path, run_manifest)

    output_hashes = {
        "raw_record_file_sha256": _file_sha256(raw_path),
        "run_manifest_file_sha256": _file_sha256(run_path),
        "runtime_manifest_file_sha256": _file_sha256(runtime_path),
    }
    completion = _Completion(output_hashes)
    map_sha256 = _digest("job-map")
    row = {
        "array_index": 0,
        "argv": [
            "python",
            str(runner_path),
            "--subject",
            "1",
            "--seed",
            "20260720",
            "--config",
            str(protocol_path.resolve()),
            "--execution-config",
            str(execution_path.resolve()),
            "--score-inputs",
            str(score_path),
            "--model-manifest",
            str(model_path),
            "--output-dir",
            str(output_dir),
            "--project-root",
            str(project_root),
            "--config-id",
            "stage1_cost_v1",
            "--expected-config-sha256",
            protocol.sha256,
            "--expected-execution-config-sha256",
            execution.sha256,
            "--expected-input-bundle-sha256",
            input_bundle_sha256,
            "--run-key",
            "stage1-cost-test",
            "--device",
            "cuda",
        ],
        "config_id": "stage1_cost_v1",
        "config_sha256": protocol.sha256,
        "expected_completion_schema": {
            "payload_type": "samga_brain_rw.stage1_cost_completion",
            "required_output_hashes": sorted(output_hashes),
            "schema_version": 1,
        },
        "gres": "gpu:a40:1",
        "input_bundle_sha256": input_bundle_sha256,
        "partition": "i64m1tga40u",
        "role": "cost-benchmark",
        "run_key": "stage1-cost-test",
        "stage": "stage-1-cost-benchmark",
    }
    job_map = {
        "array_bounds": [0, 0],
        "payload_sha256": map_sha256,
        "row_count": 1,
        "rows": [row],
        "stage": "stage-1-cost-benchmark",
    }
    completion.bind_current_claim(
        generation=generation,
        job_map_sha256=map_sha256,
        row_sha256=sha256_json(row),
        claim_sha256=_digest("claim"),
    )
    authority_identity = {
        "array_index": 0,
        "attempt_payload_sha256": (
            None
            if generation == 1
            else _digest("authority-attempt-payload")
        ),
        "attempt_record_sha256": (
            None
            if generation == 1
            else _digest("authority-attempt-record")
        ),
        "claim_sha256": _digest("claim"),
        "generation": generation,
        "job_map_sha256": map_sha256,
        "path": str(
            tmp_path
            / ".job-claims"
            / f"generation-{generation:06d}"
            / "execution.json"
        ),
        "payload_sha256": _digest("authority-execution-payload"),
        "row_sha256": sha256_json(row),
        "scheduler_job_id": "123456_0",
        "sha256": _digest("authority-execution"),
    }

    def load_execution_authority(
        payload: object,
        actual_row: object,
        **expected: object,
    ) -> dict[str, object]:
        if payload is not job_map or actual_row is not row:
            pytest.fail("cost capability selected an unsealed execution row")
        if expected != {
            "expected_claim_sha256": _digest("claim"),
            "expected_generation": generation,
        }:
            raise ValueError(
                "cost execution authority current claim identity changed"
            )
        return copy.deepcopy(authority_identity)

    fake_job_maps = SimpleNamespace(
        load_job_map=lambda path, expected_sha256: (
            job_map
            if path == tmp_path / "job-map.json"
            and expected_sha256 == map_sha256
            else pytest.fail("cost capability loaded an unsealed job map")
        ),
        load_job_completion=lambda payload, actual_row: (
            completion
            if payload is job_map and actual_row is row
            else pytest.fail("cost capability selected an unsealed row")
        ),
        load_cost_execution_authority=load_execution_authority,
    )
    cost_capability._load_job_map_module = lambda: fake_job_maps
    return (
        tmp_path / "job-map.json",
        map_sha256,
        completion,
        role_paths["brainrw"]["brainrw_checkpoint"],
    )


def test_sealed_completion_issues_revalidating_stage1_cost_capability(
    experiment_root: Path,
    tmp_path: Path,
) -> None:
    from samga_brain_rw.cost_capability import (
        load_validated_stage1_cost_capability,
    )

    job_map, map_sha256, completion, _ = _sealed_fixture(
        experiment_root,
        tmp_path,
    )
    capability = load_validated_stage1_cost_capability(job_map, map_sha256)

    assert isinstance(capability, ValidatedStage1CostCapability)
    identity = capability.identity_payload()
    assert identity["artifact_type"] == (
        "samga_brain_rw.validated_stage1_cost_capability"
    )
    assert identity["scope"] == "val-dev"
    assert identity["two_encoder_count"] == 2
    assert identity["branch_measured_ms_per_query"] == {
        "internvit": 0.01,
        "brainrw": 0.005,
    }
    assert len(identity["score_inputs"]) == 6
    assert identity["score_inputs_sha256"] == sha256_json(
        identity["score_inputs"]
    )
    assert [entry["config_id"] for entry in identity["operator_complexity_keys"]] == [
        config.config_id for config in enumerate_stage1_configs()
    ]
    assert capability.proof_sha256 == sha256_json(identity)
    assert capability.measured_branch_cost("internvit") == 0.01
    assert (
        capability.operator_complexity_key(
            enumerate_stage1_configs()[0].config_id
        )
        == (0, 6, 9)
    )
    capability.revalidate()
    assert completion.revalidation_count >= 2


def test_sealed_capability_satisfies_stage1_snapshot_interface(
    experiment_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from samga_brain_rw import stage1
    from samga_brain_rw.cost_capability import (
        load_validated_stage1_cost_capability,
    )

    job_map, map_sha256, _, _ = _sealed_fixture(
        experiment_root,
        tmp_path,
    )
    capability = load_validated_stage1_cost_capability(job_map, map_sha256)
    expected_inputs = capability.identity_payload()["score_inputs"]
    monkeypatch.setattr(
        stage1,
        "_cost_score_inputs",
        lambda cells: copy.deepcopy(expected_inputs),
    )

    identity, proof, branch_costs, operator_keys = (
        stage1._validated_cost_capability_snapshot(capability, ())
    )

    assert identity == capability.identity_payload()
    assert proof == capability.proof_sha256
    assert branch_costs == {"internvit": 0.01, "brainrw": 0.005}
    assert set(operator_keys) == {
        config.config_id for config in enumerate_stage1_configs()
    }


def test_cost_capability_is_not_constructible_from_raw_record_or_public_state(
    experiment_root: Path,
) -> None:
    from samga_brain_rw.cost_capability import SealedStage1CostCapability

    protocol = load_cost_protocol(
        experiment_root / "configs/stage1_cost_v1.json"
    )
    raw = build_raw_stage1_cost_record(
        protocol,
        benchmark_alternating_branches(
            protocol,
            {"internvit": lambda: None, "brainrw": lambda: None},
            clock_ns=_DurationClock([1_000_000] * 100),
            synchronize=lambda: None,
        ),
        runtime_reference=_runtime_reference(),
        model_reference=_raw_model_reference(),
        job_claim_reference={
            "authority_execution_file_sha256": _digest(
                "authority-execution"
            ),
            "authority_execution_payload_sha256": _digest(
                "authority-execution-payload"
            ),
            "attempt_id": "attempt-0000",
            "attempt_index": 0,
            "claim_id": "stage1-cost-a40",
            "schema_version": 1,
            "slurm_job_id": "123456_0",
            "slurm_partition": "debug",
            "unverified_claim_sha256": _digest("claim"),
            "unverified_previous_record_sha256": None,
        },
        input_reference=_raw_input_reference(),
    )

    with pytest.raises(TypeError, match="sealed.*job completion|controlled"):
        SealedStage1CostCapability(raw)


def test_cost_capability_revalidation_detects_bound_model_file_mutation(
    experiment_root: Path,
    tmp_path: Path,
) -> None:
    from samga_brain_rw.cost_capability import (
        load_validated_stage1_cost_capability,
    )

    job_map, map_sha256, _, bound_model_file = _sealed_fixture(
        experiment_root,
        tmp_path,
    )
    capability = load_validated_stage1_cost_capability(job_map, map_sha256)
    bound_model_file.write_bytes(b"mutated")

    with pytest.raises(ValueError, match="model.*SHA|file.*SHA"):
        capability.revalidate()


def test_cost_capability_revalidation_detects_runner_mutation(
    experiment_root: Path,
    tmp_path: Path,
) -> None:
    from samga_brain_rw.cost_capability import (
        load_validated_stage1_cost_capability,
    )

    job_map, map_sha256, _, _ = _sealed_fixture(
        experiment_root,
        tmp_path,
    )
    capability = load_validated_stage1_cost_capability(job_map, map_sha256)
    runner_path = (
        tmp_path
        / "project/experiments/samga_brain_rw/scripts/run_stage1_cost.py"
    )
    runner_path.write_bytes(runner_path.read_bytes() + b"\n# changed\n")

    with pytest.raises(ValueError, match="runner|input-bundle"):
        capability.revalidate()


def test_cost_capability_rejects_completion_hash_drift(
    experiment_root: Path,
    tmp_path: Path,
) -> None:
    from samga_brain_rw.cost_capability import (
        load_validated_stage1_cost_capability,
    )

    job_map, map_sha256, completion, _ = _sealed_fixture(
        experiment_root,
        tmp_path,
    )
    completion.output_hashes = copy.deepcopy(completion.output_hashes)
    completion.output_hashes["raw_record_file_sha256"] = _digest("wrong")

    with pytest.raises(ValueError, match="completion.*hash|raw record.*hash"):
        load_validated_stage1_cost_capability(job_map, map_sha256)


def test_model_manifest_rejects_missing_required_role(
    experiment_root: Path,
    tmp_path: Path,
) -> None:
    from samga_brain_rw.cost_capability import (
        load_stage1_cost_model_manifest,
    )

    _sealed_fixture(experiment_root, tmp_path)
    model_path = (tmp_path / "model-manifest.json").resolve()
    model = json.loads(model_path.read_text(encoding="utf-8"))
    model["branches"]["internvit"]["files"] = [
        value
        for value in model["branches"]["internvit"]["files"]
        if value["role"] != "internvit_feature_extractor_code"
    ]
    _write_json(model_path, model)

    with pytest.raises(ValueError, match="required.*role|missing"):
        load_stage1_cost_model_manifest(model_path)


def test_model_manifest_rejects_invalid_raw_parameter_dtype_schema(
    experiment_root: Path,
    tmp_path: Path,
) -> None:
    from samga_brain_rw.cost_capability import (
        load_stage1_cost_model_manifest,
    )

    _sealed_fixture(experiment_root, tmp_path)
    model_path = (tmp_path / "model-manifest.json").resolve()
    model = json.loads(model_path.read_text(encoding="utf-8"))
    model["raw_model_reference"]["branches"]["internvit"][
        "parameter_dtypes"
    ] = {"foundation": "float32", "task": "float32"}
    model["raw_model_reference_sha256"] = sha256_json(
        model["raw_model_reference"]
    )
    _write_json(model_path, model)

    with pytest.raises(ValueError, match="dtype|parameter"):
        load_stage1_cost_model_manifest(model_path)


def test_model_manifest_rejects_raw_checkpoint_not_bound_to_checkpoint_file(
    experiment_root: Path,
    tmp_path: Path,
) -> None:
    from samga_brain_rw.cost_capability import (
        load_stage1_cost_model_manifest,
    )

    _sealed_fixture(experiment_root, tmp_path)
    model_path = (tmp_path / "model-manifest.json").resolve()
    model = json.loads(model_path.read_text(encoding="utf-8"))
    model["raw_model_reference"]["branches"]["brainrw"][
        "checkpoint_sha256"
    ] = _digest("not-the-bound-checkpoint")
    model["raw_model_reference_sha256"] = sha256_json(
        model["raw_model_reference"]
    )
    _write_json(model_path, model)

    with pytest.raises(ValueError, match="checkpoint"):
        load_stage1_cost_model_manifest(model_path)


def test_model_manifest_rejects_raw_weight_aggregate_not_bound_to_files(
    experiment_root: Path,
    tmp_path: Path,
) -> None:
    from samga_brain_rw.cost_capability import (
        load_stage1_cost_model_manifest,
    )

    _sealed_fixture(experiment_root, tmp_path)
    model_path = (tmp_path / "model-manifest.json").resolve()
    model = json.loads(model_path.read_text(encoding="utf-8"))
    model["raw_model_reference"]["branches"]["internvit"][
        "weights_sha256"
    ] = _digest("not-the-bound-weight-aggregate")
    model["raw_model_reference_sha256"] = sha256_json(
        model["raw_model_reference"]
    )
    _write_json(model_path, model)

    with pytest.raises(ValueError, match="weight|aggregate"):
        load_stage1_cost_model_manifest(model_path)


@pytest.mark.parametrize(
    "role",
    (
        "internvit_config",
        "internvit_preprocessor_config",
        "internvit_weight_shard_1",
        "internvit_weight_shard_2",
        "internvit_weight_shard_3",
    ),
)
def test_model_manifest_rejects_reblessed_foundation_pin_drift(
    experiment_root: Path,
    tmp_path: Path,
    role: str,
) -> None:
    from samga_brain_rw import cost_capability

    _sealed_fixture(experiment_root, tmp_path)
    model_path = (tmp_path / "model-manifest.json").resolve()
    model = json.loads(model_path.read_text(encoding="utf-8"))
    branch = model["branches"]["internvit"]
    entry = next(value for value in branch["files"] if value["role"] == role)
    bound_path = Path(entry["path"])
    bound_path.write_bytes(b"attacker-replaced-foundation-bytes")
    entry["sha256"] = _file_sha256(bound_path)
    raw_branch = model["raw_model_reference"]["branches"]["internvit"]
    for field_name, roles in (
        cost_capability._MODEL_AGGREGATE_ROLES["internvit"].items()
    ):
        raw_branch[field_name] = cost_capability._model_file_aggregate_sha256(
            branch["files"],
            roles,
        )
    model["raw_model_reference_sha256"] = sha256_json(
        model["raw_model_reference"]
    )
    _write_json(model_path, model)

    with pytest.raises(ValueError, match="pinned|semantic.*foundation"):
        cost_capability.load_stage1_cost_model_manifest(model_path)


def test_stable_file_hash_rejects_symbolic_leaf_and_parent(
    tmp_path: Path,
) -> None:
    from samga_brain_rw.cost_capability import stable_regular_file_sha256

    real_parent = (tmp_path / "real").resolve()
    real_parent.mkdir()
    target = real_parent / "target.bin"
    target.write_bytes(b"sealed")
    leaf_link = (tmp_path / "leaf-link.bin").resolve()
    leaf_link.symlink_to(target)
    parent_link = (tmp_path / "parent-link").resolve()
    parent_link.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic|regular"):
        stable_regular_file_sha256(leaf_link)
    with pytest.raises(ValueError, match="symbolic|regular"):
        stable_regular_file_sha256(parent_link / "target.bin")


def test_representative_model_checkpoint_must_match_score_binding() -> None:
    from samga_brain_rw.cost_capability import (
        _validate_representative_model_score_binding,
    )

    raw_inputs = _raw_input_reference()
    scores = _score_inputs(raw_inputs)
    models = _raw_model_reference()
    models["branches"]["internvit"]["checkpoint_sha256"] = _digest(  # type: ignore[index]
        "wrong-representative-checkpoint"
    )

    with pytest.raises(ValueError, match="checkpoint/score"):
        _validate_representative_model_score_binding(
            scores,
            {"raw_model_reference": models},
        )


def test_cost_capability_loads_raw_record_for_current_recovery_generation(
    experiment_root: Path,
    tmp_path: Path,
) -> None:
    from samga_brain_rw.cost_capability import (
        load_validated_stage1_cost_capability,
    )

    job_map, map_sha256, _, _ = _sealed_fixture(
        experiment_root,
        tmp_path,
        generation=2,
    )

    capability = load_validated_stage1_cost_capability(job_map, map_sha256)

    assert capability.identity_payload()["raw_record_sha256"]


def test_cost_capability_rejects_raw_claim_not_bound_to_current_completion(
    experiment_root: Path,
    tmp_path: Path,
) -> None:
    from samga_brain_rw.cost_capability import (
        load_validated_stage1_cost_capability,
    )

    job_map, map_sha256, completion, _ = _sealed_fixture(
        experiment_root,
        tmp_path,
    )
    completion.document["payload"]["claim_sha256"] = _digest("other-claim")  # type: ignore[index]
    completion.document["payload_sha256"] = sha256_json(
        completion.document["payload"]
    )

    with pytest.raises(ValueError, match="claim|generation"):
        load_validated_stage1_cost_capability(job_map, map_sha256)


def test_cost_capability_rejects_rehashed_self_attested_job_id_forgery(
    experiment_root: Path,
    tmp_path: Path,
) -> None:
    from samga_brain_rw.cost_capability import (
        load_validated_stage1_cost_capability,
    )

    job_map, map_sha256, completion, _ = _sealed_fixture(
        experiment_root,
        tmp_path,
    )
    output_dir = tmp_path / "cost-output"
    raw_path = output_dir / "stage1-cost-attempt-0000.json"
    run_path = output_dir / "run-manifest.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    claim = raw["self_attested_job_claim_reference"]
    claim["slurm_job_id"] = "999999_0"
    raw["self_attested_job_claim_reference_sha256"] = sha256_json(claim)
    raw["record_sha256"] = sha256_json(
        {key: value for key, value in raw.items() if key != "record_sha256"}
    )
    _write_json(raw_path, raw)
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["raw_record_file_sha256"] = _file_sha256(raw_path)
    run["raw_record_sha256"] = raw["record_sha256"]
    _write_json(run_path, run)
    completion.output_hashes = copy.deepcopy(completion.output_hashes)
    completion.output_hashes["raw_record_file_sha256"] = _file_sha256(raw_path)
    completion.output_hashes["run_manifest_file_sha256"] = _file_sha256(
        run_path
    )
    completion.document["payload"]["output_hashes"] = copy.deepcopy(  # type: ignore[index]
        completion.output_hashes
    )
    completion.document["payload_sha256"] = sha256_json(
        completion.document["payload"]
    )

    with pytest.raises(ValueError, match="scheduler|job.*ID|execution"):
        load_validated_stage1_cost_capability(job_map, map_sha256)


def test_score_input_manifest_factory_requires_exact_twelve_bindings() -> None:
    from samga_brain_rw.cost_capability import (
        build_stage1_cost_score_input_manifest,
    )

    raw_inputs = _raw_input_reference()
    score_inputs = _score_inputs(raw_inputs)
    document = build_stage1_cost_score_input_manifest(
        score_inputs=score_inputs,
        raw_input_reference=raw_inputs,
    )

    assert document["scope"] == "val-dev"
    assert len(document["score_inputs"]) == 6
    assert sum(
        len(
            [
                document["score_inputs"][index][branch_id]
                for branch_id in ("internvit", "brainrw")
            ]
        )
        for index in range(6)
    ) == 12
    assert document["score_inputs_sha256"] == sha256_json(score_inputs)

    missing = copy.deepcopy(score_inputs)
    del missing[-1]["brainrw"]["run_proof_sha256"]
    with pytest.raises(ValueError, match="schema|run_proof"):
        build_stage1_cost_score_input_manifest(
            score_inputs=missing,
            raw_input_reference=raw_inputs,
        )
