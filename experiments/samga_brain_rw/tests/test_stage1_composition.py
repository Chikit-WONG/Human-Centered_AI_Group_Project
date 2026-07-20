from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np
import pytest

import samga_brain_rw.stage1 as stage1_module
from samga_brain_rw.config import SemanticConfig, make_run_key
from samga_brain_rw.fusion import (
    common_alignment_payload,
    enumerate_stage1_configs,
)
from samga_brain_rw.hashing import canonical_json_bytes, sha256_json
from samga_brain_rw.inference_cost import RawStage1CostRecord
from samga_brain_rw.registry import CandidateDecision
from samga_brain_rw.runtime_contract import PINNED_SEMANTIC_ENVIRONMENT
from samga_brain_rw.scores import (
    ScoreArtifact,
    independent_retrieval_metrics,
)
from samga_brain_rw.stage1 import (
    BRAINRW_BRANCH_ID,
    BRAINRW_CONFIG_ID,
    BRAINRW_EPOCHS,
    BRAINRW_STAGE,
    INTERNVIT_BRANCH_ID,
    INTERNVIT_CONFIG_ID,
    INTERNVIT_EPOCHS,
    INTERNVIT_STAGE,
    PILOT_COORDINATES,
    STAGE1_FUSION_CONFIG_SHA256,
    Stage1ComponentBinding,
    Stage1CompositionCell,
    Stage1CompositionOutcome,
    ValidatedComponentRunProof,
    ValidatedStage1CostCapability,
    _score_composition_gate,
    compose_stage1,
)


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = EXPERIMENT_ROOT / "configs"
INTERNVIT_SEMANTIC = SemanticConfig.from_path(
    CONFIG_ROOT / "internvit_baseline_v1.json"
)
BRAINRW_SEMANTIC = SemanticConfig.from_path(CONFIG_ROOT / "brainrw_clip_lora_v1.json")
FUSION_SEMANTIC = SemanticConfig.from_path(CONFIG_ROOT / "stage1_fusion_v1.json")
FUSION_GRID = enumerate_stage1_configs()
TARGET_FUSION_ID = "s1-temp-ti100-tc100-a050"


def _clone(value: object) -> object:
    if isinstance(value, Mapping):
        native: object = {str(key): _clone(child) for key, child in value.items()}
    elif isinstance(value, (list, tuple)):
        native = [_clone(child) for child in value]
    else:
        native = value
    return json.loads(canonical_json_bytes(native).decode("utf-8"))


def _digest(label: str, **identity: object) -> str:
    return sha256_json({"identity": identity, "label": label})


def _git_digest(label: str, **identity: object) -> str:
    return _digest(label, **identity)[:40]


def _runtime_metadata() -> dict[str, object]:
    semantic_environment = dict(PINNED_SEMANTIC_ENVIRONMENT)
    contract = {
        "accelerator": "NVIDIA A40",
        "device_type": "cuda",
        "dtype": "bfloat16",
        "schema_version": 1,
    }
    evidence = {
        "accelerator_name": "NVIDIA A40",
        "bf16_supported": True,
        "cuda_available": True,
        "cuda_capability": [8, 6],
        "cuda_device_count": 1,
        "cuda_device_index": 0,
        "cuda_version": "12.8",
        "device_type": "cuda",
        "dtype": "bfloat16",
        "schema_version": 1,
        "torch_version": "2.11.0+cu128",
        "total_memory_bytes": 48 * 1024**3,
    }
    semantic_sha256 = sha256_json(semantic_environment)
    return {
        "training_semantic_environment": semantic_environment,
        "training_semantic_environment_sha256": semantic_sha256,
        "evaluation_semantic_environment": semantic_environment,
        "evaluation_semantic_environment_sha256": semantic_sha256,
        "evaluation_runtime_contract": contract,
        "evaluation_runtime_contract_sha256": sha256_json(contract),
        "evaluation_runtime_evidence": evidence,
        "evaluation_runtime_evidence_sha256": sha256_json(evidence),
    }


def _schedule(
    branch_id: str,
) -> tuple[dict[str, object], str]:
    if branch_id == INTERNVIT_BRANCH_ID:
        config = INTERNVIT_SEMANTIC.canonical_payload()
        payload = {
            "artifact_type": "samga_brain_rw.stage1_component_schedule",
            "branch_id": branch_id,
            "config_id": INTERNVIT_CONFIG_ID,
            "config_sha256": INTERNVIT_SEMANTIC.sha256,
            "epochs": config["task"]["epochs"],  # type: ignore[index]
            "schema_version": 1,
            "task": config["task"],
        }
    else:
        config = BRAINRW_SEMANTIC.canonical_payload()
        payload = {
            "artifact_type": "samga_brain_rw.stage1_component_schedule",
            "branch_id": branch_id,
            "config_id": BRAINRW_CONFIG_ID,
            "config_sha256": BRAINRW_SEMANTIC.sha256,
            "epochs": config["training"]["epochs"],  # type: ignore[index]
            "optimizer": config["optimizer"],
            "schema_version": 1,
            "training": config["training"],
        }
    return payload, sha256_json(payload)


def _common_source_record(
    subject: int,
    seed: int,
    *,
    protocol_sha256: str,
) -> dict[str, object]:
    source_body = {
        "protocol_sha256": protocol_sha256,
        "role": "val-dev",
        "seed": seed,
        "subject": subject,
    }
    source_bytes = canonical_json_bytes(source_body)
    return {
        "manifest_sha256": _digest(
            "subject-manifest",
            subject=subject,
            protocol_sha256=protocol_sha256,
        ),
        "records_sha256": _digest(
            "manifest-records",
            subject=subject,
            protocol_sha256=protocol_sha256,
        ),
        "role": "val-dev",
        "role_payload_sha256": _digest(
            "val-dev-role",
            subject=subject,
            protocol_sha256=protocol_sha256,
        ),
        "source_manifest_sha256": _digest(
            "source-manifest",
            subject=subject,
        ),
        "source_payload_byte_count": len(source_bytes),
        "source_payload_path": (f"/safe/stage1/sub-{subject:02d}/seed-{seed}/train.pt"),
        "source_payload_sha256": sha256_json(source_body),
    }


def _score_matrices(
    *,
    variant: int = 0,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...]]:
    query_ids = tuple(f"q{index}" for index in range(6))
    gallery_ids = (*query_ids, "x0", "x1")
    internvit = np.full((6, 8), -1.0, dtype=np.float32)
    brainrw = np.full((6, 8), -1.0, dtype=np.float32)
    for index in range(6):
        competitor = (index + 1) % 6
        internvit_correct = index % 2 == 0
        for matrix, correct in (
            (internvit, internvit_correct),
            (brainrw, not internvit_correct),
        ):
            matrix[index, index] = 0.9 if correct else 0.4
            matrix[index, competitor] = 0.2 if correct else 0.8
    # This changes real payload bytes without changing any ranking.
    internvit[0, 7] = np.float32(-0.99 + variant * 0.01)
    return internvit, brainrw, query_ids, gallery_ids


@dataclass(frozen=True)
class _TestValidatedRunProof(ValidatedComponentRunProof):
    """Test-only nominal capability standing in for the pending public loader."""

    _identity: Mapping[str, object]
    _proof_sha256: str
    _completion: Mapping[str, object]

    def revalidate(self) -> None:
        identity = self.identity_payload()
        completion = self.completion_document()
        if self._proof_sha256 != sha256_json(identity):
            raise ValueError("test validated run proof SHA-256 mismatch")
        if identity["completion_sha256"] != sha256_json(completion):
            raise ValueError("test completion SHA-256 mismatch")
        for field in (
            "branch_id",
            "input_bundle_sha256",
            "protocol_sha256",
            "run_key",
            "scope",
            "seed",
            "split_role",
            "stage",
            "subject",
        ):
            if completion[field] != identity[field]:
                raise ValueError(f"test completion/run proof mismatch for {field}")
        if completion["output_hashes"] != identity["completion_output_hashes"]:
            raise ValueError("test completion output hash mismatch")

    def identity_payload(self) -> dict[str, object]:
        value = _clone(self._identity)
        assert isinstance(value, dict)
        return value

    def completion_document(self) -> dict[str, object]:
        value = _clone(self._completion)
        assert isinstance(value, dict)
        return value

    @property
    def proof_sha256(self) -> str:
        return self._proof_sha256


@dataclass(frozen=True)
class _TestValidatedStage1CostCapability(ValidatedStage1CostCapability):
    """Test-only pure-core fake; no production cost issuer exists here."""

    _identity: Mapping[str, object]
    _proof_sha256: str
    _branch_costs: Mapping[str, object]
    _operator_keys: Mapping[str, object]

    def revalidate(self) -> None:
        identity = self.identity_payload()
        if self._proof_sha256 != sha256_json(identity):
            raise ValueError("test cost capability proof SHA-256 mismatch")
        if identity["branch_measured_ms_per_query"] != _clone(self._branch_costs):
            raise ValueError("test cost capability branch costs mutated")
        expected_keys = [
            {
                "config_id": config.config_id,
                "operator_complexity_key": list(
                    self._operator_keys[config.config_id]  # type: ignore[arg-type]
                ),
            }
            for config in FUSION_GRID
        ]
        if identity["operator_complexity_keys"] != expected_keys:
            raise ValueError("test cost capability operator keys mutated")

    def identity_payload(self) -> dict[str, object]:
        value = _clone(self._identity)
        assert isinstance(value, dict)
        return value

    @property
    def proof_sha256(self) -> str:
        return self._proof_sha256

    def measured_branch_cost(self, branch_id: str) -> float:
        return float(self._branch_costs[branch_id])

    def operator_complexity_key(
        self,
        config_id: str,
    ) -> tuple[int, int, int]:
        raw = self._operator_keys[config_id]
        assert isinstance(raw, tuple)
        return raw  # type: ignore[return-value]


def _metadata(
    *,
    branch_id: str,
    checkpoint_sha256: str,
    config_sha256: str,
    git_sha: str,
    protocol_sha256: str,
    seed: int,
    source_records: list[dict[str, object]],
    subject: int,
    stage: str,
) -> dict[str, object]:
    value: dict[str, object] = {
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": config_sha256,
        "git_sha": git_sha,
        "protocol_sha256": protocol_sha256,
        "seed": seed,
        "source_records": source_records,
        "split_role": "val-dev",
        "stage": stage,
        "subject": subject,
    }
    if branch_id == BRAINRW_BRANCH_ID:
        value.update(_runtime_metadata())
    return value


def _test_proof(
    *,
    artifact: ScoreArtifact,
    branch_id: str,
    input_bundle_sha256: str,
    manifest_sha256: str,
    records_sha256: str,
    role_payload_sha256: str,
    run_key: str,
    run_manifest_sha256: str,
    schedule: Mapping[str, object],
    schedule_sha256: str,
    source_manifest_sha256: str,
    source_payload_sha256: str,
) -> _TestValidatedRunProof:
    is_internvit = branch_id == INTERNVIT_BRANCH_ID
    recipe = INTERNVIT_SEMANTIC if is_internvit else BRAINRW_SEMANTIC
    config_id = INTERNVIT_CONFIG_ID if is_internvit else BRAINRW_CONFIG_ID
    epochs = INTERNVIT_EPOCHS if is_internvit else BRAINRW_EPOCHS
    alignment = common_alignment_payload(artifact)
    semantic_environment = dict(PINNED_SEMANTIC_ENVIRONMENT)
    completion_output_hashes = {
        "final_checkpoint_sha256": artifact.metadata["checkpoint_sha256"],
        "run_manifest_sha256": run_manifest_sha256,
    }
    if is_internvit:
        completion_output_hashes["parity_sha256"] = _digest(
            "baseline-parity",
            subject=artifact.metadata["subject"],
            seed=artifact.metadata["seed"],
            score_payload_sha256=artifact.verified.payload_sha256,
        )
    else:
        completion_output_hashes.update(
            {
                "score_envelope_sha256": (artifact.verified.envelope_sha256),
                "score_payload_sha256": artifact.verified.payload_sha256,
            }
        )
    completion = {
        "artifact_type": "test.sealed_stage1_component_completion",
        "branch_id": branch_id,
        "claim_sha256": _digest(
            "claim",
            branch_id=branch_id,
            subject=artifact.metadata["subject"],
            seed=artifact.metadata["seed"],
        ),
        "input_bundle_sha256": input_bundle_sha256,
        "job_map_sha256": _digest("job-map", branch_id=branch_id),
        "output_hashes": completion_output_hashes,
        "protocol_sha256": artifact.metadata["protocol_sha256"],
        "row_sha256": _digest(
            "job-row",
            branch_id=branch_id,
            subject=artifact.metadata["subject"],
            seed=artifact.metadata["seed"],
        ),
        "run_key": run_key,
        "schema_version": 1,
        "scope": artifact.scope,
        "seed": artifact.metadata["seed"],
        "split_role": artifact.metadata["split_role"],
        "stage": artifact.metadata["stage"],
        "subject": artifact.metadata["subject"],
    }
    source_records = _clone(artifact.metadata["source_records"])
    assert isinstance(source_records, list)
    identity = {
        "alignment": alignment,
        "alignment_sha256": sha256_json(alignment),
        "artifact_type": "samga_brain_rw.validated_component_run_proof",
        "branch_id": branch_id,
        "checkpoint_sha256": artifact.metadata["checkpoint_sha256"],
        "completion_output_hashes": completion_output_hashes,
        "completion_sha256": sha256_json(completion),
        "epochs": epochs,
        "gallery_ids_sha256": artifact.gallery_ids_sha256,
        "git_sha": artifact.metadata["git_sha"],
        "input_bundle_sha256": input_bundle_sha256,
        "manifest_sha256": manifest_sha256,
        "protocol_sha256": artifact.metadata["protocol_sha256"],
        "query_ids_sha256": artifact.query_ids_sha256,
        "recipe_config_id": config_id,
        "recipe_config_sha256": recipe.sha256,
        "records_sha256": records_sha256,
        "resolved_config_sha256": artifact.metadata["config_sha256"],
        "role_payload_sha256": role_payload_sha256,
        "run_key": run_key,
        "run_manifest_sha256": run_manifest_sha256,
        "schedule": _clone(schedule),
        "schedule_sha256": schedule_sha256,
        "schema_version": 1,
        "scope": artifact.scope,
        "score_envelope_sha256": artifact.verified.envelope_sha256,
        "score_payload_sha256": artifact.verified.payload_sha256,
        "seed": artifact.metadata["seed"],
        "semantic_environment": semantic_environment,
        "semantic_environment_sha256": sha256_json(semantic_environment),
        "source_manifest_sha256": source_manifest_sha256,
        "source_payload_sha256": source_payload_sha256,
        "source_records": source_records,
        "source_records_sha256": artifact.metadata["source_records_sha256"],
        "split_role": artifact.metadata["split_role"],
        "stage": artifact.metadata["stage"],
        "subject": artifact.metadata["subject"],
    }
    return _TestValidatedRunProof(
        _identity=identity,
        _proof_sha256=sha256_json(identity),
        _completion=completion,
    )


def _save_component(
    root: Path,
    *,
    branch_id: str,
    matrix: np.ndarray,
    query_ids: Sequence[str],
    gallery_ids: Sequence[str],
    protocol_sha256: str,
    seed: int,
    subject: int,
    stage_override: str | None = None,
) -> Stage1ComponentBinding:
    is_internvit = branch_id == INTERNVIT_BRANCH_ID
    stage = INTERNVIT_STAGE if is_internvit else BRAINRW_STAGE
    if stage_override is not None:
        stage = stage_override
    config_id = INTERNVIT_CONFIG_ID if is_internvit else BRAINRW_CONFIG_ID
    recipe = INTERNVIT_SEMANTIC if is_internvit else BRAINRW_SEMANTIC
    resolved_config_sha256 = (
        _digest(
            "resolved-internvit-config",
            recipe_sha256=recipe.sha256,
            seed=seed,
            subject=subject,
        )
        if is_internvit
        else recipe.sha256
    )
    common_record = _common_source_record(
        subject,
        seed,
        protocol_sha256=protocol_sha256,
    )
    input_bundle_sha256 = sha256_json(
        {
            "branch_id": branch_id,
            "record": common_record,
            "resolved_config_sha256": resolved_config_sha256,
        }
    )
    run_key = make_run_key(
        stage,
        config_id,
        subject,
        seed,
        resolved_config_sha256,
        input_bundle_sha256,
    )
    source_record = (
        {**common_record, "run_key": run_key} if is_internvit else common_record
    )
    checkpoint_sha256 = _digest(
        "terminal-checkpoint",
        branch_id=branch_id,
        resolved_config_sha256=resolved_config_sha256,
        run_key=run_key,
    )
    git_sha = _git_digest("source-tree", branch_id=branch_id)
    directory = root / f"sub-{subject:02d}" / f"seed-{seed}" / branch_id
    ScoreArtifact.save(
        directory,
        matrix,
        query_ids,
        gallery_ids,
        _metadata(
            branch_id=branch_id,
            checkpoint_sha256=checkpoint_sha256,
            config_sha256=resolved_config_sha256,
            git_sha=git_sha,
            protocol_sha256=protocol_sha256,
            seed=seed,
            source_records=[source_record],
            subject=subject,
            stage=stage,
        ),
    )
    artifact = ScoreArtifact.load(
        directory,
        allowed_scopes={"val-dev"},
    )
    schedule, schedule_sha256 = _schedule(branch_id)
    run_manifest_sha256 = _digest(
        "run-manifest",
        branch_id=branch_id,
        checkpoint_sha256=checkpoint_sha256,
        input_bundle_sha256=input_bundle_sha256,
        run_key=run_key,
        schedule_sha256=schedule_sha256,
    )
    proof = _test_proof(
        artifact=artifact,
        branch_id=branch_id,
        input_bundle_sha256=input_bundle_sha256,
        manifest_sha256=str(common_record["manifest_sha256"]),
        records_sha256=str(common_record["records_sha256"]),
        role_payload_sha256=str(common_record["role_payload_sha256"]),
        run_key=run_key,
        run_manifest_sha256=run_manifest_sha256,
        schedule=schedule,
        schedule_sha256=schedule_sha256,
        source_manifest_sha256=str(common_record["source_manifest_sha256"]),
        source_payload_sha256=str(common_record["source_payload_sha256"]),
    )
    return Stage1ComponentBinding(
        branch_id=branch_id,
        score=artifact,
        run_proof=proof,
    )


def _build_cells(
    root: Path,
    *,
    matrix_variant: int = 0,
    protocol_overrides: Mapping[tuple[int, int], str] | None = None,
    ids_override: tuple[int, int] | None = None,
) -> tuple[Stage1CompositionCell, ...]:
    internvit_matrix, brainrw_matrix, query_ids, gallery_ids = _score_matrices(
        variant=matrix_variant
    )
    common_protocol = _digest("locked-val-dev-protocol", version=1)
    cells: list[Stage1CompositionCell] = []
    for subject, seed in PILOT_COORDINATES:
        protocol_sha256 = (protocol_overrides or {}).get(
            (subject, seed), common_protocol
        )
        cell_queries = query_ids
        cell_galleries = gallery_ids
        if ids_override == (subject, seed):
            cell_queries = tuple(f"r{index}" for index in range(6))
            cell_galleries = (*cell_queries, "y0", "y1")
        cell_root = root / f"cell-{subject:02d}-{seed}"
        internvit = _save_component(
            cell_root,
            branch_id=INTERNVIT_BRANCH_ID,
            matrix=internvit_matrix,
            query_ids=cell_queries,
            gallery_ids=cell_galleries,
            protocol_sha256=protocol_sha256,
            seed=seed,
            subject=subject,
        )
        brainrw = _save_component(
            cell_root,
            branch_id=BRAINRW_BRANCH_ID,
            matrix=brainrw_matrix,
            query_ids=cell_queries,
            gallery_ids=cell_galleries,
            protocol_sha256=protocol_sha256,
            seed=seed,
            subject=subject,
        )
        cells.append(
            Stage1CompositionCell(
                subject=subject,
                seed=seed,
                internvit=internvit,
                brainrw=brainrw,
            )
        )
    return tuple(cells)


def _cost_capability(
    cells: Sequence[Stage1CompositionCell],
) -> _TestValidatedStage1CostCapability:
    branch_costs: dict[str, object] = {
        INTERNVIT_BRANCH_ID: 0.2,
        BRAINRW_BRANCH_ID: 0.1,
    }
    family_keys = {
        "temperature_convex": (0, 0, 5),
        "zscore_convex": (0, 6, 9),
        "rrf": (2, 0, 5),
    }
    operator_keys: dict[str, object] = {
        config.config_id: family_keys[config.family] for config in FUSION_GRID
    }
    score_inputs = stage1_module._cost_score_inputs(cells)
    identity = {
        "artifact_type": ("samga_brain_rw.validated_stage1_cost_capability"),
        "branch_measured_ms_per_query": dict(branch_costs),
        "measurement_protocol_sha256": _digest("cost-protocol"),
        "operator_complexity_keys": [
            {
                "config_id": config.config_id,
                "operator_complexity_key": list(
                    operator_keys[config.config_id]  # type: ignore[arg-type]
                ),
            }
            for config in FUSION_GRID
        ],
        "raw_record_sha256": _digest("raw-cost-record"),
        "runtime_evidence_sha256": _digest("cost-runtime-evidence"),
        "schema_version": 1,
        "scope": "val-dev",
        "score_inputs": score_inputs,
        "score_inputs_sha256": sha256_json(score_inputs),
        "two_encoder_count": 2,
    }
    return _TestValidatedStage1CostCapability(
        _identity=identity,
        _proof_sha256=sha256_json(identity),
        _branch_costs=branch_costs,
        _operator_keys=operator_keys,
    )


@pytest.fixture(scope="module")
def stage1_inputs(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[
    tuple[Stage1CompositionCell, ...],
    _TestValidatedStage1CostCapability,
]:
    cells = _build_cells(tmp_path_factory.mktemp("stage1-artifacts"))
    return cells, _cost_capability(cells)


@pytest.fixture(scope="module")
def outcome(
    stage1_inputs: tuple[
        tuple[Stage1CompositionCell, ...],
        _TestValidatedStage1CostCapability,
    ],
) -> Stage1CompositionOutcome:
    cells, costs = stage1_inputs
    return compose_stage1(
        cells,
        semantic_config=FUSION_SEMANTIC,
        cost_capability=costs,
    )


def _metrics_payload(metrics: object) -> dict[str, object]:
    return {
        "gallery_count": metrics.gallery_count,  # type: ignore[attr-defined]
        "query_count": metrics.query_count,  # type: ignore[attr-defined]
        "top1_count": metrics.top1_count,  # type: ignore[attr-defined]
        "top1_rate": metrics.top1_rate,  # type: ignore[attr-defined]
        "top5_count": metrics.top5_count,  # type: ignore[attr-defined]
        "top5_rate": metrics.top5_rate,  # type: ignore[attr-defined]
    }


def _alter_proof_identity(
    proof: _TestValidatedRunProof,
    **changes: object,
) -> _TestValidatedRunProof:
    identity = proof.identity_payload()
    identity.update(changes)
    return replace(
        proof,
        _identity=identity,
        _proof_sha256=sha256_json(identity),
    )


def test_bindings_hold_real_loaded_scores_and_no_caller_metrics(
    stage1_inputs: tuple[
        tuple[Stage1CompositionCell, ...],
        _TestValidatedStage1CostCapability,
    ],
) -> None:
    cells, _ = stage1_inputs
    binding = cells[0].internvit

    assert isinstance(binding.score, ScoreArtifact)
    assert binding.score.verified.payload_sha256 == (binding.score_payload_sha256)
    assert binding.score.verified.envelope_sha256 == (binding.score_envelope_sha256)
    assert binding.score.metrics.top1_rate == 0.5
    assert "top1" not in binding.to_payload()
    assert "top5" not in binding.to_payload()
    assert (
        binding.to_payload()["run_key"]
        == (binding.run_proof.identity_payload()["run_key"])
    )


def test_component_requires_nominal_validated_proof_not_hash_mapping(
    stage1_inputs: tuple[
        tuple[Stage1CompositionCell, ...],
        _TestValidatedStage1CostCapability,
    ],
) -> None:
    cells, _ = stage1_inputs
    binding = cells[0].internvit

    with pytest.raises(TypeError, match="ValidatedComponentRunProof"):
        Stage1ComponentBinding(
            branch_id=INTERNVIT_BRANCH_ID,
            score=binding.score,
            run_proof=binding.run_proof.identity_payload(),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("protocol_sha256", _digest("fake-protocol")),
        ("source_records_sha256", _digest("fake-source")),
        ("scope", "val-confirm"),
        ("split_role", "train"),
        ("score_payload_sha256", _digest("fake-score")),
        ("resolved_config_sha256", _digest("fake-resolved-config")),
        ("checkpoint_sha256", _digest("fake-checkpoint")),
        ("alignment_sha256", _digest("fake-alignment")),
    ],
)
def test_binding_rejects_fake_or_cross_bound_proof_identity(
    stage1_inputs: tuple[
        tuple[Stage1CompositionCell, ...],
        _TestValidatedStage1CostCapability,
    ],
    field: str,
    replacement: object,
) -> None:
    cells, _ = stage1_inputs
    original = cells[0].internvit
    assert isinstance(original.run_proof, _TestValidatedRunProof)
    forged = _alter_proof_identity(
        original.run_proof,
        **{field: replacement},
    )

    with pytest.raises(ValueError, match=field.replace("_", " ") + "|mismatch"):
        Stage1ComponentBinding(
            branch_id=INTERNVIT_BRANCH_ID,
            score=original.score,
            run_proof=forged,
        )


def test_binding_rejects_bad_capability_hash_and_branch_exchange(
    stage1_inputs: tuple[
        tuple[Stage1CompositionCell, ...],
        _TestValidatedStage1CostCapability,
    ],
) -> None:
    cells, _ = stage1_inputs
    internvit = cells[0].internvit
    brainrw = cells[0].brainrw
    assert isinstance(internvit.run_proof, _TestValidatedRunProof)

    bad_hash = replace(
        internvit.run_proof,
        _proof_sha256=_digest("unrelated-proof"),
    )
    with pytest.raises(ValueError, match="proof SHA-256"):
        replace(internvit, run_proof=bad_hash)

    with pytest.raises(ValueError, match="branch|stage"):
        Stage1ComponentBinding(
            branch_id=INTERNVIT_BRANCH_ID,
            score=brainrw.score,
            run_proof=brainrw.run_proof,
        )


def test_binding_reloads_envelope_to_reject_replaced_in_memory_metadata(
    stage1_inputs: tuple[
        tuple[Stage1CompositionCell, ...],
        _TestValidatedStage1CostCapability,
    ],
) -> None:
    cells, _ = stage1_inputs
    original = cells[0].internvit
    assert isinstance(original.run_proof, _TestValidatedRunProof)
    forged_git = _git_digest("forged-in-memory-metadata")
    metadata = _clone(original.score.metadata)
    provenance = _clone(original.score.provenance)
    assert isinstance(metadata, dict)
    assert isinstance(provenance, dict)
    metadata["git_sha"] = forged_git
    provenance["git_sha"] = forged_git
    forged_score = replace(
        original.score,
        metadata=metadata,
        provenance=provenance,
    )
    forged_proof = _alter_proof_identity(
        original.run_proof,
        git_sha=forged_git,
    )

    with pytest.raises(ValueError, match="loaded|envelope|in-memory"):
        Stage1ComponentBinding(
            branch_id=INTERNVIT_BRANCH_ID,
            score=forged_score,
            run_proof=forged_proof,
        )


def test_internvit_resolved_config_may_vary_but_recipe_is_locked(
    stage1_inputs: tuple[
        tuple[Stage1CompositionCell, ...],
        _TestValidatedStage1CostCapability,
    ],
    outcome: Stage1CompositionOutcome,
) -> None:
    cells, _ = stage1_inputs
    resolved = {cell.internvit.resolved_config_sha256 for cell in cells}
    recipes = {cell.internvit.recipe_config_sha256 for cell in cells}

    assert len(resolved) == 6
    assert recipes == {INTERNVIT_SEMANTIC.sha256}
    spec = outcome.composition_spec_payload()
    assert spec["components"]["internvit"]["recipe_config_sha256"] == (
        INTERNVIT_SEMANTIC.sha256
    )


def test_cell_calls_real_alignment_and_locks_exact_terminal_stages(
    tmp_path: Path,
) -> None:
    internvit_matrix, _brainrw_matrix, query_ids, gallery_ids = _score_matrices()
    protocol = _digest("alignment-stage-protocol")
    with pytest.raises(ValueError, match="stage0|terminal stage"):
        _save_component(
            tmp_path / "stage2",
            branch_id=INTERNVIT_BRANCH_ID,
            matrix=internvit_matrix,
            query_ids=query_ids,
            gallery_ids=gallery_ids,
            protocol_sha256=protocol,
            seed=42,
            subject=1,
            stage_override="stage2",
        )


def test_cross_cell_protocol_and_ordered_ids_must_match(
    tmp_path: Path,
) -> None:
    drift_coordinate = PILOT_COORDINATES[-1]
    protocol_cells = _build_cells(
        tmp_path / "protocol-drift",
        protocol_overrides={drift_coordinate: _digest("different-protocol")},
    )
    with pytest.raises(ValueError, match="protocol"):
        compose_stage1(
            protocol_cells,
            semantic_config=FUSION_SEMANTIC,
            cost_capability=_cost_capability(protocol_cells),
        )

    ids_cells = _build_cells(
        tmp_path / "ids-drift",
        ids_override=drift_coordinate,
    )
    with pytest.raises(ValueError, match="query|gallery|ordered"):
        compose_stage1(
            ids_cells,
            semantic_config=FUSION_SEMANTIC,
            cost_capability=_cost_capability(ids_cells),
        )


def test_cost_capability_binds_exact_six_inputs_and_preregistered_keys(
    stage1_inputs: tuple[
        tuple[Stage1CompositionCell, ...],
        _TestValidatedStage1CostCapability,
    ],
) -> None:
    cells, costs = stage1_inputs
    identity = costs.identity_payload()

    assert costs.proof_sha256 == sha256_json(identity)
    assert identity["artifact_type"] == (
        "samga_brain_rw.validated_stage1_cost_capability"
    )
    assert len(identity["operator_complexity_keys"]) == 47
    assert len(identity["score_inputs"]) == 6
    assert identity["score_inputs_sha256"] == sha256_json(identity["score_inputs"])
    assert (
        identity["score_inputs"][0]["internvit"]["score_payload_sha256"]
        == cells[0].internvit.score.verified.payload_sha256
    )
    assert (
        identity["score_inputs"][0]["internvit"]["run_proof_sha256"]
        == cells[0].internvit.run_proof.proof_sha256
    )
    assert identity["score_inputs"][0]["query_count"] == len(
        cells[0].internvit.score.query_ids
    )
    assert identity["score_inputs"][0]["gallery_count"] == len(
        cells[0].internvit.score.gallery_ids
    )
    assert costs.measured_branch_cost(INTERNVIT_BRANCH_ID) == 0.2
    assert costs.measured_branch_cost(BRAINRW_BRANCH_ID) == 0.1
    assert costs.operator_complexity_key(TARGET_FUSION_ID) == (0, 0, 5)
    assert identity["two_encoder_count"] == 2


def test_compose_rejects_mapping_and_raw_cost_record(
    stage1_inputs: tuple[
        tuple[Stage1CompositionCell, ...],
        _TestValidatedStage1CostCapability,
    ],
) -> None:
    cells, costs = stage1_inputs
    with pytest.raises(TypeError, match="ValidatedStage1CostCapability"):
        compose_stage1(
            cells,
            semantic_config=FUSION_SEMANTIC,
            cost_capability=costs.identity_payload(),  # type: ignore[arg-type]
        )
    raw_record = object.__new__(RawStage1CostRecord)
    with pytest.raises(TypeError, match="ValidatedStage1CostCapability"):
        compose_stage1(
            cells,
            semantic_config=FUSION_SEMANTIC,
            cost_capability=raw_record,  # type: ignore[arg-type]
        )
    assert "Stage1CostEvidence" not in vars(stage1_module)


def test_compose_recomputes_both_branches_and_complete_47_by_6_grid(
    stage1_inputs: tuple[
        tuple[Stage1CompositionCell, ...],
        _TestValidatedStage1CostCapability,
    ],
    outcome: Stage1CompositionOutcome,
) -> None:
    cells, costs = stage1_inputs
    selection = outcome.selection_payload()
    branch_results = {
        value["branch_id"]: value for value in selection["branch_results"]
    }
    fusion_results = {
        value["config_id"]: value for value in selection["fusion_results"]
    }

    assert len(fusion_results) == 47
    assert all(len(value["cells"]) == 6 for value in fusion_results.values())
    for index, cell in enumerate(cells):
        assert branch_results[INTERNVIT_BRANCH_ID]["cells"][index][
            "metrics"
        ] == _metrics_payload(
            independent_retrieval_metrics(
                cell.internvit.score.similarity,
                cell.internvit.score.query_ids,
                cell.internvit.score.gallery_ids,
            )
        )
        assert branch_results[BRAINRW_BRANCH_ID]["cells"][index][
            "metrics"
        ] == _metrics_payload(
            independent_retrieval_metrics(
                cell.brainrw.score.similarity,
                cell.brainrw.score.query_ids,
                cell.brainrw.score.gallery_ids,
            )
        )
        for config in FUSION_GRID:
            direct = independent_retrieval_metrics(
                config.apply(
                    cell.internvit.score.similarity,
                    cell.brainrw.score.similarity,
                    gallery_ids=cell.internvit.score.gallery_ids,
                ),
                cell.internvit.score.query_ids,
                cell.internvit.score.gallery_ids,
            )
            observed = fusion_results[config.config_id]["cells"][index]
            assert observed["metrics"] == _metrics_payload(direct)
            assert observed["operator_complexity_key"] == list(
                costs.operator_complexity_key(config.config_id)
            )
            assert observed["two_encoder_count"] == 2
            assert "measured_inference_cost" not in observed
            assert observed["score_inputs"]["alignment_sha256"] == (
                cell.alignment_sha256
            )
            assert (
                observed["score_inputs"]["internvit"]["score_payload_sha256"]
                == cell.internvit.score.verified.payload_sha256
            )
            assert (
                observed["score_inputs"]["brainrw"]["score_envelope_sha256"]
                == cell.brainrw.score.verified.envelope_sha256
            )


def test_global_control_runs_once_and_fusion_uses_exact_tie_key(
    monkeypatch: pytest.MonkeyPatch,
    stage1_inputs: tuple[
        tuple[Stage1CompositionCell, ...],
        _TestValidatedStage1CostCapability,
    ],
) -> None:
    cells, costs = stage1_inputs
    calls = {"control": 0}
    original_control = stage1_module.select_global_stronger_single_branch

    def control_once(values: object) -> str:
        calls["control"] += 1
        return original_control(values)

    monkeypatch.setattr(
        stage1_module,
        "select_global_stronger_single_branch",
        control_once,
    )
    result = compose_stage1(
        cells,
        semantic_config=FUSION_SEMANTIC,
        cost_capability=costs,
    )

    assert calls == {"control": 1}
    assert result.control_branch_id == BRAINRW_BRANCH_ID
    fusion_results = result.selection_payload()["fusion_results"]
    expected_winner = min(
        fusion_results,
        key=lambda item: (
            -sum(cell["metrics"]["top1_rate"] for cell in item["cells"]) / 6,
            -sum(cell["metrics"]["top5_rate"] for cell in item["cells"]) / 6,
            tuple(item["operator_complexity_key"]),
            item["config_id"],
        ),
    )
    assert result.winner_config_id == expected_winner["config_id"]
    assert result.winner_config_id == "s1-temp-ti050-tc050-a050"
    assert result.gate.passed


def test_actual_semantic_config_and_full_selection_semantics_are_bound(
    outcome: Stage1CompositionOutcome,
) -> None:
    selection = outcome.selection_payload()
    spec = outcome.composition_spec_payload()

    assert FUSION_SEMANTIC.sha256 == STAGE1_FUSION_CONFIG_SHA256
    assert selection["semantic_config_sha256"] == FUSION_SEMANTIC.sha256
    assert selection["semantic_config"] == (FUSION_SEMANTIC.canonical_payload())
    assert spec["fusion"]["semantic_config_sha256"] == (FUSION_SEMANTIC.sha256)
    assert spec["fusion"]["semantic_config"]["candidates"] == [
        config.to_dict() for config in FUSION_GRID
    ]
    assert spec["selection"]["fusion_tie_break"] == [
        "macro_mean_top1",
        "macro_mean_top5",
        "operator_complexity_key",
        "config_id",
    ]
    assert spec["gate"] == {
        "mean_top1_delta_minimum": "0.003",
        "mean_top5_delta_minimum": "-0.002",
        "positive_cells_minimum": 4,
        "subject_top1_delta_floor": "-0.02",
        "winner_only": True,
    }


def test_composition_spec_stores_distinct_full_60_and_25_epoch_schedules(
    outcome: Stage1CompositionOutcome,
) -> None:
    spec = outcome.composition_spec_payload()
    internvit = spec["components"]["internvit"]
    brainrw = spec["components"]["brainrw"]

    assert internvit["epochs"] == 60
    assert brainrw["epochs"] == 25
    assert internvit["schedule"]["task"]["epochs"] == 60
    assert brainrw["schedule"]["training"]["epochs"] == 25
    assert internvit["schedule_sha256"] == sha256_json(internvit["schedule"])
    assert brainrw["schedule_sha256"] == sha256_json(brainrw["schedule"])
    assert internvit["schedule_sha256"] != brainrw["schedule_sha256"]
    json.dumps(spec, allow_nan=False, sort_keys=True)


def test_full_loser_grid_and_complexity_are_in_selection_and_evidence_hashes(
    outcome: Stage1CompositionOutcome,
) -> None:
    selection = outcome.selection_payload()
    loser_index = next(
        index
        for index, value in enumerate(selection["fusion_results"])
        if value["config_id"] != outcome.winner_config_id
    )
    changed = _clone(selection)
    assert isinstance(changed, dict)
    changed["fusion_results"][loser_index]["cells"][0]["metrics"]["top1_count"] += 1
    changed["fusion_results"][loser_index]["operator_complexity_key"][0] += 1

    assert sha256_json(changed) != outcome.selection_sha256
    with pytest.raises(TypeError, match="issuer|compose_stage1"):
        replace(outcome, _selection_payload=changed)
    evidence = outcome.evidence_payload()
    assert evidence["selection"] == selection
    assert evidence["selection_sha256"] == outcome.selection_sha256


def test_real_nonranking_score_change_changes_outcome_evidence(
    tmp_path: Path,
    outcome: Stage1CompositionOutcome,
) -> None:
    changed_cells = _build_cells(
        tmp_path / "changed-real-score",
        matrix_variant=1,
    )
    changed = compose_stage1(
        changed_cells,
        semantic_config=FUSION_SEMANTIC,
        cost_capability=_cost_capability(changed_cells),
    )

    assert changed.control_branch_id == outcome.control_branch_id
    assert changed.winner_config_id == outcome.winner_config_id
    assert changed.winner_top1 == outcome.winner_top1
    assert changed.winner_top5 == outcome.winner_top5
    assert changed.evidence_sha256 != outcome.evidence_sha256


def test_gate_uses_inclusive_exact_four_thresholds() -> None:
    control_top1 = (0.50,) * 6
    control_top5 = (0.90,) * 6
    deltas = (0.0145, 0.0145, 0.0145, 0.0145, -0.02, -0.02)
    winner_top1 = tuple(
        baseline + delta for baseline, delta in zip(control_top1, deltas, strict=True)
    )
    winner_top5 = (0.898,) * 6

    gate = _score_composition_gate(
        coordinates=PILOT_COORDINATES,
        winner_top1=winner_top1,
        winner_top5=winner_top5,
        control_top1=control_top1,
        control_top5=control_top5,
    )

    assert gate.passed
    assert gate.mean_top1_delta == pytest.approx(0.003)
    assert gate.mean_top5_delta == pytest.approx(-0.002)
    assert gate.positive_cells == 4
    assert gate.worst_subject_top1_delta == pytest.approx(-0.02)
    assert all(gate.criteria.values())

    below = list(winner_top1)
    below[0] -= 0.000001
    failed = _score_composition_gate(
        coordinates=PILOT_COORDINATES,
        winner_top1=tuple(below),
        winner_top5=winner_top5,
        control_top1=control_top1,
        control_top5=control_top5,
    )
    assert not failed.passed
    assert not failed.criteria["mean_top1_delta"]


def test_outcome_recomputes_every_body_and_candidate_decision(
    outcome: Stage1CompositionOutcome,
) -> None:
    assert outcome.dependency_sha256 == sha256_json(outcome.dependency_payload())
    assert outcome.control_sha256 == sha256_json(outcome.control_payload())
    assert outcome.selection_sha256 == sha256_json(outcome.selection_payload())
    assert outcome.composite_sha256 == sha256_json(outcome.composite_payload())
    assert outcome.composition_spec_sha256 == sha256_json(
        outcome.composition_spec_payload()
    )
    assert outcome.evidence_sha256 == sha256_json(outcome.evidence_payload())
    decision = CandidateDecision.from_document(outcome.candidate_decision_document())
    assert decision.candidate_id == outcome.winner_config_id
    assert decision.control_id == outcome.control_branch_id
    assert decision.config_sha256 == FUSION_SEMANTIC.sha256
    assert decision.hyperparameters_sha256 == sha256_json(
        next(
            config.to_dict()
            for config in FUSION_GRID
            if config.config_id == outcome.winner_config_id
        )
    )
    assert decision.schedule_sha256 == outcome.composition_spec_sha256
    assert decision.candidate_matrix_sha256 == outcome.composite_sha256
    assert decision.control_matrix_sha256 == outcome.control_sha256
    assert decision.gate.to_payload() == outcome.gate.to_payload()
    assert len(decision.component_sha256s) == 12
    json.dumps(outcome.to_payload(), allow_nan=False, sort_keys=True)


def test_outcome_rejects_public_construction_and_dataclass_replace(
    outcome: Stage1CompositionOutcome,
) -> None:
    constructor_values = {
        item.name: getattr(outcome, item.name) for item in fields(outcome)
    }

    with pytest.raises(TypeError, match="issuer|compose_stage1"):
        Stage1CompositionOutcome(**constructor_values)
    with pytest.raises(TypeError, match="issuer|compose_stage1"):
        replace(outcome, winner_top1=outcome.winner_top1)


def test_cost_identity_rejects_rehashed_empty_six_input_list(
    outcome: Stage1CompositionOutcome,
) -> None:
    dependency = outcome.dependency_payload()
    _, bindings = outcome._validate_dependency(dependency)
    identity = outcome.selection_payload()["cost_capability_identity"]
    identity["score_inputs"] = [{} for _ in PILOT_COORDINATES]
    identity["score_inputs_sha256"] = sha256_json(identity["score_inputs"])

    with pytest.raises(ValueError, match="exact six score inputs"):
        stage1_module._validate_serialized_cost_score_inputs(
            identity,
            bindings,
            dependency,
        )


@pytest.mark.parametrize(
    "field_name",
    ["config_id", "config_sha256", "schedule_sha256"],
)
def test_control_rejects_rehashed_selected_dependency_drift(
    outcome: Stage1CompositionOutcome,
    field_name: str,
) -> None:
    dependency = outcome.dependency_payload()
    _, bindings = outcome._validate_dependency(dependency)
    selection = outcome.selection_payload()
    selected = next(
        item
        for item in selection["branch_results"]
        if item["branch_id"] == outcome.control_branch_id
    )
    control = outcome.control_payload()
    control[field_name] = (
        "forged-config-id"
        if field_name == "config_id"
        else _digest(f"forged-control-{field_name}")
    )

    with pytest.raises(ValueError, match="control evidence.*cross-binding"):
        outcome._validate_control(control, selected, bindings)


def test_composite_rejects_config_body_different_from_selected_winner(
    outcome: Stage1CompositionOutcome,
) -> None:
    selection = outcome.selection_payload()
    selected = next(
        item
        for item in selection["fusion_results"]
        if item["config_id"] == outcome.winner_config_id
    )
    composite = outcome.composite_payload()
    composite["config"] = next(
        config.to_dict()
        for config in FUSION_GRID
        if config.config_id != outcome.winner_config_id
    )

    with pytest.raises(ValueError, match="composite.*cross-binding"):
        outcome._validate_composite(composite, selected)


def test_outcome_replace_rejects_nan_empty_decision_and_body_drift(
    outcome: Stage1CompositionOutcome,
) -> None:
    with pytest.raises(TypeError, match="issuer|compose_stage1"):
        replace(
            outcome,
            winner_top1=(float("nan"), *outcome.winner_top1[1:]),
        )
    with pytest.raises(TypeError, match="issuer|compose_stage1"):
        replace(outcome, candidate_decision_payload={})

    spec = outcome.composition_spec_payload()
    spec["components"]["brainrw"]["epochs"] = 26
    with pytest.raises(TypeError, match="issuer|compose_stage1"):
        replace(outcome, _composition_spec_payload=spec)

    dependency = outcome.dependency_payload()
    dependency["cells"][0]["internvit_binding_sha256"] = _digest("drifted-binding")
    with pytest.raises(TypeError, match="issuer|compose_stage1"):
        replace(outcome, _dependency_payload=dependency)


def test_outcome_rejects_rehashed_extra_spec_field_and_nested_proof_drift(
    outcome: Stage1CompositionOutcome,
) -> None:
    spec = outcome.composition_spec_payload()
    spec["unregistered_semantics"] = {"winner": "caller-choice"}
    with pytest.raises(TypeError, match="issuer|compose_stage1"):
        replace(
            outcome,
            composition_spec_sha256=sha256_json(spec),
            _composition_spec_payload=spec,
        )

    dependency = outcome.dependency_payload()
    cell = dependency["cells"][0]
    binding = cell["internvit_binding"]
    binding["score_payload_sha256"] = _digest("forged-serialized-score-payload")
    cell["internvit_binding_sha256"] = sha256_json(binding)
    cell["dependency"]["internvit_binding_sha256"] = cell["internvit_binding_sha256"]
    cell["dependency_sha256"] = sha256_json(cell["dependency"])
    with pytest.raises(TypeError, match="issuer|compose_stage1"):
        replace(
            outcome,
            dependency_sha256=sha256_json(dependency),
            _dependency_payload=dependency,
        )


def test_outcome_rejects_parseable_but_cross_bound_candidate_decision(
    outcome: Stage1CompositionOutcome,
) -> None:
    decision = CandidateDecision.from_document(outcome.candidate_decision_document())
    different = replace(
        decision,
        candidate_id=next(
            config.config_id
            for config in FUSION_GRID
            if config.config_id != outcome.winner_config_id
        ),
    )

    with pytest.raises(TypeError, match="issuer|compose_stage1"):
        replace(
            outcome,
            candidate_decision_payload=different.to_document(),
        )


def test_compose_rejects_semantic_config_id_or_forged_body(
    stage1_inputs: tuple[
        tuple[Stage1CompositionCell, ...],
        _TestValidatedStage1CostCapability,
    ],
) -> None:
    cells, costs = stage1_inputs
    with pytest.raises(TypeError, match="SemanticConfig"):
        compose_stage1(
            cells,
            semantic_config="stage1_fusion_v1",  # type: ignore[arg-type]
            cost_capability=costs,
        )

    forged_payload = FUSION_SEMANTIC.canonical_payload()
    forged_payload["selection"]["metric_tie_break"] = [  # type: ignore[index]
        "config_id",
        "lower_compute",
    ]
    forged = SemanticConfig(
        _canonical=canonical_json_bytes(forged_payload).decode("utf-8")
    )
    with pytest.raises(ValueError, match="semantic|SHA-256|locked"):
        compose_stage1(
            cells,
            semantic_config=forged,
            cost_capability=costs,
        )
