"""Pure Stage 1 composition from verified branch score artifacts.

The composition boundary deliberately does not accept branch metrics or
fusion observations.  It accepts two loaded :class:`ScoreArtifact` values per
pilot cell, plus a nominal capability issued by an upstream run/completion
verifier.  Branch and fusion metrics are then independently recomputed here.

The public run-proof and cost-proof finalizers are intentionally outside this
module.  ``ValidatedComponentRunProof`` and
``ValidatedStage1CostCapability`` are nominal consumer interfaces: ordinary
dictionaries, raw timing records, and caller-supplied digest fields are not
proof capabilities.  This module deliberately ships no production cost
capability implementation or factory.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from numbers import Integral, Real
from types import MappingProxyType
from typing import Literal

import numpy as np

from .config import SemanticConfig, make_run_key
from .fusion import (
    BranchValidationMetrics,
    assert_aligned,
    common_alignment_payload,
    enumerate_stage1_configs,
    select_global_stronger_single_branch,
)
from .hashing import sha256_json
from .registry import CandidateDecision
from .scores import (
    SCORE_PAYLOAD_TYPE,
    RetrievalMetrics,
    ScoreArtifact,
    independent_retrieval_metrics,
)
from .statistics import GateDecision


INTERNVIT_BRANCH_ID = "internvit"
BRAINRW_BRANCH_ID = "brainrw"
INTERNVIT_CONFIG_ID = "internvit_baseline_v1"
BRAINRW_CONFIG_ID = "brainrw_clip_lora_v1"
INTERNVIT_STAGE = "stage0"
BRAINRW_STAGE = "brainrw-clip-lora"
INTERNVIT_EPOCHS = 60
BRAINRW_EPOCHS = 25

INTERNVIT_RECIPE_CONFIG_SHA256 = (
    "db3a696a31ceba0699c4039ca73130f75edd7d2ad69ce3e55c9ab7a5ecfc27de"
)
BRAINRW_RECIPE_CONFIG_SHA256 = (
    "e723a86d33af92afd7421f88e8f7c050ccaefad11d43ad3e4e395ba2e924ed83"
)
STAGE1_FUSION_CONFIG_SHA256 = (
    "27cd33027c2fa322121f0c42732d2ecbe62f2544d6497652ae40f90b7dd8dc78"
)

PILOT_SUBJECTS = (1, 5, 8)
PILOT_SEEDS = (42, 43)
PILOT_COORDINATES = tuple(
    (subject, seed) for subject in PILOT_SUBJECTS for seed in PILOT_SEEDS
)

COMPONENT_BINDING_TYPE = "samga_brain_rw.stage1_component_binding"
CELL_DEPENDENCY_TYPE = "samga_brain_rw.stage1_cell_dependency"
DEPENDENCY_SET_TYPE = "samga_brain_rw.stage1_dependency_set"
VALIDATED_COST_CAPABILITY_TYPE = "samga_brain_rw.validated_stage1_cost_capability"
_RETIRED_COST_EVIDENCE_TYPE = "samga_brain_rw.stage1_cost_evidence"
CONTROL_EVIDENCE_TYPE = "samga_brain_rw.stage1_control_evidence"
SELECTION_EVIDENCE_TYPE = "samga_brain_rw.stage1_selection_evidence"
COMPOSITE_TYPE = "samga_brain_rw.stage1_score_composite"
COMPOSITION_SPEC_TYPE = "samga_brain_rw.stage1_composition_spec"
COMPOSITION_EVIDENCE_TYPE = "samga_brain_rw.stage1_composition_evidence"
COMPOSITION_OUTCOME_TYPE = "samga_brain_rw.stage1_composition_outcome"
_OUTCOME_ISSUER_TOKEN = object()
VALIDATED_RUN_PROOF_TYPE = "samga_brain_rw.validated_component_run_proof"
COMPONENT_SCHEDULE_TYPE = "samga_brain_rw.stage1_component_schedule"

Stage1BranchId = Literal["internvit", "brainrw"]
Stage1Status = Literal["passed", "failed"]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_BRANCH_IDS = (INTERNVIT_BRANCH_ID, BRAINRW_BRANCH_ID)
_BRANCH_SPECS: dict[str, tuple[str, str, int, str]] = {
    INTERNVIT_BRANCH_ID: (
        INTERNVIT_CONFIG_ID,
        INTERNVIT_STAGE,
        INTERNVIT_EPOCHS,
        INTERNVIT_RECIPE_CONFIG_SHA256,
    ),
    BRAINRW_BRANCH_ID: (
        BRAINRW_CONFIG_ID,
        BRAINRW_STAGE,
        BRAINRW_EPOCHS,
        BRAINRW_RECIPE_CONFIG_SHA256,
    ),
}
_RUN_PROOF_KEYS = frozenset(
    {
        "alignment",
        "alignment_sha256",
        "artifact_type",
        "branch_id",
        "checkpoint_sha256",
        "completion_output_hashes",
        "completion_sha256",
        "epochs",
        "gallery_ids_sha256",
        "git_sha",
        "input_bundle_sha256",
        "manifest_sha256",
        "protocol_sha256",
        "query_ids_sha256",
        "recipe_config_id",
        "recipe_config_sha256",
        "records_sha256",
        "resolved_config_sha256",
        "role_payload_sha256",
        "run_key",
        "run_manifest_sha256",
        "schedule",
        "schedule_sha256",
        "schema_version",
        "scope",
        "score_envelope_sha256",
        "score_payload_sha256",
        "seed",
        "semantic_environment",
        "semantic_environment_sha256",
        "source_manifest_sha256",
        "source_payload_sha256",
        "source_records",
        "source_records_sha256",
        "split_role",
        "stage",
        "subject",
    }
)
_SERIALIZED_BINDING_KEYS = (_RUN_PROOF_KEYS - {"artifact_type", "schema_version"}) | {
    "artifact_type",
    "run_proof_sha256",
    "schema_version",
}
_PROOF_SHA_FIELDS = (
    "alignment_sha256",
    "checkpoint_sha256",
    "completion_sha256",
    "gallery_ids_sha256",
    "input_bundle_sha256",
    "manifest_sha256",
    "protocol_sha256",
    "query_ids_sha256",
    "recipe_config_sha256",
    "records_sha256",
    "resolved_config_sha256",
    "role_payload_sha256",
    "run_manifest_sha256",
    "schedule_sha256",
    "score_envelope_sha256",
    "score_payload_sha256",
    "semantic_environment_sha256",
    "source_manifest_sha256",
    "source_payload_sha256",
    "source_records_sha256",
)
_METRIC_KEYS = frozenset(
    {
        "gallery_count",
        "query_count",
        "top1_count",
        "top1_rate",
        "top5_count",
        "top5_rate",
    }
)
_COST_PROTOCOL_KEYS = frozenset(
    {
        "aggregation",
        "artifact_type",
        "cuda_synchronize",
        "measured_runs",
        "raw_unit",
        "runtime_evidence_sha256",
        "schema_version",
        "scope",
        "timing_boundary",
        "unit",
        "warmup_runs",
    }
)
_GATE_SPEC = {
    "mean_top1_delta_minimum": "0.003",
    "mean_top5_delta_minimum": "-0.002",
    "positive_cells_minimum": 4,
    "subject_top1_delta_floor": "-0.02",
    "winner_only": True,
}
_SELECTION_SPEC = {
    "control_tie_break": [
        "macro_mean_top1",
        "macro_mean_top5",
        "lower_branch_measured_ms_per_query",
        "branch_id",
    ],
    "fusion_tie_break": [
        "macro_mean_top1",
        "macro_mean_top5",
        "operator_complexity_key",
        "config_id",
    ],
    "global_control_once": True,
    "global_fusion_winner_once": True,
}
_EXPECTED_SEMANTIC_SELECTION = {
    "branch_score_tie_break": "gallery_id_utf8_bytewise",
    "constant_row": "all_zero",
    "final_score_tie_break": "gallery_id_utf8_bytewise",
    "metric_tie_break": ["lower_compute", "config_id"],
    "retrieval": "standard_independent_cosine",
    "scope": "val-dev",
    "temperature_softmax": False,
    "zscore_variance": "population_ddof0",
}


def _require_exact_keys(
    value: Mapping[str, object],
    expected: frozenset[str],
    context: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{context} keys mismatch: missing={missing}, extra={extra}")


def _require_sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256")
    return value


def _require_git_sha(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _GIT_SHA_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase Git digest")
    return value


def _require_safe_id(value: object, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or _SAFE_ID_RE.fullmatch(value) is None
        or value in {".", ".."}
    ):
        raise ValueError(f"{field_name} must be a safe nonempty identifier")
    return value


def _require_coordinate(subject: object, seed: object) -> tuple[int, int]:
    if (
        isinstance(subject, bool)
        or not isinstance(subject, int)
        or not 1 <= subject <= 10
    ):
        raise ValueError("subject must be an integer in 1..10")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return subject, seed


def _require_nonnegative_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError(f"{field_name} must be a nonnegative integer")
    return int(value)


def _require_positive_integer(value: object, field_name: str) -> int:
    result = _require_nonnegative_integer(value, field_name)
    if result == 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return result


def _require_rate(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be a finite rate in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field_name} must be a finite rate in [0, 1]")
    return result


def _require_cost(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be finite and nonnegative")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field_name} must be finite and nonnegative")
    return result


def _deep_freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(child) for key, child in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(child) for child in value)
    return value


def _deep_thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(child) for child in value]
    return value


def _mapping_payload(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    result = _deep_thaw(value)
    if not isinstance(result, dict):
        raise AssertionError(f"{context} did not thaw to a mapping")
    # This also rejects non-JSON values and non-finite floats.
    sha256_json(result)
    return result


def _sequence_payload(value: object, context: str) -> list[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{context} must be a sequence")
    result = _deep_thaw(tuple(value))
    if not isinstance(result, list):
        raise AssertionError(f"{context} did not thaw to a list")
    sha256_json(result)
    return result


def _cell_id(subject: int, seed: int) -> str:
    return f"{subject:02d}/{seed}"


def _metric_payload(metrics: RetrievalMetrics) -> dict[str, object]:
    if not isinstance(metrics, RetrievalMetrics):
        raise TypeError("metrics must be independently computed RetrievalMetrics")
    return {
        "gallery_count": metrics.gallery_count,
        "query_count": metrics.query_count,
        "top1_count": metrics.top1_count,
        "top1_rate": metrics.top1_rate,
        "top5_count": metrics.top5_count,
        "top5_rate": metrics.top5_rate,
    }


def _validate_metric_payload(
    value: object,
    context: str,
) -> dict[str, object]:
    payload = _mapping_payload(value, context)
    _require_exact_keys(payload, _METRIC_KEYS, context)
    query_count = _require_positive_integer(
        payload["query_count"], f"{context}.query_count"
    )
    gallery_count = _require_positive_integer(
        payload["gallery_count"], f"{context}.gallery_count"
    )
    top1_count = _require_nonnegative_integer(
        payload["top1_count"], f"{context}.top1_count"
    )
    top5_count = _require_nonnegative_integer(
        payload["top5_count"], f"{context}.top5_count"
    )
    top1_rate = _require_rate(payload["top1_rate"], f"{context}.top1_rate")
    top5_rate = _require_rate(payload["top5_rate"], f"{context}.top5_rate")
    if top1_count > top5_count or top5_count > query_count:
        raise ValueError(f"{context} has impossible Top-1/Top-5 counts")
    if top1_rate != top1_count / query_count:
        raise ValueError(f"{context} Top-1 rate/count mismatch")
    if top5_rate != top5_count / query_count:
        raise ValueError(f"{context} Top-5 rate/count mismatch")
    if top5_rate < top1_rate:
        raise ValueError(f"{context} Top-5 must be >= Top-1")
    return {
        "gallery_count": gallery_count,
        "query_count": query_count,
        "top1_count": top1_count,
        "top1_rate": top1_rate,
        "top5_count": top5_count,
        "top5_rate": top5_rate,
    }


class ValidatedComponentRunProof(ABC):
    """Nominal capability returned by a trusted run/completion verifier.

    The Stage 1 module only consumes this interface.  It does not provide a
    public constructor from paths, mappings, or digest strings.
    """

    @abstractmethod
    def revalidate(self) -> None:
        """Recheck the upstream run, completion, and output capabilities."""

    @abstractmethod
    def identity_payload(self) -> dict[str, object]:
        """Return the JSON-native identity captured by the verifier."""

    @property
    @abstractmethod
    def proof_sha256(self) -> str:
        """Digest of :meth:`identity_payload`."""


class ValidatedStage1CostCapability(ABC):
    """Nominal capability issued by a future trusted cost finalizer.

    Stage 1 only consumes and revalidates this interface.  No production
    implementation, mapping constructor, raw-record constructor, or factory is
    issued by this module.
    """

    @abstractmethod
    def revalidate(self) -> None:
        """Recheck trusted runtime, protocol, model/run, and raw-record proof."""

    @abstractmethod
    def identity_payload(self) -> dict[str, object]:
        """Return the exact six-input cost identity captured by the issuer."""

    @property
    @abstractmethod
    def proof_sha256(self) -> str:
        """Digest of :meth:`identity_payload`."""

    @abstractmethod
    def measured_branch_cost(self, branch_id: str) -> float:
        """Return validated median branch milliseconds per query."""

    @abstractmethod
    def operator_complexity_key(
        self,
        config_id: str,
    ) -> tuple[int, int, int]:
        """Return the preregistered unitless operator-complexity key."""


def _require_loaded_score_state(score: ScoreArtifact) -> None:
    """Reject manually constructed or in-memory-mutated score dataclasses."""

    reloaded = ScoreArtifact.load(
        score.directory,
        allowed_scopes={"val-dev"},
    )
    if (
        reloaded.verified.payload_sha256 != score.verified.payload_sha256
        or reloaded.verified.envelope_sha256 != score.verified.envelope_sha256
        or reloaded.query_ids != score.query_ids
        or reloaded.gallery_ids != score.gallery_ids
        or _deep_thaw(reloaded.metadata) != _deep_thaw(score.metadata)
        or _deep_thaw(reloaded.provenance) != _deep_thaw(score.provenance)
        or _metric_payload(reloaded.metrics) != _metric_payload(score.metrics)
        or reloaded.similarity.dtype != score.similarity.dtype
        or reloaded.similarity.shape != score.similarity.shape
        or not np.array_equal(reloaded.similarity, score.similarity)
    ):
        raise ValueError(
            "in-memory ScoreArtifact differs from the freshly loaded "
            "verified envelope/payload"
        )


def _validate_component_identity(
    branch_id: Stage1BranchId,
    score: ScoreArtifact,
    proof: ValidatedComponentRunProof,
) -> dict[str, object]:
    if branch_id not in _BRANCH_SPECS:
        raise ValueError("branch_id must be internvit or brainrw")
    if not isinstance(score, ScoreArtifact):
        raise TypeError("score must be a loaded ScoreArtifact")
    if not isinstance(proof, ValidatedComponentRunProof):
        raise TypeError("run_proof must be a ValidatedComponentRunProof capability")

    _require_loaded_score_state(score)
    proof.revalidate()
    identity = _mapping_payload(
        proof.identity_payload(),
        "validated component run proof identity",
    )
    _require_exact_keys(
        identity,
        _RUN_PROOF_KEYS,
        "validated component run proof identity",
    )
    proof_sha256 = _require_sha256(
        proof.proof_sha256,
        "validated component run proof SHA-256",
    )
    if proof_sha256 != sha256_json(identity):
        raise ValueError("validated component run proof SHA-256 mismatch")
    if (
        identity["artifact_type"] != VALIDATED_RUN_PROOF_TYPE
        or identity["schema_version"] != 1
    ):
        raise ValueError("validated component run proof has the wrong type")

    for field_name in _PROOF_SHA_FIELDS:
        _require_sha256(
            identity[field_name],
            f"validated component run proof {field_name}",
        )
    _require_git_sha(
        identity["git_sha"],
        "validated component run proof git_sha",
    )
    expected_config, expected_stage, expected_epochs, expected_recipe_sha = (
        _BRANCH_SPECS[branch_id]
    )
    if (
        identity["branch_id"] != branch_id
        or identity["recipe_config_id"] != expected_config
        or identity["recipe_config_sha256"] != expected_recipe_sha
        or identity["stage"] != expected_stage
        or identity["epochs"] != expected_epochs
    ):
        raise ValueError(
            "validated component run proof branch/config/terminal stage/epoch "
            "identity differs from the locked Stage 1 component"
        )
    subject, seed = _require_coordinate(identity["subject"], identity["seed"])
    if identity["scope"] != "val-dev" or identity["split_role"] != "val-dev":
        raise ValueError("validated component run proof scope/split mismatch")

    schedule = _mapping_payload(
        identity["schedule"],
        "validated component schedule",
    )
    if identity["schedule_sha256"] != sha256_json(schedule):
        raise ValueError("validated component schedule SHA-256 mismatch")
    required_schedule = {
        "artifact_type": COMPONENT_SCHEDULE_TYPE,
        "branch_id": branch_id,
        "config_id": expected_config,
        "config_sha256": expected_recipe_sha,
        "epochs": expected_epochs,
        "schema_version": 1,
    }
    for field_name, expected in required_schedule.items():
        if schedule.get(field_name) != expected:
            raise ValueError(f"validated component schedule {field_name} mismatch")

    expected_run_key = make_run_key(
        expected_stage,
        expected_config,
        subject,
        seed,
        str(identity["resolved_config_sha256"]),
        str(identity["input_bundle_sha256"]),
    )
    if identity["run_key"] != expected_run_key:
        raise ValueError("validated component run_key mismatch")

    score.verified.revalidate()
    score.verified.revalidate_envelope()
    if (
        score.verified.artifact.payload_type != SCORE_PAYLOAD_TYPE
        or score.verified.artifact.payload_path != score.directory / "similarity.npy"
        or score.verified.artifact.envelope_path != score.directory / "metadata.json"
    ):
        raise ValueError("score is not the loaded terminal score capability")
    if score.scope != "val-dev":
        raise PermissionError("Stage 1 score composition is val-dev only")
    metadata = _mapping_payload(score.metadata, "score metadata")
    provenance = _mapping_payload(score.provenance, "score provenance")
    expected_metadata = {
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "config_sha256": identity["resolved_config_sha256"],
        "git_sha": identity["git_sha"],
        "protocol_sha256": identity["protocol_sha256"],
        "seed": seed,
        "source_records_sha256": identity["source_records_sha256"],
        "split_role": identity["split_role"],
        "stage": expected_stage,
        "subject": subject,
        "query_ids_sha256": identity["query_ids_sha256"],
        "gallery_ids_sha256": identity["gallery_ids_sha256"],
    }
    for field_name, expected in expected_metadata.items():
        if metadata.get(field_name) != expected:
            raise ValueError(f"score/proof {field_name} mismatch")
        if provenance.get(field_name) != expected:
            raise ValueError(f"score provenance/proof {field_name} mismatch")
    if (
        score.query_ids_sha256 != identity["query_ids_sha256"]
        or score.gallery_ids_sha256 != identity["gallery_ids_sha256"]
        or score.verified.payload_sha256 != identity["score_payload_sha256"]
        or score.verified.envelope_sha256 != identity["score_envelope_sha256"]
    ):
        raise ValueError("score payload/envelope/ordered-ID proof mismatch")

    source_records = _sequence_payload(
        identity["source_records"],
        "validated component source_records",
    )
    metadata_source_records = _sequence_payload(
        metadata["source_records"],
        "score source_records",
    )
    if source_records != metadata_source_records:
        raise ValueError("score/proof source_records mismatch")
    if sha256_json(source_records) != identity["source_records_sha256"]:
        raise ValueError("score/proof source_records SHA-256 mismatch")
    if not source_records or not isinstance(source_records[0], Mapping):
        raise ValueError("validated component source_records must be nonempty")
    first_record = source_records[0]
    record_crossbind = {
        "manifest_sha256": "manifest_sha256",
        "records_sha256": "records_sha256",
        "role_payload_sha256": "role_payload_sha256",
        "source_manifest_sha256": "source_manifest_sha256",
        "source_payload_sha256": "source_payload_sha256",
    }
    for record_field, identity_field in record_crossbind.items():
        if first_record.get(record_field) != identity[identity_field]:
            raise ValueError(f"score/proof {identity_field} mismatch")
    if first_record.get("role") != "val-dev":
        raise ValueError("score source role/split mismatch")
    if branch_id == INTERNVIT_BRANCH_ID:
        if any(
            not isinstance(record, Mapping)
            or record.get("run_key") != identity["run_key"]
            for record in source_records
        ):
            raise ValueError("InternViT source-record run_key mismatch")

    alignment = common_alignment_payload(score)
    proof_alignment = _mapping_payload(
        identity["alignment"],
        "validated component alignment",
    )
    if proof_alignment != alignment or identity["alignment_sha256"] != sha256_json(
        alignment
    ):
        raise ValueError("score/proof alignment mismatch")

    semantic_environment = _mapping_payload(
        identity["semantic_environment"],
        "validated component semantic environment",
    )
    if identity["semantic_environment_sha256"] != sha256_json(semantic_environment):
        raise ValueError("semantic environment SHA-256 mismatch")
    if branch_id == BRAINRW_BRANCH_ID:
        for prefix in ("training", "evaluation"):
            if (
                metadata.get(f"{prefix}_semantic_environment") != semantic_environment
                or metadata.get(f"{prefix}_semantic_environment_sha256")
                != identity["semantic_environment_sha256"]
            ):
                raise ValueError(f"BrainRW {prefix} semantic environment mismatch")

    outputs = _mapping_payload(
        identity["completion_output_hashes"],
        "validated completion output hashes",
    )
    expected_output_names = (
        {
            "final_checkpoint_sha256",
            "parity_sha256",
            "run_manifest_sha256",
        }
        if branch_id == INTERNVIT_BRANCH_ID
        else {
            "final_checkpoint_sha256",
            "run_manifest_sha256",
            "score_envelope_sha256",
            "score_payload_sha256",
        }
    )
    if set(outputs) != expected_output_names:
        raise ValueError("validated completion output hash schema mismatch")
    for field_name, digest in outputs.items():
        _require_sha256(digest, f"completion output {field_name}")
    if (
        outputs["final_checkpoint_sha256"] != identity["checkpoint_sha256"]
        or outputs["run_manifest_sha256"] != identity["run_manifest_sha256"]
    ):
        raise ValueError("completion/run checkpoint or manifest mismatch")
    if branch_id == BRAINRW_BRANCH_ID and (
        outputs["score_payload_sha256"] != identity["score_payload_sha256"]
        or outputs["score_envelope_sha256"] != identity["score_envelope_sha256"]
    ):
        raise ValueError("completion/BrainRW score output mismatch")

    recomputed_metrics = independent_retrieval_metrics(
        score.similarity,
        score.query_ids,
        score.gallery_ids,
    )
    if _metric_payload(recomputed_metrics) != _metric_payload(score.metrics):
        raise ValueError("loaded score metrics differ from independent ranking")
    return identity


@dataclass(frozen=True)
class Stage1ComponentBinding:
    """A terminal score capability cross-bound to a verified run proof."""

    branch_id: Stage1BranchId
    score: ScoreArtifact
    run_proof: ValidatedComponentRunProof
    _identity: Mapping[str, object] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        identity = _validate_component_identity(
            self.branch_id,
            self.score,
            self.run_proof,
        )
        object.__setattr__(self, "_identity", _deep_freeze(identity))

    def revalidate(self) -> None:
        current = _validate_component_identity(
            self.branch_id,
            self.score,
            self.run_proof,
        )
        if current != _deep_thaw(self._identity):
            raise ValueError("validated component run proof identity changed")

    def _value(self, field_name: str) -> object:
        return self._identity[field_name]

    @property
    def subject(self) -> int:
        return int(self._value("subject"))

    @property
    def seed(self) -> int:
        return int(self._value("seed"))

    @property
    def stage(self) -> str:
        return str(self._value("stage"))

    @property
    def recipe_config_id(self) -> str:
        return str(self._value("recipe_config_id"))

    @property
    def recipe_config_sha256(self) -> str:
        return str(self._value("recipe_config_sha256"))

    @property
    def resolved_config_sha256(self) -> str:
        return str(self._value("resolved_config_sha256"))

    @property
    def schedule_sha256(self) -> str:
        return str(self._value("schedule_sha256"))

    @property
    def epochs(self) -> int:
        return int(self._value("epochs"))

    @property
    def protocol_sha256(self) -> str:
        return str(self._value("protocol_sha256"))

    @property
    def alignment_sha256(self) -> str:
        return str(self._value("alignment_sha256"))

    @property
    def scope(self) -> str:
        return str(self._value("scope"))

    @property
    def split_role(self) -> str:
        return str(self._value("split_role"))

    @property
    def query_ids_sha256(self) -> str:
        return str(self._value("query_ids_sha256"))

    @property
    def gallery_ids_sha256(self) -> str:
        return str(self._value("gallery_ids_sha256"))

    @property
    def score_payload_sha256(self) -> str:
        return str(self._value("score_payload_sha256"))

    @property
    def score_envelope_sha256(self) -> str:
        return str(self._value("score_envelope_sha256"))

    @property
    def checkpoint_sha256(self) -> str:
        return str(self._value("checkpoint_sha256"))

    @property
    def run_manifest_sha256(self) -> str:
        return str(self._value("run_manifest_sha256"))

    def schedule_payload(self) -> dict[str, object]:
        return _mapping_payload(self._value("schedule"), "component schedule")

    def alignment_payload(self) -> dict[str, object]:
        return _mapping_payload(self._value("alignment"), "component alignment")

    def to_payload(self) -> dict[str, object]:
        identity = _mapping_payload(self._identity, "component identity")
        return {
            "artifact_type": COMPONENT_BINDING_TYPE,
            **{
                key: value
                for key, value in identity.items()
                if key not in {"artifact_type", "schema_version"}
            },
            "run_proof_sha256": self.run_proof.proof_sha256,
            "schema_version": 1,
        }

    @property
    def binding_sha256(self) -> str:
        return sha256_json(self.to_payload())


@dataclass(frozen=True)
class Stage1CellDependency:
    """The exact two-component and common-data identity of one pilot cell."""

    subject: int
    seed: int
    internvit_binding_sha256: str
    brainrw_binding_sha256: str
    alignment_sha256: str
    protocol_sha256: str
    query_ids_sha256: str
    gallery_ids_sha256: str
    scope: str
    split_role: str

    def __post_init__(self) -> None:
        _require_coordinate(self.subject, self.seed)
        for field_name in (
            "internvit_binding_sha256",
            "brainrw_binding_sha256",
            "alignment_sha256",
            "protocol_sha256",
            "query_ids_sha256",
            "gallery_ids_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name)
        if self.scope != "val-dev" or self.split_role != "val-dev":
            raise ValueError("Stage 1 dependency must be val-dev")

    @classmethod
    def from_components(
        cls,
        internvit: Stage1ComponentBinding,
        brainrw: Stage1ComponentBinding,
    ) -> "Stage1CellDependency":
        if not isinstance(internvit, Stage1ComponentBinding) or not isinstance(
            brainrw, Stage1ComponentBinding
        ):
            raise TypeError("cell dependencies require component bindings")
        if internvit.branch_id != INTERNVIT_BRANCH_ID:
            raise ValueError("InternViT dependency has the wrong branch")
        if brainrw.branch_id != BRAINRW_BRANCH_ID:
            raise ValueError("BrainRW dependency has the wrong branch")
        assert_aligned(internvit.score, brainrw.score)
        coordinate = (internvit.subject, internvit.seed)
        if coordinate != (brainrw.subject, brainrw.seed):
            raise ValueError("component subject/seed coordinates do not align")
        if internvit.alignment_payload() != brainrw.alignment_payload():
            raise ValueError("component common alignment proof differs")
        return cls(
            subject=coordinate[0],
            seed=coordinate[1],
            internvit_binding_sha256=internvit.binding_sha256,
            brainrw_binding_sha256=brainrw.binding_sha256,
            alignment_sha256=internvit.alignment_sha256,
            protocol_sha256=internvit.protocol_sha256,
            query_ids_sha256=internvit.query_ids_sha256,
            gallery_ids_sha256=internvit.gallery_ids_sha256,
            scope=internvit.scope,
            split_role=internvit.split_role,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "alignment_sha256": self.alignment_sha256,
            "artifact_type": CELL_DEPENDENCY_TYPE,
            "brainrw_binding_sha256": self.brainrw_binding_sha256,
            "gallery_ids_sha256": self.gallery_ids_sha256,
            "internvit_binding_sha256": self.internvit_binding_sha256,
            "protocol_sha256": self.protocol_sha256,
            "query_ids_sha256": self.query_ids_sha256,
            "schema_version": 1,
            "scope": self.scope,
            "seed": self.seed,
            "split_role": self.split_role,
            "subject": self.subject,
        }

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_payload())


@dataclass(frozen=True)
class Stage1CompositionCell:
    """One aligned Stage 1 cell; it contains no caller-reported metrics."""

    subject: int
    seed: int
    internvit: Stage1ComponentBinding
    brainrw: Stage1ComponentBinding
    _alignment: Mapping[str, object] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        coordinate = _require_coordinate(self.subject, self.seed)
        if not isinstance(self.internvit, Stage1ComponentBinding):
            raise TypeError("internvit must be a Stage1ComponentBinding")
        if not isinstance(self.brainrw, Stage1ComponentBinding):
            raise TypeError("brainrw must be a Stage1ComponentBinding")
        self.internvit.revalidate()
        self.brainrw.revalidate()
        if (
            self.internvit.branch_id != INTERNVIT_BRANCH_ID
            or self.internvit.stage != INTERNVIT_STAGE
        ):
            raise ValueError("Stage 1 InternViT must be terminal stage0")
        if (
            self.brainrw.branch_id != BRAINRW_BRANCH_ID
            or self.brainrw.stage != BRAINRW_STAGE
        ):
            raise ValueError(
                "Stage 1 BrainRW must be the brainrw-clip-lora terminal stage"
            )
        if (self.internvit.subject, self.internvit.seed) != coordinate:
            raise ValueError("InternViT binding coordinate mismatch")
        if (self.brainrw.subject, self.brainrw.seed) != coordinate:
            raise ValueError("BrainRW binding coordinate mismatch")
        # Explicitly call the canonical fusion alignment guard.
        assert_aligned(self.internvit.score, self.brainrw.score)
        left = common_alignment_payload(self.internvit.score)
        right = common_alignment_payload(self.brainrw.score)
        if left != right:
            raise ValueError("Stage 1 branch common provenance differs")
        object.__setattr__(self, "_alignment", _deep_freeze(left))

    def revalidate(self) -> None:
        self.internvit.revalidate()
        self.brainrw.revalidate()
        if (
            self.internvit.stage != INTERNVIT_STAGE
            or self.brainrw.stage != BRAINRW_STAGE
        ):
            raise ValueError("Stage 1 terminal stage identity changed")
        assert_aligned(self.internvit.score, self.brainrw.score)
        current = common_alignment_payload(self.internvit.score)
        if current != _deep_thaw(self._alignment):
            raise ValueError("Stage 1 cell alignment identity changed")

    def alignment_payload(self) -> dict[str, object]:
        return _mapping_payload(self._alignment, "cell alignment")

    @property
    def alignment_sha256(self) -> str:
        return sha256_json(self.alignment_payload())

    @property
    def dependency(self) -> Stage1CellDependency:
        return Stage1CellDependency.from_components(
            self.internvit,
            self.brainrw,
        )


def _ordered_stage1_cells(
    cells: Sequence[Stage1CompositionCell],
) -> tuple[Stage1CompositionCell, ...]:
    if isinstance(cells, (str, bytes, bytearray)):
        raise TypeError("Stage 1 cells must be a sequence")
    values = tuple(cells)
    if any(not isinstance(value, Stage1CompositionCell) for value in values):
        raise TypeError("Stage 1 cells must be Stage1CompositionCell values")
    coordinates = tuple((value.subject, value.seed) for value in values)
    if len(set(coordinates)) != len(coordinates):
        raise ValueError("duplicate Stage 1 subject/seed cell")
    if len(values) != 6 or set(coordinates) != set(PILOT_COORDINATES):
        raise ValueError(
            "Stage 1 requires the exact six-cell coordinate grid "
            "subjects=(1,5,8), seeds=(42,43)"
        )
    by_coordinate = {(value.subject, value.seed): value for value in values}
    ordered = tuple(by_coordinate[coordinate] for coordinate in PILOT_COORDINATES)
    for cell in ordered:
        cell.revalidate()

    first = ordered[0]
    common_identity = {
        "gallery_ids": first.internvit.score.gallery_ids,
        "gallery_ids_sha256": first.internvit.gallery_ids_sha256,
        "protocol_sha256": first.internvit.protocol_sha256,
        "query_ids": first.internvit.score.query_ids,
        "query_ids_sha256": first.internvit.query_ids_sha256,
        "scope": first.internvit.scope,
        "split_role": first.internvit.split_role,
    }
    for cell in ordered[1:]:
        actual = {
            "gallery_ids": cell.internvit.score.gallery_ids,
            "gallery_ids_sha256": cell.internvit.gallery_ids_sha256,
            "protocol_sha256": cell.internvit.protocol_sha256,
            "query_ids": cell.internvit.score.query_ids,
            "query_ids_sha256": cell.internvit.query_ids_sha256,
            "scope": cell.internvit.scope,
            "split_role": cell.internvit.split_role,
        }
        differing = sorted(
            key for key in common_identity if actual[key] != common_identity[key]
        )
        if differing:
            raise ValueError(
                "Stage 1 cells have a cross-cell protocol/query/gallery/"
                f"scope mismatch: {differing}"
            )

    for branch_id in _BRANCH_IDS:
        bindings = tuple(
            cell.internvit if branch_id == INTERNVIT_BRANCH_ID else cell.brainrw
            for cell in ordered
        )
        locked = {
            (
                binding.recipe_config_id,
                binding.recipe_config_sha256,
                binding.schedule_sha256,
                binding.epochs,
                binding._value("semantic_environment_sha256"),
            )
            for binding in bindings
        }
        if len(locked) != 1:
            raise ValueError(
                f"{branch_id} recipe/schedule/environment identity "
                "drifts across Stage 1 cells"
            )
        schedules = {sha256_json(binding.schedule_payload()) for binding in bindings}
        if schedules != {bindings[0].schedule_sha256}:
            raise ValueError(f"{branch_id} schedule body drifts across Stage 1 cells")
    return ordered


def _component_score_input(
    binding: Stage1ComponentBinding,
) -> dict[str, object]:
    return {
        "binding_sha256": binding.binding_sha256,
        "checkpoint_sha256": binding.checkpoint_sha256,
        "resolved_config_sha256": binding.resolved_config_sha256,
        "run_proof_sha256": binding.run_proof.proof_sha256,
        "score_envelope_sha256": binding.score_envelope_sha256,
        "score_payload_sha256": binding.score_payload_sha256,
    }


def _cost_score_inputs(
    cells: Sequence[Stage1CompositionCell],
) -> list[dict[str, object]]:
    ordered = _ordered_stage1_cells(cells)
    return [
        {
            "alignment_sha256": cell.alignment_sha256,
            "brainrw": _component_score_input(cell.brainrw),
            "cell_id": _cell_id(cell.subject, cell.seed),
            "gallery_count": len(cell.internvit.score.gallery_ids),
            "gallery_ids_sha256": cell.internvit.gallery_ids_sha256,
            "internvit": _component_score_input(cell.internvit),
            "query_count": len(cell.internvit.score.query_ids),
            "query_ids_sha256": cell.internvit.query_ids_sha256,
            "seed": cell.seed,
            "subject": cell.subject,
        }
        for cell in ordered
    ]


def _validate_cost_protocol(value: object) -> dict[str, object]:
    protocol = _mapping_payload(value, "cost measurement protocol")
    _require_exact_keys(
        protocol,
        _COST_PROTOCOL_KEYS,
        "cost measurement protocol",
    )
    expected_literals = {
        "aggregation": ("arithmetic_mean_elapsed_nanoseconds_divided_by_total_queries"),
        "artifact_type": "samga_brain_rw.stage1_cost_protocol",
        "cuda_synchronize": "before_and_after",
        "raw_unit": "nanoseconds",
        "schema_version": 1,
        "scope": "val-dev",
        "timing_boundary": "encoder_or_fusion_operator",
        "unit": "milliseconds_per_query",
    }
    for field_name, expected in expected_literals.items():
        if protocol[field_name] != expected:
            raise ValueError(f"cost measurement protocol {field_name} mismatch")
    _require_nonnegative_integer(
        protocol["warmup_runs"],
        "cost measurement protocol warmup_runs",
    )
    _require_positive_integer(
        protocol["measured_runs"],
        "cost measurement protocol measured_runs",
    )
    _require_sha256(
        protocol["runtime_evidence_sha256"],
        "cost measurement protocol runtime_evidence_sha256",
    )
    return protocol


def _raw_elapsed(
    value: object,
    *,
    measured_runs: int,
    context: str,
) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{context} raw measurements must be a sequence")
    raw = tuple(
        _require_positive_integer(item, f"{context}[{index}]")
        for index, item in enumerate(value)
    )
    if len(raw) != measured_runs:
        raise ValueError(
            f"{context} must contain exactly {measured_runs} raw measurements"
        )
    return raw


def _milliseconds_per_query(
    elapsed_nanoseconds: Sequence[int],
    total_query_count: int,
) -> float:
    mean_nanoseconds = math.fsum(elapsed_nanoseconds) / len(elapsed_nanoseconds)
    return mean_nanoseconds / total_query_count / 1_000_000.0


def _measurement_with_sha256(
    body: Mapping[str, object],
) -> dict[str, object]:
    value = dict(body)
    value["measurement_sha256"] = sha256_json(body)
    return value


@dataclass(frozen=True)
class _RetiredStage1CostEvidence:
    """Raw, recomputable cost evidence bound to all six score inputs.

    This proves arithmetic and identity closure, not that a trusted timing
    producer ran.  A future upstream verifier can issue this document after
    controlled measurement without changing the composition contract.
    """

    measurement_protocol: Mapping[str, object]
    score_inputs: tuple[Mapping[str, object], ...]
    branch_measurements: tuple[Mapping[str, object], ...]
    fusion_measurements: tuple[Mapping[str, object], ...]
    evidence_sha256: str

    def __new__(cls, *_: object, **__: object) -> "_RetiredStage1CostEvidence":
        raise TypeError(
            "raw Stage 1 cost evidence construction is retired; a trusted "
            "ValidatedStage1CostCapability issuer is required"
        )

    def __post_init__(self) -> None:
        protocol = _validate_cost_protocol(self.measurement_protocol)
        score_inputs = _sequence_payload(self.score_inputs, "cost score_inputs")
        branch_measurements = _sequence_payload(
            self.branch_measurements, "cost branch_measurements"
        )
        fusion_measurements = _sequence_payload(
            self.fusion_measurements, "cost fusion_measurements"
        )
        object.__setattr__(self, "measurement_protocol", _deep_freeze(protocol))
        object.__setattr__(self, "score_inputs", tuple(_deep_freeze(score_inputs)))
        object.__setattr__(
            self,
            "branch_measurements",
            tuple(_deep_freeze(branch_measurements)),
        )
        object.__setattr__(
            self,
            "fusion_measurements",
            tuple(_deep_freeze(fusion_measurements)),
        )
        _validate_cost_payload(self.to_payload())
        _require_sha256(self.evidence_sha256, "cost evidence SHA-256")
        if self.evidence_sha256 != sha256_json(self.to_payload()):
            raise ValueError("cost evidence SHA-256 mismatch")

    @classmethod
    def from_raw_measurements(
        cls,
        *,
        cells: Sequence[Stage1CompositionCell],
        measurement_protocol: Mapping[str, object],
        branch_elapsed_nanoseconds: Mapping[str, Sequence[int]],
        fusion_operator_elapsed_nanoseconds: Mapping[str, Sequence[int]],
    ) -> "_RetiredStage1CostEvidence":
        ordered = _ordered_stage1_cells(cells)
        protocol = _validate_cost_protocol(measurement_protocol)
        if not isinstance(branch_elapsed_nanoseconds, Mapping):
            raise TypeError("branch raw measurements must be a mapping")
        if set(branch_elapsed_nanoseconds) != set(_BRANCH_IDS):
            raise ValueError("branch raw measurements require internvit and brainrw")
        grid = enumerate_stage1_configs()
        expected_fusion_ids = {config.config_id for config in grid}
        if not isinstance(fusion_operator_elapsed_nanoseconds, Mapping):
            raise TypeError("fusion raw measurements must be a mapping")
        if set(fusion_operator_elapsed_nanoseconds) != expected_fusion_ids:
            missing = sorted(
                expected_fusion_ids - set(fusion_operator_elapsed_nanoseconds)
            )
            extra = sorted(
                set(fusion_operator_elapsed_nanoseconds) - expected_fusion_ids
            )
            raise ValueError(
                "fusion raw measurements require the exact 47-config grid: "
                f"missing={missing}, extra={extra}"
            )
        measured_runs = int(protocol["measured_runs"])
        total_query_count = sum(len(cell.internvit.score.query_ids) for cell in ordered)
        score_inputs = _cost_score_inputs(ordered)
        score_inputs_sha256 = sha256_json(score_inputs)

        branch_entries: list[dict[str, object]] = []
        branch_costs: dict[str, float] = {}
        branch_measurement_sha256s: dict[str, str] = {}
        for branch_id in _BRANCH_IDS:
            raw = _raw_elapsed(
                branch_elapsed_nanoseconds[branch_id],
                measured_runs=measured_runs,
                context=f"{branch_id} raw elapsed_nanoseconds",
            )
            branch_score_inputs = [
                {
                    "alignment_sha256": item["alignment_sha256"],
                    "cell_id": item["cell_id"],
                    "score": item[branch_id],
                }
                for item in score_inputs
            ]
            cost = _milliseconds_per_query(raw, total_query_count)
            body = {
                "branch_id": branch_id,
                "elapsed_nanoseconds": list(raw),
                "encoder_count": 1,
                "measured_inference_cost": cost,
                "score_inputs_sha256": sha256_json(branch_score_inputs),
                "total_query_count": total_query_count,
            }
            entry = _measurement_with_sha256(body)
            branch_entries.append(entry)
            branch_costs[branch_id] = cost
            branch_measurement_sha256s[branch_id] = str(entry["measurement_sha256"])

        fusion_entries: list[dict[str, object]] = []
        for config in grid:
            raw = _raw_elapsed(
                fusion_operator_elapsed_nanoseconds[config.config_id],
                measured_runs=measured_runs,
                context=(f"{config.config_id} raw operator elapsed_nanoseconds"),
            )
            operator_cost = _milliseconds_per_query(raw, total_query_count)
            total_cost = (
                branch_costs[INTERNVIT_BRANCH_ID]
                + branch_costs[BRAINRW_BRANCH_ID]
                + operator_cost
            )
            body = {
                "branch_measurement_sha256s": dict(branch_measurement_sha256s),
                "config_id": config.config_id,
                "elapsed_nanoseconds": list(raw),
                "encoder_count": 2,
                "measured_inference_cost": total_cost,
                "operator_cost_milliseconds_per_query": operator_cost,
                "score_inputs_sha256": score_inputs_sha256,
                "total_query_count": total_query_count,
            }
            fusion_entries.append(_measurement_with_sha256(body))

        payload = {
            "artifact_type": _RETIRED_COST_EVIDENCE_TYPE,
            "branch_measurements": branch_entries,
            "fusion_measurements": fusion_entries,
            "measurement_protocol": protocol,
            "measurement_protocol_sha256": sha256_json(protocol),
            "schema_version": 1,
            "score_inputs": score_inputs,
            "score_inputs_sha256": score_inputs_sha256,
            "scope": "val-dev",
        }
        return cls(
            measurement_protocol=protocol,
            score_inputs=tuple(score_inputs),
            branch_measurements=tuple(branch_entries),
            fusion_measurements=tuple(fusion_entries),
            evidence_sha256=sha256_json(payload),
        )

    @classmethod
    def from_document(
        cls,
        document: Mapping[str, object],
    ) -> "_RetiredStage1CostEvidence":
        value = _mapping_payload(document, "cost evidence document")
        expected = frozenset(
            {
                "artifact_type",
                "branch_measurements",
                "evidence_sha256",
                "fusion_measurements",
                "measurement_protocol",
                "measurement_protocol_sha256",
                "schema_version",
                "score_inputs",
                "score_inputs_sha256",
                "scope",
            }
        )
        _require_exact_keys(value, expected, "cost evidence document")
        return cls(
            measurement_protocol=value["measurement_protocol"],  # type: ignore[arg-type]
            score_inputs=tuple(value["score_inputs"]),  # type: ignore[arg-type]
            branch_measurements=tuple(
                value["branch_measurements"]  # type: ignore[arg-type]
            ),
            fusion_measurements=tuple(
                value["fusion_measurements"]  # type: ignore[arg-type]
            ),
            evidence_sha256=value["evidence_sha256"],  # type: ignore[arg-type]
        )

    def measurement_protocol_payload(self) -> dict[str, object]:
        return _mapping_payload(
            self.measurement_protocol,
            "cost measurement protocol",
        )

    def score_inputs_payload(self) -> list[dict[str, object]]:
        values = _sequence_payload(self.score_inputs, "cost score_inputs")
        if any(not isinstance(value, dict) for value in values):
            raise AssertionError("cost score inputs must be mappings")
        return values  # type: ignore[return-value]

    def to_payload(self) -> dict[str, object]:
        protocol = self.measurement_protocol_payload()
        score_inputs = self.score_inputs_payload()
        branches = _sequence_payload(
            self.branch_measurements, "cost branch measurements"
        )
        fusion = _sequence_payload(self.fusion_measurements, "cost fusion measurements")
        return {
            "artifact_type": _RETIRED_COST_EVIDENCE_TYPE,
            "branch_measurements": branches,
            "fusion_measurements": fusion,
            "measurement_protocol": protocol,
            "measurement_protocol_sha256": sha256_json(protocol),
            "schema_version": 1,
            "score_inputs": score_inputs,
            "score_inputs_sha256": sha256_json(score_inputs),
            "scope": "val-dev",
        }

    def to_document(self) -> dict[str, object]:
        return {
            **self.to_payload(),
            "evidence_sha256": self.evidence_sha256,
        }

    def branch_cost(self, branch_id: str) -> float:
        for raw in self.branch_measurements:
            if raw["branch_id"] == branch_id:
                return float(raw["measured_inference_cost"])
        raise ValueError(f"unknown Stage 1 branch cost: {branch_id}")

    def fusion_cost(self, config_id: str) -> float:
        for raw in self.fusion_measurements:
            if raw["config_id"] == config_id:
                return float(raw["measured_inference_cost"])
        raise ValueError(f"unknown Stage 1 fusion cost: {config_id}")


def _validate_cost_payload(payload: Mapping[str, object]) -> None:
    expected_keys = frozenset(
        {
            "artifact_type",
            "branch_measurements",
            "fusion_measurements",
            "measurement_protocol",
            "measurement_protocol_sha256",
            "schema_version",
            "score_inputs",
            "score_inputs_sha256",
            "scope",
        }
    )
    _require_exact_keys(payload, expected_keys, "cost evidence")
    if (
        payload["artifact_type"] != _RETIRED_COST_EVIDENCE_TYPE
        or payload["schema_version"] != 1
        or payload["scope"] != "val-dev"
    ):
        raise ValueError("cost evidence type/scope mismatch")
    protocol = _validate_cost_protocol(payload["measurement_protocol"])
    if payload["measurement_protocol_sha256"] != sha256_json(protocol):
        raise ValueError("cost measurement protocol SHA-256 mismatch")
    score_inputs = _sequence_payload(payload["score_inputs"], "cost score_inputs")
    if len(score_inputs) != 6:
        raise ValueError("cost evidence must bind exactly six score inputs")
    if payload["score_inputs_sha256"] != sha256_json(score_inputs):
        raise ValueError("cost score input SHA-256 mismatch")
    coordinates: list[tuple[int, int]] = []
    for index, raw in enumerate(score_inputs):
        item = _mapping_payload(raw, f"cost score_inputs[{index}]")
        expected_input_keys = frozenset(
            {
                "alignment_sha256",
                "brainrw",
                "cell_id",
                "gallery_count",
                "gallery_ids_sha256",
                "internvit",
                "query_count",
                "query_ids_sha256",
                "seed",
                "subject",
            }
        )
        _require_exact_keys(item, expected_input_keys, f"cost score_inputs[{index}]")
        coordinate = _require_coordinate(item["subject"], item["seed"])
        if item["cell_id"] != _cell_id(*coordinate):
            raise ValueError("cost score input cell_id mismatch")
        coordinates.append(coordinate)
        _require_positive_integer(item["query_count"], "cost score input query_count")
        _require_positive_integer(
            item["gallery_count"], "cost score input gallery_count"
        )
        for field_name in (
            "alignment_sha256",
            "gallery_ids_sha256",
            "query_ids_sha256",
        ):
            _require_sha256(
                item[field_name],
                f"cost score input {field_name}",
            )
        for branch_id in _BRANCH_IDS:
            component = _mapping_payload(
                item[branch_id],
                f"cost score input {branch_id}",
            )
            _require_exact_keys(
                component,
                frozenset(
                    {
                        "binding_sha256",
                        "checkpoint_sha256",
                        "resolved_config_sha256",
                        "run_proof_sha256",
                        "score_envelope_sha256",
                        "score_payload_sha256",
                    }
                ),
                f"cost score input {branch_id}",
            )
            for field_name, digest in component.items():
                _require_sha256(
                    digest,
                    f"cost score input {branch_id}.{field_name}",
                )
    if tuple(coordinates) != PILOT_COORDINATES:
        raise ValueError("cost score inputs are not the ordered six-cell pilot")
    expected_total_query_count = sum(
        int(item["query_count"])  # type: ignore[index]
        for item in score_inputs
    )

    measured_runs = int(protocol["measured_runs"])
    branches = _sequence_payload(
        payload["branch_measurements"],
        "cost branch_measurements",
    )
    if len(branches) != 2:
        raise ValueError("cost evidence requires exactly two branch measurements")
    branch_costs: dict[str, float] = {}
    branch_sha256s: dict[str, str] = {}
    total_query_counts: set[int] = set()
    for index, raw in enumerate(branches):
        entry = _mapping_payload(raw, f"cost branch_measurements[{index}]")
        expected = frozenset(
            {
                "branch_id",
                "elapsed_nanoseconds",
                "encoder_count",
                "measured_inference_cost",
                "measurement_sha256",
                "score_inputs_sha256",
                "total_query_count",
            }
        )
        _require_exact_keys(entry, expected, f"cost branch_measurements[{index}]")
        branch_id = entry["branch_id"]
        if branch_id not in _BRANCH_IDS or branch_id in branch_costs:
            raise ValueError("duplicate or unknown branch cost measurement")
        raw_elapsed = _raw_elapsed(
            entry["elapsed_nanoseconds"],
            measured_runs=measured_runs,
            context=f"{branch_id} raw elapsed_nanoseconds",
        )
        query_count = _require_positive_integer(
            entry["total_query_count"],
            f"{branch_id} total_query_count",
        )
        if query_count != expected_total_query_count:
            raise ValueError(f"{branch_id} total_query_count differs from score inputs")
        total_query_counts.add(query_count)
        if entry["encoder_count"] != 1:
            raise ValueError("single branch cost must bind one encoder")
        if entry["score_inputs_sha256"] != sha256_json(
            [
                {
                    "alignment_sha256": item["alignment_sha256"],
                    "cell_id": item["cell_id"],
                    "score": item[branch_id],
                }
                for item in score_inputs  # type: ignore[index]
            ]
        ):
            raise ValueError(f"{branch_id} cost score-input binding mismatch")
        expected_cost = _milliseconds_per_query(raw_elapsed, query_count)
        actual_cost = _require_cost(
            entry["measured_inference_cost"],
            f"{branch_id} measured_inference_cost",
        )
        if actual_cost != expected_cost:
            raise ValueError(f"{branch_id} derived cost mismatch")
        body = {
            key: value for key, value in entry.items() if key != "measurement_sha256"
        }
        if entry["measurement_sha256"] != sha256_json(body):
            raise ValueError(f"{branch_id} measurement SHA-256 mismatch")
        branch_costs[str(branch_id)] = actual_cost
        branch_sha256s[str(branch_id)] = str(entry["measurement_sha256"])
    if set(branch_costs) != set(_BRANCH_IDS):
        raise ValueError("cost evidence branch set mismatch")
    if len(total_query_counts) != 1:
        raise ValueError("cost evidence query counts differ")

    fusion = _sequence_payload(
        payload["fusion_measurements"],
        "cost fusion_measurements",
    )
    grid = enumerate_stage1_configs()
    if len(fusion) != 47:
        raise ValueError("cost evidence requires all 47 fusion measurements")
    actual_ids: list[str] = []
    for index, (raw, config) in enumerate(zip(fusion, grid, strict=True)):
        entry = _mapping_payload(raw, f"cost fusion_measurements[{index}]")
        expected = frozenset(
            {
                "branch_measurement_sha256s",
                "config_id",
                "elapsed_nanoseconds",
                "encoder_count",
                "measured_inference_cost",
                "measurement_sha256",
                "operator_cost_milliseconds_per_query",
                "score_inputs_sha256",
                "total_query_count",
            }
        )
        _require_exact_keys(entry, expected, f"cost fusion_measurements[{index}]")
        config_id = _require_safe_id(entry["config_id"], "fusion cost config_id")
        actual_ids.append(config_id)
        if config_id != config.config_id:
            raise ValueError("fusion cost grid order/identity mismatch")
        branch_hashes = _mapping_payload(
            entry["branch_measurement_sha256s"],
            "fusion branch measurement hashes",
        )
        if branch_hashes != branch_sha256s:
            raise ValueError("fusion cost branch evidence mismatch")
        raw_elapsed = _raw_elapsed(
            entry["elapsed_nanoseconds"],
            measured_runs=measured_runs,
            context=f"{config_id} raw operator elapsed_nanoseconds",
        )
        query_count = _require_positive_integer(
            entry["total_query_count"],
            f"{config_id} total_query_count",
        )
        if query_count not in total_query_counts:
            raise ValueError("fusion cost query-count mismatch")
        if entry["encoder_count"] != 2:
            raise ValueError("fusion cost must bind two encoders")
        if entry["score_inputs_sha256"] != payload["score_inputs_sha256"]:
            raise ValueError("fusion cost score-input binding mismatch")
        operator_cost = _milliseconds_per_query(raw_elapsed, query_count)
        if (
            _require_cost(
                entry["operator_cost_milliseconds_per_query"],
                f"{config_id} operator cost",
            )
            != operator_cost
        ):
            raise ValueError(f"{config_id} operator cost mismatch")
        expected_cost = (
            branch_costs[INTERNVIT_BRANCH_ID]
            + branch_costs[BRAINRW_BRANCH_ID]
            + operator_cost
        )
        if (
            _require_cost(
                entry["measured_inference_cost"],
                f"{config_id} measured_inference_cost",
            )
            != expected_cost
        ):
            raise ValueError(f"{config_id} derived fusion cost mismatch")
        body = {
            key: value for key, value in entry.items() if key != "measurement_sha256"
        }
        if entry["measurement_sha256"] != sha256_json(body):
            raise ValueError(f"{config_id} measurement SHA-256 mismatch")
    if actual_ids != [config.config_id for config in grid]:
        raise ValueError("fusion cost evidence does not contain the exact grid")


def _require_operator_complexity_key(
    value: object,
    context: str,
) -> tuple[int, int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 3
        or any(type(item) is not int or item < 0 for item in value)
    ):
        raise ValueError(
            f"{context} must be a unitless tuple of three nonnegative integers"
        )
    return (value[0], value[1], value[2])


def _validated_cost_capability_snapshot(
    capability: ValidatedStage1CostCapability,
    cells: Sequence[Stage1CompositionCell],
) -> tuple[
    dict[str, object],
    str,
    dict[str, float],
    dict[str, tuple[int, int, int]],
]:
    if not isinstance(capability, ValidatedStage1CostCapability):
        raise TypeError("cost_capability must be a ValidatedStage1CostCapability")
    capability.revalidate()
    identity = _mapping_payload(
        capability.identity_payload(),
        "validated Stage 1 cost capability identity",
    )
    _require_exact_keys(
        identity,
        frozenset(
            {
                "artifact_type",
                "branch_measured_ms_per_query",
                "measurement_protocol_sha256",
                "operator_complexity_keys",
                "raw_record_sha256",
                "runtime_evidence_sha256",
                "schema_version",
                "scope",
                "score_inputs",
                "score_inputs_sha256",
                "two_encoder_count",
            }
        ),
        "validated Stage 1 cost capability identity",
    )
    if (
        identity["artifact_type"] != VALIDATED_COST_CAPABILITY_TYPE
        or identity["schema_version"] != 1
        or identity["scope"] != "val-dev"
        or identity["two_encoder_count"] != 2
    ):
        raise ValueError("validated cost capability identity/scope mismatch")
    proof_sha256 = _require_sha256(
        capability.proof_sha256,
        "validated cost capability proof SHA-256",
    )
    if proof_sha256 != sha256_json(identity):
        raise ValueError("validated cost capability proof SHA-256 mismatch")
    expected_score_inputs = _cost_score_inputs(cells)
    score_inputs = _sequence_payload(
        identity["score_inputs"],
        "validated cost capability score_inputs",
    )
    if (
        score_inputs != expected_score_inputs
        or len(score_inputs) != 6
        or identity["score_inputs_sha256"] != sha256_json(score_inputs)
    ):
        raise ValueError(
            "validated cost capability does not bind the exact six score inputs"
        )
    for field_name in (
        "measurement_protocol_sha256",
        "raw_record_sha256",
        "runtime_evidence_sha256",
    ):
        _require_sha256(
            identity[field_name],
            f"validated cost capability {field_name}",
        )

    raw_branch_costs = _mapping_payload(
        identity["branch_measured_ms_per_query"],
        "validated cost capability branch costs",
    )
    if set(raw_branch_costs) != set(_BRANCH_IDS):
        raise ValueError("validated cost capability branch set mismatch")
    branch_costs: dict[str, float] = {}
    for branch_id in _BRANCH_IDS:
        cost = _require_cost(
            capability.measured_branch_cost(branch_id),
            f"{branch_id} branch measured milliseconds per query",
        )
        if raw_branch_costs[branch_id] != cost:
            raise ValueError("validated cost capability branch cost mismatch")
        branch_costs[branch_id] = cost

    raw_operator_keys = _sequence_payload(
        identity["operator_complexity_keys"],
        "validated cost capability operator complexity keys",
    )
    grid = enumerate_stage1_configs()
    if len(raw_operator_keys) != len(grid):
        raise ValueError("validated cost capability must bind all 47 operator keys")
    operator_keys: dict[str, tuple[int, int, int]] = {}
    for index, (raw, config) in enumerate(zip(raw_operator_keys, grid, strict=True)):
        entry = _mapping_payload(
            raw,
            f"validated operator complexity key[{index}]",
        )
        _require_exact_keys(
            entry,
            frozenset({"config_id", "operator_complexity_key"}),
            f"validated operator complexity key[{index}]",
        )
        if entry["config_id"] != config.config_id:
            raise ValueError("validated operator complexity grid mismatch")
        identity_key = _require_operator_complexity_key(
            entry["operator_complexity_key"],
            f"{config.config_id} identity operator complexity key",
        )
        capability_key = _require_operator_complexity_key(
            capability.operator_complexity_key(config.config_id),
            f"{config.config_id} capability operator complexity key",
        )
        if identity_key != capability_key:
            raise ValueError("validated cost capability operator complexity mismatch")
        operator_keys[config.config_id] = capability_key

    capability.revalidate()
    if (
        capability.proof_sha256 != proof_sha256
        or capability.identity_payload() != identity
    ):
        raise ValueError("validated cost capability mutated during composition")
    return identity, proof_sha256, branch_costs, operator_keys


def _serialized_cost_capability_values(
    value: object,
    proof_value: object,
) -> tuple[
    dict[str, object],
    str,
    dict[str, float],
    dict[str, tuple[int, int, int]],
]:
    identity = _mapping_payload(value, "serialized cost capability identity")
    _require_exact_keys(
        identity,
        frozenset(
            {
                "artifact_type",
                "branch_measured_ms_per_query",
                "measurement_protocol_sha256",
                "operator_complexity_keys",
                "raw_record_sha256",
                "runtime_evidence_sha256",
                "schema_version",
                "scope",
                "score_inputs",
                "score_inputs_sha256",
                "two_encoder_count",
            }
        ),
        "serialized cost capability identity",
    )
    if (
        identity["artifact_type"] != VALIDATED_COST_CAPABILITY_TYPE
        or identity["schema_version"] != 1
        or identity["scope"] != "val-dev"
        or identity["two_encoder_count"] != 2
    ):
        raise ValueError("serialized cost capability identity/scope mismatch")
    proof_sha256 = _require_sha256(
        proof_value,
        "serialized cost capability proof SHA-256",
    )
    if proof_sha256 != sha256_json(identity):
        raise ValueError("serialized cost capability proof SHA-256 mismatch")
    score_inputs = _sequence_payload(
        identity["score_inputs"],
        "serialized cost capability score_inputs",
    )
    if len(score_inputs) != 6 or identity["score_inputs_sha256"] != sha256_json(
        score_inputs
    ):
        raise ValueError("serialized cost capability six-input binding mismatch")
    for field_name in (
        "measurement_protocol_sha256",
        "raw_record_sha256",
        "runtime_evidence_sha256",
    ):
        _require_sha256(
            identity[field_name],
            f"serialized cost capability {field_name}",
        )
    raw_costs = _mapping_payload(
        identity["branch_measured_ms_per_query"],
        "serialized cost capability branch costs",
    )
    if set(raw_costs) != set(_BRANCH_IDS):
        raise ValueError("serialized cost capability branch set mismatch")
    branch_costs = {
        branch_id: _require_cost(
            raw_costs[branch_id],
            f"{branch_id} serialized branch cost",
        )
        for branch_id in _BRANCH_IDS
    }
    entries = _sequence_payload(
        identity["operator_complexity_keys"],
        "serialized cost capability operator keys",
    )
    grid = enumerate_stage1_configs()
    if len(entries) != len(grid):
        raise ValueError("serialized cost capability requires 47 operator keys")
    operator_keys: dict[str, tuple[int, int, int]] = {}
    for index, (raw, config) in enumerate(zip(entries, grid, strict=True)):
        entry = _mapping_payload(raw, f"serialized operator key[{index}]")
        _require_exact_keys(
            entry,
            frozenset({"config_id", "operator_complexity_key"}),
            f"serialized operator key[{index}]",
        )
        if entry["config_id"] != config.config_id:
            raise ValueError("serialized operator complexity grid mismatch")
        operator_keys[config.config_id] = _require_operator_complexity_key(
            entry["operator_complexity_key"],
            f"{config.config_id} serialized operator complexity key",
        )
    return identity, proof_sha256, branch_costs, operator_keys


def _validate_serialized_cost_score_inputs(
    identity: Mapping[str, object],
    bindings: Mapping[str, list[dict[str, object]]],
    dependency: Mapping[str, object],
) -> None:
    """Cross-bind serialized cost inputs to the exact dependency cells."""

    dependency_cells = _sequence_payload(
        dependency.get("cells"),
        "serialized cost dependency cells",
    )
    if len(dependency_cells) != 6:
        raise ValueError(
            "serialized cost capability requires the exact six score inputs"
        )
    expected: list[dict[str, object]] = []
    for index, raw_cell in enumerate(dependency_cells):
        cell = _mapping_payload(
            raw_cell,
            f"serialized cost dependency cells[{index}]",
        )
        subject, seed = _require_coordinate(
            cell.get("subject"),
            cell.get("seed"),
        )
        internvit = bindings[INTERNVIT_BRANCH_ID][index]
        brainrw = bindings[BRAINRW_BRANCH_ID][index]
        alignment = _mapping_payload(
            cell.get("alignment"),
            f"serialized cost dependency cells[{index}].alignment",
        )
        query_ids = _sequence_payload(
            alignment.get("query_ids"),
            f"serialized cost dependency cells[{index}].query_ids",
        )
        gallery_ids = _sequence_payload(
            alignment.get("gallery_ids"),
            f"serialized cost dependency cells[{index}].gallery_ids",
        )
        expected.append(
            {
                "alignment_sha256": cell["alignment_sha256"],
                "brainrw": _component_from_binding_payload(
                    brainrw,
                    str(cell["brainrw_binding_sha256"]),
                ),
                "cell_id": _cell_id(subject, seed),
                "gallery_count": len(gallery_ids),
                "gallery_ids_sha256": internvit["gallery_ids_sha256"],
                "internvit": _component_from_binding_payload(
                    internvit,
                    str(cell["internvit_binding_sha256"]),
                ),
                "query_count": len(query_ids),
                "query_ids_sha256": internvit["query_ids_sha256"],
                "seed": seed,
                "subject": subject,
            }
        )
    actual = _sequence_payload(
        identity.get("score_inputs"),
        "serialized cost capability score_inputs",
    )
    if actual != expected or identity.get("score_inputs_sha256") != sha256_json(
        expected
    ):
        raise ValueError(
            "serialized cost capability does not bind the exact six score inputs"
        )


def _validate_semantic_payload(
    payload: Mapping[str, object],
    sha256: str,
) -> None:
    if sha256 != STAGE1_FUSION_CONFIG_SHA256:
        raise ValueError("Stage 1 semantic config SHA-256 differs from locked v1")
    if sha256_json(payload) != sha256:
        raise ValueError("Stage 1 semantic config body/SHA-256 mismatch")
    if (
        payload.get("schema_version") != 1
        or payload.get("config_type") != "stage1_fusion"
        or payload.get("config_id") != "stage1_fusion_v1"
    ):
        raise ValueError("Stage 1 semantic config type/id mismatch")
    grid_payload = [config.to_dict() for config in enumerate_stage1_configs()]
    if payload.get("candidates") != grid_payload:
        raise ValueError("Stage 1 semantic config candidate grid mismatch")
    if payload.get("selection") != _EXPECTED_SEMANTIC_SELECTION:
        raise ValueError("Stage 1 semantic config selection semantics mismatch")
    formulas = {config.family: config.formula for config in enumerate_stage1_configs()}
    if payload.get("formulas") != formulas:
        raise ValueError("Stage 1 semantic config formula semantics mismatch")


def _validate_semantic_config(
    config: SemanticConfig,
) -> dict[str, object]:
    if not isinstance(config, SemanticConfig):
        raise TypeError("semantic_config must be a SemanticConfig")
    payload = config.canonical_payload()
    _validate_semantic_payload(payload, config.sha256)
    return payload


def _branch_cell_result(
    cell: Stage1CompositionCell,
    binding: Stage1ComponentBinding,
    metrics: RetrievalMetrics,
    cost: float,
) -> dict[str, object]:
    return {
        "cell_id": _cell_id(cell.subject, cell.seed),
        "branch_measured_ms_per_query": cost,
        "metrics": _metric_payload(metrics),
        "score_input": {
            "alignment_sha256": cell.alignment_sha256,
            "gallery_ids_sha256": binding.gallery_ids_sha256,
            **_component_score_input(binding),
            "query_ids_sha256": binding.query_ids_sha256,
        },
        "seed": cell.seed,
        "subject": cell.subject,
    }


def _fusion_score_inputs(
    cell: Stage1CompositionCell,
) -> dict[str, object]:
    return {
        "alignment_sha256": cell.alignment_sha256,
        "brainrw": _component_score_input(cell.brainrw),
        "gallery_ids_sha256": cell.internvit.gallery_ids_sha256,
        "internvit": _component_score_input(cell.internvit),
        "query_ids_sha256": cell.internvit.query_ids_sha256,
    }


def _fusion_cell_result(
    cell: Stage1CompositionCell,
    metrics: RetrievalMetrics,
    operator_complexity_key: tuple[int, int, int],
) -> dict[str, object]:
    return {
        "cell_id": _cell_id(cell.subject, cell.seed),
        "metrics": _metric_payload(metrics),
        "operator_complexity_key": list(operator_complexity_key),
        "score_inputs": _fusion_score_inputs(cell),
        "seed": cell.seed,
        "subject": cell.subject,
        "two_encoder_count": 2,
    }


def _mean_rate(values: Sequence[float]) -> float:
    decimals = [Decimal(str(value)) for value in values]
    return float(sum(decimals, Decimal(0)) / Decimal(len(decimals)))


def _score_composition_gate(
    *,
    coordinates: Sequence[tuple[int, int]],
    winner_top1: Sequence[float],
    winner_top5: Sequence[float],
    control_top1: Sequence[float],
    control_top5: Sequence[float],
) -> GateDecision:
    """Apply the exact four Stage 1 gates to the already selected winner."""

    coordinate_values = tuple(coordinates)
    if coordinate_values != PILOT_COORDINATES:
        raise ValueError("Stage 1 gate requires the ordered six-cell pilot")
    vectors: dict[str, tuple[float, ...]] = {}
    for name, raw in (
        ("winner_top1", winner_top1),
        ("winner_top5", winner_top5),
        ("control_top1", control_top1),
        ("control_top5", control_top5),
    ):
        values = tuple(
            _require_rate(value, f"Stage 1 gate {name}[{index}]")
            for index, value in enumerate(raw)
        )
        if len(values) != 6:
            raise ValueError(f"Stage 1 gate {name} must contain six cells")
        vectors[name] = values
    if any(
        top5 < top1
        for top1, top5 in zip(
            vectors["winner_top1"],
            vectors["winner_top5"],
            strict=True,
        )
    ):
        raise ValueError("Stage 1 winner Top-5 must be >= Top-1")
    if any(
        top5 < top1
        for top1, top5 in zip(
            vectors["control_top1"],
            vectors["control_top5"],
            strict=True,
        )
    ):
        raise ValueError("Stage 1 control Top-5 must be >= Top-1")

    top1_deltas: list[Decimal] = []
    top5_deltas: list[Decimal] = []
    subject_values: dict[int, list[Decimal]] = {
        subject: [] for subject in PILOT_SUBJECTS
    }
    positive_cells = 0
    for index, (subject, _seed) in enumerate(coordinate_values):
        top1_delta = Decimal(str(vectors["winner_top1"][index])) - Decimal(
            str(vectors["control_top1"][index])
        )
        top5_delta = Decimal(str(vectors["winner_top5"][index])) - Decimal(
            str(vectors["control_top5"][index])
        )
        top1_deltas.append(top1_delta)
        top5_deltas.append(top5_delta)
        subject_values[subject].append(top1_delta)
        if top1_delta > 0:
            positive_cells += 1
    count = Decimal(6)
    mean_top1 = sum(top1_deltas, Decimal(0)) / count
    mean_top5 = sum(top5_deltas, Decimal(0)) / count
    subject_means = tuple(
        (
            subject,
            sum(subject_values[subject], Decimal(0))
            / Decimal(len(subject_values[subject])),
        )
        for subject in PILOT_SUBJECTS
    )
    worst_subject = min(value for _, value in subject_means)
    criteria = {
        "mean_top1_delta": mean_top1 >= Decimal("0.003"),
        "mean_top5_delta": mean_top5 >= Decimal("-0.002"),
        "positive_cells": positive_cells >= 4,
        "subject_floor": worst_subject >= Decimal("-0.02"),
    }
    return GateDecision(
        gate_kind="pilot",
        stage=1,
        passed=all(criteria.values()),
        mean_top1_delta=float(mean_top1),
        mean_top5_delta=float(mean_top5),
        ci95=None,
        positive_cells=positive_cells,
        positive_subjects=sum(value > 0 for _, value in subject_means),
        worst_subject_top1_delta=float(worst_subject),
        subject_mean_top1_deltas=tuple(
            (subject, float(value)) for subject, value in subject_means
        ),
        criteria=criteria,
        initialization_evidence=(),
    )


def _result_vectors(
    result: Mapping[str, object],
    *,
    cell_field: str,
    context: str,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    cells = _sequence_payload(result.get("cells"), f"{context}.cells")
    if len(cells) != 6:
        raise ValueError(f"{context} must contain six cells")
    coordinates: list[tuple[int, int]] = []
    top1: list[float] = []
    top5: list[float] = []
    for index, raw in enumerate(cells):
        item = _mapping_payload(raw, f"{context}.cells[{index}]")
        selection_fields = (
            {"branch_measured_ms_per_query"}
            if cell_field == "score_input"
            else {"operator_complexity_key", "two_encoder_count"}
        )
        _require_exact_keys(
            item,
            frozenset(
                {
                    "cell_id",
                    cell_field,
                    "metrics",
                    "seed",
                    "subject",
                    *selection_fields,
                }
            ),
            f"{context}.cells[{index}]",
        )
        coordinate = _require_coordinate(item.get("subject"), item.get("seed"))
        coordinates.append(coordinate)
        if item.get("cell_id") != _cell_id(*coordinate):
            raise ValueError(f"{context} cell_id mismatch")
        if cell_field == "score_input":
            _require_cost(
                item.get("branch_measured_ms_per_query"),
                f"{context} branch measured milliseconds per query",
            )
        else:
            _require_operator_complexity_key(
                item.get("operator_complexity_key"),
                f"{context} operator complexity key",
            )
            if item.get("two_encoder_count") != 2:
                raise ValueError(f"{context} must bind two encoders")
        metrics = _validate_metric_payload(
            item.get("metrics"),
            f"{context}.cells[{index}].metrics",
        )
        top1.append(float(metrics["top1_rate"]))
        top5.append(float(metrics["top5_rate"]))
        if cell_field not in item:
            raise ValueError(f"{context} is missing {cell_field}")
    if tuple(coordinates) != PILOT_COORDINATES:
        raise ValueError(f"{context} has the wrong cell coordinates")
    return tuple(top1), tuple(top5)


def _component_from_binding_payload(
    payload: Mapping[str, object],
    binding_sha256: str,
) -> dict[str, object]:
    return {
        "binding_sha256": binding_sha256,
        "checkpoint_sha256": payload["checkpoint_sha256"],
        "resolved_config_sha256": payload["resolved_config_sha256"],
        "run_proof_sha256": payload["run_proof_sha256"],
        "score_envelope_sha256": payload["score_envelope_sha256"],
        "score_payload_sha256": payload["score_payload_sha256"],
    }


def _validate_serialized_component_binding(
    value: object,
    *,
    branch_id: str,
    coordinate: tuple[int, int],
) -> dict[str, object]:
    payload = _mapping_payload(value, f"serialized {branch_id} binding")
    _require_exact_keys(
        payload,
        frozenset(_SERIALIZED_BINDING_KEYS),
        f"serialized {branch_id} binding",
    )
    expected_config, expected_stage, expected_epochs, expected_recipe_sha = (
        _BRANCH_SPECS[branch_id]
    )
    if (
        payload["artifact_type"] != COMPONENT_BINDING_TYPE
        or payload["schema_version"] != 1
        or payload["branch_id"] != branch_id
        or (payload["subject"], payload["seed"]) != coordinate
        or payload["recipe_config_id"] != expected_config
        or payload["recipe_config_sha256"] != expected_recipe_sha
        or payload["stage"] != expected_stage
        or payload["epochs"] != expected_epochs
        or payload["scope"] != "val-dev"
        or payload["split_role"] != "val-dev"
    ):
        raise ValueError(f"serialized {branch_id} binding identity mismatch")
    for field_name in _PROOF_SHA_FIELDS:
        _require_sha256(
            payload[field_name],
            f"serialized {branch_id} binding {field_name}",
        )
    _require_git_sha(
        payload["git_sha"],
        f"serialized {branch_id} binding git_sha",
    )
    proof_sha256 = _require_sha256(
        payload["run_proof_sha256"],
        f"serialized {branch_id} run proof SHA-256",
    )
    proof_identity = {
        "artifact_type": VALIDATED_RUN_PROOF_TYPE,
        "schema_version": 1,
        **{
            key: payload[key]
            for key in _RUN_PROOF_KEYS
            if key not in {"artifact_type", "schema_version"}
        },
    }
    if proof_sha256 != sha256_json(proof_identity):
        raise ValueError("serialized component run proof SHA-256 mismatch")

    schedule = _mapping_payload(
        payload["schedule"],
        f"serialized {branch_id} schedule",
    )
    if (
        payload["schedule_sha256"] != sha256_json(schedule)
        or schedule.get("artifact_type") != COMPONENT_SCHEDULE_TYPE
        or schedule.get("schema_version") != 1
        or schedule.get("branch_id") != branch_id
        or schedule.get("config_id") != expected_config
        or schedule.get("config_sha256") != expected_recipe_sha
        or schedule.get("epochs") != expected_epochs
    ):
        raise ValueError(f"serialized {branch_id} schedule mismatch")
    expected_run_key = make_run_key(
        expected_stage,
        expected_config,
        coordinate[0],
        coordinate[1],
        str(payload["resolved_config_sha256"]),
        str(payload["input_bundle_sha256"]),
    )
    if payload["run_key"] != expected_run_key:
        raise ValueError(f"serialized {branch_id} run_key mismatch")

    source_records = _sequence_payload(
        payload["source_records"],
        f"serialized {branch_id} source_records",
    )
    if (
        not source_records
        or not isinstance(source_records[0], Mapping)
        or payload["source_records_sha256"] != sha256_json(source_records)
    ):
        raise ValueError(f"serialized {branch_id} source-record binding mismatch")
    first_record = source_records[0]
    for record_field, proof_field in (
        ("manifest_sha256", "manifest_sha256"),
        ("records_sha256", "records_sha256"),
        ("role_payload_sha256", "role_payload_sha256"),
        ("source_manifest_sha256", "source_manifest_sha256"),
        ("source_payload_sha256", "source_payload_sha256"),
    ):
        if first_record.get(record_field) != payload[proof_field]:
            raise ValueError(f"serialized {branch_id} {proof_field} mismatch")
    if first_record.get("role") != "val-dev":
        raise ValueError(f"serialized {branch_id} source role mismatch")
    if branch_id == INTERNVIT_BRANCH_ID and any(
        not isinstance(record, Mapping) or record.get("run_key") != payload["run_key"]
        for record in source_records
    ):
        raise ValueError("serialized InternViT source run_key mismatch")

    alignment = _mapping_payload(
        payload["alignment"],
        f"serialized {branch_id} alignment",
    )
    if (
        payload["alignment_sha256"] != sha256_json(alignment)
        or alignment.get("subject") != coordinate[0]
        or alignment.get("seed") != coordinate[1]
        or alignment.get("scope") != "val-dev"
        or alignment.get("split_role") != "val-dev"
        or alignment.get("protocol_sha256") != payload["protocol_sha256"]
        or alignment.get("query_ids_sha256") != payload["query_ids_sha256"]
        or alignment.get("gallery_ids_sha256") != payload["gallery_ids_sha256"]
    ):
        raise ValueError(f"serialized {branch_id} alignment mismatch")
    semantic_environment = _mapping_payload(
        payload["semantic_environment"],
        f"serialized {branch_id} semantic environment",
    )
    if payload["semantic_environment_sha256"] != sha256_json(semantic_environment):
        raise ValueError(f"serialized {branch_id} semantic environment mismatch")

    outputs = _mapping_payload(
        payload["completion_output_hashes"],
        f"serialized {branch_id} completion outputs",
    )
    expected_output_names = (
        {
            "final_checkpoint_sha256",
            "parity_sha256",
            "run_manifest_sha256",
        }
        if branch_id == INTERNVIT_BRANCH_ID
        else {
            "final_checkpoint_sha256",
            "run_manifest_sha256",
            "score_envelope_sha256",
            "score_payload_sha256",
        }
    )
    if set(outputs) != expected_output_names:
        raise ValueError(f"serialized {branch_id} completion schema mismatch")
    for field_name, digest in outputs.items():
        _require_sha256(
            digest,
            f"serialized {branch_id} completion {field_name}",
        )
    if (
        outputs["final_checkpoint_sha256"] != payload["checkpoint_sha256"]
        or outputs["run_manifest_sha256"] != payload["run_manifest_sha256"]
    ):
        raise ValueError(f"serialized {branch_id} completion identity mismatch")
    if branch_id == BRAINRW_BRANCH_ID and (
        outputs["score_payload_sha256"] != payload["score_payload_sha256"]
        or outputs["score_envelope_sha256"] != payload["score_envelope_sha256"]
    ):
        raise ValueError("serialized BrainRW completion score identity mismatch")
    return payload


@dataclass(frozen=True, init=False)
class Stage1CompositionOutcome:
    """Issuer-sealed Stage 1 selection, gate, and registry handoff."""

    status: Stage1Status
    passed: bool
    control_branch_id: Stage1BranchId
    winner_config_id: str
    control_top1: tuple[float, ...]
    control_top5: tuple[float, ...]
    winner_top1: tuple[float, ...]
    winner_top5: tuple[float, ...]
    gate: GateDecision
    dependency_sha256: str
    control_sha256: str
    selection_sha256: str
    composite_sha256: str
    composition_spec_sha256: str
    evidence_sha256: str
    candidate_decision_payload: Mapping[str, object]
    _dependency_payload: Mapping[str, object] = field(repr=False, compare=False)
    _control_payload: Mapping[str, object] = field(repr=False, compare=False)
    _selection_payload: Mapping[str, object] = field(repr=False, compare=False)
    _composite_payload: Mapping[str, object] = field(repr=False, compare=False)
    _composition_spec_payload: Mapping[str, object] = field(repr=False, compare=False)
    _evidence_payload: Mapping[str, object] = field(repr=False, compare=False)

    def __new__(cls, *_: object, **__: object) -> "Stage1CompositionOutcome":
        raise TypeError("Stage1CompositionOutcome is issuer-sealed; use compose_stage1")

    @classmethod
    def _issue(
        cls,
        *,
        issuer_token: object,
        **values: object,
    ) -> "Stage1CompositionOutcome":
        if issuer_token is not _OUTCOME_ISSUER_TOKEN:
            raise TypeError(
                "Stage1CompositionOutcome requires the compose_stage1 issuer"
            )
        expected = set(cls.__dataclass_fields__)
        if set(values) != expected:
            missing = sorted(expected - set(values))
            extra = sorted(set(values) - expected)
            raise TypeError(
                "Stage1CompositionOutcome issuer fields mismatch: "
                f"missing={missing}, extra={extra}"
            )
        instance = object.__new__(cls)
        for field_name, value in values.items():
            object.__setattr__(instance, field_name, value)
        instance.__post_init__()
        return instance

    def __post_init__(self) -> None:
        if self.status not in {"passed", "failed"}:
            raise ValueError("composition status must be passed or failed")
        if self.passed != (self.status == "passed"):
            raise ValueError("composition status/passed mismatch")
        if self.control_branch_id not in _BRANCH_IDS:
            raise ValueError("invalid Stage 1 control branch")
        _require_safe_id(self.winner_config_id, "winner_config_id")
        if not isinstance(self.gate, GateDecision):
            raise TypeError("composition gate must be a GateDecision")
        if self.gate.gate_kind != "pilot" or self.gate.stage != 1:
            raise ValueError("composition requires the Stage 1 pilot gate")

        rate_vectors: dict[str, tuple[float, ...]] = {}
        for field_name in (
            "control_top1",
            "control_top5",
            "winner_top1",
            "winner_top5",
        ):
            raw = tuple(getattr(self, field_name))
            if len(raw) != 6:
                raise ValueError(f"{field_name} must contain six cells")
            values = tuple(
                _require_rate(
                    value,
                    f"{field_name}[{index}] finite rate",
                )
                for index, value in enumerate(raw)
            )
            rate_vectors[field_name] = values
            object.__setattr__(self, field_name, values)
        for top1_name, top5_name in (
            ("control_top1", "control_top5"),
            ("winner_top1", "winner_top5"),
        ):
            if any(
                top5 < top1
                for top1, top5 in zip(
                    rate_vectors[top1_name],
                    rate_vectors[top5_name],
                    strict=True,
                )
            ):
                raise ValueError(f"{top5_name} must be >= {top1_name}")

        hashes = {
            "dependency SHA-256": self.dependency_sha256,
            "control SHA-256": self.control_sha256,
            "selection SHA-256": self.selection_sha256,
            "composite SHA-256": self.composite_sha256,
            "composition spec SHA-256": self.composition_spec_sha256,
            "evidence SHA-256": self.evidence_sha256,
        }
        for context, digest in hashes.items():
            _require_sha256(digest, context)

        dependency = _mapping_payload(self._dependency_payload, "dependency payload")
        control = _mapping_payload(self._control_payload, "control payload")
        selection = _mapping_payload(self._selection_payload, "selection payload")
        composite = _mapping_payload(self._composite_payload, "composite payload")
        spec = _mapping_payload(
            self._composition_spec_payload, "composition spec payload"
        )
        evidence = _mapping_payload(self._evidence_payload, "evidence payload")
        body_checks = (
            (
                "dependency SHA-256",
                self.dependency_sha256,
                dependency,
            ),
            ("control SHA-256", self.control_sha256, control),
            ("selection SHA-256", self.selection_sha256, selection),
            ("composite SHA-256", self.composite_sha256, composite),
            (
                "composition spec SHA-256",
                self.composition_spec_sha256,
                spec,
            ),
            ("evidence SHA-256", self.evidence_sha256, evidence),
        )
        for context, declared, body in body_checks:
            if declared != sha256_json(body):
                raise ValueError(f"{context} mismatch")

        (
            component_hashes,
            dependency_bindings,
        ) = self._validate_dependency(dependency)
        self._validate_spec(spec, dependency_bindings)
        (
            selected_control,
            selected_winner,
            cost_proof_sha256,
        ) = self._validate_selection(selection, dependency_bindings)
        self._validate_control(
            control,
            selected_control,
            dependency_bindings,
        )
        self._validate_composite(composite, selected_winner)

        control_top1, control_top5 = _result_vectors(
            selected_control,
            cell_field="score_input",
            context="selected control",
        )
        winner_top1, winner_top5 = _result_vectors(
            selected_winner,
            cell_field="score_inputs",
            context="selected winner",
        )
        if (
            control_top1 != self.control_top1
            or control_top5 != self.control_top5
            or winner_top1 != self.winner_top1
            or winner_top5 != self.winner_top5
        ):
            raise ValueError("outcome metrics differ from selected results")
        recomputed_gate = _score_composition_gate(
            coordinates=PILOT_COORDINATES,
            winner_top1=winner_top1,
            winner_top5=winner_top5,
            control_top1=control_top1,
            control_top5=control_top5,
        )
        if recomputed_gate.to_payload() != self.gate.to_payload():
            raise ValueError("outcome gate differs from selected metrics")
        if self.gate.passed != self.passed:
            raise ValueError("composition outcome/gate result mismatch")

        try:
            decision_document = _mapping_payload(
                self.candidate_decision_payload,
                "candidate decision document",
            )
            decision = CandidateDecision.from_document(decision_document)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid candidate decision document") from exc
        expected_winner_config = selected_winner["config"]
        if not isinstance(expected_winner_config, Mapping):
            raise ValueError("selected winner config must be a mapping")
        expected_control_config_sha = dependency_bindings[self.control_branch_id][0][
            "recipe_config_sha256"
        ]
        decision_checks = {
            "stage": 1,
            "candidate_id": self.winner_config_id,
            "control_id": self.control_branch_id,
            "scope": "val-dev",
            "config_sha256": STAGE1_FUSION_CONFIG_SHA256,
            "control_config_sha256": expected_control_config_sha,
            "hyperparameters_sha256": sha256_json(dict(expected_winner_config)),
            "schedule_sha256": self.composition_spec_sha256,
            "candidate_matrix_sha256": self.composite_sha256,
            "control_matrix_sha256": self.control_sha256,
            "absolute_top1": _mean_rate(winner_top1),
            "absolute_top5": _mean_rate(winner_top5),
        }
        for field_name, expected in decision_checks.items():
            if getattr(decision, field_name) != expected:
                suffix = "winner" if field_name == "candidate_id" else field_name
                raise ValueError(f"candidate decision {suffix} cross-binding mismatch")
        if (
            decision.gate.to_payload() != self.gate.to_payload()
            or decision.locked
            or decision.component_sha256s != tuple(sorted(component_hashes))
        ):
            raise ValueError("candidate decision evidence cross-binding mismatch")

        self._validate_evidence(
            evidence,
            dependency=dependency,
            control=control,
            selection=selection,
            composite=composite,
            spec=spec,
            decision=decision_document,
            cost_proof_sha256=cost_proof_sha256,
        )

        for field_name, body in (
            ("candidate_decision_payload", decision_document),
            ("_dependency_payload", dependency),
            ("_control_payload", control),
            ("_selection_payload", selection),
            ("_composite_payload", composite),
            ("_composition_spec_payload", spec),
            ("_evidence_payload", evidence),
        ):
            object.__setattr__(self, field_name, _deep_freeze(body))

    def _validate_dependency(
        self,
        payload: Mapping[str, object],
    ) -> tuple[list[str], dict[str, list[dict[str, object]]]]:
        _require_exact_keys(
            payload,
            frozenset(
                {
                    "artifact_type",
                    "cells",
                    "schema_version",
                    "scope",
                }
            ),
            "dependency set",
        )
        if (
            payload.get("artifact_type") != DEPENDENCY_SET_TYPE
            or payload.get("schema_version") != 1
            or payload.get("scope") != "val-dev"
        ):
            raise ValueError("Stage 1 dependency set type/scope mismatch")
        cells = _sequence_payload(payload.get("cells"), "dependency cells")
        if len(cells) != 6:
            raise ValueError("dependency set must contain six cells")
        coordinates: list[tuple[int, int]] = []
        component_hashes: list[str] = []
        bindings: dict[str, list[dict[str, object]]] = {
            INTERNVIT_BRANCH_ID: [],
            BRAINRW_BRANCH_ID: [],
        }
        for index, raw in enumerate(cells):
            item = _mapping_payload(raw, f"dependency cells[{index}]")
            _require_exact_keys(
                item,
                frozenset(
                    {
                        "alignment",
                        "alignment_sha256",
                        "brainrw_binding",
                        "brainrw_binding_sha256",
                        "dependency",
                        "dependency_sha256",
                        "internvit_binding",
                        "internvit_binding_sha256",
                        "seed",
                        "subject",
                    }
                ),
                f"dependency cells[{index}]",
            )
            coordinate = _require_coordinate(item.get("subject"), item.get("seed"))
            coordinates.append(coordinate)
            alignment = _mapping_payload(
                item["alignment"],
                f"dependency cells[{index}].alignment",
            )
            alignment_sha256 = _require_sha256(
                item["alignment_sha256"],
                f"dependency cells[{index}].alignment_sha256",
            )
            if alignment_sha256 != sha256_json(alignment):
                raise ValueError("dependency cell alignment SHA-256 mismatch")
            for branch_id in _BRANCH_IDS:
                binding_key = f"{branch_id}_binding"
                digest_key = f"{branch_id}_binding_sha256"
                binding = _validate_serialized_component_binding(
                    item.get(binding_key),
                    branch_id=branch_id,
                    coordinate=coordinate,
                )
                digest = _require_sha256(
                    item.get(digest_key),
                    f"dependency {digest_key}",
                )
                if digest != sha256_json(binding):
                    raise ValueError(f"dependency {branch_id} binding SHA-256 mismatch")
                if binding["alignment"] != alignment or (
                    binding["alignment_sha256"] != alignment_sha256
                ):
                    raise ValueError(f"dependency {branch_id} alignment mismatch")
                component_hashes.append(digest)
                bindings[branch_id].append(binding)
            dependency_body = _mapping_payload(
                item.get("dependency"), "cell dependency body"
            )
            dependency_sha = _require_sha256(
                item.get("dependency_sha256"),
                "cell dependency SHA-256",
            )
            if dependency_sha != sha256_json(dependency_body):
                raise ValueError("cell dependency SHA-256 mismatch")
            expected_dependency_body = {
                "alignment_sha256": alignment_sha256,
                "artifact_type": CELL_DEPENDENCY_TYPE,
                "brainrw_binding_sha256": item["brainrw_binding_sha256"],
                "gallery_ids_sha256": bindings[INTERNVIT_BRANCH_ID][index][
                    "gallery_ids_sha256"
                ],
                "internvit_binding_sha256": item["internvit_binding_sha256"],
                "protocol_sha256": bindings[INTERNVIT_BRANCH_ID][index][
                    "protocol_sha256"
                ],
                "query_ids_sha256": bindings[INTERNVIT_BRANCH_ID][index][
                    "query_ids_sha256"
                ],
                "schema_version": 1,
                "scope": "val-dev",
                "seed": coordinate[1],
                "split_role": "val-dev",
                "subject": coordinate[0],
            }
            if dependency_body != expected_dependency_body:
                raise ValueError("cell dependency component cross-binding mismatch")
        if tuple(coordinates) != PILOT_COORDINATES:
            raise ValueError("dependency set has the wrong pilot coordinates")
        common_fields = (
            "gallery_ids_sha256",
            "protocol_sha256",
            "query_ids_sha256",
            "scope",
            "split_role",
        )
        for field_name in common_fields:
            values = {
                binding[field_name]
                for branch_bindings in bindings.values()
                for binding in branch_bindings
            }
            if len(values) != 1:
                raise ValueError(f"dependency cross-cell {field_name} mismatch")
        return component_hashes, bindings

    def _validate_spec(
        self,
        payload: Mapping[str, object],
        bindings: Mapping[str, list[dict[str, object]]],
    ) -> None:
        _require_exact_keys(
            payload,
            frozenset(
                {
                    "artifact_type",
                    "components",
                    "cost",
                    "fusion",
                    "gate",
                    "schema_version",
                    "scope",
                    "selection",
                    "stage",
                }
            ),
            "composition spec",
        )
        if (
            payload.get("artifact_type") != COMPOSITION_SPEC_TYPE
            or payload.get("schema_version") != 1
            or payload.get("scope") != "val-dev"
            or payload.get("stage") != 1
            or payload.get("gate") != _GATE_SPEC
            or payload.get("selection") != _SELECTION_SPEC
        ):
            raise ValueError("composition spec type/gate/selection mismatch")
        fusion = _mapping_payload(payload.get("fusion"), "composition spec fusion")
        _require_exact_keys(
            fusion,
            frozenset({"semantic_config", "semantic_config_sha256"}),
            "composition spec fusion",
        )
        semantic_payload = _mapping_payload(
            fusion.get("semantic_config"),
            "composition spec semantic config",
        )
        _validate_semantic_payload(
            semantic_payload,
            _require_sha256(
                fusion.get("semantic_config_sha256"),
                "composition spec semantic config SHA-256",
            ),
        )
        components = _mapping_payload(
            payload.get("components"), "composition spec components"
        )
        if set(components) != set(_BRANCH_IDS):
            raise ValueError("composition spec component set mismatch")
        cost = _mapping_payload(payload.get("cost"), "composition spec cost")
        _require_exact_keys(
            cost,
            frozenset(
                {
                    "capability_identity",
                    "capability_proof_sha256",
                }
            ),
            "composition spec cost",
        )
        _serialized_cost_capability_values(
            cost["capability_identity"],
            cost["capability_proof_sha256"],
        )
        for branch_id in _BRANCH_IDS:
            component = _mapping_payload(
                components[branch_id],
                f"composition spec {branch_id}",
            )
            _require_exact_keys(
                component,
                frozenset(
                    {
                        "epochs",
                        "recipe_config_id",
                        "recipe_config_sha256",
                        "schedule",
                        "schedule_sha256",
                    }
                ),
                f"composition spec {branch_id}",
            )
            expected_config, _, expected_epochs, expected_sha = _BRANCH_SPECS[branch_id]
            schedule = _mapping_payload(
                component.get("schedule"),
                f"composition spec {branch_id} schedule",
            )
            if (
                component.get("recipe_config_id") != expected_config
                or component.get("recipe_config_sha256") != expected_sha
                or component.get("epochs") != expected_epochs
                or component.get("schedule_sha256") != sha256_json(schedule)
            ):
                raise ValueError(
                    f"composition spec {branch_id} recipe/schedule mismatch"
                )
            for binding in bindings[branch_id]:
                if (
                    binding.get("recipe_config_id") != expected_config
                    or binding.get("recipe_config_sha256") != expected_sha
                    or binding.get("epochs") != expected_epochs
                    or binding.get("schedule") != schedule
                    or binding.get("schedule_sha256") != component["schedule_sha256"]
                ):
                    raise ValueError(
                        f"composition spec {branch_id} differs from dependency"
                    )

    def _validate_selection(
        self,
        payload: Mapping[str, object],
        bindings: Mapping[str, list[dict[str, object]]],
    ) -> tuple[
        dict[str, object],
        dict[str, object],
        str,
    ]:
        _require_exact_keys(
            payload,
            frozenset(
                {
                    "artifact_type",
                    "branch_results",
                    "composition_spec_sha256",
                    "control_branch_id",
                    "cost_capability_identity",
                    "cost_capability_proof_sha256",
                    "dependency_sha256",
                    "fusion_results",
                    "schema_version",
                    "scope",
                    "selection_semantics",
                    "semantic_config",
                    "semantic_config_sha256",
                    "stage",
                    "winner_config_id",
                }
            ),
            "selection evidence",
        )
        if (
            payload.get("artifact_type") != SELECTION_EVIDENCE_TYPE
            or payload.get("schema_version") != 1
            or payload.get("scope") != "val-dev"
            or payload.get("stage") != 1
            or payload.get("selection_semantics") != _SELECTION_SPEC
            or payload.get("dependency_sha256") != self.dependency_sha256
            or payload.get("composition_spec_sha256") != self.composition_spec_sha256
        ):
            raise ValueError("selection evidence type/cross-binding mismatch")
        semantic = _mapping_payload(
            payload.get("semantic_config"),
            "selection semantic config",
        )
        _validate_semantic_payload(
            semantic,
            _require_sha256(
                payload.get("semantic_config_sha256"),
                "selection semantic config SHA-256",
            ),
        )
        (
            cost_identity,
            cost_proof_sha256,
            branch_costs,
            operator_keys,
        ) = _serialized_cost_capability_values(
            payload.get("cost_capability_identity"),
            payload.get("cost_capability_proof_sha256"),
        )
        dependency = self.dependency_payload()
        _validate_serialized_cost_score_inputs(
            cost_identity,
            bindings,
            dependency,
        )
        spec_cost = _mapping_payload(
            self.composition_spec_payload()["cost"],
            "selection composition-spec cost",
        )
        if (
            spec_cost["capability_identity"] != cost_identity
            or spec_cost["capability_proof_sha256"] != cost_proof_sha256
        ):
            raise ValueError("selection cost capability differs from composition spec")

        branch_results_raw = _sequence_payload(
            payload.get("branch_results"),
            "selection branch_results",
        )
        if len(branch_results_raw) != 2:
            raise ValueError("selection requires exactly two branches")
        branch_results: dict[str, dict[str, object]] = {}
        branch_selection_keys: dict[str, tuple[float, float, float, str]] = {}
        for index, raw in enumerate(branch_results_raw):
            result = _mapping_payload(raw, f"selection branch_results[{index}]")
            _require_exact_keys(
                result,
                frozenset(
                    {
                        "branch_id",
                        "branch_measured_ms_per_query",
                        "cells",
                    }
                ),
                f"selection branch_results[{index}]",
            )
            branch_id = result.get("branch_id")
            if branch_id not in _BRANCH_IDS or branch_id in branch_results:
                raise ValueError("selection branch result identity mismatch")
            top1, top5 = _result_vectors(
                result,
                cell_field="score_input",
                context=f"selection branch {branch_id}",
            )
            cost = branch_costs[str(branch_id)]
            if result.get("branch_measured_ms_per_query") != cost:
                raise ValueError("selection branch cost mismatch")
            result_cells = _sequence_payload(
                result["cells"], f"selection branch {branch_id} cells"
            )
            for cell_index, raw_cell in enumerate(result_cells):
                item = _mapping_payload(
                    raw_cell,
                    f"selection branch {branch_id} cells[{cell_index}]",
                )
                if item["branch_measured_ms_per_query"] != cost:
                    raise ValueError("selection branch cell cost mismatch")
                binding = bindings[str(branch_id)][cell_index]
                metrics = _validate_metric_payload(
                    item["metrics"],
                    (f"selection branch {branch_id} cells[{cell_index}].metrics"),
                )
                alignment = _mapping_payload(
                    binding["alignment"],
                    f"selection {branch_id} alignment",
                )
                if (
                    metrics["query_count"] != len(alignment["query_ids"])  # type: ignore[arg-type]
                    or metrics["gallery_count"] != len(alignment["gallery_ids"])  # type: ignore[arg-type]
                ):
                    raise ValueError("selection branch metric shape/input mismatch")
                expected_input = {
                    "alignment_sha256": binding["alignment_sha256"],
                    "gallery_ids_sha256": binding["gallery_ids_sha256"],
                    **_component_from_binding_payload(
                        binding,
                        str(
                            self.dependency_payload()["cells"][cell_index][
                                f"{branch_id}_binding_sha256"
                            ]
                        ),
                    ),
                    "query_ids_sha256": binding["query_ids_sha256"],
                }
                if item["score_input"] != expected_input:
                    raise ValueError(
                        "selection branch score-input cross-binding mismatch"
                    )
            branch_results[str(branch_id)] = result
            branch_selection_keys[str(branch_id)] = (
                -math.fsum(top1) / 6.0,
                -math.fsum(top5) / 6.0,
                cost,
                str(branch_id),
            )
        if set(branch_results) != set(_BRANCH_IDS):
            raise ValueError("selection branch result set mismatch")

        fusion_raw = _sequence_payload(
            payload.get("fusion_results"),
            "selection fusion_results",
        )
        grid = enumerate_stage1_configs()
        if len(fusion_raw) != 47:
            raise ValueError("selection requires the complete 47-config grid")
        fusion_results: dict[str, dict[str, object]] = {}
        fusion_selection_keys: dict[
            str, tuple[float, float, tuple[int, int, int], str]
        ] = {}
        dependency_cells = dependency["cells"]
        for index, (raw, config) in enumerate(zip(fusion_raw, grid, strict=True)):
            result = _mapping_payload(raw, f"selection fusion_results[{index}]")
            _require_exact_keys(
                result,
                frozenset(
                    {
                        "cells",
                        "config",
                        "config_id",
                        "operator_complexity_key",
                        "two_encoder_count",
                    }
                ),
                f"selection fusion_results[{index}]",
            )
            if (
                result.get("config_id") != config.config_id
                or result.get("config") != config.to_dict()
                or config.config_id in fusion_results
            ):
                raise ValueError("selection fusion config identity mismatch")
            top1, top5 = _result_vectors(
                result,
                cell_field="score_inputs",
                context=f"selection fusion {config.config_id}",
            )
            operator_key = operator_keys[config.config_id]
            if (
                _require_operator_complexity_key(
                    result.get("operator_complexity_key"),
                    "selection fusion operator complexity key",
                )
                != operator_key
                or result.get("two_encoder_count") != 2
            ):
                raise ValueError("selection fusion complexity mismatch")
            cells = _sequence_payload(
                result["cells"],
                f"selection fusion {config.config_id} cells",
            )
            for cell_index, raw_cell in enumerate(cells):
                item = _mapping_payload(
                    raw_cell,
                    (f"selection fusion {config.config_id} cells[{cell_index}]"),
                )
                if (
                    _require_operator_complexity_key(
                        item["operator_complexity_key"],
                        "selection fusion cell operator complexity key",
                    )
                    != operator_key
                    or item["two_encoder_count"] != 2
                ):
                    raise ValueError("selection fusion cell complexity mismatch")
                internvit = bindings[INTERNVIT_BRANCH_ID][cell_index]
                brainrw = bindings[BRAINRW_BRANCH_ID][cell_index]
                dependency_cell = dependency_cells[cell_index]
                metrics = _validate_metric_payload(
                    item["metrics"],
                    (
                        f"selection fusion {config.config_id} "
                        f"cells[{cell_index}].metrics"
                    ),
                )
                alignment = _mapping_payload(
                    internvit["alignment"],
                    f"selection fusion {config.config_id} alignment",
                )
                if (
                    metrics["query_count"] != len(alignment["query_ids"])  # type: ignore[arg-type]
                    or metrics["gallery_count"] != len(alignment["gallery_ids"])  # type: ignore[arg-type]
                ):
                    raise ValueError("selection fusion metric shape/input mismatch")
                expected_inputs = {
                    "alignment_sha256": internvit["alignment_sha256"],
                    "brainrw": _component_from_binding_payload(
                        brainrw,
                        dependency_cell["brainrw_binding_sha256"],
                    ),
                    "gallery_ids_sha256": internvit["gallery_ids_sha256"],
                    "internvit": _component_from_binding_payload(
                        internvit,
                        dependency_cell["internvit_binding_sha256"],
                    ),
                    "query_ids_sha256": internvit["query_ids_sha256"],
                }
                if item["score_inputs"] != expected_inputs:
                    raise ValueError(
                        "selection fusion score-input cross-binding mismatch"
                    )
            fusion_results[config.config_id] = result
            fusion_selection_keys[config.config_id] = (
                -math.fsum(top1) / 6.0,
                -math.fsum(top5) / 6.0,
                operator_key,
                config.config_id,
            )

        if payload.get("control_branch_id") != self.control_branch_id:
            raise ValueError("selection control branch mismatch")
        if payload.get("winner_config_id") != self.winner_config_id:
            raise ValueError("selection winner config mismatch")
        recomputed_control = min(
            branch_selection_keys,
            key=branch_selection_keys.__getitem__,
        )
        recomputed_winner = min(
            fusion_selection_keys,
            key=fusion_selection_keys.__getitem__,
        )
        if recomputed_control != self.control_branch_id:
            raise ValueError(
                "selection global control does not match full branch results"
            )
        if recomputed_winner != self.winner_config_id:
            raise ValueError(
                "selection global winner does not match full 47-by-6 results"
            )
        return (
            branch_results[self.control_branch_id],
            fusion_results[self.winner_config_id],
            cost_proof_sha256,
        )

    def _validate_control(
        self,
        payload: Mapping[str, object],
        selected: Mapping[str, object],
        bindings: Mapping[str, list[dict[str, object]]],
    ) -> None:
        _require_exact_keys(
            payload,
            frozenset(
                {
                    "artifact_type",
                    "branch_id",
                    "branch_result",
                    "config_id",
                    "config_sha256",
                    "dependency_sha256",
                    "schedule_sha256",
                    "schema_version",
                    "scope",
                    "selection_sha256",
                }
            ),
            "control evidence",
        )
        branch_bindings = bindings.get(self.control_branch_id)
        if not isinstance(branch_bindings, list) or len(branch_bindings) != 6:
            raise ValueError("control evidence dependency binding mismatch")
        selected_binding = branch_bindings[0]
        if (
            payload.get("artifact_type") != CONTROL_EVIDENCE_TYPE
            or payload.get("schema_version") != 1
            or payload.get("scope") != "val-dev"
            or payload.get("branch_id") != self.control_branch_id
            or payload.get("branch_result") != selected
            or payload.get("config_id") != selected_binding["recipe_config_id"]
            or payload.get("config_sha256") != selected_binding["recipe_config_sha256"]
            or payload.get("schedule_sha256") != selected_binding["schedule_sha256"]
            or payload.get("dependency_sha256") != self.dependency_sha256
            or payload.get("selection_sha256") != self.selection_sha256
        ):
            raise ValueError("control evidence cross-binding mismatch")

    def _validate_composite(
        self,
        payload: Mapping[str, object],
        selected: Mapping[str, object],
    ) -> None:
        _require_exact_keys(
            payload,
            frozenset(
                {
                    "artifact_type",
                    "composition_spec_sha256",
                    "config",
                    "config_id",
                    "dependency_sha256",
                    "schema_version",
                    "scope",
                    "selection_sha256",
                    "semantic_config_sha256",
                    "stage",
                    "winner_result",
                }
            ),
            "winner-only composite",
        )
        if (
            payload.get("artifact_type") != COMPOSITE_TYPE
            or payload.get("schema_version") != 1
            or payload.get("scope") != "val-dev"
            or payload.get("stage") != 1
            or payload.get("config_id") != self.winner_config_id
            or payload.get("config") != selected.get("config")
            or payload.get("winner_result") != selected
            or payload.get("dependency_sha256") != self.dependency_sha256
            or payload.get("selection_sha256") != self.selection_sha256
            or payload.get("composition_spec_sha256") != self.composition_spec_sha256
            or payload.get("semantic_config_sha256") != STAGE1_FUSION_CONFIG_SHA256
        ):
            raise ValueError("winner-only composite cross-binding mismatch")

    def _validate_evidence(
        self,
        payload: Mapping[str, object],
        *,
        dependency: Mapping[str, object],
        control: Mapping[str, object],
        selection: Mapping[str, object],
        composite: Mapping[str, object],
        spec: Mapping[str, object],
        decision: Mapping[str, object],
        cost_proof_sha256: str,
    ) -> None:
        expected = {
            "artifact_type": COMPOSITION_EVIDENCE_TYPE,
            "candidate_decision": decision,
            "candidate_decision_sha256": sha256_json(decision),
            "composite": composite,
            "composite_sha256": self.composite_sha256,
            "composition_spec": spec,
            "composition_spec_sha256": self.composition_spec_sha256,
            "control": control,
            "control_branch_id": self.control_branch_id,
            "control_sha256": self.control_sha256,
            "cost_capability_proof_sha256": cost_proof_sha256,
            "dependency": dependency,
            "dependency_sha256": self.dependency_sha256,
            "gate": self.gate.to_payload(),
            "schema_version": 1,
            "scope": "val-dev",
            "selection": selection,
            "selection_sha256": self.selection_sha256,
            "stage": 1,
            "winner_config_id": self.winner_config_id,
        }
        if payload != expected:
            raise ValueError("composition evidence body cross-binding mismatch")

    def dependency_payload(self) -> dict[str, object]:
        return _mapping_payload(self._dependency_payload, "dependency payload")

    def control_payload(self) -> dict[str, object]:
        return _mapping_payload(self._control_payload, "control payload")

    def selection_payload(self) -> dict[str, object]:
        return _mapping_payload(self._selection_payload, "selection payload")

    def composite_payload(self) -> dict[str, object]:
        return _mapping_payload(self._composite_payload, "composite payload")

    def composition_spec_payload(self) -> dict[str, object]:
        return _mapping_payload(
            self._composition_spec_payload,
            "composition spec payload",
        )

    def evidence_payload(self) -> dict[str, object]:
        return _mapping_payload(self._evidence_payload, "evidence payload")

    def candidate_decision_document(self) -> dict[str, object]:
        return _mapping_payload(
            self.candidate_decision_payload,
            "candidate decision document",
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "artifact_type": COMPOSITION_OUTCOME_TYPE,
            "candidate_decision": self.candidate_decision_document(),
            "composite": self.composite_payload(),
            "composite_sha256": self.composite_sha256,
            "composition_spec": self.composition_spec_payload(),
            "composition_spec_sha256": self.composition_spec_sha256,
            "control": self.control_payload(),
            "control_branch_id": self.control_branch_id,
            "control_sha256": self.control_sha256,
            "control_top1": list(self.control_top1),
            "control_top5": list(self.control_top5),
            "dependency": self.dependency_payload(),
            "dependency_sha256": self.dependency_sha256,
            "evidence": self.evidence_payload(),
            "evidence_sha256": self.evidence_sha256,
            "gate": self.gate.to_payload(),
            "passed": self.passed,
            "schema_version": 1,
            "scope": "val-dev",
            "selection": self.selection_payload(),
            "selection_sha256": self.selection_sha256,
            "stage": 1,
            "status": self.status,
            "winner_config_id": self.winner_config_id,
            "winner_top1": list(self.winner_top1),
            "winner_top5": list(self.winner_top5),
        }


def compose_stage1(
    cells: Sequence[Stage1CompositionCell],
    *,
    semantic_config: SemanticConfig,
    cost_capability: ValidatedStage1CostCapability,
) -> Stage1CompositionOutcome:
    """Recompute, globally select, and gate the exact Stage 1 score grid."""

    ordered = _ordered_stage1_cells(cells)
    semantic_payload = _validate_semantic_config(semantic_config)
    (
        cost_identity,
        cost_proof_sha256,
        branch_costs,
        operator_keys,
    ) = _validated_cost_capability_snapshot(cost_capability, ordered)

    dependency_cells: list[dict[str, object]] = []
    for cell in ordered:
        dependency = cell.dependency
        dependency_cells.append(
            {
                "alignment": cell.alignment_payload(),
                "alignment_sha256": cell.alignment_sha256,
                "brainrw_binding": cell.brainrw.to_payload(),
                "brainrw_binding_sha256": cell.brainrw.binding_sha256,
                "dependency": dependency.to_payload(),
                "dependency_sha256": dependency.sha256,
                "internvit_binding": cell.internvit.to_payload(),
                "internvit_binding_sha256": cell.internvit.binding_sha256,
                "seed": cell.seed,
                "subject": cell.subject,
            }
        )
    dependency_payload = {
        "artifact_type": DEPENDENCY_SET_TYPE,
        "cells": dependency_cells,
        "schema_version": 1,
        "scope": "val-dev",
    }
    dependency_sha256 = sha256_json(dependency_payload)

    first = ordered[0]
    composition_spec_payload = {
        "artifact_type": COMPOSITION_SPEC_TYPE,
        "components": {
            INTERNVIT_BRANCH_ID: {
                "epochs": first.internvit.epochs,
                "recipe_config_id": first.internvit.recipe_config_id,
                "recipe_config_sha256": (first.internvit.recipe_config_sha256),
                "schedule": first.internvit.schedule_payload(),
                "schedule_sha256": first.internvit.schedule_sha256,
            },
            BRAINRW_BRANCH_ID: {
                "epochs": first.brainrw.epochs,
                "recipe_config_id": first.brainrw.recipe_config_id,
                "recipe_config_sha256": first.brainrw.recipe_config_sha256,
                "schedule": first.brainrw.schedule_payload(),
                "schedule_sha256": first.brainrw.schedule_sha256,
            },
        },
        "cost": {
            "capability_identity": cost_identity,
            "capability_proof_sha256": cost_proof_sha256,
        },
        "fusion": {
            "semantic_config": semantic_payload,
            "semantic_config_sha256": semantic_config.sha256,
        },
        "gate": dict(_GATE_SPEC),
        "schema_version": 1,
        "scope": "val-dev",
        "selection": _deep_thaw(_deep_freeze(_SELECTION_SPEC)),
        "stage": 1,
    }
    composition_spec_sha256 = sha256_json(composition_spec_payload)

    cell_ids = tuple(_cell_id(cell.subject, cell.seed) for cell in ordered)
    branch_results: list[dict[str, object]] = []
    branch_metrics_for_selection: list[BranchValidationMetrics] = []
    branch_result_by_id: dict[str, dict[str, object]] = {}
    for branch_id in _BRANCH_IDS:
        cost = branch_costs[branch_id]
        cells_payload: list[dict[str, object]] = []
        top1: list[float] = []
        top5: list[float] = []
        for cell in ordered:
            binding = (
                cell.internvit if branch_id == INTERNVIT_BRANCH_ID else cell.brainrw
            )
            # Do not trust ScoreArtifact.metrics as the sole source.
            metrics = independent_retrieval_metrics(
                binding.score.similarity,
                binding.score.query_ids,
                binding.score.gallery_ids,
            )
            cells_payload.append(_branch_cell_result(cell, binding, metrics, cost))
            top1.append(metrics.top1_rate)
            top5.append(metrics.top5_rate)
        result = {
            "branch_id": branch_id,
            "branch_measured_ms_per_query": cost,
            "cells": cells_payload,
        }
        branch_results.append(result)
        branch_result_by_id[branch_id] = result
        branch_metrics_for_selection.append(
            BranchValidationMetrics(
                branch_id=branch_id,
                cell_ids=cell_ids,
                top1=tuple(top1),
                top5=tuple(top5),
                measured_inference_cost=cost,
            )
        )

    # One global control selection, never one selection per cell.
    control_branch_id = select_global_stronger_single_branch(
        tuple(branch_metrics_for_selection)
    )
    if control_branch_id not in _BRANCH_IDS:
        raise AssertionError("unknown Stage 1 control branch")

    fusion_results: list[dict[str, object]] = []
    fusion_selection_keys: dict[
        str, tuple[float, float, tuple[int, int, int], str]
    ] = {}
    fusion_result_by_id: dict[str, dict[str, object]] = {}
    grid = enumerate_stage1_configs()
    for config in grid:
        operator_key = operator_keys[config.config_id]
        cells_payload: list[dict[str, object]] = []
        top1: list[float] = []
        top5: list[float] = []
        for cell in ordered:
            # Every candidate is applied to the real aligned score matrices.
            fused = config.apply(
                cell.internvit.score.similarity,
                cell.brainrw.score.similarity,
                gallery_ids=cell.internvit.score.gallery_ids,
            )
            metrics = independent_retrieval_metrics(
                fused,
                cell.internvit.score.query_ids,
                cell.internvit.score.gallery_ids,
            )
            cells_payload.append(_fusion_cell_result(cell, metrics, operator_key))
            top1.append(metrics.top1_rate)
            top5.append(metrics.top5_rate)
        result = {
            "cells": cells_payload,
            "config": config.to_dict(),
            "config_id": config.config_id,
            "operator_complexity_key": list(operator_key),
            "two_encoder_count": 2,
        }
        fusion_results.append(result)
        fusion_result_by_id[config.config_id] = result
        fusion_selection_keys[config.config_id] = (
            -math.fsum(top1) / 6.0,
            -math.fsum(top5) / 6.0,
            operator_key,
            config.config_id,
        )

    # One global fusion selection over all complete 47-by-6 results.
    winner_config_id = min(
        fusion_selection_keys,
        key=fusion_selection_keys.__getitem__,
    )
    if winner_config_id not in fusion_result_by_id:
        raise AssertionError("unknown Stage 1 fusion winner")

    selection_payload = {
        "artifact_type": SELECTION_EVIDENCE_TYPE,
        "branch_results": branch_results,
        "composition_spec_sha256": composition_spec_sha256,
        "control_branch_id": control_branch_id,
        "cost_capability_identity": cost_identity,
        "cost_capability_proof_sha256": cost_proof_sha256,
        "dependency_sha256": dependency_sha256,
        "fusion_results": fusion_results,
        "schema_version": 1,
        "scope": "val-dev",
        "selection_semantics": _deep_thaw(_deep_freeze(_SELECTION_SPEC)),
        "semantic_config": semantic_payload,
        "semantic_config_sha256": semantic_config.sha256,
        "stage": 1,
        "winner_config_id": winner_config_id,
    }
    selection_sha256 = sha256_json(selection_payload)

    selected_control = branch_result_by_id[control_branch_id]
    control_binding = (
        first.internvit if control_branch_id == INTERNVIT_BRANCH_ID else first.brainrw
    )
    control_payload = {
        "artifact_type": CONTROL_EVIDENCE_TYPE,
        "branch_id": control_branch_id,
        "branch_result": selected_control,
        "config_id": control_binding.recipe_config_id,
        "config_sha256": control_binding.recipe_config_sha256,
        "dependency_sha256": dependency_sha256,
        "schedule_sha256": control_binding.schedule_sha256,
        "schema_version": 1,
        "scope": "val-dev",
        "selection_sha256": selection_sha256,
    }
    control_sha256 = sha256_json(control_payload)

    selected_winner = fusion_result_by_id[winner_config_id]
    winner_config = next(
        config for config in grid if config.config_id == winner_config_id
    )
    composite_payload = {
        "artifact_type": COMPOSITE_TYPE,
        "composition_spec_sha256": composition_spec_sha256,
        "config": winner_config.to_dict(),
        "config_id": winner_config_id,
        "dependency_sha256": dependency_sha256,
        "schema_version": 1,
        "scope": "val-dev",
        "selection_sha256": selection_sha256,
        "semantic_config_sha256": semantic_config.sha256,
        "stage": 1,
        "winner_result": selected_winner,
    }
    composite_sha256 = sha256_json(composite_payload)

    control_top1, control_top5 = _result_vectors(
        selected_control,
        cell_field="score_input",
        context="selected control",
    )
    winner_top1, winner_top5 = _result_vectors(
        selected_winner,
        cell_field="score_inputs",
        context="selected winner",
    )
    gate = _score_composition_gate(
        coordinates=PILOT_COORDINATES,
        winner_top1=winner_top1,
        winner_top5=winner_top5,
        control_top1=control_top1,
        control_top5=control_top5,
    )

    component_sha256s = tuple(
        binding.binding_sha256
        for cell in ordered
        for binding in (cell.internvit, cell.brainrw)
    )
    decision = CandidateDecision(
        stage=1,
        candidate_id=winner_config_id,
        control_id=control_branch_id,
        scope="val-dev",
        config_sha256=semantic_config.sha256,
        control_config_sha256=control_binding.recipe_config_sha256,
        hyperparameters_sha256=sha256_json(winner_config.to_dict()),
        schedule_sha256=composition_spec_sha256,
        component_sha256s=component_sha256s,
        candidate_matrix_sha256=composite_sha256,
        control_matrix_sha256=control_sha256,
        absolute_top1=_mean_rate(winner_top1),
        absolute_top5=_mean_rate(winner_top5),
        gate=gate,
        locked=False,
    )
    decision_document = decision.to_document()

    evidence_payload = {
        "artifact_type": COMPOSITION_EVIDENCE_TYPE,
        "candidate_decision": decision_document,
        "candidate_decision_sha256": sha256_json(decision_document),
        "composite": composite_payload,
        "composite_sha256": composite_sha256,
        "composition_spec": composition_spec_payload,
        "composition_spec_sha256": composition_spec_sha256,
        "control": control_payload,
        "control_branch_id": control_branch_id,
        "control_sha256": control_sha256,
        "cost_capability_proof_sha256": cost_proof_sha256,
        "dependency": dependency_payload,
        "dependency_sha256": dependency_sha256,
        "gate": gate.to_payload(),
        "schema_version": 1,
        "scope": "val-dev",
        "selection": selection_payload,
        "selection_sha256": selection_sha256,
        "stage": 1,
        "winner_config_id": winner_config_id,
    }
    evidence_sha256 = sha256_json(evidence_payload)
    return Stage1CompositionOutcome._issue(
        issuer_token=_OUTCOME_ISSUER_TOKEN,
        status="passed" if gate.passed else "failed",
        passed=gate.passed,
        control_branch_id=control_branch_id,
        winner_config_id=winner_config_id,
        control_top1=control_top1,
        control_top5=control_top5,
        winner_top1=winner_top1,
        winner_top5=winner_top5,
        gate=gate,
        dependency_sha256=dependency_sha256,
        control_sha256=control_sha256,
        selection_sha256=selection_sha256,
        composite_sha256=composite_sha256,
        composition_spec_sha256=composition_spec_sha256,
        evidence_sha256=evidence_sha256,
        candidate_decision_payload=decision_document,
        _dependency_payload=dependency_payload,
        _control_payload=control_payload,
        _selection_payload=selection_payload,
        _composite_payload=composite_payload,
        _composition_spec_payload=composition_spec_payload,
        _evidence_payload=evidence_payload,
    )


__all__ = [
    "BRAINRW_BRANCH_ID",
    "BRAINRW_CONFIG_ID",
    "BRAINRW_EPOCHS",
    "BRAINRW_RECIPE_CONFIG_SHA256",
    "BRAINRW_STAGE",
    "CELL_DEPENDENCY_TYPE",
    "COMPONENT_BINDING_TYPE",
    "COMPOSITE_TYPE",
    "COMPOSITION_EVIDENCE_TYPE",
    "COMPOSITION_OUTCOME_TYPE",
    "COMPOSITION_SPEC_TYPE",
    "CONTROL_EVIDENCE_TYPE",
    "DEPENDENCY_SET_TYPE",
    "INTERNVIT_BRANCH_ID",
    "INTERNVIT_CONFIG_ID",
    "INTERNVIT_EPOCHS",
    "INTERNVIT_RECIPE_CONFIG_SHA256",
    "INTERNVIT_STAGE",
    "PILOT_COORDINATES",
    "PILOT_SEEDS",
    "PILOT_SUBJECTS",
    "SELECTION_EVIDENCE_TYPE",
    "STAGE1_FUSION_CONFIG_SHA256",
    "Stage1CellDependency",
    "Stage1ComponentBinding",
    "Stage1CompositionCell",
    "Stage1CompositionOutcome",
    "VALIDATED_COST_CAPABILITY_TYPE",
    "ValidatedComponentRunProof",
    "ValidatedStage1CostCapability",
    "compose_stage1",
]
