"""Deterministic, validation-only Stage 1 score fusion.

Every numerical path converts inputs to ``float64`` and rejects non-finite
values before computing a result.  Score-artifact alignment deliberately
compares only shared experiment provenance: branch-specific checkpoints and
configs, stages, run keys, and runtime attestations are expected to differ,
while scope, protocol, cell identity, common source-record identity, and
ordered IDs must agree exactly.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Literal

import numpy as np

from .hashing import ordered_ids_sha256, sha256_json

if TYPE_CHECKING:
    from .scores import ScoreArtifact


FusionFamily = Literal["zscore_convex", "temperature_convex", "rrf"]

ZSCORE_FORMULA = (
    "alpha * zscore_ddof0(S_I) + (1 - alpha) * zscore_ddof0(S_C)"
)
TEMPERATURE_FORMULA = "alpha * S_I / T_I + (1 - alpha) * S_C / T_C"
RRF_FORMULA = "w / (k + rank_I) + (1 - w) / (k + rank_C)"

_SCORE_PAYLOAD_TYPE = "samga_brain_rw.score_matrix"
_SAMGA_TERMINAL_STAGES = frozenset({"stage0", "stage2"})
_BRAINRW_TERMINAL_STAGES = frozenset({"brainrw-clip-lora"})
_ALL_TERMINAL_STAGES = _SAMGA_TERMINAL_STAGES | _BRAINRW_TERMINAL_STAGES
_TRAINING_SMOKE_STAGE = "training_smoke/in_loop"
_SHARED_METADATA_KEYS = (
    "protocol_sha256",
    "subject",
    "seed",
    "stage",
    "split_role",
    "source_records",
    "source_records_sha256",
    "query_ids_sha256",
    "gallery_ids_sha256",
    "ordered_ids",
)
_SHARED_PROVENANCE_KEYS = (
    "protocol_sha256",
    "subject",
    "seed",
    "stage",
    "split_role",
    "source_records_sha256",
    "query_ids_sha256",
    "gallery_ids_sha256",
)
_COMMON_SOURCE_RECORD_KEYS = (
    "manifest_sha256",
    "records_sha256",
    "role",
    "role_payload_sha256",
    "source_manifest_sha256",
    "source_payload_byte_count",
    "source_payload_path",
    "source_payload_sha256",
)
_COMMON_SOURCE_SHA256_KEYS = frozenset(
    {
        "manifest_sha256",
        "records_sha256",
        "role_payload_sha256",
        "source_manifest_sha256",
        "source_payload_sha256",
    }
)
_LOWER_HEX = frozenset("0123456789abcdef")


def _as_score_matrix(scores: np.ndarray, name: str) -> np.ndarray:
    raw = np.asarray(scores)
    if raw.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 score matrix")
    if raw.shape[0] == 0 or raw.shape[1] == 0:
        raise ValueError(f"{name} must have non-empty query and gallery axes")
    if (
        np.issubdtype(raw.dtype, np.bool_)
        or np.issubdtype(raw.dtype, np.complexfloating)
        or not np.issubdtype(raw.dtype, np.number)
    ):
        raise TypeError(f"{name} must contain real numeric scores")
    matrix = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite scores")
    return matrix


def _aligned_matrices(
    internvit: np.ndarray,
    clip: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    internvit_matrix = _as_score_matrix(internvit, "internvit")
    clip_matrix = _as_score_matrix(clip, "clip")
    if internvit_matrix.shape != clip_matrix.shape:
        raise ValueError(
            "internvit and clip score matrices must have the same shape"
        )
    return internvit_matrix, clip_matrix


def _finite_scalar(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _unit_interval(value: object, name: str) -> float:
    result = _finite_scalar(value, name)
    if result < 0.0 or result > 1.0:
        raise ValueError(f"{name} must be between 0 and 1 inclusive")
    return result


def _gallery_utf8_keys(
    gallery_ids: Sequence[str],
    gallery_count: int,
) -> tuple[bytes, ...]:
    if isinstance(gallery_ids, (str, bytes, bytearray)):
        raise TypeError("gallery_ids must be a sequence of strings")
    identifiers = tuple(gallery_ids)
    if len(identifiers) != gallery_count:
        raise ValueError("gallery_ids must align exactly with score columns")
    if any(not isinstance(identifier, str) or not identifier for identifier in identifiers):
        raise ValueError("gallery_ids must contain non-empty strings")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate gallery IDs are forbidden")
    try:
        return tuple(identifier.encode("utf-8") for identifier in identifiers)
    except UnicodeEncodeError as exc:
        raise ValueError("gallery IDs must be valid UTF-8 strings") from exc


def _deep_thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    return value


def querywise_zscore(scores: np.ndarray) -> np.ndarray:
    """Return row-wise population z-scores as ``float64``.

    A row whose population variance is exactly zero maps to exact zeros.  No
    epsilon is introduced.
    """

    matrix = _as_score_matrix(scores, "scores")
    means = matrix.mean(axis=1, keepdims=True, dtype=np.float64)
    variances = matrix.var(axis=1, keepdims=True, ddof=0, dtype=np.float64)
    normalized = np.zeros(matrix.shape, dtype=np.float64)
    nonconstant = variances[:, 0] != 0.0
    if np.any(nonconstant):
        normalized[nonconstant] = (
            matrix[nonconstant] - means[nonconstant]
        ) / np.sqrt(variances[nonconstant])
    return normalized


def convex_fusion(
    internvit: np.ndarray,
    clip: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Fuse query-wise population z-scores by InternViT weight ``alpha``."""

    internvit_matrix, clip_matrix = _aligned_matrices(internvit, clip)
    weight = _unit_interval(alpha, "alpha")
    return (
        weight * querywise_zscore(internvit_matrix)
        + (1.0 - weight) * querywise_zscore(clip_matrix)
    )


def temperature_fusion(
    internvit: np.ndarray,
    clip: np.ndarray,
    alpha: float,
    internvit_temperature: float,
    clip_temperature: float,
) -> np.ndarray:
    """Fuse raw scores divided by positive scalar temperatures."""

    internvit_matrix, clip_matrix = _aligned_matrices(internvit, clip)
    weight = _unit_interval(alpha, "alpha")
    internvit_scale = _finite_scalar(
        internvit_temperature, "internvit_temperature"
    )
    clip_scale = _finite_scalar(clip_temperature, "clip_temperature")
    if internvit_scale <= 0.0 or clip_scale <= 0.0:
        raise ValueError("temperatures must be strictly positive")
    return (
        weight * internvit_matrix / internvit_scale
        + (1.0 - weight) * clip_matrix / clip_scale
    )


def ranked_gallery_indices(
    scores: np.ndarray,
    gallery_ids: Sequence[str],
) -> np.ndarray:
    """Return descending column indices with UTF-8 bytewise ID tie breaks."""

    matrix = _as_score_matrix(scores, "scores")
    utf8_keys = _gallery_utf8_keys(gallery_ids, matrix.shape[1])
    ranking = np.empty(matrix.shape, dtype=np.int64)
    for row_index, row in enumerate(matrix):
        ranking[row_index] = sorted(
            range(matrix.shape[1]),
            key=lambda column: (-float(row[column]), utf8_keys[column]),
        )
    return ranking


def _ordinal_ranks(
    scores: np.ndarray,
    gallery_ids: Sequence[str],
) -> np.ndarray:
    ordering = ranked_gallery_indices(scores, gallery_ids)
    ranks = np.empty(ordering.shape, dtype=np.int64)
    ordinal = np.arange(1, ordering.shape[1] + 1, dtype=np.int64)
    for row_index in range(ordering.shape[0]):
        ranks[row_index, ordering[row_index]] = ordinal
    return ranks


def reciprocal_rank_fusion(
    internvit: np.ndarray,
    clip: np.ndarray,
    k: int,
    internvit_weight: float,
    gallery_ids: Sequence[str],
) -> np.ndarray:
    """Fuse one-based ordinal branch ranks as ``float64``.

    Branch ties are resolved by UTF-8 bytewise gallery-ID order before ranks
    are assigned.  Final score ties can be ordered with
    :func:`ranked_gallery_indices`, which applies the identical rule.
    """

    internvit_matrix, clip_matrix = _aligned_matrices(internvit, clip)
    if isinstance(k, (bool, np.bool_)) or type(k) is not int:
        raise TypeError("k must be an integer")
    if k <= 0:
        raise ValueError("k must be strictly positive")
    weight = _unit_interval(internvit_weight, "internvit_weight")
    # Validate once before ranking either branch.
    _gallery_utf8_keys(gallery_ids, internvit_matrix.shape[1])
    internvit_ranks = _ordinal_ranks(internvit_matrix, gallery_ids)
    clip_ranks = _ordinal_ranks(clip_matrix, gallery_ids)
    return (
        weight / (float(k) + internvit_ranks.astype(np.float64))
        + (1.0 - weight) / (float(k) + clip_ranks.astype(np.float64))
    )


@dataclass(frozen=True)
class FusionConfig:
    """One member of the sealed 47-configuration Stage 1 grid."""

    config_id: str
    family: FusionFamily
    formula: str
    alpha: float | None = None
    internvit_temperature: float | None = None
    clip_temperature: float | None = None
    k: int | None = None
    internvit_weight: float | None = None
    rank_origin: int | None = None
    score_tie_break: str | None = None
    final_tie_break: str | None = None

    def apply(
        self,
        internvit: np.ndarray,
        clip: np.ndarray,
        *,
        gallery_ids: Sequence[str] | None = None,
    ) -> np.ndarray:
        if self.family == "zscore_convex":
            if self.alpha is None:
                raise ValueError("zscore config is missing alpha")
            return convex_fusion(internvit, clip, self.alpha)
        if self.family == "temperature_convex":
            if (
                self.alpha is None
                or self.internvit_temperature is None
                or self.clip_temperature is None
            ):
                raise ValueError("temperature config is incomplete")
            return temperature_fusion(
                internvit,
                clip,
                self.alpha,
                self.internvit_temperature,
                self.clip_temperature,
            )
        if self.family == "rrf":
            if self.k is None or self.internvit_weight is None:
                raise ValueError("RRF config is incomplete")
            if gallery_ids is None:
                raise ValueError("RRF requires gallery_ids for deterministic ties")
            return reciprocal_rank_fusion(
                internvit,
                clip,
                self.k,
                self.internvit_weight,
                gallery_ids,
            )
        raise ValueError(f"unsupported fusion family: {self.family!r}")

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "config_id": self.config_id,
            "family": self.family,
            "formula": self.formula,
        }
        if self.family == "zscore_convex":
            value["alpha"] = self.alpha
        elif self.family == "temperature_convex":
            value.update(
                {
                    "internvit_temperature": self.internvit_temperature,
                    "clip_temperature": self.clip_temperature,
                    "alpha": self.alpha,
                }
            )
        elif self.family == "rrf":
            value.update(
                {
                    "k": self.k,
                    "internvit_weight": self.internvit_weight,
                    "rank_origin": self.rank_origin,
                    "score_tie_break": self.score_tie_break,
                    "final_tie_break": self.final_tie_break,
                }
            )
        return value


def enumerate_stage1_configs() -> tuple[FusionConfig, ...]:
    """Return the sealed 11 + 27 + 9 configuration grid."""

    configs: list[FusionConfig] = []
    for index in range(11):
        alpha = index / 10
        configs.append(
            FusionConfig(
                config_id=f"s1-z-a{index * 10:03d}",
                family="zscore_convex",
                formula=ZSCORE_FORMULA,
                alpha=alpha,
            )
        )

    temperatures = ((0.5, "050"), (1.0, "100"), (2.0, "200"))
    alphas = ((0.25, "025"), (0.5, "050"), (0.75, "075"))
    for internvit_temperature, internvit_token in temperatures:
        for clip_temperature, clip_token in temperatures:
            for alpha, alpha_token in alphas:
                configs.append(
                    FusionConfig(
                        config_id=(
                            f"s1-temp-ti{internvit_token}-tc{clip_token}"
                            f"-a{alpha_token}"
                        ),
                        family="temperature_convex",
                        formula=TEMPERATURE_FORMULA,
                        alpha=alpha,
                        internvit_temperature=internvit_temperature,
                        clip_temperature=clip_temperature,
                    )
                )

    weights = ((0.25, "025"), (0.5, "050"), (0.75, "075"))
    for k in (10, 30, 60):
        for weight, weight_token in weights:
            configs.append(
                FusionConfig(
                    config_id=f"s1-rrf-k{k:03d}-w{weight_token}",
                    family="rrf",
                    formula=RRF_FORMULA,
                    k=k,
                    internvit_weight=weight,
                    rank_origin=1,
                    score_tie_break="gallery_id_utf8_bytewise",
                    final_tie_break="gallery_id_utf8_bytewise",
                )
            )

    if len(configs) != 47 or len({item.config_id for item in configs}) != 47:
        raise AssertionError("internal Stage 1 grid construction is not 47-unique")
    return tuple(configs)


def _require_common_source_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA-256 hex")
    return value


def _common_source_records(
    source_records: object,
    *,
    split_role: object,
) -> list[dict[str, object]]:
    records = _deep_thaw(source_records)
    if not isinstance(records, list) or not records:
        raise ValueError("score artifact source_records must be a non-empty list")
    common_records: list[dict[str, object]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(
                f"score artifact source_records[{index}] must be a mapping"
            )
        missing = [
            key for key in _COMMON_SOURCE_RECORD_KEYS if key not in record
        ]
        if missing:
            raise ValueError(
                "score artifact common source provenance is missing fields "
                f"at record {index}: {missing}"
            )
        common = {key: record[key] for key in _COMMON_SOURCE_RECORD_KEYS}
        for key in _COMMON_SOURCE_SHA256_KEYS:
            _require_common_source_sha256(
                common[key],
                f"source_records[{index}].{key}",
            )
        if common["role"] != split_role:
            raise ValueError(
                "score artifact source-record role differs from split role"
            )
        byte_count = common["source_payload_byte_count"]
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count <= 0
        ):
            raise ValueError(
                "score artifact source payload byte count must be positive"
            )
        source_path = common["source_payload_path"]
        if (
            not isinstance(source_path, str)
            or not source_path
            or "\x00" in source_path
            or not PurePosixPath(source_path).is_absolute()
        ):
            raise ValueError(
                "score artifact source payload path must be absolute POSIX"
            )
        common_records.append(common)
    return common_records


def _artifact_alignment_binding(
    artifact: ScoreArtifact,
    *,
    allowed_stages: frozenset[str],
) -> dict[str, object]:
    from .scores import ScoreArtifact

    if not isinstance(artifact, ScoreArtifact):
        raise TypeError("alignment requires loaded ScoreArtifact values")
    artifact.verified.revalidate()
    if artifact.verified.scope != "val-dev":
        raise PermissionError("score fusion can consume only val-dev artifacts")
    if artifact.verified.artifact.payload_type != _SCORE_PAYLOAD_TYPE:
        raise ValueError("score artifact has the wrong typed payload")
    if not isinstance(artifact.metadata, Mapping):
        raise ValueError("score artifact metadata must be a mapping")
    if not isinstance(artifact.provenance, Mapping):
        raise ValueError("score artifact provenance must be a mapping")

    missing_metadata = [
        key for key in _SHARED_METADATA_KEYS if key not in artifact.metadata
    ]
    if missing_metadata:
        raise ValueError(
            f"score artifact is missing alignment metadata: {missing_metadata}"
        )

    stage = artifact.metadata["stage"]
    if (
        stage == _TRAINING_SMOKE_STAGE
        or artifact.metadata.get("training_complete") is False
    ):
        raise ValueError(
            "score fusion rejects partial training-smoke artifacts"
        )
    if stage not in allowed_stages:
        raise ValueError(
            "score fusion requires a terminal branch stage; "
            f"received {stage!r}"
        )

    query_ids = tuple(artifact.query_ids)
    gallery_ids = tuple(artifact.gallery_ids)
    expected_query_hash = ordered_ids_sha256(query_ids)
    expected_gallery_hash = ordered_ids_sha256(gallery_ids)
    if artifact.metadata["query_ids_sha256"] != expected_query_hash:
        raise ValueError("score artifact query ID hash mismatch")
    if artifact.metadata["gallery_ids_sha256"] != expected_gallery_hash:
        raise ValueError("score artifact gallery ID hash mismatch")
    ordered_ids = tuple(artifact.metadata["ordered_ids"])
    if ordered_ids != query_ids + gallery_ids:
        raise ValueError("score artifact ordered IDs do not align")
    if artifact.metadata["split_role"] != artifact.verified.scope:
        raise ValueError("score artifact scope and split provenance mismatch")

    source_records = _deep_thaw(artifact.metadata["source_records"])
    expected_source_hash = sha256_json(source_records)
    if artifact.metadata["source_records_sha256"] != expected_source_hash:
        raise ValueError("score artifact source-record hash mismatch")
    common_source_records = _common_source_records(
        source_records,
        split_role=artifact.metadata["split_role"],
    )

    missing_provenance = [
        key for key in _SHARED_PROVENANCE_KEYS if key not in artifact.provenance
    ]
    if missing_provenance:
        raise ValueError(
            f"score artifact is missing alignment provenance: {missing_provenance}"
        )
    for key in _SHARED_PROVENANCE_KEYS:
        if artifact.provenance[key] != artifact.metadata[key]:
            raise ValueError(
                f"score artifact metadata/provenance mismatch for {key}"
            )

    return {
        "scope": artifact.verified.scope,
        "protocol_sha256": artifact.metadata["protocol_sha256"],
        "subject": artifact.metadata["subject"],
        "seed": artifact.metadata["seed"],
        "split_role": artifact.metadata["split_role"],
        "common_source_records": common_source_records,
        "common_source_records_sha256": sha256_json(common_source_records),
        "query_ids": query_ids,
        "query_ids_sha256": expected_query_hash,
        "gallery_ids": gallery_ids,
        "gallery_ids_sha256": expected_gallery_hash,
        "ordered_ids": ordered_ids,
        "ordered_ids_sha256": ordered_ids_sha256(ordered_ids),
    }


def common_alignment_payload(
    artifact: ScoreArtifact,
) -> dict[str, object]:
    """Return JSON-native common data identity for one terminal branch."""

    binding = _artifact_alignment_binding(
        artifact,
        allowed_stages=_ALL_TERMINAL_STAGES,
    )
    payload = _deep_thaw(binding)
    if not isinstance(payload, dict):
        raise AssertionError("common alignment binding must be a mapping")
    return payload


def assert_aligned(left: ScoreArtifact, right: ScoreArtifact) -> None:
    """Require a terminal SAMGA/BrainRW pair with exact common data identity."""

    left_binding = _artifact_alignment_binding(
        left,
        allowed_stages=_SAMGA_TERMINAL_STAGES,
    )
    right_binding = _artifact_alignment_binding(
        right,
        allowed_stages=_BRAINRW_TERMINAL_STAGES,
    )
    if left_binding != right_binding:
        differing = sorted(
            key
            for key in left_binding
            if left_binding[key] != right_binding[key]
        )
        raise ValueError(
            f"score artifacts have an alignment/provenance mismatch: {differing}"
        )


@dataclass(frozen=True)
class BranchValidationMetrics:
    """Six-cell validation measurements for one single branch."""

    branch_id: str
    cell_ids: tuple[str, ...]
    top1: tuple[float, ...]
    top5: tuple[float, ...]
    measured_inference_cost: float

    @property
    def top1_by_cell(self) -> tuple[float, ...]:
        return self.top1

    @property
    def top5_by_cell(self) -> tuple[float, ...]:
        return self.top5


@dataclass(frozen=True)
class FusionValidationMetrics:
    """Six-cell validation measurements for one sealed fusion config."""

    config_id: str
    cell_ids: tuple[str, ...]
    top1: tuple[float, ...]
    top5: tuple[float, ...]
    measured_inference_cost: float


def _validated_branch_key(
    branch: BranchValidationMetrics,
) -> tuple[tuple[str, ...], tuple[float, float, float, str]]:
    if not isinstance(branch, BranchValidationMetrics):
        raise TypeError("branches must contain BranchValidationMetrics values")
    if not isinstance(branch.branch_id, str) or not branch.branch_id:
        raise ValueError("branch_id must be a non-empty string")
    cell_ids = tuple(branch.cell_ids)
    if len(cell_ids) != 6 or len(set(cell_ids)) != 6:
        raise ValueError("each branch must contain exactly six unique val-dev cells")
    if any(not isinstance(cell_id, str) or not cell_id for cell_id in cell_ids):
        raise ValueError("cell IDs must be non-empty strings")
    top1 = tuple(branch.top1)
    top5 = tuple(branch.top5)
    if len(top1) != 6 or len(top5) != 6:
        raise ValueError("Top-1 and Top-5 must each contain exactly six cells")
    validated_metrics: list[tuple[float, ...]] = []
    for values, name in ((top1, "Top-1"), (top5, "Top-5")):
        converted = tuple(_finite_scalar(value, name) for value in values)
        if any(value < 0.0 or value > 1.0 for value in converted):
            raise ValueError(f"{name} values must be in [0, 1]")
        validated_metrics.append(converted)
    cost = _finite_scalar(
        branch.measured_inference_cost, "measured_inference_cost"
    )
    if cost < 0.0:
        raise ValueError("measured_inference_cost must be non-negative")
    mean_top1 = math.fsum(validated_metrics[0]) / 6.0
    mean_top5 = math.fsum(validated_metrics[1]) / 6.0
    return (
        tuple(sorted(cell_ids)),
        (-mean_top1, -mean_top5, cost, branch.branch_id),
    )


def _validated_fusion_key(
    candidate: FusionValidationMetrics,
) -> tuple[tuple[str, ...], tuple[float, float, float, str]]:
    if not isinstance(candidate, FusionValidationMetrics):
        raise TypeError(
            "candidates must contain FusionValidationMetrics values"
        )
    if not isinstance(candidate.config_id, str) or not candidate.config_id:
        raise ValueError("config_id must be a non-empty string")
    cell_ids = tuple(candidate.cell_ids)
    if len(cell_ids) != 6 or len(set(cell_ids)) != 6:
        raise ValueError(
            "each fusion config must contain exactly six unique val-dev cells"
        )
    if any(not isinstance(cell_id, str) or not cell_id for cell_id in cell_ids):
        raise ValueError("cell IDs must be non-empty strings")
    top1 = tuple(candidate.top1)
    top5 = tuple(candidate.top5)
    if len(top1) != 6 or len(top5) != 6:
        raise ValueError("Top-1 and Top-5 must each contain exactly six cells")
    validated_metrics: list[tuple[float, ...]] = []
    for values, name in ((top1, "Top-1"), (top5, "Top-5")):
        converted = tuple(_finite_scalar(value, name) for value in values)
        if any(value < 0.0 or value > 1.0 for value in converted):
            raise ValueError(f"{name} values must be in [0, 1]")
        validated_metrics.append(converted)
    cost = _finite_scalar(
        candidate.measured_inference_cost, "measured_inference_cost"
    )
    if cost < 0.0:
        raise ValueError("measured_inference_cost must be non-negative")
    mean_top1 = math.fsum(validated_metrics[0]) / 6.0
    mean_top5 = math.fsum(validated_metrics[1]) / 6.0
    return (
        tuple(sorted(cell_ids)),
        (-mean_top1, -mean_top5, cost, candidate.config_id),
    )


def select_stronger_single_branch(
    branches: Sequence[BranchValidationMetrics],
) -> str:
    """Lock one branch globally over six cells using the sealed tie order."""

    if isinstance(branches, (str, bytes, bytearray)):
        raise TypeError("branches must be a sequence of validation metrics")
    candidates = tuple(branches)
    if len(candidates) < 2:
        raise ValueError("at least two branches are required")
    validated = [
        (branch, *_validated_branch_key(branch)) for branch in candidates
    ]
    branch_ids = [branch.branch_id for branch, _, _ in validated]
    if len(set(branch_ids)) != len(branch_ids):
        raise ValueError("branch IDs must be unique")
    expected_cells = validated[0][1]
    if any(cell_ids != expected_cells for _, cell_ids, _ in validated[1:]):
        raise ValueError("branches must report the same six val-dev cells")
    return min(validated, key=lambda item: item[2])[0].branch_id


def select_best_fusion_config(
    candidates: Sequence[FusionValidationMetrics],
) -> str:
    """Select globally from the complete sealed 47-config validation grid."""

    if isinstance(candidates, (str, bytes, bytearray)):
        raise TypeError("candidates must be a sequence of validation metrics")
    values = tuple(candidates)
    expected_ids = {
        config.config_id for config in enumerate_stage1_configs()
    }
    actual_ids = [
        candidate.config_id
        for candidate in values
        if isinstance(candidate, FusionValidationMetrics)
    ]
    if (
        len(values) != 47
        or len(actual_ids) != len(values)
        or len(set(actual_ids)) != 47
        or set(actual_ids) != expected_ids
    ):
        missing = sorted(expected_ids - set(actual_ids))
        extra = sorted(set(actual_ids) - expected_ids)
        raise ValueError(
            "fusion selection requires the exact unique 47-config grid: "
            f"missing={missing}, extra={extra}"
        )
    validated = [
        (candidate, *_validated_fusion_key(candidate))
        for candidate in values
    ]
    expected_cells = validated[0][1]
    if any(cell_ids != expected_cells for _, cell_ids, _ in validated[1:]):
        raise ValueError(
            "fusion configs must report the same six val-dev cells"
        )
    return min(validated, key=lambda item: item[2])[0].config_id


def select_stage1_fusion_config(
    candidates: Sequence[FusionValidationMetrics],
) -> str:
    """Alias naming the sealed Stage 1 selection operation explicitly."""

    return select_best_fusion_config(candidates)


def select_global_stronger_single_branch(
    branches: Sequence[BranchValidationMetrics],
) -> str:
    """Explicit alias emphasizing that selection is never per-cell."""

    return select_stronger_single_branch(branches)


__all__ = [
    "BranchValidationMetrics",
    "FusionConfig",
    "FusionValidationMetrics",
    "assert_aligned",
    "common_alignment_payload",
    "convex_fusion",
    "enumerate_stage1_configs",
    "querywise_zscore",
    "ranked_gallery_indices",
    "reciprocal_rank_fusion",
    "select_best_fusion_config",
    "select_global_stronger_single_branch",
    "select_stage1_fusion_config",
    "select_stronger_single_branch",
    "temperature_fusion",
]
