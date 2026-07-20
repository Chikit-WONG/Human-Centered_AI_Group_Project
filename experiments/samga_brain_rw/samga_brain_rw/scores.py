"""Strict development score artifacts and independent retrieval metrics.

A score artifact is deliberately a three-file, typed ``val-dev`` bundle.
``metadata.json`` is written last and contains the completion marker and every
digest needed to bind the NumPy matrix, ordered IDs, provenance, and canonical
predictions CSV.  Loading is fail-closed: access and byte-level validation
finish before NumPy is allowed to interpret the payload.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import stat
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from .runtime_contract import require_pinned_semantic_environment

import numpy as np

from .access import TypedArtifact, VerifiedArtifact, verify_typed_artifacts
from .hashing import canonical_json_bytes, ordered_ids_sha256, sha256_json


SCORE_PAYLOAD_TYPE = "samga_brain_rw.score_matrix"
SCORE_SCOPE = "val-dev"
BRAINRW_TERMINAL_STAGE = "brainrw-clip-lora"
TRAINING_SMOKE_STAGE = "training_smoke/in_loop"
_EVALUATION_RUNTIME_KEYS = frozenset(
    {
        "evaluation_semantic_environment",
        "evaluation_semantic_environment_sha256",
        "training_semantic_environment",
        "training_semantic_environment_sha256",
        "evaluation_runtime_contract",
        "evaluation_runtime_contract_sha256",
        "evaluation_runtime_evidence",
        "evaluation_runtime_evidence_sha256",
    }
)
_TRAINING_PROGRESS_KEYS = frozenset(
    {"global_step", "planned_steps", "training_complete"}
)
_BRAINRW_RUNTIME_CONTRACT = {
    "accelerator": "NVIDIA A40",
    "device_type": "cuda",
    "dtype": "bfloat16",
    "schema_version": 1,
}
_BRAINRW_RUNTIME_EVIDENCE_KEYS = frozenset(
    {
        "accelerator_name",
        "bf16_supported",
        "cuda_available",
        "cuda_capability",
        "cuda_device_count",
        "cuda_device_index",
        "cuda_version",
        "device_type",
        "dtype",
        "schema_version",
        "torch_version",
        "total_memory_bytes",
    }
)
_BUNDLE_FILES = frozenset(
    {"metadata.json", "predictions.csv", "similarity.npy"}
)
_PREDICTION_COLUMNS = (
    "query_index",
    "query_id",
    "target_gallery_id",
    "predicted_gallery_id",
    "target_rank",
    "top1",
    "top5",
)
_INPUT_METADATA_KEYS = frozenset(
    {
        "checkpoint_sha256",
        "config_sha256",
        "git_sha",
        "protocol_sha256",
        "seed",
        "source_records",
        "split_role",
        "stage",
        "subject",
    }
)
_ENVELOPE_KEYS = frozenset(
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
_PROVENANCE_KEYS = frozenset(
    {
        "checkpoint_sha256",
        "config_sha256",
        "gallery_ids_sha256",
        "git_sha",
        "protocol_sha256",
        "query_ids_sha256",
        "seed",
        "source_records_sha256",
        "split_role",
        "stage",
        "subject",
    }
)
_BOUND_METADATA_KEYS = frozenset(
    {
        "checkpoint_sha256",
        "complete",
        "config_sha256",
        "gallery_ids",
        "gallery_ids_sha256",
        "git_sha",
        "ordered_ids",
        "prediction_columns",
        "prediction_row_count",
        "predictions_sha256",
        "protocol_sha256",
        "query_ids",
        "query_ids_sha256",
        "retrieval_metrics",
        "seed",
        "similarity_c_contiguous",
        "similarity_dtype",
        "similarity_shape",
        "source_records",
        "source_records_sha256",
        "split_role",
        "stage",
        "subject",
    }
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
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_SUBJECT_TEST_FILENAME_RE = re.compile(
    r"^sub-\d{2}_test\.json$", re.IGNORECASE
)
_FORMAL_TEST_RECORD_SHA256 = (
    "02d7e33b3fe8e5a571f8db232ca5fa86abb0c16981876ec84feae7ba64636f1a"
)
_MAX_METADATA_BYTES = 16 * 1024 * 1024
_MAX_PREDICTIONS_BYTES = 256 * 1024 * 1024
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


@dataclass(frozen=True)
class RetrievalPrediction:
    """One independently ranked query."""

    query_index: int
    query_id: str
    target_gallery_id: str
    predicted_gallery_id: str
    target_rank: int
    top1: bool
    top5: bool


@dataclass(frozen=True)
class RetrievalMetrics:
    """Exact aggregate counts plus the per-query ranking decisions."""

    query_count: int
    gallery_count: int
    top1_count: int
    top5_count: int
    top1_rate: float
    top5_rate: float
    predictions: tuple[RetrievalPrediction, ...]

    @property
    def target_ranks(self) -> tuple[int, ...]:
        return tuple(item.target_rank for item in self.predictions)


@dataclass(frozen=True)
class ScoreArtifact:
    """A loaded, immutable score bundle."""

    directory: Path
    similarity: np.ndarray
    query_ids: tuple[str, ...]
    gallery_ids: tuple[str, ...]
    metadata: Mapping[str, object]
    provenance: Mapping[str, object]
    metrics: RetrievalMetrics
    verified: VerifiedArtifact

    @property
    def scope(self) -> str:
        return self.verified.scope

    @property
    def query_ids_sha256(self) -> str:
        return _require_sha256(self.metadata["query_ids_sha256"], "query_ids_sha256")

    @property
    def gallery_ids_sha256(self) -> str:
        return _require_sha256(
            self.metadata["gallery_ids_sha256"], "gallery_ids_sha256"
        )

    @staticmethod
    def save(
        directory: Path,
        similarity: np.ndarray,
        query_ids: Sequence[str],
        gallery_ids: Sequence[str],
        metadata: Mapping[str, object],
    ) -> None:
        """Exclusively create one complete, fsynced ``val-dev`` bundle."""

        matrix, queries, galleries = _validate_score_inputs(
            similarity,
            query_ids,
            gallery_ids,
        )
        source = _validate_input_metadata(metadata)
        metrics = independent_retrieval_metrics(matrix, queries, galleries)
        source_records = source["source_records"]
        if not isinstance(source_records, list):
            raise AssertionError("validated source_records must be a list")

        query_hash = ordered_ids_sha256(queries)
        gallery_hash = ordered_ids_sha256(galleries)
        source_hash = sha256_json(source_records)
        predictions_bytes = _predictions_csv_bytes(metrics)
        predictions_hash = _sha256_bytes(predictions_bytes)

        array_buffer = io.BytesIO()
        np.save(array_buffer, matrix, allow_pickle=False)
        similarity_bytes = array_buffer.getvalue()
        similarity_hash = _sha256_bytes(similarity_bytes)

        provenance = {
            "checkpoint_sha256": source["checkpoint_sha256"],
            "config_sha256": source["config_sha256"],
            "gallery_ids_sha256": gallery_hash,
            "git_sha": source["git_sha"],
            "protocol_sha256": source["protocol_sha256"],
            "query_ids_sha256": query_hash,
            "seed": source["seed"],
            "source_records_sha256": source_hash,
            "split_role": SCORE_SCOPE,
            "stage": source["stage"],
            "subject": source["subject"],
        }
        if source["stage"] == TRAINING_SMOKE_STAGE:
            provenance.update(
                {
                    key: source[key]
                    for key in _TRAINING_PROGRESS_KEYS
                }
            )
        if source["stage"] == BRAINRW_TERMINAL_STAGE:
            provenance.update(
                {
                    key: source[key]
                    for key in _EVALUATION_RUNTIME_KEYS
                }
            )
        metric_payload = _metrics_payload(metrics)
        bound_metadata = {
            "checkpoint_sha256": source["checkpoint_sha256"],
            "complete": True,
            "config_sha256": source["config_sha256"],
            "gallery_ids": list(galleries),
            "gallery_ids_sha256": gallery_hash,
            "git_sha": source["git_sha"],
            "ordered_ids": [*queries, *galleries],
            "prediction_columns": list(_PREDICTION_COLUMNS),
            "prediction_row_count": len(queries),
            "predictions_sha256": predictions_hash,
            "protocol_sha256": source["protocol_sha256"],
            "query_ids": list(queries),
            "query_ids_sha256": query_hash,
            "retrieval_metrics": metric_payload,
            "seed": source["seed"],
            "similarity_c_contiguous": True,
            "similarity_dtype": str(matrix.dtype),
            "similarity_shape": [int(value) for value in matrix.shape],
            "source_records": source_records,
            "source_records_sha256": source_hash,
            "split_role": SCORE_SCOPE,
            "stage": source["stage"],
            "subject": source["subject"],
        }
        if source["stage"] == TRAINING_SMOKE_STAGE:
            bound_metadata.update(
                {
                    key: source[key]
                    for key in _TRAINING_PROGRESS_KEYS
                }
            )
        if source["stage"] == BRAINRW_TERMINAL_STAGE:
            bound_metadata.update(
                {
                    key: source[key]
                    for key in _EVALUATION_RUNTIME_KEYS
                }
            )
        envelope = {
            "schema_version": 1,
            "payload_type": SCORE_PAYLOAD_TYPE,
            "scope": SCORE_SCOPE,
            "source_records_sha256": source_hash,
            "ordered_ids_sha256": ordered_ids_sha256(
                [*queries, *galleries]
            ),
            "payload_sha256": similarity_hash,
            "provenance": provenance,
            "provenance_sha256": sha256_json(provenance),
            "metadata": bound_metadata,
            "metadata_sha256": sha256_json(bound_metadata),
        }
        metadata_bytes = canonical_json_bytes(envelope) + b"\n"
        _publish_bundle(
            Path(directory),
            similarity_bytes=similarity_bytes,
            predictions_bytes=predictions_bytes,
            metadata_bytes=metadata_bytes,
        )

    @classmethod
    def load(
        cls,
        directory: Path,
        allowed_scopes: Collection[str],
    ) -> "ScoreArtifact":
        """Load one strict score bundle after all pre-NumPy guards pass."""

        _require_development_scopes(allowed_scopes)
        bundle = _absolute_path(Path(directory))
        _preflight_path(bundle, "score bundle")
        return _load_score_bundle(bundle)


def _load_score_bundle(bundle: Path) -> ScoreArtifact:
    bundle_fd = _open_directory_components(
        bundle,
        create=False,
        context="score bundle",
    )
    initial_directory_stat = os.fstat(bundle_fd)
    try:
        if not stat.S_ISDIR(initial_directory_stat.st_mode):
            raise ValueError("score bundle must be a directory")
        _require_exact_bundle_files(bundle_fd)

        descriptor = TypedArtifact(
            payload_type=SCORE_PAYLOAD_TYPE,
            payload_path=bundle / "similarity.npy",
            envelope_path=bundle / "metadata.json",
        )
        verified = verify_typed_artifacts(SCORE_SCOPE, [descriptor])[0]
        _require_exact_bundle_files(bundle_fd)

        envelope_raw, envelope_stat, envelope_digest = (
            _read_relative_regular_bytes(
                bundle_fd,
                "metadata.json",
                context="score metadata",
                limit=_MAX_METADATA_BYTES,
            )
        )
        _require_verified_file(
            envelope_stat,
            envelope_digest,
            expected_identity=(
                verified.envelope_device,
                verified.envelope_inode,
                verified.envelope_size,
                verified.envelope_mtime_ns,
                verified.envelope_ctime_ns,
            ),
            expected_sha256=verified.envelope_sha256,
            context="verified envelope",
        )
        envelope = _parse_json_object(envelope_raw, "score metadata")
        (
            bound,
            provenance,
            queries,
            galleries,
            declared_shape,
            declared_dtype,
        ) = _validate_envelope(envelope, verified)

        prediction_bytes, _, prediction_digest = (
            _read_relative_regular_bytes(
                bundle_fd,
                "predictions.csv",
                context="predictions CSV",
                limit=_MAX_PREDICTIONS_BYTES,
            )
        )
        if prediction_digest != bound["predictions_sha256"]:
            raise ValueError("predictions SHA-256 mismatch")

        _require_exact_bundle_files(bundle_fd)
        matrix = _load_relative_similarity(bundle_fd, verified)
        _validate_loaded_matrix(
            matrix,
            queries,
            galleries,
            declared_shape=declared_shape,
            declared_dtype=declared_dtype,
        )
        metrics = independent_retrieval_metrics(
            matrix,
            queries,
            galleries,
        )
        if _metrics_payload(metrics) != bound["retrieval_metrics"]:
            raise ValueError(
                "declared retrieval metrics do not match scores"
            )
        expected_predictions = _predictions_csv_bytes(metrics)
        if prediction_bytes != expected_predictions:
            raise ValueError(
                "predictions CSV does not match independently ranked scores"
            )

        immutable_matrix = _immutable_matrix_copy(matrix)
        _require_exact_bundle_files(bundle_fd)
        final_directory_stat = os.fstat(bundle_fd)
        if _stat_identity(initial_directory_stat) != _stat_identity(
            final_directory_stat
        ):
            raise ValueError("score bundle directory changed during load")
        _require_bundle_path_identity(bundle, initial_directory_stat)
        return ScoreArtifact(
            directory=bundle,
            similarity=immutable_matrix,
            query_ids=queries,
            gallery_ids=galleries,
            metadata=_deep_freeze(bound),
            provenance=_deep_freeze(provenance),
            metrics=metrics,
            verified=verified,
        )
    finally:
        os.close(bundle_fd)


def independent_retrieval_metrics(
    scores: np.ndarray,
    query_ids: Sequence[str],
    gallery_ids: Sequence[str],
) -> RetrievalMetrics:
    """Rank targets by ID with deterministic UTF-8 tie-breaking."""

    matrix, queries, galleries = _validate_score_inputs(
        scores,
        query_ids,
        gallery_ids,
        copy=False,
    )
    gallery_index = {identifier: index for index, identifier in enumerate(galleries)}
    missing = [identifier for identifier in queries if identifier not in gallery_index]
    if missing:
        raise ValueError(
            f"query target IDs are missing from the gallery: {missing!r}"
        )

    gallery_utf8 = tuple(identifier.encode("utf-8") for identifier in galleries)
    predictions: list[RetrievalPrediction] = []
    for query_index, query_id in enumerate(queries):
        row = matrix[query_index]
        ranking = sorted(
            range(len(galleries)),
            key=lambda index: (-row[index].item(), gallery_utf8[index]),
        )
        target_index = gallery_index[query_id]
        target_rank = ranking.index(target_index) + 1
        predicted_id = galleries[ranking[0]]
        predictions.append(
            RetrievalPrediction(
                query_index=query_index,
                query_id=query_id,
                target_gallery_id=query_id,
                predicted_gallery_id=predicted_id,
                target_rank=target_rank,
                top1=target_rank == 1,
                top5=target_rank <= 5,
            )
        )

    values = tuple(predictions)
    top1_count = sum(item.top1 for item in values)
    top5_count = sum(item.top5 for item in values)
    return RetrievalMetrics(
        query_count=len(queries),
        gallery_count=len(galleries),
        top1_count=top1_count,
        top5_count=top5_count,
        top1_rate=top1_count / len(queries),
        top5_rate=top5_count / len(queries),
        predictions=values,
    )


def _validate_score_inputs(
    similarity: np.ndarray,
    query_ids: Sequence[str],
    gallery_ids: Sequence[str],
    *,
    copy: bool = True,
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    if not isinstance(similarity, np.ndarray):
        raise TypeError("similarity must be a NumPy array")
    if similarity.ndim != 2:
        raise ValueError("similarity must be a two-dimensional matrix")
    if not np.issubdtype(similarity.dtype, np.floating):
        raise ValueError("similarity dtype must be floating point")
    queries = _normalize_ids(query_ids, "query")
    galleries = _normalize_ids(gallery_ids, "gallery")
    if similarity.shape != (len(queries), len(galleries)):
        raise ValueError(
            "similarity shape must equal query-count by gallery-count"
        )
    if not bool(np.isfinite(similarity).all()):
        raise ValueError("similarity must contain only finite values")
    if copy:
        matrix = np.array(similarity, copy=True, order="C")
    else:
        matrix = similarity
    return matrix, queries, galleries


def _normalize_ids(
    values: Sequence[str],
    kind: str,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{kind}_ids must be a sequence of strings")
    identifiers = tuple(values)
    if not identifiers:
        raise ValueError(f"{kind}_ids must not be empty")
    for identifier in identifiers:
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"{kind}_ids must contain non-empty strings")
        if "\n" in identifier or "\r" in identifier:
            raise ValueError(f"{kind}_ids must not contain line breaks")
        try:
            identifier.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError(f"{kind}_ids must be valid UTF-8 strings") from exc
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"duplicate {kind} IDs are forbidden")
    return identifiers


def _validate_input_metadata(
    metadata: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    expected_keys = _INPUT_METADATA_KEYS
    if metadata.get("stage") == TRAINING_SMOKE_STAGE:
        expected_keys = expected_keys | _TRAINING_PROGRESS_KEYS
    if metadata.get("stage") == BRAINRW_TERMINAL_STAGE:
        expected_keys = expected_keys | _EVALUATION_RUNTIME_KEYS
    _strict_keys(metadata, expected_keys, "score metadata input")
    cloned = _json_clone(dict(metadata), "score metadata input")
    if not isinstance(cloned, dict):
        raise AssertionError("JSON-cloned metadata must be an object")

    for key in ("checkpoint_sha256", "config_sha256", "protocol_sha256"):
        _require_sha256(cloned[key], key)
    git_sha = cloned["git_sha"]
    if not isinstance(git_sha, str) or _GIT_SHA_RE.fullmatch(git_sha) is None:
        raise ValueError("git_sha must be a lowercase 40- or 64-hex digest")
    _require_nonnegative_int(cloned["seed"], "seed")
    subject = _require_positive_int(cloned["subject"], "subject")
    if subject > 10:
        raise ValueError("subject must be between 1 and 10")
    stage = cloned["stage"]
    if not isinstance(stage, str) or not stage:
        raise ValueError("stage must be a non-empty string")
    if stage == TRAINING_SMOKE_STAGE:
        _validate_training_smoke_progress(cloned)
    if stage == BRAINRW_TERMINAL_STAGE:
        _validate_evaluation_runtime_attestation(cloned)
    if cloned["split_role"] != SCORE_SCOPE:
        raise ValueError("score artifacts are restricted to val-dev")
    source_records = cloned["source_records"]
    if not isinstance(source_records, list):
        raise ValueError("source_records must be a JSON array")
    _reject_formal_scope_markers(source_records, path=("source_records",))
    return cloned


def _metrics_payload(metrics: RetrievalMetrics) -> dict[str, object]:
    return {
        "gallery_count": metrics.gallery_count,
        "query_count": metrics.query_count,
        "top1_count": metrics.top1_count,
        "top1_rate": metrics.top1_rate,
        "top5_count": metrics.top5_count,
        "top5_rate": metrics.top5_rate,
    }


def _validate_training_smoke_progress(
    value: Mapping[str, object],
) -> None:
    if value.get("training_complete") is not False:
        raise ValueError(
            "training-smoke score must declare training_complete=false"
        )
    global_step = _require_positive_int(
        value.get("global_step"),
        "global_step",
    )
    planned_steps = _require_positive_int(
        value.get("planned_steps"),
        "planned_steps",
    )
    if global_step >= planned_steps:
        raise ValueError(
            "training-smoke score must precede the planned terminal step"
        )


def _validate_evaluation_runtime_attestation(
    value: Mapping[str, object],
) -> None:
    semantic_environments: dict[str, dict[str, object]] = {}
    for role in ("training", "evaluation"):
        environment = require_pinned_semantic_environment(
            value.get(f"{role}_semantic_environment")
        )
        claimed_sha256 = _require_sha256(
            value.get(f"{role}_semantic_environment_sha256"),
            f"{role}_semantic_environment_sha256",
        )
        if sha256_json(environment) != claimed_sha256:
            raise ValueError(
                f"{role} semantic environment hash mismatch"
            )
        semantic_environments[role] = environment
    if (
        semantic_environments["training"]
        != semantic_environments["evaluation"]
    ):
        raise ValueError(
            "training and evaluation semantic environments differ"
        )
    contract = _require_mapping(
        value.get("evaluation_runtime_contract"),
        "evaluation runtime contract",
    )
    if dict(contract) != _BRAINRW_RUNTIME_CONTRACT:
        raise ValueError(
            "evaluation runtime contract is not CUDA+bfloat16+A40"
        )
    contract_sha256 = _require_sha256(
        value.get("evaluation_runtime_contract_sha256"),
        "evaluation_runtime_contract_sha256",
    )
    if sha256_json(contract) != contract_sha256:
        raise ValueError("evaluation runtime contract hash mismatch")

    evidence = _require_mapping(
        value.get("evaluation_runtime_evidence"),
        "evaluation runtime evidence",
    )
    _strict_keys(
        evidence,
        _BRAINRW_RUNTIME_EVIDENCE_KEYS,
        "evaluation runtime evidence",
    )
    evidence_sha256 = _require_sha256(
        value.get("evaluation_runtime_evidence_sha256"),
        "evaluation_runtime_evidence_sha256",
    )
    if sha256_json(evidence) != evidence_sha256:
        raise ValueError("evaluation runtime evidence hash mismatch")
    if (
        evidence["schema_version"] != 1
        or evidence["cuda_available"] is not True
        or evidence["bf16_supported"] is not True
        or evidence["accelerator_name"] != contract["accelerator"]
        or evidence["device_type"] != contract["device_type"]
        or evidence["dtype"] != contract["dtype"]
    ):
        raise ValueError(
            "evaluation runtime evidence differs from the contract"
        )
    capability = evidence["cuda_capability"]
    if not isinstance(capability, list) or capability != [8, 6]:
        raise ValueError(
            "evaluation runtime evidence is not an A40 capability"
        )
    device_count = _require_positive_int(
        evidence["cuda_device_count"],
        "evaluation CUDA device count",
    )
    device_index = _require_nonnegative_int(
        evidence["cuda_device_index"],
        "evaluation CUDA device index",
    )
    if device_index >= device_count:
        raise ValueError(
            "evaluation CUDA device index is out of range"
        )
    _require_positive_int(
        evidence["total_memory_bytes"],
        "evaluation CUDA total memory",
    )
    for key in ("cuda_version", "torch_version"):
        if not isinstance(evidence[key], str) or not evidence[key]:
            raise ValueError(
                f"evaluation runtime evidence {key} is invalid"
            )


def _predictions_csv_bytes(metrics: RetrievalMetrics) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(_PREDICTION_COLUMNS)
    for item in metrics.predictions:
        writer.writerow(
            (
                item.query_index,
                item.query_id,
                item.target_gallery_id,
                item.predicted_gallery_id,
                item.target_rank,
                int(item.top1),
                int(item.top5),
            )
        )
    return output.getvalue().encode("utf-8")


def _validate_envelope(
    envelope: Mapping[str, object],
    verified: VerifiedArtifact,
) -> tuple[
    Mapping[str, object],
    Mapping[str, object],
    tuple[str, ...],
    tuple[str, ...],
    tuple[int, int],
    np.dtype[object],
]:
    _strict_keys(envelope, _ENVELOPE_KEYS, "score envelope")
    if envelope["schema_version"] != 1 or type(envelope["schema_version"]) is not int:
        raise ValueError("score envelope schema_version must be 1")
    if envelope["payload_type"] != SCORE_PAYLOAD_TYPE:
        raise ValueError("score envelope payload_type mismatch")
    if envelope["scope"] != SCORE_SCOPE:
        raise PermissionError(
            f"{SCORE_SCOPE} cannot consume {envelope['scope']!r} artifacts"
        )
    if envelope["payload_sha256"] != verified.payload_sha256:
        raise ValueError("verified score payload SHA-256 binding mismatch")

    bound = _require_mapping(envelope["metadata"], "score bound metadata")
    provenance = _require_mapping(envelope["provenance"], "score provenance")
    bound_keys = _BOUND_METADATA_KEYS
    if bound.get("stage") == TRAINING_SMOKE_STAGE:
        bound_keys = bound_keys | _TRAINING_PROGRESS_KEYS
    if bound.get("stage") == BRAINRW_TERMINAL_STAGE:
        bound_keys = bound_keys | _EVALUATION_RUNTIME_KEYS
    provenance_keys = _PROVENANCE_KEYS
    if provenance.get("stage") == TRAINING_SMOKE_STAGE:
        provenance_keys = provenance_keys | _TRAINING_PROGRESS_KEYS
    if provenance.get("stage") == BRAINRW_TERMINAL_STAGE:
        provenance_keys = provenance_keys | _EVALUATION_RUNTIME_KEYS
    _strict_keys(bound, bound_keys, "score bound metadata")
    _strict_keys(provenance, provenance_keys, "score provenance")
    if sha256_json(bound) != _require_sha256(
        envelope["metadata_sha256"], "metadata_sha256"
    ):
        raise ValueError("metadata SHA-256 mismatch")
    if sha256_json(provenance) != _require_sha256(
        envelope["provenance_sha256"], "provenance_sha256"
    ):
        raise ValueError("provenance SHA-256 mismatch")
    if bound["complete"] is not True:
        raise ValueError("score bundle is not complete")
    if bound["split_role"] != SCORE_SCOPE or provenance["split_role"] != SCORE_SCOPE:
        raise PermissionError("score bundle is not a val-dev artifact")

    queries = _normalize_ids(
        _require_json_string_list(bound["query_ids"], "query_ids"),
        "query",
    )
    galleries = _normalize_ids(
        _require_json_string_list(bound["gallery_ids"], "gallery_ids"),
        "gallery",
    )
    query_hash = ordered_ids_sha256(queries)
    gallery_hash = ordered_ids_sha256(galleries)
    if query_hash != _require_sha256(bound["query_ids_sha256"], "query_ids_sha256"):
        raise ValueError("query-ID SHA-256 mismatch")
    if gallery_hash != _require_sha256(
        bound["gallery_ids_sha256"], "gallery_ids_sha256"
    ):
        raise ValueError("gallery-ID SHA-256 mismatch")
    if provenance["query_ids_sha256"] != query_hash:
        raise ValueError("query-ID SHA-256 provenance mismatch")
    if provenance["gallery_ids_sha256"] != gallery_hash:
        raise ValueError("gallery-ID SHA-256 provenance mismatch")

    ordered = _require_json_string_list(bound["ordered_ids"], "ordered_ids")
    if ordered != [*queries, *galleries]:
        raise ValueError("bound ordered IDs differ from query/gallery IDs")
    if ordered_ids_sha256(ordered) != _require_sha256(
        envelope["ordered_ids_sha256"], "ordered_ids_sha256"
    ):
        raise ValueError("ordered IDs SHA-256 mismatch")

    source_records = bound["source_records"]
    if not isinstance(source_records, list):
        raise ValueError("source_records must be a JSON array")
    source_hash = sha256_json(source_records)
    if source_hash != _require_sha256(
        bound["source_records_sha256"], "source_records_sha256"
    ):
        raise ValueError("source-record SHA-256 mismatch")
    if source_hash != envelope["source_records_sha256"]:
        raise ValueError("source-record SHA-256 envelope mismatch")
    if source_hash != provenance["source_records_sha256"]:
        raise ValueError("source-record SHA-256 provenance mismatch")
    _reject_formal_scope_markers(source_records, path=("source_records",))

    identity_keys = [
        "checkpoint_sha256",
        "config_sha256",
        "git_sha",
        "protocol_sha256",
        "seed",
        "stage",
        "subject",
    ]
    if bound["stage"] == TRAINING_SMOKE_STAGE:
        identity_keys.extend(sorted(_TRAINING_PROGRESS_KEYS))
    if bound["stage"] == BRAINRW_TERMINAL_STAGE:
        identity_keys.extend(sorted(_EVALUATION_RUNTIME_KEYS))
    for key in identity_keys:
        if bound[key] != provenance[key]:
            raise ValueError(f"{key} provenance binding mismatch")
    _validate_bound_identity_values(bound)

    columns = _require_json_string_list(
        bound["prediction_columns"], "prediction_columns"
    )
    if tuple(columns) != _PREDICTION_COLUMNS:
        raise ValueError("prediction_columns do not match the score schema")
    if _require_nonnegative_int(
        bound["prediction_row_count"], "prediction_row_count"
    ) != len(queries):
        raise ValueError("prediction_row_count mismatch")
    _require_sha256(bound["predictions_sha256"], "predictions_sha256")

    shape_value = bound["similarity_shape"]
    if (
        not isinstance(shape_value, list)
        or len(shape_value) != 2
        or any(type(value) is not int or value <= 0 for value in shape_value)
    ):
        raise ValueError("similarity_shape must contain two positive integers")
    declared_shape = (shape_value[0], shape_value[1])
    if declared_shape != (len(queries), len(galleries)):
        raise ValueError("declared similarity shape does not match bound IDs")
    if bound["similarity_c_contiguous"] is not True:
        raise ValueError("score matrix must declare C-contiguous storage")
    dtype_value = bound["similarity_dtype"]
    if not isinstance(dtype_value, str):
        raise ValueError("similarity_dtype must be a string")
    try:
        declared_dtype = np.dtype(dtype_value)
    except TypeError as exc:
        raise ValueError("similarity_dtype is not a NumPy dtype") from exc
    if (
        str(declared_dtype) != dtype_value
        or not np.issubdtype(declared_dtype, np.floating)
    ):
        raise ValueError("similarity_dtype must be a canonical floating dtype")

    declared_metrics = _require_mapping(
        bound["retrieval_metrics"], "retrieval_metrics"
    )
    _validate_declared_metrics(
        declared_metrics,
        query_count=len(queries),
        gallery_count=len(galleries),
    )
    return bound, provenance, queries, galleries, declared_shape, declared_dtype


def _validate_bound_identity_values(bound: Mapping[str, object]) -> None:
    for key in ("checkpoint_sha256", "config_sha256", "protocol_sha256"):
        _require_sha256(bound[key], key)
    git_sha = bound["git_sha"]
    if not isinstance(git_sha, str) or _GIT_SHA_RE.fullmatch(git_sha) is None:
        raise ValueError("git_sha must be a lowercase 40- or 64-hex digest")
    _require_nonnegative_int(bound["seed"], "seed")
    subject = _require_positive_int(bound["subject"], "subject")
    if subject > 10:
        raise ValueError("subject must be between 1 and 10")
    if not isinstance(bound["stage"], str) or not bound["stage"]:
        raise ValueError("stage must be a non-empty string")
    if bound["stage"] == TRAINING_SMOKE_STAGE:
        _validate_training_smoke_progress(bound)
    if bound["stage"] == BRAINRW_TERMINAL_STAGE:
        _validate_evaluation_runtime_attestation(bound)


def _validate_declared_metrics(
    value: Mapping[str, object],
    *,
    query_count: int,
    gallery_count: int,
) -> None:
    _strict_keys(value, _METRIC_KEYS, "retrieval_metrics")
    if _require_nonnegative_int(value["query_count"], "query_count") != query_count:
        raise ValueError("declared query_count mismatch")
    if (
        _require_nonnegative_int(value["gallery_count"], "gallery_count")
        != gallery_count
    ):
        raise ValueError("declared gallery_count mismatch")
    top1 = _require_nonnegative_int(value["top1_count"], "top1_count")
    top5 = _require_nonnegative_int(value["top5_count"], "top5_count")
    if top1 > top5 or top5 > query_count:
        raise ValueError("declared Top-1/Top-5 counts are invalid")
    for key in ("top1_rate", "top5_rate"):
        rate = value[key]
        if type(rate) not in (int, float) or not np.isfinite(rate):
            raise ValueError(f"{key} must be a finite number")
        if not 0.0 <= float(rate) <= 1.0:
            raise ValueError(f"{key} must be between zero and one")
    if float(value["top1_rate"]) != top1 / query_count:
        raise ValueError("declared top1_rate mismatch")
    if float(value["top5_rate"]) != top5 / query_count:
        raise ValueError("declared top5_rate mismatch")


def _validate_loaded_matrix(
    matrix: np.ndarray,
    queries: tuple[str, ...],
    galleries: tuple[str, ...],
    *,
    declared_shape: tuple[int, int],
    declared_dtype: np.dtype[object],
) -> None:
    if matrix.ndim != 2 or matrix.shape != declared_shape:
        raise ValueError("loaded similarity shape differs from metadata")
    if matrix.shape != (len(queries), len(galleries)):
        raise ValueError("loaded similarity shape differs from bound IDs")
    if matrix.dtype != declared_dtype:
        raise ValueError("loaded similarity dtype differs from metadata")
    if not np.issubdtype(matrix.dtype, np.floating):
        raise ValueError("loaded similarity dtype must be floating point")
    if not matrix.flags.c_contiguous:
        raise ValueError("loaded similarity must be C-contiguous")
    if not bool(np.isfinite(matrix).all()):
        raise ValueError("loaded similarity contains a non-finite value")


def _require_development_scopes(allowed_scopes: Collection[str]) -> None:
    if isinstance(allowed_scopes, (str, bytes, bytearray)):
        raise TypeError("allowed_scopes must be a collection of scope strings")
    try:
        scopes = frozenset(allowed_scopes)
    except TypeError as exc:
        raise TypeError("allowed_scopes must contain hashable strings") from exc
    if scopes != {SCORE_SCOPE}:
        raise PermissionError(
            "ScoreArtifact.load only permits allowed_scopes={'val-dev'}"
        )


def _require_exact_bundle_files(directory: Path | int) -> None:
    try:
        actual = frozenset(os.listdir(directory))
    except OSError as exc:
        raise ValueError("score bundle directory cannot be read safely") from exc
    if actual != _BUNDLE_FILES:
        raise ValueError(
            "score bundle file set must be exactly metadata.json, "
            "predictions.csv, and similarity.npy"
        )


def _open_directory_components(
    path: Path,
    *,
    create: bool,
    context: str,
) -> int:
    absolute = _absolute_path(path)
    descriptor = os.open(
        absolute.anchor,
        os.O_RDONLY | _O_DIRECTORY | _O_CLOEXEC,
    )
    try:
        for component in absolute.parts[1:]:
            try:
                next_descriptor = os.open(
                    component,
                    os.O_RDONLY
                    | _O_DIRECTORY
                    | _O_NOFOLLOW
                    | _O_CLOEXEC,
                    dir_fd=descriptor,
                )
            except FileNotFoundError as exc:
                if not create:
                    raise ValueError(
                        f"{context} could not be opened securely"
                    ) from exc
                try:
                    os.mkdir(component, mode=0o700, dir_fd=descriptor)
                    os.fsync(descriptor)
                    next_descriptor = os.open(
                        component,
                        os.O_RDONLY
                        | _O_DIRECTORY
                        | _O_NOFOLLOW
                        | _O_CLOEXEC,
                        dir_fd=descriptor,
                    )
                except OSError as create_exc:
                    raise ValueError(
                        f"{context} contains an unsafe component"
                    ) from create_exc
            except OSError as exc:
                raise ValueError(
                    f"{context} contains a symlink or unsafe component"
                ) from exc
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _publish_bundle(
    destination: Path,
    *,
    similarity_bytes: bytes,
    predictions_bytes: bytes,
    metadata_bytes: bytes,
) -> None:
    destination = _absolute_path(destination)
    _preflight_path(destination, "score output path")
    if destination == destination.parent or not destination.name:
        raise ValueError("score output must name a bundle directory")
    _preflight_path(destination.parent, "score output parent")
    parent_fd = _open_directory_components(
        destination.parent,
        create=True,
        context="score output parent",
    )
    bundle_fd = -1
    created_files: list[str] = []
    created_directory = False
    try:
        os.mkdir(destination.name, mode=0o700, dir_fd=parent_fd)
        created_directory = True
        bundle_fd = os.open(
            destination.name,
            os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC,
            dir_fd=parent_fd,
        )
        for name, payload in (
            ("similarity.npy", similarity_bytes),
            ("predictions.csv", predictions_bytes),
            ("metadata.json", metadata_bytes),
        ):
            created_files.append(name)
            _write_exclusive_file(bundle_fd, name, payload)
        os.fsync(bundle_fd)
        os.fsync(parent_fd)
    except BaseException:
        if created_directory:
            for name in reversed(created_files):
                try:
                    os.unlink(name, dir_fd=bundle_fd)
                except OSError:
                    pass
            if bundle_fd >= 0:
                try:
                    os.fsync(bundle_fd)
                except OSError:
                    pass
            try:
                os.rmdir(destination.name, dir_fd=parent_fd)
            except OSError:
                pass
            try:
                os.fsync(parent_fd)
            except OSError:
                pass
        raise
    finally:
        if bundle_fd >= 0:
            os.close(bundle_fd)
        os.close(parent_fd)


def _write_exclusive_file(directory_fd: int, name: str, payload: bytes) -> None:
    descriptor = os.open(
        name,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | _O_NOFOLLOW
        | _O_CLOEXEC,
        0o600,
        dir_fd=directory_fd,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _preflight_path(path: Path, context: str) -> None:
    raw = os.fspath(path)
    if not isinstance(raw, str):
        raise TypeError(f"{context} must be a text path")
    if "\x00" in raw:
        raise ValueError(f"{context} contains a NUL byte")
    lowered = raw.lower()
    if _FORMAL_TEST_RECORD_SHA256 in lowered:
        raise ValueError(
            f"{context} forbidden: contains the formal-test record digest"
        )
    for component in Path(raw).parts:
        normalized = _normalize_semantic_name(component)
        if normalized in {
            "formal_input",
            "formal_refit",
            "formal_test",
            "test_images",
            "val_confirm",
        }:
            raise ValueError(
                f"{context} forbidden: contains {component!r}"
            )
        if _SUBJECT_TEST_FILENAME_RE.fullmatch(component):
            raise ValueError(
                f"{context} forbidden: formal-test subject manifest"
            )

    absolute = _absolute_path(path)
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            return
        except OSError as exc:
            raise ValueError(f"{context} cannot be inspected safely") from exc
        if stat.S_ISLNK(mode):
            raise ValueError(f"{context} contains a symlink component")


def _open_relative_regular(
    directory_fd: int,
    name: str,
    *,
    context: str,
) -> int:
    if not name or Path(name).name != name:
        raise ValueError(f"{context} must be a single filename")
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | _O_NOFOLLOW | _O_CLOEXEC,
            dir_fd=directory_fd,
        )
    except OSError as exc:
        raise ValueError(f"{context} could not be opened securely") from exc
    value = os.fstat(descriptor)
    if not stat.S_ISREG(value.st_mode):
        os.close(descriptor)
        raise ValueError(f"{context} must be a regular file")
    return descriptor


def _read_relative_regular_bytes(
    directory_fd: int,
    name: str,
    *,
    context: str,
    limit: int,
) -> tuple[bytes, os.stat_result, str]:
    descriptor = _open_relative_regular(
        directory_fd,
        name,
        context=context,
    )
    try:
        before = os.fstat(descriptor)
        if before.st_size > limit:
            raise ValueError(f"{context} exceeds its size limit")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, limit + 1 - size),
            )
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > limit:
                raise ValueError(f"{context} exceeds its size limit")
        after = os.fstat(descriptor)
        _require_stable_stat(before, after, context)
        raw = b"".join(chunks)
        return raw, after, _sha256_bytes(raw)
    finally:
        os.close(descriptor)


def _load_relative_similarity(
    directory_fd: int,
    verified: VerifiedArtifact,
) -> np.ndarray:
    descriptor = _open_relative_regular(
        directory_fd,
        "similarity.npy",
        context="verified score payload",
    )
    try:
        before = os.fstat(descriptor)
        digest = _sha256_descriptor(descriptor)
        after = os.fstat(descriptor)
        _require_stable_stat(before, after, "verified score payload")
        _require_verified_file(
            after,
            digest,
            expected_identity=(
                verified.device,
                verified.inode,
                verified.size,
                verified.mtime_ns,
                verified.ctime_ns,
            ),
            expected_sha256=verified.payload_sha256,
            context="verified payload",
        )
        os.lseek(descriptor, 0, os.SEEK_SET)
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            try:
                loaded = np.load(handle, allow_pickle=False)
            except (OSError, TypeError, ValueError) as exc:
                raise ValueError(
                    "verified score payload is not a safe NumPy array"
                ) from exc
            if not isinstance(loaded, np.ndarray):
                raise ValueError(
                    "verified score payload must contain exactly one array"
                )
            post_load = os.fstat(handle.fileno())
            _require_verified_file(
                post_load,
                verified.payload_sha256,
                expected_identity=(
                    verified.device,
                    verified.inode,
                    verified.size,
                    verified.mtime_ns,
                    verified.ctime_ns,
                ),
                expected_sha256=verified.payload_sha256,
                context="verified payload",
            )
            post_digest = _sha256_descriptor(handle.fileno())
            final = os.fstat(handle.fileno())
            _require_stable_stat(
                post_load,
                final,
                "verified score payload",
            )
            _require_verified_file(
                final,
                post_digest,
                expected_identity=(
                    verified.device,
                    verified.inode,
                    verified.size,
                    verified.mtime_ns,
                    verified.ctime_ns,
                ),
                expected_sha256=verified.payload_sha256,
                context="verified payload",
            )
            return loaded
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _require_verified_file(
    value: os.stat_result,
    digest: str,
    *,
    expected_identity: tuple[int, int, int, int, int],
    expected_sha256: str,
    context: str,
) -> None:
    if _stat_identity(value) != expected_identity:
        raise ValueError(f"{context} file identity mismatch")
    if digest != expected_sha256:
        raise ValueError(f"{context} SHA-256 digest mismatch")


def _require_stable_stat(
    before: os.stat_result,
    after: os.stat_result,
    context: str,
) -> None:
    if _stat_identity(before) != _stat_identity(after):
        raise ValueError(f"{context} changed while it was read")


def _sha256_descriptor(descriptor: int) -> str:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            return digest.hexdigest()
        digest.update(chunk)


def _require_bundle_path_identity(
    bundle: Path,
    expected: os.stat_result,
) -> None:
    descriptor = _open_directory_components(
        bundle,
        create=False,
        context="score bundle",
    )
    try:
        if _stat_identity(os.fstat(descriptor)) != _stat_identity(expected):
            raise ValueError("score bundle directory identity changed")
    finally:
        os.close(descriptor)


def _immutable_matrix_copy(matrix: np.ndarray) -> np.ndarray:
    immutable_bytes = matrix.tobytes(order="C")
    immutable = np.ndarray(
        matrix.shape,
        dtype=matrix.dtype,
        buffer=immutable_bytes,
        order="C",
    )
    if immutable.flags.writeable:
        raise AssertionError("bytes-backed score matrix must be immutable")
    return immutable


def _parse_json_object(raw: bytes, context: str) -> dict[str, object]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{context} is not valid UTF-8") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{context} is malformed JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object")
    return value


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> object:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def _json_clone(value: object, context: str) -> object:
    try:
        return json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError(f"{context} must contain canonical JSON values") from exc


def _reject_formal_scope_markers(
    value: object,
    *,
    path: tuple[str, ...],
) -> None:
    if path == ("source_records",):
        _reject_strict_source_records(value)
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError("source_records keys must be strings")
            normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
            if (
                normalized in {"role", "scope", "split", "split_role", "subset"}
                and isinstance(child, str)
                and re.sub(r"[^a-z0-9]+", "_", child.lower()).strip("_")
                in {
                    "formal_input",
                    "formal_refit",
                    "formal_test",
                    "test",
                    "val_confirm",
                }
            ):
                raise PermissionError(
                    "source_records contain a non-val-dev scope marker"
                )
            _reject_formal_scope_markers(child, path=path + (normalized,))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_formal_scope_markers(child, path=path + (str(index),))
    elif isinstance(value, str):
        lowered = value.lower()
        if _FORMAL_TEST_RECORD_SHA256 in lowered:
            raise ValueError("source_records contain the formal-test digest")
        if any(part.lower() == "test_images" for part in re.split(r"[\\/]", value)):
            raise ValueError("source_records contain a test_images path")


def _reject_strict_source_records(
    value: object,
    *,
    semantic_path: tuple[str, ...] = (),
    test_context: bool = False,
) -> None:
    if isinstance(value, Mapping):
        normalized_items: list[tuple[str, object]] = []
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError("source_records keys must be strings")
            _reject_forbidden_source_text(key)
            normalized_items.append(
                (_normalize_semantic_name(key), child)
            )
        mapping_context = test_context or any(
            _is_test_or_formal_context(key)
            or (
                key in {"role", "scope", "split", "split_role", "subset"}
                and isinstance(child, str)
                and _normalize_semantic_name(child)
                in {
                    "formal_input",
                    "formal_refit",
                    "formal_test",
                    "test",
                    "val_confirm",
                }
            )
            for key, child in normalized_items
        )
        for normalized, child in normalized_items:
            if (
                normalized in {"role", "scope", "split", "split_role", "subset"}
                and isinstance(child, str)
                and _normalize_semantic_name(child)
                in {
                    "formal_input",
                    "formal_refit",
                    "formal_test",
                    "test",
                    "val_confirm",
                }
            ):
                raise PermissionError(
                    "source_records contain a non-val-dev scope marker"
                )
            child_context = (
                mapping_context or _is_test_or_formal_context(normalized)
            )
            if child_context and _has_sensitive_output_term(normalized):
                raise PermissionError(
                    "source_records contain test/formal score outputs"
                )
            _reject_strict_source_records(
                child,
                semantic_path=semantic_path + (normalized,),
                test_context=child_context,
            )
        return
    if isinstance(value, list):
        for child in value:
            _reject_strict_source_records(
                child,
                semantic_path=semantic_path,
                test_context=test_context,
            )
        return
    if isinstance(value, str):
        _reject_forbidden_source_text(value)
        path = semantic_path + (_normalize_semantic_name(value),)
        if (
            test_context
            or any(_is_test_or_formal_context(item) for item in path)
        ) and any(_has_sensitive_output_term(item) for item in path):
            raise PermissionError(
                "source_records contain test/formal score outputs"
            )


def _normalize_semantic_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _is_test_or_formal_context(value: str) -> bool:
    return (
        value in {"formal", "formal_input", "formal_refit", "formal_test", "test"}
        or value.startswith("formal_")
        or value.startswith("test_")
        or value.endswith("_test")
    )


def _has_sensitive_output_term(value: str) -> bool:
    return any(
        term in value
        for term in (
            "metric",
            "prediction",
            "rank",
            "score",
            "top1",
            "top_1",
            "top5",
            "top_5",
        )
    )


def _reject_forbidden_source_text(value: str) -> None:
    lowered = value.lower()
    if _FORMAL_TEST_RECORD_SHA256 in lowered:
        raise ValueError("source_records contain the formal-test digest")
    components = re.split(r"[\\/]+", value)
    if any(part.lower() == "test_images" for part in components):
        raise ValueError("source_records contain a test_images path")
    if any(
        _SUBJECT_TEST_FILENAME_RE.fullmatch(part) is not None
        for part in components
    ):
        raise ValueError(
            "source_records contain a formal-test subject manifest"
        )


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


def _require_mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a JSON object")
    return value


def _require_json_string_list(value: object, context: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) for item in value
    ):
        raise ValueError(f"{context} must be a JSON array of strings")
    return value


def _require_sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _require_nonnegative_int(value: object, context: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{context} must be a non-negative integer")
    return value


def _require_positive_int(value: object, context: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{context} must be a positive integer")
    return value


def _deep_freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(child) for key, child in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze(child) for child in value)
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(os.path.normpath(os.fspath(path))))


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


__all__ = [
    "RetrievalMetrics",
    "RetrievalPrediction",
    "SCORE_PAYLOAD_TYPE",
    "SCORE_SCOPE",
    "ScoreArtifact",
    "independent_retrieval_metrics",
]
