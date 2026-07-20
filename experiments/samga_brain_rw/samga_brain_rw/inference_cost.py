"""Hash-closed raw Stage 1 cost records and static complexity keys.

This module accepts caller-supplied callables, clocks, synchronization hooks,
and self-attested identity references.  It can check schemas, hashes, raw
arithmetic, and protocol conformance; it cannot prove that an A40, model,
SLURM job, or referenced input actually produced a measurement.  It issues no
trusted capability and its records are not accepted composition evidence.

The record keeps caller-observed branch durations separate from the
preregistered, unitless fusion operator-complexity key.  Static complexity is
never presented as latency, and the API accepts no validation observations.
"""

from __future__ import annotations

import copy
import json
import os
import re
import statistics
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter_ns
from types import MappingProxyType

from .fusion import enumerate_stage1_configs
from .hashing import sha256_json
from .statistics import write_development_json_exclusive


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
_DENIED_SCOPE_RE = re.compile(r"formal|val(?:[-_./\\]*)confirm", re.IGNORECASE)
_CANONICAL_FORMAL_TEST_RECORD_SHA256 = (
    "02d7e33b3fe8e5a571f8db232ca5fa86abb0c16981876ec84feae7ba64636f1a"
)
_COST_PROTOCOL_CONSTRUCTION_MARKER = object()
_EXPECTED_FUSION_CONFIG_SHA256 = (
    "27cd33027c2fa322121f0c42732d2ecbe62f2544d6497652ae40f90b7dd8dc78"
)
_BRANCH_IDS = ("internvit", "brainrw")
_EXPECTED_BENCHMARK = {
    "same_process": True,
    "interleave_order": "alternate_each_round",
    "warmup_runs": 10,
    "measured_runs": 50,
    "timer": "perf_counter_ns",
    "cuda_synchronize": "before_and_after",
    "statistic": "median",
    "dispersion": ["mad", "iqr"],
    "raw_unit": "nanoseconds",
    "reported_unit": "milliseconds_per_query",
}
_EXPECTED_QUALITY_GATE = {
    "mad_over_median_max": 0.05,
    "comparison_when_gate_fails": "abort_without_selection",
}
_EXPECTED_SYNTHETIC_WORKLOAD = {
    "schema_version": 1,
    "input_kind": "synthetic_preprocessed",
    "generator": "torch.Generator.manual_seed",
    "seed": 20260720,
    "query_count": 200,
    "gallery_count": 200,
    "labels_present": False,
    "metrics_computed": False,
    "timing_boundary": "preloaded_branch_callable",
    "includes": [
        "image_encoder",
        "eeg_task_encoder",
        "projection_and_normalization",
        "200x200_similarity",
    ],
    "excludes": [
        "filesystem_io",
        "model_loading",
        "image_preprocessing",
        "metric_computation",
    ],
    "branches": {
        "internvit": {
            "image_shape": [200, 3, 448, 448],
            "image_dtype": "bfloat16",
            "eeg_shape": [200, 17, 250],
            "eeg_dtype": "float32",
        },
        "brainrw": {
            "image_shape": [200, 3, 224, 224],
            "image_dtype": "bfloat16",
            "eeg_shape": [200, 17, 250],
            "eeg_dtype": "bfloat16",
        },
    },
}
_EXPECTED_COMPLEXITY_KEYS = {
    "temperature_convex": (0, 0, 5),
    "zscore_convex": (0, 6, 9),
    "rrf": (2, 0, 5),
}
_PROTOCOL_KEYS = frozenset(
    {
        "benchmark",
        "branch_ids",
        "config_id",
        "config_type",
        "fusion_config_sha256",
        "fusion_operator_complexity",
        "operator_cost_kind",
        "quality_gate",
        "schema_version",
        "scope",
        "synthetic_workload",
        "synthetic_workload_sha256",
    }
)
_RUNTIME_REFERENCE_KEYS = frozenset(
    {
        "declared_runtime_contract",
        "declared_runtime_contract_sha256",
        "declared_runtime_observation",
        "declared_runtime_observation_sha256",
        "declared_semantic_environment",
        "declared_semantic_environment_sha256",
    }
)
_RUNTIME_CONTRACT_KEYS = frozenset(
    {
        "accelerator",
        "branch_device_binding",
        "device_index",
        "device_type",
        "process_mode",
        "schema_version",
    }
)
_RUNTIME_EVIDENCE_KEYS = frozenset(
    {
        "accelerator_name",
        "bf16_supported",
        "cuda_available",
        "cuda_capability",
        "cuda_device_count",
        "cuda_device_index",
        "cuda_version",
        "schema_version",
        "torch_version",
        "total_memory_bytes",
    }
)
_MODEL_REFERENCE_KEYS = frozenset({"branches", "schema_version"})
_MODEL_BRANCH_KEYS = frozenset(
    {
        "checkpoint_sha256",
        "model_code_sha256",
        "model_config_sha256",
        "model_id",
        "parameter_dtype",
        "weights_sha256",
    }
)
_INPUT_REFERENCE_KEYS = frozenset({"cells", "provenance_scope", "schema_version"})
_INPUT_CELL_KEYS = frozenset(
    {
        "alignment_sha256",
        "branches",
        "cell_id",
        "gallery_ids_sha256",
        "query_ids_sha256",
        "seed",
        "subject",
    }
)
_INPUT_BRANCH_KEYS = frozenset(
    {
        "checkpoint_sha256",
        "input_bundle_sha256",
        "resolved_config_sha256",
        "run_key",
        "run_manifest_sha256",
        "score_envelope_sha256",
        "score_payload_sha256",
        "source_payload_sha256",
    }
)
_JOB_CLAIM_REFERENCE_KEYS = frozenset(
    {
        "attempt_id",
        "attempt_index",
        "claim_id",
        "schema_version",
        "slurm_job_id",
        "slurm_partition",
        "unverified_claim_sha256",
        "unverified_previous_record_sha256",
    }
)
_BENCHMARK_KEYS = frozenset(
    {
        "branches",
        "interleave_order",
        "measured_runs",
        "measurement_kind",
        "measurement_order",
        "measurement_order_sha256",
        "query_count_per_call",
        "raw_unit",
        "reported_unit",
        "warmup_runs",
    }
)
_BENCHMARK_BRANCH_KEYS = frozenset(
    {
        "elapsed_nanoseconds",
        "iqr_elapsed_nanoseconds",
        "mad_elapsed_nanoseconds",
        "mad_over_median",
        "median_elapsed_nanoseconds",
        "milliseconds_per_query",
        "q1_elapsed_nanoseconds",
        "q3_elapsed_nanoseconds",
    }
)
_RAW_RECORD_KEYS = frozenset(
    {
        "allowed_use",
        "artifact_type",
        "branch_benchmark",
        "branch_benchmark_sha256",
        "composition_eligible",
        "confirmation_eligible",
        "fusion_operator_complexity",
        "fusion_operator_complexity_sha256",
        "measurement_protocol",
        "measurement_protocol_sha256",
        "record_kind",
        "reference_trust",
        "schema_version",
        "self_attested_input_reference",
        "self_attested_input_reference_sha256",
        "self_attested_job_claim_reference",
        "self_attested_job_claim_reference_sha256",
        "self_attested_model_reference",
        "self_attested_model_reference_sha256",
        "self_attested_runtime_reference",
        "self_attested_runtime_reference_sha256",
        "scope",
        "trusted_capability_issued",
    }
)
_PILOT_COORDINATES = (
    (1, 42),
    (1, 43),
    (5, 42),
    (5, 43),
    (8, 42),
    (8, 43),
)


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{context} must be a string-keyed object")
    return {str(key): copy.deepcopy(item) for key, item in value.items()}


def _exact_keys(
    value: Mapping[str, object],
    expected: frozenset[str],
    context: str,
) -> None:
    if set(value) != expected:
        raise ValueError(f"{context} schema mismatch")


def _sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be lowercase SHA-256")
    return value


@dataclass(frozen=True, init=False)
class CostProtocol:
    """Strict semantic view of ``stage1_cost_v1.json``."""

    config_id: str
    scope: str
    branch_ids: tuple[str, ...]
    warmup_runs: int
    measured_runs: int
    statistic: str
    dispersion: tuple[str, ...]
    mad_ratio_max: float
    same_process: bool
    interleave_order: str
    cuda_synchronize: str
    timer: str
    synthetic_workload_sha256: str
    operator_cost_kind: str
    fusion_config_sha256: str
    _operator_complexity: Mapping[str, tuple[int, int, int]] = field(repr=False)
    _operator_families: Mapping[str, str] = field(repr=False)
    _synthetic_workload: Mapping[str, object] = field(repr=False)
    _payload: Mapping[str, object] = field(repr=False)
    _construction_marker: object = field(repr=False, compare=False)

    def __new__(cls, *_: object, **__: object) -> "CostProtocol":
        raise TypeError(
            "CostProtocol requires controlled load_cost_protocol construction"
        )

    @classmethod
    def from_payload(cls, value: object) -> "CostProtocol":
        payload = _mapping(value, "cost protocol")
        _exact_keys(payload, _PROTOCOL_KEYS, "cost protocol")
        if (
            payload["schema_version"] != 1
            or payload["config_type"] != "stage1_cost"
            or payload["config_id"] != "stage1_cost_v1"
            or payload["scope"] != "stage1-cost"
        ):
            raise ValueError("cost protocol identity/scope mismatch")
        branch_ids = payload["branch_ids"]
        if branch_ids != list(_BRANCH_IDS):
            raise ValueError("cost protocol branch_ids mismatch")
        benchmark = _mapping(payload["benchmark"], "cost benchmark")
        if benchmark != _EXPECTED_BENCHMARK:
            differing = sorted(
                key
                for key in set(benchmark) | set(_EXPECTED_BENCHMARK)
                if benchmark.get(key) != _EXPECTED_BENCHMARK.get(key)
            )
            raise ValueError(f"cost benchmark semantic mismatch: {differing}")
        quality = _mapping(payload["quality_gate"], "cost quality gate")
        if quality != _EXPECTED_QUALITY_GATE:
            raise ValueError("cost MAD quality gate mismatch")
        workload = _mapping(
            payload["synthetic_workload"],
            "synthetic workload",
        )
        if workload != _EXPECTED_SYNTHETIC_WORKLOAD:
            raise ValueError("synthetic workload semantics mismatch")
        workload_sha256 = _sha256(
            payload["synthetic_workload_sha256"],
            "synthetic workload SHA-256",
        )
        if workload_sha256 != sha256_json(workload):
            raise ValueError("synthetic workload SHA-256 mismatch")
        if payload["operator_cost_kind"] != "deterministic_operator_complexity_key":
            raise ValueError("operator complexity kind mismatch")
        fusion_sha256 = _sha256(
            payload["fusion_config_sha256"],
            "fusion config SHA-256",
        )
        if fusion_sha256 != _EXPECTED_FUSION_CONFIG_SHA256:
            raise ValueError("fusion config SHA-256 mismatch")

        raw_entries = payload["fusion_operator_complexity"]
        if not isinstance(raw_entries, list) or len(raw_entries) != 47:
            raise ValueError("fusion operator complexity requires exactly 47 configs")
        expected_configs = {
            config.config_id: config.family for config in enumerate_stage1_configs()
        }
        complexities: dict[str, tuple[int, int, int]] = {}
        families: dict[str, str] = {}
        for index, raw_entry in enumerate(raw_entries):
            entry = _mapping(
                raw_entry,
                f"fusion operator complexity[{index}]",
            )
            _exact_keys(
                entry,
                frozenset(
                    {
                        "config_id",
                        "family",
                        "operator_complexity_key",
                    }
                ),
                f"fusion operator complexity[{index}]",
            )
            config_id = entry["config_id"]
            family = entry["family"]
            if (
                not isinstance(config_id, str)
                or config_id not in expected_configs
                or config_id in complexities
            ):
                raise ValueError("fusion complexity has duplicate/unknown config ID")
            if family != expected_configs[config_id]:
                raise ValueError("fusion complexity family mismatch")
            raw_key = entry["operator_complexity_key"]
            if (
                not isinstance(raw_key, list)
                or len(raw_key) != 3
                or any(type(item) is not int or item < 0 for item in raw_key)
            ):
                raise ValueError("operator complexity key must be 3 integers")
            key = tuple(raw_key)
            if key != _EXPECTED_COMPLEXITY_KEYS[family]:
                raise ValueError("operator complexity family key mismatch")
            complexities[config_id] = key
            families[config_id] = family
        if set(complexities) != set(expected_configs):
            raise ValueError("fusion complexity does not cover exact 47 configs")

        instance = object.__new__(cls)
        fields = {
            "config_id": "stage1_cost_v1",
            "scope": "stage1-cost",
            "branch_ids": _BRANCH_IDS,
            "warmup_runs": 10,
            "measured_runs": 50,
            "statistic": "median",
            "dispersion": ("mad", "iqr"),
            "mad_ratio_max": 0.05,
            "same_process": True,
            "interleave_order": "alternate_each_round",
            "cuda_synchronize": "before_and_after",
            "timer": "perf_counter_ns",
            "synthetic_workload_sha256": workload_sha256,
            "operator_cost_kind": "deterministic_operator_complexity_key",
            "fusion_config_sha256": fusion_sha256,
            "_operator_complexity": _deep_freeze(complexities),
            "_operator_families": _deep_freeze(families),
            "_synthetic_workload": _deep_freeze(workload),
            "_payload": _deep_freeze(payload),
            "_construction_marker": _COST_PROTOCOL_CONSTRUCTION_MARKER,
        }
        for field_name, field_value in fields.items():
            object.__setattr__(instance, field_name, field_value)
        return instance

    @property
    def synthetic_workload(self) -> dict[str, object]:
        workload = _deep_thaw(self._synthetic_workload)
        if not isinstance(workload, dict):
            raise AssertionError("synthetic workload must be an object")
        return workload

    @property
    def operator_complexity(self) -> dict[str, tuple[int, int, int]]:
        return dict(self._operator_complexity)

    @property
    def operator_families(self) -> dict[str, str]:
        return dict(self._operator_families)

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_payload())

    def to_payload(self) -> dict[str, object]:
        payload = _deep_thaw(self._payload)
        if not isinstance(payload, dict):
            raise AssertionError("cost protocol payload must be an object")
        return payload


def _deep_freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class RawStage1CostRecord:
    """Immutable raw record; never a trusted execution capability."""

    _payload: Mapping[str, object] = field(repr=False)
    record_sha256: str

    def __post_init__(self) -> None:
        payload = _mapping(
            _deep_thaw(self._payload),
            "raw Stage 1 cost record",
        )
        _validate_raw_record_payload(payload)
        record_sha256 = _sha256(
            self.record_sha256,
            "raw cost record SHA-256",
        )
        if record_sha256 != sha256_json(payload):
            raise ValueError("raw cost record SHA-256 mismatch")
        object.__setattr__(
            self,
            "_payload",
            _deep_freeze(payload),
        )

    @classmethod
    def from_document(
        cls,
        value: object,
    ) -> "RawStage1CostRecord":
        document = _mapping(value, "raw Stage 1 cost record document")
        expected = _RAW_RECORD_KEYS | {"record_sha256"}
        _exact_keys(document, frozenset(expected), "raw cost record document")
        record_sha256 = document.pop("record_sha256")
        return cls(
            _payload=document,
            record_sha256=_sha256(
                record_sha256,
                "raw cost record SHA-256",
            ),
        )

    def to_payload(self) -> dict[str, object]:
        payload = _deep_thaw(self._payload)
        if not isinstance(payload, dict):
            raise AssertionError("raw cost record payload must be an object")
        return payload

    def to_document(self) -> dict[str, object]:
        return {
            **self.to_payload(),
            "record_sha256": self.record_sha256,
        }


def _alternating_order(
    branch_ids: tuple[str, ...],
    round_index: int,
) -> tuple[str, ...]:
    if round_index % 2 == 0:
        return branch_ids
    return tuple(reversed(branch_ids))


def _summarize_elapsed_nanoseconds(
    raw_ns: list[int],
    query_count: int,
) -> dict[str, object]:
    median_ns = statistics.median(raw_ns)
    absolute_deviations = [abs(elapsed_ns - median_ns) for elapsed_ns in raw_ns]
    mad_ns = statistics.median(absolute_deviations)
    q1_ns, _, q3_ns = statistics.quantiles(
        raw_ns,
        n=4,
        method="inclusive",
    )
    iqr_ns = q3_ns - q1_ns
    mad_ratio = mad_ns / median_ns
    return {
        "elapsed_nanoseconds": list(raw_ns),
        "median_elapsed_nanoseconds": median_ns,
        "mad_elapsed_nanoseconds": mad_ns,
        "q1_elapsed_nanoseconds": q1_ns,
        "q3_elapsed_nanoseconds": q3_ns,
        "iqr_elapsed_nanoseconds": iqr_ns,
        "mad_over_median": mad_ratio,
        "milliseconds_per_query": median_ns / query_count / 1_000_000,
    }


def benchmark_alternating_branches(
    protocol: CostProtocol,
    branch_callables: Mapping[str, Callable[[], object]],
    *,
    clock_ns: Callable[[], int] = perf_counter_ns,
    synchronize: Callable[[], object],
) -> dict[str, object]:
    """Benchmark two preloaded branches in one alternating process.

    The callables close over their fixed synthetic preprocessed inputs. No
    metrics, labels, predictions, model loading, or filesystem I/O enter this
    API or its timed boundary.
    """

    if not isinstance(protocol, CostProtocol):
        raise TypeError("protocol must be a CostProtocol")
    if not isinstance(branch_callables, Mapping) or set(branch_callables) != set(
        protocol.branch_ids
    ):
        raise ValueError("benchmark requires the exact branch callable set")
    callables = {
        branch_id: branch_callables[branch_id] for branch_id in protocol.branch_ids
    }
    if any(not callable(workload) for workload in callables.values()):
        raise TypeError("every branch workload must be callable")
    if not callable(clock_ns) or not callable(synchronize):
        raise TypeError("clock_ns and synchronize must be callable")

    for round_index in range(protocol.warmup_runs):
        for branch_id in _alternating_order(
            protocol.branch_ids,
            round_index,
        ):
            synchronize()
            callables[branch_id]()
            synchronize()

    samples: dict[str, list[int]] = {branch_id: [] for branch_id in protocol.branch_ids}
    measurement_order: list[str] = []
    for measured_index in range(protocol.measured_runs):
        round_index = protocol.warmup_runs + measured_index
        for branch_id in _alternating_order(
            protocol.branch_ids,
            round_index,
        ):
            synchronize()
            started_ns = clock_ns()
            callables[branch_id]()
            synchronize()
            finished_ns = clock_ns()
            if (
                type(started_ns) is not int
                or type(finished_ns) is not int
                or finished_ns <= started_ns
            ):
                raise ValueError(
                    "clock_ns must produce positive integer nanosecond intervals"
                )
            samples[branch_id].append(finished_ns - started_ns)
            measurement_order.append(branch_id)

    query_count = protocol.synthetic_workload["query_count"]
    if type(query_count) is not int or query_count <= 0:
        raise ValueError("synthetic workload query_count must be positive")
    summaries = {
        branch_id: _summarize_elapsed_nanoseconds(
            samples[branch_id],
            query_count,
        )
        for branch_id in protocol.branch_ids
    }
    failed_branches = [
        branch_id
        for branch_id in protocol.branch_ids
        if summaries[branch_id]["mad_over_median"] > protocol.mad_ratio_max
    ]
    if failed_branches:
        raise ValueError(
            "MAD/median quality gate failed for branches: " + ", ".join(failed_branches)
        )

    return {
        "measurement_kind": "caller_supplied_branch_callable_duration",
        "raw_unit": "nanoseconds",
        "reported_unit": "milliseconds_per_query",
        "query_count_per_call": query_count,
        "warmup_runs": protocol.warmup_runs,
        "measured_runs": protocol.measured_runs,
        "interleave_order": protocol.interleave_order,
        "measurement_order": measurement_order,
        "measurement_order_sha256": sha256_json(measurement_order),
        "branches": summaries,
    }


def _safe_id(value: object, context: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a safe nonempty identity")
    return value


def _reject_denied_scope_text(value: str, context: str) -> None:
    lowered = value.lower()
    if (
        _CANONICAL_FORMAL_TEST_RECORD_SHA256 in lowered
        or _DENIED_SCOPE_RE.search(lowered) is not None
    ):
        raise ValueError(f"{context} contains a forbidden development scope")


def _reject_forbidden_scope(value: object, context: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str):
                _reject_denied_scope_text(key, f"{context} key")
            _reject_forbidden_scope(item, f"{context}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_forbidden_scope(item, f"{context}[{index}]")
    elif isinstance(value, str):
        _reject_denied_scope_text(value, context)


def _reject_observation_fields(value: object, context: str) -> None:
    forbidden = {
        "label",
        "labels",
        "metric",
        "metrics",
        "observation",
        "observations",
        "prediction",
        "predictions",
        "score_matrix",
        "scores",
        "top1",
        "top5",
    }
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str) and key.lower() in forbidden:
                raise ValueError(
                    f"{context} must not contain metrics, labels, or observations"
                )
            _reject_observation_fields(item, f"{context}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_observation_fields(item, f"{context}[{index}]")


def _validate_self_attested_model_reference(value: object) -> dict[str, object]:
    binding = _mapping(value, "self-attested model reference")
    _reject_forbidden_scope(binding, "self-attested model reference")
    _exact_keys(binding, _MODEL_REFERENCE_KEYS, "self-attested model reference")
    if binding["schema_version"] != 1:
        raise ValueError("self-attested model reference schema_version mismatch")
    branches = _mapping(binding["branches"], "self-attested model branches")
    if set(branches) != set(_BRANCH_IDS):
        raise ValueError("self-attested model reference requires exact branches")
    for branch_id in _BRANCH_IDS:
        branch = _mapping(
            branches[branch_id],
            f"{branch_id} cost model binding",
        )
        _exact_keys(
            branch,
            _MODEL_BRANCH_KEYS,
            f"{branch_id} cost model binding",
        )
        _safe_id(branch["model_id"], f"{branch_id} model_id")
        if branch["parameter_dtype"] != "bfloat16":
            raise ValueError(f"{branch_id} cost model parameter_dtype must be bfloat16")
        for field_name in (
            "checkpoint_sha256",
            "model_code_sha256",
            "model_config_sha256",
            "weights_sha256",
        ):
            _sha256(
                branch[field_name],
                f"{branch_id} {field_name}",
            )
        branches[branch_id] = branch
    binding["branches"] = branches
    return binding


def _validate_self_attested_job_claim_reference(
    value: object,
) -> dict[str, object]:
    claim = _mapping(value, "self-attested job claim reference")
    _reject_forbidden_scope(claim, "self-attested job claim reference")
    _exact_keys(
        claim,
        _JOB_CLAIM_REFERENCE_KEYS,
        "self-attested job claim reference",
    )
    if claim["schema_version"] != 1:
        raise ValueError("cost job claim schema_version mismatch")
    _safe_id(claim["slurm_job_id"], "SLURM job ID")
    if claim["slurm_partition"] != "debug":
        raise ValueError("self-attested job reference must declare the debug partition")
    _safe_id(claim["claim_id"], "cost claim_id")
    _sha256(
        claim["unverified_claim_sha256"],
        "unverified claim reference SHA-256",
    )
    attempt_index = claim["attempt_index"]
    if type(attempt_index) is not int or attempt_index < 0:
        raise ValueError("cost attempt_index must be a nonnegative integer")
    expected_attempt_id = f"attempt-{attempt_index:04d}"
    if claim["attempt_id"] != expected_attempt_id:
        raise ValueError("cost attempt_id/index mismatch")
    previous = claim["unverified_previous_record_sha256"]
    if previous is not None:
        _sha256(previous, "unverified previous record reference SHA-256")
    return claim


def _validate_branch_benchmark(
    protocol: CostProtocol,
    value: object,
) -> dict[str, object]:
    benchmark = _mapping(value, "branch benchmark")
    _exact_keys(benchmark, _BENCHMARK_KEYS, "branch benchmark")
    expected_literals = {
        "interleave_order": protocol.interleave_order,
        "measured_runs": protocol.measured_runs,
        "measurement_kind": "caller_supplied_branch_callable_duration",
        "query_count_per_call": protocol.synthetic_workload["query_count"],
        "raw_unit": "nanoseconds",
        "reported_unit": "milliseconds_per_query",
        "warmup_runs": protocol.warmup_runs,
    }
    for field_name, expected in expected_literals.items():
        if benchmark[field_name] != expected:
            raise ValueError(f"branch benchmark {field_name} protocol mismatch")
    measurement_order = benchmark["measurement_order"]
    expected_order = [
        branch_id
        for measured_index in range(protocol.measured_runs)
        for branch_id in _alternating_order(
            protocol.branch_ids,
            protocol.warmup_runs + measured_index,
        )
    ]
    if measurement_order != expected_order:
        raise ValueError("branch benchmark measurement order mismatch")
    if benchmark["measurement_order_sha256"] != sha256_json(measurement_order):
        raise ValueError("branch benchmark measurement order SHA-256 mismatch")

    branches = _mapping(benchmark["branches"], "branch measurements")
    if set(branches) != set(protocol.branch_ids):
        raise ValueError("benchmark requires the exact branch measurements")
    query_count = int(expected_literals["query_count_per_call"])
    for branch_id in protocol.branch_ids:
        branch = _mapping(
            branches[branch_id],
            f"{branch_id} benchmark",
        )
        _exact_keys(
            branch,
            _BENCHMARK_BRANCH_KEYS,
            f"{branch_id} benchmark",
        )
        raw_ns = branch["elapsed_nanoseconds"]
        if (
            not isinstance(raw_ns, list)
            or len(raw_ns) != protocol.measured_runs
            or any(type(item) is not int or item <= 0 for item in raw_ns)
        ):
            raise ValueError(f"{branch_id} benchmark requires exact positive raw ns")
        expected_summary = _summarize_elapsed_nanoseconds(
            raw_ns,
            query_count,
        )
        if branch != expected_summary:
            raise ValueError(f"{branch_id} benchmark statistics are not recomputable")
        if expected_summary["mad_over_median"] > protocol.mad_ratio_max:
            raise ValueError(f"{branch_id} benchmark fails MAD quality gate")
        branches[branch_id] = branch
    benchmark["branches"] = branches
    return benchmark


def _validate_raw_record_payload(value: object) -> dict[str, object]:
    payload = _mapping(value, "raw Stage 1 cost record")
    _exact_keys(payload, _RAW_RECORD_KEYS, "raw Stage 1 cost record")
    if (
        payload["artifact_type"] != "samga_brain_rw.raw_stage1_cost_record"
        or payload["schema_version"] != 1
        or payload["scope"] != "stage1-cost"
        or payload["allowed_use"] != "unit_test_or_development_raw_cost_record"
        or payload["record_kind"] != "hash_closed_raw_measurement_record"
        or payload["reference_trust"] != "self_attested_unverified"
        or payload["trusted_capability_issued"] is not False
        or payload["composition_eligible"] is not False
        or payload["confirmation_eligible"] is not False
    ):
        raise ValueError("raw Stage 1 cost record identity/trust mismatch")

    protocol_payload = _mapping(
        payload["measurement_protocol"],
        "embedded cost protocol",
    )
    protocol = CostProtocol.from_payload(protocol_payload)
    if payload["measurement_protocol_sha256"] != protocol.sha256:
        raise ValueError("measurement protocol SHA-256 mismatch")

    runtime = validate_self_attested_runtime_reference(
        payload["self_attested_runtime_reference"]
    )
    if payload["self_attested_runtime_reference_sha256"] != sha256_json(runtime):
        raise ValueError("self-attested runtime reference SHA-256 mismatch")
    models = _validate_self_attested_model_reference(
        payload["self_attested_model_reference"]
    )
    if payload["self_attested_model_reference_sha256"] != sha256_json(models):
        raise ValueError("self-attested model reference SHA-256 mismatch")
    job_claim = _validate_self_attested_job_claim_reference(
        payload["self_attested_job_claim_reference"]
    )
    if payload["self_attested_job_claim_reference_sha256"] != sha256_json(job_claim):
        raise ValueError("self-attested job claim reference SHA-256 mismatch")
    inputs = validate_self_attested_input_reference(
        payload["self_attested_input_reference"]
    )
    if payload["self_attested_input_reference_sha256"] != sha256_json(inputs):
        raise ValueError("self-attested input reference SHA-256 mismatch")
    benchmark = _validate_branch_benchmark(
        protocol,
        payload["branch_benchmark"],
    )
    if payload["branch_benchmark_sha256"] != sha256_json(benchmark):
        raise ValueError("branch benchmark SHA-256 mismatch")

    static_entries = protocol.to_payload()["fusion_operator_complexity"]
    if payload["fusion_operator_complexity"] != static_entries:
        raise ValueError("fusion operator complexity protocol mismatch")
    if payload["fusion_operator_complexity_sha256"] != sha256_json(static_entries):
        raise ValueError("fusion operator complexity SHA-256 mismatch")
    return payload


def build_raw_stage1_cost_record(
    protocol: CostProtocol,
    branch_benchmark: object,
    *,
    runtime_reference: object,
    model_reference: object,
    job_claim_reference: object,
    input_reference: object,
) -> RawStage1CostRecord:
    """Build a raw hash-closed record from untrusted caller observations.

    Hash closure does not verify that any declared runtime, model, job, claim,
    or input produced the supplied callable durations.
    """

    if not isinstance(protocol, CostProtocol):
        raise TypeError("protocol must be a CostProtocol")
    runtime = validate_self_attested_runtime_reference(runtime_reference)
    models = _validate_self_attested_model_reference(model_reference)
    claim = _validate_self_attested_job_claim_reference(job_claim_reference)
    inputs = validate_self_attested_input_reference(input_reference)
    benchmark = _validate_branch_benchmark(protocol, branch_benchmark)
    static_entries = protocol.to_payload()["fusion_operator_complexity"]
    payload = {
        "allowed_use": "unit_test_or_development_raw_cost_record",
        "artifact_type": "samga_brain_rw.raw_stage1_cost_record",
        "branch_benchmark": benchmark,
        "branch_benchmark_sha256": sha256_json(benchmark),
        "composition_eligible": False,
        "confirmation_eligible": False,
        "fusion_operator_complexity": static_entries,
        "fusion_operator_complexity_sha256": sha256_json(static_entries),
        "measurement_protocol": protocol.to_payload(),
        "measurement_protocol_sha256": protocol.sha256,
        "record_kind": "hash_closed_raw_measurement_record",
        "reference_trust": "self_attested_unverified",
        "schema_version": 1,
        "self_attested_input_reference": inputs,
        "self_attested_input_reference_sha256": sha256_json(inputs),
        "self_attested_job_claim_reference": claim,
        "self_attested_job_claim_reference_sha256": sha256_json(claim),
        "self_attested_model_reference": models,
        "self_attested_model_reference_sha256": sha256_json(models),
        "self_attested_runtime_reference": runtime,
        "self_attested_runtime_reference_sha256": sha256_json(runtime),
        "scope": "stage1-cost",
        "trusted_capability_issued": False,
    }
    return RawStage1CostRecord(
        _payload=payload,
        record_sha256=sha256_json(payload),
    )


def load_cost_protocol(path: Path) -> CostProtocol:
    """Load a duplicate-key-safe JSON protocol and validate every semantic."""

    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(
                handle,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonfinite,
            )
    except json.JSONDecodeError as exc:
        raise ValueError("cost protocol is invalid JSON") from exc
    return CostProtocol.from_payload(value)


def operator_complexity_key(
    protocol: CostProtocol,
    config_id: str,
) -> tuple[int, int, int]:
    """Return the preregistered unitless key for one exact fusion config."""

    if (
        not isinstance(protocol, CostProtocol)
        or getattr(protocol, "_construction_marker", None)
        is not _COST_PROTOCOL_CONSTRUCTION_MARKER
    ):
        raise TypeError("protocol requires controlled load_cost_protocol construction")
    if not isinstance(config_id, str) or config_id not in protocol._operator_complexity:
        raise ValueError("unknown Stage 1 fusion config ID")
    return protocol._operator_complexity[config_id]


def publish_raw_cost_record_exclusive(
    path: Path,
    record: RawStage1CostRecord,
) -> Path:
    """Safely publish one attempt-bound development raw record."""

    if not isinstance(record, RawStage1CostRecord):
        raise TypeError("record must be RawStage1CostRecord")
    raw_path = os.fspath(path)
    if not isinstance(raw_path, str):
        raise TypeError("raw cost record output path must be text")
    _reject_denied_scope_text(raw_path, "raw cost record output path")
    output = Path(path)
    absolute = Path(os.path.abspath(os.path.normpath(raw_path)))
    _reject_denied_scope_text(
        os.fspath(absolute),
        "normalized raw cost record output path",
    )
    payload = record.to_payload()
    claim_reference = _mapping(
        payload["self_attested_job_claim_reference"],
        "self-attested job claim reference",
    )
    attempt_id = _safe_id(
        claim_reference["attempt_id"],
        "self-attested attempt_id",
    )
    expected_name = f"stage1-cost-{attempt_id}.json"
    if absolute.name != expected_name:
        raise ValueError(
            "raw record filename must bind the exact self-attested attempt_id"
        )
    write_development_json_exclusive(absolute, record.to_document())
    return output


def validate_self_attested_input_reference(
    value: object,
) -> dict[str, object]:
    """Check one self-attested, identity-only six-cell reference."""

    _reject_observation_fields(value, "self-attested input reference")
    binding = _mapping(value, "self-attested input reference")
    _reject_forbidden_scope(binding, "self-attested input reference")
    _exact_keys(binding, _INPUT_REFERENCE_KEYS, "cost input reference")
    if (
        binding["schema_version"] != 1
        or binding["provenance_scope"] != "val-dev-identities-only"
    ):
        raise ValueError("self-attested input reference identity/scope mismatch")
    cells = binding["cells"]
    if not isinstance(cells, list) or len(cells) != 6:
        raise ValueError("cost input binding requires the exact six-cell grid")
    normalized_cells: list[dict[str, object]] = []
    coordinates: list[tuple[int, int]] = []
    for index, raw_cell in enumerate(cells):
        cell = _mapping(raw_cell, f"cost input cell[{index}]")
        _exact_keys(cell, _INPUT_CELL_KEYS, f"cost input cell[{index}]")
        subject = cell["subject"]
        seed = cell["seed"]
        if type(subject) is not int or type(seed) is not int:
            raise ValueError("cost input cell subject/seed must be integers")
        coordinate = (subject, seed)
        coordinates.append(coordinate)
        if cell["cell_id"] != f"{subject:02d}/{seed}":
            raise ValueError("cost input cell_id/coordinate mismatch")
        for field_name in (
            "alignment_sha256",
            "gallery_ids_sha256",
            "query_ids_sha256",
        ):
            _sha256(cell[field_name], f"cost cell {field_name}")
        branches = _mapping(
            cell["branches"],
            f"cost input cell[{index}] branches",
        )
        if set(branches) != set(_BRANCH_IDS):
            raise ValueError("cost input cell requires exact branch provenance")
        for branch_id in _BRANCH_IDS:
            branch = _mapping(
                branches[branch_id],
                f"cost input cell[{index}] {branch_id}",
            )
            _exact_keys(
                branch,
                _INPUT_BRANCH_KEYS,
                f"cost input cell[{index}] {branch_id}",
            )
            _safe_id(
                branch["run_key"],
                f"cost input cell[{index}] {branch_id} run_key",
            )
            for field_name in _INPUT_BRANCH_KEYS - {"run_key"}:
                _sha256(
                    branch[field_name],
                    (f"cost input cell[{index}] {branch_id} {field_name}"),
                )
            branches[branch_id] = branch
        cell["branches"] = branches
        normalized_cells.append(cell)
    if tuple(coordinates) != _PILOT_COORDINATES:
        raise ValueError(
            "cost input binding must use ordered six-cell pilot coordinates"
        )
    binding["cells"] = normalized_cells
    return binding


def validate_self_attested_runtime_reference(
    value: object,
) -> dict[str, object]:
    """Check a caller's hash-closed A40 declaration without probing it."""

    reference = _mapping(value, "self-attested runtime reference")
    _reject_forbidden_scope(reference, "self-attested runtime reference")
    _exact_keys(
        reference,
        _RUNTIME_REFERENCE_KEYS,
        "self-attested runtime reference",
    )
    environment = _mapping(
        reference["declared_semantic_environment"],
        "declared semantic environment",
    )
    for field_name in (
        "cuda_version",
        "python_version",
        "torch_version",
    ):
        if (
            not isinstance(environment.get(field_name), str)
            or not environment[field_name]
        ):
            raise ValueError(f"cost semantic environment requires {field_name}")
    environment_sha256 = _sha256(
        reference["declared_semantic_environment_sha256"],
        "declared semantic environment SHA-256",
    )
    if environment_sha256 != sha256_json(environment):
        raise ValueError("semantic environment SHA-256 mismatch")

    contract = _mapping(
        reference["declared_runtime_contract"],
        "declared runtime contract",
    )
    _exact_keys(contract, _RUNTIME_CONTRACT_KEYS, "cost runtime contract")
    if (
        contract["schema_version"] != 1
        or contract["accelerator"] != "NVIDIA A40"
        or contract["device_type"] != "cuda"
        or contract["device_index"] != 0
        or contract["process_mode"] != "single_process"
        or contract["branch_device_binding"] != "same_cuda_device"
    ):
        raise ValueError("cost runtime contract requires the same A40/process/device")
    contract_sha256 = _sha256(
        reference["declared_runtime_contract_sha256"],
        "declared runtime contract SHA-256",
    )
    if contract_sha256 != sha256_json(contract):
        raise ValueError("runtime contract SHA-256 mismatch")

    runtime = _mapping(
        reference["declared_runtime_observation"],
        "declared runtime observation",
    )
    _exact_keys(runtime, _RUNTIME_EVIDENCE_KEYS, "declared runtime observation")
    if (
        runtime["schema_version"] != 1
        or runtime["accelerator_name"] != "NVIDIA A40"
        or runtime["cuda_available"] is not True
        or runtime["bf16_supported"] is not True
        or runtime["cuda_device_count"] != 1
        or runtime["cuda_device_index"] != 0
        or runtime["cuda_capability"] != [8, 6]
    ):
        raise ValueError("runtime observation must declare one NVIDIA A40")
    memory_bytes = runtime["total_memory_bytes"]
    if type(memory_bytes) is not int or memory_bytes < 40 * 1024**3:
        raise ValueError("runtime observation A40 memory is implausible")
    for field_name in ("cuda_version", "torch_version"):
        if (
            not isinstance(runtime[field_name], str)
            or not runtime[field_name]
            or runtime[field_name] != environment[field_name]
        ):
            raise ValueError(f"runtime/environment {field_name} mismatch")
    runtime_sha256 = _sha256(
        reference["declared_runtime_observation_sha256"],
        "declared runtime observation SHA-256",
    )
    if runtime_sha256 != sha256_json(runtime):
        raise ValueError("declared runtime observation SHA-256 mismatch")

    reference["declared_semantic_environment"] = environment
    reference["declared_runtime_contract"] = contract
    reference["declared_runtime_observation"] = runtime
    return reference


__all__ = [
    "CostProtocol",
    "RawStage1CostRecord",
    "benchmark_alternating_branches",
    "build_raw_stage1_cost_record",
    "load_cost_protocol",
    "operator_complexity_key",
    "publish_raw_cost_record_exclusive",
    "validate_self_attested_input_reference",
    "validate_self_attested_runtime_reference",
]
