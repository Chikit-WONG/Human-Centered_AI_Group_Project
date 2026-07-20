from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import samga_brain_rw.inference_cost as inference_cost
from samga_brain_rw.fusion import enumerate_stage1_configs
from samga_brain_rw.hashing import sha256_json


_CANONICAL_FORMAL_DIGEST = (
    "02d7e33b3fe8e5a571f8db232ca5fa86abb0c16981876ec84feae7ba64636f1a"
)


def test_inference_cost_module_exposes_only_raw_record_api() -> None:
    assert {
        "CostProtocol",
        "RawStage1CostRecord",
        "benchmark_alternating_branches",
        "build_raw_stage1_cost_record",
        "load_cost_protocol",
        "operator_complexity_key",
        "publish_raw_cost_record_exclusive",
        "validate_self_attested_input_reference",
        "validate_self_attested_runtime_reference",
    }.issubset(set(inference_cost.__all__))
    assert "Stage1CostEvidence" not in inference_cost.__all__
    assert "TrustedStage1CostEvidence" not in vars(inference_cost)
    assert "issue_trusted_cost_capability" not in vars(inference_cost)


def _cost_config(experiment_root: Path) -> Path:
    return experiment_root / "configs" / "stage1_cost_v1.json"


def test_cost_protocol_locks_the_branch_benchmark_and_quality_gate(
    experiment_root: Path,
) -> None:
    protocol = inference_cost.load_cost_protocol(_cost_config(experiment_root))

    assert protocol.config_id == "stage1_cost_v1"
    assert protocol.scope == "stage1-cost"
    assert protocol.branch_ids == ("internvit", "brainrw")
    assert protocol.warmup_runs == 10
    assert protocol.measured_runs == 50
    assert protocol.statistic == "median"
    assert protocol.dispersion == ("mad", "iqr")
    assert protocol.mad_ratio_max == 0.05
    assert protocol.same_process is True
    assert protocol.interleave_order == "alternate_each_round"
    assert protocol.cuda_synchronize == "before_and_after"
    assert protocol.timer == "perf_counter_ns"
    assert protocol.synthetic_workload["input_kind"] == ("synthetic_preprocessed")
    assert protocol.synthetic_workload_sha256 == sha256_json(
        protocol.synthetic_workload
    )
    assert protocol.sha256 == sha256_json(protocol.to_payload())


def test_cost_protocol_covers_the_exact_47_ids_with_static_family_keys(
    experiment_root: Path,
) -> None:
    protocol = inference_cost.load_cost_protocol(_cost_config(experiment_root))
    configs = enumerate_stage1_configs()

    assert set(protocol.operator_complexity) == {config.config_id for config in configs}
    keys_by_family = {
        family: {
            inference_cost.operator_complexity_key(
                protocol,
                config.config_id,
            )
            for config in configs
            if config.family == family
        }
        for family in ("temperature_convex", "zscore_convex", "rrf")
    }
    assert all(len(keys) == 1 for keys in keys_by_family.values())
    assert (
        next(iter(keys_by_family["temperature_convex"]))
        < next(iter(keys_by_family["zscore_convex"]))
        < next(iter(keys_by_family["rrf"]))
    )
    assert protocol.operator_cost_kind == ("deterministic_operator_complexity_key")
    assert (
        "latency"
        not in json.dumps(
            protocol.operator_complexity,
            sort_keys=True,
        ).lower()
    )


def test_cost_protocol_rejects_public_and_forged_construction(
    experiment_root: Path,
) -> None:
    protocol = inference_cost.load_cost_protocol(_cost_config(experiment_root))

    with pytest.raises(TypeError, match="load_cost_protocol|controlled"):
        inference_cost.CostProtocol(
            config_id=protocol.config_id,
            scope=protocol.scope,
            branch_ids=protocol.branch_ids,
            warmup_runs=protocol.warmup_runs,
            measured_runs=protocol.measured_runs,
            statistic=protocol.statistic,
            dispersion=protocol.dispersion,
            mad_ratio_max=protocol.mad_ratio_max,
            same_process=protocol.same_process,
            interleave_order=protocol.interleave_order,
            cuda_synchronize=protocol.cuda_synchronize,
            timer=protocol.timer,
            synthetic_workload_sha256=protocol.synthetic_workload_sha256,
            operator_cost_kind=protocol.operator_cost_kind,
            fusion_config_sha256=protocol.fusion_config_sha256,
            _operator_complexity=protocol.operator_complexity,
            _operator_families=protocol.operator_families,
            _synthetic_workload=protocol.synthetic_workload,
            _payload=protocol.to_payload(),
        )

    config_id = next(iter(protocol.operator_complexity))
    forged = object.__new__(inference_cost.CostProtocol)
    object.__setattr__(
        forged,
        "_operator_complexity",
        {config_id: protocol.operator_complexity[config_id]},
    )
    with pytest.raises(TypeError, match="load_cost_protocol|controlled"):
        inference_cost.operator_complexity_key(forged, config_id)


def test_cost_protocol_recursively_freezes_all_private_state(
    experiment_root: Path,
) -> None:
    protocol = inference_cost.load_cost_protocol(_cost_config(experiment_root))
    config_id = next(iter(protocol.operator_complexity))

    with pytest.raises(TypeError):
        protocol._operator_complexity[config_id] = (9, 9, 9)
    with pytest.raises(TypeError):
        protocol._operator_families[config_id] = "forged"
    with pytest.raises(TypeError):
        protocol._synthetic_workload["branches"]["internvit"]["image_shape"][0] = 1
    with pytest.raises(TypeError):
        protocol._payload["benchmark"]["warmup_runs"] = 0

    public_payload = protocol.to_payload()
    public_payload["benchmark"]["warmup_runs"] = 0
    public_workload = protocol.synthetic_workload
    public_workload["branches"]["internvit"]["image_shape"][0] = 1
    assert protocol.warmup_runs == 10
    assert protocol.synthetic_workload["branches"]["internvit"]["image_shape"] == [
        200,
        3,
        448,
        448,
    ]


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda value: value["benchmark"].__setitem__("measured_runs", 49),
            "measured_runs",
        ),
        (
            lambda value: value["quality_gate"].__setitem__(
                "mad_over_median_max", 0.051
            ),
            "MAD|mad",
        ),
        (
            lambda value: value["fusion_operator_complexity"].pop(),
            "47|config",
        ),
        (
            lambda value: value["fusion_operator_complexity"][0].__setitem__(
                "operator_complexity_key", [9, 9, 9]
            ),
            "complexity|family",
        ),
        (
            lambda value: value["synthetic_workload"].__setitem__(
                "input_kind", "val-dev"
            ),
            "synthetic|workload|scope",
        ),
    ],
)
def test_cost_protocol_rejects_any_semantic_drift(
    experiment_root: Path,
    tmp_path: Path,
    mutation: object,
    match: str,
) -> None:
    with _cost_config(experiment_root).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    mutated = copy.deepcopy(payload)
    mutation(mutated)
    path = tmp_path / "mutated-cost.json"
    path.write_text(json.dumps(mutated), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        inference_cost.load_cost_protocol(path)


class _DurationClock:
    def __init__(self, durations_ns: list[int]) -> None:
        self._durations = iter(durations_ns)
        self._now_ns = 0
        self._next_call_starts_interval = True

    def __call__(self) -> int:
        if self._next_call_starts_interval:
            self._next_call_starts_interval = False
            return self._now_ns
        self._now_ns += next(self._durations)
        self._next_call_starts_interval = True
        return self._now_ns


def test_alternating_benchmark_records_raw_ns_and_robust_statistics(
    experiment_root: Path,
) -> None:
    protocol = inference_cost.load_cost_protocol(_cost_config(experiment_root))
    calls: list[str] = []
    synchronizations: list[None] = []

    def branch(branch_id: str) -> object:
        def run() -> None:
            calls.append(branch_id)

        return run

    result = inference_cost.benchmark_alternating_branches(
        protocol,
        {
            "internvit": branch("internvit"),
            "brainrw": branch("brainrw"),
        },
        clock_ns=_DurationClock([1_000_000] * 100),
        synchronize=lambda: synchronizations.append(None),
    )

    expected_order: list[str] = []
    for round_index in range(60):
        pair = ["internvit", "brainrw"]
        if round_index % 2:
            pair.reverse()
        expected_order.extend(pair)
    assert calls == expected_order
    assert len(synchronizations) == 60 * 2 * 2
    assert result["measurement_kind"] == ("caller_supplied_branch_callable_duration")
    assert result["raw_unit"] == "nanoseconds"
    assert result["reported_unit"] == "milliseconds_per_query"
    assert result["warmup_runs"] == 10
    assert result["measured_runs"] == 50
    assert result["measurement_order"] == expected_order[20:]

    for branch_id in protocol.branch_ids:
        branch_result = result["branches"][branch_id]
        assert branch_result["elapsed_nanoseconds"] == [1_000_000] * 50
        assert branch_result["median_elapsed_nanoseconds"] == 1_000_000
        assert branch_result["mad_elapsed_nanoseconds"] == 0
        assert branch_result["q1_elapsed_nanoseconds"] == 1_000_000
        assert branch_result["q3_elapsed_nanoseconds"] == 1_000_000
        assert branch_result["iqr_elapsed_nanoseconds"] == 0
        assert branch_result["mad_over_median"] == 0
        assert branch_result["milliseconds_per_query"] == 0.005


def test_alternating_benchmark_validates_last_results_after_timing(
    experiment_root: Path,
) -> None:
    protocol = inference_cost.load_cost_protocol(_cost_config(experiment_root))
    events: list[tuple[str, object]] = []
    clock = _DurationClock([1_000_000] * 100)

    def traced_clock() -> int:
        value = clock()
        events.append(("clock", value))
        return value

    def branch(branch_id: str) -> object:
        def run() -> str:
            return f"{branch_id}-result"

        return run

    inference_cost.benchmark_alternating_branches(
        protocol,
        {
            "internvit": branch("internvit"),
            "brainrw": branch("brainrw"),
        },
        clock_ns=traced_clock,
        synchronize=lambda: None,
        result_validator=lambda branch_id, value: events.append(
            ("validate", (branch_id, value))
        ),
    )

    validation_indexes = [
        index for index, event in enumerate(events) if event[0] == "validate"
    ]
    assert len(validation_indexes) == 2
    assert min(validation_indexes) > max(
        index for index, event in enumerate(events) if event[0] == "clock"
    )
    assert [events[index][1] for index in validation_indexes] == [
        ("internvit", "internvit-result"),
        ("brainrw", "brainrw-result"),
    ]


def test_alternating_benchmark_rejects_wrong_branch_set_before_calling(
    experiment_root: Path,
) -> None:
    protocol = inference_cost.load_cost_protocol(_cost_config(experiment_root))
    called = False

    def workload() -> None:
        nonlocal called
        called = True

    with pytest.raises(ValueError, match="exact branch"):
        inference_cost.benchmark_alternating_branches(
            protocol,
            {"internvit": workload},
            clock_ns=_DurationClock([]),
            synchronize=lambda: None,
        )
    assert called is False


def test_alternating_benchmark_aborts_when_mad_gate_fails(
    experiment_root: Path,
) -> None:
    protocol = inference_cost.load_cost_protocol(_cost_config(experiment_root))

    with pytest.raises(ValueError, match="MAD.*quality"):
        inference_cost.benchmark_alternating_branches(
            protocol,
            {
                "internvit": lambda: None,
                "brainrw": lambda: None,
            },
            clock_ns=_DurationClock([1_000_000, 2_000_000] * 50),
            synchronize=lambda: None,
        )


def _digest(identity: str) -> str:
    return sha256_json({"identity": identity})


def _runtime_reference() -> dict[str, object]:
    semantic_environment = {
        "cuda_version": "12.6",
        "python_version": "3.11.11",
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
        "declared_semantic_environment": semantic_environment,
        "declared_semantic_environment_sha256": sha256_json(semantic_environment),
    }


def _model_reference() -> dict[str, object]:
    return {
        "schema_version": 1,
        "branches": {
            branch_id: {
                "checkpoint_sha256": _digest(f"{branch_id}-representative-checkpoint"),
                "model_code_sha256": _digest(f"{branch_id}-model-code"),
                "model_config_sha256": _digest(f"{branch_id}-model-config"),
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
    }


def _input_reference() -> dict[str, object]:
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
                                f"{cell_id}-{branch_id}-score-envelope"
                            ),
                            "score_payload_sha256": _digest(
                                f"{cell_id}-{branch_id}-score-payload"
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


def _job_claim_reference() -> dict[str, object]:
    return {
        "authority_execution_file_sha256": _digest(
            "stage1-cost-authority-execution"
        ),
        "authority_execution_payload_sha256": _digest(
            "stage1-cost-authority-execution-payload"
        ),
        "attempt_id": "attempt-0000",
        "attempt_index": 0,
        "claim_id": "stage1-cost-a40",
        "unverified_claim_sha256": _digest("stage1-cost-a40-claim"),
        "unverified_previous_record_sha256": None,
        "schema_version": 1,
        "slurm_job_id": "123456",
        "slurm_partition": "debug",
    }


def _stable_benchmark(
    protocol: inference_cost.CostProtocol,
) -> dict[str, object]:
    return inference_cost.benchmark_alternating_branches(
        protocol,
        {
            "internvit": lambda: None,
            "brainrw": lambda: None,
        },
        clock_ns=_DurationClock([1_000_000] * 100),
        synchronize=lambda: None,
    )


def _raw_record(
    protocol: inference_cost.CostProtocol,
    *,
    job_claim_reference: dict[str, object] | None = None,
) -> inference_cost.RawStage1CostRecord:
    return inference_cost.build_raw_stage1_cost_record(
        protocol,
        _stable_benchmark(protocol),
        runtime_reference=_runtime_reference(),
        model_reference=_model_reference(),
        job_claim_reference=(
            _job_claim_reference()
            if job_claim_reference is None
            else job_claim_reference
        ),
        input_reference=_input_reference(),
    )


def test_noop_benchmark_builds_only_a_self_attested_raw_record(
    experiment_root: Path,
) -> None:
    protocol = inference_cost.load_cost_protocol(_cost_config(experiment_root))
    runtime = _runtime_reference()
    models = _model_reference()
    inputs = _input_reference()
    job_claim = _job_claim_reference()

    record = inference_cost.build_raw_stage1_cost_record(
        protocol,
        _stable_benchmark(protocol),
        runtime_reference=runtime,
        model_reference=models,
        job_claim_reference=job_claim,
        input_reference=inputs,
    )

    assert isinstance(record, inference_cost.RawStage1CostRecord)
    payload = record.to_payload()
    assert payload["artifact_type"] == ("samga_brain_rw.raw_stage1_cost_record")
    assert payload["scope"] == "stage1-cost"
    assert payload["record_kind"] == "hash_closed_raw_measurement_record"
    assert payload["reference_trust"] == "self_attested_unverified"
    assert payload["trusted_capability_issued"] is False
    assert payload["composition_eligible"] is False
    assert payload["confirmation_eligible"] is False
    assert payload["measurement_protocol_sha256"] == protocol.sha256
    assert payload["self_attested_runtime_reference_sha256"] == sha256_json(runtime)
    assert payload["self_attested_model_reference_sha256"] == sha256_json(models)
    assert payload["self_attested_job_claim_reference_sha256"] == sha256_json(job_claim)
    assert payload["self_attested_input_reference_sha256"] == sha256_json(inputs)
    assert payload["branch_benchmark_sha256"] == sha256_json(
        payload["branch_benchmark"]
    )
    assert len(payload["fusion_operator_complexity"]) == 47
    assert {item["config_id"] for item in payload["fusion_operator_complexity"]} == set(
        protocol.operator_complexity
    )
    assert (
        "latency"
        not in json.dumps(
            payload["fusion_operator_complexity"],
            sort_keys=True,
        ).lower()
    )
    assert record.record_sha256 == sha256_json(payload)
    assert record.to_document() == {
        **payload,
        "record_sha256": record.record_sha256,
    }
    assert (
        inference_cost.RawStage1CostRecord.from_document(record.to_document()) == record
    )


def test_runtime_reference_is_only_structurally_hash_closed() -> None:
    runtime = _runtime_reference()
    assert inference_cost.validate_self_attested_runtime_reference(runtime) == runtime

    bad_hash = copy.deepcopy(runtime)
    bad_hash["declared_runtime_observation_sha256"] = _digest("wrong")
    with pytest.raises(ValueError, match="runtime observation.*SHA"):
        inference_cost.validate_self_attested_runtime_reference(bad_hash)

    wrong_gpu = copy.deepcopy(runtime)
    wrong_gpu["declared_runtime_contract"]["accelerator"] = "NVIDIA A800"
    wrong_gpu["declared_runtime_contract_sha256"] = sha256_json(
        wrong_gpu["declared_runtime_contract"]
    )
    with pytest.raises(ValueError, match="A40"):
        inference_cost.validate_self_attested_runtime_reference(wrong_gpu)


@pytest.mark.parametrize(
    "forbidden",
    [
        "formal",
        "formal-test",
        "formal-input",
        "formal-refit",
        "prefix-formal-suffix",
        "/safe/val-confirm/reference",
        "/safe/val_confirm/reference",
        _CANONICAL_FORMAL_DIGEST,
    ],
)
def test_self_attested_references_reject_every_denied_scope_alias(
    forbidden: str,
) -> None:
    runtime = _runtime_reference()
    runtime["declared_semantic_environment"]["python_version"] = forbidden
    runtime["declared_semantic_environment_sha256"] = sha256_json(
        runtime["declared_semantic_environment"]
    )

    with pytest.raises(ValueError, match="forbidden|scope"):
        inference_cost.validate_self_attested_runtime_reference(runtime)


def test_input_reference_is_exact_six_cells_and_observation_free() -> None:
    inputs = _input_reference()
    assert inference_cost.validate_self_attested_input_reference(inputs) == inputs

    missing_cell = copy.deepcopy(inputs)
    missing_cell["cells"].pop()
    with pytest.raises(ValueError, match="six-cell"):
        inference_cost.validate_self_attested_input_reference(missing_cell)

    observed = copy.deepcopy(inputs)
    observed["cells"][0]["metrics"] = {"top1": 1.0}
    with pytest.raises(ValueError, match="metrics|labels|observations"):
        inference_cost.validate_self_attested_input_reference(observed)


def test_raw_record_publication_is_exclusive(
    experiment_root: Path,
    tmp_path: Path,
) -> None:
    protocol = inference_cost.load_cost_protocol(_cost_config(experiment_root))
    record = _raw_record(protocol)
    output = tmp_path / "stage1-cost-attempt-0000.json"
    inference_cost.publish_raw_cost_record_exclusive(output, record)
    first_bytes = output.read_bytes()
    with pytest.raises(FileExistsError):
        inference_cost.publish_raw_cost_record_exclusive(output, record)
    assert output.read_bytes() == first_bytes
    assert json.loads(first_bytes) == record.to_document()


def test_previous_attempt_digest_is_only_an_unverified_reference(
    experiment_root: Path,
) -> None:
    protocol = inference_cost.load_cost_protocol(_cost_config(experiment_root))
    claim = _job_claim_reference()
    claim["attempt_id"] = "attempt-0001"
    claim["attempt_index"] = 1
    claim["unverified_previous_record_sha256"] = _digest("previous-record")

    payload = _raw_record(
        protocol,
        job_claim_reference=claim,
    ).to_payload()

    reference = payload["self_attested_job_claim_reference"]
    assert reference["unverified_previous_record_sha256"] == _digest("previous-record")
    assert "chain_verified" not in reference
    assert payload["trusted_capability_issued"] is False


@pytest.mark.parametrize(
    "denied_component",
    [
        "formal",
        "formal-input",
        "formal-refit",
        "prefix-formal-suffix",
        "val-confirm",
        _CANONICAL_FORMAL_DIGEST,
    ],
)
def test_raw_record_publisher_rejects_denied_development_paths(
    experiment_root: Path,
    tmp_path: Path,
    denied_component: str,
) -> None:
    protocol = inference_cost.load_cost_protocol(_cost_config(experiment_root))
    output = tmp_path / denied_component / "stage1-cost-attempt-0000.json"

    with pytest.raises(ValueError, match="forbidden|development"):
        inference_cost.publish_raw_cost_record_exclusive(
            output,
            _raw_record(protocol),
        )
    assert not output.exists()


def test_raw_record_publisher_rejects_alias_filename_and_symlink_parent(
    experiment_root: Path,
    tmp_path: Path,
) -> None:
    protocol = inference_cost.load_cost_protocol(_cost_config(experiment_root))
    record = _raw_record(protocol)

    with pytest.raises(ValueError, match="attempt|filename"):
        inference_cost.publish_raw_cost_record_exclusive(
            tmp_path / "cost-record.json",
            record,
        )

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    alias_parent = tmp_path / "alias-parent"
    alias_parent.symlink_to(real_parent, target_is_directory=True)
    aliased_output = alias_parent / "stage1-cost-attempt-0000.json"
    with pytest.raises(OSError):
        inference_cost.publish_raw_cost_record_exclusive(
            aliased_output,
            record,
        )
    assert not (real_parent / aliased_output.name).exists()
