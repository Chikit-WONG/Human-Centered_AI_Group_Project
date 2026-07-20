from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from samga_brain_rw.hashing import canonical_json_bytes, sha256_json
from samga_brain_rw.inference_cost import load_cost_protocol


def test_execution_plan_preregisters_exact_full_model_microbatch_semantics(
    experiment_root: Path,
) -> None:
    from samga_brain_rw.cost_capability import (
        load_stage1_cost_execution_plan,
    )

    plan = load_stage1_cost_execution_plan(
        experiment_root / "configs/stage1_cost_execution_v1.json"
    )

    assert plan.config_id == "stage1_cost_execution_v1"
    assert plan.seed == 20260720
    assert plan.query_count == 200 and plan.gallery_count == 200
    assert plan.branch_order == ("internvit", "brainrw")
    assert plan.image_microbatch_size("internvit") == 16
    assert plan.eeg_microbatch_size("internvit") == 200
    assert plan.image_microbatch_size("brainrw") == 64
    assert plan.eeg_microbatch_size("brainrw") == 200
    assert (
        plan.branch_payload("internvit")["representative_subject"],
        plan.branch_payload("internvit")["representative_seed"],
    ) == (1, 42)
    assert (
        plan.branch_payload("brainrw")["representative_subject"],
        plan.branch_payload("brainrw")["representative_seed"],
    ) == (1, 42)
    assert plan.labels_present is False
    assert plan.metrics_computed is False
    assert plan.synchronize == "protocol_before_and_after_branch_callable"
    assert plan.to_payload()["coverage"]["synthetic_generation_device"] == "cpu"
    assert plan.to_payload()["coverage"]["synthetic_distribution"] == (
        "standard_normal_float32_then_cast"
    )
    assert plan.to_payload()["coverage"]["synthetic_tensor_order"] == [
        "internvit.eeg",
        "internvit.image",
        "brainrw.eeg",
        "brainrw.image",
    ]
    assert plan.sha256 == sha256_json(plan.to_payload())


@pytest.mark.parametrize(
    ("path", "replacement", "match"),
    [
        (("branches", "internvit", "image_microbatch_size"), 8, "microbatch"),
        (("branches", "brainrw", "image_microbatch_size"), 32, "microbatch"),
        (("coverage", "query_count"), 199, "coverage|200"),
        (("observations", "metrics_computed"), True, "metrics"),
        (("runtime", "accelerator"), "NVIDIA A800", "A40"),
    ],
)
def test_execution_plan_rejects_semantic_drift(
    experiment_root: Path,
    tmp_path: Path,
    path: tuple[str, ...],
    replacement: object,
    match: str,
) -> None:
    from samga_brain_rw.cost_capability import (
        load_stage1_cost_execution_plan,
    )

    source = experiment_root / "configs/stage1_cost_execution_v1.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    target: object = payload
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = replacement  # type: ignore[index]
    destination = tmp_path / "execution.json"
    destination.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_stage1_cost_execution_plan(destination)


def test_internvit_runtime_architecture_requires_exact_45_layers(
    cost_runner_module: ModuleType,
) -> None:
    valid = SimpleNamespace(
        config=SimpleNamespace(
            hidden_size=3200,
            image_size=448,
            num_hidden_layers=45,
            patch_size=14,
        )
    )

    cost_runner_module._validate_internvit_architecture(valid)

    for layer_count in (44, 46):
        invalid = SimpleNamespace(
            config=SimpleNamespace(
                hidden_size=3200,
                image_size=448,
                num_hidden_layers=layer_count,
                patch_size=14,
            )
        )
        with pytest.raises(ValueError, match="architecture|45"):
            cost_runner_module._validate_internvit_architecture(invalid)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _load_script(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def jobmap_module(experiment_root: Path) -> ModuleType:
    return _load_script(
        experiment_root / "scripts/build_job_map.py",
        "stage1_cost_test_build_job_map",
    )


@pytest.fixture
def cost_builder_module(
    experiment_root: Path,
    jobmap_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> ModuleType:
    monkeypatch.setitem(sys.modules, "build_job_map", jobmap_module)
    return _load_script(
        experiment_root / "scripts/build_stage1_cost_job_map.py",
        "build_stage1_cost_job_map",
    )


@pytest.fixture(scope="module")
def cost_runner_module(experiment_root: Path) -> ModuleType:
    return _load_script(
        experiment_root / "scripts/run_stage1_cost.py",
        "run_stage1_cost",
    )


def _install_builder_inputs(
    module: ModuleType,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, str, str, str]:
    protocol_path = (
        project_root
        / "experiments/samga_brain_rw/configs/stage1_cost_v1.json"
    )
    protocol_path.parent.mkdir(parents=True)
    protocol_path.write_bytes(b"{\"sealed\":\"protocol\"}")
    execution_path = protocol_path.with_name("stage1_cost_execution_v1.json")
    execution_path.write_bytes(b"{\"sealed\":\"execution\"}")
    runner_path = (
        project_root
        / "experiments/samga_brain_rw/scripts/run_stage1_cost.py"
    )
    runner_path.parent.mkdir(parents=True)
    runner_path.write_bytes(b"#!/usr/bin/env python3\n# sealed runner\n")
    score_path = (
        project_root
        / "artifacts/samga_brain_rw/stage-1-cost-inputs/score-inputs.json"
    )
    model_path = score_path.with_name("model-manifest.json")
    score_path.parent.mkdir(parents=True)
    score_path.write_bytes(b"{\"sealed\":\"scores\"}")
    model_path.write_bytes(b"{\"sealed\":\"models\"}")
    score_file_sha256 = hashlib.sha256(score_path.read_bytes()).hexdigest()
    model_file_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    protocol_sha256 = _digest("protocol")
    execution_sha256 = _digest("execution")

    monkeypatch.setattr(
        module,
        "load_cost_protocol",
        lambda path: (
            SimpleNamespace(
                config_id="stage1_cost_v1",
                sha256=protocol_sha256,
                synthetic_workload={
                    "seed": 20260720,
                    "query_count": 200,
                    "gallery_count": 200,
                    "labels_present": False,
                    "metrics_computed": False,
                },
                warmup_runs=10,
                measured_runs=50,
                mad_ratio_max=0.05,
            )
            if path == protocol_path
            else pytest.fail("builder opened an unsealed protocol")
        ),
    )
    monkeypatch.setattr(
        module,
        "load_stage1_cost_execution_plan",
        lambda path: (
            SimpleNamespace(
                config_id="stage1_cost_execution_v1",
                sha256=execution_sha256,
                seed=20260720,
                query_count=200,
                gallery_count=200,
                labels_present=False,
                metrics_computed=False,
                branch_order=("internvit", "brainrw"),
            )
            if path == execution_path
            else pytest.fail("builder opened an unsealed execution plan")
        ),
    )
    monkeypatch.setattr(
        module,
        "load_stage1_cost_score_input_manifest",
        lambda path: (
            ({"score_inputs": [object()] * 6}, score_file_sha256)
            if path == score_path
            else pytest.fail("builder opened unsealed score identities")
        ),
    )
    monkeypatch.setattr(
        module,
        "load_stage1_cost_model_manifest",
        lambda path: (
            (
                {
                    "branches": {
                        "internvit": {"factory": "internvit_v2_5_plus_samga"},
                        "brainrw": {"factory": "brainrw_clip_lora"},
                    }
                },
                model_file_sha256,
            )
            if path == model_path
            else pytest.fail("builder opened an unsealed model manifest")
        ),
    )
    return (
        score_path,
        model_path,
        score_file_sha256,
        model_file_sha256,
        execution_sha256,
    )


def test_cost_job_map_is_one_exact_low_partition_a40_benchmark(
    cost_builder_module: ModuleType,
    jobmap_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_root = (tmp_path / "project").resolve()
    project_root.mkdir()
    (project_root / ".git").mkdir()
    (
        score_path,
        model_path,
        score_sha256,
        model_sha256,
        execution_sha256,
    ) = _install_builder_inputs(
        cost_builder_module,
        project_root,
        monkeypatch,
    )

    rows = cost_builder_module.build_stage1_cost_rows(
        project_root=project_root,
    )
    payload = jobmap_module.build_job_map(rows)

    assert payload["stage"] == "stage-1-cost-benchmark"
    assert payload["array_bounds"] == [0, 0]
    assert payload["row_count"] == 1
    row = payload["rows"][0]
    assert row["role"] == "cost-benchmark"
    assert row["partition"] == "i64m1tga40u"
    assert row["gres"] == "gpu:a40:1"
    assert row["cpus"] == 16
    assert row["memory"] == "64G"
    assert row["time"] == "12:00:00"
    assert (row["subject"], row["seed"]) == (1, 20260720)
    assert row["config_id"] == "stage1_cost_v1"
    assert row["expected_completion_schema"] == {
        "payload_type": "samga_brain_rw.stage1_cost_completion",
        "required_output_hashes": [
            "raw_record_file_sha256",
            "run_manifest_file_sha256",
            "runtime_manifest_file_sha256",
        ],
        "schema_version": 1,
    }
    expected_bundle = sha256_json(
        {
            "execution_config_sha256": execution_sha256,
            "model_manifest_file_sha256": model_sha256,
            "runner_file_sha256": hashlib.sha256(
                (
                    project_root
                    / "experiments/samga_brain_rw/scripts/run_stage1_cost.py"
                ).read_bytes()
            ).hexdigest(),
            "score_inputs_file_sha256": score_sha256,
        }
    )
    assert row["input_bundle_sha256"] == expected_bundle
    argv = row["argv"]
    assert argv[argv.index("--subject") + 1] == "1"
    assert argv[argv.index("--seed") + 1] == "20260720"
    assert argv[argv.index("--device") + 1] == "cuda"
    assert argv[argv.index("--score-inputs") + 1] == str(score_path)
    assert argv[argv.index("--model-manifest") + 1] == str(model_path)
    assert argv[argv.index("--expected-execution-config-sha256") + 1] == (
        execution_sha256
    )
    assert argv[argv.index("--execution-config") + 1].endswith(
        "configs/stage1_cost_execution_v1.json"
    )
    joined = "\n".join(argv)
    assert "val-confirm" not in joined
    assert "formal-test" not in joined
    assert "test_images" not in joined


def test_cost_job_map_input_bundle_changes_if_runner_bytes_change(
    cost_builder_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_root = (tmp_path / "project").resolve()
    project_root.mkdir()
    (project_root / ".git").mkdir()
    _install_builder_inputs(
        cost_builder_module,
        project_root,
        monkeypatch,
    )
    first = cost_builder_module.build_stage1_cost_rows(
        project_root=project_root,
    )[0]
    runner_path = (
        project_root
        / "experiments/samga_brain_rw/scripts/run_stage1_cost.py"
    )
    runner_path.write_bytes(runner_path.read_bytes() + b"# tampered\n")

    second = cost_builder_module.build_stage1_cost_rows(
        project_root=project_root,
    )[0]

    assert first["input_bundle_sha256"] != second["input_bundle_sha256"]


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    [
        ("seed", 1, "seed"),
        ("query_count", 199, "200"),
        ("gallery_count", 199, "200"),
        ("labels_present", True, "labels"),
        ("metrics_computed", True, "metrics"),
    ],
)
def test_cost_job_map_rejects_any_workload_protocol_drift(
    cost_builder_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    replacement: object,
    match: str,
) -> None:
    project_root = (tmp_path / "project").resolve()
    project_root.mkdir()
    (project_root / ".git").mkdir()
    _install_builder_inputs(
        cost_builder_module,
        project_root,
        monkeypatch,
    )
    original = cost_builder_module.load_cost_protocol

    def drifted(path: Path) -> object:
        protocol = original(path)
        workload = dict(protocol.synthetic_workload)
        workload[field] = replacement
        protocol.synthetic_workload = workload
        return protocol

    monkeypatch.setattr(cost_builder_module, "load_cost_protocol", drifted)

    with pytest.raises(ValueError, match=match):
        cost_builder_module.build_stage1_cost_rows(
            project_root=project_root,
        )


def test_cost_job_map_builder_exposes_no_manifest_path_overrides(
    cost_builder_module: ModuleType,
) -> None:
    parser = cost_builder_module._parser()

    assert {
        option
        for action in parser._actions
        for option in action.option_strings
    } == {"-h", "--help", "--output", "--project-root"}


def test_cost_job_map_builder_rejects_alternate_manifest_filename(
    cost_builder_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_root = (tmp_path / "project").resolve()
    project_root.mkdir()
    (project_root / ".git").mkdir()
    score_path, _, _, _, _ = _install_builder_inputs(
        cost_builder_module,
        project_root,
        monkeypatch,
    )
    score_path.rename(score_path.with_name("alternate-score-inputs.json"))

    with pytest.raises(ValueError, match="cannot be resolved"):
        cost_builder_module.build_stage1_cost_rows(
            project_root=project_root,
        )


def test_cost_job_map_parent_swap_cannot_redirect_publication(
    cost_builder_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_root = (tmp_path / "project").resolve()
    project_root.mkdir()
    (project_root / ".git").mkdir()
    _install_builder_inputs(
        cost_builder_module,
        project_root,
        monkeypatch,
    )
    output_root = project_root / "artifacts/samga_brain_rw/job_maps"
    output_root.mkdir(parents=True)
    output = output_root / "stage1-cost.json"
    detached_root = output_root.with_name("detached-job-maps")
    redirect_root = (tmp_path / "redirected-job-maps").resolve()
    redirect_root.mkdir()
    real_link = os.link
    swapped = False

    def racing_link(
        source: object,
        destination: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal swapped
        if (
            not swapped
            and Path(os.fspath(destination)).name == output.name
        ):
            output_root.rename(detached_root)
            output_root.symlink_to(redirect_root, target_is_directory=True)
            temporary_name = Path(os.fspath(source)).name
            detached_temporary = detached_root / temporary_name
            if detached_temporary.is_file():
                (redirect_root / temporary_name).write_bytes(
                    detached_temporary.read_bytes()
                )
            swapped = True
        real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(cost_builder_module.os, "link", racing_link)

    with pytest.raises(ValueError, match="symbolic|identity|changed"):
        cost_builder_module.main(
            [
                "--project-root",
                str(project_root),
                "--output",
                str(output),
            ]
        )

    assert swapped is True
    assert not (redirect_root / output.name).exists()
    assert (detached_root / output.name).is_file()


class _DurationClock:
    def __init__(self, durations_ns: list[int]) -> None:
        self._durations = iter(durations_ns)
        self._now = 0
        self._start = True

    def __call__(self) -> int:
        if self._start:
            self._start = False
            return self._now
        self._now += next(self._durations)
        self._start = True
        return self._now


def test_runner_benchmarks_only_exact_preloaded_branch_callables(
    experiment_root: Path,
    cost_runner_module: ModuleType,
) -> None:
    protocol = load_cost_protocol(
        experiment_root / "configs/stage1_cost_v1.json"
    )
    calls: list[str] = []
    measurement_order = [
        branch_id
        for round_index in range(10, 60)
        for branch_id in (
            ("internvit", "brainrw")
            if round_index % 2 == 0
            else ("brainrw", "internvit")
        )
    ]

    result = cost_runner_module.benchmark_preloaded_real_branches(
        protocol,
        {
            "internvit": lambda: (
                calls.append("internvit")
                or torch.ones((200, 200), dtype=torch.float32)
            ),
            "brainrw": lambda: (
                calls.append("brainrw")
                or torch.ones((200, 200), dtype=torch.float32)
            ),
        },
        clock_ns=_DurationClock(
            [
                2_000_000 if branch_id == "internvit" else 1_000_000
                for branch_id in measurement_order
            ]
        ),
        synchronize=lambda: None,
    )

    assert calls.count("internvit") == 60
    assert calls.count("brainrw") == 60
    assert result["query_count_per_call"] == 200
    assert result["warmup_runs"] == 10
    assert result["measured_runs"] == 50
    assert result["branches"]["internvit"]["mad_over_median"] == 0.0
    assert result["branches"]["brainrw"]["mad_over_median"] == 0.0
    forbidden = {"labels", "metrics", "top1", "top5", "predictions"}
    assert forbidden.isdisjoint(result)


def test_runner_has_no_cli_override_for_locked_benchmark_semantics(
    cost_runner_module: ModuleType,
    tmp_path: Path,
) -> None:
    parser = cost_runner_module._parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--subject",
                "1",
                "--seed",
                "20260720",
                "--config",
                str(tmp_path / "protocol.json"),
                "--score-inputs",
                str(tmp_path / "scores.json"),
                "--model-manifest",
                str(tmp_path / "models.json"),
                "--output-dir",
                str(tmp_path / "out"),
                "--project-root",
                str(tmp_path),
                "--config-id",
                "stage1_cost_v1",
                "--expected-config-sha256",
                _digest("protocol"),
                "--expected-input-bundle-sha256",
                _digest("inputs"),
                "--run-key",
                "run",
                "--device",
                "cuda",
                "--measured-runs",
                "1",
            ]
        )


def test_runner_binds_recovery_generation_to_exact_claim_file_and_slurm_job(
    cost_runner_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    map_sha256 = _digest("job-map")
    row_sha256 = _digest("row")
    claim_payload = {
        "array_index": 0,
        "generation": 2,
        "job_map_sha256": map_sha256,
        "recovered_from_claim_sha256": _digest("previous-claim"),
        "recovery_record_sha256": _digest("recovery"),
        "row_sha256": row_sha256,
    }
    claim_document = {
        "payload": claim_payload,
        "payload_sha256": sha256_json(claim_payload),
        "payload_type": "samga_brain_rw.job_claim",
        "schema_version": 1,
    }
    claim_path = (
        tmp_path / "claims/generation-000002/claim.json"
    ).resolve()
    claim_path.parent.mkdir(parents=True)
    claim_path.write_bytes(canonical_json_bytes(claim_document))
    claim_file_sha256 = hashlib.sha256(claim_path.read_bytes()).hexdigest()
    attempt_payload = {
        "array_index": 0,
        "claim_sha256": claim_file_sha256,
        "generation": 2,
        "job_map_sha256": map_sha256,
        "row_sha256": row_sha256,
        "scheduler_job_id": "123456_0",
    }
    attempt_document = {
        "payload": attempt_payload,
        "payload_sha256": sha256_json(attempt_payload),
        "payload_type": "samga_brain_rw.job_attempt",
        "schema_version": 1,
    }
    attempt_path = claim_path.with_name("attempt.json")
    attempt_path.write_bytes(canonical_json_bytes(attempt_document))
    execution_payload = {
        "array_index": 0,
        "attempt_payload_sha256": attempt_document["payload_sha256"],
        "attempt_record_sha256": hashlib.sha256(
            attempt_path.read_bytes()
        ).hexdigest(),
        "claim_sha256": claim_file_sha256,
        "generation": 2,
        "job_map_sha256": map_sha256,
        "row_sha256": row_sha256,
        "scheduler_job_id": "123456_0",
    }
    execution_document = {
        "payload": execution_payload,
        "payload_sha256": sha256_json(execution_payload),
        "payload_type": "samga_brain_rw.cost_execution_authority",
        "schema_version": 1,
    }
    execution_path = claim_path.with_name("execution.json")
    execution_path.write_bytes(canonical_json_bytes(execution_document))
    environment = {
        "SAMGA_JOB_ARRAY_INDEX": "0",
        "SAMGA_JOB_CLAIM": str(claim_path),
        "SAMGA_JOB_EXECUTION": str(execution_path),
        "SAMGA_JOB_EXECUTION_SHA256": hashlib.sha256(
            execution_path.read_bytes()
        ).hexdigest(),
        "SAMGA_JOB_MAP_SHA256": map_sha256,
        "SAMGA_JOB_ROW_SHA256": row_sha256,
        "SLURM_ARRAY_JOB_ID": "123456",
        "SLURM_ARRAY_TASK_ID": "0",
        "SLURM_JOB_PARTITION": "i64m1tga40u",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    reference = cost_runner_module._build_job_claim_reference("sealed-run")

    assert reference == {
        "authority_execution_file_sha256": hashlib.sha256(
            execution_path.read_bytes()
        ).hexdigest(),
        "authority_execution_payload_sha256": execution_document[
            "payload_sha256"
        ],
        "attempt_id": "attempt-0001",
        "attempt_index": 1,
        "claim_id": "sealed-run",
        "schema_version": 1,
        "slurm_job_id": "123456_0",
        "slurm_partition": "i64m1tga40u",
        "unverified_claim_sha256": claim_file_sha256,
        "unverified_previous_record_sha256": None,
    }
    monkeypatch.setenv("SLURM_JOB_PARTITION", "debug")
    with pytest.raises(ValueError, match="i64m1tga40u"):
        cost_runner_module._build_job_claim_reference("sealed-run")


def test_runner_publishes_attempt_named_raw_and_exact_completion_hashes(
    cost_runner_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_dir = (tmp_path / "cost-output").resolve()
    published_paths: list[Path] = []
    completion_hashes: list[dict[str, str]] = []

    class _RawRecord:
        record_sha256 = _digest("raw-record")

        @staticmethod
        def to_payload() -> dict[str, object]:
            return {
                "self_attested_job_claim_reference": {
                    "attempt_id": "attempt-0003"
                }
            }

    def publish_raw(path: Path, record: object) -> None:
        published_paths.append(path)
        path.write_bytes(b'{"raw":true}')

    def write_json(path: Path, value: dict[str, object]) -> None:
        path.write_bytes(canonical_json_bytes(value))

    monkeypatch.setattr(
        cost_runner_module,
        "publish_raw_cost_record_exclusive",
        publish_raw,
    )
    monkeypatch.setattr(
        cost_runner_module,
        "write_development_json_exclusive",
        write_json,
    )

    result = cost_runner_module._publish_sealed_outputs(
        output_dir=output_dir,
        raw_record=_RawRecord(),
        runtime_document={"runtime_evidence_sha256": _digest("runtime")},
        run_manifest_static={"scope": "stage1-cost"},
        completion_publisher=lambda values: (
            completion_hashes.append(dict(values)) or "completed"
        ),
    )

    raw_path = output_dir / "stage1-cost-attempt-0003.json"
    run_path = output_dir / "run-manifest.json"
    runtime_path = output_dir / "runtime-manifest.json"
    assert result == "completed"
    assert published_paths == [raw_path]
    assert set(completion_hashes[0]) == {
        "raw_record_file_sha256",
        "run_manifest_file_sha256",
        "runtime_manifest_file_sha256",
    }
    assert completion_hashes[0]["raw_record_file_sha256"] == (
        hashlib.sha256(raw_path.read_bytes()).hexdigest()
    )
    run_manifest = json.loads(run_path.read_text(encoding="utf-8"))
    assert run_manifest["raw_record_path"] == str(raw_path)
    assert run_manifest["runtime_manifest_path"] == str(runtime_path)
    assert run_manifest["raw_record_sha256"] == _digest("raw-record")


def test_runner_chunk_ranges_cover_every_row_once_in_order(
    cost_runner_module: ModuleType,
) -> None:
    assert cost_runner_module._chunk_ranges(200, 16) == (
        (0, 16),
        (16, 32),
        (32, 48),
        (48, 64),
        (64, 80),
        (80, 96),
        (96, 112),
        (112, 128),
        (128, 144),
        (144, 160),
        (160, 176),
        (176, 192),
        (192, 200),
    )
    assert cost_runner_module._chunk_ranges(200, 64) == (
        (0, 64),
        (64, 128),
        (128, 192),
        (192, 200),
    )
    with pytest.raises(ValueError, match="positive"):
        cost_runner_module._chunk_ranges(200, 0)


class _FakeInternEmbeddings(torch.nn.Module):
    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        values = pixels.float().mean(dim=(1, 2, 3))
        return torch.stack(
            (
                torch.stack((values, values + 1, values + 2, values + 3), dim=1),
                torch.stack((values + 4, values + 5, values + 6, values + 7), dim=1),
            ),
            dim=1,
        )


class _FakeInternLayer(torch.nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + 1


class _FakeInternFoundation(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision_model = SimpleNamespace(
            embeddings=_FakeInternEmbeddings(),
            encoder=SimpleNamespace(
                layers=torch.nn.ModuleList(
                    [_FakeInternLayer(), _FakeInternLayer()]
                )
            ),
        )


def _fake_collect_block_poolings(
    model: torch.nn.Module,
    pixels: torch.Tensor,
    *,
    captured_block_outputs: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    assert captured_block_outputs == (1, 2)
    hidden = model.vision_model.embeddings(pixels)
    values = []
    for layer in model.vision_model.encoder.layers:
        hidden = layer(hidden)
        values.append(hidden[:, 1:, :].mean(dim=1))
    patch_mean = torch.stack(values, dim=1)
    return torch.zeros_like(patch_mean), patch_mean


class _FakeSAMGATask(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.eeg_batches: list[int] = []
        self.image_batches: list[int] = []

    def encode_eeg(self, eeg: torch.Tensor) -> torch.Tensor:
        self.eeg_batches.append(eeg.shape[0])
        return eeg.float().mean(dim=(1, 2)).unsqueeze(1).repeat(1, 4)

    def encode_image(
        self,
        features: torch.Tensor,
        subject_ids: torch.Tensor,
        *,
        force_global: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert force_global is True
        assert subject_ids.tolist() == [1] * features.shape[0]
        self.image_batches.append(features.shape[0])
        return features.float().mean(dim=1), torch.ones(
            features.shape[0], features.shape[1]
        )


def test_internvit_callable_executes_full_eeg_and_image_paths_in_microbatches(
    cost_runner_module: ModuleType,
) -> None:
    task = _FakeSAMGATask()
    callable_ = cost_runner_module._make_internvit_similarity_callable(
        foundation_model=_FakeInternFoundation(),
        task_model=task,
        eeg=torch.arange(4 * 2 * 2, dtype=torch.float32).reshape(4, 2, 2),
        pixels=torch.arange(
            4 * 3 * 2 * 2, dtype=torch.bfloat16
        ).reshape(4, 3, 2, 2),
        subject=1,
        eeg_microbatch_size=3,
        image_microbatch_size=2,
        layer_ids=(1, 2),
        device=torch.device("cpu"),
        foundation_autocast=False,
        pooling_helper=_fake_collect_block_poolings,
    )

    similarity = callable_()

    assert similarity.shape == (4, 4)
    assert similarity.dtype == torch.float32
    assert bool(torch.isfinite(similarity).all())
    assert task.eeg_batches == [3, 1]
    assert task.image_batches == [2, 2]


class _FakeBrainRW(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.eeg_batches: list[int] = []
        self.image_batches: list[int] = []

    def encode_brain(self, eeg: torch.Tensor) -> torch.Tensor:
        self.eeg_batches.append(eeg.shape[0])
        values = eeg.float().mean(dim=(1, 2))
        return torch.stack((values, values + 1, values + 2), dim=1)

    def encode_image(self, pixels: torch.Tensor) -> torch.Tensor:
        self.image_batches.append(pixels.shape[0])
        values = pixels.float().mean(dim=(1, 2, 3))
        return torch.stack((values, values + 1, values + 2), dim=1)


def test_brainrw_callable_executes_full_eeg_and_image_paths_in_microbatches(
    cost_runner_module: ModuleType,
) -> None:
    model = _FakeBrainRW()
    callable_ = cost_runner_module._make_brainrw_similarity_callable(
        model=model,
        eeg=torch.arange(
            5 * 2 * 2, dtype=torch.bfloat16
        ).reshape(5, 2, 2),
        pixels=torch.arange(
            5 * 3 * 2 * 2, dtype=torch.bfloat16
        ).reshape(5, 3, 2, 2),
        eeg_microbatch_size=3,
        image_microbatch_size=2,
    )

    similarity = callable_()

    assert similarity.shape == (5, 5)
    assert similarity.dtype == torch.float32
    assert bool(torch.isfinite(similarity).all())
    assert model.eeg_batches == [3, 2]
    assert model.image_batches == [2, 2, 1]


def test_brainrw_callable_does_not_repeat_model_l2_normalization(
    cost_runner_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cost_runner_module.F,
        "normalize",
        lambda *args, **kwargs: pytest.fail(
            "BrainRW embeddings are already normalized by the model"
        ),
    )
    callable_ = cost_runner_module._make_brainrw_similarity_callable(
        model=_FakeBrainRW(),
        eeg=torch.ones((2, 2, 2), dtype=torch.bfloat16),
        pixels=torch.ones((2, 3, 2, 2), dtype=torch.bfloat16),
        eeg_microbatch_size=2,
        image_microbatch_size=2,
    )

    similarity = callable_()

    assert similarity.shape == (2, 2)


def test_similarity_finite_validation_runs_after_benchmark_timing(
    experiment_root: Path,
    cost_runner_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = load_cost_protocol(
        experiment_root / "configs/stage1_cost_v1.json"
    )
    finite_calls: list[torch.Tensor] = []
    original_isfinite = cost_runner_module.torch.isfinite
    monkeypatch.setattr(
        cost_runner_module.torch,
        "isfinite",
        lambda value: (
            finite_calls.append(value),
            original_isfinite(value),
        )[1],
    )
    callables = {
        "internvit": lambda: torch.ones((200, 200), dtype=torch.float32),
        "brainrw": lambda: torch.ones((200, 200), dtype=torch.float32),
    }

    cost_runner_module.benchmark_preloaded_real_branches(
        protocol,
        callables,
        clock_ns=_DurationClock([1_000_000] * 100),
        synchronize=lambda: None,
    )

    assert len(finite_calls) == 2


def test_real_factory_assembles_exact_two_preloaded_full_model_callables(
    experiment_root: Path,
    cost_runner_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from samga_brain_rw.cost_capability import (
        load_stage1_cost_execution_plan,
    )

    plan = load_stage1_cost_execution_plan(
        experiment_root / "configs/stage1_cost_execution_v1.json"
    )
    protocol = load_cost_protocol(
        experiment_root / "configs/stage1_cost_v1.json"
    )
    intern_task = _FakeSAMGATask()
    brainrw = _FakeBrainRW()
    calls: list[str] = []
    monkeypatch.setattr(
        cost_runner_module,
        "_load_internvit_runtime_models",
        lambda **kwargs: (
            calls.append("load-internvit")
            or (
                _FakeInternFoundation(),
                intern_task,
                1,
                _fake_collect_block_poolings,
            )
        ),
    )
    monkeypatch.setattr(
        cost_runner_module,
        "_load_brainrw_runtime_model",
        lambda **kwargs: calls.append("load-brainrw") or brainrw,
    )
    monkeypatch.setattr(
        cost_runner_module,
        "_build_synthetic_inputs",
        lambda execution_plan, device: {
            "internvit": {
                "eeg": torch.zeros(4, 2, 2),
                "pixels": torch.zeros(4, 3, 2, 2, dtype=torch.bfloat16),
            },
            "brainrw": {
                "eeg": torch.zeros(4, 2, 2, dtype=torch.bfloat16),
                "pixels": torch.zeros(4, 3, 2, 2, dtype=torch.bfloat16),
            },
        },
    )
    model_manifest = {
        "branches": {
            "internvit": {"parameters": {"representative_subject": 1}},
            "brainrw": {"parameters": {"representative_subject": 1}},
        },
        "raw_model_reference": {
            "branches": {
                "internvit": {"checkpoint_sha256": _digest("intern")},
                "brainrw": {"checkpoint_sha256": _digest("brain")},
            }
        },
    }

    callables = cost_runner_module.build_real_branch_callables(
        protocol=protocol,
        execution_plan=plan,
        model_manifest=model_manifest,
        device=torch.device("cpu"),
        runtime_environment_binding={"schema_version": 1},
    )

    assert tuple(callables) == ("internvit", "brainrw")
    assert calls == ["load-internvit", "load-brainrw"]


def test_brainrw_runtime_checkpoint_binding_rejects_contract_mismatch(
    cost_runner_module: ModuleType,
) -> None:
    environment = {"schema_version": 1, "torch": "2.10.0+cu126"}
    contract = {
        "accelerator": "NVIDIA A40",
        "device_type": "cuda",
        "dtype": "bfloat16",
        "schema_version": 1,
    }
    evidence = {
        "accelerator_name": "NVIDIA A40",
        "bf16_supported": True,
        "schema_version": 1,
    }
    runtime = SimpleNamespace(
        semantic_environment=environment,
        semantic_environment_sha256=sha256_json(environment),
        contract=contract,
        contract_sha256=sha256_json(contract),
        evidence=evidence,
        evidence_sha256=sha256_json(evidence),
    )
    payload = {
        "semantic_environment": environment,
        "semantic_environment_sha256": sha256_json(environment),
        "runtime_contract": contract,
        "runtime_contract_sha256": sha256_json(contract),
        "runtime_evidence": evidence,
        "runtime_evidence_sha256": sha256_json(evidence),
    }
    cost_runner_module._validate_brainrw_runtime_checkpoint(payload, runtime)

    drifted = dict(payload)
    drifted["runtime_contract"] = {**contract, "dtype": "float32"}
    drifted["runtime_contract_sha256"] = sha256_json(
        drifted["runtime_contract"]
    )
    with pytest.raises(ValueError, match="runtime.*contract|contract.*mismatch"):
        cost_runner_module._validate_brainrw_runtime_checkpoint(
            drifted,
            runtime,
        )
