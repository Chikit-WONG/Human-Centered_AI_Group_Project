from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import dataclass, replace as dataclass_replace
from pathlib import Path
from types import MappingProxyType, ModuleType, SimpleNamespace

import numpy as np
import pytest

import train as samga_train
from samga_brain_rw.config import make_run_key
from samga_brain_rw.hashing import canonical_json_bytes, sha256_json
from samga_brain_rw.runtime_contract import (
    PINNED_SEMANTIC_ENVIRONMENT,
    PRODUCTION_RUNTIME_CONTRACT,
    build_environment_binding,
)
from samga_brain_rw.scores import ScoreArtifact


def _h(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _load_runner(experiment_root: Path) -> ModuleType:
    path = experiment_root / "scripts" / "run_training_cell.py"
    spec = importlib.util.spec_from_file_location("run_training_cell", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_training_cell"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runner_module(experiment_root: Path) -> ModuleType:
    return _load_runner(experiment_root)


def _run_key(
    *,
    stage: int = 2,
    config_id: str = "s2-layernorm-on",
    subject: int = 1,
    seed: int = 42,
) -> str:
    return make_run_key(
        f"stage{stage}",
        config_id,
        subject,
        seed,
        _h("config"),
        _h("input"),
    )


def _argv(
    tmp_path: Path,
    *,
    mode: str,
    stage: int = 2,
    max_train_steps: int | None = None,
    input_bundle_sha256: str | None = None,
) -> list[str]:
    config_id = "s2-layernorm-on" if stage == 2 else "internvit_baseline_v1"
    input_sha256 = input_bundle_sha256 or _h("input")
    run_key = make_run_key(
        f"stage{stage}",
        config_id,
        1,
        42,
        _h("config"),
        input_sha256,
    )
    values = [
        "--mode",
        mode,
        "--stage",
        str(stage),
        "--role",
        "candidate" if stage == 2 else "baseline",
        "--subject",
        "1",
        "--seed",
        "42",
        "--resume",
        "none",
        "--config",
        str(tmp_path / "baseline.json"),
        "--manifest",
        str(tmp_path / "sub-01_protocol.json"),
        "--feature-cache",
        str(tmp_path / "features.npy"),
        "--output-dir",
        str(tmp_path / run_key),
        "--project-root",
        str(tmp_path),
        "--config-id",
        config_id,
        "--expected-config-sha256",
        _h("config"),
        "--expected-input-bundle-sha256",
        input_sha256,
        "--run-key",
        run_key,
    ]
    if stage == 2:
        values.extend(
            [
                "--stage2-config",
                str(tmp_path / "stage2.json"),
                "--candidate-id",
                config_id,
            ]
        )
    if max_train_steps is not None:
        values.extend(["--max-train-steps", str(max_train_steps)])
    return values


def test_runner_accepts_only_the_canonical_hashed_run_key_grammar(
    runner_module: ModuleType,
    tmp_path: Path,
) -> None:
    argv = _argv(tmp_path, mode="smoke", max_train_steps=1)
    run_key_position = argv.index("--run-key") + 1
    output_position = argv.index("--output-dir") + 1
    canonical = argv[run_key_position]

    assert "__config-" in canonical
    assert "__inputs-" in canonical
    assert runner_module.parse_arguments(argv).run_key == canonical

    legacy = (
        f"stage2__s2-layernorm-on__sub-01__seed-42__"
        f"{_h('config')}__{_h('input')}"
    )
    legacy_argv = list(argv)
    legacy_argv[run_key_position] = legacy
    legacy_argv[output_position] = str(tmp_path / legacy)

    with pytest.raises(SystemExit):
        runner_module.parse_arguments(legacy_argv)


def test_smoke_requires_positive_max_steps_and_full_forbids_it(
    runner_module: ModuleType,
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit):
        runner_module.parse_arguments(_argv(tmp_path, mode="smoke"))
    with pytest.raises(SystemExit):
        runner_module.parse_arguments(
            _argv(tmp_path, mode="smoke", max_train_steps=0)
        )
    with pytest.raises(SystemExit):
        runner_module.parse_arguments(
            _argv(tmp_path, mode="full", max_train_steps=1)
        )

    smoke = runner_module.parse_arguments(
        _argv(tmp_path, mode="smoke", max_train_steps=1)
    )
    full = runner_module.parse_arguments(_argv(tmp_path, mode="full"))
    assert smoke.max_train_steps == 1
    assert full.max_train_steps is None


def test_runner_device_defaults_to_cuda(
    runner_module: ModuleType,
    tmp_path: Path,
) -> None:
    arguments = runner_module.parse_arguments(
        _argv(tmp_path, mode="smoke", max_train_steps=1)
    )
    assert arguments.device == "cuda"


@pytest.mark.parametrize("unsupported", ("auto", "cpu"))
def test_runner_rejects_non_cuda_devices(
    runner_module: ModuleType,
    tmp_path: Path,
    unsupported: str,
) -> None:
    argv = _argv(tmp_path, mode="smoke", max_train_steps=1)
    argv.extend(["--device", unsupported])
    with pytest.raises(SystemExit):
        runner_module.parse_arguments(argv)


def test_stage_contract_rejects_stage2_inputs_for_stage0_and_requires_them_for_stage2(
    runner_module: ModuleType,
    tmp_path: Path,
) -> None:
    stage0 = _argv(tmp_path, mode="smoke", stage=0, max_train_steps=1)
    stage0.extend(["--stage2-config", "forbidden.json"])
    with pytest.raises(SystemExit):
        runner_module.parse_arguments(stage0)

    stage2 = _argv(tmp_path, mode="smoke", stage=2, max_train_steps=1)
    position = stage2.index("--candidate-id")
    del stage2[position : position + 2]
    with pytest.raises(SystemExit):
        runner_module.parse_arguments(stage2)


def test_stage2_training_command_emits_candidate_id_once(
    runner_module: ModuleType,
    tmp_path: Path,
    experiment_root: Path,
) -> None:
    arguments = runner_module.parse_arguments(
        _argv(tmp_path, mode="smoke", max_train_steps=1)
    )
    arguments.project_root = experiment_root.parents[1]
    command = runner_module._train_command(arguments)
    assert command.count("--candidate-id") == 1
    position = command.index("--candidate-id")
    assert command[position + 1] == arguments.candidate_id


def _training_outputs(runner_module: ModuleType, tmp_path: Path) -> object:
    return runner_module.TrainingOutputs(
        run_manifest_path=tmp_path / "run_manifest.json",
        run_manifest_sha256=_h("run-manifest-file"),
        final_checkpoint_path=tmp_path / "checkpoint_epoch001_step000000001.pt",
        final_checkpoint_sha256=_h("checkpoint"),
        in_loop_metadata_path=tmp_path / "in_loop" / "metadata.json",
        in_loop_metadata_sha256=_h("in-loop-metadata"),
    )


def test_commands_force_explicit_cuda_after_namespace_tampering(
    runner_module: ModuleType,
    tmp_path: Path,
    experiment_root: Path,
) -> None:
    arguments = runner_module.parse_arguments(
        _argv(tmp_path, mode="full")
    )
    arguments.project_root = experiment_root.parents[1]
    arguments.device = "cpu"
    outputs = _training_outputs(runner_module, tmp_path)

    commands = (
        runner_module._train_command(arguments),
        runner_module._evaluation_command(
            arguments,
            outputs,
            "saved_checkpoint",
        ),
    )
    for command in commands:
        assert command.count("--device") == 1
        position = command.index("--device")
        assert command[position + 1] == "cuda"


def test_smoke_runs_training_only_and_never_invokes_official_evaluator(
    runner_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    experiment_root: Path,
) -> None:
    arguments = runner_module.parse_arguments(
        _argv(tmp_path, mode="smoke", max_train_steps=1)
    )
    arguments.project_root = experiment_root.parents[1]
    observed: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        observed.append(list(command))
        assert kwargs["check"] is True
        return SimpleNamespace(returncode=0)

    outputs = _training_outputs(runner_module, tmp_path)
    monkeypatch.setattr(
        runner_module,
        "validate_training_outputs",
        lambda *_args, **_kwargs: outputs,
    )
    monkeypatch.delenv("SAMGA_JOB_MAP", raising=False)

    assert runner_module.run_cell(arguments, subprocess_runner=fake_run) == 0
    assert len(observed) == 1
    command = observed[0]
    assert command[1].endswith("/experiments/samga_brain_rw/train.py")
    assert "--max-train-steps" in command
    assert command[command.index("--max-train-steps") + 1] == "1"
    assert not any("evaluate.py" in value for value in command)
    assert not any("check_baseline_parity.py" in value for value in command)


def test_full_runs_three_evaluations_then_parity_then_completion(
    runner_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    experiment_root: Path,
) -> None:
    arguments = runner_module.parse_arguments(_argv(tmp_path, mode="full"))
    arguments.project_root = experiment_root.parents[1]
    observed: list[list[str]] = []
    outputs = _training_outputs(runner_module, tmp_path)
    parity = tmp_path / "baseline_parity.json"
    parity_sha = _h("parity")

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        observed.append(list(command))
        assert kwargs["check"] is True
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        runner_module,
        "validate_training_outputs",
        lambda *_args, **_kwargs: outputs,
    )
    monkeypatch.setattr(
        runner_module,
        "validate_full_outputs",
        lambda *_args, **_kwargs: (parity, parity_sha),
    )
    monkeypatch.setenv("SAMGA_JOB_MAP", str(tmp_path / "job-map.json"))

    assert runner_module.run_cell(arguments, subprocess_runner=fake_run) == 0
    assert len(observed) == 6
    assert observed[0][1].endswith("/experiments/samga_brain_rw/train.py")
    assert "--max-train-steps" not in observed[0]
    assert all(
        command[1].endswith("/experiments/samga_brain_rw/evaluate.py")
        for command in observed[1:4]
    )
    assert observed[4][1].endswith(
        "/experiments/samga_brain_rw/scripts/check_baseline_parity.py"
    )
    completion = observed[5]
    assert completion[1].endswith(
        "/experiments/samga_brain_rw/scripts/build_job_map.py"
    )
    assert completion[2] == "complete-env"
    hashes = json.loads(completion[completion.index("--output-hashes") + 1])
    assert hashes == {
        "final_checkpoint_sha256": outputs.final_checkpoint_sha256,
        "parity_sha256": parity_sha,
        "run_manifest_sha256": outputs.run_manifest_sha256,
    }


def _production_runtime() -> SimpleNamespace:
    contract = dict(PRODUCTION_RUNTIME_CONTRACT)
    evidence = {
        key: value
        for key, value in contract.items()
        if key
        not in {
            "schema_version",
            "device_type",
            "device",
        }
    }
    evidence.update(cuda_available=True, device_count=1)
    return SimpleNamespace(
        environment_binding=build_environment_binding(
            PINNED_SEMANTIC_ENVIRONMENT,
            contract,
        ),
        contract=contract,
        evidence=evidence,
    )


def _real_run_manifest(
    arguments: object,
) -> dict[str, object]:
    return samga_train.build_run_manifest(
        stage=arguments.stage,
        subject=arguments.subject,
        seed=arguments.seed,
        config_id=arguments.config_id,
        config_sha256=arguments.expected_config_sha256,
        protocol_sha256=_h("protocol"),
        cache_sha256=_h("cache"),
        git_sha="a" * 40,
        upstream_sha=samga_train.PINNED_UPSTREAM_SHA,
        data_order_sha256=_h("order"),
        candidate_spec_sha256=_h("candidate-spec"),
        run_key=arguments.run_key,
    )


def _real_run_summary(
    arguments: object,
    *,
    resume_source_checkpoint_sha256: str | None = None,
) -> dict[str, object]:
    smoke = arguments.mode == "smoke"
    global_step = arguments.max_train_steps if smoke else 60
    checkpoint_name = (
        "checkpoint_epoch001_step00000001.pt"
        if smoke
        else "checkpoint_epoch060.pt"
    )
    checkpoint_hashes = (
        {checkpoint_name: _h(f"checkpoint:{arguments.mode}")}
        if smoke
        else {
            f"checkpoint_epoch{epoch:03d}.pt": _h(
                f"checkpoint:full:{epoch}"
            )
            for epoch in range(51, 61)
        }
    )
    checkpoint_sha256 = checkpoint_hashes[checkpoint_name]
    return {
        **_real_run_manifest(arguments),
        "completed": not smoke,
        "global_step": global_step,
        "final_checkpoint": checkpoint_name,
        "final_checkpoint_sha256": checkpoint_sha256,
        "checkpoint_hashes": checkpoint_hashes,
        "in_loop_score_directory": "in_loop",
        "max_train_steps": arguments.max_train_steps,
        "resume_source_checkpoint_sha256": (
            resume_source_checkpoint_sha256
        ),
        **samga_train._runtime_manifest_metadata(_production_runtime()),
        "top1_rate": 0.1,
        "top5_rate": 0.5,
    }


def _legacy_stage0_run_summary(
    arguments: object,
) -> dict[str, object]:
    summary = _real_run_summary(arguments)
    summary["git_sha"] = (
        "aed25e2e5756cc1f08a859d385ffb116364fa2f9"
    )
    body = {
        key: summary[key]
        for key in _RUN_MANIFEST_BASE_KEYS_FOR_TEST
        if key != "run_manifest_sha256"
    }
    summary["run_manifest_sha256"] = sha256_json(body)
    summary["checkpoint_hashes"] = {
        f"checkpoint_epoch{epoch:03d}.pt": _h(
            f"legacy-checkpoint:{epoch}"
        )
        for epoch in range(1, 61)
    }
    summary["final_checkpoint_sha256"] = summary["checkpoint_hashes"][
        "checkpoint_epoch060.pt"
    ]
    return summary


_RUN_MANIFEST_BASE_KEYS_FOR_TEST = frozenset(
    {
        "schema_version",
        "payload_type",
        "stage",
        "subject",
        "seed",
        "config_id",
        "config_sha256",
        "protocol_sha256",
        "cache_sha256",
        "git_sha",
        "upstream_sha",
        "data_order_sha256",
        "candidate_spec_sha256",
        "run_key",
        "run_manifest_sha256",
    }
)


def test_runner_accepts_exact_real_smoke_and_full_run_summary_shapes(
    runner_module: ModuleType,
    tmp_path: Path,
) -> None:
    smoke = runner_module.parse_arguments(
        _argv(tmp_path, mode="smoke", max_train_steps=1)
    )
    full = runner_module.parse_arguments(_argv(tmp_path, mode="full"))

    for arguments in (smoke, full):
        summary = _real_run_summary(arguments)
        assert (
            runner_module._validate_run_manifest(summary, arguments)
            == summary
        )


def test_run_manifest_accepts_only_exact_sealed_stage0_legacy_retention(
    runner_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    arguments = runner_module.parse_arguments(
        _argv(tmp_path, mode="full", stage=0)
    )
    legacy = _legacy_stage0_run_summary(arguments)
    legacy_file_sha256 = hashlib.sha256(
        canonical_json_bytes(legacy) + b"\n"
    ).hexdigest()

    with pytest.raises(ValueError, match="checkpoint.*retention|51.*60"):
        runner_module._validate_run_manifest(
            legacy,
            arguments,
            run_manifest_file_sha256=legacy_file_sha256,
        )
    monkeypatch.setattr(
        runner_module,
        "_LEGACY_STAGE0_RUN_MANIFEST_FILE_SHA256",
        {arguments.run_key: legacy_file_sha256},
    )
    assert (
        runner_module._validate_run_manifest(
            legacy,
            arguments,
            run_manifest_file_sha256=legacy_file_sha256,
        )
        == legacy
    )

    missing = copy.deepcopy(legacy)
    del missing["checkpoint_hashes"]["checkpoint_epoch001.pt"]
    with pytest.raises(
        ValueError,
        match="legacy|checkpoint.*retention|1.*60|51.*60",
    ):
        runner_module._validate_run_manifest(
            missing,
            arguments,
            run_manifest_file_sha256=legacy_file_sha256,
        )

    nonlegacy = copy.deepcopy(legacy)
    nonlegacy["git_sha"] = "b" * 40
    body = {
        key: nonlegacy[key]
        for key in runner_module._RUN_MANIFEST_BASE_KEYS
        if key != "run_manifest_sha256"
    }
    nonlegacy["run_manifest_sha256"] = sha256_json(body)
    with pytest.raises(ValueError, match="checkpoint.*retention|51.*60"):
        runner_module._validate_run_manifest(nonlegacy, arguments)


@pytest.mark.parametrize(
    "mutation",
    (
        "full_missing_epoch_51",
        "full_extra_epoch_50",
        "smoke_extra_previous",
    ),
)
def test_run_manifest_rejects_noncanonical_checkpoint_retention(
    runner_module: ModuleType,
    tmp_path: Path,
    mutation: str,
) -> None:
    mode = "smoke" if mutation == "smoke_extra_previous" else "full"
    arguments = runner_module.parse_arguments(
        _argv(
            tmp_path,
            mode=mode,
            max_train_steps=1 if mode == "smoke" else None,
        )
    )
    summary = _real_run_summary(arguments)
    hashes = summary["checkpoint_hashes"]
    assert isinstance(hashes, dict)
    if mutation == "full_missing_epoch_51":
        del hashes["checkpoint_epoch051.pt"]
    elif mutation == "full_extra_epoch_50":
        hashes["checkpoint_epoch050.pt"] = _h("extra")
    elif mutation == "smoke_extra_previous":
        hashes["checkpoint_epoch001.pt"] = _h("previous")
    else:
        raise AssertionError("unknown mutation")

    with pytest.raises(ValueError, match="checkpoint.*retention|epochs 51|latest"):
        runner_module._validate_run_manifest(summary, arguments)


@pytest.mark.parametrize("with_transient", (False, True))
def test_smoke_manifest_accepts_late_partial_durable_prefix(
    runner_module: ModuleType,
    tmp_path: Path,
    with_transient: bool,
) -> None:
    arguments = runner_module.parse_arguments(
        _argv(tmp_path, mode="smoke", max_train_steps=525)
    )
    summary = _real_run_summary(arguments)
    hashes = {
        "checkpoint_epoch051.pt": _h("late-partial-51"),
        "checkpoint_epoch052.pt": _h("late-partial-52"),
    }
    final_name = "checkpoint_epoch052.pt"
    if with_transient:
        final_name = "checkpoint_epoch053_step00000525.pt"
        hashes[final_name] = _h("late-partial-53-step-525")
    summary["checkpoint_hashes"] = hashes
    summary["final_checkpoint"] = final_name
    summary["final_checkpoint_sha256"] = hashes[final_name]

    assert (
        runner_module._validate_checkpoint_retention_manifest(
            summary,
            arguments,
        )
        == hashes
    )


def test_smoke_manifest_rejects_gapped_late_partial_durable_retention(
    runner_module: ModuleType,
    tmp_path: Path,
) -> None:
    arguments = runner_module.parse_arguments(
        _argv(tmp_path, mode="smoke", max_train_steps=525)
    )
    summary = _real_run_summary(arguments)
    final_name = "checkpoint_epoch053_step00000525.pt"
    summary["checkpoint_hashes"] = {
        "checkpoint_epoch051.pt": _h("late-partial-51"),
        final_name: _h("late-partial-53-step-525"),
    }
    summary["final_checkpoint"] = final_name
    summary["final_checkpoint_sha256"] = summary["checkpoint_hashes"][
        final_name
    ]

    with pytest.raises(ValueError, match="contiguous|prefix|retention"):
        runner_module._validate_checkpoint_retention_manifest(
            summary,
            arguments,
        )


def _write_validator_checkpoint_bundle(
    output: Path,
    name: str,
) -> str:
    checkpoint = output / name
    checkpoint.write_bytes(f"checkpoint:{name}".encode("utf-8"))
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (output / f"{name}.meta.json").write_bytes(
        canonical_json_bytes(
            {
                "complete": True,
                "payload_sha256": digest,
                "payload_type": "samga_brain_rw.epoch_checkpoint",
                "schema_version": 1,
                "scope": "train",
            }
        )
        + b"\n"
    )
    return digest


def _verified_checkpoint_fixture(
    runner_module: ModuleType,
    arguments: object,
    summary: dict[str, object],
    path: Path,
    *,
    retention: dict[str, object] | None = None,
    run_key: str | None = None,
    payload: dict[str, object] | None = None,
    input_hashes: dict[str, object] | None = None,
) -> SimpleNamespace:
    match = runner_module._CHECKPOINT_NAME_RE.fullmatch(path.name)
    assert match is not None
    epoch = int(match.group("epoch"))
    nested_manifest = {
        key: summary[key]
        for key in runner_module._RUN_MANIFEST_BASE_KEYS
    }
    effective_run_key = run_key or arguments.run_key
    nested_manifest["run_key"] = effective_run_key
    candidate_spec = {
        "candidate_spec_sha256": summary["candidate_spec_sha256"],
        "config_id": arguments.config_id,
        "data_order_sha256": summary["data_order_sha256"],
        "input_bundle_sha256": arguments.expected_input_bundle_sha256,
        "run_key": effective_run_key,
        "trajectory_sha256": _h("trajectory"),
    }
    runtime_state = {
        "epoch_complete": match.group("partial") is None,
        "resume_source_checkpoint_sha256": summary[
            "resume_source_checkpoint_sha256"
        ],
    }
    retention_value = retention or {
        "policy": "retain_exact_epochs_51_through_60",
        "required_epochs": list(range(51, 61)),
        "retain_for_averaging": (
            path.name in runner_module._FULL_RETAINED_CHECKPOINT_NAMES
        ),
    }
    checkpoint_payload = payload if payload is not None else {}
    candidate_value = dict(
        checkpoint_payload.get("candidate_spec", {})
    )
    for key, value in candidate_spec.items():
        candidate_value.setdefault(key, value)
    checkpoint_payload["candidate_spec"] = candidate_value
    checkpoint_payload.setdefault("environment", summary["environment"])
    checkpoint_payload.setdefault("retention", retention_value)
    checkpoint_payload.setdefault("run_manifest", nested_manifest)
    checkpoint_payload.setdefault("runtime_state", runtime_state)
    checkpoint_payload.setdefault("input_hashes", input_hashes or {})
    return SimpleNamespace(
        path=path.resolve(),
        sha256=summary["checkpoint_hashes"][path.name],
        epoch=epoch,
        global_step=(
            summary["global_step"]
            if path.name == summary["final_checkpoint"]
            else epoch * 10
        ),
        subject=arguments.subject,
        seed=arguments.seed,
        config_id=arguments.config_id,
        config_sha256=arguments.expected_config_sha256,
        schedule_sha256=runner_module.SCHEDULE_SHA256,
        optimizer_stage="stage2",
        trajectory_sha256=_h("trajectory"),
        data_order_sha256=summary["data_order_sha256"],
        candidate_spec_sha256=summary["candidate_spec_sha256"],
        input_bundle_sha256=arguments.expected_input_bundle_sha256,
        run_key=effective_run_key,
        payload=MappingProxyType(checkpoint_payload),
        input_hashes=checkpoint_payload["input_hashes"],
        environment=checkpoint_payload["environment"],
        run_manifest=checkpoint_payload["run_manifest"],
        candidate_spec=checkpoint_payload["candidate_spec"],
        runtime_state=checkpoint_payload["runtime_state"],
        retention=checkpoint_payload["retention"],
        model_state_dict={"weight": object()},
    )


@pytest.mark.parametrize("with_transient", (False, True))
def test_smoke_output_validator_accepts_late_partial_durable_prefix(
    runner_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    with_transient: bool,
) -> None:
    arguments = runner_module.parse_arguments(
        _argv(tmp_path, mode="smoke", max_train_steps=525)
    )
    output = Path(arguments.output_dir)
    output.mkdir()
    summary = _real_run_summary(arguments)
    names = [
        "checkpoint_epoch051.pt",
        "checkpoint_epoch052.pt",
    ]
    if with_transient:
        names.append("checkpoint_epoch053_step00000525.pt")
    hashes = {
        name: _write_validator_checkpoint_bundle(output, name)
        for name in names
    }
    summary["checkpoint_hashes"] = hashes
    summary["final_checkpoint"] = names[-1]
    summary["final_checkpoint_sha256"] = hashes[names[-1]]
    monkeypatch.setattr(
        runner_module,
        "verify_epoch_checkpoint",
        lambda path: _verified_checkpoint_fixture(
            runner_module,
            arguments,
            summary,
            path,
        ),
    )

    observed_hashes, final = (
        runner_module._validate_retained_checkpoint_outputs(
            output,
            summary,
            arguments,
        )
    )

    assert observed_hashes == hashes
    assert final.path == (output / names[-1]).resolve()


def test_full_output_validator_requires_exact_last_ten_checkpoint_bundles(
    runner_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    arguments = runner_module.parse_arguments(_argv(tmp_path, mode="full"))
    output = Path(arguments.output_dir)
    output.mkdir()
    summary = _real_run_summary(arguments)
    summary["checkpoint_hashes"] = {
        f"checkpoint_epoch{epoch:03d}.pt": (
            _write_validator_checkpoint_bundle(
                output,
                f"checkpoint_epoch{epoch:03d}.pt",
            )
        )
        for epoch in range(51, 61)
    }
    summary["final_checkpoint_sha256"] = summary["checkpoint_hashes"][
        "checkpoint_epoch060.pt"
    ]
    loaded_epochs: list[int] = []

    def verify_retained(path: Path) -> SimpleNamespace:
        epoch = int(path.name.removeprefix("checkpoint_epoch")[:3])
        loaded_epochs.append(epoch)
        return _verified_checkpoint_fixture(
            runner_module,
            arguments,
            summary,
            path,
        )

    monkeypatch.setattr(
        runner_module,
        "verify_epoch_checkpoint",
        verify_retained,
    )
    stable_bytes = runner_module._stable_regular_bytes_at

    def sidecar_bytes_only(
        directory_fd: int,
        name: str,
        *,
        context: str,
    ) -> bytes:
        assert name.endswith(".meta.json")
        return stable_bytes(directory_fd, name, context=context)

    monkeypatch.setattr(
        runner_module,
        "_stable_regular_bytes_at",
        sidecar_bytes_only,
    )

    runner_module._validate_retained_checkpoint_outputs(
        output,
        summary,
        arguments,
    )
    assert loaded_epochs == list(range(51, 61))

    (output / "checkpoint_epoch051.pt.meta.json").unlink()
    with pytest.raises(ValueError, match="checkpoint.*retention|sidecar|exact"):
        runner_module._validate_retained_checkpoint_outputs(
            output,
            summary,
            arguments,
        )


def test_full_output_validator_rejects_retained_payload_policy_drift(
    runner_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    arguments = runner_module.parse_arguments(_argv(tmp_path, mode="full"))
    output = Path(arguments.output_dir)
    output.mkdir()
    summary = _real_run_summary(arguments)
    summary["checkpoint_hashes"] = {
        f"checkpoint_epoch{epoch:03d}.pt": (
            _write_validator_checkpoint_bundle(
                output,
                f"checkpoint_epoch{epoch:03d}.pt",
            )
        )
        for epoch in range(51, 61)
    }
    summary["final_checkpoint_sha256"] = summary["checkpoint_hashes"][
        "checkpoint_epoch060.pt"
    ]

    def verify_retained(path: Path) -> SimpleNamespace:
        epoch = int(path.name.removeprefix("checkpoint_epoch")[:3])
        return _verified_checkpoint_fixture(
            runner_module,
            arguments,
            summary,
            path,
            retention={
                "policy": "retain_exact_epochs_51_through_60",
                "required_epochs": list(range(51, 61)),
                "retain_for_averaging": epoch != 55,
            },
        )

    monkeypatch.setattr(
        runner_module,
        "verify_epoch_checkpoint",
        verify_retained,
    )

    with pytest.raises(ValueError, match="retention.*epoch|epoch.*retention"):
        runner_module._validate_retained_checkpoint_outputs(
            output,
            summary,
            arguments,
        )


def test_full_output_validator_rejects_mixed_run_checkpoint_window(
    runner_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    arguments = runner_module.parse_arguments(_argv(tmp_path, mode="full"))
    output = Path(arguments.output_dir)
    output.mkdir()
    summary = _real_run_summary(arguments)
    summary["checkpoint_hashes"] = {
        f"checkpoint_epoch{epoch:03d}.pt": (
            _write_validator_checkpoint_bundle(
                output,
                f"checkpoint_epoch{epoch:03d}.pt",
            )
        )
        for epoch in range(51, 61)
    }
    summary["final_checkpoint_sha256"] = summary["checkpoint_hashes"][
        "checkpoint_epoch060.pt"
    ]

    def verify_retained(path: Path) -> SimpleNamespace:
        epoch = int(path.name.removeprefix("checkpoint_epoch")[:3])
        run_key = arguments.run_key if epoch != 55 else "mixed-run"
        return _verified_checkpoint_fixture(
            runner_module,
            arguments,
            summary,
            path,
            run_key=run_key,
        )

    monkeypatch.setattr(
        runner_module,
        "verify_epoch_checkpoint",
        verify_retained,
    )

    with pytest.raises(ValueError, match="run.?key|mixed"):
        runner_module._validate_retained_checkpoint_outputs(
            output,
            summary,
            arguments,
        )


def test_smoke_output_validator_rejects_previous_transient_bundle(
    runner_module: ModuleType,
    tmp_path: Path,
) -> None:
    arguments = runner_module.parse_arguments(
        _argv(tmp_path, mode="smoke", max_train_steps=1)
    )
    output = Path(arguments.output_dir)
    output.mkdir()
    summary = _real_run_summary(arguments)
    final_name = str(summary["final_checkpoint"])
    final_digest = _write_validator_checkpoint_bundle(
        output,
        final_name,
    )
    summary["final_checkpoint_sha256"] = final_digest
    summary["checkpoint_hashes"] = {final_name: final_digest}
    previous_name = "checkpoint_epoch001.pt"
    _write_validator_checkpoint_bundle(output, previous_name)

    with pytest.raises(ValueError, match="checkpoint.*retention|latest|exact"):
        runner_module._validate_retained_checkpoint_outputs(
            output,
            summary,
            arguments,
        )


def test_output_validator_rejects_symlinked_checkpoint_sidecar(
    runner_module: ModuleType,
    tmp_path: Path,
) -> None:
    arguments = runner_module.parse_arguments(
        _argv(tmp_path, mode="smoke", max_train_steps=1)
    )
    output = Path(arguments.output_dir)
    output.mkdir()
    summary = _real_run_summary(arguments)
    final_name = str(summary["final_checkpoint"])
    final_digest = _write_validator_checkpoint_bundle(
        output,
        final_name,
    )
    summary["final_checkpoint_sha256"] = final_digest
    summary["checkpoint_hashes"] = {final_name: final_digest}
    sidecar = output / f"{final_name}.meta.json"
    target = output / "sidecar-target.json"
    sidecar.replace(target)
    sidecar.symlink_to(target)

    with pytest.raises(ValueError, match="sidecar|symlink|regular|safe"):
        runner_module._validate_retained_checkpoint_outputs(
            output,
            summary,
            arguments,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("old_payload_type", "identity"),
        ("unknown_summary_field", "keys"),
        ("missing_runtime_evidence", "keys"),
        ("invalid_resume_source", "resume_source"),
        ("environment_hash", "runtime contract hash"),
        ("runtime_contract", "runtime_contract.*environment"),
        ("runtime_contract_hash", "runtime contract hash"),
        ("semantic_environment_hash", "semantic environment hash"),
        ("runtime_evidence_field", "runtime_evidence.*keys"),
        ("runtime_evidence_value", "runtime evidence.*accelerator_name"),
    ],
)
def test_runner_rejects_noncanonical_run_summary_runtime_identity(
    runner_module: ModuleType,
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    arguments = runner_module.parse_arguments(
        _argv(tmp_path, mode="smoke", max_train_steps=1)
    )
    summary = copy.deepcopy(_real_run_summary(arguments))
    if mutation == "old_payload_type":
        summary["payload_type"] = "samga_brain_rw.run_manifest"
        body = {
            key: summary[key]
            for key in runner_module._RUN_MANIFEST_BASE_KEYS
            if key != "run_manifest_sha256"
        }
        summary["run_manifest_sha256"] = sha256_json(body)
    elif mutation == "unknown_summary_field":
        summary["unknown"] = "forbidden"
    elif mutation == "missing_runtime_evidence":
        del summary["runtime_evidence"]
    elif mutation == "invalid_resume_source":
        summary["resume_source_checkpoint_sha256"] = "not-a-sha"
    elif mutation == "environment_hash":
        environment = summary["environment"]
        assert isinstance(environment, dict)
        environment["runtime_contract_sha256"] = _h("wrong-contract")
    elif mutation == "runtime_contract":
        contract = summary["runtime_contract"]
        assert isinstance(contract, dict)
        contract["device"] = "cuda:1"
    elif mutation == "runtime_contract_hash":
        summary["runtime_contract_sha256"] = _h("wrong-contract")
    elif mutation == "semantic_environment_hash":
        summary["semantic_environment_sha256"] = _h("wrong-environment")
    elif mutation == "runtime_evidence_field":
        evidence = summary["runtime_evidence"]
        assert isinstance(evidence, dict)
        evidence["unknown"] = "forbidden"
    elif mutation == "runtime_evidence_value":
        evidence = summary["runtime_evidence"]
        assert isinstance(evidence, dict)
        evidence["accelerator_name"] = "forged accelerator"
    else:
        raise AssertionError("unknown mutation")

    with pytest.raises(ValueError, match=message):
        runner_module._validate_run_manifest(summary, arguments)


@pytest.mark.parametrize(
    "resume_source",
    [None, _h("resume-source")],
)
def test_runner_accepts_only_null_or_sha_resume_source(
    runner_module: ModuleType,
    tmp_path: Path,
    resume_source: str | None,
) -> None:
    arguments = runner_module.parse_arguments(
        _argv(tmp_path, mode="smoke", max_train_steps=1)
    )
    summary = _real_run_summary(
        arguments,
        resume_source_checkpoint_sha256=resume_source,
    )
    assert runner_module._validate_run_manifest(
        summary,
        arguments,
    )["resume_source_checkpoint_sha256"] == resume_source


def _input_hashes() -> dict[str, object]:
    return {
        "manifest_sha256": _h("manifest"),
        "records_sha256": _h("records"),
        "source_manifest_sha256": _h("source-manifest"),
        "source_payload_byte_count_sha256": sha256_json(123),
        "source_payload_path_sha256": sha256_json(
            "/development/sub-01/train.pt"
        ),
        "source_payload_sha256": _h("source-payload"),
        "val_dev_role_sha256": _h("val-dev-role"),
    }


def _partial_score_artifact(
    arguments: object,
    run_manifest: dict[str, object],
    checkpoint_sha256: str,
    input_hashes: dict[str, object],
) -> SimpleNamespace:
    metadata = {
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": arguments.expected_config_sha256,
        "git_sha": run_manifest["git_sha"],
        "global_step": arguments.max_train_steps,
        "planned_steps": 120,
        "protocol_sha256": run_manifest["protocol_sha256"],
        "seed": arguments.seed,
        "source_records": [
            {
                "manifest_sha256": input_hashes["manifest_sha256"],
                "records_sha256": input_hashes["records_sha256"],
                "role": "val-dev",
                "role_payload_sha256": input_hashes["val_dev_role_sha256"],
                "run_key": arguments.run_key,
                "source_manifest_sha256": input_hashes[
                    "source_manifest_sha256"
                ],
                "source_payload_byte_count": 123,
                "source_payload_path": "/development/sub-01/train.pt",
                "source_payload_sha256": input_hashes[
                    "source_payload_sha256"
                ],
            }
        ],
        "split_role": "val-dev",
        "stage": "training_smoke/in_loop",
        "subject": arguments.subject,
        "training_complete": False,
    }
    frozen = MappingProxyType(metadata)
    return SimpleNamespace(metadata=frozen, provenance=frozen)


def test_checkpoint_identity_validator_runs_before_subset_crossbinding(
    runner_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    arguments = runner_module.parse_arguments(
        _argv(tmp_path, mode="smoke", max_train_steps=1)
    )
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"transport-valid-checkpoint")
    checkpoint_sha256 = hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()
    run_manifest = _real_run_summary(arguments)
    run_manifest["final_checkpoint_sha256"] = checkpoint_sha256
    calls: list[Path] = []

    def reject_identity(path: Path) -> None:
        calls.append(path)
        raise ValueError("scientific identity drift")

    monkeypatch.setattr(
        runner_module,
        "verify_epoch_checkpoint",
        reject_identity,
    )

    with pytest.raises(ValueError, match="scientific identity drift"):
        runner_module._validate_checkpoint(
            checkpoint,
            run_manifest,
            arguments,
        )
    assert calls == [checkpoint]


def test_checkpoint_transport_hash_mismatch_precedes_identity_validation(
    runner_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    arguments = runner_module.parse_arguments(
        _argv(tmp_path, mode="smoke", max_train_steps=1)
    )
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    actual_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    run_manifest = _real_run_summary(arguments)
    run_manifest["final_checkpoint_sha256"] = actual_sha256
    monkeypatch.setattr(
        runner_module,
        "verify_epoch_checkpoint",
        lambda path: SimpleNamespace(
            path=path.resolve(),
            sha256=_h("wrong-transport"),
        ),
    )

    with pytest.raises(ValueError, match="typed checkpoint hash"):
        runner_module._validate_checkpoint(
            checkpoint,
            run_manifest,
            arguments,
        )


def test_training_output_validation_binds_partial_checkpoint_and_run_identity(
    runner_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_hashes = _input_hashes()
    input_bundle_sha256 = sha256_json(dict(sorted(input_hashes.items())))
    arguments = runner_module.parse_arguments(
        _argv(
            tmp_path,
            mode="smoke",
            max_train_steps=1,
            input_bundle_sha256=input_bundle_sha256,
        )
    )
    output = Path(arguments.output_dir)
    output.mkdir()
    in_loop = output / "in_loop"
    in_loop.mkdir()
    metadata = in_loop / "metadata.json"
    metadata.write_bytes(b'{"complete":true}\n')
    checkpoint = output / "checkpoint_epoch001_step00000001.pt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (output / f"{checkpoint.name}.meta.json").write_bytes(
        canonical_json_bytes(
            {
                "complete": True,
                "payload_sha256": checkpoint_sha,
                "payload_type": "samga_brain_rw.epoch_checkpoint",
                "schema_version": 1,
                "scope": "train",
            }
        )
        + b"\n"
    )
    base_manifest = _real_run_manifest(arguments)
    run_manifest = {
        **base_manifest,
        "completed": False,
        "global_step": 1,
        "final_checkpoint": checkpoint.name,
        "final_checkpoint_sha256": checkpoint_sha,
        "checkpoint_hashes": {checkpoint.name: checkpoint_sha},
        "in_loop_score_directory": "in_loop",
        "max_train_steps": 1,
        "resume_source_checkpoint_sha256": None,
        **samga_train._runtime_manifest_metadata(_production_runtime()),
        "top1_rate": 0.1,
        "top5_rate": 0.5,
    }
    manifest_path = output / "run_manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(run_manifest) + b"\n")
    candidate_spec = {
        "config_id": "s2-layernorm-on",
        "input_bundle_sha256": input_bundle_sha256,
        "run_key": arguments.run_key,
    }
    checkpoint_payload = {
        "payload_type": "samga_brain_rw.epoch_checkpoint",
        "subject": 1,
        "seed": 42,
        "config_sha256": _h("config"),
        "epoch": 1,
        "global_step": 1,
        "environment": run_manifest["environment"],
        "runtime_state": {
            "epoch_complete": False,
            "resume_source_checkpoint_sha256": None,
        },
        "candidate_spec": candidate_spec,
        "run_manifest": base_manifest,
        "input_hashes": input_hashes,
    }
    verifier_calls: list[Path] = []

    def verify_checkpoint(path: Path) -> SimpleNamespace:
        verifier_calls.append(path)
        return _verified_checkpoint_fixture(
            runner_module,
            arguments,
            run_manifest,
            path,
            payload=checkpoint_payload,
            input_hashes=input_hashes,
        )

    monkeypatch.setattr(
        runner_module,
        "verify_epoch_checkpoint",
        verify_checkpoint,
    )
    score_artifact = _partial_score_artifact(
        arguments,
        run_manifest,
        checkpoint_sha,
        input_hashes,
    )
    monkeypatch.setattr(
        runner_module.ScoreArtifact,
        "load",
        lambda *_args, **_kwargs: score_artifact,
    )

    validated = runner_module.validate_training_outputs(arguments)
    assert validated.final_checkpoint_sha256 == checkpoint_sha
    assert validated.run_manifest_sha256 == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    assert validated.in_loop_metadata_sha256 == hashlib.sha256(
        metadata.read_bytes()
    ).hexdigest()
    assert verifier_calls == [checkpoint]

    checkpoint_payload["runtime_state"] = {"epoch_complete": True}
    with pytest.raises(ValueError, match="partial|epoch_complete"):
        runner_module.validate_training_outputs(arguments)


def test_training_command_output_adapter_parses_sealed_argv_and_translates_exit(
    runner_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    outputs = _training_outputs(runner_module, tmp_path)
    monkeypatch.setattr(
        runner_module,
        "validate_training_outputs",
        lambda arguments: (
            outputs
            if arguments.run_key == _run_key()
            else pytest.fail("adapter parsed the wrong arguments")
        ),
    )
    sealed = [
        "python",
        str(
            tmp_path
            / "experiments/samga_brain_rw/scripts/run_training_cell.py"
        ),
        *_argv(tmp_path, mode="smoke", max_train_steps=1),
    ]

    assert runner_module.validate_training_command_outputs(
        sealed,
        expected_mode="smoke",
    ) == outputs
    with pytest.raises(ValueError, match="mode"):
        runner_module.validate_training_command_outputs(
            sealed,
            expected_mode="full",
        )

    invalid = list(sealed)
    position = invalid.index("--max-train-steps")
    del invalid[position : position + 2]
    with pytest.raises(ValueError, match="argument|parse|smoke|max"):
        runner_module.validate_training_command_outputs(invalid)
    with pytest.raises(ValueError, match="sealed|runner|argv"):
        runner_module.validate_training_command_outputs(
            ["python", "different.py", *sealed[2:]]
        )
    with pytest.raises(ValueError, match="runner|project.root|argv"):
        runner_module.validate_training_command_outputs(
            [
                "python",
                str(tmp_path / "elsewhere" / "run_training_cell.py"),
                *sealed[2:],
            ]
        )


def test_public_training_command_proof_is_one_frozen_full_capture_by_default(
    runner_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    @dataclass(frozen=True)
    class FakeProof:
        outputs: object
        checkpoint: object
        in_loop_score: object
        terminal_score: object | None = None
        completion_output_hashes: object = MappingProxyType({})
        sealed_argv: tuple[str, ...] = ()

    arguments = runner_module.parse_arguments(_argv(tmp_path, mode="full"))
    training_proof = FakeProof(
        outputs=_training_outputs(runner_module, tmp_path),
        checkpoint=SimpleNamespace(label="typed-checkpoint"),
        in_loop_score=SimpleNamespace(label="typed-in-loop-score"),
    )
    final_proof = FakeProof(
        outputs=training_proof.outputs,
        checkpoint=training_proof.checkpoint,
        in_loop_score=training_proof.in_loop_score,
        terminal_score=SimpleNamespace(label="typed-terminal-score"),
            completion_output_hashes=MappingProxyType(
                {
                    "final_checkpoint_sha256": (
                        training_proof.outputs.final_checkpoint_sha256
                    ),
                    "parity_sha256": _h("parity"),
                    "run_manifest_sha256": (
                        training_proof.outputs.run_manifest_sha256
                    ),
                }
            ),
    )
    captures: list[object] = []
    monkeypatch.setattr(
        runner_module,
        "_capture_training_run_proof",
        lambda actual, **kwargs: (
            captures.append(actual) or training_proof
            if kwargs.get("verify_static_config") is True
            and tuple(kwargs.get("sealed_argv", ())) == tuple(command)
            else pytest.fail("public proof did not request static config proof")
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runner_module,
        "_validate_full_training_proof",
        lambda actual, proof: (
            final_proof
            if actual.run_key == arguments.run_key
            and proof is training_proof
            else pytest.fail("full proof switched its training capture")
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runner_module,
        "ValidatedTrainingRunProof",
        FakeProof,
        raising=False,
    )
    monkeypatch.setattr(
        runner_module,
        "ScoreArtifact",
        SimpleNamespace,
    )
    monkeypatch.setattr(
        runner_module,
        "VerifiedEpochCheckpoint",
        SimpleNamespace,
    )
    command = [
        "python",
        str(
            tmp_path
            / "experiments/samga_brain_rw/scripts/run_training_cell.py"
        ),
        *_argv(tmp_path, mode="full"),
    ]

    proof = runner_module.validate_training_command_proof(command)

    assert proof is final_proof
    assert captures == [arguments]
    invalid_checkpoint_proof = dataclass_replace(
        final_proof,
        checkpoint=object(),
    )
    monkeypatch.setattr(
        runner_module,
        "_validate_full_training_proof",
        lambda _actual, _proof: invalid_checkpoint_proof,
    )
    with pytest.raises(TypeError, match="checkpoint|typed"):
        runner_module.validate_training_command_proof(command)
    with pytest.raises(ValueError, match="mode"):
        runner_module.validate_training_command_proof(
            command,
            expected_mode="smoke",
        )


def test_public_training_command_proof_rejects_restricted_scope_before_capture(
    runner_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    command = [
        "python",
        str(
            tmp_path
            / "experiments/samga_brain_rw/scripts/run_training_cell.py"
        ),
        *_argv(tmp_path, mode="full"),
    ]
    command[command.index("--manifest") + 1] = str(
        tmp_path / "formal-test" / "sub-01_protocol.json"
    )
    monkeypatch.setattr(
        runner_module,
        "_capture_training_run_proof",
        lambda *_args, **_kwargs: pytest.fail(
            "restricted scope reached training artifact capture"
        ),
        raising=False,
    )

    with pytest.raises(PermissionError, match="scope|development|formal"):
        runner_module.validate_training_command_proof(command)


def test_training_proof_thaws_frozen_score_source_records_for_canonical_hashing(
    runner_module: ModuleType,
) -> None:
    frozen = (
        MappingProxyType(
            {
                "role": "val-dev",
                "nested": MappingProxyType({"count": 200}),
            }
        ),
    )

    assert runner_module._thaw_json(frozen) == [
        {"nested": {"count": 200}, "role": "val-dev"}
    ]
    assert sha256_json(runner_module._thaw_json(frozen)) == sha256_json(
        [{"nested": {"count": 200}, "role": "val-dev"}]
    )


_PARITY_ROLES = {
    "in_loop": "in_loop",
    "saved_checkpoint": "saved_checkpoint",
    "repeat_emission": "repeat_emission",
    "reload_evaluation": "reload_evaluation",
}


def _load_parity_module(experiment_root: Path) -> ModuleType:
    path = experiment_root / "scripts" / "check_baseline_parity.py"
    spec = importlib.util.spec_from_file_location(
        "run_training_cell_parity_fixture",
        path,
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _real_parity_fixture(
    experiment_root: Path,
    tmp_path: Path,
) -> tuple[Path, dict[str, object], dict[str, ScoreArtifact]]:
    run_directory = (tmp_path / "typed-parity-run").resolve()
    query_ids = ["query-b", "query-a"]
    gallery_ids = [
        "query-a",
        "zeta",
        "query-b",
        "alpha",
        "other-1",
        "other-2",
    ]
    base = np.array(
        [
            [0.1, 0.9, 0.8, 0.9, 0.0, -1.0],
            [1.0, 0.2, 0.1, 0.3, 0.4, 0.5],
        ],
        dtype=np.float32,
    )
    metadata = {
        "checkpoint_sha256": _h("parity-checkpoint"),
        "config_sha256": _h("parity-config"),
        "git_sha": "a" * 40,
        "protocol_sha256": _h("parity-protocol"),
        "seed": 42,
        "source_records": [
            {"record_id": "query-b"},
            {"record_id": "query-a"},
        ],
        "split_role": "val-dev",
        "stage": "stage0",
        "subject": 1,
    }
    matrices = {
        role: base.copy()
        for role in _PARITY_ROLES
    }
    matrices["saved_checkpoint"][0, 0] += np.float32(2e-7)
    matrices["repeat_emission"][1, 1] -= np.float32(3e-7)
    matrices["reload_evaluation"][0, 4] += np.float32(4e-7)
    for role, directory in _PARITY_ROLES.items():
        ScoreArtifact.save(
            run_directory / directory,
            matrices[role],
            query_ids,
            gallery_ids,
            metadata,
        )
    parity_module = _load_parity_module(experiment_root)
    report = parity_module.build_baseline_parity_report(
        run_directory,
        scope="val-dev",
    )
    artifacts = {
        role: ScoreArtifact.load(
            run_directory / directory,
            {"val-dev"},
        )
        for role, directory in _PARITY_ROLES.items()
    }
    return run_directory, report, artifacts


def test_full_parity_validator_recomputes_all_four_typed_bundles(
    runner_module: ModuleType,
    experiment_root: Path,
    tmp_path: Path,
) -> None:
    run_directory, report, artifacts = _real_parity_fixture(
        experiment_root,
        tmp_path,
    )

    runner_module._validate_parity_report_against_artifacts(
        report,
        output_dir=run_directory,
        artifacts=artifacts,
    )


def test_full_parity_validator_rejects_same_byte_payload_replacement(
    runner_module: ModuleType,
    experiment_root: Path,
    tmp_path: Path,
) -> None:
    run_directory, report, artifacts = _real_parity_fixture(
        experiment_root,
        tmp_path,
    )
    payload = run_directory / "repeat_emission" / "similarity.npy"
    replacement = payload.with_name("similarity.replacement.npy")
    replacement.write_bytes(payload.read_bytes())
    os.replace(replacement, payload)

    with pytest.raises(ValueError, match="identity|changed|parity"):
        runner_module._validate_parity_report_against_artifacts(
            report,
            output_dir=run_directory,
            artifacts=artifacts,
        )


def test_full_parity_validator_rejects_root_swap_after_validation(
    runner_module: ModuleType,
    experiment_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_directory, report, artifacts = _real_parity_fixture(
        experiment_root,
        tmp_path,
    )
    original_sha256_json = runner_module.sha256_json
    swapped = False

    def swap_before_summary_hash(value: object) -> str:
        nonlocal swapped
        if isinstance(value, dict) and "checkpoint_sha256" in value:
            old_directory = run_directory.with_name(
                f"{run_directory.name}.replaced"
            )
            run_directory.rename(old_directory)
            run_directory.mkdir()
            swapped = True
        return original_sha256_json(value)

    monkeypatch.setattr(
        runner_module,
        "sha256_json",
        swap_before_summary_hash,
    )

    with pytest.raises(ValueError, match="directory.*identity|changed"):
        runner_module._validate_parity_report_against_artifacts(
            report,
            output_dir=run_directory,
            artifacts=artifacts,
        )
    assert swapped is True


@pytest.mark.parametrize(
    "mutation",
    (
        "root_extra",
        "directory_identity",
        "file_hash",
        "pair_identity",
        "pair_nan",
        "pair_negative",
        "matrix_after_report",
    ),
)
def test_full_parity_validator_rejects_forged_report_or_artifact(
    runner_module: ModuleType,
    experiment_root: Path,
    tmp_path: Path,
    mutation: str,
) -> None:
    run_directory, report, artifacts = _real_parity_fixture(
        experiment_root,
        tmp_path,
    )
    forged = copy.deepcopy(report)
    if mutation == "root_extra":
        forged["trusted"] = True
    elif mutation == "directory_identity":
        forged["run_directory_identity"]["inode"] += 1
    elif mutation == "file_hash":
        forged["artifacts"]["repeat_emission"]["files"][
            "similarity.npy"
        ]["sha256"] = _h("forged-payload")
    elif mutation == "pair_identity":
        forged["comparisons"][0]["right"] = "reload_evaluation"
    elif mutation == "pair_nan":
        forged["comparisons"][0][
            "max_absolute_score_difference"
        ] = float("nan")
    elif mutation == "pair_negative":
        forged["comparisons"][0][
            "max_absolute_score_difference"
        ] = -1.0
    elif mutation == "matrix_after_report":
        changed = artifacts["repeat_emission"].similarity.copy()
        changed[0, 0] += np.float32(2e-4)
        artifacts["repeat_emission"] = dataclass_replace(
            artifacts["repeat_emission"],
            similarity=changed,
        )
    else:  # pragma: no cover - parametrization invariant
        raise AssertionError(mutation)

    with pytest.raises(
        (TypeError, ValueError),
        match="parity|comparison|identity|hash|score|schema|finite",
    ):
        runner_module._validate_parity_report_against_artifacts(
            forged,
            output_dir=run_directory,
            artifacts=artifacts,
        )


def test_full_proof_loads_exact_four_score_roles_once(
    runner_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    @dataclass(frozen=True)
    class FakeProof:
        output_dir: Path
        outputs: object
        in_loop_score: object
        completion_output_hashes: object = MappingProxyType({})
        terminal_score: object | None = None
        parity_artifacts: object = MappingProxyType({})
        parity_report: object | None = None
        parity_report_bytes: bytes | None = None
        parity_sha256: str | None = None

    class FakeScore:
        loaded: list[Path] = []

        def __init__(self, role: str) -> None:
            self.role = role

        @classmethod
        def load(
            cls,
            directory: Path,
            _scopes: set[str],
        ) -> "FakeScore":
            cls.loaded.append(Path(directory))
            return cls(Path(directory).name)

    output = tmp_path / "full-proof"
    output.mkdir()
    parity = {"fixture": True}
    (output / "baseline_parity.json").write_bytes(
        canonical_json_bytes(parity) + b"\n"
    )
    proof = FakeProof(
        output_dir=output,
        outputs=SimpleNamespace(
            final_checkpoint_sha256=_h("checkpoint"),
            run_manifest_sha256=_h("manifest"),
        ),
        in_loop_score=FakeScore("in_loop"),
    )
    arguments = SimpleNamespace(mode="full")
    observed: list[dict[str, FakeScore]] = []
    monkeypatch.setattr(
        runner_module,
        "ValidatedTrainingRunProof",
        FakeProof,
    )
    monkeypatch.setattr(runner_module, "ScoreArtifact", FakeScore)
    monkeypatch.setattr(
        runner_module,
        "_validate_terminal_score_against_proof",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runner_module,
        "_validate_parity_report_against_artifacts",
        lambda _report, *, output_dir, artifacts: (
            observed.append(dict(artifacts))
            if output_dir == output
            else pytest.fail("full proof changed output directory")
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runner_module,
        "_validate_parity_report_against_score",
        lambda *_args, **_kwargs: pytest.fail(
            "full proof trusted only saved_checkpoint"
        ),
        raising=False,
    )

    validated = runner_module._validate_full_training_proof(
        arguments,
        proof,
    )

    expected_paths = [
        output / directory
        for role, directory in _PARITY_ROLES.items()
        if role != "in_loop"
    ]
    assert FakeScore.loaded == expected_paths
    assert set(observed[0]) == set(_PARITY_ROLES)
    assert dict(validated.parity_artifacts) == observed[0]
    assert validated.terminal_score is observed[0]["saved_checkpoint"]


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("checkpoint_sha256", _h("wrong-checkpoint"), "checkpoint"),
        ("config_sha256", _h("wrong-config"), "config"),
        ("git_sha", "b" * 40, "git"),
        ("protocol_sha256", _h("wrong-protocol"), "protocol"),
        ("subject", 5, "subject"),
        ("seed", 43, "seed"),
        ("stage", "stage2", "stage"),
        ("training_complete", True, "training_complete"),
        ("global_step", 2, "global_step"),
        ("planned_steps", 1, "planned_steps"),
        ("source_run_key", "different-run", "run_key"),
        ("source_manifest_sha256", _h("wrong-manifest"), "manifest"),
        ("source_role_sha256", _h("wrong-role"), "role|input"),
    ],
)
def test_training_output_validation_rejects_score_identity_mismatch(
    runner_module: ModuleType,
    tmp_path: Path,
    field: str,
    replacement: object,
    message: str,
) -> None:
    input_hashes = _input_hashes()
    input_bundle_sha256 = sha256_json(dict(sorted(input_hashes.items())))
    arguments = runner_module.parse_arguments(
        _argv(
            tmp_path,
            mode="smoke",
            max_train_steps=1,
            input_bundle_sha256=input_bundle_sha256,
        )
    )
    run_manifest = _real_run_summary(arguments)
    checkpoint_sha256 = str(run_manifest["final_checkpoint_sha256"])
    artifact = _partial_score_artifact(
        arguments,
        run_manifest,
        checkpoint_sha256,
        input_hashes,
    )
    metadata = dict(artifact.metadata)
    records = [
        dict(record)
        for record in metadata["source_records"]
    ]
    if field == "source_run_key":
        records[0]["run_key"] = replacement
    elif field == "source_manifest_sha256":
        records[0]["manifest_sha256"] = replacement
    elif field == "source_role_sha256":
        records[0]["role_payload_sha256"] = replacement
    else:
        metadata[field] = replacement
    metadata["source_records"] = records
    mutated = SimpleNamespace(
        metadata=MappingProxyType(metadata),
        provenance=artifact.provenance,
    )

    with pytest.raises(ValueError, match=message):
        runner_module._validate_in_loop_score_artifact(
            mutated,
            run_manifest=run_manifest,
            checkpoint_payload={"input_hashes": input_hashes},
            final_checkpoint_sha256=checkpoint_sha256,
            arguments=arguments,
        )


def test_runner_rejects_sealed_scope_paths_before_subprocess(
    runner_module: ModuleType,
    tmp_path: Path,
) -> None:
    argv = _argv(tmp_path, mode="smoke", max_train_steps=1)
    argv[argv.index("--manifest") + 1] = str(tmp_path / "val-confirm" / "x.json")
    arguments = runner_module.parse_arguments(argv)
    with pytest.raises((PermissionError, ValueError), match="sealed|development"):
        runner_module.run_cell(
            arguments,
            subprocess_runner=lambda *_args, **_kwargs: pytest.fail(
                "subprocess must not run"
            ),
        )
