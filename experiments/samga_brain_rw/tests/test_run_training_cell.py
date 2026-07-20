from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import MappingProxyType, ModuleType, SimpleNamespace

import pytest

import train as samga_train
from samga_brain_rw.config import make_run_key
from samga_brain_rw.hashing import canonical_json_bytes, sha256_json
from samga_brain_rw.runtime_contract import (
    PINNED_SEMANTIC_ENVIRONMENT,
    PRODUCTION_RUNTIME_CONTRACT,
    build_environment_binding,
)


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
    checkpoint_sha256 = _h(f"checkpoint:{arguments.mode}")
    return {
        **_real_run_manifest(arguments),
        "completed": not smoke,
        "global_step": global_step,
        "final_checkpoint": checkpoint_name,
        "final_checkpoint_sha256": checkpoint_sha256,
        "checkpoint_hashes": {
            checkpoint_name: checkpoint_sha256,
        },
        "in_loop_score_directory": "in_loop",
        "max_train_steps": arguments.max_train_steps,
        "resume_source_checkpoint_sha256": (
            resume_source_checkpoint_sha256
        ),
        **samga_train._runtime_manifest_metadata(_production_runtime()),
        "top1_rate": 0.1,
        "top5_rate": 0.5,
    }


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
    payload = MappingProxyType({"scientific_identity": "drifted"})
    envelope = MappingProxyType({"sidecar": "drifted"})
    calls: list[tuple[object, object]] = []

    monkeypatch.setattr(
        runner_module,
        "load_typed_torch_checkpoint",
        lambda *_args, **_kwargs: SimpleNamespace(
            payload=payload,
            envelope=envelope,
            sha256=checkpoint_sha256,
        ),
    )

    def reject_identity(
        candidate_payload: object,
        candidate_envelope: object,
    ) -> None:
        calls.append((candidate_payload, candidate_envelope))
        raise ValueError("scientific identity drift")

    monkeypatch.setattr(
        runner_module,
        "validate_epoch_checkpoint_identity",
        reject_identity,
        raising=False,
    )

    with pytest.raises(ValueError, match="scientific identity drift"):
        runner_module._validate_checkpoint(
            checkpoint,
            run_manifest,
            arguments,
        )
    assert calls == [
        ({"scientific_identity": "drifted"}, envelope)
    ]


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
        "load_typed_torch_checkpoint",
        lambda *_args, **_kwargs: SimpleNamespace(
            payload={},
            envelope={},
            sha256=_h("wrong-transport"),
        ),
    )
    monkeypatch.setattr(
        runner_module,
        "validate_epoch_checkpoint_identity",
        lambda *_args: pytest.fail(
            "identity validation must follow transport hash validation"
        ),
        raising=False,
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
    checkpoint = output / "checkpoint_epoch001_step000000001.pt"
    checkpoint.write_bytes(b"checkpoint")
    (output / f"{checkpoint.name}.meta.json").write_bytes(b"{}\n")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
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
    checkpoint_envelope = MappingProxyType(
        {"scientific_sidecar": "fixture"}
    )
    monkeypatch.setattr(
        runner_module,
        "load_typed_torch_checkpoint",
        lambda *_args, **_kwargs: SimpleNamespace(
            payload=MappingProxyType(checkpoint_payload),
            envelope=checkpoint_envelope,
            sha256=checkpoint_sha,
        ),
    )
    identity_calls: list[tuple[object, object]] = []

    def record_identity(
        payload: object,
        envelope: object,
    ) -> None:
        identity_calls.append((payload, envelope))

    monkeypatch.setattr(
        runner_module,
        "validate_epoch_checkpoint_identity",
        record_identity,
        raising=False,
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
    assert identity_calls == [
        (checkpoint_payload, checkpoint_envelope)
    ]

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
