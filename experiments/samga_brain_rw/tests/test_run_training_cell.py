from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import MappingProxyType, ModuleType, SimpleNamespace

import pytest

from samga_brain_rw.hashing import canonical_json_bytes, sha256_json


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
    return (
        f"stage{stage}__{config_id}__sub-{subject:02d}__seed-{seed}__"
        f"{_h('config')}__{_h('input')}"
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
    run_key = (
        f"stage{stage}__{config_id}__sub-01__seed-42__"
        f"{_h('config')}__{input_sha256}"
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


def test_training_output_validation_binds_partial_checkpoint_and_run_identity(
    runner_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_hashes = {"only": _h("irrelevant")}
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
    body = {
        "schema_version": 1,
        "payload_type": "samga_brain_rw.run_manifest",
        "stage": 2,
        "subject": 1,
        "seed": 42,
        "config_id": "s2-layernorm-on",
        "config_sha256": _h("config"),
        "protocol_sha256": _h("protocol"),
        "cache_sha256": _h("cache"),
        "git_sha": "a" * 40,
        "upstream_sha": "1a63745b7ff6f98dad34b0f0b8246a9b5260d9c1",
        "data_order_sha256": _h("order"),
        "candidate_spec_sha256": _h("candidate-spec"),
        "run_key": arguments.run_key,
    }
    run_manifest = {
        **body,
        "run_manifest_sha256": sha256_json(body),
        "completed": False,
        "global_step": 1,
        "final_checkpoint": checkpoint.name,
        "final_checkpoint_sha256": checkpoint_sha,
        "checkpoint_hashes": {checkpoint.name: checkpoint_sha},
        "in_loop_score_directory": "in_loop",
        "max_train_steps": 1,
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
        "runtime_state": {"epoch_complete": False},
        "candidate_spec": candidate_spec,
        "run_manifest": {**body, "run_manifest_sha256": sha256_json(body)},
        "input_hashes": input_hashes,
    }
    monkeypatch.setattr(
        runner_module,
        "load_typed_torch_checkpoint",
        lambda *_args, **_kwargs: SimpleNamespace(
            payload=MappingProxyType(checkpoint_payload),
            sha256=checkpoint_sha,
        ),
    )
    monkeypatch.setattr(
        runner_module.ScoreArtifact,
        "load",
        lambda *_args, **_kwargs: object(),
    )

    validated = runner_module.validate_training_outputs(arguments)
    assert validated.final_checkpoint_sha256 == checkpoint_sha
    assert validated.run_manifest_sha256 == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    assert validated.in_loop_metadata_sha256 == hashlib.sha256(
        metadata.read_bytes()
    ).hexdigest()

    checkpoint_payload["runtime_state"] = {"epoch_complete": True}
    with pytest.raises(ValueError, match="partial|epoch_complete"):
        runner_module.validate_training_outputs(arguments)


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
