from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import MappingProxyType, ModuleType, SimpleNamespace

import pytest

from samga_brain_rw.config import make_run_key
from samga_brain_rw.hashing import ordered_ids_sha256, sha256_json


def _h(label: str) -> str:
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
def runner_module(experiment_root: Path) -> ModuleType:
    return _load_script(
        experiment_root / "scripts" / "run_brainrw_cell.py",
        "run_brainrw_cell",
    )


def _argv(
    tmp_path: Path,
    *,
    mode: str,
    subject: int = 8,
    seed: int = 42,
) -> list[str]:
    project_root = (tmp_path / "project").resolve()
    output_root = (
        project_root
        / "artifacts"
        / "samga_brain_rw"
        / f"stage-1-brainrw-{'smoke' if mode == 'smoke' else 'pilot'}"
    )
    config_sha256 = _h("brainrw-config")
    input_bundle_sha256 = _h(f"input:{subject}")
    semantic_environment_sha256 = _h("environment")
    run_key = make_run_key(
        "brainrw-clip-lora",
        "brainrw_clip_lora_v1",
        subject,
        seed,
        config_sha256,
        input_bundle_sha256,
    )
    values = [
        "--mode",
        mode,
        "--subject",
        str(subject),
        "--seed",
        str(seed),
        "--resume",
        "none",
        "--config",
        str(
            project_root
            / "experiments/samga_brain_rw/configs/"
            "brainrw_clip_lora_v1.json"
        ),
        "--manifest",
        str(
            project_root
            / "artifacts/samga_brain_rw/protocol/manifests/"
            f"sub-{subject:02d}_protocol.json"
        ),
        "--clip-path",
        str((tmp_path / "models" / "clip").resolve()),
        "--output-dir",
        str(output_root / run_key),
        "--project-root",
        str(project_root),
        "--config-id",
        "brainrw_clip_lora_v1",
        "--expected-config-sha256",
        config_sha256,
        "--expected-input-bundle-sha256",
        input_bundle_sha256,
        "--expected-semantic-environment-sha256",
        semantic_environment_sha256,
        "--run-key",
        run_key,
        "--device",
        "cuda",
    ]
    if mode == "smoke":
        values.extend(["--max-train-steps", "1"])
    return values


def test_cli_locks_smoke_and_full_modes(
    runner_module: ModuleType,
    tmp_path: Path,
) -> None:
    smoke = runner_module.parse_arguments(_argv(tmp_path, mode="smoke"))
    assert smoke.mode == "smoke"
    assert smoke.max_train_steps == 1
    assert smoke.resume == "none"

    full = runner_module.parse_arguments(_argv(tmp_path, mode="full"))
    assert full.mode == "full"
    assert full.max_train_steps is None

    wrong_key = _argv(tmp_path, mode="smoke")
    wrong_key[wrong_key.index("--run-key") + 1] = "wrong"
    with pytest.raises(SystemExit):
        runner_module.parse_arguments(wrong_key)

    forbidden_limit = _argv(tmp_path, mode="full")
    forbidden_limit.extend(["--max-train-steps", "1"])
    with pytest.raises(SystemExit):
        runner_module.parse_arguments(forbidden_limit)


def test_cli_requires_fresh_full_and_exact_one_step_smoke(
    runner_module: ModuleType,
    tmp_path: Path,
) -> None:
    resumed_full = _argv(tmp_path, mode="full")
    resumed_full[resumed_full.index("--resume") + 1] = str(
        (tmp_path / "checkpoint.pt").resolve()
    )
    with pytest.raises(SystemExit):
        runner_module.parse_arguments(resumed_full)

    two_step_smoke = _argv(tmp_path, mode="smoke")
    two_step_smoke[
        two_step_smoke.index("--max-train-steps") + 1
    ] = "2"
    with pytest.raises(SystemExit):
        runner_module.parse_arguments(two_step_smoke)


def _schedule_binding(
    runner_module: ModuleType,
    tmp_path: Path,
    *,
    mode: str,
    planned_steps: int = 625,
    global_step: int | None = None,
) -> tuple[
    argparse.Namespace,
    SimpleNamespace,
    SimpleNamespace,
    SimpleNamespace,
    dict[str, object],
]:
    arguments = runner_module.parse_arguments(_argv(tmp_path, mode=mode))
    completed = (
        (1 if mode == "smoke" else planned_steps)
        if global_step is None
        else global_step
    )
    training_complete = mode == "full"
    checkpoint = SimpleNamespace(
        payload={
            "planned_steps": planned_steps,
            "global_step": completed,
            "steps": completed,
            "training_complete": training_complete,
            "effective_batch_size": 512,
            "resumed_from_sha256": None,
        }
    )
    config = SimpleNamespace(
        payload={
            "training": {
                "epochs": 25,
                "batch_size": 512,
            }
        }
    )
    manifest = SimpleNamespace(train_row_count=12_540)
    run_manifest = {
        "planned_steps": planned_steps,
        "completed_steps": completed,
        "training_complete": training_complete,
        "effective_batch_size": 512,
        "resumed_from_sha256": None,
    }
    return arguments, checkpoint, config, manifest, run_manifest


@pytest.mark.parametrize("mode", ("smoke", "full"))
def test_locked_schedule_accepts_only_the_625_step_recipe(
    runner_module: ModuleType,
    tmp_path: Path,
    mode: str,
) -> None:
    binding = _schedule_binding(
        runner_module,
        tmp_path,
        mode=mode,
    )
    runner_module._validate_locked_schedule(*binding)


@pytest.mark.parametrize("mode", ("smoke", "full"))
def test_locked_schedule_rejects_self_consistent_624_step_output(
    runner_module: ModuleType,
    tmp_path: Path,
    mode: str,
) -> None:
    binding = _schedule_binding(
        runner_module,
        tmp_path,
        mode=mode,
        planned_steps=624,
    )
    with pytest.raises(ValueError, match="625|planned|step|recipe"):
        runner_module._validate_locked_schedule(*binding)


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        (
            lambda config, manifest: config.payload["training"].update(
                {"epochs": 24}
            ),
            "epoch|25|recipe",
        ),
        (
            lambda config, manifest: config.payload["training"].update(
                {"batch_size": 511}
            ),
            "batch|512|recipe",
        ),
        (
            lambda config, manifest: setattr(
                manifest,
                "train_row_count",
                12_539,
            ),
            "row|12540|12,540|recipe",
        ),
    ),
)
def test_locked_schedule_binds_config_and_manifest_recipe(
    runner_module: ModuleType,
    tmp_path: Path,
    mutation: object,
    match: str,
) -> None:
    binding = _schedule_binding(
        runner_module,
        tmp_path,
        mode="full",
    )
    _, _, config, manifest, _ = binding
    mutation(config, manifest)  # type: ignore[operator]
    with pytest.raises(ValueError, match=match):
        runner_module._validate_locked_schedule(*binding)


@pytest.mark.parametrize("mode", ("smoke", "full"))
def test_locked_schedule_rejects_non_null_resume_parent(
    runner_module: ModuleType,
    tmp_path: Path,
    mode: str,
) -> None:
    binding = _schedule_binding(
        runner_module,
        tmp_path,
        mode=mode,
    )
    _, checkpoint, _, _, run_manifest = binding
    resumed_from = _h("resume-parent")
    checkpoint.payload["resumed_from_sha256"] = resumed_from
    run_manifest["resumed_from_sha256"] = resumed_from

    with pytest.raises(ValueError, match="resume|parent|fresh|provenance"):
        runner_module._validate_locked_schedule(*binding)


@pytest.mark.parametrize("mode", ("smoke", "full"))
def test_runner_executes_only_development_commands_then_completes(
    runner_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
) -> None:
    arguments = runner_module.parse_arguments(_argv(tmp_path, mode=mode))
    monkeypatch.setattr(
        runner_module,
        "_preflight_inputs",
        lambda _arguments: None,
    )
    monkeypatch.setattr(
        runner_module,
        "_validate_runtime_and_input_identity",
        lambda _arguments: None,
    )
    outputs = runner_module.BrainRWOutputs(
        run_manifest_path=arguments.output_dir / "run_manifest.json",
        run_manifest_sha256=_h("manifest"),
        checkpoint_path=arguments.output_dir / "checkpoint.pt",
        checkpoint_sha256=_h("checkpoint"),
        in_loop_metadata_path=(
            arguments.output_dir / "training_smoke/in_loop/metadata.json"
            if mode == "smoke"
            else None
        ),
        in_loop_metadata_sha256=(
            _h("in-loop") if mode == "smoke" else None
        ),
        score_directory=(
            arguments.output_dir / "val_dev_scores"
            if mode == "full"
            else None
        ),
        score_payload_sha256=(
            _h("score-payload") if mode == "full" else None
        ),
        score_envelope_sha256=(
            _h("score-envelope") if mode == "full" else None
        ),
    )
    proof = SimpleNamespace(outputs=outputs)
    monkeypatch.setattr(
        runner_module,
        "_validate_brainrw_training_once",
        lambda _arguments: proof,
    )
    monkeypatch.setattr(
        runner_module,
        "_validate_brainrw_outputs_from_proof",
        lambda _arguments, actual_proof: (
            outputs
            if actual_proof is proof
            else pytest.fail("runner changed the captured training proof")
        ),
    )
    monkeypatch.setenv("SAMGA_JOB_MAP", "/sealed/job-map.json")
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        commands.append(list(command))
        assert kwargs["check"] is True
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        assert environment["HF_DATASETS_OFFLINE"] == "1"
        assert environment["TRANSFORMERS_OFFLINE"] == "1"
        return SimpleNamespace(returncode=0)

    assert (
        runner_module.run_cell(
            arguments,
            subprocess_runner=fake_run,
        )
        == 0
    )
    assert "--scope" in commands[0]
    assert commands[0][commands[0].index("--scope") + 1] == "train"
    assert commands[0][commands[0].index("--validation-scope") + 1] == (
        "val-dev"
    )
    flattened = "\n".join(item for command in commands for item in command)
    assert "val-confirm" not in flattened
    assert "formal-test" not in flattened

    completion = commands[-1]
    assert completion[1].endswith(
        "experiments/samga_brain_rw/scripts/build_job_map.py"
    )
    assert "complete-env" in completion
    encoded = completion[completion.index("--output-hashes") + 1]
    expected_names = (
        {
            "final_checkpoint_sha256",
            "in_loop_metadata_sha256",
            "run_manifest_sha256",
        }
        if mode == "smoke"
        else {
            "final_checkpoint_sha256",
            "run_manifest_sha256",
            "score_envelope_sha256",
            "score_payload_sha256",
        }
    )
    import json

    assert set(json.loads(encoded)) == expected_names
    if mode == "smoke":
        assert len(commands) == 2
    else:
        assert len(commands) == 3
        assert commands[1][1].endswith(
            "experiments/samga_brain_rw/scripts/emit_brainrw_scores.py"
        )
        assert commands[1][commands[1].index("--scope") + 1] == "val-dev"


def test_full_runner_validates_training_once_and_threads_one_proof(
    runner_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    arguments = runner_module.parse_arguments(_argv(tmp_path, mode="full"))
    outputs = runner_module.BrainRWOutputs(
        run_manifest_path=arguments.output_dir / "run_manifest.json",
        run_manifest_sha256=_h("manifest"),
        checkpoint_path=arguments.output_dir / "checkpoint.pt",
        checkpoint_sha256=_h("checkpoint"),
        score_directory=arguments.output_dir / "val_dev_scores",
        score_payload_sha256=_h("score-payload"),
        score_envelope_sha256=_h("score-envelope"),
    )
    proof = SimpleNamespace(outputs=outputs)
    captures: list[argparse.Namespace] = []
    monkeypatch.setattr(
        runner_module,
        "_preflight_inputs",
        lambda _arguments: None,
    )
    monkeypatch.setattr(
        runner_module,
        "_validate_runtime_and_input_identity",
        lambda _arguments: None,
    )
    monkeypatch.setattr(
        runner_module,
        "_validate_brainrw_training_once",
        lambda actual: captures.append(actual) or proof,
        raising=False,
    )
    monkeypatch.setattr(
        runner_module,
        "validate_brainrw_training_outputs",
        lambda _arguments: pytest.fail(
            "run_cell must not reload training outputs after capture"
        ),
    )

    def validate(
        actual: argparse.Namespace,
        actual_proof: object,
    ) -> object:
        assert actual is arguments
        assert actual_proof is proof
        return outputs

    monkeypatch.setattr(
        runner_module,
        "_validate_brainrw_outputs_from_proof",
        validate,
    )
    monkeypatch.setenv("SAMGA_JOB_MAP", "/sealed/job-map.json")
    commands: list[list[str]] = []
    assert (
        runner_module.run_cell(
            arguments,
            subprocess_runner=lambda command, **_kwargs: commands.append(
                list(command)
            ),
        )
        == 0
    )
    assert captures == [arguments]
    assert len(commands) == 3


def test_full_runner_rejects_a_b_switch_before_completion(
    runner_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact_b, proof_a, arguments = _validated_proof_binding(
        runner_module,
        tmp_path,
    )
    artifact_b.metadata["checkpoint_sha256"] = _h("checkpoint-b")
    artifact_b.provenance["checkpoint_sha256"] = _h("checkpoint-b")
    with pytest.raises(FrozenInstanceError):
        proof_a.outputs = SimpleNamespace()

    captures: list[argparse.Namespace] = []
    monkeypatch.setattr(
        runner_module,
        "_preflight_inputs",
        lambda _arguments: None,
    )
    monkeypatch.setattr(
        runner_module,
        "_validate_runtime_and_input_identity",
        lambda _arguments: None,
    )
    monkeypatch.setattr(
        runner_module,
        "_validate_brainrw_training_once",
        lambda actual: captures.append(actual) or proof_a,
    )
    score_loads: list[Path] = []
    monkeypatch.setattr(
        runner_module.ScoreArtifact,
        "load",
        lambda path, **_kwargs: (
            score_loads.append(Path(path)) or artifact_b
        ),
    )
    real_from_proof = (
        runner_module._validate_brainrw_outputs_from_proof
    )
    proof_validations: list[object] = []

    def validate_from_proof(
        actual: argparse.Namespace,
        proof: object,
    ) -> object:
        proof_validations.append(proof)
        return real_from_proof(actual, proof)

    monkeypatch.setattr(
        runner_module,
        "_validate_brainrw_outputs_from_proof",
        validate_from_proof,
    )
    monkeypatch.setenv("SAMGA_JOB_MAP", "/sealed/job-map.json")
    commands: list[list[str]] = []

    def record_command(
        command: list[str],
        **_kwargs: object,
    ) -> None:
        commands.append(list(command))

    with pytest.raises(ValueError, match="checkpoint|proof|binding"):
        runner_module.run_cell(
            arguments,
            subprocess_runner=record_command,
        )
    assert captures == [arguments]
    assert proof_validations == [proof_a]
    assert score_loads == [arguments.output_dir / "val_dev_scores"]
    assert not any("complete-env" in command for command in commands)


def test_command_output_revalidation_reparses_exact_sealed_runner(
    runner_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    arguments = runner_module.parse_arguments(_argv(tmp_path, mode="smoke"))
    expected = SimpleNamespace(label="validated")
    monkeypatch.setattr(
        runner_module,
        "validate_brainrw_outputs",
        lambda actual: expected if actual.run_key == arguments.run_key else None,
    )
    command = [
        "python",
        str(
            arguments.project_root
            / "experiments/samga_brain_rw/scripts/run_brainrw_cell.py"
        ),
        *_argv(tmp_path, mode="smoke"),
    ]

    assert (
        runner_module.validate_brainrw_command_outputs(
            command,
            expected_mode="smoke",
        )
        is expected
    )
    command[1] = str(tmp_path / "other.py")
    with pytest.raises(ValueError, match="runner|project-root"):
        runner_module.validate_brainrw_command_outputs(
            command,
            expected_mode="smoke",
        )


def test_map_config_verifier_binds_declared_config_and_clip(
    runner_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    arguments = runner_module.parse_arguments(_argv(tmp_path, mode="smoke"))
    command = [
        "python",
        str(
            arguments.project_root
            / "experiments/samga_brain_rw/scripts/run_brainrw_cell.py"
        ),
        *_argv(tmp_path, mode="smoke"),
    ]
    verified = SimpleNamespace(
        path=arguments.config,
        payload={"config_id": arguments.config_id},
        sha256=arguments.expected_config_sha256,
        clip_path=arguments.clip_path,
    )
    monkeypatch.setattr(
        runner_module.br,
        "verify_brainrw_config",
        lambda config, clip: (
            verified
            if config == arguments.config and clip == arguments.clip_path
            else pytest.fail("map config verifier used different paths")
        ),
    )
    assert runner_module.validate_brainrw_map_config(command) is verified

    verified.clip_path = (tmp_path / "models" / "switched").resolve()
    with pytest.raises(ValueError, match="clip|CLIP|config|drift"):
        runner_module.validate_brainrw_map_config(command)


def test_preflight_rejects_sealed_scope_path_before_subprocess(
    runner_module: ModuleType,
    tmp_path: Path,
) -> None:
    argv = _argv(tmp_path, mode="smoke")
    argv[argv.index("--manifest") + 1] = str(
        tmp_path / "val-confirm" / "sub-08_protocol.json"
    )
    arguments = runner_module.parse_arguments(argv)
    with pytest.raises(PermissionError, match="sealed|scope|development"):
        runner_module._preflight_inputs(arguments)


def _score_binding(
    runner_module: ModuleType,
    tmp_path: Path,
) -> tuple[
    SimpleNamespace,
    SimpleNamespace,
    SimpleNamespace,
    argparse.Namespace,
]:
    arguments = runner_module.parse_arguments(_argv(tmp_path, mode="full"))
    ids = ("image-a", "image-b")
    manifest = SimpleNamespace(
        manifest_sha256=_h("manifest"),
        protocol_sha256=_h("protocol"),
        records_sha256=_h("records"),
        source_manifest_sha256=_h("source-manifest"),
        source_payload_byte_count=123,
        source_payload_path=Path("/safe/train-source.pt"),
        source_payload_sha256=_h("source-payload"),
        val_dev_role_sha256=_h("val-dev-role"),
        val_dev_ordered_ids=ids,
        val_dev_ordered_ids_sha256=ordered_ids_sha256(ids),
    )
    checkpoint_payload = {
        "git_sha": "a" * 40,
        "semantic_environment": {"python": "3.11"},
        "semantic_environment_sha256": sha256_json({"python": "3.11"}),
        "runtime_contract": {"device_type": "cuda"},
        "runtime_contract_sha256": sha256_json({"device_type": "cuda"}),
        "runtime_evidence": {"accelerator_name": "NVIDIA A40"},
        "runtime_evidence_sha256": sha256_json(
            {"accelerator_name": "NVIDIA A40"}
        ),
        "global_step": 625,
        "planned_steps": 625,
        "training_complete": True,
    }
    checkpoint = SimpleNamespace(
        sha256=_h("checkpoint"),
        payload=checkpoint_payload,
    )
    source_records = [
        {
            "manifest_sha256": manifest.manifest_sha256,
            "records_sha256": manifest.records_sha256,
            "role": "val-dev",
            "role_payload_sha256": manifest.val_dev_role_sha256,
            "source_manifest_sha256": manifest.source_manifest_sha256,
            "source_payload_byte_count": manifest.source_payload_byte_count,
            "source_payload_path": str(manifest.source_payload_path),
            "source_payload_sha256": manifest.source_payload_sha256,
        }
    ]
    runtime = {
        "training_semantic_environment": checkpoint_payload[
            "semantic_environment"
        ],
        "training_semantic_environment_sha256": checkpoint_payload[
            "semantic_environment_sha256"
        ],
        "evaluation_semantic_environment": checkpoint_payload[
            "semantic_environment"
        ],
        "evaluation_semantic_environment_sha256": checkpoint_payload[
            "semantic_environment_sha256"
        ],
        "evaluation_runtime_contract": checkpoint_payload[
            "runtime_contract"
        ],
        "evaluation_runtime_contract_sha256": checkpoint_payload[
            "runtime_contract_sha256"
        ],
        "evaluation_runtime_evidence": checkpoint_payload[
            "runtime_evidence"
        ],
        "evaluation_runtime_evidence_sha256": checkpoint_payload[
            "runtime_evidence_sha256"
        ],
    }
    metadata = {
        "checkpoint_sha256": checkpoint.sha256,
        "config_sha256": arguments.expected_config_sha256,
        "protocol_sha256": manifest.protocol_sha256,
        "git_sha": checkpoint_payload["git_sha"],
        "seed": arguments.seed,
        "split_role": "val-dev",
        "stage": "brainrw-clip-lora",
        "subject": arguments.subject,
        "query_ids_sha256": manifest.val_dev_ordered_ids_sha256,
        "gallery_ids_sha256": manifest.val_dev_ordered_ids_sha256,
        "ordered_ids": [*ids, *ids],
        "source_records": source_records,
        "source_records_sha256": sha256_json(source_records),
        **runtime,
    }
    provenance = {
        key: metadata[key]
        for key in (
            "checkpoint_sha256",
            "config_sha256",
            "protocol_sha256",
            "git_sha",
            "seed",
            "split_role",
            "stage",
            "subject",
            "query_ids_sha256",
            "gallery_ids_sha256",
            "source_records_sha256",
        )
    }
    provenance.update(runtime)
    artifact = SimpleNamespace(
        metadata=metadata,
        provenance=provenance,
        query_ids=ids,
        gallery_ids=ids,
    )
    return artifact, checkpoint, manifest, arguments


def _validated_proof_binding(
    runner_module: ModuleType,
    tmp_path: Path,
) -> tuple[SimpleNamespace, object, argparse.Namespace]:
    artifact, checkpoint, manifest, arguments = _score_binding(
        runner_module,
        tmp_path,
    )
    manifest.path = arguments.manifest
    manifest.train_role_sha256 = _h("train-role")
    manifest.train_row_count = 12_540
    manifest.val_dev_row_count = len(manifest.val_dev_ordered_ids)
    checkpoint.payload["resumed_from_sha256"] = None
    config = SimpleNamespace(
        path=arguments.config,
        payload=MappingProxyType(
            {"config_id": arguments.config_id}
        ),
        sha256=arguments.expected_config_sha256,
        clip_path=arguments.clip_path,
    )
    run_manifest_bytes = b'{"generation":"a"}\n'
    run_manifest_sha256 = hashlib.sha256(
        run_manifest_bytes
    ).hexdigest()
    outputs = runner_module.BrainRWOutputs(
        run_manifest_path=arguments.output_dir / "run_manifest.json",
        run_manifest_sha256=run_manifest_sha256,
        checkpoint_path=arguments.output_dir / "checkpoint.pt",
        checkpoint_sha256=checkpoint.sha256,
    )
    proof = runner_module._make_validated_proof(
        arguments=arguments,
        config=config,
        manifest=manifest,
        checkpoint=checkpoint,
        run_manifest={"resumed_from_sha256": None},
        run_manifest_bytes=run_manifest_bytes,
        outputs=outputs,
    )
    return artifact, proof, arguments


def test_score_identity_binds_manifest_checkpoint_ids_and_runtime(
    runner_module: ModuleType,
    tmp_path: Path,
) -> None:
    artifact, checkpoint, manifest, arguments = _score_binding(
        runner_module,
        tmp_path,
    )
    runner_module._validate_score_identity(
        artifact,
        arguments=arguments,
        checkpoint=checkpoint,
        manifest=manifest,
        expected_stage="brainrw-clip-lora",
    )

    for field, replacement in (
        ("protocol_sha256", _h("other-protocol")),
        ("git_sha", "b" * 40),
        ("evaluation_runtime_contract_sha256", _h("other-runtime")),
    ):
        broken_artifact, _, _, _ = _score_binding(
            runner_module,
            tmp_path,
        )
        broken_artifact.metadata[field] = replacement
        broken_artifact.provenance[field] = replacement
        with pytest.raises(ValueError, match="protocol|git|runtime|binding"):
            runner_module._validate_score_identity(
                broken_artifact,
                arguments=arguments,
                checkpoint=checkpoint,
                manifest=manifest,
                expected_stage="brainrw-clip-lora",
            )


def test_score_identity_rejects_source_or_ordered_id_cross_binding(
    runner_module: ModuleType,
    tmp_path: Path,
) -> None:
    artifact, checkpoint, manifest, arguments = _score_binding(
        runner_module,
        tmp_path,
    )
    artifact.metadata["source_records"][0]["manifest_sha256"] = _h(
        "other-manifest"
    )
    artifact.metadata["source_records_sha256"] = sha256_json(
        artifact.metadata["source_records"]
    )
    artifact.provenance["source_records_sha256"] = artifact.metadata[
        "source_records_sha256"
    ]
    with pytest.raises(ValueError, match="source|manifest|binding"):
        runner_module._validate_score_identity(
            artifact,
            arguments=arguments,
            checkpoint=checkpoint,
            manifest=manifest,
            expected_stage="brainrw-clip-lora",
        )

    artifact, checkpoint, manifest, arguments = _score_binding(
        runner_module,
        tmp_path,
    )
    artifact.query_ids = ("image-b", "image-a")
    with pytest.raises(ValueError, match="query|ID|ordered"):
        runner_module._validate_score_identity(
            artifact,
            arguments=arguments,
            checkpoint=checkpoint,
            manifest=manifest,
            expected_stage="brainrw-clip-lora",
        )
