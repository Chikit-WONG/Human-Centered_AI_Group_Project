from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from samga_brain_rw.hashing import sha256_json


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
def jobmap_module(experiment_root: Path) -> ModuleType:
    return _load_script(
        experiment_root / "scripts" / "build_job_map.py",
        "expansion_test_build_job_map",
    )


@pytest.fixture
def builder_module(
    experiment_root: Path,
    jobmap_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> ModuleType:
    monkeypatch.setitem(sys.modules, "build_job_map", jobmap_module)
    return _load_script(
        experiment_root / "scripts" / "build_stage1_expansion_job_map.py",
        "build_stage1_expansion_job_map",
    )


@pytest.fixture(scope="module")
def runner_module(experiment_root: Path) -> ModuleType:
    return _load_script(
        experiment_root / "scripts" / "run_stage1_expansion_cell.py",
        "run_stage1_expansion_cell",
    )


def _install_fake_resolvers(
    module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def identity(component: str, subject: int, seed: int) -> object:
        config_id = (
            "internvit_baseline_v1"
            if component == "internvit"
            else "brainrw_clip_lora_v1"
        )
        prefix = "stage0" if component == "internvit" else "brainrw-clip-lora"
        config_sha256 = _h(f"{component}:config:{subject}:{seed}")
        input_bundle_sha256 = _h(f"{component}:inputs:{subject}")
        run_key = module.make_run_key(
            prefix,
            config_id,
            subject,
            seed,
            config_sha256,
            input_bundle_sha256,
        )
        return module.ExpansionIdentity(
            component=component,
            config_id=config_id,
            config_sha256=config_sha256,
            input_bundle_sha256=input_bundle_sha256,
            run_key=run_key,
        )

    monkeypatch.setattr(
        module,
        "_resolve_samga_cell",
        lambda **kwargs: identity(
            "internvit", kwargs["subject"], kwargs["seed"]
        ),
    )
    monkeypatch.setattr(
        module,
        "_resolve_brainrw_cell",
        lambda **kwargs: identity(
            "brainrw", kwargs["subject"], kwargs["seed"]
        ),
    )
    monkeypatch.setattr(
        module,
        "_declared_clip_path",
        lambda path: path.parent / "fake-clip",
    )


def test_expansion_map_is_two_components_by_ten_subjects_by_five_seeds(
    builder_module: ModuleType,
    jobmap_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_root = (tmp_path / "project").resolve()
    project_root.mkdir()
    _install_fake_resolvers(builder_module, monkeypatch)

    rows = builder_module.build_stage1_expansion_rows(
        project_root=project_root,
        partition="i64m1tga40u",
        semantic_environment_sha256=_h("environment"),
        environment_binding={"binding": "fixture"},
        locked_survivor_sha256=_h("locked-survivor"),
    )
    payload = jobmap_module.build_job_map(rows)

    assert payload["stage"] == "stage-1-expansion-train"
    assert payload["row_count"] == 100
    assert {(row["role"], row["subject"], row["seed"]) for row in payload["rows"]} == {
        (role, subject, seed)
        for role in ("brainrw-component", "internvit-component")
        for subject in range(1, 11)
        for seed in range(42, 47)
    }
    assert len({row["run_key"] for row in payload["rows"]}) == 100
    for row in payload["rows"]:
        argv = row["argv"]
        assert argv[argv.index("--validation-scope") + 1] == "none"
        assert argv[argv.index("--resume") + 1] == "none"
        assert argv[argv.index("--locked-survivor-sha256") + 1] == _h(
            "locked-survivor"
        )
        assert row["partition"] == "i64m1tga40u"
        assert row["gres"] == "gpu:a40:1"
        assert row["cpus"] == 8
        assert row["memory"] == "64G"
        assert row["time"] == "04:00:00"
        assert row["expected_completion_schema"] == {
            "schema_version": 1,
            "payload_type": "samga_brain_rw.stage1_expansion_completion",
            "required_output_hashes": [
                "component_record_sha256",
                "final_checkpoint_sha256",
                "run_manifest_sha256",
            ],
        }
        text = "\n".join(argv)
        assert "val-confirm" not in text
        assert "formal-test" not in text
        assert "evaluate.py" not in text


@pytest.mark.parametrize(
    "partition",
    ("i64m1tga40u", "i64m1tga40ue", "emergency_gpua40"),
)
def test_expansion_map_accepts_only_locked_a40_partition_tiers(
    builder_module: ModuleType,
    jobmap_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    partition: str,
) -> None:
    project_root = (tmp_path / partition).resolve()
    project_root.mkdir()
    _install_fake_resolvers(builder_module, monkeypatch)
    payload = jobmap_module.build_job_map(
        builder_module.build_stage1_expansion_rows(
            project_root=project_root,
            partition=partition,
            semantic_environment_sha256=_h("environment"),
            environment_binding={"binding": "fixture"},
            locked_survivor_sha256=_h("locked-survivor"),
        )
    )
    assert {row["partition"] for row in payload["rows"]} == {partition}


@pytest.mark.parametrize("partition", ("debug", "emergency_gpu", "unknown"))
def test_expansion_map_rejects_other_partitions(
    builder_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    partition: str,
) -> None:
    project_root = (tmp_path / partition).resolve()
    project_root.mkdir()
    _install_fake_resolvers(builder_module, monkeypatch)
    with pytest.raises(ValueError, match="partition"):
        builder_module.build_stage1_expansion_rows(
            project_root=project_root,
            partition=partition,
            semantic_environment_sha256=_h("environment"),
            environment_binding={"binding": "fixture"},
            locked_survivor_sha256=_h("locked-survivor"),
        )


def _runner_argv(
    module: ModuleType,
    tmp_path: Path,
    component: str,
) -> list[str]:
    project_root = (tmp_path / "project").resolve()
    config_id = (
        "internvit_baseline_v1"
        if component == "internvit"
        else "brainrw_clip_lora_v1"
    )
    prefix = "stage0" if component == "internvit" else "brainrw-clip-lora"
    config_sha256 = _h(f"{component}:config")
    input_bundle_sha256 = _h(f"{component}:inputs")
    run_key = module.make_run_key(
        prefix,
        config_id,
        8,
        42,
        config_sha256,
        input_bundle_sha256,
    )
    argv = [
        "--component",
        component,
        "--validation-scope",
        "none",
        "--subject",
        "8",
        "--seed",
        "42",
        "--resume",
        "none",
        "--config",
        str(project_root / f"{component}.json"),
        "--manifest",
        str(project_root / "sub-08_protocol.json"),
        "--output-dir",
        str(project_root / "artifacts" / component / run_key),
        "--project-root",
        str(project_root),
        "--config-id",
        config_id,
        "--expected-config-sha256",
        config_sha256,
        "--expected-input-bundle-sha256",
        input_bundle_sha256,
        "--locked-survivor-sha256",
        _h("locked-survivor"),
        "--run-key",
        run_key,
        "--device",
        "cuda",
    ]
    if component == "internvit":
        argv.extend(
            [
                "--feature-cache",
                str(project_root / "features.npy"),
            ]
        )
    else:
        argv.extend(
            [
                "--clip-path",
                str(project_root / "clip"),
                "--expected-semantic-environment-sha256",
                _h("environment"),
            ]
        )
    return argv


@pytest.mark.parametrize("component", ("internvit", "brainrw"))
def test_expansion_runner_builds_train_only_command(
    runner_module: ModuleType,
    tmp_path: Path,
    component: str,
) -> None:
    arguments = runner_module.parse_arguments(
        _runner_argv(runner_module, tmp_path, component)
    )
    command = runner_module._training_command(arguments)

    assert command[command.index("--validation-scope") + 1] == "none"
    assert command[command.index("--scope") + 1] == "train"
    assert command[command.index("--resume") + 1] == "none"
    assert "--max-train-steps" not in command
    assert "evaluate.py" not in "\n".join(command)
    if component == "internvit":
        assert command[1].endswith("experiments/samga_brain_rw/train.py")
        assert command[command.index("--stage") + 1] == "0"
    else:
        assert command[1].endswith("experiments/samga_brain_rw/train_brainrw.py")


def test_component_record_identity_is_hash_sealed(
    runner_module: ModuleType,
) -> None:
    payload = {
        "artifact_type": "samga_brain_rw.stage1_expansion_component",
        "component": "internvit",
        "subject": 8,
        "seed": 42,
        "validation_scope": "none",
    }
    record = runner_module.seal_component_record(payload)
    assert record["schema_version"] == 1
    assert record["payload"] == payload
    assert record["payload_sha256"] == sha256_json(payload)


@pytest.mark.parametrize(
    ("component", "manifest_key", "model_key"),
    (
        ("internvit", "manifest_sha256", "model_sha256"),
        ("brainrw", "manifest", "clip_weights"),
    ),
)
def test_component_record_uses_checkpoint_provenance_layout(
    runner_module: ModuleType,
    tmp_path: Path,
    component: str,
    manifest_key: str,
    model_key: str,
) -> None:
    arguments = runner_module.parse_arguments(
        _runner_argv(runner_module, tmp_path, component)
    )
    run_manifest = {
        "git_sha": "1" * 40,
        "protocol_sha256": _h("protocol"),
    }
    checkpoint_payload = {
        "input_hashes": {
            manifest_key: _h("manifest"),
            model_key: _h("model"),
        }
    }
    payload = runner_module._component_payload(
        arguments,
        run_manifest,
        arguments.output_dir / "checkpoint.pt",
        checkpoint_payload,
        _h("checkpoint"),
        _h("run-manifest"),
    )

    assert payload["manifest_sha256"] == _h("manifest")
    assert payload["frozen_base_model_sha256"] == _h("model")
    assert payload["input_bundle_sha256"] == (
        arguments.expected_input_bundle_sha256
    )


def test_common_train_only_manifest_allows_samga_checkpoint_bound_bundle(
    runner_module: ModuleType,
    tmp_path: Path,
) -> None:
    arguments = runner_module.parse_arguments(
        _runner_argv(runner_module, tmp_path, "internvit")
    )
    runner_module._validate_common_manifest(
        {
            "subject": arguments.subject,
            "seed": arguments.seed,
            "run_key": arguments.run_key,
            "config_sha256": arguments.expected_config_sha256,
            "validation_scope": "none",
            "observed_scopes": ["train"],
            "validation_metrics": {
                "performed": False,
                "validation_scope": "none",
            },
        },
        arguments,
    )


def test_expansion_slurm_chunks_one_hundred_rows_into_ten_jobs(
    experiment_root: Path,
) -> None:
    script = (
        experiment_root / "slurm" / "expansion_train_array.slurm"
    ).read_text(encoding="utf-8")

    assert "JOB_MAP_CHUNK_STRIDE" in script
    assert "ARRAY_INDEX+=CHUNK_STRIDE" in script
    assert "timeout --signal=TERM --kill-after=5m 4h" in script
