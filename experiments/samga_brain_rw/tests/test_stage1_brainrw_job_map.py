from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import MappingProxyType, ModuleType, SimpleNamespace

import pytest


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
        "stage1_test_build_job_map",
    )


@pytest.fixture
def builder_module(
    experiment_root: Path,
    jobmap_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> ModuleType:
    monkeypatch.setitem(sys.modules, "build_job_map", jobmap_module)
    return _load_script(
        experiment_root / "scripts" / "build_stage1_brainrw_job_map.py",
        "build_stage1_brainrw_job_map",
    )


def _install_fake_identities(
    module: ModuleType,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = (
        project_root
        / "experiments/samga_brain_rw/configs/brainrw_clip_lora_v1.json"
    )
    clip_path = (project_root.parent / "models" / "declared-clip").resolve()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(
            {"clip": {"path": str(clip_path)}},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    config = SimpleNamespace(
        path=config_path,
        payload=MappingProxyType(
            {
                "config_id": "brainrw_clip_lora_v1",
            }
        ),
        sha256=_h("config"),
        clip_path=clip_path,
        clip_config_sha256=_h("clip-config"),
        clip_preprocessor_sha256=_h("clip-preprocessor"),
        clip_weights_sha256=_h("clip-weights"),
    )
    monkeypatch.setattr(
        module.br,
        "verify_brainrw_config",
        lambda actual_config, actual_clip: (
            config
            if actual_config == config_path and actual_clip == clip_path
            else pytest.fail("builder used an unsealed config/CLIP path")
        ),
    )

    def manifest(path: Path, *, expected_subject: int) -> SimpleNamespace:
        expected_path = (
            project_root
            / "artifacts/samga_brain_rw/protocol/manifests"
            / f"sub-{expected_subject:02d}_protocol.json"
        )
        assert path == expected_path
        return SimpleNamespace(
            path=path,
            subject=expected_subject,
            manifest_sha256=_h(f"manifest:{expected_subject}"),
            protocol_sha256=_h("protocol"),
            records_sha256=_h("records"),
            source_manifest_sha256=_h(f"source:{expected_subject}"),
            source_payload_sha256=_h(f"payload:{expected_subject}"),
            train_role_sha256=_h(f"train:{expected_subject}"),
            val_dev_role_sha256=_h(f"val:{expected_subject}"),
        )

    monkeypatch.setattr(
        module.br,
        "load_development_manifest_identity",
        manifest,
    )


def test_smoke_map_is_one_fixed_debug_cell(
    builder_module: ModuleType,
    jobmap_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_root = (tmp_path / "project").resolve()
    project_root.mkdir()
    _install_fake_identities(builder_module, project_root, monkeypatch)

    rows = builder_module.build_stage1_brainrw_rows(
        project_root=project_root,
        phase="smoke",
        semantic_environment_sha256=_h("environment"),
    )
    payload = jobmap_module.build_job_map(rows)

    assert payload["stage"] == "stage-1-brainrw-smoke"
    assert payload["row_count"] == 1
    row = payload["rows"][0]
    assert (row["subject"], row["seed"]) == (8, 42)
    assert row["role"] == "clip-branch"
    assert row["partition"] == "debug"
    assert row["time"] == "00:30:00"
    assert row["gres"] == "gpu:a40:1"
    assert row["cpus"] == 8 and row["memory"] == "64G"
    assert row["expected_completion_schema"] == {
        "schema_version": 1,
        "payload_type": "samga_brain_rw.brainrw_smoke_completion",
        "required_output_hashes": [
            "final_checkpoint_sha256",
            "in_loop_metadata_sha256",
            "run_manifest_sha256",
        ],
    }
    argv = row["argv"]
    assert argv[argv.index("--mode") + 1] == "smoke"
    assert argv[argv.index("--max-train-steps") + 1] == "1"
    assert argv[argv.index("--device") + 1] == "cuda"
    assert "val-confirm" not in "\n".join(argv)
    assert "formal-test" not in "\n".join(argv)


def test_pilot_map_is_exact_three_by_two_grid(
    builder_module: ModuleType,
    jobmap_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_root = (tmp_path / "project").resolve()
    project_root.mkdir()
    _install_fake_identities(builder_module, project_root, monkeypatch)

    payload = jobmap_module.build_job_map(
        builder_module.build_stage1_brainrw_rows(
            project_root=project_root,
            phase="pilot",
            semantic_environment_sha256=_h("environment"),
        )
    )

    assert payload["stage"] == "stage-1-brainrw-pilot"
    assert payload["row_count"] == 6
    assert {(row["subject"], row["seed"]) for row in payload["rows"]} == {
        (subject, seed) for subject in (1, 5, 8) for seed in (42, 43)
    }
    sub08 = [row for row in payload["rows"] if row["subject"] == 8]
    assert {row["input_bundle_sha256"] for row in sub08} == {
        sub08[0]["input_bundle_sha256"]
    }
    assert {
        row["argv"][row["argv"].index("--manifest") + 1] for row in sub08
    } == {sub08[0]["argv"][sub08[0]["argv"].index("--manifest") + 1]}
    for row in payload["rows"]:
        assert row["partition"] == "i64m1tga40u"
        assert row["time"] == "02:00:00"
        assert "--max-train-steps" not in row["argv"]
        assert row["argv"][row["argv"].index("--mode") + 1] == "full"
        assert row["expected_completion_schema"] == {
            "schema_version": 1,
            "payload_type": "samga_brain_rw.brainrw_full_completion",
            "required_output_hashes": [
                "final_checkpoint_sha256",
                "run_manifest_sha256",
                "score_envelope_sha256",
                "score_payload_sha256",
            ],
        }


@pytest.mark.parametrize(
    "pilot_partition",
    (
        "i64m1tga40u",
        "i64m1tga40ue",
        "emergency_gpua40",
    ),
)
def test_pilot_builder_accepts_exact_partition_escalation_allowlist(
    builder_module: ModuleType,
    jobmap_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    pilot_partition: str,
) -> None:
    project_root = (tmp_path / "project").resolve()
    project_root.mkdir()
    _install_fake_identities(builder_module, project_root, monkeypatch)

    rows = builder_module.build_stage1_brainrw_rows(
        project_root=project_root,
        phase="pilot",
        semantic_environment_sha256=_h("environment"),
        pilot_partition=pilot_partition,
    )
    payload = jobmap_module.build_job_map(rows)

    assert {row["partition"] for row in payload["rows"]} == {pilot_partition}
    assert {row["time"] for row in payload["rows"]} == {"02:00:00"}


@pytest.mark.parametrize("pilot_partition", ("debug", "unknown"))
def test_pilot_builder_rejects_non_escalation_partition(
    builder_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    pilot_partition: str,
) -> None:
    project_root = (tmp_path / "project").resolve()
    project_root.mkdir()
    _install_fake_identities(builder_module, project_root, monkeypatch)

    with pytest.raises(ValueError, match="pilot partition|partition.*pilot"):
        builder_module.build_stage1_brainrw_rows(
            project_root=project_root,
            phase="pilot",
            semantic_environment_sha256=_h("environment"),
            pilot_partition=pilot_partition,
        )


@pytest.mark.parametrize(
    "pilot_partition",
    (
        "i64m1tga40u",
        "i64m1tga40ue",
        "emergency_gpua40",
        "debug",
    ),
)
def test_smoke_builder_rejects_every_pilot_partition_override(
    builder_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    pilot_partition: str,
) -> None:
    project_root = (tmp_path / "project").resolve()
    project_root.mkdir()
    _install_fake_identities(builder_module, project_root, monkeypatch)

    with pytest.raises(ValueError, match="smoke|override|pilot partition"):
        builder_module.build_stage1_brainrw_rows(
            project_root=project_root,
            phase="smoke",
            semantic_environment_sha256=_h("environment"),
            pilot_partition=pilot_partition,
        )


@pytest.mark.parametrize(
    "pilot_partition",
    (
        "i64m1tga40u",
        "i64m1tga40ue",
        "emergency_gpua40",
    ),
)
def test_builder_cli_exposes_only_sealed_pilot_partitions(
    builder_module: ModuleType,
    tmp_path: Path,
    pilot_partition: str,
) -> None:
    arguments = builder_module._parser().parse_args(
        [
            "--phase",
            "pilot",
            "--project-root",
            str(tmp_path),
            "--semantic-environment-sha256",
            _h("environment"),
            "--pilot-partition",
            pilot_partition,
            "--output",
            str(tmp_path / "map.json"),
        ]
    )

    assert arguments.pilot_partition == pilot_partition


@pytest.mark.parametrize("phase", ("full", "confirmation", "formal-test"))
def test_builder_rejects_unregistered_phases(
    builder_module: ModuleType,
    tmp_path: Path,
    phase: str,
) -> None:
    with pytest.raises(ValueError, match="phase|smoke|pilot"):
        builder_module.build_stage1_brainrw_rows(
            project_root=(tmp_path / "project").resolve(),
            phase=phase,
            semantic_environment_sha256=_h("environment"),
        )


def test_builder_requires_lowercase_semantic_environment_hash(
    builder_module: ModuleType,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="semantic|SHA-256"):
        builder_module.build_stage1_brainrw_rows(
            project_root=(tmp_path / "project").resolve(),
            phase="smoke",
            semantic_environment_sha256="not-a-hash",
        )


def test_builder_rejects_verified_clip_path_drift_from_config(
    builder_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_root = (tmp_path / "project").resolve()
    project_root.mkdir()
    config_path = (
        project_root
        / "experiments/samga_brain_rw/configs/brainrw_clip_lora_v1.json"
    )
    declared = (tmp_path / "models" / "declared").resolve()
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            {"clip": {"path": str(declared)}},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        builder_module.br,
        "verify_brainrw_config",
        lambda _config, actual_clip: SimpleNamespace(
            path=config_path,
            payload=MappingProxyType({"config_id": "brainrw_clip_lora_v1"}),
            sha256=_h("config"),
            clip_path=(tmp_path / "models" / "different").resolve(),
        )
        if actual_clip == declared
        else pytest.fail("builder ignored the path declared by config"),
    )

    with pytest.raises(ValueError, match="CLIP|clip|config|drift"):
        builder_module.build_stage1_brainrw_rows(
            project_root=project_root,
            phase="smoke",
            semantic_environment_sha256=_h("environment"),
        )
