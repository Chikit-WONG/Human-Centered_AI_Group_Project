from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from samga_brain_rw.hashing import sha256_json


LAUNCHERS = (
    "preflight_debug.slurm",
    "online_parity_debug.slurm",
    "pilot_array.slurm",
    "confirmation_array.slurm",
)
QUEUE_COMMAND = [
    "squeue",
    "-h",
    "-p",
    "debug,i64m1tga40u,i64m1tga40ue,emergency_gpua40",
    "-o",
    "%.10i %.14P %.10u %.2t %.10M %.6D %R",
]

JOB_ENVIRONMENT_NAMES = (
    "SAMGA_JOB_MAP",
    "SAMGA_JOB_MAP_SHA256",
    "SAMGA_JOB_ROW_SHA256",
    "SAMGA_JOB_CLAIM",
    "SAMGA_JOB_ARRAY_INDEX",
    "SAMGA_JOB_ARRAY_MIN",
    "SAMGA_JOB_ARRAY_MAX",
)


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
    module = _load_script(
        experiment_root / "scripts" / "build_job_map.py",
        "build_job_map",
    )
    return module


@pytest.fixture(scope="module")
def submit_module(
    experiment_root: Path,
    jobmap_module: ModuleType,
) -> ModuleType:
    sys.modules["build_job_map"] = jobmap_module
    return _load_script(
        experiment_root / "scripts" / "submit_pilot.py",
        "submit_pilot",
    )


def _row(
    tmp_path: Path,
    *,
    stage: str = "stage-2-pilot",
    mode: str | None = None,
    training_runner: bool = True,
    role: str = "candidate",
    config_id: str = "s2-layernorm-on",
    subject: int = 1,
    seed: int = 42,
    partition: str = "debug",
    time: str = "00:30:00",
) -> dict[str, object]:
    stage_number = 2 if "stage-2" in stage else 0
    selected_mode = mode or ("smoke" if "smoke" in stage else "full")
    config_sha256 = _h(f"config:{config_id}")
    input_bundle_sha256 = _h(f"input:{subject}:{seed}")
    run_key = (
        f"stage{stage_number}__{config_id}__sub-{subject:02d}__seed-{seed}__"
        f"{config_sha256}__{input_bundle_sha256}"
    )
    project_root = Path(__file__).resolve().parents[3]
    output_dir = tmp_path / stage / run_key
    argv = [
        "python",
        str(project_root / "experiments/samga_brain_rw/scripts/run_training_cell.py"),
        "--mode",
        selected_mode,
        "--stage",
        str(stage_number),
        "--role",
        role,
        "--subject",
        str(subject),
        "--seed",
        str(seed),
        "--resume",
        "none",
        "--config",
        "experiments/samga_brain_rw/configs/internvit_baseline_v1.json",
        "--manifest",
        f"artifacts/samga_brain_rw/protocol/manifests/sub-{subject:02d}_protocol.json",
        "--feature-cache",
        "artifacts/samga_reproduction/features/features.npy",
        "--output-dir",
        str(output_dir),
        "--project-root",
        str(project_root),
        "--config-id",
        config_id,
        "--expected-config-sha256",
        config_sha256,
        "--expected-input-bundle-sha256",
        input_bundle_sha256,
        "--run-key",
        run_key,
    ]
    if stage_number == 2:
        argv.extend(
            [
                "--stage2-config",
                "experiments/samga_brain_rw/configs/stage2_candidates_v1.json",
                "--candidate-id",
                config_id,
            ]
        )
    if selected_mode == "smoke":
        argv.extend(["--max-train-steps", "1"])
    if not training_runner:
        argv = [
            "python",
            "legacy_development_command.py",
            "--subject",
            str(subject),
            "--seed",
            str(seed),
            "--config",
            "legacy-development.json",
        ]
    required = (
        [
            "final_checkpoint_sha256",
            "in_loop_metadata_sha256",
            "run_manifest_sha256",
        ]
        if selected_mode == "smoke"
        else [
            "final_checkpoint_sha256",
            "parity_sha256",
            "run_manifest_sha256",
        ]
    )
    required = required if training_runner else ["metrics_sha256"]
    return {
        "stage": stage,
        "role": role,
        "config_id": config_id,
        "config_sha256": config_sha256,
        "input_bundle_sha256": input_bundle_sha256,
        "subject": subject,
        "seed": seed,
        "run_key": run_key,
        "argv": argv,
        "partition": partition,
        "gres": "gpu:a40:1",
        "cpus": 8,
        "memory": "64G",
        "time": time,
        "stdout_path": "logs/samga_brain_rw/sealed_%A_%a.out",
        "stderr_path": "logs/samga_brain_rw/sealed_%A_%a.err",
        "completion_path": str(tmp_path / stage / run_key / "completion.json"),
        "expected_completion_schema": {
            "schema_version": 1,
            "payload_type": "samga_brain_rw.job_completion",
            "required_output_hashes": required,
        },
    }


def _output_hashes(
    row: dict[str, object],
    label: str,
) -> dict[str, str]:
    schema = row["expected_completion_schema"]
    assert isinstance(schema, dict)
    names = schema["required_output_hashes"]
    assert isinstance(names, list)
    return {str(name): _h(f"{label}:{name}") for name in names}


def _rehash(payload: dict[str, object]) -> None:
    body = {key: value for key, value in payload.items() if key != "payload_sha256"}
    payload["payload_sha256"] = sha256_json(body)


def _claimed_job_environment(
    jobmap_module: ModuleType,
    tmp_path: Path,
) -> tuple[
    dict[str, object],
    dict[str, object],
    Path,
    dict[str, str],
]:
    map_path = tmp_path / "env-map.json"
    payload = jobmap_module.write_job_map([_row(tmp_path)], map_path)
    row = payload["rows"][0]
    claim = jobmap_module.claim_job_row(payload, row)
    array_min, array_max = payload["array_bounds"]
    environment = {
        "SAMGA_JOB_MAP": str(map_path),
        "SAMGA_JOB_MAP_SHA256": str(payload["payload_sha256"]),
        "SAMGA_JOB_ROW_SHA256": sha256_json(row),
        "SAMGA_JOB_CLAIM": str(claim.path),
        "SAMGA_JOB_ARRAY_INDEX": str(row["array_index"]),
        "SAMGA_JOB_ARRAY_MIN": str(array_min),
        "SAMGA_JOB_ARRAY_MAX": str(array_max),
    }
    return payload, row, map_path, environment


def _install_job_environment(
    monkeypatch: pytest.MonkeyPatch,
    environment: dict[str, str],
) -> None:
    for name in JOB_ENVIRONMENT_NAMES:
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)


def test_static_launchers_lock_gpu_logs_environment_and_conda(
    experiment_root: Path,
) -> None:
    slurm_dir = experiment_root / "slurm"
    for name in LAUNCHERS:
        text = (slurm_dir / name).read_text(encoding="utf-8")
        assert "#SBATCH --gres=gpu:a40:1" in text
        assert "logs/samga_brain_rw" in text
        assert re.search(r"#SBATCH --output=logs/samga_brain_rw/\S+\.out", text)
        assert re.search(r"#SBATCH --error=logs/samga_brain_rw/\S+\.err", text)
        assert "export PYTHONPATH=experiments/samga_brain_rw" in text
        assert "export HF_DATASETS_OFFLINE=1" in text
        assert "export TRANSFORMERS_OFFLINE=1" in text
        assert "export HF_HUB_OFFLINE=1" in text
        source = "source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh"
        assert source in text
        assert "conda activate test" in text
        assert text.index(source) < text.index("conda activate test")
        assert "formal-test" not in text.lower()
        assert "/test/" not in text.lower()
        assert "sub-01_test" not in text.lower()


@pytest.mark.parametrize(
    "name",
    ("preflight_debug.slurm", "online_parity_debug.slurm"),
)
def test_debug_launchers_fit_free_partition_limit(
    experiment_root: Path,
    name: str,
) -> None:
    text = (experiment_root / "slurm" / name).read_text(encoding="utf-8")
    assert "#SBATCH --partition=debug" in text
    match = re.search(r"#SBATCH --time=(\d{2}):(\d{2}):(\d{2})", text)
    assert match is not None
    hours, minutes, seconds = (int(value) for value in match.groups())
    assert hours * 3600 + minutes * 60 + seconds <= 30 * 60


def test_array_launchers_validate_hash_bounds_and_selected_row(
    experiment_root: Path,
) -> None:
    for name in ("pilot_array.slurm", "confirmation_array.slurm"):
        text = (experiment_root / "slurm" / name).read_text(encoding="utf-8")
        assert "${JOB_MAP:?" in text
        assert "${JOB_MAP_SHA256:?" in text
        assert "SLURM_ARRAY_TASK_ID" in text
        if name == "pilot_array.slurm":
            assert "JOB_MAP_ARRAY_MIN" in text
            assert "JOB_MAP_ARRAY_MAX" in text
        else:
            assert "SLURM_ARRAY_TASK_MIN" in text
            assert "SLURM_ARRAY_TASK_MAX" in text
        assert "--job-map-sha256" in text
        assert "--array-index" in text
        assert "--array-min" in text
        assert "--array-max" in text
        assert "run-row" in text


def test_confirmation_launcher_requires_seal_and_claim_but_never_submits(
    experiment_root: Path,
) -> None:
    text = (experiment_root / "slurm" / "confirmation_array.slurm").read_text(
        encoding="utf-8"
    )
    assert "${CONFIRMATION_SEAL:?" in text
    assert "${CELL_CLAIM:?" in text
    assert "--confirmation-seal" in text
    assert "--cell-claim" in text
    executable = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert not any(line.startswith("sbatch ") for line in executable)


def test_job_map_sorts_rows_assigns_indices_and_seals_every_field(
    jobmap_module: ModuleType,
    tmp_path: Path,
) -> None:
    rows = [
        _row(
            tmp_path,
            role="control",
            config_id="s2-layernorm-off",
            subject=5,
            seed=43,
        ),
        _row(tmp_path, subject=1, seed=42),
    ]
    payload = jobmap_module.build_job_map(list(reversed(rows)))
    assert payload["schema_version"] == 1
    assert payload["payload_type"] == "samga_brain_rw.job_map"
    assert payload["array_bounds"] == [0, 1]
    assert payload["row_count"] == 2
    assert payload["payload_sha256"] == sha256_json(
        {key: value for key, value in payload.items() if key != "payload_sha256"}
    )
    assert [row["array_index"] for row in payload["rows"]] == [0, 1]
    assert payload["rows"] == sorted(
        payload["rows"],
        key=jobmap_module.job_row_sort_key,
    )
    assert set(payload["rows"][0]) == {
        "array_index",
        "stage",
        "role",
        "config_id",
        "config_sha256",
        "input_bundle_sha256",
        "run_key",
        "subject",
        "seed",
        "argv",
        "partition",
        "gres",
        "cpus",
        "memory",
        "time",
        "stdout_path",
        "stderr_path",
        "completion_path",
        "expected_completion_schema",
    }

    path = tmp_path / "stage2-pilot-map.json"
    written = jobmap_module.write_job_map(rows, path)
    assert path.exists()
    assert (
        jobmap_module.load_job_map(path, expected_sha256=written["payload_sha256"])
        == written
    )
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        jobmap_module.write_job_map(rows, path)
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "mutation",
    (
        lambda payload: payload["rows"].reverse(),
        lambda payload: payload["rows"][1].update({"array_index": 0}),
        lambda payload: payload["rows"][1].update({"array_index": 9}),
        lambda payload: payload.update({"array_bounds": [0, 9]}),
        lambda payload: payload["rows"].pop(),
    ),
)
def test_job_map_rejects_reordered_duplicate_or_out_of_range_rows(
    jobmap_module: ModuleType,
    tmp_path: Path,
    mutation: object,
) -> None:
    payload = jobmap_module.build_job_map(
        [
            _row(tmp_path, subject=1, seed=42),
            _row(tmp_path, subject=5, seed=43),
        ]
    )
    mutation(payload)  # type: ignore[operator]
    _rehash(payload)
    with pytest.raises(ValueError, match="row|array|sorted|count"):
        jobmap_module.validate_job_map(payload)


def test_job_map_rejects_duplicate_runs_mixed_resources_and_sealed_paths(
    jobmap_module: ModuleType,
    tmp_path: Path,
) -> None:
    row = _row(tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        jobmap_module.build_job_map([row, copy.deepcopy(row)])

    other_resource = _row(tmp_path, subject=5, seed=43)
    other_resource["time"] = "01:00:00"
    with pytest.raises(ValueError, match="homogeneous|resource"):
        jobmap_module.build_job_map([row, other_resource])

    for forbidden in (
        "/data/formal-test/cache.npy",
        "/data/test/cache.npy",
        "/data/sub-01_test.json",
    ):
        unsafe = copy.deepcopy(row)
        unsafe["argv"].append(forbidden)
        with pytest.raises(ValueError, match="sealed|forbidden"):
            jobmap_module.build_job_map([unsafe])

    wrong_logs = copy.deepcopy(row)
    wrong_logs["stdout_path"] = "logs/elsewhere/job.out"
    with pytest.raises(ValueError, match="logs/samga_brain_rw"):
        jobmap_module.build_job_map([wrong_logs])


def test_job_map_requires_lowercase_a40_and_debug_time_limit(
    jobmap_module: ModuleType,
    tmp_path: Path,
) -> None:
    uppercase = _row(tmp_path)
    uppercase["gres"] = "gpu:A40:1"
    with pytest.raises(ValueError, match="gpu:a40:1"):
        jobmap_module.build_job_map([uppercase])

    too_long = _row(tmp_path, time="00:30:01")
    with pytest.raises(ValueError, match="30"):
        jobmap_module.build_job_map([too_long])


def test_selected_row_requires_exact_map_hash_bounds_and_index(
    jobmap_module: ModuleType,
    tmp_path: Path,
) -> None:
    payload = jobmap_module.build_job_map(
        [
            _row(tmp_path, subject=1, seed=42),
            _row(tmp_path, subject=5, seed=43),
        ]
    )
    digest = payload["payload_sha256"]
    selected = jobmap_module.select_job_row(
        payload,
        expected_sha256=digest,
        array_index=1,
        array_min=0,
        array_max=1,
    )
    assert selected["array_index"] == 1
    assert "--subject" in selected["argv"]
    assert "--seed" in selected["argv"]
    assert "--config" in selected["argv"]

    with pytest.raises(ValueError, match="hash"):
        jobmap_module.select_job_row(
            payload,
            expected_sha256="0" * 64,
            array_index=1,
            array_min=0,
            array_max=1,
        )
    with pytest.raises(ValueError, match="bounds"):
        jobmap_module.select_job_row(
            payload,
            expected_sha256=digest,
            array_index=1,
            array_min=0,
            array_max=2,
        )
    with pytest.raises(ValueError, match="range|index"):
        jobmap_module.select_job_row(
            payload,
            expected_sha256=digest,
            array_index=2,
            array_min=0,
            array_max=1,
        )


@pytest.mark.parametrize(
    ("field", "flag", "replacement", "match"),
    (
        ("stage", "--stage", "0", "stage"),
        ("role", "--role", "control", "role"),
        ("config_id", "--config-id", "different", "config_id"),
        ("config_sha256", "--expected-config-sha256", "0" * 64, "config"),
        (
            "input_bundle_sha256",
            "--expected-input-bundle-sha256",
            "0" * 64,
            "input",
        ),
        ("run_key", "--run-key", "different", "run_key"),
        ("run_key", "--output-dir", "/tmp/different", "output|run_key"),
        ("config_id", "--candidate-id", "different", "candidate"),
    ),
)
def test_job_map_binds_runner_identity_fields_to_named_argv(
    jobmap_module: ModuleType,
    tmp_path: Path,
    field: str,
    flag: str,
    replacement: str,
    match: str,
) -> None:
    row = _row(tmp_path)
    argv = row["argv"]
    assert isinstance(argv, list)
    position = argv.index(flag)
    argv[position + 1] = replacement
    with pytest.raises(ValueError, match=match):
        jobmap_module.build_job_map([row])


def test_job_map_requires_one_shared_sealed_slurm_log_pattern(
    jobmap_module: ModuleType,
    tmp_path: Path,
) -> None:
    first = _row(tmp_path, subject=1, seed=42)
    second = _row(tmp_path, subject=5, seed=43)
    second["stdout_path"] = "logs/samga_brain_rw/different_%A_%a.out"
    with pytest.raises(ValueError, match="log|stdout|pattern|resource"):
        jobmap_module.build_job_map([first, second])

    invalid = _row(tmp_path)
    invalid["stderr_path"] = "logs/samga_brain_rw/missing-array.err"
    with pytest.raises(ValueError, match="%A|%a|pattern"):
        jobmap_module.build_job_map([invalid])


def test_training_stage_rejects_non_unified_runner_before_map_seal(
    jobmap_module: ModuleType,
    tmp_path: Path,
) -> None:
    row = _row(tmp_path)
    argv = row["argv"]
    assert isinstance(argv, list)
    argv[1] = "experiments/samga_brain_rw/train.py"
    with pytest.raises(ValueError, match="unified|runner"):
        jobmap_module.build_job_map([row])


def test_training_stage_binds_runner_to_declared_project_root(
    jobmap_module: ModuleType,
    tmp_path: Path,
) -> None:
    row = _row(tmp_path)
    argv = row["argv"]
    assert isinstance(argv, list)
    position = argv.index("--project-root")
    argv[position + 1] = str(tmp_path / "other-root")
    with pytest.raises(ValueError, match="project.root|runner"):
        jobmap_module.build_job_map([row])


def test_partial_retry_exports_full_map_bounds_and_uses_sealed_logs(
    experiment_root: Path,
    jobmap_module: ModuleType,
    submit_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    rows = [
        _row(
            tmp_path,
            subject=subject,
            seed=seed,
            partition="i64m1tga40u",
            time="04:00:00",
        )
        for subject, seed in ((1, 42), (5, 43), (8, 42))
    ]
    payload = jobmap_module.write_job_map(rows, tmp_path / "pilot.json")
    for row in (payload["rows"][0], payload["rows"][2]):
        jobmap_module.claim_job_row(payload, row)
        jobmap_module.complete_job_row(
            payload,
            row,
            {
                "final_checkpoint_sha256": _h(f"checkpoint-{row['array_index']}"),
                "parity_sha256": _h(f"parity-{row['array_index']}"),
                "run_manifest_sha256": _h(f"manifest-{row['array_index']}"),
            },
        )
    calls: list[list[str]] = []

    def fake_runner(command: list[str], **_: object) -> SimpleNamespace:
        calls.append(list(command))
        return SimpleNamespace(stdout="123\n", returncode=0)

    submit_module._submit(
        payload,
        job_map_path=tmp_path / "pilot.json",
        job_map_sha256=payload["payload_sha256"],
        slurm_script=experiment_root / "slurm" / "pilot_array.slurm",
        log_dir=Path("logs/samga_brain_rw"),
        rows=[payload["rows"][1]],
        runner=fake_runner,
    )
    command = calls[1]
    assert "--array=1-1" in command
    export = next(value for value in command if value.startswith("--export="))
    assert "JOB_MAP_ARRAY_MIN=0" in export
    assert "JOB_MAP_ARRAY_MAX=2" in export
    assert "--output=logs/samga_brain_rw/sealed_%A_%a.out" in command
    assert "--error=logs/samga_brain_rw/sealed_%A_%a.err" in command


def test_current_development_launchers_use_immutable_full_map_bounds(
    experiment_root: Path,
) -> None:
    for name in (
        "pilot_array.slurm",
        "preflight_debug.slurm",
        "online_parity_debug.slurm",
    ):
        text = (experiment_root / "slurm" / name).read_text(encoding="utf-8")
        assert "${JOB_MAP_ARRAY_MIN:?" in text
        assert "${JOB_MAP_ARRAY_MAX:?" in text
        assert 'ARRAY_MIN=${JOB_MAP_ARRAY_MIN:?' in text
        assert 'ARRAY_MAX=${JOB_MAP_ARRAY_MAX:?' in text


def test_run_row_exports_exact_array_context_to_child(
    jobmap_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    map_path = tmp_path / "run-row-map.json"
    payload = jobmap_module.write_job_map(
        [
            _row(tmp_path, subject=1, seed=42),
            _row(tmp_path, subject=5, seed=43),
        ],
        map_path,
    )
    row = payload["rows"][1]
    output_hashes = _output_hashes(row, "run-row")
    observed: dict[str, str] = {}

    def fake_run(command: object, **kwargs: object) -> SimpleNamespace:
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        observed.update(
            {
                name: environment[name]
                for name in JOB_ENVIRONMENT_NAMES
            }
        )
        assert command == row["argv"]
        assert kwargs["check"] is False
        jobmap_module.complete_job_row(payload, row, output_hashes)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(jobmap_module.subprocess, "run", fake_run)
    result = jobmap_module.main(
        [
            "run-row",
            "--job-map",
            str(map_path),
            "--job-map-sha256",
            str(payload["payload_sha256"]),
            "--array-index",
            "1",
            "--array-min",
            "0",
            "--array-max",
            "1",
        ]
    )
    assert result == 0
    assert observed == {
        "SAMGA_JOB_MAP": str(map_path),
        "SAMGA_JOB_MAP_SHA256": str(payload["payload_sha256"]),
        "SAMGA_JOB_ROW_SHA256": sha256_json(row),
        "SAMGA_JOB_CLAIM": observed["SAMGA_JOB_CLAIM"],
        "SAMGA_JOB_ARRAY_INDEX": "1",
        "SAMGA_JOB_ARRAY_MIN": "0",
        "SAMGA_JOB_ARRAY_MAX": "1",
    }
    assert Path(observed["SAMGA_JOB_CLAIM"]).is_file()
    assert jobmap_module.completion_is_valid(payload, row)


def test_complete_env_reloads_and_completes_only_selected_claim(
    jobmap_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload, row, _, environment = _claimed_job_environment(
        jobmap_module,
        tmp_path,
    )
    _install_job_environment(monkeypatch, environment)
    output_hashes = _output_hashes(row, "complete-env")

    result = jobmap_module.main(
        [
            "complete-env",
            "--output-hashes",
            json.dumps(output_hashes),
        ]
    )

    assert result == 0
    assert jobmap_module.completion_is_valid(payload, row)


def test_complete_env_rejects_recovery_before_completion_lock(
    jobmap_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload, row, _, environment = _claimed_job_environment(
        jobmap_module,
        tmp_path,
    )
    _install_job_environment(monkeypatch, environment)
    original_complete = jobmap_module.complete_job_row
    recovered: list[object] = []
    first_claim_path = Path(environment["SAMGA_JOB_CLAIM"])
    first_claim_sha256 = hashlib.sha256(
        first_claim_path.read_bytes()
    ).hexdigest()

    def recover_before_completion_lock(
        candidate_payload: object,
        candidate_row: object,
        output_hashes: object,
        **expected_claim: object,
    ) -> object:
        second = jobmap_module.recover_job_row(
            candidate_payload,
            candidate_row,
            recovery_audit_sha256=_h("race-recovery-audit"),
        )
        recovered.append(second)
        return original_complete(
            candidate_payload,
            candidate_row,
            output_hashes,
            **expected_claim,
        )

    monkeypatch.setattr(
        jobmap_module,
        "complete_job_row",
        recover_before_completion_lock,
    )
    with pytest.raises(
        ValueError,
        match="claim identity.*changed|current claim",
    ):
        jobmap_module.main(
            [
                "complete-env",
                "--output-hashes",
                json.dumps(_output_hashes(row, "race-output")),
            ]
        )

    assert len(recovered) == 1
    second = recovered[0]
    assert second.generation == 2
    assert second.path != first_claim_path
    assert second.sha256 != first_claim_sha256
    assert not Path(row["completion_path"]).exists()


@pytest.mark.parametrize("missing_name", JOB_ENVIRONMENT_NAMES)
def test_complete_env_rejects_each_missing_context_variable(
    jobmap_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    missing_name: str,
) -> None:
    _, row, _, environment = _claimed_job_environment(
        jobmap_module,
        tmp_path,
    )
    _install_job_environment(monkeypatch, environment)
    monkeypatch.delenv(missing_name)

    with pytest.raises(ValueError, match=missing_name):
        jobmap_module.main(
            [
                "complete-env",
                "--output-hashes",
                json.dumps(_output_hashes(row, "missing-env")),
            ]
        )
    assert not Path(row["completion_path"]).exists()


@pytest.mark.parametrize(
    ("tamper", "match"),
    (
        ("map-path", "SAMGA_JOB_MAP|sealed|read"),
        ("map-sha256", "hash"),
        ("row-sha256", "row hash"),
        ("claim-path", "claim path"),
        ("array-index-invalid", "SAMGA_JOB_ARRAY_INDEX"),
        ("array-index-range", "index|range"),
        ("array-min", "bounds"),
        ("array-max", "bounds"),
    ),
)
def test_complete_env_rejects_tampered_context(
    jobmap_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tamper: str,
    match: str,
) -> None:
    _, row, _, environment = _claimed_job_environment(
        jobmap_module,
        tmp_path,
    )
    if tamper == "map-path":
        environment["SAMGA_JOB_MAP"] = str(tmp_path / "missing-map.json")
    elif tamper == "map-sha256":
        environment["SAMGA_JOB_MAP_SHA256"] = _h("tampered-map")
    elif tamper == "row-sha256":
        environment["SAMGA_JOB_ROW_SHA256"] = _h("tampered-row")
    elif tamper == "claim-path":
        other_map = jobmap_module.build_job_map(
            [_row(tmp_path, subject=5, seed=43)]
        )
        other_row = other_map["rows"][0]
        other_claim = jobmap_module.claim_job_row(other_map, other_row)
        environment["SAMGA_JOB_CLAIM"] = str(other_claim.path)
    elif tamper == "array-index-invalid":
        environment["SAMGA_JOB_ARRAY_INDEX"] = "not-an-integer"
    elif tamper == "array-index-range":
        environment["SAMGA_JOB_ARRAY_INDEX"] = "1"
    elif tamper == "array-min":
        environment["SAMGA_JOB_ARRAY_MIN"] = "1"
    elif tamper == "array-max":
        environment["SAMGA_JOB_ARRAY_MAX"] = "1"
    else:
        raise AssertionError(tamper)
    _install_job_environment(monkeypatch, environment)

    with pytest.raises(ValueError, match=match):
        jobmap_module.main(
            [
                "complete-env",
                "--output-hashes",
                json.dumps(_output_hashes(row, "tampered-env")),
            ]
        )
    assert not Path(row["completion_path"]).exists()


def test_completion_is_idempotent_and_prevents_resubmission(
    jobmap_module: ModuleType,
    tmp_path: Path,
) -> None:
    payload = jobmap_module.build_job_map([_row(tmp_path)])
    row = payload["rows"][0]
    assert jobmap_module.completion_is_valid(payload, row) is False
    claim = jobmap_module.claim_job_row(payload, row)
    with pytest.raises(RuntimeError, match="recovery|active|stale"):
        jobmap_module.claim_job_row(payload, row)

    completion = jobmap_module.complete_job_row(
        payload,
        row,
        _output_hashes(row, "metrics"),
    )
    assert completion.path == Path(row["completion_path"])
    assert jobmap_module.completion_is_valid(payload, row) is True
    assert jobmap_module.should_submit_row(payload, row) is False
    original = completion.path.read_bytes()
    repeated = jobmap_module.complete_job_row(
        payload,
        row,
        _output_hashes(row, "metrics"),
    )
    assert repeated.sha256 == completion.sha256
    assert completion.path.read_bytes() == original
    assert claim.path.exists()


def test_stale_claim_recovery_is_audited_and_never_deletes_original(
    jobmap_module: ModuleType,
    tmp_path: Path,
) -> None:
    payload = jobmap_module.build_job_map([_row(tmp_path)])
    row = payload["rows"][0]
    first = jobmap_module.claim_job_row(payload, row)
    first_bytes = first.path.read_bytes()
    second = jobmap_module.recover_job_row(
        payload,
        row,
        recovery_audit_sha256=_h("audited-stale-claim"),
    )
    assert second.generation == 2
    assert first.path.read_bytes() == first_bytes
    assert first.recovery_path.exists()
    recovery = json.loads(first.recovery_path.read_text(encoding="utf-8"))
    assert recovery["payload"]["claim_sha256"] == first.sha256
    assert recovery["payload"]["recovery_audit_sha256"] == _h("audited-stale-claim")
    assert second.document["payload"]["recovered_from_claim_sha256"] == first.sha256
    assert second.path.exists()

    jobmap_module.complete_job_row(
        payload,
        row,
        _output_hashes(row, "recovered-metrics"),
    )
    assert jobmap_module.completion_is_valid(payload, row)


def test_submitter_checks_queue_and_submits_debug_smoke_before_full_pilot(
    experiment_root: Path,
    jobmap_module: ModuleType,
    submit_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    smoke = jobmap_module.write_job_map(
        [_row(tmp_path, stage="stage-2-smoke")],
        tmp_path / "smoke-map.json",
    )
    pilot = jobmap_module.write_job_map(
        [
            _row(
                tmp_path,
                subject=1,
                seed=42,
                partition="i64m1tga40u",
                time="04:00:00",
            ),
            _row(
                tmp_path,
                subject=5,
                seed=43,
                partition="i64m1tga40u",
                time="04:00:00",
            ),
        ],
        tmp_path / "pilot-map.json",
    )
    calls: list[list[str]] = []

    def fake_runner(command: list[str], **_: object) -> SimpleNamespace:
        calls.append(list(command))
        return SimpleNamespace(stdout="12345\n", returncode=0)

    script = experiment_root / "slurm" / "pilot_array.slurm"
    phase = submit_module.submit_available_pilot(
        smoke_job_map=tmp_path / "smoke-map.json",
        smoke_sha256=smoke["payload_sha256"],
        pilot_job_map=tmp_path / "pilot-map.json",
        pilot_sha256=pilot["payload_sha256"],
        slurm_script=script,
        log_dir=Path("logs/samga_brain_rw"),
        runner=fake_runner,
    )
    assert phase == "smoke-submitted"
    assert calls[0] == QUEUE_COMMAND
    assert calls[1][0] == "sbatch"
    assert "--partition=debug" in calls[1]
    assert "--gres=gpu:a40:1" in calls[1]
    assert "--array=0-0" in calls[1]
    assert str(script) == calls[1][-1]
    assert not any("i64m1tga40u" in item for item in calls[1][1:])

    smoke_row = smoke["rows"][0]
    jobmap_module.claim_job_row(smoke, smoke_row)
    jobmap_module.complete_job_row(
        smoke,
        smoke_row,
        _output_hashes(smoke_row, "smoke"),
    )
    calls.clear()
    phase = submit_module.submit_available_pilot(
        smoke_job_map=tmp_path / "smoke-map.json",
        smoke_sha256=smoke["payload_sha256"],
        pilot_job_map=tmp_path / "pilot-map.json",
        pilot_sha256=pilot["payload_sha256"],
        slurm_script=script,
        log_dir=Path("logs/samga_brain_rw"),
        runner=fake_runner,
    )
    assert phase == "pilot-submitted"
    assert calls[0] == QUEUE_COMMAND
    assert "--partition=i64m1tga40u" in calls[1]
    assert "--array=0-1" in calls[1]

    for row in pilot["rows"]:
        jobmap_module.claim_job_row(pilot, row)
        jobmap_module.complete_job_row(
            pilot,
            row,
            _output_hashes(row, f"pilot:{row['array_index']}"),
        )
    calls.clear()
    phase = submit_module.submit_available_pilot(
        smoke_job_map=tmp_path / "smoke-map.json",
        smoke_sha256=smoke["payload_sha256"],
        pilot_job_map=tmp_path / "pilot-map.json",
        pilot_sha256=pilot["payload_sha256"],
        slurm_script=script,
        log_dir=Path("logs/samga_brain_rw"),
        runner=fake_runner,
    )
    assert phase == "already-complete"
    assert calls == []


def test_submitter_refuses_confirmation_stage_without_any_command(
    experiment_root: Path,
    jobmap_module: ModuleType,
    submit_module: ModuleType,
    tmp_path: Path,
) -> None:
    confirmation = jobmap_module.write_job_map(
        [_row(tmp_path, stage="confirmation", training_runner=False)],
        tmp_path / "confirmation-map.json",
    )
    smoke = jobmap_module.write_job_map(
        [_row(tmp_path, stage="stage-2-smoke")],
        tmp_path / "smoke-map.json",
    )
    calls: list[list[str]] = []

    def fake_runner(command: list[str], **_: object) -> SimpleNamespace:
        calls.append(list(command))
        return SimpleNamespace(stdout="", returncode=0)

    with pytest.raises(ValueError, match="confirmation|current stage"):
        submit_module.submit_available_pilot(
            smoke_job_map=tmp_path / "smoke-map.json",
            smoke_sha256=smoke["payload_sha256"],
            pilot_job_map=tmp_path / "confirmation-map.json",
            pilot_sha256=confirmation["payload_sha256"],
            slurm_script=experiment_root / "slurm" / "confirmation_array.slurm",
            log_dir=tmp_path / "logs" / "samga_brain_rw",
            runner=fake_runner,
        )
    assert calls == []
