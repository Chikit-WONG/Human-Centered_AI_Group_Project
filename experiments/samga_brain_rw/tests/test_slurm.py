from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import pwd
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from samga_brain_rw.config import make_run_key
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
    _load_script(
        experiment_root / "scripts" / "run_training_cell.py",
        "run_training_cell",
    )
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
    project_root: Path | None = None,
) -> dict[str, object]:
    stage_number = 2 if "stage-2" in stage else 0
    selected_mode = mode or ("smoke" if "smoke" in stage else "full")
    config_sha256 = _h(f"config:{config_id}")
    input_bundle_sha256 = _h(f"input:{subject}:{seed}")
    run_key = make_run_key(
        f"stage{stage_number}",
        config_id,
        subject,
        seed,
        config_sha256,
        input_bundle_sha256,
    )
    selected_project_root = (
        project_root or tmp_path / "project-root"
    ).resolve()
    selected_project_root.mkdir(parents=True, exist_ok=True)
    output_dir = (
        selected_project_root
        / "artifacts"
        / "samga_brain_rw"
        / stage
        / run_key
    )
    argv = [
        "python",
        str(
            selected_project_root
            / "experiments/samga_brain_rw/scripts/run_training_cell.py"
        ),
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
        str(selected_project_root),
        "--config-id",
        config_id,
        "--expected-config-sha256",
        config_sha256,
        "--expected-input-bundle-sha256",
        input_bundle_sha256,
        "--run-key",
        run_key,
        "--device",
        "cuda",
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
        "completion_path": str(output_dir / "completion.json"),
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


def _submission_project(
    tmp_path: Path,
    experiment_root: Path,
) -> tuple[Path, Path]:
    project_root = (tmp_path / "project-root").resolve()
    project_root.mkdir(parents=True, exist_ok=True)
    (project_root / ".git").mkdir(exist_ok=True)
    relative_script = Path(
        "experiments/samga_brain_rw/slurm/pilot_array.slurm"
    )
    source = experiment_root / "slurm" / "pilot_array.slurm"
    script = project_root / relative_script
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_bytes(source.read_bytes())
    return project_root, script


def _slurm_recovery_case(
    *,
    tmp_path: Path,
    experiment_root: Path,
    jobmap_module: ModuleType,
    submit_module: ModuleType,
) -> tuple[
    dict[str, object],
    dict[str, object],
    Path,
    object,
    list[str],
    str,
]:
    project_root, script = _submission_project(tmp_path, experiment_root)
    map_path = project_root / "artifacts/smoke-map.json"
    map_path.parent.mkdir(parents=True)
    payload = jobmap_module.write_job_map(
        [
            _row(
                tmp_path,
                stage="stage-0-smoke",
                role="baseline",
                project_root=project_root,
            )
        ],
        map_path,
    )
    row = payload["rows"][0]
    claim = jobmap_module.claim_job_row(payload, row)
    claim_stat = claim.path.stat()
    start = datetime.fromtimestamp(claim_stat.st_mtime) - timedelta(seconds=2)
    end = datetime.fromtimestamp(claim_stat.st_mtime) + timedelta(seconds=2)
    job_id = "12345_0"
    command = submit_module._resource_command(
        payload,
        job_map_path=map_path,
        job_map_sha256=str(payload["payload_sha256"]),
        slurm_script=script,
        log_dir=Path("logs/samga_brain_rw"),
        indices=[0],
    )
    username = pwd.getpwuid(os.getuid()).pw_name
    fields = jobmap_module.SLURM_RECOVERY_SACCT_FIELDS
    values = {
        "JobIDRaw": "12345",
        "JobID": job_id,
        "State": "FAILED",
        "ExitCode": "2:0",
        "DerivedExitCode": "0:0",
        "Submit": (start - timedelta(seconds=1)).isoformat(timespec="seconds"),
        "Eligible": start.isoformat(timespec="seconds"),
        "Start": start.isoformat(timespec="seconds"),
        "End": end.isoformat(timespec="seconds"),
        "ElapsedRaw": "4",
        "Partition": str(row["partition"]),
        "Account": "root",
        "QOS": "debug",
        "UID": str(os.getuid()),
        "User": username,
        "JobName": "samga-pilot",
        "NodeList": "gpu-test",
        "AllocTRES": "billing=8,cpu=8,gres/gpu:a40=1,gres/gpu=1,mem=64G,node=1",
        "ReqTRES": "billing=8,cpu=8,gres/gpu:a40=1,gres/gpu=1,mem=64G,node=1",
        "TimelimitRaw": "30",
        "WorkDir": str(project_root),
        "SubmitLine": " ".join(command),
        "Cluster": "test-cluster",
    }
    sacct_line = "|".join(values[field] for field in fields) + "\n"
    for stream_name in ("stdout_path", "stderr_path"):
        pattern = project_root / str(row[stream_name])
        concrete = Path(
            str(pattern).replace("%A", "12345").replace("%a", "0")
        )
        concrete.parent.mkdir(parents=True, exist_ok=True)
        concrete.write_bytes(f"{stream_name}\n".encode())
    Path(str(row["completion_path"])).parent.mkdir(parents=True)
    return payload, row, map_path, claim, command, sacct_line


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
        assert "PROJECT_ROOT=${PROJECT_ROOT:?" in text
        assert 'cd -- "${PROJECT_ROOT}"' in text
        assert (
            'export PYTHONPATH="${PROJECT_ROOT}/'
            'experiments/samga_brain_rw"'
        ) in text
        assert (
            '"${PROJECT_ROOT}/experiments/samga_brain_rw/'
            'scripts/build_job_map.py" run-row'
        ) in text
        assert "export HF_DATASETS_OFFLINE=1" in text
        assert "export TRANSFORMERS_OFFLINE=1" in text
        assert "export HF_HUB_OFFLINE=1" in text
        source = "source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh"
        assert source in text
        assert "conda activate test" in text
        disable_nounset = text.index("set +u")
        source_index = text.index(source)
        activate_index = text.index("conda activate test")
        restore_nounset = text.index("set -u", activate_index)
        assert disable_nounset < source_index < activate_index < restore_nounset
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


def test_training_row_rejects_output_dir_with_same_run_key_in_other_parent(
    jobmap_module: ModuleType,
    tmp_path: Path,
) -> None:
    row = _row(tmp_path)
    argv = row["argv"]
    assert isinstance(argv, list)
    output_position = argv.index("--output-dir") + 1
    output_dir = Path(argv[output_position])
    argv[output_position] = str(
        output_dir.parent.parent / "other-stage" / output_dir.name
    )

    with pytest.raises(ValueError, match="completion|output"):
        jobmap_module.build_job_map([row])


def test_training_row_accepts_bound_output_below_results_root(
    jobmap_module: ModuleType,
    tmp_path: Path,
) -> None:
    row = _row(tmp_path)
    argv = row["argv"]
    assert isinstance(argv, list)
    project_root = Path(
        argv[argv.index("--project-root") + 1]
    )
    output_dir = (
        project_root
        / "results"
        / "samga_brain_rw"
        / str(row["stage"])
        / str(row["run_key"])
    )
    argv[argv.index("--output-dir") + 1] = str(output_dir)
    row["completion_path"] = str(output_dir / "completion.json")

    assert jobmap_module.build_job_map([row])["row_count"] == 1


def test_training_row_rejects_bound_paths_outside_development_roots(
    jobmap_module: ModuleType,
    tmp_path: Path,
) -> None:
    row = _row(tmp_path)
    argv = row["argv"]
    assert isinstance(argv, list)
    project_root = Path(
        argv[argv.index("--project-root") + 1]
    )
    output_dir = (
        project_root
        / "unsealed-output"
        / str(row["run_key"])
    )
    argv[argv.index("--output-dir") + 1] = str(output_dir)
    row["completion_path"] = str(output_dir / "completion.json")

    with pytest.raises(
        ValueError,
        match="artifacts|results|development|project.root|outside",
    ):
        jobmap_module.build_job_map([row])


def test_training_row_rejects_noncanonical_dotdot_paths(
    jobmap_module: ModuleType,
    tmp_path: Path,
) -> None:
    row = _row(tmp_path)
    argv = row["argv"]
    assert isinstance(argv, list)
    output_dir = Path(argv[argv.index("--output-dir") + 1])
    noncanonical = (
        f"{output_dir.parent.as_posix()}/intermediate/../"
        f"{output_dir.name}"
    )
    argv[argv.index("--output-dir") + 1] = noncanonical
    row["completion_path"] = f"{noncanonical}/completion.json"

    with pytest.raises(ValueError, match="normalized|canonical|\\.\\."):
        jobmap_module.build_job_map([row])


def test_legacy_row_rejects_relative_completion_path_during_build(
    jobmap_module: ModuleType,
    tmp_path: Path,
) -> None:
    row = _row(
        tmp_path,
        stage="confirmation",
        training_runner=False,
    )
    row["completion_path"] = "artifacts/run/completion.json"

    with pytest.raises(ValueError, match="completion_path.*absolute|normalized"):
        jobmap_module.build_job_map([row])


def test_training_row_rejects_symlinked_output_containment(
    jobmap_module: ModuleType,
    tmp_path: Path,
) -> None:
    row = _row(tmp_path)
    argv = row["argv"]
    assert isinstance(argv, list)
    project_root = Path(
        argv[argv.index("--project-root") + 1]
    )
    development_root = (
        project_root / "artifacts" / "samga_brain_rw"
    )
    development_root.mkdir(parents=True)
    outside = tmp_path / "outside-output"
    outside.mkdir()
    escape = development_root / "escape"
    escape.symlink_to(outside, target_is_directory=True)
    output_dir = escape / str(row["run_key"])
    argv[argv.index("--output-dir") + 1] = str(output_dir)
    row["completion_path"] = str(output_dir / "completion.json")

    with pytest.raises(ValueError, match="symlink|normalized|outside"):
        jobmap_module.build_job_map([row])


def test_unified_training_row_seals_explicit_cuda_device(
    jobmap_module: ModuleType,
    tmp_path: Path,
) -> None:
    row = _row(tmp_path)
    argv = row["argv"]
    assert isinstance(argv, list)
    assert argv.count("--device") == 1
    assert argv[argv.index("--device") + 1] == "cuda"
    payload = jobmap_module.build_job_map([row])
    sealed_argv = payload["rows"][0]["argv"]
    assert isinstance(sealed_argv, list)
    assert sealed_argv[sealed_argv.index("--device") + 1] == "cuda"


def test_unified_training_row_rejects_legacy_unprefixed_run_key(
    jobmap_module: ModuleType,
    tmp_path: Path,
) -> None:
    row = _row(tmp_path)
    canonical = str(row["run_key"])
    assert "__config-" in canonical
    assert "__inputs-" in canonical
    assert jobmap_module.build_job_map([row])["row_count"] == 1

    legacy = (
        f"stage2__s2-layernorm-on__sub-01__seed-42__"
        f"{row['config_sha256']}__{row['input_bundle_sha256']}"
    )
    row["run_key"] = legacy
    argv = row["argv"]
    assert isinstance(argv, list)
    argv[argv.index("--run-key") + 1] = legacy
    output_position = argv.index("--output-dir") + 1
    argv[output_position] = str(
        Path(argv[output_position]).parent / legacy
    )
    row["completion_path"] = str(
        Path(str(row["completion_path"])).parent.parent
        / legacy
        / "completion.json"
    )

    with pytest.raises(ValueError, match="run_key"):
        jobmap_module.build_job_map([row])


@pytest.mark.parametrize("unsupported", ("auto", "cpu"))
def test_unified_training_row_rejects_non_cuda_device(
    jobmap_module: ModuleType,
    tmp_path: Path,
    unsupported: str,
) -> None:
    row = _row(tmp_path)
    argv = row["argv"]
    assert isinstance(argv, list)
    argv[argv.index("--device") + 1] = unsupported
    with pytest.raises(ValueError, match="device|cuda"):
        jobmap_module.build_job_map([row])


def test_unified_training_row_rejects_missing_device_but_legacy_is_unchanged(
    jobmap_module: ModuleType,
    tmp_path: Path,
) -> None:
    row = _row(tmp_path)
    argv = row["argv"]
    assert isinstance(argv, list)
    position = argv.index("--device")
    del argv[position : position + 2]
    with pytest.raises(ValueError, match="device|cuda"):
        jobmap_module.build_job_map([row])

    legacy = _row(tmp_path, stage="confirmation", training_runner=False)
    assert "--device" not in legacy["argv"]
    assert jobmap_module.build_job_map([legacy])["row_count"] == 1


def test_partial_retry_exports_full_map_bounds_and_uses_sealed_logs(
    experiment_root: Path,
    jobmap_module: ModuleType,
    submit_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    project_root, script = _submission_project(
        tmp_path, experiment_root
    )
    rows = [
        _row(
            tmp_path,
            subject=subject,
            seed=seed,
            partition="i64m1tga40u",
            time="04:00:00",
            project_root=project_root,
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
        slurm_script=script,
        log_dir=Path("logs/samga_brain_rw"),
        rows=[payload["rows"][1]],
        runner=fake_runner,
    )
    command = calls[1]
    assert "--array=1-1" in command
    assert f"--chdir={project_root}" in command
    export = next(value for value in command if value.startswith("--export="))
    assert "JOB_MAP_ARRAY_MIN=0" in export
    assert "JOB_MAP_ARRAY_MAX=2" in export
    assert f"PROJECT_ROOT={project_root}" in export
    assert (
        f"--output={project_root}/logs/samga_brain_rw/"
        "sealed_%A_%a.out"
    ) in command
    assert (
        f"--error={project_root}/logs/samga_brain_rw/"
        "sealed_%A_%a.err"
    ) in command
    assert command[-1] == str(script)


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


def test_public_claim_cli_is_not_exposed_and_cannot_mutate_state(
    jobmap_module: ModuleType,
    tmp_path: Path,
) -> None:
    map_path = tmp_path / "no-public-claim-map.json"
    payload = jobmap_module.write_job_map([_row(tmp_path)], map_path)
    row = payload["rows"][0]

    with pytest.raises(SystemExit) as rejected:
        jobmap_module.main(
            [
                "claim",
                "--job-map",
                str(map_path),
                "--job-map-sha256",
                str(payload["payload_sha256"]),
                "--array-index",
                "0",
                "--array-min",
                "0",
                "--array-max",
                "0",
            ]
        )

    assert rejected.value.code == 2
    assert not jobmap_module._state_dir(row).exists()
    assert not Path(str(row["completion_path"])).exists()


def test_public_complete_cli_is_not_exposed_and_cannot_mutate_state(
    jobmap_module: ModuleType,
    tmp_path: Path,
) -> None:
    map_path = tmp_path / "no-public-complete-map.json"
    payload = jobmap_module.write_job_map([_row(tmp_path)], map_path)
    row = payload["rows"][0]
    claim = jobmap_module.claim_job_row(payload, row)
    claim_bytes = claim.path.read_bytes()

    with pytest.raises(SystemExit) as rejected:
        jobmap_module.main(
            [
                "complete",
                "--job-map",
                str(map_path),
                "--job-map-sha256",
                str(payload["payload_sha256"]),
                "--array-index",
                "0",
                "--array-min",
                "0",
                "--array-max",
                "0",
                "--output-hashes",
                json.dumps(_output_hashes(row, "unbound-public-cli")),
            ]
        )

    assert rejected.value.code == 2
    assert claim.path.read_bytes() == claim_bytes
    assert not Path(str(row["completion_path"])).exists()


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
        second = jobmap_module._recover_job_row_unverified_for_testing(
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


def test_claim_rejects_preexisting_job_claims_symlink_without_external_write(
    jobmap_module: ModuleType,
    tmp_path: Path,
) -> None:
    payload = jobmap_module.build_job_map([_row(tmp_path)])
    row = payload["rows"][0]
    state_dir = jobmap_module._state_dir(row)
    claims_root = state_dir.parent
    claims_root.parent.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside-claims"
    outside.mkdir()
    claims_root.symlink_to(outside, target_is_directory=True)

    with pytest.raises(
        ValueError,
        match="symlink|symbolic|state|directory",
    ):
        jobmap_module.claim_job_row(payload, row)

    assert list(outside.iterdir()) == []


def test_claim_rejects_generation_symlink_injected_after_enumeration(
    jobmap_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = jobmap_module.build_job_map([_row(tmp_path)])
    row = payload["rows"][0]
    state_dir = jobmap_module._state_dir(row)
    generation_path = state_dir / "generation-000001"
    outside = tmp_path / "outside-injected-generation"
    outside.mkdir()
    original_generation_numbers = jobmap_module._generation_numbers
    injected = False

    def inject_generation_symlink(directory: object) -> list[int]:
        nonlocal injected
        numbers = original_generation_numbers(directory)
        if not injected:
            generation_path.symlink_to(
                outside,
                target_is_directory=True,
            )
            injected = True
        return numbers

    monkeypatch.setattr(
        jobmap_module,
        "_generation_numbers",
        inject_generation_symlink,
    )

    with pytest.raises(
        ValueError,
        match="symlink|symbolic|state|generation|directory",
    ):
        jobmap_module.claim_job_row(payload, row)

    assert injected is True
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("failure_point", ("claim-publication", "rename"))
def test_first_claim_staging_failure_leaves_no_final_generation_and_retries(
    jobmap_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_point: str,
) -> None:
    payload = jobmap_module.build_job_map([_row(tmp_path)])
    row = payload["rows"][0]
    state_dir = jobmap_module._state_dir(row)
    final_generation = state_dir / "generation-000001"
    original_publish = jobmap_module._exclusive_publish_at
    original_rename = jobmap_module._rename_staged_generation

    def fail_claim_publication(
        directory_fd: int,
        name: str,
        data: bytes,
    ) -> None:
        if failure_point == "claim-publication" and name == "claim.json":
            raise RuntimeError("injected staged claim publication failure")
        original_publish(directory_fd, name, data)

    def fail_staged_rename(
        state_fd: int,
        staging_name: str,
        final_name: str,
    ) -> None:
        if failure_point == "rename":
            raise RuntimeError("injected staged generation rename failure")
        original_rename(state_fd, staging_name, final_name)

    monkeypatch.setattr(
        jobmap_module,
        "_exclusive_publish_at",
        fail_claim_publication,
    )
    monkeypatch.setattr(
        jobmap_module,
        "_rename_staged_generation",
        fail_staged_rename,
    )

    with pytest.raises(RuntimeError, match="injected staged"):
        jobmap_module.claim_job_row(payload, row)

    assert not final_generation.exists()
    assert not [
        entry
        for entry in state_dir.iterdir()
        if entry.name.startswith(".generation-staging-")
    ]

    monkeypatch.setattr(
        jobmap_module,
        "_exclusive_publish_at",
        original_publish,
    )
    monkeypatch.setattr(
        jobmap_module,
        "_rename_staged_generation",
        original_rename,
    )
    claim = jobmap_module.claim_job_row(payload, row)
    assert claim.generation == 1
    assert claim.path.is_file()


def test_post_rename_parent_fsync_failure_preserves_complete_final_claim(
    jobmap_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = jobmap_module.build_job_map([_row(tmp_path)])
    row = payload["rows"][0]
    state_dir = jobmap_module._state_dir(row)
    final_generation = state_dir / "generation-000001"
    final_claim = final_generation / "claim.json"
    original_rename = jobmap_module.os.rename
    original_fsync = jobmap_module.os.fsync
    renamed = False
    state_fd_after_rename: int | None = None
    injected = False

    def observe_rename(
        source: object,
        destination: object,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        nonlocal renamed, state_fd_after_rename
        original_rename(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )
        if destination == "generation-000001":
            renamed = True
            state_fd_after_rename = dst_dir_fd

    def fail_first_parent_fsync_after_rename(fd: int) -> None:
        nonlocal injected
        if renamed and fd == state_fd_after_rename and not injected:
            injected = True
            raise OSError("injected post-rename parent fsync failure")
        original_fsync(fd)

    monkeypatch.setattr(jobmap_module.os, "rename", observe_rename)
    monkeypatch.setattr(jobmap_module.os, "fsync", fail_first_parent_fsync_after_rename)

    with pytest.raises(OSError, match="post-rename parent fsync"):
        jobmap_module.claim_job_row(payload, row)

    assert renamed is True
    assert injected is True
    assert final_generation.is_dir()
    assert final_claim.is_file()
    assert final_claim.stat().st_size > 0
    assert not [
        entry
        for entry in state_dir.iterdir()
        if entry.name.startswith(".generation-staging-")
    ]
    assert jobmap_module.should_submit_row(payload, row) is False
    with pytest.raises(RuntimeError, match="active|stale|recovery"):
        jobmap_module.claim_job_row(payload, row)


def test_recovery_generation_staging_failure_retries_same_generation(
    jobmap_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = jobmap_module.build_job_map([_row(tmp_path)])
    row = payload["rows"][0]
    first = jobmap_module.claim_job_row(payload, row)
    original_rename = jobmap_module._rename_staged_generation

    def fail_generation_two_rename(
        state_fd: int,
        staging_name: str,
        final_name: str,
    ) -> None:
        if final_name == "generation-000002":
            raise RuntimeError("injected recovery generation rename failure")
        original_rename(state_fd, staging_name, final_name)

    monkeypatch.setattr(
        jobmap_module,
        "_rename_staged_generation",
        fail_generation_two_rename,
    )
    audit = _h("generation-two-staging-failure")
    with pytest.raises(RuntimeError, match="injected recovery"):
        jobmap_module._recover_job_row_unverified_for_testing(
            payload,
            row,
            recovery_audit_sha256=audit,
        )

    second_path = first.path.parent.parent / "generation-000002"
    assert not second_path.exists()
    assert first.recovery_path.is_file()
    assert not [
        entry
        for entry in jobmap_module._state_dir(row).iterdir()
        if entry.name.startswith(".generation-staging-")
    ]

    monkeypatch.setattr(
        jobmap_module,
        "_rename_staged_generation",
        original_rename,
    )
    second = jobmap_module._recover_job_row_unverified_for_testing(
        payload,
        row,
        recovery_audit_sha256=audit,
    )
    assert second.generation == 2
    assert second.path.is_file()


def test_recovery_rejects_generation_one_symlink_without_external_write(
    jobmap_module: ModuleType,
    tmp_path: Path,
) -> None:
    payload = jobmap_module.build_job_map([_row(tmp_path)])
    row = payload["rows"][0]
    first = jobmap_module.claim_job_row(payload, row)
    first_bytes = first.path.read_bytes()
    generation_path = first.path.parent
    outside = tmp_path / "outside-generation-one"
    generation_path.rename(outside)
    generation_path.symlink_to(outside, target_is_directory=True)

    with pytest.raises(
        ValueError,
        match="symlink|symbolic|state|generation|directory",
    ):
        jobmap_module._recover_job_row_unverified_for_testing(
            payload,
            row,
            recovery_audit_sha256=_h("symlinked-generation-one"),
        )

    assert (outside / "claim.json").read_bytes() == first_bytes
    assert {entry.name for entry in outside.iterdir()} == {"claim.json"}
    assert not (
        jobmap_module._state_dir(row) / "generation-000002"
    ).exists()


def test_attempt_rejects_generation_two_symlink_without_external_write(
    jobmap_module: ModuleType,
    tmp_path: Path,
) -> None:
    payload = jobmap_module.build_job_map([_row(tmp_path)])
    row = payload["rows"][0]
    jobmap_module.claim_job_row(payload, row)
    second = jobmap_module._recover_job_row_unverified_for_testing(
        payload,
        row,
        recovery_audit_sha256=_h("symlinked-generation-two"),
    )
    second_bytes = second.path.read_bytes()
    generation_path = second.path.parent
    outside = tmp_path / "outside-generation-two"
    generation_path.rename(outside)
    generation_path.symlink_to(outside, target_is_directory=True)

    with pytest.raises(
        ValueError,
        match="symlink|symbolic|state|generation|directory",
    ):
        jobmap_module.consume_recovery_attempt(payload, row)

    assert (outside / "claim.json").read_bytes() == second_bytes
    assert {entry.name for entry in outside.iterdir()} == {"claim.json"}


def test_completion_output_hashes_returns_copy_or_none_and_rejects_tampering(
    jobmap_module: ModuleType,
    tmp_path: Path,
) -> None:
    payload = jobmap_module.build_job_map([_row(tmp_path)])
    row = payload["rows"][0]
    assert jobmap_module.completion_output_hashes(payload, row) is None

    jobmap_module.claim_job_row(payload, row)
    hashes = _output_hashes(row, "declared-outputs")
    jobmap_module.complete_job_row(payload, row, hashes)
    loaded = jobmap_module.completion_output_hashes(payload, row)
    assert loaded == hashes
    assert loaded is not hashes

    completion_path = Path(str(row["completion_path"]))
    document = json.loads(completion_path.read_text(encoding="utf-8"))
    document["payload"]["output_hashes"]["final_checkpoint_sha256"] = _h(
        "tampered-output"
    )
    completion_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical|hash|completion"):
        jobmap_module.completion_output_hashes(payload, row)


def test_unverified_recovery_helper_is_internal_only(
    jobmap_module: ModuleType,
) -> None:
    assert not hasattr(jobmap_module, "recover_job_row")
    assert callable(
        jobmap_module._recover_job_row_unverified_for_testing
    )


def test_unverified_recovery_is_not_exposed_by_cli(
    jobmap_module: ModuleType,
    tmp_path: Path,
) -> None:
    map_path = tmp_path / "no-recovery-cli-map.json"
    payload = jobmap_module.write_job_map(
        [_row(tmp_path)],
        map_path,
    )
    row = payload["rows"][0]
    first = jobmap_module.claim_job_row(payload, row)

    with pytest.raises(SystemExit):
        jobmap_module.main(
            [
                "recover",
                "--job-map",
                str(map_path),
                "--job-map-sha256",
                str(payload["payload_sha256"]),
                "--array-index",
                "0",
                "--array-min",
                "0",
                "--array-max",
                "0",
                "--recovery-audit-sha256",
                _h("unverified-cli-audit"),
            ]
        )

    assert not first.recovery_path.exists()


def test_slurm_recovery_publishes_typed_audit_and_fresh_retry(
    experiment_root: Path,
    jobmap_module: ModuleType,
    submit_module: ModuleType,
    tmp_path: Path,
) -> None:
    payload, row, map_path, first, _, sacct_line = _slurm_recovery_case(
        tmp_path=tmp_path,
        experiment_root=experiment_root,
        jobmap_module=jobmap_module,
        submit_module=submit_module,
    )
    calls: list[list[str]] = []

    def fake_runner(command: list[str], **_: object) -> SimpleNamespace:
        calls.append(list(command))
        stdout = sacct_line if command[0] == "sacct" else ""
        return SimpleNamespace(stdout=stdout, returncode=0)

    second = jobmap_module.recover_job_row_from_slurm(
        payload,
        row,
        job_map_path=map_path,
        failed_slurm_job="12345_0",
        runner=fake_runner,
    )

    assert [command[0] for command in calls] == [
        "sacct",
        "squeue",
        "sacct",
        "squeue",
        "sacct",
    ]
    assert second.generation == 2
    assert jobmap_module.should_submit_row(payload, row) is True
    assert not second.attempt_path.exists()
    audit_path = first.path.with_name("slurm-recovery-audit.json")
    audit_bytes = audit_path.read_bytes()
    audit = json.loads(audit_bytes)
    assert audit["payload_type"] == "samga_brain_rw.slurm_recovery_audit"
    assert audit["payload"]["binding_mode"] == "legacy_claim_scheduler_binding"
    assert audit["payload"]["scheduler"]["job_id"] == "12345_0"
    recovery = json.loads(first.recovery_path.read_text(encoding="utf-8"))
    assert recovery["payload"]["recovery_audit_type"] == audit["payload_type"]
    assert recovery["payload"]["recovery_audit_sha256"] == hashlib.sha256(
        audit_bytes
    ).hexdigest()
    quarantine = recovery["payload"]["quarantine"]
    assert quarantine["file_count"] == 0
    assert not Path(str(row["completion_path"])).parent.exists()

    repeated = jobmap_module.recover_job_row_from_slurm(
        payload,
        row,
        job_map_path=map_path,
        failed_slurm_job="12345_0",
        runner=fake_runner,
    )
    assert repeated.path == second.path
    assert not (
        second.path.parent.parent / "generation-000003"
    ).exists()


def test_slurm_recovery_resumes_after_typed_audit_publication_crash(
    experiment_root: Path,
    jobmap_module: ModuleType,
    submit_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload, row, map_path, first, _, sacct_line = _slurm_recovery_case(
        tmp_path=tmp_path,
        experiment_root=experiment_root,
        jobmap_module=jobmap_module,
        submit_module=submit_module,
    )

    def fake_runner(command: list[str], **_: object) -> SimpleNamespace:
        return SimpleNamespace(
            stdout=sacct_line if command[0] == "sacct" else "",
            returncode=0,
        )

    transition = jobmap_module._transition_to_recovered_claim_locked

    def crash_after_audit(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("injected post-audit crash")

    monkeypatch.setattr(
        jobmap_module,
        "_transition_to_recovered_claim_locked",
        crash_after_audit,
    )
    with pytest.raises(RuntimeError, match="post-audit"):
        jobmap_module.recover_job_row_from_slurm(
            payload,
            row,
            job_map_path=map_path,
            failed_slurm_job="12345_0",
            runner=fake_runner,
        )
    audit_path = first.path.with_name("slurm-recovery-audit.json")
    audit_bytes = audit_path.read_bytes()
    assert not first.recovery_path.exists()

    monkeypatch.setattr(
        jobmap_module,
        "_transition_to_recovered_claim_locked",
        transition,
    )
    second = jobmap_module.recover_job_row_from_slurm(
        payload,
        row,
        job_map_path=map_path,
        failed_slurm_job="12345_0",
        runner=fake_runner,
    )
    assert second.generation == 2
    assert audit_path.read_bytes() == audit_bytes


def test_requeued_failed_job_cannot_consume_recovered_attempt(
    experiment_root: Path,
    jobmap_module: ModuleType,
    submit_module: ModuleType,
    tmp_path: Path,
) -> None:
    payload, row, map_path, _, _, sacct_line = _slurm_recovery_case(
        tmp_path=tmp_path,
        experiment_root=experiment_root,
        jobmap_module=jobmap_module,
        submit_module=submit_module,
    )

    def fake_runner(command: list[str], **_: object) -> SimpleNamespace:
        return SimpleNamespace(
            stdout=sacct_line if command[0] == "sacct" else "",
            returncode=0,
        )

    second = jobmap_module.recover_job_row_from_slurm(
        payload,
        row,
        job_map_path=map_path,
        failed_slurm_job="12345_0",
        runner=fake_runner,
    )
    with pytest.raises(RuntimeError, match="failed|old|recovery|requeue"):
        jobmap_module.consume_recovery_attempt(
            payload,
            row,
            scheduler_job_id="12345_0",
        )
    assert not second.attempt_path.exists()

    jobmap_module.consume_recovery_attempt(
        payload,
        row,
        scheduler_job_id="12346_0",
    )
    attempt = json.loads(second.attempt_path.read_text(encoding="utf-8"))
    assert attempt["payload"]["scheduler_job_id"] == "12346_0"


def test_later_slurm_recovery_binds_consumed_attempt_scheduler_identity(
    experiment_root: Path,
    jobmap_module: ModuleType,
    submit_module: ModuleType,
    tmp_path: Path,
) -> None:
    payload, row, map_path, _, _, first_sacct = _slurm_recovery_case(
        tmp_path=tmp_path,
        experiment_root=experiment_root,
        jobmap_module=jobmap_module,
        submit_module=submit_module,
    )

    def first_runner(command: list[str], **_: object) -> SimpleNamespace:
        return SimpleNamespace(
            stdout=first_sacct if command[0] == "sacct" else "",
            returncode=0,
        )

    second = jobmap_module.recover_job_row_from_slurm(
        payload,
        row,
        job_map_path=map_path,
        failed_slurm_job="12345_0",
        runner=first_runner,
    )
    jobmap_module.consume_recovery_attempt(
        payload,
        row,
        scheduler_job_id="12346_0",
    )
    attempt_mtime = second.attempt_path.stat().st_mtime
    start = datetime.fromtimestamp(attempt_mtime) - timedelta(seconds=2)
    end = datetime.fromtimestamp(attempt_mtime) + timedelta(seconds=2)
    fields = list(jobmap_module.SLURM_RECOVERY_SACCT_FIELDS)
    values = first_sacct.removesuffix("\n").split("|")
    replacements = {
        "JobIDRaw": "12346",
        "JobID": "12346_0",
        "Submit": (start - timedelta(seconds=1)).isoformat(timespec="seconds"),
        "Eligible": start.isoformat(timespec="seconds"),
        "Start": start.isoformat(timespec="seconds"),
        "End": end.isoformat(timespec="seconds"),
        "ElapsedRaw": "4",
    }
    for field, value in replacements.items():
        values[fields.index(field)] = value
    second_sacct = "|".join(values) + "\n"
    project_root = Path(
        str(row["argv"][row["argv"].index("--project-root") + 1])
    )
    for field in ("stdout_path", "stderr_path"):
        concrete = project_root / str(row[field]).replace(
            "%A", "12346"
        ).replace("%a", "0")
        concrete.write_bytes(f"second {field}\n".encode())
    output_dir = Path(str(row["completion_path"])).parent
    output_dir.mkdir()
    (output_dir / "partial.bin").write_bytes(b"second failed attempt")

    def second_runner(command: list[str], **_: object) -> SimpleNamespace:
        return SimpleNamespace(
            stdout=second_sacct if command[0] == "sacct" else "",
            returncode=0,
        )

    third = jobmap_module.recover_job_row_from_slurm(
        payload,
        row,
        job_map_path=map_path,
        failed_slurm_job="12346_0",
        runner=second_runner,
    )
    assert third.generation == 3
    audit = json.loads(
        second.slurm_recovery_audit_path.read_text(encoding="utf-8")
    )
    assert audit["payload"]["binding_mode"] == (
        "recovered_attempt_scheduler_binding"
    )
    assert audit["payload"]["attempt"]["scheduler_job_id"] == "12346_0"
    assert jobmap_module.should_submit_row(payload, row) is True


@pytest.mark.parametrize(
    ("failure", "match"),
    [
        ("live", "live|squeue"),
        ("completed", "failed terminal|COMPLETED"),
        ("drift", "changed|sacct"),
        ("wrong-user", "user|UID"),
        ("wrong-workdir", "WorkDir|work"),
        ("wrong-submit", "SubmitLine|submission"),
        ("multiple", "exactly one|ambiguous|sacct"),
    ],
)
def test_slurm_recovery_rejects_untrusted_scheduler_evidence(
    experiment_root: Path,
    jobmap_module: ModuleType,
    submit_module: ModuleType,
    tmp_path: Path,
    failure: str,
    match: str,
) -> None:
    payload, row, map_path, first, _, sacct_line = _slurm_recovery_case(
        tmp_path=tmp_path,
        experiment_root=experiment_root,
        jobmap_module=jobmap_module,
        submit_module=submit_module,
    )
    fields = list(jobmap_module.SLURM_RECOVERY_SACCT_FIELDS)

    def changed_line(field: str, value: str) -> str:
        values = sacct_line.removesuffix("\n").split("|")
        values[fields.index(field)] = value
        return "|".join(values) + "\n"

    first_sacct = sacct_line
    second_sacct = sacct_line
    queue = ""
    if failure == "live":
        queue = "12345_0|user|RUNNING|debug|gpu-test\n"
    elif failure == "completed":
        first_sacct = second_sacct = changed_line("State", "COMPLETED")
    elif failure == "drift":
        second_sacct = changed_line("NodeList", "gpu-other")
    elif failure == "wrong-user":
        first_sacct = second_sacct = changed_line("UID", "999999")
    elif failure == "wrong-workdir":
        first_sacct = second_sacct = changed_line("WorkDir", str(tmp_path))
    elif failure == "wrong-submit":
        first_sacct = second_sacct = changed_line(
            "SubmitLine",
            sacct_line.split("|")[fields.index("SubmitLine")].replace(
                str(payload["payload_sha256"]),
                "0" * 64,
            ),
        )
    elif failure == "multiple":
        first_sacct = second_sacct = sacct_line + sacct_line
    sacct_outputs = iter([first_sacct, second_sacct])

    def fake_runner(command: list[str], **_: object) -> SimpleNamespace:
        if command[0] == "squeue":
            return SimpleNamespace(stdout=queue, returncode=0)
        return SimpleNamespace(stdout=next(sacct_outputs), returncode=0)

    with pytest.raises((RuntimeError, ValueError), match=match):
        jobmap_module.recover_job_row_from_slurm(
            payload,
            row,
            job_map_path=map_path,
            failed_slurm_job="12345_0",
            runner=fake_runner,
        )
    assert not first.path.with_name("slurm-recovery-audit.json").exists()
    assert not first.recovery_path.exists()
    assert not (
        first.path.parent.parent / "generation-000002"
    ).exists()


@pytest.mark.parametrize(
    "failure",
    ("mtime", "missing-log", "symlink-log", "symlink-log-parent"),
)
def test_slurm_recovery_rejects_unbound_legacy_claim_or_logs(
    experiment_root: Path,
    jobmap_module: ModuleType,
    submit_module: ModuleType,
    tmp_path: Path,
    failure: str,
) -> None:
    payload, row, map_path, first, _, sacct_line = _slurm_recovery_case(
        tmp_path=tmp_path,
        experiment_root=experiment_root,
        jobmap_module=jobmap_module,
        submit_module=submit_module,
    )
    stderr = (
        Path(str(row["stderr_path"]).replace("%A", "12345").replace("%a", "0"))
    )
    project_root = Path(str(row["argv"][row["argv"].index("--project-root") + 1]))
    stderr = project_root / stderr
    if failure == "mtime":
        os.utime(first.path, ns=(1_000_000_000, 1_000_000_000))
    elif failure == "missing-log":
        stderr.unlink()
    elif failure == "symlink-log":
        outside = tmp_path / "outside-error.log"
        outside.write_bytes(b"outside")
        stderr.unlink()
        stderr.symlink_to(outside)
    else:
        logs = project_root / "logs"
        outside_logs = tmp_path / "outside-logs"
        logs.rename(outside_logs)
        logs.symlink_to(outside_logs, target_is_directory=True)

    def fake_runner(command: list[str], **_: object) -> SimpleNamespace:
        return SimpleNamespace(
            stdout=sacct_line if command[0] == "sacct" else "",
            returncode=0,
        )

    with pytest.raises(ValueError, match="mtime|time|log|regular|symbolic|sealed"):
        jobmap_module.recover_job_row_from_slurm(
            payload,
            row,
            job_map_path=map_path,
            failed_slurm_job="12345_0",
            runner=fake_runner,
        )
    assert not first.path.with_name("slurm-recovery-audit.json").exists()
    assert not first.recovery_path.exists()


def test_typed_slurm_recovery_audit_is_required_by_recovery_chain(
    experiment_root: Path,
    jobmap_module: ModuleType,
    submit_module: ModuleType,
    tmp_path: Path,
) -> None:
    payload, row, map_path, first, _, sacct_line = _slurm_recovery_case(
        tmp_path=tmp_path,
        experiment_root=experiment_root,
        jobmap_module=jobmap_module,
        submit_module=submit_module,
    )

    def fake_runner(command: list[str], **_: object) -> SimpleNamespace:
        return SimpleNamespace(
            stdout=sacct_line if command[0] == "sacct" else "",
            returncode=0,
        )

    jobmap_module.recover_job_row_from_slurm(
        payload,
        row,
        job_map_path=map_path,
        failed_slurm_job="12345_0",
        runner=fake_runner,
    )
    audit_path = first.path.with_name("slurm-recovery-audit.json")
    audit_path.unlink()

    with pytest.raises(ValueError, match="audit|sealed|regular"):
        jobmap_module.should_submit_row(payload, row)


def test_stale_claim_recovery_is_audited_and_never_deletes_original(
    jobmap_module: ModuleType,
    tmp_path: Path,
) -> None:
    payload = jobmap_module.build_job_map([_row(tmp_path)])
    row = payload["rows"][0]
    first = jobmap_module.claim_job_row(payload, row)
    first_bytes = first.path.read_bytes()
    second = jobmap_module._recover_job_row_unverified_for_testing(
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

    jobmap_module.consume_recovery_attempt(payload, row)
    jobmap_module.complete_job_row(
        payload,
        row,
        _output_hashes(row, "recovered-metrics"),
    )
    assert jobmap_module.completion_is_valid(payload, row)


def test_each_recovery_generation_authorizes_exactly_one_attempt(
    jobmap_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    map_path = tmp_path / "attempt-map.json"
    payload = jobmap_module.write_job_map(
        [_row(tmp_path)],
        map_path,
    )
    row = payload["rows"][0]
    first = jobmap_module.claim_job_row(payload, row)
    second = jobmap_module._recover_job_row_unverified_for_testing(
        payload,
        row,
        recovery_audit_sha256=_h("generation-2-audit"),
    )
    assert first.generation == 1
    assert second.generation == 2
    assert jobmap_module.should_submit_row(payload, row) is True

    calls: list[object] = []

    def fail_once(command: object, **kwargs: object) -> SimpleNamespace:
        calls.append(command)
        assert kwargs["check"] is False
        return SimpleNamespace(returncode=17)

    monkeypatch.setattr(jobmap_module.subprocess, "run", fail_once)
    run_argv = [
        "run-row",
        "--job-map",
        str(map_path),
        "--job-map-sha256",
        str(payload["payload_sha256"]),
        "--array-index",
        "0",
        "--array-min",
        "0",
        "--array-max",
        "0",
    ]
    assert jobmap_module.main(run_argv) == 17
    assert jobmap_module.should_submit_row(payload, row) is False
    assert second.attempt_path.is_file()

    with pytest.raises(RuntimeError, match="attempt|audit|recovery"):
        jobmap_module.main(run_argv)
    assert len(calls) == 1

    third = jobmap_module._recover_job_row_unverified_for_testing(
        payload,
        row,
        recovery_audit_sha256=_h("generation-3-audit"),
    )
    assert third.generation == 3
    assert jobmap_module.should_submit_row(payload, row) is True
    second_recovery = json.loads(
        second.recovery_path.read_text(encoding="utf-8")
    )
    attempt_sha256 = hashlib.sha256(
        second.attempt_path.read_bytes()
    ).hexdigest()
    assert (
        second_recovery["payload"]["attempt_record_sha256"]
        == attempt_sha256
    )


def test_run_row_rejects_unclaimed_unaudited_output_directory(
    jobmap_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    map_path = tmp_path / "unaudited-output-map.json"
    payload = jobmap_module.write_job_map(
        [_row(tmp_path)],
        map_path,
    )
    row = payload["rows"][0]
    output_dir = Path(str(row["completion_path"])).parent
    output_dir.mkdir(parents=True)
    partial = output_dir / "partial.bin"
    partial.write_bytes(b"unaudited partial output")
    monkeypatch.setattr(
        jobmap_module.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail(
            "unaudited output must fail before subprocess"
        ),
    )

    with pytest.raises(RuntimeError, match="output|audit|recovery"):
        jobmap_module.main(
            [
                "run-row",
                "--job-map",
                str(map_path),
                "--job-map-sha256",
                str(payload["payload_sha256"]),
                "--array-index",
                "0",
                "--array-min",
                "0",
                "--array-max",
                "0",
            ]
        )

    assert partial.read_bytes() == b"unaudited partial output"


def test_should_submit_row_rejects_unclaimed_unaudited_output_directory(
    jobmap_module: ModuleType,
    tmp_path: Path,
) -> None:
    payload = jobmap_module.build_job_map([_row(tmp_path)])
    row = payload["rows"][0]
    output_dir = Path(str(row["completion_path"])).parent
    output_dir.mkdir(parents=True)
    partial = output_dir / "partial.bin"
    partial.write_bytes(b"unaudited partial output")

    with pytest.raises(RuntimeError, match="output|audit|recovery"):
        jobmap_module.should_submit_row(payload, row)

    assert partial.read_bytes() == b"unaudited partial output"


def test_audited_recovery_atomically_quarantines_output_for_fresh_retry(
    jobmap_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    map_path = tmp_path / "quarantine-map.json"
    payload = jobmap_module.write_job_map(
        [_row(tmp_path)],
        map_path,
    )
    row = payload["rows"][0]
    output_dir = Path(str(row["completion_path"])).parent
    first = jobmap_module.claim_job_row(payload, row)
    assert not first.path.is_relative_to(output_dir)
    output_dir.mkdir(parents=True)
    (output_dir / "partial.bin").write_bytes(b"generation-1 partial")

    second = jobmap_module._recover_job_row_unverified_for_testing(
        payload,
        row,
        recovery_audit_sha256=_h("quarantine-generation-1"),
    )
    assert second.generation == 2
    assert not output_dir.exists()
    first_recovery = json.loads(
        first.recovery_path.read_text(encoding="utf-8")
    )["payload"]
    assert first_recovery["restart_mode"] == "fresh"
    quarantine = first_recovery["quarantine"]
    assert isinstance(quarantine, dict)
    assert set(quarantine) == {
        "file_count",
        "quarantine_path",
        "source_output_dir",
        "tree_sha256",
    }
    assert quarantine["source_output_dir"] == str(output_dir)
    assert re.fullmatch(r"[0-9a-f]{64}", quarantine["tree_sha256"])
    assert quarantine["file_count"] == 1
    first_quarantine = Path(quarantine["quarantine_path"])
    assert first_quarantine.is_dir()
    assert (
        first_quarantine / "partial.bin"
    ).read_bytes() == b"generation-1 partial"

    def fail_recovered_run(
        _command: object,
        **kwargs: object,
    ) -> SimpleNamespace:
        assert kwargs["check"] is False
        assert not output_dir.exists()
        output_dir.mkdir(parents=True)
        (output_dir / "partial.bin").write_bytes(
            b"generation-2 partial"
        )
        return SimpleNamespace(returncode=23)

    monkeypatch.setattr(
        jobmap_module.subprocess,
        "run",
        fail_recovered_run,
    )
    assert jobmap_module.main(
        [
            "run-row",
            "--job-map",
            str(map_path),
            "--job-map-sha256",
            str(payload["payload_sha256"]),
            "--array-index",
            "0",
            "--array-min",
            "0",
            "--array-max",
            "0",
        ]
    ) == 23
    assert output_dir.is_dir()
    assert jobmap_module.should_submit_row(payload, row) is False

    third = jobmap_module._recover_job_row_unverified_for_testing(
        payload,
        row,
        recovery_audit_sha256=_h("quarantine-generation-2"),
    )
    assert third.generation == 3
    assert not output_dir.exists()
    second_recovery = json.loads(
        second.recovery_path.read_text(encoding="utf-8")
    )["payload"]
    second_quarantine = Path(
        second_recovery["quarantine"]["quarantine_path"]
    )
    assert (
        second_quarantine / "partial.bin"
    ).read_bytes() == b"generation-2 partial"
    assert first_quarantine.is_dir()
    assert jobmap_module.should_submit_row(payload, row) is True


def test_recovery_adopts_quarantine_left_by_prepublication_crash(
    jobmap_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = jobmap_module.build_job_map([_row(tmp_path)])
    row = payload["rows"][0]
    first = jobmap_module.claim_job_row(payload, row)
    output_dir = Path(str(row["completion_path"])).parent
    output_dir.mkdir(parents=True)
    partial = output_dir / "partial.bin"
    partial.write_bytes(b"partial before recovery publication")
    original_create_record = jobmap_module._create_record

    def crash_before_recovery_publication(
        path: Path,
        payload_type: str,
        record_payload: dict[str, object],
        **record_options: object,
    ) -> object:
        if payload_type == jobmap_module.RECOVERY_TYPE:
            raise RuntimeError("injected recovery publication crash")
        return original_create_record(
            path,
            payload_type,
            record_payload,
            **record_options,
        )

    monkeypatch.setattr(
        jobmap_module,
        "_create_record",
        crash_before_recovery_publication,
    )
    audit_sha256 = _h("prepublication-crash-audit")
    with pytest.raises(RuntimeError, match="injected"):
        jobmap_module._recover_job_row_unverified_for_testing(
            payload,
            row,
            recovery_audit_sha256=audit_sha256,
        )

    base = jobmap_module._state_dir(row)
    quarantine_path = jobmap_module._quarantine_path(base, 1)
    assert not output_dir.exists()
    assert (
        quarantine_path / "partial.bin"
    ).read_bytes() == b"partial before recovery publication"
    assert not first.recovery_path.exists()

    monkeypatch.setattr(
        jobmap_module,
        "_create_record",
        original_create_record,
    )
    second = jobmap_module._recover_job_row_unverified_for_testing(
        payload,
        row,
        recovery_audit_sha256=audit_sha256,
    )

    recovery = json.loads(
        first.recovery_path.read_text(encoding="utf-8")
    )["payload"]
    quarantine = recovery["quarantine"]
    assert isinstance(quarantine, dict)
    assert quarantine["quarantine_path"] == str(quarantine_path)
    assert quarantine["source_output_dir"] == str(output_dir)
    assert quarantine["file_count"] == 1
    assert quarantine["tree_sha256"] == (
        jobmap_module._directory_tree_identity(
            quarantine_path
        )["tree_sha256"]
    )
    assert recovery["restart_mode"] == "fresh"
    assert recovery["recovery_audit_sha256"] == audit_sha256
    argv = row["argv"]
    assert isinstance(argv, list)
    assert argv[argv.index("--resume") + 1] == "none"
    assert second.generation == 2


def test_recovery_resumes_published_record_with_missing_next_claim(
    jobmap_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    map_path = tmp_path / "pending-recovery-map.json"
    payload = jobmap_module.write_job_map(
        [_row(tmp_path)],
        map_path,
    )
    row = payload["rows"][0]
    first = jobmap_module.claim_job_row(payload, row)
    assert jobmap_module.should_submit_row(payload, row) is False
    original_create_record = jobmap_module._create_record

    def crash_before_next_claim_publication(
        path: Path,
        payload_type: str,
        record_payload: dict[str, object],
        **record_options: object,
    ) -> object:
        if (
            payload_type == jobmap_module.CLAIM_TYPE
            and record_payload.get("generation") == 2
        ):
            raise RuntimeError("injected next-claim publication crash")
        return original_create_record(
            path,
            payload_type,
            record_payload,
            **record_options,
        )

    monkeypatch.setattr(
        jobmap_module,
        "_create_record",
        crash_before_next_claim_publication,
    )
    audit_sha256 = _h("pending-recovery-audit")
    with pytest.raises(RuntimeError, match="injected"):
        jobmap_module._recover_job_row_unverified_for_testing(
            payload,
            row,
            recovery_audit_sha256=audit_sha256,
        )

    recovery_bytes = first.recovery_path.read_bytes()
    recovery = json.loads(
        recovery_bytes.decode("utf-8")
    )["payload"]
    assert recovery["next_generation"] == 2
    assert recovery["recovery_audit_sha256"] == audit_sha256
    next_claim_path = (
        first.path.parent.parent
        / "generation-000002"
        / "claim.json"
    )
    assert not next_claim_path.exists()

    monkeypatch.setattr(
        jobmap_module,
        "_create_record",
        original_create_record,
    )
    with pytest.raises(ValueError, match="recovery|next claim"):
        jobmap_module.should_submit_row(payload, row)
    with pytest.raises(ValueError, match="recovery|next claim"):
        jobmap_module.complete_job_row(
            payload,
            row,
            _output_hashes(row, "must-not-complete"),
        )
    calls: list[object] = []

    def forbidden_subprocess(*_args: object, **_kwargs: object) -> object:
        calls.append(object())
        pytest.fail("pending recovery must fail before subprocess")

    monkeypatch.setattr(
        jobmap_module.subprocess,
        "run",
        forbidden_subprocess,
    )
    with pytest.raises(ValueError, match="recovery|next claim"):
        jobmap_module.main(
            [
                "run-row",
                "--job-map",
                str(map_path),
                "--job-map-sha256",
                str(payload["payload_sha256"]),
                "--array-index",
                "0",
                "--array-min",
                "0",
                "--array-max",
                "0",
            ]
        )
    assert calls == []

    with pytest.raises(
        ValueError,
        match="does not match.*audit|audit.*does not match",
    ):
        jobmap_module._recover_job_row_unverified_for_testing(
            payload,
            row,
            recovery_audit_sha256=_h("different-audit"),
        )
    assert first.recovery_path.read_bytes() == recovery_bytes
    assert not next_claim_path.exists()

    second = jobmap_module._recover_job_row_unverified_for_testing(
        payload,
        row,
        recovery_audit_sha256=audit_sha256,
    )
    assert second.generation == 2
    assert second.path == next_claim_path
    assert first.recovery_path.read_bytes() == recovery_bytes
    assert second.document["payload"][
        "recovery_record_sha256"
    ] == hashlib.sha256(recovery_bytes).hexdigest()
    assert jobmap_module.should_submit_row(payload, row) is True


def test_pending_recovery_rejects_quarantine_unbound_by_null_record(
    jobmap_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = jobmap_module.build_job_map([_row(tmp_path)])
    row = payload["rows"][0]
    first = jobmap_module.claim_job_row(payload, row)
    original_create_record = jobmap_module._create_record

    def crash_before_next_claim_publication(
        path: Path,
        payload_type: str,
        record_payload: dict[str, object],
        **record_options: object,
    ) -> object:
        if (
            payload_type == jobmap_module.CLAIM_TYPE
            and record_payload.get("generation") == 2
        ):
            raise RuntimeError("injected next-claim publication crash")
        return original_create_record(
            path,
            payload_type,
            record_payload,
            **record_options,
        )

    monkeypatch.setattr(
        jobmap_module,
        "_create_record",
        crash_before_next_claim_publication,
    )
    audit_sha256 = _h("null-quarantine-audit")
    with pytest.raises(RuntimeError, match="injected"):
        jobmap_module._recover_job_row_unverified_for_testing(
            payload,
            row,
            recovery_audit_sha256=audit_sha256,
        )

    recovery = json.loads(
        first.recovery_path.read_text(encoding="utf-8")
    )["payload"]
    assert recovery["quarantine"] is None
    quarantine_path = jobmap_module._quarantine_path(
        jobmap_module._state_dir(row),
        1,
    )
    quarantine_path.mkdir(parents=True)
    (quarantine_path / "unbound.bin").write_bytes(
        b"not sealed by recovery record"
    )
    monkeypatch.setattr(
        jobmap_module,
        "_create_record",
        original_create_record,
    )

    with pytest.raises(
        ValueError,
        match="quarantine.*not recorded|unbound.*quarantine",
    ):
        jobmap_module._recover_job_row_unverified_for_testing(
            payload,
            row,
            recovery_audit_sha256=audit_sha256,
        )
    assert not (
        first.path.parent.parent
        / "generation-000002"
        / "claim.json"
    ).exists()


def test_recovery_rejects_quarantine_root_symlink_without_external_write(
    jobmap_module: ModuleType,
    tmp_path: Path,
) -> None:
    payload = jobmap_module.build_job_map([_row(tmp_path)])
    row = payload["rows"][0]
    first = jobmap_module.claim_job_row(payload, row)
    output_dir = Path(str(row["completion_path"])).parent
    output_dir.mkdir(parents=True)
    (output_dir / "partial.bin").write_bytes(b"must remain local")

    state_dir = jobmap_module._state_dir(row)
    outside = tmp_path / "outside-quarantine-root"
    outside.mkdir()
    (state_dir / "quarantine").symlink_to(
        outside,
        target_is_directory=True,
    )

    with pytest.raises(
        ValueError,
        match="quarantine|symlink|symbolic|directory",
    ):
        jobmap_module._recover_job_row_unverified_for_testing(
            payload,
            row,
            recovery_audit_sha256=_h("quarantine-root-symlink"),
        )

    assert (output_dir / "partial.bin").read_bytes() == b"must remain local"
    assert list(outside.iterdir()) == []
    assert not first.recovery_path.exists()
    assert not (
        state_dir / "generation-000002"
    ).exists()


def test_recovery_rejects_quarantine_symlink_injected_after_validation(
    jobmap_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = jobmap_module.build_job_map([_row(tmp_path)])
    row = payload["rows"][0]
    first = jobmap_module.claim_job_row(payload, row)
    output_dir = Path(str(row["completion_path"])).parent
    output_dir.mkdir(parents=True)
    (output_dir / "partial.bin").write_bytes(b"must not escape")

    state_dir = jobmap_module._state_dir(row)
    quarantine_root = state_dir / "quarantine"
    outside = tmp_path / "outside-quarantine-race"
    outside.mkdir()
    original_open_child = jobmap_module._open_child_directory
    injected = False

    def inject_before_nofollow_open(
        parent_fd: int,
        name: str,
        path: Path,
        *,
        create: bool = False,
        exclusive: bool = False,
    ) -> object:
        nonlocal injected
        if name == "quarantine" and not injected:
            quarantine_root.symlink_to(
                outside,
                target_is_directory=True,
            )
            injected = True
        return original_open_child(
            parent_fd,
            name,
            path,
            create=create,
            exclusive=exclusive,
        )

    monkeypatch.setattr(
        jobmap_module,
        "_open_child_directory",
        inject_before_nofollow_open,
    )

    with pytest.raises(
        ValueError,
        match="quarantine|symlink|symbolic|directory",
    ):
        jobmap_module._recover_job_row_unverified_for_testing(
            payload,
            row,
            recovery_audit_sha256=_h("quarantine-race"),
        )

    assert injected is True
    assert (output_dir / "partial.bin").read_bytes() == b"must not escape"
    assert list(outside.iterdir()) == []
    assert not first.recovery_path.exists()
    assert not (
        state_dir / "generation-000002"
    ).exists()


@pytest.mark.parametrize("invalid_kind", ("file", "symlink"))
def test_recovery_rejects_invalid_preexisting_quarantine(
    jobmap_module: ModuleType,
    tmp_path: Path,
    invalid_kind: str,
) -> None:
    payload = jobmap_module.build_job_map([_row(tmp_path)])
    row = payload["rows"][0]
    first = jobmap_module.claim_job_row(payload, row)
    quarantine_path = jobmap_module._quarantine_path(
        jobmap_module._state_dir(row),
        1,
    )
    quarantine_path.parent.mkdir(parents=True)
    if invalid_kind == "file":
        quarantine_path.write_bytes(b"not a directory")
    else:
        target = tmp_path / "outside-quarantine"
        target.mkdir()
        quarantine_path.symlink_to(target, target_is_directory=True)

    with pytest.raises(
        ValueError,
        match="real directory|symbolic|symlink",
    ):
        jobmap_module._recover_job_row_unverified_for_testing(
            payload,
            row,
            recovery_audit_sha256=_h(invalid_kind),
        )
    assert not first.recovery_path.exists()


def test_submitter_checks_queue_and_submits_debug_smoke_before_full_pilot(
    experiment_root: Path,
    jobmap_module: ModuleType,
    submit_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    project_root, script = _submission_project(
        tmp_path, experiment_root
    )
    smoke = jobmap_module.write_job_map(
        [
            _row(
                tmp_path,
                stage="stage-2-smoke",
                project_root=project_root,
            )
        ],
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
                project_root=project_root,
            ),
            _row(
                tmp_path,
                subject=5,
                seed=43,
                partition="i64m1tga40u",
                time="04:00:00",
                project_root=project_root,
            ),
        ],
        tmp_path / "pilot-map.json",
    )
    calls: list[list[str]] = []

    def fake_runner(command: list[str], **_: object) -> SimpleNamespace:
        calls.append(list(command))
        return SimpleNamespace(stdout="12345\n", returncode=0)

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
    smoke_hashes = _output_hashes(smoke_row, "smoke")
    jobmap_module.claim_job_row(smoke, smoke_row)
    jobmap_module.complete_job_row(
        smoke,
        smoke_row,
        smoke_hashes,
    )

    def fake_output_validator(
        argv: object,
        *,
        expected_mode: str,
    ) -> SimpleNamespace:
        assert argv == smoke_row["argv"]
        assert expected_mode == "smoke"
        return SimpleNamespace(
            final_checkpoint_sha256=smoke_hashes[
                "final_checkpoint_sha256"
            ],
            in_loop_metadata_sha256=smoke_hashes[
                "in_loop_metadata_sha256"
            ],
            run_manifest_sha256=smoke_hashes[
                "run_manifest_sha256"
            ],
        )

    monkeypatch.setattr(
        submit_module,
        "validate_training_command_outputs",
        fake_output_validator,
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

    def forbid_completed_pilot_validation(
        *_args: object,
        **_kwargs: object,
    ) -> object:
        pytest.fail("completed pilot must not revalidate smoke outputs")

    monkeypatch.setattr(
        submit_module,
        "validate_training_command_outputs",
        forbid_completed_pilot_validation,
    )
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


def test_submitter_revalidates_every_smoke_row_before_pilot_queue(
    experiment_root: Path,
    jobmap_module: ModuleType,
    submit_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    project_root, script = _submission_project(tmp_path, experiment_root)
    smoke_path = tmp_path / "multi-smoke.json"
    smoke = jobmap_module.write_job_map(
        [
            _row(
                tmp_path,
                stage="stage-2-smoke",
                subject=subject,
                project_root=project_root,
            )
            for subject in (1, 5)
        ],
        smoke_path,
    )
    declared: dict[str, dict[str, str]] = {}
    for row in smoke["rows"]:
        hashes = _output_hashes(row, f"smoke:{row['array_index']}")
        declared[str(row["run_key"])] = hashes
        jobmap_module.claim_job_row(smoke, row)
        jobmap_module.complete_job_row(smoke, row, hashes)

    pilot_path = tmp_path / "multi-pilot.json"
    pilot = jobmap_module.write_job_map(
        [
            _row(
                tmp_path,
                partition="i64m1tga40u",
                time="04:00:00",
                project_root=project_root,
            )
        ],
        pilot_path,
    )
    events: list[str] = []

    def fake_output_validator(
        argv: object,
        *,
        expected_mode: str,
    ) -> SimpleNamespace:
        assert isinstance(argv, list)
        assert expected_mode == "smoke"
        run_key = argv[argv.index("--run-key") + 1]
        events.append(f"validate:{run_key}")
        hashes = declared[run_key]
        return SimpleNamespace(
            final_checkpoint_sha256=hashes["final_checkpoint_sha256"],
            in_loop_metadata_sha256=hashes[
                "in_loop_metadata_sha256"
            ],
            run_manifest_sha256=hashes["run_manifest_sha256"],
        )

    monkeypatch.setattr(
        submit_module,
        "validate_training_command_outputs",
        fake_output_validator,
        raising=False,
    )

    def fake_runner(command: list[str], **_: object) -> SimpleNamespace:
        events.append("queue" if command == QUEUE_COMMAND else "sbatch")
        return SimpleNamespace(stdout="12345\n", returncode=0)

    phase = submit_module.submit_available_pilot(
        smoke_job_map=smoke_path,
        smoke_sha256=smoke["payload_sha256"],
        pilot_job_map=pilot_path,
        pilot_sha256=pilot["payload_sha256"],
        slurm_script=script,
        log_dir=Path("logs/samga_brain_rw"),
        runner=fake_runner,
    )

    expected_validations = [
        f"validate:{row['run_key']}" for row in smoke["rows"]
    ]
    assert phase == "pilot-submitted"
    assert events == [*expected_validations, "queue", "sbatch"]


@pytest.mark.parametrize("failure", ("deleted", "hash-mismatch"))
def test_submitter_rejects_missing_or_tampered_smoke_artifacts_before_queue(
    experiment_root: Path,
    jobmap_module: ModuleType,
    submit_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    project_root, script = _submission_project(tmp_path, experiment_root)
    smoke_path = tmp_path / f"{failure}-smoke.json"
    smoke = jobmap_module.write_job_map(
        [
            _row(
                tmp_path,
                stage="stage-2-smoke",
                project_root=project_root,
            )
        ],
        smoke_path,
    )
    row = smoke["rows"][0]
    declared = _output_hashes(row, failure)
    jobmap_module.claim_job_row(smoke, row)
    jobmap_module.complete_job_row(smoke, row, declared)
    pilot_path = tmp_path / f"{failure}-pilot.json"
    pilot = jobmap_module.write_job_map(
        [
            _row(
                tmp_path,
                partition="i64m1tga40u",
                time="04:00:00",
                project_root=project_root,
            )
        ],
        pilot_path,
    )

    def fake_output_validator(
        _argv: object,
        *,
        expected_mode: str,
    ) -> SimpleNamespace:
        assert expected_mode == "smoke"
        if failure == "deleted":
            raise ValueError("training output cannot be opened safely")
        return SimpleNamespace(
            final_checkpoint_sha256=_h("different-checkpoint"),
            in_loop_metadata_sha256=declared["in_loop_metadata_sha256"],
            run_manifest_sha256=declared["run_manifest_sha256"],
        )

    monkeypatch.setattr(
        submit_module,
        "validate_training_command_outputs",
        fake_output_validator,
        raising=False,
    )
    commands: list[list[str]] = []
    with pytest.raises(ValueError, match="output|artifact|hash|completion"):
        submit_module.submit_available_pilot(
            smoke_job_map=smoke_path,
            smoke_sha256=smoke["payload_sha256"],
            pilot_job_map=pilot_path,
            pilot_sha256=pilot["payload_sha256"],
            slurm_script=script,
            log_dir=Path("logs/samga_brain_rw"),
            runner=lambda command, **_kwargs: commands.append(list(command)),
        )
    assert commands == []


def test_submitter_rejects_stray_output_without_calling_sbatch(
    experiment_root: Path,
    jobmap_module: ModuleType,
    submit_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    project_root, script = _submission_project(
        tmp_path,
        experiment_root,
    )
    smoke_path = tmp_path / "stray-output-smoke.json"
    smoke = jobmap_module.write_job_map(
        [
            _row(
                tmp_path,
                stage="stage-2-smoke",
                project_root=project_root,
            )
        ],
        smoke_path,
    )
    pilot_path = tmp_path / "stray-output-pilot.json"
    pilot = jobmap_module.write_job_map(
        [
            _row(
                tmp_path,
                partition="i64m1tga40u",
                time="04:00:00",
                project_root=project_root,
            )
        ],
        pilot_path,
    )
    smoke_row = smoke["rows"][0]
    output_dir = Path(str(smoke_row["completion_path"])).parent
    output_dir.mkdir(parents=True)
    (output_dir / "partial.bin").write_bytes(b"stray output")
    calls: list[list[str]] = []

    def fake_runner(command: list[str], **_: object) -> SimpleNamespace:
        calls.append(list(command))
        return SimpleNamespace(stdout="12345\n", returncode=0)

    with pytest.raises(RuntimeError, match="output|audit|recovery"):
        submit_module.submit_available_pilot(
            smoke_job_map=smoke_path,
            smoke_sha256=smoke["payload_sha256"],
            pilot_job_map=pilot_path,
            pilot_sha256=pilot["payload_sha256"],
            slurm_script=script,
            log_dir=Path("logs/samga_brain_rw"),
            runner=fake_runner,
        )

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
