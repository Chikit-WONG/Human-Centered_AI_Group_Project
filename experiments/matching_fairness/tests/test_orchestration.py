from __future__ import annotations

import importlib.util
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[1]
RUNNER = EXPERIMENT_ROOT / "run_matching_fairness.sh"
SUBMITTER_PATH = EXPERIMENT_ROOT / "scripts/submit_pipeline.py"
SLURM_ROOT = EXPERIMENT_ROOT / "slurm"
LOGS_FRAGMENT = "/test/brain-rw/logs/matching_fairness_v3/"


def _load_submitter():
    spec = importlib.util.spec_from_file_location("matching_submit_pipeline", SUBMITTER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_script(name: str):
    path = EXPERIMENT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"orchestration_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _dry_run(*arguments: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(RUNNER), *arguments],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )


def test_wrapper_is_strict_and_exposes_only_fixed_scope_options() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "set -Eeuo pipefail" in text
    help_result = _dry_run("--help")
    assert help_result.returncode == 0, help_result.stderr
    assert "--phase" in help_result.stdout
    assert "--submit" in help_result.stdout
    assert "--dry-run" in help_result.stdout
    assert "--overwrite" in help_result.stdout
    for forbidden in ("--subject", "--seed", "--model", "--results-root"):
        assert forbidden not in help_result.stdout


def test_all_dry_run_is_cwd_independent_and_fixed_to_sub08_seed42(tmp_path: Path) -> None:
    result = _dry_run("--phase", "all", "--dry-run", cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    output = result.stdout
    assert str(REPOSITORY_ROOT) in output
    assert "sub-08" in output
    assert "seed42" in output
    assert "nice" in output
    assert "atm_s" in output
    assert "our_project" in output
    assert "sub-01" not in output
    assert "seed0" not in output
    assert "--array=0-1%2" in output
    assert "afterok:<train_job_id>" in output
    assert "afterok:<native_export_job_id>:<brainrw_export_job_id>" in output
    assert "MODE=audit" in output
    assert "afterok:<native_audit_job_id>" in output
    assert "training_cells=2" in output
    assert "main_exports=3" in output
    assert "scenarios=90" in output
    assert "decoder_records=450" in output
    assert "/runs/runs" not in output


def test_dry_run_does_not_create_runtime_directories(tmp_path: Path) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    result = module.execute_pipeline(
        phase="all",
        submit=False,
        dry_run=True,
        overwrite=False,
        layout=layout,
    )
    assert result["mode"] == "dry-run"
    assert not layout.results_root.exists()
    assert not layout.logs_root.exists()


def _write_canonical_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _checkpoint_manifest(model: str) -> dict[str, object]:
    checkpoint = {
        "epoch": 1,
        "val_loss": 0.1,
        "checkpoint": "epoch_0001.pth",
        "sha256": "e" * 64,
    }
    return {
        "schema_version": 1,
        "model": model,
        "encoder_type": "NICE" if model == "nice" else "ATMS",
        "subject": "sub-08",
        "seed": 42,
        "source": {
            "url": "https://github.com/dongyangli-del/EEG_Image_decode.git",
            "branch": "develop",
            "commit": "f" * 40,
            "checkout_sha256": "a" * 64,
        },
        "inputs": {
            "training_eeg": {
                "name": "preprocessed_eeg_training.npy",
                "sha256": "b" * 64,
            },
            "training_features": {
                "name": "ViT-H-14_features_train.pt",
                "sha256": "c" * 64,
            },
        },
        "hyperparameters": {
            "epochs": 500,
            "batch_size": 1024,
            "learning_rate": 3e-4,
            "val_ratio": 0.1,
            "early_stopping_patience": 10,
            "ema_decay": 0.999,
            "logit_scale_type": "exp",
            "avg_trials": True,
            "n_chans": 63,
            "n_times": 250,
        },
        "encoder_behavior": {
            "use_subject_id": model == "atm_s",
            "normalize_feats": model == "atm_s",
        },
        "checkpoints": [checkpoint],
        "selection": {
            "epoch": 1,
            "val_loss": 0.1,
            "checkpoint": "epoch_0001.pth",
        },
        "best_checkpoint": {"name": "best_val.pth", "sha256": "e" * 64},
        "history": {"name": "history.csv", "sha256": "d" * 64},
        "stopped_early": False,
    }


def _write_sealed_fairness_inputs(root: Path, protocol: Path) -> tuple[Path, Path]:
    import numpy as np

    from matching_fairness.artifacts import ScoreArtifact, write_score_artifact
    from matching_fairness.native_export import (
        _ASSET_RELATIVE_PATHS,
        _formal_artifact_inventory,
        _score_artifact_sha256,
    )
    from matching_fairness.provenance import sha256_file, sha256_path
    from matching_fairness.trial_splits import build_trial_manifest

    ids = tuple(f"image-{index:03d}" for index in range(200))
    sessions = np.tile(np.repeat(np.arange(4), 20), (200, 1))
    trial_manifest = root / "trial_manifest.json"
    _write_canonical_json(
        trial_manifest,
        build_trial_manifest(ids, sessions, seed=42),
    )
    trial_sha256 = sha256_file(trial_manifest)
    protocol_sha256 = sha256_file(protocol)
    matrices = root / "matrices"
    checkpoints = root / "checkpoints"
    standard_similarity = np.eye(200, dtype=np.float32)
    eeg_a_similarity = standard_similarity.copy()
    eeg_b_similarity = standard_similarity.copy()
    rows = np.arange(200)
    eeg_a_similarity[rows, (rows + 1) % 200] = 0.01
    eeg_b_similarity[rows, (rows + 2) % 200] = 0.02
    similarity_by_half = {
        "standard": standard_similarity,
        "a": eeg_a_similarity,
        "b": eeg_b_similarity,
    }
    native_metrics = {
        "top1_count": 200,
        "top5_count": 200,
        "sample_count": 200,
    }
    source_lock = {
        "url": "https://github.com/dongyangli-del/EEG_Image_decode.git",
        "branch": "develop",
        "commit": "f" * 40,
        "checkout_sha256": "a" * 64,
    }
    asset_hashes = {
        _ASSET_RELATIVE_PATHS[0]: "b" * 64,
        _ASSET_RELATIVE_PATHS[1]: "7" * 64,
        _ASSET_RELATIVE_PATHS[2]: "c" * 64,
        _ASSET_RELATIVE_PATHS[3]: "8" * 64,
    }
    asset_lock = {
        "repo_id": "LidongYang/EEG_Image_decode",
        "repo_type": "dataset",
        "asset_root": "/sealed/assets",
        "files": {
            name: {"bytes": 1, "sha256": digest}
            for name, digest in asset_hashes.items()
        },
    }
    role_by_directory = {"standard": "standard", "eeg_a": "a", "eeg_b": "b"}

    for model_index, model in enumerate(("nice", "atm_s")):
        checkpoint_path = checkpoints / model / "checkpoint_manifest.json"
        _write_canonical_json(checkpoint_path, _checkpoint_manifest(model))
        checkpoint_manifest_sha256 = sha256_file(checkpoint_path)
        for half_index, (directory, half) in enumerate(role_by_directory.items()):
            metadata = {
                "model_slug": model,
                "trial_half": half,
                "checkpoint_role": "val_selected_formal",
                "checkpoint": "/sealed/best_val.pth",
                "checkpoint_sha256": "e" * 64,
                "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
                "source_lock": source_lock,
                "asset_lock_manifest_sha256": "9" * 64,
                "asset_lock": asset_lock,
                "input_sha256": {
                    "test_eeg": asset_hashes[_ASSET_RELATIVE_PATHS[1]],
                    "test_features": asset_hashes[_ASSET_RELATIVE_PATHS[3]],
                    "trial_manifest": trial_sha256,
                },
                "trial_manifest_sha256": trial_sha256,
                "subject": "sub-08",
                "seed": 42,
                "logit_scale_type": "exp",
                "effective_logit_scale": 1.0,
                "query_embeddings_sha256": (
                    f"{model_index * 3 + half_index + 1:064x}"
                ),
                "native_metrics": native_metrics,
            }
            write_score_artifact(
                matrices / model / directory,
                ScoreArtifact(
                    similarity=similarity_by_half[half],
                    query_ids=ids,
                    gallery_entry_ids=ids,
                    gallery_canonical_ids=ids,
                    target_canonical_ids=ids,
                    metadata=metadata,
                ),
            )

    brain_inputs = {
        "protocol_sha256": protocol_sha256,
        "trial_manifest_sha256": trial_sha256,
        "brain_test_sha256": "3" * 64,
        "evaluator_sha256": "4" * 64,
        "test_image_tree_sha256": "5" * 64,
        "model_content_sha256": {
            "brain_model": "6" * 64,
            "vision_adapter": "7" * 64,
            "pretrained_vision_base": "8" * 64,
        },
    }
    brain_root = matrices / "our_project"
    for half_index, (directory, half) in enumerate(role_by_directory.items()):
        metadata = {
            "model_slug": "our_project",
            "trial_half": half,
            "checkpoint_role": "fixed_formal",
            "checkpoint": "/sealed/brain_model",
            "checkpoint_content_sha256": "6" * 64,
            "similarity": "cosine",
            "query_embeddings_sha256": f"{half_index + 10:064x}",
            "subject": "sub-08",
            "seed": 42,
            "trial_manifest_sha256": trial_sha256,
            "protocol_sha256": protocol_sha256,
            "brain_test_sha256": "3" * 64,
            "model_content_sha256": brain_inputs["model_content_sha256"],
            "evaluator_version": "AIAA3800-BRAINRW-FORMAL-v1",
            "evaluator_sha256": "4" * 64,
            "runtime_inputs": {
                "test_image_tree_sha256": "5" * 64,
                "selected_channel_indices": list(range(46, 63)),
                "time_slice": [0, 250],
                "dataset_name": "things",
                "expected_sample_count": 200,
            },
            "native_metrics": native_metrics,
        }
        write_score_artifact(
            brain_root / directory,
            ScoreArtifact(
                similarity=similarity_by_half[half],
                query_ids=ids,
                gallery_entry_ids=ids,
                gallery_canonical_ids=ids,
                target_canonical_ids=ids,
                metadata=metadata,
            ),
        )
        run = brain_root / "runs" / directory
        run.mkdir(parents=True)
        _write_canonical_json(run / "metrics.json", native_metrics)
        (run / "predictions.csv").write_text(
            "query_id,predicted_id\nimage-000,image-000\n",
            encoding="utf-8",
        )
    _write_canonical_json(
        brain_root / "export_manifest.json",
        {
            "schema_version": 1,
            "scope": "fixed_formal_export",
            "checkpoint_role": "fixed_formal",
            "model_slug": "our_project",
            "subject": "sub-08",
            "seed": 42,
            "artifacts": {
                directory: {
                    "path": directory,
                    "sha256": _score_artifact_sha256(brain_root / directory),
                }
                for directory in role_by_directory
            },
            "runs": {
                directory: {
                    "path": f"runs/{directory}",
                    "sha256": sha256_path(brain_root / "runs" / directory),
                }
                for directory in role_by_directory
            },
            "inputs": brain_inputs,
        },
    )

    inventory = _formal_artifact_inventory(
        [
            matrices / model / directory
            for model in ("nice", "atm_s", "our_project")
            for directory in role_by_directory
        ],
        expected_image_count=200,
    )
    audit_run = {
        "epoch": 1,
        "checkpoint": "/sealed/epoch_0001.pth",
        "checkpoint_sha256": "e" * 64,
        "effective_logit_scale": 1.0,
        "top1_count": 200,
        "top5_count": 200,
        "sample_count": 200,
    }
    for model in ("nice", "atm_s"):
        _write_canonical_json(
            matrices / model / "best_test_audit.json",
            {
                "schema_version": 1,
                "scope": "best_test_audit_only",
                "model_slug": model,
                "checkpoint_policy": "every_epoch_checkpoint",
                "fairness_artifact_created": False,
                "formal_artifact_inventory": inventory,
                "runs": [audit_run],
                "best_test": audit_run,
            },
        )
    return matrices, trial_manifest


def _file_hash_snapshot(root: Path, protocol: Path | None = None) -> dict[str, str]:
    files = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    if protocol is not None:
        files["<formal-protocol>"] = hashlib.sha256(protocol.read_bytes()).hexdigest()
    return files


def test_fixture_pipeline_is_byte_stable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario_runner = _load_script("run_scenarios")
    production_calls: list[Path] = []
    production_run = scenario_runner.run_scenarios

    def counted_production_run(**kwargs):
        production_calls.append(Path(kwargs["output_dir"]))
        return production_run(**kwargs)

    monkeypatch.setattr(scenario_runner, "run_scenarios", counted_production_run)
    aggregate_results = _load_script("aggregate_results").aggregate_results
    protocol = EXPERIMENT_ROOT / "configs/protocol_sub08_seed42.json"
    sealed_root = tmp_path / "sealed-inputs"
    sealed_root.mkdir()
    matrices, trial_manifest = _write_sealed_fairness_inputs(sealed_root, protocol)
    original_sealed_inputs = _file_hash_snapshot(sealed_root, protocol)
    input_snapshots: list[dict[str, str]] = []
    aggregate_input_snapshots: list[dict[str, str]] = []
    run_snapshots: list[dict[str, str]] = []
    roots: list[Path] = []

    for name in ("fixture-a", "fixture-b"):
        root = tmp_path / name / "matching_fairness_v3"
        root.mkdir(parents=True)
        shutil.copytree(sealed_root / "checkpoints", root / "checkpoints")
        shutil.copytree(matrices, root / "matrices")
        input_snapshots.append(_file_hash_snapshot(sealed_root, protocol))
        aggregate_input_snapshots.append(
            {
                f"checkpoints/{name}": digest
                for name, digest in _file_hash_snapshot(root / "checkpoints").items()
            }
            | {
                f"matrices/{name}": digest
                for name, digest in _file_hash_snapshot(root / "matrices").items()
            }
        )

        count = scenario_runner.run_scenarios(
            protocol_path=protocol,
            artifact_root=matrices,
            trial_manifest_path=trial_manifest,
            output_dir=root / "runs",
        )
        assert count == 450
        assert len(tuple((root / "runs").rglob("summary.json"))) == 90
        assert len(tuple((root / "runs").rglob("per_query.csv"))) == 90
        assert _file_hash_snapshot(sealed_root, protocol) == original_sealed_inputs
        run_snapshots.append(_file_hash_snapshot(root / "runs"))
        roots.append(root)

    assert len(production_calls) == 2
    assert input_snapshots[0] == input_snapshots[1] == original_sealed_inputs
    assert aggregate_input_snapshots[0] == aggregate_input_snapshots[1]
    assert run_snapshots[0] == run_snapshots[1]

    compared = (
        "aggregate_metrics.csv",
        "aggregate_summary.json",
        "RESULTS.md",
        "RESULTS_ZH.md",
    )
    hashes = []
    for root in roots:
        aggregate = aggregate_results(root)
        hashes.append(
            {
                filename: hashlib.sha256((aggregate / filename).read_bytes()).hexdigest()
                for filename in compared
            }
        )
    assert hashes[0] == hashes[1]


@pytest.mark.parametrize(
    ("phase", "included", "excluded"),
    [
        ("preflight", "preflight.py", "train_native.py"),
        ("train", "train_native_array.slurm", "export_brainrw.slurm"),
        ("export", "export_brainrw.slurm", "fairness_cpu.slurm"),
        ("match", "run_scenarios.py", "aggregate_results.py"),
        ("aggregate", "aggregate_results.py", "run_scenarios.py"),
    ],
)
def test_phase_dry_run_is_scoped(
    phase: str, included: str, excluded: str
) -> None:
    result = _dry_run("--phase", phase, "--dry-run")
    assert result.returncode == 0, result.stderr
    assert included in result.stdout
    assert excluded not in result.stdout


@pytest.mark.parametrize("bad", ["sub-01", "seed0", "nice", "--results-root"])
def test_rejects_scope_broadening_arguments(bad: str) -> None:
    result = _dry_run("--phase", "all", "--dry-run", bad)
    assert result.returncode != 0


def test_invalid_array_id_fails_closed() -> None:
    module = _load_submitter()
    assert module.model_for_array_id(0) == "nice"
    assert module.model_for_array_id(1) == "atm_s"
    for value in (-1, 2, 8):
        with pytest.raises(ValueError, match="array"):
            module.model_for_array_id(value)


def test_slurm_resources_logs_offline_flags_and_array_scope() -> None:
    expected = {
        "train_native_array.slurm": (
            "i64m1tga40u", "gpu:a40:1", "8", "64G", "08:00:00", "0-1%2"
        ),
        "export_native_array.slurm": (
            "i64m1tga40u", "gpu:a40:1", "4", "48G", "03:00:00", "0-1%2"
        ),
        "export_brainrw.slurm": (
            "debug", "gpu:a40:1", "4", "32G", "00:30:00", None
        ),
        "fairness_cpu.slurm": (
            "i64m512u", None, "4", "16G", "02:00:00", None
        ),
    }
    for name, (partition, gres, cpus, memory, walltime, array) in expected.items():
        text = (SLURM_ROOT / name).read_text(encoding="utf-8")
        assert f"#SBATCH --partition={partition}" in text
        assert f"#SBATCH --cpus-per-task={cpus}" in text
        assert f"#SBATCH --mem={memory}" in text
        assert f"#SBATCH --time={walltime}" in text
        if gres is None:
            assert "#SBATCH --gres=" not in text
        else:
            assert f"#SBATCH --gres={gres}" in text
        if array is None:
            assert "#SBATCH --array=" not in text
        else:
            assert f"#SBATCH --array={array}" in text
        output_lines = [line for line in text.splitlines() if line.startswith("#SBATCH --output=")]
        error_lines = [line for line in text.splitlines() if line.startswith("#SBATCH --error=")]
        assert len(output_lines) == len(error_lines) == 1
        assert LOGS_FRAGMENT in output_lines[0]
        assert LOGS_FRAGMENT in error_lines[0]
        assert "HF_DATASETS_OFFLINE=1" in text
        assert "TRANSFORMERS_OFFLINE=1" in text
        assert "HF_HUB_OFFLINE=1" in text
        assert "PYTHONPATH=" in text


def test_slurm_scripts_reject_unknown_array_ids_and_use_internal_cells() -> None:
    for name, cell in (
        ("train_native_array.slurm", "train-native"),
        ("export_native_array.slurm", "export-native"),
    ):
        text = (SLURM_ROOT / name).read_text(encoding="utf-8")
        assert "SLURM_ARRAY_TASK_ID" in text
        assert "--internal-cell" in text
        assert cell in text
        assert "0|1" in text
        assert "invalid SLURM_ARRAY_TASK_ID" in text


@pytest.mark.parametrize(
    "name", ["train_native_array.slurm", "export_native_array.slurm"]
)
def test_slurm_array_scripts_fail_before_environment_activation_for_bad_id(
    name: str,
) -> None:
    environment = dict(os.environ)
    environment["SLURM_ARRAY_TASK_ID"] = "2"
    result = subprocess.run(
        ["bash", str(SLURM_ROOT / name)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 64
    assert "invalid SLURM_ARRAY_TASK_ID" in result.stderr


class _SubmissionRunner:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = iter(outputs)
        self.calls: list[list[str]] = []

    def __call__(self, argv, **kwargs):
        self.calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout=next(self.outputs), stderr="")


class _FailAtRunner:
    def __init__(self, fail_index: int) -> None:
        self.fail_index = fail_index
        self.calls: list[list[str]] = []

    def __call__(self, argv, **_kwargs):
        index = len(self.calls)
        self.calls.append(list(argv))
        if index == self.fail_index:
            raise subprocess.CalledProcessError(1, argv, stderr="scheduler rejected")
        return subprocess.CompletedProcess(argv, 0, stdout=f"{100 + index}\n", stderr="")


def test_submit_pipeline_uses_argv_and_exact_afterok_dependencies(tmp_path: Path) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    runner = _SubmissionRunner(["101\n", "202\n", "303\n", "404\n", "505\n"])
    result = module.submit_all(layout=layout, overwrite=False, runner=runner)
    assert result == {
        "train_job_id": 101,
        "native_export_job_id": 202,
        "brainrw_export_job_id": 303,
        "native_audit_job_id": 404,
        "final_job_id": 505,
    }
    assert runner.calls[0][0] == "sbatch"
    assert "--parsable" in runner.calls[0]
    assert "--dependency=afterok:101" in runner.calls[1]
    assert not any(item.startswith("--dependency=") for item in runner.calls[2])
    assert "--dependency=afterok:202:303" in runner.calls[3]
    assert any("MODE=audit" in item for item in runner.calls[3])
    assert "--dependency=afterok:404" in runner.calls[4]
    assert all(isinstance(call, list) for call in runner.calls)
    submission = json.loads(layout.submission_manifest.read_text(encoding="utf-8"))
    assert submission["subject"] == "sub-08"
    assert submission["seed"] == 42
    assert submission["schema_version"] == 3
    assert submission["mode"] == "all"
    assert set(submission["requests"]) == {"all"}
    request = submission["requests"]["all"]
    assert request["phase"] == "all"
    assert request["state"] == "completed"
    assert request["failure"] is None
    assert request["job_order"] == [
        "train", "native_export", "brainrw_export", "native_audit", "final"
    ]
    assert {
        f"{name}_job_id": request["jobs"][name]["job_id"]
        for name in request["job_order"]
    } == result
    assert all(
        row["state"] == "submitted"
        and row["token"].startswith("mf-")
        and isinstance(row["argv"], list)
        for row in request["jobs"].values()
    )


def test_existing_submission_manifest_prevents_duplicate_submit(tmp_path: Path) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    layout.submission_manifest.parent.mkdir(parents=True)
    layout.submission_manifest.write_text("{}\n", encoding="utf-8")
    runner = _SubmissionRunner(["1\n"])
    with pytest.raises((FileExistsError, RuntimeError), match="submission|ledger"):
        module.submit_all(layout=layout, overwrite=False, runner=runner)
    assert runner.calls == []


@pytest.mark.parametrize("state", ("active", "completed", "failed", "unknown"))
def test_preexisting_submission_state_never_auto_resubmits(
    tmp_path: Path, state: str
) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    layout.submission_manifest.parent.mkdir(parents=True)
    layout.submission_manifest.write_text(
        json.dumps({"schema_version": 2, "state": state}) + "\n",
        encoding="utf-8",
    )
    runner = _SubmissionRunner(["1\n"])
    with pytest.raises(
        (FileExistsError, RuntimeError, ValueError), match="submission|ledger"
    ):
        module.submit_all(layout=layout, overwrite=True, runner=runner)
    assert runner.calls == []


@pytest.mark.parametrize("fail_index", range(5))
def test_submission_failure_persists_every_known_id_and_stops_dag(
    tmp_path: Path, fail_index: int
) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    runner = _FailAtRunner(fail_index)
    with pytest.raises(subprocess.CalledProcessError):
        module.submit_all(layout=layout, overwrite=False, runner=runner)
    ledger = json.loads(layout.submission_manifest.read_text(encoding="utf-8"))
    request = ledger["requests"]["all"]
    assert request["state"] == "failed"
    assert request["failure"]["stage"] == request["job_order"][fail_index]
    assert len(runner.calls) == fail_index + 1
    for index, name in enumerate(request["job_order"]):
        row = request["jobs"][name]
        if index < fail_index:
            assert row["state"] == "submitted"
            assert row["job_id"] == 100 + index
        elif index == fail_index:
            assert row["state"] == "failed"
            assert row["token"].startswith("mf-")
            assert row["job_id"] is None
        else:
            assert row["state"] == "planned"
    with pytest.raises((FileExistsError, RuntimeError)):
        module.submit_all(
            layout=layout,
            overwrite=True,
            runner=_SubmissionRunner(["999\n"]),
        )


def test_submission_intent_is_durable_before_each_scheduler_call(tmp_path: Path) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    calls: list[list[str]] = []

    def runner(argv, **_kwargs):
        ledger = json.loads(layout.submission_manifest.read_text(encoding="utf-8"))
        request = ledger["requests"]["all"]
        name = request["job_order"][len(calls)]
        assert request["state"] == "active"
        assert request["jobs"][name]["state"] == "submitting"
        assert request["jobs"][name]["token"].startswith("mf-")
        assert request["jobs"][name]["argv"] == list(argv)
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout=f"{201 + len(calls)}\n", stderr="")

    module.submit_all(layout=layout, overwrite=False, runner=runner)
    assert len(calls) == 5


def test_job_id_ledger_write_failure_stops_with_durable_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    real_replace = module._replace_submission_ledger
    calls = 0

    def fail_second_write(path: Path, payload: dict[str, object]) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected job-id ledger failure")
        real_replace(path, payload)

    monkeypatch.setattr(module, "_replace_submission_ledger", fail_second_write)
    runner = _SubmissionRunner(["301\n", "302\n"])
    with pytest.raises(OSError, match="job-id ledger"):
        module.submit_all(layout=layout, overwrite=False, runner=runner)
    ledger = json.loads(layout.submission_manifest.read_text(encoding="utf-8"))
    request = ledger["requests"]["all"]
    assert len(runner.calls) == 1
    assert request["jobs"]["train"]["state"] == "submitting"
    assert request["jobs"]["train"]["token"].startswith("mf-")
    assert request["jobs"]["train"]["job_id"] is None


def test_no_clobber_reservation_fsyncs_parent_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_submitter()
    destination = tmp_path / "manifests/submission.json"
    observed: list[Path] = []
    monkeypatch.setattr(
        module, "_fsync_directory", lambda path: observed.append(Path(path))
    )
    module._atomic_write_json_noclobber(destination, {"state": "active"})
    assert observed == [destination.parent]


def test_repeat_phase_submit_is_guarded_by_same_durable_ledger(tmp_path: Path) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    module.submit_phase(
        phase="train",
        layout=layout,
        overwrite=False,
        runner=_SubmissionRunner(["401\n"]),
    )
    duplicate = _SubmissionRunner(["402\n"])
    with pytest.raises((FileExistsError, RuntimeError), match="submission|ledger"):
        module.submit_phase(
            phase="train", layout=layout, overwrite=True, runner=duplicate
        )
    assert duplicate.calls == []


def test_audit_and_final_dependencies_cannot_be_bypassed(tmp_path: Path) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    commands = module.submission_commands(layout=layout, overwrite=False)
    assert commands["native_audit"]["depends_on"] == (
        "native_export", "brainrw_export"
    )
    assert commands["final"]["depends_on"] == ("native_audit",)


def test_phase_submit_supports_train_and_corrected_export_dag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    train_runner = _SubmissionRunner(["11\n"])
    train = module.submit_phase(
        phase="train", layout=layout, overwrite=False, runner=train_runner
    )
    assert train == {"train_job_id": 11}
    assert "train_native_array.slurm" in train_runner.calls[0][-1]

    monkeypatch.setattr(module, "_checkpoint_matches_inputs", lambda *_args: True)
    export_runner = _SubmissionRunner(["21\n", "22\n", "23\n"])
    export = module.submit_phase(
        phase="export", layout=layout, overwrite=False, runner=export_runner
    )
    assert export == {
        "native_export_job_id": 21,
        "brainrw_export_job_id": 22,
        "native_audit_job_id": 23,
    }
    assert "--dependency=afterok:21:22" in export_runner.calls[2]
    assert any("MODE=audit" in item for item in export_runner.calls[2])


def test_same_layout_completed_train_then_export_uses_phase_scoped_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    train = module.submit_phase(
        phase="train",
        layout=layout,
        overwrite=False,
        runner=_SubmissionRunner(["11\n"]),
    )
    monkeypatch.setattr(
        module,
        "_checkpoint_matches_inputs",
        lambda _layout, model: model in {"nice", "atm_s"},
    )
    export_runner = _SubmissionRunner(["21\n", "22\n", "23\n"])

    export = module.submit_phase(
        phase="export",
        layout=layout,
        overwrite=False,
        runner=export_runner,
    )

    assert train == {"train_job_id": 11}
    assert export == {
        "native_export_job_id": 21,
        "brainrw_export_job_id": 22,
        "native_audit_job_id": 23,
    }
    ledger = json.loads(layout.submission_manifest.read_text(encoding="utf-8"))
    assert ledger["schema_version"] == 3
    assert ledger["mode"] == "phased"
    assert set(ledger["requests"]) == {"train", "export"}
    assert ledger["requests"]["train"]["state"] == "completed"
    assert ledger["requests"]["export"]["state"] == "completed"


def test_same_layout_export_rejects_missing_current_checkpoints_before_submit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    module.submit_phase(
        phase="train",
        layout=layout,
        overwrite=False,
        runner=_SubmissionRunner(["11\n"]),
    )
    monkeypatch.setattr(
        module,
        "_checkpoint_matches_inputs",
        lambda _layout, _model: False,
    )
    runner = _SubmissionRunner(["21\n"])

    with pytest.raises(ValueError, match="checkpoint"):
        module.submit_phase(
            phase="export",
            layout=layout,
            overwrite=False,
            runner=runner,
        )

    assert runner.calls == []


@pytest.mark.parametrize(
    "tamper",
    ("completed_with_planned_job", "foreign_token", "completed_with_failure"),
)
def test_phase_ledger_rejects_cross_field_tamper_before_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    module.submit_phase(
        phase="train",
        layout=layout,
        overwrite=False,
        runner=_SubmissionRunner(["11\n"]),
    )
    ledger = json.loads(layout.submission_manifest.read_text(encoding="utf-8"))
    request = ledger["requests"]["train"]
    job = request["jobs"]["train"]
    if tamper == "completed_with_planned_job":
        job.update(
            {
                "state": "planned",
                "argv": None,
                "job_id": None,
                "error": None,
            }
        )
    elif tamper == "foreign_token":
        job["token"] = f"mf-{'0' * 32}-train"
    else:
        request["failure"] = {"stage": "train", "error": "forged"}
    layout.submission_manifest.write_text(
        json.dumps(ledger, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "_checkpoint_matches_inputs", lambda *_args: True)
    runner = _SubmissionRunner(["21\n"])

    with pytest.raises(RuntimeError, match="ledger|request|job"):
        module.submit_phase(
            phase="export",
            layout=layout,
            overwrite=False,
            runner=runner,
        )

    assert runner.calls == []


def test_same_layout_repeat_export_and_all_phased_mix_are_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    monkeypatch.setattr(module, "_checkpoint_matches_inputs", lambda *_args: True)
    module.submit_phase(
        phase="train",
        layout=layout,
        overwrite=False,
        runner=_SubmissionRunner(["11\n"]),
    )
    module.submit_phase(
        phase="export",
        layout=layout,
        overwrite=False,
        runner=_SubmissionRunner(["21\n", "22\n", "23\n"]),
    )

    for submit in (
        lambda runner: module.submit_phase(
            phase="export", layout=layout, overwrite=True, runner=runner
        ),
        lambda runner: module.submit_all(
            layout=layout, overwrite=True, runner=runner
        ),
    ):
        runner = _SubmissionRunner(["99\n"])
        with pytest.raises((FileExistsError, RuntimeError), match="ledger|phase|submission"):
            submit(runner)
        assert runner.calls == []


def test_active_train_request_blocks_concurrent_export_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    monkeypatch.setattr(module, "_checkpoint_matches_inputs", lambda *_args: True)
    nested = _SubmissionRunner(["21\n"])

    def train_runner(argv, **_kwargs):
        with pytest.raises((FileExistsError, RuntimeError), match="active|ledger|phase"):
            module.submit_phase(
                phase="export",
                layout=layout,
                overwrite=False,
                runner=nested,
            )
        return subprocess.CompletedProcess(argv, 0, stdout="11\n", stderr="")

    module.submit_phase(
        phase="train",
        layout=layout,
        overwrite=False,
        runner=train_runner,
    )
    assert nested.calls == []


def test_native_audit_fails_before_all_nine_artifacts(tmp_path: Path) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    with pytest.raises(ValueError, match="nine"):
        module.run_internal_cell(
            layout=layout,
            cell="export-native",
            array_id=0,
            export_mode="audit",
            overwrite=False,
        )


def test_export_resume_does_not_skip_stale_input_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    matrix = layout.matrix_dir("nice")
    matrix.mkdir(parents=True)
    monkeypatch.setattr(module, "_matrix_complete", lambda _path: True)
    monkeypatch.setattr(
        module,
        "_matrix_matches_inputs",
        lambda _layout, _model: False,
        raising=False,
    )
    with pytest.raises(ValueError, match="partial|mismatch|provenance"):
        module.run_internal_cell(
            layout=layout,
            cell="export-native",
            array_id=0,
            export_mode="main",
            overwrite=False,
        )


@pytest.mark.parametrize(
    ("cell", "array_id", "model"),
    [
        ("export-native", 0, "nice"),
        ("export-brainrw", None, "our_project"),
    ],
)
def test_new_export_must_bind_current_input_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cell: str,
    array_id: int | None,
    model: str,
) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    monkeypatch.setattr(module, "_run", lambda _command: None)
    monkeypatch.setattr(module, "_matrix_complete", lambda _path: True)
    monkeypatch.setattr(
        module,
        "_matrix_matches_inputs",
        lambda _layout, candidate: candidate != model,
    )
    with pytest.raises(RuntimeError, match="current|provenance"):
        module.run_internal_cell(
            layout=layout,
            cell=cell,
            array_id=array_id,
            export_mode="main",
            overwrite=False,
        )


def test_audit_resume_does_not_skip_stale_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    audit = layout.matrix_dir("nice") / "best_test_audit.json"
    audit.parent.mkdir(parents=True)
    audit.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(module, "_require_all_nine_artifacts", lambda _layout: None)
    monkeypatch.setattr(
        module,
        "_audit_matches_inputs",
        lambda _layout, _model: False,
        raising=False,
    )
    with pytest.raises(ValueError, match="partial|mismatch|audit"):
        module.run_internal_cell(
            layout=layout,
            cell="export-native",
            array_id=0,
            export_mode="audit",
            overwrite=False,
        )


def test_aggregate_fails_before_matching_and_audits(tmp_path: Path) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    with pytest.raises(ValueError, match="matching|audit"):
        module.run_internal_cell(
            layout=layout,
            cell="aggregate",
            array_id=None,
            export_mode="main",
            overwrite=False,
        )


@pytest.mark.parametrize(
    "output",
    ["", "abc\n", "12;cluster\n", "12 extra\n", "0\n", "-1\n"],
)
def test_submit_pipeline_rejects_non_exact_parsable_job_ids(
    tmp_path: Path, output: str
) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    runner = _SubmissionRunner([output])
    with pytest.raises(RuntimeError, match="job ID"):
        module.submit_all(layout=layout, overwrite=False, runner=runner)
    ledger = json.loads(layout.submission_manifest.read_text(encoding="utf-8"))
    request = ledger["requests"]["all"]
    assert request["state"] == "unknown"
    assert request["jobs"]["train"]["state"] == "unknown"
    assert request["jobs"]["train"]["token"].startswith("mf-")
    assert request["jobs"]["train"]["job_id"] is None


def test_dry_run_never_invokes_submission_runner(tmp_path: Path) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dry-run attempted to submit")

    rendered = module.render_submission_plan(layout=layout, overwrite=False)
    assert "sbatch" in rendered
    result = module.execute_pipeline(
        phase="all",
        submit=False,
        dry_run=True,
        overwrite=False,
        layout=layout,
        runner=forbidden,
    )
    assert result["mode"] == "dry-run"


@pytest.mark.parametrize("phase", ("train", "export", "all"))
def test_public_heavy_phase_requires_submit_or_dry_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    monkeypatch.setattr(
        module,
        "run_internal_cell",
        lambda **_kwargs: pytest.fail("heavy phase attempted local execution"),
    )
    monkeypatch.setattr(
        module,
        "run_preflight",
        lambda *_args, **_kwargs: pytest.fail("all phase attempted local execution"),
    )
    with pytest.raises(ValueError, match="submit|dry-run"):
        module.execute_pipeline(
            phase=phase,
            submit=False,
            dry_run=False,
            overwrite=False,
            layout=layout,
        )


def test_orphan_preflight_phase_manifest_fails_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    orphan = module.phase_manifest_path(layout, "preflight")
    orphan.parent.mkdir(parents=True)
    orphan.write_text("{}\n", encoding="utf-8")
    assert not layout.logs_root.exists()
    monkeypatch.setattr(module, "_ensure_source_and_assets", lambda _layout: None)
    monkeypatch.setattr(
        module,
        "_run",
        lambda _command: pytest.fail("orphan state was mutated before rejection"),
    )
    with pytest.raises(ValueError, match="preflight|partial|orphan"):
        module.run_preflight(layout, overwrite=False)
    assert not layout.logs_root.exists()


@pytest.mark.parametrize("present", ("root", "lock"))
def test_asset_root_and_lock_partial_pair_fails_without_fetch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    present: str,
) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    layout.source_checkout.mkdir(parents=True)
    layout.source_lock.parent.mkdir(parents=True, exist_ok=True)
    layout.source_lock.write_text("{}\n", encoding="utf-8")
    import matching_fairness.provenance as provenance

    monkeypatch.setattr(
        provenance,
        "inspect_checkout",
        lambda _path: SimpleNamespace(to_dict=lambda: {}),
    )
    if present == "root":
        layout.asset_root.mkdir(parents=True)
    else:
        layout.asset_lock.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "_run",
        lambda _command: pytest.fail("partial asset state triggered a fetch"),
    )
    with pytest.raises(ValueError, match="asset|partial"):
        module._ensure_source_and_assets(layout)


def _write_brainrw_snapshot_fixture(layout) -> bytes:
    import numpy as np
    import torch

    image_ids = [f"image-{index:03d}" for index in range(200)]
    sessions = np.tile(np.repeat(np.arange(4), 20), (200, 1))
    payload = {
        "eeg": torch.empty((200, 80, 63, 250), device="meta"),
        "label": np.tile(np.arange(200)[:, None], (1, 80)),
        "img": np.asarray([[f"{image_id}.jpg"] * 80 for image_id in image_ids]),
        "text": np.asarray([[image_id] * 80 for image_id in image_ids]),
        "session": sessions,
        "ch_names": [f"channel-{index}" for index in range(63)],
        "times": np.arange(250),
    }
    layout.brainrw_test.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, layout.brainrw_test)
    encoded = layout.brainrw_test.read_bytes()
    layout.preflight_manifest.parent.mkdir(parents=True, exist_ok=True)
    layout.preflight_manifest.write_text(
        json.dumps(
            {
                "brainrw": {
                    "path": str(layout.brainrw_test),
                    "sha256": hashlib.sha256(encoded).hexdigest(),
                    "eeg_shape": [200, 80, 63, 250],
                    "image_count": 200,
                    "session_counts": {"0": 20, "1": 20, "2": 20, "3": 20},
                }
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return encoded


def test_trial_manifest_deserializes_only_verified_snapshot_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    expected = _write_brainrw_snapshot_fixture(layout)
    real_load = torch.load
    observed: list[bytes] = []

    def guarded_load(source, *args, **kwargs):
        assert isinstance(source, io.BytesIO)
        observed.append(source.getvalue())
        return real_load(source, *args, **kwargs)

    monkeypatch.setattr(torch, "load", guarded_load)
    manifest = module._trial_manifest_payload(layout)
    assert manifest["seed"] == 42
    assert observed == [expected]


@pytest.mark.parametrize("mutation", ("replace", "in_place"))
def test_trial_snapshot_mutation_is_rejected_before_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    import torch

    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    _write_brainrw_snapshot_fixture(layout)
    if mutation == "replace":
        replacement = layout.brainrw_test.with_suffix(".replacement")
        replacement.write_bytes(b"replacement")
        os.replace(replacement, layout.brainrw_test)
    else:
        with layout.brainrw_test.open("ab") as stream:
            stream.write(b"in-place mutation")
    monkeypatch.setattr(
        torch,
        "load",
        lambda *_args, **_kwargs: pytest.fail("unverified bytes were deserialized"),
    )
    with pytest.raises(ValueError, match="hash|SHA-256|preflight"):
        module._trial_manifest_payload(layout)


def test_trial_snapshot_rejects_symlink_before_deserialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    _write_brainrw_snapshot_fixture(layout)
    target = layout.brainrw_test.with_suffix(".real")
    layout.brainrw_test.rename(target)
    layout.brainrw_test.symlink_to(target)
    monkeypatch.setattr(
        torch,
        "load",
        lambda *_args, **_kwargs: pytest.fail("symlinked bytes were deserialized"),
    )
    with pytest.raises(ValueError, match="symlink|regular file|nofollow"):
        module._trial_manifest_payload(layout)


def test_trial_snapshot_hash_mismatch_never_reaches_deserializer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    _write_brainrw_snapshot_fixture(layout)
    preflight = json.loads(layout.preflight_manifest.read_text(encoding="utf-8"))
    preflight["brainrw"]["sha256"] = "0" * 64
    layout.preflight_manifest.write_text(json.dumps(preflight) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        torch,
        "load",
        lambda *_args, **_kwargs: pytest.fail("hash mismatch was deserialized"),
    )
    with pytest.raises(ValueError, match="hash|SHA-256|preflight"):
        module._trial_manifest_payload(layout)


def _publish_real_brainrw_fixture(module, layout) -> None:
    import numpy as np

    from matching_fairness.artifacts import ScoreArtifact, write_score_artifact
    from matching_fairness.provenance import sha256_file, sha256_path

    layout.protocol.parent.mkdir(parents=True, exist_ok=True)
    layout.protocol.write_text('{"formal":true}\n', encoding="utf-8")
    layout.trial_manifest.parent.mkdir(parents=True, exist_ok=True)
    layout.trial_manifest.write_text('{"seed":42}\n', encoding="utf-8")
    layout.brainrw_test.parent.mkdir(parents=True, exist_ok=True)
    layout.brainrw_test.write_bytes(b"brainrw-test")
    layout.official_test_images.mkdir(parents=True, exist_ok=True)
    (layout.official_test_images / "image-000.jpg").write_bytes(b"image")
    for name in ("brain_model", "vision_model"):
        path = layout.brainrw_model_root / name
        path.mkdir(parents=True, exist_ok=True)
        (path / "weights.bin").write_bytes(name.encode("ascii"))
    layout.clip_root.mkdir(parents=True, exist_ok=True)
    (layout.clip_root / "weights.bin").write_bytes(b"clip")
    evaluator = layout.repository_root / "scripts/evaluate_retrieval.py"
    model_content = {
        "brain_model": sha256_path(layout.brainrw_model_root / "brain_model"),
        "vision_adapter": sha256_path(layout.brainrw_model_root / "vision_model"),
        "pretrained_vision_base": sha256_path(layout.clip_root),
    }
    ids = tuple(f"image-{index:03d}" for index in range(200))
    exporter = _load_script("export_brainrw_scores")
    arguments = exporter.build_parser().parse_args(
        module.phase_commands(layout)["export_brainrw"][0][2:]
    )

    def runner(command: list[str], *, check: bool) -> object:
        assert check is True
        half = command[command.index("--trial-half") + 1]
        artifact_path = Path(command[command.index("--score-artifact-output") + 1])
        metadata = {
            "model_slug": "our_project",
            "trial_half": half,
            "checkpoint_role": "fixed_formal",
            "checkpoint": str(layout.brainrw_model_root / "brain_model"),
            "checkpoint_content_sha256": model_content["brain_model"],
            "similarity": "cosine",
            "query_embeddings_sha256": {
                "standard": "1" * 64,
                "a": "2" * 64,
                "b": "3" * 64,
            }[half],
            "subject": "sub-08",
            "seed": 42,
            "trial_manifest_sha256": sha256_file(layout.trial_manifest),
            "protocol_sha256": sha256_file(layout.protocol),
            "brain_test_sha256": sha256_file(layout.brainrw_test),
            "model_content_sha256": model_content,
            "evaluator_version": "AIAA3800-BRAINRW-FORMAL-v1",
            "evaluator_sha256": sha256_file(evaluator),
            "runtime_inputs": {
                "test_image_tree_sha256": sha256_path(layout.official_test_images),
                "selected_channel_indices": list(range(46, 63)),
                "time_slice": [0, 250],
                "dataset_name": "things",
                "expected_sample_count": 200,
            },
            "native_metrics": {
                "top1_count": 200,
                "top5_count": 200,
                "sample_count": 200,
            },
        }
        write_score_artifact(
            artifact_path,
            ScoreArtifact(
                similarity=np.eye(200, dtype=np.float32),
                query_ids=ids,
                gallery_entry_ids=ids,
                gallery_canonical_ids=ids,
                target_canonical_ids=ids,
                metadata=metadata,
            ),
        )
        metrics = Path(command[command.index("--metrics-output") + 1])
        predictions = Path(command[command.index("--predictions-output") + 1])
        metrics.parent.mkdir(parents=True, exist_ok=True)
        metrics.write_text('{"top1_count":200,"top5_count":200}\n', encoding="utf-8")
        predictions.write_text("query,prediction\n", encoding="utf-8")
        return object()

    exporter.export_brainrw_scores(arguments, runner=runner)


def test_orchestrator_accepts_real_brainrw_publisher_schema(tmp_path: Path) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    _publish_real_brainrw_fixture(module, layout)
    assert module._matrix_matches_inputs(layout, "our_project") is True


def test_orchestrator_delegates_to_shared_brainrw_tree_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import matching_fairness.formal_artifacts as formal_artifacts

    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    _publish_real_brainrw_fixture(module, layout)
    monkeypatch.setattr(
        formal_artifacts,
        "validate_brainrw_export_tree",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("shared validator sentinel")
        ),
    )

    assert module._matrix_matches_inputs(layout, "our_project") is False


@pytest.mark.parametrize(
    "tamper", ("missing", "extra", "artifact_hash", "run_hash", "symlink")
)
def test_orchestrator_rejects_brainrw_export_tree_tampering(
    tmp_path: Path, tamper: str
) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    _publish_real_brainrw_fixture(module, layout)
    root = layout.matrix_dir("our_project")
    manifest_path = root / "export_manifest.json"
    if tamper == "missing":
        manifest_path.unlink()
    elif tamper == "extra":
        (root / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    elif tamper in {"artifact_hash", "run_hash"}:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if tamper == "artifact_hash":
            payload["artifacts"]["standard"]["sha256"] = "0" * 64
        else:
            payload["runs"]["standard"]["sha256"] = "0" * 64
        manifest_path.write_text(
            json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        member = root / "runs/standard/predictions.csv"
        outside = tmp_path / "outside.csv"
        outside.write_text("outside", encoding="utf-8")
        member.unlink()
        member.symlink_to(outside)
    assert module._matrix_matches_inputs(layout, "our_project") is False


def _write_complete_artifact(path: Path, token: str) -> None:
    path.mkdir(parents=True)
    (path / "metadata.json").write_text(
        json.dumps({"token": token}, sort_keys=True) + "\n", encoding="utf-8"
    )
    (path / "similarity.npy").write_bytes(b"matrix")


def _write_exact_audit_fixture(module, layout, model: str = "nice") -> None:
    checkpoint = {
        "schema_version": 1,
        "model": model,
        "encoder_type": "NICE" if model == "nice" else "ATMS",
        "subject": "sub-08",
        "seed": 42,
        "source": {
            "url": "https://github.com/dongyangli-del/EEG_Image_decode.git",
            "branch": "develop",
            "commit": "1" * 40,
            "checkout_sha256": "2" * 64,
        },
        "inputs": {
            "training_eeg": {
                "name": "preprocessed_eeg_training.npy", "sha256": "3" * 64
            },
            "training_features": {
                "name": "ViT-H-14_features_train.pt", "sha256": "4" * 64
            },
        },
        "hyperparameters": {
            "epochs": 500, "batch_size": 1024, "learning_rate": 3e-4,
            "val_ratio": 0.1, "early_stopping_patience": 10,
            "ema_decay": 0.999, "logit_scale_type": "exp",
            "avg_trials": True, "n_chans": 63, "n_times": 250,
        },
        "encoder_behavior": {
            "use_subject_id": model == "atm_s",
            "normalize_feats": model == "atm_s",
        },
        "checkpoints": [
            {
                "epoch": 1, "val_loss": 0.3,
                "checkpoint": "epoch_0001.pth", "sha256": "5" * 64,
            },
            {
                "epoch": 2, "val_loss": 0.2,
                "checkpoint": "epoch_0002.pth", "sha256": "6" * 64,
            },
        ],
        "selection": {
            "epoch": 2, "val_loss": 0.2, "checkpoint": "epoch_0002.pth"
        },
        "best_checkpoint": {"name": "best_val.pth", "sha256": "6" * 64},
        "history": {"name": "history.csv", "sha256": "7" * 64},
        "stopped_early": True,
    }
    checkpoint_path = layout.checkpoint_dir(model) / "checkpoint_manifest.json"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(
        json.dumps(checkpoint, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    inventory = []
    trial_half = {"standard": "standard", "eeg_a": "a", "eeg_b": "b"}
    for artifact_model in ("nice", "atm_s", "our_project"):
        for half in ("standard", "eeg_a", "eeg_b"):
            artifact = layout.matrix_dir(artifact_model) / half
            _write_complete_artifact(artifact, f"{artifact_model}-{half}")
            inventory.append(
                {
                    "model_slug": artifact_model,
                    "trial_half": trial_half[half],
                    "path": str(artifact),
                    "sha256": module._score_artifact_sha256(artifact),
                }
            )
    inventory.sort(key=lambda row: (row["model_slug"], row["trial_half"]))
    runs = [
        {
            "epoch": 1, "checkpoint": "epoch_0001.pth",
            "checkpoint_sha256": "5" * 64, "effective_logit_scale": 1.0,
            "top1_count": 100, "top5_count": 150, "sample_count": 200,
        },
        {
            "epoch": 2, "checkpoint": "epoch_0002.pth",
            "checkpoint_sha256": "6" * 64, "effective_logit_scale": 1.1,
            "top1_count": 101, "top5_count": 151, "sample_count": 200,
        },
    ]
    audit = {
        "schema_version": 1,
        "scope": "best_test_audit_only",
        "model_slug": model,
        "checkpoint_policy": "every_epoch_checkpoint",
        "fairness_artifact_created": False,
        "formal_artifact_inventory": inventory,
        "runs": runs,
        "best_test": dict(runs[1]),
    }
    audit_path = layout.matrix_dir(model) / "best_test_audit.json"
    audit_path.write_text(
        json.dumps(audit, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_audit_resume_accepts_task8_exact_current_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    _write_exact_audit_fixture(module, layout)
    monkeypatch.setattr(module, "_checkpoint_matches_inputs", lambda *_args: True)
    monkeypatch.setattr(module, "_matrix_matches_inputs", lambda *_args: True)
    assert module._audit_matches_inputs(layout, "nice") is True


@pytest.mark.parametrize(
    "tamper", ("epoch", "hash", "order", "count", "policy", "best", "extra")
)
def test_audit_resume_rejects_task8_contract_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    _write_exact_audit_fixture(module, layout)
    monkeypatch.setattr(module, "_checkpoint_matches_inputs", lambda *_args: True)
    monkeypatch.setattr(module, "_matrix_matches_inputs", lambda *_args: True)
    path = layout.matrix_dir("nice") / "best_test_audit.json"
    audit = json.loads(path.read_text(encoding="utf-8"))
    if tamper == "epoch":
        audit["runs"][1]["epoch"] = 1
    elif tamper == "hash":
        audit["runs"][1]["checkpoint_sha256"] = "0" * 64
    elif tamper == "order":
        audit["runs"].reverse()
    elif tamper == "count":
        audit["runs"][0]["top5_count"] = 201
    elif tamper == "policy":
        audit["checkpoint_policy"] = "best_only"
    elif tamper == "best":
        audit["best_test"] = dict(audit["runs"][0])
    else:
        audit["unexpected"] = True
    path.write_text(
        json.dumps(audit, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    assert module._audit_matches_inputs(layout, "nice") is False


def test_audit_resume_requires_current_checkpoint_and_standard_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    _write_exact_audit_fixture(module, layout)
    monkeypatch.setattr(module, "_checkpoint_matches_inputs", lambda *_args: False)
    monkeypatch.setattr(module, "_matrix_matches_inputs", lambda *_args: True)
    assert module._audit_matches_inputs(layout, "nice") is False
    monkeypatch.setattr(module, "_checkpoint_matches_inputs", lambda *_args: True)
    monkeypatch.setattr(module, "_matrix_matches_inputs", lambda *_args: False)
    assert module._audit_matches_inputs(layout, "nice") is False


def test_resume_matching_skips_only_complete_hash_bound_output(tmp_path: Path) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    layout.protocol.parent.mkdir(parents=True, exist_ok=True)
    layout.protocol.write_text('{}\n', encoding="utf-8")
    layout.trial_manifest.parent.mkdir(parents=True)
    layout.trial_manifest.write_text('{}\n', encoding="utf-8")
    for model in ("nice", "atm_s", "our_project"):
        for half in ("standard", "eeg_a", "eeg_b"):
            _write_complete_artifact(layout.matrix_dir(model) / half, f"{model}-{half}")
    (layout.runs_root / "scenario_manifest.json").parent.mkdir(parents=True)
    (layout.runs_root / "scenario_manifest.json").write_text("{}\n", encoding="utf-8")
    for model in ("nice", "atm_s", "our_project"):
        for index in range(30):
            suite = "standard" if index < 27 else "duplicate_eeg"
            cell = layout.runs_root / model / "subj08" / "seed42" / suite / f"{index:02d}_cell"
            cell.mkdir(parents=True)
            (cell / "summary.json").write_text("{}\n", encoding="utf-8")
            (cell / "per_query.csv").write_text("header\n", encoding="utf-8")
    module.write_phase_manifest(layout, "match")
    assert module.phase_action(layout, "match", overwrite=False) == "skip"

    module.phase_manifest_path(layout, "match").write_text(
        '{"tampered":true}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="manifest|hash"):
        module.phase_action(layout, "match", overwrite=False)


def test_partial_output_fails_and_overwrite_is_phase_scoped(tmp_path: Path) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    layout.runs_root.mkdir(parents=True)
    (layout.runs_root / "partial.txt").write_text("partial", encoding="utf-8")
    layout.aggregate_root.mkdir(parents=True)
    marker = layout.aggregate_root / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="partial|mismatch"):
        module.phase_action(layout, "match", overwrite=False)
    assert module.phase_action(layout, "match", overwrite=True) == "run"
    assert not layout.runs_root.exists()
    assert marker.read_text(encoding="utf-8") == "keep"


def test_overwrite_never_removes_checkpoints_or_source_assets(tmp_path: Path) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    checkpoint = layout.checkpoint_dir("nice")
    checkpoint.mkdir(parents=True)
    (checkpoint / "partial.pth").write_bytes(b"checkpoint")
    layout.source_lock.parent.mkdir(parents=True)
    layout.source_lock.write_text("locked\n", encoding="utf-8")
    layout.asset_lock.write_text("locked\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint"):
        module.phase_action(layout, "train", overwrite=True, array_id=0)
    assert checkpoint.exists()
    assert layout.source_lock.exists()
    assert layout.asset_lock.exists()


def test_training_resume_does_not_skip_stale_input_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    checkpoint = layout.checkpoint_dir("nice")
    checkpoint.mkdir(parents=True)
    monkeypatch.setattr(module, "_checkpoint_complete", lambda _path: True)
    monkeypatch.setattr(
        module,
        "_checkpoint_matches_inputs",
        lambda _layout, _model: False,
        raising=False,
    )
    with pytest.raises(ValueError, match="checkpoint|mismatch"):
        module.phase_action(layout, "train", overwrite=False, array_id=0)


def test_runtime_output_symlinks_are_rejected(tmp_path: Path) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    layout.results_root.parent.mkdir(parents=True)
    layout.results_root.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        module.prepare_runtime_directories(layout)


def test_rendered_commands_match_actual_cli_option_names(tmp_path: Path) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    commands = module.phase_commands(layout)
    train = commands["train"][0]
    assert "--source-checkout" in train and "--training-eeg" in train
    native = commands["export_native"][0]
    for option in (
        "--asset-lock", "--test-features", "--trial-manifest", "--checkpoint-dir"
    ):
        assert option in native
    brainrw = commands["export_brainrw"][0]
    for option in (
        "--brain-model-path", "--vision-adapter-path",
        "--pretrained-model-name-or-path", "--trial-split-manifest",
    ):
        assert option in brainrw
    match = commands["match"][0]
    assert match[match.index("--output-dir") + 1] == str(layout.runs_root)
    assert "/runs/runs" not in " ".join(match)
    aggregate = commands["aggregate"][0]
    assert aggregate[aggregate.index("--results-root") + 1] == str(layout.results_root)


def test_rendered_argv_is_accepted_by_actual_cli_parsers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    commands = module.phase_commands(layout)
    assert _load_script("train_native").build_parser().parse_args(
        commands["train"][0][2:]
    ).model == "nice"
    assert _load_script("export_native_scores").build_parser().parse_args(
        commands["export_native"][1][2:]
    ).model == "atm_s"
    assert _load_script("export_brainrw_scores").build_parser().parse_args(
        commands["export_brainrw"][0][2:]
    ).subject_id == 8
    aggregate = _load_script("aggregate_results").build_parser().parse_args(
        commands["aggregate"][0][2:]
    )
    assert aggregate.results_root == layout.results_root

    scenario_module = _load_script("run_scenarios")
    monkeypatch.setattr(sys, "argv", commands["match"][0][1:])
    scenario = scenario_module.parse_args()
    assert scenario.artifact_root == layout.matrices_root
    assert scenario.output_dir == layout.runs_root


def test_matrix_paths_match_task7_and_task8_direct_contract(tmp_path: Path) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    commands = module.phase_commands(layout)
    for model in ("nice", "atm_s", "our_project"):
        assert layout.matrix_dir(model) == layout.matrices_root / model
        assert "subj08" not in layout.matrix_dir(model).parts
        assert "seed42" not in layout.matrix_dir(model).parts
    emitted = "\n".join(
        " ".join(command)
        for command_group in commands.values()
        for command in command_group
    )
    assert "/matrices/nice/subj08" not in emitted
    assert "/matrices/atm_s/subj08" not in emitted
    assert "/matrices/our_project/subj08" not in emitted
    assert commands["match"][0][commands["match"][0].index("--artifact-root") + 1] == str(
        layout.matrices_root
    )
