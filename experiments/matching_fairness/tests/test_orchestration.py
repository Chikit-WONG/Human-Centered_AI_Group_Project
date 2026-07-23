from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

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
    assert submission["jobs"] == result


def test_existing_submission_manifest_prevents_duplicate_submit(tmp_path: Path) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    layout.submission_manifest.parent.mkdir(parents=True)
    layout.submission_manifest.write_text("{}\n", encoding="utf-8")
    runner = _SubmissionRunner(["1\n"])
    with pytest.raises(FileExistsError, match="submission"):
        module.submit_all(layout=layout, overwrite=False, runner=runner)
    assert runner.calls == []


def test_audit_and_final_dependencies_cannot_be_bypassed(tmp_path: Path) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    commands = module.submission_commands(layout=layout, overwrite=False)
    assert commands["native_audit"]["depends_on"] == (
        "native_export", "brainrw_export"
    )
    assert commands["final"]["depends_on"] == ("native_audit",)


def test_phase_submit_supports_train_and_corrected_export_dag(tmp_path: Path) -> None:
    module = _load_submitter()
    train_layout = module.RuntimeLayout.for_test(tmp_path / "train")
    train_runner = _SubmissionRunner(["11\n"])
    train = module.submit_phase(
        phase="train", layout=train_layout, overwrite=False, runner=train_runner
    )
    assert train == {"train_job_id": 11}
    assert "train_native_array.slurm" in train_runner.calls[0][-1]

    export_layout = module.RuntimeLayout.for_test(tmp_path / "export")
    export_runner = _SubmissionRunner(["21\n", "22\n", "23\n"])
    export = module.submit_phase(
        phase="export", layout=export_layout, overwrite=False, runner=export_runner
    )
    assert export == {
        "native_export_job_id": 21,
        "brainrw_export_job_id": 22,
        "native_audit_job_id": 23,
    }
    assert "--dependency=afterok:21:22" in export_runner.calls[2]
    assert any("MODE=audit" in item for item in export_runner.calls[2])


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


def _write_complete_artifact(path: Path, token: str) -> None:
    path.mkdir(parents=True)
    (path / "metadata.json").write_text(
        json.dumps({"token": token}, sort_keys=True) + "\n", encoding="utf-8"
    )
    (path / "similarity.npy").write_bytes(b"matrix")


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
