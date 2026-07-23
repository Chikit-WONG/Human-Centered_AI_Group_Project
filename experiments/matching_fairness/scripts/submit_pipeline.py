#!/usr/bin/env python3
"""Fixed-scope orchestration for the sealed sub-08 / seed-42 experiment."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Callable, Iterator, Mapping, Sequence
import uuid


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

SUBJECT = "sub-08"
SEED = 42
MODELS = ("nice", "atm_s", "our_project")
NATIVE_MODELS = ("nice", "atm_s")
ARTIFACT_HALVES = ("standard", "eeg_a", "eeg_b")
CHANNELS = "P7,P5,P3,P1,Pz,P2,P4,P6,P8,PO7,PO3,POz,PO4,PO8,O1,Oz,O2"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_AGGREGATE_FILES = {
    "RESULTS.md",
    "RESULTS_ZH.md",
    "aggregate_metrics.csv",
    "aggregate_summary.json",
    "presentation_duplicate_eeg.md",
    "presentation_standard.md",
}


@dataclass(frozen=True)
class RuntimeLayout:
    experiment_root: Path
    repository_root: Path
    results_root: Path
    logs_root: Path
    protocol: Path
    source_checkout: Path
    source_lock: Path
    asset_root: Path
    asset_lock: Path
    preflight_manifest: Path
    trial_manifest: Path
    things_root: Path
    brainrw_test: Path
    official_test_images: Path
    brainrw_model_root: Path
    clip_root: Path

    @classmethod
    def fixed(cls) -> "RuntimeLayout":
        final_project = REPOSITORY_ROOT.parent
        brainrw_root = final_project / "test/brain-rw"
        results = brainrw_root / "results/matching_fairness_v3"
        eeg_project = Path(
            "/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/EEG_Project"
        )
        things = eeg_project / "EEG_Recon-RL/datasets/things_eeg_data"
        asset_root = eeg_project / "models/EEG_Image_decode_assets"
        manifests = results / "manifests"
        return cls(
            experiment_root=EXPERIMENT_ROOT,
            repository_root=REPOSITORY_ROOT,
            results_root=results,
            logs_root=brainrw_root / "logs/matching_fairness_v3",
            protocol=EXPERIMENT_ROOT / "configs/protocol_sub08_seed42.json",
            source_checkout=eeg_project
            / "reference_code/codes_for_papers/EEG_Image_decode",
            source_lock=manifests / "upstream_lock.json",
            asset_root=asset_root,
            asset_lock=manifests / "assets_lock.json",
            preflight_manifest=manifests / "preflight.json",
            trial_manifest=manifests / "trial_split_sub08_seed42.json",
            things_root=things,
            brainrw_test=things
            / "Preprocessed_data_250Hz_whiten/sub-08/test.pt",
            official_test_images=things / "test_images",
            brainrw_model_root=brainrw_root / "runs/seed42/subj08",
            clip_root=eeg_project / "models/CLIP-ViT-B-32-laion2B-s34B-b79K",
        )

    @classmethod
    def for_test(cls, root: Path) -> "RuntimeLayout":
        root = Path(root)
        runtime = root / "runtime"
        results = runtime / "results/matching_fairness_v3"
        manifests = results / "manifests"
        things = root / "things"
        return cls(
            experiment_root=EXPERIMENT_ROOT,
            repository_root=REPOSITORY_ROOT,
            results_root=results,
            logs_root=runtime / "logs/matching_fairness_v3",
            protocol=root / "protocol_sub08_seed42.json",
            source_checkout=root / "upstream",
            source_lock=manifests / "upstream_lock.json",
            asset_root=root / "assets",
            asset_lock=manifests / "assets_lock.json",
            preflight_manifest=manifests / "preflight.json",
            trial_manifest=manifests / "trial_split_sub08_seed42.json",
            things_root=things,
            brainrw_test=things / "Preprocessed_data_250Hz_whiten/sub-08/test.pt",
            official_test_images=things / "test_images",
            brainrw_model_root=root / "brainrw_model/subj08/seed42",
            clip_root=root / "clip",
        )

    @property
    def manifests_root(self) -> Path:
        return self.results_root / "manifests"

    @property
    def runs_root(self) -> Path:
        return self.results_root / "runs"

    @property
    def aggregate_root(self) -> Path:
        return self.results_root / "aggregate"

    @property
    def submission_manifest(self) -> Path:
        return self.manifests_root / "submission.json"

    @property
    def matrices_root(self) -> Path:
        return self.results_root / "matrices"

    def matrix_dir(self, model: str) -> Path:
        _validate_model(model)
        # Tasks 7 and 8 consume matrices/<model>/<half> directly. Subject and
        # seed remain sealed in every artifact and orchestration manifest.
        return self.matrices_root / model

    def checkpoint_dir(self, model: str) -> Path:
        if model not in NATIVE_MODELS:
            raise ValueError(f"checkpoint model must be NICE or ATM-S: {model}")
        return self.results_root / "checkpoints" / model

    @property
    def training_eeg(self) -> Path:
        return self.asset_root / (
            "Preprocessed_data_250Hz/sub-08/preprocessed_eeg_training.npy"
        )

    @property
    def test_eeg(self) -> Path:
        return self.asset_root / "Preprocessed_data_250Hz/sub-08/preprocessed_eeg_test.npy"

    @property
    def training_features(self) -> Path:
        return self.asset_root / "ViT-H-14_features_train.pt"

    @property
    def test_features(self) -> Path:
        return self.asset_root / "ViT-H-14_features_test.pt"


def model_for_array_id(array_id: int) -> str:
    if isinstance(array_id, bool) or array_id not in (0, 1):
        raise ValueError(f"array ID must be exactly 0 or 1, found {array_id!r}")
    return NATIVE_MODELS[array_id]


def phase_commands(layout: RuntimeLayout) -> dict[str, list[list[str]]]:
    python = sys.executable
    scripts = layout.experiment_root / "scripts"
    source_common = [
        "--source-checkout", str(layout.source_checkout),
        "--source-lock", str(layout.source_lock),
    ]
    train: list[list[str]] = []
    export_native: list[list[str]] = []
    for model in NATIVE_MODELS:
        train.append(
            [
                python, str(scripts / "train_native.py"),
                "--protocol", str(layout.protocol),
                *source_common,
                "--training-eeg", str(layout.training_eeg),
                "--training-features", str(layout.training_features),
                "--output-dir", str(layout.results_root),
                "--model", model,
            ]
        )
        export_native.append(
            [
                python, str(scripts / "export_native_scores.py"),
                "--protocol", str(layout.protocol),
                *source_common,
                "--asset-root", str(layout.asset_root),
                "--asset-lock", str(layout.asset_lock),
                "--test-eeg", str(layout.test_eeg),
                "--test-features", str(layout.test_features),
                "--test-images", str(layout.official_test_images),
                "--trial-manifest", str(layout.trial_manifest),
                "--checkpoint-dir", str(layout.checkpoint_dir(model)),
                "--output-dir", str(layout.matrix_dir(model)),
                "--model", model,
                "--device", "cuda",
                "--mode", "main",
            ]
        )
    brainrw = [
        python, str(scripts / "export_brainrw_scores.py"),
        "--protocol", str(layout.protocol),
        "--brain-model-path", str(layout.brainrw_model_root / "brain_model"),
        "--vision-adapter-path", str(layout.brainrw_model_root / "vision_model"),
        "--pretrained-model-name-or-path", str(layout.clip_root),
        "--brain-directory", str(layout.things_root / "Preprocessed_data_250Hz_whiten"),
        "--image-directory", str(layout.things_root),
        "--selected-channels", CHANNELS,
        "--trial-split-manifest", str(layout.trial_manifest),
        "--output-dir", str(layout.matrix_dir("our_project")),
        "--dataset-name", "things",
        "--subject-id", "8",
        "--time-slice", "0,250",
        "--batch-size", "100",
        "--num-workers", "0",
        "--device", "cuda",
        "--dtype", "bf16",
        "--cache-dir", str(layout.brainrw_model_root / "cache/matching_fairness_v3"),
        "--seed", "42",
        "--expected-num-samples", "200",
        "--expected-top1-count", "182",
        "--expected-top5-count", "199",
        "--local-files-only",
    ]
    return {
        "fetch_source": [[
            python, str(scripts / "fetch_upstream.py"),
            "--path", str(layout.source_checkout),
            "--manifest", str(layout.source_lock),
        ]],
        "fetch_assets": [[
            python, str(scripts / "fetch_assets.py"),
            "--asset-root", str(layout.asset_root),
            "--manifest", str(layout.asset_lock),
        ]],
        "preflight": [[
            python, str(scripts / "preflight.py"),
            "--protocol", str(layout.protocol),
            "--checkout", str(layout.source_checkout),
            "--asset-root", str(layout.asset_root),
            "--brainrw-test", str(layout.brainrw_test),
            "--official-test-images", str(layout.official_test_images),
            "--manifest", str(layout.preflight_manifest),
        ]],
        "train": train,
        "export_native": export_native,
        "export_brainrw": [brainrw],
        "match": [[
            python, str(scripts / "run_scenarios.py"),
            "--protocol", str(layout.protocol),
            "--artifact-root", str(layout.matrices_root),
            "--trial-manifest", str(layout.trial_manifest),
            "--output-dir", str(layout.runs_root),
        ]],
        "aggregate": [[
            python, str(scripts / "aggregate_results.py"),
            "--results-root", str(layout.results_root),
        ]],
    }


def _native_audit_command(layout: RuntimeLayout, model: str) -> list[str]:
    command = list(phase_commands(layout)["export_native"][NATIVE_MODELS.index(model)])
    command[command.index("--mode") + 1] = "audit"
    for artifact_model in MODELS:
        for half in ARTIFACT_HALVES:
            command.extend(
                ["--formal-artifact", str(layout.matrix_dir(artifact_model) / half)]
            )
    return command


def _sbatch_argv(
    script: Path,
    *,
    dependency: str | None,
    overwrite: bool,
    export_mode: str | None = None,
) -> list[str]:
    variables = [
        "ALL",
        f"MATCHING_FAIRNESS_OVERWRITE={1 if overwrite else 0}",
    ]
    if export_mode is not None:
        if export_mode not in {"main", "audit"}:
            raise ValueError("native export mode must be main or audit")
        variables.append(f"MATCHING_FAIRNESS_EXPORT_MODE={export_mode}")
    command = ["sbatch", "--parsable", f"--export={','.join(variables)}"]
    if dependency is not None:
        command.append(f"--dependency={dependency}")
    command.append(str(script))
    return command


def submission_commands(
    *, layout: RuntimeLayout, overwrite: bool
) -> dict[str, Mapping[str, object]]:
    slurm = layout.experiment_root / "slurm"
    return {
        "train": {
            "argv": _sbatch_argv(
                slurm / "train_native_array.slurm",
                dependency=None,
                overwrite=overwrite,
            ),
            "depends_on": (),
        },
        "native_export": {
            "argv": _sbatch_argv(
                slurm / "export_native_array.slurm",
                dependency="afterok:<train_job_id>",
                overwrite=overwrite,
                export_mode="main",
            ),
            "depends_on": ("train",),
        },
        "brainrw_export": {
            "argv": _sbatch_argv(
                slurm / "export_brainrw.slurm",
                dependency=None,
                overwrite=overwrite,
            ),
            "depends_on": (),
        },
        "native_audit": {
            "argv": _sbatch_argv(
                slurm / "export_native_array.slurm",
                dependency="afterok:<native_export_job_id>:<brainrw_export_job_id>",
                overwrite=overwrite,
                export_mode="audit",
            ),
            "depends_on": ("native_export", "brainrw_export"),
        },
        "final": {
            "argv": _sbatch_argv(
                slurm / "fairness_cpu.slurm",
                dependency="afterok:<native_audit_job_id>",
                overwrite=overwrite,
            ),
            "depends_on": ("native_audit",),
        },
    }


def render_submission_plan(*, layout: RuntimeLayout, overwrite: bool) -> str:
    commands = submission_commands(layout=layout, overwrite=overwrite)
    lines = [
        "scope: subject=sub-08 seed=42 models=nice,atm_s,our_project",
        "native array: 0=nice 1=atm_s --array=0-1%2",
        "counts: training_cells=2 main_exports=3 native_audit_cells=2 "
        "scenarios=90 decoder_records=450",
        "preflight (local, no formal test metric):",
    ]
    for name in ("fetch_source", "fetch_assets", "preflight"):
        lines.append(f"  {name}: {shlex.join(phase_commands(layout)[name][0])}")
    lines.append("  trial_manifest: Task4 session-balanced 10/10 real-trial split")
    for name in ("train", "native_export", "brainrw_export", "native_audit", "final"):
        entry = commands[name]
        lines.append(f"{name}: {shlex.join(entry['argv'])}")
    lines.extend(
        [
            "dependency graph:",
            "  preflight -> train",
            "  train -> native_export (afterok:<train_job_id>)",
            "  preflight -> brainrw_export (parallel; no training dependency)",
            "  native_export + brainrw_export -> native_audit "
            "(afterok:<native_export_job_id>:<brainrw_export_job_id>; MODE=audit)",
            "  native_audit -> match -> aggregate "
            "(afterok:<native_audit_job_id>)",
            f"results_root: {layout.results_root}",
            f"logs_root: {layout.logs_root}",
        ]
    )
    return "\n".join(lines) + "\n"


def render_phase_plan(
    *, layout: RuntimeLayout, phase: str, overwrite: bool
) -> str:
    if phase == "all":
        return render_submission_plan(layout=layout, overwrite=overwrite)
    commands = phase_commands(layout)
    header = [
        "scope: subject=sub-08 seed=42 models=nice,atm_s,our_project",
        f"phase: {phase}",
    ]
    if phase == "preflight":
        for name in ("fetch_source", "fetch_assets", "preflight"):
            header.append(f"{name}: {shlex.join(commands[name][0])}")
        header.append("trial_manifest: Task4 session-balanced 10/10 real-trial split")
    elif phase == "train":
        header.extend(
            [
                "native array: 0=nice 1=atm_s --array=0-1%2",
                shlex.join(
                    submission_commands(layout=layout, overwrite=overwrite)["train"]["argv"]
                ),
            ]
        )
    elif phase == "export":
        entries = submission_commands(layout=layout, overwrite=overwrite)
        # A standalone export phase assumes the sealed training phase already finished.
        native_main = _sbatch_argv(
            layout.experiment_root / "slurm/export_native_array.slurm",
            dependency=None,
            overwrite=overwrite,
            export_mode="main",
        )
        header.extend(
            [
                "native array: 0=nice 1=atm_s --array=0-1%2",
                f"native_export: {shlex.join(native_main)}",
                f"brainrw_export: {shlex.join(entries['brainrw_export']['argv'])}",
                "native_audit: " + shlex.join(entries["native_audit"]["argv"]),
                "native_export + brainrw_export -> native_audit "
                "(afterok:<native_export_job_id>:<brainrw_export_job_id>; MODE=audit)",
            ]
        )
    elif phase in {"match", "aggregate"}:
        header.append(shlex.join(commands[phase][0]))
    else:
        raise ValueError(f"unknown phase: {phase}")
    return "\n".join(header) + "\n"


def _parse_job_id(output: str) -> int:
    if re.fullmatch(r"[1-9][0-9]*\n?", output) is None:
        raise RuntimeError(f"sbatch did not return an exact numeric job ID: {output!r}")
    return int(output.rstrip("\n"))


def _submit(
    argv: Sequence[str],
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> int:
    completed = runner(
        list(argv),
        check=True,
        text=True,
        capture_output=True,
    )
    return _parse_job_id(completed.stdout)


def submit_all(
    *,
    layout: RuntimeLayout,
    overwrite: bool,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, int]:
    return _submit_workflow(
        phase="all",
        layout=layout,
        overwrite=overwrite,
        runner=runner,
    )


def submit_phase(
    *,
    phase: str,
    layout: RuntimeLayout,
    overwrite: bool,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, int]:
    if phase not in {"train", "export"}:
        raise ValueError("phase submission is supported only for train or export")
    return _submit_workflow(
        phase=phase,
        layout=layout,
        overwrite=overwrite,
        runner=runner,
    )


_WORKFLOW_ORDER = {
    "train": ("train",),
    "export": ("native_export", "brainrw_export", "native_audit"),
    "all": (
        "train", "native_export", "brainrw_export", "native_audit", "final"
    ),
}


@contextmanager
def _submission_lock(path: Path) -> Iterator[None]:
    lock_path = path.with_name(f"{path.name}.lock")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as error:
        raise RuntimeError("could not open the submission ledger lock") from error
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another submission phase holds the active ledger lock") from error
        yield
    finally:
        os.close(descriptor)


def _new_submission_request(phase: str, overwrite: bool) -> dict[str, object]:
    request_id = uuid.uuid4().hex
    order = _WORKFLOW_ORDER[phase]
    return {
        "phase": phase,
        "state": "active",
        "overwrite": overwrite,
        "request_id": request_id,
        "job_order": list(order),
        "jobs": {
            name: {
                "state": "planned",
                "token": f"mf-{request_id}-{name.replace('_', '-')}",
                "argv": None,
                "job_id": None,
                "error": None,
            }
            for name in order
        },
        "failure": None,
    }


def _new_submission_ledger(
    *,
    mode: str,
    phase: str,
    overwrite: bool,
) -> dict[str, object]:
    return {
        "schema_version": 3,
        "subject": SUBJECT,
        "seed": SEED,
        "models": list(MODELS),
        "mode": mode,
        "requests": {phase: _new_submission_request(phase, overwrite)},
    }


def _read_submission_ledger(path: Path) -> dict[str, object]:
    from matching_fairness.reporting import _canonical_json, _read_regular_file

    payload = _canonical_json(
        _read_regular_file(path, "submission ledger"),
        "submission ledger",
    )
    if set(payload) != {
        "schema_version", "subject", "seed", "models", "mode", "requests"
    } or (
        payload.get("schema_version") != 3
        or payload.get("subject") != SUBJECT
        or payload.get("seed") != SEED
        or payload.get("models") != list(MODELS)
        or payload.get("mode") not in {"all", "phased"}
        or not isinstance(payload.get("requests"), Mapping)
    ):
        raise RuntimeError("submission ledger schema/identity is invalid")
    requests = payload["requests"]
    assert isinstance(requests, Mapping)
    expected_request_keys = {"all"} if payload["mode"] == "all" else {"train", "export"}
    if not set(requests).issubset(expected_request_keys) or not requests:
        raise RuntimeError("submission ledger phase set is invalid")
    if payload["mode"] == "all" and set(requests) != {"all"}:
        raise RuntimeError("all-mode ledger must contain exactly the all request")
    for phase, request in requests.items():
        _validate_submission_request(str(phase), request)
    return dict(payload)


def _validate_submission_request(phase: str, value: object) -> None:
    if phase not in _WORKFLOW_ORDER or not isinstance(value, Mapping):
        raise RuntimeError("submission request phase is invalid")
    if set(value) != {
        "phase", "state", "overwrite", "request_id", "job_order", "jobs", "failure"
    } or (
        value.get("phase") != phase
        or value.get("state") not in {"active", "completed", "failed", "unknown"}
        or not isinstance(value.get("overwrite"), bool)
        or re.fullmatch(r"[0-9a-f]{32}", str(value.get("request_id", ""))) is None
        or value.get("job_order") != list(_WORKFLOW_ORDER[phase])
        or not isinstance(value.get("jobs"), Mapping)
    ):
        raise RuntimeError("submission request schema/identity is invalid")
    jobs = value["jobs"]
    assert isinstance(jobs, Mapping)
    if set(jobs) != set(_WORKFLOW_ORDER[phase]):
        raise RuntimeError("submission request job set is invalid")
    request_id = str(value["request_id"])
    states: list[str] = []
    for name in _WORKFLOW_ORDER[phase]:
        row = jobs[name]
        if not isinstance(row, Mapping) or set(row) != {
            "state", "token", "argv", "job_id", "error"
        }:
            raise RuntimeError("submission job ledger row is invalid")
        state = str(row["state"])
        argv = row["argv"]
        job_id = row["job_id"]
        error = row["error"]
        valid_argv = (
            isinstance(argv, list)
            and bool(argv)
            and all(isinstance(argument, str) and argument for argument in argv)
        )
        if (
            state not in {"planned", "submitting", "submitted", "failed", "unknown"}
            or row["token"]
            != f"mf-{request_id}-{str(name).replace('_', '-')}"
            or (
                state == "planned"
                and (argv is not None or job_id is not None or error is not None)
            )
            or (
                state == "submitting"
                and (not valid_argv or job_id is not None or error is not None)
            )
            or (
                state == "submitted"
                and (
                    not valid_argv
                    or isinstance(job_id, bool)
                    or not isinstance(job_id, int)
                    or job_id <= 0
                    or error is not None
                )
            )
            or (
                state in {"failed", "unknown"}
                and (
                    not valid_argv
                    or job_id is not None
                    or not isinstance(error, str)
                    or not error
                )
            )
        ):
            raise RuntimeError("submission job ledger row value is invalid")
        states.append(state)

    request_state = str(value["state"])
    failure = value["failure"]
    if request_state == "active":
        if failure is not None or any(state in {"failed", "unknown"} for state in states):
            raise RuntimeError("active submission request state is inconsistent")
        unfinished = False
        submitting_seen = False
        for state in states:
            if state == "submitted":
                if unfinished:
                    raise RuntimeError("active submission job order is inconsistent")
            elif state == "submitting":
                if unfinished or submitting_seen:
                    raise RuntimeError("active submission job order is inconsistent")
                unfinished = True
                submitting_seen = True
            else:
                unfinished = True
    elif request_state == "completed":
        if failure is not None or any(state != "submitted" for state in states):
            raise RuntimeError("completed submission request state is inconsistent")
    else:
        if (
            not isinstance(failure, Mapping)
            or set(failure) != {"stage", "error"}
            or failure.get("stage") not in _WORKFLOW_ORDER[phase]
            or not isinstance(failure.get("error"), str)
            or not failure["error"]
        ):
            raise RuntimeError("failed submission request record is invalid")
        failed_index = _WORKFLOW_ORDER[phase].index(str(failure["stage"]))
        expected_states = ["submitted"] * failed_index + [request_state]
        expected_states.extend(
            ["planned"] * (len(_WORKFLOW_ORDER[phase]) - failed_index - 1)
        )
        failed_row = jobs[str(failure["stage"])]
        assert isinstance(failed_row, Mapping)
        if states != expected_states or failed_row["error"] != failure["error"]:
            raise RuntimeError("failed submission request state is inconsistent")


def _replace_submission_ledger(path: Path, payload: Mapping[str, object]) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError("submission ledger must remain a regular file")
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.replace-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _workflow_command(
    *,
    name: str,
    phase: str,
    layout: RuntimeLayout,
    overwrite: bool,
    job_ids: Mapping[str, int],
    token: str,
) -> list[str]:
    slurm = layout.experiment_root / "slurm"
    if name == "train":
        command = _sbatch_argv(
            slurm / "train_native_array.slurm", dependency=None, overwrite=overwrite
        )
    elif name == "native_export":
        dependency = (
            f"afterok:{job_ids['train']}" if phase == "all" else None
        )
        command = _sbatch_argv(
            slurm / "export_native_array.slurm",
            dependency=dependency,
            overwrite=overwrite,
            export_mode="main",
        )
    elif name == "brainrw_export":
        command = _sbatch_argv(
            slurm / "export_brainrw.slurm", dependency=None, overwrite=overwrite
        )
    elif name == "native_audit":
        command = _sbatch_argv(
            slurm / "export_native_array.slurm",
            dependency=(
                f"afterok:{job_ids['native_export']}:{job_ids['brainrw_export']}"
            ),
            overwrite=overwrite,
            export_mode="audit",
        )
    elif name == "final":
        command = _sbatch_argv(
            slurm / "fairness_cpu.slurm",
            dependency=f"afterok:{job_ids['native_audit']}",
            overwrite=overwrite,
        )
    else:
        raise ValueError(f"unknown submission job: {name}")
    command.insert(-1, f"--job-name={token}")
    return command


def _submit_workflow(
    *,
    phase: str,
    layout: RuntimeLayout,
    overwrite: bool,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> dict[str, int]:
    prepare_runtime_directories(layout)
    if phase not in _WORKFLOW_ORDER:
        raise ValueError(f"unknown submission phase: {phase}")
    with _submission_lock(layout.submission_manifest):
        exists = os.path.lexists(layout.submission_manifest)
        if phase in {"all", "train"}:
            if exists:
                _read_submission_ledger(layout.submission_manifest)
                raise FileExistsError(
                    "submission ledger already contains an all/phased request"
                )
            ledger = _new_submission_ledger(
                mode="all" if phase == "all" else "phased",
                phase=phase,
                overwrite=overwrite,
            )
            _atomic_write_json_noclobber(layout.submission_manifest, ledger)
        else:
            if not exists:
                raise ValueError(
                    "export submission requires a completed train request and checkpoints"
                )
            ledger = _read_submission_ledger(layout.submission_manifest)
            requests = ledger["requests"]
            assert isinstance(requests, dict)
            if ledger["mode"] != "phased" or "all" in requests:
                raise RuntimeError("all and phased submission modes cannot be mixed")
            train = requests.get("train")
            if not isinstance(train, Mapping) or train.get("state") != "completed":
                raise RuntimeError("export requires a completed train ledger request")
            if "export" in requests:
                raise FileExistsError("export submission phase already exists in ledger")
            if not all(
                _checkpoint_matches_inputs(layout, model) for model in NATIVE_MODELS
            ):
                raise ValueError("export requires both current validation-selected checkpoints")
            requests["export"] = _new_submission_request("export", overwrite)
            _replace_submission_ledger(layout.submission_manifest, ledger)

        requests = ledger["requests"]
        assert isinstance(requests, dict)
        request = requests[phase]
        assert isinstance(request, dict)
        order = _WORKFLOW_ORDER[phase]
        jobs = request["jobs"]
        assert isinstance(jobs, dict)
        job_ids: dict[str, int] = {}
        for name in order:
            row = jobs[name]
            assert isinstance(row, dict)
            command = _workflow_command(
                name=name,
                phase=phase,
                layout=layout,
                overwrite=overwrite,
                job_ids=job_ids,
                token=str(row["token"]),
            )
            row["state"] = "submitting"
            row["argv"] = command
            _replace_submission_ledger(layout.submission_manifest, ledger)
            try:
                job_id = _submit(command, runner)
            except subprocess.CalledProcessError as error:
                row["state"] = "failed"
                row["error"] = str(error)
                request["state"] = "failed"
                request["failure"] = {"stage": name, "error": str(error)}
                _replace_submission_ledger(layout.submission_manifest, ledger)
                raise
            except (OSError, RuntimeError) as error:
                row["state"] = "unknown"
                row["error"] = str(error)
                request["state"] = "unknown"
                request["failure"] = {"stage": name, "error": str(error)}
                _replace_submission_ledger(layout.submission_manifest, ledger)
                raise
            row["state"] = "submitted"
            row["job_id"] = job_id
            job_ids[name] = job_id
            _replace_submission_ledger(layout.submission_manifest, ledger)
        request["state"] = "completed"
        _replace_submission_ledger(layout.submission_manifest, ledger)
        return {f"{name}_job_id": job_ids[name] for name in order}


def prepare_runtime_directories(layout: RuntimeLayout) -> None:
    for path in (layout.results_root, layout.logs_root, layout.manifests_root):
        _reject_symlink_target(path)
    layout.results_root.mkdir(parents=True, exist_ok=True)
    layout.logs_root.mkdir(parents=True, exist_ok=True)
    layout.manifests_root.mkdir(parents=True, exist_ok=True)


def _reject_symlink_target(path: Path) -> None:
    cursor = Path(path)
    while True:
        if os.path.lexists(cursor) and cursor.is_symlink():
            raise ValueError(f"runtime path must not contain a symlink: {cursor}")
        if cursor == cursor.parent:
            break
        cursor = cursor.parent


def phase_manifest_path(layout: RuntimeLayout, phase: str) -> Path:
    if phase not in {"preflight", "match", "aggregate"}:
        raise ValueError(f"phase does not use an orchestration manifest: {phase}")
    return layout.manifests_root / f"orchestration_{phase}.json"


def _hash_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required input must be a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_tree(path: Path) -> str:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"required input must be a regular directory: {path}")
    digest = hashlib.sha256()
    for entry in sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix()):
        if entry.is_symlink():
            raise ValueError(f"provenance tree must not contain symlinks: {entry}")
        relative = entry.relative_to(path).as_posix().encode("utf-8")
        digest.update(b"D\0" if entry.is_dir() else b"F\0")
        digest.update(relative)
        digest.update(b"\0")
        if entry.is_file():
            digest.update(bytes.fromhex(_hash_file(entry)))
    return digest.hexdigest()


def _hash_path(path: Path) -> str:
    return _hash_file(path) if path.is_file() and not path.is_symlink() else _hash_tree(path)


def _phase_input_paths(layout: RuntimeLayout, phase: str) -> dict[str, Path]:
    if phase == "match":
        paths = {"protocol": layout.protocol, "trial_manifest": layout.trial_manifest}
        for model in MODELS:
            for half in ARTIFACT_HALVES:
                artifact = layout.matrix_dir(model) / half
                paths[f"{model}:{half}:metadata"] = artifact / "metadata.json"
                paths[f"{model}:{half}:similarity"] = artifact / "similarity.npy"
        return paths
    if phase == "aggregate":
        paths = {
            "runs": layout.runs_root,
            "nice_audit": layout.matrix_dir("nice") / "best_test_audit.json",
            "atm_s_audit": layout.matrix_dir("atm_s") / "best_test_audit.json",
            "nice_checkpoint": layout.checkpoint_dir("nice") / "checkpoint_manifest.json",
            "atm_s_checkpoint": layout.checkpoint_dir("atm_s") / "checkpoint_manifest.json",
        }
        return paths
    if phase == "preflight":
        return {
            "protocol": layout.protocol,
            "source_lock": layout.source_lock,
            "asset_lock": layout.asset_lock,
            "brainrw_test": layout.brainrw_test,
            "official_test_images": layout.official_test_images,
        }
    raise ValueError(f"unsupported phase manifest: {phase}")


def _phase_inputs(layout: RuntimeLayout, phase: str) -> dict[str, str]:
    return {
        name: _hash_path(path)
        for name, path in sorted(_phase_input_paths(layout, phase).items())
    }


def _phase_output(layout: RuntimeLayout, phase: str) -> Path:
    return {
        "match": layout.runs_root,
        "aggregate": layout.aggregate_root,
    }[phase]


def _atomic_write_json_noclobber(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(f"manifest already exists: {path}") from None
    finally:
        temporary.unlink(missing_ok=True)
    _fsync_directory(path.parent)


def write_phase_manifest(layout: RuntimeLayout, phase: str) -> Path:
    if phase == "match" and not _match_output_complete(layout.runs_root):
        raise ValueError("matching output is partial")
    if phase == "aggregate" and not _aggregate_output_complete(layout.aggregate_root):
        raise ValueError("aggregate output is partial")
    output = _phase_output(layout, phase)
    payload = {
        "schema_version": 1,
        "phase": phase,
        "subject": SUBJECT,
        "seed": SEED,
        "input_sha256": _phase_inputs(layout, phase),
        "output_sha256": _hash_tree(output),
    }
    destination = phase_manifest_path(layout, phase)
    _atomic_write_json_noclobber(destination, payload)
    return destination


def _read_json(path: Path) -> object:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"manifest must be a regular file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"manifest is not valid UTF-8 JSON: {path}") from error


def _expected_phase_manifest(layout: RuntimeLayout, phase: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "phase": phase,
        "subject": SUBJECT,
        "seed": SEED,
        "input_sha256": _phase_inputs(layout, phase),
        "output_sha256": _hash_tree(_phase_output(layout, phase)),
    }


def _match_output_complete(path: Path) -> bool:
    if path.is_symlink() or not path.is_dir():
        return False
    files = [entry for entry in path.rglob("*") if entry.is_file()]
    if any(entry.is_symlink() for entry in path.rglob("*")):
        return False
    return (
        (path / "scenario_manifest.json").is_file()
        and sum(entry.name == "summary.json" for entry in files) == 90
        and sum(entry.name == "per_query.csv" for entry in files) == 90
        and len(files) == 181
    )


def _aggregate_output_complete(path: Path) -> bool:
    if path.is_symlink() or not path.is_dir():
        return False
    if {entry.name for entry in path.iterdir()} != _AGGREGATE_FILES:
        return False
    if any(entry.is_symlink() or not entry.is_file() for entry in path.iterdir()):
        return False
    try:
        summary = _read_json(path / "aggregate_summary.json")
    except ValueError:
        return False
    return (
        isinstance(summary, Mapping)
        and summary.get("subject") == SUBJECT
        and summary.get("seed") == SEED
        and summary.get("record_count") == 450
    )


def _safe_remove_derived(path: Path, root: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"refusing to remove symlinked derived output: {path}")
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError as error:
        raise ValueError(f"derived output escapes its fixed root: {path}") from error
    if path.is_dir():
        shutil.rmtree(path)
    elif os.path.lexists(path):
        path.unlink()


def phase_action(
    layout: RuntimeLayout,
    phase: str,
    *,
    overwrite: bool,
    array_id: int | None = None,
) -> str:
    if phase == "train":
        if array_id is None:
            raise ValueError("training resume check requires an array ID")
        checkpoint = layout.checkpoint_dir(model_for_array_id(array_id))
        if not os.path.lexists(checkpoint):
            return "run"
        model = model_for_array_id(array_id)
        if _checkpoint_matches_inputs(layout, model):
            return "skip"
        raise ValueError(
            "checkpoint output is partial or mismatched; checkpoints are never removed "
            "by --overwrite"
        )
    if phase not in {"match", "aggregate"}:
        raise ValueError(f"unsupported public phase resume check: {phase}")
    output = _phase_output(layout, phase)
    manifest = phase_manifest_path(layout, phase)
    complete = (
        _match_output_complete(output)
        if phase == "match"
        else _aggregate_output_complete(output)
    )
    if not os.path.lexists(output) and not os.path.lexists(manifest):
        return "run"
    if overwrite:
        _safe_remove_derived(output, layout.results_root)
        _safe_remove_derived(manifest, layout.manifests_root)
        return "run"
    if not complete or not manifest.is_file() or manifest.is_symlink():
        raise ValueError(f"{phase} output is partial or its resume manifest is missing")
    if _read_json(manifest) != _expected_phase_manifest(layout, phase):
        raise ValueError(f"{phase} resume manifest or input hash mismatch")
    return "skip"


def _checkpoint_complete(path: Path) -> bool:
    try:
        manifest = _read_json(path / "checkpoint_manifest.json")
    except ValueError:
        return False
    if not isinstance(manifest, Mapping):
        return False
    best = manifest.get("best_checkpoint")
    history = manifest.get("history")
    if not isinstance(best, Mapping) or not isinstance(history, Mapping):
        return False
    best_path = path / str(best.get("name", ""))
    history_path = path / str(history.get("name", ""))
    try:
        return (
            best.get("sha256") == _hash_file(best_path)
            and history.get("sha256") == _hash_file(history_path)
            and manifest.get("subject") == SUBJECT
            and manifest.get("seed") == SEED
            and manifest.get("model") == path.name
        )
    except ValueError:
        return False


def _checkpoint_matches_inputs(layout: RuntimeLayout, model: str) -> bool:
    path = layout.checkpoint_dir(model)
    if not _checkpoint_complete(path):
        return False
    try:
        from matching_fairness.config import Protocol

        payload = _read_json(path / "checkpoint_manifest.json")
        source = _read_json(layout.source_lock)
        if not isinstance(payload, Mapping) or not isinstance(source, Mapping):
            return False
        expected_keys = {
            "schema_version", "model", "encoder_type", "subject", "seed",
            "source", "inputs", "hyperparameters", "encoder_behavior",
            "checkpoints", "selection", "best_checkpoint", "history",
            "stopped_early",
        }
        encoder = {
            "nice": {
                "encoder_type": "NICE",
                "use_subject_id": False,
                "normalize_feats": False,
            },
            "atm_s": {
                "encoder_type": "ATMS",
                "use_subject_id": True,
                "normalize_feats": True,
            },
        }[model]
        if (
            set(payload) != expected_keys
            or payload.get("schema_version") != 1
            or payload.get("model") != model
            or payload.get("encoder_type") != encoder["encoder_type"]
            or payload.get("subject") != SUBJECT
            or payload.get("seed") != SEED
            or payload.get("source") != source
            or not isinstance(payload.get("stopped_early"), bool)
        ):
            return False
        protocol = Protocol.load(layout.protocol)
        protocol.assert_formal_scope()
        training = protocol.native_training
        expected_hyperparameters = {
            "epochs": int(training["epochs"]),
            "batch_size": int(training["batch_size"]),
            "learning_rate": float(training["lr"]),
            "val_ratio": float(training["val_ratio"]),
            "early_stopping_patience": int(training["early_stopping_patience"]),
            "ema_decay": float(training["ema_decay"]),
            "logit_scale_type": str(training["logit_scale_type"]),
            "avg_trials": bool(training["avg_trials"]),
            "n_chans": int(training["n_chans"]),
            "n_times": int(training["n_times"]),
        }
        expected_inputs = {
            "training_eeg": {
                "name": layout.training_eeg.name,
                "sha256": _hash_file(layout.training_eeg),
            },
            "training_features": {
                "name": layout.training_features.name,
                "sha256": _hash_file(layout.training_features),
            },
        }
        expected_behavior = {
            "use_subject_id": encoder["use_subject_id"],
            "normalize_feats": encoder["normalize_feats"],
        }
        if (
            payload.get("hyperparameters") != expected_hyperparameters
            or payload.get("inputs") != expected_inputs
            or payload.get("encoder_behavior") != expected_behavior
        ):
            return False
        records = payload.get("checkpoints")
        if not isinstance(records, list) or not records:
            return False
        identities: set[int] = set()
        for record in records:
            if not isinstance(record, Mapping) or set(record) != {
                "epoch", "val_loss", "checkpoint", "sha256"
            }:
                return False
            epoch = record.get("epoch")
            loss = record.get("val_loss")
            if (
                isinstance(epoch, bool)
                or not isinstance(epoch, int)
                or epoch <= 0
                or epoch in identities
                or isinstance(loss, bool)
                or not isinstance(loss, (int, float))
                or not math.isfinite(float(loss))
                or record.get("checkpoint") != f"epoch_{epoch:04d}.pth"
                or record.get("sha256") != _hash_file(path / str(record["checkpoint"]))
            ):
                return False
            identities.add(epoch)
        selected = min(records, key=lambda row: (float(row["val_loss"]), int(row["epoch"])))
        expected_selection = {
            "epoch": selected["epoch"],
            "val_loss": selected["val_loss"],
            "checkpoint": selected["checkpoint"],
        }
        best = payload.get("best_checkpoint")
        return (
            payload.get("selection") == expected_selection
            and isinstance(best, Mapping)
            and best.get("name") == "best_val.pth"
            and best.get("sha256") == selected["sha256"]
            and best.get("sha256") == _hash_file(path / "best_val.pth")
        )
    except (OSError, TypeError, ValueError):
        return False


def _native_matrix_complete(path: Path) -> bool:
    if path.is_symlink() or not path.is_dir():
        return False
    names = {entry.name for entry in path.iterdir()}
    allowed = [set(ARTIFACT_HALVES), set(ARTIFACT_HALVES) | {"best_test_audit.json"}]
    if names not in allowed or any(entry.is_symlink() for entry in path.iterdir()):
        return False
    try:
        from matching_fairness.artifacts import read_score_artifact

        for half in ARTIFACT_HALVES:
            read_score_artifact(path / half)
    except (OSError, ValueError):
        return False
    return True


def _brainrw_matrix_complete(path: Path) -> bool:
    try:
        from matching_fairness.formal_artifacts import validate_brainrw_export_tree

        validate_brainrw_export_tree(path, expected_image_count=200)
        return True
    except (OSError, TypeError, ValueError):
        return False


def _matrix_complete(path: Path) -> bool:
    """Backward-compatible structural dispatcher used by older unit fixtures."""
    return (
        _brainrw_matrix_complete(path)
        if path.name == "our_project"
        else _native_matrix_complete(path)
    )


def _brainrw_export_manifest_matches(layout: RuntimeLayout) -> bool:
    from matching_fairness.formal_artifacts import validate_brainrw_export_tree
    from matching_fairness.provenance import sha256_path
    expected_inputs = {
        "protocol_sha256": _hash_file(layout.protocol),
        "trial_manifest_sha256": _hash_file(layout.trial_manifest),
        "brain_test_sha256": _hash_file(layout.brainrw_test),
        "evaluator_sha256": _hash_file(
            layout.repository_root / "scripts/evaluate_retrieval.py"
        ),
        "test_image_tree_sha256": sha256_path(layout.official_test_images),
        "model_content_sha256": {
            "brain_model": sha256_path(layout.brainrw_model_root / "brain_model"),
            "vision_adapter": sha256_path(layout.brainrw_model_root / "vision_model"),
            "pretrained_vision_base": sha256_path(layout.clip_root),
        },
    }
    validate_brainrw_export_tree(
        layout.matrix_dir("our_project"),
        expected_image_count=200,
        expected_inputs=expected_inputs,
    )
    return True


def _matrix_matches_inputs(layout: RuntimeLayout, model: str) -> bool:
    """Require complete artifacts to bind the current fixed formal inputs."""
    path = layout.matrix_dir(model)
    complete = (
        _brainrw_matrix_complete(path)
        if model == "our_project"
        else _native_matrix_complete(path)
    )
    if not complete:
        return False
    try:
        from matching_fairness.artifacts import independent_ranks, read_score_artifact
        from matching_fairness.provenance import sha256_path

        artifacts = {
            half: read_score_artifact(path / half) for half in ARTIFACT_HALVES
        }
        expected_halves = {
            "standard": "standard",
            "eeg_a": "a",
            "eeg_b": "b",
        }
        trial_hash = _hash_file(layout.trial_manifest)
        for directory, artifact in artifacts.items():
            metadata = artifact.metadata
            if (
                metadata.get("model_slug") != model
                or metadata.get("trial_half") != expected_halves[directory]
                or metadata.get("subject") != SUBJECT
                or metadata.get("seed") != SEED
                or metadata.get("trial_manifest_sha256") != trial_hash
            ):
                return False

        if model in NATIVE_MODELS:
            source = _read_json(layout.source_lock)
            assets = _read_json(layout.asset_lock)
            checkpoint_manifest = layout.checkpoint_dir(model) / "checkpoint_manifest.json"
            checkpoint = _read_json(checkpoint_manifest)
            if not isinstance(checkpoint, Mapping):
                return False
            best = checkpoint.get("best_checkpoint")
            if not isinstance(best, Mapping):
                return False
            input_hashes = {
                "test_eeg": _hash_file(layout.test_eeg),
                "test_features": _hash_file(layout.test_features),
                "trial_manifest": trial_hash,
            }
            identity = {
                "source_lock": source,
                "asset_lock_manifest_sha256": _hash_file(layout.asset_lock),
                "asset_lock": assets,
                "checkpoint_manifest_sha256": _hash_file(checkpoint_manifest),
                "checkpoint_sha256": best.get("sha256"),
                "input_sha256": input_hashes,
                "checkpoint_role": "val_selected_formal",
                "logit_scale_type": "exp",
            }
            for artifact in artifacts.values():
                metadata = artifact.metadata
                if any(metadata.get(key) != value for key, value in identity.items()):
                    return False
            return True

        if model != "our_project":
            return False
        evaluator = layout.repository_root / "scripts/evaluate_retrieval.py"
        content = {
            "brain_model": sha256_path(layout.brainrw_model_root / "brain_model"),
            "vision_adapter": sha256_path(layout.brainrw_model_root / "vision_model"),
            "pretrained_vision_base": sha256_path(layout.clip_root),
        }
        expected_runtime = {
            "test_image_tree_sha256": sha256_path(layout.official_test_images),
            "selected_channel_indices": list(range(46, 63)),
            "time_slice": [0, 250],
            "dataset_name": "things",
            "expected_sample_count": 200,
        }
        identity = {
            "checkpoint": str(layout.brainrw_model_root / "brain_model"),
            "checkpoint_role": "fixed_formal",
            "checkpoint_content_sha256": content["brain_model"],
            "model_content_sha256": content,
            "protocol_sha256": _hash_file(layout.protocol),
            "brain_test_sha256": _hash_file(layout.brainrw_test),
            "evaluator_sha256": _hash_file(evaluator),
            "evaluator_version": "AIAA3800-BRAINRW-FORMAL-v1",
            "runtime_inputs": expected_runtime,
            "similarity": "cosine",
        }
        expected_fields = {
            "model_slug", "trial_half", "checkpoint_role", "checkpoint",
            "checkpoint_content_sha256", "similarity", "query_embeddings_sha256",
            "subject", "seed", "trial_manifest_sha256", "protocol_sha256",
            "brain_test_sha256", "model_content_sha256", "evaluator_version",
            "evaluator_sha256", "runtime_inputs", "native_metrics",
        }
        for directory, artifact in artifacts.items():
            metadata = artifact.metadata
            ranks = independent_ranks(artifact)
            expected_metrics = {
                "top1_count": int((ranks <= 1).sum()),
                "top5_count": int((ranks <= 5).sum()),
                "sample_count": len(ranks),
            }
            if (
                set(metadata) != expected_fields
                or artifact.similarity.shape != (200, 200)
                or metadata.get("trial_half") != expected_halves[directory]
                or metadata.get("native_metrics") != expected_metrics
                or any(metadata.get(key) != value for key, value in identity.items())
            ):
                return False
        return (
            artifacts["eeg_a"].metadata["query_embeddings_sha256"]
            != artifacts["eeg_b"].metadata["query_embeddings_sha256"]
            and _brainrw_export_manifest_matches(layout)
        )
    except (OSError, TypeError, ValueError):
        return False


def _score_artifact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for name in ("metadata.json", "similarity.npy"):
        digest.update(name.encode("ascii"))
        digest.update(bytes.fromhex(_hash_file(path / name)))
    return digest.hexdigest()


def _audit_matches_inputs(layout: RuntimeLayout, model: str) -> bool:
    if model not in NATIVE_MODELS:
        return False
    audit_path = layout.matrix_dir(model) / "best_test_audit.json"
    try:
        from matching_fairness.reporting import (
            _canonical_json,
            _read_regular_file,
            _validate_audit_manifest,
            _validate_checkpoint_manifest,
        )

        if not _checkpoint_matches_inputs(layout, model) or not _matrix_matches_inputs(
            layout, model
        ):
            return False
        checkpoint_path = layout.checkpoint_dir(model) / "checkpoint_manifest.json"
        checkpoint = _canonical_json(
            _read_regular_file(checkpoint_path, "checkpoint manifest"),
            "checkpoint manifest",
        )
        audit = _canonical_json(
            _read_regular_file(audit_path, "best-test audit"),
            "best-test audit",
        )
        _selection, checkpoint_identities = _validate_checkpoint_manifest(
            checkpoint, model
        )
        _best, inventory, audit_identities = _validate_audit_manifest(audit, model)
        if checkpoint_identities != audit_identities:
            return False
        expected_inventory: list[dict[str, object]] = []
        expected_digests: dict[tuple[str, str], str] = {}
        trial_half = {"standard": "standard", "eeg_a": "a", "eeg_b": "b"}
        for artifact_model in MODELS:
            for half in ARTIFACT_HALVES:
                artifact = layout.matrix_dir(artifact_model) / half
                digest = _score_artifact_sha256(artifact)
                expected_inventory.append({
                    "model_slug": artifact_model,
                    "trial_half": trial_half[half],
                    "path": str(artifact),
                    "sha256": digest,
                })
                expected_digests[(artifact_model, half)] = digest
        expected_inventory.sort(
            key=lambda entry: (entry["model_slug"], entry["trial_half"])
        )
        return (
            audit.get("formal_artifact_inventory") == expected_inventory
            and inventory == expected_digests
        )
    except (OSError, TypeError, ValueError):
        return False


def _derived_action(
    *, path: Path, root: Path, complete: bool, overwrite: bool, label: str
) -> str:
    if not os.path.lexists(path):
        return "run"
    if overwrite:
        _safe_remove_derived(path, root)
        return "run"
    if complete:
        return "skip"
    raise ValueError(f"{label} output is partial or mismatched")


def _require_all_nine_artifacts(layout: RuntimeLayout) -> None:
    for model in MODELS:
        if not _matrix_matches_inputs(layout, model):
            raise ValueError(
                "native audit requires all nine complete formal ScoreArtifacts"
            )


def _run(command: Sequence[str], runner: Callable[..., object] = subprocess.run) -> None:
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(EXPERIMENT_ROOT)
        if not existing
        else f"{EXPERIMENT_ROOT}{os.pathsep}{existing}"
    )
    runner(list(command), check=True, env=environment)


def _ensure_source_and_assets(layout: RuntimeLayout) -> None:
    commands = phase_commands(layout)
    source_pair = (layout.source_checkout.exists(), layout.source_lock.exists())
    if source_pair == (False, False):
        _run(commands["fetch_source"][0])
    elif source_pair != (True, True):
        raise ValueError("source checkout/lock is partial; refusing implicit replacement")
    else:
        from matching_fairness.provenance import inspect_checkout

        expected = _read_json(layout.source_lock)
        if not isinstance(expected, Mapping) or inspect_checkout(
            layout.source_checkout
        ).to_dict() != dict(expected):
            raise ValueError("source checkout does not match immutable source lock")

    asset_pair = (
        os.path.lexists(layout.asset_root),
        os.path.lexists(layout.asset_lock),
    )
    if asset_pair == (False, False):
        _run(commands["fetch_assets"][0])
    elif asset_pair != (True, True):
        raise ValueError("asset root/lock is partial; refusing implicit replacement")
    else:
        from scripts.fetch_assets import inventory_assets

        expected = _read_json(layout.asset_lock)
        if (
            not isinstance(expected, Mapping)
            or expected.get("asset_root") != str(layout.asset_root)
            or expected.get("files") != inventory_assets(layout.asset_root)
        ):
            raise ValueError("official assets do not match immutable asset lock")


def _verified_brainrw_snapshot(layout: RuntimeLayout) -> bytes:
    preflight = _read_json(layout.preflight_manifest)
    if not isinstance(preflight, Mapping):
        raise ValueError("preflight manifest must be a mapping")
    brainrw = preflight.get("brainrw")
    expected_keys = {
        "path", "sha256", "eeg_shape", "image_count", "session_counts"
    }
    if (
        not isinstance(brainrw, Mapping)
        or set(brainrw) != expected_keys
        or brainrw.get("path") != str(layout.brainrw_test)
        or _SHA256.fullmatch(str(brainrw.get("sha256", ""))) is None
        or brainrw.get("eeg_shape") != [200, 80, 63, 250]
        or brainrw.get("image_count") != 200
        or brainrw.get("session_counts")
        != {"0": 20, "1": 20, "2": 20, "3": 20}
    ):
        raise ValueError("preflight BrainRW identity/hash schema is invalid")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(layout.brainrw_test, flags)
    except OSError as error:
        raise ValueError("BrainRW test snapshot must be a non-symlink regular file") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("BrainRW test snapshot must be a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity:
        raise ValueError("BrainRW test snapshot changed while it was being read")
    encoded = b"".join(chunks)
    if hashlib.sha256(encoded).hexdigest() != brainrw["sha256"]:
        raise ValueError("BrainRW test snapshot SHA-256 does not match preflight")
    return encoded


def _trial_manifest_payload(layout: RuntimeLayout) -> dict[str, object]:
    import numpy as np
    import torch

    from matching_fairness.trial_splits import build_trial_manifest

    encoded = _verified_brainrw_snapshot(layout)
    payload = torch.load(io.BytesIO(encoded), map_location="cpu", weights_only=False)
    expected_keys = {"eeg", "label", "img", "text", "session", "ch_names", "times"}
    if not isinstance(payload, Mapping) or set(payload) != expected_keys:
        raise ValueError("BrainRW test.pt does not use the expected exact data schema")
    eeg_shape = tuple(int(value) for value in getattr(payload["eeg"], "shape", ()))
    if eeg_shape != (200, 80, 63, 250):
        raise ValueError("BrainRW EEG must have shape (200, 80, 63, 250)")
    images = np.asarray(payload["img"])
    sessions = np.asarray(payload["session"])
    labels = np.asarray(payload["label"])
    texts = np.asarray(payload["text"])
    if (
        images.shape != (200, 80)
        or sessions.shape != images.shape
        or labels.shape != images.shape
        or texts.shape != images.shape
    ):
        raise ValueError("BrainRW label/image/text/session arrays must have shape (200, 80)")
    channels = payload["ch_names"]
    times = np.asarray(payload["times"])
    if (
        not isinstance(channels, (list, tuple))
        or len(channels) != 63
        or any(not isinstance(value, str) or not value for value in channels)
        or len(set(channels)) != 63
        or times.shape != (250,)
    ):
        raise ValueError("BrainRW channel/time metadata schema is invalid")
    pairs: list[tuple[str, object]] = []
    for image_row, session_row in zip(images, sessions):
        ids = {Path(str(value)).stem for value in image_row.tolist()}
        if len(ids) != 1:
            raise ValueError("BrainRW image row contains mixed canonical identities")
        pairs.append((next(iter(ids)), session_row))
    pairs.sort(key=lambda item: item[0])
    image_ids = tuple(item[0] for item in pairs)
    if len(image_ids) != 200 or len(set(image_ids)) != 200:
        raise ValueError("formal trial manifest requires 200 unique image IDs")
    ordered_sessions = np.stack([np.asarray(item[1]) for item in pairs])
    return build_trial_manifest(image_ids, ordered_sessions, seed=SEED)


def _ensure_trial_manifest(layout: RuntimeLayout) -> None:
    from matching_fairness.trial_splits import validate_trial_manifest

    expected = _trial_manifest_payload(layout)
    if layout.trial_manifest.exists():
        actual = _read_json(layout.trial_manifest)
        if actual != expected:
            raise ValueError("existing trial manifest does not match real session identities")
        validate_trial_manifest(actual, tuple(expected["image_ids"]))
        return
    _atomic_write_json_noclobber(layout.trial_manifest, expected)
    validate_trial_manifest(expected, tuple(expected["image_ids"]))


def run_preflight(layout: RuntimeLayout, *, overwrite: bool) -> str:
    phase_manifest = phase_manifest_path(layout, "preflight")
    state = (
        os.path.lexists(layout.preflight_manifest),
        os.path.lexists(layout.trial_manifest),
        os.path.lexists(phase_manifest),
    )
    if any(state) and not all(state):
        raise ValueError("preflight output is partial or orphaned")
    prepare_runtime_directories(layout)
    _ensure_source_and_assets(layout)
    outputs_exist = all(state)
    if outputs_exist and not overwrite:
        if (
            layout.preflight_manifest.is_file()
            and layout.trial_manifest.is_file()
            and phase_manifest.is_file()
        ):
            expected = {
                "schema_version": 1,
                "phase": "preflight",
                "subject": SUBJECT,
                "seed": SEED,
                "input_sha256": _phase_inputs(layout, "preflight"),
                "output_sha256": {
                    "preflight": _hash_file(layout.preflight_manifest),
                    "trial_manifest": _hash_file(layout.trial_manifest),
                },
            }
            if _read_json(phase_manifest) == expected:
                return "skip"
        raise ValueError("preflight output is partial or its input hashes mismatch")
    if overwrite:
        for path in (layout.preflight_manifest, layout.trial_manifest, phase_manifest):
            _safe_remove_derived(path, layout.manifests_root)
    _run(phase_commands(layout)["preflight"][0])
    _ensure_trial_manifest(layout)
    payload = {
        "schema_version": 1,
        "phase": "preflight",
        "subject": SUBJECT,
        "seed": SEED,
        "input_sha256": _phase_inputs(layout, "preflight"),
        "output_sha256": {
            "preflight": _hash_file(layout.preflight_manifest),
            "trial_manifest": _hash_file(layout.trial_manifest),
        },
    }
    _atomic_write_json_noclobber(phase_manifest, payload)
    return "run"


def run_internal_cell(
    *,
    layout: RuntimeLayout,
    cell: str,
    array_id: int | None,
    export_mode: str,
    overwrite: bool,
) -> str:
    commands = phase_commands(layout)
    if cell == "train-native":
        if array_id is None:
            raise ValueError("train-native requires an array ID")
        action = phase_action(layout, "train", overwrite=overwrite, array_id=array_id)
        if action == "run":
            _run(commands["train"][array_id])
            if not _checkpoint_matches_inputs(
                layout, model_for_array_id(array_id)
            ):
                raise RuntimeError("native training did not publish a complete checkpoint")
        return action
    if cell == "export-native":
        if array_id is None:
            raise ValueError("export-native requires an array ID")
        model = model_for_array_id(array_id)
        matrix = layout.matrix_dir(model)
        if export_mode == "main":
            action = _derived_action(
                path=matrix,
                root=layout.matrices_root,
                complete=_matrix_matches_inputs(layout, model),
                overwrite=overwrite,
                label=f"{model} native export",
            )
            if action == "run":
                _run(commands["export_native"][array_id])
                if not _matrix_matches_inputs(layout, model):
                    raise RuntimeError(
                        "native export did not publish three complete artifacts "
                        "bound to the current input provenance"
                    )
            return action
        if export_mode != "audit":
            raise ValueError("native export mode must be main or audit")
        _require_all_nine_artifacts(layout)
        audit = matrix / "best_test_audit.json"
        action = _derived_action(
            path=audit,
            root=matrix,
            complete=_audit_matches_inputs(layout, model),
            overwrite=overwrite,
            label=f"{model} native audit",
        )
        if action == "run":
            _run(_native_audit_command(layout, model))
            if not _audit_matches_inputs(layout, model):
                raise RuntimeError(
                    "native audit did not publish a current inventory-bound manifest"
                )
        return action
    if cell == "export-brainrw":
        matrix = layout.matrix_dir("our_project")
        action = _derived_action(
            path=matrix,
            root=layout.matrices_root,
            complete=_matrix_matches_inputs(layout, "our_project"),
            overwrite=overwrite,
            label="BrainRW export",
        )
        if action == "run":
            _run(commands["export_brainrw"][0])
            if not _matrix_matches_inputs(layout, "our_project"):
                raise RuntimeError(
                    "BrainRW export did not publish three complete artifacts "
                    "bound to the current input provenance"
                )
        return action
    if cell == "match":
        action = phase_action(layout, "match", overwrite=overwrite)
        if action == "run":
            _require_all_nine_artifacts(layout)
            for model in NATIVE_MODELS:
                if not _audit_matches_inputs(layout, model):
                    raise ValueError("matching cannot start before both native audits")
            _run(commands["match"][0])
            write_phase_manifest(layout, "match")
        return action
    if cell == "aggregate":
        if phase_action(layout, "match", overwrite=False) != "skip":
            raise ValueError("aggregate requires a complete hash-bound matching phase")
        action = phase_action(layout, "aggregate", overwrite=overwrite)
        if action == "run":
            _run(commands["aggregate"][0])
            write_phase_manifest(layout, "aggregate")
        return action
    if cell == "final":
        match_action = run_internal_cell(
            layout=layout,
            cell="match",
            array_id=None,
            export_mode="main",
            overwrite=overwrite,
        )
        aggregate_action = run_internal_cell(
            layout=layout,
            cell="aggregate",
            array_id=None,
            export_mode="main",
            overwrite=overwrite,
        )
        return f"match={match_action},aggregate={aggregate_action}"
    if cell == "trial-manifest":
        _ensure_trial_manifest(layout)
        return "run"
    raise ValueError(f"unknown internal cell: {cell}")


def execute_pipeline(
    *,
    phase: str,
    submit: bool,
    dry_run: bool,
    overwrite: bool,
    layout: RuntimeLayout,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, object]:
    if submit and dry_run:
        raise ValueError("--submit and --dry-run are mutually exclusive")
    if dry_run:
        rendered = render_phase_plan(
            layout=layout, phase=phase, overwrite=overwrite
        )
        return {"mode": "dry-run", "phase": phase, "rendered": rendered}
    if not submit and phase in {"train", "export", "all"}:
        raise ValueError(
            f"heavy phase {phase!r} requires --submit or --dry-run"
        )
    if submit:
        if phase == "all":
            run_preflight(layout, overwrite=overwrite)
            submitted = submit_all(
                layout=layout, overwrite=overwrite, runner=runner
            )
        elif phase in {"train", "export"}:
            submitted = submit_phase(
                phase=phase,
                layout=layout,
                overwrite=overwrite,
                runner=runner,
            )
        else:
            raise ValueError(
                "--submit is supported only with --phase train, export, or all"
            )
        return {"mode": "submitted", **submitted}
    if phase == "preflight":
        return {"mode": "local", "preflight": run_preflight(layout, overwrite=overwrite)}
    if phase == "train":
        return {"mode": "local", "train": [
            run_internal_cell(
                layout=layout, cell="train-native", array_id=index,
                export_mode="main", overwrite=overwrite,
            )
            for index in (0, 1)
        ]}
    if phase == "export":
        native = [
            run_internal_cell(
                layout=layout, cell="export-native", array_id=index,
                export_mode="main", overwrite=overwrite,
            )
            for index in (0, 1)
        ]
        brainrw = run_internal_cell(
            layout=layout, cell="export-brainrw", array_id=None,
            export_mode="main", overwrite=overwrite,
        )
        audits = [
            run_internal_cell(
                layout=layout, cell="export-native", array_id=index,
                export_mode="audit", overwrite=overwrite,
            )
            for index in (0, 1)
        ]
        return {"mode": "local", "native": native, "brainrw": brainrw, "audit": audits}
    if phase == "match":
        return {"mode": "local", "match": run_internal_cell(
            layout=layout, cell="match", array_id=None,
            export_mode="main", overwrite=overwrite,
        )}
    if phase == "aggregate":
        return {"mode": "local", "aggregate": run_internal_cell(
            layout=layout, cell="aggregate", array_id=None,
            export_mode="main", overwrite=overwrite,
        )}
    if phase == "all":
        run_preflight(layout, overwrite=overwrite)
        execute_pipeline(
            phase="train", submit=False, dry_run=False,
            overwrite=overwrite, layout=layout, runner=runner,
        )
        execute_pipeline(
            phase="export", submit=False, dry_run=False,
            overwrite=overwrite, layout=layout, runner=runner,
        )
        return {"mode": "local", "final": run_internal_cell(
            layout=layout, cell="final", array_id=None,
            export_mode="main", overwrite=overwrite,
        )}
    raise ValueError(f"unknown phase: {phase}")


def _validate_model(model: str) -> None:
    if model not in MODELS:
        raise ValueError(f"model is outside the fixed formal scope: {model}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fixed sub-08 / seed-42 matching-fairness pipeline"
    )
    parser.add_argument(
        "--phase",
        choices=("preflight", "train", "export", "match", "aggregate", "all"),
        default="all",
    )
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--internal-cell",
        choices=("train-native", "export-native", "export-brainrw", "match", "aggregate", "final", "trial-manifest"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--array-id", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--export-mode", choices=("main", "audit"), default="main", help=argparse.SUPPRESS
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    if arguments.submit and arguments.dry_run:
        build_parser().error("--submit and --dry-run are mutually exclusive")
    layout = RuntimeLayout.fixed()
    if arguments.internal_cell is not None:
        result: object = run_internal_cell(
            layout=layout,
            cell=arguments.internal_cell,
            array_id=arguments.array_id,
            export_mode=arguments.export_mode,
            overwrite=arguments.overwrite,
        )
    else:
        result = execute_pipeline(
            phase=arguments.phase,
            submit=arguments.submit,
            dry_run=arguments.dry_run,
            overwrite=arguments.overwrite,
            layout=layout,
        )
        if arguments.dry_run:
            print(result["rendered"], end="")
            return
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
