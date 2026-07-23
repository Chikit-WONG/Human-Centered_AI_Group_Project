from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SUBMITTER_PATH = EXPERIMENT_ROOT / "scripts/submit_pipeline.py"
REQUEST_ID = "3ae8dc60c2df4166b7d4021f48146487"
LEDGER_SHA256 = "2125615c73c156bea4137c1c764aba6b7893e94cb64d819b6856b8a93b4042be"
REASON = "spool-entrypoint-bug"
APPROVED_LEDGER = Path(__file__).with_name("fixtures") / "approved_submission.json"
ORDER = ("train", "native_export", "brainrw_export", "native_audit", "final")
OLD_IDS = (10047830, 10047831, 10047832, 10047833, 10047834)
RAW_STATES = (
    "CANCELLED by 203817",
    "CANCELLED by 203817",
    "FAILED",
    "CANCELLED by 203817",
    "CANCELLED by 203817",
)


def _load_submitter():
    spec = importlib.util.spec_from_file_location(
        "matching_recovery_submitter", SUBMITTER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_original(module, layout, *, mutate=None) -> tuple[bytes, str]:
    del module
    encoded = APPROVED_LEDGER.read_bytes()
    assert hashlib.sha256(encoded).hexdigest() == LEDGER_SHA256
    if mutate is not None:
        ledger = json.loads(encoded)
        mutate(ledger)
        encoded = (
            json.dumps(ledger, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode()
    layout.submission_manifest.parent.mkdir(parents=True, exist_ok=True)
    layout.submission_manifest.write_bytes(encoded)
    return encoded, hashlib.sha256(encoded).hexdigest()


def _sacct_stdout(
    *,
    states: tuple[str, ...] = RAW_STATES,
    names: tuple[str, ...] | None = None,
    ids: tuple[int, ...] = OLD_IDS,
    exit_codes: tuple[str, ...] | None = None,
) -> str:
    if names is None:
        names = tuple(f"mf-{REQUEST_ID}-{name.replace('_', '-')}" for name in ORDER)
    if exit_codes is None:
        exit_codes = ("0:15", "0:15", "2:0", "0:15", "0:15")
    return "".join(
        f"{job_id}|{name}|{state}|{exit_code}|2026-07-22T10:00:00|"
        f"2026-07-22T10:01:00|2026-07-22T10:02:00|\n"
        for job_id, name, state, exit_code in zip(ids, names, states, exit_codes)
    )


class _RecoveryRunner:
    def __init__(
        self,
        layout,
        *,
        sacct_stdout: str | None = None,
        new_ids: tuple[int, ...] = (2001, 2002, 2003, 2004, 2005),
        fail_sacct: BaseException | None = None,
        fail_sbatch_at: int | None = None,
        uncertain_sbatch_at: int | None = None,
        before_sbatch=None,
    ) -> None:
        self.layout = layout
        stdout = _sacct_stdout() if sacct_stdout is None else sacct_stdout
        self.sacct_stdout = stdout.encode("utf-8") if isinstance(stdout, str) else stdout
        self.new_ids = new_ids
        self.fail_sacct = fail_sacct
        self.fail_sbatch_at = fail_sbatch_at
        self.uncertain_sbatch_at = uncertain_sbatch_at
        self.before_sbatch = before_sbatch
        self.calls: list[list[str]] = []
        self.call_kwargs: list[dict[str, object]] = []
        self.sbatch_calls = 0

    def __call__(self, argv, **kwargs):
        call = list(argv)
        self.calls.append(call)
        self.call_kwargs.append(dict(kwargs))
        if call[0] == "sacct":
            if self.fail_sacct is not None:
                raise self.fail_sacct
            return subprocess.CompletedProcess(
                call, 0, stdout=self.sacct_stdout, stderr=b""
            )
        assert call[0] == "sbatch"
        index = self.sbatch_calls
        self.sbatch_calls += 1
        if self.before_sbatch is not None:
            self.before_sbatch(index, call)
        if index == self.fail_sbatch_at:
            raise subprocess.CalledProcessError(1, call, stderr="scheduler rejected")
        if index == self.uncertain_sbatch_at:
            raise OSError("scheduler response lost")
        return subprocess.CompletedProcess(
            call, 0, stdout=f"{self.new_ids[index]}\n", stderr=""
        )


def _recover(module, layout, sha256: str, runner):
    return module.recover_failed_all(
        layout=layout,
        original_request_id=REQUEST_ID,
        original_ledger_sha256=sha256,
        recovery_reason=REASON,
        overwrite=False,
        runner=runner,
    )


def test_successful_recovery_is_audited_noclobber_and_uses_only_new_dag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    original, sha256 = _write_original(module, layout)
    fsyncs: list[Path] = []
    real_fsync = module._fsync_directory

    def fsync(path: Path) -> None:
        fsyncs.append(Path(path))
        real_fsync(path)

    def before_sbatch(index: int, argv: list[str]) -> None:
        assert layout.submission_recovery_manifest.is_file()
        assert layout.submission_manifest.read_bytes() == original
        assert layout.manifests_root in fsyncs
        ledger = json.loads(layout.submission_recovery_manifest.read_text())
        name = ORDER[index]
        assert ledger["request"]["jobs"][name]["state"] == "submitting"
        assert ledger["request"]["jobs"][name]["argv"] == argv

    monkeypatch.setattr(module, "_fsync_directory", fsync)
    runner = _RecoveryRunner(layout, before_sbatch=before_sbatch)
    result = _recover(module, layout, sha256, runner)

    assert result == {
        f"{name}_job_id": 2001 + index for index, name in enumerate(ORDER)
    }
    assert layout.submission_manifest.read_bytes() == original
    assert runner.calls[0] == [
        "sacct",
        "-X",
        "--duplicates",
        "--noheader",
        "--parsable2",
        "--jobs",
        ",".join(map(str, OLD_IDS)),
        "--format=JobIDRaw,JobName%128,State%64,ExitCode,Submit,Start,End",
    ]
    assert len(runner.calls) == 6
    assert runner.call_kwargs[0]["text"] is False
    assert all(kwargs["text"] is True for kwargs in runner.call_kwargs[1:])
    ledger = json.loads(layout.submission_recovery_manifest.read_text())
    assert ledger["schema_version"] == 1
    assert ledger["kind"] == "matching_fairness_submission_recovery"
    assert ledger["subject"] == "sub-08" and ledger["seed"] == 42
    assert ledger["models"] == ["nice", "atm_s", "our_project"]
    assert ledger["original_ledger"]["sha256"] == sha256
    assert ledger["original_ledger"]["request_id"] == REQUEST_ID
    assert ledger["original_ledger"]["request_state"] == "completed"
    assert ledger["original_ledger"]["overwrite"] is False
    assert ledger["original_ledger"]["job_order"] == list(ORDER)
    assert ledger["original_ledger"]["jobs"] == json.loads(original)["requests"]["all"][
        "jobs"
    ]
    evidence = ledger["scheduler_verification"]
    assert evidence["argv"] == runner.calls[0]
    assert (
        evidence["stdout_sha256"]
        == hashlib.sha256(_sacct_stdout().encode()).hexdigest()
    )
    assert evidence["checked_at_utc"].endswith("Z")
    assert [evidence["jobs"][name]["raw_state"] for name in ORDER] == list(RAW_STATES)
    assert [evidence["jobs"][name]["normalized_state"] for name in ORDER] == [
        "CANCELLED",
        "CANCELLED",
        "FAILED",
        "CANCELLED",
        "CANCELLED",
    ]
    request = ledger["request"]
    assert request["state"] == "completed" and request["overwrite"] is False
    assert request["request_id"] != REQUEST_ID
    assert all(
        row["token"].startswith(f"mf-{request['request_id']}-")
        for row in request["jobs"].values()
    )
    assert not set(OLD_IDS) & {row["job_id"] for row in request["jobs"].values()}
    sbatches = runner.calls[1:]
    assert all("MATCHING_FAIRNESS_OVERWRITE=0" in " ".join(call) for call in sbatches)
    expected_root = f"MATCHING_FAIRNESS_EXPERIMENT_ROOT={layout.experiment_root}"
    assert all(expected_root in " ".join(call) for call in sbatches)
    assert "--dependency=afterok:2001" in sbatches[1]
    assert not any(value.startswith("--dependency=") for value in sbatches[2])
    assert "--dependency=afterok:2002:2003" in sbatches[3]
    assert "--dependency=afterok:2004" in sbatches[4]


def test_recovery_is_hard_bound_to_the_approved_incident_constants() -> None:
    module = _load_submitter()
    assert module._APPROVED_RECOVERY_REQUEST_ID == REQUEST_ID
    assert module._APPROVED_RECOVERY_LEDGER_SHA256 == LEDGER_SHA256


def test_self_consistent_substituted_completed_ledger_is_rejected_before_scheduler(
    tmp_path: Path,
) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    substituted_id = "1" * 32
    ledger = module._new_submission_ledger(mode="all", phase="all", overwrite=False)
    request = ledger["requests"]["all"]
    request["request_id"] = substituted_id
    request["state"] = "completed"
    for name, job_id in zip(ORDER, OLD_IDS):
        request["jobs"][name] = {
            "state": "submitted",
            "token": f"mf-{substituted_id}-{name.replace('_', '-')}",
            "argv": ["sbatch", f"substituted-{name}.slurm"],
            "job_id": job_id,
            "error": None,
        }
    encoded = (
        json.dumps(ledger, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    substituted_sha = hashlib.sha256(encoded).hexdigest()
    layout.submission_manifest.parent.mkdir(parents=True, exist_ok=True)
    layout.submission_manifest.write_bytes(encoded)
    names = tuple(
        f"mf-{substituted_id}-{name.replace('_', '-')}" for name in ORDER
    )
    runner = _RecoveryRunner(layout, sacct_stdout=_sacct_stdout(names=names))

    with pytest.raises(ValueError, match="approved|incident|request|SHA"):
        module.recover_failed_all(
            layout=layout,
            original_request_id=substituted_id,
            original_ledger_sha256=substituted_sha,
            recovery_reason=REASON,
            overwrite=False,
            runner=runner,
        )

    assert runner.calls == []
    assert not os.path.lexists(layout.submission_recovery_manifest)


def test_scheduler_evidence_hashes_exact_raw_crlf_bytes(tmp_path: Path) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    _original, sha256 = _write_original(module, layout)
    raw_stdout = _sacct_stdout().replace("\n", "\r\n").encode("utf-8")
    runner = _RecoveryRunner(layout, sacct_stdout=raw_stdout)

    _recover(module, layout, sha256, runner)

    evidence = json.loads(layout.submission_recovery_manifest.read_text())[
        "scheduler_verification"
    ]
    assert runner.call_kwargs[0]["text"] is False
    assert evidence["stdout_sha256"] == hashlib.sha256(raw_stdout).hexdigest()


def test_invalid_utf8_sacct_bytes_fail_before_recovery_or_submission(
    tmp_path: Path,
) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    _original, sha256 = _write_original(module, layout)
    runner = _RecoveryRunner(
        layout,
        sacct_stdout=_sacct_stdout().encode("utf-8") + b"\xff",
    )

    with pytest.raises(ValueError, match="UTF-8|utf-8"):
        _recover(module, layout, sha256, runner)

    assert len(runner.calls) == 1 and runner.calls[0][0] == "sacct"
    assert not os.path.lexists(layout.submission_recovery_manifest)


@pytest.mark.parametrize("state", ("CANCELLED by 203817", "FAILED"))
def test_actual_incident_state_spellings_are_accepted(
    tmp_path: Path, state: str
) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    _original, sha256 = _write_original(module, layout)
    states = tuple(state for _ in ORDER)
    runner = _RecoveryRunner(layout, sacct_stdout=_sacct_stdout(states=states))
    _recover(module, layout, sha256, runner)
    evidence = json.loads(layout.submission_recovery_manifest.read_text())[
        "scheduler_verification"
    ]
    expected = "CANCELLED" if state.startswith("CANCELLED") else "FAILED"
    assert {row["normalized_state"] for row in evidence["jobs"].values()} == {expected}


@pytest.mark.parametrize("content", (b"", b"{}\n", b"not-json\n"))
def test_any_existing_recovery_file_blocks_before_scheduler(
    tmp_path: Path, content: bytes
) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    _original, sha256 = _write_original(module, layout)
    layout.submission_recovery_manifest.write_bytes(content)
    runner = _RecoveryRunner(layout)
    with pytest.raises(FileExistsError, match="recovery"):
        _recover(module, layout, sha256, runner)
    assert runner.calls == []


def test_existing_recovery_symlink_blocks_before_scheduler(tmp_path: Path) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    _original, sha256 = _write_original(module, layout)
    target = tmp_path / "target"
    target.write_text("target")
    layout.submission_recovery_manifest.symlink_to(target)
    runner = _RecoveryRunner(layout)
    with pytest.raises(FileExistsError, match="recovery"):
        _recover(module, layout, sha256, runner)
    assert runner.calls == []


@pytest.mark.parametrize(
    ("mutation", "sha_override"),
    (
        (lambda ledger: ledger.update(schema_version=2), None),
        (lambda ledger: ledger.update(mode="phased"), None),
        (lambda ledger: ledger["requests"]["all"].update(state="active"), None),
        (lambda ledger: ledger["requests"]["all"].update(request_id="0" * 32), None),
        (
            lambda ledger: ledger["requests"]["all"]["jobs"]["final"].update(
                job_id=OLD_IDS[0]
            ),
            None,
        ),
        (None, "0" * 64),
    ),
)
def test_wrong_original_identity_blocks_without_recovery_or_scheduler(
    tmp_path: Path, mutation, sha_override: str | None
) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    _original, sha256 = _write_original(module, layout, mutate=mutation)
    runner = _RecoveryRunner(layout)
    with pytest.raises(
        (ValueError, RuntimeError),
        match="original|ledger|SHA|request|mode|state|unique",
    ):
        _recover(module, layout, sha_override or sha256, runner)
    assert runner.calls == []
    assert not os.path.lexists(layout.submission_recovery_manifest)


@pytest.mark.parametrize(
    "error",
    (
        subprocess.CalledProcessError(1, ["sacct"], stderr="failed"),
        OSError("sacct unavailable"),
    ),
)
def test_scheduler_query_failure_creates_no_recovery_ledger(
    tmp_path: Path, error: BaseException
) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    _original, sha256 = _write_original(module, layout)
    runner = _RecoveryRunner(layout, fail_sacct=error)
    with pytest.raises(type(error)):
        _recover(module, layout, sha256, runner)
    assert len(runner.calls) == 1 and runner.calls[0][0] == "sacct"
    assert not os.path.lexists(layout.submission_recovery_manifest)


@pytest.mark.parametrize(
    "stdout",
    (
        "",
        _sacct_stdout().replace(_sacct_stdout().splitlines(True)[0], ""),
        _sacct_stdout() + _sacct_stdout().splitlines(True)[0],
        _sacct_stdout(
            names=("wrong",)
            + tuple(f"mf-{REQUEST_ID}-{name.replace('_', '-')}" for name in ORDER[1:])
        ),
        _sacct_stdout(exit_codes=("bad", "0:15", "2:0", "0:15", "0:15")),
        _sacct_stdout() + "99999999|extra-root|FAILED|1:0||||\n",
    ),
)
def test_malformed_or_incomplete_sacct_output_fails_closed(
    tmp_path: Path, stdout: str
) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    _original, sha256 = _write_original(module, layout)
    runner = _RecoveryRunner(layout, sacct_stdout=stdout)
    with pytest.raises((ValueError, RuntimeError), match="sacct|record|job|exit|name"):
        _recover(module, layout, sha256, runner)
    assert len(runner.calls) == 1
    assert not os.path.lexists(layout.submission_recovery_manifest)


@pytest.mark.parametrize("state", ("COMPLETED", "RUNNING", "PENDING", "MYSTERY", ""))
def test_non_unsuccessful_scheduler_states_fail_closed(
    tmp_path: Path, state: str
) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    _original, sha256 = _write_original(module, layout)
    states = (state,) + RAW_STATES[1:]
    runner = _RecoveryRunner(layout, sacct_stdout=_sacct_stdout(states=states))
    with pytest.raises((ValueError, RuntimeError), match="state|terminal"):
        _recover(module, layout, sha256, runner)
    assert len(runner.calls) == 1
    assert not os.path.lexists(layout.submission_recovery_manifest)


@pytest.mark.parametrize("root_name", ("checkpoints", "matrices", "runs", "aggregate"))
@pytest.mark.parametrize("kind", ("nonempty", "file", "symlink"))
def test_unsafe_derived_output_roots_block_before_scheduler(
    tmp_path: Path, root_name: str, kind: str
) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    _original, sha256 = _write_original(module, layout)
    root = layout.results_root / root_name
    if kind == "nonempty":
        root.mkdir(parents=True)
        (root / "artifact").write_text("unexpected")
    elif kind == "file":
        root.parent.mkdir(parents=True, exist_ok=True)
        root.write_text("unexpected")
    else:
        target = tmp_path / f"{root_name}-target"
        target.mkdir()
        root.symlink_to(target, target_is_directory=True)
    runner = _RecoveryRunner(layout)
    with pytest.raises(
        (ValueError, RuntimeError), match="output|empty|directory|symlink"
    ):
        _recover(module, layout, sha256, runner)
    assert runner.calls == []
    assert not os.path.lexists(layout.submission_recovery_manifest)


@pytest.mark.parametrize("fail_index", (0, 2, 4))
def test_recovery_sbatch_failure_is_durable_and_stops_downstream(
    tmp_path: Path, fail_index: int
) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    _original, sha256 = _write_original(module, layout)
    runner = _RecoveryRunner(layout, fail_sbatch_at=fail_index)
    with pytest.raises(subprocess.CalledProcessError):
        _recover(module, layout, sha256, runner)
    request = json.loads(layout.submission_recovery_manifest.read_text())["request"]
    assert request["state"] == "failed"
    assert request["failure"]["stage"] == ORDER[fail_index]
    assert len(runner.calls) == fail_index + 2
    assert [request["jobs"][name]["state"] for name in ORDER] == (
        ["submitted"] * fail_index + ["failed"] + ["planned"] * (4 - fail_index)
    )


def test_recovery_sbatch_uncertainty_is_durable_and_stops_downstream(
    tmp_path: Path,
) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    _original, sha256 = _write_original(module, layout)
    runner = _RecoveryRunner(layout, uncertain_sbatch_at=1)
    with pytest.raises(OSError, match="response lost"):
        _recover(module, layout, sha256, runner)
    request = json.loads(layout.submission_recovery_manifest.read_text())["request"]
    assert request["state"] == "unknown"
    assert [request["jobs"][name]["state"] for name in ORDER] == [
        "submitted",
        "unknown",
        "planned",
        "planned",
        "planned",
    ]


def test_accepted_job_ledger_write_failure_leaves_durable_submitting_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    _original, sha256 = _write_original(module, layout)
    real_replace = module._replace_submission_ledger
    writes = 0

    def fail_job_id_write(path: Path, payload) -> None:
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("injected recovery job-id write failure")
        real_replace(path, payload)

    monkeypatch.setattr(module, "_replace_submission_ledger", fail_job_id_write)
    runner = _RecoveryRunner(layout)
    with pytest.raises(OSError, match="job-id write"):
        _recover(module, layout, sha256, runner)
    request = json.loads(layout.submission_recovery_manifest.read_text())["request"]
    assert len(runner.calls) == 2
    assert request["jobs"]["train"]["state"] == "submitting"
    assert request["jobs"]["train"]["job_id"] is None


def test_original_mutation_between_jobs_is_recorded_before_next_sbatch(
    tmp_path: Path,
) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    original, sha256 = _write_original(module, layout)

    def mutate_after_first(index: int, _argv: list[str]) -> None:
        if index == 0:
            layout.submission_manifest.write_bytes(original + b" ")

    runner = _RecoveryRunner(layout, before_sbatch=mutate_after_first)
    with pytest.raises((ValueError, RuntimeError), match="original|changed|SHA"):
        _recover(module, layout, sha256, runner)
    request = json.loads(layout.submission_recovery_manifest.read_text())["request"]
    assert len(runner.calls) == 2
    assert request["state"] == "failed"
    assert request["failure"]["stage"] == "native_export"
    assert request["jobs"]["native_export"]["state"] == "failed"


def test_normal_all_submit_remains_blocked_by_original_ledger(tmp_path: Path) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    original, _sha256 = _write_original(module, layout)
    runner = _RecoveryRunner(layout)
    with pytest.raises((FileExistsError, RuntimeError), match="ledger|submission"):
        module.submit_all(layout=layout, overwrite=False, runner=runner)
    assert runner.calls == []
    assert layout.submission_manifest.read_bytes() == original


@pytest.mark.parametrize(
    "overrides",
    (
        {"phase": "train"},
        {"submit": False},
        {"dry_run": True},
        {"overwrite": True},
        {"internal_cell": "train-native"},
        {"array_id": 0},
        {"export_mode": "audit"},
        {"original_request_id": "bad"},
        {"original_request_id": "0" * 32},
        {"original_ledger_sha256": "bad"},
        {"original_ledger_sha256": "a" * 64},
        {"recovery_reason": "generic-retry"},
    ),
)
def test_invalid_recovery_cli_combinations_fail_before_scheduler(
    tmp_path: Path, overrides: dict[str, object]
) -> None:
    module = _load_submitter()
    values = {
        "phase": "all",
        "submit": True,
        "dry_run": False,
        "overwrite": False,
        "internal_cell": None,
        "array_id": None,
        "export_mode": "main",
        "recover_failed_all": True,
        "original_request_id": REQUEST_ID,
        "original_ledger_sha256": LEDGER_SHA256,
        "recovery_reason": REASON,
    }
    values.update(overrides)
    with pytest.raises(ValueError, match="recover|recovery|request|SHA|reason"):
        module._validate_recovery_cli(**values)


def test_recovery_metadata_without_flag_is_rejected() -> None:
    module = _load_submitter()
    with pytest.raises(ValueError, match="recovery"):
        module._validate_recovery_cli(
            phase="all",
            submit=True,
            dry_run=False,
            overwrite=False,
            internal_cell=None,
            array_id=None,
            export_mode="main",
            recover_failed_all=False,
            original_request_id=REQUEST_ID,
            original_ledger_sha256=None,
            recovery_reason=None,
        )


@pytest.mark.parametrize("state", ("active", "completed", "failed", "unknown"))
def test_second_recovery_attempt_is_impossible_for_every_ledger_state(
    tmp_path: Path, state: str
) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    _original, sha256 = _write_original(module, layout)
    layout.submission_recovery_manifest.write_text(
        json.dumps({"schema_version": 1, "request": {"state": state}}) + "\n"
    )
    runner = _RecoveryRunner(layout)
    with pytest.raises(FileExistsError, match="recovery"):
        _recover(module, layout, sha256, runner)
    assert runner.calls == []


def test_layout_exposes_fixed_recovery_manifest(tmp_path: Path) -> None:
    module = _load_submitter()
    layout = module.RuntimeLayout.for_test(tmp_path)
    assert layout.submission_recovery_manifest == (
        layout.manifests_root / "submission_recovery.json"
    )
