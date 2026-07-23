# Second GPFS Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and execute one audited, no-clobber second recovery of the fixed sub-08 / seed-42 matching-fairness DAG after the first recovery failed only because GPFS rejected `renameat2(RENAME_NOREPLACE)`.

**Architecture:** Preserve `submission.json` and `submission_recovery.json` byte-for-byte. A dedicated recovery script validates both fixed ledger identities, validates the first recovery's exact completed request and five submitted jobs, queries `sacct` for terminal-unsuccessful evidence, requires all derived output roots to be absent or empty, then reserves `submission_recovery_2.json` before the first `sbatch`. The existing five-stage DAG and dependency construction are reused without overwrite; every scheduler call is preceded by byte-level revalidation of both predecessor ledgers.

**Tech Stack:** Python 3.11, pytest, JSON/SHA-256, POSIX file locking and fsync helpers already in `submit_pipeline.py`, SLURM `sacct`/`sbatch`.

## Global Constraints

- Work only on the existing `ckw` branch. Do not create or switch branches or worktrees.
- The implementation baseline is `33fa0aee28f7d9f846d9bdab87ad1d70495148a5`.
- Do not push until the complete experiment and documentation are reviewed.
- Use strict RED/GREEN TDD. The implementer must not invoke `sbatch`, `scancel`, or mutate formal runtime/log files.
- The original ledger path is `/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/Class/AIAA3800_L01-Human-Centered_Artificial_Intelligence/Final_Project/test/brain-rw/results/matching_fairness_v3/manifests/submission.json`; its exact SHA-256 is `2125615c73c156bea4137c1c764aba6b7893e94cb64d819b6856b8a93b4042be` and request ID is `3ae8dc60c2df4166b7d4021f48146487`.
- The first recovery ledger path is `/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/Class/AIAA3800_L01-Human-Centered_Artificial_Intelligence/Final_Project/test/brain-rw/results/matching_fairness_v3/manifests/submission_recovery.json`; its exact SHA-256 is `2b665cc2d2d69338328c716332d81a412a3829d382a5ec53331cb86c71e05022` and request ID is `82bbdd064b904c1fb2579b79a7b46353`.
- The first recovery job IDs are exactly `10048700`, `10048701`, `10048702`, `10048703`, `10048704` for `train`, `native_export`, `brainrw_export`, `native_audit`, `final` respectively.
- The second recovery reason is exactly `gpfs-renameat2-unsupported`.
- The second ledger is exactly `manifests/submission_recovery_2.json`. Any existing file, directory, symlink, or broken symlink at that path permanently blocks submission.
- `submission.json` and `submission_recovery.json` must remain byte-identical before, during, and after implementation and submission.
- The derived roots `checkpoints`, `matrices`, `runs`, and `aggregate` must be absent or empty non-symlink directories before `sacct`; partial or nonempty output is fail-closed.
- The new DAG must use `overwrite=False`, new request/job IDs, the existing five-job dependency graph, and no job ID from either predecessor DAG.
- The new script must use raw `sacct --parsable2` bytes and preserve their SHA-256 in the ledger.
- Scheduler uncertainty after an `sbatch` call must be durable as `unknown`; an explicit scheduler rejection must be durable as `failed`. Never retry automatically.
- GPFS fallback safety is already provided by commit `33fa0ae`; this task must not alter score publication logic.

---

### Task 1: Add a dedicated, incident-bound second recovery entry point

**Files:**
- Create: `experiments/matching_fairness/scripts/recover_failed_publication.py`
- Create: `experiments/matching_fairness/tests/test_recovery_publication.py`
- Create: `experiments/matching_fairness/tests/fixtures/approved_submission_recovery.json`
- Modify: `experiments/matching_fairness/README.md`
- Modify: `experiments/matching_fairness/README_ZH.md`

**Interfaces:**
- Consumes: `submit_pipeline.RuntimeLayout`, `_WORKFLOW_ORDER`, `_read_original_submission_snapshot`, `_require_recovery_outputs_absent_or_empty`, `_verify_terminal_scheduler_state`, `_new_submission_request`, `_validate_submission_request`, `_workflow_command`, `_submit`, `_submission_lock`, `_atomic_write_json_noclobber`, `_replace_submission_ledger`, and `_assert_original_ledger_unchanged`.
- Produces: `recover_failed_publication(*, layout: RuntimeLayout, original_request_id: str, original_ledger_sha256: str, prior_recovery_request_id: str, prior_recovery_ledger_sha256: str, recovery_reason: str, runner=subprocess.run) -> dict[str, int]` and a CLI with `--submit`, `--original-request-id`, `--original-ledger-sha256`, `--prior-recovery-request-id`, `--prior-recovery-ledger-sha256`, and `--recovery-reason`.

- [ ] **Step 1: Freeze the exact prior recovery fixture**

Copy the already immutable formal `submission_recovery.json` bytes into `tests/fixtures/approved_submission_recovery.json` with `apply_patch`, then assert in the new test module:

```python
ORIGINAL_REQUEST_ID = "3ae8dc60c2df4166b7d4021f48146487"
ORIGINAL_LEDGER_SHA256 = "2125615c73c156bea4137c1c764aba6b7893e94cb64d819b6856b8a93b4042be"
PRIOR_REQUEST_ID = "82bbdd064b904c1fb2579b79a7b46353"
PRIOR_LEDGER_SHA256 = "2b665cc2d2d69338328c716332d81a412a3829d382a5ec53331cb86c71e05022"
PRIOR_JOB_IDS = (10048700, 10048701, 10048702, 10048703, 10048704)
REASON = "gpfs-renameat2-unsupported"

def test_fixed_incident_constants_match_approved_bytes():
    assert hashlib.sha256(APPROVED_ORIGINAL.read_bytes()).hexdigest() == ORIGINAL_LEDGER_SHA256
    assert hashlib.sha256(APPROVED_PRIOR.read_bytes()).hexdigest() == PRIOR_LEDGER_SHA256
```

- [ ] **Step 2: Write failing validation and success-path tests**

Write tests that load the dedicated script by `importlib`, create `RuntimeLayout.for_test(tmp_path)`, install exact predecessor ledger bytes, and use a fake runner with one raw-byte `sacct` response plus five numeric `sbatch` responses. The success test must assert:

```python
assert result == {
    "train_job_id": 3001,
    "native_export_job_id": 3002,
    "brainrw_export_job_id": 3003,
    "native_audit_job_id": 3004,
    "final_job_id": 3005,
}
assert original_path.read_bytes() == original_bytes
assert prior_path.read_bytes() == prior_bytes
assert second["kind"] == "matching_fairness_submission_recovery_2"
assert second["recovery_reason"] == REASON
assert second["predecessors"]["original_submission"]["sha256"] == ORIGINAL_LEDGER_SHA256
assert second["predecessors"]["first_recovery"]["sha256"] == PRIOR_LEDGER_SHA256
assert second["scheduler_verification"]["stdout_sha256"] == hashlib.sha256(raw_sacct).hexdigest()
assert second["request"]["state"] == "completed"
```

Also assert exact dependencies `afterok:3001`, none for BrainRW, `afterok:3002:3003`, and `afterok:3004`; every `sbatch` contains `MATCHING_FAIRNESS_OVERWRITE=0` and the absolute `MATCHING_FAIRNESS_EXPERIMENT_ROOT`.

- [ ] **Step 3: Write failing fail-closed tests**

Parameterize tests for all of these pre-scheduler failures:

```python
INVALID_METADATA = (
    {"original_request_id": "0" * 32},
    {"original_ledger_sha256": "0" * 64},
    {"prior_recovery_request_id": "0" * 32},
    {"prior_recovery_ledger_sha256": "0" * 64},
    {"recovery_reason": "generic-retry"},
)
SECOND_PATH_CONTENTS = (b"", b"{}\n", b"not-json\n")
UNSAFE_ROOT_KINDS = ("nonempty", "file", "symlink", "broken-symlink")
```

Add separate tests for a second-ledger directory, a second-ledger symlink, malformed or self-consistent substituted predecessor ledgers, a prior ledger whose original cross-reference is wrong, a prior request not `completed`, a prior job missing/duplicated/not `submitted`, scheduler query failure, malformed/missing/duplicate/unexpected `sacct` records, invalid UTF-8, and any `COMPLETED`, `RUNNING`, or `PENDING` scheduler state. All must assert zero `sbatch` calls and no second ledger.

- [ ] **Step 4: Write failing durability and race tests**

Test that the second ledger is durably reserved before the first `sbatch`, that both predecessor byte strings are checked before every `sbatch`, and that mutating either predecessor after the first accepted job records the next stage as `failed` without another scheduler call. Add explicit tests for scheduler rejection, scheduler-response uncertainty, accepted-job ledger-write failure, duplicate IDs against either predecessor DAG, duplicate IDs within the new DAG, and two concurrent recovery callers where exactly one reaches `sacct`/`sbatch` and the other receives `FileExistsError`.

- [ ] **Step 5: Run RED**

Run:

```bash
PYTHONPATH=experiments/matching_fairness \
  /hpc2hdd/home/ckwong627/miniconda3/envs/eeg_recon/bin/python -m pytest \
  experiments/matching_fairness/tests/test_recovery_publication.py -q
```

Expected: failures because `recover_failed_publication.py` and its interfaces do not yet exist. Preserve the exact failed/passed counts in the SDD report.

- [ ] **Step 6: Implement the fixed constants and strict prior-ledger reader**

The new script must define these exact constants and path helper:

```python
_REASON = "gpfs-renameat2-unsupported"
_ORIGINAL_REQUEST_ID = "3ae8dc60c2df4166b7d4021f48146487"
_ORIGINAL_LEDGER_SHA256 = "2125615c73c156bea4137c1c764aba6b7893e94cb64d819b6856b8a93b4042be"
_PRIOR_REQUEST_ID = "82bbdd064b904c1fb2579b79a7b46353"
_PRIOR_LEDGER_SHA256 = "2b665cc2d2d69338328c716332d81a412a3829d382a5ec53331cb86c71e05022"
_PRIOR_JOB_IDS = (10048700, 10048701, 10048702, 10048703, 10048704)

def second_recovery_manifest(layout: RuntimeLayout) -> Path:
    return layout.manifests_root / "submission_recovery_2.json"
```

Read each predecessor with the existing regular-file and canonical-JSON helpers. Require exact hashes, request IDs, subject `sub-08`, seed `42`, model list `nice/atm_s/our_project`, exact five-job order, prior request `completed`, `overwrite is False`, exact job IDs/tokens, and exact prior-to-original cross-reference. Return immutable byte snapshots and prior job rows.

- [ ] **Step 7: Implement no-clobber recovery and durable failure semantics**

Inside `_submission_lock(layout.submission_manifest)`, perform this exact order:

```python
if os.path.lexists(second_path):
    raise FileExistsError("second recovery ledger already exists")
original_bytes, _, original_jobs = _read_original_submission_snapshot(
    path=layout.submission_manifest,
    expected_sha256=original_ledger_sha256,
    expected_request_id=original_request_id,
)
prior_bytes, prior_payload, prior_jobs = _read_prior_recovery_snapshot(
    path=layout.submission_recovery_manifest,
    expected_sha256=prior_recovery_ledger_sha256,
    expected_request_id=prior_recovery_request_id,
    original_jobs=original_jobs,
)
_require_recovery_outputs_absent_or_empty(layout)
verification = _verify_terminal_scheduler_state(original_jobs=prior_jobs, runner=runner)
_assert_snapshot_unchanged(original_path, original_bytes, ORIGINAL_LEDGER_SHA256)
_assert_snapshot_unchanged(prior_path, prior_bytes, PRIOR_LEDGER_SHA256)
_atomic_write_json_noclobber(second_path, recovery)
```

For each name in `_WORKFLOW_ORDER["all"]`, persist `submitting` plus exact argv, revalidate both snapshots, call `_submit`, reject any ID from the union of both predecessor job sets and new IDs, then persist `submitted`. Preserve existing `failed` versus `unknown` semantics. Finally persist request state `completed` and return the five IDs.

- [ ] **Step 8: Add the dedicated CLI and bilingual operator documentation**

The parser must require `--submit` and all five metadata values; it must not expose `--overwrite` or `--dry-run`. Document this exact command in both READMEs:

```bash
PYTHONPATH=experiments/matching_fairness \
python experiments/matching_fairness/scripts/recover_failed_publication.py \
  --submit \
  --original-request-id 3ae8dc60c2df4166b7d4021f48146487 \
  --original-ledger-sha256 2125615c73c156bea4137c1c764aba6b7893e94cb64d819b6856b8a93b4042be \
  --prior-recovery-request-id 82bbdd064b904c1fb2579b79a7b46353 \
  --prior-recovery-ledger-sha256 2b665cc2d2d69338328c716332d81a412a3829d382a5ec53331cb86c71e05022 \
  --recovery-reason gpfs-renameat2-unsupported
```

The documentation must say this is not a generic retry interface, both predecessor ledgers remain immutable, any `submission_recovery_2.json` blocks reuse, and a scheduler uncertainty requires manual audit rather than rerun.

- [ ] **Step 9: Run GREEN and regression suites**

Run the focused test, then:

```bash
PYTHONPATH=experiments/matching_fairness \
  /hpc2hdd/home/ckwong627/miniconda3/envs/eeg_recon/bin/python -m pytest \
  experiments/matching_fairness/tests/test_recovery.py \
  experiments/matching_fairness/tests/test_recovery_publication.py \
  experiments/matching_fairness/tests/test_orchestration.py -q

PYTHONPATH=experiments/matching_fairness \
  /hpc2hdd/home/ckwong627/miniconda3/envs/eeg_recon/bin/python -m pytest \
  experiments/matching_fairness/tests \
  tests/test_things_trial_selection.py \
  tests/test_train_clip_lora_grad_clip.py --disable-warnings -q
```

Also run `ruff check` on changed Python files, `python -m py_compile` on the new script, and `git diff --check`. Record explicit exit codes. Re-hash both formal predecessor ledgers and all formal runtime/log paths; every baseline hash must match.

- [ ] **Step 10: Commit and obtain an independent review**

Commit only the plan, script, tests, fixture, and bilingual README changes on `ckw`:

```bash
git add docs/superpowers/plans/2026-07-23-second-gpfs-recovery.md \
  experiments/matching_fairness/scripts/recover_failed_publication.py \
  experiments/matching_fairness/tests/test_recovery_publication.py \
  experiments/matching_fairness/tests/fixtures/approved_submission_recovery.json \
  experiments/matching_fairness/README.md \
  experiments/matching_fairness/README_ZH.md
git commit -m "fix(fairness): add audited GPFS recovery"
```

A fresh reviewer must inspect the diff and run only the focused recovery suites. All findings must be fixed and re-reviewed before Task 2.

---

### Task 2: Execute and monitor the formal second recovery

**Files:**
- Create at runtime only: `/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/Class/AIAA3800_L01-Human-Centered_Artificial_Intelligence/Final_Project/test/brain-rw/results/matching_fairness_v3/manifests/submission_recovery_2.json`
- Preserve exactly: `submission.json`, `submission_recovery.json`, all prior logs

**Interfaces:**
- Consumes: independently reviewed `recover_failed_publication.py` and the five terminal-unsuccessful first-recovery jobs.
- Produces: one new five-job SLURM DAG and a durable second-recovery ledger.

- [ ] **Step 1: Perform a no-write pre-submit audit**

Verify branch `ckw`, clean tracked worktree, exact two ledger hashes, no live jobs with either prior token, all first-recovery jobs terminal-unsuccessful in exact seven-field `sacct` output, and all four derived roots absent or empty. Do not delete any file to make the check pass.

- [ ] **Step 2: Submit exactly once**

Run the exact documented CLI command once. Capture stdout, stderr, exit code, resulting ledger SHA-256, new request ID, and all five new job IDs. If the CLI returns uncertainty or partial submission, stop and audit; never invoke it again.

- [ ] **Step 3: Monitor without changing code**

Use `squeue` and `sacct` to follow the exact five new job IDs. Do not modify tracked source while jobs are pending/running. If a job fails, preserve all logs and ledger bytes and diagnose before any new authorization request.

- [ ] **Step 4: Verify formal outputs**

After all five jobs complete, run the existing formal verifier and aggregation tests. Require complete sealed matrices for `nice`, `atm_s`, and `our_project`, all 30 scenarios per model, all five decoders, both standard and duplicate-EEG suites, and the expected 450 decoder records.

---

### Task 3: Publish the bilingual result report

**Files:**
- Create: `experiments/matching_fairness/RESULTS.md`
- Create: `experiments/matching_fairness/RESULTS_ZH.md`

**Interfaces:**
- Consumes: sealed aggregate JSON/CSV and all manifests from Task 2.
- Produces: mutually linked English and Chinese reports with identical numbers.

- [ ] **Step 1: Generate reports only from sealed aggregate artifacts**

Use the repository's existing report generator. Do not hand-copy metrics. Include model × scenario × decoder results for NICE, ATM-S, and Our project; standard and duplicate-EEG suites; top-1/top-5 only where the decoder defines them; and matching-fairness caveats.

- [ ] **Step 2: Verify report identity and links**

Assert both reports exist, link to each other, contain the exact subject/seed/checkpoint rule, and reproduce aggregate values without rounding disagreement.

- [ ] **Step 3: Commit, review, and push `ckw`**

Run the frozen full test suite and `git diff --check`, obtain an independent review, commit result files on `ckw`, then push `ckw` to `origin/ckw`. Do not create a PR or another branch unless the user separately requests it.
