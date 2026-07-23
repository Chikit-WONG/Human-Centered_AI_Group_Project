# NICE / ATM-S / Our Project Matching-Fairness Experiment

English | [中文](README_ZH.md)

This experiment is a fixed-scope comparison of **NICE**, **ATM-S**, and
**Our project (BrainRW)** on THINGS-EEG2. The formal delivery unit is exactly
`sub-08 / seed-42`; it is not an all-subject or multi-seed benchmark.

No formal scores are recorded in this guide. After the sealed run finishes,
the generated reports are written to `aggregate/RESULTS.md` and
`aggregate/RESULTS_ZH.md` under the results root described below.

## Research question and reporting boundary

The standard paper metric ranks each EEG query against the image gallery
independently. The four additional decoders instead inspect the complete
query-by-gallery score matrix, so they are a **transductive batch-level
analysis**, not replacements for standard retrieval. This experiment asks
whether their apparent benefit survives when the known one-to-one test-set
structure is weakened by missing queries, missing images, duplicate gallery
entries, or multiple real EEG measurements for one image.

All five decoders receive the same similarity matrix and identity mapping for
a given model and scenario. Ground-truth targets are used only after decoding
to score the predictions. The formal headline remains Independent Top-1 and
Top-5; constrained matching is reported separately.

Because the sealed scope contains only one subject-seed cell, conclusions must
say **“observed on sub-08 / seed-42.”** The outputs support exact counts,
percentages, and per-query transitions, but not cross-subject or cross-seed
significance claims.

## Models, source lock, and checkpoint sealing

NICE and ATM-S come from the official
[`dongyangli-del/EEG_Image_decode`](https://github.com/dongyangli-del/EEG_Image_decode)
repository's `develop` branch. Preflight fetches the then-current branch head,
checks it out at detached HEAD, requires a clean source tree, and records the
remote URL, branch, exact commit, and tracked-tree SHA-256 in:

```text
test/brain-rw/results/matching_fairness_v3/manifests/upstream_lock.json
```

Every later native training or export stage verifies that lock. The run never
silently substitutes a local NICE/ATM-S implementation for the locked official
source.

NICE and ATM-S use the paper-native training configuration except for one
predeclared protocol correction: the formal checkpoint is selected by the
lowest **validation contrastive loss** (`val_ratio=0.1`, patience `10`; exact
ties choose the earlier epoch). The formal test set is sealed until that
checkpoint, configuration, and hashes are frozen. Test Top-1/Top-5 do not
select the formal checkpoint.

The native exporter also produces `best_test_audit.json`. This is an explicitly
test-selected **reproduction diagnostic only**. It is isolated from the 30
fairness scenarios and must never be described as an unbiased formal result.
Our project uses its pre-existing fixed formal BrainRW checkpoint and audited
evaluation configuration; it is not reselected for this experiment.

Before matching, each model must pass the native-versus-unified Independent
Top-1/Top-5 parity gate on the standard `200 x 200` matrix. A source, hash,
identity-order, shape, finiteness, or parity mismatch fails closed.

## Scenario suites

### Standard suite: 27 scenarios

The standard artifact averages all 80 trials per image. One canonical manifest
is generated with seed `42` and reused by all three models:

```text
drop_query       in {0, 5, 10}
drop_gallery     in {0, 5, 10}
drop_pair        = 0
duplicate_gallery in {0, 10, 20}
```

This gives exactly `3 x 3 x 1 x 3 = 27` scenarios per model.
`drop_query` leaves unmatched real images as distractors; `drop_gallery`
creates EEG queries whose correct image is absent; and `duplicate_gallery`
adds a new gallery entry that shares the original canonical image identity.

### Real duplicate-EEG suite: 3 scenarios

The duplicate-query suite does not copy EEG rows. For every image, each of the
four sessions contains 20 real trials. A deterministic SHA-256 ordering applies
an exact **10/10 per-session split**: 10 trials from each session go to half A
and the other 10 go to half B. Therefore
EEG-A and EEG-B are disjoint, session-balanced averages of 40 real trials each.

- `dupq0`: 200 EEG-A queries x 200 images;
- `dupq10`: append 10 corresponding EEG-B queries, giving `210 x 200`;
- `dupq20`: append 20 corresponding EEG-B queries, giving `220 x 200`.

The ten duplicated identities are a strict subset of the twenty. These three
40-trial scenarios are a robustness suite and their absolute scores must not be
compared with the 80-trial standard suite. Strict one-to-one assignment has a
structural Top-1 ceiling of `200/210 = 95.24%` for `dupq10` and
`200/220 = 90.91%` for `dupq20`; unmatched counts are therefore essential.

Across three models, the complete run contains `3 x (27 + 3) = 90`
model-scenario cells.

## Decoder semantics

| Decoder | Semantics | Top-5 |
|---|---|---|
| Independent | Stable per-row ranking; gallery reuse is allowed | Defined and reported |
| Greedy | Process queries by independent Top-1 confidence, then choose the best unused entry | Undefined |
| Hungarian | Maximize global score with `linear_sum_assignment`; rectangular matrices match only `min(Q,G)` pairs | Undefined |
| Stable Matching | Query-proposing Gale-Shapley with deterministic preferences and tie-breaking | Undefined |
| Sinkhorn | Transport plan with temperature `0.05`, at most `500` iterations, tolerance `1e-8`, followed by row argmax; not strict one-to-one | Undefined |

Greedy, Hungarian, and Stable Matching emit one assignment per matched query;
Sinkhorn emits one row-argmax choice from a transport plan. None produces a
ranked list of five candidates, so an “assignment Top-5” would be invented and
is deliberately not reported. Only Independent reports standard Top-5.
Sinkhorn non-convergence is retained but explicitly flagged rather than hidden.

## Environment

Run commands from the repository root:

```bash
cd /hpc2hdd/home/ckwong627/workdir/new_sub_workdir/Class/AIAA3800_L01-Human-Centered_Artificial_Intelligence/Final_Project/Human-Centered_AI_Group_Project
source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
```

The environment declaration uses the explicit official
`https://conda.anaconda.org/conda-forge` URL plus `nodefaults`, so user-level
channel aliases cannot redirect dependency resolution to a mirror and
Anaconda Defaults are never used to resolve or download this environment.
Conda 26 may still query notice/ToS metadata for globally configured channels
before reading the file; the command never enables or performs ToS acceptance.
Conda creates only the isolated Python 3.12/pip skeleton; setuptools 75.8.0
is pinned as part of that build toolchain because the locked CLIP setup still
imports `pkg_resources`. The
same one-command environment creation then installs exact pip versions. The
PyTorch trio is pinned to `2.5.0+cu124` / `0.20.0+cu124` / `2.5.0+cu124` from
the official CUDA 12.4 PyTorch wheel index, and OpenAI CLIP is pinned to commit
`a9b1bf5920416aaeaec965c25dd9e8f98c864f16`. The command disables pip build
isolation so CLIP uses that pinned toolchain, and clears only an inherited
`LD_LIBRARY_PATH` while pip invokes the system Git client. This prevents
Conda's libffi from being injected into Git and does not modify any existing
environment.

Create the isolated native environment once:

```bash
PIP_NO_BUILD_ISOLATION=1 env -u LD_LIBRARY_PATH conda env create -n atm_native \
  -f experiments/matching_fairness/configs/atm_native_environment.yml
```

If `atm_native` already exists, reconcile it with the declaration instead of
creating it again:

```bash
PIP_NO_BUILD_ISOLATION=1 env -u LD_LIBRARY_PATH conda env update -n atm_native --prune \
  -f experiments/matching_fairness/configs/atm_native_environment.yml
```

Only if the command repeatedly fails while Git fetches the locked CLIP commit
with the same GitHub TLS/transport error, use the already prefetched exact
checkout below. First verify that its HEAD is the expected commit and that the
checkout is clean:

```bash
git -C /hpc2hdd/home/ckwong627/workdir/new_sub_workdir/EEG_Project/models/openai_clip_a9b1bf5920416aaeaec965c25dd9e8f98c864f16_shallow \
  rev-parse HEAD
git -C /hpc2hdd/home/ckwong627/workdir/new_sub_workdir/EEG_Project/models/openai_clip_a9b1bf5920416aaeaec965c25dd9e8f98c864f16_shallow \
  status --short
```

`rev-parse` must print `a9b1bf5920416aaeaec965c25dd9e8f98c864f16`, and
`status --short` must print nothing. Then install only that missing VCS package
without contacting a package index:

```bash
PIP_NO_INDEX=1 PIP_NO_BUILD_ISOLATION=1 env -u LD_LIBRARY_PATH \
  conda run -n atm_native python -m pip install \
  --no-build-isolation --no-deps \
  "git+file:///hpc2hdd/home/ckwong627/workdir/new_sub_workdir/EEG_Project/models/openai_clip_a9b1bf5920416aaeaec965c25dd9e8f98c864f16_shallow@a9b1bf5920416aaeaec965c25dd9e8f98c864f16"
```

The current `atm_native` was completed with this fallback after the Conda
skeleton and all declared scientific wheels had installed successfully; the
remote VCS step itself did not complete reliably in this run.

Then run the exact version/import gate:

```bash
conda run -n atm_native python \
  experiments/matching_fairness/scripts/preflight.py --environment-only
```

Do not install these dependencies into or update the existing `test` or
`eeg_recon` environments. Native NICE/ATM-S jobs activate `atm_native`;
BrainRW export and CPU matching/aggregation use the already established
`eeg_recon` environment in their SLURM scripts.

## Preflight, dry-run, and submission

Activate the native environment before using the wrapper because formal
preflight runs locally:

```bash
conda activate atm_native
```

Run source/data/environment/provenance preflight without submitting a job:

```bash
bash experiments/matching_fairness/run_matching_fairness.sh --phase preflight
```

Render the complete fixed DAG without creating runtime directories or
submitting jobs:

```bash
bash experiments/matching_fairness/run_matching_fairness.sh \
  --phase all --dry-run
```

The dry-run must state exactly **2 training cells, 3 main model exports,
2 native audit cells, 90 scenarios, and 450 decoder records**. The formal DAG
is:

```text
preflight -> native training array -> native main export array
preflight -------------------------> BrainRW main export
native main exports + BrainRW export -> native audit array
native audits -> matching -> aggregation
```

After reviewing the dry-run, submit the complete DAG once:

```bash
bash experiments/matching_fairness/run_matching_fairness.sh \
  --phase all --submit
```

The submission ledger is written before each `sbatch` call and binds the job
IDs and dependencies. A repeated `all --submit` is intentionally rejected; it
is not a resume command.

## Phased execution and resume behavior

For deliberate phased execution, first complete preflight, then submit native
training:

```bash
bash experiments/matching_fairness/run_matching_fairness.sh \
  --phase train --submit
```

After both validation-selected checkpoints are complete and current, submit
the three main exports and two native audit cells:

```bash
bash experiments/matching_fairness/run_matching_fairness.sh \
  --phase export --submit
```

Once those jobs succeed, the hash-bound CPU phases can be run or resumed
locally:

```bash
bash experiments/matching_fairness/run_matching_fairness.sh --phase match
bash experiments/matching_fairness/run_matching_fairness.sh --phase aggregate
```

Completed preflight, matching, and aggregation phases are verified against
their manifests and skipped when current. Partial, orphaned, hash-mismatched,
failed, or unknown states fail closed and require inspection of the ledger and
logs before recovery. `--overwrite` is only for an intentional, reviewed
replacement of derived artifacts; it is not a generic retry or resume switch
and does not erase submission-ledger history.

### Audited one-time failed-DAG recovery

The recovery command below is a fixed incident path for the reviewed
spool-entrypoint failure only. It is **not** overwrite, resume, or a generic
retry. Under the submission lock it verifies the immutable original ledger and
all five exact root jobs against authoritative `sacct -X` terminal-unsuccessful
records, requires the checkpoint/matrix/run/aggregate roots to be absent or
empty, and then reserves `manifests/submission_recovery.json` before the first
new `sbatch` call. The original `submission.json` remains byte-identical.

```bash
bash experiments/matching_fairness/run_matching_fairness.sh \
  --phase all \
  --submit \
  --recover-failed-all \
  --original-request-id 3ae8dc60c2df4166b7d4021f48146487 \
  --original-ledger-sha256 2125615c73c156bea4137c1c764aba6b7893e94cb64d819b6856b8a93b4042be \
  --recovery-reason spool-entrypoint-bug
```

Any existing recovery path—even empty, malformed, failed, completed, or a
symlink—permanently blocks another recovery attempt. A scheduler mismatch,
nonterminal/successful old job, unsafe output root, or original-ledger mutation
fails closed; do not delete or edit either ledger to force a retry.

The aggregation phase can also be invoked directly after a complete matching
tree:

```bash
PYTHONPATH=experiments/matching_fairness \
/hpc2hdd/home/ckwong627/miniconda3/envs/eeg_recon/bin/python \
  experiments/matching_fairness/scripts/aggregate_results.py \
  --results-root /hpc2hdd/home/ckwong627/workdir/new_sub_workdir/Class/AIAA3800_L01-Human-Centered_Artificial_Intelligence/Final_Project/test/brain-rw/results/matching_fairness_v3
```

## Outputs and logs

Large checkpoints, score matrices, scenario ledgers, and generated reports are
kept outside Git under:

```text
/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/Class/AIAA3800_L01-Human-Centered_Artificial_Intelligence/Final_Project/test/brain-rw/results/matching_fairness_v3/
  manifests/                         # source, asset, preflight, trial, phase, and submission locks
  checkpoints/{nice,atm_s}/          # validation-selected native checkpoints
  matrices/{nice,atm_s,our_project}/ # standard, eeg_a, and eeg_b score artifacts
  runs/<model>/subj08/seed42/
    standard/                         # 27 scenario JSON/CSV pairs
    duplicate_eeg/                    # 3 scenario JSON/CSV pairs
  aggregate/
    RESULTS.md
    RESULTS_ZH.md
    aggregate_metrics.csv
    aggregate_summary.json
    presentation_standard.md
    presentation_duplicate_eeg.md
```

All scheduler stdout/stderr files are kept under:

```text
/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/Class/AIAA3800_L01-Human-Centered_Artificial_Intelligence/Final_Project/test/brain-rw/logs/matching_fairness_v3/
```

## Verification before formal submission

The cluster's `test` environment currently does not contain `pytest`, so the
repository-only complete suite and fixture smoke use the already compatible
`eeg_recon` test runner without changing that environment. Run:

```bash
PYTHONPATH=experiments/matching_fairness \
/hpc2hdd/home/ckwong627/miniconda3/envs/eeg_recon/bin/python -m pytest \
  experiments/matching_fairness/tests \
  tests/test_things_trial_selection.py \
  tests/test_train_clip_lora_grad_clip.py -v
```

Run the native-import subset in the declared environment:

```bash
PYTHONPATH=experiments/matching_fairness \
/hpc2hdd/home/ckwong627/miniconda3/envs/atm_native/bin/python -m pytest \
  experiments/matching_fairness/tests/test_native_training.py \
  experiments/matching_fairness/tests/test_native_export.py -v
```

Finally, verify that two independent fixture runs produce byte-identical
aggregate CSV, summary JSON, and both reports:

```bash
PYTHONPATH=experiments/matching_fairness \
/hpc2hdd/home/ckwong627/miniconda3/envs/eeg_recon/bin/python -m pytest \
  experiments/matching_fairness/tests/test_orchestration.py::test_fixture_pipeline_is_byte_stable -v
```
