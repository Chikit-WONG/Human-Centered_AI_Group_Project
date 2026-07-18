# SAMGA-LoRA execution audit log

This log records protocol-relevant execution decisions. Generated metrics and
machine-readable provenance remain under the ignored `artifacts/samga_lora`
tree.

## Frozen baseline and inputs

- Work began from commit `a97b97a110c0fea7d4adafd5abce477c6cce525c` on the new
  branch `experiment/samga-lora`; the prior result was tagged locally as
  `clip-lora-baseline-v1`.
- Ten subject-specific train/test manifests passed row-count, channel, image-path,
  concept-disjointness, and cross-subject row-hash checks.
- Shared CLIP feature caches were produced by job `9988393`: both array cells
  completed successfully.

## W1

- The first two scheduler attempts (`9988377`, `9988386`) failed before Python
  training because of, respectively, SLURM spool-path resolution and strict-shell
  interaction with a third-party conda activation hook. The wrappers were fixed;
  neither attempt created a cache or checkpoint.
- Cache/online and Frozen/zero-LoRA parity job `9988409` completed successfully.
- Frozen/LoRA smoke array `9988410` completed successfully. Both task gradients
  were nonzero, the LoRA visual gradient was nonzero, checkpoint reconstruction
  passed, and both validation paths used standard independent 200-way retrieval.
- After correcting the normalization protocol described below, smoke job
  `9988618` was first submitted to `debug` and cancelled without starting when
  all debug A40s remained occupied. Replacement job `9988639` ran on
  `i64m1tga40u`; both cells completed successfully. Frozen visual gradient was
  exactly zero, LoRA visual gradient was nonzero, paired task-initialization
  hashes matched, and the recorded flags were `eeg_l2norm=false` and
  `image_l2norm=true`.
- The final preflight report was regenerated after the protocol correction. It
  passed and records Python/package versions, all manifest row hashes, the CLIP
  config hash, and the local `model.safetensors` SHA-256 hash.

## W2

- Pilot job `9988419` was stopped before use after an audit found that LoRA
  creation consumed RNG before task-model initialization. Its outputs were moved
  to `pilot_v1_invalid_rng` and are excluded from selection.
- Training now resets the seed immediately before constructing the task model and
  records a SHA-256 hash of its initial state. The selector requires exact hash
  equality for every paired Frozen/LoRA cell.
- Corrected-RNG pilot array `9988445` was subsequently stopped and moved to
  `pilot_v2_invalid_both_l2norm`: a second audit found that it normalized both
  EEG and image features, whereas the released paper setup and launchers apply
  L2 normalization only to image features. It is excluded from selection.
- The loss now exposes and records both normalization flags, defaulting to the
  released SAMGA configuration (`eeg_l2norm=false`, `image_l2norm=true`).
- The corrected official-loss pilot was submitted as job `9988646`. It is the
  only pilot root eligible for selection; the two invalid roots above remain
  quarantined and are never read by the selector.
- After tasks 0--14 had started, the shared A40 nodes were reserved for
  higher-priority partitions. Still-pending tasks 15--23 were cancelled before
  start and resubmitted unchanged to `i64m1tga40ue` as job `9988770`; no run
  directory was duplicated or overwritten.
- Completed LoRA pilot cells took at most about 27 minutes, while the initial
  launcher requested 12 hours. To permit scheduler backfill, still-pending tasks
  18--23 were cancelled before start and resubmitted unchanged with a one-hour
  limit as job `9988802`. The versioned pilot/formal launchers now request one
  and two hours, respectively; these limits remain comfortably above measured
  runtimes and do not alter scientific settings.

## W3 preparation

- Before pilot selection/source locking, the formal launcher was set to four
  allocated CPUs, four data-loader workers, and 48 GiB requested memory per GPU
  job. This is an execution resource setting, not a model or data change, and is
  applied uniformly to all 100 formal cells. The 48 GiB request is conservative
  relative to measured GPU/model/data usage while allowing scheduler backfill.
  The formal wrapper is included in the selector's source-hash lock.
- A direct 100-task formal-array submission was rejected before job creation by
  the cluster's `normal` QOS (`MaxSubmitJobsPU=10`). The launcher now defaults
  to a ten-task chunk, and `submit_formal_chunk.sh` validates/submits the
  contiguous chunks 0--9 through 90--99 sequentially. This scheduling-only
  change does not alter the locked formal wrapper or scientific protocol.
- Formal chunk 0--9 was initially submitted as job `9988980` with a two-hour
  limit, but was cancelled while entirely pending because that limit prevented
  reservation backfill; it created no run directory or test metric. The chunk
  submitter now requests 30 minutes for cached Frozen chunks 0--49 and one hour
  for selected-epoch-25 LoRA chunks 50--99, both well above measured runtimes.
- Its 30-minute shared-partition replacement, job `9988982`, was also cancelled
  while all tasks were pending because the nodes remained reserved for higher
  partitions. The chunk submitter now accepts an explicitly audited A40
  partition argument so unchanged chunks can follow the documented shared ->
  exclusive -> emergency escalation policy.
- Chunk 0--9 was then submitted to `i64m1tga40ue` as job `9988984`, but all ten
  tasks remained pending behind higher-priority work, so it too was cancelled
  before start. After both lower tiers were demonstrably blocked, unchanged
  chunk 0--9 was submitted to `emergency_gpua40` as job `9988986`; eight tasks
  started immediately and two waited only on the per-user eight-GPU limit.
- Frozen chunks 0--9, 10--19, 20--29, 30--39, and 40--49 completed as emergency
  jobs `9988986`, `9989000`, `9989012`, `9989034`, and `9989048`; all 50 cells
  exited 0 and produced one checkpoint, metric file, and prediction file.
- LoRA chunk 50--59 was submitted as job `9989064`. Once cells 50--57 completed
  and only 58--59 remained active, non-overlapping range 60--67 was submitted as
  job `9989159`. This rolling eight-cell scheduling keeps the QOS maximum of ten
  submitted tasks and eight running GPUs utilized without changing any formal
  index, seed, subject, or training configuration.
- After cells 60--65 completed and only 66--67 remained active, non-overlapping
  range 68--75 was submitted as job `9989248`, continuing the same rolling
  policy.
- After cells 68--73 completed and only 74--75 remained active, non-overlapping
  range 76--83 was submitted as job `9989309`.
- After cells 76--81 completed and only 82--83 remained active, non-overlapping
  range 84--91 was submitted as job `9989382`.
- After cells 84--89 completed and only 90--91 remained active, the final
  non-overlapping range 92--99 was submitted as job `9989444`.
- All 100 formal cells completed with `ExitCode 0:0`: 50 Frozen controls and
  50 paired LoRA runs. Every cell contains one epoch-25 checkpoint, metric
  file, 200-row prediction file, run configuration, training history, and
  completion marker; all formal stderr files are empty.
- The locked-source check still passed after the final task. Strict aggregation
  and an independent recomputation over all 50 pairs agreed exactly: Frozen
  Top-1/Top-5 = 76.17%/95.79%, LoRA = 82.68%/97.67%, paired changes =
  +6.51/+1.88 percentage points, and the two-way subject/seed bootstrap Top-1
  95% interval = [+5.22, +7.69] points. The pre-registered success criterion
  passed.

## Exploratory inferred-InternViT follow-up

- Because the confirmatory CLIP gate passed, the pre-registered exploratory
  follow-up was opened. The SAMGA paper/repository specifies 3200-dimensional
  layers 20/24/28/32/36 at 448-pixel resolution but does not disclose an exact
  checkpoint identifier or feature-extraction implementation.
- `OpenGVLab/InternViT-6B-448px-V1-5` was selected as a plausible, explicitly
  inferred candidate and pinned to revision
  `03e138c81d3fd538c77439fd43a42c067d827427`. The exploratory code defines
  embedding output as hidden-state zero and captures indices 20/24/28/32/36.
- Official Hub metadata supplied the three shard SHA-256 values. The cluster's
  Hugging Face HEAD/redirect path could query metadata but could not sustain
  snapshot downloads, so byte-identical LFS objects are transferred through a
  ModelScope CDN copy and must match all three official hashes before model
  loading. This transport change does not change model bytes.
- The follow-up is isolated under `exploratory_internvit/`, uses frozen features,
  released-launcher seed 2025, a fixed epoch-60 checkpoint, one test evaluation,
  and standard independent retrieval. It is never labelled as an exact SAMGA
  paper reproduction or mixed into the confirmatory CLIP result.
- Debug smoke job `9989675` was cancelled while still pending after the debug
  queue reached seven pending jobs and the shared A40 partition had no pending
  work. Its shared-partition replacement, `9989678`, exited before loading the
  model because the new exploratory SLURM wrapper incorrectly resolved its
  source path from Slurm's spool copy. No feature cache was created. All three
  exploratory wrappers now resolve the repository from `SLURM_SUBMIT_DIR` (or
  an explicit `SAMGA_PROJECT_ROOT`) and pass `bash -n`.
- Corrected smoke jobs `9989682` (batch 16) and `9989687` (batch 32) both
  completed on `i64m1tga40u`. Each produced a finite 64 x 5 x 3200 float16
  cache with the identical SHA-256
  `ec4f8a491ea536916b7f694f91abb4739e1faf12aee8b72c215ff1ac34565f00`.
  Their extraction times were 12.01 and 11.35 seconds and peak allocated CUDA
  memory was 15.35 and 20.36 GiB, respectively. The full sharded extraction
  therefore uses batch 32.
- Full eight-shard train plus one-test extraction was submitted as shared A40
  array job `9989695`. Each train shard covers a precomputed, non-overlapping
  integer interval of the 16,540 manifest rows; strict merging remains gated on
  all nine tasks exiting successfully and matching the pinned provenance.
- All nine tasks of job `9989695` exited `0:0`. The eight train shards each
  completed in 6:01--6:21 and merged into a complete 16,540 x 5 x 3,200 cache;
  the test task completed in 1:00 and produced 200 x 5 x 3,200 features. The
  merged train/test SHA-256 values are respectively
  `b9b7d2eeae87f1f56565a0c5fdd2c3bae28751994964452a2a8a7ad8c6f4ea73`
  and `629109bd2a43cd8ac48d36d0eea7cce61f80674e6fa401cd514d01335230f976`.
  Both caches passed provenance, shape, layer, finite-value, and shared-record
  validation against all ten subject manifests.
- Training smoke job `9989736` was cancelled while pending in a congested debug
  queue and replaced unchanged on shared A40 as `9989738`. During that run a
  configuration audit found that the wrapper-only 3,200-dimensional width was
  inferable from weights but absent as an explicit checkpoint-config field.
  The job was cancelled before checkpoint/test evaluation and its partial
  directory was quarantined under
  `exploratory_seed2025_invalid_missing_image_dim`. The wrapper now injects
  `image_dim=3200` into both `run_config.json` and checkpoint configuration;
  unit coverage was added. Corrected subject-01 smoke job `9989748` is the only
  eligible subject-01 run.
- Corrected subject-01 job `9989748` completed `0:0` in 2:15. Its epoch-60
  checkpoint reload passed, its stderr was empty, and standard independent
  retrieval was 76.0%/96.0% over 200 queries. Remaining subjects 02--10 ran as
  shared A40 array `9989754`; all nine cells completed `0:0` in 2:04--2:25 and
  all stderr files were empty.
- Strict aggregation re-derived every target rank from all ten 200-row
  prediction files and checked the model revision/hashes, cache hashes, explicit
  image width, manifest provenance, 60-record training history, completion
  marker, sole epoch-60 checkpoint, checkpoint reload, normalization flags, and
  no-Hungarian protocol. An independent prediction-only audit agreed: Top-1
  1661/2000 = 83.05%, Top-5 1960/2000 = 98.00%.
- Exploratory result-file SHA-256 values: `summary.json`
  `41a2366c5b9195cf8aee75a6e50d577ed5437e76cb9dc40a43c5639e41d45efc`,
  `per_subject_metrics.csv`
  `ad73380d24e698ffa5f87de3c5043f3c106f5b7e9b52dc1faddeb47d2d309b86`,
  `RESULTS_EN.md`
  `41d3bb9e25cc4501614770c31132c08abcf12596332a6705fcccb122912c362a`,
  and `RESULTS_ZH.md`
  `b8d59a1b145b4e104260e3b2d6c6118dc948420280c2aaf5e55b7b26e93f61c7`.
- Final verification passed: 23 repository tests, Python compile, all Bash/SLURM
  syntax checks, confirmatory source-lock verification, `git diff --check`, six
  relevant Markdown relative-link audits, balanced fenced-code markers, and
  bilingual key-result token checks. Runtime experiments remained in `test`;
  tests used the compatible `eeg_recon` environment because the cluster's
  existing `test` environment does not include the optional pytest package.
  The final pytest transcript is under `logs/samga_lora/final-tests.out`, and no
  SAMGA job remains queued or running.
