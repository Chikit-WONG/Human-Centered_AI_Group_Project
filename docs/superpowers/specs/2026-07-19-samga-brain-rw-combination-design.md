# SAMGA + brain-rw Prioritized Experiment Design

[中文](./2026-07-19-samga-brain-rw-combination-design-zh.md)

**Date:** 2026-07-19  
**Status:** Confirmed for implementation<br>
**Repository:** `Human-Centered_AI_Group_Project`  
**Experiment namespace:** `samga_brain_rw`

## 1. Objective

Determine whether the transferable parts of brain-rw—especially visual low-rank
adaptation, two-timescale optimization, and complementary CLIP similarities—can
improve SAMGA on standard THINGS-EEG2 zero-shot EEG-to-image retrieval.

The project has two distinct targets:

1. **Reproduction target:** beat the current fixed-epoch local SAMGA
   exploratory benchmark, Top-1 **89.02%** and Top-5 **98.87%**.
2. **Paper-numeric target:** exceed the SAMGA paper's reported Top-1
   **91.30%** under this project's locked standard independent 200-way
   retrieval definition.

A result between 89.02% and 91.30% is an improvement over the local
reproduction. A value above 91.30% may be described only as numerically higher
under the local locked protocol, not as a strict reproduction or proof that the
method exceeds SAMGA under the paper's unspecified checkpoint, extractor, and
seed details. Top-5 is already near saturation and is treated mainly as a
non-regression metric.

## 2. Evidence motivating the design

- The audited brain-rw recipe uses a BrainMLP, LAION CLIP ViT-B/32 rank-32
  visual LoRA, and TTUR (brain learning rate `5e-4`, visual learning rate
  `5e-5`). Its fixed-epoch 10-subject, 5-seed result is 86.66% Top-1 and
  98.38% Top-5.
- A controlled SAMGA-on-CLIP experiment found 82.68% / 97.67% with visual
  LoRA versus 76.17% / 95.79% when frozen: paired gains of +6.51 and +1.88
  percentage points. This supports testing LoRA/TTUR, but it does not establish
  the same gain for InternViT.
- The fixed-epoch local inferred-InternViT-V2.5 benchmark is 89.02% / 98.87%.
  It remains exploratory because the feature/checkpoint semantics were
  historically informed by observed test results. A 91.82% test-selected
  diagnostic is additionally excluded because it directly uses the formal test
  set for checkpoint selection.
- Existing InternViT caches support frozen-backbone experiments only. A module
  trained after cached features is a feature-space adapter, not InternViT LoRA.
  True LoRA changes internal hidden states and therefore requires online image
  forwarding, followed by cache regeneration if the adapter is frozen.

## 3. Non-negotiable evaluation protocol

### 3.1 Split hierarchy and test resealing

The preflight must find exactly 1,654 non-test training concepts; otherwise it
aborts and requires a reviewed protocol amendment. Partition them as follows:

1. Canonicalize each concept ID as UTF-8.
2. Compute `SHA256("AIAA3800-SAMGA-SPLIT-v1\n" + concept_id)` and sort by the
   hexadecimal digest, breaking an impossible digest tie by concept ID.
3. Assign ranks 1–200 to `val-dev`, ranks 201–400 to `val-confirm`, and the
   remaining 1,254 concepts to `train`.
4. For each validation concept, canonicalize all stimulus IDs and select the
   one with the smallest
   `SHA256("AIAA3800-SAMGA-STIM-v1\n" + split + "\n" + concept_id + "\n" +
   stimulus_id)` digest as its sole EEG query and matching gallery image.
5. Exclude every stimulus from the 400 validation concepts from gradient
   updates during development, and commit query/gallery IDs, concept IDs, and
   hashes before training.

This creates four non-overlapping decision roles:

1. **Train:** 1,254 concepts used for development-stage gradient updates.
2. **Development validation (`val-dev`):** exactly 200 independent EEG queries
   against a 200-image gallery, used for pilot selection on subjects 01, 05,
   and 08 with seeds 42 and 43.
3. **Confirmation validation (`val-confirm`):** a disjoint 200-query,
   200-image task used once to compare locked survivors on the same 10-subject
   x 5-seed grid.
4. **Formal test:** the original 200-query, 200-image task, resealed for all new
   model decisions.

After `val-confirm` locks one final candidate, retrain that candidate and its
strict matched control from scratch on the union of all 1,654 non-test concepts
with the already locked epoch, schedule, and hyperparameters. This refit has no
validation, early stopping, or checkpoint selection. It restores the full
training-data scope before the one-time formal test.

- All architecture choice, hyperparameter choice, checkpoint epoch, stage
  promotion, and stopping decisions use `val-dev`.
- Cross-family ranking uses `val-confirm` only after each survivor's
  configuration is frozen.
- The formal 200-way test set has been observed in historical work, so it is not
  a pristine holdout. It is nevertheless resealed during Stages 0–5: no new
  metrics, predictions, or error analyses may be inspected.
- Validation selects exactly one final candidate. Only that candidate and its
  paired locked baseline enter the formal 10-subject, 5-seed evaluation.
- Test-best epoch, test early stopping, and iterative decisions based on test
  results are prohibited.
- Existing test-exposed checkpoints may be used only for engineering smoke
  checks or explicitly labelled diagnostics, never for the final claim.

### 3.2 Retrieval definition

- Subjects: `01` through `10`.
- Final seeds: `42, 43, 44, 45, 46`.
- Standard independent 200-way cosine-similarity retrieval.
- Primary metric: macro mean Top-1 over the 50 subject-seed cells.
- Secondary metric: macro mean Top-5.
- Hungarian/global one-to-one assignment is excluded from the primary
  experiment. If retained at all, it is reported separately as transductive
  analysis and never compared with standard Top-1.

### 3.3 Paired comparisons

For every candidate-versus-baseline cell:

- use the same subject and seed;
- reuse task-model initialization and data-order hashes where architecture
  permits;
- match warm-up, effective batch size, optimization steps, validation schedule,
  and locked stopping epoch;
- record any unavoidable mismatch in the run manifest;
- calculate per-cell paired deltas and a two-way subject/seed cluster-bootstrap
  95% confidence interval.

Every final candidate reports its absolute difference from the historical
89.02% / 98.87% benchmark and its paired difference from a strict matched
causal control:

- Stage 1: the stronger single branch;
- Stage 2: frozen-feature SAMGA with matched parameter/optimization controls;
- Stage 3: the online-frozen control;
- Stage 4: the shared-frozen control;
- Stage 5: the better of the strongest single branch and locked Stage 1 fusion.

### 3.4 Final success gate

The selected method is considered a robust local improvement only if all of the
following hold on the final 10 x 5 test grid:

- absolute mean Top-1 is above the historical 89.02% benchmark;
- mean paired Top-1 gain over the strict matched control is at least **+0.50
  percentage points** (`>= 0.005` when metrics use the `[0, 1]` scale);
- the two-way cluster-bootstrap 95% CI lower bound for Top-1 gain is above zero;
- mean paired Top-5 gain is no worse than **-0.20 percentage points**
  (`>= -0.002` on the `[0, 1]` scale);
- at least 8 of 10 subjects have a positive seed-mean Top-1 delta;
- no subject's seed-mean Top-1 delta is below **-2.00 percentage points**
  (`>= -0.02` on the `[0, 1]` scale).

A result above 91.30% may be described as numerically above the paper headline
only under this project's locked protocol. It is not an exact reproduction or
a proof of superiority under the paper's protocol because the precise
checkpoint, extractor, and seeds are not specified. Compute cost, trainable
parameter count, peak GPU memory, training time, number of inference encoders,
and inference latency must accompany accuracy.

## 4. Prioritized experiment ladder

All stages below operate on validation until a single final candidate is locked.
A failed stage does not justify opening the formal test set.

### Stage 0 — Protocol, provenance, and baseline lock

Before adding a method:

1. Keep the upstream SAMGA checkout read-only and record its commit.
2. Lock the inferred InternViT V2.5 checkpoint, preprocessing, input resolution,
   output layers, token exclusion, pooling, and normalization semantics.
3. Record model/checkpoint hashes and feature-cache manifests.
4. Reproduce the chosen frozen baseline through the new evaluator without
   reading formal-test metrics during development.
5. Verify that query/gallery IDs and ordering are explicit, unique, and hashed.
6. If author-provided extractor details become available, treat them as a new
   pre-registered baseline rather than silently changing the locked baseline.

Exit condition: deterministic baseline parity on train/validation data and a
complete provenance manifest.

### Stage 1 — Validation-only score-level fusion diagnostic

This is the cheapest test of whether InternViT-SAMGA and brain-rw/CLIP-LoRA make
complementary errors.

1. Emit per-query similarity matrices from the frozen InternViT-SAMGA branch
   and a newly trained brain-rw/CLIP-LoRA branch with formal-test loading
   disabled. A historical replay is permitted only for engineering diagnostics,
   not candidate selection.
2. Require exact query and gallery ID/hash equality before fusion.
3. Evaluate exactly this grid on `val-dev`:
   - query-wise z-normalized convex fusion with InternViT weight
     `alpha in {0.0, 0.1, ..., 1.0}`;
   - temperature-calibrated convex fusion with each branch temperature in
     `{0.5, 1.0, 2.0}` and `alpha in {0.25, 0.50, 0.75}`;
   - reciprocal-rank fusion with `k in {10, 30, 60}` and InternViT branch
     weight in `{0.25, 0.50, 0.75}`.
4. Select calibration and fusion weights on `val-dev` only.
5. Compare against the stronger single branch, not merely the weaker branch.
6. Resolve exact metric ties by lower compute, then lexicographic config ID;
   do not inspect individual formal-test examples.

This is an ensemble result, not a single-backbone SAMGA improvement. It must
report the cost of two image encoders. It also gates Stage 5: if CLIP errors are
not complementary, the train-time dual-branch fusion is not pursued.

Pilot promotion gate on subjects 01, 05, and 08 with seeds 42 and 43:

- `val-dev` Top-1 gain over the best single branch at least +0.30 percentage
  points (`>= 0.003` on the `[0, 1]` scale);
- at least 4 of 6 paired cells are positive;
- mean Top-5 change no worse than -0.20 percentage points (`>= -0.002`);
- no pilot subject's seed-mean Top-1 drops by more than 2 percentage points
  (`>= -0.02`).

### Stage 2 — Cheap frozen-feature improvements and controlled adapter

Use existing InternViT caches to test these exact one-factor candidates:

1. explicit per-layer feature LayerNorm on/off;
2. train-fitted whitening on/off, with all statistics estimated from train only;
3. shared versus separate per-layer image pre-projectors;
4. arithmetic checkpoint averaging over the last 5 or 10 locked-schedule
   checkpoints, and SWA over the same two windows;
5. a residual feature-space low-rank adapter.

Candidates are evaluated separately; post-hoc combinations are not allowed in
this protocol.

For layer feature `h_l`, the initial adapter candidate is:

```text
h'_l = h_l + gamma_l * B_l GELU(A_l LayerNorm(h_l))
```

`B_l` is zero-initialized so the candidate is exactly the baseline at step
zero, and `gamma_l` starts at 1.0. The exact adapter grid is:

- rank in `{8, 16, 32}`;
- adapter-to-task learning-rate ratio in `{0.05, 0.10}`;
- all other optimizer and schedule values copied from the locked baseline.

The adapter must be labelled **cached InternViT feature adapter**, never
InternViT LoRA. Resolve ties by higher `val-dev` Top-5, then lower parameter
count, then lexicographic config ID.

Because SAMGA already contains a trainable image projection, the adapter is
compared with:

- adapter-off identity;
- a dense bottleneck MLP control matched within 1% trainable parameters;
- separate per-layer pre-projectors with comparable parameter count;
- identical optimizer, data order, and training schedule for every control.

Promotion gate on the same 3 x 2 pilot:

- mean `val-dev` Top-1 gain at least +0.50 percentage points (`>= 0.005`);
- at least 4 of 6 paired cells are positive;
- mean Top-5 change no worse than -0.20 percentage points (`>= -0.002`);
- no pilot subject's seed-mean Top-1 drops by more than 2 percentage points
  (`>= -0.02`).

Only the best validation candidate from this stage remains eligible.

### Stage 3 — True late-block InternViT LoRA feasibility pilot

This is the primary causal test of transferring brain-rw's visual adaptation.
It is deliberately piloted before expensive expansion.

1. Warm up SAMGA with locked frozen caches for epochs 1–45 of the 60-epoch
   schedule.
2. For epochs 46–60, load InternViT V2.5 online and target transformer blocks
   28 through 36 inclusive, using the locked extractor's layer-number convention.
3. In each target block, adapt attention query/key/value/output projections and
   MLP `fc1`/`fc2` linear modules.
4. Stage 0 writes the exact concrete module paths to `lora_targets_v1.json`.
   Missing or ambiguous semantic mappings abort the pilot; they are not replaced
   ad hoc.
5. Evaluate exactly four configurations: rank in `{4, 8}` crossed with
   visual-adapter-to-task learning-rate ratio in `{0.01, 0.05}`. Set LoRA alpha
   to twice the rank and dropout to `0.05`.
6. Keep every base InternViT parameter frozen and retain the locked task-model
   learning rate and optimizer.
7. Continue with BF16, gradient checkpointing, a memory-safe microbatch, and
   recorded gradient accumulation.
8. Emit hidden states 20, 24, 28, 32, and 36 with the locked token exclusion,
   patch pooling, and normalization semantics.

Before optimization:

- online-frozen outputs over a fixed 32-sample parity set must match the cache
  with `rtol=1e-3`, `atol=1e-4`, and mean cosine at least `0.9999`;
- cached and online-frozen Top-1/Top-5 predictions and aggregate metrics must
  be identical; report mean top-20 set overlap diagnostically, but do not
  require the complete 200-item ordering to match position by position;
- zero-LoRA output must match online frozen output;
- only LoRA parameters and intended SAMGA parameters may receive gradients;
- save/reload must preserve outputs and optimizer state.

Each LoRA pilot cell has an online-frozen paired control with identical warm-up,
online steps, batch order, and optimizer switch. The pilot grid is subjects 01,
05, and 08 with seeds 42 and 43. Evaluate all four locked configurations. Choose
by mean `val-dev` Top-1, then Top-5, then lower rank, then lower LR ratio. No
additional ranks, blocks, modules, epochs, or ratios may be introduced without
a reviewed version of this design.

Promotion uses the exact Stage 2 +0.50-point gate relative to the online-frozen
control. If no configuration passes, this family stops. OOM handling may lower
microbatch and increase accumulation, but it must preserve and record effective
batch size.

### Stage 4 — Shared true LoRA, freeze, and rebuild caches

If Stage 3 passes its promotion gate, prefer the more economical shared-adapter
path before 50 independent online runs:

1. For each seed, train one subject-aware shared InternViT LoRA on pooled
   training subjects; gradient updates use train concepts only.
2. Reuse the exact single rank, target set, schedule, and LR ratio selected in
   Stage 3; Stage 4 introduces no new hyperparameters.
3. Include a paired shared-frozen control with the same schedule.
4. Lock the LoRA and all choices before creating any formal-test cache.
5. Regenerate deterministic train and validation InternViT caches. Regenerate
   the test cache only if this becomes the single locked final candidate.
6. Train downstream SAMGA on train only; model selection uses `val-dev` only.

Each of the five final seeds gets an independently trained shared adapter so
uncertainty includes adapter-training randomness.

Promotion is measured against the shared-frozen control on the 3 x 2 pilot:

- mean `val-dev` Top-1 gain at least +0.50 percentage points (`>= 0.005`);
- at least 4 of 6 paired cells are positive;
- mean Top-5 change no worse than -0.20 percentage points (`>= -0.002`);
- no pilot subject's seed-mean Top-1 drops by more than 2 percentage points
  (`>= -0.02`).

If shared LoRA fails but the Stage 3 joint candidate passed, only the already
locked Stage 3 configuration remains eligible for confirmation; no new rank,
target, schedule, or LR choice is introduced.

### Stage 5 — Conditional train-time dual-branch late fusion

Implement this stage only if Stage 1 passes its promotion gate. Do not alter
SAMGA's subject-aware router or pretend CLIP is an InternViT granularity. Keep
two explicit branches:

1. Branch I is the locked InternViT-SAMGA model.
2. Branch C is the controlled SAMGA-on-CLIP branch using LAION CLIP ViT-B/32.
3. Both branches receive the same EEG query and emit ID-aligned normalized
   cosine-similarity matrices `S_I` and `S_C`.

Fuse them with three trainable scalars:

```text
lambda = sigmoid(a)
T_I = exp(t_I)
T_C = exp(t_C)
S_fused = lambda * S_I / T_I + (1 - lambda) * S_C / T_C
loss = CE(S_fused, y) + 0.25 * CE(S_I, y) + 0.25 * CE(S_C, y)
```

Initialize `a = t_I = t_C = 0`. During epochs 1–45, train the two task branches
with their locked frozen-visual schedules. During epochs 46–60, continue the
task branches and optimize the fusion scalars. In the frozen-CLIP arm, CLIP has
no gradients. In the CLIP-LoRA arm, use rank 32, alpha 32, dropout `0.05`,
target `q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`, `fc2`, and
`visual_projection`, and use a visual-to-task LR ratio of `0.10`.

Compare exactly four arms on `val-dev`: InternViT only, CLIP only, dual branch
with frozen CLIP, and dual branch with CLIP-LoRA/TTUR.

Promotion is measured against the better of the strongest single branch and
the locked Stage 1 score fusion:

- mean `val-dev` Top-1 gain at least +0.50 percentage points (`>= 0.005`);
- at least 4 of 6 paired cells are positive;
- mean Top-5 change no worse than -0.20 percentage points (`>= -0.002`);
- no pilot subject's seed-mean Top-1 drops by more than 2 percentage points
  (`>= -0.02`).

If the gate fails, the learned branch has no survivor. Any reported result is
labelled multi-backbone and includes the cost of both encoders.

## 5. Candidate selection without test leakage

Each stage writes `val-dev` metrics and decisions to a registry.

After all applicable stages:

1. Discard candidates that failed their `val-dev` promotion gate.
2. Freeze at most one survivor configuration per stage.
3. Evaluate every survivor and its strict matched control exactly once on the
   same `val-confirm` 10-subject x 5-seed grid. Do not change any
   hyperparameters after this evaluation.
4. Require the confirmation gate:
   - mean paired Top-1 gain at least +0.50 percentage points (`>= 0.005`);
   - two-way cluster-bootstrap 95% CI lower bound above zero;
   - mean paired Top-5 change at least -0.20 percentage points (`>= -0.002`);
   - positive seed-mean Top-1 delta for at least 8 of 10 subjects;
   - no subject seed-mean Top-1 delta below -2.00 points (`>= -0.02`).
5. Rank confirmation survivors by absolute mean `val-confirm` Top-1.
   Matched-control deltas are causal gates, not cross-family ranking scores.
6. If two absolute Top-1 values differ by at most 0.10 percentage points
   (`0.001` on the `[0, 1]` scale), choose fewer inference encoders, then lower
   measured FLOPs, then fewer trainable parameters, then higher absolute Top-5,
   then lexicographic config ID.
7. Lock one configuration, common epoch policy, and all hashes.
8. Retrain the selected candidate and strict matched control from scratch on all
   1,654 non-test concepts using the locked schedule. Perform no further
   validation or checkpoint selection.
9. Create a hash-sealed final-run manifest that later runs refuse to overwrite.
10. Run the paired baseline and selected candidate on the formal 10 x 5 grid
    exactly once.

If multiple candidates are intentionally taken to formal test, this departure
must be pre-registered and the analysis must apply an appropriate family-wise
multiple-comparison correction. The default design allows one candidate.

## 6. Implementation layout and interfaces

New work belongs under:

```text
experiments/samga_brain_rw/
  configs/
  samga_brain_rw/
  scripts/
  slurm/
  tests/
```

Heavy/generated outputs use:

```text
artifacts/samga_brain_rw/
logs/samga_brain_rw/
results/samga_brain_rw/
```

The implementation must:

- take dataset, model, cache, artifact, result, and log paths through CLI/config
  rather than hard-coded home paths;
- default to offline model/data loading;
- leave the reference SAMGA repository unchanged;
- support resumable, idempotent runs keyed by subject, seed, stage, and config
  hash;
- emit similarity matrices with ordered query/gallery IDs for fusion;
- emit compact JSON/CSV/Markdown summaries separately from large checkpoints;
- store git SHA, upstream SHA, environment versions, model/cache hashes, seed,
  subject, SLURM job ID, effective batch size, and config hash in every manifest.

## 7. Verification strategy

### Unit and integration tests

- zero-initialized feature adapter is exactly identity;
- adapter and control shapes/parameter counts are correct;
- zero-LoRA and online-frozen outputs match;
- base InternViT weights remain frozen and have no gradients;
- only intended LoRA modules receive gradients;
- checkpoint save/reload is deterministic;
- cached and online sample IDs/order/hashes agree;
- test split cannot be opened by training or model-selection code;
- paired initialization/data-order hashes agree;
- score-fusion calibration uses validation only;
- aggregator rejects missing/duplicate 10 x 5 cells and computes paired
  confidence intervals correctly.

### Execution checks

1. Run syntax, import, unit, and CPU feature-adapter tests locally.
2. Run one-sample and one-batch GPU smoke tests on `debug`.
3. Run one subject-seed end-to-end smoke on `debug` if it fits 30 minutes.
4. Submit longer pilot arrays to `i64m1tga40u` after checking queue state.
5. Expand only candidates that pass validation gates.

All `.out` and `.err` files go under `logs/samga_brain_rw/`.

## 8. Failure handling and stopping rules

- **Missing score matrices/checkpoints:** deterministically re-evaluate locked,
  test-resealed checkpoints; do not retrain to obtain a favorable score.
- **ID/order mismatch:** stop fusion and repair provenance; never align rows by
  position alone.
- **Online/cache mismatch:** stop true-LoRA work until extraction semantics are
  reconciled.
- **OOM:** reduce microbatch, enable/extend checkpointing, and preserve
  effective batch size through accumulation.
- **No validation gain:** record the negative result and stop that family.
- **Stage 1 lacks complementarity:** skip Stage 5.
- **Feature adapter only beats an unmatched control:** do not attribute the
  gain to low rank.
- **Shared LoRA fails:** retain the best validation survivor; do not inspect
  test to choose a fallback.
- **All methods fail:** publish the clean negative result and the verified
  baseline rather than using Hungarian or test-selected checkpoints.

## 9. Deliverables

- reproducible configs and launchers for every attempted stage;
- validation registry with promotion decisions;
- manifests and SLURM job maps;
- compact paired pilot summaries;
- one final 10-subject x 5-seed result table for the locked candidate and
  baseline;
- paired confidence intervals, per-subject breakdown, and compute-cost table;
- English and Chinese experiment documentation updated only after results are
  available.

## 10. Explicitly deferred options

Full InternViT fine-tuning, QLoRA of broad backbone regions, periodic stale-cache
refresh, EEG-encoder replacement, and distillation remain possible follow-up
ablations. They are lower priority because they either weaken attribution,
increase implementation/compute risk, or approximate true joint adaptation.
They require a new reviewed design rather than automatic launch under this
experiment.
