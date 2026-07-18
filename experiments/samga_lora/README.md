# SAMGA + Visual LoRA

This directory contains an independent, leakage-controlled implementation of the
SAMGA training protocol plus the project's visual LoRA/TTUR intervention. The
upstream SAMGA checkout is treated as a read-only behavioral reference because
it does not currently include a license and its released training loop selects
epochs on the test set.

The confirmatory comparison changes exactly one factor: a frozen CLIP ViT-B/32
teacher versus rank-32 LoRA on that same teacher. Both arms use the same EEGProject,
17 posterior channels, trial averaging, five visual layers, subject-aware router,
linear projections, two-stage objective, train/validation rows, batch size, seeds,
and standard independent retrieval evaluator. Hungarian assignment is excluded.

## Confirmatory result

The complete ten-subject x five-seed paired grid passed the pre-registered
criterion:

| Arm | Top-1 | Top-5 |
|---|---:|---:|
| Frozen CLIP SAMGA | 76.17% +/- 0.30 | 95.79% +/- 0.32 |
| SAMGA + visual LoRA/TTUR | **82.68% +/- 0.36** | **97.67% +/- 0.17** |
| Paired change | **+6.51 points**, 95% CI **[+5.22, +7.69]** | **+1.88 points**, 95% CI **[+1.30, +2.50]** |

The arm-level uncertainty is sample SD across the five seed-level ten-subject
macro scores. Paired intervals use 10,000 two-way subject/seed cluster-bootstrap
resamples. Top-1 improved in 48/50 paired cells and every subject's mean change
was positive. All 100 formal cells exited successfully, retained 200-row
predictions and epoch-25 checkpoints, and passed strict re-derivation and source
provenance checks.

This controlled CLIP result identifies the LoRA/TTUR intervention effect; it is
not the SAMGA paper's InternViT 91.3%/98.8% result. The separately labelled
`exploratory_internvit/` path tests a plausible pinned InternViT model without
claiming an exact paper reproduction.

## Exploratory inferred-InternViT result

The optional follow-up used frozen
`OpenGVLab/InternViT-6B-448px-V1-5` features at pinned revision
`03e138c81d3fd538c77439fd43a42c067d827427`, hidden-state indices
20/24/28/32/36, seed 2025, and fixed epoch 60. All ten subject runs passed
checkpoint reload and independent prediction re-derivation:

| Run | Top-1 | Top-5 |
|---|---:|---:|
| Inferred InternViT frozen SAMGA | **83.05%** (1661/2000) | **98.00%** (1960/2000) |

This one-seed diagnostic uses no Hungarian decoding or test-set checkpoint
selection. It is not an exact reproduction of the paper's five-seed
91.3%/98.8% because the exact checkpoint, extractor, and hidden-state semantics
are not released. See `exploratory_internvit/README.md` for the reproducible,
strictly separated path.

## Layout

- `samga_lora/`: data, model, loss, provenance, and retrieval utilities.
- `scripts/`: manifest/cache preparation, gated runners, selection, and aggregation.
- `slurm/`: W1 smoke, W2 pilot, and gated W3 formal arrays.
- `configs/protocol.json`: machine-readable locked protocol.
- Runtime artifacts are written to `artifacts/samga_lora`, logs to
  `logs/samga_lora`, and final summaries to `results/samga_lora`; all are ignored
  by Git.

## Execution order

From the repository root, activate the `test` conda environment and build the ten
subject manifests. Extract the shared frozen train/test caches with
`slurm/cache_array.slurm`, then run `slurm/smoke_array.slurm`. Only after both smoke
cells and cache parity pass should `slurm/pilot_array.slurm` be submitted. Run
`scripts/select_pilot.py` to lock one visual learning-rate ratio and one epoch.
The formal array refuses to read the test set unless `pilot_selection.json` records
a passed gate. The cluster's normal QOS allows ten submitted jobs per user, so
submit the formal grid as the sequential chunks `0`, `10`, ..., `90`:

```bash
bash experiments/samga_lora/scripts/submit_formal_chunk.sh 0
# Wait for that chunk to finish, then repeat with 10, 20, ..., 90.
```

The optional second argument selects an A40 partition when the shared queue is
blocked by reservations, for example `... submit_formal_chunk.sh 0
i64m1tga40ue`. The default remains the low-priority shared partition.

The full rationale and thresholds are in the paired
[English protocol plan](../../docs/protocols/SAMGA_LORA_EXPERIMENT_PLAN_EN.md)
and [中文协议计划](../../docs/protocols/SAMGA_LORA_EXPERIMENT_PLAN_ZH.md).

## Environment

The cluster jobs activate the existing `test` environment (Python 3.10.18,
PyTorch 2.10.0+cu126, Transformers 4.57.6, and PEFT 0.18.1; the optional
InternViT extractor also uses Timm 1.0.26). The exact Python
package pins are listed in `requirements.txt`; the PyTorch build itself must be
chosen for the target CUDA driver. Runtime code writes JSON/JSONL directly and
does not require TensorBoard. Repository tests can be run with any compatible
environment that also provides `pytest`:

```bash
PYTHONPATH=experiments/samga_lora python -m pytest -q experiments/samga_lora/tests
```
