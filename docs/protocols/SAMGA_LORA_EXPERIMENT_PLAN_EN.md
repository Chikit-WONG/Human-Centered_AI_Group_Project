# SAMGA x LoRA/TTUR Experiment Plan

English | [简体中文](SAMGA_LORA_EXPERIMENT_PLAN_ZH.md)

## Primary question

Under the same CLIP ViT-B/32 visual backbone, does adding rank-32 visual LoRA
with a smaller vision learning rate improve SAMGA over an otherwise identical
frozen-vision control on intra-subject THINGS-EEG2 retrieval?

## Locked protocol

- Standard independent 200-way Top-1/Top-5; no Hungarian decoding.
- Seventeen posterior channels and trial averaging.
- CLIP blocks 4, 6, 8, 10, and 12; CLS token followed by the frozen CLIP
  post-layer normalization.
- SAMGA EEGProject, subject-aware router, linear projectors, shared encoder,
  stage-1 MMD plus symmetric contrastive loss, and stage-2 contrastive loss.
- Match the released SAMGA launcher by applying L2 normalization to image
  features, but not EEG features, inside the contrastive training objective;
  evaluation still uses cosine similarity.
- Rank/alpha 32 LoRA on q/k/v/out projections and both MLP linear layers.
- Seeds 42--46. The test set remains sealed until the pilot has selected one
  vision-learning-rate ratio and one shared stopping epoch from a concept-
  disjoint validation set.

## Execution gates

1. Validate manifests, feature-cache parity, gradients, checkpoint reload, and
   a debug smoke run.
2. Pilot subjects 01, 05, and 08 at seeds 42 and 43. Compare vision LR ratios
   0.05, 0.10, and 0.20 against a shared frozen control.
3. Expand only if the best validation configuration improves mean Top-1 by at
   least 0.5 percentage points, is positive in at least four of six paired
   cells, and has no subject degradation worse than 2 points.
4. Retrain all ten subjects at seeds 42--46 on all training concepts, evaluate
   the held-out test set once, and aggregate paired differences.
5. If the CLIP experiment succeeds, run a clearly labelled exploratory frozen
   reproduction with inferred InternViT-6B-448px-V1-5 features.

## Success criterion

The confirmatory mean paired Top-1 gain must be at least 0.5 percentage points
with a two-way subject/seed cluster-bootstrap 95% confidence interval whose
lower bound is above zero. Top-5 must not decrease by more than 0.2 points.

## Execution record (2026-07-16)

- Preflight, cache/online parity, gradient, reload, and smoke gates passed.
- The six-pair pilot passed and locked vision LR ratio 0.20 and epoch 25.
- All 100 formal cells completed successfully with empty stderr: 50 Frozen and
  50 LoRA cells over the complete ten-subject x five-seed grid.
- Frozen: Top-1 76.17% +/- 0.30, Top-5 95.79% +/- 0.32.
- LoRA/TTUR: Top-1 82.68% +/- 0.36, Top-5 97.67% +/- 0.17.
- Paired change: Top-1 +6.51 points, two-way bootstrap 95% CI
  [+5.22, +7.69]; Top-5 +1.88 points, 95% CI [+1.30, +2.50].
- The pre-registered confirmatory success criterion passed. The inferred-
  InternViT frozen follow-up was therefore opened as a separate exploratory run.
- The exploratory follow-up completed for all ten subjects at seed 2025 and
  fixed epoch 60: Top-1 83.05% (1661/2000), Top-5 98.00% (1960/2000).
  Every checkpoint reload and 200-row prediction audit passed. Because the
  paper does not disclose the exact checkpoint/extractor/layer semantics and
  reports five seeds, this value remains an inferred-model one-seed diagnostic,
  not an exact reproduction of the paper's 91.3%/98.8%.
