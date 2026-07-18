# Audited SAMGA public-code reproduction attempt

English | [简体中文](README_ZH.md)

This experiment is a best-effort numerical reproduction of SAMGA's
THINGS-EEG2 intra-subject retrieval result. It runs the released training code
without editing the SAMGA source tree, but it is not a paper-exact
reproduction: the paper and repository do not identify the visual checkpoint,
release the feature extractor, define the layer/token semantics, or disclose
the five random-seed values. The reproduced result therefore remains an
audited, explicitly assumption-dependent result.

This experiment is separate from
[`experiments/samga_lora`](../samga_lora/README.md), which is a
leakage-controlled Frozen-versus-LoRA attribution study using CLIP ViT-B/32.
It also does not replace the course project's original CLIP/LoRA headline
result.

## Scope and public-material ambiguities

The local paper is arXiv v1 and the official source is pinned at clean commit
`1a63745b7ff6f98dad34b0f0b8246a9b5260d9c1`. As of 2026-07-18, the repository
has no release, checkpoint, precomputed feature download, feature-extraction
script, or published five-seed list. The open
[feature-version question](https://github.com/LinJiang8/SAMGA/issues/1) has no
author answer.

Several paper statements differ from the public launcher:

| Item | Paper | Released `intra.sh` / code default used by the launcher |
|---|---|---|
| Batch size | 1024 | 512 |
| Seeds | Mean over five undisclosed seeds | Seed 2025 only |
| Retrieval temperature | Learnable | Fixed unless `--t_learnable` is added |
| Layer dropout | Applied; probability undisclosed | `0.0` |
| Training similarity | Cosine | Image-side L2 only during training |
| Early stopping | Applied; rule undisclosed | Patience 10, driven by formal-test Top-1 |

The PDF itself never names InternViT. The repository directory convention
implies five 3,200-dimensional layer features at layers 20/24/28/32/36, but
does not resolve the model version, checkpoint, hidden-state offset, token
pooling, pre/post-LayerNorm representation, or image processor. These gaps are
why the protocol below is labelled inferred rather than author-confirmed.

## Pinned source, model, data, and feature protocol

Downloaded model assets are confined to the user-requested model root:

```text
/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/EEG_Project/models
```

The selected model is:

```text
OpenGVLab/InternViT-6B-448px-V2_5
revision 9d1a4344077479c93d42584b6941c64d795d508d
/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/EEG_Project/models/InternViT-6B-448px-V2_5/9d1a4344077479c93d42584b6941c64d795d508d
```

`model_provenance.json` has SHA-256
`56f4218ae3f636e32521a719808767313c093a3d85a78053bfb1404d1463342c`;
the three official weight shards have SHA-256 values
`9818659d13d932da8bc0c3b8ee15f5b5d68d8c94d66eb525be566066630111da`,
`4f0c10e72d6f6513f421baa6ec843d5508657435059c1d18b6b5fd7789f9d5b7`, and
`d21c4fe0bc4af1425cfae1a59a8f5fbb00fde9d8e2888325a60913ac61b0494d`.
See [DOWNLOADER_SAFETY.md](DOWNLOADER_SAFETY.md)
for the fail-closed download design and
[V2_5_FEATURE_PIPELINE.md](V2_5_FEATURE_PIPELINE.md) for extraction and
verification commands.

The selected feature assumption is:

- resize and center-crop to 448 pixels using the pinned processor;
- ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation
  `[0.229, 0.224, 0.225]`;
- actual block outputs 20, 24, 28, 32, and 36;
- mean of patch tokens, excluding CLS;
- no additional per-vector LayerNorm or L2 normalization before SAMGA;
- float16 cache on disk, loaded as float32 by the training code.

The resulting feature provenance SHA-256 is
`d12c29387738cdd76fedd547221e33ada2db2fe12c123be7ef904e3f58732fb1`.
The feature choice was screened first on Subject 08 and then checked on
Subjects 01 and 05, so even fixed-epoch results are exploratory rather than a
prospectively locked replication.

No dataset was downloaded for this run. Existing MVNN-whitened THINGS-EEG2
files were reused from:

```text
/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/EEG_Project/EEG_Recon-RL/datasets/things_eeg_data/Preprocessed_data_250Hz_whiten
```

They were converted to SAMGA's layout without re-normalization. Source `.pt`
train/test tensors have shapes `[16540,4,63,250]` and `[200,80,63,250]`.
After conversion, the SAMGA `.npy` tensors have shapes
`[1654,10,4,63,250]` and `[200,1,80,63,250]`. Source and converted storage
are float16; the released loader converts arrays to float32. The generated
data provenance SHA-256 is
`b76b3db21010d4a55da7855e2e6cd3a00c4439c8855b645d09d02bfbd5e46463`.

Training used the existing `test` environment: Python 3.10.18, PyTorch
2.10.0+cu126, TorchVision 0.25.0+cu126, Transformers 4.57.6, Timm 1.0.26,
NumPy 1.26.4, SciPy 1.15.3, and scikit-learn 1.7.2.

## Metric definitions and checkpoint-selection guardrails

All results use standard independent 200-way cosine retrieval. Hungarian
assignment is not used.

- **Actual final/stopping epoch:** `top1 acc` and `top5 acc` in the released
  CSV. They are fixed epoch-60 metrics only when all 60 epochs completed.
- **Per-epoch test-selected diagnostic:** the released code evaluates the
  formal test set every epoch and selects the epoch with the highest test
  Top-1, breaking exact Top-1 ties by lower test loss. Its Top-5 is the
  companion value at that Top-1-selected epoch, not independently maximized.
- **Patience-10 endpoint:** the released early-stopping rule is also driven by
  formal-test Top-1. The stopping-epoch metric therefore has test leakage too;
  it is not validation-selected.

The paper's 91.30% Top-1 and 98.80% Top-5 are reference values, not ground
truth for an incompletely specified public protocol.

## Verified results

For seed 2025, batch 512, 60 epochs, and early stopping disabled, all ten
subjects completed and passed CSV-versus-log and configuration audits:

| Seed-2025 protocol | Top-1 | Top-5 | Top-1 gap | Top-5 gap |
|---|---:|---:|---:|---:|
| Epoch 60, no early stopping | 89.55% | 98.65% | -1.75 points | -0.15 points |
| Test-set-selected epoch diagnostic | 91.95% | 98.95% | +0.65 points | +0.15 points |

The first row has cross-subject sample SDs of 3.90 and 1.36 points; the second
has cross-subject sample SDs of 4.04 and 1.40 points. The selected epoch is
40.0 ± 11.96 (mean ± cross-subject sample SD). The second row must never be
presented as a leakage-free final-project estimate.

### Released-launcher-compatible seed-2025 confirmation

The batch-512, patience-10 confirmation completed all ten subjects:

| Metric interpretation | Top-1 mean ± cross-subject SD | Top-5 mean ± cross-subject SD | Top-1 gap | Top-5 gap |
|---|---:|---:|---:|---:|
| Actual stopping/final epoch | 88.95% ± 4.78 | 98.90% ± 1.26 | -2.35 points | +0.10 points |
| Test-set-selected diagnostic | 91.50% ± 4.00 | 98.75% ± 1.46 | +0.20 points | -0.05 points |

Actual stopping epochs are 34, 31, 40, 30, 33, 47, 60, 30, 32, and 37 for Subjects 01–10. Their mean is 37.4 ± 9.55, with range 30–60. The stopping rule monitors formal-test Top-1, so the endpoint row is test-conditioned; the selected row adds direct best-epoch selection on the same test set.

### Project-defined five-seed stability grid

Seeds 42–46 are project-defined because the paper does not disclose its five seed values. All 50 cells completed 60 epochs with early stopping disabled:

| Seed | Epoch-60 Top-1 | Epoch-60 Top-5 | Test-selected Top-1 | Companion Top-5 |
|---:|---:|---:|---:|---:|
| 42 | 88.75% | 98.80% | 91.85% | 98.70% |
| 43 | 89.10% | 98.95% | 91.70% | 98.75% |
| 44 | 89.40% | 98.85% | 91.65% | 99.10% |
| 45 | 88.55% | 98.90% | 91.75% | 98.95% |
| 46 | 89.30% | 98.85% | 92.15% | 98.85% |
| Mean ± seed-level SD | **89.02% ± 0.36** | **98.87% ± 0.06** | **91.82% ± 0.20** | **98.87% ± 0.16** |

Each seed is first macro-averaged over ten subjects; the displayed SD is the sample SD across those five seed-level means. It is not the pooled SD across 50 subject–seed cells. Relative to the paper headline, the epoch-60 gaps are -2.28/+0.07 points and the test-selected gaps are +0.52/+0.07 points for Top-1/Top-5.

Detailed local reports are generated under:

```text
results/samga_reproduction
```

That directory is intentionally ignored by Git; the core results and
guardrails are therefore inlined here for remote GitHub readers.

## Reproduction workflow

Run commands from the project root. First follow
[DOWNLOADER_SAFETY.md](DOWNLOADER_SAFETY.md), then extract and verify the
pinned features using [V2_5_FEATURE_PIPELINE.md](V2_5_FEATURE_PIPELINE.md).
Prepare the existing EEG assets and convert the selected train/test feature
caches into the five-layer filenames expected by the released SAMGA loader:

```bash
source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate test
python experiments/samga_reproduction/prepare_official_assets.py eeg \
  --source-root /hpc2hdd/home/ckwong627/workdir/new_sub_workdir/EEG_Project/EEG_Recon-RL/datasets/things_eeg_data/Preprocessed_data_250Hz_whiten \
  --output-root "$PWD/artifacts/samga_reproduction/data/preprocessed_eeg" \
  --subjects 1 2 3 4 5 6 7 8 9 10

python experiments/samga_reproduction/prepare_official_assets.py features \
  --train-cache "$PWD/artifacts/samga_reproduction/features/internvit_v2_5_multi_variant/variants/train_idx0_patch_mean/features.npy" \
  --test-cache "$PWD/artifacts/samga_reproduction/features/internvit_v2_5_multi_variant/variants/test_idx0_patch_mean/features.npy" \
  --output-dir "$PWD/artifacts/samga_reproduction/features/v2_5_idx0_patch_mean_raw" \
  --normalization none
```

Run one fixed-60 subject:

```bash
SAMGA_VARIANT=v2_5_idx0_patch_mean_raw \
SAMGA_FEATURE_ROOT="$PWD/artifacts/samga_reproduction/features/v2_5_idx0_patch_mean_raw" \
SAMGA_SEED=2025 \
SAMGA_BATCH_SIZE=512 \
SAMGA_EARLY_STOP_PATIENCE=0 \
SAMGA_NUM_EPOCHS=60 \
bash experiments/samga_reproduction/run_official_cell.sh 8
```

Use `official_array.slurm` for Subjects 01–10. Create the log directory before
`sbatch`, because SLURM opens output files before the script starts. Export
every main-protocol value explicitly:

```bash
mkdir -p logs/samga_reproduction
sbatch --array=0-9%2 \
  --export=ALL,SAMGA_VARIANT=v2_5_idx0_patch_mean_raw,SAMGA_FEATURE_ROOT="$PWD/artifacts/samga_reproduction/features/v2_5_idx0_patch_mean_raw",SAMGA_SEED=2025,SAMGA_BATCH_SIZE=512,SAMGA_EARLY_STOP_PATIENCE=0,SAMGA_NUM_EPOCHS=60 \
  experiments/samga_reproduction/official_array.slurm
```

For the released-launcher early-stopping confirmation, use the distinct
variant `v2_5_idx0_patch_mean_raw_launcher_p10`, point it at the same feature
root, and set `SAMGA_EARLY_STOP_PATIENCE=10`. The runner refuses malformed
subjects, seeds, variants, switch values, and missing assets before training.

Create an isolated result snapshot before aggregation. The aggregator rejects
duplicates, symlinks, malformed CSVs, unexpected cells, and incomplete
expected matrices:

```bash
python experiments/samga_reproduction/aggregate_official_results.py \
  --input-root /absolute/path/to/clean/source_cells \
  --output-dir /absolute/path/to/clean/aggregate \
  --expected-subjects 1-10 \
  --expected-seeds 2025
```

The released SAMGA source tree must remain clean at commit `1a63745...`.
Generated EEG layouts, features, checkpoints, and run outputs live under
`artifacts/samga_reproduction`; curated local reports live under
`results/samga_reproduction`; downloaded weights live only under
`EEG_Project/models`.

## Sensitivity-only switches

The three switches below are off by default. The reported main protocol also
sets its variant, feature root, batch 512, patience, seed, and 60-epoch limit
explicitly; the runner's generic batch default is not the reported protocol.
These opt-in switches expose ablations already supported by the released
`train.py`:

| Environment variable | Default | Accepted values | Appended CLI when enabled |
|---|---:|---|---|
| `SAMGA_EEG_L2NORM` | `0` | `0` or `1` | `--eeg_l2norm` |
| `SAMGA_T_LEARNABLE` | `0` | `0` or `1` | `--t_learnable` |
| `SAMGA_ROUTER_LAYER_DROPOUT` | `0` | finite number in `[0, 1)` | `--router_layer_dropout VALUE` when nonzero |

Use a distinct `SAMGA_VARIANT` for every protocol. For example:

```bash
SAMGA_VARIANT=v2_5-patch-eegl2-tlearn-rdrop025 \
SAMGA_FEATURE_ROOT=/absolute/path/to/v2_5-patch-features \
SAMGA_EEG_L2NORM=1 \
SAMGA_T_LEARNABLE=1 \
SAMGA_ROUTER_LAYER_DROPOUT=0.25 \
bash experiments/samga_reproduction/run_official_cell.sh 8
```

## Limitations

- The model version and feature semantics are inferred because the authors did
  not publish them. A numerically close result does not prove configuration
  identity.
- V2.5 patch-mean semantics were selected after a limited subject screen.
- Seed 2025 is the only released-launcher seed. Any 42–46 grid is
  project-defined because the paper's five seed values are undisclosed.
- Both per-epoch best selection and the public patience-10 early-stopping rule
  inspect the formal test set.
- The paper and launcher disagree on several material hyperparameters.
- GPU retraining is seeded but not guaranteed bitwise deterministic.
- Trial-averaged decoding is not single-trial decoding.
