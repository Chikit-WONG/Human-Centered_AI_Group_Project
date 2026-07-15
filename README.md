# EEG-to-Image Retrieval with Joint Brain–Vision Alignment

English | [简体中文](README_ZH.md)

Course project for **AIAA3800 — Human-Centered Artificial Intelligence**.

This repository studies whether non-invasive EEG recordings can be mapped into a visual-semantic embedding space and used to retrieve the image that a person viewed. The formal standard protocol trains a separate brain encoder and LoRA-adapted CLIP vision encoder for each of the ten THINGS-EEG2 subjects (`sub-01` through `sub-10`). A global one-to-one Hungarian decoder is retained only as a transductive Subject 08 ablation.

> **Scope.** Standard Top-1/Top-5 results cover all ten subjects at seed `42` and are reported per subject and as a ten-subject aggregate. Each subject has an independently trained model. The Hungarian result covers only `sub-08` and is excluded from the aggregate.

## Highlights

- Maps trial-averaged posterior EEG signals to the 512-dimensional CLIP image space.
- Jointly trains a brain MLP and rank-32 LoRA adapters on CLIP ViT-B/32.
- Uses different learning rates for the brain and vision branches (TTUR-style optimization).
- Reports a fixed final-checkpoint result for every subject rather than selecting the best test epoch.
- Trains and evaluates all ten subjects independently, then aggregates standard Top-1/Top-5.
- Provides unit tests, independent checkpoint-reload checks, and per-query predictions for every standard run, plus similarity-matrix provenance for the `sub-08` Hungarian ablation.

## Method

![EEG-to-image retrieval architecture: dual-side CLIP alignment for training and Top-1/Top-5 image retrieval at inference](asserts/Architecture.png)

For EEG query embedding $b_i$ and gallery image embedding $v_j$, both L2-normalized, the retrieval score is

```math
S_{ij} = b_i^\top v_j.
```

Standard retrieval ranks each row independently:

```math
\hat{j}_i = \underset{j}{\arg\,\max}\; S_{ij}.
```

The optional Hungarian decoder instead solves one global bijection:

```math
\hat{\pi} = \underset{\pi \in \mathrm{Perm}(N)}{\arg\,\max}
\sum_{i=1}^{N} S_{i,\pi(i)}.
```

The second protocol can assign an image that is not a query's row-wise maximum because it optimizes the complete one-to-one matching.

## Verified Results

### Ten-subject standard independent retrieval

Each subject has a separately trained model. The official result uses the fixed checkpoint saved after epoch 25 and evaluates 200 held-out queries against 200 unique gallery images per subject. Two fresh save/reload evaluations produced identical metrics and per-query predictions for every subject. The new array trained `sub-01`–`sub-07` and `sub-09`–`sub-10`; `sub-08` reuses the previously completed, repeat-verified run under the identical protocol.

| Subject | Top-1 | Top-5 | Correct@1 | Correct@5 |
|---|---:|---:|---:|---:|
| sub-01 | 86.0% | 96.5% | 172/200 | 193/200 |
| sub-02 | 90.5% | 100.0% | 181/200 | 200/200 |
| sub-03 | 85.0% | 97.0% | 170/200 | 194/200 |
| sub-04 | 83.5% | 97.0% | 167/200 | 194/200 |
| sub-05 | 84.0% | 98.0% | 168/200 | 196/200 |
| sub-06 | 94.0% | 99.5% | 188/200 | 199/200 |
| sub-07 | 86.0% | 98.0% | 172/200 | 196/200 |
| sub-08 | 91.0% | 99.5% | 182/200 | 199/200 |
| sub-09 | 82.5% | 98.0% | 165/200 | 196/200 |
| sub-10 | 91.0% | 99.5% | 182/200 | 199/200 |
| **Ten-subject mean / pooled count** | **87.35%** | **98.30%** | **1747/2000** | **1966/2000** |

The between-subject population standard deviation is 3.74 percentage points for Top-1 and 1.19 points for Top-5. The macro mean equals pooled accuracy because every subject contributes 200 queries. The random 200-way baselines are 0.5% Top-1 and 2.5% Top-5.

### Comparison with prior EEG-to-image retrieval work

The table below uses the closest identifiable protocol in the literature: **intra-subject, 200-way, zero-shot retrieval on THINGS-EEG2**, averaged over all ten subjects. Each EEG query is ranked independently against the 200 held-out stimulus images; the Hungarian result is therefore excluded. Values are the headline results from each paper's main comparison table. The largest value in each metric column is bold, and our results are additionally highlighted in blue.

| Method | Publication status | Top-1 | Top-5 | Protocol note |
|---|---|---:|---:|---|
| [NICE (exact-image reimplementation by Li et al.)](https://proceedings.neurips.cc/paper_files/paper/2024/file/ba5f1233efa77787ff9ec015877dbd1f-Paper-Conference.pdf) | NeurIPS 2024, Table 8 | 21.52% | 51.57% | Validation-selected checkpoint; exact stimulus-image gallery |
| [ATM-S](https://proceedings.neurips.cc/paper_files/paper/2024/file/ba5f1233efa77787ff9ec015877dbd1f-Paper-Conference.pdf) | NeurIPS 2024, Table 8 | 26.13% | 55.32% | Formal proceedings result; 63 EEG channels |
| [UBP](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_Bridging_the_Vision-Brain_Gap_with_an_Uncertainty-Aware_Blur_Prior_CVPR_2025_paper.html) | CVPR 2025 | 50.90% | 79.70% | 17 channels; trial averaging; blurred-image gallery representation |
| [Hierarchical visual embeddings](https://openreview.net/forum?id=IEq71qS8B7) | ICLR 2026 | 75.70% | 94.60% | 17 channels; RN50 + CLIP-B/32 + VAE fusion |
| [EEGiT](https://openaccess.thecvf.com/content/CVPR2026/html/Zhou_EEGiT_Teaching_Vision_Transformers_to_Understand_the_EEG_signal_CVPR_2026_paper.html) | CVPR 2026 | 70.40% | 95.10% | Pretrained ViT transferred to the EEG encoder |
| [Shallow Alignment](https://arxiv.org/abs/2601.21948) | arXiv 2026 preprint | 82.60% | 97.70% | Five-seed mean; best intermediate visual layer |
| [HCF](https://arxiv.org/abs/2603.07077) | arXiv 2026 preprint | 84.60% | 98.20% | Hierarchically fused intermediate visual features |
| [SAMGA](https://arxiv.org/abs/2604.17782) | arXiv 2026 preprint | **91.30%** | **98.80%** | Five-seed mean; 60 epochs with early stopping |
| Our project (standard retrieval) | Course project, fixed protocol | $`\color{blue}{\mathbf{87.35\%}}`$ | $`\color{blue}{\mathbf{98.30\%}}`$ | One seed; 17 channels; fixed epoch-25 checkpoint |

Although our project does not take first place in this selected comparison, **ranking second on both metrics is still a strong result**: Top-1 $`\color{blue}{\mathbf{87.35\%}}`$ and Top-5 $`\color{blue}{\mathbf{98.30\%}}`$. Our project exceeds every peer-reviewed literature row in the table; the only higher row is SAMGA, which is currently a non-peer-reviewed preprint. This is **not** an unqualified state-of-the-art claim: the papers use different visual targets, pretrained encoders, seed counts, and checkpoint-selection rules. HCF and Shallow Alignment are also preprints, whereas our project reports one seed. The hierarchical-visual-embedding paper reports 94.60% in its main table, although the ten displayed per-subject Top-5 values average to approximately 94.91%.

Two further reporting decisions prevent protocol mixing:

- The original NICE-GA paper's 15.6% Top-1 and 42.8% Top-5 EEG results are **not** included. They evaluate 200-way class-template identification using other images from each concept, rather than retrieving the exact stimulus image. The NICE row above is Li et al.'s later exact-image reimplementation under the ATM retrieval protocol.
- The final NeurIPS ATM paper reports 26.13%/55.32% in its formal Table 8. The often-cited 28.64%/58.47% comes from a different arXiv/ablation reporting rule, so it is not mixed into this table.

Although every test concept in this split has exactly one stimulus image, making concept identity and image identity one-to-one at scoring time, the gallery representation still matters. Our project ranks the actual test-image embeddings; it does not replace them with class templates. All literature values above are paper-reported rather than rerun inside this repository.

### Subject 08 Hungarian one-to-one ablation

| Evaluation protocol (`sub-08` only) | Top-1 / assignment accuracy | Top-5 |
|---|---:|---:|
| Standard independent per-query retrieval | **182/200 (91.0%)** | **199/200 (99.5%)** |
| Global Hungarian one-to-one assignment | **200/200 (100.0%)** | N/A |

### Interpreting the Hungarian result

The Hungarian result is a **transductive closed-set ablation**, not a replacement for standard Top-1:

- it jointly observes the full test query batch;
- it assumes that the 200 queries and 200 gallery images form a known bijection;
- every gallery image must be used exactly once;
- a single global assignment returns one image per query, so it has no directly comparable Top-5.

In the `sub-08` run, independent Top-1 predictions covered only 183 unique gallery images. Hungarian decoding changed 18 assignments, converting all 18 standard Top-1 errors to correct matches without changing any correct match to an error. Nine predeclared row/column orderings produced the same mapped assignment, ruling out an aligned-order tie-break explanation for the 100% result.

The recommended reporting convention is therefore:

- **Primary result:** ten-subject standard mean Top-1 87.35% and Top-5 98.30%, accompanied by the per-subject table, pooled counts, and population standard deviations above.
- **Secondary ablation:** `sub-08` global one-to-one assignment accuracy 100.0%, compared with that subject's standard Top-1 91.0% and Top-5 99.5%.

Hungarian assignment is not used in any ten-subject score.

## Experiment Configuration

| Component | Setting |
|---|---|
| Dataset | THINGS-EEG2 |
| Verified subjects / seed | `sub-01`–`sub-10` (trained separately) / `42` |
| Per-subject loaded train EEG tensor | `(16540, 4, 63, 250)` |
| Per-subject loaded test EEG tensor | `(200, 80, 63, 250)` |
| Trial handling | Average 4 train trials and 80 test trials separately |
| EEG channels | `P7,P5,P3,P1,Pz,P2,P4,P6,P8,PO7,PO3,POz,PO4,PO8,O1,Oz,O2` |
| Time window | `[0, 250)` samples |
| Brain encoder | MLP with residual projection blocks |
| Vision encoder | CLIP ViT-B/32 |
| Vision adaptation | LoRA rank 32, all linear layers |
| Embedding dimension | 512 |
| Brain / vision learning rates | `5e-4` / `5e-5` |
| Scheduler / weight decay | Cosine / `0.05` |
| Train / evaluation batch size | 512 / 100 |
| Training | 25 epochs, bf16, gradient checkpointing |
| Evaluation scope | 200 queries × 200-image gallery per subject (2,000 queries total) |
| Formal hardware | One NVIDIA A40 per subject job |

## Repository Layout

```text
.
├── main/
│   ├── data.py                     # THINGS-EEG/image loading and ID matching
│   ├── models_brain.py             # EEG encoder backbones
│   ├── models_clip.py              # Brain–CLIP alignment model
│   └── models_diffusion.py         # Experimental reconstruction components
├── scripts/
│   ├── evaluate_retrieval.py       # Standard and Hungarian evaluation
│   ├── aggregate_subject_metrics.py # Validate/aggregate ten-subject metrics
│   ├── finalize_results.py         # Standard-result validation/reporting
│   ├── finalize_hungarian_results.py
│   ├── run_subject_reproduction.sh # Generic single-subject reproduction
│   ├── run_sub08_reproduction.sh   # Legacy Subject 08-specific wrapper
│   ├── run_hungarian_evaluation.sh # Site-specific Hungarian wrapper
│   ├── submit_subject_array.slurm  # Ten-subject SLURM array launcher
│   └── submit_*.slurm              # Other HKUST(GZ) SLURM launchers
├── tests/
│   └── test_hungarian_assignment.py
├── docs/                            # Internal technical notes
├── train_clip_lora.py               # Main training entry point
├── vanilla.py                       # Experimental reconstruction path
├── enhance.py                       # Experimental retrieval refinement
└── graph.py                         # Experimental graph-based refinement
```

Generated checkpoints, caches, logs, plans, and result artifacts are intentionally excluded by `.gitignore`.

## Environment Setup

Run the commands in this section from the repository root. Each formal subject run used Linux, one NVIDIA A40, and the following fully tested software stack:

| Package | Tested version |
|---|---:|
| Python | 3.10.20 |
| PyTorch | 2.11.0 + CUDA 12.8 |
| TorchVision | 0.26.0 + CUDA 12.8 |
| Transformers | 5.12.1 |
| Datasets | 5.0.0 |
| Accelerate | 1.14.0 |
| PEFT | 0.19.1 |
| Diffusers | 0.38.0 |
| Safetensors | 0.8.0 |
| NumPy | 2.2.6 |
| SciPy | 1.15.3 |
| Pillow | 12.2.0 |
| tqdm | 4.68.3 |
| einops | 0.8.2 |

`diffusers` is part of the core environment because `main/models_clip.py` imports one of its model classes even when only retrieval is run. SciPy is required by the evaluation entry point and provides the Hungarian solver.

### Option A: reuse the verified cluster environment

On the project cluster, `eeg_recon` is the environment used to produce the reported metrics. If Conda is available in the current shell, activate it directly:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate eeg_recon

python --version
which python
```

No additional installation is needed for standard or Hungarian retrieval. The existing `test` environment is **not** recommended for the formal run: its package versions differ from the table above and it currently has a `libstdc++`/`GLIBCXX` import conflict on the cluster.

To preserve `eeg_recon` unchanged while creating a separate working copy:

```bash
conda create --name eeg-retrieval --clone eeg_recon -y
conda activate eeg-retrieval
```

### Option B: create the tested environment from scratch

Create a clean Conda environment, install the matching CUDA 12.8 PyTorch wheels first, and then install the remaining pinned dependencies:

```bash
conda create --name eeg-retrieval python=3.10.20 pip -y
conda activate eeg-retrieval
python -m pip install --upgrade pip

python -m pip install \
  torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install \
  transformers==5.12.1 \
  datasets==5.0.0 \
  accelerate==1.14.0 \
  peft==0.19.1 \
  diffusers==0.38.0 \
  safetensors==0.8.0 \
  numpy==2.2.6 \
  scipy==1.15.3 \
  Pillow==12.2.0 \
  tqdm==4.68.3 \
  einops==0.8.2
```

The CUDA wheel must match the target machine's NVIDIA driver. If CUDA 12.8 is unsuitable, select a compatible PyTorch build from the [official installation guide](https://pytorch.org/get-started/locally/) and keep the remaining package versions pinned. Do not mix independently selected PyTorch and TorchVision builds.

### Verify the installation

Run this import check before submitting a training job:

```bash
python - <<'PY'
import sys

import accelerate
import datasets
import diffusers
import peft
import scipy
import torch
import torchvision
import transformers
from scipy.optimize import linear_sum_assignment

from main.models_clip import BrainCLIPModel

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("TorchVision:", torchvision.__version__)
print("Transformers:", transformers.__version__)
print("Datasets:", datasets.__version__)
print("Accelerate:", accelerate.__version__)
print("PEFT:", peft.__version__)
print("Diffusers:", diffusers.__version__)
print("SciPy:", scipy.__version__)
print("Compiled CUDA:", torch.version.cuda)
print("CUDA visible on this node:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("Core retrieval imports: OK")
PY

python -m unittest discover -s tests -v
```

`CUDA visible on this node: False` is expected on a login node without an allocated GPU. Repeat the check inside the SLURM GPU allocation before training; the formal run should report the assigned A40. The single-GPU reproduction command below uses one `torchrun` process, so running `accelerate launch` with the repository's two-process `accelerate_config.yaml` is not equivalent.

### Optional reconstruction dependencies

The standard and Hungarian retrieval results do not require the experimental reconstruction utilities. To use `vanilla.py`, `enhance.py`, `graph.py`, or the reconstruction metrics, also install:

```bash
python -m pip install \
  scikit-image==0.25.2 \
  clip-anytorch==2.6.0
```

Those paths also require separately downloaded SDXL/IP-Adapter weights. Experiment trackers such as Weights & Biases, SwanLab, or TensorBoard are optional and are only needed when selected through `--report_to`.

## Data and Pretrained Model

The dataset and model weights are not distributed in this repository.

Download THINGS-EEG2 from the [THINGS initiative](https://things-initiative.org/) or its [OSF repository](https://osf.io/3jk45/), then prepare the 250 Hz whitened files expected by the loader:

```text
things_eeg_data/
├── Preprocessed_data_250Hz_whiten/
│   ├── sub-01/
│   │   ├── train.pt
│   │   └── test.pt
│   ├── ...
│   └── sub-10/
│       ├── train.pt
│       └── test.pt
├── training_images/
│   └── **/*.jpg
└── test_images/
    └── **/*.jpg
```

The CLIP model must be available in a local Hugging Face-compatible directory containing its configuration, image processor, and weights, for example:

```text
CLIP-ViT-B-32-laion2B-s34B-b79K/
├── config.json
├── preprocessor_config.json
└── model.safetensors
```

Set portable paths before running:

```bash
export PROJECT_ROOT="$(pwd)"
export THINGS_ROOT="/path/to/things_eeg_data"
export BRAIN_DIR="$THINGS_ROOT/Preprocessed_data_250Hz_whiten"
export CLIP_PATH="/path/to/CLIP-ViT-B-32-laion2B-s34B-b79K"
export SUBJECT_ID=1
printf -v SUBJECT_PADDED '%02d' "$SUBJECT_ID"
export OUTPUT_DIR="$PROJECT_ROOT/runs/all_subjects/seed42/subj${SUBJECT_PADDED}"
export RESULTS_DIR="$PROJECT_ROOT/results/all_subjects/seed42/subj${SUBJECT_PADDED}"
export CHANNELS="P7,P5,P3,P1,Pz,P2,P4,P6,P8,PO7,PO3,POz,PO4,PO8,O1,Oz,O2"

mkdir -p "$OUTPUT_DIR/cache" "$RESULTS_DIR"
```

For a fully offline run:

```bash
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

## Training

The following command reproduces one subject without relying on site-specific wrapper paths. For the formal ten-subject protocol, set `SUBJECT_ID` to each value from 1 through 10 and run a separate training job; do not combine subjects in one model.

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=1 \
  train_clip_lora.py \
  --dataset_name things \
  --brain_directory "$BRAIN_DIR" \
  --image_directory "$THINGS_ROOT" \
  --cache_dir "$OUTPUT_DIR/cache" \
  --subject_ids "$SUBJECT_ID" \
  --eval_subject_ids "$SUBJECT_ID" \
  --brain_column eeg \
  --brain_backbone brain_mlp \
  --dropout 0.1 \
  --pretrained_model_name_or_path "$CLIP_PATH" \
  --lora_rank 32 \
  --lora_layers all-linear \
  --gradient_checkpointing \
  --time_slice 0,250 \
  --avg_trials \
  --selected_channels "$CHANNELS" \
  --learning_rate 5e-4 \
  --vision_learning_rate 5e-5 \
  --lr_scheduler_type cosine \
  --weight_decay 0.05 \
  --seed 42 \
  --dataloader_num_workers 8 \
  --mixed_precision bf16 \
  --output_dir "$OUTPUT_DIR" \
  --metrics_jsonl "$OUTPUT_DIR/validation_metrics.jsonl" \
  --save_total_limit 1 \
  --checkpointing_steps epoch \
  --validation_steps epoch \
  --num_train_epochs 25 \
  --per_device_train_batch_size 512 \
  --per_device_eval_batch_size 100
```

The generic wrapper performs a smoke run or a 25-epoch formal run followed by two fresh checkpoint-reload evaluations. It refuses to overwrite an existing formal run by default:

```bash
bash scripts/run_subject_reproduction.sh smoke --subject-id 1
bash scripts/run_subject_reproduction.sh formal --subject-id 1
```

Run a smoke test before the formal job if using new hardware or a newly prepared dataset.

## Evaluation

Define the common evaluation arguments in Bash:

```bash
EVAL_ARGS=(
  --brain-model-path "$OUTPUT_DIR/brain_model"
  --vision-adapter-path "$OUTPUT_DIR/vision_model"
  --pretrained-model-name-or-path "$CLIP_PATH"
  --brain-directory "$BRAIN_DIR"
  --image-directory "$THINGS_ROOT"
  --dataset-name things
  --subject-id "$SUBJECT_ID"
  --selected-channels "$CHANNELS"
  --time-slice 0,250
  --batch-size 100
  --num-workers 0
  --device cuda
  --dtype bf16
  --cache-dir "$OUTPUT_DIR/cache"
  --seed 42
  --expected-num-samples 200
  --local-files-only
)
```

### Standard independent retrieval

```bash
python scripts/evaluate_retrieval.py \
  "${EVAL_ARGS[@]}" \
  --metrics-output "$RESULTS_DIR/sub${SUBJECT_PADDED}_seed42_formal_metrics.json" \
  --predictions-output "$RESULTS_DIR/sub${SUBJECT_PADDED}_seed42_formal_predictions.csv"
```

### Hungarian one-to-one ablation

This ablation was verified only for `sub-08`. First set `SUBJECT_ID=8`, recompute `SUBJECT_PADDED=08`, point `OUTPUT_DIR` and `RESULTS_DIR` to its run, and then rebuild the complete `EVAL_ARGS=(...)` array above so Bash captures the updated values. The evaluator still writes the standard per-query metrics while adding a separate constrained-assignment namespace and CSV:

```bash
python scripts/evaluate_retrieval.py \
  "${EVAL_ARGS[@]}" \
  --enable-hungarian \
  --metrics-output "$RESULTS_DIR/sub08_hungarian_metrics.json" \
  --predictions-output "$RESULTS_DIR/sub08_hungarian_standard_predictions.csv" \
  --hungarian-output "$RESULTS_DIR/sub08_hungarian_assignment.csv" \
  --similarity-output "$RESULTS_DIR/sub08_cosine_similarity.npz"
```

Do not label `assignment_accuracy` as standard Top-1, and do not invent a Hungarian Top-5 for a single global assignment.

## Tests

```bash
python -m unittest -v tests/test_hungarian_assignment.py
```

The tests cover solver optimality, collision resolution, invalid matrices, non-diagonal ID mappings, and row/column permutation invariance for a unique optimum.

## SLURM Wrappers

The repository includes launchers used on the HKUST(GZ) cluster:

```bash
# All ten standard subject runs; the array permits at most two concurrent jobs.
sbatch scripts/submit_subject_array.slurm smoke
sbatch scripts/submit_subject_array.slurm formal

# Subject 08-specific legacy/Hungarian launchers.
sbatch scripts/submit_sub08.slurm formal
sbatch scripts/submit_hungarian_eval.slurm
```

After all ten standard runs finish, validate and aggregate them:

```bash
python scripts/aggregate_subject_metrics.py \
  --results-root "$PROJECT_ROOT/results/all_subjects/seed42" \
  --subjects 1-10 \
  --seed 42 \
  --expected-epochs 25
```

The aggregator checks query/gallery cardinality, all 25 validation records, metric/count agreement, repeat-reload prediction identity, retrieval protocol, saved model configurations, CLIP base path, and key environment versions. It writes `summary.json`, `per_subject_metrics.csv`, `RESULTS_EN.md`, and `RESULTS_ZH.md` under `results/all_subjects/seed42/`.

These shell and SLURM files currently contain site-specific absolute paths. Before using them in another clone or cluster, update:

- `PROJECT_ROOT`, `THINGS_ROOT`, `BRAIN_DIR`, and `CLIP_PATH`;
- `#SBATCH --chdir`, `--output`, and `--error`;
- the Conda activation path and environment name;
- partition, GPU, CPU, memory, and time requests.

The direct training and evaluation commands above are the portable reference commands.

## Reproducibility Policy

- The official metric is evaluated from the fixed final checkpoint after epoch 25.
- Test-set peak epochs are diagnostic only and are not used for checkpoint selection.
- The ten subjects use distinct models; there is no cross-subject pooling or joint training.
- Query and gallery identity are matched by unique image ID rather than by assuming a diagonal target.
- Every standard evaluation is repeated after an independent model reload.
- The ten-subject aggregate contains only standard independent Top-1/Top-5 scores.
- The `sub-08` Hungarian evaluation saves the full similarity matrix, ID ordering, hashes, transition ledger, and assignment output.
- Ground-truth labels are used only after the assignment is solved; they are not part of the Hungarian objective.
- Multiple predeclared row/column orderings are audited so aligned input order cannot silently determine an exact tie.

## Limitations and Responsible Use

- Results cover all ten THINGS-EEG2 subjects but only one random seed, so across-seed uncertainty is not measured.
- Each subject uses an independent model; this experiment does not evaluate cross-subject generalization.
- The reproduced training loop invokes gradient clipping before `backward()`, so the configured maximum gradient norm does not affect these runs. All ten subjects use this same behavior; correcting the order would define a different protocol and require a full rerun.
- Trial averaging uses repeated presentations and is not equivalent to single-trial decoding.
- The Hungarian ablation was verified only for `sub-08`; it must not be extrapolated to all ten subjects. It also requires the complete query batch and a known one-to-one gallery prior, so it is not an online single-query retrieval protocol.
- Dataset, preprocessing, and model-weight versions can materially affect the result.
- The reconstruction path is experimental; no formal reconstruction metric is claimed in this README.
- EEG is sensitive human-participant data. Follow the dataset's consent, privacy, licensing, and redistribution requirements, and do not interpret this research system as a clinical or diagnostic tool.

## References

- Gifford, A. T., Dwivedi, K., Roig, G., & Cichy, R. M. (2022). [A large and rich EEG dataset for modeling human visual object recognition](https://doi.org/10.1016/j.neuroimage.2022.119754). *NeuroImage, 264*, 119754.
- Song, Y., Liu, B., Li, X., Shi, N., Wang, Y., & Gao, X. (2024). [Decoding Natural Images from EEG for Object Recognition](https://openreview.net/forum?id=dhLIno8FmH). *ICLR 2024*. Code: [NICE-EEG](https://github.com/eeyhsong/NICE-EEG).
- Li, D., Wei, C., Li, S., Zou, J., & Liu, Q. (2024). [Visual Decoding and Reconstruction via EEG Embeddings with Guided Diffusion](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ba5f1233efa77787ff9ec015877dbd1f-Abstract-Conference.html). *NeurIPS 2024*. Code: [EEG Image Decode](https://github.com/dongyangli-del/EEG_Image_decode).
- Wu, H., Li, Q., Zhang, C., He, Z., & Ying, X. (2025). [Bridging the Vision-Brain Gap with an Uncertainty-Aware Blur Prior](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_Bridging_the_Vision-Brain_Gap_with_an_Uncertainty-Aware_Blur_Prior_CVPR_2025_paper.html). *CVPR 2025*.
- Zheng, J., Jia, H., Li, M., Zheng, Y., Zeng, Y., Gao, Y., & Liang, C. (2026). [Learning Brain Representation with Hierarchical Visual Embeddings](https://openreview.net/forum?id=IEq71qS8B7). *ICLR 2026*.
- Zhou, J., Xu, C., Wang, W., Yang, E., & Deng, C. (2026). [EEGiT: Teaching Vision Transformers to Understand the EEG Signal](https://openaccess.thecvf.com/content/CVPR2026/html/Zhou_EEGiT_Teaching_Vision_Transformers_to_Understand_the_EEG_signal_CVPR_2026_paper.html). *CVPR 2026*.
- Du, Y., Dai, S., Song, Y., Thompson, P. M., Tang, H., & Zhan, L. (2026). [Deep Models, Shallow Alignment: Uncovering the Granularity Mismatch in Neural Decoding](https://arxiv.org/abs/2601.21948). *arXiv preprint*.
- Tang, J., Jiang, S., Su, F., & Zhao, Z. (2026). [Aligning What EEG Can See: Structural Representations for Brain-Vision Matching](https://arxiv.org/abs/2603.07077). *arXiv preprint*.
- Jiang, L., She, Q., Xu, J., Xu, H., Wu, D., & Kuang, Z. (2026). [Subject-Aware Multi-Granularity Alignment for Zero-Shot EEG-to-Image Retrieval](https://arxiv.org/abs/2604.17782). *arXiv preprint*.

## License

No open-source license has been declared for this course-project repository. Add an explicit license before redistributing the code or accepting external contributions.
