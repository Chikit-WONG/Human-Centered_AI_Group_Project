#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "${EXPERIMENT_ROOT}/../.." && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/hpc2hdd/home/ckwong627/workdir/new_sub_workdir}"
DATA_ROOT="${THINGS_EEG_ROOT:-${WORKSPACE_ROOT}/EEG_Project/EEG_Recon-RL/datasets/things_eeg_data}"
BRAIN_ROOT="${THINGS_EEG_BRAIN_ROOT:-${DATA_ROOT}/Preprocessed_data_250Hz_whiten}"
CLIP_PATH="${CLIP_PATH:-${WORKSPACE_ROOT}/EEG_Project/models/CLIP-ViT-B-32-laion2B-s34B-b79K}"
ARTIFACT_ROOT="${SAMGA_ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts/samga_lora}"
MANIFEST_ROOT="${ARTIFACT_ROOT}/manifests"
CACHE_ROOT="${ARTIFACT_ROOT}/feature_cache"
SMOKE_ROOT="${ARTIFACT_ROOT}/smoke"
PILOT_ROOT="${ARTIFACT_ROOT}/pilot"
FORMAL_ROOT="${ARTIFACT_ROOT}/formal"
LOCKED_CONFIG="${ARTIFACT_ROOT}/pilot_selection.json"
RESULT_ROOT="${PROJECT_ROOT}/results/samga_lora"

# Some third-party conda activation hooks read optional variables. Keep strict
# nounset semantics for our scripts, but do not impose them on those hooks.
set +u
source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate test
set -u
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p "${ARTIFACT_ROOT}" "${PROJECT_ROOT}/logs/samga_lora"
