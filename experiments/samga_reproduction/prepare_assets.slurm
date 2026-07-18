#!/usr/bin/env bash
#SBATCH --job-name=samga-assets
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/samga_reproduction/samga-assets-%j.out
#SBATCH --error=logs/samga_reproduction/samga-assets-%j.err

set -euo pipefail
PROJECT_ROOT="${SAMGA_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?submit from the project root}}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/hpc2hdd/home/ckwong627/workdir/new_sub_workdir}"
REPRO_ROOT="${PROJECT_ROOT}/artifacts/samga_reproduction"
SCRIPT="${PROJECT_ROOT}/experiments/samga_reproduction/prepare_official_assets.py"
SOURCE_EEG="${WORKSPACE_ROOT}/EEG_Project/EEG_Recon-RL/datasets/things_eeg_data/Preprocessed_data_250Hz_whiten"
SOURCE_FEATURES="${PROJECT_ROOT}/artifacts/samga_lora/internvit/feature_cache"

set +u
source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate test
set -u
export PYTHONUNBUFFERED=1

cd "${PROJECT_ROOT}"
python "${SCRIPT}" eeg \
  --source-root "${SOURCE_EEG}" \
  --output-root "${REPRO_ROOT}/data/preprocessed_eeg" \
  --subjects "${SAMGA_PREP_SUBJECTS:-8}"

python "${SCRIPT}" features \
  --train-cache "${SOURCE_FEATURES}/internvit_layers_20_24_28_32_36_train.npy" \
  --test-cache "${SOURCE_FEATURES}/internvit_layers_20_24_28_32_36_test.npy" \
  --output-dir "${REPRO_ROOT}/features/v1_5_idx0_cls_raw" \
  --normalization none
