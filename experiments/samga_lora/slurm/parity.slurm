#!/usr/bin/env bash
#SBATCH --job-name=samga-parity
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a40:1
#SBATCH --output=logs/samga_lora/%x-%j.out
#SBATCH --error=logs/samga_lora/%x-%j.err

set -euo pipefail
PROJECT_ROOT="${SAMGA_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?submit from the project root}}"
cd "${PROJECT_ROOT}"
SCRIPT_DIR="${PROJECT_ROOT}/experiments/samga_lora/scripts"
source "${SCRIPT_DIR}/common.sh"

python "${EXPERIMENT_ROOT}/scripts/check_cache_parity.py" \
  --manifest "${MANIFEST_ROOT}/sub-01_train.json" \
  --cache "${CACHE_ROOT}/clip_layers_4_6_8_10_12_train.npy" \
  --clip-path "${CLIP_PATH}" \
  --output "${ARTIFACT_ROOT}/cache_parity.json" \
  --check-lora-init
