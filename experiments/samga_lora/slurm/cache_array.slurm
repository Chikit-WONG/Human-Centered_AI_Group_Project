#!/usr/bin/env bash
#SBATCH --job-name=samga-cache
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a40:1
#SBATCH --array=0-1
#SBATCH --output=logs/samga_lora/%x-%A_%a.out
#SBATCH --error=logs/samga_lora/%x-%A_%a.err

set -euo pipefail
PROJECT_ROOT="${SAMGA_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?submit from the project root}}"
cd "${PROJECT_ROOT}"
SCRIPT_DIR="${PROJECT_ROOT}/experiments/samga_lora/scripts"
source "${SCRIPT_DIR}/common.sh"

if [[ "${SLURM_ARRAY_TASK_ID}" == "0" ]]; then
  SPLIT=train
else
  SPLIT=test
fi
python "${EXPERIMENT_ROOT}/scripts/extract_clip_features.py" \
  --manifest "${MANIFEST_ROOT}/sub-01_${SPLIT}.json" \
  --clip-path "${CLIP_PATH}" \
  --output "${CACHE_ROOT}/clip_layers_4_6_8_10_12_${SPLIT}.npy" \
  --batch-size 256 \
  --num-workers 8
