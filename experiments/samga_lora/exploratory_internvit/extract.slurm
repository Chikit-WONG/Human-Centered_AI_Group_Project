#!/usr/bin/env bash
#SBATCH --job-name=samga-internvit-cache
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --array=0-1%2
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/samga_lora/samga-internvit-cache-%A_%a.out
#SBATCH --error=logs/samga_lora/samga-internvit-cache-%A_%a.err

set -euo pipefail
PROJECT_ROOT="${SAMGA_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?submit from the project root}}"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/experiments/samga_lora/scripts/common.sh"

MODEL_PATH="${INTERNVIT_MODEL_PATH:-${ARTIFACT_ROOT}/internvit/model-03e138c81d3f}"
FEATURE_ROOT="${ARTIFACT_ROOT}/internvit/feature_cache"
mkdir -p "${FEATURE_ROOT}"
if (( SLURM_ARRAY_TASK_ID == 0 )); then
  SPLIT=train
else
  SPLIT=test
fi
if [[ -n "${INTERNVIT_MAX_ROWS:-}" ]]; then
  OUTPUT="${FEATURE_ROOT}/smoke_${INTERNVIT_MAX_ROWS}_internvit_layers_20_24_28_32_36_${SPLIT}.npy"
else
  OUTPUT="${FEATURE_ROOT}/internvit_layers_20_24_28_32_36_${SPLIT}.npy"
fi
ARGS=(
  --manifest "${MANIFEST_ROOT}/sub-01_${SPLIT}.json"
  --model-path "${MODEL_PATH}"
  --output "${OUTPUT}"
  --layer-ids 20 24 28 32 36
  --batch-size "${INTERNVIT_BATCH_SIZE:-4}"
  --num-workers 4
)
if [[ -n "${INTERNVIT_MAX_ROWS:-}" ]]; then
  ARGS+=(--max-rows "${INTERNVIT_MAX_ROWS}")
fi
if [[ "${INTERNVIT_OVERWRITE:-0}" == "1" ]]; then
  ARGS+=(--overwrite)
fi
python "${EXPERIMENT_ROOT}/exploratory_internvit/extract_features.py" "${ARGS[@]}"
