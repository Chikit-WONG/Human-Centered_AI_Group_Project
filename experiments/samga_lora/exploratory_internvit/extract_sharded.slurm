#!/usr/bin/env bash
#SBATCH --job-name=samga-internvit-shard
#SBATCH --partition=i64m1tga40u
#SBATCH --time=04:00:00
#SBATCH --array=0-8%8
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/samga_lora/samga-internvit-shard-%A_%a.out
#SBATCH --error=logs/samga_lora/samga-internvit-shard-%A_%a.err

set -euo pipefail
PROJECT_ROOT="${SAMGA_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?submit from the project root}}"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/experiments/samga_lora/scripts/common.sh"

SHARDS="${INTERNVIT_SHARDS:-8}"
if (( SHARDS <= 0 || SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID > SHARDS )); then
  echo "Expected array indices 0..${SHARDS}" >&2
  exit 2
fi
MODEL_PATH="${INTERNVIT_MODEL_PATH:-${ARTIFACT_ROOT}/internvit/model-03e138c81d3f}"
FEATURE_ROOT="${ARTIFACT_ROOT}/internvit/feature_cache"
SHARD_ROOT="${FEATURE_ROOT}/shards"
mkdir -p "${SHARD_ROOT}"
if (( SLURM_ARRAY_TASK_ID < SHARDS )); then
  SPLIT=train
  TOTAL=16540
  START=$((TOTAL * SLURM_ARRAY_TASK_ID / SHARDS))
  END=$((TOTAL * (SLURM_ARRAY_TASK_ID + 1) / SHARDS))
  OUTPUT="${SHARD_ROOT}/train_rows_${START}_${END}.npy"
  ROW_ARGS=(--start-row "${START}" --end-row "${END}")
else
  SPLIT=test
  OUTPUT="${FEATURE_ROOT}/internvit_layers_20_24_28_32_36_test.npy"
  ROW_ARGS=()
fi
ARGS=(
  --manifest "${MANIFEST_ROOT}/sub-01_${SPLIT}.json"
  --model-path "${MODEL_PATH}"
  --output "${OUTPUT}"
  --layer-ids 20 24 28 32 36
  --batch-size "${INTERNVIT_BATCH_SIZE:-16}"
  --num-workers 4
  "${ROW_ARGS[@]}"
)
if [[ "${INTERNVIT_OVERWRITE:-0}" == "1" ]]; then
  ARGS+=(--overwrite)
fi
python "${EXPERIMENT_ROOT}/exploratory_internvit/extract_features.py" "${ARGS[@]}"
