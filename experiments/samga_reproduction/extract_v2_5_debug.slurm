#!/usr/bin/env bash
#SBATCH --job-name=samga-v2_5-features
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --array=0-8%2
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/samga_reproduction/samga-v2_5-features-%A_%a.out
#SBATCH --error=logs/samga_reproduction/samga-v2_5-features-%A_%a.err

set -euo pipefail

PROJECT_ROOT="${SAMGA_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?submit from the project root}}"
MODEL_PATH="${INTERNVIT_V2_5_MODEL_PATH:?set INTERNVIT_V2_5_MODEL_PATH to the fixed local checkpoint}"
MANIFEST_ROOT="${PROJECT_ROOT}/artifacts/samga_lora/manifests"
FEATURE_ROOT="${SAMGA_V2_5_FEATURE_ROOT:-${PROJECT_ROOT}/artifacts/samga_reproduction/features/internvit_v2_5_multi_variant}"
SCRIPT="${PROJECT_ROOT}/experiments/samga_reproduction/extract_v2_5_features.py"

set +u
source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate test
set -u

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/logs/samga_reproduction"

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID > 8 )); then
  echo "Expected SLURM_ARRAY_TASK_ID in 0..8" >&2
  exit 2
fi

if (( SLURM_ARRAY_TASK_ID < 8 )); then
  SPLIT="train"
  SHARD_INDEX="${SLURM_ARRAY_TASK_ID}"
  SHARD_COUNT=8
else
  SPLIT="test"
  SHARD_INDEX=0
  SHARD_COUNT=1
fi

if [[ -n "${INTERNVIT_V2_5_MAX_ROWS:-}" ]]; then
  FEATURE_ROOT="${FEATURE_ROOT}/debug_rows_${INTERNVIT_V2_5_MAX_ROWS}"
fi
OUTPUT_DIRECTORY="${FEATURE_ROOT}/shards/${SPLIT}/shard_$(printf '%02d' "${SHARD_INDEX}")"

ARGS=(
  --manifest "${MANIFEST_ROOT}/sub-01_${SPLIT}.json"
  --model-path "${MODEL_PATH}"
  --output-directory "${OUTPUT_DIRECTORY}"
  --shard-index "${SHARD_INDEX}"
  --shard-count "${SHARD_COUNT}"
  --batch-size "${INTERNVIT_V2_5_BATCH_SIZE:-16}"
  --num-workers "${INTERNVIT_V2_5_NUM_WORKERS:-4}"
  --seed "${INTERNVIT_V2_5_SEED:-0}"
  --device cuda
)
if [[ -n "${INTERNVIT_V2_5_MAX_ROWS:-}" ]]; then
  ARGS+=(--max-rows "${INTERNVIT_V2_5_MAX_ROWS}")
fi

python "${SCRIPT}" "${ARGS[@]}"
