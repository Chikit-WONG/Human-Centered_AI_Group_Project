#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../scripts/common.sh"

INDEX="${1:?usage: run_cell.sh INDEX}"
if (( INDEX < 0 || INDEX >= 10 )); then
  echo "InternViT exploratory index must be in 0..9" >&2
  exit 2
fi
SUBJECT=$((INDEX + 1))
SUBJECT_PADDED="$(printf '%02d' "${SUBJECT}")"
SEED="${INTERNVIT_SEED:-2025}"
EPOCH="${INTERNVIT_EPOCH:-60}"
IMAGE_DIM=3200
LAYERS=(20 24 28 32 36)
INTERNVIT_ROOT="${ARTIFACT_ROOT}/internvit"
MODEL_PATH="${INTERNVIT_MODEL_PATH:-${INTERNVIT_ROOT}/model-03e138c81d3f}"
FEATURE_ROOT="${INTERNVIT_ROOT}/feature_cache"
RUN_DIR="${INTERNVIT_ROOT}/exploratory_seed${SEED}/sub-${SUBJECT_PADDED}"
TRAIN_MANIFEST="${MANIFEST_ROOT}/sub-${SUBJECT_PADDED}_train.json"
TEST_MANIFEST="${MANIFEST_ROOT}/sub-${SUBJECT_PADDED}_test.json"
TRAIN_CACHE="${FEATURE_ROOT}/internvit_layers_20_24_28_32_36_train.npy"
TEST_CACHE="${FEATURE_ROOT}/internvit_layers_20_24_28_32_36_test.npy"
CHECKPOINT="${RUN_DIR}/checkpoint_epoch$(printf '%03d' "${EPOCH}").pt"

for path in "${MODEL_PATH}/config.json" "${TRAIN_CACHE}" "${TEST_CACHE}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Required InternViT artifact is missing: ${path}" >&2
    exit 3
  fi
done

if [[ ! -f "${RUN_DIR}/completion.json" ]]; then
  python "${EXPERIMENT_ROOT}/exploratory_internvit/entry.py" train \
    --image-dim "${IMAGE_DIM}" \
    --run-id "internvit-frozen-sub${SUBJECT_PADDED}-seed${SEED}" \
    --manifest "${TRAIN_MANIFEST}" \
    --output-dir "${RUN_DIR}" \
    --subset formal_train \
    --vision-mode frozen \
    --vision-lr-ratio 0 \
    --feature-cache "${TRAIN_CACHE}" \
    --clip-path "${MODEL_PATH}" \
    --subject-id "${SUBJECT}" \
    --seed "${SEED}" \
    --layer-ids "${LAYERS[@]}" \
    --prior-center 28 \
    --num-epochs "${EPOCH}" \
    --stage1-epochs 20 \
    --candidate-epochs "${EPOCH}" \
    --batch-size 512 \
    --num-workers 4
fi

if [[ ! -f "${RUN_DIR}/checkpoint_verification.json" ]]; then
  python "${EXPERIMENT_ROOT}/exploratory_internvit/entry.py" verify \
    --image-dim "${IMAGE_DIM}" \
    --checkpoint "${CHECKPOINT}" \
    --output "${RUN_DIR}/checkpoint_verification.json"
fi

if [[ ! -f "${RUN_DIR}/test_metrics.json" ]]; then
  EVAL_ARGS=(
    --checkpoint "${CHECKPOINT}"
    --test-manifest "${TEST_MANIFEST}"
    --test-feature-cache "${TEST_CACHE}"
    --metrics-output "${RUN_DIR}/test_metrics.json"
    --predictions-output "${RUN_DIR}/test_predictions.csv"
    --batch-size 100
    --num-workers 4
  )
  if [[ -f "${RUN_DIR}/test_predictions.csv" ]]; then
    EVAL_ARGS+=(--overwrite)
  fi
  python "${EXPERIMENT_ROOT}/exploratory_internvit/entry.py" evaluate \
    --image-dim "${IMAGE_DIM}" "${EVAL_ARGS[@]}"
fi
