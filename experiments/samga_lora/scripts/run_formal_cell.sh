#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

INDEX="${1:?usage: run_formal_cell.sh INDEX}"
if (( INDEX < 0 || INDEX >= 100 )); then
  echo "Formal index must be in 0..99" >&2
  exit 2
fi
read -r GATE EPOCH RATIO < <(python -c \
  'import json,sys; d=json.load(open(sys.argv[1])); print(int(d["gate_passed"]), d["selected"]["epoch"], d["selected"]["vision_lr_ratio"])' \
  "${LOCKED_CONFIG}")
if [[ "${GATE}" != "1" ]]; then
  echo "Pilot gate did not pass; refusing formal test experiment" >&2
  exit 3
fi
python "${EXPERIMENT_ROOT}/scripts/verify_locked_source.py" \
  --locked-config "${LOCKED_CONFIG}" \
  --experiment-root "${EXPERIMENT_ROOT}"

if (( INDEX < 50 )); then
  MODE=frozen
  CELL=${INDEX}
else
  MODE=lora
  CELL=$((INDEX - 50))
fi
SUBJECT=$((CELL / 5 + 1))
SEED=$((CELL % 5 + 42))
SUBJECT_PADDED="$(printf '%02d' "${SUBJECT}")"
RUN_DIR="${FORMAL_ROOT}/${MODE}/sub-${SUBJECT_PADDED}/seed-${SEED}"
TRAIN_MANIFEST="${MANIFEST_ROOT}/sub-${SUBJECT_PADDED}_train.json"
TEST_MANIFEST="${MANIFEST_ROOT}/sub-${SUBJECT_PADDED}_test.json"
CHECKPOINT="${RUN_DIR}/checkpoint_epoch$(printf '%03d' "${EPOCH}").pt"

if [[ ! -f "${RUN_DIR}/completion.json" ]]; then
  ARGS=(
    --run-id "formal-${MODE}-sub${SUBJECT_PADDED}-seed${SEED}"
    --manifest "${TRAIN_MANIFEST}"
    --output-dir "${RUN_DIR}"
    --subset formal_train
    --vision-mode "${MODE}"
    --vision-lr-ratio "${RATIO}"
    --clip-path "${CLIP_PATH}"
    --subject-id "${SUBJECT}"
    --seed "${SEED}"
    --num-epochs "${EPOCH}"
    --stage1-epochs 20
    --candidate-epochs "${EPOCH}"
    --batch-size 512
    --num-workers 4
  )
  if [[ "${MODE}" == "frozen" ]]; then
    ARGS+=(--feature-cache "${CACHE_ROOT}/clip_layers_4_6_8_10_12_train.npy")
  fi
  python "${EXPERIMENT_ROOT}/train.py" "${ARGS[@]}"
fi

if [[ ! -f "${RUN_DIR}/test_metrics.json" ]]; then
  EVAL_ARGS=(
    --checkpoint "${CHECKPOINT}"
    --test-manifest "${TEST_MANIFEST}"
    --metrics-output "${RUN_DIR}/test_metrics.json"
    --predictions-output "${RUN_DIR}/test_predictions.csv"
    --batch-size 100
    --num-workers 4
  )
  if [[ "${MODE}" == "frozen" ]]; then
    EVAL_ARGS+=(--test-feature-cache "${CACHE_ROOT}/clip_layers_4_6_8_10_12_test.npy")
  fi
  if [[ -f "${RUN_DIR}/test_predictions.csv" ]]; then
    EVAL_ARGS+=(--overwrite)
  fi
  python "${EXPERIMENT_ROOT}/evaluate.py" "${EVAL_ARGS[@]}"
fi
