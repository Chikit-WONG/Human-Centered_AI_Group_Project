#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

INDEX="${1:?usage: run_smoke_cell.sh INDEX}"
if [[ "${INDEX}" == "0" ]]; then
  MODE=frozen
elif [[ "${INDEX}" == "1" ]]; then
  MODE=lora
else
  echo "Smoke index must be 0 or 1" >&2
  exit 2
fi

SUBJECT=8
SEED=42
RUN_DIR="${SMOKE_ROOT}/${MODE}"
MANIFEST="${MANIFEST_ROOT}/sub-08_train.json"
ARGS=(
  --run-id "smoke-${MODE}"
  --manifest "${MANIFEST}"
  --validation-manifest "${MANIFEST}"
  --output-dir "${RUN_DIR}"
  --subset pilot_train
  --vision-mode "${MODE}"
  --clip-path "${CLIP_PATH}"
  --subject-id "${SUBJECT}"
  --seed "${SEED}"
  --num-epochs 1
  --stage1-epochs 1
  --candidate-epochs 1
  --batch-size 512
  --eval-batch-size 100
  --num-workers 4
  --max-train-steps 1
)
if [[ "${MODE}" == "frozen" ]]; then
  ARGS+=(
    --feature-cache "${CACHE_ROOT}/clip_layers_4_6_8_10_12_train.npy"
    --validation-feature-cache "${CACHE_ROOT}/clip_layers_4_6_8_10_12_train.npy"
  )
fi

if [[ -f "${RUN_DIR}/completion.json" ]]; then
  echo "Smoke run already complete: ${RUN_DIR}"
else
  python "${EXPERIMENT_ROOT}/train.py" "${ARGS[@]}"
fi
python "${EXPERIMENT_ROOT}/scripts/verify_checkpoint.py" \
  --checkpoint "${RUN_DIR}/checkpoint_epoch001.pt" \
  --output "${RUN_DIR}/checkpoint_reload.json"
