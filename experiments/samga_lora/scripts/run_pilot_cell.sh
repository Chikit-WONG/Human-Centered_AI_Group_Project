#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

INDEX="${1:?usage: run_pilot_cell.sh INDEX}"
SUBJECTS=(1 5 8)
SEEDS=(42 43)
RATIOS=(0.05 0.10 0.20)

if (( INDEX < 0 || INDEX >= 24 )); then
  echo "Pilot index must be in 0..23" >&2
  exit 2
fi
if (( INDEX < 6 )); then
  MODE=frozen
  CELL=${INDEX}
  FAMILY=frozen
  RATIO=0.10
else
  MODE=lora
  LORA_INDEX=$((INDEX - 6))
  RATIO_INDEX=$((LORA_INDEX / 6))
  CELL=$((LORA_INDEX % 6))
  RATIO="${RATIOS[${RATIO_INDEX}]}"
  FAMILY="lora-ratio-${RATIO}"
fi
SUBJECT_INDEX=$((CELL / 2))
SEED_INDEX=$((CELL % 2))
SUBJECT="${SUBJECTS[${SUBJECT_INDEX}]}"
SEED="${SEEDS[${SEED_INDEX}]}"
SUBJECT_PADDED="$(printf '%02d' "${SUBJECT}")"
RUN_DIR="${PILOT_ROOT}/${FAMILY}/sub-${SUBJECT_PADDED}/seed-${SEED}"
MANIFEST="${MANIFEST_ROOT}/sub-${SUBJECT_PADDED}_train.json"

if [[ -f "${RUN_DIR}/completion.json" ]]; then
  echo "Pilot run already complete: ${RUN_DIR}"
  exit 0
fi

ARGS=(
  --run-id "pilot-${FAMILY}-sub${SUBJECT_PADDED}-seed${SEED}"
  --manifest "${MANIFEST}"
  --validation-manifest "${MANIFEST}"
  --output-dir "${RUN_DIR}"
  --subset pilot_train
  --vision-mode "${MODE}"
  --vision-lr-ratio "${RATIO}"
  --clip-path "${CLIP_PATH}"
  --subject-id "${SUBJECT}"
  --seed "${SEED}"
  --num-epochs 60
  --stage1-epochs 20
  --candidate-epochs 20,25,30,35,40,45,50,55,60
  --batch-size 512
  --eval-batch-size 100
  --num-workers 8
)
if [[ "${MODE}" == "frozen" ]]; then
  ARGS+=(
    --feature-cache "${CACHE_ROOT}/clip_layers_4_6_8_10_12_train.npy"
    --validation-feature-cache "${CACHE_ROOT}/clip_layers_4_6_8_10_12_train.npy"
  )
fi
python "${EXPERIMENT_ROOT}/train.py" "${ARGS[@]}"
