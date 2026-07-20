#!/usr/bin/env bash
set -euo pipefail

SUBJECT="${1:?subject is required}"
SEED="${2:?seed is required}"
CONFIG="${3:?baseline config is required}"
MANIFEST="${4:?protocol manifest is required}"
FEATURE_CACHE="${5:?feature cache is required}"
STAGE2_CONFIG="${6:?stage2 config is required}"
CANDIDATE_ID="${7:?candidate id is required}"
RESUME="${8:?resume is required}"
OUTPUT_DIR="${9:?output directory is required}"
PROJECT_ROOT="${10:?project root is required}"

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="${PROJECT_ROOT}/experiments/samga_brain_rw"

TRAIN_EXTRA_ARGS=()
ADAPTER_RANK="${SAMGA_ADAPTER_RANK:-}"
ADAPTER_LR_RATIO="${SAMGA_ADAPTER_LR_RATIO:-}"
if [[ -n "${ADAPTER_RANK}" || -n "${ADAPTER_LR_RATIO}" ]]; then
  if [[ -z "${ADAPTER_RANK}" || -z "${ADAPTER_LR_RATIO}" ]]; then
    echo "SAMGA_ADAPTER_RANK and SAMGA_ADAPTER_LR_RATIO must be set together" >&2
    exit 2
  fi
  TRAIN_EXTRA_ARGS+=(
    --adapter-rank "${ADAPTER_RANK}"
    --adapter-lr-ratio "${ADAPTER_LR_RATIO}"
  )
fi

WHITENING_ARTIFACT="${SAMGA_WHITENING_ARTIFACT:-}"
if [[ -n "${WHITENING_ARTIFACT}" ]]; then
  TRAIN_EXTRA_ARGS+=(
    --whitening-artifact "${WHITENING_ARTIFACT}"
  )
fi

python "${PROJECT_ROOT}/experiments/samga_brain_rw/train.py" \
  --scope train \
  --validation-scope val-dev \
  --stage 2 \
  --subject "${SUBJECT}" \
  --seed "${SEED}" \
  --device cuda \
  --resume "${RESUME}" \
  --config "${CONFIG}" \
  --manifest "${MANIFEST}" \
  --feature-cache "${FEATURE_CACHE}" \
  --stage2-config "${STAGE2_CONFIG}" \
  --candidate-id "${CANDIDATE_ID}" \
  "${TRAIN_EXTRA_ARGS[@]}" \
  --output-dir "${OUTPUT_DIR}"

test -d "${OUTPUT_DIR}/in_loop"

python "${PROJECT_ROOT}/experiments/samga_brain_rw/evaluate.py" \
  --scope val-dev \
  --subject "${SUBJECT}" \
  --seed "${SEED}" \
  --device cuda \
  --config "${CONFIG}" \
  --manifest "${MANIFEST}" \
  --feature-cache "${FEATURE_CACHE}" \
  --checkpoint "${OUTPUT_DIR}/checkpoint_epoch060.pt" \
  --output-dir "${OUTPUT_DIR}/saved_checkpoint"

python "${PROJECT_ROOT}/experiments/samga_brain_rw/evaluate.py" \
  --scope val-dev \
  --subject "${SUBJECT}" \
  --seed "${SEED}" \
  --device cuda \
  --config "${CONFIG}" \
  --manifest "${MANIFEST}" \
  --feature-cache "${FEATURE_CACHE}" \
  --checkpoint "${OUTPUT_DIR}/checkpoint_epoch060.pt" \
  --output-dir "${OUTPUT_DIR}/repeat_emission"

python "${PROJECT_ROOT}/experiments/samga_brain_rw/evaluate.py" \
  --scope val-dev \
  --subject "${SUBJECT}" \
  --seed "${SEED}" \
  --device cuda \
  --config "${CONFIG}" \
  --manifest "${MANIFEST}" \
  --feature-cache "${FEATURE_CACHE}" \
  --checkpoint "${OUTPUT_DIR}/checkpoint_epoch060.pt" \
  --output-dir "${OUTPUT_DIR}/reload_evaluation"

python \
  "${PROJECT_ROOT}/experiments/samga_brain_rw/scripts/check_baseline_parity.py" \
  --run-dir "${OUTPUT_DIR}" \
  --scope val-dev \
  --output "${OUTPUT_DIR}/baseline_parity.json"

if [[ -n "${SAMGA_JOB_MAP:-}" ]]; then
  FINAL_CHECKPOINT_SHA256="$(
    sha256sum "${OUTPUT_DIR}/checkpoint_epoch060.pt" | awk '{print $1}'
  )"
  PARITY_SHA256="$(
    sha256sum "${OUTPUT_DIR}/baseline_parity.json" | awk '{print $1}'
  )"
  RUN_MANIFEST_SHA256="$(
    sha256sum "${OUTPUT_DIR}/run_manifest.json" | awk '{print $1}'
  )"
  python \
    "${PROJECT_ROOT}/experiments/samga_brain_rw/scripts/build_job_map.py" \
    complete-env \
    --output-hashes \
    "{\"final_checkpoint_sha256\":\"${FINAL_CHECKPOINT_SHA256}\",\"parity_sha256\":\"${PARITY_SHA256}\",\"run_manifest_sha256\":\"${RUN_MANIFEST_SHA256}\"}"
fi
