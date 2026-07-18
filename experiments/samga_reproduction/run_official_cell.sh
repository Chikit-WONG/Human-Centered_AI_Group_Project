#!/usr/bin/env bash
set -euo pipefail

SUBJECT="${1-}"
if [[ ! "${SUBJECT}" =~ ^([1-9]|10)$ ]]; then
  echo "SUBJECT must be a decimal integer in 1..10" >&2
  exit 2
fi

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/hpc2hdd/home/ckwong627/workdir/new_sub_workdir}"
PROJECT_ROOT="${SAMGA_PROJECT_ROOT:-${WORKSPACE_ROOT}/Class/AIAA3800_L01-Human-Centered_Artificial_Intelligence/Final_Project/Human-Centered_AI_Group_Project}"
SAMGA_ROOT="${SAMGA_REFERENCE_ROOT:-${WORKSPACE_ROOT}/EEG_Project/reference_code/codes_for_papers/SAMGA}"
REPRO_ROOT="${SAMGA_REPRO_ROOT:-${PROJECT_ROOT}/artifacts/samga_reproduction}"
EEG_ROOT="${SAMGA_EEG_ROOT:-${REPRO_ROOT}/data/preprocessed_eeg}"
VARIANT="${SAMGA_VARIANT-}"
VARIANT_PATTERN='^[A-Za-z0-9][A-Za-z0-9._-]*$'
if [[ ! "${VARIANT}" =~ ${VARIANT_PATTERN} ]]; then
  echo "SAMGA_VARIANT must match [A-Za-z0-9][A-Za-z0-9._-]* (got '${VARIANT}')" >&2
  exit 2
fi
FEATURE_ROOT="${SAMGA_FEATURE_ROOT:-${REPRO_ROOT}/features/${VARIANT}}"
SEED="${SAMGA_SEED-2025}"
BATCH_SIZE="${SAMGA_BATCH_SIZE-1024}"
EARLY_STOP_PATIENCE="${SAMGA_EARLY_STOP_PATIENCE-0}"
NUM_EPOCHS="${SAMGA_NUM_EPOCHS-60}"
EEG_L2NORM="${SAMGA_EEG_L2NORM-0}"
T_LEARNABLE="${SAMGA_T_LEARNABLE-0}"
ROUTER_LAYER_DROPOUT="${SAMGA_ROUTER_LAYER_DROPOUT-0}"

if [[ ! "${SEED}" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "SAMGA_SEED must be an integer in 0..4294967295 (got '${SEED}')" >&2
  exit 2
fi
SEED_LENGTH="${#SEED}"
if (( SEED_LENGTH > 10 )) \
  || { (( SEED_LENGTH == 10 )) && [[ "${SEED}" > "4294967295" ]]; }; then
  echo "SAMGA_SEED must be an integer in 0..4294967295 (got '${SEED}')" >&2
  exit 2
fi
if [[ ! "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SAMGA_BATCH_SIZE must be a positive decimal integer (got '${BATCH_SIZE}')" >&2
  exit 2
fi
if [[ ! "${NUM_EPOCHS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SAMGA_NUM_EPOCHS must be a positive decimal integer (got '${NUM_EPOCHS}')" >&2
  exit 2
fi
if [[ ! "${EARLY_STOP_PATIENCE}" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "SAMGA_EARLY_STOP_PATIENCE must be a non-negative decimal integer (got '${EARLY_STOP_PATIENCE}')" >&2
  exit 2
fi

SUBJECT_PADDED="$(printf '%02d' "${SUBJECT}")"
CELL_ROOT="${REPRO_ROOT}/official_runs/${VARIANT}/seed${SEED}/sub-${SUBJECT_PADDED}-b${BATCH_SIZE}-p${EARLY_STOP_PATIENCE}"

validate_boolean_switch() {
  local name="${1}"
  local value="${2}"
  if [[ "${value}" != "0" && "${value}" != "1" ]]; then
    echo "${name} must be 0 or 1 (got '${value}')" >&2
    exit 2
  fi
}

normalize_unsigned_decimal() {
  local raw="${1}"
  local leading_zeroes="${raw%%[!0]*}"
  NORMALIZED_UNSIGNED_DECIMAL="${raw#"${leading_zeroes}"}"
  if [[ -z "${NORMALIZED_UNSIGNED_DECIMAL}" ]]; then
    NORMALIZED_UNSIGNED_DECIMAL="0"
  fi
}

unsigned_decimal_leq_small() {
  local raw="${1}"
  local small="${2}"
  normalize_unsigned_decimal "${raw}"
  local normalized="${NORMALIZED_UNSIGNED_DECIMAL}"
  if (( ${#normalized} < ${#small} )); then
    return 0
  fi
  if (( ${#normalized} > ${#small} )); then
    return 1
  fi
  [[ "${normalized}" == "${small}" || "${normalized}" < "${small}" ]]
}

unsigned_decimal_geq_small() {
  local raw="${1}"
  local small="${2}"
  normalize_unsigned_decimal "${raw}"
  local normalized="${NORMALIZED_UNSIGNED_DECIMAL}"
  if (( ${#normalized} > ${#small} )); then
    return 0
  fi
  if (( ${#normalized} < ${#small} )); then
    return 1
  fi
  [[ "${normalized}" == "${small}" || "${normalized}" > "${small}" ]]
}

decimal_exponent_leq_threshold() {
  local raw="${1}"
  local threshold="${2}"
  local negative=0
  local magnitude="${raw}"

  if [[ "${magnitude}" == -* ]]; then
    negative=1
    magnitude="${magnitude#-}"
  elif [[ "${magnitude}" == +* ]]; then
    magnitude="${magnitude#+}"
  fi

  normalize_unsigned_decimal "${magnitude}"
  magnitude="${NORMALIZED_UNSIGNED_DECIMAL}"
  if [[ "${magnitude}" == "0" ]]; then
    negative=0
  fi

  if (( negative == 1 )); then
    if (( threshold >= 0 )); then
      return 0
    fi
    local threshold_magnitude="$((-threshold))"
    if unsigned_decimal_geq_small "${magnitude}" "${threshold_magnitude}"; then
      return 0
    fi
    return 1
  fi

  if (( threshold < 0 )); then
    return 1
  fi
  if unsigned_decimal_leq_small "${magnitude}" "${threshold}"; then
    return 0
  fi
  return 1
}

validate_router_layer_dropout() {
  local value="${1}"
  local number_pattern='^[+-]?(([0-9]+([.][0-9]*)?)|([.][0-9]+))([eE][+-]?[0-9]+)?$'
  ROUTER_LAYER_DROPOUT_ENABLED=0
  if [[ ! "${value}" =~ ${number_pattern} ]]; then
    echo "SAMGA_ROUTER_LAYER_DROPOUT must be a finite number in [0, 1) (got '${value}')" >&2
    exit 2
  fi

  local negative=0
  local unsigned="${value}"
  if [[ "${unsigned}" == -* ]]; then
    negative=1
    unsigned="${unsigned#-}"
  elif [[ "${unsigned}" == +* ]]; then
    unsigned="${unsigned#+}"
  fi

  local mantissa="${unsigned}"
  local exponent="0"
  if [[ "${unsigned}" == *[eE]* ]]; then
    mantissa="${unsigned%%[eE]*}"
    exponent="${unsigned#*[eE]}"
  fi

  local coefficient="${mantissa//./}"
  if [[ ! "${coefficient}" =~ [1-9] ]]; then
    return 0
  fi
  if (( negative == 1 )); then
    echo "SAMGA_ROUTER_LAYER_DROPOUT must be a finite number in [0, 1) (got '${value}')" >&2
    exit 2
  fi

  local fractional_part=""
  if [[ "${mantissa}" == *.* ]]; then
    fractional_part="${mantissa#*.}"
  fi
  normalize_unsigned_decimal "${coefficient}"
  local significant_digits="${NORMALIZED_UNSIGNED_DECIMAL}"
  local fractional_length="${#fractional_part}"
  local significant_length="${#significant_digits}"
  local exponent_threshold="$((fractional_length - significant_length))"
  if ! decimal_exponent_leq_threshold "${exponent}" "${exponent_threshold}"; then
    echo "SAMGA_ROUTER_LAYER_DROPOUT must be a finite number in [0, 1) (got '${value}')" >&2
    exit 2
  fi
  ROUTER_LAYER_DROPOUT_ENABLED=1
}

validate_boolean_switch "SAMGA_EEG_L2NORM" "${EEG_L2NORM}"
validate_boolean_switch "SAMGA_T_LEARNABLE" "${T_LEARNABLE}"
validate_router_layer_dropout "${ROUTER_LAYER_DROPOUT}"

OPTIONAL_TRAIN_ARGS=()
if [[ "${EEG_L2NORM}" == "1" ]]; then
  OPTIONAL_TRAIN_ARGS+=(--eeg_l2norm)
fi
if [[ "${T_LEARNABLE}" == "1" ]]; then
  OPTIONAL_TRAIN_ARGS+=(--t_learnable)
fi
if [[ "${ROUTER_LAYER_DROPOUT_ENABLED}" == "1" ]]; then
  OPTIONAL_TRAIN_ARGS+=(--router_layer_dropout "${ROUTER_LAYER_DROPOUT}")
fi

if find "${CELL_ROOT}" -name result.csv -print -quit 2>/dev/null | grep -q .; then
  echo "Completed result already exists under ${CELL_ROOT}"
  exit 0
fi

for path in \
  "${EEG_ROOT}/info.json" \
  "${EEG_ROOT}/sub-${SUBJECT_PADDED}/train.npy" \
  "${EEG_ROOT}/sub-${SUBJECT_PADDED}/test.npy" \
  "${FEATURE_ROOT}/image_train_layer20.npy" \
  "${FEATURE_ROOT}/image_test_layer20.npy"; do
  if [[ ! -f "${path}" ]]; then
    echo "Required reproduction asset is missing: ${path}" >&2
    exit 3
  fi
done

mkdir -p "${CELL_ROOT}" "${PROJECT_ROOT}/logs/samga_reproduction"
set +u
source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate test
set -u
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "${SAMGA_ROOT}"
python train.py \
  --batch_size "${BATCH_SIZE}" \
  --stage2_learning_rate 5e-5 \
  --learning_rate 1e-4 \
  --output_name "official-${VARIANT}-sub${SUBJECT_PADDED}-seed${SEED}-b${BATCH_SIZE}-p${EARLY_STOP_PATIENCE}" \
  --eeg_encoder_type EEGProject \
  --train_subject_ids "${SUBJECT}" \
  --test_subject_ids "${SUBJECT}" \
  --softplus \
  --num_epochs "${NUM_EPOCHS}" \
  --image_feature_dir "${FEATURE_ROOT}" \
  --text_feature_dir "" \
  --eeg_data_dir "${EEG_ROOT}" \
  --device cuda:0 \
  --output_dir "${CELL_ROOT}" \
  --selected_channels P7 P5 P3 P1 Pz P2 P4 P6 P8 PO7 PO3 POz PO4 PO8 O1 Oz O2 \
  --eeg_aug \
  --eeg_aug_type smooth \
  --frozen_eeg_prior \
  --img_l2norm \
  --projector linear \
  --feature_dim 512 \
  --data_average \
  --save_weights \
  --stage1_mmd_start 0.9 \
  --stage1_mmd_end 0.5 \
  --stage1_epochs 20 \
  --early_stop_patience "${EARLY_STOP_PATIENCE}" \
  --use_multilayer_router \
  --layer_ids 20 24 28 32 36 \
  --layer_prior_center 28 \
  --layer_prior_strength 1.0 \
  --router_eval_mode global \
  --seed "${SEED}" \
  "${OPTIONAL_TRAIN_ARGS[@]}"
