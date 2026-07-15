#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<'USAGE'
Run one THINGS-EEG subject with the verified Brain-RW retrieval protocol.

Usage:
  bash scripts/run_subject_reproduction.sh smoke --subject-id N [options]
  bash scripts/run_subject_reproduction.sh formal --subject-id N [options]

Modes:
  smoke   Train for one optimizer step, save/reload, then evaluate.
  formal  Train for 25 epochs, then run two independent reload evaluations.

Required:
  --subject-id N             THINGS-EEG subject in the range 1..10.

Options:
  --seed N                   Random seed (default: 42).
  --epochs N                 Override the mode's epoch count.
  --max-train-steps N|none   Override the optimizer-step limit.
  --output-dir PATH          Override the checkpoint/run directory.
  --results-dir PATH         Override the evaluation output directory.
  --resume-from-checkpoint PATH
                              Resume optimizer/model state from a checkpoint.
  --skip-eval                Stop after training.
  -h, --help                 Show this help message.

Environment overrides:
  THINGS_ROOT, BRAIN_DIR, CLIP_PATH, CONDA_SH, CONDA_ENV
USAGE
}

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly THINGS_ROOT="${THINGS_ROOT:-/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/EEG_Project/EEG_Recon-RL/datasets/things_eeg_data}"
readonly BRAIN_DIR="${BRAIN_DIR:-${THINGS_ROOT}/Preprocessed_data_250Hz_whiten}"
readonly CLIP_PATH="${CLIP_PATH:-/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/EEG_Project/models/CLIP-ViT-B-32-laion2B-s34B-b79K}"
readonly CONDA_SH="${CONDA_SH:-/hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh}"
readonly CONDA_ENV="${CONDA_ENV:-eeg_recon}"
readonly CHANNELS="P7,P5,P3,P1,Pz,P2,P4,P6,P8,PO7,PO3,POz,PO4,PO8,O1,Oz,O2"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

MODE="${1:-}"
if [[ "${MODE}" != "smoke" && "${MODE}" != "formal" ]]; then
    usage >&2
    exit 2
fi
shift

SUBJECT_ID=""
SEED=42
OUTPUT_DIR=""
RESULTS_DIR=""
RESUME_FROM_CHECKPOINT=""
SKIP_EVAL=false
if [[ "${MODE}" == "smoke" ]]; then
    EPOCHS=1
    MAX_TRAIN_STEPS=1
else
    EPOCHS=25
    MAX_TRAIN_STEPS=""
fi

while (($#)); do
    case "$1" in
        --subject-id)
            [[ $# -ge 2 ]] || { echo "error: --subject-id requires a value" >&2; exit 2; }
            SUBJECT_ID="$2"
            shift 2
            ;;
        --seed)
            [[ $# -ge 2 ]] || { echo "error: --seed requires a value" >&2; exit 2; }
            SEED="$2"
            shift 2
            ;;
        --epochs)
            [[ $# -ge 2 ]] || { echo "error: --epochs requires a value" >&2; exit 2; }
            EPOCHS="$2"
            shift 2
            ;;
        --max-train-steps)
            [[ $# -ge 2 ]] || { echo "error: --max-train-steps requires a value" >&2; exit 2; }
            if [[ "$2" == "none" ]]; then
                MAX_TRAIN_STEPS=""
            else
                MAX_TRAIN_STEPS="$2"
            fi
            shift 2
            ;;
        --output-dir)
            [[ $# -ge 2 ]] || { echo "error: --output-dir requires a value" >&2; exit 2; }
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --results-dir)
            [[ $# -ge 2 ]] || { echo "error: --results-dir requires a value" >&2; exit 2; }
            RESULTS_DIR="$2"
            shift 2
            ;;
        --resume-from-checkpoint)
            [[ $# -ge 2 ]] || { echo "error: --resume-from-checkpoint requires a value" >&2; exit 2; }
            RESUME_FROM_CHECKPOINT="$2"
            shift 2
            ;;
        --skip-eval)
            SKIP_EVAL=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

[[ "${SUBJECT_ID}" =~ ^([1-9]|10)$ ]] || {
    echo "error: --subject-id must be an integer from 1 to 10" >&2
    exit 2
}
[[ "${SEED}" =~ ^[0-9]+$ ]] || {
    echo "error: --seed must be a non-negative integer" >&2
    exit 2
}
[[ "${EPOCHS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "error: --epochs must be a positive integer" >&2
    exit 2
}
if [[ -n "${MAX_TRAIN_STEPS}" && ! "${MAX_TRAIN_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: --max-train-steps must be a positive integer or 'none'" >&2
    exit 2
fi

SUBJECT_PADDED="$(printf '%02d' "$((10#${SUBJECT_ID}))")"
readonly SUBJECT_ID SUBJECT_PADDED SEED

if [[ -z "${OUTPUT_DIR}" ]]; then
    if [[ "${MODE}" == "smoke" ]]; then
        OUTPUT_DIR="${PROJECT_ROOT}/runs/all_subjects/smoke/seed${SEED}/subj${SUBJECT_PADDED}"
    else
        OUTPUT_DIR="${PROJECT_ROOT}/runs/all_subjects/seed${SEED}/subj${SUBJECT_PADDED}"
    fi
fi
if [[ -z "${RESULTS_DIR}" ]]; then
    if [[ "${MODE}" == "smoke" ]]; then
        RESULTS_DIR="${OUTPUT_DIR}/evaluation"
    else
        RESULTS_DIR="${PROJECT_ROOT}/results/all_subjects/seed${SEED}/subj${SUBJECT_PADDED}"
    fi
fi

OUTPUT_DIR="$(realpath -m "${OUTPUT_DIR}")"
RESULTS_DIR="$(realpath -m "${RESULTS_DIR}")"
readonly OUTPUT_DIR RESULTS_DIR

required_paths=(
    "${PROJECT_ROOT}/train_clip_lora.py"
    "${BRAIN_DIR}/sub-${SUBJECT_PADDED}/train.pt"
    "${BRAIN_DIR}/sub-${SUBJECT_PADDED}/test.pt"
    "${THINGS_ROOT}/training_images"
    "${THINGS_ROOT}/test_images"
    "${CLIP_PATH}/config.json"
    "${CLIP_PATH}/preprocessor_config.json"
    "${CLIP_PATH}/model.safetensors"
    "${CONDA_SH}"
)
if [[ "${SKIP_EVAL}" == false ]]; then
    required_paths+=("${PROJECT_ROOT}/scripts/evaluate_retrieval.py")
fi
for path in "${required_paths[@]}"; do
    [[ -e "${path}" ]] || { echo "error: required path does not exist: ${path}" >&2; exit 1; }
done

if [[ -z "${RESUME_FROM_CHECKPOINT}" ]]; then
    existing_outputs=(
        "${OUTPUT_DIR}/validation_metrics.jsonl"
        "${OUTPUT_DIR}/brain_model"
        "${OUTPUT_DIR}/vision_model"
    )
    for path in "${existing_outputs[@]}"; do
        [[ ! -e "${path}" ]] || {
            echo "error: refusing to overwrite existing run artifact: ${path}" >&2
            echo "use a fresh --output-dir or explicitly resume from a checkpoint" >&2
            exit 1
        }
    done
else
    [[ -d "${RESUME_FROM_CHECKPOINT}" ]] || {
        echo "error: resume checkpoint directory does not exist: ${RESUME_FROM_CHECKPOINT}" >&2
        exit 1
    }
    RESUME_FROM_CHECKPOINT="$(realpath -e "${RESUME_FROM_CHECKPOINT}")"
    [[ "$(basename "${RESUME_FROM_CHECKPOINT}")" =~ ^(epoch|step)_[0-9]+$ ]] || {
        echo "error: resume checkpoint must be named epoch_N or step_N" >&2
        exit 1
    }
    [[ "$(dirname "${RESUME_FROM_CHECKPOINT}")" == "${OUTPUT_DIR}" ]] || {
        echo "error: resume checkpoint must belong to --output-dir: ${OUTPUT_DIR}" >&2
        exit 1
    }
fi

if [[ "${SKIP_EVAL}" == false ]]; then
    evaluation_outputs=(
        "${RESULTS_DIR}/sub${SUBJECT_PADDED}_seed${SEED}_${MODE}_metrics.json"
        "${RESULTS_DIR}/sub${SUBJECT_PADDED}_seed${SEED}_${MODE}_predictions.csv"
    )
    if [[ "${MODE}" == "formal" ]]; then
        evaluation_outputs+=(
            "${RESULTS_DIR}/sub${SUBJECT_PADDED}_seed${SEED}_formal_repeat_metrics.json"
            "${RESULTS_DIR}/sub${SUBJECT_PADDED}_seed${SEED}_formal_repeat_predictions.csv"
        )
    fi
    for path in "${evaluation_outputs[@]}"; do
        [[ ! -e "${path}" ]] || {
            echo "error: refusing to overwrite existing evaluation artifact: ${path}" >&2
            echo "use a fresh --results-dir" >&2
            exit 1
        }
    done
fi

mkdir -p "${OUTPUT_DIR}/cache" "${RESULTS_DIR}"
cd "${PROJECT_ROOT}"

# shellcheck source=/dev/null
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export SWANLAB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

RUN_NAME="brainrw-things-eeg-sub${SUBJECT_PADDED}-clip-b32-r32-seed${SEED}-${MODE}"

echo "mode=${MODE}"
echo "subject_id=${SUBJECT_ID}"
echo "seed=${SEED}"
echo "conda_env=${CONDA_DEFAULT_ENV:-unknown}"
echo "python=$(command -v python)"
echo "things_root=${THINGS_ROOT}"
echo "output_dir=${OUTPUT_DIR}"
echo "results_dir=${RESULTS_DIR}"
echo "epochs=${EPOCHS}"
echo "max_train_steps=${MAX_TRAIN_STEPS:-none}"
echo "resume_from_checkpoint=${RESUME_FROM_CHECKPOINT:-none}"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader || true

train_args=(
    --dataset_name things
    --brain_directory "${BRAIN_DIR}"
    --image_directory "${THINGS_ROOT}"
    --cache_dir "${OUTPUT_DIR}/cache"
    --subject_ids "${SUBJECT_ID}"
    --eval_subject_ids "${SUBJECT_ID}"
    --brain_column eeg
    --brain_backbone brain_mlp
    --dropout 0.1
    --pretrained_model_name_or_path "${CLIP_PATH}"
    --lora_rank 32
    --lora_layers all-linear
    --gradient_checkpointing
    --time_slice 0,250
    --avg_trials
    --selected_channels "${CHANNELS}"
    --learning_rate 5.0e-4
    --vision_learning_rate 5.0e-5
    --lr_scheduler_type cosine
    --weight_decay 0.05
    --seed "${SEED}"
    --dataloader_num_workers 8
    --mixed_precision bf16
    --output_dir "${OUTPUT_DIR}"
    --metrics_jsonl "${OUTPUT_DIR}/validation_metrics.jsonl"
    --run_name "${RUN_NAME}"
    --save_total_limit 1
    --checkpointing_steps epoch
    --validation_steps epoch
    --num_train_epochs "${EPOCHS}"
    --per_device_train_batch_size 512
    --per_device_eval_batch_size 100
)
if [[ -n "${MAX_TRAIN_STEPS}" ]]; then
    train_args+=(--max_train_steps "${MAX_TRAIN_STEPS}")
fi
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
    train_args+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

torchrun --standalone --nnodes=1 --nproc-per-node=1 \
    train_clip_lora.py "${train_args[@]}"

if [[ "${SKIP_EVAL}" == true ]]; then
    echo "Training complete; evaluation skipped."
    exit 0
fi

for path in "${OUTPUT_DIR}/brain_model" "${OUTPUT_DIR}/vision_model"; do
    [[ -d "${path}" ]] || { echo "error: expected saved model not found: ${path}" >&2; exit 1; }
done

METRICS_OUTPUT="${RESULTS_DIR}/sub${SUBJECT_PADDED}_seed${SEED}_${MODE}_metrics.json"
PREDICTIONS_OUTPUT="${RESULTS_DIR}/sub${SUBJECT_PADDED}_seed${SEED}_${MODE}_predictions.csv"

run_evaluation() {
    local metrics_output="$1"
    local predictions_output="$2"
    python scripts/evaluate_retrieval.py \
        --brain-model-path "${OUTPUT_DIR}/brain_model" \
        --vision-adapter-path "${OUTPUT_DIR}/vision_model" \
        --pretrained-model-name-or-path "${CLIP_PATH}" \
        --brain-directory "${BRAIN_DIR}" \
        --image-directory "${THINGS_ROOT}" \
        --dataset-name things \
        --subject-id "${SUBJECT_ID}" \
        --selected-channels "${CHANNELS}" \
        --time-slice 0,250 \
        --batch-size 100 \
        --num-workers 0 \
        --device cuda \
        --dtype bf16 \
        --cache-dir "${OUTPUT_DIR}/cache" \
        --seed "${SEED}" \
        --expected-num-samples 200 \
        --local-files-only \
        --metrics-output "${metrics_output}" \
        --predictions-output "${predictions_output}"
}

run_evaluation "${METRICS_OUTPUT}" "${PREDICTIONS_OUTPUT}"

if [[ "${MODE}" == "formal" ]]; then
    run_evaluation \
        "${RESULTS_DIR}/sub${SUBJECT_PADDED}_seed${SEED}_formal_repeat_metrics.json" \
        "${RESULTS_DIR}/sub${SUBJECT_PADDED}_seed${SEED}_formal_repeat_predictions.csv"
fi

echo "metrics=${METRICS_OUTPUT}"
echo "predictions=${PREDICTIONS_OUTPUT}"
