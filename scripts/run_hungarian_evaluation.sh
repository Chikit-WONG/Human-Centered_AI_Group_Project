#!/usr/bin/env bash
set -Eeuo pipefail

readonly PROJECT_ROOT="/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/Class/AIAA3800_L01-Human-Centered_Artificial_Intelligence/Final_Project/test/brain-rw"
readonly THINGS_ROOT="/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/EEG_Project/EEG_Recon/datasets/things_eeg_data"
readonly BRAIN_DIR="${THINGS_ROOT}/Preprocessed_data_250Hz_whiten"
readonly CLIP_PATH="/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/EEG_Project/models/CLIP-ViT-B-32-laion2B-s34B-b79K"
readonly MODEL_DIR="${PROJECT_ROOT}/runs/seed42/subj08"
readonly RESULTS_DIR="${PROJECT_ROOT}/results"
readonly CONDA_SH="/hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh"
readonly CHANNELS="P7,P5,P3,P1,Pz,P2,P4,P6,P8,PO7,PO3,POz,PO4,PO8,O1,Oz,O2"

required_paths=(
    "${MODEL_DIR}/brain_model"
    "${MODEL_DIR}/vision_model"
    "${BRAIN_DIR}/sub-08/test.pt"
    "${THINGS_ROOT}/test_images"
    "${CLIP_PATH}/config.json"
    "${PROJECT_ROOT}/scripts/evaluate_retrieval.py"
    "${PROJECT_ROOT}/scripts/finalize_hungarian_results.py"
    "${PROJECT_ROOT}/tests/test_hungarian_assignment.py"
    "${CONDA_SH}"
)
for path in "${required_paths[@]}"; do
    [[ -e "${path}" ]] || {
        echo "error: required path does not exist: ${path}" >&2
        exit 1
    }
done

mkdir -p "${RESULTS_DIR}" "${MODEL_DIR}/cache"
cd "${PROJECT_ROOT}"

# shellcheck source=/dev/null
source "${CONDA_SH}"
conda activate eeg_recon

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export SWANLAB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "conda_env=${CONDA_DEFAULT_ENV:-unknown}"
echo "python=$(command -v python)"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader

python -m unittest -v tests/test_hungarian_assignment.py

run_evaluation() {
    local suffix="$1"
    local metrics_output="$2"
    local standard_predictions_output="$3"
    local assignment_output="$4"
    local similarity_output="$5"
    echo "evaluation_pass=${suffix}"
    python scripts/evaluate_retrieval.py \
        --brain-model-path "${MODEL_DIR}/brain_model" \
        --vision-adapter-path "${MODEL_DIR}/vision_model" \
        --pretrained-model-name-or-path "${CLIP_PATH}" \
        --brain-directory "${BRAIN_DIR}" \
        --image-directory "${THINGS_ROOT}" \
        --dataset-name things \
        --subject-id 8 \
        --selected-channels "${CHANNELS}" \
        --time-slice 0,250 \
        --batch-size 100 \
        --num-workers 0 \
        --device cuda \
        --dtype bf16 \
        --cache-dir "${MODEL_DIR}/cache" \
        --seed 42 \
        --expected-num-samples 200 \
        --expected-top1-count 182 \
        --expected-top5-count 199 \
        --local-files-only \
        --enable-hungarian \
        --metrics-output "${metrics_output}" \
        --predictions-output "${standard_predictions_output}" \
        --hungarian-output "${assignment_output}" \
        --similarity-output "${similarity_output}"
}

run_evaluation \
    primary \
    "${RESULTS_DIR}/sub08_seed42_formal_hungarian_assignment_metrics.json" \
    "${RESULTS_DIR}/sub08_seed42_formal_hungarian_standard_predictions.csv" \
    "${RESULTS_DIR}/sub08_seed42_formal_hungarian_assignment_predictions.csv" \
    "${RESULTS_DIR}/sub08_seed42_formal_cosine_similarity.npz"

run_evaluation \
    repeat \
    "${RESULTS_DIR}/sub08_seed42_formal_hungarian_assignment_repeat_metrics.json" \
    "${RESULTS_DIR}/sub08_seed42_formal_hungarian_standard_repeat_predictions.csv" \
    "${RESULTS_DIR}/sub08_seed42_formal_hungarian_assignment_repeat_predictions.csv" \
    "${RESULTS_DIR}/sub08_seed42_formal_cosine_similarity_repeat.npz"

python scripts/finalize_hungarian_results.py --results-dir "${RESULTS_DIR}"

echo "Hungarian evaluation passes completed."
