#!/usr/bin/env bash
#SBATCH --job-name=mf3_export_brainrw
#SBATCH --partition=debug
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/Class/AIAA3800_L01-Human-Centered_Artificial_Intelligence/Final_Project/test/brain-rw/logs/matching_fairness_v3/export_brainrw_%j.out
#SBATCH --error=/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/Class/AIAA3800_L01-Human-Centered_Artificial_Intelligence/Final_Project/test/brain-rw/logs/matching_fairness_v3/export_brainrw_%j.err

set -Eeuo pipefail
readonly REQUESTED_EXPERIMENT_ROOT="${MATCHING_FAIRNESS_EXPERIMENT_ROOT:-}"
if [[ -z "${REQUESTED_EXPERIMENT_ROOT}" ]]; then
    echo "error: MATCHING_FAIRNESS_EXPERIMENT_ROOT must be set" >&2
    exit 64
fi
if [[ "${REQUESTED_EXPERIMENT_ROOT}" != /* ]]; then
    echo "error: MATCHING_FAIRNESS_EXPERIMENT_ROOT must be an absolute path" >&2
    exit 64
fi
if [[ ! -d "${REQUESTED_EXPERIMENT_ROOT}" ]]; then
    echo "error: MATCHING_FAIRNESS_EXPERIMENT_ROOT directory does not exist" >&2
    exit 64
fi
if ! EXPERIMENT_ROOT="$(cd -- "${REQUESTED_EXPERIMENT_ROOT}" && pwd -P)"; then
    echo "error: could not canonicalize MATCHING_FAIRNESS_EXPERIMENT_ROOT" >&2
    exit 64
fi
readonly EXPERIMENT_ROOT
readonly SUBMITTER="${EXPERIMENT_ROOT}/scripts/submit_pipeline.py"
if [[ ! -f "${SUBMITTER}" ]]; then
    echo "error: MATCHING_FAIRNESS_EXPERIMENT_ROOT is missing scripts/submit_pipeline.py" >&2
    exit 64
fi
readonly CONDA_SH="/hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh"
source "${CONDA_SH}"
conda activate eeg_recon

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="${EXPERIMENT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

arguments=(
    "${SUBMITTER}"
    --internal-cell export-brainrw
)
if [[ "${MATCHING_FAIRNESS_OVERWRITE:-0}" == "1" ]]; then
    arguments+=(--overwrite)
elif [[ "${MATCHING_FAIRNESS_OVERWRITE:-0}" != "0" ]]; then
    echo "error: MATCHING_FAIRNESS_OVERWRITE must be 0 or 1" >&2
    exit 64
fi
exec python "${arguments[@]}"
