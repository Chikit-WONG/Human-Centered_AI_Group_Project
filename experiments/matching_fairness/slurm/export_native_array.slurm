#!/usr/bin/env bash
#SBATCH --job-name=mf3_export_native
#SBATCH --partition=i64m1tga40u
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/Class/AIAA3800_L01-Human-Centered_Artificial_Intelligence/Final_Project/test/brain-rw/logs/matching_fairness_v3/export_native_%A_%a.out
#SBATCH --error=/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/Class/AIAA3800_L01-Human-Centered_Artificial_Intelligence/Final_Project/test/brain-rw/logs/matching_fairness_v3/export_native_%A_%a.err

set -Eeuo pipefail
case "${SLURM_ARRAY_TASK_ID:-}" in
    0|1) ;;
    *)
        echo "error: invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-unset}; expected 0 or 1" >&2
        exit 64
        ;;
esac
readonly EXPORT_MODE="${MATCHING_FAIRNESS_EXPORT_MODE:-main}"
case "${EXPORT_MODE}" in
    main|audit) ;;
    *)
        echo "error: invalid MATCHING_FAIRNESS_EXPORT_MODE=${EXPORT_MODE}" >&2
        exit 64
        ;;
esac

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
conda activate atm_native

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
    --internal-cell export-native
    --array-id "${SLURM_ARRAY_TASK_ID}"
    --export-mode "${EXPORT_MODE}"
)
if [[ "${MATCHING_FAIRNESS_OVERWRITE:-0}" == "1" ]]; then
    arguments+=(--overwrite)
elif [[ "${MATCHING_FAIRNESS_OVERWRITE:-0}" != "0" ]]; then
    echo "error: MATCHING_FAIRNESS_OVERWRITE must be 0 or 1" >&2
    exit 64
fi
exec python "${arguments[@]}"
