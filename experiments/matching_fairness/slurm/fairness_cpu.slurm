#!/usr/bin/env bash
#SBATCH --job-name=mf3_fairness_cpu
#SBATCH --partition=i64m512u
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/Class/AIAA3800_L01-Human-Centered_Artificial_Intelligence/Final_Project/test/brain-rw/logs/matching_fairness_v3/fairness_cpu_%j.out
#SBATCH --error=/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/Class/AIAA3800_L01-Human-Centered_Artificial_Intelligence/Final_Project/test/brain-rw/logs/matching_fairness_v3/fairness_cpu_%j.err

set -Eeuo pipefail
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly EXPERIMENT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
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
    "${EXPERIMENT_ROOT}/scripts/submit_pipeline.py"
    --internal-cell final
)
if [[ "${MATCHING_FAIRNESS_OVERWRITE:-0}" == "1" ]]; then
    arguments+=(--overwrite)
elif [[ "${MATCHING_FAIRNESS_OVERWRITE:-0}" != "0" ]]; then
    echo "error: MATCHING_FAIRNESS_OVERWRITE must be 0 or 1" >&2
    exit 64
fi
exec python "${arguments[@]}"
