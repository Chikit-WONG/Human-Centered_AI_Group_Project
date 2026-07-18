#!/usr/bin/env bash
#SBATCH --job-name=samga-smoke
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:a40:1
#SBATCH --array=0-1
#SBATCH --output=logs/samga_lora/%x-%A_%a.out
#SBATCH --error=logs/samga_lora/%x-%A_%a.err

set -euo pipefail
PROJECT_ROOT="${SAMGA_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?submit from the project root}}"
cd "${PROJECT_ROOT}"
SCRIPT_DIR="${PROJECT_ROOT}/experiments/samga_lora/scripts"
bash "${SCRIPT_DIR}/run_smoke_cell.sh" "${SLURM_ARRAY_TASK_ID}"
