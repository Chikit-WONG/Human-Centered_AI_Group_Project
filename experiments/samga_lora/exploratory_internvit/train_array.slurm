#!/usr/bin/env bash
#SBATCH --job-name=samga-internvit
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --array=0-9%2
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=logs/samga_lora/samga-internvit-%A_%a.out
#SBATCH --error=logs/samga_lora/samga-internvit-%A_%a.err

set -euo pipefail
PROJECT_ROOT="${SAMGA_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?submit from the project root}}"
cd "${PROJECT_ROOT}"
bash "${PROJECT_ROOT}/experiments/samga_lora/exploratory_internvit/run_cell.sh" "${SLURM_ARRAY_TASK_ID}"
