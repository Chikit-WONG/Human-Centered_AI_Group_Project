#!/usr/bin/env bash
#SBATCH --job-name=samga-formal
#SBATCH --partition=i64m1tga40u
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:a40:1
# The cluster's normal QOS permits at most ten submitted jobs per user. Submit
# the ten contiguous chunks 0-9, 10-19, ..., 90-99 sequentially with
# scripts/submit_formal_chunk.sh.
#SBATCH --array=0-9%10
#SBATCH --output=logs/samga_lora/%x-%A_%a.out
#SBATCH --error=logs/samga_lora/%x-%A_%a.err

set -euo pipefail
PROJECT_ROOT="${SAMGA_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?submit from the project root}}"
cd "${PROJECT_ROOT}"
SCRIPT_DIR="${PROJECT_ROOT}/experiments/samga_lora/scripts"
bash "${SCRIPT_DIR}/run_formal_cell.sh" "${SLURM_ARRAY_TASK_ID}"
