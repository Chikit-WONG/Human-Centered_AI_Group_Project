#!/usr/bin/env bash
#SBATCH --job-name=brainrw-all-subs
#SBATCH --partition=debug
#SBATCH --array=1-10%2
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --chdir=/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/Class/AIAA3800_L01-Human-Centered_Artificial_Intelligence/Final_Project/Human-Centered_AI_Group_Project
#SBATCH --output=/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/Class/AIAA3800_L01-Human-Centered_Artificial_Intelligence/Final_Project/Human-Centered_AI_Group_Project/logs/slurm-all-subjects-%A_%a.out
#SBATCH --error=/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/Class/AIAA3800_L01-Human-Centered_Artificial_Intelligence/Final_Project/Human-Centered_AI_Group_Project/logs/slurm-all-subjects-%A_%a.err

set -Eeuo pipefail

readonly PROJECT_ROOT="/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/Class/AIAA3800_L01-Human-Centered_Artificial_Intelligence/Final_Project/Human-Centered_AI_Group_Project"
MODE="${1:-formal}"
if (($#)); then
    shift
fi

[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || {
    echo "error: submit_subject_array.slurm must run as a SLURM array" >&2
    exit 2
}

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

srun --unbuffered bash "${PROJECT_ROOT}/scripts/run_subject_reproduction.sh" \
    "${MODE}" --subject-id "${SLURM_ARRAY_TASK_ID}" "$@"
