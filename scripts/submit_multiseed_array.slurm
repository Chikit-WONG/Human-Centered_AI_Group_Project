#!/usr/bin/env bash
#SBATCH --job-name=brainrw-5seed
#SBATCH --partition=debug
#SBATCH --array=0-39%2
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --chdir=/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/Class/AIAA3800_L01-Human-Centered_Artificial_Intelligence/Final_Project/Human-Centered_AI_Group_Project
#SBATCH --output=/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/Class/AIAA3800_L01-Human-Centered_Artificial_Intelligence/Final_Project/Human-Centered_AI_Group_Project/logs/slurm-5seed-%A_%a.out
#SBATCH --error=/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/Class/AIAA3800_L01-Human-Centered_Artificial_Intelligence/Final_Project/Human-Centered_AI_Group_Project/logs/slurm-5seed-%A_%a.err

set -Eeuo pipefail

readonly PROJECT_ROOT="/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/Class/AIAA3800_L01-Human-Centered_Artificial_Intelligence/Final_Project/Human-Centered_AI_Group_Project"
MODE="${1:-formal}"
if (($#)); then
    shift
fi

# seed 42 is deliberately absent: all ten subjects were already verified.
# Positional arguments allow the same script to run a different seed list.
if (($#)); then
    SEEDS=("$@")
else
    SEEDS=(43 44 45 46)
fi
readonly MODE SEEDS

[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || {
    echo "error: submit_multiseed_array.slurm must run as a SLURM array" >&2
    exit 2
}
[[ "${SLURM_ARRAY_TASK_ID}" =~ ^[0-9]+$ ]] || {
    echo "error: invalid SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
}

readonly TASK_INDEX="${SLURM_ARRAY_TASK_ID}"
readonly TOTAL_TASKS="$(( ${#SEEDS[@]} * 10 ))"
if ((TASK_INDEX >= TOTAL_TASKS)); then
    echo "error: task ${TASK_INDEX} is outside 0..$((TOTAL_TASKS - 1))" >&2
    exit 2
fi

readonly SEED_INDEX="$((TASK_INDEX / 10))"
readonly SUBJECT_ID="$((TASK_INDEX % 10 + 1))"
readonly SEED="${SEEDS[SEED_INDEX]}"
[[ "${SEED}" =~ ^[0-9]+$ ]] || {
    echo "error: seed must be an integer: ${SEED}" >&2
    exit 2
}

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "array_task_index=${TASK_INDEX}"
echo "subject_id=${SUBJECT_ID}"
echo "seed=${SEED}"

srun --unbuffered bash "${PROJECT_ROOT}/scripts/run_subject_reproduction.sh" \
    "${MODE}" --subject-id "${SUBJECT_ID}" --seed "${SEED}"
