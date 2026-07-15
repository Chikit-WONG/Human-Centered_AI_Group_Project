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

# The default 0-39 array is exactly the 4 seeds below × 10 subjects. Seed 42 is
# deliberately absent because all ten subjects were already verified. A custom
# seed count requires an explicit matching submission range, for example:
#   sbatch --array=0-19%2 scripts/submit_multiseed_array.slurm formal 51 52
if (($#)); then
    SEEDS=("$@")
else
    SEEDS=(43 44 45 46)
fi
readonly MODE SEEDS

for candidate_seed in "${SEEDS[@]}"; do
    [[ "${candidate_seed}" =~ ^[0-9]+$ ]] || {
        echo "error: every seed must be a non-negative integer: ${candidate_seed}" >&2
        exit 2
    }
done

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
for array_field in \
    SLURM_ARRAY_TASK_MIN SLURM_ARRAY_TASK_MAX \
    SLURM_ARRAY_TASK_STEP SLURM_ARRAY_TASK_COUNT; do
    [[ "${!array_field:-}" =~ ^[0-9]+$ ]] || {
        echo "error: missing or invalid ${array_field}; submit this script with sbatch --array" >&2
        exit 2
    }
done
readonly EXPECTED_ARRAY_MAX="$((TOTAL_TASKS - 1))"
if ((SLURM_ARRAY_TASK_MIN != 0 \
    || SLURM_ARRAY_TASK_MAX != EXPECTED_ARRAY_MAX \
    || SLURM_ARRAY_TASK_STEP != 1 \
    || SLURM_ARRAY_TASK_COUNT != TOTAL_TASKS)); then
    echo "error: ${#SEEDS[@]} seed(s) require exactly ${TOTAL_TASKS} array tasks (0-${EXPECTED_ARRAY_MAX}); submitted array is min=${SLURM_ARRAY_TASK_MIN}, max=${SLURM_ARRAY_TASK_MAX}, step=${SLURM_ARRAY_TASK_STEP}, count=${SLURM_ARRAY_TASK_COUNT}" >&2
    echo "resubmit with: sbatch --array=0-${EXPECTED_ARRAY_MAX}%2 scripts/submit_multiseed_array.slurm ${MODE} ${SEEDS[*]}" >&2
    exit 2
fi
if ((TASK_INDEX >= TOTAL_TASKS)); then
    echo "error: task ${TASK_INDEX} is outside 0..$((TOTAL_TASKS - 1))" >&2
    exit 2
fi

readonly SEED_INDEX="$((TASK_INDEX / 10))"
readonly SUBJECT_ID="$((TASK_INDEX % 10 + 1))"
readonly SEED="${SEEDS[SEED_INDEX]}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "array_task_index=${TASK_INDEX}"
echo "subject_id=${SUBJECT_ID}"
echo "seed=${SEED}"

srun --unbuffered bash "${PROJECT_ROOT}/scripts/run_subject_reproduction.sh" \
    "${MODE}" --subject-id "${SUBJECT_ID}" --seed "${SEED}"
