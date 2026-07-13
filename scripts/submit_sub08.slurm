#!/usr/bin/env bash
#SBATCH --job-name=brainrw-s08
#SBATCH --partition=debug
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --chdir=/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/Class/AIAA3800_L01-Human-Centered_Artificial_Intelligence/Final_Project/test/brain-rw
#SBATCH --output=/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/Class/AIAA3800_L01-Human-Centered_Artificial_Intelligence/Final_Project/test/brain-rw/logs/slurm-sub08-%j.out
#SBATCH --error=/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/Class/AIAA3800_L01-Human-Centered_Artificial_Intelligence/Final_Project/test/brain-rw/logs/slurm-sub08-%j.err

set -Eeuo pipefail

readonly PROJECT_ROOT="/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/Class/AIAA3800_L01-Human-Centered_Artificial_Intelligence/Final_Project/test/brain-rw"

# Submit the smoke test first:
#   sbatch scripts/submit_sub08.slurm smoke
# Once it succeeds, request the full 25-epoch run with:
#   sbatch scripts/submit_sub08.slurm formal
MODE="${1:-smoke}"
if (($#)); then
    shift
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

srun --unbuffered bash "${PROJECT_ROOT}/scripts/run_sub08_reproduction.sh" \
    "${MODE}" "$@"
