#!/bin/bash
#SBATCH --job-name=samga-preflight
#SBATCH --partition=debug
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=logs/samga_brain_rw/preflight_%j.out
#SBATCH --error=logs/samga_brain_rw/preflight_%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:?set the verified repository root}
cd -- "${PROJECT_ROOT}"

JOB_MAP=${JOB_MAP:?set JOB_MAP to the sealed preflight job map}
JOB_MAP_SHA256=${JOB_MAP_SHA256:?set JOB_MAP_SHA256 to its sealed hash}
ARRAY_INDEX=${SLURM_ARRAY_TASK_ID:-0}
ARRAY_MIN=${JOB_MAP_ARRAY_MIN:?set immutable full job-map minimum}
ARRAY_MAX=${JOB_MAP_ARRAY_MAX:?set immutable full job-map maximum}

export PYTHONPATH="${PROJECT_ROOT}/experiments/samga_brain_rw"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

set +u
source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate test
set -u

python \
  "${PROJECT_ROOT}/experiments/samga_brain_rw/scripts/build_job_map.py" run-row \
  --job-map "${JOB_MAP}" \
  --job-map-sha256 "${JOB_MAP_SHA256}" \
  --array-index "${ARRAY_INDEX}" \
  --array-min "${ARRAY_MIN}" \
  --array-max "${ARRAY_MAX}"
