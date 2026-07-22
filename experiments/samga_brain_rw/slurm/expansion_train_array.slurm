#!/bin/bash
#SBATCH --job-name=s1-expand
#SBATCH --partition=i64m1tga40u
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-18:00:00
#SBATCH --output=logs/samga_brain_rw/stage1_expansion_%A_%a.out
#SBATCH --error=logs/samga_brain_rw/stage1_expansion_%A_%a.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:?set the verified repository root}
cd -- "${PROJECT_ROOT}"

JOB_MAP=${JOB_MAP:?set JOB_MAP to the sealed Stage 1 expansion map}
JOB_MAP_SHA256=${JOB_MAP_SHA256:?set JOB_MAP_SHA256 to its sealed hash}
FIRST_INDEX=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
ARRAY_MIN=${JOB_MAP_ARRAY_MIN:?set immutable full job-map minimum}
ARRAY_MAX=${JOB_MAP_ARRAY_MAX:?set immutable full job-map maximum}
CHUNK_STRIDE=${JOB_MAP_CHUNK_STRIDE:?set immutable chunk stride}
if [[ ! ${CHUNK_STRIDE} =~ ^[1-9][0-9]*$ ]]; then
  echo "JOB_MAP_CHUNK_STRIDE must be a positive integer" >&2
  exit 2
fi
if (( FIRST_INDEX < ARRAY_MIN || FIRST_INDEX > ARRAY_MAX )); then
  echo "SLURM array task is outside the sealed job-map bounds" >&2
  exit 2
fi

export PYTHONPATH="${PROJECT_ROOT}/experiments/samga_brain_rw"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

set +u
source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate test
set -u

for ((
  ARRAY_INDEX=FIRST_INDEX;
  ARRAY_INDEX<=ARRAY_MAX;
  ARRAY_INDEX+=CHUNK_STRIDE
)); do
  timeout --signal=TERM --kill-after=5m 4h \
    python \
    "${PROJECT_ROOT}/experiments/samga_brain_rw/scripts/build_job_map.py" run-row \
    --job-map "${JOB_MAP}" \
    --job-map-sha256 "${JOB_MAP_SHA256}" \
    --array-index "${ARRAY_INDEX}" \
    --array-min "${ARRAY_MIN}" \
    --array-max "${ARRAY_MAX}"
done
