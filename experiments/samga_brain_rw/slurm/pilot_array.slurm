#!/bin/bash
#SBATCH --job-name=samga-pilot
#SBATCH --partition=i64m1tga40u
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/samga_brain_rw/pilot_%A_%a.out
#SBATCH --error=logs/samga_brain_rw/pilot_%A_%a.err

set -euo pipefail

JOB_MAP=${JOB_MAP:?set JOB_MAP to the sealed pilot job map}
JOB_MAP_SHA256=${JOB_MAP_SHA256:?set JOB_MAP_SHA256 to its sealed hash}
ARRAY_INDEX=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
ARRAY_MIN=${JOB_MAP_ARRAY_MIN:?set immutable full job-map minimum}
ARRAY_MAX=${JOB_MAP_ARRAY_MAX:?set immutable full job-map maximum}

export PYTHONPATH=experiments/samga_brain_rw
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate test

python experiments/samga_brain_rw/scripts/build_job_map.py run-row \
  --job-map "${JOB_MAP}" \
  --job-map-sha256 "${JOB_MAP_SHA256}" \
  --array-index "${ARRAY_INDEX}" \
  --array-min "${ARRAY_MIN}" \
  --array-max "${ARRAY_MAX}"
