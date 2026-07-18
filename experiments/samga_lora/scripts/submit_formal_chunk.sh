#!/usr/bin/env bash
set -euo pipefail

START="${1:?usage: submit_formal_chunk.sh START [PARTITION], where START is 0,10,...,90}"
PARTITION="${2:-i64m1tga40u}"
if ! [[ "${START}" =~ ^[0-9]+$ ]] || (( START < 0 || START > 90 || START % 10 != 0 )); then
  echo "START must be one of 0,10,...,90" >&2
  exit 2
fi
case "${PARTITION}" in
  i64m1tga40u|i64m1tga40ue|emergency_gpua40) ;;
  *) echo "Unsupported A40 partition: ${PARTITION}" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${EXPERIMENT_ROOT}/../.." && pwd)"
LOCKED_CONFIG="${PROJECT_ROOT}/artifacts/samga_lora/pilot_selection.json"

python - "${LOCKED_CONFIG}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    locked = json.load(handle)
if not locked.get("gate_passed"):
    raise SystemExit("Pilot gate is absent or did not pass")
PY

if squeue -h -u "${USER}" -n samga-formal | grep -q .; then
  echo "A samga-formal chunk is already submitted; wait for it to finish" >&2
  exit 3
fi

END=$((START + 9))
if (( START < 50 )); then
  TIME_LIMIT="00:30:00"
else
  TIME_LIMIT="01:00:00"
fi
cd "${PROJECT_ROOT}"
sbatch --array="${START}-${END}%10" --time="${TIME_LIMIT}" --partition="${PARTITION}" \
  "${EXPERIMENT_ROOT}/slurm/formal_array.slurm"
