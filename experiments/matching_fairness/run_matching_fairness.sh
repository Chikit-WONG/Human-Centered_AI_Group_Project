#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly SUBMITTER="${SCRIPT_DIR}/scripts/submit_pipeline.py"

[[ -f "${SUBMITTER}" ]] || {
    echo "error: missing fixed pipeline submitter: ${SUBMITTER}" >&2
    exit 1
}

export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
exec python "${SUBMITTER}" "$@"
