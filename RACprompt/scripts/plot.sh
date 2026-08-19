#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/rac_opd.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/storage-shared/nlp/minhpn19/RACprompt/outputs}"

if [[ -z "${RUN_NAME:-}" ]]; then
  LATEST_RUN_FILE="${OUTPUT_ROOT}/latest_run.txt"
  if [[ ! -s "${LATEST_RUN_FILE}" ]]; then
    echo "Cannot find latest run: missing ${LATEST_RUN_FILE}. Set RUN_NAME explicitly." >&2
    exit 1
  fi
  IFS= read -r RUN_NAME < "${LATEST_RUN_FILE}"
fi

echo "Plotting run: ${RUN_NAME}"
python3 analyze.py \
  --config "${CONFIG}" \
  --set "training.run_name=${RUN_NAME}" \
  --set "paths.output_root=${OUTPUT_ROOT}" \
  "$@"
