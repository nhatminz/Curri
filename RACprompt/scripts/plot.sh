#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/rac_opd.yaml}"
RUN_NAME="${RUN_NAME:-rac_opd_qwen3}"
python3 analyze.py --config "${CONFIG}" --set "training.run_name=${RUN_NAME}" "$@"

