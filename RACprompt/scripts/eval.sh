#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/rac_opd.yaml}"
CHECKPOINT="${CHECKPOINT:-latest}"
EVAL_SAMPLES="${EVAL_SAMPLES:-4}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-2048}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/storage-shared/nlp/minhpn19/RACprompt/outputs}"

if [[ -z "${RUN_NAME:-}" ]]; then
  LATEST_RUN_FILE="${OUTPUT_ROOT}/latest_run.txt"
  if [[ ! -s "${LATEST_RUN_FILE}" ]]; then
    echo "Cannot find latest run: missing ${LATEST_RUN_FILE}. Set RUN_NAME explicitly." >&2
    exit 1
  fi
  IFS= read -r RUN_NAME < "${LATEST_RUN_FILE}"
fi

echo "Evaluating run: ${RUN_NAME}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "-1" ]]; then
  IFS=',' read -r -a RAC_VISIBLE_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
  NUM_GPUS="${#RAC_VISIBLE_GPUS[@]}"
else
  NUM_GPUS="$(python3 -c 'import torch; print(torch.cuda.device_count())')"
fi
if [[ "${NUM_GPUS}" -lt 1 ]]; then
  echo "No visible CUDA GPU for mandatory vLLM evaluation." >&2
  exit 1
fi

python3 evaluate.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --set "training.run_name=${RUN_NAME}" \
  --set "paths.output_root=${OUTPUT_ROOT}" \
  --set "evaluation.num_samples_per_problem=${EVAL_SAMPLES}" \
  --set "evaluation.max_new_tokens=${EVAL_MAX_NEW_TOKENS}" \
  --set "evaluation.tensor_parallel_size=1" \
  --set "evaluation.data_parallel_size=${NUM_GPUS}" \
  "$@"
