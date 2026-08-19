#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/rac_opd.yaml}"
RUN_NAME="${RUN_NAME:-rac_opd_qwen3}"
CHECKPOINT="${CHECKPOINT:-latest}"
EVAL_SAMPLES="${EVAL_SAMPLES:-4}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-2048}"

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
  --set "evaluation.num_samples_per_problem=${EVAL_SAMPLES}" \
  --set "evaluation.max_new_tokens=${EVAL_MAX_NEW_TOKENS}" \
  --set "evaluation.tensor_parallel_size=1" \
  --set "evaluation.data_parallel_size=${NUM_GPUS}" \
  "$@"

