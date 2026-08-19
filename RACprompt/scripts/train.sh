#!/usr/bin/env bash
set -euo pipefail

# Common experiment controls (override as environment variables).
LR="${LR:-1e-6}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
MAX_STEPS="${MAX_STEPS:-0}"
EXTRA_STEPS="${EXTRA_STEPS:-100}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-1024}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
EVAL_EVERY="${EVAL_EVERY:-50}"
SAVE_EVERY="${SAVE_EVERY:-50}"
CRITICAL_STATES="${CRITICAL_STATES:-24}"
M_BRANCHES="${M_BRANCHES:-4}"
ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-auto}"
RUN_NAME="${RUN_NAME:-rac_opd_qwen3}"
CONFIG="${CONFIG:-configs/rac_opd.yaml}"
RESUME="${RESUME:-}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "-1" ]]; then
  IFS=',' read -r -a RAC_VISIBLE_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
  NUM_GPUS="${#RAC_VISIBLE_GPUS[@]}"
else
  NUM_GPUS="$(python3 -c 'import torch; print(torch.cuda.device_count())')"
fi
if [[ "${NUM_GPUS}" -lt 1 ]]; then
  echo "No visible CUDA GPU. Set CUDA_VISIBLE_DEVICES before training." >&2
  exit 1
fi

ARGS=(
  --config "${CONFIG}"
  --set "training.learning_rate=${LR}"
  --set "training.global_batch_size=${GLOBAL_BATCH_SIZE}"
  --set "training.max_steps=${MAX_STEPS}"
  --set "training.extra_steps=${EXTRA_STEPS}"
  --set "training.run_name=${RUN_NAME}"
  --set "data.max_prompt_tokens=${MAX_PROMPT_TOKENS}"
  --set "rollout.max_new_tokens=${MAX_NEW_TOKENS}"
  --set "evaluation.every_steps=${EVAL_EVERY}"
  --set "checkpoint.save_every_steps=${SAVE_EVERY}"
  --set "critical.target=${CRITICAL_STATES}"
  --set "critical.branch_candidates=${M_BRANCHES}"
  --set "rollout.backend=${ROLLOUT_BACKEND}"
)
if [[ -n "${RESUME}" ]]; then
  ARGS+=(--resume "${RESUME}")
fi

torchrun --standalone --nproc_per_node="${NUM_GPUS}" train.py "${ARGS[@]}" "$@"

