# RAC-OPD: Recoverability-Aware Curriculum for On-Policy Distillation

This repository implements the dynamic RAC-OPD curriculum for Qwen3. The behavior rollout is always sampled from the current student. Sparse counterfactual teacher-preferred branches are one-token, `no_grad` diagnostics that update only the *future* prompt distribution; they never modify rollout tokens, become training targets, or weight the current loss.

## Method and implementation

- Rank 0 owns the curriculum RNG, samples exactly 32 prompt indices with replacement, broadcasts them, and splits them unevenly when necessary.
- Prompt memory starts at 0.5, optionally ages toward 0.5, and mixes softmax priority with 10% uniform exploration.
- Each prompt gets one current-policy rollout. The default is HF/PyTorch generation with BF16 and current in-memory weights.
- `Dplus` is exact full-vocabulary positive correction mass; `C` is teacher mass on student top-32 support.
- The default 24 critical states combine 12 segment maxima, 8 global `Dplus` peaks, 4 compatibility changes, NMS, and gap-relaxed fill.
- Critical-state selection never uses a TA-OPD `D*C` score.
- Up to four strictly positive teacher-preferred tokens inside student top-32 support create one-step diagnostic branches.
- Branch scoring reuses the critical-prefix KV cache where the installed Transformers cache API supports it; a batched fallback is explicit and logged.
- `A`, weighted next-state compatibility `F`, and bridgeability `B=A*F` produce bottleneck-sensitive geometric recoverability `R`.
- Prompt priority is `T=G*R`; EMA updates affect only later sampling, so mastered prompts naturally lose priority.
- Training re-forwards only the original rollout and minimizes token-mean student-top-16 reverse KL, with no curriculum loss weights or token mask.
- DDP loss scaling remains the exact global mean for uneven rank splits and never changes LR or global batch with GPU count.
- Checkpoints include student/optimizer/scheduler, curriculum arrays, per-rank RNG states, schema metadata, and resolved configuration.
- Full MATH-500, AIME 2024, and AIME 2025 evaluation uses vLLM at step 0, each interval, and final step.
- Logs, raw evaluation generations, prompt/critical-state CSVs, TensorBoard data, PNGs, and PDFs are produced automatically.

## Fresh B200 environment

Run on the B200 host from a fresh environment:

```bash
cd /workspace/storage-shared/nlp/minhpn19/RACprompt

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
pip install vllm==0.23.0 --extra-index-url https://download.pytorch.org/whl/cu129
pip install -r requirements.txt
```

The dedicated vLLM installation line is intentional. B200/GB200 needs CUDA 12.8 or newer, and the vLLM 0.23.0 CUDA 12.9 wheel brings its compatible PyTorch 2.11 build. Do not first install an unrelated PyTorch wheel into this venv. If the B200 image is standardized on another CUDA ABI, use the matching wheel from the [official vLLM GPU installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/) and keep vLLM/PyTorch paired.

Verify the environment and run unit tests:

```bash
python - <<'PY'
import torch, transformers, vllm
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("bf16", torch.cuda.is_bf16_supported(), "gpus", torch.cuda.device_count())
print("transformers", transformers.__version__, "vllm", vllm.__version__)
PY
python -m pytest -q
```

## Training commands

The launcher derives process count only from currently visible GPUs and keeps global batch 32.

```bash
# 1 GPU
CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh

# 2 GPUs
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train.sh

# 3 GPUs (split 11/11/10, still exactly 32 globally)
CUDA_VISIBLE_DEVICES=0,1,2 bash scripts/train.sh

# 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train.sh
```

Common overrides are environment variables; no Python edits are needed:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
LR=8e-7 \
GLOBAL_BATCH_SIZE=32 \
MAX_STEPS=700 \
EXTRA_STEPS=100 \
EVAL_EVERY=25 \
SAVE_EVERY=25 \
MAX_PROMPT_TOKENS=1024 \
MAX_NEW_TOKENS=2048 \
CRITICAL_STATES=24 \
ROLLOUT_BACKEND=auto \
RUN_NAME=rac_opd_qwen3_exp1 \
bash scripts/train.sh
```

`MAX_STEPS=0` resolves to `ceil(N/GLOBAL_BATCH_SIZE)+EXTRA_STEPS`. `CRITICAL_STATES` accepts 1–50; 24 is the research default. `M_BRANCHES` is intentionally fixed at 4 by config validation. Additional settings can be appended as normal CLI overrides:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh \
  --set curriculum.enable_staleness_decay=false \
  --set rollout.temperature=1.0 \
  --set critical.stats_top_k=32
```

The training vLLM fast path is guarded because public colocated weight-transfer APIs are version-specific. In this delivered version, `ROLLOUT_BACKEND=auto` explicitly logs why it selects the exact current-weight Transformers backend. It never reloads vLLM from disk per step and never silently permits stale rollouts. `ROLLOUT_BACKEND=vllm` fails until a version-pinned, validated in-memory weight-sync adapter is supplied.

## Resume

Resume uses the next optimizer step and does not repeat step-0 evaluation. Exact RNG continuation requires the same GPU/world size as the saved run.

```bash
# Latest checkpoint in outputs/<RUN_NAME>/checkpoints
CUDA_VISIBLE_DEVICES=0,1,2 RUN_NAME=rac_opd_qwen3 RESUME=latest bash scripts/train.sh

# Explicit checkpoint
CUDA_VISIBLE_DEVICES=0,1,2 RUN_NAME=rac_opd_qwen3 \
RESUME=/workspace/storage-shared/nlp/minhpn19/RACprompt/outputs/rac_opd_qwen3/checkpoints/step_000100 \
bash scripts/train.sh
```

## Standalone full evaluation

All three complete datasets are evaluated; `EVAL_SAMPLES=4` is the default. The script requests vLLM data-parallel replicas (`TP=1`) across all visible GPUs when supported by the installed offline API.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 RUN_NAME=rac_opd_qwen3 CHECKPOINT=latest bash scripts/eval.sh

# Evaluate an explicit model directory as step 0
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --model /workspace/storage-shared/models/Qwen3-1.7B \
  --step 0
```

## Regenerate plots

```bash
RUN_NAME=rac_opd_qwen3 bash scripts/plot.sh

# Or point directly at a run directory
python analyze.py \
  --output_dir /workspace/storage-shared/nlp/minhpn19/RACprompt/outputs/rac_opd_qwen3
```

## Local smoke mode

Real defaults always use the B200 paths. For a tiny local model/dataset, override paths explicitly; dry run then caps the pool at eight records, uses one step and at most 32 response tokens, and disables full evaluation:

```bash
python train.py --dry_run \
  --set paths.student_model=/path/to/tiny/causal-lm \
  --set paths.teacher_model=/path/to/tiny/causal-lm \
  --set paths.train_data=/path/to/tiny.jsonl \
  --set paths.output_root=/tmp/racprompt_outputs
```

## Output layout

Runs are written to `/workspace/storage-shared/nlp/minhpn19/RACprompt/outputs/<run_name>/` with resolved config, JSONL/CSV/TensorBoard logs, per-step full evaluation generations, resumable checkpoints, prompt usage and critical-state analysis CSVs, and all required PNG/PDF plots.

