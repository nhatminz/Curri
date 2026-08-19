# Codex Implementation Prompt — RAC-OPD Dynamic Recoverability Curriculum

You are implementing a research-grade training codebase for a new **Recoverability-Aware Curriculum for On-Policy Distillation (RAC-OPD)**. Work in the local development directory:

`/mnt/hdd/nhatminh/RACprompt/`

The code will later be copied to the B200 machine and run from:

`/workspace/storage-shared/nlp/minhpn19/RACprompt`

All runtime defaults/configs must therefore use the **B200 paths below**, not the local development paths. Do not assume the B200 data/model paths exist on the local machine. Build the project so it can be syntax-tested/unit-tested locally without loading the real models.

Do not simplify the algorithm into TA-OPD, PACED, or static prompt ranking. The main contribution is a **dynamic prompt curriculum based on one-step recoverability of teacher-preferred corrections**.

---

## 1. Research goal and non-negotiable method semantics

We train:

- Student: `/workspace/storage-shared/models/Qwen3-1.7B`
- Frozen teacher: `/workspace/storage-shared/models/Qwen3-8B`
- Training data: `/workspace/storage-shared/nlp/minhpn19/data/DAPO-Math-17k-Processed`
- Use the **entire training dataset as the prompt pool**.

Evaluation datasets:

- MATH-500: `/workspace/storage-shared/nlp/minhpn19/data/eval/math500`
- AIME 2024: `/workspace/storage-shared/nlp/minhpn19/data/eval/aime24`
- AIME 2025: `/workspace/storage-shared/nlp/minhpn19/data/eval/aime25`

### Core idea

For a prompt `q`, the student first produces a normal on-policy rollout. After the full rollout is available, identify sparse but informative **critical states**. At each critical state, ask:

> If the student followed a teacher-preferred correction that is currently reachable by the student, would the **next state** become easier for teacher–student distillation?

This is a **counterfactual diagnostic only**.

### Absolutely do NOT do the following

1. **Do not use `B_t` to alter the token sampled by the main student rollout.**
2. **Do not teacher-guide the behavior policy.**
3. **Do not train on the counterfactual branches.**
4. **Do not weight the current batch OPD loss by `T(q)` in the main implementation.**
5. **Do not use TA-OPD score to choose critical states.**
6. **Do not pre-generate teacher trajectories for the whole dataset.**
7. **Do not perform full teacher continuation rollouts from every student prefix.**
8. **Do not use stale student weights for on-policy rollouts.**

The main rollout must remain:

`y ~ pi_student_current(. | q)`

The counterfactual branches are `no_grad` measurements used only to update the **future prompt-sampling distribution**.

All measurements for iteration `k` must be computed using the same student checkpoint `theta_k`, before the optimizer changes the model to `theta_{k+1}`.

---

# 2. Exact per-iteration pipeline

Implement the training loop in this order.

## Step 1 — Sample a global batch of prompts

Maintain one dynamic memory score per training prompt:

`M_q in [0, 1]`

Initialize all prompts optimistically with:

`M_q = 0.5`

At training step `k`, optionally age stale scores toward the prior:

`M_eff(q,k) = M0 + (M_q - M0) * exp(-(k - last_seen_q) / age_tau_steps)`

Default:

- `M0 = 0.5`
- `age_tau_steps = 200`
- make staleness decay configurable and easy to disable.

Convert effective scores to a normalized categorical sampling distribution:

`P_k(q) = eps_explore / N + (1 - eps_explore) * softmax(M_eff(q,k) / curriculum_temperature)`

Defaults:

- `eps_explore = 0.10`
- `curriculum_temperature = 0.25`

Requirements:

- sampling is **with replacement**;
- `sum_q P_k(q) == 1` numerically;
- no prompt should ever receive exactly zero probability;
- prompt probabilities must be recomputed as scores change;
- a prompt that becomes mastered must naturally lose priority when its new `T(q)` drops;
- prompt sampling semantics must be independent of GPU count.

**Distributed requirement:** rank 0 owns the global curriculum RNG and samples exactly `global_batch_size` prompt indices, then broadcasts/splits those indices across ranks. This prevents GPU count from changing the curriculum distribution.

Default global batch size:

`global_batch_size = 32`

Do not exceed 32 by default.

For world sizes that do not divide 32 (e.g. 3 or 5 GPUs), split the 32 examples unevenly across ranks (difference at most one) and correctly scale each rank's loss so the all-reduced gradient equals the mean over the exact 32-example global batch. Do not silently change global batch size and do not scale the learning rate with GPU count.

---

## Step 2 — Generate the normal student rollout

For each selected prompt, generate one student response from the **current** student policy.

Default training generation:

- `rollouts_per_prompt = 1`
- `max_new_tokens = 2048`
- `temperature = 1.0`
- `top_p = 1.0`
- repetition penalty `1.0`
- preserve the dataset/model's correct Qwen3 prompt/chat template; inspect dataset/model metadata instead of hardcoding an incompatible template.
- max prompt tokens default `1024`.

The rollout must remain truly on-policy with respect to the current student parameters.

### Rollout backend

Engineering priority is speed, but correctness is mandatory.

Implement a backend abstraction:

- `rollout_backend=auto`
- preferred fast path: **vLLM with exact current-weight synchronization**;
- correctness fallback: optimized Hugging Face/PyTorch batched generation.

If using vLLM for training rollouts:

- use a supported colocated weight-transfer mechanism (prefer current public vLLM CUDA-IPC weight transfer / sleep-mode APIs where available);
- sync the student weights after every optimizer step and before the next rollout;
- never silently use stale weights;
- use BF16, **no quantization**, unless explicitly enabled by a non-default research flag;
- if the installed vLLM version/API cannot provide reliable per-step current-weight synchronization, automatically fall back to the PyTorch/HF rollout backend and log a clear warning.

Do not restart/reload a full vLLM model from disk every training step.

---

## Step 3 — Score the completed student trajectory with student and teacher

Only after the rollout is complete, score each valid response position `t` using the same `theta_k` student and the frozen teacher:

`p_t(v) = pi_student(v | s_t)`

`q_t(v) = pi_teacher(v | s_t)`

where `s_t = (prompt, student_response_<t)`.

Teacher:

- always frozen;
- `.eval()`;
- `torch.no_grad()` / `torch.inference_mode()`;
- BF16.

For diagnostics, compute in numerically stable FP32 where appropriate.

Do not retain full-vocabulary logits for the entire batch longer than necessary. Chunk/microbatch scoring and immediately reduce to the statistics needed below.

---

# 3. Critical-state selection — use this method, NOT TA-OPD

Default target:

`critical_states_target = 24`

This number is intentionally not tiny because the machine may use B200 180GB GPUs and we care more about a strong signal than minimal diagnostic compute.

Use three complementary sources of critical states.

## 3.1 Teacher correction magnitude

At every valid response position compute the full-vocabulary positive teacher correction mass:

`Dplus_t = sum_v max(q_t(v) - p_t(v), 0)`

For normalized probability distributions this lies in `[0,1]` and equals total variation distance.

This is the main criticality signal.

Compute it without retaining both full-vocabulary probability tensors for all positions at once.

## 3.2 Current compatibility

Let:

`S_student_t = TopK(p_t, K_stats)`

with default:

`K_stats = 32`

Define:

`C_t = sum_{v in S_student_t} q_t(v)`

This is teacher probability mass lying on the student's current top-K support.

Use `C_t` only as a compatibility statistic and for detecting **changes**; do not multiply it by `Dplus_t` as TA-OPD does for selection.

## 3.3 Selector construction

Default parameters:

- `critical_states_target = 24`
- `critical_num_segments = 12`
- `critical_global_peaks = 8`
- `critical_change_points = 4`
- `critical_change_lag = 32`
- `critical_min_gap_tokens = 32`

Build the critical-state set in this order:

### A. Temporal coverage: 12 states

Split the valid response positions into 12 approximately equal temporal segments.

For each non-empty segment, choose the position with maximum `Dplus_t`.

This ensures coverage across early, middle, and late reasoning.

### B. Global correction peaks: +8 states

Add up to 8 highest-`Dplus_t` positions globally.

Use greedy non-maximum suppression so a newly added state must be at least `critical_min_gap_tokens` away from already selected states when possible.

### C. Compatibility change points: +4 states

For valid `t >= critical_change_lag`, compute:

`DeltaC_t = abs(C_t - C_{t-critical_change_lag})`

Add up to 4 largest `DeltaC_t` positions, again deduplicating and respecting the minimum-gap rule when possible.

### D. Fill logic

After deduplication, if fewer than 24 states were selected:

1. fill from remaining highest-`Dplus_t` positions respecting the gap;
2. if still fewer and the trajectory is long enough, progressively relax the gap;
3. if the trajectory itself has fewer than 24 valid positions, simply use all valid positions.

Track the selection reason(s) for each state:

- `segment`
- `global_peak`
- `change_point`
- `fill`

This metadata is required later for analysis plots.

Do not use TA-OPD score as the critical-state selector.

---

# 4. Counterfactual one-step recoverability at each critical state

At a selected critical state `t`, use the original `p_t`, `q_t`.

## 4.1 Current accessibility A_t

Let the student reachable set be:

`S_student_t = TopK(p_t, K_stats)`

Compute:

`A_t = [sum_{v in S_student_t} max(q_t(v)-p_t(v), 0)] / [Dplus_t + eps_num]`

Clamp numerical noise into `[0,1]`.

Interpretation: what fraction of the teacher's desired positive correction is already reachable inside the student's high-probability support?

## 4.2 Choose M counterfactual teacher-preferred reachable tokens

Set:

`M = 4`

This value is fixed as the main default.

For tokens in `S_student_t`, define:

`alpha_t(v) = max(q_t(v) - p_t(v), 0)`

Choose the top 4 tokens with largest strictly positive `alpha_t(v)`.

Exclude invalid control tokens (`pad`, `bos`, etc.). Do not create a lookahead branch after terminal EOS. If EOS is among candidates, handle it explicitly and log it rather than pretending it has a normal next state.

If fewer than 4 valid positive candidates exist, use the available candidates.

## 4.3 One-step counterfactual lookahead

For each selected token `v`, create only the hypothetical next prefix:

`s_t^v = s_t + v`

Then evaluate exactly one next-token distribution from both current student and frozen teacher:

`p_next^v(u) = pi_student(u | s_t^v)`

`q_next^v(u) = pi_teacher(u | s_t^v)`

Define next-state compatibility:

`Cplus_t_v = sum_{u in TopK(p_next^v, K_stats)} q_next^v(u)`

This branch is diagnostic only:

- `no_grad`;
- do not append it to the main rollout;
- do not use it as a training target;
- do not backpropagate through it.

### Engineering requirement for branch probes

Do **not** naively re-prefill the full prefix separately for all 4 branches.

Exploit prefix/KV reuse:

- prefill a critical prefix once per model when using the PyTorch path, then score the 4 candidate one-token continuations in a batch using cache reuse where supported;
- with vLLM, use prefix caching / batched requests so the 4 branches sharing a prefix do not repeat unnecessary prefill work;
- microbatch branch probes to stay memory-safe.

The implementation should be profiled. Log time spent in branch probing separately.

## 4.4 Future compatibility F_t

Use the current teacher correction mass as branch weights:

`F_t = sum_v alpha_t(v) * Cplus_t_v / [sum_v alpha_t(v) + eps_num]`

If no valid positive branch exists, mark the state invalid for recoverability aggregation rather than inventing a high score.

## 4.5 Transition bridgeability B_t

Define:

`B_t = A_t * F_t`

Clamp to `[0,1]`.

Interpretation:

- `A_t`: can the student follow the teacher correction now?
- `F_t`: if it follows such a correction, is the next state still compatible?
- `B_t`: is this teacher-guided transition recoverable/bridgeable?

---

# 5. Prompt-level score

Use only valid critical states.

## 5.1 Recoverability R(q)

Use a bottleneck-sensitive geometric mean:

`R(q) = exp(mean_t(log(clamp(B_t, 1e-6, 1.0))))`

Do not replace this with a plain arithmetic average in the main method.

## 5.2 Need-to-learn / corrective information G(q)

Use the mean teacher correction magnitude over the same valid critical states:

`G(q) = mean_t(Dplus_t)`

Since `Dplus_t in [0,1]`, keep this simple and interpretable.

This prevents prompts where student and teacher are already nearly identical from receiving high priority merely because they are compatible.

## 5.3 Final prompt teachability T(q)

`T(q) = G(q) * R(q)`

Clamp numerical noise into `[0,1]`.

The intended lifecycle is:

- not ready: high `G`, low `R` -> low/moderate priority;
- ready to learn: high `G`, high `R` -> high priority;
- mastered: low `G`, even if high `R` -> priority falls again.

## 5.4 Update prompt memory

After diagnosing prompt `q` at step `k`:

`M_q <- beta * M_q + (1-beta) * T(q)`

Default:

`prompt_score_ema_beta = 0.80`

Update:

- `last_seen_q = k`
- `prompt_usage_count[q] += 1`
- optionally keep running stats of `G`, `R`, `T`, rollout length, number of valid critical states.

**Important:** the newly computed `T(q)` affects only future prompt sampling. It must not reweight the current batch's OPD loss in the main experiment.

---

# 6. OPD training objective

After the diagnostics for the current batch are complete, perform the actual training update on the **original student-generated trajectories only**.

Implement vanilla top-K reverse-KL OPD as the default, matching the common student-top-K formulation.

Default:

`opd_top_k = 16`

For each response state, let:

`S_t = TopK(p_student_t, opd_top_k)`

Restrict and renormalize student and teacher probabilities on `S_t`:

`p_bar_t`, `q_bar_t`

Then:

`L_t = KL(p_bar_t || q_bar_t)`

Aggregate using token mean over valid response positions and then the exact global-batch mean.

No TA-OPD token mask in the main method.

No `T(q)` loss weighting in the main method.

Teacher has no gradient.

Student is full fine-tuned unless a config explicitly requests otherwise.

### Training defaults

Use these defaults but expose them in config and the bash launcher:

- optimizer: AdamW, fused when safely supported
- learning rate: `1e-6`
- weight decay: `0.01`
- scheduler: constant by default
- max grad norm: `1.0`
- precision: BF16
- global batch size: `32`
- max prompt tokens: `1024`
- max response / rollout new tokens: `2048`
- rollout temperature: `1.0`
- rollout top-p: `1.0`
- rollouts per prompt: `1`
- no quantization for student or teacher by default
- gradient checkpointing: configurable, default OFF on B200 unless memory profiling shows it is needed, because it can reduce speed
- use FlashAttention/SDPA best available implementation without changing model semantics
- enable TF32 only for operations where it does not replace required BF16 model math unexpectedly; do not trade model quality for throughput.

### Memory-safe implementation order

Preferred robust implementation:

**Phase A: no-grad diagnostics**

1. rollout;
2. teacher/student scoring;
3. critical-state selection;
4. counterfactual branch probes;
5. compute `G`, `R`, `T`;
6. update curriculum memory.

Then free unnecessary diagnostic tensors/cache.

**Phase B: train forward/backward**

Re-forward the original rollout batch with gradients to compute the OPD loss, then backward and optimizer step.

Do not hold an enormous full-vocabulary training graph alive while running all counterfactual diagnostics.

---

# 7. Number of training steps

Sampling is with replacement, so do not define training only by classical epochs.

Let:

`N = len(full_training_dataset)`

Default automatic step count:

`max_steps = ceil(N / global_batch_size) + extra_steps`

Default:

`extra_steps = 100`

For roughly 17K prompts and batch 32 this is around 630 steps.

Implement:

- if config/bash `MAX_STEPS > 0`, use it exactly;
- if `MAX_STEPS <= 0`, compute the formula above;
- log the resolved number of steps.

Also report an “equivalent draws per dataset item” statistic:

`max_steps * global_batch_size / N`

---

# 8. Evaluation requirements

Evaluation must run on the **full** datasets, never a subset:

- full MATH-500
- full AIME 2024
- full AIME 2025

Use vLLM for evaluation for speed.

Mandatory eval steps:

1. **step 0 before any optimizer update**
2. every `eval_every_steps`
3. **final step**, even if it is not divisible by the interval.

Default:

`eval_every_steps = 50`

Evaluation defaults:

- BF16, no quantization
- `temperature = 0.6`
- `top_p = 0.95`
- `num_samples_per_problem = 4`
- `max_new_tokens = 2048`
- expose all in config/bash.

If periodic evaluation becomes the dominant runtime, keep the default at 4 rather than silently evaluating a subset. The user can change the sample count from the bash script.

For every problem save:

- prompt/problem ID
- generated response(s)
- extracted final answer(s)
- ground-truth answer
- correctness per sample
- parse/verifier failure info

Primary plotted metric:

`mean accuracy over all generated samples` for each benchmark at each eval step.

Also save any useful secondary metric you can compute robustly, but do not clutter the main plot.

### Math answer verification

Inspect the actual dataset schemas first.

Implement a robust evaluator that:

- extracts `\\boxed{...}` when available;
- handles common AIME integer-answer formats;
- normalizes whitespace/sign/fractions;
- uses symbolic/equivalence checking where a maintained library is available;
- logs parse failures instead of silently marking malformed data without explanation;
- never relies on an LLM judge for these three benchmarks.

Do not hardcode field names without schema detection/configurable field mapping.

---

# 9. Required plots and analysis artifacts

Create plots automatically at the end of training and update them after each evaluation when cheap.

All plots must also have their raw CSV/JSON data saved.

## 9.1 Evaluation curves

One line plot containing three curves:

- MATH-500
- AIME 2024
- AIME 2025

x-axis: training step

y-axis: evaluation mean accuracy

Include step 0 and final step.

Save PNG and PDF.

## 9.2 Training loss curve

Plot:

- raw OPD loss vs step
- optionally an EMA-smoothed OPD loss in the same figure

Save PNG/PDF plus CSV.

## 9.3 Prompt usage analysis

After training:

- histogram of how many times each prompt was sampled;
- report:
  - max usage count
  - mean
  - median
  - p90/p95/p99
  - number and percentage of prompts never sampled
  - top 20 most-used prompt IDs and counts
- save full per-prompt CSV containing:
  - prompt ID/index
  - usage count
  - current memory score
  - last seen step
  - latest/EMA `G`
  - latest/EMA `R`
  - latest/EMA `T`

## 9.4 Critical-state position analysis

Record every selected critical state with:

- training step
- prompt ID
- absolute response-token position
- response length
- normalized position `t / response_length`
- selection reason
- `Dplus_t`
- `C_t`
- `A_t`
- `F_t`
- `B_t`

Create at least:

1. histogram/density of normalized critical-state positions in `[0,1]`;
2. histogram of absolute token positions;
3. a stacked or faceted plot by selection reason (`segment`, `global_peak`, `change_point`, `fill`);
4. optionally a 2D heatmap of training step vs normalized position.

This analysis is important because I want to know where critical states tend to occur.

---

# 10. Logging and profiling

Log enough information to debug both the research method and engineering performance.

Use:

- human-readable console logging
- `tqdm`
- JSONL structured training log
- CSV summaries
- TensorBoard if simple to support

Per-step log at minimum:

- step
- learning rate
- OPD loss
- grad norm
- rollout mean/min/max length
- rollout clip/truncation ratio
- mean `Dplus`
- mean `A`
- mean `F`
- mean `B`
- mean prompt `G`
- mean prompt `R`
- mean prompt `T`
- min/max prompt sampling probability in current distribution
- sampling entropy/effective sample size of prompt distribution
- number of valid critical states
- number of branch probes
- tokens generated
- rollout tokens/sec
- wall time:
  - prompt sampling
  - rollout
  - teacher/student diagnostic scoring
  - critical-state selection
  - counterfactual branch probing
  - train forward/backward
  - optimizer step
  - evaluation when applicable
- peak GPU memory allocated/reserved per rank

Write the fully resolved config into the run output directory.

---

# 11. Checkpointing and exact resume

Training can be interrupted. Implement resumable checkpoints.

Default:

- `save_every_steps = 50`
- `keep_last_n_checkpoints = 3`
- always save final checkpoint.

Checkpoint must contain enough state to resume training without resetting the curriculum:

- student weights
- optimizer state
- scheduler state
- current training step
- prompt memory `M_q`
- prompt last-seen steps
- prompt usage counts
- running `G/R/T` stats if used
- global curriculum RNG state
- Python RNG
- NumPy RNG
- PyTorch CPU RNG
- CUDA RNG states
- data/schema metadata
- resolved config

Teacher does not need to be checkpointed because it is frozen and loaded from its fixed path.

Provide:

`--resume_from_checkpoint /path/to/checkpoint`

and an automatic `--resume latest` option if practical.

Resume must continue from the next step rather than rerunning step 0 evaluation.

---

# 12. Multi-GPU behavior: CUDA_VISIBLE_DEVICES controls everything

The project must work with a variable number of B200 GPUs:

- 1 GPU
- 2 GPUs
- 3 GPUs
- 4 GPUs
- 5 GPUs
- more, subject to global batch constraints

The user should only need to change:

`CUDA_VISIBLE_DEVICES=0`

or

`CUDA_VISIBLE_DEVICES=0,1`

or

`CUDA_VISIBLE_DEVICES=0,1,2`

etc.

The bash launcher must detect the number of visible GPUs and launch the correct number of distributed processes automatically.

Use PyTorch `torchrun` + DDP as the default training parallelism because Qwen3-1.7B and the frozen 8B teacher fit comfortably on a B200 180GB and DDP avoids unnecessary sharding complexity.

Do not use `torch.nn.DataParallel`.

Important invariance requirements:

- exact global batch stays 32 regardless of GPU count;
- no automatic LR scaling with GPU count;
- same prompt-sampling distribution semantics;
- same loss definition;
- same generation hyperparameters;
- BF16 on every GPU;
- no automatic FP8/int8 quantization;
- distributed acceleration must not deliberately reduce model/training quality.

Support `world_size=1` through the same code path when possible.

Use NCCL for multi-GPU.

Only rank 0 should write shared logs/plots/checkpoints unless files are explicitly rank-sharded.

Use barriers carefully around evaluation/checkpoint operations.

---

# 13. vLLM engineering requirements

Use vLLM where it genuinely improves throughput while preserving exact method semantics.

### Evaluation

vLLM is mandatory for evaluation.

Prefer data parallel replication (`TP=1`) for Qwen3-1.7B across multiple B200s when this gives higher throughput than tensor parallelism, because the model is small. Make this configurable and benchmark-friendly.

### Training rollout

Prefer vLLM only if exact current-weight synchronization can be maintained every optimizer step.

Implement clean abstraction:

- `VLLMRolloutBackend`
- `TransformersRolloutBackend`

`auto` mode should select the fast valid backend.

Do not invent private vLLM APIs. Use current supported public APIs and inspect the installed version signatures before coding integration. For colocated training/inference, prefer official CUDA-IPC weight transfer and sleep-mode functionality if supported by the pinned vLLM version.

If exact vLLM current-weight sync is not robust in the chosen environment, use the optimized PyTorch rollout backend rather than violating on-policy semantics.

### Counterfactual probes

If practical, vLLM/prefix caching may be used for the no-grad probes, but do not make the entire training code depend on a fragile custom server if a cached PyTorch implementation is faster/simpler.

---

# 14. Project structure

Create a clean codebase approximately like this; adjust filenames if a better modular organization is obvious, but keep responsibilities separated:

```text
RACprompt/
├── README.md
├── requirements.txt
├── configs/
│   └── rac_opd.yaml
├── scripts/
│   ├── train.sh
│   ├── eval.sh
│   └── plot.sh
├── racprompt/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── distributed.py
│   ├── models.py
│   ├── rollout.py
│   ├── scoring.py
│   ├── critical_states.py
│   ├── recoverability.py
│   ├── curriculum.py
│   ├── opd_loss.py
│   ├── trainer.py
│   ├── evaluator.py
│   ├── verifier.py
│   ├── checkpoint.py
│   ├── logging_utils.py
│   └── plotting.py
├── train.py
├── evaluate.py
├── analyze.py
└── tests/
    ├── test_curriculum.py
    ├── test_critical_states.py
    ├── test_distributed_batch.py
    ├── test_recoverability.py
    └── test_checkpoint_state.py
```

Use typed dataclasses/Pydantic/config objects where useful. Keep research formulas close to code with concise comments.

---

# 15. Config and bash launcher

`configs/rac_opd.yaml` must expose all important parameters.

`scripts/train.sh` must put the most commonly changed parameters at the top, for example:

```bash
LR=1e-6
GLOBAL_BATCH_SIZE=32
MAX_STEPS=0                 # <=0 means auto: ceil(N/B)+EXTRA_STEPS
EXTRA_STEPS=100
MAX_NEW_TOKENS=2048
EVAL_EVERY=50
SAVE_EVERY=50
CRITICAL_STATES=24
M_BRANCHES=4
ROLLOUT_BACKEND=auto
RUN_NAME=rac_opd_qwen3
```

Then call the Python trainer with overrides.

The script must derive `NUM_GPUS` from the current `CUDA_VISIBLE_DEVICES`/PyTorch-visible devices and use `torchrun --standalone --nproc_per_node=$NUM_GPUS`.

Do not force a fixed GPU count inside Python or bash.

---

# 16. Data loading requirements

The actual files may be JSON, JSONL, Parquet, Arrow, or Hugging Face `save_to_disk` datasets.

Implement robust loading:

1. inspect path contents;
2. auto-detect supported format;
3. print the detected schema;
4. allow config overrides for prompt/question/answer/id fields;
5. assign a stable integer `prompt_id` if one is absent.

Do not load a small subset for the real training run.

Use the full DAPO-Math-17k-Processed dataset.

Do not assume evaluation field names; inspect and map them.

---

# 17. Requirements file

Create `requirements.txt`.

Pin or constrain versions that are mutually compatible with:

- NVIDIA B200 / CUDA environment
- BF16
- Qwen3
- vLLM current-weight transfer APIs used by the code
- PyTorch distributed training
- Transformers
- datasets
- safetensors
- tqdm
- matplotlib
- pandas
- PyYAML
- TensorBoard
- math verification dependencies chosen by the evaluator

Do not add unnecessary packages.

If PyTorch/vLLM installation on B200 requires a CUDA-specific wheel/index that should not be encoded blindly into `requirements.txt`, document the exact prerequisite install command in `README.md` before `pip install -r requirements.txt`.

---

# 18. README: exact commands from a fresh venv

At the end, README must contain explicit commands starting from:

```bash
cd /workspace/storage-shared/nlp/minhpn19/RACprompt

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Then show training commands for different GPU counts:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train.sh
CUDA_VISIBLE_DEVICES=0,1,2 bash scripts/train.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train.sh
```

Show:

- how to override LR/batch/max steps/eval interval;
- how to resume from latest checkpoint;
- how to run standalone evaluation;
- how to regenerate plots.

Do not require the user to edit Python source code for routine hyperparameter changes.

---

# 19. Output directory convention

Use:

`/workspace/storage-shared/nlp/minhpn19/RACprompt/outputs/<run_name>/`

Suggested contents:

```text
outputs/<run_name>/
├── config_resolved.yaml
├── logs/
│   ├── train.jsonl
│   ├── train_metrics.csv
│   └── eval_metrics.csv
├── eval/
│   ├── step_000000/
│   ├── step_000050/
│   └── ...
├── checkpoints/
│   ├── step_000050/
│   └── ...
├── analysis/
│   ├── prompt_usage.csv
│   └── critical_states.csv_or_parquet
└── plots/
    ├── eval_curves.png
    ├── eval_curves.pdf
    ├── loss_curve.png
    ├── prompt_usage_hist.png
    ├── critical_state_normalized_position.png
    ├── critical_state_absolute_position.png
    └── critical_state_by_reason.png
```

---

# 20. Performance engineering priorities

This is a research experiment on B200 180GB; optimize aggressively without changing the algorithm.

Priority order:

1. exact method semantics;
2. no stale on-policy weights;
3. maximize GPU utilization;
4. minimize duplicated full-prefix forward work;
5. avoid storing full-vocab logits longer than needed;
6. batch teacher scoring;
7. batch the 4 counterfactual branches;
8. reuse KV/prefix cache for branch probes;
9. use BF16;
10. use FlashAttention/SDPA;
11. fused optimizer where supported;
12. pinned-memory/asynchronous CPU->GPU data transfer;
13. avoid unnecessary CPU synchronization and `.item()` in inner loops;
14. profile each phase.

Do not make an optimization that changes the objective, uses approximate/stale student weights, changes generation sampling, quantizes the models, or changes the global batch in a way that could confound model performance.

---

# 21. Required correctness tests / smoke checks

Before considering the project complete, implement and run lightweight tests that do not require the real 8B model.

At minimum verify:

1. prompt sampling probabilities sum to 1;
2. every prompt has nonzero exploration probability;
3. mastered prompt score can decrease after a low new `T(q)`;
4. global batch splitting produces exactly 32 examples for world sizes 1, 2, 3, 4, 5;
5. distributed loss-scaling formula gives the same global mean as a single-process reference;
6. critical-state selector returns <=24 unique valid positions, with temporal coverage;
7. selector never uses TA-OPD `D*C` score;
8. `A_t`, `F_t`, `B_t`, `R(q)`, `G(q)`, `T(q)` remain finite and in expected ranges;
9. geometric mean handles small values safely;
10. counterfactual branches never alter the original rollout tokens;
11. counterfactual branches are detached/no-grad;
12. checkpoint save/load restores curriculum scores, usage counts, step, optimizer, and RNG state;
13. eval schedule includes step 0, every 50 by default, and final step;
14. resume does not repeat step-0 evaluation.

Add a `--dry_run` or smoke-test mode that can use a tiny public/local model if available, but do not change real runtime defaults.

---

# 22. Final deliverables from you, Codex

Do not stop after writing a single script. Produce the full runnable project.

When finished:

1. show the final file tree;
2. summarize the RAC-OPD implementation in <=15 concise bullets;
3. report which tests you ran locally and their status;
4. clearly state any part that could not be runtime-tested because B200 paths/models are unavailable locally;
5. give the exact fresh-venv commands;
6. give exact 1-GPU, 2-GPU, 3-GPU, and resume commands;
7. mention how to change LR, global batch size, max steps, eval frequency, max tokens, critical states, and rollout backend;
8. do not claim the vLLM training-rollout fast path works unless it was actually validated against the installed API; keep the correct PyTorch fallback operational.

The code should prioritize being **research-correct, resumable, analyzable, and fast on variable numbers of B200 GPUs**.
