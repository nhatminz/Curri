# RAC-OPD: Recoverability-Aware Curriculum for On-Policy Distillation

Repository này triển khai đầy đủ RAC-OPD cho Qwen3. Student luôn sinh rollout từ đúng trọng số hiện tại. Các nhánh teacher-preferred chỉ là phép đo phản thực một bước trong `no_grad`; chúng không sửa rollout, không trở thành target huấn luyện và không reweight loss của batch hiện tại.

## 1. Mặc định thí nghiệm

| Thành phần | Giá trị mặc định |
|---|---|
| Student | `/workspace/storage-shared/models/Qwen3-1.7B` |
| Frozen teacher | `/workspace/storage-shared/models/Qwen3-8B` |
| Training pool | `/workspace/storage-shared/nlp/minhpn19/data/DAPO-Math-17k-Processed` |
| MATH-500 | `/workspace/storage-shared/nlp/minhpn19/data/eval/math500` |
| AIME 2024 | `/workspace/storage-shared/nlp/minhpn19/data/eval/aime24` |
| AIME 2025 | `/workspace/storage-shared/nlp/minhpn19/data/eval/aime25` |
| Output root | `/workspace/storage-shared/nlp/minhpn19/RACprompt/outputs` |
| Global batch | 32, không phụ thuộc số GPU |
| Critical states | 24 |
| Counterfactual branches/state | 4 |
| OPD top-K | 16 |
| Precision | BF16, không quantization |

Các giá trị này nằm trong `configs/rac_opd.yaml`. Dataset loader tự nhận diện Hugging Face `save_to_disk`, JSON, JSONL hoặc Parquet, in schema đã phát hiện và dùng toàn bộ split được chọn.

## 2. Cài đặt mới trên B200 không có Internet

B200 này không truy cập được `download.pytorch.org`, PyPI hoặc GitHub. Chỉ dùng ba Nexus mirror nội bộ xuất hiện trong cấu hình của máy. Khối lệnh dưới đây chủ động bỏ qua mọi pip config kế thừa và không chứa URL Internet nào:

```bash
cd /workspace/storage-shared/nlp/minhpn19/RACprompt

python3 -m venv .venv
source .venv/bin/activate

# Ép pip chỉ dùng Nexus nội bộ. /dev/null vô hiệu hóa pip.conf cũ có URL ngoài.
export PIP_CONFIG_FILE=/dev/null
export PIP_INDEX_URL=http://10.30.154.118:8888/repository/pypi.org/simple/
export PIP_EXTRA_INDEX_URL="http://10.30.154.118:8888/repository/python/simple/ http://10.30.154.118:8888/repository/pypi-official/simple/"
export PIP_TRUSTED_HOST=10.30.154.118

python -m pip install --upgrade pip setuptools wheel
python -m pip install vllm==0.23.0
python -m pip install -r requirements.txt
```

Không thêm lại `--extra-index-url https://download.pytorch.org/...`: pip sẽ luôn thử kết nối URL đó và bị retry vì máy không có route ra Internet.

Trước khi cài, có thể kiểm tra Nexus đã mirror đúng version hay chưa. Lệnh này cũng chỉ truy cập mạng nội bộ:

```bash
python -m pip index versions vllm
python -m pip index versions torch
```

Nếu output có `vllm 0.23.0` và `torch 2.11.0`, tiếp tục dùng khối cài đặt phía trên. Có thể kiểm tra toàn bộ index đang có hiệu lực bằng:

```bash
python -m pip config debug
```

Output chỉ được chứa host nội bộ `10.30.154.118`; không được có URL bắt đầu bằng `https://download.pytorch.org`, `https://pypi.org` hoặc URL GitHub.

### Nếu Nexus chưa có vLLM/PyTorch cần thiết

Máy hoàn toàn offline chỉ cài được package đã có trong Nexus hoặc wheel được chép vào shared storage. Trên một máy Linux x86_64 có Internet, dùng cùng Python major/minor với B200 để tạo wheelhouse:

```bash
cd /path/to/RACprompt
python3 -m venv wheel-download-env
source wheel-download-env/bin/activate
python -m pip install --upgrade pip

mkdir -p racprompt-wheelhouse
python -m pip download \
  --dest racprompt-wheelhouse \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  vllm==0.23.0 \
  -r requirements.txt

tar -czf racprompt-wheelhouse-cu129.tar.gz racprompt-wheelhouse
```

Chép `racprompt-wheelhouse-cu129.tar.gz` vào shared storage, rồi trên B200:

```bash
cd /workspace/storage-shared/nlp/minhpn19/RACprompt
source .venv/bin/activate

tar -xzf /path/on/shared-storage/racprompt-wheelhouse-cu129.tar.gz
python -m pip install \
  --no-index \
  --find-links ./racprompt-wheelhouse \
  vllm==0.23.0
python -m pip install \
  --no-index \
  --find-links ./racprompt-wheelhouse \
  -r requirements.txt
```

`--no-index` bảo đảm pip không thử truy cập bất kỳ mạng nào. Wheelhouse phải được tạo cho cùng kiến trúc, phiên bản Python và CUDA stack của B200.

Kiểm tra môi trường:

```bash
python - <<'PY'
import torch
import transformers
import vllm

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("BF16 supported:", torch.cuda.is_bf16_supported())
print("visible GPUs:", torch.cuda.device_count())
print("transformers:", transformers.__version__)
print("vllm:", vllm.__version__)
PY

python -m pytest -q
```

Kết quả mong đợi là CUDA/BF16 đều khả dụng và toàn bộ unit tests pass.

## 3. Run name tự động

Khi không truyền `RUN_NAME`, `scripts/train.sh` tạo tên tại thời điểm bấm chạy:

```text
rac_opd_qwen3_YYYYMMDD_HHMMSS
```

Ví dụ:

```text
rac_opd_qwen3_20260819_143527
```

Tên được tạo đúng một lần trước `torchrun`, vì vậy mọi DDP rank dùng cùng một output directory. Launcher luôn in:

```text
RAC-OPD run name: rac_opd_qwen3_20260819_143527
RAC-OPD output:   /workspace/storage-shared/nlp/minhpn19/RACprompt/outputs/rac_opd_qwen3_20260819_143527
```

Sau khi trainer khởi tạo thành công, file sau được cập nhật:

```text
/workspace/storage-shared/nlp/minhpn19/RACprompt/outputs/latest_run.txt
```

Xem run gần nhất bằng:

```bash
cat /workspace/storage-shared/nlp/minhpn19/RACprompt/outputs/latest_run.txt
```

Nếu muốn một tên cố định thay vì timestamp:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
RUN_NAME=rac_opd_ablation_no_aging \
bash scripts/train.sh
```

Khi chạy trực tiếp `train.py`, `training.run_name=auto` trong YAML cũng được rank 0 chuyển thành tên timestamp rồi broadcast cho các rank.

## 4. Chạy training

Launcher chỉ nhìn `CUDA_VISIBLE_DEVICES` để xác định số process. Global batch vẫn đúng 32 và learning rate không tự scale theo số GPU.

```bash
# 1 GPU: local batch 32
CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh

# 2 GPU: 16/16
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train.sh

# 3 GPU: 11/11/10, tổng vẫn là 32
CUDA_VISIBLE_DEVICES=0,1,2 bash scripts/train.sh

# 4 GPU: 8/8/8/8
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train.sh

# 5 GPU: 7/7/6/6/6
CUDA_VISIBLE_DEVICES=0,1,2,3,4 bash scripts/train.sh
```

Một lệnh thí nghiệm đầy đủ:

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
bash scripts/train.sh
```

Không có `RUN_NAME` trong lệnh trên nên tên sẽ được sinh tự động.

### Các biến thường dùng

| Biến shell | Mặc định | Ý nghĩa |
|---|---:|---|
| `LR` | `1e-6` | AdamW learning rate; không scale theo GPU |
| `GLOBAL_BATCH_SIZE` | `32` | Tổng số prompt mỗi optimizer step |
| `MAX_STEPS` | `0` | `<=0` dùng công thức tự động |
| `EXTRA_STEPS` | `100` | Số step cộng thêm khi `MAX_STEPS<=0` |
| `MAX_PROMPT_TOKENS` | `1024` | Giới hạn token prompt training |
| `MAX_NEW_TOKENS` | `2048` | Giới hạn rollout response |
| `EVAL_EVERY` | `50` | Khoảng cách evaluation |
| `SAVE_EVERY` | `50` | Khoảng cách checkpoint |
| `CRITICAL_STATES` | `24` | Số state mục tiêu, hợp lệ từ 1 đến 50 |
| `M_BRANCHES` | `4` | Main method cố định bằng 4 |
| `ROLLOUT_BACKEND` | `auto` | Chọn backend rollout an toàn |
| `RUN_NAME` | timestamp tự động | Tên thủ công nếu cần |
| `OUTPUT_ROOT` | B200 output path | Root chứa tất cả run |
| `CONFIG` | `configs/rac_opd.yaml` | File cấu hình gốc |
| `RESUME` | rỗng | `latest` hoặc checkpoint cụ thể |

Khi `MAX_STEPS=0`, trainer dùng:

```text
ceil(len(full_training_dataset) / GLOBAL_BATCH_SIZE) + EXTRA_STEPS
```

Có thể override mọi trường YAML bằng `--set` đặt sau lệnh script:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train.sh \
  --set curriculum.enable_staleness_decay=false \
  --set curriculum.temperature=0.30 \
  --set critical.stats_top_k=48 \
  --set rollout.temperature=0.9 \
  --set training.opd_top_k=24
```

Các `--set` bổ sung nằm cuối command nên có quyền ưu tiên cao hơn biến mặc định của launcher.

## 5. Resume chính xác

Checkpoint lưu student, optimizer, scheduler, curriculum memory, usage counts, step và RNG của từng rank. Resume tiếp tục từ step kế tiếp và không chạy lại evaluation step 0.

Để resume run gần nhất mà không cần nhớ timestamp:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
RESUME=latest \
bash scripts/train.sh
```

Launcher đọc run name từ `outputs/latest_run.txt`, sau đó tìm checkpoint mới nhất trong run đó.

Resume checkpoint cụ thể; nếu không truyền `RUN_NAME`, launcher suy ra tên run từ path:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
RESUME=/workspace/storage-shared/nlp/minhpn19/RACprompt/outputs/rac_opd_qwen3_20260819_143527/checkpoints/step_000100 \
bash scripts/train.sh
```

Resume latest của một run được chỉ định rõ:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
RUN_NAME=rac_opd_qwen3_20260819_143527 \
RESUME=latest \
bash scripts/train.sh
```

Để khôi phục chính xác RNG/DDP, phải dùng cùng số GPU và cùng thứ tự `CUDA_VISIBLE_DEVICES` như lúc tạo checkpoint. Có thể đặt `MAX_STEPS` lớn hơn để kéo dài run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
RESUME=latest \
MAX_STEPS=900 \
bash scripts/train.sh
```

## 6. Evaluation đầy đủ

Training tự chạy full MATH-500, AIME 2024 và AIME 2025 tại:

1. step 0 trước optimizer update;
2. mỗi `EVAL_EVERY` step;
3. final step, kể cả khi không chia hết cho interval.

Mỗi problem mặc định sinh bốn samples bằng vLLM BF16. Evaluation chạy trong process riêng để giải phóng hoàn toàn vLLM engine trước khi training tiếp tục.

Đánh giá checkpoint mới nhất của run mới nhất:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval.sh
```

Đánh giá run cụ thể:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
RUN_NAME=rac_opd_qwen3_20260819_143527 \
CHECKPOINT=latest \
bash scripts/eval.sh
```

Thay số sample hoặc độ dài response evaluation:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
EVAL_SAMPLES=8 \
EVAL_MAX_NEW_TOKENS=3072 \
bash scripts/eval.sh
```

Đánh giá trực tiếp một model không thuộc checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --model /workspace/storage-shared/models/Qwen3-1.7B \
  --step 0 \
  --output_dir /workspace/storage-shared/nlp/minhpn19/RACprompt/outputs/base_qwen3_eval
```

`scripts/eval.sh` ưu tiên data parallel với `TP=1` trên số GPU nhìn thấy nếu installed vLLM hỗ trợ offline `data_parallel_size`.

## 7. Theo dõi training

Lấy đường dẫn run gần nhất:

```bash
OUTPUT_ROOT=/workspace/storage-shared/nlp/minhpn19/RACprompt/outputs
LATEST_RUN="$(cat "${OUTPUT_ROOT}/latest_run.txt")"
RUN_DIR="${OUTPUT_ROOT}/${LATEST_RUN}"
echo "${RUN_DIR}"
```

Theo dõi JSONL hoặc CSV:

```bash
tail -f "${RUN_DIR}/logs/train.jsonl"
```

```bash
column -s, -t < "${RUN_DIR}/logs/train_metrics.csv" | less -S
```

TensorBoard:

```bash
tensorboard \
  --logdir "${RUN_DIR}/logs/tensorboard" \
  --host 0.0.0.0 \
  --port 6006
```

Mỗi training step log loss, gradient norm, rollout lengths, truncation ratio, `Dplus/A/F/B/G/R/T`, entropy/ESS của sampling distribution, số branch probe, throughput, thời gian từng phase và peak GPU memory của từng rank.

## 8. Tạo lại plots

Tạo plots cho run mới nhất:

```bash
bash scripts/plot.sh
```

Tạo plots cho run cụ thể:

```bash
RUN_NAME=rac_opd_qwen3_20260819_143527 bash scripts/plot.sh
```

Hoặc truyền thẳng output directory:

```bash
python analyze.py \
  --output_dir /workspace/storage-shared/nlp/minhpn19/RACprompt/outputs/rac_opd_qwen3_20260819_143527
```

Plots bao gồm evaluation curves, loss raw/EMA, histogram prompt usage, phân bố vị trí critical state chuẩn hóa/tuyệt đối và phân bố theo selection reason. Dữ liệu gốc luôn được giữ trong CSV/JSON.

## 9. Output layout

```text
outputs/
├── latest_run.txt
└── rac_opd_qwen3_YYYYMMDD_HHMMSS/
    ├── config_resolved.yaml
    ├── logs/
    │   ├── train.jsonl
    │   ├── train_metrics.csv
    │   ├── eval_metrics.csv
    │   └── tensorboard/
    ├── eval/
    │   ├── step_000000/
    │   ├── step_000050/
    │   └── ...
    ├── checkpoints/
    │   ├── step_000050/
    │   │   ├── student/
    │   │   └── trainer_state.pt
    │   └── ...
    ├── analysis/
    │   ├── critical_states.csv
    │   ├── prompt_usage.csv
    │   ├── prompt_usage_summary.json
    │   └── loss_curve.csv
    └── plots/
        ├── eval_curves.png
        ├── eval_curves.pdf
        ├── loss_curve.png
        ├── loss_curve.pdf
        ├── prompt_usage_hist.png
        ├── critical_state_normalized_position.png
        ├── critical_state_absolute_position.png
        └── critical_state_by_reason.png
```

`config_resolved.yaml` chứa run name timestamp và `max_steps` đã resolve, nên đây là file cần lưu cùng kết quả để tái lập thí nghiệm.

## 10. Local dry run

Runtime mặc định luôn giữ path B200. Muốn smoke test local phải override model và dataset rõ ràng:

```bash
python train.py --dry_run \
  --set paths.student_model=/path/to/tiny/causal-lm \
  --set paths.teacher_model=/path/to/tiny/causal-lm \
  --set paths.train_data=/path/to/tiny.jsonl \
  --set paths.output_root=/tmp/racprompt_outputs
```

Dry run dùng tối đa tám prompt, một optimizer step, tối đa 32 response tokens và tắt full evaluation. Nó không thay đổi default của run thật.

## 11. Ghi chú backend và xử lý lỗi

### Pip báo `Network is unreachable` với `download.pytorch.org`

Shell hiện tại vẫn còn external index từ lệnh cũ hoặc từ `pip.conf`. Thiết lập lại ba biến internal-only rồi cài lại:

```bash
export PIP_CONFIG_FILE=/dev/null
export PIP_INDEX_URL=http://10.30.154.118:8888/repository/pypi.org/simple/
export PIP_EXTRA_INDEX_URL="http://10.30.154.118:8888/repository/python/simple/ http://10.30.154.118:8888/repository/pypi-official/simple/"
export PIP_TRUSTED_HOST=10.30.154.118

python -m pip install vllm==0.23.0
python -m pip install -r requirements.txt
```

Không dùng `--extra-index-url https://download.pytorch.org/whl/cu129` trên B200 offline.

### Thấy cảnh báo fallback từ vLLM training rollout

Đây là hành vi an toàn. Public colocated weight-transfer API phụ thuộc version. `ROLLOUT_BACKEND=auto` chỉ dùng vLLM training rollout khi có adapter đồng bộ current weights đã được xác thực; bản hiện tại fallback sang HF/PyTorch với đúng model trong memory. Nó không reload model từ disk mỗi step và không bao giờ âm thầm sinh rollout từ stale weights.

Evaluation vẫn bắt buộc dùng vLLM.

### Không tìm thấy `latest_run.txt`

Chưa có run nào khởi tạo thành công trong `OUTPUT_ROOT`, hoặc bạn đang trỏ nhầm root. Chỉ định rõ:

```bash
OUTPUT_ROOT=/correct/output/root \
RUN_NAME=known_run_name \
RESUME=latest \
bash scripts/train.sh
```

### Dataset field không được nhận diện

Xem schema được in trong console, sau đó override:

```bash
bash scripts/train.sh \
  --set data.prompt_field=question \
  --set data.answer_field=answer \
  --set data.id_field=problem_id
```

### CUDA OOM

Không giảm global batch hoặc số critical state trước khi xác định phase gây OOM. Các lựa chọn ít ảnh hưởng semantics hơn:

```bash
bash scripts/train.sh \
  --set rollout.batch_size_per_device=2 \
  --set critical.branch_microbatch_size=2 \
  --set training.gradient_checkpointing=true
```

`GLOBAL_BATCH_SIZE`, learning rate, generation sampling và precision không tự thay đổi theo số GPU.

### Resume báo world size không khớp

Checkpoint lưu RNG riêng cho từng rank. Dùng lại đúng số GPU của run gốc; code chủ động từ chối resume “exact” bằng world size khác.

## 12. Các invariant nghiên cứu quan trọng

- Main rollout luôn là `student_current`, không teacher guidance.
- Critical selector chỉ dùng temporal coverage, `Dplus` peaks và thay đổi compatibility; không dùng TA-OPD `D*C`.
- Counterfactual branch chỉ dài đúng một token correction rồi đo next-state distribution.
- Không backpropagate và không train trên branch.
- `R` là geometric mean có floor, không phải arithmetic mean.
- `T=G*R` chỉ cập nhật future prompt sampling memory.
- OPD loss chỉ dùng original student rollout, không `T(q)` weighting và không token mask.
- Teacher luôn frozen, BF16, eval và no-grad.
- Global batch và loss semantics độc lập số GPU.
- Step 0, periodic và final evaluation đều chạy trên toàn bộ ba benchmark.
