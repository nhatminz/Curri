from __future__ import annotations

import ast
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, get_type_hints

import yaml


@dataclass
class PathsConfig:
    student_model: str = "/workspace/storage-shared/models/Qwen3-1.7B"
    teacher_model: str = "/workspace/storage-shared/models/Qwen3-8B"
    train_data: str = (
        "/workspace/storage-shared/nlp/minhpn19/data/DAPO-Math-17k-Processed"
    )
    math500: str = "/workspace/storage-shared/nlp/minhpn19/data/eval/math500"
    aime24: str = "/workspace/storage-shared/nlp/minhpn19/data/eval/aime24"
    aime25: str = "/workspace/storage-shared/nlp/minhpn19/data/eval/aime25"
    output_root: str = "/workspace/storage-shared/nlp/minhpn19/RACprompt/outputs"


@dataclass
class DataConfig:
    train_split: str = "train"
    eval_split: Optional[str] = None
    prompt_field: Optional[str] = None
    answer_field: Optional[str] = None
    id_field: Optional[str] = None
    prompt_template_mode: str = "auto"  # auto, chat, raw
    enable_thinking: Optional[bool] = None
    max_prompt_tokens: int = 1024


@dataclass
class CurriculumConfig:
    initial_score: float = 0.5
    age_tau_steps: float = 200.0
    enable_staleness_decay: bool = True
    eps_explore: float = 0.10
    temperature: float = 0.25
    ema_beta: float = 0.80
    seed: int = 2025


@dataclass
class CriticalConfig:
    target: int = 24
    num_segments: int = 12
    global_peaks: int = 8
    change_points: int = 4
    change_lag: int = 32
    min_gap_tokens: int = 32
    stats_top_k: int = 32
    branch_candidates: int = 4
    eps_num: float = 1.0e-12
    score_position_chunk: int = 32
    branch_microbatch_size: int = 4


@dataclass
class RolloutConfig:
    backend: str = "auto"
    max_new_tokens: int = 2048
    temperature: float = 1.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    rollouts_per_prompt: int = 1
    batch_size_per_device: int = 0  # 0 means all local prompts
    allow_vllm_training_backend: bool = True


@dataclass
class TrainingConfig:
    run_name: str = "rac_opd_qwen3"
    global_batch_size: int = 32
    max_steps: int = 0
    extra_steps: int = 100
    learning_rate: float = 1.0e-6
    weight_decay: float = 0.01
    scheduler: str = "constant"
    max_grad_norm: float = 1.0
    opd_top_k: int = 16
    gradient_checkpointing: bool = False
    seed: int = 1337
    diagnostic_sequence_microbatch_size: int = 1
    attn_implementation: str = "auto"
    fused_adamw: bool = True
    tf32: bool = True
    dry_run: bool = False


@dataclass
class EvalConfig:
    every_steps: int = 50
    temperature: float = 0.6
    top_p: float = 0.95
    num_samples_per_problem: int = 4
    max_new_tokens: int = 2048
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1
    gpu_memory_utilization: float = 0.35
    max_model_len: Optional[int] = None
    enabled: bool = True


@dataclass
class CheckpointConfig:
    save_every_steps: int = 50
    keep_last_n: int = 3
    resume: Optional[str] = None


@dataclass
class LoggingConfig:
    tensorboard: bool = True
    log_every_steps: int = 1
    plot_after_eval: bool = True


@dataclass
class RACConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    critical: CriticalConfig = field(default_factory=CriticalConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def validate(self) -> None:
        if self.training.global_batch_size <= 0:
            raise ValueError("global_batch_size must be positive")
        if not 1 <= self.critical.target <= 50:
            raise ValueError("critical.target must be in [1, 50]")
        if self.critical.branch_candidates != 4:
            raise ValueError("The main RAC-OPD method fixes branch_candidates at 4")
        if self.rollout.rollouts_per_prompt != 1:
            raise ValueError(
                "The main implementation requires one on-policy rollout per prompt"
            )
        if self.rollout.max_new_tokens <= 0 or self.data.max_prompt_tokens <= 0:
            raise ValueError("Prompt and response token limits must be positive")
        if not 0.0 < self.curriculum.eps_explore <= 1.0:
            raise ValueError("eps_explore must be in (0, 1]")
        if self.curriculum.temperature <= 0:
            raise ValueError("curriculum temperature must be positive")
        if self.training.opd_top_k <= 0 or self.critical.stats_top_k <= 0:
            raise ValueError("top-k values must be positive")


def _construct_dataclass(cls: type, values: Mapping[str, Any]) -> Any:
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for item in dataclasses.fields(cls):
        if item.name not in values:
            continue
        value = values[item.name]
        typ = hints.get(item.name, item.type)
        if dataclasses.is_dataclass(typ) and isinstance(value, Mapping):
            value = _construct_dataclass(typ, value)
        kwargs[item.name] = value
    return cls(**kwargs)


def _parse_override(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        try:
            return ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return raw


def _set_nested(mapping: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = mapping
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            child = {}
            cursor[part] = child
        cursor = child
    cursor[parts[-1]] = value


def load_config(path: str | Path, overrides: list[str] | None = None) -> RACConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping")
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Override must be KEY=VALUE, got {override!r}")
        key, value = override.split("=", 1)
        _set_nested(raw, key, _parse_override(value))
    config = _construct_dataclass(RACConfig, raw)
    config.validate()
    return config


def config_to_dict(config: RACConfig) -> dict[str, Any]:
    return dataclasses.asdict(config)


def save_config(config: RACConfig, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config_to_dict(config), handle, sort_keys=False)
