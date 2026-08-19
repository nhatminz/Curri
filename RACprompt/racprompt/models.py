from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

from .config import RACConfig


LOGGER = logging.getLogger(__name__)


@dataclass
class ModelBundle:
    student: torch.nn.Module
    teacher: torch.nn.Module
    tokenizer: Any


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _model_kwargs(config: RACConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "trust_remote_code": False,
    }
    if config.training.attn_implementation != "auto":
        kwargs["attn_implementation"] = config.training.attn_implementation
    return kwargs


def load_models(
    config: RACConfig, device: torch.device, student_model_path: str | None = None
) -> ModelBundle:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("transformers is required for training") from exc
    resolved_student_path = student_model_path or config.paths.student_model
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_student_path, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs either a pad or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    kwargs = _model_kwargs(config)
    LOGGER.info("Loading current student from %s", resolved_student_path)
    student = AutoModelForCausalLM.from_pretrained(resolved_student_path, **kwargs).to(
        device
    )
    LOGGER.info("Loading frozen teacher from %s", config.paths.teacher_model)
    teacher = AutoModelForCausalLM.from_pretrained(
        config.paths.teacher_model, **kwargs
    ).to(device)
    student_vocab = int(student.get_output_embeddings().weight.shape[0])
    teacher_vocab = int(teacher.get_output_embeddings().weight.shape[0])
    if student_vocab != teacher_vocab or len(tokenizer) > student_vocab:
        raise ValueError(
            "Student, teacher, and tokenizer must share one token vocabulary for exact "
            f"full-vocabulary corrections; student={student_vocab}, teacher={teacher_vocab}, "
            f"tokenizer={len(tokenizer)}"
        )
    teacher.eval()
    teacher.requires_grad_(False)
    student.config.use_cache = not config.training.gradient_checkpointing
    if config.training.gradient_checkpointing:
        student.gradient_checkpointing_enable()
    return ModelBundle(student=student, teacher=teacher, tokenizer=tokenizer)
