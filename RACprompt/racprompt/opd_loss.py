from __future__ import annotations

import torch

from .models import unwrap_model
from .rollout import Rollout
from .scoring import _response_logits


def sequence_opd_loss(
    student_model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    rollout: Rollout,
    top_k: int,
    device: torch.device,
) -> torch.Tensor:
    """Student-top-K reverse KL, token-mean for one original rollout."""
    if not rollout.response_ids:
        # Keep a differentiable zero for the rare empty generation.
        parameter = next(unwrap_model(student_model).parameters())
        return parameter.sum() * 0.0
    student = student_model
    teacher = unwrap_model(teacher_model)
    ids = torch.tensor([rollout.full_ids], dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    student_output = student(input_ids=ids, attention_mask=mask, use_cache=False)
    student_logits = _response_logits(
        student_output.logits, len(rollout.prompt_ids), len(rollout.response_ids)
    )[0]
    k = min(int(top_k), student_logits.shape[-1])
    top_ids = torch.topk(student_logits.detach(), k=k, dim=-1).indices
    selected_student = torch.gather(student_logits, dim=-1, index=top_ids).float()
    with torch.no_grad():
        teacher_output = teacher(input_ids=ids, attention_mask=mask, use_cache=False)
        teacher_logits = _response_logits(
            teacher_output.logits, len(rollout.prompt_ids), len(rollout.response_ids)
        )[0]
        selected_teacher = torch.gather(teacher_logits, dim=-1, index=top_ids).float()
        log_q_bar = torch.log_softmax(selected_teacher, dim=-1)
    log_p_bar = torch.log_softmax(selected_student, dim=-1)
    p_bar = log_p_bar.exp()
    token_kl = torch.sum(p_bar * (log_p_bar - log_q_bar), dim=-1)
    loss = token_kl.mean()
    del (
        student_output,
        teacher_output,
        student_logits,
        teacher_logits,
        selected_student,
        selected_teacher,
    )
    return loss
