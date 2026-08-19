from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from .models import unwrap_model
from .rollout import Rollout


LOGGER = logging.getLogger(__name__)


@dataclass
class TrajectoryStatistics:
    dplus: np.ndarray
    compatibility: np.ndarray
    topk_ids: np.ndarray
    topk_student_prob: np.ndarray
    topk_teacher_prob: np.ndarray

    @property
    def topk_positive_correction(self) -> np.ndarray:
        return np.maximum(self.topk_teacher_prob - self.topk_student_prob, 0.0)


def _response_logits(
    logits: torch.Tensor, prompt_length: int, response_length: int
) -> torch.Tensor:
    if response_length <= 0:
        return logits[:, :0, :]
    start = prompt_length - 1
    end = start + response_length
    if start < 0 or end > logits.shape[1]:
        raise ValueError("Invalid prompt/response alignment")
    return logits[:, start:end, :]


class DiagnosticScorer:
    def __init__(
        self,
        student: torch.nn.Module,
        teacher: torch.nn.Module,
        device: torch.device,
        stats_top_k: int = 32,
        position_chunk_size: int = 32,
        branch_microbatch_size: int = 4,
    ) -> None:
        self.student = student
        self.teacher = teacher
        self.device = device
        self.stats_top_k = stats_top_k
        self.position_chunk_size = position_chunk_size
        self.branch_microbatch_size = max(1, int(branch_microbatch_size))
        self._cache_fallback_warned = False

    def score(self, rollout: Rollout) -> TrajectoryStatistics:
        response_length = len(rollout.response_ids)
        if response_length == 0:
            empty = np.empty((0,), dtype=np.float32)
            empty_topk = np.empty((0, self.stats_top_k), dtype=np.float32)
            return TrajectoryStatistics(
                empty,
                empty.copy(),
                empty_topk.astype(np.int64),
                empty_topk,
                empty_topk.copy(),
            )
        ids = torch.tensor([rollout.full_ids], dtype=torch.long, device=self.device)
        mask = torch.ones_like(ids)
        student = unwrap_model(self.student)
        teacher = unwrap_model(self.teacher)
        student_was_training = student.training
        student.eval()
        with torch.inference_mode():
            student_output = student(
                input_ids=ids, attention_mask=mask, use_cache=False
            )
            student_logits = _response_logits(
                student_output.logits, len(rollout.prompt_ids), response_length
            )[0]
            del student_output
            teacher_output = teacher(
                input_ids=ids, attention_mask=mask, use_cache=False
            )
            teacher_logits = _response_logits(
                teacher_output.logits, len(rollout.prompt_ids), response_length
            )[0]
            del teacher_output
            dplus_parts: list[torch.Tensor] = []
            compatibility_parts: list[torch.Tensor] = []
            ids_parts: list[torch.Tensor] = []
            p_parts: list[torch.Tensor] = []
            q_parts: list[torch.Tensor] = []
            k = min(self.stats_top_k, student_logits.shape[-1])
            for start in range(0, response_length, self.position_chunk_size):
                end = min(start + self.position_chunk_size, response_length)
                p = torch.softmax(student_logits[start:end].float(), dim=-1)
                q = torch.softmax(teacher_logits[start:end].float(), dim=-1)
                dplus_parts.append(torch.clamp(q - p, min=0).sum(dim=-1).cpu())
                p_top, top_ids = torch.topk(p, k=k, dim=-1)
                q_top = torch.gather(q, dim=-1, index=top_ids)
                compatibility_parts.append(q_top.sum(dim=-1).cpu())
                ids_parts.append(top_ids.cpu())
                p_parts.append(p_top.cpu())
                q_parts.append(q_top.cpu())
                del p, q, p_top, q_top, top_ids
            del student_logits, teacher_logits, ids, mask
        student.train(student_was_training)
        return TrajectoryStatistics(
            dplus=torch.cat(dplus_parts).numpy(),
            compatibility=torch.cat(compatibility_parts).numpy(),
            topk_ids=torch.cat(ids_parts).numpy(),
            topk_student_prob=torch.cat(p_parts).numpy(),
            topk_teacher_prob=torch.cat(q_parts).numpy(),
        )

    @staticmethod
    def _repeat_cache(cache: Any, repeats: int) -> Any:
        duplicated = copy.deepcopy(cache)
        if hasattr(duplicated, "batch_repeat_interleave"):
            duplicated.batch_repeat_interleave(repeats)
            return duplicated
        if isinstance(duplicated, (tuple, list)):
            return type(duplicated)(
                type(layer)(
                    tensor.repeat_interleave(repeats, dim=0) for tensor in layer
                )
                for layer in duplicated
            )
        raise TypeError(f"Unsupported cache type: {type(cache).__name__}")

    def _branch_next_logits_cached(
        self,
        model: torch.nn.Module,
        prefix_ids: Sequence[int],
        candidate_ids: Sequence[int],
    ) -> torch.Tensor:
        prefix = torch.tensor([list(prefix_ids)], dtype=torch.long, device=self.device)
        candidate = torch.tensor(
            candidate_ids, dtype=torch.long, device=self.device
        ).unsqueeze(1)
        with torch.inference_mode():
            prefetched = model(
                input_ids=prefix, attention_mask=torch.ones_like(prefix), use_cache=True
            )
            repeated_cache = self._repeat_cache(
                prefetched.past_key_values, len(candidate_ids)
            )
            attention = torch.ones(
                (len(candidate_ids), len(prefix_ids) + 1),
                dtype=torch.long,
                device=self.device,
            )
            output = model(
                input_ids=candidate,
                attention_mask=attention,
                past_key_values=repeated_cache,
                use_cache=False,
            )
            logits = output.logits[:, -1, :]
            del prefetched, output, repeated_cache, prefix, candidate, attention
        return logits

    def _branch_next_logits_full(
        self,
        model: torch.nn.Module,
        prefix_ids: Sequence[int],
        candidate_ids: Sequence[int],
    ) -> torch.Tensor:
        branches = [list(prefix_ids) + [int(candidate)] for candidate in candidate_ids]
        ids = torch.tensor(branches, dtype=torch.long, device=self.device)
        with torch.inference_mode():
            output = model(
                input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False
            )
            logits = output.logits[:, -1, :]
            del output, ids
        return logits

    def probe_next_compatibility(
        self, prefix_ids: Sequence[int], candidate_ids: Sequence[int]
    ) -> tuple[np.ndarray, bool]:
        if not candidate_ids:
            return np.empty((0,), dtype=np.float32), True
        student = unwrap_model(self.student)
        teacher = unwrap_model(self.teacher)
        student_was_training = student.training
        student.eval()
        used_cache = True
        result_parts: list[np.ndarray] = []
        try:
            for start in range(0, len(candidate_ids), self.branch_microbatch_size):
                chunk = candidate_ids[start : start + self.branch_microbatch_size]
                try:
                    student_logits = self._branch_next_logits_cached(
                        student, prefix_ids, chunk
                    )
                    teacher_logits = self._branch_next_logits_cached(
                        teacher, prefix_ids, chunk
                    )
                except (TypeError, RuntimeError, AttributeError, ValueError) as exc:
                    used_cache = False
                    if not self._cache_fallback_warned:
                        LOGGER.warning(
                            "KV branch-cache reuse unsupported by this model/API (%s); using batched full-prefix fallback",
                            exc,
                        )
                        self._cache_fallback_warned = True
                    student_logits = self._branch_next_logits_full(
                        student, prefix_ids, chunk
                    )
                    teacher_logits = self._branch_next_logits_full(
                        teacher, prefix_ids, chunk
                    )
                with torch.inference_mode():
                    k = min(self.stats_top_k, student_logits.shape[-1])
                    top_ids = torch.topk(student_logits.float(), k=k, dim=-1).indices
                    teacher_prob = torch.softmax(teacher_logits.float(), dim=-1)
                    values = torch.gather(teacher_prob, dim=-1, index=top_ids).sum(
                        dim=-1
                    )
                    result_parts.append(values.cpu().numpy())
                    del student_logits, teacher_logits, top_ids, teacher_prob, values
        finally:
            student.train(student_was_training)
        return np.concatenate(result_parts), used_cache


def choose_branch_candidates(
    topk_ids: np.ndarray,
    topk_student_prob: np.ndarray,
    topk_teacher_prob: np.ndarray,
    count: int,
    invalid_control_ids: set[int],
    eos_ids: set[int],
) -> tuple[np.ndarray, np.ndarray, int]:
    alpha = np.maximum(
        np.asarray(topk_teacher_prob, dtype=np.float64)
        - np.asarray(topk_student_prob, dtype=np.float64),
        0.0,
    )
    order = np.argsort(-alpha, kind="stable")
    chosen_ids: list[int] = []
    chosen_alpha: list[float] = []
    eos_candidates = 0
    for index in order:
        token = int(topk_ids[index])
        if alpha[index] <= 0:
            break
        if token in eos_ids:
            eos_candidates += 1
            continue
        if token in invalid_control_ids:
            continue
        chosen_ids.append(token)
        chosen_alpha.append(float(alpha[index]))
        if len(chosen_ids) >= count:
            break
    return (
        np.asarray(chosen_ids, dtype=np.int64),
        np.asarray(chosen_alpha, dtype=np.float64),
        eos_candidates,
    )
