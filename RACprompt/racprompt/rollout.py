from __future__ import annotations

import importlib.util
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .config import RolloutConfig
from .models import unwrap_model


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Rollout:
    prompt_index: int
    prompt_id: str
    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    clipped: bool

    @property
    def full_ids(self) -> tuple[int, ...]:
        return self.prompt_ids + self.response_ids


@dataclass(frozen=True)
class RolloutRequest:
    prompt_index: int
    prompt_id: str
    prompt_ids: tuple[int, ...]


class RolloutBackend(ABC):
    @abstractmethod
    def generate(self, requests: Sequence[RolloutRequest]) -> list[Rollout]:
        raise NotImplementedError

    def sync_weights(self, step: int) -> None:
        """Synchronize the exact current student weights before a rollout."""


class TransformersRolloutBackend(RolloutBackend):
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        config: RolloutConfig,
        device: torch.device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device

    @staticmethod
    def _trim_response(
        tokens: list[int], eos_ids: set[int], pad_id: int | None
    ) -> list[int]:
        result: list[int] = []
        for token in tokens:
            if pad_id is not None and token == pad_id and token not in eos_ids:
                break
            result.append(int(token))
            if token in eos_ids:
                break
        return result

    def generate(self, requests: Sequence[RolloutRequest]) -> list[Rollout]:
        if not requests:
            return []
        model = unwrap_model(self.model)
        was_training = model.training
        previous_cache = getattr(model.config, "use_cache", True)
        model.eval()
        model.config.use_cache = True
        batch_limit = self.config.batch_size_per_device or len(requests)
        results: list[Rollout] = []
        eos_value = self.tokenizer.eos_token_id
        eos_ids = (
            set(eos_value if isinstance(eos_value, list) else [eos_value])
            if eos_value is not None
            else set()
        )
        pad_id = self.tokenizer.pad_token_id
        try:
            for start in range(0, len(requests), batch_limit):
                chunk = list(requests[start : start + batch_limit])
                max_len = max(len(request.prompt_ids) for request in chunk)
                input_ids = torch.full(
                    (len(chunk), max_len),
                    int(pad_id),
                    dtype=torch.long,
                    device=self.device,
                )
                attention_mask = torch.zeros_like(input_ids)
                for row, request in enumerate(chunk):
                    values = torch.tensor(
                        request.prompt_ids, dtype=torch.long, device=self.device
                    )
                    input_ids[row, -values.numel() :] = values
                    attention_mask[row, -values.numel() :] = 1
                with torch.inference_mode():
                    output = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        do_sample=True,
                        max_new_tokens=self.config.max_new_tokens,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        repetition_penalty=self.config.repetition_penalty,
                        pad_token_id=pad_id,
                        eos_token_id=eos_value,
                        use_cache=True,
                    )
                generated = output[:, max_len:].cpu().tolist()
                for request, raw_tokens in zip(chunk, generated):
                    response = self._trim_response(raw_tokens, eos_ids, pad_id)
                    results.append(
                        Rollout(
                            prompt_index=request.prompt_index,
                            prompt_id=request.prompt_id,
                            prompt_ids=request.prompt_ids,
                            response_ids=tuple(response),
                            clipped=len(response) >= self.config.max_new_tokens
                            and (not response or response[-1] not in eos_ids),
                        )
                    )
                del output, input_ids, attention_mask
        finally:
            model.config.use_cache = previous_cache
            model.train(was_training)
        return results


class VLLMRolloutBackend(RolloutBackend):
    """Guarded integration point: enabled only with a validated exact-weight synchronizer.

    vLLM's colocated weight-transfer API is version-specific. The default project never
    guesses a private API or reloads from disk per step; auto mode therefore uses the HF
    path unless an installed adapter can prove and perform an exact in-memory sync.
    """

    def __init__(
        self, *args: Any, weight_sync_adapter: Any | None = None, **kwargs: Any
    ):
        if importlib.util.find_spec("vllm") is None:
            raise RuntimeError("vLLM is not installed")
        if weight_sync_adapter is None or not callable(
            getattr(weight_sync_adapter, "sync_from_model", None)
        ):
            raise RuntimeError(
                "No validated public vLLM current-weight synchronization adapter is available; "
                "using vLLM here could make rollouts stale"
            )
        self.weight_sync_adapter = weight_sync_adapter
        raise NotImplementedError(
            "A version-pinned colocated adapter must construct the vLLM engine"
        )

    def generate(self, requests: Sequence[RolloutRequest]) -> list[Rollout]:
        raise NotImplementedError


def create_rollout_backend(
    name: str,
    model: torch.nn.Module,
    tokenizer: Any,
    config: RolloutConfig,
    device: torch.device,
) -> RolloutBackend:
    normalized = name.lower()
    if normalized not in {"auto", "transformers", "vllm"}:
        raise ValueError(f"Unknown rollout backend: {name}")
    if normalized == "vllm" and not config.allow_vllm_training_backend:
        raise RuntimeError(
            "vLLM training rollout was explicitly requested but disabled"
        )
    if normalized in {"auto", "vllm"} and config.allow_vllm_training_backend:
        try:
            return VLLMRolloutBackend(
                model=model, tokenizer=tokenizer, config=config, device=device
            )
        except (RuntimeError, NotImplementedError) as exc:
            if normalized == "vllm":
                raise RuntimeError(
                    f"Requested vLLM training rollout is unsafe: {exc}"
                ) from exc
            LOGGER.warning(
                "vLLM training rollout disabled: %s. Falling back to exact HF generation.",
                exc,
            )
    return TransformersRolloutBackend(model, tokenizer, config, device)
