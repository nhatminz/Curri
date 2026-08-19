from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PromptScore:
    prompt_index: int
    prompt_id: str
    teachability: float
    need: float
    recoverability: float
    rollout_length: int
    valid_critical_states: int


class CurriculumState:
    def __init__(
        self,
        prompt_ids: list[str],
        initial_score: float = 0.5,
        age_tau_steps: float = 200.0,
        enable_staleness_decay: bool = True,
        eps_explore: float = 0.10,
        temperature: float = 0.25,
        ema_beta: float = 0.80,
        seed: int = 2025,
    ) -> None:
        if not prompt_ids:
            raise ValueError("The prompt pool cannot be empty")
        self.prompt_ids = [str(item) for item in prompt_ids]
        self.initial_score = float(initial_score)
        self.age_tau_steps = float(age_tau_steps)
        self.enable_staleness_decay = bool(enable_staleness_decay)
        self.eps_explore = float(eps_explore)
        self.temperature = float(temperature)
        self.ema_beta = float(ema_beta)
        count = len(prompt_ids)
        self.scores = np.full(count, initial_score, dtype=np.float64)
        self.last_seen = np.full(count, -1, dtype=np.int64)
        self.usage_counts = np.zeros(count, dtype=np.int64)
        self.latest_g = np.full(count, np.nan, dtype=np.float64)
        self.latest_r = np.full(count, np.nan, dtype=np.float64)
        self.latest_t = np.full(count, np.nan, dtype=np.float64)
        self.ema_g = np.full(count, np.nan, dtype=np.float64)
        self.ema_r = np.full(count, np.nan, dtype=np.float64)
        self.ema_t = np.full(count, np.nan, dtype=np.float64)
        self.rng = np.random.default_rng(seed)

    def effective_scores(self, step: int) -> np.ndarray:
        if not self.enable_staleness_decay or self.age_tau_steps <= 0:
            return self.scores.copy()
        ages = np.maximum(step - self.last_seen, 0)
        decay = np.exp(-ages.astype(np.float64) / self.age_tau_steps)
        return self.initial_score + (self.scores - self.initial_score) * decay

    def probabilities(self, step: int) -> np.ndarray:
        effective = self.effective_scores(step)
        centered = effective / self.temperature
        centered -= centered.max()
        weights = np.exp(centered)
        softmax = weights / weights.sum(dtype=np.float64)
        count = len(self.scores)
        probabilities = self.eps_explore / count + (1.0 - self.eps_explore) * softmax
        probabilities /= probabilities.sum(dtype=np.float64)
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities <= 0):
            raise FloatingPointError("Invalid curriculum probability distribution")
        return probabilities

    def sample(
        self, step: int, global_batch_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        probabilities = self.probabilities(step)
        indices = self.rng.choice(
            len(probabilities), size=global_batch_size, replace=True, p=probabilities
        ).astype(np.int64)
        return indices, probabilities

    @staticmethod
    def _ema(old: float, new: float, beta: float) -> float:
        return new if np.isnan(old) else beta * old + (1.0 - beta) * new

    def update(self, result: PromptScore, step: int) -> None:
        index = int(result.prompt_index)
        t_value = float(np.clip(result.teachability, 0.0, 1.0))
        self.scores[index] = (
            self.ema_beta * self.scores[index] + (1.0 - self.ema_beta) * t_value
        )
        self.last_seen[index] = int(step)
        self.usage_counts[index] += 1
        self.latest_g[index] = result.need
        self.latest_r[index] = result.recoverability
        self.latest_t[index] = t_value
        self.ema_g[index] = self._ema(self.ema_g[index], result.need, self.ema_beta)
        self.ema_r[index] = self._ema(
            self.ema_r[index], result.recoverability, self.ema_beta
        )
        self.ema_t[index] = self._ema(self.ema_t[index], t_value, self.ema_beta)

    def state_dict(self) -> dict[str, Any]:
        return {
            "prompt_ids": self.prompt_ids,
            "scores": self.scores,
            "last_seen": self.last_seen,
            "usage_counts": self.usage_counts,
            "latest_g": self.latest_g,
            "latest_r": self.latest_r,
            "latest_t": self.latest_t,
            "ema_g": self.ema_g,
            "ema_r": self.ema_r,
            "ema_t": self.ema_t,
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if list(state["prompt_ids"]) != self.prompt_ids:
            raise ValueError(
                "Checkpoint prompt IDs/order differ from the current dataset"
            )
        for name in (
            "scores",
            "last_seen",
            "usage_counts",
            "latest_g",
            "latest_r",
            "latest_t",
            "ema_g",
            "ema_r",
            "ema_t",
        ):
            value = np.asarray(state[name])
            if value.shape != self.scores.shape:
                raise ValueError(
                    f"Invalid curriculum array shape for {name}: {value.shape}"
                )
            setattr(self, name, value.copy())
        self.rng.bit_generator.state = state["rng_state"]
