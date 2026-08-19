from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class StateRecoverability:
    dplus: float
    accessibility: float
    future_compatibility: float
    bridgeability: float
    valid: bool


@dataclass(frozen=True)
class PromptRecoverability:
    need: float
    recoverability: float
    teachability: float
    valid_states: int


def accessibility(
    dplus: float, positive_mass_on_student_topk: float, eps: float = 1e-12
) -> float:
    return float(np.clip(positive_mass_on_student_topk / (dplus + eps), 0.0, 1.0))


def future_compatibility(
    alpha: np.ndarray, next_compatibility: np.ndarray, eps: float = 1e-12
) -> float | None:
    alpha = np.asarray(alpha, dtype=np.float64)
    next_compatibility = np.asarray(next_compatibility, dtype=np.float64)
    valid = np.isfinite(alpha) & np.isfinite(next_compatibility) & (alpha > 0)
    if not np.any(valid):
        return None
    value = np.sum(alpha[valid] * next_compatibility[valid]) / (
        np.sum(alpha[valid]) + eps
    )
    return float(np.clip(value, 0.0, 1.0))


def state_recoverability(
    dplus: float,
    positive_mass_on_student_topk: float,
    alpha: np.ndarray,
    next_compatibility: np.ndarray,
    eps: float = 1e-12,
) -> StateRecoverability:
    a_value = accessibility(dplus, positive_mass_on_student_topk, eps)
    f_value = future_compatibility(alpha, next_compatibility, eps)
    if f_value is None:
        return StateRecoverability(
            float(dplus), a_value, float("nan"), float("nan"), False
        )
    b_value = float(np.clip(a_value * f_value, 0.0, 1.0))
    return StateRecoverability(float(dplus), a_value, f_value, b_value, True)


def geometric_mean_bridgeability(values: Iterable[float], floor: float = 1e-6) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("At least one finite bridgeability value is required")
    return float(np.exp(np.mean(np.log(np.clip(array, floor, 1.0)))))


def aggregate_prompt(states: Iterable[StateRecoverability]) -> PromptRecoverability:
    valid = [state for state in states if state.valid]
    if not valid:
        return PromptRecoverability(0.0, 0.0, 0.0, 0)
    need = float(np.clip(np.mean([state.dplus for state in valid]), 0.0, 1.0))
    recoverability = geometric_mean_bridgeability(
        state.bridgeability for state in valid
    )
    teachability = float(np.clip(need * recoverability, 0.0, 1.0))
    return PromptRecoverability(need, recoverability, teachability, len(valid))
