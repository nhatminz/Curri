from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CriticalState:
    position: int
    reasons: tuple[str, ...]


def _far_enough(position: int, selected: dict[int, set[str]], gap: int) -> bool:
    return all(abs(position - other) >= gap for other in selected)


def select_critical_states(
    dplus: np.ndarray,
    compatibility: np.ndarray,
    target: int = 24,
    num_segments: int = 12,
    global_peaks: int = 8,
    change_points: int = 4,
    change_lag: int = 32,
    min_gap_tokens: int = 32,
) -> list[CriticalState]:
    """Select using Dplus coverage/peaks and compatibility changes, never Dplus*C."""
    dplus = np.asarray(dplus, dtype=np.float64)
    compatibility = np.asarray(compatibility, dtype=np.float64)
    if dplus.ndim != 1 or compatibility.shape != dplus.shape:
        raise ValueError("dplus and compatibility must be same-length vectors")
    length = dplus.size
    if length == 0 or target <= 0:
        return []
    target = min(int(target), length)
    selected: dict[int, set[str]] = {}

    def add(position: int, reason: str, gap: int, force_existing: bool = True) -> bool:
        position = int(position)
        if position in selected:
            if force_existing:
                selected[position].add(reason)
            return False
        if len(selected) >= target or not _far_enough(position, selected, gap):
            return False
        selected[position] = {reason}
        return True

    # A: exact temporal coverage via approximately equal non-empty segments.
    for segment in np.array_split(
        np.arange(length, dtype=np.int64), min(num_segments, length)
    ):
        if segment.size:
            position = int(segment[np.argmax(dplus[segment])])
            # Segment maxima are always retained; adjacent short segments may be closer than the NMS gap.
            if position in selected:
                selected[position].add("segment")
            elif len(selected) < target:
                selected[position] = {"segment"}

    # B: correction peaks, with greedy NMS against all existing choices.
    added = 0
    for position in np.argsort(-dplus, kind="stable"):
        if int(position) in selected:
            selected[int(position)].add("global_peak")
            continue
        if add(int(position), "global_peak", min_gap_tokens):
            added += 1
            if added >= global_peaks:
                break

    # C: largest lagged compatibility changes. Dplus is deliberately absent.
    if change_lag > 0 and length > change_lag:
        candidates = np.arange(change_lag, length, dtype=np.int64)
        changes = np.abs(compatibility[change_lag:] - compatibility[:-change_lag])
        order = candidates[np.argsort(-changes, kind="stable")]
        added = 0
        for position in order:
            if int(position) in selected:
                selected[int(position)].add("change_point")
                continue
            if add(int(position), "change_point", min_gap_tokens):
                added += 1
                if added >= change_points:
                    break

    # D: highest remaining Dplus; progressively relax the gap when necessary.
    gap_schedule: list[int] = []
    gap = max(0, int(min_gap_tokens))
    while gap > 1:
        gap_schedule.append(gap)
        gap //= 2
    gap_schedule.extend([1, 0])
    for relaxed_gap in dict.fromkeys(gap_schedule):
        for position in np.argsort(-dplus, kind="stable"):
            if len(selected) >= target:
                break
            if int(position) not in selected:
                add(int(position), "fill", relaxed_gap)
        if len(selected) >= target:
            break

    return [
        CriticalState(position=position, reasons=tuple(sorted(reasons)))
        for position, reasons in sorted(selected.items())
    ]
