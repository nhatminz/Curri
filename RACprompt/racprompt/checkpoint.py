from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import RACConfig, config_to_dict
from .curriculum import CurriculumState
from .models import unwrap_model


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def checkpoint_step(path: str | Path) -> int:
    name = Path(path).name
    if not name.startswith("step_"):
        raise ValueError(f"Not a step checkpoint: {path}")
    return int(name.removeprefix("step_"))


def find_latest_checkpoint(checkpoint_root: str | Path) -> Path | None:
    root = Path(checkpoint_root)
    candidates: list[tuple[int, Path]] = []
    for path in root.glob("step_*") if root.exists() else []:
        if path.is_dir() and (path / "trainer_state.pt").exists():
            try:
                candidates.append((checkpoint_step(path), path))
            except ValueError:
                pass
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def resolve_resume_path(value: str | None, checkpoint_root: str | Path) -> Path | None:
    if not value:
        return None
    if value.lower() == "latest":
        latest = find_latest_checkpoint(checkpoint_root)
        if latest is None:
            raise FileNotFoundError(f"No checkpoint found under {checkpoint_root}")
        return latest
    path = Path(value)
    if not (path / "trainer_state.pt").exists():
        raise FileNotFoundError(f"Invalid checkpoint: {path}")
    return path


def save_checkpoint(
    checkpoint_root: str | Path,
    step: int,
    student: torch.nn.Module,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    curriculum: CurriculumState,
    rng_states_by_rank: list[dict[str, Any]],
    data_metadata: dict[str, Any],
    config: RACConfig,
    keep_last_n: int,
) -> Path:
    root = Path(checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)
    final_path = root / f"step_{step:06d}"
    temporary = root / f".step_{step:06d}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    model_dir = temporary / "student"
    unwrap_model(student).save_pretrained(model_dir, safe_serialization=True)
    tokenizer.save_pretrained(model_dir)
    torch.save(
        {
            "step": int(step),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "curriculum": curriculum.state_dict(),
            "rng_states_by_rank": rng_states_by_rank,
            "data_metadata": data_metadata,
            "config": config_to_dict(config),
        },
        temporary / "trainer_state.pt",
    )
    with (temporary / "complete.json").open("w", encoding="utf-8") as handle:
        json.dump({"step": step}, handle)
    if final_path.exists():
        shutil.rmtree(final_path)
    os.replace(temporary, final_path)
    rotate_checkpoints(root, keep_last_n, protect=final_path)
    return final_path


def rotate_checkpoints(
    root: str | Path, keep_last_n: int, protect: Path | None = None
) -> None:
    if keep_last_n <= 0:
        return
    candidates: list[tuple[int, Path]] = []
    for path in Path(root).glob("step_*"):
        try:
            candidates.append((checkpoint_step(path), path))
        except ValueError:
            pass
    candidates.sort(reverse=True)
    for _, path in candidates[keep_last_n:]:
        if protect is None or path.resolve() != protect.resolve():
            shutil.rmtree(path)


def load_training_state(
    checkpoint: str | Path,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    curriculum: CurriculumState,
    rank: int,
) -> dict[str, Any]:
    state = torch.load(
        Path(checkpoint) / "trainer_state.pt", map_location="cpu", weights_only=False
    )
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    curriculum.load_state_dict(state["curriculum"])
    rng_states = state["rng_states_by_rank"]
    if rank >= len(rng_states):
        raise ValueError(
            f"Checkpoint has RNG state for {len(rng_states)} ranks, but resume rank={rank}. "
            "Exact resume requires the same world size."
        )
    restore_rng_state(rng_states[rank])
    return state
