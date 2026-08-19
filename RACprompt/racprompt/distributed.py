from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

try:
    import torch
    import torch.distributed as dist
except (
    ImportError
):  # Pure batching helpers remain locally testable without the GPU stack.
    torch = None  # type: ignore[assignment]
    dist = None  # type: ignore[assignment]


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed(timeout_minutes: int = 60) -> DistributedContext:
    if torch is None:
        raise ImportError("PyTorch is required for distributed training")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        if world_size > 1:
            raise RuntimeError("NCCL multi-process training requires CUDA")
        return DistributedContext(0, 0, 1, torch.device("cpu"))
    torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl", timeout=timedelta(minutes=timeout_minutes))
    return DistributedContext(
        rank, local_rank, world_size, torch.device("cuda", local_rank)
    )


def split_sizes(total: int, world_size: int) -> list[int]:
    if total < 0 or world_size <= 0:
        raise ValueError("total must be nonnegative and world_size positive")
    base, remainder = divmod(total, world_size)
    return [base + (rank < remainder) for rank in range(world_size)]


def split_offsets(total: int, world_size: int) -> list[tuple[int, int]]:
    sizes = split_sizes(total, world_size)
    offsets: list[tuple[int, int]] = []
    start = 0
    for size in sizes:
        offsets.append((start, start + size))
        start += size
    return offsets


def broadcast_indices(
    indices: torch.Tensor | None, total: int, context: DistributedContext
) -> torch.Tensor:
    if context.is_main:
        if indices is None or indices.numel() != total:
            raise ValueError("Rank 0 must provide exactly total sampled indices")
        shared = indices.to(device=context.device, dtype=torch.long)
    else:
        shared = torch.empty(total, device=context.device, dtype=torch.long)
    if context.world_size > 1:
        dist.broadcast(shared, src=0)
    start, end = split_offsets(total, context.world_size)[context.rank]
    return shared[start:end].cpu()


def ddp_local_loss_scale(world_size: int, global_batch_size: int) -> float:
    """Scale a local sum so DDP's gradient average is the exact global mean."""
    return float(world_size) / float(global_batch_size)


def gather_objects(local: Any, context: DistributedContext) -> list[Any] | None:
    if context.world_size == 1:
        return [local]
    gathered: list[Any] | None = (
        [None] * context.world_size if context.is_main else None
    )
    dist.gather_object(local, gathered, dst=0)
    return gathered


def broadcast_object(value: Any, context: DistributedContext) -> Any:
    if context.world_size == 1:
        return value
    payload = [value if context.is_main else None]
    dist.broadcast_object_list(payload, src=0, device=context.device)
    return payload[0]


def barrier(context: DistributedContext) -> None:
    if context.world_size > 1:
        dist.barrier()


def reduce_sum(value: float, context: DistributedContext) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=context.device)
    if context.world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def cleanup_distributed() -> None:
    if dist is not None and dist.is_initialized():
        dist.destroy_process_group()
