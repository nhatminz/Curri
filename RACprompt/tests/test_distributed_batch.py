import numpy as np

from racprompt.distributed import ddp_local_loss_scale, split_sizes


def test_global_batch_split_is_exact_for_world_sizes_one_to_five():
    for world_size in range(1, 6):
        sizes = split_sizes(32, world_size)
        assert sum(sizes) == 32
        assert max(sizes) - min(sizes) <= 1


def test_ddp_loss_scaling_matches_single_process_global_mean():
    values = np.linspace(0.1, 3.2, 32)
    reference = values.mean()
    for world_size in range(1, 6):
        sizes = split_sizes(32, world_size)
        cursor = 0
        scaled_local_losses = []
        for size in sizes:
            local_sum = values[cursor : cursor + size].sum()
            cursor += size
            scaled_local_losses.append(local_sum * ddp_local_loss_scale(world_size, 32))
        # DDP averages one scaled gradient/loss contribution from each rank.
        assert np.isclose(np.mean(scaled_local_losses), reference)
