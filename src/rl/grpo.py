from typing import Callable

import torch 
import numpy as np

def compute_group_normalized_rewards(
                                    reward_fn: Callable[[str, str], dict[str, float]],
                                    rollout_responses: list[str],
                                    repeated_ground_truths: list[str],
                                    group_size: int,
                                    advantage_eps: float,
                                    normalize_by_std: bool,
                                    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    
    raw_rewards: list[float] = [reward_fn(response, ground_truth)['reward'] for response, ground_truth in zip(rollout_responses, repeated_ground_truths, strict=True)]
    assert len(raw_rewards) % group_size == 0
    
    advantages: list[float] = []
    num_groups = len(raw_rewards) // group_size
    for group_idx in range(num_groups):
        group_rewards = raw_rewards[group_idx*group_size: (group_idx+1) * group_size]
        mean = sum(group_rewards) / group_size
        denom = np.std(group_rewards, ddof=1) + advantage_eps if normalize_by_std else 1

        advantages.extend([(reward - mean) / denom for reward in group_rewards])

    return torch.tensor(advantages), torch.Tensor(raw_rewards), {}

    

    
    