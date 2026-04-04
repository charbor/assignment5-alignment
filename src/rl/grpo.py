from typing import Callable, Literal

import torch 
import numpy as np
from jaxtyping import Float

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

def compute_naive_policy_gradient_loss(
        raw_rewards_or_advantages: torch.Tensor,
        policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    return - policy_log_probs * raw_rewards_or_advantages

def compute_grpo_clip_loss(
        advantages: Float[torch.Tensor, "batch 1"], # batch_size, 1
        policy_log_probs: Float[torch.Tensor, "batch seq"], # batch_size seq_len
        old_log_probs: Float[torch.Tensor, "batch seq"], # batch_size seq_len
        cliprange: float
        ) -> tuple[Float[torch.Tensor, "batch seq"], dict[str, torch.Tensor]]: 
    
    policy_ratio = torch.exp(policy_log_probs - old_log_probs)
    clipped_advantages = torch.where(advantages >= 0, (1 + cliprange) * advantages, (1 - cliprange) * advantages)

    token_losses: Float[torch.Tensor, "batch seq"] = - torch.min(policy_ratio * advantages, clipped_advantages)
    return token_losses, {}


def compute_policy_gradient_loss(
        policy_log_probs: Float[torch.Tensor, "batch seq"],
        loss_type: Literal['no_baseline', 'reinforce_with_baseline', 'grpo_clip'],
        raw_rewards: Float[torch.Tensor, "batch 1"] | None = None,
        advantages: Float[torch.Tensor, 'batch 1'] | None = None,
        old_log_probs: Float[torch.Tensor, "batch seq"] | None = None,
        cliprange: float | None = None
) -> tuple[Float[torch.Tensor, 'batch seq'], dict[str, torch.Tensor]]:
    match loss_type:
        case 'no_baseline':
            assert raw_rewards is not None
            return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {}
        case 'reinforce_with_baseline':
            assert advantages is not None
            return compute_naive_policy_gradient_loss(advantages, policy_log_probs), {}
        case 'grpo_clip':
            assert advantages is not None and old_log_probs is not None and cliprange is not None
            return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    raise ValueError(f"Unknown loss_type: {loss_type}")


    
    

def masked_mean(tensor: torch.Tensor,
                mask: torch.Tensor,
                dim: int | None = None):
    return torch.sum(tensor * mask, dim = dim) / torch.sum(mask, dim=dim)

    
def grpo_minibatch_train_step(
        policy_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        gradient_accumulation_steps: int,
        loss_type: Literal['no_baseline', 'reinforce_with_baseline', 'grpo_clip'],
        raw_rewards: torch.Tensor | None = None,
        advantages: torch.Tensor | None = None,
        old_log_probs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    '''Forward and backwards pass on microbatch.
    Return loss, metadata
    '''
    pass