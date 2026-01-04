from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch.distributions import Categorical


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lam: float,
    last_values: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generalized Advantage Estimation.

    rewards, values, dones: shape [T, N] (or [T] for single-env)
    last_values: shape [N] or scalar (bootstrap value for final state)
    """
    # Ensure rewards/values/dones are at least 2D [T, N]
    if rewards.ndim == 1:
        rewards = rewards.unsqueeze(1)
    if values.ndim == 1:
        values = values.unsqueeze(1)
    if dones.ndim == 1:
        dones = dones.unsqueeze(1)

    # Normalize last_values to shape [N]
    if last_values.ndim == 0:
        last_values = last_values.expand(values.shape[1])
    elif last_values.ndim > 1:
        last_values = last_values.view(-1)

    values_ext = torch.cat([values, last_values.unsqueeze(0)], dim=0)

    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros_like(values_ext[0])

    for t in reversed(range(T)):
        mask = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * values_ext[t + 1] * mask - values_ext[t]
        last_adv = delta + gamma * lam * mask * last_adv
        advantages[t] = last_adv

    returns = advantages + values
    return advantages, returns


def ppo_clip_loss(
    logits: torch.Tensor,
    actions: torch.Tensor,
    old_logprobs: torch.Tensor,
    values: torch.Tensor,
    old_values: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    clip_eps: float,
    entropy_coef: float,
    value_coef: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute PPO clipped policy loss + value loss + entropy.
    """
    dist = Categorical(logits=logits)
    logprobs = dist.log_prob(actions)
    entropy = dist.entropy().mean()

    ratios = torch.exp(logprobs - old_logprobs)
    adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    unclipped = ratios * adv
    clipped = torch.clamp(ratios, 1 - clip_eps, 1 + clip_eps) * adv
    policy_loss = -torch.min(unclipped, clipped).mean()

    value_clipped = old_values + torch.clamp(values - old_values, -clip_eps, clip_eps)
    value_losses = (values - returns).pow(2)
    value_losses_clipped = (value_clipped - returns).pow(2)
    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

    stats = {
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "entropy": entropy.detach(),
        "approx_kl": 0.5 * (logprobs - old_logprobs).pow(2).mean().detach(),
    }
    return loss, stats
