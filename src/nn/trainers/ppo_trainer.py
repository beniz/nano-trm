from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule

from src.nn.rl.rollout_buffer import RolloutBuffer
from src.nn.rl.ppo_utils import compute_gae, ppo_clip_loss


class TRMPPOTrainer(LightningModule):
    """
    Lightning wrapper that adds PPO on top of a TRM backbone.

    The backbone stays model-only; this class owns rollouts, losses, and optimizer steps.
    """

    def __init__(
        self,
        backbone: nn.Module,
        policy_head: nn.Module,
        value_head: nn.Module,
        rollout_buffer: RolloutBuffer,
        optimizer_cfg: Optional[Dict[str, Any]] = None,
        scheduler_cfg: Optional[Dict[str, Any]] = None,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        ppo_epochs: int = 4,
        minibatch_size: int = 64,
    ) -> None:
        super().__init__()
        self.automatic_optimization = False  # manual optimization for PPO

        self.backbone = backbone
        self.policy_head = policy_head
        self.value_head = value_head
        self.rollout_buffer = rollout_buffer

        self.optimizer_cfg = optimizer_cfg or {}
        self.scheduler_cfg = scheduler_cfg or {}

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size

    def policy_value(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs the backbone and heads to produce policy logits and value.

        Assumes full-grid observation; adapt feature selection as needed.
        """
        feats = self.backbone.forward_features(batch)
        policy_logits = self.policy_head(feats["policy_input"])
        values = self.value_head(feats["value_input"]).squeeze(-1)
        return policy_logits, values

    def collect_rollouts(self) -> None:
        """
        Step environments with the current policy and fill the rollout buffer.

        Implement env reset/step and storage here.
        """
        raise NotImplementedError("Implement rollout collection loop")

    def ppo_update(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Run PPO epochs over minibatches from the rollout buffer.
        """
        opt = self.optimizers()

        advantages, returns = compute_gae(
            rewards=batch["rewards"],
            values=batch["values"],
            dones=batch["dones"],
            gamma=self.gamma,
            lam=self.gae_lambda,
        )

        stats: Dict[str, float] = {}
        for _ in range(self.ppo_epochs):
            for mb in self.rollout_buffer.iter_minibatches(self.minibatch_size):
                logits, values = self.policy_value(mb["obs"])

                loss, loss_terms = ppo_clip_loss(
                    logits=logits,
                    actions=mb["actions"],
                    old_logprobs=mb["logprobs"],
                    values=values,
                    old_values=mb["values"],
                    advantages=advantages[mb["indices"]],
                    returns=returns[mb["indices"]],
                    clip_eps=self.clip_eps,
                    entropy_coef=self.entropy_coef,
                    value_coef=self.value_coef,
                )

                opt.zero_grad()
                self.manual_backward(loss)
                opt.step()

                # Collect last minibatch stats for logging
                stats = {k: float(v) for k, v in loss_terms.items()}

        return loss, stats

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """
        Orchestrates rollout collection and PPO updates.

        The `batch` argument can be ignored if you source mazes internally.
        """
        # Fill buffer using current policy
        self.collect_rollouts()

        # Run PPO updates
        loss, stats = self.ppo_update(batch)

        # Log PPO metrics
        for name, value in stats.items():
            self.log(f"ppo/{name}", value, on_step=True, on_epoch=False, prog_bar=(name == "policy_loss"))

        return loss

    def configure_optimizers(self):
        """
        Build optimizer/scheduler for backbone + heads.
        """
        raise NotImplementedError("Configure optimizers for PPO training")
