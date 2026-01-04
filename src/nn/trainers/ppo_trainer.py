from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from lightning import LightningModule
from torch.distributions import Categorical
import hydra
from tqdm import tqdm

from src.nn.rl.maze_env import MazeEnv
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
        rollout_steps: int = 128,
        env_class: Any = MazeEnv,
    ) -> None:
        super().__init__()
        self.automatic_optimization = False  # manual optimization for PPO
        # Instantiate nested configs if they are DictConfig/Plain.
        self.backbone = backbone if isinstance(backbone, nn.Module) else hydra.utils.instantiate(backbone)
        self.policy_head = (
            policy_head if isinstance(policy_head, nn.Module) else hydra.utils.instantiate(policy_head)
        )
        self.value_head = value_head if isinstance(value_head, nn.Module) else hydra.utils.instantiate(value_head)
        self.rollout_buffer = (
            rollout_buffer
            if isinstance(rollout_buffer, RolloutBuffer)
            else hydra.utils.instantiate(rollout_buffer)
        )

        self.optimizer_cfg = optimizer_cfg or {}
        self.scheduler_cfg = scheduler_cfg or {}

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.rollout_steps = rollout_steps
        self.env_class = env_class

        self.envs: List[MazeEnv] = []
        self.last_values: Optional[torch.Tensor] = None
        self.num_envs: Optional[int] = None
        self.train_reward_accum: List[float] = []
        self.train_path_accum: List[int] = []
        self.test_envs: List[MazeEnv] = []
        self.test_mazes: Optional[np.ndarray] = None
        self.test_reward_accum: List[float] = []
        self.test_path_accum: List[int] = []
        self._ppo_epoch_counter: int = 0  # track PPO epochs for logging cadence
        self.eval_config: Optional[Dict[str, Any]] = None  # set externally for periodic eval

    def policy_value(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs the backbone and heads to produce policy logits and value.
        Expects obs shape [B, H, W] tokenized grid.
        """
        puzzle_ids = torch.zeros(obs.shape[0], dtype=torch.int32, device=obs.device)
        feats = self.backbone.forward_features({"input": obs, "puzzle_identifiers": puzzle_ids})
        policy_logits = self.policy_head(feats["policy_input"])
        values = self.value_head(feats["value_input"]).squeeze(-1)
        return policy_logits, values

    def _ensure_envs(self, num_envs: int, mazes: np.ndarray) -> List[MazeEnv]:
        if not self.envs or len(self.envs) != num_envs:
            self.envs = [self.env_class() for _ in range(num_envs)]
        # Reset each env with provided maze
        obs_list = []
        for idx, (env, maze) in enumerate(zip(self.envs, mazes)):
            obs = env.reset(maze)
            if obs is None:
                raise RuntimeError(f"_ensure_envs: env.reset returned None for index {idx}")
            obs_list.append(obs)
        return obs_list

    def collect_rollouts(self, mazes: torch.Tensor) -> None:
        """
        Step environments with the current policy and fill the rollout buffer.
        `mazes` is expected to be shape [N, H, W] (tokens).
        """
        if mazes.numel() == 0:
            print("[PPO] Empty batch of mazes; skipping rollout collection.")
            self.rollout_buffer.reset()
            return
        mazes_np = mazes.detach().cpu().numpy()
        if mazes_np.ndim == 2:
            # Flattened sequences -> reshape to grids
            grid_size = int(mazes_np.shape[1] ** 0.5)
            mazes_np = mazes_np.reshape(mazes_np.shape[0], grid_size, grid_size)
        self.num_envs = mazes_np.shape[0]
        if self.num_envs == 0:
            raise RuntimeError("collect_rollouts received zero mazes in batch.")
        obs_list = self._ensure_envs(self.num_envs, mazes_np)
        if len(obs_list) == 0:
            raise RuntimeError(f"[PPO] No observations after env reset (mazes shape={mazes_np.shape}).")

        self.rollout_buffer.reset(self.num_envs)
        device = self.device
        episode_buffers: List[List[Dict[str, torch.Tensor]]] = [[] for _ in range(self.num_envs)]
        any_kept = False
        reward_sums = [0.0 for _ in range(self.num_envs)]
        step_counts = [0 for _ in range(self.num_envs)]

        for t in tqdm(
            range(self.rollout_steps),
            desc="Collect rollouts",
            leave=False,
            total=max(self.rollout_steps, 1),
        ):
            if len(obs_list) == 0:
                raise RuntimeError(
                    f"[PPO] obs_list became empty during rollouts at step {t} (mazes shape={mazes_np.shape})"
                )
            obs_np = np.stack(obs_list)  # [N, H, W]
            obs_t = torch.tensor(obs_np, device=device, dtype=torch.long)

            logits, values = self.policy_value(obs_t)
            dist = Categorical(logits=logits)
            actions = dist.sample()
            logprobs = dist.log_prob(actions)

            actions_np = actions.detach().cpu().numpy()
            next_obs_list = []
            rewards: List[float] = []
            dones: List[bool] = []

            for idx, (env, act) in enumerate(zip(self.envs, actions_np)):
                next_obs, reward, done, info = env.step(int(act))
                next_obs_list.append(next_obs)
                rewards.append(reward)
                dones.append(done)

                reward_sums[idx] += float(reward)
                step_counts[idx] += 1

                step_entry = {
                    "t": t,
                    "obs": obs_t[idx].detach(),
                    "actions": actions[idx].detach(),
                    "rewards": torch.tensor(float(reward), device=device, dtype=torch.float32),
                    "dones": torch.tensor(bool(done), device=device, dtype=torch.bool),
                    "values": values[idx].detach(),
                    "logprobs": logprobs[idx].detach(),
                }
                episode_buffers[idx].append(step_entry)

                if done and info.get("reached_goal", False):
                    path_len = info.get("steps", self.rollout_steps)
                    ep_reward = reward_sums[idx]
                    for entry in episode_buffers[idx]:
                        t_entry = entry.pop("t")
                        self.rollout_buffer.add(t_entry, idx, **entry)
                    any_kept = True
                    self.train_reward_accum.append(ep_reward)
                    self.train_path_accum.append(int(path_len))
                    episode_buffers[idx] = []
                    reward_sums[idx] = 0.0
                    step_counts[idx] = 0
                elif done:
                    # Episode ended without goal: keep stats but do not add to buffer
                    self.train_reward_accum.append(reward_sums[idx])
                    self.train_path_accum.append(step_counts[idx])
                    episode_buffers[idx] = []
                    reward_sums[idx] = 0.0
                    step_counts[idx] = 0

            # Use the freshly computed observations for the next timestep
            obs_list = next_obs_list
        # Prepare final observations for bootstrapping values (one per env)
        final_obs_list = []
        for env, base_maze in zip(self.envs, mazes_np):
            if env.done:
                final_obs_list.append(env.reset(base_maze))
            else:
                final_obs_list.append(env.grid)

        # Bootstrap value for final states
        obs_np = np.stack(final_obs_list)
        obs_t = torch.tensor(obs_np, device=device, dtype=torch.long)
        _, last_values = self.policy_value(obs_t)
        self.last_values = last_values.detach()

        # Include truncated episodes (not finished within rollout) in stats
        for r_sum, steps in zip(reward_sums, step_counts):
            if steps > 0:
                self.train_reward_accum.append(r_sum)
                self.train_path_accum.append(steps)

        # Log rollout-level stats (avg reward/path) after collection
        #print('len train_reward_accum=', len(self.train_reward_accum))
        if self.train_reward_accum:
            avg_train_reward = float(np.mean(self.train_reward_accum))
            avg_train_path = float(np.mean(self.train_path_accum))
            self.log("ppo/train_avg_reward", avg_train_reward, on_step=True, on_epoch=False, prog_bar=False)
            self.log("ppo/train_avg_path_len", avg_train_path, on_step=True, on_epoch=False, prog_bar=False)
            self.train_reward_accum.clear()
            self.train_path_accum.clear()
        if not any_kept:
            # No successful episodes; clear buffer to avoid KeyErrors downstream
            print("[PPO] No goal-reaching rollouts collected this cycle.")
            self.rollout_buffer.reset()

    def _eval_envs(self, mazes_np: np.ndarray) -> Tuple[float, float]:
        """Greedy eval (argmax policy) on provided mazes. Returns avg_reward, avg_path_len."""
        if self.test_mazes is None or self.test_mazes.shape[0] != mazes_np.shape[0]:
            self.test_envs = [self.env_class() for _ in range(mazes_np.shape[0])]
            self.test_mazes = mazes_np

        rewards = []
        paths = []
        for env, maze in tqdm(
            list(zip(self.test_envs, mazes_np)),
            desc="Eval rollouts",
            leave=False,
            total=len(mazes_np),
        ):
            obs = env.reset(maze)
            done = False
            ep_reward = 0.0
            steps = 0
            while not done and steps < maze.size:
                obs_t = torch.tensor(obs, device=self.device, dtype=torch.long).unsqueeze(0)
                logits, _ = self.policy_value(obs_t)
                action = torch.argmax(logits, dim=-1).item()
                obs, r, done, info = env.step(action)
                ep_reward += float(r)
                steps += 1
                if done:
                    steps = info.get("steps", steps)
            rewards.append(ep_reward)
            paths.append(steps)
        return float(np.mean(rewards)), float(np.mean(paths))

    def ppo_update(self) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Run PPO epochs over minibatches from the rollout buffer.
        """
        opt = self.optimizers()
        if not isinstance(opt, list):
            opt = [opt]
        opt = opt[0]  # single optimizer expected

        if len(self.rollout_buffer) == 0:
            # Nothing collected (no goal-reaching episodes); skip update
            zero = torch.tensor(0.0, device=self.device, requires_grad=True)
            return zero, {}

        rollouts = self.rollout_buffer.as_batch()
        # Debug shapes (commented out; enable for debugging rollout/GAE shapes)
        # print(
        #     "[PPO] rollouts shapes:",
        #     {k: tuple(v.shape) for k, v in rollouts.items()},
        #     "last_values:",
        #     None if self.last_values is None else tuple(self.last_values.shape),
        # )

        advantages, returns = compute_gae(
            rewards=rollouts["rewards"],
            values=rollouts["values"],
            dones=rollouts["dones"],
            gamma=self.gamma,
            lam=self.gae_lambda,
            last_values=self.last_values,
        )

        # Flatten time/env dims
        def flatten(x: torch.Tensor) -> torch.Tensor:
            return x.reshape(-1, *x.shape[2:]) if x.ndim > 2 else x.reshape(-1)

        flat_obs = flatten(rollouts["obs"])
        flat_actions = flatten(rollouts["actions"])
        flat_old_logprobs = flatten(rollouts["logprobs"])
        flat_old_values = flatten(rollouts["values"])
        flat_adv = flatten(advantages)
        flat_returns = flatten(returns)

        stats: Dict[str, float] = {}
        for _ in tqdm(
            range(self.ppo_epochs),
            desc="PPO epochs",
            leave=False,
            total=max(self.ppo_epochs - 1, 1),
        ):
            mb_iter = list(self.rollout_buffer.iter_minibatches(self.minibatch_size))
            total_mb = len(mb_iter)
            accum_steps = max(1, int(self.optimizer_cfg.get("grad_accum_steps", 1)))
            opt.zero_grad()

            for i, mb in enumerate(mb_iter, 1):
                logits, values = self.policy_value(mb["obs"])

                loss, loss_terms = ppo_clip_loss(
                    logits=logits,
                    actions=mb["actions"],
                    old_logprobs=mb["logprobs"],
                    values=values,
                    old_values=mb["values"],
                    advantages=flat_adv[mb["indices"]],
                    returns=flat_returns[mb["indices"]],
                    clip_eps=self.clip_eps,
                    entropy_coef=self.entropy_coef,
                    value_coef=self.value_coef,
                )

                loss = loss / accum_steps
                self.manual_backward(loss)

                if i % accum_steps == 0 or i == total_mb:
                    opt.step()
                    opt.zero_grad()

                stats = {k: float(v) for k, v in loss_terms.items()}
                # print(
                #     f"loss={loss.item():.4f} policy={stats.get('policy_loss', 0):.4f} "
                #     f"value={stats.get('value_loss', 0):.4f} entropy={stats.get('entropy', 0):.4f}",
                #     end="\r",
                # )
            self._ppo_epoch_counter += 1

        return loss, stats

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """
        Orchestrates rollout collection and PPO updates.
        Expects batch to contain maze grids (key: 'maze' or 'input') shaped [N, H, W].
        """
        mazes = batch.get("maze") if isinstance(batch, dict) else None
        if mazes is None and isinstance(batch, dict):
            mazes = batch.get("input")
        if mazes is None:
            raise ValueError("PPO trainer expects batch['maze'] or batch['input'] with maze grids.")
        if mazes.numel() == 0:
            raise RuntimeError("Received empty maze batch in training_step; check dataloader.")

        # Fill buffer using current policy
        self.collect_rollouts(mazes)

        # Run PPO updates
        loss, stats = self.ppo_update()

        # Log PPO metrics every full PPO cycle
        if self._ppo_epoch_counter % self.ppo_epochs == 0:
            for name, value in stats.items():
                self.log(f"ppo/{name}", value, on_step=True, on_epoch=False, prog_bar=(name == "policy_loss"))

        # Optional test eval (if batch supplies test_maze)
        if isinstance(batch, dict) and batch.get("test_maze") is not None:
            test_mazes = batch["test_maze"].detach().cpu().numpy()
            avg_test_reward, avg_test_path = self._eval_envs(test_mazes)
            self.log("ppo/test_avg_reward", avg_test_reward, on_step=True, on_epoch=False, prog_bar=False)
            self.log("ppo/test_avg_path_len", avg_test_path, on_step=True, on_epoch=False, prog_bar=False)

        # Eval on val set after each rollout+update cycle (if provided at init)
        eval_cfg = getattr(self, "eval_config", None)
        if eval_cfg and self.trainer:
            interval = max(1, int(eval_cfg.get("interval", 1)))
            if self._ppo_epoch_counter % interval == 0:
                eval_mazes = eval_cfg.get("mazes")
                if eval_mazes is not None:
                    avg_eval_reward, avg_eval_path = self._eval_envs(eval_mazes)
                    self.log("ppo/eval_avg_reward", avg_eval_reward, on_step=True, on_epoch=False, prog_bar=False)
                    self.log("ppo/eval_avg_path_len", avg_eval_path, on_step=True, on_epoch=False, prog_bar=False)

        return loss

    def configure_optimizers(self):
        """
        Build optimizer/scheduler for backbone + heads.
        """
        # Exclude q_head (unused in PPO). If you want to train it later, add it back here.
        def param_filter(module: nn.Module):
            for name, p in module.named_parameters():
                if "q_head" in name:
                    continue
                yield p

        params = list(param_filter(self.backbone)) + list(self.policy_head.parameters()) + list(
            self.value_head.parameters()
        )
        opt = torch.optim.AdamW(
            params,
            lr=self.optimizer_cfg.get("lr", 3e-4),
            weight_decay=self.optimizer_cfg.get("weight_decay", 0.01),
        )
        return opt
