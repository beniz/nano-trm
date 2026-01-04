from __future__ import annotations

from typing import Dict, Iterator

import torch


class RolloutBuffer:
    """
    Simple rollout buffer for PPO.
    Stores transitions and yields minibatches by index.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.storage: Dict[str, torch.Tensor] = {}
        self.ptr = 0
        self.num_envs: int | None = None

    def reset(self, num_envs: int | None = None) -> None:
        self.storage = {}
        self.ptr = 0
        if num_envs is not None:
            self.num_envs = num_envs

    def add(self, t: int, env_idx: int, **kwargs) -> None:
        """
        Store transition at time t for environment env_idx.
        Expects reset(num_envs) to have been called so we know num_envs.
        """
        if self.num_envs is None:
            raise RuntimeError("Call reset(num_envs=...) before adding to RolloutBuffer.")
        if not self.storage:
            for k, v in kwargs.items():
                shape = (self.capacity, self.num_envs) + tuple(v.shape)
                self.storage[k] = torch.zeros(shape, dtype=v.dtype, device=v.device)
        for k, v in kwargs.items():
            self.storage[k][t, env_idx] = v
        self.ptr = max(self.ptr, t + 1)

    def __len__(self) -> int:
        return self.ptr

    def as_batch(self) -> Dict[str, torch.Tensor]:
        return {k: v[: self.ptr] for k, v in self.storage.items()}

    def iter_minibatches(self, minibatch_size: int) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Flattens time and env dimensions for sampling minibatches.
        Expects stored tensors shaped [T, N, ...].
        """
        if self.ptr == 0:
            return
        # Flatten first two dims (T * N, ...)
        flat_storage = {k: v[: self.ptr].reshape(-1, *v.shape[2:]) for k, v in self.storage.items()}
        total = next(iter(flat_storage.values())).shape[0]
        idx = torch.randperm(total, device=next(iter(flat_storage.values())).device)
        for start in range(0, total, minibatch_size):
            mb_idx = idx[start : start + minibatch_size]
            minibatch = {k: v[mb_idx] for k, v in flat_storage.items()}
            minibatch["indices"] = mb_idx
            yield minibatch
