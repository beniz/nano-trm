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

    def reset(self) -> None:
        self.storage = {}
        self.ptr = 0

    def add(self, **kwargs) -> None:
        if not self.storage:
            for k, v in kwargs.items():
                shape = (self.capacity,) + tuple(v.shape[1:])
                self.storage[k] = torch.zeros(shape, dtype=v.dtype, device=v.device)
        idx = self.ptr
        for k, v in kwargs.items():
            self.storage[k][idx] = v
        self.ptr += 1

    def __len__(self) -> int:
        return self.ptr

    def as_batch(self) -> Dict[str, torch.Tensor]:
        return {k: v[: self.ptr] for k, v in self.storage.items()}

    def iter_minibatches(self, minibatch_size: int) -> Iterator[Dict[str, torch.Tensor]]:
        idx = torch.randperm(self.ptr)
        for start in range(0, self.ptr, minibatch_size):
            mb_idx = idx[start : start + minibatch_size]
            minibatch = {k: v[mb_idx] for k, v in self.storage.items()}
            minibatch["indices"] = mb_idx
            yield minibatch
