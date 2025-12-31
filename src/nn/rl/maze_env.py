from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


class MazeEnv:
    """
    Minimal full-grid maze environment wrapper.

    Replace internals with your maze state, transitions, and rewards.
    """

    def __init__(self, max_steps: int = 200, reward_goal: float = 1.0, reward_step: float = -0.01):
        self.max_steps = max_steps
        self.reward_goal = reward_goal
        self.reward_step = reward_step
        self.t = 0
        self.state = None
        self.done = False

    def reset(self, maze: np.ndarray) -> np.ndarray:
        """Reset environment with a new maze."""
        self.state = maze.copy()
        self.t = 0
        self.done = False
        return self._observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Apply action and return (obs, reward, done, info)."""
        if self.done:
            return self._observation(), 0.0, True, {}

        # TODO: apply action to state; determine terminal and reward
        reward = self.reward_step
        self.t += 1

        if self._reached_goal():
            reward += self.reward_goal
            self.done = True
        elif self.t >= self.max_steps:
            self.done = True

        return self._observation(), reward, self.done, {}

    def _observation(self) -> np.ndarray:
        """Return full-grid observation (placeholder)."""
        return self.state

    def _reached_goal(self) -> bool:
        """Check goal condition (placeholder)."""
        return False
