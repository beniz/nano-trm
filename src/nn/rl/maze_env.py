from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


class MazeEnv:
    """
    Full-grid maze environment with 4-way actions.

    Reward scheme (per user spec):
    - Step penalty: -1 / (num_cells)
    - Goal reward: 0 (the incentive is purely shortest-path via step cost)
    """

    ACTIONS = {
        0: (-1, 0),  # up
        1: (1, 0),   # down
        2: (0, -1),  # left
        3: (0, 1),   # right
    }

    def __init__(self, max_steps: int | None = None) -> None:
        self.max_steps = max_steps
        self.t = 0
        self.done = False

        # Token mapping supports either raw chars (ord values) or pre-tokenized grids (1-5).
        self.token_map: Dict[str, int] = {}
        self.grid: np.ndarray | None = None
        self.agent_pos: Tuple[int, int] | None = None
        self.goal_pos: Tuple[int, int] | None = None
        self.step_cost: float = 0.0

    def _init_token_map(self, grid: np.ndarray) -> None:
        # Detect if grid is char-coded (ASCII) or token-coded (1-5 per build_maze_dataset).
        unique_vals = np.unique(grid)
        if unique_vals.max() <= 5:
            # Token-coded (#=1, space=2, S=3, G=4, path=5)
            self.token_map = {"wall": 1, "space": 2, "start": 3, "goal": 4, "path": 5}
        else:
            # ASCII-coded
            self.token_map = {
                "wall": ord("#"),
                "space": ord(" "),
                "start": ord("S"),
                "goal": ord("G"),
                "path": ord("o"),
            }

    def _find_positions(self, grid: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        start_idxs = np.argwhere(grid == self.token_map["start"])
        goal_idxs = np.argwhere(grid == self.token_map["goal"])
        if start_idxs.size == 0 or goal_idxs.size == 0:
            raise ValueError("Maze must contain start 'S' and goal 'G'")
        return tuple(start_idxs[0]), tuple(goal_idxs[0])

    def reset(self, maze: np.ndarray) -> np.ndarray:
        """Reset environment with a new maze (expects a 2D grid)."""
        if maze.ndim != 2:
            raise ValueError(f"Expected 2D maze grid, got shape {maze.shape}")

        self._init_token_map(maze)
        self.grid = maze.copy()
        self.agent_pos, self.goal_pos = self._find_positions(self.grid)

        h, w = self.grid.shape
        num_cells = h * w
        self.step_cost = -1.0 / float(num_cells)
        # Always cap episodes to grid size (num cells)
        self.max_steps = num_cells

        self.t = 0
        self.done = False
        return self._observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Apply action and return (obs, reward, done, info)."""
        if self.done:
            return self._observation(), 0.0, True, {"terminal": True}

        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action {action}, expected 0-3")

        dr, dc = self.ACTIONS[action]
        r, c = self.agent_pos
        nr, nc = r + dr, c + dc

        # Check bounds and walls
        if (
            nr < 0
            or nr >= self.grid.shape[0]
            or nc < 0
            or nc >= self.grid.shape[1]
            or self.grid[nr, nc] == self.token_map["wall"]
        ):
            # Invalid move: stay in place, still incur step cost
            nr, nc = r, c

        self.agent_pos = (nr, nc)
        self._update_grid_agent()

        self.t += 1
        reached_goal = self.agent_pos == self.goal_pos
        timeup = self.t >= self.max_steps
        self.done = reached_goal or timeup

        reward = 0.0 if reached_goal else self.step_cost
        info = {"reached_goal": reached_goal, "time_limit": timeup, "steps": self.t}
        return self._observation(), reward, self.done, info

    def _update_grid_agent(self) -> None:
        # Clear previous agent positions (set to space unless goal)
        # This is a simple single-agent marker update; adjust if you need trails.
        self.grid[self.grid == self.token_map["start"]] = self.token_map["space"]
        r, c = self.agent_pos
        # Keep goal token if standing on it
        if (r, c) == self.goal_pos:
            self.grid[r, c] = self.token_map["goal"]
        else:
            self.grid[r, c] = self.token_map["start"]

    def _observation(self) -> np.ndarray:
        """Return full-grid observation (tokens)."""
        return self.grid
