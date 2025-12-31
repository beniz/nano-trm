from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class MazeDataset(Dataset):
    """Lightweight dataset that serves maze grids for RL; can optionally load solutions."""

    def __init__(
        self,
        split_dir: str,
        grid_size: int,
        pad_value: int = 0,
        load_solutions: bool = False,
    ):
        self.grid_size = grid_size
        self.pad_value = pad_value
        inputs_path = os.path.join(split_dir, "all__inputs.npy")
        if os.path.exists(inputs_path):
            self.inputs = np.load(inputs_path)
        else:
            self.inputs = np.empty((0, grid_size * grid_size), dtype=np.uint8)

        self.labels = None
        if load_solutions:
            labels_path = os.path.join(split_dir, "all__labels.npy")
            if os.path.exists(labels_path):
                self.labels = np.load(labels_path)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int):
        grid = self.inputs[idx].reshape(self.grid_size, self.grid_size)
        maze = torch.from_numpy(grid.astype(np.int64))
        sample = {
            "maze": maze,
            "input": maze,  # for compatibility with trainers expecting 'input'
            "puzzle_identifiers": torch.tensor(0, dtype=torch.int32),  # single id per puzzle
        }
        if self.labels is not None:
            solution = torch.from_numpy(
                self.labels[idx].reshape(self.grid_size, self.grid_size).astype(np.int64)
            )
            sample["solution"] = solution
        return sample


class MazeRLDataModule(LightningDataModule):
    """
    DataModule for RL PPO training on mazes.

    Serves mazes as token grids; no labels are provided.
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 0,
        pad_value: int = 0,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pad_value = pad_value

        # These will be set in setup from metadata
        self.grid_size: Optional[int] = None
        self.vocab_size: Optional[int] = None
        self.seq_len: Optional[int] = None
        self.num_train_puzzles: int = 0
        self.num_puzzles: int = 0
        self.pad_value_resolved: int = pad_value
        self.max_grid_size: Optional[int] = None

        self.train_dataset: Optional[MazeDataset] = None
        self.val_dataset: Optional[MazeDataset] = None
        self.test_dataset: Optional[MazeDataset] = None

    def prepare_data(self) -> None:
        # No download; assumes data_dir already populated.
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        meta_path = os.path.join(self.data_dir, "metadata.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"metadata.json not found in {self.data_dir}")

        with open(meta_path) as f:
            meta = json.load(f)

        self.grid_size = int(meta["max_grid_size"])
        self.max_grid_size = self.grid_size
        self.vocab_size = int(meta["vocab_size"])
        self.seq_len = int(meta["seq_len"])

        train_dir = os.path.join(self.data_dir, "train")
        val_dir = os.path.join(self.data_dir, "val")
        test_dir = os.path.join(self.data_dir, "test")

        if stage in ("fit", None):
            self.train_dataset = MazeDataset(train_dir, self.grid_size, pad_value=self.pad_value)
            self.val_dataset = MazeDataset(val_dir, self.grid_size, pad_value=self.pad_value)
            self.num_train_puzzles = len(self.train_dataset)
            self.num_puzzles = len(self.train_dataset)

        if stage in ("test", None):
            self.test_dataset = MazeDataset(test_dir, self.grid_size, pad_value=self.pad_value)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        if self.val_dataset is None or len(self.val_dataset) == 0:
            return []
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        if self.test_dataset is None or len(self.test_dataset) == 0:
            return []
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
