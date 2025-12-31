import warnings

warnings.filterwarnings("ignore")

import shutil
from pathlib import Path
from typing import Optional

import lightning
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger

from src.nn.utils import (
    RankedLogger,
    extras,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)
import hydra
from omegaconf import DictConfig, OmegaConf



def update_model_config(cfg: DictConfig, datamodule: LightningDataModule):
    """
    Propagate datamodule metadata into the model/backbone config.
    Supports both direct model configs and nested backbone configs (PPO).
    """
    target_cfg = cfg.model.backbone if "backbone" in cfg.model else cfg.model

    for attr in ["num_puzzles", "batch_size", "pad_value", "max_grid_size", "vocab_size", "grid_size", "seq_len"]:
        if hasattr(datamodule, attr) and attr in target_cfg:
            setattr(target_cfg, attr, getattr(datamodule, attr))

    if "seq_len" in target_cfg and hasattr(datamodule, "seq_len"):
        target_cfg.seq_len = datamodule.seq_len
    elif "max_grid_size" in target_cfg and getattr(target_cfg, "max_grid_size", None):
        target_cfg.seq_len = target_cfg.max_grid_size * target_cfg.max_grid_size
    elif hasattr(datamodule, "grid_size"):
        target_cfg.max_grid_size = getattr(datamodule, "grid_size")
        target_cfg.seq_len = target_cfg.max_grid_size * target_cfg.max_grid_size

    log.info(
        f"Setting model config from data module (where available): "
        f"num_puzzles={getattr(datamodule, 'num_puzzles', 'n/a')} "
        f"batch_size={getattr(datamodule, 'batch_size', 'n/a')} "
        f"vocab_size={getattr(datamodule, 'vocab_size', 'n/a')}"
    )


@task_wrapper
def train(cfg: DictConfig) -> Optional[float]:
    # Set seed for random number generators in pytorch, numpy and python.random.
    if cfg.get("seed"):
        lightning.seed_everything(cfg.seed, workers=True)

    output_dir = Path(cfg["paths"]["output_dir"])

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    # Setup datamodule to get num_puzzles
    datamodule.setup(stage="fit")

    update_model_config(cfg, datamodule)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model_kwargs = {}
    target = getattr(cfg.model, "_target_", "")
    # Only pass output_dir to models that accept it (skip PPO trainer wrapper)
    if "ppo_trainer" not in str(target):
        model_kwargs["output_dir"] = output_dir
    model: LightningModule = hydra.utils.instantiate(cfg.model, **model_kwargs)

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    loggers: list[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=loggers, enable_progress_bar=False
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": loggers,
        "trainer": trainer,
    }

    if loggers:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting training!")

    datamodule.setup(stage="fit")
    log.info(f"Val check interval: {cfg.trainer.get('val_check_interval', 'default (1.0)')}")
    log.info(
        f"Check val every n epoch: {cfg.trainer.get('check_val_every_n_epoch', 'default (1.0)')}"
    )
    log.info(f"Steps per epoch: {len(datamodule.train_dataset) // cfg.data.batch_size}")
    log.info(f"Max epochs: {cfg.trainer.max_epochs}")
    log.info(f"Batch size: {cfg.data.batch_size}")

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    log.info("Training finished!")

    OmegaConf.save(cfg, output_dir / "config.yaml", resolve=True)

    if cfg.save_dir is not None:
        save_dir = cfg.save_dir
        log.info(f"Uploading training output to: {save_dir}")
        shutil.copytree(output_dir, save_dir)


@hydra.main(version_base="1.3", config_path="./configs", config_name="train.yaml")
def main(cfg: DictConfig):
    """
    Main entry point for training.

    Args:
        cfg: DictConfig configuration composed by Hydra.
    """
    # Apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # Train the model
    return train(cfg)


if __name__ == "__main__":
    main()
