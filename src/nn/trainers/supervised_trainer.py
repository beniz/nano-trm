from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
from lightning import LightningModule

from src.nn.models.trm import TRMModule
from src.nn.modules.utils import compute_lr
from src.nn.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

try:
    from adam_atan2 import AdamATan2
except ImportError:
    AdamATan2 = None


class SupervisedTRMModule(LightningModule):
    """
    Lightning wrapper that keeps TRM architecture intact while moving the training loop out of the model.
    """

    def __init__(self, **backbone_kwargs) -> None:
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()

        # Instantiate the original TRM as a backbone (architecture + losses).
        self.backbone = TRMModule(**backbone_kwargs)
        self.manual_step = 0
        self.total_steps = float("inf")

    def setup(self, stage: str) -> None:
        # Compute total steps for LR scheduling (mirrors original TRM.setup logic).
        if stage == "fit" and hasattr(self.trainer, "datamodule") and self.trainer.datamodule:
            dm = self.trainer.datamodule
            samples_per_epoch = dm.num_train_groups if hasattr(dm, "num_train_groups") else len(dm.train_dataset)
            steps_per_epoch = samples_per_epoch // dm.batch_size
            if self.trainer.max_epochs > 0:
                self.total_steps = steps_per_epoch * self.trainer.max_epochs
            else:
                self.total_steps = float("inf")

            log.info("Training configuration (supervised wrapper):")
            log.info(f"  Steps per epoch: {steps_per_epoch}")
            log.info(f"  Total steps: {self.total_steps}")

    def forward(self, batch: Dict[str, torch.Tensor]):
        return self.backbone.forward(batch)

    def _grad_monitoring(self) -> None:
        with torch.no_grad():
            total_grad_norm = (
                torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), max_norm=float("inf")).item()
            )

            grad_metrics = {}
            if hasattr(self.backbone, "lenet") and self.backbone.lenet.layers:
                first_layer = self.backbone.lenet.layers[0]
                last_layer = self.backbone.lenet.layers[-1]
                if hasattr(first_layer, "self_attn") and first_layer.self_attn.qkv_proj.weight.grad is not None:
                    grad_metrics["first_attn"] = first_layer.self_attn.qkv_proj.weight.grad.norm().item()
                if hasattr(last_layer, "mlp") and last_layer.mlp.down_proj.weight.grad is not None:
                    grad_metrics["last_mlp"] = last_layer.mlp.down_proj.weight.grad.norm().item()

            if getattr(self.backbone, "lm_head", None) is not None and self.backbone.lm_head.weight.grad is not None:
                grad_metrics["lm_head"] = self.backbone.lm_head.weight.grad.norm().item()
            if getattr(self.backbone, "q_head", None) is not None and self.backbone.q_head.weight.grad is not None:
                grad_metrics["q_head"] = self.backbone.q_head.weight.grad.norm().item()

            self.log("grad/total_norm", total_grad_norm, on_step=True, prog_bar=True)
            if "first_attn" in grad_metrics and "last_mlp" in grad_metrics:
                ratio = grad_metrics["first_attn"] / (grad_metrics["last_mlp"] + 1e-8)
                self.log("grad/flow_ratio", ratio, on_step=True, prog_bar=True)
            for name, value in grad_metrics.items():
                self.log(f"grad/{name}", value, on_step=True)

            if total_grad_norm < 1e-6 or total_grad_norm > 100:
                log.warning(f"Step {self.manual_step}: Gradient norm={total_grad_norm:.2e}")

    def _gate_monitoring(self) -> None:
        with torch.no_grad():
            if not hasattr(self.backbone, "lenet"):
                return
            for layer_idx, layer in enumerate(self.backbone.lenet.layers):
                if hasattr(layer, "self_attn") and getattr(layer.self_attn, "gate_proj", None) is not None:
                    gate_proj = layer.self_attn.gate_proj
                    weight_norm = gate_proj.weight.norm().item()
                    bias_mean = gate_proj.bias.mean().item() if gate_proj.bias is not None else 0
                    effective_gate = torch.sigmoid(torch.tensor(bias_mean)).item()

                    self.log(f"gate/layer{layer_idx}/weight_norm", weight_norm)
                    self.log(f"gate/layer{layer_idx}/bias_mean", bias_mean)
                    self.log(f"gate/layer{layer_idx}/effective_gate", effective_gate)

                    if gate_proj.weight.grad is not None:
                        self.log(f"gate/layer{layer_idx}/grad_norm", gate_proj.weight.grad.norm().item())

    def _log_train_metrics(self, metrics: dict, lr_this_step: float, batch_size: int) -> None:
        self.log("train/lr", lr_this_step, on_step=True)
        if metrics.get("count", 0) > 0:
            with torch.no_grad():
                count = metrics["count"]
                self.log("train/accuracy", metrics.get("accuracy", 0) / count, on_step=True)
                self.log(
                    "train/exact_accuracy",
                    metrics.get("exact_accuracy", 0) / count,
                    prog_bar=True,
                    on_step=True,
                )
                self.log("train/q_halt_accuracy", metrics.get("q_halt_accuracy", 0) / count, on_step=True)
                self.log("train/steps", metrics.get("steps", 0) / count, prog_bar=True, on_step=True)

                self.log("train/lm_loss", metrics.get("lm_loss", 0) / batch_size, on_step=True)
                self.log("train/q_halt_loss", metrics.get("q_halt_loss", 0) / batch_size, on_step=True)

                avg_halt_steps = metrics.get("steps", 0) / metrics["count"]
                early_halt_rate = avg_halt_steps < self.backbone.hparams.N_supervision
                self.log("train/early_halt_rate", early_halt_rate, on_step=True)

                if self.backbone.hparams.use_sigreg and "sigreg_loss" in metrics:
                    self.log("train/sigreg_loss", metrics["sigreg_loss"], on_step=True)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        opt_list = self.optimizers()
        if not isinstance(opt_list, list):
            opt_list = [opt_list]

        batch_size = batch["input"].shape[0]

        if getattr(self.backbone, "carry", None) is None:
            self.backbone.carry = self.backbone.initial_carry(batch)

        self.backbone.carry, loss, metrics, _ = self.backbone.compute_loss_and_metrics(self.backbone.carry, batch)

        scaled_loss = loss / batch_size
        self.manual_backward(scaled_loss)

        self._grad_monitoring()
        self._gate_monitoring()

        torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), max_norm=1.0)

        current_step = self.manual_step
        lr_this_step = None

        base_lrs: List[float] = [self.backbone.hparams.learning_rate]
        if len(opt_list) > 1:
            base_lrs.append(self.backbone.hparams.learning_rate_emb)

        for opt, base_lr in zip(opt_list, base_lrs):
            if current_step < self.backbone.hparams.warmup_steps:
                lr_this_step = compute_lr(
                    base_lr=base_lr,
                    lr_warmup_steps=self.backbone.hparams.warmup_steps,
                    lr_min_ratio=self.backbone.hparams.lr_min_ratio,
                    current_step=current_step,
                    total_steps=self.total_steps,
                )
            else:
                lr_this_step = base_lr

            if hasattr(opt, "_optimizer"):
                for param_group in opt._optimizer.param_groups:
                    param_group["lr"] = lr_this_step
                opt._optimizer.step()
                opt._optimizer.zero_grad()
            else:
                for param_group in opt.param_groups:
                    param_group["lr"] = lr_this_step
                opt.step()
                opt.zero_grad()

        self._log_train_metrics(metrics, lr_this_step, batch_size)

        assert not torch.isnan(metrics.get("lm_loss")), f"LM loss is NaN at step {self.manual_step}"
        self.manual_step += 1
        self.backbone.manual_step = self.manual_step

        return loss.detach()

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        batch_size = batch["input"].shape[0]

        with torch.no_grad():
            carry = self.backbone.initial_carry(batch)
            accumulated_metrics = {}
            total_loss = 0.0
            n_steps = 0

            while True:
                carry, loss, metrics, all_halted = self.backbone.compute_loss_and_metrics(carry, batch)
                for k, v in metrics.items():
                    accumulated_metrics[k] = accumulated_metrics.get(k, 0) + v.item()

                total_loss += loss.item()
                n_steps += 1
                if all_halted:
                    break

            count = accumulated_metrics.get("count", batch_size)
            if count > 0:
                avg_metrics = {
                    "val/loss": total_loss / (n_steps * batch_size),
                    "val/accuracy": accumulated_metrics.get("accuracy", 0) / count,
                    "val/exact_accuracy": accumulated_metrics.get("exact_accuracy", 0) / count,
                    "val/q_halt_accuracy": accumulated_metrics.get("q_halt_accuracy", 0) / count,
                    "val/steps": accumulated_metrics.get("steps", 0) / count,
                    "val/lm_loss": accumulated_metrics.get("lm_loss", 0) / (n_steps * batch_size),
                    "val/q_halt_loss": accumulated_metrics.get("q_halt_loss", 0) / (n_steps * batch_size),
                }
            else:
                avg_metrics = {
                    f"val/{k}": 0.0
                    for k in [
                        "loss",
                        "accuracy",
                        "exact_accuracy",
                        "q_halt_accuracy",
                        "steps",
                        "lm_loss",
                        "q_halt_loss",
                    ]
                }

            for name, value in avg_metrics.items():
                self.log(
                    name,
                    value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=(name in ["val/loss", "val/exact_accuracy"]),
                    sync_dist=True,
                )
            return avg_metrics

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        return self.validation_step(batch, batch_idx)

    def on_train_epoch_start(self):
        if hasattr(self.trainer, "datamodule") and self.trainer.datamodule is not None:
            dm = self.trainer.datamodule
            if hasattr(dm, "on_train_epoch_start"):
                dm.on_train_epoch_start(self.current_epoch)

    def configure_optimizers(self):
        base_lr = self.backbone.hparams.learning_rate
        embedding_lr = self.backbone.hparams.learning_rate_emb

        optimizers = []

        if AdamATan2 is not None:
            main_opt = AdamATan2(
                self.backbone.parameters(),
                lr=base_lr,
                weight_decay=self.backbone.hparams.weight_decay,
                betas=(0.9, 0.95),
            )
        else:
            main_opt = torch.optim.AdamW(
                self.backbone.parameters(),
                lr=base_lr,
                weight_decay=self.backbone.hparams.weight_decay,
                betas=(0.9, 0.95),
            )
        optimizers.append(main_opt)

        # If the backbone exposes a sparse embedding optimizer factory, use it (preserves original behavior)
        if hasattr(self.backbone, "create_sparse_optimizer"):
            sparse_opt = self.backbone.create_sparse_optimizer(
                lr=embedding_lr,
                weight_decay=self.backbone.hparams.weight_decay,
            )
            if sparse_opt is not None:
                optimizers.append(sparse_opt)

        return optimizers
