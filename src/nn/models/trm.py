"""
HRM PyTorch Lightning Module - Following Figure 2 pseudocode exactly
"""

import os
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.nn.utils.constants import IGNORE_LABEL_ID

from src.nn.modules.sparse_embeddings import (
    CastedSparseEmbedding,
    CastedSparseEmbeddingSignSGD_Distributed,
)
from src.nn.modules.trm_block import (
    CastedEmbedding,
    CastedLinear,
    ReasoningBlock,
    ReasoningBlockConfig,
    ReasoningModule,
    RotaryEmbedding,
    RotaryEmbedding2D,
)
#from src.nn.modules.sigreg import compute_sigreg_loss
from src.nn.modules.utils import stablemax_cross_entropy, trunc_normal_init_
from src.nn.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


@dataclass
class TRMInnerCarry:
    z_H: torch.Tensor  # High-level state (y = the solution representation)
    z_L: torch.Tensor  # Low-level state (z = the problem representation)


@dataclass
class TRMCarry:
    """Carry structure for maintaining state across steps."""

    inner_carry: TRMInnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]  # Stores current batch data


class TRMModule(nn.Module):
    """
    HRM implementation following Figure 2 pseudocode exactly.

    This class is now a pure nn.Module. Training, logging, and optimizer
    logic are handled by external wrappers (e.g., SupervisedTRMModule).
    """

    def __init__(
        self,
        hidden_size: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,  # min(2, hidden_size // 64)
        max_grid_size: int = 30,
        H_cycles: int = 3,
        L_cycles: int = 6,
        N_supervision: int = 16,
        N_supervision_val: int = 16,
        ffn_expansion: int = 2,
        learning_rate: float = 1e-4,
        learning_rate_emb: float = 1e-2,
        weight_decay: float = 0.01,
        warmup_steps: int = 2000,
        halt_exploration_prob: float = 0.1,
        puzzle_emb_dim: int = 512,  # Puzzle embedding dimension
        puzzle_emb_len: int = 16,  # How many tokens for puzzle embedding
        rope_theta: int = 10000,
        pos_emb_type: str = None,
        lr_min_ratio: float = 1.0,
        use_sigreg: bool = False,
        use_mlp: bool = False,
        attn_gate_type: str = None,  # None, "headwise", "elementwise"
        vocab_size: int = 0,  # Should be set from datamodule
        num_puzzles: int = 0,  # Should be set from datamodule
        batch_size: int = 0,  # Should be set from datamodule
        pad_value: int = -1,  # Should be set from datamodule
        seq_len: int = 0,  # Should be set from datamodule
        output_dir: str = None,
    ):
        super().__init__()
        self.hparams = SimpleNamespace(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            max_grid_size=max_grid_size,
            H_cycles=H_cycles,
            L_cycles=L_cycles,
            N_supervision=N_supervision,
            N_supervision_val=N_supervision_val,
            ffn_expansion=ffn_expansion,
            learning_rate=learning_rate,
            learning_rate_emb=learning_rate_emb,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            halt_exploration_prob=halt_exploration_prob,
            puzzle_emb_dim=puzzle_emb_dim,
            puzzle_emb_len=puzzle_emb_len,
            rope_theta=rope_theta,
            pos_emb_type=pos_emb_type,
            lr_min_ratio=lr_min_ratio,
            use_sigreg=use_sigreg,
            use_mlp=use_mlp,
            attn_gate_type=attn_gate_type,
            vocab_size=vocab_size,
            num_puzzles=num_puzzles,
            batch_size=batch_size,
            pad_value=pad_value,
            seq_len=seq_len,
            output_dir=output_dir,
        )

        self.forward_dtype = torch.float32

        # Token embeddings
        self.embed_scale = math.sqrt(hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        log.info(f"Creating TRM with vocab size={vocab_size}, seq_len={seq_len}, puzzle_emb_len={puzzle_emb_len} {pos_emb_type=} and attn_gate_type={attn_gate_type}")

        self.input_embedding = CastedEmbedding(
            vocab_size, hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype
        )

        if pos_emb_type == "2d":
            log.info("Using 2D Rotary Embeddings")
            self.pos_embedding = RotaryEmbedding2D(
            dim=hidden_size // num_heads,
            prefix_len=puzzle_emb_len,
            max_grid_size=int(math.sqrt(seq_len)),  # e.g., 9 for seq_len=81
            base=rope_theta,
        )
        elif pos_emb_type == "1d":
            log.info("Using 1D Rotary Embeddings")
            self.pos_embedding = RotaryEmbedding(
                dim=hidden_size // num_heads,
                max_position_embeddings=seq_len + puzzle_emb_len,
                base=rope_theta,
            )
        else:
            log.info("Not using Rotary Embeddings")

        # a single network (not two separate networks)
        reasoning_config = ReasoningBlockConfig(
            hidden_size=hidden_size,
            num_heads=num_heads,
            expansion=ffn_expansion,
            rms_norm_eps=1e-5,
            seq_len=seq_len,
            mlp_t=use_mlp,
            puzzle_emb_ndim=puzzle_emb_dim,
            puzzle_emb_len=puzzle_emb_len,
            attn_gate_type=attn_gate_type,
        )

        self.lenet = ReasoningModule(
            layers=[ReasoningBlock(reasoning_config) for _ in range(num_layers)]
        )

        self.lm_head = CastedLinear(hidden_size, vocab_size, bias=False)
        self.q_head = CastedLinear(hidden_size, 1, bias=True) # learn to stop, not to continue

        with torch.no_grad():
            self.q_head.weight.zero_()
            if self.q_head.bias is not None:
                self.q_head.bias.fill_(-5.0)  # Strong negative bias

        # State for carry (persisted across training steps)
        self.carry = None

        self.z_H_init = nn.Buffer(
            trunc_normal_init_(torch.empty(hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )
        self.z_L_init = nn.Buffer(
            trunc_normal_init_(torch.empty(hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )

        if self.hparams.use_sigreg:
            self.sigreg_beta = 1.0
            self.sigreg_slices = 256

        # Add puzzle embeddings
        if puzzle_emb_dim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                num_embeddings=num_puzzles,
                embedding_dim=puzzle_emb_dim,
                batch_size=batch_size,
                init_std=0.0,  # Reference uses 0 init
                cast_to=self.forward_dtype,
            )
            self.puzzle_emb_len = puzzle_emb_len
            log.info(f"Created puzzle_emb with num_puzzles={num_puzzles}, batch_size={batch_size}")
            log.info(f"puzzle_emb.local_weights.shape: {self.puzzle_emb.local_weights.shape}")
            log.info(f"puzzle_emb.weights.shape: {self.puzzle_emb.weights.shape}")
        else:
            log.info("puzzle_emb_dim <= 0, not creating puzzle embeddings")
            self.puzzle_emb = None
            self.puzzle_emb_len = 0

        self.carry = None
        self.manual_step = 0

    def maybe_compile_inner_forward(self):
        """Optionally compile inner_forward for speed when CUDA + torch.compile are available."""
        if "DISABLE_COMPILE" in os.environ:
            return
        if not hasattr(torch, "compile"):
            return
        device = next(self.parameters()).device
        if device.type != "cuda":
            return
        try:
            log.info("Compiling inner_forward with torch.compile...")
            self.inner_forward = torch.compile(
                self.inner_forward,
                mode="reduce-overhead",
                fullgraph=False,
            )
            log.info("Compilation successful")
        except Exception as e:
            log.warning(f"torch.compile failed, running uncompiled: {e}")

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Token embedding
        embedding = self.input_embedding(input.to(torch.int32))

        # Puzzle embeddings
        if self.hparams.puzzle_emb_dim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)

            pad_count = self.puzzle_emb_len * self.hparams.hidden_size - puzzle_embedding.shape[-1]

            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat(
                (
                    puzzle_embedding.view(-1, self.puzzle_emb_len, self.hparams.hidden_size),
                    embedding,
                ),
                dim=-2,
            )

        # Scale
        return self.embed_scale * embedding

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["input"].shape[0]
        device = batch["input"].device

        return TRMCarry(
            inner_carry=self.empty_carry(
                batch_size, device
            ),  # Empty is expected, it will be reseted in first pass as all sequences are halted.
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=device),  # Default to halted
            current_data={k: torch.empty_like(v, device=device) for k, v in batch.items()},
        )

    def empty_carry(self, batch_size: int, device: torch.device) -> TRMInnerCarry:
        return TRMInnerCarry(
            z_H=torch.empty(
                batch_size,
                self.hparams.seq_len + self.puzzle_emb_len,
                self.hparams.hidden_size,
                dtype=self.forward_dtype,
                device=device,
            ),
            z_L=torch.empty(
                batch_size,
                self.hparams.seq_len + self.puzzle_emb_len,
                self.hparams.hidden_size,
                dtype=self.forward_dtype,
                device=device,
            ),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: TRMInnerCarry) -> TRMInnerCarry:
        return TRMInnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.z_H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.z_L_init, carry.z_L),
        )
    
    def inner_forward(
        self, carry: TRMInnerCarry, batch: Dict[str, torch.Tensor], *, detach_carry: bool = True
    ) -> Tuple[TRMInnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.pos_embedding() if hasattr(self, "pos_embedding") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["input"], batch["puzzle_identifiers"])

        # Forward iterations
        z_H, z_L = carry.z_H, carry.z_L
        # H_cycles-1 without grad
        with torch.no_grad():
            for _ in range(self.hparams.H_cycles - 1):
                for _ in range(self.hparams.L_cycles):
                    z_L = self.lenet(z_L, z_H + input_embeddings, **seq_info)
                z_H = self.lenet(z_H, z_L, **seq_info)
        # 1 with grad
        for _ in range(self.hparams.L_cycles):
            z_L = self.lenet(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.lenet(z_H, z_L, **seq_info)

        if self.training and self.hparams.use_sigreg:
            if "compute_sigreg_loss" not in globals():
                raise RuntimeError("Sigreg loss requested but compute_sigreg_loss is not imported.")
            self._sigreg_loss = compute_sigreg_loss(
                z_H, z_L, global_step=self.manual_step, num_slices=self.sigreg_slices
            )
    
        # LM Outputs
        if detach_carry:
            z_H_carry = z_H.detach()
            z_L_carry = z_L.detach()
        else:
            z_H_carry = z_H
            z_L_carry = z_L

        new_carry = TRMInnerCarry(z_H=z_H_carry, z_L=z_L_carry)  # New carry (optionally detached)
        output = self.lm_head(z_H)[:, self.puzzle_emb_len :]
        q_logits = self.q_head(z_H[:, 0]).to(
            torch.float32
        )  # Q-head; uses the first puzzle_emb position

        return new_carry, output, q_logits[..., 0]

    def forward(
        self, carry: TRMCarry, batch: Dict[str, torch.Tensor]
    ) -> Tuple[TRMCarry, Dict[str, torch.Tensor]]:
        # Update data, carry (removing halted sequences)
        new_inner_carry = self.reset_carry(carry.halted, carry.inner_carry)

        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {
            k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v)
            for k, v in carry.current_data.items()
        }

        # Forward inner model
        new_inner_carry, logits, q_halt_logits = self.inner_forward(
            new_inner_carry, new_current_data
        )

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
        }

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            n_supervision_steps = (
                self.hparams.N_supervision if self.training else self.hparams.N_supervision_val
            )

            is_last_step = new_steps >= n_supervision_steps

            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and (self.hparams.N_supervision > 1):
                # Halt signal
                # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes

                halted = halted | (q_halt_logits > 0)

                # Exploration
                min_halt_steps = (
                    torch.rand_like(q_halt_logits) < self.hparams.halt_exploration_prob
                ) * torch.randint_like(new_steps, low=2, high=self.hparams.N_supervision + 1)
                halted = halted & (new_steps >= min_halt_steps)

        return TRMCarry(new_inner_carry, new_steps, halted, new_current_data), outputs

    def compute_loss_and_metrics(self, carry, batch):
        """Compute loss and metrics without circular reference."""
        # Get model outputs
        new_carry, outputs = self.forward(carry, batch)
        labels = new_carry.current_data["output"]

        with torch.no_grad():
            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)

            # Correctness
            mask = labels != IGNORE_LABEL_ID
            loss_counts = mask.sum(-1)

            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts

            # Metrics (halted)
            valid_metrics = new_carry.halted & (loss_counts > 0)

            metrics = {
                "count": valid_metrics.sum(),
                "accuracy": torch.where(
                    valid_metrics, (is_correct.float() / loss_divisor).sum(-1), 0
                ).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                "q_halt_accuracy": (
                    valid_metrics & ((outputs["q_halt_logits"].squeeze() >= 0) == seq_is_correct)
                ).sum(),
                "steps": torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # Compute losses: These are per-sequence losses that will be summed
        lm_loss = (
            stablemax_cross_entropy(
                outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask
            )
            / loss_divisor
        ).sum()

        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"],
            seq_is_correct.to(outputs["q_halt_logits"].dtype),
            reduction="sum",
        )
        metrics.update(
            {
                "lm_loss": lm_loss.detach(),
                "q_halt_loss": q_halt_loss.detach(),
            }
        )

        total_loss = lm_loss + 0.5 * q_halt_loss

        if self.training and self.hparams.use_sigreg:
            total_loss = total_loss + self.sigreg_beta * self._sigreg_loss
            metrics.update({"sigreg_loss": self._sigreg_loss.detach()})

        return new_carry, total_loss, metrics, new_carry.halted.all()

    def create_sparse_optimizer(
        self, lr: float, weight_decay: float, world_size: int = 1
    ):
        """
        Optional helper to build the sparse embedding optimizer, matching the previous Lightning configure_optimizers.
        """
        if self.puzzle_emb is None:
            return None

        # Force sparse embedding local weights to be leaf tensors
        self.puzzle_emb.local_weights = self.puzzle_emb.local_weights.detach().requires_grad_(True)

        return CastedSparseEmbeddingSignSGD_Distributed(
            self.puzzle_emb.buffers(),
            lr=lr,
            weight_decay=weight_decay,
            world_size=world_size,
        )

    def forward_features(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Returns features suitable for downstream heads (e.g., policy/value).
        Uses a fresh carry per call (no persistent state).
        """
        obs = batch["input"]
        if obs.dim() > 2:
            obs = obs.view(obs.shape[0], -1)
        puzzle_ids = batch.get("puzzle_identifiers")
        if puzzle_ids is None:
            puzzle_ids = torch.zeros(obs.shape[0], dtype=torch.int32, device=obs.device)
        if puzzle_ids.dim() == 0:
            puzzle_ids = puzzle_ids.view(1).expand(obs.shape[0])
        proc_batch = {
            "input": obs.to(torch.int64),
            "puzzle_identifiers": puzzle_ids.to(torch.int32),
        }

        carry = self.initial_carry(proc_batch)
        inner_carry, _, _ = self.inner_forward(carry.inner_carry, proc_batch, detach_carry=False)
        # Use z_H as shared representation; policy/value heads can slice as needed.
        feats = {
            "z_H": inner_carry.z_H,
            "z_L": inner_carry.z_L,
            "policy_input": inner_carry.z_H[:, 0],            # first token (puzzle_emb position)
            "value_input": inner_carry.z_H.mean(dim=1),        # simple mean pool over sequence
        }
        return feats
