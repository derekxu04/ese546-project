from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from einops.layers.torch import Reduce, Rearrange
import math

# -----------------------------------------------------------------------------
# Configuration helpers
# -----------------------------------------------------------------------------

@dataclass
class HRMConfig:
    vocab_size: int = 10    # digits 0-9 by default
    hidden_size: int = 256
    seq_len: int = 81       # flattened 9x9 Sudoku grid
    num_heads: int = 4
    ff_multiplier: float = 4.0
    dropout: float = 0.0
    num_layers: int = 2
    num_low_refinements: int = 2        # T in HRM - low-level updates per high-level
    num_refinement_blocks: int = 3      # N in HRM - number of high-level updates
    max_supervision_steps: int = 12     # max number of deep supervision steps during training
    halt_prob_threshold: float = 0.5    # threshold for halt probability to stop inference early
    spatial_mask_ratio: float = 0.3
    spatial_min_targets: int = 8
    spatial_mask_token: int = 0
    jepa_predictor_hidden: int = 256
    jepa_predictor_layers: int = 1
    stopgrad_target: bool = True

    @classmethod
    def from_dict(cls, cfg: Dict) -> "HRMConfig":
        valid = {k: cfg[k] for k in cfg if k in cls.__dataclass_fields__}
        return cls(**valid)

    def to_dict(self) -> Dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# -----------------------------------------------------------------------------
# Building blocks
# -----------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Transformer-style block using PyTorch primitives."""

    def __init__(self, hidden_size: int, num_heads: int, ff_multiplier: float, dropout: float):
        super().__init__()
        self.norm_1 = nn.LayerNorm(hidden_size)
        # all-to-all attention, no masking - this makes sense because for sudoku it is not autoregressive
        # instead, it ingests the whole puzzle (including the blanks) and predicts the entire solution in parallel using non-causal attention
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        ff_hidden = int(hidden_size * ff_multiplier)
        self.norm_2 = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, ff_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, hidden_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states):
        x = self.norm_1(hidden_states)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        hidden_states = hidden_states + self.dropout(attn_out)

        y = self.norm_2(hidden_states)
        hidden_states = hidden_states + self.dropout(self.ff(y))
        return hidden_states


class LowLevelNetwork(nn.Module):
    """Low-level network (L_net) that handles rapid, detailed computations."""

    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    ff_multiplier=config.ff_multiplier,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )

    def forward(self, zL, zH, x):
        # Combine all inputs - low-level state attends to itself, high-level guidance, and inputs
        hidden_states = zL + zH + x
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class HighLevelNetwork(nn.Module):
    """High-level network (H_net) that handles slow, abstract planning."""

    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    ff_multiplier=config.ff_multiplier,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )

    def forward(self, zH, zL):
        # High-level state attends to itself and low-level feedback
        hidden_states = zH + zL
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


# -----------------------------------------------------------------------------
# Hierarchical Reasoning Model with JEPA
#
# Based on HRM pseudocode:
#
# def hrm(z, x, N=2, T=2):
#     x = input_embedding(x)
#     zH, zL = z
#     with torch.no_grad():
#         for _i in range(N * T - 1):
#             zL = L_net(zL, zH, x)
#             if (_i + 1) % T == 0:
#                 zH = H_net(zH, zL)
#     # 1-step grad
#     zL = L_net(zL, zH, x)
#     zH = H_net(zH, zL)
#     return (zH, zL), output_head(zH)
#
# Combined with JEPA spatial masking loss for learning representations
# -----------------------------------------------------------------------------

class HierarchicalReasoningModel(nn.Module):
    """Hierarchical Reasoning Model with JEPA spatial masking.

    Combines HRM's two-level hierarchical processing with JEPA's self-supervised
    representation learning via masked token prediction.
    """

    def __init__(self, config):
        super().__init__()
        if isinstance(config, dict):
            config = HRMConfig.from_dict(config)
        self.config = config

        # Embedding scaling 
        self.embed_scale = math.sqrt(config.hidden_size)

        # tokens -> embeddings
        self.input_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.seq_len, config.hidden_size))

        # Hierarchical networks
        self.L_net = LowLevelNetwork(config)  # Fast, detailed processing
        self.H_net = HighLevelNetwork(config)  # Slow, abstract planning

        # Initial states
        # Reference uses buffers (no grad) but we keep as parameters for flexibility
        self.zH_init = nn.Parameter(torch.randn(config.hidden_size))
        self.zL_init = nn.Parameter(torch.randn(config.hidden_size))

        # Prediction heads
        # Output from high-level state (abstract planning determines final answer)
        self.to_pred = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Halt probability from first token only 
        self.to_halt_pred = nn.Sequential(
            nn.Linear(config.hidden_size, 1, bias=False),
            nn.Sigmoid(),
            Rearrange('... 1 -> ...')
        )

        # JEPA spatial predictor
        # Predicts masked latent representations from visible context
        predictor_layers = []
        in_dim = config.hidden_size
        for _ in range(max(1, config.jepa_predictor_layers)):
            predictor_layers.extend([
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, config.jepa_predictor_hidden),
                nn.GELU(),
                nn.Linear(config.jepa_predictor_hidden, config.hidden_size),
            ])
            in_dim = config.hidden_size
        self.spatial_predictor = nn.Sequential(*predictor_layers)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.input_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.to_pred.weight, mean=0.0, std=0.02)

    def get_initial(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get initial high-level and low-level states, expanded for batch.

        Args:
            batch_size: Number of samples in batch
            device: Device to place tensors on

        Returns:
            (zH, zL): Initial high-level and low-level states [batch, seq_len, hidden]
        """
        # Broadcast single vector to [batch, seq_len, hidden]
        zH = self.zH_init.view(1, 1, -1).expand(batch_size, self.config.seq_len, -1).to(device)
        zL = self.zL_init.view(1, 1, -1).expand(batch_size, self.config.seq_len, -1).to(device)
        return zH, zL

    def hrm_step(self, z: Tuple[torch.Tensor, torch.Tensor], x: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Single HRM reasoning step with hierarchical updates.

        Performs N * T low-level updates with N high-level updates.
        First (N*T - 1) updates are done without gradients for efficiency.
        Final update retains gradients for learning.
        """
        zH, zL = z
        N = self.config.num_refinement_blocks
        T = self.config.num_low_refinements

        # First (N*T - 1) updates without gradients
        with torch.no_grad():
            for i in range(N * T - 1):
                zL = self.L_net(zL, zH, x)
                # Update high-level every T low-level steps
                if (i + 1) % T == 0:
                    zH = self.H_net(zH, zL)

        # Final update with gradients (for learning)
        zL = self.L_net(zL, zH, x)
        zH = self.H_net(zH, zL)

        # Output from high-level state
        y_hat = self.to_pred(zH)

        return (zH, zL), y_hat

    @torch.no_grad()
    def predict(self, x):
        """Inference with adaptive computation and early halting."""
        batch_size = x.shape[0]

        # Prepare inputs with embedding scaling 
        inputs = self.embed_scale * (self.input_embed(x) + self.pos_embed)
        zH, zL = self.get_initial(batch_size, x.device)

        active_batch_indices = torch.arange(batch_size, device=x.device, dtype=torch.float32)

        preds = []
        exited_step_indices = []
        exited_batch_indices = []

        for step in range(self.config.max_supervision_steps):
            is_last = step == self.config.max_supervision_steps - 1

            # Run HRM step
            (zH, zL), logits = self.hrm_step((zH, zL), inputs)

            # Check halt condition 
            halt_prob = self.to_halt_pred(zH[:, 0])
            should_halt = (halt_prob >= self.config.halt_prob_threshold) | is_last

            # Collect predictions for halting samples
            if not should_halt.any():
                continue

            preds.append(logits[should_halt])
            exited_step_indices.extend([step] * should_halt.sum().item())
            exited_batch_indices.append(active_batch_indices[should_halt])

            if is_last:
                continue

            # Continue with non-halted samples
            inputs = inputs[~should_halt]
            zH = zH[~should_halt]
            zL = zL[~should_halt]
            active_batch_indices = active_batch_indices[~should_halt]

            if zH.numel() == 0:
                break

        # Gather and sort results
        preds = torch.cat(preds).to(x.device).argmax(dim=-1)
        exited_step_indices = torch.tensor(exited_step_indices, device=x.device)
        exited_batch_indices = torch.cat(exited_batch_indices).to(x.device)
        sort_indices = exited_batch_indices.argsort(dim=-1)

        return preds[sort_indices], exited_step_indices[sort_indices]

    def forward(self, x, zH, zL):
        """Forward pass for training with deep supervision."""
        # Apply embedding scaling 
        inputs = self.embed_scale * (self.input_embed(x) + self.pos_embed)
        (zH, zL), logits = self.hrm_step((zH, zL), inputs)

        # Halt probability from first token only 
        halt_prob = self.to_halt_pred(zH[:, 0])

        return logits, halt_prob, zH, zL

    def encode_latents(self, tokens: torch.Tensor) -> torch.Tensor:
        """Encode tokens into latent representations using high-level state.

        Used by JEPA to get target and context representations.
        """
        batch_size = tokens.shape[0]
        zH, zL = self.get_initial(batch_size, tokens.device)
        # Apply embedding scaling 
        inputs = self.embed_scale * (self.input_embed(tokens) + self.pos_embed)
        # Run full HRM step to get high-level representation
        (zH, _), _ = self.hrm_step((zH, zL), inputs)
        return zH

    def _sample_target_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample random mask for JEPA spatial masking."""
        mask = torch.rand(batch_size, self.config.seq_len, device=device) < self.config.spatial_mask_ratio
        min_targets = max(1, self.config.spatial_min_targets)
        needs_fill = mask.sum(dim=1, keepdim=True) < min_targets
        if needs_fill.any():
            filler = torch.topk(torch.rand(batch_size, self.config.seq_len, device=device), min_targets, dim=1).indices
            mask.scatter_(1, filler, True)
        return mask

    def spatial_jepa_loss(
        self,
        puzzle_inputs: torch.Tensor,
        solution_labels: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute JEPA spatial masking loss.

        Predicts masked cell latents from visible context.
        Helps learn better representations for hierarchical reasoning.
        """
        if target_mask is None:
            target_mask = self._sample_target_mask(puzzle_inputs.shape[0], puzzle_inputs.device)

        context_tokens = puzzle_inputs.clone()
        context_tokens[target_mask] = self.config.spatial_mask_token

        context_latents = self.encode_latents(context_tokens)
        target_latents = self.encode_latents(solution_labels)
        if self.config.stopgrad_target:
            target_latents = target_latents.detach()

        predicted = self.spatial_predictor(context_latents)
        diff = (predicted - target_latents).pow(2)
        mask = target_mask.unsqueeze(-1).to(diff.dtype)
        loss = (diff * mask).sum() / mask.sum().clamp_min(1.0)
        return loss, target_mask


if __name__ == "__main__":  # small sanity check
    config = HRMConfig()
    model = HierarchicalReasoningModel(config)
    batch_size = 2
    seq_len = config.seq_len
    dummy_input = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    zH, zL = model.get_initial()
    zH = zH.expand(batch_size, -1, -1)
    zL = zL.expand(batch_size, -1, -1)
    logits, halt_prob, zH_new, zL_new = model(dummy_input, zH, zL)
    print("Logits shape:", logits.shape)  # Expected: (batch_size, seq_len, vocab_size)
    print("Halt probabilities shape:", halt_prob.shape)  # Expected: (batch_size,)
    print("zH shape:", zH_new.shape)  # Expected: (batch_size, seq_len, hidden_size)
    print("zL shape:", zL_new.shape)  # Expected: (batch_size, seq_len, hidden_size)
