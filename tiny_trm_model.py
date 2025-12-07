"""Simplified Tiny Recursive Model (TRM).

The goal of this module is to expose a tiny-yet-flexible recursive reasoning
model that mirrors the ideas from "Less is More: Recursive Reasoning with Tiny
Networks" while remaining easy to read and tune.  Everything needed to build
the model lives in this single file so that experiments outside of the original
codebase are straightforward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn


# -----------------------------------------------------------------------------
# Configuration helpers
# -----------------------------------------------------------------------------


@dataclass
class TinyTRMConfig:
    """Configuration for :class:`TinyRecursiveModel`.

    You can pass a plain ``dict`` with matching keys to ``TinyRecursiveModel``
    and it will be converted into this dataclass automatically.
    """

    vocab_size: int = 10  # digits 0-9 by default
    seq_len: int = 81     # flattened 9x9 Sudoku grid
    hidden_size: int = 256
    num_heads: int = 4
    ff_multiplier: float = 4.0
    num_layers: int = 2
    H_cycles: int = 3
    L_cycles: int = 6
    dropout: float = 0.0
    num_puzzle_types: int = 0  # optional per-puzzle embedding

    @classmethod
    def from_dict(cls, cfg: Dict) -> "TinyTRMConfig":
        valid = {k: cfg[k] for k in cfg if k in cls.__dataclass_fields__}
        return cls(**valid)

    def to_dict(self) -> Dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# -----------------------------------------------------------------------------
# Building blocks
# -----------------------------------------------------------------------------


class TinyRecursiveBlock(nn.Module):
    """Transformer-style block using PyTorch primitives."""

    def __init__(self, hidden_size: int, num_heads: int, ff_multiplier: float, dropout: float) -> None:
        super().__init__()
        self.norm_1 = nn.LayerNorm(hidden_size)
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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.norm_1(hidden_states)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        hidden_states = hidden_states + self.dropout(attn_out)

        y = self.norm_2(hidden_states)
        hidden_states = hidden_states + self.dropout(self.ff(y))
        return hidden_states


class TinyReasoner(nn.Module):
    """Stack of :class:`TinyRecursiveBlock` layers with input injection."""

    def __init__(self, config: TinyTRMConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TinyRecursiveBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    ff_multiplier=config.ff_multiplier,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )

    def forward(self, hidden_states: torch.Tensor, injected_context: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + injected_context
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


# -----------------------------------------------------------------------------
# Tiny Recursive Model
# -----------------------------------------------------------------------------


class TinyRecursiveModel(nn.Module):
    """Self-contained Tiny Recursive Model suitable for small experiments."""

    def __init__(self, config: TinyTRMConfig | Dict) -> None:
        super().__init__()
        if isinstance(config, dict):
            config = TinyTRMConfig.from_dict(config)
        self.config = config

        self.token_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.seq_len, config.hidden_size))
        if config.num_puzzle_types > 0:
            self.puzzle_embed = nn.Embedding(config.num_puzzle_types, config.hidden_size)
        else:
            self.puzzle_embed = None

        self.reasoner = TinyReasoner(config)
        self.head = nn.Linear(config.hidden_size, config.vocab_size)

        # Learned initial latents
        init_scale = 0.02
        self.init_high = nn.Parameter(init_scale * torch.randn(1, config.seq_len, config.hidden_size))
        self.init_low = nn.Parameter(init_scale * torch.randn(1, config.seq_len, config.hidden_size))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        if self.puzzle_embed is not None:
            nn.init.normal_(self.puzzle_embed.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)

    def forward(
        self,
        inputs: torch.Tensor,
        puzzle_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            inputs: ``(batch, seq_len)`` tensor of token IDs.
            puzzle_ids: optional tensor with shape ``(batch,)`` identifying the
                puzzle family.  Only used when ``num_puzzle_types > 0``.
        """

        if inputs.shape[-1] != self.config.seq_len:
            raise ValueError(f"Expected seq_len={self.config.seq_len}, got {inputs.shape[-1]}")

        emb = self.token_embed(inputs) + self.pos_embed
        if self.puzzle_embed is not None and puzzle_ids is not None:
            emb = emb + self.puzzle_embed(puzzle_ids).unsqueeze(1)

        batch_size = inputs.shape[0]
        z_high = self.init_high.expand(batch_size, -1, -1)
        z_low = self.init_low.expand(batch_size, -1, -1)

        for _ in range(self.config.H_cycles):
            for _ in range(self.config.L_cycles):
                z_low = self.reasoner(z_low, emb)
            z_high = self.reasoner(z_high, z_low)

        logits = self.head(z_high)
        return logits

    @torch.no_grad()
    def predict(self, inputs: torch.Tensor, puzzle_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self(inputs, puzzle_ids=puzzle_ids)
        return torch.argmax(logits, dim=-1)


def build_tiny_trm(config: Optional[Dict] = None) -> TinyRecursiveModel:
    """Factory helper used by external scripts."""

    if config is None:
        config = TinyTRMConfig().to_dict()
    return TinyRecursiveModel(config)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":  # small sanity check
    dummy_config = TinyTRMConfig(hidden_size=64, num_layers=1, H_cycles=2, L_cycles=2)
    model = TinyRecursiveModel(dummy_config)
    inputs = torch.randint(0, dummy_config.vocab_size, (2, dummy_config.seq_len))
    logits = model(inputs)
    print("logits:", logits.shape)
    print("params:", count_parameters(model))
