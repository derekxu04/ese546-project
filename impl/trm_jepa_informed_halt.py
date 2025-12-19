from __future__ import annotations

import torch 
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Optional
from einops.layers.torch import Reduce, Rearrange

# -----------------------------------------------------------------------------
# Configuration helpers
# -----------------------------------------------------------------------------

@dataclass
class TRMConfig:
    vocab_size: int = 10    # digits 0-9 by default
    hidden_size: int = 256
    seq_len: int = 81       # flattened 9x9 Sudoku grid
    num_heads: int = 4
    ff_multiplier: float = 4.0
    dropout: float = 0.0
    num_layers: int = 2
    num_latent_refinements: int = 6     # T in paper - 1 output refinement per N latent refinements
    num_refinement_blocks: int = 3      # n in paper
    max_supervision_steps: int = 12     # max number of deep supervision steps during training
    halt_prob_threshold: float = 0.5    # threshold for halt probability to stop inference early
    spatial_mask_ratio: float = 0.3
    spatial_min_targets: int = 8
    spatial_mask_token: int = 0
    jepa_predictor_hidden: int = 256
    jepa_predictor_layers: int = 1
    stopgrad_target: bool = True
    use_jepa_informed_halt: bool = True  # whether to use JEPA reconstruction quality to inform halting
    jepa_halt_weight: float = 0.5        # weight for JEPA signal in halting decision (alpha in q_hat = Q_head(y) + alpha * (1 - jepa_loss))
    jepa_halt_threshold: float = 0.1     # JEPA reconstruction loss threshold for confident halting

    @classmethod
    def from_dict(cls, cfg: Dict) -> "TRMConfig":
        valid = {k: cfg[k] for k in cfg if k in cls.__dataclass_fields__}
        return cls(**valid)
    
    def to_dict(self) -> Dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# -----------------------------------------------------------------------------
# Building blocks
# -----------------------------------------------------------------------------

class TinyRecursiveBlock(nn.Module):
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


class TinyReasoner(nn.Module):
    """Stack of :class:`TinyRecursiveBlock` layers with input injection."""

    def __init__(self, config):
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

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


# -----------------------------------------------------------------------------
# Tiny Recursive Model

# Based on following pseudocode from paper:
#
# def latent recursion(x, y, z, n=6):
#     for i in range(n): # latent reasoning
#         z = net(x, y, z)
#     y = net(y, z) # refine output answer
#     return y, z
#
# def deep recursion(x, y, z, n=6, T=3):
#     # recursing T−1 times to improve y and z (no gradients needed)
#     with torch.no_grad():
#         for j in range(T−1):
#             y, z = latent recursion(x, y, z, n)
#     # recursing once to improve y and z
#     y, z = latent recursion(x, y, z, n)
#     return (y.detach(), z.detach()), output head(y), Q head(y)
#
# # Deep Supervision
# for x input, y true in train dataloader:
#     y, z = y init, z init
#     for step in range(N supervision):
#         x = input embedding(x input)
#         (y, z), y hat, q hat = deep recursion(x, y, z)
#         loss = softmax cross entropy(y hat, y true)
#         loss += binary cross entropy(q hat, (y hat == y true))
#         loss.backward()
#         opt.step()
#         opt.zero_grad()
#         if q hat > 0: # early−stopping
#             break
# -----------------------------------------------------------------------------

class TinyRecursiveModel(nn.Module):
    """Self-contained Tiny Recursive Model suitable for small experiments."""

    def __init__(self, config):
        super().__init__()
        if isinstance(config, dict):
            config = TRMConfig.from_dict(config)
        self.config = config

        # tokens -> embeddings
        self.input_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.seq_len, config.hidden_size))

        self.network = TinyReasoner(config)

        # initial latents (learned)
        # note: these have shape (1, seq_len, hidden_size), could just do hidden_size, but different positions on sudoku might have different priors
        # it is not that many extra parameters, but could do ablation study on this, my hunch is that reduce down to hidden_size minimally impacts performance
        init_scale = 0.02
        self.output_init_embed = nn.Parameter(init_scale * torch.randn(1, config.seq_len, config.hidden_size))
        self.latent_init_embed = nn.Parameter(init_scale * torch.randn(1, config.seq_len, config.hidden_size))

        # prediction heads
        # reverse embedding
        self.to_pred = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # predicts q_hat (halt probability)
        self.to_halt_pred = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.Linear(config.hidden_size, 1, bias = False),
            nn.Sigmoid(),
            Rearrange('... 1 -> ...')
        )

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
    
    def get_initial(self):
        outputs = self.output_init_embed
        latents = self.latent_init_embed

        return outputs, latents

    # inputs, outputs, latents: (batch_size, seq_len, hidden_size)
    def latent_recursion(self, inputs, outputs, latents):
        # in the paper, they only use one network to do both the latent update and the output update
        # the network learns to refine latents if input is passed in, else it refines the output

        for _ in range(self.config.num_latent_refinements):
            latents = self.network(inputs + outputs + latents)
        outputs = self.network(outputs + latents)

        return outputs, latents
        
    # inputs, ouputs, latents: (batch_size, seq_len, hidden_size)
    def deep_recursion(self, inputs, outputs, latents):
        # recurse T-1 times to improve y and z (no gradients needed)
        with torch.no_grad():
            for _ in range(self.config.num_refinement_blocks - 1):
                outputs, latents = self.latent_recursion(inputs, outputs, latents)
        # recurse once to improve y and z
        outputs, latents = self.latent_recursion(inputs, outputs, latents)

        return outputs, latents

    @torch.no_grad()
    def predict(self, x):
        batch_size = x.shape[0]

        inputs = self.input_embed(x) + self.pos_embed
        outputs, latents = self.get_initial()

        active_batch_indices = torch.arange(batch_size, device=x.device, dtype=torch.float32)
        active_inputs = x.clone()  # Track active inputs for JEPA confidence

        preds = []
        exited_step_indices = []
        exited_batch_indices = []

        for step in range(self.config.max_supervision_steps):
            is_last = step == self.config.max_supervision_steps - 1

            outputs, latents = self.deep_recursion(inputs, outputs, latents)

            # Get current predictions for JEPA confidence
            logits = self.to_pred(outputs)
            current_preds = torch.argmax(logits, dim=-1)

            # Compute halt probability
            halt_prob = self.to_halt_pred(outputs)

            # Use JEPA-informed halting if enabled
            if self.config.use_jepa_informed_halt:
                jepa_confidence = self.compute_jepa_confidence(outputs, active_inputs, current_preds)
                # Combine halt_prob with JEPA confidence
                informed_halt_prob = halt_prob + self.config.jepa_halt_weight * jepa_confidence
                informed_halt_prob = torch.clamp(informed_halt_prob, 0.0, 1.0)

                # Halting criteria: (informed_halt >= threshold) AND (jepa_confidence is high)
                # This ensures we only halt when both q_hat and JEPA agree the solution is good
                should_halt = (informed_halt_prob >= self.config.halt_prob_threshold) | is_last
            else:
                should_halt = (halt_prob >= self.config.halt_prob_threshold) | is_last

            # check if any in the batch should halt
            if not should_halt.any():
                continue

            preds.append(logits[should_halt])
            exited_step_indices.extend([step] * should_halt.sum().item())
            exited_batch_indices.append(active_batch_indices[should_halt])

            if is_last:
                continue

            # for next round
            inputs = inputs[~should_halt]
            outputs = outputs[~should_halt]
            latents = latents[~should_halt]
            active_batch_indices = active_batch_indices[~should_halt]
            active_inputs = active_inputs[~should_halt]

            if outputs.numel() == 0:
                break

        preds = torch.cat(preds).to(x.device).argmax(dim = -1)
        exited_step_indices = torch.tensor(exited_step_indices, device=x.device)

        exited_batch_indices = torch.cat(exited_batch_indices).to(x.device)
        sort_indices = exited_batch_indices.argsort(dim = -1)

        return preds[sort_indices], exited_step_indices[sort_indices]
    
    # during training, there is only one iteration of deep recursion per batch (unlike inference)
    def forward(self, x, outputs, latents, solution_labels: Optional[torch.Tensor] = None):
        inputs = self.input_embed(x) + self.pos_embed
        outputs, latents = self.deep_recursion(inputs, outputs, latents)

        logits = self.to_pred(outputs)
        halt_prob = self.to_halt_pred(outputs)

        # Compute JEPA-informed halt probability if enabled and labels provided
        jepa_confidence = None
        informed_halt_prob = halt_prob
        if self.config.use_jepa_informed_halt and solution_labels is not None:
            jepa_confidence = self.compute_jepa_confidence(outputs, x, solution_labels)
            # Combine halt_prob with JEPA confidence:
            # informed_halt = halt_prob + alpha * jepa_confidence
            # Higher JEPA confidence -> higher halt probability (more likely to stop)
            informed_halt_prob = halt_prob + self.config.jepa_halt_weight * jepa_confidence
            # Clamp to [0, 1] since we're combining probabilities
            informed_halt_prob = torch.clamp(informed_halt_prob, 0.0, 1.0)

        return logits, informed_halt_prob, outputs, latents, jepa_confidence

    def encode_latents(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size = tokens.shape[0]
        outputs, latents = self.get_initial()
        outputs = outputs.expand(batch_size, -1, -1).to(tokens.device)
        latents = latents.expand(batch_size, -1, -1).to(tokens.device)
        inputs = self.input_embed(tokens) + self.pos_embed
        outputs, _ = self.deep_recursion(inputs, outputs, latents)
        return outputs

    def _sample_target_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
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

    def compute_jepa_confidence(
        self,
        outputs: torch.Tensor,
        puzzle_inputs: torch.Tensor,
        solution_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute JEPA-based confidence score from reconstruction quality.

        Lower reconstruction loss = higher confidence that the representation is complete.
        Returns a per-sample confidence score in [0, 1].

        Args:
            outputs: Current output embeddings [B, seq_len, hidden_size]
            puzzle_inputs: Input puzzle tokens [B, seq_len]
            solution_labels: Solution tokens [B, seq_len]

        Returns:
            confidence: Per-sample confidence scores [B], where higher = more confident
        """
        # Use fixed mask for consistency during inference
        batch_size = puzzle_inputs.shape[0]
        target_mask = self._sample_target_mask(batch_size, puzzle_inputs.device)

        # Compute JEPA reconstruction loss
        context_tokens = puzzle_inputs.clone()
        context_tokens[target_mask] = self.config.spatial_mask_token

        context_latents = self.encode_latents(context_tokens)
        target_latents = self.encode_latents(solution_labels)
        if self.config.stopgrad_target:
            target_latents = target_latents.detach()

        predicted = self.spatial_predictor(context_latents)
        diff = (predicted - target_latents).pow(2)

        # Compute per-sample reconstruction loss
        mask = target_mask.unsqueeze(-1).to(diff.dtype)
        per_sample_loss = (diff * mask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp_min(1.0)

        # Convert loss to confidence: confidence = 1 / (1 + loss)
        # This gives ~1.0 when loss is near 0, and approaches 0 as loss increases
        confidence = 1.0 / (1.0 + per_sample_loss)

        return confidence

if __name__ == "__main__":  # small sanity check
    config = TRMConfig()
    model = TinyRecursiveModel(config)
    batch_size = 2
    seq_len = config.seq_len
    dummy_input = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    dummy_labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    outputs, latents = model.get_initial()
    logits, halt_prob, _, _, jepa_conf = model(
        dummy_input,
        outputs.expand(batch_size, -1, -1),
        latents.expand(batch_size, -1, -1),
        solution_labels=dummy_labels
    )
    print("Logits shape:", logits.shape)  # Expected: (batch_size, seq_len, vocab_size)
    print("Halt probabilities shape:", halt_prob.shape)  # Expected: (batch_size,)
    if jepa_conf is not None:
        print("JEPA confidence shape:", jepa_conf.shape)  # Expected: (batch_size,)
        print("JEPA confidence values:", jepa_conf)