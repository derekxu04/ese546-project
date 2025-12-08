import torch 
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict
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

        preds = []
        exited_step_indices = []
        exited_batch_indices = []

        for step in range(self.config.max_supervision_steps):
            is_last = step == self.config.max_supervision_steps - 1

            outputs, latents = self.deep_recursion(inputs, outputs, latents)

            halt_prob = self.to_halt_pred(outputs)
            should_halt = (halt_prob >= self.config.halt_prob_threshold) | is_last

            # check if any in the batch should halt
            if not should_halt.any():
                continue
                
            logits = self.to_pred(outputs[should_halt])
            preds.append(logits)
            exited_step_indices.extend([step] * should_halt.sum().item())
            exited_batch_indices.append(active_batch_indices[should_halt])

            if is_last:
                continue
                
            # for next round
            inputs = inputs[~should_halt]
            outputs = outputs[~should_halt]
            latents = latents[~should_halt]
            active_batch_indices = active_batch_indices[~should_halt]

            if outputs.numel() == 0:
                break

        preds = torch.cat(preds).to(x.device).argmax(dim = -1)
        exited_step_indices = torch.tensor(exited_step_indices, device=x.device)

        exited_batch_indices = torch.cat(exited_batch_indices).to(x.device)
        sort_indices = exited_batch_indices.argsort(dim = -1)

        return preds[sort_indices], exited_step_indices[sort_indices]
    
    # during training, there is only one iteration of deep recursion per batch (unlike inference) 
    def forward(self, x, outputs, latents):
        inputs = self.input_embed(x) + self.pos_embed
        outputs, latents = self.deep_recursion(inputs, outputs, latents)

        logits = self.to_pred(outputs)
        halt_prob = self.to_halt_pred(outputs)

        return logits, halt_prob, outputs, latents

if __name__ == "__main__":  # small sanity check
    config = TRMConfig()
    model = TinyRecursiveModel(config)
    batch_size = 2
    seq_len = config.seq_len
    dummy_input = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    outputs, latents = model.get_initial()
    logits, halt_prob, _, _ = model(dummy_input, outputs.expand(batch_size, -1, -1), latents.expand(batch_size, -1, -1))
    print("Logits shape:", logits.shape)  # Expected: (batch_size, seq_len, vocab_size)
    print("Halt probabilities shape:", halt_prob.shape)  # Expected: (batch_size,)