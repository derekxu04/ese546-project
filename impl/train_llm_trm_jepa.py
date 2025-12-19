import copy
import json
import os
import argparse
from pathlib import Path
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_messages(model_name, messages):
    if "google/gemma" in model_name:
        full_messages = copy.deepcopy(messages)[1:3]
        full_messages[0]["content"] = messages[0]["content"] + "\n\n" + full_messages[0]["content"]
        return full_messages
    return messages


def load_and_prepare_dataset(data_file, tokenizer, model_name, max_length=512):
    dataset = load_dataset('json', data_files=data_file)['train']
    print(f"Loaded {len(dataset)} examples")

    def tokenize_conversations(examples):
        input_ids_list, labels_list, attention_mask_list = [], [], []
        input_only_ids_list, input_only_mask_list = [], []  # For JEPA

        for messages in examples['messages']:
            # Full conversation (for main task)
            full_messages = get_messages(model_name, messages)
            formatted_chat = tokenizer.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=False
            )
            tokenized = tokenizer(
                formatted_chat, truncation=True, max_length=max_length,
                padding="max_length", return_tensors=None
            )
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            labels = create_masked_labels(messages, tokenizer, input_ids, attention_mask)

            input_ids_list.append(input_ids)
            labels_list.append(labels)
            attention_mask_list.append(attention_mask)

            # Input only (question) for JEPA context
            user_message = [msg for msg in messages if msg['role'] == 'user'][0]
            input_text = tokenizer.apply_chat_template(
                [user_message], tokenize=False, add_generation_prompt=False
            )
            input_tok = tokenizer(
                input_text, truncation=True, max_length=max_length,
                padding="max_length", return_tensors=None
            )
            input_only_ids_list.append(input_tok["input_ids"])
            input_only_mask_list.append(input_tok["attention_mask"])

        return {
            "input_ids": input_ids_list,
            "labels": labels_list,
            "attention_mask": attention_mask_list,
            "input_only_ids": input_only_ids_list,
            "input_only_mask": input_only_mask_list,
        }

    def create_masked_labels(messages, tokenizer, input_ids, attention_mask):
        labels = [-100] * len(input_ids)
        for i, mask in enumerate(attention_mask):
            if mask == 0:
                labels[i] = -100

        for msg in messages:
            if msg['role'] == 'assistant':
                assistant_content = msg['content']
                assistant_tokens = tokenizer.encode(assistant_content, add_special_tokens=False)
                decoded_assistant = [tokenizer.decode(item) for item in assistant_tokens]
                decoded_input = [tokenizer.decode(item) for item in input_ids]

                for i in range(len(input_ids) - len(assistant_tokens) + 1):
                    if attention_mask[i] == 1 and decoded_input[i:i+len(assistant_tokens)] == decoded_assistant:
                        for j in range(i, min(i + len(assistant_tokens), len(input_ids))):
                            if attention_mask[j] == 1:
                                labels[j] = input_ids[j]
                        break
        return labels

    tokenized_dataset = dataset.map(
        tokenize_conversations, batched=True, remove_columns=dataset.column_names
    )
    return tokenized_dataset


class TinyRecursiveBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class JEPAPredictor(nn.Module):
    def __init__(self, hidden_size: int, predictor_hidden: int = 256, num_layers: int = 1):
        super().__init__()
        layers = []
        in_dim = hidden_size
        for _ in range(max(1, num_layers)):
            layers.extend([
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, predictor_hidden),
                nn.GELU(),
                nn.Linear(predictor_hidden, hidden_size),
            ])
            in_dim = hidden_size
        self.predictor = nn.Sequential(*layers)

    def forward(self, x):
        return self.predictor(x)


class LLMTRMJEPAModel(nn.Module):
    """LLM + TRM-JEPA"""

    def __init__(
        self,
        llm_model,
        vocab_size: int,
        hidden_size: int,
        num_heads: int = 8,
        num_layers: int = 2,
        num_latent_refinements: int = 2,
        num_refinement_blocks: int = 1,
        jepa_predictor_hidden: int = 256,
        jepa_predictor_layers: int = 1,
        spatial_mask_ratio: float = 0.3,
        spatial_min_targets: int = 8,
        stopgrad_target: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.llm = llm_model
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_latent_refinements = num_latent_refinements
        self.num_refinement_blocks = num_refinement_blocks
        self.spatial_mask_ratio = spatial_mask_ratio
        self.spatial_min_targets = spatial_min_targets
        self.stopgrad_target = stopgrad_target

        # Freeze LLM
        for param in self.llm.parameters():
            param.requires_grad = False

        self.input_proj = nn.Linear(self.llm.config.hidden_size, hidden_size)

        # Learned initial states
        self.output_init = nn.Parameter(0.02 * torch.randn(1, 1, hidden_size))
        self.latent_init = nn.Parameter(0.02 * torch.randn(1, 1, hidden_size))

        # Refinement network
        self.refinement_layers = nn.ModuleList([
            TinyRecursiveBlock(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Output heads
        self.to_logits = nn.Linear(hidden_size, vocab_size, bias=False)
        self.to_halt = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())

        # JEPA predictor
        self.jepa_predictor = JEPAPredictor(hidden_size, jepa_predictor_hidden, jepa_predictor_layers)

    def refinement_network(self, x):
        for layer in self.refinement_layers:
            x = layer(x)
        return x

    def latent_recursion(self, inputs, outputs, latents):
        for _ in range(self.num_latent_refinements):
            latents = self.refinement_network(inputs + outputs + latents)
        outputs = self.refinement_network(outputs + latents)
        return outputs, latents

    def deep_recursion(self, inputs, outputs, latents):
        with torch.no_grad():
            for _ in range(self.num_refinement_blocks - 1):
                outputs, latents = self.latent_recursion(inputs, outputs, latents)
        outputs, latents = self.latent_recursion(inputs, outputs, latents)
        return outputs, latents

    def encode_with_llm(self, input_ids, attention_mask):
        """Encode using frozen LLM."""
        with torch.no_grad():
            llm_outputs = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            llm_hidden = llm_outputs.hidden_states[-1]
        return self.input_proj(llm_hidden)

    def encode_latents(self, input_ids, attention_mask):
        """Encode tokens to latents through full model"""
        batch_size, seq_len = input_ids.shape
        inputs = self.encode_with_llm(input_ids, attention_mask)
        outputs = self.output_init.expand(batch_size, seq_len, -1).to(inputs.device)
        latents = self.latent_init.expand(batch_size, seq_len, -1).to(inputs.device)
        outputs, _ = self.deep_recursion(inputs, outputs, latents)
        return outputs

    def forward(self, inputs, outputs, latents):
        """Single forward"""
        outputs, latents = self.deep_recursion(inputs, outputs, latents)
        logits = self.to_logits(outputs)
        halt_prob = self.to_halt(outputs.mean(dim=1))
        return logits, halt_prob, outputs, latents

    def _sample_target_mask(self, batch_size: int, seq_len: int, device: torch.device):
        mask = torch.rand(batch_size, seq_len, device=device) < self.spatial_mask_ratio
        min_targets = max(1, self.spatial_min_targets)
        needs_fill = mask.sum(dim=1, keepdim=True) < min_targets
        if needs_fill.any():
            filler = torch.topk(torch.rand(batch_size, seq_len, device=device), min_targets, dim=1).indices
            mask.scatter_(1, filler, True)
        return mask

    def spatial_jepa_loss(
        self,
        input_only_ids,
        input_only_mask,
        full_ids,
        full_mask,
    ):
        """JEPA loss - adapted from trm_jepa.py:296-318 for LLM.

        For Sudoku: context=masked_puzzle, target=solution
        For LLM: context=full_conv_with_some_tokens_masked, target=full_conv_unmasked

        This is a denoising/self-prediction objective on the conversation.
        """
        batch_size = full_ids.shape[0]
        seq_len = full_ids.shape[1]
        device = full_ids.device

        # Sample mask on FULL sequence
        target_mask = self._sample_target_mask(batch_size, seq_len, device)
        valid_mask = (full_mask > 0).bool()
        target_mask = target_mask & valid_mask

        # Context: full conversation with some tokens masked
        # (analogous to puzzle_inputs with positions masked in Sudoku)
        context_tokens = full_ids.clone()
        context_tokens[target_mask] = self.llm.config.pad_token_id  # Mask with pad token

        # Encode context (masked conversation)
        context_latents = self.encode_latents(context_tokens, full_mask)

        # Target: full conversation (unmasked)
        target_latents = self.encode_latents(full_ids, full_mask)
        if self.stopgrad_target:
            target_latents = target_latents.detach()

        # Predict
        predicted = self.jepa_predictor(context_latents)

        # MSE on masked positions
        diff = (predicted - target_latents).pow(2)
        mask_expand = target_mask.unsqueeze(-1).to(diff.dtype)
        loss = (diff * mask_expand).sum() / (mask_expand.sum() * self.hidden_size).clamp_min(1.0)

        return loss


def train_epoch(model, dataloader, optimizer, device, config, epoch):
    """Train with deep supervision"""
    model.train()

    stats = {"loss": 0.0, "ce": 0.0, "halt": 0.0, "jepa": 0.0, "opt_steps": 0}

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        input_only_ids = batch['input_only_ids'].to(device)
        input_only_mask = batch['input_only_mask'].to(device)

        # JEPA loss computed ONCE per batch
        jepa_loss = model.spatial_jepa_loss(
            input_only_ids,
            input_only_mask,
            input_ids,
            attention_mask,
        )

        # Encode inputs
        inputs = model.encode_with_llm(input_ids, attention_mask)
        batch_size, seq_len = input_ids.shape

        # Initialize states
        outputs = model.output_init.expand(batch_size, seq_len, -1)
        latents = model.latent_init.expand(batch_size, seq_len, -1)

        # Active samples
        x = inputs
        y = labels
        active_outputs = outputs
        active_latents = latents

        # Deep supervision
        for sup_step in range(config['max_supervision_steps']):
            is_last = sup_step == config['max_supervision_steps'] - 1

            # Forward
            logits, halt_prob, next_outputs, next_latents = model(
                x, active_outputs, active_latents
            )

            # CE loss
            ce_loss = F.cross_entropy(
                logits.reshape(-1, model.vocab_size),
                y.reshape(-1),
                ignore_index=-100
            )

            # Halt loss
            preds = logits.argmax(dim=-1)
            correct_mask = (preds == y) | (y == -100)
            all_correct = correct_mask.all(dim=1).float()
            halt_loss = F.binary_cross_entropy(halt_prob.squeeze(), all_correct)

            # Total loss 
            total_loss = (
                ce_loss +
                config['halt_loss_weight'] * halt_loss +
                config['jepa_loss_weight'] * jepa_loss.detach()  
            )

            # Optimize
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            optimizer.step()

            stats["loss"] += total_loss.item()
            stats["ce"] += ce_loss.item()
            stats["halt"] += halt_loss.item()
            stats["jepa"] += jepa_loss.item()
            stats["opt_steps"] += 1

            # Early stopping 
            should_halt = (halt_prob.squeeze() >= config['halt_prob_threshold']) | is_last
            if should_halt.all():
                break

            # Detach and filter 
            next_outputs = next_outputs.detach()
            next_latents = next_latents.detach()

            keep = ~should_halt
            x = x[keep]
            y = y[keep]
            active_outputs = next_outputs[keep]
            active_latents = next_latents[keep]

            if x.numel() == 0:
                break

        if batch_idx % config['log_interval'] == 0:
            print(f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | "
                  f"Loss: {total_loss.item():.4f} | CE: {ce_loss.item():.4f} | "
                  f"Halt: {halt_loss.item():.4f} | JEPA: {jepa_loss.item():.4f}")

    return {k: v / max(stats["opt_steps"], 1) for k, v in stats.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-1B-Instruct')
    parser.add_argument('--data_file', type=str, default='llm-jepa/datasets/gsm8k_test.jsonl')
    parser.add_argument('--output_dir', type=str, default='runs/llm_trm_jepa_final')
    parser.add_argument('--max_length', type=int, default=512)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    model_cfg = config['model']
    train_cfg = config['training']
    full_cfg = {**model_cfg, **train_cfg}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load LLM
    print(f"Loading: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        low_cpu_mem_usage=True,
        use_cache=False,
    )

    # Create model
    print("Creating LLM+TRM-JEPA")
    model = LLMTRMJEPAModel(
        llm_model=llm_model,
        vocab_size=len(tokenizer),
        hidden_size=model_cfg['hidden_size'],
        num_heads=model_cfg['num_heads'],
        num_layers=model_cfg['num_layers'],
        num_latent_refinements=model_cfg['num_latent_refinements'],
        num_refinement_blocks=model_cfg['num_refinement_blocks'],
        jepa_predictor_hidden=model_cfg['jepa_predictor_hidden'],
        jepa_predictor_layers=model_cfg['jepa_predictor_layers'],
        spatial_mask_ratio=model_cfg['spatial_mask_ratio'],
        spatial_min_targets=model_cfg.get('spatial_min_targets', 8),
        stopgrad_target=model_cfg['stopgrad_target'],
        dropout=model_cfg.get('dropout', 0.0),
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Params: {trainable:,} / {total:,}")

    # Dataset
    dataset = load_and_prepare_dataset(args.data_file, tokenizer, args.model_name, args.max_length)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=train_cfg['batch_size'], shuffle=True, num_workers=0
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg['lr'],
        weight_decay=train_cfg['weight_decay'],
    )

    # Train
    print("Training...")
    for epoch in range(train_cfg['epochs']):
        stats = train_epoch(model, dataloader, optimizer, device, full_cfg, epoch)
        print(f"\nEpoch {epoch}: Loss={stats['loss']:.4f} CE={stats['ce']:.4f} "
              f"Halt={stats['halt']:.4f} JEPA={stats['jepa']:.4f}\n")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))
    tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
