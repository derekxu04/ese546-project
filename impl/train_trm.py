"""Training script for the Tiny Recursive Model defined in impl/trm.py.

This mirrors the reference/train_trm.py workflow but binds to the newer
TinyRecursiveModel implementation that exposes explicit latent recursion and
halting heads.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Use the impl-local Sudoku helpers to avoid modifying sys.path.
from data_sudoku import SudokuDataConfig, SudokuDataset, prepare_sudoku_dataset
from trm import TinyRecursiveModel, TRMConfig


RUN_CONFIG: Dict = {
    "model": {
        "hidden_size": 256,
        "num_heads": 4,
        "num_layers": 2,
        "num_latent_refinements": 6,
        "num_refinement_blocks": 3,
        "max_supervision_steps": 4,
        "halt_prob_threshold": 0.5,
        "vocab_size": 10,
        "seq_len": 81,
        "dropout": 0.0,
    },
    "data": {
        "dataset_dir": "data/sudoku-toy",
        "train_subset": 2000,
        "force_download": False,
        "use_builtin_sample": True,
    },
    "training": {
        "epochs": 1000,
        "batch_size": 128,
        "lr": 3e-4,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "eval_interval": 1,
        "run_eval": True,
        "log_interval": 100,
        "halt_loss_weight": 1.0,
        "seed": 0,
        "num_workers": 0,
        "output_dir": "runs/tiny_trm_impl_sudoku",
        "device": None,
    },
}


@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    grad_clip: float
    eval_interval: int
    run_eval: bool
    log_interval: int
    halt_loss_weight: float
    seed: int
    num_workers: int
    output_dir: str
    device: Optional[str]


def deep_update(base: Dict, override: Dict) -> Dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_run_config(path: Optional[str]) -> Dict:
    cfg = copy.deepcopy(RUN_CONFIG)
    if path is None:
        return cfg
    with open(path) as handle:
        user_cfg = json.load(handle)
    return deep_update(cfg, user_cfg)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_dataloaders(cfg: SudokuDataConfig, train_cfg: TrainingConfig) -> Dict[str, DataLoader]:
    prepare_sudoku_dataset(cfg)
    train_ds = SudokuDataset(cfg.dataset_dir, split="train")
    test_split = "test" if (Path(cfg.dataset_dir) / "test.npz").exists() else "train"
    eval_ds = SudokuDataset(cfg.dataset_dir, split=test_split)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
    )
    return {"train": train_loader, "eval": eval_loader}


def prepare_initial_states(model: TinyRecursiveModel, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    outputs, latents = model.get_initial()
    outputs = outputs.expand(batch_size, -1, -1)
    latents = latents.expand(batch_size, -1, -1)
    return outputs, latents


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    preds = torch.argmax(logits, dim=-1)
    token_acc = (preds == labels).float().mean().item()
    puzzle_acc = (preds == labels).all(dim=-1).float().mean().item()
    return {"token_acc": token_acc, "puzzle_acc": puzzle_acc}


def compute_loss(
    logits: torch.Tensor,
    halt_prob: torch.Tensor,
    labels: torch.Tensor,
    halt_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ce = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
    preds = torch.argmax(logits.detach(), dim=-1)
    puzzle_correct = (preds == labels).all(dim=-1).float()
    halt_loss = F.binary_cross_entropy(halt_prob, puzzle_correct)
    total = ce + halt_loss_weight * halt_loss
    return total, ce.detach(), halt_loss.detach(), puzzle_correct.mean().detach()


def train_epoch(
    model: TinyRecursiveModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    halt_loss_weight: float,
    log_interval: int,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_token_acc = 0.0
    total_batches = 0
    total_halt_loss = 0.0
    total_ce_loss = 0.0
    avg_halt_acc = 0.0

    for step, batch in enumerate(loader, start=1):
        inputs = batch["inputs"].to(device)
        labels = batch["labels"].to(device)
        init_outputs, init_latents = prepare_initial_states(model, inputs.size(0))
        init_outputs = init_outputs.to(device)
        init_latents = init_latents.to(device)

        logits, halt_prob, _, _ = model(inputs, init_outputs, init_latents)
        loss, ce_loss, halt_loss, halt_acc = compute_loss(
            logits, halt_prob, labels, halt_loss_weight
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        metrics = compute_metrics(logits.detach(), labels)
        total_loss += loss.item()
        total_token_acc += metrics["token_acc"]
        total_ce_loss += ce_loss.item()
        total_halt_loss += halt_loss.item()
        avg_halt_acc += halt_acc.item()
        total_batches += 1

        if log_interval > 0 and step % log_interval == 0:
            print(
                f"    step {step:05d} | loss {loss.item():.4f} | token acc {metrics['token_acc']:.3f}"
            )

    denom = max(total_batches, 1)
    return {
        "loss": total_loss / denom,
        "token_acc": total_token_acc / denom,
        "ce_loss": total_ce_loss / denom,
        "halt_loss": total_halt_loss / denom,
        "halt_acc": avg_halt_acc / denom,
    }


def evaluate(
    model: TinyRecursiveModel,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    # total_loss = 0.0
    total_token_acc = 0.0
    total_puzzle_acc = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            labels = batch["labels"].to(device)
            preds, _ = model.predict(inputs)
            # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

            token_acc = (preds == labels).float().mean().item()
            puzzle_acc = (preds == labels).all(dim=-1).float().mean().item()

            # total_loss += loss.item()
            total_token_acc += token_acc
            total_puzzle_acc += puzzle_acc
            total_batches += 1

    denom = max(total_batches, 1)
    return {
        # "loss": total_loss / denom,
        "token_acc": total_token_acc / denom,
        "puzzle_acc": total_puzzle_acc / denom,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TRM (impl version) on Sudoku")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config override")
    args = parser.parse_args()

    cfg_dict = load_run_config(args.config)
    model_cfg = TRMConfig.from_dict(cfg_dict["model"])
    data_cfg = SudokuDataConfig(**cfg_dict["data"])
    training_cfg = TrainingConfig(**cfg_dict["training"])

    set_seed(training_cfg.seed)
    device_str = training_cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    loaders = build_dataloaders(data_cfg, training_cfg)
    model = TinyRecursiveModel(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg.lr,
        weight_decay=training_cfg.weight_decay,
    )

    output_dir = Path(training_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    print(f"Model parameters: {count_parameters(model):,}")
    for epoch in range(1, training_cfg.epochs + 1):
        train_stats = train_epoch(
            model,
            loaders["train"],
            optimizer,
            device,
            training_cfg.grad_clip,
            training_cfg.halt_loss_weight,
            training_cfg.log_interval,
        )
        print(
            "Epoch {ep:03d} | train loss {tot:.4f} (ce {ce:.4f} + halt {halt:.4f}) | token acc {acc:.3f} | halt acc {ht:.3f}".format(
                ep=epoch,
                tot=train_stats["loss"],
                ce=train_stats["ce_loss"],
                halt=train_stats["halt_loss"],
                acc=train_stats["token_acc"],
                ht=train_stats["halt_acc"],
            )
        )

        if training_cfg.run_eval and epoch % training_cfg.eval_interval == 0:
            eval_stats = evaluate(
                model,
                loaders["eval"],
                device,
            )
            print(
                "           eval token acc {t_acc:.3f} | puzzle acc {p_acc:.3f}".format(
                    # loss=eval_stats["loss"],
                    t_acc=eval_stats["token_acc"],
                    p_acc=eval_stats["puzzle_acc"],
                )
            )
            if eval_stats["puzzle_acc"] > best_acc:
                best_acc = eval_stats["puzzle_acc"]
                checkpoint = {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "eval_stats": eval_stats,
                    "config": cfg_dict,
                }
                torch.save(checkpoint, output_dir / "best.pt")
                print("           Saved new best checkpoint.")

    torch.save({"model_state": model.state_dict(), "config": cfg_dict}, output_dir / "last.pt")
    print("Training complete. Checkpoints stored in", output_dir)


if __name__ == "__main__":
    main()
