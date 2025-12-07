"""Training script for Tiny Recursive Model with JEPA-style latent alignment.

This script extends ``train_trm.py`` by encoding both the puzzle inputs and the
ground-truth solutions through TRM, then minimizing a latent-space distance
between their final hidden representations in addition to the token-level loss.
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

from data_sudoku import SudokuDataConfig, SudokuDataset, prepare_sudoku_dataset
from tiny_trm_model import TinyRecursiveModel, TinyTRMConfig, build_tiny_trm, count_parameters


RUN_CONFIG: Dict = {
    "model": {
        "hidden_size": 256,
        "num_heads": 4,
        "num_layers": 2,
        "H_cycles": 3,
        "L_cycles": 6,
        "vocab_size": 10,
        "seq_len": 81,
        "dropout": 0.05,
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
        "log_interval": 50,
        "latent_loss_weight": 10000.0,
        "latent_pool": "mean",  # "mean" or "cls" (first token)
        "normalize_latent": True,
        "stopgrad_target": True,
        "seed": 0,
        "num_workers": 0,
        "output_dir": "runs/tiny_trm_sudoku_jepa",
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
    log_interval: int
    latent_loss_weight: float
    latent_pool: str
    normalize_latent: bool
    stopgrad_target: bool
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


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    preds = torch.argmax(logits, dim=-1)
    token_acc = (preds == labels).float().mean().item()
    puzzle_acc = (preds == labels).all(dim=-1).float().mean().item()
    return {"token_acc": token_acc, "puzzle_acc": puzzle_acc}


def pool_hidden_states(hidden: torch.Tensor, method: str) -> torch.Tensor:
    if method == "mean":
        return hidden.mean(dim=1)
    if method == "cls":
        return hidden[:, 0]
    raise ValueError(f"Unknown pooling method '{method}'")


def compute_latent_loss(
    model: TinyRecursiveModel,
    labels: torch.Tensor,
    puzzle_ids: Optional[torch.Tensor],
    predicted_hidden: torch.Tensor,
    pool_method: str,
    normalize: bool,
    stopgrad_target: bool,
) -> torch.Tensor:
    with torch.no_grad() if stopgrad_target else torch.enable_grad():
        target_logits, target_hidden = model(labels, puzzle_ids=puzzle_ids, return_hidden=True)
    target_repr = pool_hidden_states(target_hidden, pool_method)
    if normalize:
        target_repr = F.normalize(target_repr, dim=-1)

    pred_repr = pool_hidden_states(predicted_hidden, pool_method)
    if normalize:
        pred_repr = F.normalize(pred_repr, dim=-1)

    if stopgrad_target:
        target_repr = target_repr.detach()
    latent_loss = F.mse_loss(pred_repr, target_repr)
    return latent_loss


def train_epoch(
    model: TinyRecursiveModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    log_interval: int,
    train_cfg: TrainingConfig,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_latent = 0.0
    total_batches = 0
    total_acc = 0.0

    for step, batch in enumerate(loader, start=1):
        inputs = batch["inputs"].to(device)
        labels = batch["labels"].to(device)

        logits, hidden_states = model(inputs, return_hidden=True)
        base_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

        latent_loss = torch.tensor(0.0, device=device)
        if train_cfg.latent_loss_weight > 0:
            latent_loss = compute_latent_loss(
                model,
                labels,
                puzzle_ids=None,
                predicted_hidden=hidden_states,
                pool_method=train_cfg.latent_pool,
                normalize=train_cfg.normalize_latent,
                stopgrad_target=train_cfg.stopgrad_target,
            )

        loss = base_loss + train_cfg.latent_loss_weight * latent_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        metrics = compute_metrics(logits.detach(), labels)
        total_loss += base_loss.item()
        total_latent += latent_loss.item()
        total_acc += metrics["token_acc"]
        total_batches += 1

        if log_interval > 0 and step % log_interval == 0:
            print(
                f"    step {step:05d} | ce {base_loss.item():.4f} | latent {latent_loss.item():.4f} | token acc {metrics['token_acc']:.3f}"
            )

    return {
        "ce_loss": total_loss / max(total_batches, 1),
        "latent_loss": total_latent / max(total_batches, 1),
        "token_acc": total_acc / max(total_batches, 1),
    }


def evaluate(
    model: TinyRecursiveModel,
    loader: DataLoader,
    device: torch.device,
    train_cfg: TrainingConfig,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_latent = 0.0
    total_token_acc = 0.0
    total_puzzle_acc = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            labels = batch["labels"].to(device)
            logits, hidden_states = model(inputs, return_hidden=True)
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            latent_loss = torch.tensor(0.0, device=device)
            if train_cfg.latent_loss_weight > 0:
                latent_loss = compute_latent_loss(
                    model,
                    labels,
                    puzzle_ids=None,
                    predicted_hidden=hidden_states,
                    pool_method=train_cfg.latent_pool,
                    normalize=train_cfg.normalize_latent,
                    stopgrad_target=train_cfg.stopgrad_target,
                )
            metrics = compute_metrics(logits, labels)
            total_loss += ce_loss.item()
            total_latent += latent_loss.item()
            total_token_acc += metrics["token_acc"]
            total_puzzle_acc += metrics["puzzle_acc"]
            total_batches += 1
    return {
        "ce_loss": total_loss / max(total_batches, 1),
        "latent_loss": total_latent / max(total_batches, 1),
        "token_acc": total_token_acc / max(total_batches, 1),
        "puzzle_acc": total_puzzle_acc / max(total_batches, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Tiny Recursive Model with JEPA loss on Sudoku")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config override")
    args = parser.parse_args()

    cfg_dict = load_run_config(args.config)
    model_cfg = TinyTRMConfig.from_dict(cfg_dict["model"])
    data_cfg = SudokuDataConfig(**cfg_dict["data"])
    training_cfg = TrainingConfig(**cfg_dict["training"])

    set_seed(training_cfg.seed)
    device_str = training_cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    loaders = build_dataloaders(data_cfg, training_cfg)
    model = build_tiny_trm(model_cfg.to_dict()).to(device)
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
            training_cfg.log_interval,
            training_cfg,
        )
        print(
            f"Epoch {epoch:03d} | ce {train_stats['ce_loss']:.4f} | latent {train_stats['latent_loss']:.4f} | token acc {train_stats['token_acc']:.3f}"
        )

        if epoch % training_cfg.eval_interval == 0:
            eval_stats = evaluate(model, loaders["eval"], device, training_cfg)
            print(
                "           eval |\n"
                f"             ce {eval_stats['ce_loss']:.4f} | latent {eval_stats['latent_loss']:.4f} | token acc {eval_stats['token_acc']:.3f} | puzzle acc {eval_stats['puzzle_acc']:.3f}"
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
