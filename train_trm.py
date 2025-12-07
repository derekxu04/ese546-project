"""Training script for the simplified Tiny Recursive Model (TRM).

Edit the ``RUN_CONFIG`` dictionary below or provide a JSON override via
``--config path/to/config.json`` to change hyperparameters without touching the
rest of the code.
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
        "use_builtin_sample": False, # True for overfit to small dataset
    },
    "training": {
        "epochs": 20,
        "batch_size": 128,
        "lr": 3e-4,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "eval_interval": 1,
        "log_interval": 100,
        "seed": 0,
        "num_workers": 0,
        "output_dir": "runs/tiny_trm_sudoku",
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


def train_epoch(
    model: TinyRecursiveModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    log_interval: int,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_batches = 0
    total_acc = 0.0
    for step, batch in enumerate(loader, start=1):
        inputs = batch["inputs"].to(device)
        labels = batch["labels"].to(device)
        logits = model(inputs)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        metrics = compute_metrics(logits.detach(), labels)
        total_loss += loss.item()
        total_acc += metrics["token_acc"]
        total_batches += 1

        if log_interval > 0 and step % log_interval == 0:
            print(
                f"    step {step:05d} | loss {loss.item():.4f} | token acc {metrics['token_acc']:.3f}"
            )

    return {
        "loss": total_loss / max(total_batches, 1),
        "token_acc": total_acc / max(total_batches, 1),
    }


def evaluate(model: TinyRecursiveModel, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_token_acc = 0.0
    total_puzzle_acc = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            labels = batch["labels"].to(device)
            logits = model(inputs)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            metrics = compute_metrics(logits, labels)
            total_loss += loss.item()
            total_token_acc += metrics["token_acc"]
            total_puzzle_acc += metrics["puzzle_acc"]
            total_batches += 1
    return {
        "loss": total_loss / max(total_batches, 1),
        "token_acc": total_token_acc / max(total_batches, 1),
        "puzzle_acc": total_puzzle_acc / max(total_batches, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Tiny Recursive Model on Sudoku")
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
        )
        print(f"Epoch {epoch:03d} | train loss {train_stats['loss']:.4f} | token acc {train_stats['token_acc']:.3f}")

        if epoch % training_cfg.eval_interval == 0:
            eval_stats = evaluate(model, loaders["eval"], device)
            print(
                f"           eval loss {eval_stats['loss']:.4f} | token acc {eval_stats['token_acc']:.3f} | puzzle acc {eval_stats['puzzle_acc']:.3f}"
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
