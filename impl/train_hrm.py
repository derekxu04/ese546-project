"""Training script for the Hierarchical Reasoning Model defined in impl/hrm.py.

Implements deep supervision training for HRM with two-level hierarchical states
(zH and zL) instead of TRM's single-level (y and z).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Use the impl-local Sudoku helpers to avoid modifying sys.path.
from data_sudoku import SudokuDataConfig, SudokuDataset, prepare_sudoku_dataset
from hrm import HierarchicalReasoningModel, HRMConfig


CONFIG_DIR = Path(__file__).parent / "configs"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "hrm_default.json"


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
    checkpoint_interval: int
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
    with open(DEFAULT_CONFIG_PATH) as handle:
        cfg = json.load(handle)
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


def prepare_initial_states(model: HierarchicalReasoningModel, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare initial hierarchical states for HRM.

    Returns:
        (zH, zL): Initial high-level and low-level states
    """
    return model.get_initial(batch_size, device)


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
    model: HierarchicalReasoningModel,
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
    total_samples = 0
    start_time = time.time()

    for step, batch in enumerate(loader, start=1):
        inputs = batch["inputs"].to(device)
        labels = batch["labels"].to(device)
        init_zH, init_zL = prepare_initial_states(model, inputs.size(0), device)

        # HRM forward pass with hierarchical states
        logits, halt_prob, _, _ = model(inputs, init_zH, init_zL)
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
        total_samples += inputs.size(0)

        if log_interval > 0 and step % log_interval == 0:
            print(
                f"    step {step:05d} | loss {loss.item():.4f} | token acc {metrics['token_acc']:.3f}"
            )

    elapsed_time = time.time() - start_time
    denom = max(total_batches, 1)
    return {
        "loss": total_loss / denom,
        "token_acc": total_token_acc / denom,
        "ce_loss": total_ce_loss / denom,
        "halt_loss": total_halt_loss / denom,
        "halt_acc": avg_halt_acc / denom,
        "epoch_time": elapsed_time,
        "samples_per_sec": total_samples / elapsed_time if elapsed_time > 0 else 0,
    }


def evaluate(
    model: HierarchicalReasoningModel,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_token_acc = 0.0
    total_puzzle_acc = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            labels = batch["labels"].to(device)
            preds, _ = model.predict(inputs)

            token_acc = (preds == labels).float().mean().item()
            puzzle_acc = (preds == labels).all(dim=-1).float().mean().item()

            total_token_acc += token_acc
            total_puzzle_acc += puzzle_acc
            total_batches += 1

    denom = max(total_batches, 1)
    return {
        "token_acc": total_token_acc / denom,
        "puzzle_acc": total_puzzle_acc / denom,
    }


def plot_training_curves(history: Dict[str, list], output_dir: Path) -> None:
    """Plot and save training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot total loss
    axes[0, 0].plot(history["epochs"], history["train_loss"], label="Total Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Plot token accuracy
    axes[0, 1].plot(history["epochs"], history["train_token_acc"], label="Train Token Acc")
    if history["eval_token_acc"]:
        axes[0, 1].plot(history["eval_epochs"], history["eval_token_acc"], label="Eval Token Acc", marker='o')
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_title("Token Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Plot CE and halt loss
    axes[1, 0].plot(history["epochs"], history["ce_loss"], label="CE Loss")
    axes[1, 0].plot(history["epochs"], history["halt_loss"], label="Halt Loss")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].set_title("Loss Components")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Plot halt accuracy and puzzle accuracy
    axes[1, 1].plot(history["epochs"], history["halt_acc"], label="Halt Acc")
    if history["eval_puzzle_acc"]:
        axes[1, 1].plot(history["eval_epochs"], history["eval_puzzle_acc"], label="Eval Puzzle Acc", marker='o')
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].set_title("Halt & Puzzle Accuracy")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HRM (impl version) on Sudoku")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config override")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    cfg_dict = load_run_config(args.config)
    model_cfg = HRMConfig.from_dict(cfg_dict["model"])
    data_cfg = SudokuDataConfig(**cfg_dict["data"])
    training_cfg = TrainingConfig(**cfg_dict["training"])  # type: ignore[arg-type]

    set_seed(training_cfg.seed)
    device_str = training_cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    loaders = build_dataloaders(data_cfg, training_cfg)
    model = HierarchicalReasoningModel(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg.lr,
        weight_decay=training_cfg.weight_decay,
    )

    start_epoch = 1
    best_acc = 0.0
    # Training history for plotting
    history = {
        "epochs": [],
        "train_loss": [],
        "train_token_acc": [],
        "ce_loss": [],
        "halt_loss": [],
        "halt_acc": [],
        "eval_epochs": [],
        "eval_token_acc": [],
        "eval_puzzle_acc": [],
    }

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer_state = checkpoint.get("optimizer_state")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        best_acc = checkpoint.get("best_acc", checkpoint.get("eval_stats", {}).get("puzzle_acc", 0.0))
        start_epoch = checkpoint.get("epoch", 0) + 1
        # Restore history if available
        history = checkpoint.get("history", history)
        print(f"Resumed from {args.resume} at epoch {start_epoch - 1}")

    output_dir = Path(training_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    last_epoch = start_epoch - 1
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"HRM Configuration: N={model_cfg.num_refinement_blocks}, T={model_cfg.num_low_refinements}")
    print(f"Total updates per step: {model_cfg.num_refinement_blocks * model_cfg.num_low_refinements} low-level, {model_cfg.num_refinement_blocks} high-level")

    for epoch in range(start_epoch, training_cfg.epochs + 1):
        train_stats = train_epoch(
            model,
            loaders["train"],
            optimizer,
            device,
            training_cfg.grad_clip,
            training_cfg.halt_loss_weight,
            training_cfg.log_interval,
        )

        # Record training metrics
        history["epochs"].append(epoch)
        history["train_loss"].append(train_stats["loss"])
        history["train_token_acc"].append(train_stats["token_acc"])
        history["ce_loss"].append(train_stats["ce_loss"])
        history["halt_loss"].append(train_stats["halt_loss"])
        history["halt_acc"].append(train_stats["halt_acc"])

        print(
            "Epoch {ep:03d} | train loss {tot:.4f} (ce {ce:.4f} + halt {halt:.4f}) | token acc {acc:.3f} | halt acc {ht:.3f} | time {time:.2f}s | throughput {thr:.1f} samples/s".format(
                ep=epoch,
                tot=train_stats["loss"],
                ce=train_stats["ce_loss"],
                halt=train_stats["halt_loss"],
                acc=train_stats["token_acc"],
                ht=train_stats["halt_acc"],
                time=train_stats["epoch_time"],
                thr=train_stats["samples_per_sec"],
            )
        )

        if training_cfg.run_eval and epoch % training_cfg.eval_interval == 0:
            eval_stats = evaluate(
                model,
                loaders["eval"],
                device,
            )

            # Record evaluation metrics
            history["eval_epochs"].append(epoch)
            history["eval_token_acc"].append(eval_stats["token_acc"])
            history["eval_puzzle_acc"].append(eval_stats["puzzle_acc"])

            print(
                "           eval token acc {t_acc:.3f} | puzzle acc {p_acc:.3f}".format(
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
                    "best_acc": best_acc,
                    "config": cfg_dict,
                    "history": history,
                }
                torch.save(checkpoint, output_dir / "best.pt")
                print("           Saved new best checkpoint.")

        # Save periodic checkpoint every checkpoint_interval epochs
        if epoch % training_cfg.checkpoint_interval == 0:
            checkpoint = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "best_acc": best_acc,
                "config": cfg_dict,
                "history": history,
            }
            torch.save(checkpoint, output_dir / f"checkpoint_epoch_{epoch}.pt")
            print(f"           Saved checkpoint at epoch {epoch}")

        last_epoch = epoch

    # Plot training curves
    plot_training_curves(history, output_dir)
    print(f"Training curves saved to {output_dir / 'training_curves.png'}")

    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": last_epoch,
            "best_acc": best_acc,
            "config": cfg_dict,
            "history": history,
        },
        output_dir / "last.pt",
    )
    print("Training complete. Checkpoints stored in", output_dir)


if __name__ == "__main__":
    main()
