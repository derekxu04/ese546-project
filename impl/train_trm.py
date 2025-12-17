"""Training script for the Tiny Recursive Model defined in impl/trm.py.

This mirrors the reference/train_trm.py workflow but binds to the newer
TinyRecursiveModel implementation that exposes explicit latent recursion and
halting heads.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import time
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Subset

# Use the impl-local Sudoku helpers to avoid modifying sys.path.
from data_sudoku import SudokuDataConfig, SudokuDataset, prepare_sudoku_dataset
from trm import TinyRecursiveModel, TRMConfig


CONFIG_DIR = Path(__file__).parent / "configs"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "trm_default.json"


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
    #test_split = "test" if (Path(cfg.dataset_dir) / "test.npz").exists() else "train"
    #eval_ds = SudokuDataset(cfg.dataset_dir, split=test_split)
    test_path = Path(cfg.dataset_dir) / "test.npz"
    assert test_path.exists(), (
        "Missing test.npz — refusing to fall back to train split "
        "(this would cause train/test leakage)."
    )
    eval_ds = SudokuDataset(cfg.dataset_dir, split="test")
    # IMPORTANT: Apply train_subset at runtime even if train.npz already exists.
    if cfg.train_subset is not None and len(train_ds) > cfg.train_subset:
        g = torch.Generator()
        g.manual_seed(cfg.seed)
        idx = torch.randperm(len(train_ds), generator=g)[: cfg.train_subset].tolist()
        train_ds = Subset(train_ds, idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=(train_cfg.device == "cuda" or (train_cfg.device is None and torch.cuda.is_available())),
        persistent_workers=(train_cfg.num_workers > 0),

    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=(train_cfg.device == "cuda" or (train_cfg.device is None and torch.cuda.is_available())),
        persistent_workers=(train_cfg.num_workers > 0),
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
    puzzle_correct = (preds == labels).all(dim=-1).float()  # (b,)

    halt_loss = F.binary_cross_entropy(halt_prob, puzzle_correct)

    total = ce + halt_loss_weight * halt_loss
    return total, ce.detach(), halt_loss.detach(), puzzle_correct.detach()


def train_epoch(
    model: TinyRecursiveModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    halt_loss_weight: float,
    log_interval: int,
    eval_loader: Optional[DataLoader] = None,
    eval_interval: int = 100,
) -> Dict[str, float]:

    model.train()

    total_loss = 0.0
    total_ce_loss = 0.0
    total_halt_loss = 0.0
    total_token_acc = 0.0
    total_puzzle_acc = 0.0
    total_test_loss = 0.0
    total_test_token_acc = 0.0
    total_test_puzzle_acc = 0.0

    total_optimizer_steps = 0  # counts supervision steps, not batches
    total_examples = 0  # count supervised examples processed (for throughput)

    for batch_idx, batch in enumerate(loader, start=1):
        x_full = batch["inputs"].to(device)
        y_full = batch["labels"].to(device)

        # initial (y,z) latents for this batch
        outputs, latents = prepare_initial_states(model, x_full.size(0))
        outputs = outputs.to(device)
        latents = latents.to(device)

        # active set (we'll shrink like predict())
        x = x_full
        y = y_full
        active_outputs = outputs
        active_latents = latents

        # We do up to N deep supervision steps per batch
        for sup_step in range(model.config.max_supervision_steps):
            is_last = (sup_step == model.config.max_supervision_steps - 1)

            logits, halt_prob, next_outputs, next_latents = model(
                x, active_outputs, active_latents
            )

            loss, ce_loss, halt_loss, puzzle_correct = compute_loss(
                logits, halt_prob, y, halt_loss_weight
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            # Metrics for this supervision step (only on current active subset)
            with torch.no_grad():
                preds = torch.argmax(logits, dim=-1)
                token_acc = (preds == y).float().mean().item()
                puzzle_acc = (preds == y).all(dim=-1).float().mean().item()

            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_halt_loss += halt_loss.item()
            total_token_acc += token_acc
            total_puzzle_acc += puzzle_acc
            total_optimizer_steps += 1
            # number of supervised examples in this opt step (active set size)
            total_examples += y.size(0)

            if log_interval > 0 and total_optimizer_steps % log_interval == 0:
                print(
                    f"    optstep {total_optimizer_steps:05d} | "
                    f"sup {sup_step:02d} | loss {loss.item():.4f} | token acc {token_acc:.3f} | puzzle acc {puzzle_acc:.3f}"
                )

            # Early stopping mask (per-sample), like inference.
            with torch.no_grad():
                should_halt = (halt_prob >= model.config.halt_prob_threshold) | is_last

            if should_halt.all():
                break

            next_outputs = next_outputs.detach()
            next_latents = next_latents.detach()

            # Shrink active set for the next supervision iteration
            keep = ~should_halt
            x = x[keep]
            y = y[keep]
            active_outputs = next_outputs[keep]
            active_latents = next_latents[keep]

            if x.numel() == 0:
                break

    # Compute test loss once per epoch
    if eval_loader is not None:
        test_stats = compute_test_loss(model, eval_loader, device)
        total_test_loss = test_stats["loss"]
        total_test_token_acc = test_stats["token_acc"]
        total_test_puzzle_acc = test_stats["puzzle_acc"]
        model.train()  # Switch back to train mode

    denom = max(total_optimizer_steps, 1)
    return {
        "loss": total_loss / denom,
        "token_acc": total_token_acc / denom,
        "puzzle_acc": total_puzzle_acc / denom,
        "ce_loss": total_ce_loss / denom,
        "halt_loss": total_halt_loss / denom,
        # interpret "halt_acc" as average puzzle_correct rate seen at training steps
        # (not identical to "halt head accuracy", but still useful)
        "halt_acc": total_puzzle_acc / denom,
        "test_loss": total_test_loss if eval_loader else 0.0,
        "test_token_acc": total_test_token_acc if eval_loader else 0.0,
        "test_puzzle_acc": total_test_puzzle_acc if eval_loader else 0.0,
        "examples": total_examples,
    }



def compute_test_loss(
    model: TinyRecursiveModel,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Compute test loss and accuracy on batches from the eval loader."""
    model.eval()
    total_loss = 0.0
    total_token_acc = 0.0
    total_puzzle_acc = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            labels = batch["labels"].to(device)
            preds, _ = model.predict(inputs)
            
            # Compute loss by running forward with initialized states
            #outputs, latents = prepare_initial_states(model, inputs.size(0))
            #outputs = outputs.to(device)
            #latents = latents.to(device)
            outputs, latents = model.get_initial()
            outputs = outputs.expand(inputs.size(0), -1, -1).to(device)
            latents = latents.expand(inputs.size(0), -1, -1).to(device)

            logits, _, _, _ = model(inputs, outputs, latents)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

            token_acc = (preds == labels).float().mean().item()
            puzzle_acc = (preds == labels).all(dim=-1).float().mean().item()

            total_loss += loss.item()
            total_token_acc += token_acc
            total_puzzle_acc += puzzle_acc
            total_batches += 1

    denom = max(total_batches, 1)
    return {
        "loss": total_loss / denom,
        "token_acc": total_token_acc / denom,
        "puzzle_acc": total_puzzle_acc / denom,
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
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--metrics-csv", type=str, default=None, help="Path to output CSV metrics file")
    args = parser.parse_args()

    cfg_dict = load_run_config(args.config)
    model_cfg = TRMConfig.from_dict(cfg_dict["model"])
    data_cfg = SudokuDataConfig(**cfg_dict["data"])
    training_cfg = TrainingConfig(**cfg_dict["training"])

    set_seed(training_cfg.seed)
    device_str = training_cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Fallback to CPU if CUDA requested but unavailable
    if device_str == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available; falling back to CPU")
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"Using device: {device_str}")

    loaders = build_dataloaders(data_cfg, training_cfg)
    model = TinyRecursiveModel(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg.lr,
        weight_decay=training_cfg.weight_decay,
    )

    start_epoch = 1
    best_acc = 0.0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer_state = checkpoint.get("optimizer_state")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        best_acc = checkpoint.get("best_acc", checkpoint.get("eval_stats", {}).get("puzzle_acc", 0.0))
        start_epoch = checkpoint.get("epoch", 0) + 1
        print(f"Resumed from {args.resume} at epoch {start_epoch - 1}")

    output_dir = Path(training_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = None
    metrics_csv_path = args.metrics_csv
    if metrics_csv_path is not None:
        metrics_path = Path(metrics_csv_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_file = open(metrics_path, 'w', newline='')
        csv_writer = csv.writer(metrics_file)
        csv_writer.writerow([
            'Epoch',
            'Train Total Loss',
            'Train Token Acc',
            'Train CE Loss',
            'Train Halt Loss',
            'Train Halt Acc',
            'Test CE Loss',
            'Test Token Acc',
            'Test Puzzle Acc',
            'Eval Token Acc',
            'Eval Puzzle Acc',
            'Epoch Time (s)',
            'Peak GPU MB',
            'Throughput (examples/s)'
        ])

    last_epoch = start_epoch - 1
    print(f"Model parameters: {count_parameters(model):,}")
    # Initialize placeholders for eval stats when not running evaluation
    eval_stats = {"token_acc": "", "puzzle_acc": ""}
    for epoch in range(start_epoch, training_cfg.epochs + 1):
        # reset peak memory counters and time the epoch
        if device.type == 'cuda':
            try:
                torch.cuda.reset_peak_memory_stats(device)
            except Exception:
                pass
        start_time = time.time()
        train_stats = train_epoch(
            model,
            loaders["train"],
            optimizer,
            device,
            training_cfg.grad_clip,
            training_cfg.halt_loss_weight,
            training_cfg.log_interval,
            loaders["eval"],
            training_cfg.eval_interval,
        )
        epoch_time = time.time() - start_time

        # peak GPU memory in MB (if available)
        peak_gpu_mb = ''
        if device.type == 'cuda':
            try:
                peak_gpu_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            except Exception:
                peak_gpu_mb = ''

        throughput = ''
        if isinstance(train_stats.get('examples', None), (int, float)) and epoch_time > 0:
            throughput = train_stats.get('examples', 0) / epoch_time

        print(
            "Epoch {ep:03d} | train loss {tot:.4f} (ce {ce:.4f} + halt {halt:.4f}) | token acc {acc:.3f} | halt acc {ht:.3f} | time {t:.2f}s | thr {thr:.1f} ex/s".format(
                ep=epoch,
                tot=train_stats["loss"],
                ce=train_stats["ce_loss"],
                halt=train_stats["halt_loss"],
                acc=train_stats["token_acc"],
                ht=train_stats["halt_acc"],
                t=epoch_time,
                thr=(throughput if throughput != '' else 0.0),
            )
        )
        # run evaluation
        if training_cfg.run_eval and epoch % training_cfg.eval_interval == 0:
            eval_stats = evaluate(
                model,
                loaders["eval"],
                device,
            )
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
                }
                torch.save(checkpoint, output_dir / "best.pt")
                print("           Saved new best checkpoint.")

        # write basic epoch metrics
        if metrics_file is not None:
            csv_writer.writerow([
                epoch,
                train_stats['loss'],
                train_stats['token_acc'],
                train_stats['ce_loss'],
                train_stats['halt_loss'],
                train_stats['halt_acc'],
                train_stats.get('test_loss', ''),
                train_stats.get('test_token_acc', ''),
                train_stats.get('test_puzzle_acc', ''),
                (eval_stats['token_acc'] if training_cfg.run_eval else ''),
                (eval_stats['puzzle_acc'] if training_cfg.run_eval else ''),
                epoch_time,
                (peak_gpu_mb if peak_gpu_mb != '' else ''),
                (throughput if throughput != '' else ''),
            ])
            metrics_file.flush()
        # periodically save checkpoints every 100 epochs
        if epoch % 100 == 0:
            checkpoint = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "eval_stats": eval_stats,
                "best_acc": best_acc,
                "config": cfg_dict,
            }
            torch.save(checkpoint, output_dir / f"checkpoint_epoch_{epoch}.pt")
            print(f"           Saved checkpoint at epoch {epoch}.")

        # (Evaluation handled above when `run_eval` was True)
        last_epoch = epoch

    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": last_epoch,
            "best_acc": best_acc,
            "config": cfg_dict,
        },
        output_dir / "last.pt",
    )
    print("Training complete. Checkpoints stored in", output_dir)
    if metrics_file is not None:
        metrics_file.close()

    # If metrics CSV requested, flush/close is handled by context manager; nothing special to do here.


if __name__ == "__main__":
    main()
