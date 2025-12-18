"""Training script for TinyRecursiveModel+JEPA (impl version).

Combines token-level cross-entropy + halting loss with a spatial JEPA loss that
predicts masked cell latents from the visible context.
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

from data_sudoku import SudokuDataConfig, SudokuDataset, prepare_sudoku_dataset
from trm_jepa import TinyRecursiveModel, TRMConfig



CONFIG_DIR = Path(__file__).parent / "configs"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "trm_jepa_default.json"


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
    halt_warmup_epochs: int
    jepa_loss_weight: float
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
    # Download/cache Sudoku splits and wrap into loaders.
    prepare_sudoku_dataset(cfg)
    train_ds = SudokuDataset(cfg.dataset_dir, split="train")
    #test_split = "test" if (Path(cfg.dataset_dir) / "test.npz").exists() else "train"
    #eval_ds = SudokuDataset(cfg.dataset_dir, split=test_split)
    # Require a real held-out test split to avoid train/test leakage.
    test_path = Path(cfg.dataset_dir) / "test.npz"
    assert test_path.exists(), (
        "Missing test.npz — refusing to fall back to train split "
        "(this would cause train/test leakage)."
    )
    eval_ds = SudokuDataset(cfg.dataset_dir, split="test")

    # Optional deterministic subsample to keep runs manageable.
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
    # Base latent states shared across the batch (expand to batch size).
    outputs, latents = model.get_initial()
    outputs = outputs.expand(batch_size, -1, -1)
    latents = latents.expand(batch_size, -1, -1)
    return outputs, latents


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    # Token- and puzzle-level accuracy.
    preds = torch.argmax(logits, dim=-1)
    token_acc = (preds == labels).float().mean().item()
    puzzle_acc = (preds == labels).all(dim=-1).float().mean().item()
    return {"token_acc": token_acc, "puzzle_acc": puzzle_acc}


def compute_losses(
    logits: torch.Tensor,
    halt_prob: torch.Tensor,
    labels: torch.Tensor,
    halt_loss_weight: float,
    jepa_loss: torch.Tensor,
    jepa_weight: float,
) -> Dict[str, torch.Tensor]:
    # Cross-entropy over all tokens.
    ce = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
    preds = torch.argmax(logits.detach(), dim=-1)
    puzzle_correct = (preds == y).all(dim=-1).float()
    puzzle_correct = 0.9 * puzzle_correct + 0.05 # Label smoothing for halt targets

    # Supervise halting on puzzle correctness with light smoothing.
    halt_loss = F.binary_cross_entropy(halt_prob, puzzle_correct)
    halt_loss = halt_loss.clamp(max=10.0)


    if train_cfg.halt_warmup_epochs > 0 and epoch < train_cfg.halt_warmup_epochs:
        halt_term = 0.0
    else:
        halt_weight = train_cfg.halt_loss_weight
        if epoch > 200:
            halt_weight *= 0.1

        halt_term = halt_weight * halt_loss

    jepa_term = train_cfg.jepa_loss_weight * jepa_loss

    total_loss = ce + halt_term + jepa_term
    return {
        "total": total,
        "ce": ce.detach(),
        "halt": halt_loss.detach(),
        "jepa": jepa_loss.detach(),
        "halt_term": halt_term.detach(),
        "jepa_term": jepa_term.detach(),
        "halt_acc": puzzle_correct.mean().detach(),
    }


def train_epoch(
    model: TinyRecursiveModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    train_cfg: TrainingConfig,
    eval_loader: Optional[DataLoader] = None,
    eval_interval: int = 100,
) -> Dict[str, float]:
    model.train()

    stats = {
        "loss": 0.0,
        "token_acc": 0.0,
        "ce_loss": 0.0,
        "halt_loss": 0.0,
        "jepa_raw": 0.0,
        "jepa_term": 0.0,
        "halt_acc": 0.0,
        "test_loss": 0.0,
        "test_token_acc": 0.0,
        "test_puzzle_acc": 0.0,
        "opt_steps": 0,
        "examples": 0,
    }

    for batch in loader:
        x_full = batch["inputs"].to(device)
        y_full = batch["labels"].to(device)

        jepa_loss, _ = model.spatial_jepa_loss(x_full, y_full)

        outputs, latents = prepare_initial_states(model, x_full.size(0))
        outputs = outputs.to(device)
        latents = latents.to(device)

        x = x_full
        y = y_full
        active_outputs = outputs
        active_latents = latents

        for sup_step in range(model.config.max_supervision_steps):
            is_last = sup_step == model.config.max_supervision_steps - 1

            logits, halt_prob, next_outputs, next_latents = model(
                x, active_outputs, active_latents
            )
            ce = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            preds = logits.argmax(dim=-1)
            puzzle_correct = (preds == y).all(dim=-1).float()

            halt_loss = F.binary_cross_entropy(halt_prob, puzzle_correct)

            # Combine CE + halting + JEPA (JEPA detached to avoid double-backprop).
            total_loss = (
                ce
                + train_cfg.halt_loss_weight * halt_loss
                + train_cfg.jepa_loss_weight * jepa_loss.detach()
            )


            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            with torch.no_grad():
                token_acc = (preds == y).float().mean().item()
                puzzle_acc = puzzle_correct.mean().item()

            stats["loss"] += total_loss.item()
            stats["token_acc"] += token_acc
            stats["ce_loss"] += ce.item()
            stats["halt_loss"] += halt_loss.item()
            stats["jepa_raw"] += jepa_loss.item()
            stats["jepa_term"] += (train_cfg.jepa_loss_weight * jepa_loss).item()
            stats["halt_acc"] += puzzle_acc
            stats["opt_steps"] += 1
            # count supervised examples processed in this optimization step
            stats["examples"] += y.size(0)

            # Early-stop samples that already met halt threshold.
            should_halt = (halt_prob >= model.config.halt_prob_threshold) | is_last
            if should_halt.all():
                break

            next_outputs = next_outputs.detach()
            next_latents = next_latents.detach()

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
        stats["test_loss"] = test_stats["loss"]
        stats["test_token_acc"] = test_stats["token_acc"]
        stats["test_puzzle_acc"] = test_stats["puzzle_acc"]
        model.train()  # Switch back to train mode

    denom = max(stats["opt_steps"], 1)
    return {
        "loss": stats["loss"] / denom,
        "token_acc": stats["token_acc"] / denom,
        "ce_loss": stats["ce_loss"] / denom,
        "halt_loss": stats["halt_loss"] / denom,
        "jepa_raw": stats["jepa_raw"] / denom,
        "jepa_term": stats["jepa_term"] / denom,
        "halt_acc": stats["halt_acc"] / denom,
        "test_loss": stats["test_loss"] if eval_loader else 0.0,
        "test_token_acc": stats["test_token_acc"] if eval_loader else 0.0,
        "test_puzzle_acc": stats["test_puzzle_acc"] if eval_loader else 0.0,
        "examples": stats["examples"],
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TRM+JEPA (impl version) on Sudoku")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config override")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--metrics-csv", type=str, default=None, help="Path to output CSV metrics file")
    args = parser.parse_args()

    cfg_dict = load_run_config(args.config)
    model_cfg = TRMConfig.from_dict(cfg_dict["model"])
    data_cfg = SudokuDataConfig(**cfg_dict["data"])
    training_cfg = TrainingConfig(**cfg_dict["training"])  # type: ignore[arg-type]

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
            'Train JEPA Raw',
            'Train JEPA Weighted',
            'Train Halt Acc',
            'Test CE Loss',
            'Test Token Acc',
            'Test Puzzle Acc',
            'Eval Token Acc',
            'Eval Puzzle Acc',
            'Epoch Time (s)',
            'Peak GPU MB',
            'Throughput (examples/s)',
        ])

    # Initialize placeholder for eval stats
    eval_stats = {"token_acc": "", "puzzle_acc": ""}
    last_epoch = start_epoch - 1
    print(f"Model parameters: {count_parameters(model):,}")
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
            training_cfg,
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
            "Epoch {ep:03d} | train loss {tot:.4f} "
            "(ce {ce:.4f} + halt {halt:.4f} + jepa {jepa:.4f}) "
            "| token acc {acc:.3f} | halt acc {ht:.3f} "
            "| time {t:.2f}s | thr {thr:.1f} ex/s".format(
                ep=epoch,
                tot=train_stats["loss"],
                ce=train_stats["ce_loss"],
                halt=train_stats["halt_loss"],
                jepa=train_stats["jepa_term"],   # ✅ weighted JEPA
                acc=train_stats["token_acc"],
                ht=train_stats["halt_acc"],
                t=epoch_time,
                thr=(throughput if throughput != '' else 0.0),
            )
        )
        # run evaluation if requested
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

        # write epoch metrics
        if metrics_file is not None:
            csv_writer.writerow([
                epoch,
                train_stats['loss'],
                train_stats['token_acc'],
                train_stats['ce_loss'],
                train_stats['halt_loss'],
                train_stats['jepa_raw'],
                train_stats['jepa_term'],
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


if __name__ == "__main__":
    main()
