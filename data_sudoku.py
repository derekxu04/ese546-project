"""Utilities for preparing and loading Sudoku data."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from huggingface_hub import hf_hub_download


BUILTIN_PUZZLES: Tuple[Tuple[str, str], ...] = (
    (
        "530070000600195000098000060800060003400803001700020006060000280000419005000080079",
        "534678912672195348198342567859761423426853791713924856961537284287419635345286179",
    ),
    (
        "200080300060070084030500209000105408000000000402706000301007040720040060004010003",
        "245981376169273584837564219976135428318429657452796831391857642723648195584312963",
    ),
    (
        "000000907000420180000705026100904000050000040000507009920108000034059000507000000",
        "461238957795426183382715426176943852259871643843567219928184765634759891517692534",
    ),
)


@dataclass
class SudokuDataConfig:
    """Configuration for data preparation and loading."""

    dataset_dir: str = "data/sudoku-small"
    # repo_id: str = "sapientinc/sudoku-extreme"
    repo_id: str = "SakanaAI/Sudoku-Bench"
    splits: Tuple[str, ...] = ("train", "test")
    train_subset: Optional[int] = None
    seed: int = 0
    force_download: bool = False
    use_builtin_sample: bool = False

    def path(self) -> Path:
        return Path(self.dataset_dir)


# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------


def _decode_board(board: str) -> np.ndarray:
    vals = [0 if ch == '.' else int(ch) for ch in board.strip()]
    if len(vals) != 81:
        raise ValueError("Each Sudoku board must contain exactly 81 values.")
    return np.array(vals, dtype=np.int64)


def _read_split(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    boards: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    with open(csv_path, newline="") as handle:
        reader = csv.reader(handle)
        next(reader)  # skip header
        for _source, puzzle, solution, _rating in reader:
            boards.append(_decode_board(puzzle))
            labels.append(_decode_board(solution))
    return np.stack(boards, axis=0), np.stack(labels, axis=0)


def _subsample(inputs: np.ndarray, labels: np.ndarray, limit: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.choice(inputs.shape[0], size=limit, replace=False)
    return inputs[idx], labels[idx]


def prepare_sudoku_dataset(config: SudokuDataConfig) -> Dict[str, Path]:
    """Download and cache Sudoku splits as compressed numpy files."""

    root = config.path()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "config.json"

    if config.use_builtin_sample:
        _write_builtin_dataset(root, config)
        with open(manifest_path, "w") as f:
            json.dump(asdict(config), f, indent=2)
        return {split: root / f"{split}.npz" for split in config.splits}

    output_paths: Dict[str, Path] = {}
    for split in config.splits:
        target = root / f"{split}.npz"
        output_paths[split] = target
        if target.exists() and not config.force_download:
            continue

        csv_file = hf_hub_download(
            repo_id=config.repo_id,
            filename=f"{split}.csv",
            repo_type="dataset",
            force_download=config.force_download,
        )
        inputs, labels = _read_split(Path(csv_file))
        if split == "train" and config.train_subset is not None and inputs.shape[0] > config.train_subset:
            inputs, labels = _subsample(inputs, labels, config.train_subset, config.seed)

        np.savez_compressed(target, inputs=inputs, labels=labels)

    with open(manifest_path, "w") as f:
        json.dump(asdict(config), f, indent=2)

    return output_paths


def _write_builtin_dataset(root: Path, config: SudokuDataConfig) -> None:
    inputs = []
    labels = []
    for puzzle, solution in BUILTIN_PUZZLES:
        inputs.append(_decode_board(puzzle))
        labels.append(_decode_board(solution))

    inputs_arr = np.stack(inputs, axis=0)
    labels_arr = np.stack(labels, axis=0)

    for split in config.splits:
        np.savez_compressed(root / f"{split}.npz", inputs=inputs_arr, labels=labels_arr)


# -----------------------------------------------------------------------------
# PyTorch Dataset
# -----------------------------------------------------------------------------


class SudokuDataset(Dataset):
    """Simple Dataset returning flattened Sudoku boards."""

    def __init__(self, root: str, split: str = "train") -> None:
        self.root = Path(root)
        self.split = split
        npz_path = self.root / f"{split}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing {npz_path}. Run prepare_sudoku_dataset first.")

        data = np.load(npz_path)
        self.inputs = torch.from_numpy(data["inputs"].astype(np.int64))
        self.labels = torch.from_numpy(data["labels"].astype(np.int64))

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "inputs": self.inputs[idx],
            "labels": self.labels[idx],
        }


if __name__ == "__main__":
    cfg = SudokuDataConfig(train_subset=100)
    paths = prepare_sudoku_dataset(cfg)
    print("Prepared:", paths)
    ds = SudokuDataset(cfg.dataset_dir, split="train")
    sample = ds[0]
    print("Sample keys:", sample.keys())
