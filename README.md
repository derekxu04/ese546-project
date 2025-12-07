# Tiny Recursive Model Sandbox

Lightweight playground for experimenting with the Tiny Recursive Model (TRM) from *Less is More: Recursive Reasoning with Tiny Networks*. This repo exposes a minimal PyTorch implementation, a Sudoku data loader, and a training script that can run on laptops without hugging huge dependencies.

## Repo Contents

- `tiny_trm_model.py` – standalone TRM implementation (config + model + helper funcs).
- `data_sudoku.py` – utilities for preparing Sudoku data. Supports a built-in toy dataset (default) or the full `sapientinc/sudoku-extreme` release from Hugging Face.
- `train_trm.py` – command-line training script with editable configuration.

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install torch torchvision torchaudio numpy einops tqdm huggingface_hub
```

Feel free to add additional packages as you explore.

## Running Training

```bash
source .venv/bin/activate
python train_trm.py
```

Key behavior:
- By default, the script uses the **built-in toy dataset** defined in `data_sudoku.py` so nothing large is downloaded.
- The resolved configuration is stored in `RUN_CONFIG` inside `train_trm.py`. Modify that dictionary or supply `--config custom.json` (same structure) to tweak hyperparameters, dataset paths, etc.
- Model checkpoints are saved under `runs/tiny_trm_sudoku/` (`best.pt`, `last.pt`).

## Dataset Options

`data_sudoku.py` supports two modes:

1. **Built-in toy dataset** (default): set `use_builtin_sample=True` in the data config. A handful of puzzles are embedded directly in the repo so there is zero download cost—ideal for quick sanity checks or when running on low-memory machines.
2. **Full Sudoku Extreme dataset**: set `use_builtin_sample=False`. The script will download CSV splits from Hugging Face. Optional knobs:
   - `train_subset`: randomly sample at most this many puzzles before caching.
   - `splits`: choose which splits to fetch (e.g., `("train",)` to skip test data).
   - `force_download`: re-fetch even if `.npz` files already exist.

## Customizing Training

Inside `RUN_CONFIG`:
- `model` – controls TRM depth, width, recursion cycles, etc.
- `data` – points to dataset directory and selection flags described above.
- `training` – epochs, batch size, learning rate, device override, logging cadence.

Example override file (`tiny.json`):
```json
{
  "data": {
    "use_builtin_sample": false,
    "train_subset": 1000
  },
  "training": {
    "batch_size": 32,
    "epochs": 5
  }
}
```
Run with `python train_trm.py --config tiny.json`.

## Next Steps

- Swap Sudoku for another puzzle dataset by adding a new loader.
- Extend `tiny_trm_model.py` with additional features (e.g., Rope, ACT halting).
- Integrate wandb or TensorBoard logging for richer monitoring.

Happy experimenting! 🎲
