# Tiny Recursive Model Sandbox

Lightweight playground for experimenting with the Tiny Recursive Model (TRM) from *Less is More: Recursive Reasoning with Tiny Networks*. All actively maintained code lives inside the `impl/` directory.

## Key Components (all under `impl/`)

- `trm.py` – primary TRM implementation with latent recursion, halting head, and prediction helpers.
- `data_sudoku.py` – Sudoku dataset utilities (built-in sample + Hugging Face downloader/subsampler).
- `train_trm.py` – training loop for the `impl/trm.py` model using standard cross-entropy plus halting supervision.
- `train_trm_jepa.py` – training variant that augments the base loss with a JEPA-style latent alignment term.

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install torch torchvision torchaudio numpy einops tqdm huggingface_hub
```

Feel free to add additional packages as you explore.

## Running Training (impl version)

```bash
source .venv/bin/activate
python impl/train_trm.py
```

Key behavior:
- Uses the **built-in toy dataset** (three puzzles) unless you flip `use_builtin_sample=False` in the data config.
- All knobs live in `impl/train_trm.py`’s `RUN_CONFIG`. Override by editing the dict or passing `--config custom.json` (same structure).
- Checkpoints land in `runs/tiny_trm_impl_sudoku/` (`best.pt`, `last.pt`).
- Training logs report token accuracy, the cross-entropy/halting-loss split, and the average halt target.

### JEPA-style Training

```bash
source .venv/bin/activate
python impl/train_trm_jepa.py
```

This variant feeds both puzzle and solution through TRM and adds a latent-alignment objective, inspired by [LLM-JEPA](https://arxiv.org/abs/2509.14252). Extra knobs in `RUN_CONFIG["training"]`:

- `latent_loss_weight`: scales the JEPA penalty (set to `0.0` to disable).
- `latent_pool`: `"mean"` (default) or `"cls"` to aggregate sequence states.
- `normalize_latent`: whether to L2-normalize representations before computing MSE.
- `stopgrad_target`: if `True`, the target (solution) branch is stop-grad, mimicking JEPA's target encoder.
- Checkpoints go under `runs/tiny_trm_sudoku_jepa/`.

## Dataset Options

`impl/data_sudoku.py` supports two modes:

1. **Built-in toy dataset** (default): set `use_builtin_sample=True` in the data config. A handful of puzzles are embedded directly in the repo so there is zero download cost—ideal for quick sanity checks or when running on low-memory machines.
2. **Hugging Face dataset** (e.g., `sapientinc/sudoku-extreme` or `SakanaAI/Sudoku-Bench`): set `use_builtin_sample=False`. Optional knobs:
  - `train_subset`: randomly keep only `train_subset` puzzles before caching.
  - `splits`: choose which splits to fetch (e.g., `("train",)` to skip test data).
  - `force_download`: re-fetch even if `.npz` files already exist.
  - `repo_id`: point to a lighter dataset if you want quicker downloads.

## Customizing Training

Inside `impl/train_trm.py::RUN_CONFIG`:
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
Run with `python impl/train_trm.py --config tiny.json`.

## Next Steps

- Swap Sudoku for another puzzle dataset by adding a new loader inside `impl/`.
- Extend `impl/trm.py` with additional features (e.g., Rope, ACT halting).
- Integrate wandb or TensorBoard logging for richer monitoring.

Happy experimenting! 🎲
