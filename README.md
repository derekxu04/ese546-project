# Tiny Recursive Model Sandbox

Experimental playground for TRM and TRM+JEPA on Sudoku. Actively maintained code lives in `impl/`; experiments and plots are orchestrated from `scripts/` and saved under `runs/` and `plots/`.

## Repository Layout

- `impl/`
  - `trm.py` — core Tiny Recursive Model with latent recursion + halting.
  - `trm_jepa.py` — TRM with JEPA-style latent alignment head.
  - `train_trm.py` — training loop for TRM (cross-entropy + halting loss).
  - `train_trm_jepa.py` — training loop for TRM+JEPA (adds latent alignment loss).
  - `data_sudoku.py` — dataset prep/downloader (Hugging Face `sapientinc/sudoku-extreme`); supports subsampling via `train_subset`.
  - `configs/` — default configs for the impl trainers.

- `scripts/`
  - `run_experiments.py` — experiment driver: preview sweep, selection, full training, and comparison plots.
  - `compare_runs.py` — plot multiple runs (TRM vs TRM+JEPA) from CSV metrics.
  - `plot_metrics.py` — single-run metric plots from one CSV.
  - `plot_metrics.py` / `compare_runs.py` expect CSVs produced by the impl trainers.

- `runs/`
  - `metrics/` — CSV logs for each run (used by plotting scripts).
  - `experiments_*` — model checkpoints and outputs per experiment.
  - `experiment_summary.csv` — selection log from `run_experiments.py`.

- `plots/`
  - Generated PNGs from `plot_metrics.py` and `compare_runs.py` (train/test CE loss, token accuracy, puzzle accuracy).

## Quickstart

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install torch torchvision torchaudio numpy einops tqdm huggingface_hub pandas matplotlib
```

If you need a zero-download sanity check, set `use_builtin_sample=true` in a config (uses a tiny embedded 3-puzzle set).

## Training (impl)

### Plain TRM
```bash
source venv/bin/activate
python impl/train_trm.py --config impl/configs/trm_default.json
```

### TRM + JEPA
```bash
source venv/bin/activate
python impl/train_trm_jepa.py --config impl/configs/trm_jepa_default.json
```

Both trainers download `sapientinc/sudoku-extreme` by default and honor `train_subset` (set to `50_000` in configs to keep runs manageable). Test split is always held out; no fallback to train.

## Experiment Driver (scripts/run_experiments.py)

End-to-end sweep and plotting:
```bash
source venv/bin/activate
python scripts/run_experiments.py \
  --out-dir runs/experiments_main \
  --preview-count 0 \
  --preview-epochs 15 \
  --preview-subset 512 \
  --refine-k 1 \
  --refine-metric token_acc \
  --full-epochs 500 \
  --hidden-sizes 128,256 \
  --trm-layers 1,2,3 \
  --jepa-weights 1e-04,5e-04,1e-03 \
  --parallel --gpus 0 --max-procs 1 --force
```

Phases:
- Preview: short runs on a subset to score configs (TRM + JEPA families).
- Selection: top-k per family by `token_acc`.
- Full: long runs on selected configs.
- Plots: comparison PNGs in `plots/` and CSVs in `runs/metrics/`.

## Plotting

- Single run: `python scripts/plot_metrics.py --csv runs/metrics/<run>.csv --out plots/<dest>`
- Compare runs: `python scripts/compare_runs.py --csv <run1>.csv <run2>.csv --labels run1 run2 --out plots/<dest>`

Outputs include train/test CE loss and token/puzzle accuracy curves. Comparison plots style TRM vs JEPA differently.

## Config overrides (keep it small when needed)

You can override any config field with a JSON file. Example:

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

Run with:

```bash
python impl/train_trm.py --config tiny.json
```

## Data Notes

- Default dataset: `sapientinc/sudoku-extreme` (~3.8M train / 422k test). Configs set `train_subset=50_000` for faster iterations.
- Subsampling is reproducible via `train_subset` and `seed` in `data_sudoku.py` configs.
- Test split (`test.npz`) is mandatory; scripts assert its presence to avoid train/test leakage.