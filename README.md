# Tiny Recursive Model / Hierarchical Reasoning Model / LLM TRM-JEPA Sandbox

Experimental playground exploring recursive reasoning architectures on structured problems. This repository implements:

- **TRM (Tiny Recursive Model)**: A compact architecture that iteratively refines predictions through latent recursion and adaptive computation (halting mechanism). The model learns when to stop reasoning via a learned halting probability.

- **HRM (Hierarchical Recursive Model)**: Extends TRM with separate low-level (L_net) and high-level (H_net) reasoning networks, enabling hierarchical multi-scale processing.

- **JEPA (Joint-Embedding Predictive Architecture)**: Augments TRM/HRM with spatial latent alignment losses inspired by self-supervised learning, encouraging structured representations.

- **LLM + TRM + JEPA**: Applies TRM-style iterative reasoning to fine-tune large language models on mathematical reasoning tasks.

All actively maintained code lives in `impl/`; experiments and plots are orchestrated from `scripts/` and saved under `runs/` and `plots/`.

## Repository Layout

### Core Implementations (`impl/`)

**Base Models:**
- `trm.py` — Tiny Recursive Model: single shared network for latent + output refinement with learned halting.
- `trm_jepa.py` — TRM + JEPA: adds spatial masking predictor for self-supervised latent alignment.
- `hrm.py` — Hierarchical Recursive Model: separate L_net (low-level) and H_net (high-level) architectures.
- `hrm_jepa.py` — HRM + JEPA: hierarchical reasoning with latent alignment objectives.

**Training Scripts:**
- `train_trm.py` — trains TRM on Sudoku with cross-entropy + halting supervision.
- `train_trm_jepa.py` — trains TRM+JEPA with additional spatial JEPA loss (masked cell prediction).
- `train_hrm.py` — trains HRM on Sudoku with hierarchical supervision.
- `train_hrm_jepa.py` — trains HRM+JEPA combining hierarchical + JEPA objectives.
- `train_llm_trm_jepa.py` — fine-tunes LLMs (Gemma/Llama) with TRM-inspired iterative reasoning on GSM8k math problems.

**Data & Config:**
- `data_sudoku.py` — downloads/caches Sudoku data from Hugging Face (`sapientinc/sudoku-extreme`), supports deterministic subsampling.
- `configs/` — JSON configs for all trainers (model architecture, data, training hyperparameters).

### Experiment Orchestration (`scripts/`)

- `run_experiments.py` — **main experiment driver**: runs hyperparameter sweeps with preview → selection → full training pipeline, generates comparison plots automatically.
- `compare_runs.py` — plots multiple runs side-by-side (e.g., TRM vs TRM+JEPA) from CSV metrics.
- `plot_metrics.py` — generates individual training curves from a single CSV log.

### Outputs (`runs/` and `plots/`)

- `runs/metrics/` — CSV logs with epoch-by-epoch metrics (loss, accuracy, throughput).
- `runs/experiments_*` — model checkpoints (`.pt` files), training logs, intermediate outputs.
- `runs/experiment_summary.csv` — summary of hyperparameter search and selected configs.
- `plots/` — generated visualizations: train/test CE loss, token/puzzle accuracy curves, TRM vs JEPA comparisons.

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

### HRM (Hierarchical Recursive Model)
```bash
source venv/bin/activate
python impl/train_hrm.py --config impl/configs/hrm_default.json
```

### HRM + JEPA
```bash
source venv/bin/activate
python impl/train_hrm_jepa.py --config impl/configs/hrm_jepa_default.json
```

### LLM + TRM + JEPA (GSM8k fine-tuning)
```bash
source venv/bin/activate
python impl/train_llm_trm_jepa.py --config impl/configs/llm_trm_jepa_gsm8k.json
```

Sudoku trainers (TRM/HRM) download `sapientinc/sudoku-extreme` by default and honor `train_subset` (set to `50_000` in configs to keep runs manageable). Test split is always held out; no fallback to train.

## Experiment Driver (scripts/run_experiments.py)

Automates hyperparameter sweeps with intelligent selection and plotting. The workflow:

1. **Preview Phase**: Runs all configs for a few epochs on a small subset (512 samples) to quickly score them.
2. **Selection**: Picks top-k per family (TRM vs JEPA) based on a specified metric (e.g., `token_acc`).
3. **Full Training**: Trains selected configs for many epochs (500) on the full dataset (50k samples).
4. **Comparison Plots**: Automatically generates side-by-side plots comparing TRM vs TRM+JEPA performance.

Example command:
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

**Key flags:**
- `--preview-subset`: number of training samples for quick preview runs.
- `--refine-k`: how many configs per family to select for full training.
- `--refine-metric`: metric to rank preview runs (`token_acc`, `eval_puzzle_acc`, etc.).
- `--hidden-sizes`, `--trm-layers`, `--jepa-weights`: define the hyperparameter grid.
- `--parallel --gpus 0 --max-procs 1`: run experiments sequentially on GPU 0.

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

**Sudoku (TRM/HRM experiments):**
- Default: `sapientinc/sudoku-extreme` with ~3.8M training puzzles and 422k held-out test puzzles.
- Configs use `train_subset=50_000` for manageable training times (full dataset available if needed).
- Subsampling is deterministic (fixed seed) for reproducibility.
- **Test split is mandatory**: scripts enforce held-out evaluation to prevent train/test leakage.

**GSM8k (LLM experiments):**
- Grade-school math reasoning dataset with ~7.5k training examples.
- Used by `train_llm_trm_jepa.py` to fine-tune LLMs (Gemma, Llama) with TRM-inspired iterative reasoning.
- Each problem consists of a question and chain-of-thought solution.

**Quick sanity checks:**
- Set `use_builtin_sample=true` in a config to use 3 embedded puzzles (zero download, instant startup).

## Architecture Notes

**TRM (Tiny Recursive Model):**
- Uses a single shared Transformer to iteratively refine both latents and outputs.
- Learns when to stop via a halting head (supervised by puzzle correctness).
- Combines multiple latent refinements per output update for efficient reasoning.

**HRM (Hierarchical Recursive Model):**
- Separates reasoning into low-level (L_net) and high-level (H_net) networks.
- L_net handles fine-grained token updates; H_net coordinates global structure.
- Enables multi-scale processing for complex reasoning tasks.

**JEPA (Joint-Embedding Predictive Architecture):**
- Adds self-supervised spatial masking: predict masked cell representations from visible context.
- Encourages structured latent representations aligned with problem structure.
- Combined with task loss (CE + halting) for end-to-end training.

## Tips

- Start with `train_subset=1000` and `epochs=10` for quick prototyping.
- Use `--preview-count 0` to run all configs in the grid (or set a limit like `--preview-count 4`).
- Monitor `runs/experiment_summary.csv` to see which configs were selected.
- Check `plots/refined_full_training/` for automatic TRM vs JEPA comparisons.
- For debugging, set `use_builtin_sample=true` and `epochs=3` in a config override.

Happy experimenting! 🎲
