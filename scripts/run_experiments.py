import argparse
import json
import os
import subprocess
import time
from itertools import product
from copy import deepcopy
import csv

try:
    import torch as _torch
    _has_torch = True
except Exception:
    _torch = None
    _has_torch = False


DEFAULTS = {
    "epochs": 10,
    "batch_size": 128,
    "lr": 0.0003,
}

SUMMARY_CSV = os.path.join("runs", "experiment_summary.csv")


# -----------------------------
# I/O helpers
# -----------------------------
def write_override(base_name: str, overrides: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{base_name}.json")
    with open(path, "w") as f:
        json.dump(overrides, f, indent=2)
    return path


def read_best_metric(csv_path: str, refine_metric: str, maximize: bool):
    """
    Reads a CSV and returns the best value over all rows for the requested metric.
    - refine_metric: one of {token_acc, eval_token_acc, eval_puzzle_acc, train_loss}
    - maximize: True for accuracy, False for loss
    Returns -inf / +inf if unreadable or metric missing.
    """

    import os
    import pandas as pd

    # -----------------------------
    # Guardrails
    # -----------------------------
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return float("-inf") if maximize else float("inf")

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return float("-inf") if maximize else float("inf")

    if df.empty:
        return float("-inf") if maximize else float("inf")

    # -----------------------------
    # Metric name mapping
    # -----------------------------
    METRIC_MAP = {
        "token_acc": [
            "Train Token Acc",
            "Token Accuracy",
            "token_acc",
        ],
        "eval_token_acc": [
            "Eval Token Acc",
        ],
        "eval_puzzle_acc": [
            "Eval Puzzle Acc",
        ],
        "train_loss": [
            "Train Total Loss",
            "Train Loss",
        ],
    }

    if refine_metric not in METRIC_MAP:
        raise ValueError(f"Unknown refine_metric: {refine_metric}")

    # Normalize column names once
    norm_cols = {
        c.lower().strip().replace(" ", "_"): c
        for c in df.columns
    }

    # -----------------------------
    # Find the first valid column
    # -----------------------------
    for cand in METRIC_MAP[refine_metric]:
        key = cand.lower().strip().replace(" ", "_")
        if key in norm_cols:
            col = norm_cols[key]
            series = df[col].dropna()

            if series.empty:
                continue

            return float(series.max() if maximize else series.min())

    # -----------------------------
    # Metric not found
    # -----------------------------
    return float("-inf") if maximize else float("inf")

def metric_spec(refine_metric: str):
    """
    Returns (candidates, maximize_bool) for the chosen refine metric.
    """
    if refine_metric == "eval_puzzle_acc":
        return (["Eval Puzzle Acc", "eval_puzzle_acc", "eval_puzzle_accuracy"], True)
    if refine_metric == "eval_token_acc":
        return (["Eval Token Acc", "eval_token_acc", "eval_token_accuracy"], True)
    if refine_metric == "token_acc":
        return (["Token Accuracy", "token_acc", "token_accuracy"], True)
    if refine_metric == "train_loss":
        return (["Train Loss", "train_loss", "loss"], False)
    # fallback
    return (["Eval Puzzle Acc", "eval_puzzle_acc"], True)


# -----------------------------
# Parallel runner
# -----------------------------
def run_experiments_parallel(
    exps,
    gen_dir,
    metrics_dir,
    epochs,
    subset,
    suffix,
    parallel,
    gpu_list,
    max_procs,
    force,
):
    """
    Runs each experiment for a fixed epochs (and optional subset).
    Writes metrics CSVs named f"{name}_{suffix}.csv".
    Returns list of (exp, metrics_csv_path).
    """
    os.makedirs(gen_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    running = []  # (Popen, exp_name, csv_path)

    def spawn(cmd, env, exp_name, csv_path):
        print("Spawning:", " ".join(cmd), "CUDA_VISIBLE_DEVICES=", env.get("CUDA_VISIBLE_DEVICES"))
        p = subprocess.Popen(cmd, env=env)
        running.append((p, exp_name, csv_path))

    outputs = []

    for idx, exp in enumerate(exps):
        exp_name = f"{exp['name']}_{suffix}"

        overrides = deepcopy(exp["override"])
        overrides.setdefault("training", {})
        overrides["training"]["epochs"] = epochs

        overrides.setdefault("data", {})
        if subset is not None:
            overrides["data"]["train_subset"] = subset

        cfg_path = write_override(exp_name, overrides, gen_dir)
        metrics_csv = os.path.join(metrics_dir, f"{exp_name}.csv")

        if os.path.exists(metrics_csv) and not force:
            print(f"Skipping {exp_name} because {metrics_csv} exists (use --force to re-run)")
            outputs.append((exp, metrics_csv))
            continue

        cmd = ["python3", exp["script"], "--config", cfg_path, "--metrics-csv", metrics_csv]

        if not parallel:
            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=True)
            outputs.append((exp, metrics_csv))
            continue

        # parallel mode
        env = os.environ.copy()
        if gpu_list:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_list[idx % len(gpu_list)])

        # throttle
        while len(running) >= max_procs:
            for p, n, c in list(running):
                ret = p.poll()
                if ret is not None:
                    running.remove((p, n, c))
                    if ret != 0:
                        raise RuntimeError(f"Process {n} failed with return code {ret}")
            if len(running) >= max_procs:
                time.sleep(0.5)

        spawn(cmd, env, exp_name, metrics_csv)

    # wait remaining
    while running:
        for p, n, c in list(running):
            ret = p.poll()
            if ret is not None:
                running.remove((p, n, c))
                if ret != 0:
                    raise RuntimeError(f"Process {n} failed with return code {ret}")
                else:
                    # success
                    pass
        if running:
            time.sleep(0.5)

    # add outputs from parallel-spawned runs too
    # (sequential ones already appended)
    # In this implementation, we append outputs when we *schedule* sequential runs,
    # but for parallel we didn't append — so we recompute expected csv paths.
    scheduled = set((exp["name"], exp["script"]) for exp in exps)
    for exp in exps:
        exp_name = f"{exp['name']}_{suffix}"
        metrics_csv = os.path.join(metrics_dir, f"{exp_name}.csv")
        if (exp["name"], exp["script"]) in scheduled and (exp, metrics_csv) not in outputs:
            outputs.append((exp, metrics_csv))

    return outputs


# -----------------------------
# Selection helpers
# -----------------------------
def split_families(experiments):
    trm = []
    jepa = []
    for exp in experiments:
        if exp["name"].startswith("trm_"):
            trm.append(exp)
        elif exp["name"].startswith("jepa_"):
            jepa.append(exp)
        else:
            # default: treat as TRM-like
            trm.append(exp)
    return trm, jepa


def select_top_k_per_family(preview_outputs, refine_metric, k_per_family=1):
    """
    preview_outputs: list of (exp, csv_path) with preview metrics
    Returns: chosen_trm_exps, chosen_jepa_exps (lists)
    """
    cands, maximize = metric_spec(refine_metric)

    trm_scored = []
    jepa_scored = []

    for exp, csv_path in preview_outputs:
        score = read_best_metric(csv_path, refine_metric, maximize=maximize)
        if exp["name"].startswith("trm_"):
            trm_scored.append((score, exp, csv_path))
        elif exp["name"].startswith("jepa_"):
            jepa_scored.append((score, exp, csv_path))
        else:
            trm_scored.append((score, exp, csv_path))

    trm_scored.sort(key=lambda x: x[0], reverse=maximize)
    jepa_scored.sort(key=lambda x: x[0], reverse=maximize)

    chosen_trm = [x[1] for x in trm_scored[:k_per_family]] if trm_scored else []
    chosen_jepa = [x[1] for x in jepa_scored[:k_per_family]] if jepa_scored else []


    write_header = not os.path.exists(SUMMARY_CSV)

    with open(SUMMARY_CSV, "a", newline="") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow([
                "exp_name",
                "family",
                "hidden_size",
                "num_layers",
                "jepa_weight",
                "preview_metric",
                "preview_score",
                "selected",
                "metrics_csv",
            ])

        for score, exp, csv_path in trm_scored:
            cfg = exp["override"]["model"]
            writer.writerow([
                exp["name"],
                "TRM",
                cfg.get("hidden_size"),
                cfg.get("num_layers"),
                "",
                refine_metric,
                score,
                exp in chosen_trm,
                os.path.basename(csv_path),
            ])

        for score, exp, csv_path in jepa_scored:
            cfg = exp["override"]["model"]
            writer.writerow([
                exp["name"],
                "JEPA",
                cfg.get("hidden_size"),
                "",
                exp["override"]["training"]["jepa_loss_weight"],
                refine_metric,
                score,
                exp in chosen_jepa,
                os.path.basename(csv_path),
            ])


    print("\n=== Preview selection ===")
    if trm_scored:
        print("TRM ranked (top first):")
        for s, e, p in trm_scored[: min(10, len(trm_scored))]:
            print(f"  {e['name']}: score={s:.6f}  csv={os.path.basename(p)}")
    else:
        print("No TRM preview results found.")

    if jepa_scored:
        print("JEPA ranked (top first):")
        for s, e, p in jepa_scored[: min(10, len(jepa_scored))]:
            print(f"  {e['name']}: score={s:.6f}  csv={os.path.basename(p)}")
    else:
        print("No JEPA preview results found.")

    print("\nChosen TRM:", [e["name"] for e in chosen_trm])
    print("Chosen JEPA:", [e["name"] for e in chosen_jepa])
    print("=========================\n")

    return chosen_trm, chosen_jepa


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--out-dir", default="runs/experiments", help="Directory for run outputs")
    parser.add_argument("--device", default=None, help="Device override (e.g., cpu or cuda)")
    parser.add_argument("--quick", action="store_true", help="Run quick configs (smaller default epochs)")
    parser.add_argument("--hidden-sizes", type=str, default=None, help="Comma-separated hidden sizes")
    parser.add_argument("--trm-layers", type=str, default=None, help="Comma-separated TRM num_layers values")
    parser.add_argument("--jepa-weights", type=str, default=None, help="Comma-separated JEPA loss weights")
    parser.add_argument("--force", action="store_true", help="Force re-run experiments even if metrics exist")
    parser.add_argument("--parallel", action="store_true", help="Run experiments in parallel")
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated GPU device ids to use (e.g., 0,1).")
    parser.add_argument("--max-procs", type=int, default=None, help="Maximum concurrent processes")

    # Two-stage knobs
    parser.add_argument("--preview-epochs", type=int, default=3, help="Epochs for preview runs")
    parser.add_argument("--preview-count", type=int, default=0, help="Number of preview configs to run (0=all)")
    parser.add_argument("--preview-subset", type=int, default=256, help="Train subset size for preview runs")
    parser.add_argument("--train-subset", type=int, default=None, help="Train subset for FULL runs (optional)")

    parser.add_argument(
        "--refine-metric",
        choices=["eval_puzzle_acc", "eval_token_acc", "token_acc", "train_loss"],
        default="eval_puzzle_acc",
        help="Metric to rank preview runs",
    )
    parser.add_argument("--refine-k", type=int, default=1, help="Top-k per family to run fully (default 1)")
    parser.add_argument("--full-epochs", type=int, default=1000, help="Epochs for full runs")
    parser.add_argument("--generate-only", action="store_true", help="Only generate override configs and selection; do not start training")

    # legacy flags (ignored in this rewrite)
    parser.add_argument("--halving", action="store_true", help="(ignored) successive halving not implemented here")
    parser.add_argument("--rungs", type=str, default="3,8,20", help="(ignored)")
    parser.add_argument("--keep-frac", type=float, default=0.5, help="(ignored)")

    args = parser.parse_args()

    # GPU list
    gpu_list = None
    if args.gpus:
        gpu_list = [s.strip() for s in args.gpus.split(",") if s.strip()]
    if args.parallel and not gpu_list:
        gpu_list = ["0"]

    # Adjust parallelism defaults for available devices
    # If CUDA requested but PyTorch isn't available or no CUDA GPUs present, warn and fall back to CPU+sequential runs
    if args.device == "cuda":
        if not _has_torch:
            print(
                "WARNING: --device cuda requested but PyTorch not available; falling back to CPU and disabling parallel to avoid OOM."
            )
            args.device = "cpu"
            args.parallel = False
        else:
            if not _torch.cuda.is_available():
                print(
                    "WARNING: --device cuda requested but no CUDA GPUs available; falling back to CPU and disabling parallel to avoid OOM."
                )
                args.device = "cpu"
                args.parallel = False

    # If parallel requested but only one GPU is provided, disable parallel to avoid oversubscription
    gpu_count = len(gpu_list) if gpu_list else 0
    if args.parallel and gpu_count <= 1:
        print(f"Only {gpu_count} GPU(s) available; running sequentially to avoid over-subscription.")
        args.parallel = False

    # Cap max_procs to the number of GPUs if GPUs are used, otherwise default to 1
    max_procs = args.max_procs or (len(gpu_list) if gpu_list else 1)
    if gpu_count > 0:
        max_procs = min(max_procs, gpu_count)

    # Directories
    gen_dir = os.path.join("configs", "generated")
    metrics_dir = os.path.join("runs", "metrics")
    os.makedirs(gen_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # Build experiment grids
    trm_hidden_sizes = [128, 256] if not args.hidden_sizes else [int(h) for h in args.hidden_sizes.split(",")]
    trm_num_layers = [1, 2] if not args.trm_layers else [int(l) for l in args.trm_layers.split(",")]

    jepa_hidden_sizes = [128, 256] if not args.hidden_sizes else [int(h) for h in args.hidden_sizes.split(",")]
    jepa_weights = [5e-6, 1e-4] if not args.jepa_weights else [float(w) for w in args.jepa_weights.split(",")]

    trm_experiments = []
    for hs, nl in product(trm_hidden_sizes, trm_num_layers):
        overrides = {
            "model": {"hidden_size": hs, "num_layers": nl},
            "training": {
                "epochs": 0,  # filled later
                "batch_size": 128,
                "lr": 0.0003,
                "output_dir": os.path.join(args.out_dir, f"trm_h{hs}_l{nl}"),
            },
            "data": {},
        }
        if args.device is not None:
            overrides["training"]["device"] = args.device
        elif args.parallel and gpu_list:
            overrides["training"]["device"] = "cuda"

        trm_experiments.append(
            {"name": f"trm_h{hs}_l{nl}", "script": "impl/train_trm.py", "override": overrides}
        )

    jepa_experiments = []
    for hs, jw in product(jepa_hidden_sizes, jepa_weights):
        jw_label = str(jw).replace(".", "p").replace("-", "m")
        overrides = {
            "model": {"hidden_size": hs, "spatial_mask_ratio": 0.3},
            "training": {
                "epochs": 0,  # filled later
                "batch_size": 128,
                "lr": 0.0003,
                "jepa_loss_weight": jw,
                "output_dir": os.path.join(args.out_dir, f"jepa_h{hs}_w{jw_label}"),
            },
            "data": {},
        }
        if args.device is not None:
            overrides["training"]["device"] = args.device
        elif args.parallel and gpu_list:
            overrides["training"]["device"] = "cuda"

        jepa_experiments.append(
            {"name": f"jepa_h{hs}_w{jw_label}", "script": "impl/train_trm_jepa.py", "override": overrides}
        )

    experiments = trm_experiments + jepa_experiments

    # -----------------------------
    # Phase A: Preview ALL configs
    # -----------------------------
    print(f"\n[Phase A] Preview: running {len(experiments)} configs for {args.preview_epochs} epochs "
          f"(subset={args.preview_subset})\n")

    # Limit preview configs if requested
    preview_count = args.preview_count if args.preview_count and args.preview_count > 0 else len(experiments)

    # When the user asks to limit the number of preview configs, make sure we include
    # at least one config from each family (TRM and JEPA) if they exist. This ensures
    # that the hyperparameter search covers both families and that the refinement
    # stage can pick the best from each family.
    if preview_count >= len(experiments):
        preview_exps = experiments[:]
    else:
        trm_list, jepa_list = split_families(experiments)
        preview_exps = []
        i = 0
        # Interleave TRM and JEPA configs while we still need more previews
        while len(preview_exps) < preview_count and (i < len(trm_list) or i < len(jepa_list)):
            if i < len(trm_list) and len(preview_exps) < preview_count:
                preview_exps.append(trm_list[i])
            if i < len(jepa_list) and len(preview_exps) < preview_count:
                preview_exps.append(jepa_list[i])
            i += 1
        # If still short (uneven family sizes), append remaining experiments in original order
        if len(preview_exps) < preview_count:
            for e in experiments:
                if e not in preview_exps:
                    preview_exps.append(e)
                    if len(preview_exps) >= preview_count:
                        break

    # Report how many preview configs per family were selected
    t_sel, j_sel = split_families(preview_exps)
    print(f"Preview selection: {len(preview_exps)} configs (TRM={len(t_sel)}, JEPA={len(j_sel)})")

    preview_outputs = run_experiments_parallel(
        exps=preview_exps,
        gen_dir=gen_dir,
        metrics_dir=metrics_dir,
        epochs=args.preview_epochs,
        subset=args.preview_subset,
        suffix="preview",
        parallel=args.parallel,
        gpu_list=gpu_list,
        max_procs=max_procs,
        force=args.force,
    )

    # -----------------------------
    # Phase B: Select best TRM + best JEPA
    # -----------------------------
    k = max(1, int(args.refine_k))
    chosen_trm, chosen_jepa = select_top_k_per_family(
        preview_outputs=preview_outputs,
        refine_metric=args.refine_metric,
        k_per_family=k,
    )

    chosen = chosen_trm + chosen_jepa
    if not chosen:
        print("No chosen experiments from preview; exiting.")
        return

    # If user requested generate-only, write the final override configs for the
    # chosen experiments (with full epochs) and exit without running training.
    if args.generate_only:
        print("--generate-only requested: writing override configs for chosen experiments and exiting.")
        for exp in chosen:
            overrides = deepcopy(exp["override"])
            overrides.setdefault("training", {})
            overrides["training"]["epochs"] = args.full_epochs
            # Respect a train_subset override passed on the CLI
            if args.train_subset is not None:
                overrides.setdefault("data", {})
                overrides["data"]["train_subset"] = args.train_subset
            base_name = f"{exp['name']}_full"
            path = write_override(base_name, overrides, gen_dir)
            print(f"  Wrote: {path}")
        print("Done generating configs.")
        return

    # -----------------------------
    # Phase C: Full training ONLY chosen
    # -----------------------------
    print(f"\n[Phase C] Full training: running {len(chosen)} chosen configs for {args.full_epochs} epochs\n")

    full_subset = args.train_subset  # can be None (meaning full data)
    full_outputs = run_experiments_parallel(
        exps=chosen,
        gen_dir=gen_dir,
        metrics_dir=metrics_dir,
        epochs=args.full_epochs,
        subset=full_subset,
        suffix="full",
        parallel=args.parallel,
        gpu_list=gpu_list,
        max_procs=max_procs,
        force=args.force,
    )

    # -----------------------------
    # Optional: Compare plots
    # -----------------------------
    full_csvs = []
    full_labels = []
    for exp, csv_path in full_outputs:
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            full_csvs.append(csv_path)
            full_labels.append(os.path.basename(csv_path).replace(".csv", ""))

    if full_csvs:
        out_refined = os.path.join("plots", "refined_full_training")
        os.makedirs(out_refined, exist_ok=True)
        compare_cmd = (
            ["python3", "scripts/compare_runs.py", "--csv"]
            + full_csvs
            + ["--labels"]
            + full_labels
            + ["--out", out_refined]
        )
        print("Creating refined full training plots:", " ".join(compare_cmd))
        subprocess.run(compare_cmd, check=True)
        print("Refined plots in", out_refined)

    print("\nAll done.\n")


if __name__ == "__main__":
    main()
