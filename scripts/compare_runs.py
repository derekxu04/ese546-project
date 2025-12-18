import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def compare(csv_paths, labels, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    def is_jepa(label: str) -> bool:
        return "jepa" in label.lower()

    def style(is_jepa_run):
        return {
            "linestyle": "--" if is_jepa_run else "-",
            "marker": "o" if is_jepa_run else None,
            "linewidth": 2,
        }

    def get_column(df, *candidates):
        """Return the first candidate column that exists in df, or None."""
        for col in candidates:
            if col in df.columns:
                return col
        return None

    runs = []
    for csv_path, label in zip(csv_paths, labels):
        df = pd.read_csv(csv_path)
        runs.append((df, label, is_jepa(label)))

    def get_train_token_acc(df):
        return df.get("Train Token Acc", df.get("Token Accuracy"))

    print("\n[Sanity check] Train vs Test Token Accuracy difference:")
    for df, label, _ in runs:
        train_acc = get_train_token_acc(df)
        test_acc = df.get("Test Token Acc")

        if train_acc is not None and test_acc is not None:
            diff = (train_acc - test_acc).abs().max()
            print(f"  {label}: max |train-test| token acc = {diff:.6f}")
        else:
            print(f"  {label}: token accuracy columns missing")


    # Plot 1: Train CE Loss
    plt.figure(figsize=(8, 5))
    for df, label, jepa_flag in runs:
        train_ce_col = get_column(df, "Train CE Loss", "CE Loss")
        if train_ce_col:
            plt.plot(df['Epoch'], df[train_ce_col], label=label)
    plt.xlabel('Epoch')
    plt.ylabel('Train CE Loss')
    plt.title('Train CE Loss Comparison')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'train_ce_loss.png'))
    plt.close()

    # Plot 2: Test CE Loss (TRM vs JEPA)
    plt.figure(figsize=(8, 5))
    for df, label, jepa_flag in runs:
        test_ce_col = get_column(df, "Test CE Loss", "Test Loss")
        if test_ce_col and df[test_ce_col].notna().any():
            mask = df[test_ce_col].notna()
            plt.plot(
                df.loc[mask, "Epoch"],
                df.loc[mask, test_ce_col],
                label=label,
                **style(jepa_flag),
            )
    plt.xlabel("Epoch")
    plt.ylabel("Test CE Loss")
    plt.title("Test CE Loss (TRM vs TRM+JEPA)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "compare_test_ce_loss.png"))
    plt.close()

    # Plot 3: Train Token Accuracy (TRM vs JEPA)
    plt.figure(figsize=(8, 5))
    for df, label, jepa_flag in runs:
        train_acc_col = get_column(df, "Train Token Acc", "Token Accuracy")
        if train_acc_col:
            plt.plot(
                df["Epoch"],
                df[train_acc_col],
                label=label,
                **style(jepa_flag),
            )
    plt.xlabel("Epoch")
    plt.ylabel("Train Token Accuracy")
    plt.title("Train Token Accuracy (TRM vs TRM+JEPA)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "compare_train_token_accuracy.png"))
    plt.close()

    # Plot 3b: Train Token Accuracy (Zoomed)
    plt.figure(figsize=(8, 5))
    for df, label, jepa_flag in runs:
        train_acc_col = get_column(df, "Train Token Acc", "Token Accuracy")
        if train_acc_col:
            plt.plot(
                df["Epoch"],
                df[train_acc_col],
                label=label,
                **style(jepa_flag),
            )

    plt.xlabel("Epoch")
    plt.ylabel("Train Token Accuracy")
    plt.title("Train Token Accuracy (Zoomed)")
    plt.ylim(0.95, 1.001)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "compare_train_token_accuracy_zoomed.png"))
    plt.close()


    # Plot 4: Test Token Accuracy (TRM vs JEPA)
    plt.figure(figsize=(8, 5))
    for df, label, jepa_flag in runs:
        test_acc_col = get_column(df, "Test Token Acc")
        if test_acc_col and df[test_acc_col].notna().any():
            mask = df[test_acc_col].notna()
            plt.plot(
                df.loc[mask, "Epoch"],
                df.loc[mask, test_acc_col],
                label=label,
                **style(jepa_flag),
            )
    plt.xlabel("Epoch")
    plt.ylabel("Test Token Accuracy")
    plt.title("Test Token Accuracy (TRM vs TRM+JEPA)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "compare_test_token_accuracy.png"))
    plt.close()

    # Plot 4b: Test Token Accuracy (Zoomed)
    plt.figure(figsize=(8, 5))
    for df, label, jepa_flag in runs:
        test_acc_col = get_column(df, "Test Token Acc")
        if test_acc_col and df[test_acc_col].notna().any():
            mask = df[test_acc_col].notna()
            plt.plot(
                df.loc[mask, "Epoch"],
                df.loc[mask, test_acc_col],
                label=label,
                **style(jepa_flag),
            )

    plt.xlabel("Epoch")
    plt.ylabel("Test Token Accuracy")
    plt.title("Test Token Accuracy (Zoomed)")
    plt.ylim(0.95, 1.001)   # 👈 KEY CHANGE
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "compare_test_token_accuracy_zoomed.png"))
    plt.close()


    # Plot 5: Test Puzzle Accuracy (TRM vs JEPA)
    plt.figure(figsize=(8, 5))
    for df, label, jepa_flag in runs:
        puzzle_acc_col = get_column(df, "Test Puzzle Acc")
        if puzzle_acc_col and df[puzzle_acc_col].notna().any():
            mask = df[puzzle_acc_col].notna()
            plt.plot(
                df.loc[mask, "Epoch"],
                df.loc[mask, puzzle_acc_col],
                label=label,
                **style(jepa_flag),
            )
    plt.xlabel("Epoch")
    plt.ylabel("Puzzle Accuracy")
    plt.title("Test Puzzle Accuracy (TRM vs TRM+JEPA)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "compare_test_puzzle_accuracy.png"))
    plt.close()

    print("Generated comparison plots in", out_dir)

    # Plot 6: Train − Test Token Accuracy Gap
    plt.figure(figsize=(8, 5))
    for df, label, jepa_flag in runs:
        if "Train Token Acc" in df.columns and "Test Token Acc" in df.columns:
            gap = df["Train Token Acc"] - df["Test Token Acc"]
            plt.plot(
                df["Epoch"],
                gap,
                label=label,
                **style(jepa_flag),
            )

    plt.xlabel("Epoch")
    plt.ylabel("Train - Test Token Accuracy")
    plt.title("Generalization Gap (Token Accuracy)")
    plt.ylim(-0.002, 0.002)

    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "token_acc_gap.png"))
    plt.close()

    # Plot 7: Token Error Rate (log scale)
    plt.figure(figsize=(8, 5))
    for df, label, jepa_flag in runs:
        if "Train Token Acc" in df.columns:
            error = 1.0 - df["Train Token Acc"]
            plt.plot(
                df["Epoch"],
                error,
                label=label,
                **style(jepa_flag),
            )

    plt.xlabel("Epoch")
    plt.ylabel("Train Token Error Rate")
    plt.title("Train Token Error Rate (log scale)")
    plt.yscale("log")
    plt.grid(alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "token_error_rate_log.png"))
    plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", nargs="+", required=True, help="Paths to metrics CSVs")
    parser.add_argument("--labels", nargs="+", required=True, help="Labels for runs")
    parser.add_argument("--out", default="plots", help="Output directory for comparison plots")
    args = parser.parse_args()

    compare(args.csv, args.labels, args.out)

