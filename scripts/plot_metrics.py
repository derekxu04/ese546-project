import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_metrics(csv_path: str, out_dir: str):
    df = pd.read_csv(csv_path)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loaded CSV with columns: {df.columns.tolist()}")

    # -----------------------------
    # Sanity checks for test metrics
    # -----------------------------
    if "Test Loss" in df.columns:
        n_test = df["Test Loss"].notna().sum()
        print(f"Test Loss points: {n_test}")
        if n_test < 5:
            print(
                "WARNING: very few test-loss points found. "
                "This is expected if eval_interval > 1."
            )

    if "Test Token Acc" in df.columns:
        n_test_acc = df["Test Token Acc"].notna().sum()
        print(f"Test Token Acc points: {n_test_acc}")

    # ============================================================
    # Plot 1: CE-only loss (TRAIN vs TEST)  <-- MAIN COMPARISON
    # ============================================================
    if "CE Loss" in df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(
            df["Epoch"],
            df["CE Loss"],
            label="Train CE Loss",
            linewidth=2,
        )

        if "Test Loss" in df.columns and df["Test Loss"].notna().any():
            mask = df["Test Loss"].notna()
            plt.plot(
                df.loc[mask, "Epoch"],
                df.loc[mask, "Test Loss"],
                label="Test CE Loss",
                linewidth=2,
                linestyle="--",
                marker="o",
            )

        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("Train vs Test CE Loss")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "ce_loss.png"), dpi=120)
        plt.close()
    else:
        print("Skipping CE-loss plot (column 'CE Loss' not found).")

    # ============================================================
    # Plot 2: Total training objective (TRAIN ONLY)
    # ============================================================
    if "Train Loss" in df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(
            df["Epoch"],
            df["Train Loss"],
            label="Train Total Loss",
            linewidth=2,
        )
        plt.xlabel("Epoch")
        plt.ylabel("Total Training Loss")
        plt.title("Training Objective (CE + Halt + JEPA)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "train_total_loss.png"), dpi=120)
        plt.close()
    else:
        print("Skipping total train-loss plot (column 'Train Loss' not found).")

    # ============================================================
    # Plot 3: Token accuracy (TRAIN vs TEST)
    # ============================================================
    if "Token Accuracy" in df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(
            df["Epoch"],
            df["Token Accuracy"],
            label="Train Token Accuracy",
            linewidth=2,
        )

        if "Test Token Acc" in df.columns and df["Test Token Acc"].notna().any():
            mask = df["Test Token Acc"].notna()
            plt.plot(
                df.loc[mask, "Epoch"],
                df.loc[mask, "Test Token Acc"],
                label="Test Token Accuracy",
                linewidth=2,
                linestyle="--",
                marker="o",
            )

        plt.xlabel("Epoch")
        plt.ylabel("Token Accuracy")
        plt.title("Token Accuracy vs Epoch")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "token_accuracy.png"), dpi=120)
        plt.close()
    else:
        print("Skipping token-accuracy plot (column 'Token Accuracy' not found).")

    # ============================================================
    # Plot 4: Puzzle accuracy (TEST ONLY)
    # ============================================================
    if "Test Puzzle Acc" in df.columns and df["Test Puzzle Acc"].notna().any():
        plt.figure(figsize=(10, 6))
        mask = df["Test Puzzle Acc"].notna()
        plt.plot(
            df.loc[mask, "Epoch"],
            df.loc[mask, "Test Puzzle Acc"],
            label="Test Puzzle Accuracy",
            linewidth=2,
            marker="o",
        )

        plt.xlabel("Epoch")
        plt.ylabel("Puzzle Accuracy")
        plt.title("Test Puzzle Accuracy vs Epoch")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "test_puzzle_accuracy.png"), dpi=120)
        plt.close()
    else:
        print("Skipping puzzle-accuracy plot (no test puzzle accuracy found).")

    print(f"Plots written to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to CSV metrics file",
    )
    parser.add_argument(
        "--out",
        default="plots",
        help="Output directory for plots",
    )
    args = parser.parse_args()

    plot_metrics(args.csv, args.out)
