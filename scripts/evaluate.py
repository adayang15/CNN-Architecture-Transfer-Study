"""Aggregate all experiment JSON results and generate comparison figures.

Usage:
    python scripts/evaluate.py --results results/
    python scripts/evaluate.py --results results/ --output results/figures/

Outputs:
    results/figures/accuracy_comparison.png   — Bar chart of test accuracy per experiment
    results/figures/params_vs_accuracy.png    — Scatter: trainable params vs best accuracy
    results/figures/model_size_comparison.png — Bar chart of model size in MB
    A summary table is printed to stdout.
"""

import argparse
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for Colab / headless)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_results(results_dir: str) -> list:
    """Load all per-experiment JSON files from results_dir."""
    pattern = os.path.join(results_dir, "*.json")
    paths   = sorted(glob.glob(pattern))
    records = []
    for p in paths:
        with open(p) as f:
            records.append(json.load(f))
    return records


def print_table(records: list):
    """Print a markdown-style summary table."""
    header = (
        f"{'Experiment':<40} {'Pretrained':>10} {'CutMix':>7} "
        f"{'Best Acc':>9} {'Params(M)':>10} {'Size(MB)':>9} {'Inf(ms)':>8}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in records:
        print(
            f"{r['experiment_name']:<40} {str(r.get('pretrained','?')):>10} "
            f"{str(r.get('cutmix','?')):>7} "
            f"{r.get('best_test_acc', 0)*100:>8.2f}% "
            f"{r.get('total_params_M', 0):>10.2f} "
            f"{r.get('model_size_mb', 0):>9.1f} "
            f"{r.get('inference_ms', 0):>8.2f}"
        )
    print(sep)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_accuracy_comparison(records: list, out_dir: str):
    names = [r["experiment_name"] for r in records]
    accs  = [r.get("best_test_acc", 0) * 100 for r in records]

    # Short labels for readability
    short = [n.replace("resnet18", "R18").replace("resnet50", "R50")
              .replace("mobilenetv2", "MV2").replace("_pretrained", "+PT")
              .replace("_scratch", "-S").replace("_cutmix", "+CM")
             for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(short, accs, color="#4C72B0", edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=9)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12)
    ax.set_title("CIFAR-100 Test Accuracy by Experiment", fontsize=14)
    ax.set_ylim(0, min(100, max(accs) + 10))
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=25, ha="right", fontsize=9)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "accuracy_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_params_vs_accuracy(records: list, out_dir: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in records:
        x = r.get("trainable_params_M", 0)
        y = r.get("best_test_acc", 0) * 100
        ax.scatter(x, y, s=80, zorder=5)
        ax.annotate(
            r["experiment_name"].replace("_pretrained", "").replace("_scratch", ""),
            (x, y), textcoords="offset points", xytext=(5, 3), fontsize=7,
        )
    ax.set_xlabel("Trainable Parameters (M)", fontsize=12)
    ax.set_ylabel("Best Test Accuracy (%)", fontsize=12)
    ax.set_title("Parameter Efficiency: Trainable Params vs Accuracy", fontsize=13)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "params_vs_accuracy.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_model_size(records: list, out_dir: str):
    names  = [r["experiment_name"] for r in records]
    sizes  = [r.get("model_size_mb", 0) for r in records]
    short  = [n.replace("resnet18", "R18").replace("resnet50", "R50")
               .replace("mobilenetv2", "MV2").replace("_pretrained", "+PT")
               .replace("_scratch", "-S").replace("_cutmix", "+CM")
              for n in names]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(short, sizes, color="#55A868", edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, fmt="%.1f MB", padding=3, fontsize=9)
    ax.set_ylabel("Model Size (MB)", fontsize=12)
    ax.set_title("Model Size Comparison", fontsize=14)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=25, ha="right", fontsize=9)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "model_size_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate and compare CNN experiments")
    parser.add_argument("--results", default="results",          help="Directory with JSON result files")
    parser.add_argument("--output",  default="results/figures",  help="Directory to save figures")
    return parser.parse_args()


def main():
    args    = parse_args()
    records = load_results(args.results)

    if not records:
        print(f"No JSON result files found in '{args.results}'. Run scripts/train.py first.")
        return

    print(f"\nLoaded {len(records)} experiment(s).\n")
    print_table(records)

    plot_accuracy_comparison(records, args.output)
    plot_params_vs_accuracy(records, args.output)
    plot_model_size(records, args.output)

    print("\nAll figures saved.")


if __name__ == "__main__":
    main()
