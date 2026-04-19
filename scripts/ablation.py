"""Ablation study: compare all 5 configurations across data fractions.

This script subsamples the training set to 10 %, 25 %, and 100 % and trains
each configuration for a reduced number of epochs.  Results are saved as JSON
and a data-efficiency plot is generated.

Usage:
    python scripts/ablation.py --results results/ablation/ --epochs 10
    python scripts/ablation.py --epochs 5 --fractions 0.1 0.5 1.0
"""

import argparse
import os
import sys
import json
import glob

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
import torchvision

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils   import load_config, setup_logging, set_seed, save_results
from src.data    import get_transforms
from src.models  import build_model
from src.trainer import CNNTrainer


BASE_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs", "base.yaml",
)

ALL_CONFIGS = [
    "configs/resnet18_scratch.yaml",
    "configs/resnet18_pretrained.yaml",
    "configs/resnet50_pretrained.yaml",
    "configs/mobilenetv2_pretrained.yaml",
    "configs/resnet18_pretrained_cutmix.yaml",
]


def subsample_dataset(dataset, fraction: float, seed: int = 42):
    """Return a random Subset of `dataset` with `fraction` of samples."""
    import random
    n = len(dataset)
    k = max(1, int(n * fraction))
    rng = random.Random(seed)
    indices = rng.sample(range(n), k)
    return Subset(dataset, indices)


def run_ablation(
    config_path: str,
    fraction: float,
    epochs_override: int,
    device: torch.device,
    data_dir: str,
    results_dir: str,
    num_workers: int,
) -> dict:
    config = load_config(config_path, BASE_CONFIG)
    config["epochs"] = epochs_override      # override for speed

    name      = config["experiment_name"]
    img_size  = config.get("img_size", 224)
    batch_sz  = config.get("batch_size", 128)

    set_seed(config.get("seed", 42))

    train_tf, test_tf = get_transforms(img_size=img_size, augment=True)
    full_train = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=train_tf
    )
    test_ds = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=test_tf
    )

    sub_train   = subsample_dataset(full_train, fraction, seed=config.get("seed", 42))
    train_loader = DataLoader(sub_train, batch_size=batch_sz, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_ds,   batch_size=batch_sz, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    model   = build_model(config)
    trainer = CNNTrainer(model, config, device)
    history = trainer.train(train_loader, test_loader)

    best_acc = max(r["test_acc"] for r in history)
    result   = {
        "experiment_name": name,
        "fraction":        fraction,
        "n_train":         len(sub_train),
        "epochs":          epochs_override,
        "best_test_acc":   round(best_acc, 4),
    }
    key  = f"{name}_frac{int(fraction*100):03d}"
    path = os.path.join(results_dir, f"{key}.json")
    save_results(result, path)
    return result


def plot_data_efficiency(all_results: list, out_dir: str):
    """Line plot: x=data fraction, y=best accuracy, one line per experiment."""
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in all_results:
        grouped[r["experiment_name"]].append((r["fraction"], r["best_test_acc"]))

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, points in sorted(grouped.items()):
        points.sort(key=lambda t: t[0])
        xs = [p[0] * 100 for p in points]
        ys = [p[1] * 100 for p in points]
        short = (name.replace("resnet18", "R18").replace("resnet50", "R50")
                     .replace("mobilenetv2", "MV2").replace("_pretrained", "+PT")
                     .replace("_scratch", "-S").replace("_cutmix", "+CM"))
        ax.plot(xs, ys, marker="o", label=short)

    ax.set_xlabel("Training Data Fraction (%)", fontsize=12)
    ax.set_ylabel("Best Test Accuracy (%)", fontsize=12)
    ax.set_title("Data Efficiency: Accuracy vs Training Set Size", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "data_efficiency.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Data-fraction ablation study")
    parser.add_argument("--results",   default="results/ablation", help="Output directory")
    parser.add_argument("--data",      default="./data",           help="CIFAR-100 data root")
    parser.add_argument("--epochs",    default=10, type=int,       help="Epochs per run")
    parser.add_argument("--fractions", nargs="+", type=float,
                        default=[0.1, 0.25, 1.0],                  help="Data fractions to test")
    parser.add_argument("--device",    default=None,               help="cuda / cpu")
    parser.add_argument("--workers",   default=4, type=int,        help="DataLoader workers")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.results, exist_ok=True)
    logger = setup_logging("ablation", args.results)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info("Device: %s", device)

    all_results = []
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    for cfg_rel in ALL_CONFIGS:
        cfg_path = os.path.join(project_root, cfg_rel)
        if not os.path.exists(cfg_path):
            logger.warning("Config not found, skipping: %s", cfg_path)
            continue
        for frac in sorted(args.fractions):
            logger.info("Running %s @ %.0f%% data ...", cfg_rel, frac * 100)
            r = run_ablation(
                config_path     = cfg_path,
                fraction        = frac,
                epochs_override = args.epochs,
                device          = device,
                data_dir        = args.data,
                results_dir     = args.results,
                num_workers     = args.workers,
            )
            all_results.append(r)
            logger.info(
                "  → best_acc=%.2f%% (n_train=%d)", r["best_test_acc"] * 100, r["n_train"]
            )

    # Also load any pre-existing ablation JSONs
    for p in glob.glob(os.path.join(args.results, "*.json")):
        with open(p) as f:
            try:
                d = json.load(f)
                if "fraction" in d and d not in all_results:
                    all_results.append(d)
            except Exception:
                pass

    fig_dir = os.path.join(args.results, "figures")
    plot_data_efficiency(all_results, fig_dir)
    logger.info("Ablation study complete.")


if __name__ == "__main__":
    main()
