"""Train a single CNN experiment on CIFAR-100.

Usage:
    python scripts/train.py --config configs/resnet18_pretrained.yaml
    python scripts/train.py --config configs/resnet18_pretrained_cutmix.yaml --results results/
    python scripts/train.py --config configs/resnet18_scratch.yaml --device cpu
"""

import argparse
import os
import sys
import torch

# Allow running from the project root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    load_config, setup_logging, set_seed,
    count_parameters, get_model_size_mb, measure_inference_time,
    save_results, Timer,
)
from src.data    import get_dataloaders
from src.models  import build_model
from src.trainer import CNNTrainer


BASE_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs", "base.yaml",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN on CIFAR-100")
    parser.add_argument("--config",  required=True,           help="Path to experiment YAML config")
    parser.add_argument("--base",    default=BASE_CONFIG,     help="Path to base YAML config")
    parser.add_argument("--results", default="results",       help="Directory to save results")
    parser.add_argument("--data",    default="./data",        help="CIFAR-100 data root directory")
    parser.add_argument("--device",  default=None,            help="Device: cuda / cpu (auto-detect if omitted)")
    parser.add_argument("--workers", default=4,  type=int,    help="DataLoader worker count")
    return parser.parse_args()


def main():
    args   = parse_args()
    config = load_config(args.config, args.base)
    name   = config["experiment_name"]

    logger = setup_logging(name, args.results)
    logger.info("Experiment: %s", name)
    logger.info("Config: %s", config)

    # ── Seed ──────────────────────────────────────────────────────────
    seed = config.get("seed", 42)
    set_seed(seed)
    logger.info("Seed: %d", seed)

    # ── Device ────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Data ──────────────────────────────────────────────────────────
    img_size   = config.get("img_size", 224)
    batch_size = config.get("batch_size", 128)

    logger.info("Loading CIFAR-100 (img_size=%d, batch_size=%d) ...", img_size, batch_size)
    train_loader, test_loader, _ = get_dataloaders(
        data_dir    = args.data,
        img_size    = img_size,
        batch_size  = batch_size,
        num_workers = args.workers,
    )

    # ── Model ─────────────────────────────────────────────────────────
    model   = build_model(config)
    params  = count_parameters(model)
    size_mb = get_model_size_mb(model)
    logger.info(
        "Model params: %.2fM total | %.2fM trainable | Size: %.1f MB",
        params["total_M"], params["trainable_M"], size_mb,
    )

    # ── Training ──────────────────────────────────────────────────────
    trainer  = CNNTrainer(model, config, device)
    log_path = os.path.join(args.results, f"{name}_history.csv")

    logger.info("Starting training for %d epochs ...", config.get("epochs", 50))
    with Timer() as total_t:
        history = trainer.train(train_loader, test_loader, log_path=log_path)

    logger.info("Training complete in %.1f s", total_t.elapsed)

    # ── Final evaluation ──────────────────────────────────────────────
    best_acc = max(r["test_acc"] for r in history)
    final_acc = history[-1]["test_acc"]
    logger.info("Best test accuracy : %.4f (%.2f%%)", best_acc, best_acc * 100)
    logger.info("Final test accuracy: %.4f (%.2f%%)", final_acc, final_acc * 100)

    # ── Inference speed ───────────────────────────────────────────────
    inf_device = "cuda" if device.type == "cuda" else "cpu"
    inf_time   = measure_inference_time(
        model, input_size=(1, 3, img_size, img_size), device=inf_device
    )
    logger.info("Inference time: %.2f ms/sample (%s)", inf_time, inf_device)

    # ── Save results ──────────────────────────────────────────────────
    results = {
        "experiment_name":      name,
        "architecture":         config["strategy"]["architecture"],
        "pretrained":           config["strategy"].get("pretrained", False),
        "cutmix":               config["strategy"].get("cutmix", False),
        "best_test_acc":        round(best_acc, 4),
        "final_test_acc":       round(final_acc, 4),
        "total_params_M":       params["total_M"],
        "trainable_params_M":   params["trainable_M"],
        "model_size_mb":        size_mb,
        "inference_ms":         inf_time,
        "total_training_s":     round(total_t.elapsed, 1),
        "epochs":               config.get("epochs", 50),
        "config":               config,
    }

    out_path = os.path.join(args.results, f"{name}.json")
    save_results(results, out_path)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
