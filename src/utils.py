"""Shared utilities: config, logging, seeding, model profiling, and result saving."""

import os
import json
import time
import random
import logging
import csv
import yaml
import numpy as np
import torch


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str, base_path: str = None) -> dict:
    """Load a YAML config and merge with base.yaml if provided."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if base_path and os.path.exists(base_path):
        with open(base_path) as f:
            base = yaml.safe_load(f)
        config = deep_merge(base, config)
    return config


def setup_logging(experiment_name: str, results_dir: str) -> logging.Logger:
    """Set up console + file logging."""
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, f"{experiment_name}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
    )
    return logging.getLogger(experiment_name)


def set_seed(seed: int = 42):
    """Set random seed across random, numpy, torch, and cuda."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Timer:
    """Context manager for timing code blocks."""
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, *args):
        self.elapsed = round(time.time() - self.start, 2)


def count_parameters(model) -> dict:
    """Return total, trainable, and frozen parameter counts."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_M":     round(total / 1e6, 2),
        "trainable_M": round(trainable / 1e6, 2),
    }


def get_model_size_mb(model) -> float:
    """Calculate model size in MB from parameter and buffer memory."""
    param_size  = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return round((param_size + buffer_size) / 1024 / 1024, 2)


def measure_inference_time(model, input_size=(1, 3, 224, 224), device="cuda", n_runs=100) -> float:
    """Measure average inference time in ms over n_runs forward passes."""
    model.eval()
    dummy = torch.randn(input_size).to(device)

    with torch.no_grad():
        for _ in range(10):           # warmup
            model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()

    return round((time.time() - start) / n_runs * 1000, 2)


def save_results(results: dict, path: str):
    """Save results dict to JSON."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def save_training_log(history: list, path: str):
    """Save per-epoch training history to CSV."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not history:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
