"""CNN backbone factory for CIFAR-100 transfer learning experiments.

Supported architectures:
    - resnet18    (torchvision ResNet-18)
    - resnet50    (torchvision ResNet-50)
    - mobilenetv2 (torchvision MobileNetV2)

All models replace the original 1000-class head with a new linear layer
for `num_classes` outputs.  When `pretrained=True` the backbone is
initialised with ImageNet-1k weights and the new head is randomly initialised.
"""

import logging
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)


def _replace_head(model, arch: str, num_classes: int) -> nn.Module:
    """Replace the final classification layer with a new Linear(*, num_classes)."""
    if arch in ("resnet18", "resnet50"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif arch == "mobilenetv2":
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch!r}")
    return model


def build_model(config: dict) -> nn.Module:
    """Construct a CNN model from a strategy config dict.

    Expected keys inside config['strategy']:
        architecture (str):  One of resnet18 / resnet50 / mobilenetv2.
        pretrained   (bool): Load ImageNet weights if True.
        num_classes  (int):  Number of output classes (default 100).

    Args:
        config: Merged experiment config dict (base + experiment YAML).

    Returns:
        nn.Module ready for training (parameters are NOT moved to device here).
    """
    strategy    = config.get("strategy", {})
    arch        = strategy["architecture"].lower()
    pretrained  = strategy.get("pretrained", False)
    num_classes = config.get("num_classes", 100)

    weights_arg = "DEFAULT" if pretrained else None

    if arch == "resnet18":
        model = models.resnet18(weights=weights_arg)
    elif arch == "resnet50":
        model = models.resnet50(weights=weights_arg)
    elif arch == "mobilenetv2":
        model = models.mobilenet_v2(weights=weights_arg)
    else:
        raise ValueError(
            f"Unsupported architecture: {arch!r}. "
            "Choose from: resnet18, resnet50, mobilenetv2."
        )

    model = _replace_head(model, arch, num_classes)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(
        "Built %s | pretrained=%s | params: %.2fM total, %.2fM trainable",
        arch, pretrained, total / 1e6, trainable / 1e6,
    )

    return model
