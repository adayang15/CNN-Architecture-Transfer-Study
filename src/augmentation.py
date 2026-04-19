"""CutMix data augmentation for image classification.

Reference:
    Yun et al., "CutMix: Training Strategy that Makes Use of Sample Mixing"
    ICCV 2019 — https://arxiv.org/abs/1905.04899

Usage:
    images, labels_a, labels_b, lam = cutmix_data(images, labels, alpha=1.0)
    loss = cutmix_criterion(criterion, outputs, labels_a, labels_b, lam)
"""

import numpy as np
import torch


def rand_bbox(img_size: tuple, lam: float) -> tuple:
    """Compute a random bounding box whose area fraction equals (1 - lam).

    Args:
        img_size: (H, W) of the image.
        lam:      Lambda value sampled from Beta(alpha, alpha).

    Returns:
        (x1, y1, x2, y2) integer pixel coordinates (top-left, bottom-right).
    """
    H, W = img_size
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)

    # Uniformly sample patch centre
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    return x1, y1, x2, y2


def cutmix_data(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 1.0,
) -> tuple:
    """Apply CutMix to a batch of images and labels.

    A rectangular patch from a randomly shuffled image replaces the
    corresponding region of each image in the batch.

    Args:
        images: Float tensor of shape (B, C, H, W).
        labels: Long tensor of shape (B,) with class indices.
        alpha:  Concentration parameter for the Beta distribution.
                alpha=1.0 gives Uniform(0, 1); larger values concentrate near 0.5.

    Returns:
        mixed_images: Tensor (B, C, H, W) after mixing.
        labels_a:     Original labels (B,).
        labels_b:     Labels of the shuffled images (B,).
        lam:          Actual area lambda after clipping (scalar float).
    """
    lam = np.random.beta(alpha, alpha)

    B, C, H, W = images.size()
    rand_index = torch.randperm(B, device=images.device)

    labels_a = labels
    labels_b = labels[rand_index]

    x1, y1, x2, y2 = rand_bbox((H, W), lam)

    mixed_images = images.clone()
    mixed_images[:, :, y1:y2, x1:x2] = images[rand_index, :, y1:y2, x1:x2]

    # Recompute lambda from the actual box area (may differ due to clipping)
    lam = 1.0 - (x2 - x1) * (y2 - y1) / (W * H)

    return mixed_images, labels_a, labels_b, lam


def cutmix_criterion(
    criterion,
    outputs: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Compute the CutMix loss as a convex combination of two cross-entropy terms.

    Args:
        criterion: A loss function callable (e.g. nn.CrossEntropyLoss()).
        outputs:   Model logits (B, num_classes).
        labels_a:  Original labels (B,).
        labels_b:  Shuffled labels (B,).
        lam:       Mixing coefficient (area fraction kept from original image).

    Returns:
        Scalar loss tensor.
    """
    return lam * criterion(outputs, labels_a) + (1.0 - lam) * criterion(outputs, labels_b)
