"""CIFAR-100 data loading with ImageNet-compatible transforms."""

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


# ImageNet normalization statistics
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


def get_transforms(img_size: int = 224, augment: bool = True):
    """Return train and test transform pipelines.

    Args:
        img_size: Target spatial resolution (resize + center-crop).
        augment:  Whether to apply random augmentations to the train split.

    Returns:
        Tuple of (train_transform, test_transform).
    """
    # Test / val: deterministic resize + center-crop + normalize
    test_transform = transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),   # proportional resize
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])

    if augment:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                   saturation=0.4, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])
    else:
        train_transform = test_transform

    return train_transform, test_transform


def get_dataloaders(
    data_dir: str = "./data",
    img_size: int = 224,
    batch_size: int = 128,
    num_workers: int = 4,
    augment: bool = True,
    pin_memory: bool = True,
) -> tuple:
    """Download CIFAR-100 and return (train_loader, test_loader, class_names).

    Args:
        data_dir:    Root directory for dataset storage.
        img_size:    Spatial resolution fed to the model (default 224 for ImageNet backbones).
        batch_size:  Mini-batch size for both loaders.
        num_workers: DataLoader worker processes.
        augment:     Apply random augmentations to training split.
        pin_memory:  Pin CPU memory for faster GPU transfer.

    Returns:
        (train_loader, test_loader, class_names) where class_names is a list of 100 strings.
    """
    train_tf, test_tf = get_transforms(img_size=img_size, augment=augment)

    train_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=True,  download=True, transform=train_tf
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=test_tf
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    class_names = train_dataset.classes  # list of 100 class name strings
    return train_loader, test_loader, class_names
