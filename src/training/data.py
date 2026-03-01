"""Data Loading Module

Handles dataset loading for CIFAR-10/100, Tiny-ImageNet, CUB-200, etc.
"""

from __future__ import annotations

from typing import Optional, Tuple, Any
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder


def get_transforms(
    dataset: str,
    split: str = 'train',
    image_size: int = 224,
    augment: bool = True,
) -> transforms.Compose:
    """Get transforms for a dataset

    Args:
        dataset: Dataset name
        split: 'train' or 'val'
        image_size: Target image size
        augment: Whether to apply data augmentation

    Returns:
        Composed transforms
    """
    if split == 'train' and augment:
        # Training transforms with augmentation
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        # Validation transforms (no augmentation)
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.143)),  # Resize to slightly larger
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def create_dataset(
    name: str,
    split: str = 'train',
    transform: Optional[transforms.Compose] = None,
    root: str = './data',
) -> Dataset:
    """Create a dataset

    Args:
        name: Dataset name (cifar10, cifar100, tiny-imagenet, cub200, mnist)
        split: 'train' or 'val'
        transform: Transforms to apply
        root: Root directory for datasets

    Returns:
        Dataset instance
    """
    root = Path(root)

    name_lower = name.lower()

    if name_lower == 'cifar10':
        # CIFAR-10
        train = (split == 'train')
        return CIFAR10(
            root=str(root / 'cifar-10-batches-py' if (root / 'cifar-10-batches-py').exists() else root),
            train=train,
            download=False,
            transform=transform,
        )

    elif name_lower == 'cifar100':
        # CIFAR-100
        train = (split == 'train')
        return CIFAR100(
            root=str(root),
            train=train,
            download=False,
            transform=transform,
        )

    elif name_lower == 'tiny-imagenet':
        # Tiny-ImageNet
        # Structure: root/tiny-imagenet-200/train/ or root/tiny-imagenet-200/val/
        tiny_imagenet_root = root / 'tiny-imagenet-200'

        if split == 'train':
            data_path = tiny_imagenet_root / 'train'
        else:
            data_path = tiny_imagenet_root / 'val'

        return ImageFolder(root=str(data_path), transform=transform)

    elif name_lower == 'cub200':
        # CUB-200-2011
        cub_root = root / 'CUB_200_2011'

        if split == 'train':
            data_path = cub_root / 'images'  # Will be filtered
        else:
            data_path = cub_root / 'images'

        return ImageFolder(root=str(data_path), transform=transform)

    elif name_lower == 'mnist':
        from torchvision.datasets import MNIST
        train = (split == 'train')
        return MNIST(
            root=str(root),
            train=train,
            download=False,
            transform=transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]) if transform is None else transform,
        )

    else:
        raise ValueError(f"Unknown dataset: {name}. Supported: cifar10, cifar100, tiny-imagenet, cub200, mnist")


__all__ = [
    'create_dataset',
    'get_transforms',
]
