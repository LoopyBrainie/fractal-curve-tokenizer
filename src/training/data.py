"""Data Loading Module

Handles dataset loading for CIFAR-10/100, Tiny-ImageNet, CUB-200, etc.
Supports both torchvision datasets and Hugging Face datasets library.
"""

from __future__ import annotations

from typing import Optional, Tuple, Any, Union, Dict
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder

# Optional HF datasets import
try:
    from datasets import load_dataset, Dataset as HFDataset, IterableDataset as HFIterableDataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    HFDataset = None
    HFIterableDataset = None


class HFDatasetWrapper(Dataset):
    """Wrapper for regular (non-streaming) Hugging Face datasets with on-the-fly transforms.

    Uses HF's with_transform for efficient batch-level transforms.
    """

    def __init__(
        self,
        hf_dataset: HFDataset,
        image_size: int = 64,
        augment: bool = True,
    ):
        """Initialize HF Dataset wrapper.

        Args:
            hf_dataset: Hugging Face dataset (non-streaming)
            image_size: Target image size
            augment: Whether to apply data augmentation
        """
        self._dataset = hf_dataset
        self.image_size = image_size
        self.augment = augment

        # Build transform pipeline
        if augment:
            self._transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self._transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        self._dataset = hf_dataset.with_transform(self._collate_fn)

    def _collate_fn(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Collate function for batch processing.

        Args:
            batch: Dict with 'img' (list of PIL Images) and 'label' (list of ints)

        Returns:
            Tuple of (pixel_values tensor [B, C, H, W], labels tensor [B])
        """
        pixel_values = torch.stack([
            self._transform(img.convert("RGB")) for img in batch["img"]
        ])
        labels = torch.tensor(batch["label"], dtype=torch.long)
        return (pixel_values, labels)

    def __len__(self) -> int:
        """Return length for DataLoader compatibility."""
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single item from the dataset."""
        return self._dataset[idx]


class HFStreamingDatasetWrapper(IterableDataset):
    """Wrapper for streaming Hugging Face IterableDatasets with on-the-fly transforms.

    Transforms are applied during iteration since streaming datasets don't support
    random access indexing.
    """

    def __init__(
        self,
        hf_dataset: HFIterableDataset,
        image_size: int = 64,
        augment: bool = True,
    ):
        """Initialize streaming HF Dataset wrapper.

        Args:
            hf_dataset: Hugging Face IterableDataset (streaming)
            image_size: Target image size
            augment: Whether to apply data augmentation
        """
        self._dataset = hf_dataset
        self.image_size = image_size
        self.augment = augment

        # Build transform pipeline
        if augment:
            self._transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self._transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def _apply_transform(self, item: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply transform to a single item.

        Args:
            item: Dict with 'img' (PIL Image) and 'label' (int)

        Returns:
            Tuple of (pixel_values tensor [C, H, W], label tensor)
        """
        pixel_values = self._transform(item["img"].convert("RGB"))
        label = torch.tensor(item["label"], dtype=torch.long)
        return (pixel_values, label)

    def __iter__(self):
        """Iterate over the dataset with transforms applied."""
        for item in self._dataset:
            yield self._apply_transform(item)

    def __len__(self) -> int:
        """Return estimated length for DataLoader compatibility.

        Streaming datasets don't have a finite length, so we return a large
        estimated value. The DataLoader will iterate until StopIteration is
        raised by the underlying iterator.
        """
        # Return estimated size based on typical CIFAR-10/TinyImageNet sizes
        # This is used by DataLoader to calculate num_batches
        return 50000  # Approximate, will stop when iterator exhausted


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


# Dataset metadata registry: image_size, num_classes, and optional HF path
# hf_path: the Hugging Face dataset path to use when --use-hf-dataset is set
DATASET_INFO = {
    # Local torchvision datasets (with optional HF path)
    'cifar10': {'image_size': 32, 'num_classes': 10, 'hf_path': 'uoft-cs/cifar10'},
    'cifar100': {'image_size': 32, 'num_classes': 100, 'hf_path': 'uoft-cs/cifar100'},
    'tiny-imagenet': {'image_size': 64, 'num_classes': 200, 'hf_path': 'zh-plus/tiny-imagenet'},
    'cub200': {'image_size': 224, 'num_classes': 200, 'hf_path': None},
    'mnist': {'image_size': 28, 'num_classes': 10, 'hf_path': None},
    'imagenet': {'image_size': 224, 'num_classes': 1000, 'hf_path': None},
}


def get_dataset_info(dataset_name: str) -> Dict[str, int]:
    """Get dataset metadata (image_size, num_classes) for a given dataset.

    Args:
        dataset_name: Dataset name (e.g., 'tiny-imagenet', 'cifar10', etc.)

    Returns:
        Dict with 'image_size', 'num_classes', and optionally 'hf_path'
    """
    name_lower = dataset_name.lower()
    info = DATASET_INFO.get(name_lower) or DATASET_INFO.get(dataset_name)

    if info is None:
        # Try to load from HF dataset to get metadata
        if '/' in dataset_name and HF_AVAILABLE:
            try:
                hf_dataset = load_dataset(dataset_name, split='train', streaming=True)
                features = hf_dataset.features
                # Get image size from first sample
                sample = next(iter(hf_dataset))
                img = sample.get('image') or sample.get('pixel_values')
                if img is not None:
                    if hasattr(img, 'size'):
                        info = {'image_size': img.size[0], 'num_classes': len(features['label'].names) if hasattr(features.get('label'), 'names') else features['label'].num_classes}
                    elif isinstance(img, torch.Tensor):
                        info = {'image_size': img.shape[-1], 'num_classes': features['label'].num_classes if hasattr(features.get('label'), 'num_classes') else 200}
                # Register for future use
                if info:
                    DATASET_INFO[dataset_name] = info
            except Exception:
                pass

    if info is None:
        # Default fallback
        info = {'image_size': 64, 'num_classes': 200, 'hf_path': None}

    return info


def get_hf_path(dataset_name: str) -> Optional[str]:
    """Get the Hugging Face dataset path for a given local dataset name.

    Args:
        dataset_name: Local dataset name (e.g., 'cifar10', 'tiny-imagenet')

    Returns:
        HF dataset path (e.g., 'uoft-cs/cifar10') or None if not available
    """
    info = get_dataset_info(dataset_name)
    return info.get('hf_path')


def create_hf_dataset(
    name: str,
    split: str = 'train',
    image_size: int = 64,
    augment: bool = True,
    shuffle: bool = True,
    buffer_size: int = 1000,
    seed: int = 42,
    **kwargs,
) -> Dataset:
    """Create a dataset using Hugging Face datasets library with streaming mode.

    High-performance loading pipeline:
    1. Load from Hub in streaming mode (no full download required)
    2. Apply shuffle buffer for approximate global randomization
    3. Use with_transform for efficient batch-level on-the-fly transforms

    Args:
        name: Dataset name on Hugging Face Hub (e.g., 'zh-plus/tiny-imagenet')
        split: 'train' or 'val'
        image_size: Target image size for resizing
        augment: Whether to apply data augmentation
        shuffle: Whether to apply shuffle buffer (default: True for training)
        buffer_size: Shuffle buffer size (default: 1000)
        seed: Random seed for reproducibility (default: 42)
        **kwargs: Additional arguments passed to load_dataset

    Returns:
        Dataset compatible with PyTorch DataLoader

    Example:
        ds = create_hf_dataset("zh-plus/tiny-imagenet", split="train")
        loader = DataLoader(ds, batch_size=64, num_workers=4, pin_memory=True)
    """
    if not HF_AVAILABLE:
        raise ImportError(
            "Hugging Face datasets library is not installed. "
            "Install with: pip install datasets"
        )

    # Map split names (HF uses 'train', 'validation', and/or 'test')
    # Some datasets like uoft-cs/cifar10 only have 'train' and 'test'
    if split == 'train':
        hf_split = 'train'
    else:
        # Try 'validation' first, fall back to 'test' if not available
        hf_split = 'validation'

    # 1. Load dataset in streaming mode
    try:
        hf_dataset = load_dataset(
            name,
            split=hf_split,
            streaming=True,
            **kwargs,
        )
    except ValueError as e:
        if 'validation' in str(e) and split in ('val', 'validation'):
            # Fall back to 'test' split if 'validation' doesn't exist
            hf_split = 'test'
            hf_dataset = load_dataset(
                name,
                split=hf_split,
                streaming=True,
                **kwargs,
            )
        else:
            raise

    # 2. Apply shuffle buffer for approximate global randomization
    # This is essential for streaming - without it, data comes in class order
    if shuffle:
        hf_dataset = hf_dataset.shuffle(buffer_size=buffer_size, seed=seed)

    # 3. Wrap with appropriate dataset class based on streaming mode
    # Streaming datasets use HFStreamingDatasetWrapper (IterableDataset)
    # Non-streaming datasets use HFDatasetWrapper (Dataset)
    if isinstance(hf_dataset, HFIterableDataset):
        return HFStreamingDatasetWrapper(
            hf_dataset=hf_dataset,
            image_size=image_size,
            augment=augment,
        )
    else:
        return HFDatasetWrapper(
            hf_dataset=hf_dataset,
            image_size=image_size,
            augment=augment,
        )


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
    'create_hf_dataset',
    'get_transforms',
    'get_dataset_info',
    'get_hf_path',
    'DATASET_INFO',
    'HFDatasetWrapper',
]
