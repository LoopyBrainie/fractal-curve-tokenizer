# Fractal Curve Tokenizer

[English](README.md) | [中文](README_zh.md)

A next-generation Vision Transformer (ViT) architecture that leverages fractal geometry and Hilbert curves for adaptive, multi-scale image tokenization.

## Overview

Fractal Curve Tokenizer introduces a novel approach to image tokenization in Vision Transformers. Instead of dividing images into a fixed grid of patches, it uses a recursive, adaptive splitting strategy based on image complexity (variance, edges, texture). This allows the model to allocate more tokens to complex regions and fewer to uniform areas, optimizing computational efficiency and capturing multi-scale features.

Key technologies include:

* **Learnable Fractal Tokenization**: Uses a **MiniCNN** and **REINFORCE** (with Gumbel-Softmax) to learn optimal image splitting strategies dynamically during training.
* **Hilbert Curve Traversal**: Preserves 2D spatial locality when flattening tokens into a 1D sequence using true recursive Hilbert curves.
* **Advanced Positional Embedding**: Encodes hierarchical depth and path history to maintain structural context.
* **Batch Processing with Padding**: Efficiently handles variable-length token sequences within a batch.

## Features

* **Adaptive Resolution**: Automatically adjusts token density based on image content complexity.
* **Differentiable Tokenizer**: The splitting decision is fully differentiable and optimized end-to-end.
* **Spatial Locality Preservation**: Uses Hilbert curves to maintain better spatial relationships than raster scan order.
* **Robust Regularization**: Integrated **DropPath** (Stochastic Depth) and Entropy Regularization to prevent overfitting.
* **Efficient Batch Training**: Optimized `pad_sequence` and masking implementation for high-speed training.
* **Flexible Architecture**: Supports both `NextGenerationFractalViT` (full feature set) and `SimpleFractalViT` (lightweight, backward compatible).

## Installation

Ensure you have uv and PyTorch installed.

```bash
# Clone the repository
git clone https://github.com/LoopyBrainie/fractal-curve-tokenizer.git
cd fractal-curve-tokenizer

# Install dependencies
pip install -r requirements.txt
# OR if using uv/poetry (recommended)
uv sync
```

## Usage

### Basic Inference

```python
import torch
from vit_pytorch.fractal_vit import NextGenerationFractalViT

# Initialize model
model = NextGenerationFractalViT(
    image_size=256,
    num_classes=1000,
    dim=512,
    depth=6,
    heads=8,
    mlp_dim=1024,
    min_patch_size=(4, 4),
    max_level=5
)

# Forward pass
img = torch.randn(1, 3, 256, 256)
logits = model(img) # (1, 1000)
```

### Training

The project includes a robust training script located at `examples/training/train_fractal_vit.py`. This script supports training on CIFAR-10, CIFAR-100, and MNIST.

#### Basic Training Command

```bash
python examples/training/train_fractal_vit.py --dataset cifar10 --epochs 50 --batch-size 64
```

#### Full Argument List

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--dataset` | str | `cifar10` | Dataset to use: `cifar10`, `cifar100`, `mnist`, `imagenet`, `coco`, `caltech256`, `tiny-imagenet`. |
| `--data-root` | str | `None` | Path to the root directory of the dataset (required for ImageNet/COCO/TinyImageNet). |
| `--epochs` | int | `50` | Number of training epochs. |
| `--batch-size` | int | `64` | Batch size for training. |
| `--lr` | float | `5e-4` | Initial learning rate. |
| `--weight-decay` | float | `0.01` | Weight decay for optimizer. |
| `--val-split` | float | `0.1` | Fraction of training data to use for validation. |
| `--subset-size` | int | `None` | Limit the number of training samples (for debugging). |
| `--dim` | int | `192` | Model embedding dimension. |
| `--depth` | int | `8` | Depth of the Transformer. |
| `--heads` | int | `8` | Number of attention heads. |
| `--dim-head` | int | `32` | Dimension of each attention head. |
| `--dropout` | float | `0.1` | Dropout rate. |
| `--emb-dropout` | float | `0.1` | Embedding dropout rate. |
| `--max-level` | int | `4` | Maximum recursion level for fractal tokenization. |
| `--pool` | str | `cls` | Pooling method: `cls` or `mean`. |
| `--use-simple` | flag | `False` | Use `SimpleFractalViT` instead of `NextGenerationFractalViT`. |
| `--no-learnable-split` | flag | `False` | Disable the learnable split decision network (use heuristic). |
| `--quick-test` | flag | `False` | Run a quick 5-epoch test on a small subset. |
| `--use-amp` | flag | `False` | Enable Automatic Mixed Precision (AMP) training. |
| `--gradient-clip` | float | `1.0` | Gradient clipping value. |
| `--num-workers` | int | `2` | Number of data loading workers. |
| `--seed` | int | `42` | Random seed for reproducibility. |
| `--disable-hilbert-bias` | flag | `False` | Disable Hilbert-path attention bias (faster on CPU). |
| `--force-next-gen` | flag | `False` | Force using `NextGenerationFractalViT` even on CPU. |
| `--device` | str | `auto` | Device to use: `auto`, `cpu`, `cuda`. |

#### Output

Training artifacts are saved in the `experiments/` directory, organized by timestamp:

* `checkpoints/`: Saved model weights (`best.pth`).
* `logs/`: Training logs.
* `visualizations/`: Loss and accuracy curves.
* `training_history.json`: Detailed metrics for every epoch.

## Testing

The project uses `pytest` for testing. The test suite has been reorganized into unit and integration tests.

```bash
# Run all tests
uv run pytest

# Run only unit tests
uv run pytest tests/unit

# Run only integration tests
uv run pytest tests/integration
```

## Project Structure

```text
fractal-curve-tokenizer/
├── src/
│   └── vit_pytorch/
│       ├── fractal_vit.py          # Main model definitions
│       ├── fractal_curve_tokenizer.py # Adaptive tokenizer logic
│       ├── positional.py           # Advanced positional embeddings
│       ├── transformer.py          # Transformer blocks with masking support
│       └── ...
├── examples/
│   └── training/
│       └── train_fractal_vit.py    # Main training script
├── tests/
│   ├── unit/                       # Unit tests for components
│   └── integration/                # Integration tests for workflows
├── experiments/                    # Training outputs (ignored by git)
└── workspace/                      # Local data and models (ignored by git)
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
