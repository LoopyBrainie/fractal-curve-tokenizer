# Fractal Curve Tokenizer

[English](README.md) | [中文](README_zh.md)

A next-generation Vision Transformer (ViT) architecture that leverages fractal geometry and Hilbert curves for adaptive, multi-scale image tokenization.

## Overview

Fractal Curve Tokenizer introduces a novel approach to image tokenization in Vision Transformers. Instead of dividing images into a fixed grid of patches, it uses a recursive, adaptive splitting strategy based on image complexity (variance, edges, texture). This allows the model to allocate more tokens to complex regions and fewer to uniform areas, optimizing computational efficiency and capturing multi-scale features.

Key technologies include:

* **Adaptive Fractal Tokenization**: Recursively splits image patches based on content complexity.
* **Hilbert Curve Traversal**: Preserves 2D spatial locality when flattening tokens into a 1D sequence using true recursive Hilbert curves.
* **Advanced Positional Embedding**: Encodes hierarchical depth and path history to maintain structural context.
* **Batch Processing with Padding**: Efficiently handles variable-length token sequences within a batch.

## Features

* **Adaptive Resolution**: Automatically adjusts token density based on image content.
* **Spatial Locality Preservation**: Uses Hilbert curves to maintain better spatial relationships than raster scan order.
* **Multi-Scale Feature Extraction**: Captures features at various scales simultaneously.
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

#### Key Arguments

* `--dataset`: Choose dataset (`cifar10`, `cifar100`, `mnist`). Default: `cifar10`.
* `--epochs`: Number of training epochs. Default: `50`.
* `--batch-size`: Batch size. Default: `64`.
* `--lr`: Learning rate. Default: `5e-4`.
* `--quick-test`: Run a short 5-epoch training on a small subset of data to verify the pipeline.

    ```bash
    python examples/training/train_fractal_vit.py --quick-test
    ```

* `--use-simple`: Force the use of `SimpleFractalViT` (lighter model) instead of the full `NextGenerationFractalViT`. Recommended for CPU training or baseline comparisons.
* `--device`: Manually specify device (`cpu`, `cuda`, `auto`). Default: `auto`.
* `--num-workers`: Number of data loading workers. Default: `2`.

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
