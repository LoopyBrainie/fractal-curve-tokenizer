# Fractal Curve Tokenizer

[English](README.md) | [中文](README_zh.md)

A next-generation Vision Transformer (ViT) architecture that leverages fractal geometry and Hilbert curves for adaptive, multi-scale image tokenization.

## Overview

Fractal Curve Tokenizer introduces a novel approach to image tokenization in Vision Transformers. Instead of dividing images into a fixed grid of patches, it uses **Hilbert curve traversal** to preserve 2D spatial locality when flattening tokens into a 1D sequence.

**Key innovations:**

* **Streaming Fractal Tokenizer**: End-to-end differentiable tokenization using multi-scale convolution pyramid and Gumbel-Softmax scale selection (V2).
* **Hilbert Curve Traversal**: Preserves 2D spatial locality using space-filling curves with proven locality properties.
* **Low-Rank Hilbert Attention Bias**: Efficient $O(N \cdot r)$ attention bias computation instead of $O(N^2)$.
* **SwiGLU FFN**: Modern feed-forward architecture with gated linear units (LLaMA/PaLM style).
* **Advanced Positional Embedding**: Encodes hierarchical depth and quadrant path history.

## Architecture

```
Input Image (B, C, H, W)
        │
        ▼
┌─────────────────────────────┐
│ StreamingFractalTokenizerV2 │  ← Multi-scale ConvPyramid + Gumbel-Softmax
│ - MultiScalePatchEncoder    │
│ - Hilbert Reordering        │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│ AdvancedFractalPosition     │  ← Depth + Path Embedding + Fusion
│ Embedding                   │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│ EnhancedFractalTransformer  │  ← Level-Aware LayerNorm
│ - HilbertAwareAttention     │  ← Low-Rank Hilbert Bias
│ - SwiGLU FFN                │  ← Gate * Value projection
│ - DropPath                  │
└─────────────────────────────┘
        │
        ▼
    MLP Head → Logits
```

## Visualizations

The following visualizations demonstrate the core concepts of Hilbert curve tokenization.

### Hilbert Curve Orders

Comparison of Hilbert curves at different orders, showing how the space-filling pattern scales:

![Hilbert Order Comparison](workspace/visualizations/fractal_curves/hilbert_order_comparison.png)

### Locality Preservation

Demonstrates how Hilbert curves preserve 2D spatial locality in 1D sequences:

![Locality Preservation](workspace/visualizations/fractal_curves/hilbert_locality.png)

### Quadtree Structure

Visualization of the recursive quadtree decomposition with Hilbert traversal order:

![Quadtree Structure](workspace/visualizations/fractal_curves/quadtree_structure.png)

### 2D to 1D Mapping

How 2D grid positions are mapped to a 1D token sequence via Hilbert curve:

![2D to 1D Mapping](workspace/visualizations/fractal_curves/2d_to_1d_mapping.png)

### Multi-Scale Hierarchy

Visualization of multi-scale patch extraction at different resolutions:

![Multi-Scale Hierarchy](workspace/visualizations/fractal_curves/multiscale_hierarchy.png)

### Mixed-Level Adaptive Tokenization

Demonstrates how the model adaptively selects different patch sizes based on region complexity:

![Mixed-Level Segmentation](workspace/visualizations/fractal_curves/mixed_level_segmentation.png)

### Hilbert Curve Growth Animation

![Hilbert Growth](workspace/visualizations/fractal_curves/hilbert_growth.gif)

## Features

* **End-to-End Differentiable**: No REINFORCE required - fully differentiable with Gumbel-Softmax.
* **Spatial Locality Preservation**: Hilbert curves maintain 2D neighborhood relationships in 1D.
* **Efficient Attention**: Low-rank Hilbert bias reduces memory from $O(N^2)$ to $O(N \cdot r)$.
* **Modern FFN**: SwiGLU with optional level adaptation for layer-aware processing.
* **Robust Regularization**: Integrated DropPath (Stochastic Depth) and entropy regularization.
* **Flexible Architecture**: Supports `NextGenerationFractalViT` with streaming tokenizers and adaptive splitting.

## Installation

Ensure you have uv and PyTorch installed.

```bash
# Clone the repository
git clone https://github.com/LoopyBrainie/fractal-curve-tokenizer.git
cd fractal-curve-tokenizer

# Install dependencies (recommended: uv)
uv sync

# Or with pip
pip install -e .
```

## Usage

### Basic Inference

```python
import torch
from vit_pytorch import NextGenerationFractalViT

# Initialize model with recommended settings
model = NextGenerationFractalViT(
    image_size=224,
    num_classes=1000,
    dim=384,
    depth=6,
    heads=6,
    mlp_dim=768,
    tokenizer_type='streaming_v2',  # Recommended: Gumbel-Softmax
    bias_mode='low_rank',           # Recommended: Memory efficient
    ffn_type='swiglu_level',        # Recommended: SwiGLU + Level Adaptation
)

# Forward pass
img = torch.randn(1, 3, 224, 224)
logits = model(img)  # (1, 1000)
```

### Tokenizer Types

| Type | Class | Description | Status |
|------|-------|-------------|--------|
| `streaming_v2` | `StreamingFractalTokenizerV2` | Gumbel-Softmax adaptive scale | ✅ **Recommended** |
| `streaming` | `StreamingFractalTokenizer` | Fixed multi-scale convolution | ✅ Stable |
| `legacy` | `FractalHilbertTokenizer` | BFS + REINFORCE | ⚠️ Deprecated |

### Standalone Tokenizer

```python
from vit_pytorch import StreamingFractalTokenizerV2

tokenizer = StreamingFractalTokenizerV2(
    image_size=224,
    dim=384,
    scales=[4, 8, 16],
    temperature=1.0,
)

images = torch.randn(2, 3, 224, 224)
output = tokenizer.tokenize(images)

# Access tokens and levels
tokens = output.sequences[0].tokens      # (N, 384)
levels = output.sequences[0].get_levels() # (N, info_len)
```

### Training

The project includes a comprehensive training script at `examples/training/train_fractal_vit.py`.

#### Quick Start

```bash
# CIFAR-10 with recommended settings
uv run python examples/training/train_fractal_vit.py \
    --dataset cifar10 \
    --tokenizer-type streaming_v2 \
    --bias-mode low_rank \
    --ffn-type swiglu_level \
    --epochs 50

# Quick test (5 epochs, 512 samples)
uv run python examples/training/train_fractal_vit.py --quick-test --use-amp
```

#### Key Training Parameters

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--tokenizer-type` | `streaming_v2` | Tokenizer: `streaming_v2`, `streaming`, `legacy` |
| `--bias-mode` | `low_rank` | Hilbert bias: `low_rank`, `hierarchical`, `original` |
| `--ffn-type` | `swiglu_level` | FFN type: `swiglu_level`, `swiglu`, `gelu` |
| `--dataset` | `cifar10` | Dataset: `cifar10`, `cifar100`, `mnist`, `tiny-imagenet` |
| `--epochs` | `50` | Number of training epochs |
| `--batch-size` | `64` | Training batch size |
| `--lr` | `5e-4` | Initial learning rate |
| `--dim` | `192` | Model embedding dimension |
| `--depth` | `8` | Transformer depth (layers) |
| `--heads` | `8` | Number of attention heads |
| `--use-amp` | `False` | Enable mixed precision training |

#### FFN Architecture Selection

| FFN Type | Parameters | Speed | Use Case |
|----------|-----------|-------|----------|
| `gelu` | Baseline | 1.0x | Legacy comparison |
| `swiglu` | -12% | 1.15x | Lightweight inference |
| `swiglu_level` | -6% | 1.10x | **Default, best accuracy** |

#### Supported Datasets

| Dataset | Auto-Download | Classes | Image Size |
|---------|--------------|---------|------------|
| `cifar10` | ✅ | 10 | 32×32 |
| `cifar100` | ✅ | 100 | 32×32 |
| `mnist` | ✅ | 10 | 28×28 |
| `tiny-imagenet` | ❌ Manual | 200 | 64×64 |

#### Training Output

```
experiments/
└── fractal_vit_20251214_142110/
    ├── checkpoints/
    │   └── best.pth              # Best model weights
    ├── logs/
    │   ├── config.json           # Training configuration
    │   └── metrics.json          # Per-epoch metrics
    └── visualizations/           # Training curves
```

## Benchmarking

### Model Performance

```bash
uv run python -m tests.benchmarks.benchmark_fractal_vit
```

### Compare with Standard ViT

```bash
uv run python -m tests.benchmarks.compare_fractal_vs_standard \
    --output-dir benchmark_results \
    --image-size 64 \
    --num-epochs 20
```

### Evaluate Pretrained Model

```bash
uv run python -m tests.benchmarks.evaluate_pretrained \
    --checkpoint experiments/.../checkpoints/best.pth \
    --visualize
```

## Testing

```bash
# Run all tests
uv run pytest

# Unit tests only
uv run pytest tests/unit

# Integration tests only
uv run pytest tests/integration
```

**Test Status**: 120 passed, 1 skipped

## Project Structure

```text
fractal-curve-tokenizer/
├── src/
│   └── vit_pytorch/
│       ├── __init__.py             # Package entry with exports
│       ├── fractal_vit.py          # Main model definitions
│       ├── streaming_tokenizer.py  # StreamingFractalTokenizer V1/V2
│       ├── transformer.py          # Transformer with level-aware norm
│       ├── attention.py            # Hilbert-aware attention + Low-Rank Bias
│       ├── feedforward.py          # SwiGLU FFN + Level Adaptation
│       ├── positional.py           # Depth + Path positional embedding
│       ├── hilbert.py              # Hilbert curve d ↔ (x,y) mapping
│       ├── tokenization.py         # Base classes and data structures
│       ├── constants.py            # Hyperparameter defaults
│       ├── features.py             # Token feature computation
│       ├── utils.py                # Utility functions
│       └── _deprecated/            # Deprecated modules (v1.0 removal)
│           ├── fractal_curve_tokenizer.py  # Legacy BFS + REINFORCE
│           └── token_processor.py          # Legacy token processor
├── examples/
│   └── training/
│       └── train_fractal_vit.py    # Training script
├── tests/
│   ├── unit/                       # Unit tests
│   ├── integration/                # Integration tests
│   └── benchmarks/                 # Performance benchmarks
├── documents/                      # Project documentation
├── experiments/                    # Training outputs (gitignored)
└── workspace/                      # Local data/models (gitignored)
```

## Module Hierarchy

| Layer | Module | Description |
|-------|--------|-------------|
| **L4** Application | `fractal_vit.py` | `NextGenerationFractalViT` |
| **L3** Pipeline | `streaming_tokenizer.py` | Multi-scale tokenization with Hilbert reorder |
| | `transformer.py` | Level-aware transformer blocks |
| **L2** Component | `attention.py` | Hilbert bias with low-rank decomposition |
| | `feedforward.py` | SwiGLU + Level adaptation |
| | `positional.py` | Depth + Path embedding |
| **L1** Foundation | `hilbert.py` | Space-filling curve algorithms |
| | `tokenization.py` | Base classes, `TokenizerOutput` |
| | `constants.py` | Default hyperparameters |

## Mathematical Formalization

**Core Pipeline:**
$$I \xrightarrow{T} (T, L) \xrightarrow{E_{pos}} T' \xrightarrow{\text{Transformer}} X' \xrightarrow{\text{Pool}} z \xrightarrow{\text{MLP}} \hat{y}$$

**Hilbert Curve Mapping:**
$$H: [0, n^2) \leftrightarrow [0, n)^2$$

**Low-Rank Hilbert Bias:**
$$B_{hilbert}[i,j] = \phi(p_i)^T \cdot \psi(p_j), \quad \phi, \psi: \mathbb{R}^d \to \mathbb{R}^r$$

**SwiGLU FFN:**
$$\text{SwiGLU}(x) = W_{out} \cdot (\text{Swish}(W_{gate} \cdot x) \odot W_{value} \cdot x)$$

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
