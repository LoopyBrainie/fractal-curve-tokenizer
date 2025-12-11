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

The project includes a comprehensive training script at `examples/training/train_fractal_vit.py` with support for multiple datasets and advanced training features.

#### Quick Start

```bash
# CIFAR-10 with default settings
uv run python examples/training/train_fractal_vit.py --dataset cifar10 --epochs 50

# Tiny ImageNet with SwiGLU FFN (recommended)
uv run python examples/training/train_fractal_vit.py \
    --dataset tiny-imagenet \
    --ffn-type swiglu_level \
    --epochs 100 \
    --warmup-epochs 10

# Quick test (5 epochs, 512 samples)
uv run python examples/training/train_fractal_vit.py --quick-test --use-amp
```

#### FFN Architecture Selection

The training script supports three Feed-Forward Network variants:

| FFN Type | Parameters | Speed | Recommendation |
|----------|-----------|-------|----------------|
| `gelu` | 379K | Baseline | Legacy/baseline only |
| `swiglu` | 334K (-11.9%) | Fast | Lightweight tasks |
| `swiglu_level` | 357K (-5.8%) | Medium | **Default, best balance** |

**Example:**
```bash
# Use SwiGLU + Level Adaptation (default)
uv run python examples/training/train_fractal_vit.py --ffn-type swiglu_level

# Use lightweight SwiGLU
uv run python examples/training/train_fractal_vit.py --ffn-type swiglu
```

#### Supported Datasets

| Dataset | Auto-Download | Classes | Image Size | Notes |
|---------|--------------|---------|------------|-------|
| `cifar10` | ✅ | 10 | 32×32 | Default dataset |
| `cifar100` | ✅ | 100 | 32×32 | More challenging |
| `mnist` | ✅ | 10 | 28×28 | Grayscale digits |
| `tiny-imagenet` | ❌ Manual | 200 | 64×64 | Requires download |

**Tiny ImageNet Setup:**
```bash
# Download and extract manually
cd data
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
# Expected structure:
# data/tiny-imagenet-200/train/n01443537/images/*.JPEG
# data/tiny-imagenet-200/val/images/*.JPEG
```

#### Key Training Parameters

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--dataset` | str | `cifar10` | Dataset: `cifar10`, `cifar100`, `mnist`, `tiny-imagenet` |
| `--epochs` | int | `50` | Number of training epochs |
| `--batch-size` | int | `64` | Training batch size |
| `--lr` | float | `5e-4` | Initial learning rate |
| `--warmup-epochs` | int | `10` | **[NEW]** Number of warmup epochs |
| `--weight-decay` | float | `0.01` | Weight decay (L2 regularization) |
| `--val-split` | float | `0.1` | Validation split ratio |
| `--dim` | int | `192` | Model embedding dimension |
| `--depth` | int | `8` | Transformer depth (layers) |
| `--heads` | int | `8` | Number of attention heads |
| `--dim-head` | int | `32` | Dimension per attention head |
| `--max-level` | int | `4` | Maximum fractal recursion level |
| `--ffn-type` | str | `swiglu_level` | **[NEW]** FFN type: `gelu`, `swiglu`, `swiglu_level` |
| `--pool` | str | `cls` | Pooling method: `cls` or `mean` |
| `--dropout` | float | `0.1` | Dropout rate |
| `--gradient-clip` | float | `1.0` | Gradient clipping value |
| `--use-amp` | flag | `False` | Enable mixed precision training |
| `--accum-steps` | int | `1` | Gradient accumulation steps |
| `--num-workers` | int | `4` | Data loading workers |
| `--no-learnable-split` | flag | `False` | Disable learnable tokenization |
| `--quick-test` | flag | `False` | Quick 5-epoch test on 512 samples |

#### Performance Optimizations

The training script includes automatic performance enhancements:

| Optimization | Auto-Enabled | Expected Speedup | Requirements |
|-------------|--------------|------------------|--------------|
| TF32 Acceleration | ✅ CUDA | ~8× matmul speed | Ampere+ GPU (RTX 30/40) |
| cuDNN Benchmark | ✅ CUDA | 5-15% | Fixed input sizes |
| Spawn Multiprocessing | ✅ Always | Stability | CUDA compatibility |
| Persistent Workers | ✅ Always | Faster epochs | num_workers > 0 |
| Mixed Precision (AMP) | ⚙️ `--use-amp` | 20-40% | Modern GPU |
| Gradient Accumulation | ⚙️ `--accum-steps` | Larger batch | Limited VRAM |

**Example: Maximum Performance**
```bash
# RTX 3090/4090 optimal settings
uv run python examples/training/train_fractal_vit.py \
    --dataset cifar100 \
    --ffn-type swiglu_level \
    --batch-size 128 \
    --use-amp \
    --num-workers 4 \
    --warmup-epochs 10
```

**Example: Limited VRAM**
```bash
# Simulate batch size 256 with 8GB VRAM
uv run python examples/training/train_fractal_vit.py \
    --batch-size 64 \
    --accum-steps 4 \
    --use-amp
```

#### Training Output

Training artifacts are automatically saved in `experiments/` directory, organized by timestamp:

```
experiments/
└── fractal_vit_20251211_142110/
    ├── checkpoints/
    │   └── best.pth              # Best model weights
    ├── logs/
    │   ├── config.json           # Training configuration
    │   ├── metrics.json          # Per-epoch metrics
    │   └── final.json            # Final results + tokenization analysis
    └── visualizations/           # (Optional) Visualization plots
```

**Detailed logs include:**
* Train/validation loss and accuracy per epoch
* Learning rate schedule
* Token count statistics and adaptivity analysis
* Variance-token correlation assessment
* GPU information and training time

## Benchmarking

The project includes comprehensive benchmarking tools to evaluate model performance, tokenization efficiency, convergence behavior, and comparison with standard ViT.

### 1. Model Performance Benchmark

Benchmark forward/backward pass timing and tokenization analysis:

```bash
uv run python -m tests.benchmarks.benchmark_fractal_vit
```

**Key Metrics:**
- Forward/backward pass timing (ms)
- Tokenization time and token count distribution
- Model parameters and memory usage
- Images per second throughput

**Results:** Saved to `benchmark_results/` with JSON metrics and optional visualizations.

### 2. Convergence Analysis

Analyze training convergence on synthetic classification tasks:

```bash
uv run python -m tests.benchmarks.check_convergence \
  --output-dir benchmark_results \
  --image-size 32 \
  --num-epochs 20 \
  --num-runs 3
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--output-dir` | str | `benchmark_results` | Output directory for results |
| `--image-size` | int | `32` | Image size for synthetic data |
| `--num-classes` | int | `4` | Number of classes |
| `--num-epochs` | int | `30` | Training epochs per run |
| `--num-runs` | int | `3` | Number of repeated runs |

**Tested Scenarios:**
- Color classification (simple task)
- Pattern classification (moderate difficulty)
- Complexity classification (challenging task)

**Output Metrics:**
- Convergence speed (epochs to reach threshold)
- Final train/validation accuracy and loss
- Overfitting detection
- Loss variance and gradient statistics

### 3. Standard ViT Comparison

Compare FractalViT variants against standard patch-based ViT:

```bash
uv run python -m tests.benchmarks.compare_fractal_vs_standard \
  --output-dir benchmark_results \
  --image-size 64 \
  --num-epochs 20 \
  --batch-size 8
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--output-dir` | str | `benchmark_results` | Output directory |
| `--image-size` | int | `64` | Input image size |
| `--num-classes` | int | `10` | Number of classes |
| `--batch-size` | int | `8` | Batch size |
| `--num-epochs` | int | `20` | Training epochs |
| `--plot` | flag | `False` | Generate comparison plots |

**Comparison Metrics:**
- Tokenization efficiency (adaptive vs fixed patches)
- Computational performance (forward/backward speed)
- Memory usage
- Training convergence and final accuracy
- Parameter count

### 4. Pretrained Model Evaluation

Evaluate trained `.pth` checkpoints with comprehensive metrics and visualizations:

```bash
uv run python -m tests.benchmarks.evaluate_pretrained \
  --checkpoint experiments/fractal_vit_simple_20251208/checkpoints/best.pth \
  --visualize \
  --num-samples 500 \
  --output-dir benchmark_results/pretrained_eval
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--checkpoint` | str | **required** | Path to `.pth` checkpoint file |
| `--output-dir` | str | `{checkpoint_dir}/evaluation` | Output directory for results |
| `--visualize` | flag | `False` | Generate visualization plots |
| `--num-samples` | int | `None` | Number of test samples (None = all) |
| `--batch-size` | int | `32` | Evaluation batch size |

**Evaluation Metrics:**
- Test accuracy (top-1 and top-5)
- Throughput (images/second)
- Tokenization analysis (token count distribution)
- Image complexity vs token count correlation

**Generated Visualizations (if `--visualize`):**
- `tokenization_analysis.png`: Sample images with token distribution
- `complexity_vs_tokens.png`: Scatter plot of complexity vs tokens
- `performance_summary.png`: Accuracy and throughput charts
- `confusion_matrix.png`: Confusion matrix (for small sample sizes)
- `evaluation_results.json`: Complete metrics in JSON format

**Example with Multiple Checkpoints:**

```bash
# Evaluate your best model
uv run python -m tests.benchmarks.evaluate_pretrained \
  -c experiments/fractal_vit_simple_20251208_222817/checkpoints/best.pth \
  --visualize

# Quick evaluation on subset
uv run python -m tests.benchmarks.evaluate_pretrained \
  -c workspace/models/fractal_vit/fractal_vit_simple_best.pth \
  --num-samples 100 \
  -o quick_eval
```

### 5. CNN Ablation Study

Evaluate the impact of using CNN features for split decisions versus using only handcrafted features.

```bash
uv run python tests/benchmarks/benchmark_cnn_ablation.py --dataset cifar10 --epochs 20
```

**Arguments:**
- `--dataset`: Dataset to use (`cifar10`, `tiny-imagenet`).
- `--epochs`: Number of training epochs.
- `--batch-size`: Batch size (default: 64).
- `--use-cnn`: Enable CNN features (default: False in benchmark to test baseline).

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
│   ├── benchmarks/                 # Comprehensive benchmarking suite
│   │   ├── benchmark_fractal_vit.py    # Performance benchmarks
│   │   ├── check_convergence.py        # Convergence analysis
│   │   ├── compare_fractal_vs_standard.py # ViT comparison
│   │   ├── evaluate_pretrained.py      # Pretrained model evaluation
│   │   └── benchmark_metrics.py        # Core metrics definitions
│   ├── unit/                       # Unit tests for components
│   └── integration/                # Integration tests for workflows
├── experiments/                    # Training outputs (ignored by git)
├── benchmark_results/              # Benchmark outputs and visualizations
└── workspace/                      # Local data and models (ignored by git)
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
