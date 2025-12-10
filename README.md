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
# CIFAR-10 (auto-downloads)
uv run python examples/training/train_fractal_vit.py --dataset cifar10 --epochs 50 --batch-size 64

# Tiny ImageNet (auto-downloads ~237 MB)
uv run python examples/training/train_fractal_vit.py --dataset tiny-imagenet --quick-test
```

**Note:** Datasets like CIFAR-10, CIFAR-100, MNIST, and Tiny ImageNet are automatically downloaded to `workspace/data/` on first use. Large datasets (ImageNet, COCO) require manual download.

#### Full Argument List

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--dataset` | str | `cifar10` | Dataset to use: `cifar10`, `cifar100`, `mnist`, `imagenet`, `coco`, `caltech256`, `tiny-imagenet`. |
| `--data-root` | str | `workspace/data` | Path to the root directory of the dataset. Auto-created for supported datasets. |
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
| `--use-simple` | flag | `False` | **[DEPRECATED]** Ignored, always uses `NextGenerationFractalViT`. |
| `--no-learnable-split` | flag | `False` | Disable the learnable split decision network (use heuristic). |
| `--quick-test` | flag | `False` | Run a quick 5-epoch test on a small subset. |
| `--use-amp` | flag | `False` | Enable Automatic Mixed Precision (AMP) training. |
| `--gradient-clip` | float | `1.0` | Gradient clipping value. |
| `--accum-steps` | int | `1` | **[NEW]** Gradient accumulation steps for larger effective batch size. |
| `--num-workers` | int | `-1` | Number of data loading workers. `-1` for auto-detect. |
| `--seed` | int | `42` | Random seed for reproducibility. |
| `--bias-mode` | str | `low_rank` | Hilbert bias mode: `original`, `low_rank`, `hierarchical`. |
| `--low-rank-r` | int | `32` | Rank for low-rank Hilbert bias factorization. |
| `--device` | str | `auto` | Device to use: `auto`, `cpu`, `cuda`. |

#### Training Optimizations (v2.0)

The training script includes several performance optimizations:

| Optimization | Description | Expected Speedup |
|-------------|-------------|------------------|
| `torch.compile()` | PyTorch 2.0+ JIT compilation (CUDA only) | 15-40% |
| `cudnn.benchmark` | cuDNN auto-tuner for fixed input sizes | 5-15% |
| TF32 enabled | TensorFloat-32 for Ampere+ GPUs | 5-10% |
| Evaluation AMP | Mixed precision during validation/test | 20-30% inference |
| `set_to_none=True` | Faster gradient zeroing | 1-3% |
| `pin_memory` + `non_blocking` | Async CPU-GPU data transfer | Variable |
| Graceful interruption | Ctrl+C saves checkpoint before exit | - |

**Gradient Accumulation Example:**

```bash
# Simulate batch size 256 with limited GPU memory (64 x 4 = 256)
uv run python examples/training/train_fractal_vit.py \
    --batch-size 64 --accum-steps 4 --use-amp
```

**Container Environment (Docker/Podman):**

```bash
# Recommended for --shm-size=4g containers
python train_fractal_vit.py --num-workers 8 --use-amp
```

#### Output

Training artifacts are saved in the `experiments/` directory, organized by timestamp:

* `checkpoints/`: Saved model weights (`best.pth`).
* `logs/`: Training logs.
* `visualizations/`: Loss and accuracy curves.
* `training_history.json`: Detailed metrics for every epoch.

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
