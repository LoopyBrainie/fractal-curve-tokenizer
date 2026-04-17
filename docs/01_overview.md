# Chapter 1: System Architecture

## 1.1 Overview

The **Fractal Curve Tokenizer** project introduces a Vision Transformer (ViT) architecture that replaces traditional fixed-grid patching with **content-adaptive quadtree tokenization** ordered by the **Hilbert space-filling curve**. By leveraging the locality-preserving properties of the Hilbert curve, the model dynamically allocates more tokens to complex image regions while maintaining a spatially coherent sequence for the transformer.

---

## 1.2 Motivation and Key Innovations

Standard ViTs suffer from two fundamental limitations:

1. **Locality Blindness**: Raster-scan serialization destroys 2D spatial proximity - vertically adjacent patches become distant in sequence
2. **Scale Invariance Violation**: Uniform 16×16 patches waste computation on homogeneous regions (sky, walls) while under-resolving complex areas (edges, textures)

This project addresses these via:

### Hilbert Locality Preservation

Hilbert ordering ensures that adjacent tokens in the sequence are spatially proximate in 2D space, providing a strong inductive bias:

$$\|p_1 - p_2\|_2 \leq C \cdot |H^{-1}(p_1) - H^{-1}(p_2)|^{1/2}$$

### Adaptive Quadtree Splitting

Instead of a fixed grid, the `StreamingFractalTokenizerV3` uses a differentiable splitter to decompose the image into variable-sized patches based on local complexity:

$$\text{Split}(R) \iff C(R) > \tau_d$$

### LCA-Based Attention

By utilizing the **Lowest Common Ancestor (LCA)** in the quadtree hierarchy, the model implements a geometric attention bias with ~100 parameters (vs $O(N^2)$ in standard ViTs):

$$B[i,j] = \text{LCAEmbed}(\text{LCA}(i, j))$$

### Efficiency

Adaptive tokenization achieves a **~40× reduction** in attention matrix size compared to standard ViT-16 (for 224×224 images):

| Metric | Standard ViT-16 | Fractal ViT |
|:-------|:----------------|:------------|
| Token Count | ~196 (14×14 patches) | ~32-64 |
| Attention Matrix | $N^2 \approx 38K$ | $N^2 \approx 1-4K$ |
| **Reduction** | - | **~40×** |

> **Note**: Complexity remains $O(N^2 \cdot D)$. The efficiency gain comes from token count reduction, not asymptotic complexity.

---

## 1.3 System Architecture Pipeline

**Fractal ViT Data Flow**

```
Image (B, C, H, W)
        │
        ▼
┌───────────────────────────────────────────────┐
│         StreamingFractalTokenizerV3            │
│  ┌─────────────────────────────────────────┐ │
│  │ SharedConv (Feature Extraction)          │ │
│  │ HilbertOptimalSplitter (Decision)        │ │
│  │ ROI-Align (Region Pooling)              │ │
│  │ HilbertSort (Curve Ordering)             │ │
│  └─────────────────────────────────────────┘ │
└───────────────────────────────────────────────┘
        │
        ▼
   (tokens, levels_info)
        │
        ▼
┌───────────────────────────────────────────────┐
│       FractalPositionEmbedding                 │
│       (Depth + Path Encoding)                 │
└───────────────────────────────────────────────┘
        │
        ▼
   Level-Aware LayerNorm
        │
        ▼
┌───────────────────────────────────────────────┐
│     FractalTransformerBlock × L                │
│  ┌─────────────────────────────────────────┐ │
│  │ ManifoldNativeAttention (LCA Bias)      │ │
│  │ Level-Aware LayerNorm                    │ │
│  │ AdaptiveFractalFeedForward (SwiGLU)      │ │
│  └─────────────────────────────────────────┘ │
└───────────────────────────────────────────────┘
        │
        ▼
   Global Pooling (CLS / Mean)
        │
        ▼
┌───────────────────────────────────────────────┐
│       MLP Head + LogitsClamp                  │
└───────────────────────────────────────────────┘
        │
        ▼
   Class Logits (B, num_classes)
```

---

## 1.4 Architecture Layers

The codebase follows a strict **4-layer hierarchy** for modularity and dependency management:

| Layer | Name | Purpose | Key Entities |
|:------|:-----|:--------|:--------------|
| **L4** | **Application** | High-level model assembly | `FractalCurveViT` |
| **L3** | **Pipeline** | Tokenization and Transformer stages | `StreamingFractalTokenizer`, `FractalTransformer` |
| **L2** | **Components** | Specialized layers and logic | `HilbertOptimalSplitter`, `ManifoldNativeAttention`, `SwiGLUFFN` |
| **L1** | **Foundation** | Mathematical primitives and config | `HilbertCurve`, `LevelsInfo`, `FractalConfig`, `SplitterProtocol` |

### Import Rules (L1→L2→L3→L4)

```python
# L1 imports: None (base)
# L2 imports: L1 only
from vit_pytorch.core.curve_hilbert import HilbertCurve
from vit_pytorch.core.levels_info import LevelsInfo

# L3 imports: L1, L2
from vit_pytorch.modules.tokenizer import StreamingFractalTokenizerV3

# L4 imports: All
from vit_pytorch import FractalCurveViT
```

> **Wrong**: `from vit_pytorch.modules.base_splitter import CoreSplitter`
> **Correct**: `from vit_pytorch.core.splitter_protocol import CoreSplitter`

---

## 1.5 Code Entity Relationship

```
                    Natural Language Space
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
         "L4 uses L3"          "L3 uses L2 Protocol"
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
              Implementation      ┌───────┴───────┐
                                 ▼               ▼
                          FractalCurveViT   StreamingFractalTokenizer
                          +forward()          +tokenize()
                          +analyze()          +split()
                                               │
                              ┌────────────────┼────────────────┐
                              ▼                ▼                ▼
                       HilbertOptimalSplitter ManifoldNativeAttention SwiGLUFFN
                       +forward()           +forward()         +forward()
                       +update_candidates()
```

---

## 1.6 Model-Trainer Interface

**Principle**: Model defines capabilities, Trainer decides usage.

`forward()` returns `TrainingStats`:

```python
@dataclass
class TrainingStats:
    logits: Tensor              # Classification logits
    num_tokens: Tensor         # Token count per sample
    depth_used: Tensor         # Max depth used
    depth_distribution: Tensor  # Token count per depth
    features: Optional[Tensor]  # Final layer features
    transformer_tokens: Tensor  # Tokens after transformer
    # Auxiliary outputs collected from layers:
    # - splitter_output (entropy, budget_loss, etc.)
    # - attention_outputs[i] (geometric_bias_mean, etc.)
    # - ffn_outputs[i] (level_mixing_mean, etc.)
```

---

## 1.7 Document Structure

| Chapter | Content |
|:--------|:--------|
| [00_introduction](00_introduction.md) | Project background and motivation |
| [01_overview](01_overview.md) | System architecture (this chapter) |
| [02_data_structures](02_data_structures.md) | Core mathematical foundations |
| [03_fractal_tokenizer](03_fractal_tokenizer.md) | Tokenization pipeline |
| [04_positional_embedding](04_positional_embedding.md) | Position encoding |
| [05_attention_mechanism](05_attention_mechanism.md) | Manifold-native attention |
| [06_feedforward_network](06_feedforward_network.md) | Feed-forward networks |
| [07_transformer_encoder](07_transformer_encoder.md) | Transformer blocks |
| [08_fractal_vit_model](08_fractal_vit_model.md) | Complete model |
| [09_training_system](09_training_system.md) | Training infrastructure |
| [10_testing_qa](10_testing_qa.md) | Testing and benchmarking |
| [appendix](appendix.md) | Appendix |

---

## 1.8 Quick Start

### Installation

```bash
# Using uv (recommended)
uv sync

# Or pip
pip install -e .
```

### Basic Usage

```python
import torch
from vit_pytorch import FractalCurveViT

model = FractalCurveViT(
    image_size=224,
    num_classes=1000,
    dim=384,
    num_layers=12,
    heads=6,
    mlp_dim=768,
    tokenizer_type='streaming_v3',
    bias_mode='lca',
    ffn_type='swiglu_level',
)

images = torch.randn(4, 3, 224, 224)
stats = model(images)
print(f"Logits shape: {stats.logits.shape}")  # (4, 1000)
print(f"Token count: {stats.num_tokens.mean().item():.1f}")
```

### Training

```bash
# CUB-200 (dynamic resolution)
uv run python src/training/train_fractal_vit.py \
    --dataset cub200 \
    --image-size None \
    --use-amp \
    --compile
```

---

## 1.9 Project Status

| Metric | Value |
|:-------|:------|
| Version | 0.8.x |
| Test Coverage | 295+ tests passing |
| Tokenizer | V3 (Variable Depth Tokens) |
| Attention Bias | LCA (recommended) |
| FFN | SwiGLU + Level Adaptation |

> **Next**: [00_introduction.md](00_introduction.md) - Project Background and Motivation
