# Introduction

> **Fractal Curve Tokenizer for Vision Transformers**

## Background and Motivation

Traditional Vision Transformers (ViT) partition images into fixed-size patches (e.g., 16×16), ignoring the inherent multi-scale structure of visual content. This work explores an alternative: **content-adaptive tokenization** guided by the Hilbert space-filling curve.

### Problem Statement

Standard ViT tokenization exhibits two fundamental limitations:

1. **Scale Invariance Violation**: Uniform patches cannot represent both fine textures and coarse semantics efficiently
2. **Locality Blindness**: The linear sequence of patches does not encode spatial proximity

### Hypothesis

The **Hilbert curve's locality-preserving property** provides a natural inductive bias for vision transformers, enabling:

- Spatially coherent token sequences
- Hierarchical attention patterns via LCA (Lowest Common Ancestor) relationships
- Multi-scale representation through adaptive quadtree splitting

---

## Mathematical Foundation

### Hilbert Curve

The **Hilbert curve** is a continuous fractal mapping that fills a 2D space while preserving locality:

$$H: [0, n^2) \leftrightarrow [0, n) \times [0, n)$$

**Locality Bound**: For any two points $p_1, p_2$ in the 2D grid:

$$\|p_1 - p_2\|_2 \leq C \cdot |H^{-1}(p_1) - H^{-1}(p_2)|^{1/2}$$

where $C$ is a dimension-dependent constant. This ensures that adjacent positions in the Hilbert sequence correspond to spatially proximate locations.

### Quadtree-Hilbert Isomorphism

A fundamental property underlies this work: the bijection between quadtree paths and Hilbert curve segments:

$$\text{QuadtreePath}(R) = [q_1, q_2, \ldots, q_d] \iff \text{HilbertSegment}(R) = H|_{[a,b]}$$

This enables adaptive quadtree-based splitting to maintain Hilbert-ordered token sequences.

---

## Architecture Overview

| Component | Implementation | Description |
|:----------|:---------------|:------------|
| Tokenizer | `StreamingFractalTokenizerV3` | Adaptive quadtree + Hilbert reordering |
| Position Encoding | `FractalPositionEmbedding` | Depth + path encoding |
| Attention Bias | `LCAHilbertBias` | ~100 parameters, explicit geometric meaning |
| FFN | `SwiGLUFFN` | Gated activation with level adaptation |

---

## Quick Start

### Installation

```bash
# Using uv (recommended)
uv sync

# Or pip
pip install -e .
```

### Basic Usage

```python
from vit_pytorch import FractalCurveViT

model = FractalCurveViT(
    image_size=224,
    num_classes=1000,
    dim=384,
    depth=6,
    heads=6,
    mlp_dim=768,
    tokenizer_type='streaming_v3',  # Variable Depth Tokens
    hilbert_bias_mode='lca',        # LCA Hilbert Bias
)

images = torch.randn(4, 3, 224, 224)
logits = model(images)  # (4, 1000)
```

---

## Document Structure

| Chapter | Content |
|:--------|:--------|
| [01_overview](01_overview.md) | System architecture and data flow |
| [02_data_structures](02_data_structures.md) | Core data structures |
| [03_fractal_tokenizer](03_fractal_tokenizer.md) | Tokenization pipeline |
| [04_positional_embedding](04_positional_embedding.md) | Position encoding |
| [05_attention_mechanism](05_attention_mechanism.md) | Hilbert-aware attention |
| [06_feedforward_network](06_feedforward_network.md) | Feed-forward networks |
| [07_transformer_encoder](07_transformer_encoder.md) | Transformer encoder |
| [08_fractal_vit_model](08_fractal_vit_model.md) | Complete model |
| [09_training_system](09_training_system.md) | Training system |
| [10_testing_qa](10_testing_qa.md) | Testing and QA |
| [11_issues_roadmap](11_issues_roadmap.md) | Development history |
| [appendix](appendix.md) | Appendix |

---

## Project Status

| Metric | Value |
|:-------|:------|
| Version | 0.8.x |
| Test Coverage | 295+ tests passing |
| Tokenizer | V3 (Variable Depth Tokens) |
| Hilbert Bias | LCA (recommended) |
| FFN | SwiGLU + Level Adaptation |

> **Next**: [01_overview.md](01_overview.md) - System Architecture
