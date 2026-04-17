# Chapter 0: Project Background and Motivation

## 0.1 Problem Statement: Limitations of Fixed-Patch ViTs

Traditional Vision Transformers partition images into fixed-size grids (e.g., 16×16 patches). This approach suffers from two primary inefficiencies:

### 0.1.1 Scale Invariance Violation

Uniform grids treat high-entropy regions (edges, textures) and low-entropy regions (sky, uniform backgrounds) with the same resolution. This leads to:
- **Redundant computation** in simple areas (wasting tokens on homogeneous regions)
- **Information loss** in complex regions (insufficient resolution for fine details)

### 0.1.2 Locality Blindness

Standard raster-scan serialization of patches destroys 2D spatial proximity. Two patches that are vertically adjacent in an image become distant in the linear sequence, forcing the Transformer to learn spatial relationships from scratch without any geometric inductive bias.

---

## 0.2 Mathematical Foundation

### 0.2.1 Hilbert Locality Bound

The Hilbert curve ($H$) is a continuous fractal mapping:

$$H: [0, n^2) \leftrightarrow [0, n) \times [0, n)$$

It is characterized by a strict **locality bound**:

$$\|p_1 - p_2\|_2 \leq C \cdot |H^{-1}(p_1) - H^{-1}(p_2)|^{1/2}$$

This ensures that **points close in the 1D Hilbert sequence are guaranteed to be spatially proximate in 2D space**. The exponent 1/2 reflects the fractal dimension of the Hilbert curve.

### 0.2.2 Quadtree-Hilbert Isomorphism

A core technical insight: there exists a **bijection** between quadtree paths and Hilbert curve segments. Any region $R$ defined by a quadtree path can be exactly mapped to a continuous segment of the Hilbert curve:

$$\text{QuadtreePath}(R) = [q_1, q_2, \ldots, q_d] \iff \text{HilbertSegment}(R) = H|_{[a,b]}$$

This isomorphism enables:
- Variable-sized quadtree regions to map to contiguous Hilbert segments
- Spatial relationships preserved through the token ordering
- Hierarchical structure encoded in the sequence

### 0.2.3 LCA-Based Geometry

The hierarchical nature of the quadtree allows for the calculation of the **Lowest Common Ancestor (LCA)** between any two tokens:

$$\text{LCA}(i, j) = \text{Length}(\text{CommonPrefix}(\text{Path}(i), \text{Path}(j)))$$

The depth of the LCA serves as a proxy for spatial and structural distance:
- **High LCA depth** → tokens are in nearby quadrants → stronger attention bias
- **Low LCA depth** → tokens are in distant regions → weaker attention bias

This provides **geometric attention bias with minimal parameters** (~100 params vs $O(N^2)$ in standard ViTs).

---

## 0.3 Architecture Overview

### Current Implementation: V3 Variable Depth Tokens

The system uses **content-adaptive quadtree splitting** with independent per-region split decisions:

$$\text{Split}(R) \iff C(R) > \tau_d$$

where $C(R)$ is a learnable complexity measure and $\tau_d$ is a depth-dependent threshold. Each region makes its own decision without competing with other scales.

### Key Components

| Component | Purpose |
|:----------|:--------|
| `StreamingFractalTokenizerV3` | Adaptive token generation |
| `HilbertOptimalSplitter` | Region splitting decisions |
| `ManifoldNativeAttention` | LCA-based geometric attention |
| `SwiGLUFFN` | Level-adaptive feed-forward network |

---

## 0.4 System Mapping: Concept to Code

### Tokenization Pipeline Data Flow

```
Image (B, C, H, W)
        │
        ▼
┌───────────────────────────────────────────────┐
│       StreamingFractalTokenizerV3             │
│  ┌─────────────────────────────────────────┐ │
│  │ SharedConv (Feature Extraction)         │ │
│  │ HilbertOptimalSplitter (Decision)       │ │
│  │ ROI-Align (Region Pooling)              │ │
│  │ HilbertSort (Curve Ordering)            │ │
│  └─────────────────────────────────────────┘ │
└───────────────────────────────────────────────┘
        │
        ▼
   TokenizerOutput (tokens, levels_info)
        │
        ▼
┌───────────────────────────────────────────────┐
│       FractalPositionEmbedding                 │
│       (Depth + Path Encoding)                 │
└───────────────────────────────────────────────┘
        │
        ▼
   FractalTransformerBlock × L
        │
        ▼
   Class Logits
```

### Architecture Component Hierarchy

<<<<<<< Updated upstream
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
=======
```
FractalCurveViT (L4)
├── tokenizer: StreamingFractalTokenizerV3
│   ├── patch_embed: HilbertNativePatchEmbed
│   └── splitter: CoreSplitter → HilbertOptimalSplitter
├── transformer: FractalTransformer
│   └── layers: FractalTransformerBlock[]
│       ├── attention: ManifoldNativeAttention
│       │   └── bias: LCAHilbertBias
│       └── ffn: AdaptiveFractalFeedForward
└── pos_drop: nn.Dropout
>>>>>>> Stashed changes
```

---

## 0.5 Efficiency Gains

By using variable-depth tokens, the model significantly reduces the sequence length $N$ processed by the Transformer:

| Metric | Standard ViT-16 | Fractal ViT |
|:-------|:----------------|:------------|
| Image Size | 224×224 | 224×224 |
| Patch Size | 16×16 | Variable (4×4 to 64×64) |
| Token Count ($N$) | ~196 (14×14) | ~32-64 |
| Attention Matrix ($N^2$) | ~38K | ~1-4K |
| **Reduction Factor** | - | **~40×** |

> **Note**: Attention complexity remains $O(N^2 \cdot D)$. The ~40× efficiency gain comes from token count reduction ($N \approx 32-64$ vs $196$), not asymptotic complexity change.

### Why Variable Token Count Works

- Standard ViT with 16×16 patches on 224×224 image: $N = (224/16)^2 = 196$ tokens
- Fractal ViT with adaptive splitting: $N \approx 32-64$ tokens (depending on image complexity)
- The model learns to allocate more tokens to complex regions (edges, textures) and fewer to simple regions (background)

---

## 0.6 Key Innovations Summary

| Innovation | Description | Benefit |
|:-----------|:------------|:--------|
| **Hilbert Locality** | Space-filling curve preserves 2D proximity in 1D sequence | Strong geometric inductive bias |
| **Adaptive Quadtree** | Content-dependent splitting based on complexity | Efficient token allocation |
| **LCA Attention Bias** | Geometric bias from quadtree hierarchy | ~100 params instead of $O(N^2)$ |
| **Scale-Aware Residual** | Parent-to-child information flow | Prevents information washout |

---

## 0.7 Document Navigation

| Chapter | Content |
|:--------|:--------|
| [00_introduction](00_introduction.md) | Project background and motivation (this chapter) |
| [01_overview](01_overview.md) | System architecture overview |
| [02_data_structures](02_data_structures.md) | Core mathematical foundations |
| [03_fractal_tokenizer](03_fractal_tokenizer.md) | Tokenization pipeline |

> **Next**: [01_overview.md](01_overview.md) - System Architecture
