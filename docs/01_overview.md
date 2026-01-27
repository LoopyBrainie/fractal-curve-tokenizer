# Chapter 1: System Architecture

## 1.1 Overview

This chapter describes the complete data flow and module structure of the Fractal Curve ViT architecture.

### High-Level Pipeline

$$I \xrightarrow{\text{Tokenize}} (T, L) \xrightarrow{E_{pos}} T' \xrightarrow{\text{CLS}} [c; T'] \xrightarrow{\text{Transformer}} X' \xrightarrow{\text{Pool}} z \xrightarrow{\text{MLP}} \hat{y}$$

where:
- $I \in \mathbb{R}^{B \times C \times H \times W}$: Input image batch
- $T \in \mathbb{R}^{B \times N \times D}$: Token embeddings
- $L \in \mathbb{Z}^{B \times N \times \text{Info}}$: Level information (depth + quadtree path)
- $X' \in \mathbb{R}^{B \times (N+1) \times D}$: Encoded sequence (with CLS token)
- $\hat{y} \in \mathbb{R}^{B \times C_{out}}$: Class logits

---

## 1.2 Data Flow Diagram

```
Input Image (B, C, H, W)
        │
        ▼
┌───────────────────────────────────────────────┐
│       StreamingFractalTokenizerV3             │
│  ┌─────────────────────────────────────────┐  │
│  │ 1. Learnable Complexity: C_theta(R)     │  │
│  │ 2. Differentiable Quadtree Split        │  │
│  │ 3. Region Pooling via ROI-Align         │  │
│  │ 4. Hilbert Curve Reordering             │  │
│  └─────────────────────────────────────────┘  │
└───────────────────────────────────────────────┘
        │
        ▼
   (tokens, levels_info)
        │
        ▼
┌───────────────────────────────────────────────┐
│       FractalPositionEmbedding                │
│  E_pos(i) = Fusion(E_depth(d_i) + E_path(i))  │
└───────────────────────────────────────────────┘
        │
        ▼
   T' = T + E_pos
        │
        ▼
┌───────────────────────────────────────────────┐
│            Add CLS Token                      │
│       [CLS; T'] → (B, N+1, D)                 │
└───────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────┐
│         FractalTransformer × L                │
│  ┌─────────────────────────────────────────┐  │
│  │ Level-Aware LayerNorm                   │  │
│  │ HilbertAwareMultiScaleAttention         │  │
│  │   + LCA Hilbert Bias                    │  │
│  │ DropPath + Residual                     │  │
│  │ Level-Aware LayerNorm                   │  │
│  │ AdaptiveFractalFeedForward (SwiGLU)     │  │
│  │ DropPath + Residual                     │  │
│  └─────────────────────────────────────────┘  │
└───────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────┐
│     Pooling: CLS or Mean                      │
└───────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────┐
│     MLP Head: LN → Linear → GELU → Linear     │
└───────────────────────────────────────────────┘
        │
        ▼
   Logits (B, num_classes)
```

---

## 1.3 Module Hierarchy

| Layer | Module | File | Core Functionality |
|:------|:-------|:-----|:-------------------|
| **L4** | Application | `model_fractal_vit.py` | `FractalCurveViT` |
| **L3** | Pipeline | `tokenizer_streaming.py` | `StreamingFractalTokenizerV3` |
|        |          | `block_transformer.py` | `FractalTransformer` |
| **L2** | Components | `gumbel_topk_splitter.py` | `GumbelTopKSplitter` (Scheme D/E) |
|        |            | `attn_hilbert_bias.py` | `HilbertAwareMultiScaleAttention`, `LCAHilbertBias` |
|        |            | `ffn_swiglu.py` | `SwiGLUFFN`, `AdaptiveFractalFeedForward` |
|        |            | `embed_fractal_position.py` | `FractalPositionEmbedding` |
| **L1** | Foundation | `curve_hilbert.py` | `HilbertCurve`, `PseudoHilbertCurve` |

---

## 1.4 Key Innovations

### 1.4.1 Variable Depth Tokens (V3)

Unlike fixed-grid tokenization, V3 performs **content-adaptive quadtree splitting**:

$$\text{Split}(R) \iff C(R) > \tau_d$$

where:
- $C(R) = \alpha \cdot \frac{\text{Var}(R)}{\text{Var}(R) + \sigma_0^2} + (1-\alpha) \cdot \frac{G(R)}{G(R) + g_0^2}$
- $\tau_d = \tau_0 \cdot \gamma^d$ (depth-dependent threshold)

### 1.4.2 LCA Hilbert Bias

Attention bias derived from quadtree LCA (Lowest Common Ancestor) depth:

$$B[i,j] = \text{LCAEmbed}(\text{LCA}(i, j))$$

This provides explicit geometric meaning with only ~100 learnable parameters.

### 1.4.3 SwiGLU FFN with Level Adaptation

$$\text{SwiGLU}(x) = W_{out} \cdot (\text{Swish}(W_{gate} \cdot x) \odot W_{value} \cdot x)$$

Extended with level-adaptive residual:

$$\text{Output} = (1 - \alpha_d) \cdot \text{FFN}(x) + \alpha_d \cdot \text{Adapter}([x; E_{level}(d)])$$

---

## 1.5 Configuration

### FractalConfig

```python
from vit_pytorch import FractalConfig

config = FractalConfig(
    d_model=384,
    num_heads=6,
    hilbert_bias_mode='lca',  # 'lca', 'low_rank', 'hierarchical'
    max_depth=4,
)
```

### Model Instantiation

```python
from vit_pytorch import FractalCurveViT

model = FractalCurveViT(
    image_size=224,
    num_classes=1000,
    dim=384,
    depth=6,
    heads=6,
    mlp_dim=768,
    tokenizer_type='streaming_v3',
    bias_mode='lca',
    ffn_type='swiglu_level',
)
```

---

## 1.6 Complexity Analysis

> **Key Clarification**: The "40× reduction" refers to **token count reduction**, not asymptotic complexity. Token count reduces from ~307K (standard ViT 16×16 patches for 224×224) to ~32 (V3 variable-depth tokens).

| Operation | Complexity | Notes |
|:----------|:-----------|:------|
| Tokenization | $O(N_{cand} \cdot D)$ | $N_{cand} = \sum_{d=0}^{D} 4^d$ (e.g., 85 for D=3) |
| Hilbert Reordering | $O(N \log N)$ | Sort by Hilbert index |
| LCA Computation | $O(N^2)$ | Cached, amortized $O(1)$ |
| Attention | $O(N^2 \cdot D)$ | Standard transformer (N ≈ 32) |
| **Effective Computation** | ~40× reduction | $N_{V3} \approx 32$ vs $N_{ViT} \approx 307K$ |

> **Note**: Complete dynamic depth adaptation is deferred due to fundamental incompatibility with GPU SIMT parallelism. Current implementation uses fixed `depth // 2` for parallel efficiency.

> **Next**: [02_data_structures.md](02_data_structures.md) - Core Data Structures
