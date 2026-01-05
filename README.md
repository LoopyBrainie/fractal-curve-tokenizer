# Fractal Curve Tokenizer

[English](README.md) | [中文](README_zh.md)

A Vision Transformer with **Hilbert curve tokenization** and **adaptive multi-scale patch selection**. Fractal ViT dynamically allocates tokens based on image complexity—more tokens for detailed regions, fewer for uniform backgrounds.

## Fractal ViT vs. Standard ViT

| | Standard ViT | Fractal ViT |
|---|---|---|
| **Tokenization** | Fixed 16×16 grid | Adaptive quadtree (4×4 ~ 32×32) |
| **Token Count** | Fixed: N = HW/P² | Variable: N ∈ [N_min, N_max] |
| **Patch Ordering** | Raster scan (row-major) | Hilbert curve (locality-preserving) |
| **Position Encoding** | Learnable N² bias | LCA-based hierarchical bias (~100 params) |
| **Inductive Bias** | None (data-driven) | Spatial locality + multi-scale structure |

### The Problem with Fixed Grids

Standard ViT treats all image regions equally:

```
Standard ViT: 4 tokens (uniform 16×16)     Fractal ViT: 5 tokens (adaptive)
┌────────────┬────────────┐                ┌──────────────────────────┐
│            │            │                │                          │
│   16×16    │   16×16    │                │          32×32           │
│  (sky)     │  (sky)     │                │      (uniform sky)       │
├────────────┼────────────┤       →        ├──────┬──────┬────────────┤
│            │            │                │ 4×4  │ 4×4  │            │
│   16×16    │   16×16    │                │(eye) │(eye) │    8×8     │
│  (face)    │  (face)    │                └──────┴──────┴────────────┘
└────────────┴────────────┘
```

Fractal ViT allocates **fine-grained tokens** to complex regions (eyes, edges) and **coarse tokens** to uniform areas (sky, background).

## Key Technical Contributions

### 1. Hilbert Curve Ordering

Patches are ordered along a **Hilbert space-filling curve** instead of row-major raster scan:

$$H: [0, n^2) \leftrightarrow [0, n) \times [0, n)$$

**Why Hilbert?** The curve preserves 2D locality in the 1D sequence:
$$\|p_1 - p_2\|_2 \leq C \cdot |H^{-1}(p_1) - H^{-1}(p_2)|^{1/2}$$

This means spatially adjacent patches remain close in the attention sequence, improving the effectiveness of local attention patterns.

![Hilbert Curve Orders](workspace/visualizations/hilbert_curve.png)

*Hilbert curve (order=4) generated directly from the architecture utility.*

### 2. Adaptive Quadtree Tokenization

Instead of fixed patch sizes, we use a **content-aware quadtree** that splits regions based on local complexity:

$$C(R) = \alpha \cdot \frac{\text{Var}(R)}{\text{Var}(R) + \sigma_0^2} + (1-\alpha) \cdot \frac{G(R)}{G(R) + g_0^2}$$

- **Var(R)**: Local pixel variance (texture)
- **G(R)**: Gradient energy (edges)
- **Split if**: $C(R) > \tau_0 \cdot \gamma^d$ (depth-dependent threshold)

![Quadtree Structure](workspace/visualizations/quadtree.png)

*Synthetic adaptive quadtree driven by the architecture splitting heuristic (no training data).*

### 3. LCA-Based Attention Bias

We replace the standard N² learnable position bias with a **Lowest Common Ancestor (LCA)** formulation:

$$B[i,j] = \tau_h \cdot \text{Embed}(\text{LCA}(i,j))$$

where LCA(i,j) is the tree depth at which paths to tokens i and j first diverge. This provides:
- **O(log N) parameters** vs O(N²) for standard ViT
- **Explicit geometric meaning**: nearby patches share deeper ancestors
- **Per-head learnable temperature** τ_h for adaptive scaling

![LCA Bias Matrix](workspace/visualizations/lca_bias.png)

*LCA depth matrix derived purely from quadtree paths (architecture-level bias).* 

## Architecture

```
Image (B, C, H, W)
       │
       ▼
┌─────────────────────────────────────┐
│  StreamingFractalTokenizerV3        │
│  ├─ SharedConv: feature extraction  │
│  ├─ AdaptiveSplit: quadtree split   │
│  ├─ ROI-Align: variable-size pool   │
│  └─ HilbertSort: curve ordering     │
└─────────────────────────────────────┘
       │
       ▼ (T ∈ ℝ^{B×N×D}, L ∈ ℤ^{B×N×(1+depth)})
┌─────────────────────────────────────┐
│  FractalPositionEmbedding           │
│  └─ Depth + Quadrant path encoding  │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  FractalTransformer (×L layers)     │
│  ├─ HilbertAttention + LCA Bias     │
│  └─ SwiGLU FFN + Level Adaptation   │
└─────────────────────────────────────┘
       │
       ▼
   [CLS] Pool → MLP Head → Logits
```

## Quick Start

```python
import torch
from vit_pytorch import FractalCurveViT

model = FractalCurveViT(
    image_size=224,
    num_classes=1000,
    dim=384,
    depth=6,
    heads=6,
    tokenizer_type="streaming_v3",
    hilbert_bias_mode="lca",
)

img = torch.randn(1, 3, 224, 224)
logits = model(img)  # (1, 1000)
```

## Configuration

### Splitting Schemes

| Scheme | Description |
|--------|-------------|
| `balanced_greedy` | Fast priority-queue splitting with 2:1 balance constraint |
| `fixed_budget_dp` | Dynamic programming for strict token budget |
| `learnable` | End-to-end differentiable (Gumbel-Softmax) |

### Attention Bias Modes

| Mode | Params | Description |
|------|--------|-------------|
| `lca` | ~100 | **Recommended.** LCA embedding table |
| `low_rank` | ~50K | Factorized bias: B = ΦΨᵀ |
| `hierarchical` | ~5K | Per-level learned bias |

## Installation

```bash
git clone https://github.com/LoopyBrainie/fractal-curve-tokenizer.git
cd fractal-curve-tokenizer
pip install -e .  # or: uv sync
```

## Training

```bash
# CIFAR-10
python examples/training/train_fractal_vit.py --dataset cifar10 --epochs 50

# With learnable splitting
python examples/training/train_fractal_vit.py \
    --dataset tiny-imagenet \
    --split-scheme learnable \
    --epochs 100
```

## Testing

```bash
pytest tests/  # 295 test cases
```

## Related Work

- **ViT** (Dosovitskiy et al., 2020): Fixed 16×16 patches, raster ordering
- **Swin Transformer**: Shifted windows, but still uniform grid
- **Quadtree Transformer**: Similar quadtree idea, different attention mechanism
- **Fractal ViT (Ours)**: Hilbert ordering + LCA bias + adaptive splitting

## License

MIT License - see [LICENSE](LICENSE).
