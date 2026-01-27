# Chapter 5: Attention Mechanism

## 5.1 Overview

The `HilbertAwareMultiScaleAttention` extends standard multi-head attention with **Hilbert curve-derived attention biases**, encoding spatial proximity and hierarchical relationships explicitly. It supports **affine modulation** (I31-3) for area-aware attention biasing.

**Complexity Note**: Attention complexity remains $O(N^2 \cdot D)$. The ~40× efficiency gain comes from token count reduction ($N \approx 32$ vs $307K$), not asymptotic complexity change.

**Temperature Selection**: The default $\tau_h \approx 1.5$ is chosen to:
1. Provide meaningful bias magnitude (not too small to be ignored)
2. Allow gradient flow through the Softplus parameterization
3. Balance spatial locality prior strength

---

## 5.2 Mathematical Formulation

### 5.2.1 Standard Multi-Head Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

### 5.2.2 Hilbert-Aware Attention

$$\text{HilbertAttn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} \cdot \sigma_{scale} + \alpha_h \cdot B_{hilbert} + \alpha_l \cdot B_{level}\right) V$$

where:
- $\sigma_{scale}$: Learnable depth-dependent scaling factor
- $\alpha_h$: Hilbert bias scale constant (`HILBERT_BIAS_SCALE`)
- $\alpha_l$: Level bias scale constant (`LEVEL_BIAS_SCALE`)
- $B_{hilbert}$: LCA-based Hilbert curve bias
- $B_{level}$: Relative level depth bias

### 5.2.3 Bias Scale Constants

To ensure proper gradient magnitudes, bias terms are scaled by constants:

| Constant | Value | Purpose |
|:---------|:------|:--------|
| `HILBERT_BIAS_SCALE` | 1.0 | Scale for LCA-based spatial bias |
| `LEVEL_BIAS_SCALE` | 0.1 | Scale for relative level bias |

The scaled attention formula:

$$\text{scores} = \frac{QK^T}{\sqrt{d_k}} \cdot \sigma_{scale} + HILBERT\_BIAS\_SCALE \cdot B_{hilbert} + LEVEL\_BIAS\_SCALE \cdot B_{level}$$

---

## 5.3 Hilbert Bias Modes

### 5.3.1 LCA Hilbert Bias (Standard)

This bias encodes the tree distance between two tokens using their Lowest Common Ancestor (LCA) in the quadtree structure.

**Mathematical definition**:

$$B[i,j] = \tau_h \cdot \text{LCAEmbed}(\text{LCA}(i, j))$$

where:
- $\text{LCA}(i, j) \in \{0, \dots, d_{max}\}$: Depth of the smallest quadtree region containing both $R_i$ and $R_j$.
- $\text{LCAEmbed}: \mathbb{Z} \to \mathbb{R}^H$: Learnable embedding table.
- $\tau_h \in \mathbb{R}^H$: Per-head temperature parameter.

**Temperature Parameter ($\tau_h$)**:

To ensure the bias strength is positive and adaptive, we use a Softplus parameterization:

$$\tau_h = \text{Softplus}(\gamma_h)$$

Initialized such that $\tau \approx 1.5$, enhancing the prior for spatial locality.

**P11-3: Region-Based LCA Computation**:

For accurate LCA, compute directly from region boundaries:

$$\text{Path}(R) = \text{bit}(cx, D-d) + 2 \cdot \text{bit}(cy, D-d)$$
$$\text{LCA}(i, j) = \text{Length}(\text{CommonPrefix}(\text{Path}(i), \text{Path}(j)))$$

> **Note**: This replaces fragile index arithmetic with direct geometric computation.

---

## 5.4 Level Bias

### 5.4.1 Relative Level Embedding

Encodes the relationship between tokens at different scales (e.g., parent-child vs. peer-peer).

$$B_{level}[i,j] = W_{rel}[\text{clamp}(d_i - d_j + L, 0, 2L)]$$

where:
- $d_i, d_j$: Depths of tokens $i, j$
- $L$: Maximum relative depth range
- $W_{rel}$: Embedding table of size $(2L+1) \times H$

### 5.4.2 Level Scaling

Scales the attention logits based on the depth of the query to stabilize training across scales.

$$\sigma_{scale}(d) = \text{Softplus}(\text{LevelScaleEmb}(d))$$

Deeper tokens (finer resolution) typically learn smaller scaling factors to broaden their attention span or vice-versa.

---

## 5.5 Affine Modulated Bias (I31-3)

The **AffineModulatedBias** enhances spatial attention with area-aware modulation, enabling the model to learn size-dependent attention patterns.

### 5.5.1 Mathematical Formulation

**Area Encoding** (NeRF-style Fourier features):

$$f_{area} = \frac{\log(s_{patch} + 1)}{\log(S_{total} + 1)}$$
$$\gamma(f) = [\sin(2^k \pi f), \cos(2^k \pi f)]_{k=0}^{L-1}$$

where:
- $s_{patch}$: Patch area
- $S_{total}$: Total image area
- $L$: Number of Fourier levels

**Affine Modulation**:

$$B_{\text{final}} = \gamma(s_i, s_j) \odot B_{\text{spatial}} + \beta(s_i, s_j)$$

where:
- $\gamma(s_i, s_j) = \sigma(\text{MLP}_\gamma(p_s))$: Learnable scale factor
- $\beta(s_i, s_j) = \text{MLP}_\beta(p_s)$: Learnable bias factor
- $p_s = \text{area\_emb}[i] \cdot \text{area\_emb}[j]$: Area similarity

**Residual Connection**:

$$B_{\text{combined}} = B_{\text{spatial}} + \alpha \cdot (B_{\text{final}} - B_{\text{spatial}})$$

where $\alpha$ is a learnable zero-initialized parameter for gradual activation.

### 5.5.2 AreaEncoder Architecture

```
Input: regions [B, N, 4]
   │
   ▼
┌─────────────────────┐
│ Normalized Area     │
│ f = log(area+1) /   │
│     log(total+1)    │
└─────────────────────┘
   │
   ▼
┌─────────────────────┐
│ Fourier Features    │
│ [sin(2^kπf),        │
│  cos(2^kπf)]_k      │
│  → 2L dimensions    │
└─────────────────────┘
   │
   ▼
┌─────────────────────┐
│ MLP Projection      │
│ Linear(2L) → hidden │
│ → Linear(hidden) →  │
│   Linear(hidden) →  │
│   dim               │
└─────────────────────┘
   │
   ▼
Output: area_emb [B, N, dim]
```

### 5.5.3 AffineModulatedBias Architecture

```
regions [B, N, 4]
      │
      ├──────────────────┐
      ▼                  ▼
┌─────────────┐   ┌─────────────┐
│   LCA       │   │   Area      │
│   Embedding │   │   Encoder   │
└─────────────┘   └─────────────┘
      │                  │
      ▼                  ▼
┌─────────────┐   ┌─────────────┐
│   Spatial   │   │   Area      │
│   Bias      │   │   Similarity│
│ [B,dim,N,N] │   │   Matrix    │
└─────────────┘   └─────────────┘
      │                  │
      └────────┬─────────┘
               ▼
        ┌─────────────┐
        │   Affine    │
        │   Modulation│
        │ γ·B + β     │
        └─────────────┘
               │
               ▼
        ┌─────────────┐
        │   Residual  │
        │ B + α(·-B)  │
        └─────────────┘
               │
               ▼
Output: bias [B, dim, N, N]
```

---

## 5.6 Shape-Scale Bias (I31)

The **ShapeScaleEncoder** captures region geometry for enhanced attention bias.

### 5.6.1 Mathematical Formulation

**Aspect Ratio** (log-transformed for symmetry):

$$r = \log(w/h)$$

**Normalized Area**:

$$s = \frac{w \cdot W_{patch}}{W_{total} \cdot H_{total}}$$

**Gated Combination**:

$$g = \sigma(\text{MLP}([r; s]))$$
$$E_{shape}(R) = \text{MLP}([r \cdot g; s \cdot (1-g)])$$

**Shape-Scale Similarity Matrix**:

$$B_{shape}[i,j] = E_{shape}(R_i) \cdot E_{shape}(R_j)^T$$

### 5.6.2 Combined Bias

$$B_{final} = B_{LCA} + \tau \cdot B_{shape}$$

where $\tau$ is a learnable zero-initialized weight.

---

## 5.7 Bias Mode Comparison

| Mode | Parameters | Complexity | Geometric Meaning |
|:-----|:-----------|:-----------|:------------------|
| `lca` | ~100 | $O(N^2)$ | Explicit (LCA depth) |
| `affine_modulation` | ~1K | $O(N^2)$ | Area-aware spatial bias |
| `shape_scale` | ~1K | $O(N^2)$ | Geometry-aware bias |

**Recommendation**: Use `lca` mode for efficiency, `affine_modulation` for improved performance on datasets with size variation.

---

## 5.8 Implementation

### Class: HilbertAwareMultiScaleAttention

```python
class HilbertAwareMultiScaleAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        max_level: int = 8,
        use_hilbert_bias: bool = True,
        use_level_scaling: bool = True,
        lca_temperature: Optional[float] = 1.5,
        learnable_temperature: bool = True,
        use_affine_modulation: bool = False,
        fourier_levels: int = 4,
    ):
        """
        Args:
            dim: Input dimension
            heads: Number of attention heads
            dim_head: Dimension per head
            dropout: Dropout rate
            max_level: Maximum quadtree depth
            use_hilbert_bias: Enable LCA-based Hilbert bias
            use_level_scaling: Enable depth-dependent scaling
            lca_temperature: Initial temperature for LCA bias
            learnable_temperature: Whether temperature is learnable
            use_affine_modulation: Enable I31-3 area-aware bias
            fourier_levels: Number of Fourier frequency levels
        """
```

### Class: LCAHilbertBias

```python
class LCAHilbertBias(HilbertBiasBase):
    def __init__(
        self,
        max_depth: int,
        heads: int,
        lca_temperature: Optional[float] = 1.5,
        learnable_temperature: bool = True,
    ):
        """
        Uses P6-2 learnable temperature with Softplus parameterization.
        Supports both levels_info and region-based LCA computation.
        """
```

### Class: AffineModulatedBias

```python
class AffineModulatedBias(nn.Module):
    def __init__(
        self,
        dim: int,
        max_depth: int,
        enable_area_modulation: bool = True,
        fourier_levels: int = 4,
    ):
        """
        Args:
            dim: Attention dimension
            max_depth: Maximum quadtree depth
            enable_area_modulation: Enable area-aware modulation
            fourier_levels: Number of Fourier frequency levels
        """
```

---

## 5.9 Usage Example

### Basic Configuration

```python
from vit_pytorch import HilbertAwareMultiScaleAttention

attn = HilbertAwareMultiScaleAttention(
    dim=384,
    heads=6,
    dim_head=64,
    max_level=8,
    use_hilbert_bias=True,
    use_level_scaling=True,
    lca_temperature=1.5,
    learnable_temperature=True,
)
```

### With Affine Modulation (I31-3)

```python
attn = HilbertAwareMultiScaleAttention(
    dim=384,
    heads=6,
    max_level=8,
    use_affine_modulation=True,
    fourier_levels=4,
)

# During forward pass, provide regions and image_size
x = torch.randn(2, 100, 384)
regions = torch.zeros(2, 100, 4)  # Region boundaries
image_size = 224

out = attn(x, regions=regions, image_size=image_size)
```

### Direct LCA Computation from Regions

```python
from vit_pytorch.attn_hilbert_bias import LCAHilbertBias

lca_bias = LCAHilbertBias(
    max_depth=8,
    heads=6,
    lca_temperature=1.5,
)

# Compute bias directly from regions (P11-3 recommended)
regions = torch.randn(2, 50, 4)  # [B, N, 4]
image_size = 224

bias = lca_bias.forward_from_regions(regions, image_size)
# Output: [B, H, N, N]
```

---

## 5.10 Cache Optimization (I30-9)

For efficiency, LCA depth computation is cached across transformer layers:

**Cache Key**: `data_ptr` + `torch_version`

**Benefits**:
- Eliminates redundant LCA computation per layer
- ~6x speedup for 6-layer transformers
- Automatic invalidation on tensor modification

```python
# Cache is automatically managed
lca_bias = LCAHilbertBias(max_depth=8, heads=6)

# Manually clear cache if needed
lca_bias.clear_cache()
```

---

## 5.11 Numerical Stability

### Variance Epsilon

Layer normalization uses $\epsilon = 10^{-5}$ for variance stability:

$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

### Softplus Temperature

Temperature parameterization ensures positivity:

$$\tau_h = \text{Softplus}(\gamma_h) = \log(1 + e^{\gamma_h})$$

> **Next**: [06_feedforward_network.md](06_feedforward_network.md) - Feed-Forward Networks
