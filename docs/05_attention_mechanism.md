# Chapter 5: Attention Mechanism

## 5.1 Overview

The `ManifoldNativeAttention` (MNA) is the core attention mechanism of the Fractal Curve ViT, designed to operate on the non-Euclidean manifold of variable-resolution fractal tokens.

**Key Innovation**: Unlike standard ViT that treats tokens as a flat sequence, this mechanism explicitly models **hierarchical and spatial relationships** derived from the Hilbert curve and quadtree structure.

**Efficiency**: Achieves ~40× reduction in attention matrix complexity by operating on sparse, non-uniform token sets (N ≈ 32-64 vs 307K for standard ViT).

### 5.1.1 Architecture

```
X_{l+1} = X_l + Attn(X_l) + FractalResidual(X_l)

Attn = BandedAttention(QK + B_manifold)
B_manifold = GeometricLatentDecoder(ξ)
```

### 5.1.2 Complexity Analysis

| Metric | Standard ViT | Fractal ViT |
|:-------|:-------------|:------------|
| Sequence Length (N) | ~196 | ~32-64 |
| Attention Matrix | $O(N^2)$ | $O(N \cdot W)$ |
| **Effective Reduction** | - | **~40×** |

> **Note**: Complexity remains $O(N^2 \cdot D)$. The efficiency gain comes from token count reduction, not asymptotic complexity change.

---

## 5.2 Geometric Latent Decoder (ξ-space)

The `GeometricLatentDecoder` maps tokens from discrete quadtree paths into a continuous **5-dimensional geometric feature space (ξ-space)**.

### 5.2.1 Coordinate Reconstruction

Tokens are reconstructed into 2D coordinates from their `LevelsInfo` paths using vectorized bitwise operations:

$$x = \sum_{k=0}^{d-1} \text{bit}_k(q_k, 0) \cdot 2^{max\_level-k-1}$$

$$y = \sum_{k=0}^{d-1} \text{bit}_k(q_k, 1) \cdot 2^{max\_level-k-1}$$

This avoids the $O(N \cdot D)$ serial overhead of traditional quadtree traversal.

### 5.2.2 Unified Geometric Feature Vector (ξ_ij)

The feature vector $\xi_{ij}$ between token $i$ and $j$ consists of:

| Component | Formula | Description |
|:----------|:--------|:------------|
| **Normalized Hilbert Distance** | $\Delta h_{ij} / N^2$ | Hilbert index difference |
| **LCA Depth Bias** | $2^{d_{LCA}}$ | Lowest Common Ancestor depth |
| **Euclidean Distance** | $\|pos_i - pos_j\|_2$ | Physical 2D distance |
| **Area Ratio** | $area_i / area_j$ | Patch size relationship |
| **Rotational Similarity** | $1$ if parents share quadrant, else $0$ | Spatial orientation |

---

## 5.3 Poincaré Disk Distance

To handle the multi-scale nature of fractal tokens, the model computes distances on a **Poincaré disk**, a model of hyperbolic geometry.

### 5.3.1 Mathematical Formulation

The 2D coordinates are mapped to the unit disk:

$$u = \tanh(r/2) \cdot \frac{x - c}{\|x - c\|}$$

The hyperbolic distance is:

$$d_H(u, v) = \text{acosh}\left(1 + \frac{2\|u-v\|^2}{(1-\|u\|^2)(1-\|v\|^2)}\right)$$

### 5.3.2 Numerical Stability (I-NAN)

To prevent gradient explosions and NaN values:

| Protection | Method |
|:-----------|:-------|
| **FP32 Enforcement** | Internal computations use FP32 to prevent FP16 underflow |
| **Epsilon Protection** | `acosh(x)` requires $x \geq 1 + \epsilon$ |
| **Output Clamping** | Distances clamped to $[0, 10.0]$ |

---

## 5.4 Hierarchical Attention Bias (LCA)

### 5.4.1 LCA Matrix Calculation

The attention scores are augmented with a bias $B_{hilbert}$ derived from the **Lowest Common Ancestor (LCA)** of token pairs:

$$B[i,j] = \text{LCAEmbed}(\text{LCA}(i, j))$$

The `get_lca_matrix` function computes the depth of the shared ancestor for every token pair. Tokens within the same quadtree branch receive a higher attention bias.

### 5.4.2 Bias Scale Constants

| Constant | Value | Purpose |
|:---------|:------|:--------|
| `HILBERT_BIAS_SCALE` | 1.0 | Scale for LCA-based spatial bias |
| `LEVEL_BIAS_SCALE` | 1.0 | Scale for relative level bias |

The scaled attention formula:

$$\text{scores} = \frac{QK^T}{\sqrt{d_k}} \cdot \sigma_{scale} + HILBERT\_BIAS\_SCALE \cdot B_{hilbert} + LEVEL\_BIAS\_SCALE \cdot B_{level}$$

> **Note (I122-2)**: The original τ_h temperature parameter has been removed. Bias strength is controlled by `HILBERT_BIAS_SCALE × √d_k`.

---

## 5.5 Level Bias

### 5.5.1 Relative Level Embedding

Encodes the relationship between tokens at different scales:

$$B_{level}[i,j] = W_{rel}[\text{clamp}(d_i - d_j + L, 0, 2L)]$$

where:
- $d_i, d_j$: Depths of tokens $i, j$
- $L$: Maximum relative depth range
- $W_{rel}$: Embedding table of size $(2L+1) \times H$

### 5.5.2 Level Scaling

Scales attention logits based on query depth:

$$\sigma_{scale}(d) = \text{Softplus}(\text{LevelScaleEmb}(d))$$

---

## 5.6 Hierarchical Soft-Hard Attention (I160-2)

> **Recommended**: Use with `DeterministicNeighborSplitter` for hard locality guarantees

### 5.6.1 Three-Region Partitioning

| Region | LCA Depth Condition | Attention Behavior |
|:-------|:-------------------|:-------------------|
| **HARD_ZERO** | $\ell < \ell_{min}$ | Force exclusion (mask = 0) |
| **SOFT_POSITIVE** | $\ell_{min} \leq \ell < \ell_{soft}$ | Soft bias encouragement |
| **HARD_ONE** | $\ell \geq \ell_{soft}$ | Full encouragement (mask = 1) |

Default values: $\ell_{min} = 1$ (quadrant boundary), $\ell_{soft} = 2$ (sub-quadrant boundary)

### 5.6.2 Mathematical Formulation

**Hierarchical Mask**:

$$M(\ell) = \begin{cases} 0 & \text{if } \ell < \ell_{min} \\ \sigma(\ell - \ell_{soft}/2) & \text{if } \ell_{min} \leq \ell < \ell_{soft} \\ 1 & \text{if } \ell \geq \ell_{soft} \end{cases}$$

**Attention Formula**:

$$\tilde{A}_{ij} = \frac{QK^T}{\sqrt{d_k}}[i,j] + \alpha(\ell_{ij}) \cdot B_{hilbert}[i,j]$$

---

## 5.7 Affine Modulated Bias (I31-3)

The **AffineModulatedBias** enhances spatial attention with area-aware modulation.

### 5.7.1 Area Encoding (NeRF-style Fourier Features)

$$f_{area} = \frac{\log(s_{patch} + 1)}{\log(S_{total} + 1)}$$

$$\gamma(f) = [\sin(2^k \pi f), \cos(2^k \pi f)]_{k=0}^{L-1}$$

### 5.7.2 Affine Modulation

$$B_{final} = \gamma(s_i, s_j) \odot B_{spatial} + \beta(s_i, s_j)$$

---

## 5.8 Shape-Scale Bias (I31)

The **ShapeScaleEncoder** captures region geometry for enhanced attention bias.

### 5.8.1 Aspect Ratio

$$r = \log(w/h)$$

### 5.8.2 Gated Combination

$$g = \sigma(\text{MLP}([r; s]))$$

$$E_{shape}(R) = \text{MLP}([r \cdot g; s \cdot (1-g)])$$

---

## 5.9 Scale-Aware Residual (Fractal Residuals)

**ScaleAwareResidual** enables parent-to-child information flow through learned residual connections.

### 5.9.1 Mathematical Formulation

$$X_{l+1}^{(child)} = X_l^{(parent)} \cdot \sigma(g_d) + \text{Attn}(X_l)$$

where $\sigma(g_d) = \text{sigmoid}(\text{Linear}(d))$ is a learnable gate based on depth.

### 5.9.2 Key Property

| Gate Value | Behavior |
|:-----------|:---------|
| Close to 1 | Child features dominated by parent projection (information flows up) |
| Close to 0 | Child features are independent |

---

## 5.10 Cartesian 2D RoPE (I167-4)

**Cartesian2DRoPE** implements rotary position embedding based on physical 2D coordinates $(x, y)$.

### 5.10.1 Mathematical Formulation

Given position $i$ with physical coordinates $p_i = (x_i, y_i)$:

$$\theta_i = \text{atan2}(y_i, x_i)$$

**RoPE application**:

$$\text{RoPE}(q_i, k_i) = \begin{pmatrix} \cos(\theta_i / 2) & -\sin(\theta_i / 2) \\ \sin(\theta_i / 2) & \cos(\theta_i / 2) \end{pmatrix} \begin{pmatrix} q_i^{(0)} \\ q_i^{(1)} \end{pmatrix}$$

### 5.10.2 Implementation

```python
class Cartesian2DRoPE(nn.Module):
    def __init__(self, dim: int, max_level: int = 8):
        self.dim = dim
        self.max_level = max_level

    def forward(self, q: Tensor, paths: Tensor, depths: Tensor) -> Tensor:
        coords = coords_from_paths(paths, depths, self.max_level)
        angles = torch.atan2(coords[..., 1], coords[..., 0])
        cos_angle, sin_angle = torch.cos(angles), torch.sin(angles)
        # Apply rotation...
```

---

## 5.11 Diagnostics and Monitoring

### 5.11.1 CLSAttentionTracker

The `CLSAttentionTracker` monitors attention flow from the `[CLS]` token to different quadtree depths.

**Link Broken Detection**: When `[CLS]` loses contact with global context (Depth 0 attention < 20%), the tracker signals to adjust `HILBERT_BIAS_SCALE`.

### 5.11.2 get_stats() Diagnostics

The `get_stats()` method returns real-time metrics:

| Metric | Description |
|:-------|:------------|
| `manifold_bias_std` | Dispersion of the geometric bias |
| `poincare_dist_mean` | Average hyperbolic distance between tokens |
| `effective_rank` | SVD-based rank of attention matrix (detects mode collapse) |

---

## 5.12 Implementation

### Class: ManifoldNativeAttention

```python
class ManifoldNativeAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        max_level: int = 8,
        use_cartesian_rope: bool = True,   # I167-4
        use_scale_aware_residual: bool = True,
        use_poincare_distance: bool = True,
        band_width: int = 32,
    ):
        """
        Args:
            dim: Input dimension
            heads: Number of attention heads
            dim_head: Dimension per head
            dropout: Dropout rate
            max_level: Maximum quadtree depth
            use_cartesian_rope: Enable Cartesian 2D RoPE
            use_scale_aware_residual: Enable fractal residuals
            use_poincare_distance: Enable Poincaré disk distance
            band_width: Band width for Hilbert-banded attention
        """
```

### Class: LCAHilbertBias

```python
class LCAHilbertBias(nn.Module):
    def __init__(
        self,
        max_depth: int,
        heads: int,
        lca_temperature: Optional[float] = None,  # Removed in I122-2
        learnable_temperature: bool = False,      # Removed in I122-2
    ):
        # Note: lca_temperature removed in I122-2
        # Bias strength now controlled by hilbert_bias_scale × √d_k
```

---

## 5.13 Bias Mode Comparison

| Mode | Parameters | Complexity | Geometric Meaning |
|:-----|:-----------|:-----------|:------------------|
| `lca` | ~100 | $O(N^2)$ | Explicit (LCA depth) |
| `affine_modulation` | ~1K | $O(N^2)$ | Area-aware spatial bias |
| `shape_scale` | ~1K | $O(N^2)$ | Geometry-aware bias |
| `hierarchical_soft_hard` | ~100 | $O(N^2)$ | Hard locality guarantee |

**Recommendation**: Use `lca` for efficiency, `hierarchical_soft_hard` with `DeterministicNeighborSplitter` for best Hilbert locality.

---

## 5.14 Cache Optimization (I30-9)

For efficiency, LCA depth computation is cached across transformer layers:

**Cache Key**: `data_ptr` + `torch_version`

| Benefit | Description |
|:---------|:------------|
| LCA Computation | Eliminates redundant computation per layer |
| Speedup | ~6× speedup for 6-layer transformers |
| Invalidation | Automatic on tensor modification |

```python
# Cache is automatically managed
lca_bias = LCAHilbertBias(max_depth=8, heads=8)

# Manually clear cache if needed
lca_bias.clear_cache()
```

---

## 5.15 Document Navigation

| Chapter | Content |
|:--------|:--------|
| [05_attention_mechanism](05_attention_mechanism.md) | Manifold-native attention (this chapter) |
| [06_feedforward_network](06_feedforward_network.md) | Feed-forward networks |
| [07_transformer_encoder](07_transformer_encoder.md) | Transformer blocks |

> **Next**: [06_feedforward_network.md](06_feedforward_network.md) - Feed-Forward Networks
