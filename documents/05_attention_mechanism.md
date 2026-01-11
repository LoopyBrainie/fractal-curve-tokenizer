# Chapter 5: Attention Mechanism

## 5.1 Overview

The `HilbertAwareMultiScaleAttention` extends standard multi-head attention with **Hilbert curve-derived attention biases**, encoding spatial proximity and hierarchical relationships explicitly.

---

## 5.2 Mathematical Formulation

### 5.2.1 Standard Multi-Head Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

### 5.2.2 Hilbert-Aware Attention

$$\text{HilbertAttn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} \cdot \sigma_{scale} + B_{hilbert} + B_{level}\right) V$$

where:
- $\sigma_{scale}$: Learnable depth-dependent scaling factor
- $B_{hilbert}$: LCA-based Hilbert curve bias
- $B_{level}$: Relative level depth bias

This formulation allows the model to differentiate between spatial relationships (Hilbert) and scale relationships (Level).

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

**LCA Computation**:
Instead of relying on fragile index arithmetic, we compute LCA directly from region coordinates or path vectors:
$$\text{Path}(R) = [q_1, \dots, q_d]$$
$$\text{LCA}(i, j) = \text{Length}(\text{CommonPrefix}(\text{Path}(i), \text{Path}(j)))$$

*Note: Low-Rank Hilbert Bias and Hierarchical Hilbert Bias have been removed in favor of the more efficient and geometrically meaningful LCA Bias.*

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

## 5.5 Bias Mode Comparison

| Mode | Parameters | Complexity | Geometric Meaning |
|:-----|:-----------|:-----------|:------------------|
| `lca` | ~100 | $O(N^2)$ | ✓ Explicit (LCA depth) |
| **Old Modes** | | | **Deprecated** |

**Recommendation**: Use `lca` mode (default) for all models.

---

## 5.6 Implementation

### Class: HilbertAwareMultiScaleAttention

```python
class HilbertAwareMultiScaleAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        bias_mode: str = 'lca',
        lca_temperature: float = 1.5,
        learnable_temperature: bool = True,
        # ...
    ):
        ...
```

The `forward` method automatically normalizes `levels_info` to ensure compatibility between batch and non-batch execution paths.

### Forward Pass

```python
    # ... inside forward ...
    # Add Hilbert bias
    hilbert_bias = self.hilbert_bias(levels_info)
    if hilbert_bias is not None:
        scores = scores + hilbert_bias
    
    # Add level bias
    level_bias = self._get_level_bias(levels_info)
    scores = scores + level_bias
```

---

## 5.7 LCA Hilbert Bias Details

### Class: LCAHilbertBias

The bias computation is vectorized for efficiency:

```python
class LCAHilbertBias(HilbertBiasBase):
    def __init__(self, ...):
        # Uses P6-2 learnable temperature
        self._init_temperature(lca_temperature, learnable_temperature)
    
    def _compute_lca_depths(self, levels_info):
        # ... comparison logic ...
        matches = (path_i == path_j)
        cumulative_match = matches.cumprod(dim=-1)
        lca_depths = cumulative_match.sum(dim=-1)
        return lca_depths
```

---

## 5.8 Usage Example

```python
from vit_pytorch import HilbertAwareMultiScaleAttention

attn = HilbertAwareMultiScaleAttention(
    dim=384,
    heads=6,
    bias_mode='lca',
    lca_temperature=1.5,
)

# levels_info: [B, N, max_depth+1] containing depth + quadtree path
x = torch.randn(2, 100, 384)
out = attn(x, levels_info)
```

> **Next**: [06_feedforward_network.md](06_feedforward_network.md) - Feed-Forward Networks
