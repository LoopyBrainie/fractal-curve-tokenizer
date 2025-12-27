# Chapter 5: Attention Mechanism

## 5.1 Overview

The `HilbertAwareMultiScaleAttention` extends standard multi-head attention with **Hilbert curve-derived attention biases**, encoding spatial proximity and hierarchical relationships.

---

## 5.2 Mathematical Formulation

### 5.2.1 Standard Multi-Head Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

### 5.2.2 Hilbert-Aware Attention

$$\text{HilbertAttn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} \cdot \sigma_{scale} + B_{hilbert} + B_{level}\right) V$$

where:
- $\sigma_{scale}$: Level-dependent attention scaling
- $B_{hilbert}$: Hilbert curve-derived bias
- $B_{level}$: Relative level bias

---

## 5.3 Hilbert Bias Modes

### 5.3.1 LCA Hilbert Bias (Recommended)

**Mathematical definition**:

$$B[i,j] = \tau_h \cdot \text{LCAEmbed}(\text{LCA}(i, j))$$

where:
- $\text{LCA}(i, j)$: Lowest Common Ancestor depth in quadtree
- $\text{LCAEmbed}: [0, d_{max}] \to \mathbb{R}^H$: Learnable embedding
- $\tau_h$: Temperature parameter (learnable)

**Properties**:
- Parameter count: ~100 (99.8% reduction vs low-rank)
- Explicit geometric meaning: LCA depth ≈ spatial distance
- Efficient computation via path comparison

**LCA Computation**:

$$\text{LCA}(i, j) = \max\{k : p_i[1:k] = p_j[1:k]\}$$

where $p_i, p_j$ are quadtree paths.

### 5.3.2 Low-Rank Hilbert Bias

**Mathematical definition**:

$$B[i,j] = \phi(p_i)^T \cdot \psi(p_j)$$

where $\phi, \psi: \mathbb{R}^d \to \mathbb{R}^r$ are learnable projections.

**Properties**:
- Parameter count: ~50K
- Complexity: $O(N \cdot r)$ vs $O(N^2)$
- Memory efficient for large sequences

### 5.3.3 Hierarchical Hilbert Bias

**Mathematical definition**:

$$B[i,j] = \sum_{\ell=1}^{L} b^{(\ell)}(q_i^{(\ell)}, q_j^{(\ell)})$$

where $b^{(\ell)}: [0,3]^2 \to \mathbb{R}$ are level-specific bias functions.

**Properties**:
- Interpretable per-level contributions
- Factorized representation

---

## 5.4 Level Bias

### 5.4.1 Relative Level Embedding

$$B_{level}[i,j] = W_{rel}[\text{clamp}(d_i - d_j + L, 0, 2L)]$$

where:
- $d_i, d_j$: Depths of tokens $i, j$
- $L$: Maximum relative depth
- $W_{rel} \in \mathbb{R}^{(2L+1) \times H}$: Learnable embedding

### 5.4.2 Level Scaling

Depth-dependent attention scaling:

$$\sigma_{scale}(d) = \text{LevelScaleEmb}(d)$$

Deeper tokens (finer resolution) use smaller scaling factors.

---

## 5.5 Bias Mode Comparison

| Mode | Parameters | Complexity | Geometric Meaning |
|:-----|:-----------|:-----------|:------------------|
| `lca` | ~100 | $O(N^2)$ | ✓ Explicit (LCA depth) |
| `low_rank` | ~50K | $O(N \cdot r)$ | Learned |
| `hierarchical` | ~1K | $O(N^2 \cdot L)$ | ✓ Per-level |

**Recommendation**: Use `lca` mode for most applications.

---

## 5.6 Implementation

### Class: HilbertAwareMultiScaleAttention

```python
class HilbertAwareMultiScaleAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        max_level: int = 50,
        bias_mode: BiasMode = 'lca',
        low_rank_r: int = 32,
        lca_temperature: Optional[float] = 1.5,
        learnable_temperature: bool = True,
    ):
        ...
```

### Forward Pass

```python
def forward(
    self,
    x: Tensor,
    levels_info: Tensor,
    attention_mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Args:
        x: (B, N, D) - Input tokens
        levels_info: (B, N, max_depth+1) - Level information
        attention_mask: (B, 1, 1, N) - Attention mask
    
    Returns:
        output: (B, N, D) - Attended tokens
    """
    B, N, D = x.shape
    
    # QKV projection
    q, k, v = self.to_qkv(x).chunk(3, dim=-1)
    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
    
    # Attention scores
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dim_head)
    
    # Apply level scaling
    level_scale = self._get_level_scale(levels_info)
    scores = scores * level_scale
    
    # Add Hilbert bias
    hilbert_bias = self.hilbert_bias(levels_info)
    if hilbert_bias is not None:
        scores = scores + hilbert_bias
    
    # Add level bias
    level_bias = self._get_level_bias(levels_info)
    scores = scores + level_bias
    
    # Apply mask
    if attention_mask is not None:
        scores = scores.masked_fill(~attention_mask, float('-inf'))
    
    # Softmax and output
    attn = F.softmax(scores, dim=-1)
    attn = self.dropout(attn)
    
    out = torch.matmul(attn, v)
    out = rearrange(out, 'b h n d -> b n (h d)')
    
    return self.to_out(out)
```

---

## 5.7 LCA Hilbert Bias Details

### Class: LCAHilbertBias

```python
class LCAHilbertBias(HilbertBiasBase):
    def __init__(
        self,
        num_heads: int,
        max_lca_depth: int = 10,
        temperature: float = 1.5,
        learnable_temperature: bool = True,
    ):
        super().__init__()
        
        # LCA depth embedding: max_lca_depth+1 possible depths
        self.lca_embedding = nn.Embedding(max_lca_depth + 1, num_heads)
        
        # Temperature parameter
        if learnable_temperature:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.register_buffer('temperature', torch.tensor(temperature))
```

### LCA Computation (Vectorized)

```python
def _compute_lca_depths(self, levels_info: Tensor) -> Tensor:
    """
    Compute LCA depths for all token pairs.
    
    Args:
        levels_info: (B, N, max_depth+1)
    
    Returns:
        lca_depths: (B, N, N)
    """
    paths = levels_info[:, :, 1:]  # (B, N, max_depth)
    B, N, D = paths.shape
    
    # Compare paths: (B, N, 1, D) vs (B, 1, N, D)
    path_i = paths.unsqueeze(2)
    path_j = paths.unsqueeze(1)
    
    # Find first mismatch
    matches = (path_i == path_j)  # (B, N, N, D)
    cumulative_match = matches.cumprod(dim=-1)  # (B, N, N, D)
    lca_depths = cumulative_match.sum(dim=-1)  # (B, N, N)
    
    return lca_depths
```

---

## 5.8 Usage Example

```python
from vit_pytorch import HilbertAwareMultiScaleAttention

# LCA mode (recommended)
attn_lca = HilbertAwareMultiScaleAttention(
    dim=384,
    heads=6,
    dim_head=64,
    bias_mode='lca',
    lca_temperature=1.5,
    learnable_temperature=True,
)

# Low-rank mode
attn_lr = HilbertAwareMultiScaleAttention(
    dim=384,
    heads=6,
    dim_head=64,
    bias_mode='low_rank',
    low_rank_r=32,
)

x = torch.randn(2, 100, 384)
levels_info = torch.zeros(2, 100, 5, dtype=torch.long)

out = attn_lca(x, levels_info)  # (2, 100, 384)
```

---

## 5.9 Attention Visualization

The Hilbert bias creates structured attention patterns:

```
Without Hilbert Bias:        With LCA Hilbert Bias:
┌─────────────────┐          ┌─────────────────┐
│ ░░░░░░░░░░░░░░░ │          │ ████░░░░░░░░░░░ │
│ ░░░░░░░░░░░░░░░ │          │ ████░░░░░░░░░░░ │
│ ░░░░░░░░░░░░░░░ │          │ ░░░░████░░░░░░░ │
│ ░░░░░░░░░░░░░░░ │    →     │ ░░░░████░░░░░░░ │
│ ░░░░░░░░░░░░░░░ │          │ ░░░░░░░░████░░░ │
│ ░░░░░░░░░░░░░░░ │          │ ░░░░░░░░░░░░███ │
└─────────────────┘          └─────────────────┘
  (Uniform)                    (Hierarchical clusters)
```

Tokens sharing a common ancestor attend more strongly to each other.

> **Next**: [06_feedforward_network.md](06_feedforward_network.md) - Feed-Forward Networks
