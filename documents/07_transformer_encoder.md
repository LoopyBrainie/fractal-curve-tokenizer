# Chapter 7: Transformer Encoder

## 7.1 Overview

The `FractalTransformer` stacks multiple `FractalTransformerBlock` layers with **level-aware normalization**, **Hilbert-aware attention**, and **level aggregation**.

---

## 7.2 Mathematical Formulation

### 7.2.1 Transformer Block

$$x' = x + \text{DropPath}(\text{Attn}(\text{LN}_1(x)))$$
$$x'' = x' + \text{DropPath}(\text{FFN}(\text{LN}_2(x')))$$

### 7.2.2 Level-Aware Layer Normalization

$$\text{LevelNorm}(x, d) = \gamma_d \cdot \frac{x - \mu}{\sigma} + \beta_d$$

where $\gamma_d, \beta_d \in \mathbb{R}^D$ are depth-dependent learnable parameters.

### 7.2.3 Level Aggregation

$$s_d = \sigma(\text{Embed}_{level}(d)) \in (0, 1)^D$$
$$r = W_2 \cdot \text{ReLU}(W_1 \cdot x)$$
$$x' = x + \lambda \cdot (r \odot s_d)$$

where $\lambda$ is a learnable scaling factor.

---

## 7.3 FractalTransformer

### Class Definition

```python
class FractalTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int = 8,
        dim_head: int = 64,
        mlp_dim: int = None,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        max_level: int = 50,
        use_hilbert_bias: bool = True,
        bias_mode: str = 'lca',
        ffn_type: str = 'swiglu_level',
    ):
        ...
```

### Parameters

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `dim` | int | - | Model dimension |
| `depth` | int | - | Number of transformer layers |
| `heads` | int | 8 | Number of attention heads |
| `dim_head` | int | 64 | Dimension per head |
| `mlp_dim` | int | dim × 4 | FFN hidden dimension |
| `dropout` | float | 0.0 | Dropout rate |
| `drop_path` | float | 0.0 | DropPath rate |
| `bias_mode` | str | 'lca' | Hilbert bias mode |
| `ffn_type` | str | 'swiglu_level' | FFN type |

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
        x: (B, N, D) - Input sequence
        levels_info: (B, N, max_depth+1) - Level information
        attention_mask: (B, 1, 1, N) - Attention mask
    
    Returns:
        x: (B, N, D) - Encoded sequence
    """
    # Layer stack
    for layer in self.layers:
        x = layer(x, levels_info, attention_mask)
    
    # Level aggregation
    x = self._apply_level_aggregation(x, levels_info)
    
    # Final normalization
    x = self.final_norm(x)
    
    return x
```

---

## 7.4 FractalTransformerBlock

### Block Structure

```
Input x
    │
    ▼
┌───────────────────────────┐
│  Level-Aware LayerNorm 1  │
└───────────────────────────┘
    │
    ▼
┌───────────────────────────┐
│ HilbertAwareMultiScaleAttn│
│   + LCA/LowRank Bias      │
└───────────────────────────┘
    │
    ▼
┌───────────────────────────┐
│   DropPath + Residual     │
│   x = x + drop(attn) × w₁ │
└───────────────────────────┘
    │
    ▼
┌───────────────────────────┐
│  Level-Aware LayerNorm 2  │
└───────────────────────────┘
    │
    ▼
┌───────────────────────────┐
│ AdaptiveFractalFeedForward│
│   (SwiGLU + Level Adapt)  │
└───────────────────────────┘
    │
    ▼
┌───────────────────────────┐
│   DropPath + Residual     │
│   x = x + drop(ffn) × w₂  │
└───────────────────────────┘
    │
    ▼
Output x
```

### Implementation

```python
class FractalTransformerBlock(nn.Module):
    def forward(
        self,
        x: Tensor,
        levels_info: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # Pre-norm attention
        norm1_x = self._apply_level_aware_norm(
            x, levels_info, self.norm1_gamma, self.norm1_beta, self.default_norm1
        )
        attn_out = self.attention(norm1_x, levels_info, attention_mask)
        x = x + self.drop_path(attn_out * self.residual_weights[0])
        
        # Pre-norm FFN
        norm2_x = self._apply_level_aware_norm(
            x, levels_info, self.norm2_gamma, self.norm2_beta, self.default_norm2
        )
        ff_out = self.ff(norm2_x, levels_info)
        x = x + self.drop_path(ff_out * self.residual_weights[1])
        
        return x
```

---

## 7.5 Level-Aware Layer Normalization

### Implementation

```python
def _apply_level_aware_norm(
    self,
    x: Tensor,
    levels_info: Tensor,
    gamma: nn.Parameter,
    beta: nn.Parameter,
    default_norm: nn.LayerNorm,
) -> Tensor:
    """
    Apply depth-dependent layer normalization.
    
    Each depth level has its own scale (γ) and shift (β) parameters.
    """
    # Extract depths
    depths = extract_depths(levels_info, self.max_level)  # (B, N)
    
    # Get per-token parameters
    gamma_d = gamma[depths]  # (B, N, D)
    beta_d = beta[depths]    # (B, N, D)
    
    # Standard normalization
    x_norm = default_norm(x)  # (B, N, D)
    
    # Apply depth-specific affine transform
    return x_norm * gamma_d + beta_d
```

### Purpose

Different resolution tokens (depths) have different statistical properties. Level-aware normalization allows the model to learn depth-specific transformations.

---

## 7.6 DropPath (Stochastic Depth)

### Mathematical Definition

**Training**:
$$\text{DropPath}(x) = \begin{cases} 0 & \text{with probability } p \\ \frac{x}{1-p} & \text{otherwise} \end{cases}$$

**Inference**:
$$\text{DropPath}(x) = x$$

### Purpose

1. Regularization via random layer dropping
2. Reduces effective network depth during training
3. Acts as implicit ensemble

### Implementation

```python
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        
        return x * mask / keep_prob
```

---

## 7.7 Level Aggregation

After the layer stack, a learnable aggregator combines information across depths:

```python
def _apply_level_aggregation(self, x: Tensor, levels_info: Tensor) -> Tensor:
    """
    Level-aware aggregation (ARCH-R2).
    
    Formula:
        s_d = σ(Embed_level(d))
        r = W₂ · ReLU(W₁ · x)
        x' = x + λ · (r ⊙ s_d)
    """
    depths = extract_depths(levels_info, self.max_level)
    
    # Level-specific scaling
    scale = torch.sigmoid(self._level_aggregator_scale(depths))  # (B, N, D)
    
    # Bottleneck refinement
    refined = self._level_aggregator_bottleneck(x)  # (B, N, D)
    
    # Residual update
    return x + refined * scale * self._aggregator_scale
```

---

## 7.8 Usage Example

```python
from vit_pytorch import FractalTransformer

transformer = FractalTransformer(
    dim=384,
    depth=6,
    heads=6,
    dim_head=64,
    mlp_dim=768,
    dropout=0.1,
    drop_path=0.1,
    bias_mode='lca',
    ffn_type='swiglu_level',
)

x = torch.randn(2, 100, 384)
levels_info = torch.zeros(2, 100, 5, dtype=torch.long)
mask = torch.ones(2, 1, 1, 100, dtype=torch.bool)

output = transformer(x, levels_info, mask)  # (2, 100, 384)
```

> **Next**: [08_fractal_vit_model.md](08_fractal_vit_model.md) - Complete Model
