# Chapter 7: Transformer Encoder

## 7.1 Overview

The `FractalTransformer` stacks multiple `FractalTransformerBlock` layers with **level-aware normalization**, **Hilbert-aware attention**, and **level aggregation**. It features **depth-aware residual gating** (STAB-5) for adaptive information flow.

---

## 7.2 Mathematical Formulation

### 7.2.1 Transformer Block

$$x' = x + \text{DropPath}(\text{Attn}(\text{LN}_1(x)))$$
$$x'' = x' + \text{DropPath}(\text{FFN}(\text{LN}_2(x')))$$

### 7.2.2 Layer Norm

The implementation uses standard `nn.LayerNorm` for both attention and FFN sub-layers (I106-2):

$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sigma} + \beta$$

Hilbert bias handles scale calibration across different token depths, eliminating the need for depth-dependent normalization parameters.

### 7.2.3 Residual Gate (STAB-5, I34-10)

The residual gate modulates the contribution of attention and FFN outputs using **tanh** activation:

$$g_{raw} \in \mathbb{R}, \quad g = \tanh(g_{raw}) \in [-1, 1]$$
$$x' = x + (1 + g) \cdot \text{DropPath}(\text{Attn}(\text{LN}(x)))$$
$$x'' = x' + (1 + g) \cdot \text{DropPath}(\text{FFN}(\text{LN}(x')))$$

where:
- Gate range: $[0, 2]$ (values $>1$ amplify, $<1$ attenuate)
- $1 + g$ ensures residual connection is always open (avoiding dead paths)

**Rationale**: Using $\tanh$ directly instead of $2 \cdot \sigma$ provides more stable gradient flow for values near 0, and the additive form $(1 + g)$ guarantees non-zero residual paths.

**V-Shaped Gate Pattern (I24-6)**:

Based on training observations:
- $d=0$ (global): $w \approx 0.7$ - Suppress global information
- $d=1$: $w \approx 1.0$ - Keep original
- $d=2$: $w \approx 0.85$ - Slight suppression
- $d=3$ (fine-grained): $w \approx 0.65$ - Suppress details

**Initialization**:

Using inverse sigmoid to achieve target values:

$$g = \log\left(\frac{w/2}{1 - w/2}\right)$$

For $w=0.7$: $g \approx -0.36$

### 7.2.4 Level Aggregation

$$s_d = \sigma(\text{Embed}_{level}(d)) \in (0, 1)^D$$
$$r = W_2 \cdot \text{ReLU}(W_1 \cdot x)$$
$$x' = x + \lambda \cdot (r \odot s_d)$$

where:
- $\lambda$: Learnable scaling factor (initialized to 0.2)
- $\odot$: Element-wise multiplication

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
        max_level: int = 8,
        drop_path_rate: float = 0.1,
        ffn_type: str = 'swiglu_level',
        use_checkpoint: bool = False,
        lca_temperature: float = 1.5,
        learnable_temperature: bool = True,
        use_affine_modulation: bool = False,
        fourier_levels: int = 4,
    ):
        """
        Args:
            dim: Model dimension
            depth: Number of transformer layers
            heads: Number of attention heads
            dim_head: Dimension per head
            mlp_dim: MLP hidden dimension
            dropout: Dropout rate
            drop_path: DropPath rate
            max_level: Maximum quadtree depth (P11-2: should match tokenizer.max_depth)
            drop_path_rate: Maximum DropPath rate (linearly increased)
            ffn_type: FFN type ('gelu', 'swiglu', 'swiglu_level')
            use_checkpoint: Gradient checkpointing for memory efficiency
            lca_temperature: LCA bias temperature
            learnable_temperature: Whether temperature is learnable
            use_affine_modulation: Enable I31-3 area-aware bias
            fourier_levels: Fourier frequency levels for area encoding
        """
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
| `max_level` | int | 8 | Maximum quadtree depth |
| `ffn_type` | str | 'swiglu_level' | FFN type |
| `use_checkpoint` | bool | False | Gradient checkpointing |
| `lca_temperature` | float | 1.5 | LCA bias temperature |
| `learnable_temperature` | bool | True | Learnable temperature |
| `use_affine_modulation` | bool | False | Area-aware bias (I31-3) |
| `fourier_levels` | int | 4 | Fourier frequency levels |

---

## 7.4 FractalTransformerBlock

### Block Structure

```
Input x [B, N, D]
      │
      ▼
┌─────────────────────────────────┐
│  Level-Aware LayerNorm 1        │
│  γ_1(d), β_1(d) per depth       │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│ HilbertAwareMultiScaleAttention │
│ + LCA Bias                      │
│ + Level Bias                    │
│ + Affine Modulation (optional)  │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│   Residual Gate (STAB-5)        │
│   w_1(d) = 2·σ(g_1[d]) ∈ [0,2] │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│   DropPath                      │
│   x = x + w_1 · drop(attn)      │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│  Level-Aware LayerNorm 2        │
│  γ_2(d), β_2(d) per depth       │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│ AdaptiveFractalFeedForward      │
│ (SwiGLU + Level Adaptation)     │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│   Residual Gate (STAB-5)        │
│   w_2(d) = 2·σ(g_2[d]) ∈ [0,2] │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│   DropPath                      │
│   x = x + w_2 · drop(ffn)       │
└─────────────────────────────────┘
      │
      ▼
Output x [B, N, D]
```

### Implementation

```python
class FractalTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        max_level: int = 8,
        drop_path: float = 0.0,
        ffn_type: FFNType = 'swiglu_level',
        lca_temperature: Optional[float] = 1.5,
        learnable_temperature: bool = True,
        use_affine_modulation: bool = False,
        fourier_levels: int = 4,
    ):
        super().__init__()
        self.dim = dim
        self.max_level = max_level

        # Attention
        self.attention = HilbertAwareMultiScaleAttention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            max_level=max_level,
            lca_temperature=lca_temperature,
            learnable_temperature=learnable_temperature,
            use_affine_modulation=use_affine_modulation,
            fourier_levels=fourier_levels,
        )

        # FFN
        self.ff = AdaptiveFractalFeedForward(
            dim=dim,
            hidden_dim=mlp_dim,
            dropout=dropout,
            max_level=max_level,
            ffn_type=ffn_type,
        )

        # STAB-5: Residual Gate (level-aware)
        self._residual_gate = nn.Embedding(max_level + 1, 2)

        # Level-aware LayerNorm
        self.norm1_gamma = nn.Embedding(max_level + 1, dim)
        self.norm1_beta = nn.Embedding(max_level + 1, dim)
        self.norm2_gamma = nn.Embedding(max_level + 1, dim)
        self.norm2_beta = nn.Embedding(max_level + 1, dim)

        # DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
```

### Forward Pass

```python
def forward(
    self,
    x: torch.Tensor,
    levels_info: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    regions: Optional[torch.Tensor] = None,
    image_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Args:
        x: (B, N, D) - Input sequence
        levels_info: (B, N, max_depth+1) - Level information
        attention_mask: (B, 1, 1, N) - Attention mask
        regions: (B, N, 4) - Region boundaries for LCA computation
        image_size: int - Image size for region-based LCA

    Returns:
        x: (B, N, D) - Encoded sequence
    """
    # Extract depths for level-aware operations
    if levels_info is not None and levels_info.numel() > 0:
        depths = extract_depths(levels_info, self.max_level)
        gate_raw = self._residual_gate(depths)
        gate = torch.sigmoid(gate_raw) * 2  # [B, N, 2] ∈ [0, 2]
        w1 = gate[:, :, 0].unsqueeze(-1)
        w2 = gate[:, :, 1].unsqueeze(-1)
    else:
        # Default gate for CLS token
        default_gate = torch.sigmoid(self._residual_gate.weight[0]) * 2
        w1 = default_gate[0]
        w2 = default_gate[1]

    # Level-aware LayerNorm 1
    norm1_x = self._apply_level_aware_norm(
        x, levels_info, self.norm1_gamma, self.norm1_beta, self.default_norm1
    )

    # Attention
    attn_out = self.attention(
        norm1_x,
        levels_info=levels_info,
        attention_mask=attention_mask,
        regions=regions,
        image_size=image_size,
    )

    # Residual with gate
    x = x + self.drop_path(attn_out * w1)

    # Level-aware LayerNorm 2
    norm2_x = self._apply_level_aware_norm(
        x, levels_info, self.norm2_gamma, self.norm2_beta, self.default_norm2
    )

    # FFN
    ff_out = self.ff(norm2_x, levels_info)

    # Residual with gate
    x = x + self.drop_path(ff_out * w2)

    return x
```

---

## 7.5 Level-Aware Layer Normalization

### Implementation

```python
def _apply_level_aware_norm(
    self,
    x: torch.Tensor,
    levels_info: Optional[torch.Tensor],
    gamma_emb: nn.Embedding,
    beta_emb: nn.Embedding,
    default_norm: nn.LayerNorm,
) -> torch.Tensor:
    """
    Apply depth-dependent layer normalization.

    Each depth level has its own scale (γ) and shift (β) parameters.

    P11-16: When levels_info is None, use depth 0 as default.
    """
    if x.dim() != 3:
        raise ValueError(f"Expected x to be 3D [B, S, D], got {x.dim()}D")

    # Handle None levels_info (P11-16)
    if levels_info is None or levels_info.numel() == 0:
        depths = torch.zeros(x.shape[0], x.shape[1], dtype=torch.long, device=x.device)
    elif levels_info.dim() == 2:
        depths = extract_depths(levels_info, self.max_level)
        gamma = gamma_emb(depths).unsqueeze(0)
        beta = beta_emb(depths).unsqueeze(0)
    else:
        depths = extract_depths(levels_info, self.max_level)
        gamma = gamma_emb(depths)
        beta = beta_emb(depths)

    # Manual LayerNorm
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (x - mean) / torch.sqrt(var + 1e-5)

    return x_norm * gamma + beta
```

### Purpose

Different resolution tokens (depths) have different statistical properties:
- **Shallow tokens** (large regions): Higher variance, more global information
- **Deep tokens** (small regions): Lower variance, more local details

Level-aware normalization allows the model to learn depth-specific transformations.

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
3. Acts as implicit ensemble of networks of varying depth
4. Linearly increasing drop rate: $p_l = \frac{l}{L} \cdot p_{max}$

### Implementation

```python
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)

        if self.scale_by_keep:
            random_tensor.div_(keep_prob)

        return x * random_tensor
```

---

## 7.7 Level Aggregation

After the layer stack, a learnable aggregator combines information across depths:

### Mathematical Formulation

$$s_d = \sigma(\text{Embed}_{level}(d)) \in (0, 1)^D$$
$$r = W_2 \cdot \text{ReLU}(W_1 \cdot x)$$
$$x' = x + \lambda \cdot (r \odot s_d)$$

### Implementation

```python
class FractalTransformer(nn.Module):
    def _apply_level_aggregation(self, x: torch.Tensor, levels_info: torch.Tensor) -> torch.Tensor:
        """
        Level-aware aggregation (ARCH-R2).

        Formula:
            s_d = σ(Embed_level(d))
            r = W_2 · ReLU(W_1 · x)
            x' = x + λ · (r ⊙ s_d)
        """
        depths = extract_depths(levels_info, self.max_level)
        scale = torch.sigmoid(self._level_aggregator_scale(depths))

        # Adjust shape for broadcasting
        if scale.dim() == 2:
            scale = scale.unsqueeze(0)

        refined = self._level_aggregator_bottleneck(x)
        aggregated = refined * scale

        return x + aggregated * self._aggregator_scale
```

---

## 7.8 Gradient Checkpointing

For memory-efficient training on long sequences:

```python
def forward(
    self,
    x: torch.Tensor,
    levels_info: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    regions: Optional[torch.Tensor] = None,
    image_size: Optional[int] = None,
) -> torch.Tensor:
    batch_size, seq_len, dim = x.shape

    for layer in self.layers:
        if self.use_checkpoint and self.training:
            # Recompute activations to save memory
            x = checkpoint(
                layer,
                x,
                levels_info,
                attention_mask,
                regions,
                image_size,
                use_reentrant=False,
            )
        else:
            x = layer(
                x,
                levels_info=levels_info,
                attention_mask=attention_mask,
                regions=regions,
                image_size=image_size,
            )

    # Level aggregation
    if levels_info is not None and levels_info.numel() > 0:
        x = self._apply_level_aggregation(x, levels_info)

    x = self.final_norm(x)
    return x
```

---

## 7.9 Usage Example

```python
from vit_pytorch import FractalTransformer

transformer = FractalTransformer(
    dim=384,
    depth=6,
    heads=6,
    dim_head=64,
    mlp_dim=768,
    dropout=0.1,
    drop_path_rate=0.1,
    ffn_type='swiglu_level',
    use_checkpoint=False,
    lca_temperature=1.5,
    learnable_temperature=True,
)

x = torch.randn(2, 100, 384)
levels_info = torch.zeros(2, 100, 9, dtype=torch.long)
mask = torch.ones(2, 1, 1, 100, dtype=torch.bool)
regions = torch.rand(2, 100, 4)
image_size = 224

output = transformer(x, levels_info, mask, regions=regions, image_size=image_size)
# Output: (2, 100, 384)
```

---

## 7.10 Numerical Stability Constants

All constants are centralized in `constants.py`:

| Constant | Value | Purpose |
|:---------|:------|:--------|
| `GUMBEL_EPSILON` | 1e-8 | Gumbel noise stability |
| `LOG_EPSILON` | 1e-8 | Logarithm stability |
| `DIVISION_EPSILON` | 1e-8 | Division stability |
| `PROB_EPSILON` | 1e-5 | Probability clamping |
| `LAYER_NORM_EPS` | 1e-5 | LayerNorm variance |
| `SPLITTER_TEMP_START` | 1.0 | Initial Gumbel temperature |
| `SPLITTER_TEMP_END` | 0.5 | Final Gumbel temperature |
| `TEMPERATURE_MIN` | 0.3 | Minimum temperature (gradient explosion below) |
| `DEPTH_KL_WEIGHT` | 0.5 | Depth balance KL weight |
| `DEPTH_QUOTA_TARGET` | - | Quota target distribution |
| `LEARNABLE_QUOTA_ENABLED` | True | Scheme E quota allocation |
| `QUOTA_MIN_PER_DEPTH` | 2 | Minimum quota per depth |
| `THRESHOLD_VAR_REG_WEIGHT` | 0.3 | Threshold variance regularization |

---

## 7.11 Complexity Analysis

### Per Block

| Operation | Time | Space |
|:----------|:-----|:------|
| Attention | $O(B \cdot H \cdot N^2 \cdot d)$ | $O(B \cdot H \cdot N^2)$ |
| FFN | $O(B \cdot N \cdot D \cdot D_{ff})$ | $O(B \cdot N \cdot D_{ff})$ |
| Level Norm | $O(B \cdot N \cdot D)$ | $O(B \cdot N \cdot D)$ |
| Residual Gate | $O(B \cdot N)$ | $O(1)$ |

### Full Transformer (L layers)

| Metric | Value |
|:-------|:------|
| Time | $O(L \cdot B \cdot H \cdot N^2 \cdot d)$ |
| Space (no checkpointing) | $O(L \cdot B \cdot H \cdot N^2)$ |
| Space (with checkpointing) | $O(B \cdot H \cdot N^2 + L \cdot B \cdot N \cdot D)$ |

---

> **Next**: [08_fractal_vit_model.md](08_fractal_vit_model.md) - Complete Model
