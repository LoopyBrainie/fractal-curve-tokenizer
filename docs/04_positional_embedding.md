# Chapter 4: Positional Embedding

## 4.1 Overview

The `FractalPositionEmbedding` encodes token positions using both **depth** and **quadtree path** information, providing a hierarchical position encoding tailored for variable-depth tokenization. It supports **Area-Enhanced** encoding (I31-3) for area-aware position representation.

---

## 4.2 Mathematical Formulation

### 4.2.1 Position Encoding Definition

$$E_{pos}(i) = \text{Fusion}(E_{depth}(d_i) + E_{path}(p_i))$$

where:
- $d_i \in [0, d_{max}]$: Depth of token $i$
- $p_i = [q_1, \ldots, q_d]$: Quadtree path of token $i$
- $E_{depth}: \mathbb{Z} \to \mathbb{R}^D$: Depth embedding
- $E_{path}: [0,3]^{d_{max}} \to \mathbb{R}^D$: Path embedding
- $\text{Fusion}: \mathbb{R}^D \to \mathbb{R}^D$: Fusion network

### 4.2.2 Depth Embedding

Learnable embedding indexed by depth:

$$E_{depth}(d) = W_{depth}[d], \quad W_{depth} \in \mathbb{R}^{(d_{max}+1) \times D}$$

### 4.2.3 Path Embedding (STAB-4 Fix)

Aggregated quadrant embeddings along the path with depth normalization:

**Original (Unstable)**:

$$E_{path}(p) = \sum_{i=1}^{d} W_{quad}[L_i \cdot 4 + q_i]$$

This leads to $\|E_{path}\| \propto \sqrt{d}$, causing variance imbalance across depths.

**Normalized (STAB-4)**:

$$E_{path}(p) = \frac{1}{\sqrt{d}} \sum_{i=1}^{d} W_{quad}[L_i \cdot 4 + q_i]$$

where:
- $W_{quad} \in \mathbb{R}^{(d_{max} \cdot 4) \times D}$: Flattened level-quadrant embeddings
- $L_i$: Level index of the $i$-th step
- $q_i$: Quadrant index of the $i$-th step
- $\frac{1}{\sqrt{d}}$: Normalization factor maintaining constant variance across depths

**Masking for Variable Depths**:

Only valid path steps contribute to the sum:

$$\text{mask}[j] = \begin{cases} 1 & \text{if } j < d_i \\ 0 & \text{otherwise} \end{cases}$$
$$E_{path}(p) = \frac{1}{\sqrt{\sum_j \text{mask}[j]}} \sum_j \text{mask}[j] \cdot W_{quad}[j]$$

### 4.2.4 Fusion Network

Two-layer MLP with residual connection:

$$\text{Fusion}(x) = x + \text{MLP}(x)$$

where $\text{MLP}(x) = W_2 \cdot \text{GELU}(W_1 \cdot x)$.

---

## 4.3 Area-Enhanced Position Embedding (I31-3)

The `AreaEnhancedPositionEmbedding` extends the base encoding with area information.

### 4.3.1 Mathematical Formulation

$$E_{pos}(i) = \text{Fusion}(E_{depth}(d_i) + E_{path}(p_i) + \lambda \cdot E_{area}(R_i))$$

where:
- $E_{area}(R_i)$: Area embedding from `AreaEncoder`
- $\lambda$: Learnable scale parameter (zero-initialized)

### 4.3.2 AreaEncoder (Shared with Attention)

The same `AreaEncoder` is used for both position embedding and attention bias:

**Area Score**:

$$f_{area} = \frac{\log(s_{patch} + 1)}{\log(S_{total} + 1)}$$

**Fourier Features**:

$$\gamma(f) = [\sin(2^k \pi f), \cos(2^k \pi f)]_{k=0}^{L-1}$$

**MLP Projection**:

$$E_{area}(R) = \text{MLP}(\gamma(f_{area}))$$

### 4.3.3 Residual Injection

Area embedding is injected via residual connection:

$$E_{pos} = E_{base} + \lambda \cdot E_{area}$$

With $\lambda$ initialized to 0, the model can gradually learn to use area information.

---

## 4.4 Geometric Interpretation

### 4.4.1 Quadrant Encoding

The quadrant indices encode spatial position within each level:

```
Level 0 (root):     Level 1:              Level 2:
┌─────────────┐     ┌──────┬──────┐       ┌───┬───┬───┬───┐
│             │     │  2   │  3   │       │ 2 │ 3 │ 2 │ 3 │
│      0      │  →  ├──────┼──────┤   →   ├───┼───┼───┼───┤
│             │     │  0   │  1   │       │ 0 │ 1 │ 0 │ 1 │
└─────────────┘     └──────┴──────┘       ├───┼───┼───┼───┤
                                          │ 2 │ 3 │ 2 │ 3 │
                                          ├───┼───┼───┼───┤
                                          │ 0 │ 1 │ 0 │ 1 │
                                          └───┴───┴───┴───┘
```

### 4.4.2 Path Uniqueness

Each quadtree path uniquely identifies a spatial region:

$$\text{Region}([q_1, \ldots, q_d]) = \bigcap_{i=1}^{d} \text{Quadrant}(q_i, i)$$

### 4.4.3 Hilbert Compatibility

The path encoding preserves Hilbert curve locality:

$$|H^{-1}(p_i) - H^{-1}(p_j)| \propto \|E_{path}(p_i) - E_{path}(p_j)\|_2$$

---

## 4.5 Implementation

### Class: FractalPositionEmbedding

```python
class FractalPositionEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_level: int = 8,
        max_seq_len: int = 10000,
        use_hilbert_encoding: bool = True,
        use_spatial_encoding: bool = True,
        dropout: float = 0.1,
    ):
        """
        Args:
            dim: Embedding dimension
            max_level: Maximum quadtree depth (P11-2: should match tokenizer.max_depth)
            max_seq_len: Maximum sequence length
            use_hilbert_encoding: Enable Hilbert path encoding
            use_spatial_encoding: Enable spatial encoding
            dropout: Dropout rate for position embedding (I27-2)
        """
```

### Class: AreaEnhancedPositionEmbedding

```python
class AreaEnhancedPositionEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_level: int = 8,
        fourier_levels: int = 4,
        use_hilbert_encoding: bool = True,
        use_spatial_encoding: bool = True,
        dropout: float = 0.1,
    ):
        """
        Args:
            dim: Embedding dimension
            max_level: Maximum quadtree depth
            fourier_levels: Number of Fourier frequency levels
            use_hilbert_encoding: Enable Hilbert encoding
            use_spatial_encoding: Enable spatial encoding
            dropout: Dropout rate
        """
```

### Forward Pass (Base)

```python
def forward(
    self,
    levels_info: torch.Tensor,
    regions: Optional[torch.Tensor] = None,
    image_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Args:
        levels_info: (B, N, max_depth+1) - [depth, q_1, q_2, ..., q_d]
        regions: (B, N, 4) - Region boundaries [x1, y1, x2, y2]
        image_size: int or (W, H) - Image dimensions

    Returns:
        position_embedding: (B, N, D)
    """
    depths = levels_info[..., 0].clamp(0, self.max_level).long()
    paths = levels_info[..., 1:].long()

    # Depth embedding
    depth_emb = self.depth_embedding(depths)

    # Path embedding with normalization (STAB-4)
    level_offsets = torch.arange(paths.shape[-1], device=levels_info.device) * 4
    flat_indices = (paths + level_offsets).clamp(0, self.max_level * 4 - 1)
    path_embs = self.quadrant_embedding(flat_indices)

    # Mask and normalize
    seq_indices = torch.arange(paths.shape[-1], device=levels_info.device)
    mask = seq_indices < depths.unsqueeze(-1)
    path_count = mask.sum(dim=-1, keepdim=True).clamp(min=1).float()
    path_final = (path_embs * mask.unsqueeze(-1)).sum(dim=-2) / torch.sqrt(path_count)

    # Fusion
    combined = depth_emb + path_final
    return self.fusion_network(combined)
```

### Forward Pass (Area-Enhanced)

```python
def forward(
    self,
    levels_info: torch.Tensor,
    regions: Optional[torch.Tensor] = None,
    image_size: Optional[int] = None,
) -> torch.Tensor:
    # Base position encoding
    pos_emb = self.base_embedding(levels_info)

    # Area encoding (if regions provided)
    if regions is not None and image_size is not None:
        area_emb = self.area_encoder(regions, image_size)

        # Handle CLS token alignment
        if area_emb.shape[1] == pos_emb.shape[1] + 1:
            area_emb = area_emb[:, 1:, :]

        # Residual injection
        pos_emb = pos_emb + self.area_scale * area_emb

    return pos_emb
```

---

## 4.6 Comparison with Standard Position Embeddings

| Method | Encoding | Hierarchical | Adaptive | Area-Aware |
|:-------|:---------|:-------------|:---------|:-----------|
| Sinusoidal | $\sin(pos / 10000^{2i/d})$ | No | No | No |
| Learned 1D | $W[pos]$ | No | No | No |
| Learned 2D | $W_x[x] + W_y[y]$ | No | No | No |
| RoPE | Rotation matrices | No | No | No |
| **Fractal** | $E_{depth}(d) + E_{path}(p)$ | Yes | Yes | No |
| **Fractal+Area** | $E_{depth} + E_{path} + \lambda \cdot E_{area}$ | Yes | Yes | Yes |

---

## 4.7 Dropout in Position Embedding (I27-2)

Position embedding acts as an **information bottleneck**, so dropout should be conservative:

**Recommended Configuration**:

| Parameter | Value | Rationale |
|:----------|:------|:----------|
| `dropout` | 0.1 | ~0.5 × main transformer dropout |
| Maximum | 0.2 | Higher values risk position information loss |

**Analysis**:
- $p_{pos} > 0.2$: Position information degradation → model cannot learn spatial relationships
- $p_{pos} < 0.05$: Insufficient regularization → overfitting to specific positions

**Empirical Formula**:

$$p_{pos} \approx 0.5 \times p_{transformer}$$

---

## 4.8 Usage Example

### Basic Configuration

```python
from vit_pytorch import FractalPositionEmbedding

pos_embedding = FractalPositionEmbedding(
    dim=384,
    max_level=8,
    use_hilbert_encoding=True,
    use_spatial_encoding=True,
    dropout=0.1,
)

# levels_info: (B, N, max_depth+1)
# Format: [depth, q_1, q_2, ..., q_{max_depth}]
levels_info = torch.zeros(2, 100, 9, dtype=torch.long)
levels_info[:, :, 0] = 2  # All tokens at depth 2

tokens = torch.randn(2, 100, 384)
tokens_with_pos = tokens + pos_embedding(levels_info)
```

### With Area Enhancement (I31-3)

```python
from vit_pytorch import AreaEnhancedPositionEmbedding

pos_embedding = AreaEnhancedPositionEmbedding(
    dim=384,
    max_level=8,
    fourier_levels=4,
    dropout=0.1,
)

# With regions for area encoding
regions = torch.rand(2, 100, 4)  # [B, N, 4] - [x1, y1, x2, y2]
image_size = 224

tokens_with_pos = tokens + pos_embedding(levels_info, regions=regions, image_size=image_size)
```

---

## 4.9 Attention Bias Integration

The position embedding and attention bias share the same `AreaEncoder`:

```
┌─────────────────────────────────────────────────────────────┐
│                    Position Embedding                        │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐  │
│  │  Depth Emb  │ +  │  Path Emb   │ +  │ Area Emb (opt)  │  │
│  │  [0, D]     │    │  (norm)     │    │ [I31-3]         │  │
│  └─────────────┘    └─────────────┘    └─────────────────┘  │
│                         │                                      │
│                         ▼                                      │
│                   ┌───────────┐                                │
│                   │  Fusion   │                                │
│                   │  Network  │                                │
│                   └───────────┘                                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  Token + Position   │
                    └─────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    Attention Bias                            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐  │
│  │ LCA Embed   │ +  │ Level Bias  │ +  │ Area Mod (opt)  │  │
│  │ [B,H,N,N]   │    │ [B,H,N,N]   │    │ [I31-3]         │  │
│  └─────────────┘    └─────────────┘    └─────────────────┘  │
│                         │                                      │
│                         ▼                                      │
│                   ┌───────────┐                                │
│                   │   Scale   │                                │
│                   │  Factors  │                                │
│                   └───────────┘                                │
└─────────────────────────────────────────────────────────────┘
```

---

## 4.10 Constants Reference

| Constant | Value | Purpose |
|:---------|:------|:--------|
| `EMBEDDING_INIT_STD` | 0.02 | Embedding weight initialization std |
| `HILBERT_BIAS_SCALE` | 1.0 | Attention bias scale |
| `LEVEL_BIAS_SCALE` | 1.0 | Level bias scale |

> **Next**: [05_attention_mechanism.md](05_attention_mechanism.md) - Hilbert-Aware Attention
