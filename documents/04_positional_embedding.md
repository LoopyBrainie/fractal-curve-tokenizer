# Chapter 4: Positional Embedding

## 4.1 Overview

The `FractalPositionEmbedding` encodes token positions using both **depth** and **quadtree path** information, providing a hierarchical position encoding tailored for variable-depth tokenization.

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

### 4.2.3 Path Embedding

Aggregated quadrant embeddings along the path:

$$E_{path}(p) = \sum_{i=1}^{d} W_{level}[i] \odot W_{quad}[q_i]$$

where:
- $W_{level} \in \mathbb{R}^{d_{max} \times D}$: Level-specific weights
- $W_{quad} \in \mathbb{R}^{4 \times D}$: Quadrant embeddings

### 4.2.4 Fusion Network

Two-layer MLP with residual connection:

$$\text{Fusion}(x) = x + \text{MLP}(x)$$

where $\text{MLP}(x) = W_2 \cdot \text{GELU}(W_1 \cdot x)$.

---

## 4.3 Geometric Interpretation

### 4.3.1 Quadrant Encoding

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

### 4.3.2 Path Uniqueness

Each quadtree path uniquely identifies a spatial region:

$$\text{Region}([q_1, \ldots, q_d]) = \bigcap_{i=1}^{d} \text{Quadrant}(q_i, i)$$

### 4.3.3 Hilbert Compatibility

The path encoding preserves Hilbert curve locality:

$$|H^{-1}(p_i) - H^{-1}(p_j)| \propto \|E_{path}(p_i) - E_{path}(p_j)\|_2$$

---

## 4.4 Implementation

### Class: FractalPositionEmbedding

```python
class FractalPositionEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_level: int = 50,
        use_fusion: bool = True,
    ):
        super().__init__()
        
        # Depth embedding
        self.depth_embedding = nn.Embedding(max_level + 1, dim)
        
        # Path embedding components
        self.level_embedding = nn.Embedding(max_level, dim)
        self.quadrant_embedding = nn.Embedding(4, dim)
        
        # Fusion network
        if use_fusion:
            self.fusion = nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, dim),
            )
        else:
            self.fusion = nn.Identity()
```

### Forward Pass

```python
def forward(self, levels_info: Tensor) -> Tensor:
    """
    Args:
        levels_info: (B, N, max_depth+1) - [depth, q_1, q_2, ..., q_d]
    
    Returns:
        position_embedding: (B, N, D)
    """
    depths = levels_info[:, :, 0]  # (B, N)
    paths = levels_info[:, :, 1:]  # (B, N, max_depth)
    
    # Depth embedding
    depth_emb = self.depth_embedding(depths)  # (B, N, D)
    
    # Path embedding
    path_emb = self._encode_path(paths, depths)  # (B, N, D)
    
    # Combine and fuse
    combined = depth_emb + path_emb
    return combined + self.fusion(combined)
```

---

## 4.5 Comparison with Standard Position Embeddings

| Method | Encoding | Hierarchical | Adaptive |
|:-------|:---------|:-------------|:---------|
| Sinusoidal | $\sin(pos / 10000^{2i/d})$ | ✗ | ✗ |
| Learned 1D | $W[pos]$ | ✗ | ✗ |
| Learned 2D | $W_x[x] + W_y[y]$ | ✗ | ✗ |
| RoPE | Rotation matrices | ✗ | ✗ |
| **Fractal** | $E_{depth}(d) + E_{path}(p)$ | ✓ | ✓ |

### Advantages

1. **Depth awareness**: Tokens at different resolutions receive distinct encodings
2. **Path specificity**: Spatial location encoded via quadtree path
3. **Variable length**: Works with any number of tokens
4. **Fusion flexibility**: MLP can learn complex interactions

---

## 4.6 Usage Example

```python
from vit_pytorch import FractalPositionEmbedding

pos_embedding = FractalPositionEmbedding(
    dim=384,
    max_level=50,
    use_fusion=True,
)

# levels_info: (B, N, max_depth+1)
# Format: [depth, q_1, q_2, ..., q_{max_depth}]
levels_info = torch.zeros(2, 100, 5, dtype=torch.long)
levels_info[:, :, 0] = 2  # All tokens at depth 2
levels_info[:, :, 1] = 1  # First quadrant index
levels_info[:, :, 2] = 3  # Second quadrant index

tokens = torch.randn(2, 100, 384)
tokens_with_pos = tokens + pos_embedding(levels_info)
```

> **Next**: [05_attention_mechanism.md](05_attention_mechanism.md) - Hilbert-Aware Attention
