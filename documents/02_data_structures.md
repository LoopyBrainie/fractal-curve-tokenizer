# Chapter 2: Core Data Structures

## 2.1 Overview

This chapter defines the fundamental data structures that flow through the Fractal Curve ViT pipeline.

---

## 2.2 TokenizerOutput

The unified output structure from all tokenizers.

### Definition

```python
@dataclass
class TokenizerOutput:
    sequences: List[TokenSequence]  # Per-image token sequences
```

### TokenSequence

```python
@dataclass  
class TokenSequence:
    tokens: Tensor           # (N, D) - token embeddings
    attention_mask: Tensor   # (N,) - valid token mask
    metadata: Dict[str, Any] # Additional information
    
    def get_levels(self) -> Tensor:
        """Extract levels_info from metadata."""
        return self.metadata.get('levels', None)
```

### Access Patterns

```python
# Per-image access
for seq in output.sequences:
    tokens = seq.tokens          # (N_i, D)
    levels = seq.get_levels()    # (N_i, max_depth+1)
    
# Batch access (with padding)
tokens, levels, mask = output.to_batch()  # (B, N_max, D), (B, N_max, Info), (B, N_max)
```

---

## 2.3 levels_info Tensor

The hierarchical position encoding for each token.

### Shape

$$L \in \mathbb{Z}^{B \times N \times (d_{max} + 1)}$$

### Structure

| Index | Content | Range | Description |
|:------|:--------|:------|:------------|
| `[:, :, 0]` | Depth | $[0, d_{max}]$ | Quadtree depth of the token |
| `[:, :, 1:]` | Path | $[0, 3]^{d_{max}}$ | Quadtree path (quadrant indices) |

### Quadrant Encoding

```
Quadrant indices (Hilbert-compatible):
    
    ┌─────┬─────┐
    │  2  │  3  │
    ├─────┼─────┤
    │  0  │  1  │
    └─────┴─────┘
```

### Mathematical Interpretation

For a token at depth $d$ with path $[q_1, q_2, \ldots, q_d]$:

$$\text{position}(t) = \sum_{i=1}^{d} q_i \cdot 4^{d-i}$$

This maps bijectively to a Hilbert curve segment.

### Example

```python
# Token at depth 2, path [1, 3] (bottom-right → top-right)
levels_info[b, t] = [2, 1, 3, 0, 0, 0]
#                    ^  ^  ^  ^^^^^^^
#                    |  |  |  padding (unused)
#                    |  |  └── q_2 = 3
#                    |  └───── q_1 = 1
#                    └──────── depth = 2
```

---

## 2.4 FractalConfig

Unified configuration dataclass for the entire system.

### Definition

```python
@dataclass
class FractalConfig:
    # Model dimensions
    d_model: int = 384
    num_heads: int = 6
    
    # Tokenizer configuration
    image_size: int = 224
    min_patch_size: int = 4
    max_depth: int = 4
    
    # Hilbert bias configuration
    hilbert_bias_mode: str = 'lca'  # 'lca', 'low_rank', 'hierarchical'
    low_rank_r: int = 32
    
    # Attention parameters
    lca_temperature: float = 1.5
    learnable_temperature: bool = True
    
    # FFN configuration
    ffn_type: str = 'swiglu_level'
```

### Derived Properties

```python
@property
def num_scales(self) -> int:
    """Number of scales in the quadtree."""
    return self.max_depth + 1

@property
def patch_sizes(self) -> Tuple[int, ...]:
    """Available patch sizes from fine to coarse."""
    return tuple(self.min_patch_size * (2 ** i) for i in range(self.num_scales))
```

---

## 2.5 AdaptiveSplitConfig

Configuration for content-adaptive quadtree splitting.

### Complexity Function Parameters

| Parameter | Symbol | Default | Description |
|:----------|:-------|:--------|:------------|
| `alpha` | $\alpha$ | 0.5 | Variance weight in $[0, 1]$ |
| `sigma_0_sq` | $\sigma_0^2$ | 0.01 | Variance normalization constant |
| `g_0_sq` | $g_0^2$ | 0.08 | Gradient normalization constant |

### Threshold Function Parameters

| Parameter | Symbol | Default | Description |
|:----------|:-------|:--------|:------------|
| `tau_0` | $\tau_0$ | 0.15 | Root threshold |
| `gamma` | $\gamma$ | 0.85 | Threshold decay factor |
| `max_depth` | $d_{max}$ | 4 | Maximum split depth |

### Splitting Scheme

```python
class SplitScheme(Enum):
    BALANCED_GREEDY = "balanced_greedy"  # Scheme B: Greedy with 2:1 balance
    FIXED_BUDGET_DP = "fixed_budget_dp"  # Scheme C: DP with token budget
    LEARNABLE = "learnable"              # Scheme L: End-to-end learnable
```

---

## 2.6 QuadtreeNode

Internal representation of a quadtree node during splitting.

### Definition

```python
@dataclass
class QuadtreeNode:
    x: int              # Top-left x coordinate
    y: int              # Top-left y coordinate
    size: int           # Region size (pixels)
    depth: int          # Quadtree depth
    path: List[int]     # Quadrant path from root
    complexity: float   # Computed complexity C(R)
    
    @property
    def region(self) -> Tuple[int, int, int, int]:
        """Return (x, y, x+size, y+size) bounding box."""
        return (self.x, self.y, self.x + self.size, self.y + self.size)
```

### Invariants

1. **Size constraint**: $\text{size} = \text{image\_size} / 2^{\text{depth}}$
2. **Path length**: $\text{len(path)} = \text{depth}$
3. **Alignment**: $(x, y)$ aligned to $\text{size}$-pixel grid

---

## 2.7 HilbertIndex

Mapping between 2D coordinates and Hilbert curve positions.

### Mathematical Definition

$$H: [0, n^2) \leftrightarrow [0, n) \times [0, n)$$

### Implementation

```python
class HilbertCurve:
    def __init__(self, order: int):
        """Initialize Hilbert curve of given order.
        
        Args:
            order: Log2 of grid size (e.g., order=4 → 16×16 grid)
        """
        self.order = order
        self.n = 2 ** order
        
    def d2xy(self, d: int) -> Tuple[int, int]:
        """Convert Hilbert index to (x, y) coordinates."""
        ...
        
    def xy2d(self, x: int, y: int) -> int:
        """Convert (x, y) coordinates to Hilbert index."""
        ...
```

### Locality Property

For any two points $p_1, p_2$:

$$\|p_1 - p_2\|_2 \leq C \cdot |H^{-1}(p_1) - H^{-1}(p_2)|^{1/2}$$

---

## 2.8 Tensor Shape Conventions

### Input/Output Shapes

| Tensor | Shape | Description |
|:-------|:------|:------------|
| Image | $(B, C, H, W)$ | Input image batch |
| Tokens | $(B, N, D)$ | Token embeddings |
| Levels | $(B, N, d_{max}+1)$ | Level information |
| Attention Mask | $(B, 1, 1, N)$ | Broadcast-compatible mask |
| Hilbert Bias | $(B, H, N, N)$ | Per-head attention bias |
| Logits | $(B, C_{out})$ | Classification output |

### Dimension Notation

| Symbol | Meaning | Typical Value |
|:-------|:--------|:--------------|
| $B$ | Batch size | 32 |
| $C$ | Image channels | 3 |
| $H, W$ | Image height/width | 224 |
| $N$ | Number of tokens | 16-196 |
| $D$ | Model dimension | 384 |
| $H$ | Number of heads | 6 |
| $d_{max}$ | Maximum depth | 4 |

> **Next**: [03_fractal_tokenizer.md](03_fractal_tokenizer.md) - Tokenization Pipeline
