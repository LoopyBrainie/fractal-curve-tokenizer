# Chapter 2: Core Mathematical Foundations

## 2.1 Overview

This chapter describes the mathematical primitives that underpin the Fractal Curve Tokenizer system. The architecture relies on the isomorphism between quadtree structures and the Hilbert space-filling curve to transform 2D spatial information into 1-dimensional sequences while preserving maximum locality.

---

## 2.2 Hilbert Curve Algorithms

The system implements several variants of the Hilbert curve to handle different hardware constraints and image geometries.

### 2.2.1 FastBitwiseHilbert

A **vectorized implementation** using Gray-code and Morton encoding (bit-interleaving) to perform 2D-to-1D mapping without Python loops.

**Key Features**:
- Bitwise operations for O(1) per-element conversion
- No recursion overhead
- Optimized for GPU tensor operations

**Source**: `src/vit_pytorch/core/fast_bitwise_hilbert.py`

### 2.2.2 HilbertCurve (Butz Algorithm)

The standard recursive implementation used for square grids where $N = 2^k$.

```python
class HilbertCurve:
    def __init__(self, level: int):
        self.level = level
        self.n = 2 ** level

    def d_to_xy(self, d: int) -> Tuple[int, int]:
        """Convert Hilbert index to 2D coordinates."""
        # Butz algorithm implementation
        ...

    def xy_to_d(self, x: int, y: int) -> int:
        """Convert 2D coordinates to Hilbert index."""
        ...
```

**Source**: `src/vit_pytorch/core/curve_hilbert.py`

### 2.2.3 PseudoHilbertCurve

Extends the locality benefits to **non-square or non-power-of-two grids**. Useful for arbitrary image dimensions.

### 2.2.4 HilbertTopologyCache

A **lazy precomputation module** that stores `coord_to_idx` and `idx_to_coord` tensors to avoid redundant calculations during training.

```python
class HilbertTopologyCache:
    def __init__(self, max_level: int = 8):
        self.max_level = max_level
        self._coord_to_idx = {}  # Lazy initialization
        self._idx_to_coord = {}

    def get_coords(self, indices: Tensor) -> Tensor:
        """Convert Hilbert indices to 2D coordinates."""
        if self.max_level not in self._idx_to_coord:
            self._precompute(self.max_level)
        return self._idx_to_coord[self.max_level][indices]
```

**Cache Key**: `data_ptr` + `torch_version` for automatic invalidation

**Source**: `src/vit_pytorch/core/hilbert_topology_cache.py`

---

## 2.3 Quadtree and LevelsInfo Data Structures

### 2.3.1 LevelsInfo Tensor

The hierarchical nature of fractal tokenization is represented by the `LevelsInfo` data structure, which flattens a quadtree into a tensor format suitable for GPU processing.

**Tensor Definition**:

$$L \in \mathbb{Z}^{B \times N \times (D+1)}$$

where:

| Index | Content | Range | Description |
|:------|:--------|:------|:------------|
| `L[:, :, 0]` | Depth | $[0, D_{max}]$ | Quadtree depth of the token |
| `L[:, :, 1:]` | Path | $[0, 3]^{D_{max}}$ | Quadrant path (quadrant indices) |

### 2.3.2 Quadrant Encoding

```
Quadrant indices (Hilbert-compatible):
    ┌─────┬─────┐
    │  2  │  3  │
    ├─────┼─────┤
    │  0  │  1  │
    └─────┴─────┘
```

### 2.3.3 Coordinate Reconstruction

Tokens are reconstructed into 2D coordinates from their `LevelsInfo` paths using vectorized bitwise operations:

$$x = \sum_{k=0}^{d-1} \text{bit}_k(q_k, 0) \cdot 2^{max\_level-k-1}$$

$$y = \sum_{k=0}^{d-1} \text{bit}_k(q_k, 1) \cdot 2^{max\_level-k-1}$$

This avoids the $O(N \cdot D)$ serial overhead of traditional quadtree traversal.

### 2.3.4 LCA Matrix

The system computes the **Lowest Common Ancestor (LCA)** between any two tokens to determine their relative geometric bias in attention mechanisms:

$$\text{LCA}(i, j) = \text{Length}(\text{CommonPrefix}(\text{Path}(i), \text{Path}(j)))$$

The LCA depth provides a measure of spatial proximity in the quadtree hierarchy.

### 2.3.5 Precomputed LUTs

For `torch.compile` compatibility, the system uses **eager initialization** of Hilbert Look-Up Tables (LUTs) up to depth 8.

```python
# Precomputed LUTs for fast lookup
_LUT_DEPTH_8 = self._build_lut(max_level=8)
```

**Source**: `src/vit_pytorch/core/levels_info.py`

---

## 2.4 Configuration and Splitter Protocols

### 2.4.1 FractalConfig

Automatically derives image geometry, determining `max_level` and whether to use `PseudoHilbertCurve` based on input resolution.

```python
@dataclass
class FractalConfig:
    image_size: int | Tuple[int, int]
    min_patch_size: int = 4
    max_level_limit: int = 8
    tokenizer_type: str = 'streaming_v3'

    def __post_init__(self):
        # Auto-derive max_level from image size
        if isinstance(self.image_size, int):
            self.max_level = int(np.log2(self.image_size // self.min_patch_size))
        else:
            self.max_level = min(
                int(np.log2(self.image_size[0] // self.min_patch_size)),
                int(np.log2(self.image_size[1] // self.min_patch_size))
            )
```

### 2.4.2 CoreSplitter Protocol

Standardized interface for different splitting strategies:

```python
class CoreSplitter(Protocol):
    @property
    def num_candidates(self) -> int:
        """Number of candidate regions."""
        ...

    def forward(
        self,
        features: Tensor,
        image_size: Tuple[int, int]
    ) -> SplitResult:
        """Perform splitting decision."""
        ...

    def update_candidates(self, image_size: Tuple[int, int]) -> None:
        """Update candidate regions for new image size."""
        ...
```

### 2.4.3 Tree Consistency Axioms (H1SS)

The `HilbertOptimalSplitter` follows six axioms to ensure valid quadtree splitting:

| Axiom | Description |
|:------|:------------|
| **A1** | Coverage: Root covers entire image |
| **A2** | Containment: Each child ⊆ parent |
| **A3** | Disjointness: Sibling regions don't overlap |
| **A4** | Tree Depth: Max depth ≤ $D_{max}$ |
| **A5** | Monotonicity: Parent selected ⇒ at least one child selected |
| **A6** | Continuity: Selected regions form connected subgraph |

---

## 2.5 Mathematical Concepts to Code Entities Mapping

```
┌─────────────────────────────────────────────────────────────────┐
│              Mathematical Concept Space                          │
├─────────────────────────────────────────────────────────────────┤
│  Hilbert Curve Mapping  │  Quadtree Decomposition  │  Spatial Indexing  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ uses
┌─────────────────────────────────────────────────────────────────┐
│                 Code Entity Space (L1 Foundation)               │
├─────────────────────────────────────────────────────────────────┤
│  FastBitwiseHilbert         │  HilbertCurve        │  HilbertTopologyCache  │
│  [core/fast_bitwise_hilbert.py]  [core/curve_hilbert.py]  [core/hilbert_topology_cache.py]  │
├─────────────────────────────────────────────────────────────────┤
│  LevelsInfo                      │  PatternEncoder  │  SplitterProtocol  │
│  [core/levels_info.py]              [core/pattern_encoder.py]  [core/splitter_protocol.py]  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2.6 SDS Metric (Spatial Discontinuity Score)

The **SDS (Structure Distortion Score)** validates Hilbert locality preservation:

$$\text{SDS} = \frac{1}{N^2} \sum_{i,j} \left| \|p_i - p_j\|_2 - \frac{|h_i - h_j|}{H_{\max}} \right|$$

where:

- $p_i, p_j$: 2D physical coordinates
- $h_i, h_j$: Hilbert indices
- $H_{\max}$: Maximum Hilbert index for the grid

**Interpretation**:

- SDS ≈ 0: Perfect locality preservation (spatial distance ∝ Hilbert distance)
- SDS >> 0: Locality distortion (random ordering)

**Source**: `src/vit_pytorch/core/curve_hilbert.py` - `SDSMetric` class

---

## 2.7 Key Abbreviations

| Abbreviation | Full Term | Context |
|:------------|:---------|:--------|
| **SDS** | Spatial Discontinuity Score | Hilbert locality validation |
| **LUT** | Look-Up Table | Precomputed Hilbert mappings |
| **LCA** | Lowest Common Ancestor | Quadtree hierarchy |
| **H1SS** | Hilbert-Optimal Splitter (6 Axioms) | Default splitter |

---

## 2.8 Document Navigation

| Chapter | Content |
|:--------|:--------|
| [02_data_structures](02_data_structures.md) | Core mathematical foundations (this chapter) |
| [03_fractal_tokenizer](03_fractal_tokenizer.md) | Tokenization pipeline |
| [04_positional_embedding](04_positional_embedding.md) | Position encoding |

> **Next**: [03_fractal_tokenizer.md](03_fractal_tokenizer.md) - Fractal Tokenization Pipeline
