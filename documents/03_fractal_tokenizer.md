# Chapter 3: Fractal Tokenizer

## 3.1 Overview

The `StreamingFractalTokenizerV3` implements **Variable Depth Tokenization** via adaptive quadtree splitting and Hilbert curve reordering. It adopts the **Gumbel-Top-K (Scheme D)** mechanism to ensure 100% gradient coverage and parallel execution, replacing earlier BFS-based approaches.

---

## 3.2 Mathematical Formulation

### 3.2.1 Tokenization Pipeline

$$I \xrightarrow{\text{SharedConv}} F \xrightarrow{\text{ParallelEval}} \{S_i, \text{logits}_i\}_{i=1}^{N_{cand}} \xrightarrow{\text{TopK}} \{R_j\}_{j=1}^{K} \xrightarrow{\text{Consist}} \{T_k\} \xrightarrow{\text{Sort}}$$

where:
- $I \in \mathbb{R}^{C \times H \times W}$: Input image
- $N_{cand}$: Total number of candidate quadtree regions ($\sum_{d=0}^{D} 4^d$) (e.g., 85 for D=3)
- $R_j$: Selected regions
- $T_k$: Final consistent set of tokens (no overlaps)

### 3.2.2 Gumbel-Top-K Decision (Scheme D)

Instead of making threshold decisions per region, we evaluate all candidates in parallel and select the Top-K with the highest scores.

**Decision Logits**:

$$\text{logits}_i = \text{MLP}(\text{ROI}(F, R_i)) + b_{explore} + b_{log}(d_i) + \beta \cdot \gamma^{d_i} - \tau_{d_i}$$

where:
- $b_{log}(d_i) = \log(N_{total} / N_{d_i})$: **Log-Compensation Bias** (I21) to balance selection probability across depths.
- $\tau_{d_i}$: Learnable per-depth threshold.
- $b_{explore}$: Annealed exploration bias.
- $\gamma^{d_i}$: Depth penalty term (optional).

**Stochastic Selection**:

$$z_i = \text{logits}_i + g_i, \quad g_i \sim \text{Gumbel}(0, 1)$$
$$\text{selected\_indices} = \text{TopK}(\{z_i/\tau\}_{i=1}^{N_{cand}}, K)$$

This formulation provides gradients for **all** candidates via the Straight-Through Estimator (STE), unlike thresholding which kills gradients for rejected regions.

### 3.2.3 Tree Consistency

The raw Top-K selection may violate the tree structure (e.g., selecting both a parent and its child). We enforce consistency via a vectorized operation:

$$\text{consistent}(i) \iff i \in \text{TopK} \land \forall c \in \text{children}(i), c \notin \text{TopK}$$

This insures that if a parent is selected, its children are ignored, maintaining a valid partition (or subset thereof).

---

## 3.3 Splitting Schemes

### 3.3.1 Scheme D: Gumbel-Top-K (Current)

| Feature | Description |
|:--------|:------------|
| **Parallelism** | 100% (All regions evaluated in one batch) |
| **Gradient Flow** | 100% (STE allows gradients to flow to unselected regions) |
| **Hilbert Locality** | 100% (Strict adherence to Hilbert curve ordering) |
| **Complexity** | $O(N_{cand})$ parallel evaluation |

*Note: Previous schemes A (BFS), B (Relaxation), and C (Fixed Budget) are deprecated.*

---

## 3.4 Token Embedding

### 3.4.1 HilbertNativePatchEmbed

Region-to-token embedding satisfying four constraints:

| Constraint | Description | Implementation |
|:-----------|:------------|:---------------|
| **C1** | Scale equivariance | Shared conv + adaptive pooling |
| **C2** | Depth awareness | Learnable depth modulation |
| **C3** | Hilbert compatibility | Quadtree path preserved |
| **C4** | Differentiability | ROI-Align for smooth gradients |

**Mathematical formulation**:

$$t_i = \text{Pool}(F[R_i]) \cdot \sigma_d + E_d$$

where:
- $F$: Shared convolutional features
- $\text{Pool}$: Adaptive pooling to fixed size
- $\sigma_d$: Depth-dependent scale (learnable)
- $E_d$: Depth embedding

### 3.4.2 ROI-Align

For smooth gradient flow across region boundaries:

$$\text{ROIAlign}(F, R) = \text{BilinearInterpolate}(F, \text{SamplePoints}(R))$$

This avoids quantization artifacts from integer rounding.

---

## 3.5 Hilbert Reordering

After splitting and embedding, tokens are sorted by their Hilbert curve index:

$$\pi(i) = \text{argsort}(H^{-1}(\text{center}(R_i)))$$

### Isomorphism

$$\text{QuadtreePath}(R) = [q_1, \ldots, q_d] \iff \text{HilbertSegment}(R) = H|_{[a,b]}$$

This ensures that:
1. Spatially adjacent regions have nearby sequence positions
2. LCA relationships are preserved for attention bias

---

## 3.6 Implementation

### Class: StreamingFractalTokenizerV3

```python
class StreamingFractalTokenizerV3(BaseTokenizer):
    def __init__(
        self,
        image_size: int = 224,
        d_model: int = 384,
        config: Optional[AdaptiveSplitConfig] = None,
        # ...
    ):
        ...
```

The core logic is delegated to `GumbelTopKSplitter`.

### Diagnostic Metrics

- **Scale Entropy**: Measures diversity of selected depths.
- **Top-K Overlap**: (Debug) How often parents and children are both in Top-K (before consistency check).

---

## 3.7 Usage Example

```python
from vit_pytorch import StreamingFractalTokenizerV3

tokenizer = StreamingFractalTokenizerV3(
    image_size=224,
    d_model=384,
)

# Returns TokenizerOutput with aligned tokens and level info
output = tokenizer.tokenize(images)
```

> **Next**: [04_positional_embedding.md](04_positional_embedding.md) - Position Encoding
