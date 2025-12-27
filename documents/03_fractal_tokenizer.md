# Chapter 3: Fractal Tokenizer

## 3.1 Overview

The `StreamingFractalTokenizerV3` implements **Variable Depth Tokenization** via adaptive quadtree splitting and Hilbert curve reordering. This chapter provides a complete mathematical specification.

---

## 3.2 Mathematical Formulation

### 3.2.1 Tokenization Pipeline

$$I \xrightarrow{\text{Split}} \{R_i\}_{i=1}^{N} \xrightarrow{\text{Embed}} \{t_i\}_{i=1}^{N} \xrightarrow{\text{Sort}} \{t_{\pi(i)}\}_{i=1}^{N}$$

where:
- $I \in \mathbb{R}^{C \times H \times W}$: Input image
- $R_i$: Quadtree region (axis-aligned rectangle)
- $t_i \in \mathbb{R}^D$: Token embedding
- $\pi$: Hilbert curve permutation

### 3.2.2 Complexity Function

The splitting decision is based on a normalized complexity measure:

$$C(R) = \alpha \cdot C_{var}(R) + (1 - \alpha) \cdot C_{grad}(R)$$

where:

$$C_{var}(R) = \frac{\text{Var}(R)}{\text{Var}(R) + \sigma_0^2}, \quad C_{grad}(R) = \frac{G(R)}{G(R) + g_0^2}$$

- $\text{Var}(R)$: Pixel variance within region $R$
- $G(R)$: Gradient energy (sum of squared gradients)
- $\alpha \in [0, 1]$: Balance parameter (default: 0.5)
- $\sigma_0^2, g_0^2$: Normalization constants

### 3.2.3 Depth-Dependent Threshold

$$\tau_d = \tau_0 \cdot \gamma^d$$

where:
- $\tau_0$: Root threshold (default: 0.15)
- $\gamma$: Decay factor (default: 0.85)
- $d$: Current depth

**Splitting criterion**:
$$\text{Split}(R) \iff C(R) > \tau_d \land d < d_{max} \land \text{size}(R) \geq \text{min\_size}$$

---

## 3.3 Splitting Schemes

### 3.3.1 Balanced Greedy Splitting (Scheme B)

**Algorithm**:

```
Input: Image I, config cfg
Output: Set of leaf regions {R_i}

1. Initialize priority queue Q with root region
2. While Q is not empty:
   a. Pop region R with highest complexity
   b. If Split(R):
      - Add 4 children to Q
   c. Else:
      - Add R to output set
3. Post-process for 2:1 balance constraint
```

**2:1 Balance Constraint**: Adjacent regions differ by at most 1 level in depth.

$$\forall R_i, R_j \text{ adjacent}: |d_i - d_j| \leq 1$$

This ensures smooth transitions and is enforced via iterative refinement.

### 3.3.2 Fixed Budget DP Splitting (Scheme C)

**Objective**:
$$\min_{\{R_i\}} \sum_{i=1}^{N} C(R_i) \quad \text{s.t.} \quad |\{R_i\}| = K$$

**Algorithm**: Dynamic programming on quadtree structure.

```
Input: Image I, token budget K
Output: Optimal split with exactly K tokens

1. Compute complexity for all possible regions
2. DP on quadtree: dp[node][budget] = min complexity
3. Backtrack to recover optimal split
```

### 3.3.3 Learnable Splitting (Scheme L)

**End-to-end differentiable splitting** via Gumbel-Softmax:

$$\text{SplitProb}(R) = \sigma(\text{MLP}([C_{var}, C_{grad}, d, \ldots]))$$

Temperature-annealed sampling:
$$z = \text{GumbelSoftmax}(\log p, \tau(t))$$

where $\tau(t) = \tau_{max} \cdot (\tau_{min}/\tau_{max})^{t/T}$.

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

### Quadtree-Hilbert Isomorphism

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
        base_patch_size: int = 4,
        max_depth: int = 4,
        split_scheme: str = 'balanced_greedy',
        config: Optional[AdaptiveSplitConfig] = None,
    ):
        ...
    
    def tokenize(self, images: Tensor) -> TokenizerOutput:
        """Convert images to variable-depth tokens."""
        ...
```

### Key Methods

| Method | Description |
|:-------|:------------|
| `tokenize(images)` | Main entry point |
| `_compute_complexity(features)` | Compute $C(R)$ for all regions |
| `_split_regions(complexity)` | Apply splitting scheme |
| `_embed_regions(features, regions)` | ROI-Align + depth embedding |
| `_hilbert_sort(tokens, levels)` | Sort by Hilbert index |

---

## 3.7 Diagnostics

### Training Statistics

```python
stats = tokenizer.get_split_stats()
# {
#     'num_tokens': [48, 52, ...],        # Tokens per image
#     'depth_distributions': [{0: 4, 1: 16, 2: 28}, ...],
#     'mean_complexity': 0.42,
# }

entropy = tokenizer.get_scale_entropy()  # Depth distribution entropy
```

### Depth Distribution

A healthy tokenizer should show diverse depth distributions:

| Metric | Target | Interpretation |
|:-------|:-------|:---------------|
| Entropy | > 1.5 | Good diversity |
| Max depth usage | > 10% | Using fine scales |
| Min depth usage | > 5% | Using coarse scales |

---

## 3.8 Usage Example

```python
from vit_pytorch import StreamingFractalTokenizerV3
from vit_pytorch.split_adaptive import AdaptiveSplitConfig

# Custom configuration
config = AdaptiveSplitConfig(
    alpha=0.5,           # Variance/gradient balance
    tau_0=0.15,          # Root threshold
    gamma=0.85,          # Decay factor
    max_depth=4,         # Maximum depth
    scheme='balanced_greedy',
)

tokenizer = StreamingFractalTokenizerV3(
    image_size=224,
    d_model=384,
    config=config,
)

images = torch.randn(4, 3, 224, 224)
output = tokenizer.tokenize(images)

for i, seq in enumerate(output.sequences):
    print(f"Image {i}: {seq.tokens.shape[0]} tokens")
    print(f"  Depths: {seq.get_levels()[:, 0].unique().tolist()}")
```

> **Next**: [04_positional_embedding.md](04_positional_embedding.md) - Position Encoding
