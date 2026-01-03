# Chapter 3: Fractal Tokenizer

## 3.1 Overview

The `StreamingFractalTokenizerV3` implements **Variable Depth Tokenization** via adaptive quadtree splitting and Hilbert curve reordering. This chapter provides a complete mathematical specification.

---

## 3.2 Mathematical Formulation

### 3.2.1 Tokenization Pipeline

$$I \xrightarrow{\text{SharedConv}} F \xrightarrow{\text{Split}} \{R_i\}_{i=1}^{N} \xrightarrow{\text{Embed}} \{t_i\}_{i=1}^{N} \xrightarrow{\text{Sort}} \{t_{\pi(i)}\}_{i=1}^{N}$$

where:
- $I \in \mathbb{R}^{C \times H \times W}$: Input image
- $F \in \mathbb{R}^{D \times H' \times W'}$: Shared feature map
- $R_i$: Quadtree region (axis-aligned rectangle)
- $t_i \in \mathbb{R}^D$: Token embedding
- $\pi$: Hilbert curve permutation

### 3.2.2 Learnable Complexity Function

The splitting decision is based on a learnable complexity measure estimated from region features:

$$C_\theta(R) = \sigma(\text{MLP}(\text{ROI}(F, R)))$$

where:
- $\text{ROI}(F, R)$: ROI-Aligned features for region $R$
- $\text{MLP}$: Multi-layer perceptron
- $\sigma$: Sigmoid activation

This replaces the previous heuristic-based complexity (variance + gradient) with an end-to-end differentiable estimator.

### 3.2.3 Learnable Threshold

The splitting threshold is depth-dependent and learnable:

$$\tau_d = \tau_{base,d} + \delta_d$$

where:
- $\tau_{base,d}$: Base threshold for depth $d$
- $\delta_d$: Learnable offset parameter

**Splitting criterion**:
$$\text{Split}(R) \iff C_\theta(R) > \tau_d \land d < d_{max} \land \text{size}(R) \geq \text{min\_size}$$

The decision is made differentiable via Gumbel-Softmax during training.

---

## 3.3 Splitting Schemes

### 3.3.1 Learnable Splitting (Scheme L)

**End-to-end differentiable splitting** via Gumbel-Softmax (Standard):

$$\text{SplitProb}(R) = \sigma(\text{MLP}(\text{ROI}(F, R)))$$

Temperature-annealed sampling:
$$z = \text{GumbelSoftmax}(\log p, \tau(t))$$

*Note: Scheme B (Balanced Greedy) and Scheme C (Fixed Budget DP) have been deprecated in favor of the unified Learnable Splitter.*

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
