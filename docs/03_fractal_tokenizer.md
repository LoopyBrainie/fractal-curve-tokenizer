# Chapter 3: Fractal Tokenizer

## 3.1 Overview

The `StreamingFractalTokenizerV3` implements **Variable Depth Tokenization** via adaptive quadtree splitting and Hilbert curve reordering. The recommended splitter is **H1SS (Hilbert Splitter with Stable Selection)** using Entmax sparse activation, providing full gradient flow and stable train/eval consistency.

**Efficiency Note**: The ~40× computational reduction comes from token count reduction ($N_{V3} \approx 32$ vs $N_{ViT} \approx 307K$), not from asymptotic complexity change. Attention remains $O(N^2 \cdot D)$, but with $N$ reduced by ~40×.

**Architecture**: Token depth is determined by the H1SS selection mechanism with Entmax sparse activation. Transformer effective depth is fixed at `num_layers // 2`.

---

## 3.2 Mathematical Formulation

### 3.2.1 Tokenization Pipeline

$$I \xrightarrow{\text{Splitter}} \text{split\_result} \xrightarrow{\text{Embedding}} \{T_i, L_i\} \xrightarrow{\text{HilbertSort}} \{T_i', L_i'\}$$

where:
- $I \in \mathbb{R}^{B \times C \times H \times W}$: Input batch
- $\text{split\_result}$: Selected regions from H1SS/H-entmax
- $T_i$: Token embeddings
- $L_i$: Level (depth) information

### 3.2.2 H1SS: Hilbert Splitter with Stable Selection

H1SS is based on six axioms (A1-A6) for optimal token selection:

| Axiom | Description |
|:------|:------------|
| A1 | 1D Hilbert manifold convolution |
| A2 | No Gumbel perturbation |
| A3 | Entmax sparse activation |
| A4 | Tree consistency soft constraint |
| A5 | Single Entmax projection |
| A6 | < 10K parameters |

**α-Entmax Definition**:

$$\text{entmax}_\alpha(z) = \arg\max_{p \in \Delta^{n-1}} (p^T z + H_\alpha(p))$$

where $H_\alpha(p) = \frac{1}{1-\alpha} \log \sum_i p_i^\alpha$ is the Rényi entropy.

**Properties**:
- $\alpha \to 1$: Degrades to Softmax
- $\alpha > 1$: Produces sparse distributions
- $\alpha = 1.5$: May cause hard truncation (I107), reducing gradient flow

**Alpha Warmup Strategy (I107)**:

H1SS uses an alpha warmup schedule to prevent hard truncation during early training:

$$\alpha(t) = \begin{cases} 1.2 + 0.3 \cdot \frac{t}{T_{warmup}} & t < T_{warmup} \\ \min(2.0, 1.5 + 0.5 \cdot \frac{t - T_{warmup}}{T_{schedule} - T_{warmup}}) & t \geq T_{warmup} \end{cases}$$

Default schedule: $T_{warmup} = 10$, $T_{schedule} = 20$

This ensures:
- Early training ($\alpha \approx 1.2$): Soft selection, full gradient flow
- Late training ($\alpha \to 1.5-2.0$): Sparse selection, improved efficiency

**Selection (I122-1 Updated)**:

The original STE formulation:

$$\text{st\_mask} = \text{hard\_mask} - \text{soft\_mask.detach()} + \alpha \cdot \text{soft\_mask}$$

Has been replaced with **entmax sparse selection** for stability:

$$\text{selected\_mask} = \text{entmax}_\alpha(z) \cdot K$$

where $K$ is the target token count, and $\alpha$ is the entmax parameter. For $\alpha < 1.9$, entmax degrades to softmax, providing direct gradient flow without STE approximation.

### 3.2.3 H-entmax: Hilbert-Ordered Entmax (Alternative)

**Hilbert-Ordered Entmax Splitter** uses α-Entmax (α=1.5) for exact sparsity with full gradient flow:

| Feature | Description |
|:--------|:------------|
| **Parallelism** | 100% |
| **Gradient Flow** | **100%** (full, no STE) |
| **Hilbert Locality** | 100% |
| **Sparsity** | Exact α-Entmax |

### 3.2.4 Tree Consistency

The tree consistency constraint prevents selecting both a parent and its child:

$$\text{consistent}(i) \iff i \in \text{Selected} \land \forall c \in \text{children}(i), c \notin \text{Selected}$$

**Tree Constraint Loss** (H1SS):

$$\mathcal{L}_{tree} = \sum_{i} \sum_{c \in \text{children}(i)} \max(0, p_i - p_c + \epsilon)$$

where $p_i$ is the selection probability.

**Jump Loss** (Hilbert Continuity):

The Jump Loss encourages selection of spatially contiguous regions by penalizing large Hilbert distance jumps:

$$\mathcal{L}_{jump} = \mathbb{E}[(\Delta h - 1)^2_+]$$

where:

- $\Delta h = |h_{i+1} - h_i|$: Absolute difference in Hilbert indices between adjacent selected tokens
- $(\cdot)_+ = \max(0, \cdot)$: ReLU activation

**Why ReLU over exponential (I107)**:

| Form | $\Delta h = 1$ | $\Delta h \to \infty$ | Behavior |
|:-----|:----------------|:---------------------|:---------|
| $\exp(-\gamma \Delta h)$ | $0.37$ (high penalty) | $0$ (no penalty) | Incorrect: Large jumps go unpunished |
| $(\Delta h - 1)^2_+$ | $0$ (no penalty) | $\infty$ (quadratic growth) | Correct: Penalizes large jumps proportionally |

The ReLU form correctly implements the intuition that small Hilbert jumps ($\Delta h \leq 1$, which corresponds to adjacent regions) should be allowed, while large jumps should be quadratically penalized.

## 3.3 Splitter Comparison

| Splitter | Gradient | Sparsity | Parameters | Status |
|:---------|:---------|:---------|:-----------|:-------|
| GumbelTopK | ~37% (STE) | Soft | Medium | Legacy |
| DeterministicNeighbor | 100% | Soft | Medium | Legacy |
| **H1SS (HilbertOptimal)** | **100%** | **Sparse** | **< 10K** | ✓ **Recommended** |
| **H-entmax (HilbertOrderedEntmax)** | **100%** | **Sparse** | **Medium** | ✓ **Alternative** |

### H1SS Parameters

```python
HilbertOptimalSplitter(
    feature_dim=256,
    hidden_dim=64,
    max_level_limit=8,
    K_min=8,
    K_max=64,
    entmax_alpha=1.2,  # α-Entmax alpha
    tree_constraint_weight=0.1,
    temperature_init=1.0,
    temperature_min=0.3,
    jump_loss_weight=0.1,
    density_field_hidden_dim=32,
)
```

### H-entmax Parameters

The H-entmax functionality is provided by `HilbertOptimalSplitter` with α-Entmax activation:

```python
# Use HilbertOptimalSplitter with entmax_alpha for H-entmax behavior
HilbertOptimalSplitter(
    feature_dim=256,
    hidden_dim=64,
    max_level_limit=8,
    K_min=8,
    K_max=64,
    entmax_alpha=1.5,  # α for α-Entmax (full sparse selection)
    tree_constraint_weight=0.1,
    temperature_init=1.0,
    temperature_min=0.3,
)
```

---

## 3.4 DeterministicNeighborSplitter (Legacy)

> **Note**: This splitter is **Legacy**. Use **H1SS (HilbertOptimalSplitter)** instead.

### Overview

`DeterministicNeighborSplitter` was previously recommended for its deterministic behavior and 100% gradient coverage, but has been superseded by H1SS which offers better parameter efficiency (< 10K parameters) and sparse activation.

### Comparison

| Dimension | GumbelTopK | DeterministicNeighbor | H1SS |
|:---------|:------------|:---------------------|:-----|
| **Randomness Source** | $g \sim Gumbel(0,1)$ | None | None |
| **Gradient Coverage** | ~37% (STE) | 100% (direct) | **100%** |
| **Parameters** | Medium | Medium | **< 10K** |
| **Sparsity** | Soft | Soft | **Sparse** |
| **Status** | Legacy | Legacy | ✓ **Recommended** |

---

## 3.5 Token Embedding

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

### 3.4.3 Shape-Scale Enhancement (I31-2)

$$t_i' = t_i + E_{shape}(R_i)$$

where $E_{shape}$ is computed by the Shape-Scale Encoder.

---

## 3.5 Hilbert Reordering

After splitting and embedding, tokens are sorted by their Hilbert curve index:

$$\pi(i) = \text{argsort}(H^{-1}(\text{center}(R_i)))$$

### Isomorphism

$$\text{QuadtreePath}(R) = [q_1, \ldots, q_d] \iff \text{HilbertSegment}(R) = H|_{[a,b]}$$

This ensures that:
1. Spatially adjacent regions have nearby sequence positions
2. LCA relationships are preserved for attention bias

### P11-3: Region-Based LCA Computation

For accurate attention bias, LCA is computed directly from region boundaries:

$$\text{Path}(R) = \text{bit}(cx, D-d) + 2 \cdot \text{bit}(cy, D-d)$$
$$\text{LCA}(i, j) = \text{Length}(\text{CommonPrefix}(\text{Path}(i), \text{Path}(j)))$$

### I161-1: Depth-Root Normalized Hilbert Index

The raw Hilbert index $H \in [0, 4^d)$ varies across depths, making direct comparison problematic. The **depth-root normalization** addresses this:

$$H_{norm} = \left(\frac{H}{4^d}\right)^{\frac{1}{d}}$$

**Properties**:

- $H_{norm} \in [0, 1]$ for all depths $d$
- $H_{norm}$ at depth $d$ naturally aligns with $H_{norm}$ at depth $d-1$ (parent level)
- Preserves Hilbert curve self-similarity across scales

**Why This Matters**:

Without normalization, a depth-0 token with $H=100$ would appear "farther" than a depth-3 token with $H=10$, despite the depth-3 token covering a smaller, more specific region. Normalization ensures scale-invariant comparison while preserving spatial locality ordering.

## 3.6 Implementation

### Class: StreamingFractalTokenizerV3

```python
class StreamingFractalTokenizerV3(BaseTokenizer):
    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]] = 224,
        channels: int = 3,
        d_model: int = 256,
        base_patch_size: int = 4,
        min_patch_size: Union[int, Tuple[int, int]] = 4,
        max_level: Optional[int] = None,
        use_hilbert_order: bool = True,
        depth_scale_range: Optional[Tuple[float, float]] = (0.5, 2.0),
        use_interpolated_pooling: bool = False,
        use_dynamic_weight: bool = False,
    ):
        """
        Args:
            image_size: Input image size (int or (W, H) tuple)
            channels: Number of input channels
            d_model: Model dimension
            base_patch_size: Base patch size for level 0
            min_patch_size: Target minimum patch size
            max_level: Maximum quadtree level (auto-computed if None, I30-17)
            use_hilbert_order: Enable Hilbert curve ordering
            depth_scale_range: P6-1 depth scale range for sigmoid parameterization
            use_interpolated_pooling: Enable interpolated pooling (I-PHASE4)
            use_dynamic_weight: Enable dynamic weight (C3 scale equivariance)
        """
        ...
```

### Core Logic: HilbertOptimalSplitter (H1SS)

```python
class HilbertOptimalSplitter(nn.Module):
    def forward(self, features: Tensor) -> TensorSplitResult:
        """
        Returns:
            TensorSplitResult with:
            - regions: [M, 4] Selected region coordinates
            - depths: [M] Region depths
            - batch_indices: [M] Batch indices
            - hilbert_indices: [M] Hilbert curve indices
            - selected_mask: [B, N] Entmax sparse selection mask
            - logits: [B, N] Raw logits
            - probs: [B, N] Split probabilities
            - num_selected_per_batch: [B] Token count per sample
        """
```

### Diagnostic Metrics

- **Scale Entropy**: Measures diversity of selected depths.
- **Top-K Overlap**: (Debug) How often parents and children are both in Top-K (before consistency check).
- **Quota Distribution**: $\pi_d$ values from Scheme E learnable quota.

---

## 3.7 Usage Example

### Basic Usage with H1SS (Recommended)

```python
from vit_pytorch import StreamingFractalTokenizerV3

# H1SS (Hilbert Splitter with Stable Selection)
# Note: Splitter is now a separate component passed to tokenize()
tokenizer = StreamingFractalTokenizerV3(
    image_size=224,
    d_model=256,
    base_patch_size=4,
    depth_scale_range=(0.5, 2.0),  # P6-1: sigmoid parameterization
)

# Returns TokenizerOutput with aligned tokens and level info
output = tokenizer.tokenize(images)

# Access results
tokens = output.tokens          # [B, N, D]
levels_info = output.levels     # [B, N, max_level+1]
regions = output.regions        # [B, N, 4] - region boundaries
split_probs = output.split_probs  # [B, N] - selection confidence
```

### Direct Splitter Usage

```python
from vit_pytorch.layers.splitters.hilbert_optimal_splitter import HilbertOptimalSplitter

splitter = HilbertOptimalSplitter(
    feature_dim=256,
    hidden_dim=64,
    max_level_limit=8,
    K_min=8,
    K_max=64,
    entmax_alpha=1.2,
    tree_constraint_weight=0.1,
)
```

---

## 3.8 Temperature Annealing

Temperature annealing controls exploration vs. exploitation for H1SS and H-entmax:

$$\tau(t) = \tau_{start} \cdot \left(\frac{\tau_{end}}{\tau_{start}}\right)^{t / T_{total}}$$

**Default Schedule (H1SS)**:

| Parameter | Value |
|:----------|:------|
| $\tau_{start}$ | 1.0 |
| $\tau_{end}$ | 0.3 |
| Schedule | Exponential decay |

---

## 3.9 Constants Reference

All numerical stability constants are centralized in `constants.py`:

| Constant | Value | Mathematical Basis |
|:---------|:------|:-------------------|
| `EPS` | $1e-6$ | General FP stability |
| `FP16_SAFE_EPSILON` | $1e-6$ | FP16 safe lower bound |
| `TEMPERATURE_MIN` | $0.3$ | Prevents gradient saturation (I113-10: τ=0.3 时梯度强度 ≈ 3.3) |
| `LOGIT_CLAMP_BOUND` | $10.0$ | $\text{softmax}(x > 10) \approx 0.99995$ |
| `GRAD_CLAMP_BOUND` | $20.0$ | Gradient magnitude bound |
| `SCALE_CLAMP_BOUND` | $15.0$ | Depth scale clamp bound |
| `SPLITTER_TEMP_START` | $1.0$ | Initial temperature |
| `SPLITTER_TEMP_END` | $0.3$ | Final temperature |
| `LEARNABLE_QUOTA_ENABLED` | `True` | Enable Scheme E quota |
| `QUOTA_MIN_PER_DEPTH` | $2$ | Minimum tokens per depth |
| `QUOTA_MIN_RATIO` | $0.02$ | Minimum quota ratio |
| `K_COVERAGE_BASE` | $0.25$ | Base token coverage |
| `K_MIN_HARD_LIMIT` | $8$ | Minimum absolute K |

### Mathematical Formulation of Key Constants

**Level Aggregator Initialization**:

$$\gamma = \log(e^{1.3133} - 1) \approx 0.26$$

This gives $\text{softplus}(\gamma) \approx 1.0$, ensuring initial scale is neutral.

**Gumbel Temperature Bound**:

$$\tau \geq 0.3$$

Below this threshold, softmax gradients vanish exponentially. The bound is derived from:

$$\frac{\partial \text{softmax}(x/\tau)}{\partial x} = \frac{\text{softmax}(x/\tau)(1 - \text{softmax}(x/\tau))}{\tau}$$

For $\tau < 0.3$, gradient magnitude drops below $10^{-5}$.

> **Next**: [04_positional_embedding.md](04_positional_embedding.md) - Position Encoding
