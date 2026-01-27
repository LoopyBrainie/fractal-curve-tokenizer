# Chapter 3: Fractal Tokenizer

## 3.1 Overview

The `StreamingFractalTokenizerV3` implements **Variable Depth Tokenization** via adaptive quadtree splitting and Hilbert curve reordering. It adopts the **Gumbel-Top-K (Scheme D)** mechanism with **Learnable Quota Allocation (Scheme E)** for ~K/N gradient coverage (~37.6%) and parallel execution, replacing earlier BFS-based approaches.

**Efficiency Note**: The ~40× computational reduction comes from token count reduction ($N_{V3} \approx 32$ vs $N_{ViT} \approx 307K$), not from asymptotic complexity change. Attention remains $O(N^2 \cdot D)$, but with $N$ reduced by ~40×.

**Architecture**: Token depth is determined by the Gumbel-Top-K selection mechanism. Transformer effective depth is fixed at `depth // 2`.

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

**Decision Logits** (I30-4 Updated):

$$\text{logits}_i = \text{MLP}(\text{ROI}(F, R_i)) + b_{explore} + \beta \cdot \gamma^{d_i} - \tau_{d_i}$$

where:
- $\text{MLP}(\text{ROI}(F, R_i))$: Learnable complexity score (I30-4)
- $\tau_{d_i}$: Learnable per-depth threshold.
- $b_{explore}$: Annealed exploration bias.
- $\gamma^{d_i}$: Depth penalty term (optional).

> **Note (I30-4)**: The Log-Compensation Bias $b_{log}(d_i) = \log(N_{total} / N_{d_i})$ has been removed in favor of the learnable quota mechanism (Scheme E).

**Stochastic Selection**:

$$z_i = \text{logits}_i + g_i, \quad g_i \sim \text{Gumbel}(0, 1)$$
$$\text{selected\_indices} = \text{TopK}(\{z_i/\tau\}_{i=1}^{N_{cand}}, K)$$

This formulation provides gradients for **selected** candidates via the Straight-Through Estimator (STE), while unselected candidates receive attenuated gradients (~20x reduction).

### 3.2.3 Learnable Quota Allocation (Scheme E)

When `LEARNABLE_QUOTA_ENABLED=True`, the model learns an optimal token quota distribution across depths.

**Quota Calculation**:

$$\pi_d = \text{softmax}(\phi_d), \quad \phi \in \mathbb{R}^{D+1}$$
$$K_d = \text{round}(\pi_d \cdot K_{total})$$
$$K_d = \max(K_d, K_{min})$$

where:
- $\phi_d$: Learnable logit for depth $d$
- $\pi_d$: Learned probability distribution over depths
- $K_d$: Quota allocated to depth $d$
- $K_{min}$: Minimum quota per depth (default: 2)

**Hierarchical Top-K Selection**:

For each depth $d$, select exactly $K_d$ tokens from regions at that depth, then merge results across depths.

### 3.2.4 Tree Consistency

The raw Top-K selection may violate the tree structure (e.g., selecting both a parent and its child). We enforce consistency via a vectorized operation:

$$\text{consistent}(i) \iff i \in \text{TopK} \land \forall c \in \text{children}(i), c \notin \text{TopK}$$

This insures that if a parent is selected, its children are ignored, maintaining a valid partition (or subset thereof).

### 3.2.5 Shape-Scale Encoder (I31)

For enhanced token representation, a shape-scale encoder captures region geometry:

**Aspect Ratio**:
$$r = \log(w/h) \quad \text{(log-transformed for symmetry)}$$

**Normalized Area**:
$$s = \frac{w \cdot W_{patch}}{W_{total} \cdot H_{total}}$$

**Gated Combination**:
$$g = \sigma(\text{MLP}([r; s]))$$
$$E_{shape}(R) = \text{MLP}([r \cdot g; s \cdot (1-g)])$$

---

## 3.3 Splitting Schemes

### 3.3.1 Scheme D: Gumbel-Top-K (Base)

| Feature | Description |
|:--------|:------------|
| **Parallelism** | 100% (All regions evaluated in one batch) |
| **Gradient Flow** | ~37.6% (K/N) - STE provides gradients to selected tokens, attenuated for unselected |
| **Hilbert Locality** | 100% (Strict adherence to Hilbert curve ordering) |
| **Complexity** | $O(N_{cand})$ parallel evaluation |

### 3.3.2 Scheme E: Learnable Quota (Extension)

| Feature | Description |
|:--------|:------------|
| **Adaptive Budget** | Learns optimal token distribution across depths |
| **Depth Balance** | Prevents over-allocation to shallow or deep tokens |
| **Interpretability** | $\pi_d$ reveals model's preferred depth distribution |

### 3.3.3 Scheme Comparison

| Scheme | Parallelism | Gradient | Quota | Status |
|:-------|:------------|:---------|:------|:-------|
| A (BFS) | ~30% | Partial | Fixed | Deprecated |
| B (Relaxation) | ~60% | Partial | Fixed | Deprecated |
| C (Fixed Budget) | 100% | STE | Fixed | Deprecated |
| **D (Gumbel-Top-K)** | **100%** | **STE** | **Fixed** | **Current** |
| **E (Learnable Quota)** | **100%** | **STE** | **Learned** | **Recommended** |

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
        min_patch_size: int = 4,
        max_depth: Optional[int] = None,
        K_min: int = 8,
        K_max: int = 64,
        splitter_dropout: float = 0.15,
    ):
        """
        Args:
            image_size: Input image size
            d_model: Model dimension
            base_patch_size: Base patch size for level 0
            min_patch_size: Target minimum patch size
            max_depth: Maximum quadtree depth (auto-computed if None)
            K_min: Minimum token count (I23-2)
            K_max: Maximum token count
            splitter_dropout: Dropout for splitter MLP
        """
        ...
```

### Core Logic: GumbelTopKSplitter

```python
class GumbelTopKSplitter(nn.Module):
    def forward(self, features: Tensor) -> GumbelTopKResult:
        """
        Returns:
            GumbelTopKResult with:
            - regions: [M, 4] Selected region coordinates
            - depths: [M] Region depths
            - batch_indices: [M] Batch indices
            - hilbert_indices: [M] Hilbert curve indices
            - selected_mask: [B, N] STE gradient mask
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

```python
from vit_pytorch import StreamingFractalTokenizerV3

tokenizer = StreamingFractalTokenizerV3(
    image_size=224,
    d_model=384,
    base_patch_size=4,
    K_min=8,
    K_max=64,
)

# Returns TokenizerOutput with aligned tokens and level info
output = tokenizer.tokenize(images)

# Access results
tokens = output.tokens          # [B, N, D]
levels_info = output.levels     # [B, N, max_depth+1]
regions = output.regions        # [B, N, 4] - region boundaries
split_probs = output.split_probs  # [B, N] - selection confidence
```

### Advanced: Scheme E with Learnable Quota

```python
from vit_pytorch import GumbelTopKSplitter

splitter = GumbelTopKSplitter(
    dim=384,
    max_depth=6,
    K_total=32,
    learnable_quota=True,    # Enable Scheme E
    quota_init_logits=None,  # Auto-initialization
)

# After training, inspect learned quota distribution
quota_probs = F.softmax(splitter.quota_logits, dim=0)
# quota_probs[d] = probability of allocating tokens to depth d
```

---

## 3.8 Temperature Annealing

The Gumbel-Softmax temperature controls exploration vs. exploitation:

$$\tau(t) = \tau_{start} \cdot \left(\frac{\tau_{end}}{\tau_{start}}\right)^{t / T_{total}}$$

**Default Schedule**:

| Parameter | Value |
|:----------|:------|
| $\tau_{start}$ | 1.0 |
| $\tau_{end}$ | 0.5 |
| Schedule | Exponential decay |

---

## 3.9 Constants Reference

All numerical stability constants are centralized in `constants.py`:

| Constant | Value | Mathematical Basis |
|:---------|:------|:-------------------|
| `GUMBEL_EPSILON` | $1e-8$ | FP16 safe lower bound |
| `LOG_EPSILON` | $1e-8$ | Logarithm stability |
| `DIVISION_EPSILON` | $1e-8$ | Division stability |
| `PROB_EPSILON` | $1e-5$ | Probability clamping |
| `LOGIT_CLAMP_BOUND` | $50.0$ | $\text{softmax}(x > 50) \approx \text{one-hot}$ |
| `GRAD_CLAMP_BOUND` | $20.0$ | $P(\|grad\| > 20) \approx 10^{-6}$ |
| `TEMPERATURE_MIN` | $0.3$ | Prevents gradient saturation |

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
