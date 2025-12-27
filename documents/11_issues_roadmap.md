# Chapter 11: Development History

> **Last Updated**: December 2024 | **Status**: Active Development

## 11.1 Overview

This chapter documents the evolution of the Fractal Curve Tokenizer project, including architecture decisions, performance optimizations, and lessons learned.

---

## 11.2 Architecture Evolution

### Timeline

| Version | Date | Architecture | Status |
|:--------|:-----|:-------------|:-------|
| V1 | Nov 2024 | Fixed multi-scale convolution | ⚠️ Removed |
| V2 | Dec 2024 | Gumbel-Softmax scale selection | ⚠️ Removed |
| V3 | Dec 2024 | Variable Depth Tokens (quadtree) | ✓ Current |

### V1: Fixed Multi-Scale (Removed)

**Approach**: Fixed convolutional pyramid with predetermined scale ratios.

**Limitation**: No content adaptation—same tokenization for all images.

### V2: Gumbel-Softmax (Removed)

**Approach**: Learnable scale selection via Gumbel-Softmax.

**Problem**: Scale collapse. Mathematical proof showed softmax weights inevitably converge to finest scale:

$$\lim_{t \to \infty} \alpha_{s_0} = 1, \quad \text{where } s_0 = \arg\max_s H(F_s)$$

**Observation**: At epoch 17, scale_0 reached 99.99% weight.

### V3: Variable Depth Tokens (Current)

**Approach**: Content-adaptive quadtree splitting with explicit complexity thresholds.

**Key insight**: Avoid softmax competition by using independent per-region split decisions:

$$\text{Split}(R) \iff C(R) > \tau_d$$

**Result**: Stable depth distributions throughout training.

---

## 11.3 Key Improvements

### Architecture (ARCH)

| ID | Description | Status |
|:---|:------------|:-------|
| ARCH-P0 | Streaming tokenizer implementation | ✓ Done |
| ARCH-VDT | Variable Depth Tokens redesign | ✓ Done |
| ARCH-P1 | Deprecated module removal | ✓ Done |
| ARCH-R1 | Global context attention removed | ✓ Done |
| ARCH-R2 | Level aggregator added | ✓ Done |

### Performance (PERF)

| ID | Description | Impact | Status |
|:---|:------------|:-------|:-------|
| PERF-1 | LCA Hilbert Bias | 99.8% param reduction | ✓ Done |
| PERF-2 | SwiGLU FFN | 9.4% model size reduction | ✓ Done |
| PERF-3 | Attention mask vectorization | O(B×S²) → O(B) | ✓ Done |
| PERF-4 | Integral image caching | O(1) region stats | ✓ Done |
| PERF-5 | LCA computation caching | ~8× speedup | ✓ Done |

### Bug Fixes (P0)

| ID | Description | Status |
|:---|:------------|:-------|
| P0-1 | REINFORCE baseline (EMA) | ✓ Fixed |
| P0-2 | Attention mask propagation | ✓ Fixed |
| P0-3 | Hilbert bias mode config | ✓ Fixed |
| P0-4 | TokenizerOutput convenience properties | ✓ Fixed |
| P0-C1 | Cross-scale attention collapse | ✓ Replaced with VDT |

### Quality (P2)

| ID | Description | Status |
|:---|:------------|:-------|
| P2-1 | Test coverage (295+ tests) | ✓ Done |
| P2-2 | Data augmentation strategies | ✓ Done |
| P2-3 | Type annotations (98%) | ✓ Done |
| P2-4 | Mathematical docstrings | ✓ Done |

---

## 11.4 Lessons Learned

### 1. Softmax Scale Competition

**Problem**: When using softmax to weight multiple scales, the finest scale always wins because it retains the most information.

**Solution**: Use independent binary decisions per region instead of competitive softmax.

### 2. Gumbel-Softmax Gradient Sparsity

**Problem**: Straight-through estimator only propagates gradients to selected options.

**Solution**: Content-adaptive thresholding with differentiable complexity estimation.

### 3. LCA Distance vs Learned Embeddings

**Finding**: Simple LCA depth embedding (~100 params) outperforms complex learned path embeddings (~50K params).

**Reason**: LCA depth has explicit geometric meaning (spatial distance), requiring less learning.

### 4. FFN Capacity

**Problem**: Original mlp_dim = 2× dim was too small.

**Solution**: mlp_dim = 4× dim (standard ViT ratio) improved accuracy 3-5%.

### 5. Regularization Balance

**Problem**: dropout=0.3 + drop_path=0.2 caused under-fitting.

**Solution**: Reduced to 0.1 each, improving accuracy 3-5%.

---

## 11.5 Removed Components

The following components have been removed from the codebase:

| Component | Reason |
|:----------|:-------|
| `FractalHilbertTokenizer` | Replaced by streaming tokenizer |
| `EnhancedFractalTokenProcessor` | Merged into V3 |
| `StreamingFractalTokenizer` (V1) | Superseded by V3 |
| `StreamingFractalTokenizerV2` | Scale collapse issue |
| `CrossScaleAttention` | Mathematical collapse proven |
| `_deprecated/` directory | Cleanup complete |
| `original` bias mode | Replaced by LCA |

---

## 11.6 Current Recommendations

### Model Configuration

```python
model = FractalCurveViT(
    image_size=224,
    num_classes=1000,
    dim=384,
    depth=6,
    heads=6,
    mlp_dim=768,                    # 4× dim
    dropout=0.1,                    # Reduced from 0.3
    tokenizer_type='streaming_v3',  # V3 only
    hilbert_bias_mode='lca',        # ~100 params
    ffn_type='swiglu_level',        # SwiGLU + level adapt
)
```

### Training Configuration

```python
optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=0.03)
scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[5])
```

---

## 11.7 Future Directions

| Area | Description | Priority |
|:-----|:------------|:---------|
| Hierarchical attention | Multi-resolution attention patterns | Medium |
| 3D extension | Video and volumetric data | Low |
| Efficiency | FlashAttention integration | High |
| Scaling | Larger model variants | Medium |

> **Next**: [appendix.md](appendix.md) - Appendix
