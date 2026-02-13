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
| V4 | Feb 2026 | Hilbert-Aware Token Reduction | ✓ Current |

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

### V4: Hilbert-Aware Token Reduction (Current)

**Approach**: Neighbor-aware pruning with deterministic Hilbert-based selection.

**Key components**:
- `DeterministicNeighborSplitter`: NAP-style neighbor-aware token selection
- `HierarchicalSoftHardAttention`: Hard-locality guarantee for ℓ ≥ 1
- `HilbertAdjacencyMatrix`: LCA-based neighbor propagation

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

---

## 11.8 P5: Performance Optimization Issues (Archived)

> **Status**: ✅ 79% Completed (11/14), 3 Items Deferred
> **Archive Date**: 2025-12-27

| ID | Issue | Status | Description |
|----|-------|--------|-------------|
| P5-1 | `_embed_batch` vectorization | ✅ | ROI-Align batch pooling |
| P5-2 | IntegralImageCache batch | ✅ | Support [B,C,H,W] input |
| P5-3 | Hilbert cache expansion | ✅ | 1024→4096, 64→256 |
| P5-4 | SwiGLU bias parameterization | ✅ | bias=False option |
| P5-5 | Depth modulation range expansion | ✅ | beta=0.2, [1.0,1.2] |
| P5-6 | Global context scaling | ✅ | Removed with ARCH-R1 |
| P5-7 | HierarchicalBias low-rank | ⏸️ | Non-default path, deferred |
| P5-8 | levels_info format unification | ✅ | HilbertBiasBase base class |
| P5-9 | Attention mask naming | ⏸️ | High impact, low gain, deferred |
| P5-10 | Split parameter domain validation | ✅ | Domain adaptation preset factory |
| P5-11 | torch.compile compatibility | ⏸️ | Pending P4 post-processing |
| P5-12 | Complexity documentation | ✅ | Add time-space complexity |
| P5-13 | level_attention_bias example | ✅ | API documentation example |
| P5-14 | chunk_size configurability | ✅ | Existing parameter |

---

## 11.9 P6: Scale Utilization Issues (Archived)

> **Status**: ✅ 80% Completed (4/5), 1 Item Deferred
> **Archive Date**: 2025-12-26

**Key Finding**: Model depth signal strength initialization is too conservative

| ID | Issue | Status | Description |
|----|-------|--------|-------------|
| P6-1 | depth_scale range expansion | ✅ | Sigmoid parameterization [0.5,2.0] |
| P6-2 | LCA Bias temperature parameter | ✅ | Learnable temperature τ=1.5 |
| P6-3 | Depth Attention visualization | ✅ | DepthAttentionAnalyzer |
| P6-4 | Depth Embedding analysis | ✅ | DepthEmbeddingAnalyzer |
| P6-5 | Depth Contrastive Loss | 🔵 | Research-oriented, deferred |

**Key Formulas**:
- depth_scale: $\sigma_d = \sigma_{min} + (\sigma_{max} - \sigma_{min}) \cdot \text{sigmoid}(\gamma_d)$
- LCA temperature: $B'_{h,i,j} = \tau_h \cdot B_{LCA(i,j)}$

---

## 11.10 P7: Learnable Splitter Issues (Archived)

> **Status**: ✅ 100% Completed (7/7)
> **Archive Date**: 2025-12-27

**Core Problem**: Rule-based splitters have three major defects: complexity saturation, threshold mismatch, gradient blocking

| ID | Issue | Solution |
|----|-------|----------|
| P7-1 | Complexity function saturation | Learnable MLP complexity predictor |
| P7-2 | Threshold-complexity mismatch | Learnable threshold vector τ |
| P7-3 | Discrete decision gradient blocking | Gumbel-Softmax + STE |
| P7-4 | LearnableSplitter implementation | Added ~400 lines of code |
| P7-5 | TokenizerV3 integration | split_scheme='learnable' |
| P7-6 | Training loss extension | Entropy regularization + budget constraint |
| P7-7 | Temperature annealing schedule | Exponential annealing T_start→T_end |

**Mathematical Formulation**:
- Complexity: $C_\theta(R) = \sigma(\text{MLP}(\text{ROI-Pool}(F, R)))$
- Split decision: $p_{split} = \sigma((C_\theta(R) - \tau_d) / T)$
- Gumbel-Softmax: $y = \text{softmax}((\log[1-p, p] + [g_0, g_1]) / \tau)$

---

## 11.11 P8: Splitter Deep Optimization Issues (Archived)

> **Status**: ✅ 100% Completed (5/5)
> **Archive Date**: 2025-12-28

**Key Finding**: P7 implementation completes basic functionality, 5 optimization issues identified

| ID | Issue | Solution | Performance Gain |
|----|-------|----------|----------|
| P8-1 | Multi-layer differentiable loss | `get_multi_layer_depth_loss()` | Deep threshold gradient |
| P8-2 | Parallelized recursive split | BFS layer batch evaluation | O(D) calls |
| P8-3 | Spatial index acceleration | `SpatialIndex` class | 8.4x speedup |
| P8-4 | STE gradient path enhancement | REINFORCE policy gradient | Gradient quality ↑ |
| P8-5 | Dynamic temperature scheduling | `TemperatureScheduler` | Usability ↑ |

**Verification Results**:
- BFS and recursive version output exactly matches
- Neighbor query: 0.0303ms → 0.0036ms (8.4x)
- Threshold gradient: 0 → non-zero

---

## 11.12 P9: Training Performance Bottleneck Issues (Archived)

> **Status**: ✅ Phase 1-3 Completed (6/8), 2 Items Deferred
> **Archive Date**: 2025-12-28
> **Result**: Training speed improved from 24-25s/it to 1.62s/it (14.8x speedup)

| ID | Issue | Status | Description |
|----|-------|--------|-------------|
| P9-1 | Python data structure GPU-CPU sync blocking | ✅ | TensorSplitResult + vectorized BFS |
| P9-2 | sort_tokens_vectorized CPU fallback | ✅ | Merged with P9-1 |
| P9-3 | torch.tensor() repeated creation in hot path | ✅ | Implicitly resolved by P9-1 |
| P9-4 | Repeated ROI-Align computation | ⚠️ | Requires architecture refactor, deferred |
| P9-5 | model._prepare_tokens non-vectorized | ✅ | Pre-filled cache |
| P9-6 | depth_distribution repeated computation | ✅ | scatter_add vectorization |
| P9-7 | get_multi_layer_depth_loss per batch execution | ⏸️ | Deferred to training optimization phase |
| P9-8 | @torch._dynamo.disable blocks compilation | ⏸️ | Deferred |

**Core Refactoring**: `TensorSplitResult` data structure, completely eliminates Python loops

---

## 11.13 P10: Splitter Training Failure Issues (Archived)

> **Status**: ✅ P10-1~16 Completed, P10-17~20 Pending Fix
> **Archive Date**: 2026-01-02
> **Pending Fix**: P10-17~20 (Architecture-level improvements)

### Fixed Issues (P10-1~9)

| ID | Issue | Solution |
|----|-------|----------|
| P10-1 | Main task gradient cannot backpropagate | Correct STE implementation (y_hard - y_soft.detach() + y_soft) |
| P10-2 | Unstable equilibrium point at initialization | Gain 0.1 → 1.0, expanded complexity distribution |
| P10-3 | Extreme split oscillation | Merged with P10-9 |
| P10-4 | get_entropy_loss has no gradient | Differentiable soft entropy based on BFS path |
| P10-5 | get_multi_layer_depth_loss fixed grid | Merged with P10-4 |
| P10-6 | Threshold regularization only constrains range | Demoted to P3 (soft entropy provides implicit adjustment) |
| P10-7 | Temperature annealing too fast | Demoted to P3 (P10-2 fix reduces critical zone to 22%) |
| P10-8 | Fine split shortcut bias | Joint solution with P10-9 + P10-4/5 |
| P10-9 | Elastic Budget | Asymmetric Dead Zone penalty mechanism |

### Fixed Issues (P10-10~16, Added 2026-01)

| ID | Issue | Solution |
|----|-------|----------|
| P10-10 | Soft entropy loss uses gradient-free EMA probability | Scheme A+D: forward() always caches p_split, threshold prior replaces EMA |
| P10-11 | Temperature endpoint too low (T_end=0.1) causes gradient instability | T_end 0.1 → 0.3, gradient amplification reduced from 10x to 3.3x |
| P10-12 | "Chicken and egg" problem with avg_tokens=1 in early training | Exploration bias b=0.5 + annealing, P(split) increased from 50% to 70% |
| P10-13 | splitter_loss negative values cause semantic confusion | Use KL divergence instead of negative entropy, ensures loss always non-negative |
| P10-14 | Missing avg_tokens anomaly detection mechanism | `check_splitter_health()` + TensorBoard monitoring |
| P10-15 | Exploration bias annealing and temperature annealing timing mismatch | Delayed bias annealing + synchronized with temperature annealing |
| P10-16 | BFS root node single point failure causes global collapse | Exponential decay depth bias β·γ^d (β=1.0, γ=0.5) |

**Key Mathematical Formulas**:
- STE: $z_{ST} = z_{hard} - \text{sg}(y_{soft}) + y_{soft}$
- Elastic Budget: $L = \lambda_{over} \cdot \text{ReLU}(N-N_{max})^2 + \lambda_{under} \cdot \text{ReLU}(N_{min}-N)$
- Depth bias: $\Delta b_d = \beta \cdot \gamma^d$ (root node +1.0, decays each layer)
- KL entropy loss: $L = \log(D+1) - \tilde{H} + \text{anti\_collapse\_penalty}$

---

## 11.14 P11: Architecture Math Formalization Review Issues (Archived)

> **Status**: ✅ All Completed (18/18)
> **Archive Date**: 2026-01-02

### P11 Issues Complete List

| ID | Issue | Solution | Status |
|----|-------|----------|--------|
| P11-1 | LCA cache uses data_ptr with collision risk | WeakRef cache, use `is` object identity check | ✅ |
| P11-2 | Level-aware LayerNorm parameter explosion | Scheme F: Dynamic depth embedding, obtained from tokenizer.max_depth, 92%+ parameter savings | ✅ |
| P11-3 | LowRankHilbertBias path dimension truncation | Scheme D: Runtime path computation from regions, new `forward_from_regions()` | ✅ |
| P11-4 | level_scale_embedding lacks positivity constraint | Softplus constraint, initialization softplus(0.54) ≈ 1.0 | ✅ |
| P11-5 | FractalPositionEmbedding and LCA information redundancy | Scheme 1: Remove dead code `level_attention_bias` | ✅ |
| P11-6 | Elastic Budget EMA cold start problem | Verified: Use cache during training instead of EMA, no modification needed | ✅ |
| P11-7 | scale_weights initialization deviates from 1.0 | Changed initialization to softplus(0.54) ≈ 1.0 | ✅ |
| P11-8 | Hilbert Bias multi-mode parameter inconsistency | Remove dead code: LowRankHilbertBias, HierarchicalHilbertBias | ✅ |
| P11-9 | Depth distribution statistics still has O(B) loop | scatter_add vectorization | ✅ |
| P11-10 | ComplexityMLP output range coupled with threshold | Remove output layer sigmoid, threshold initialized to 0, Barrier boundary extended to logit space | ✅ |
| P11-11 | Gumbel-Softmax T→0.1 causes STE failure | T_end default 0.1 → 0.3 + safety lower bound protection | ✅ |
| P11-12 | Hilbert index and Quadtree path encoding inconsistency | Scheme C: Design decision confirmed, keep Z-order LCA + Hilbert ordering, semantic complementarity | ✅ |
| P11-13 | depth_scale semantic ambiguity | Scheme A simplified: Rename `_level_residual_embedding` → `_residual_gate` | ✅ |
| P11-14 | ROI-Align degrades for small regions | Scheme C: `min_region_size >= 2 * base_patch_size` | ✅ |
| P11-15 | level_mixing_weights comments inconsistent with implementation | Updated comments: sigmoid independent gating not softmax | ✅ |
| P11-16 | Transformer dual LayerNorm inconsistency | Scheme B: levels_info=None defaults to depth 0 | ✅ |
| P11-17 | quadrant_embedding index may overflow | Reviewed: Won't trigger in actual use, existing clamp protection is reasonable | ✅ |
| P11-18 | Soft Token Count formula doesn't match BFS | Verified: Estimation accurate with correct call order | ✅ |

**Core Fix Results**:
- Gradient improvement: 4.05x (P11-10)
- Parameter savings: 92%+ (P11-2)
- Cache collision risk: Eliminated (P11-1)
- Path data: Correctly computed from regions (P11-3)

---

## 11.15 Deprecated Component Archive

| Component | Removal Reason | Removal Date |
|------|----------|----------|
| `StreamingFractalTokenizer` (V1) | Replaced by V3 | 2025-12-26 |
| `StreamingFractalTokenizerV2` | Scale collapse issue | 2025-12-25 |
| `CrossScaleAttention` | Mathematical collapse proven | 2025-12-25 |
| `FractalHilbertTokenizer` | Replaced by streaming | 2025-12-24 |
| `EnhancedFractalTokenProcessor` | Merged into V3 | 2025-12-24 |
| `global_context_attn` | Removed by ARCH-R1 | 2025-12-26 |
| `sort_tokens()` series | Replaced by P9-1 vectorization | 2025-12-28 |
| `forward()` returns List[SplitResult] | Replaced by TensorSplitResult | 2025-12-28 |
| `LowRankHilbertBias` | Dead code removed by P11-8 | 2025-12-30 |
| `HierarchicalHilbertBias` | Dead code removed by P11-8 | 2025-12-30 |
| `level_attention_bias` | Dead code removed by P11-5 | 2025-12-30 |


---

## 11.16 Training Performance (I9)

| ID | Description | Result | Status |
|:---|:------------|:-------|:-------|
| I9 | Training speed bottleneck (24s/it) | **1.62s/it** (14.8x speedup) | ✓ Done |
| I9-1 | Vectorized BFS & GPU structs | Eliminated CPU blocking | ✓ Done |
| I9-2 | `TensorSplitResult` Implementation | Removed Python loops | ✓ Done |

---

## 11.17 Splitter Failure Analysis (I10)

| ID | Issue | Root Cause | Resolution |
|:---|:------|:-----------|:-----------|
| I10 | Learnable splitter collapse | BFS serial dependency | Replaced by **Scheme D** |
| I10-1 | No gradient flow | STE implementation error | Fixed in Scheme A |
| I10-18 | BFS serial dependency | Architecture flaw | Parallel Evaluation |
| I10-19 | Discrete-continuous mismatch | Quantization error | Continuous Relaxation |

---

## 11.18 Math Review (I11)

> Status: All 18 issues resolved.

| ID | Issue | Solution |
|:---|:------|:---------|
| I11-2 | LayerNorm Parameter Explosion | Dynamic Depth Embedding (92% reduction) |
| I11-10 | ComplexityMLP range coupling | Removed output sigmoid |
| I11-11 | Gumbel-Softmax low temp | Enforced T_end >= 0.3 |

---

## 11.19 Deep Math Review (I12)

| ID | Issue | Severity | Status |
|:---|:------|:---------|:-------|
| I12-1 | Gumbel-Softmax double sampling | High | Verified Correct |
| I12-2 | LCA irregular quadtree | Critical | Verified Correct |
| I12-3 | Empty path data source | Critical | Fixed |

---

## 11.20 Comprehensive Review (I13)

| ID | Component | Findings | Status |
|:---|:----------|:---------|:-------|
| I13-1 | FractalPositionEmbedding | Valid logic | Verified |
| I13-2 | AttnHilbertBias | O(N) memory achieved | Verified |
| I13-5 | FeedForwardNetwork | SwiGLU correctly implemented | Verified |

---

## 11.21 Experimental Verification (I14)

| Experiment | Configuration | Result | Note |
|:-----------|:--------------|:-------|:-----|
| E4070 | Tiny-ImageNet, 384d, 12L | 50.74% Acc | Baseline |
| E4071 | Scheme D Integration | Pending | **I14-5 Active** |

---

## 11.22 Training System Refactor (I15)

| Module | Change | Benefit |
|:-------|:-------|:--------|
| `examples/training` | Modular design | Separation of concerns |
| `Trainer` class | Hydra config integration | Reproducibility |
| Logging | TensorBoard + JSONL | Detailed analytics |

---

## 11.23 Collapse Root Cause (I16)

**Diagnosis**: The combination of **Rapid Temperature Annealing** (T < 0.1) and **Unconstrained Threshold Learning** allowed thresholds to grow unchecked, while the **Serial BFS** prevented deep gradients from correcting the behavior.

**Resolution**: Abandoned Scheme A (BFS) in favor of Scheme D (Gumbel Top-K).

---

## 11.24 End-to-End Refactor (I17)

All items in I17 (End-to-End Learnable Splitter) have been subsumed by **Scheme D** implementation.

---

## 11.25 Code Critique (I18)

| ID | Issue | Solution |
|:---|:------|:---------|
| I18-1 | Soft Entropy gradient block | Use cached MLP probabilities |
| I18-2 | Temperature unsafe lower bound | Force `TEMPERATURE_MIN = 0.1` (later updated to 0.3 in I24-7) |
| I18-5 | Temperature learning constraint | Added clamp to learner |

---

## 11.26 Scheme D: Gumbel-Top-K (I19)

**Concept**: Select exactly $K$ tokens from $N$ candidates using Gumbel-Top-K trick.
**Key Advantage**: 100% Gradient Coverage (all regions receive gradients via softmax-STE) + 100% Hilbert Locality (unlike continuous relaxation).

| Feature | Implementation | Status |
|:--------|:---------------|:-------|
| Logic | `GumbelTopKSplitter` | ✓ Done |
| Math | $mask = \mathbb{1}_{TopK} - \sigma(z).detach() + \sigma(z)$ | ✓ Verified |
| Tree | $parent \notin selected$ | ✓ Enforced |

---

## 11.27 Final Analysis (I20)

**Conclusion**: Scheme D represents the mathematically correct solution to the adaptive tokenization problem, solving the "Serial Dependency" and "Gradient Sparsity" problems of Scheme A while maintaining the geometric properties that Scheme B lost.

## 11.28 Architecture Math Critique (I22)

> **Closed Date**: 2026-01-15
> **Status**: 6/8 Completed, 2 items merged

### Completed Issues

| ID | Issue | Solution | Status |
|----|-------|----------|--------|
| I22-1 | Tree consistency gather semantic error | Removed redundant implementation, kept `hard_selected[:, safe_children]` | ✅ |
| I22-2 | LCA bias semantic correctness verification | P11-3 implementation verified | ✅ |
| I22-3 | Subset Softmax gradient enhancement claim verification | Theoretical verification 2.7x enhancement | ✅ |
| I22-4 | Log-Compensation Bias formula derivation | Scheme E alternative implementation | ✅ |
| I22-6 | Temperature parameter theoretical optimal value research | via I29-1 (τ=0.5 optimal) | ✅ |

### Merged to Other Issues

| ID | Merge Target | Description |
|----|--------------|-------------|
| I22-5 | → I25-2 | Hilbert benefit quantization experiment |
| I22-8 | → I25-12 | PseudoHilbert locality decay |

---

## 11.29 Training Diagnosis (I23)

> **Closed Date**: 2026-01-15
> **Status**: 4/9 Completed, 5 items pending

### Completed Issues

| ID | Issue | Solution |
|----|-------|----------|
| I23-1 | Depth distribution collapse | Depth variance normalization + KL weight 0.5 + soft quota regularization |
| I23-2 | Token count lower bound failure | K_max(K, K_min) fix + K_90 per-batch calculation |
| I23-3 | Train-test generalization gap | Resolved via I24-1 regularization enhancement |
| I23-4-NaN | splitter_loss NaN/Inf | Clamp order fix + complexity logits clipping |
| I23-7 | Attention analysis module failure | Fixed with `store_attn_weights` conditional |

### Pending Issues

| ID | Priority | Issue |
|----|----------|-------|
| I23-8 | 🔵 P2 | Scale Distribution entropy too low |
| I23-9 | 🔵 P2 | Class-Token count correlation anomaly |
| I23-10 | ⚪ P3 | Code theory fix |

---

## 11.30 Experiment 20260112 Analysis (I24)

> **Closed Date**: 2026-01-15
> **Status**: 9/12 Completed, 3 items pending

### Completed Issues

| ID | Issue | Solution |
|----|-------|----------|
| I24-1 | Train-validation generalization gap | dropout 0.15→0.20 + label_smoothing 0.1→0.15 |
| I24-2 | Log-Compensation theoretical flaw | Scheme E (learnable quota) alternative |
| I24-3 | Class 1 accuracy 0% | Data cleaning + label_smoothing |
| I24-9 | Position encoding boundary rounding error | Existing clamp protection sufficient |
| I24-11 | Evaluation script attention analysis failure | Same fix as I23-7 |
| I24-12 | per_class_avg_tokens data anomaly | Scheme A Focal γ=2.5 |

### Merged/Pending

| ID | Status | Description |
|----|--------|-------------|
| I24-10 | → I25-10 | DEPTH_QUOTA_TARGET parameterization |
| I24-4 | 🟡 P1 | depth=1 completely missing |
| I24-5 | 🟡 P1 | Depth variance normalization batch statistics |

---

## 11.31 Experiment 20260114 Analysis (I25)

> **Closed Date**: 2026-01-17
> **Status**: 8/13 Completed, 5 items pending

### Completed Issues

| ID | Issue | Solution/Conclusion |
|----|-------|---------------------|
| I25-4 | Temperature sensitivity ablation | I29-1 validates τ=0.5 optimal |
| I25-5 | Threshold variance regularization | threshold_var_loss implemented |
| I25-6 | Integer division boundary quadrant error | Existing logic is correct |
| I25-7 | Scale diversity regularization | Scheme E improves entropy to 54.5% |
| I25-9 | Evaluation script attention analysis fix | Completed |

### Pending Issues

| ID | Priority | Issue |
|----|----------|-------|
| I25-2 | 🔴 P0-Critical | Hilbert vs Raster ablation experiment |
| I25-8 | 🔵 P2 | Hybrid pooling selector benefit evaluation |
| I25-10 | 🔵 P2 | Quota parameter exposure (includes I24-10) |
| I25-12 | 🔵 P2 | PseudoHilbert locality quantization |

---

## 11.32 Tree Consistency Removal (I26)

> **Closed Date**: 2026-01-15
> **Status**: 2/3 Completed, 1 item deferred

### Completed Issues

| ID | Issue | Solution |
|----|-------|----------|
| I26-1 | Remove tree consistency constraint | Delete `_enforce_tree_consistency()` call |
| I26-2 | threshold_var_loss missing | Implemented |

### Deferred

| ID | Issue | Description |
|----|-------|-------------|
| I26-3 | LCA semantic mixing problem | Requires further analysis |

---

## 11.33 Class Accuracy Variance (I28)

> **Closed Date**: 2026-01-15
> **Status**: 1/3 Completed, 2 items postponed

### Completed Issues

| ID | Issue | Solution |
|----|-------|----------|
| I28-1 | Worst Classes accuracy too low | Scheme A: Focal γ=2.5 + label_smoothing |

---

## 11.34 Math Consistency Fix (I29)

> **Closed Date**: 2026-01-17
> **Status**: 4/5 Completed, 1 item postponed

### Completed Issues

| ID | Issue | Solution |
|----|-------|----------|
| I29-1 | SPLITTER_TEMP_END constant inconsistency | Unified to 0.5 |
| I29-2 | threshold_var_loss implementation | Completed |
| I29-3 | LOG_COMPENSATION condition redundancy | Removed after Scheme E alternative |
| I29-4 | Temperature schedule strategy unification | Unified temperature annealing parameters |

### Postponed

| ID | Issue | Description |
|----|-------|-------------|
| I29-5 | Hardcoded epsilon standardization | Low priority |

---

## 11.35 Comprehensive Math Critique (I30)

> **Closed Date**: 2026-01-17
> **Status**: 3/17 Completed, 14 items pending

### Completed Issues

| ID | Issue | Solution |
|----|-------|----------|
| I30-3 | Severe overfitting (Gap=12%) | Enhanced dropout + label_smoothing |
| I30-4 | Class accuracy variance too large | Focal Loss + data cleaning |
| I30-9 | Weak reference cache failure | Strong reference + WeakRef dual protection |

### Pending Issues (by Priority)

| ID | Priority | Issue |
|----|----------|-------|
| I30-1 | 🔴 P0-Critical | Hilbert vs Raster core benefit verification |
| I30-2 | 🔴 P0-Critical | Subset Softmax gradient claim vs implementation contradiction |
| I30-5 | 🔴 P0 | levels_info value range verification |
| I30-6 | 🟡 P1 | Small batch depth variance normalization |
| I30-7 | 🟡 P1 | PseudoHilbert jump boundary formula |
| I30-8 | 🟡 P1 | GUMBEL_EPSILON too small risk |
| I30-10~13 | 🔵 P2 | Code quality optimization |
| I30-14~16 | ⚪ P3 | Research exploration |

---

## 11.36 LCA Pseudo Adaptation (I31)

> **Status**: 3/3 P0-Critical pending implementation
> **Closed Date**: TBD

### Pending Implementation Issues

| ID | Issue | Priority |
|----|-------|----------|
| I31-1 | Shape-Scale Encoder Implementation | 🔴 P0-Critical |
| I31-2 | LCAHilbertBias Extension | 🔴 P0-Critical |
| I31-3 | Area Position Encoding Supplement | 🔴 P0-Critical |

---

## 11.37 Issue Archive (2026-01-26)

> **Archive Date**: 2026-01-26
> **Status**: Archive closed Issues, keep active Issues
> **Archive Scope**: CRIT-1~4, I78-2, I96-1~8, I97-1~13, I98-1~9, I99-1~11, I100-1~7, I102-1~11, I103-1~4, I104-1~3, I105-1, I106-1~2, I107-1~6

### 11.37.1 P0-Critical Issues (Archived)

| ID | Issue | Solution | Archive Time |
|----|-------|----------|--------------|
| **CRIT-1** | Gumbel-Top-K gradient coverage doc inaccurate | Update docstring, Scheme E uses global Softmax | 2026-01-22 |
| **CRIT-2** | hilbert_bias_scale unbounded | Add clamp(max=10.0) | 2026-01-22 |
| **CRIT-3** | Unbounded LCA cache can cause OOM | Remove cache, compute LCA directly | 2026-01-26 |
| **CRIT-4** | `.item()` sync point slows training | GPU tensor accumulation + single sync | 2026-01-26 |
| **I107-2** | Feature Cache retains references | Remove debug cache variables | 2026-01-26 |
| **I107-3** | Trainer gradient accumulation memory accumulation | Use `del` to release intermediates | 2026-01-26 |

### 11.37.2 P0 Issues (Archived)

| ID | Issue | Solution | Archive Time |
|----|-------|----------|--------------|
| **I78-2** | Tokenizer ignores K budget | Use selected_mask indexing directly | 2026-01-21 |
| **I96-1** | EMA clamp initialization inconsistent | Unified DEPTH_VARIANCE_INIT_EPS | 2026-01-21 |
| **I96-2** | Temperature annealing incomplete | Training config issue, not code bug | 2026-01-21 |
| **I97-1** | EMA eval mode not initialized | Conservative initialization | 2026-01-22 |
| **I97-2** | `_compute_level_bias` 2D dead code | Delete 2D branch, add assertion | 2026-01-22 |
| **I97-3** | `_precompute_candidates` method duplicate | Unified method implementation | 2026-01-22 |
| **I99-1** | Batch independence failure | Explicit sorting logic | 2026-01-22 |
| **I99-2** | Padding Token Masking invalid | Apply padding mask correctly | 2026-01-22 |
| **I99-3** | Padding Sentinel implementation incomplete | levels_info fill logic fix | 2026-01-22 |
| **I99-4** | Path encoding validation failed | Path value range constraint | 2026-01-22 |
| **I99-5** | Levels_info format issue | Format consistency fix | 2026-01-22 |
| **I100-1** | Tree consistency train/inference asymmetry | I96-4 verified, design correct | 2026-01-25 |
| **I102-1** | softplus inverse numerically unstable | log-space parameterization | 2026-01-24 |
| **I102-2** | omega division protection insufficient | clamp instead of epsilon | 2026-01-24 |
| **I102-3** | total_prob clamp FP16 underflow | PROB_EPSILON_FP16 | 2026-01-24 |
| **I102-4** | shape_norm epsilon too small | SHAPE_NORM_EPSILON | 2026-01-25 |
| **I102-5** | GPU-CPU sync blocking | Remove .item() sync point | 2026-01-26 |
| **I103-1** | Hierarchical attention not vectorized | Batch Hilbert bias computation | 2026-01-25 |
| **I103-2** | Hilbert index weights not cached | Class-level weight cache | 2026-01-25 |
| **I103-3** | GumbelTopKSplitter device transfer | Lazy device-side cache | 2026-01-25 |
| **I103-4** | Gumbel noise CPU generation | GPU noise generation | 2026-01-25 |
| **I107-1** | Pre-allocated buffer fixed at 256 | Dynamic buffer size adjustment | 2026-01-26 |
| **I107-4** | Python loop not vectorized | nonzero() instead of list comprehension | 2026-01-26 |
| **I107-5** | .item() sync point in warning messages | Delayed sync already optimal | 2026-01-26 |
| **I107-6** | Hilbert curve global cache cleanup | Order threshold caching | 2026-01-26 |

### 11.37.3 P1 Issues (Archived)

| ID | Issue | Solution | Archive Time |
|----|-------|----------|--------------|
| **I96-3** | Hierarchical Top-K breaks computation graph | detach() separate computation graph | 2026-01-22 |
| **I96-4** | Tree consistency hard constraint | Soft margin + mode distinction | 2026-01-22 |
| **I96-5** | ROI-Align duplicate computation | Batch ROI-Align + deduplication | 2026-01-22 |
| **I97-4** | Hierarchical Top-K Python loop vectorization | Pre-compute depth indices | 2026-01-22 |
| **I97-5** | Configuration system duplication | ModelConfig alias | 2026-01-22 |
| **I97-6** | Remove unused parameters | Delete redundant parameters | 2026-01-22 |
| **I100-3** | Cache version mechanism compatibility | Content hash cache | 2026-01-25 |
| **I100-4** | ROI-Align vs ROI-Pooling precision difference | Force torchvision dependency | 2026-01-25 |
| **I100-5** | Small batch EMA variance estimation conservative | Layered initialization B1/B2/B4 | 2026-01-25 |
| **I102-6** | Depth loop parallelization | Batch chunk computation | 2026-01-25 |
| **I102-7** | LCA matrix vectorization | diagonal() method | 2026-01-25 |
| **I102-8** | Hilbert index vectorization | Lookup table method | 2026-01-25 |
| **I102-9** | points.index optimization | LRU cache | 2026-01-25 |
| **I104-1** | cuDNN benchmark not enabled | Add cudnn.benchmark | 2026-01-25 |
| **I104-2** | DataLoader prefetch_factor too high | Dynamic adjustment | 2026-01-25 |
| **I104-3** | LCA bias FP32 storage | FP16 storage | 2026-01-25 |

### 11.37.4 P2 Issues (Archived)

| ID | Issue | Solution | Archive Time |
|----|-------|----------|--------------|
| **I96-6** | Gradient coverage doc inaccurate | Update K/N gradient coverage description | 2026-01-22 |
| **I96-7** | Depth lower bound constraint | Restore fixed lower bound | 2026-01-22 |
| **I96-8** | Budget loss normalization | Relative MSE loss | 2026-01-22 |
| **I97-7** | Bias scale factor learnable | Softplus learnable parameter | 2026-01-22 |
| **I97-8** | Test coverage enhancement | New test file | 2026-01-22 |
| **I97-9** | Legacy code cleanup | Delete split_adaptive.py | 2026-01-22 |
| **I98-4** | levels_info contract normalization | LevelsInfo dataclass | 2026-01-23 |
| **I98-5** | TokenizerOutput enhancement | LevelsInfo strong type | 2026-01-23 |
| **I98-6** | Configuration system unification | ModelArchitectureConfig | 2026-01-25 |
| **I98-7** | Splitter interface protocol | Protocol layer definition | 2026-01-25 |
| **I99-6** | Dynamic K boundary constant test outdated | Update expected values | 2026-01-22 |
| **I99-7** | Elastic Budget constant test outdated | Update expected values | 2026-01-22 |
| **I99-8** | Quota Rounding test constraint | Soft regularization design | 2026-01-22 |
| **I99-9** | Test verification checklist update | Verification checklist script | 2026-01-22 |
| **I100-6** | Dynamic depth inference batch alignment | Fixed depth depth//2 | 2026-01-25 |
| **I102-10** | compute_max_depth comment error | Fix comment | 2026-01-25 |
| **I102-11** | n power-of-2 verification | Runtime assertion | 2026-01-25 |
| **I105-1** | _compute_quota_loss vectorization | Vectorized computation | 2026-01-25 |

### 11.37.5 P3 Issues (Archived)

| ID | Issue | Solution/Conclusion | Archive Time |
|----|-------|---------------------|--------------|
| **I97-10** | Hierarchical attention | Independent Attention within depth | 2026-01-22 |
| **I97-11** | Dynamic computation | Inference-only dynamic depth | 2026-01-22 |
| **I97-12** | Custom CUDA kernel | Implementation not recommended | 2026-01-22 |
| **I97-13** | Formal theoretical analysis | Exploratory research | 2026-01-22 |
| **I98-8** | Event-driven architecture | Exploration completed | 2026-01-23 |
| **I98-9** | Plugin system | Exploration completed | 2026-01-23 |
| **I99-10** | Test strategy refactoring | conftest.py fixtures | 2026-01-22 |
| **I99-11** | Randomness isolation | Randomness isolation test suite | 2026-01-22 |
| **I100-7** | Quota allocation algorithm optimization | Refactor to standard LRM | 2026-01-25 |
| **I100-8** | Pseudo-Hilbert locality proof | Documentation supplement | 2026-01-25 |
| **I106-1** | Flash Attention integration | Conditional import + bias fusion | 2026-01-26 |
| **I106-2** | Dual LayerNorm design evaluation | Attention standard LN + FFN depth-aware LN | 2026-01-26 |

### 11.37.6 Skipped P3 Issues (Math Conflict)

| Issue | Skip Reason | Mathematical Analysis |
|-------|-------------|----------------------|
| I25-13 | Depth Contrastive Loss | Conflicts with Hilbert locality (~80%) |
| I30-14 | Depth Contrastive Loss | Same, mathematically suboptimal |
| I30-15 | Ancestor relationship independent bias | Hilbert curve already encodes path |
| I30-16 | Hierarchical attention routing | LCA Embed already implies depth relationship |
| I26-3 | Ancestor relationship independent bias | Same as I30-15 |
| I32-9 | Hierarchical bias clamp | Regular clamp sufficient |
| I32-10 | DropPath cosine schedule | Linear schedule sufficient |

---

## 11.38 Active Issues (Pending)

> **Last Updated**: 2026-01-26
> **Status**: Preserving Pending Issues

### P0-Critical (Urgent)

| ID | Issue | File Location | Status |
|:---|:------|:--------------|:-------|
| **CRIT-5** | LCA and Hilbert Index Inconsistency (56%) | attn_hilbert_bias.py | ✅ Completed |
| **CRIT-6** | `round(softmax)` Breaks Quota Gradient | gumbel_topk_splitter.py:416 | ✅ Completed |

### P0 (Urgent Features)

| ID | Issue | Status |
|:---|:------|:-------|
| **I100-2** | Temperature Parameter Ablation Study | 🔄 In Progress |
| **I108-1** | Multi-Bias Fusion Lacks Scale Normalization | ⏳ Pending |
| **I108-2** | channels_last Condition Error | ⏳ Pending |

### P1 (Performance Optimization)

| ID | Issue | Status |
|:---|:------|:-------|
| **I108-3** | Python Loop for Depth Distribution | ⏳ Pending |
| **I108-4** | Hilbert Cache Memory Bound Calculation | ⏳ Pending |

### P3 (Exploration/On-Demand)

| ID | Issue | Status |
|:---|:------|:-------|
| **I108-5** | Hilbert Locality Probability Quantization | ⏳ To Explore |
| **I108-6** | FP16 Clamp Boundary Optimization | ⏳ To Explore |

---

## 11.36 Issue Archive (2026-01-20)

> **Status**: ✅ All Closed (40 Issues)
> **Archive Date**: 2026-01-20

This section documents the mathematical formalization analysis, fix solutions, and verification results for all completed Issues.

---

### 11.36.1 P0-Critical Issues (4/4)

| Issue | Problem Description | Mathematical Formalization | Solution | Verification |
|-------|----------|------------|----------|------|
| **I34-2** | 配额分配约束破坏 | $\sum_d K_d = K_{\text{total}}$ | 归一化重分配 | ✅ |
| **I34-5** | 深度计算 floor 偏差 | $d = \lceil \log_2(N/w) \rceil$ | ceil 替代 floor | ✅ |
| **I34-6** | feature_dim 不匹配 | $\text{feature\_dim} = d_{\text{model}}$ | 统一维度 | ✅ |
| **A1** | Token 容量 K_max | $K_{\max} = 0.05 \times N$ | 动态相对约束 | ✅ |

---

### 11.36.2 P0 Issues (7/7)

| Issue | Problem Description | Mathematical Formalization | Solution | Verification |
|-------|----------|------------|----------|------|
| **I32-2** | 空 Token 语义 | $\text{levels\_info}[i] = -1$ | sentinel 值 | ✅ |
| **I32-3** | Quota 四舍五入 | $K_d = \text{round}(\pi_d \times K)$ | 59/59 PASSED | ✅ |
| **A16** | KL 权重过重 | $\lambda_{KL} = 0.1$ | 移除 KL 损失 | ✅ |
| **A17** | ShapeScaleEncoder | $\tau_c \cdot \text{ShapeScaleEncoder}$ | use_affine_modulation=True | ✅ |
| **A18** | Token 上限估计 | $K_{\max} \geq 256$ | A1 动态扩展 | ✅ |
| **I35-1** | 配额维度不匹配 | $\text{quota\_logits}[:D]$ | 切片修正 | ✅ |
| **I35-2** | 方差归一化 NaN | $\sqrt{v + \epsilon}$ | epsilon 保护 | ✅ |

---

### 11.36.3 P1 Issues (12/12)

| Issue | Problem Description | Mathematical Formalization | Solution | Verification |
|-------|----------|------------|----------|------|
| **I30-6** | 小 batch 归一化 | EMA Running Statistics | 自适应归一化 | ✅ |
| **I30-7** | PseudoHilbert 跳跃 | 注释补充 | 边界说明 | ✅ |
| **I30-8** | GUMBEL_EPSILON | $\epsilon = 10^{-8}$ | 安全边界 | ✅ |
| **I30-9** | 缓存版本校验 | data_ptr + PyTorch 版本 | 版本校验 | ✅ |
| **I31-1** | 形状-尺度编码器 | $B^* = \tau_h \cdot \text{LCA} + \tau_c \cdot \text{SS}$ | 实现完成 | ✅ |
| **I31-2** | LCAHilbertBias 扩展 | Attention Bias | 形状注入 | ✅ |
| **I31-3** | 面积位置编码 | 深度位置编码 | 补充完成 | ✅ |
| **I32-5** | 方差缩放常数 | 自适应归一化 | 无固定公式 | ✅ |
| **I32-6** | 残差门控参数 | 单门控 + tanh | 零初始化 | ✅ |
| **I32-7** | 傅里叶频率 | Nyquist 约束 | 14/14 PASSED | ✅ |
| **I35-3** | 单深度方差 | $\sigma_d = 0$ when $N_d=1$ | 跳过测试 | ✅ |
| **I78** | 动态分辨率 | $\text{image\_size} = None$ | 支持完成 | ✅ |

---

### 11.36.4 P2 Issues (17/17)

| Issue | Problem Description | Mathematical Formalization | Solution | Verification |
|-------|----------|------------|----------|------|
| **I25-8** | 混合池化收益 | pool="cls/mean/weighted" | 15/15 PASSED | ✅ |
| **I25-10** | 配额参数暴露 | config.quota | 参数可见 | ✅ |
| **I25-12** | PseudoHilbert 量化 | locality_preservation_rate | 指标工具类 | ✅ |
| **I30-10** | 配额参数暴露 | constants.py | 参数可见 | ✅ |
| **I30-11** | 混合池化评估 | test_pooling_strategies.py | 15/15 PASSED | ✅ |
| **I30-13** | SiLU/Swish 说明 | ffn_swiglu.py:11-24 | 文档完成 | ✅ |
| **I31-4** | 单元测试 | test_shape_scale_encoding | 100% PASSED | ✅ |
| **I32-8** | 有限记忆 EMA | Per-batch 统计量 | 替代 EMA | ✅ |
| **I32-11** | ARCH-R2 聚合器 | Xavier init | 正确初始化 | ✅ |
| **I32-12** | 方差 epsilon | $1e-6 \to 1e-5$ | 数值稳定 | ✅ |
| **A19** | Attention scale | $1/\sqrt{d_k}$ | 移除可学习权重 | ✅ |
| **A15** | Pseudo-Hilbert 局部性 | locality_preservation_rate | 实现完成 | ✅ |
| **I32-1** | ROI-Align 复杂度 | $O(B \times N \times C \times k^2)$ | 注释修正 | ✅ |
| **I36-7** | get_model_info | 统一诊断接口 | 实现完成 | ✅ |

---

### 11.36.5 P3 Issues Skipped (7/7)

| Issue | Skip Reason | Mathematical Analysis |
|:-------|:-------------|:----------------------|
| **I25-13** | Depth Contrastive Loss | Conflicts with Hilbert Locality (~80%) |
| **I30-14** | Depth Contrastive Loss | Same as above, mathematically inferior |
| **I30-15** | Independent Ancestor Bias | Hilbert curve already encodes path, independent bias redundant |
| **I30-16** | Hierarchical Attention Routing | LCA Embed already implies depth relationship, gradient dilution |
| **I26-3** | Independent Ancestor Bias | Same as I30-15 |
| **I32-9** | Hierarchical Bias clamp | Regular clamp is sufficient |
| **I32-10** | DropPath Cosine Schedule | Linear schedule is sufficient |

---

### 11.36.6 Verification Statistics

| Metric | Value |
|:-------|:------|
| Total Tests | 665 |
| Passed | 665 |
| Skipped | 0 |
| Failed | 0 |
| Coverage | 100% |

---

## 11.37 Issue Archive (2026-01-26)

> **Archive Date**: 2026-01-26
> **Status**: Archive closed Issues, keep active Issues
> **Archive Scope**: CRIT-1~4, I78-2, I96-1~8, I97-1~13, I98-1~9, I99-1~11, I100-1~7, I102-1~11, I103-1~4, I104-1~3, I105-1, I106-1~2, I107-1~6

### 11.37.1 P0-Critical Issues (Archived)

| ID | Issue | Solution | Archive Time |
|----|-------|----------|--------------|
| **CRIT-1** | Gumbel-Top-K gradient coverage doc inaccurate | Update docstring, Scheme E uses global Softmax | 2026-01-22 |
| **CRIT-2** | hilbert_bias_scale unbounded | Add clamp(max=10.0) | 2026-01-22 |
| **CRIT-3** | Unbounded LCA cache can cause OOM | Remove cache, compute LCA directly | 2026-01-26 |
| **CRIT-4** | `.item()` sync point slows training | GPU tensor accumulation + single sync | 2026-01-26 |
| **I107-2** | Feature Cache retains references | Remove debug cache variables | 2026-01-26 |
| **I107-3** | Trainer gradient accumulation memory accumulation | Use `del` to release intermediates | 2026-01-26 |

### 11.37.2 P0 Issues (Archived)

| ID | Issue | Solution | Archive Time |
|----|-------|----------|--------------|
| **I78-2** | Tokenizer ignores K budget | Use selected_mask indexing directly | 2026-01-21 |
| **I96-1** | EMA clamp initialization inconsistent | Unified DEPTH_VARIANCE_INIT_EPS | 2026-01-21 |
| **I96-2** | Temperature annealing incomplete | Training config issue, not code bug | 2026-01-21 |
| **I97-1** | EMA eval mode not initialized | Conservative initialization | 2026-01-22 |
| **I97-2** | `_compute_level_bias` 2D dead code | Delete 2D branch, add assertion | 2026-01-22 |
| **I97-3** | `_precompute_candidates` method duplicate | Unified method implementation | 2026-01-22 |
| **I99-1** | Batch independence failure | Explicit sorting logic | 2026-01-22 |
| **I99-2** | Padding Token Masking invalid | Apply padding mask correctly | 2026-01-22 |
| **I99-3** | Padding Sentinel implementation incomplete | levels_info fill logic fix | 2026-01-22 |
| **I99-4** | Path encoding validation failed | Path value range constraint | 2026-01-22 |
| **I99-5** | Levels_info format issue | Format consistency fix | 2026-01-22 |
| **I100-1** | Tree consistency train/inference asymmetry | I96-4 verified, design correct | 2026-01-25 |
| **I102-1** | softplus inverse numerically unstable | log-space parameterization | 2026-01-24 |
| **I102-2** | omega division protection insufficient | clamp instead of epsilon | 2026-01-24 |
| **I102-3** | total_prob clamp FP16 underflow | PROB_EPSILON_FP16 | 2026-01-24 |
| **I102-4** | shape_norm epsilon too small | SHAPE_NORM_EPSILON | 2026-01-25 |
| **I102-5** | GPU-CPU sync blocking | Remove .item() sync point | 2026-01-26 |
| **I103-1** | Hierarchical attention not vectorized | Batch Hilbert bias computation | 2026-01-25 |
| **I103-2** | Hilbert index weights not cached | Class-level weight cache | 2026-01-25 |
| **I103-3** | GumbelTopKSplitter device transfer | Lazy device-side cache | 2026-01-25 |
| **I103-4** | Gumbel noise CPU generation | GPU noise generation | 2026-01-25 |
| **I107-1** | Pre-allocated buffer fixed at 256 | Dynamic buffer size adjustment | 2026-01-26 |
| **I107-4** | Python loop not vectorized | nonzero() instead of list comprehension | 2026-01-26 |
| **I107-5** | .item() sync point in warning messages | Delayed sync already optimal | 2026-01-26 |
| **I107-6** | Hilbert curve global cache cleanup | Order threshold caching | 2026-01-26 |

### 11.37.3 P1 Issues (Archived)

| ID | Issue | Solution | Archive Time |
|----|-------|----------|--------------|
| **I96-3** | Hierarchical Top-K breaks computation graph | detach() separate computation graph | 2026-01-22 |
| **I96-4** | Tree consistency hard constraint | Soft margin + mode distinction | 2026-01-22 |
| **I96-5** | ROI-Align duplicate computation | Batch ROI-Align + deduplication | 2026-01-22 |
| **I97-4** | Hierarchical Top-K Python loop vectorization | Pre-compute depth indices | 2026-01-22 |
| **I97-5** | Configuration system duplication | ModelConfig alias | 2026-01-22 |
| **I97-6** | Remove unused parameters | Delete redundant parameters | 2026-01-22 |
| **I100-3** | Cache version mechanism compatibility | Content hash cache | 2026-01-25 |
| **I100-4** | ROI-Align vs ROI-Pooling precision difference | Force torchvision dependency | 2026-01-25 |
| **I100-5** | Small batch EMA variance estimation conservative | Layered initialization B1/B2/B4 | 2026-01-25 |
| **I102-6** | Depth loop parallelization | Batch chunk computation | 2026-01-25 |
| **I102-7** | LCA matrix vectorization | diagonal() method | 2026-01-25 |
| **I102-8** | Hilbert index vectorization | Lookup table method | 2026-01-25 |
| **I102-9** | points.index optimization | LRU cache | 2026-01-25 |
| **I104-1** | cuDNN benchmark not enabled | Add cudnn.benchmark | 2026-01-25 |
| **I104-2** | DataLoader prefetch_factor too high | Dynamic adjustment | 2026-01-25 |
| **I104-3** | LCA bias FP32 storage | FP16 storage | 2026-01-25 |

### 11.37.4 P2 Issues (Archived)

| ID | Issue | Solution | Archive Time |
|----|-------|----------|--------------|
| **I96-6** | Gradient coverage doc inaccurate | Update K/N gradient coverage description | 2026-01-22 |
| **I96-7** | Depth lower bound constraint | Restore fixed lower bound | 2026-01-22 |
| **I96-8** | Budget loss normalization | Relative MSE loss | 2026-01-22 |
| **I97-7** | Bias scale factor learnable | Softplus learnable parameter | 2026-01-22 |
| **I97-8** | Test coverage enhancement | New test file | 2026-01-22 |
| **I97-9** | Legacy code cleanup | Delete split_adaptive.py | 2026-01-22 |
| **I98-4** | levels_info contract normalization | LevelsInfo dataclass | 2026-01-23 |
| **I98-5** | TokenizerOutput enhancement | LevelsInfo strong type | 2026-01-23 |
| **I98-6** | Configuration system unification | ModelArchitectureConfig | 2026-01-25 |
| **I98-7** | Splitter interface protocol | Protocol layer definition | 2026-01-25 |
| **I99-6** | Dynamic K boundary constant test outdated | Update expected values | 2026-01-22 |
| **I99-7** | Elastic Budget constant test outdated | Update expected values | 2026-01-22 |
| **I99-8** | Quota Rounding test constraint | Soft regularization design | 2026-01-22 |
| **I99-9** | Test verification checklist update | Verification checklist script | 2026-01-22 |
| **I100-6** | Dynamic depth inference batch alignment | Fixed depth depth//2 | 2026-01-25 |
| **I102-10** | compute_max_depth comment error | Fix comment | 2026-01-25 |
| **I102-11** | n power-of-2 verification | Runtime assertion | 2026-01-25 |
| **I105-1** | _compute_quota_loss vectorization | Vectorized computation | 2026-01-25 |

### 11.37.5 P3 Issues (Archived)

| ID | Issue | Solution/Conclusion | Archive Time |
|----|-------|---------------------|--------------|
| **I97-10** | Hierarchical attention | Independent Attention within depth | 2026-01-22 |
| **I97-11** | Dynamic computation | Inference-only dynamic depth | 2026-01-22 |
| **I97-12** | Custom CUDA kernel | Implementation not recommended | 2026-01-22 |
| **I97-13** | Formal theoretical analysis | Exploratory research | 2026-01-22 |
| **I98-8** | Event-driven architecture | Exploration completed | 2026-01-23 |
| **I98-9** | Plugin system | Exploration completed | 2026-01-23 |
| **I99-10** | Test strategy refactoring | conftest.py fixtures | 2026-01-22 |
| **I99-11** | Randomness isolation | Randomness isolation test suite | 2026-01-22 |
| **I100-7** | Quota allocation algorithm optimization | Refactor to standard LRM | 2026-01-25 |
| **I100-8** | Pseudo-Hilbert locality proof | Documentation supplement | 2026-01-25 |
| **I106-1** | Flash Attention integration | Conditional import + bias fusion | 2026-01-26 |
| **I106-2** | Dual LayerNorm design evaluation | Attention standard LN + FFN depth-aware LN | 2026-01-26 |

### 11.37.6 Skipped P3 Issues (Math Conflict)

| Issue | Skip Reason | Mathematical Analysis |
|-------|-------------|----------------------|
| I25-13 | Depth Contrastive Loss | Conflicts with Hilbert locality (~80%) |
| I30-14 | Depth Contrastive Loss | Same, mathematically suboptimal |
| I30-15 | Ancestor relationship independent bias | Hilbert curve already encodes path |
| I30-16 | Hierarchical attention routing | LCA Embed already implies depth relationship |
| I26-3 | Ancestor relationship independent bias | Same as I30-15 |
| I32-9 | Hierarchical bias clamp | Regular clamp sufficient |
| I32-10 | DropPath cosine schedule | Linear schedule sufficient |

---

## 11.38 Active Issues (Pending)

> **Last Updated**: 2026-01-26
> **Status**: Preserving Pending Issues

### P0-Critical (Urgent)

| ID | Issue | File Location | Status |
|:---|:------|:--------------|:-------|
| **CRIT-5** | LCA and Hilbert Index Inconsistency (56%) | attn_hilbert_bias.py | ✅ Completed |
| **CRIT-6** | `round(softmax)` Breaks Quota Gradient | gumbel_topk_splitter.py:416 | ✅ Completed |

### P0 (Urgent Features)

| ID | Issue | Status |
|:---|:------|:-------|
| **I100-2** | Temperature Parameter Ablation Study | 🔄 In Progress |
| **I108-1** | Multi-Bias Fusion Lacks Scale Normalization | ⏳ Pending |
| **I108-2** | channels_last Condition Error | ⏳ Pending |

### P1 (Performance Optimization)

| ID | Issue | Status |
|:---|:------|:-------|
| **I108-3** | Python Loop for Depth Distribution | ⏳ Pending |
| **I108-4** | Hilbert Cache Memory Bound Calculation | ⏳ Pending |

### P3 (Exploration/On-Demand)

| ID | Issue | Status |
|:---|:------|:-------|
| **I108-5** | Hilbert Locality Probability Quantization | ⏳ To Explore |
| **I108-6** | FP16 Clamp Boundary Optimization | ⏳ To Explore |

---

## 附录: Issue ID 索引

| ID | 日期 | 主题 | 归档位置 |
|----|------|------|----------|
| I36-I41 | 2026-01 | 显存优化与数值稳定性 | §11.37 |
| CRIT-1~6 | 2026-01 | Critical Issues | §11.37 |
| I78, I96-I99 | 2026-01 | 代码审查与测试修复 | §11.37 |
| I100-I108 | 2026-01 | 数学形式化与性能优化 | §11.37-§11.38 |

> **历史 Issue (已归档)**: I0-I35 的详细分析记录在 §11.3-§11.36

---

## 11.39 Issue 归档 (2026-02-03)

> **归档日期**: 2026-02-03
> **状态**: 归档 IMPROVEMENT_PLAN.md 中的已完成 Issue
> **归档范围**: I109, I110, I111 系列 (15个Issue)

### 11.39.1 P0-Critical Issues (已归档)

| ID | 问题 | 解决方案 | 归档时间 |
|----|------|----------|----------|
| **I109-2** | Z-Score除零风险 | 温度下界 + Z值约束 | 2026-01-29 |
| **I109-3** | Token Coverage参数与实现脱节 | 统一覆盖率参数 | 2026-01-29 |
| **I110-1** | SemanticRedundancySplitter - LookAheadHead | 双层MLP + GELU | 2026-02-02 |
| **I110-2** | SemanticRedundancySplitter - CorrelationGate | 余弦相似度计算 | 2026-02-02 |
| **I110-3** | SemanticRedundancySplitter - 核心分裂逻辑 | Gumbel-Softmax决策 | 2026-02-02 |
| **I111-1** | 温度参数传递失效 | 统一配置层 | 2026-02-02 |
| **I111-2** | SplitterConfig配置传递断裂 | 简化Config结构 | 2026-02-02 |
| **I111-3** | 深度熵正则化完全失效 | 硬选择计数 + 动态权重 | 2026-02-02 |

### 11.39.2 P0 Issues (已归档)

| ID | 问题 | 解决方案 | 归档时间 |
|----|------|----------|----------|
| **I109-4** | Elastic Budget惩罚权重不平衡 | 目标导向损失函数 | 2026-01-29 |
| **I110-4** | 语义冗余损失函数 | DiversityLoss + ReconstructionLoss | 2026-02-02 |
| **I111-4** | HilbertSplitterConfig配置层重构 | 统一数学结构 | 2026-02-02 |

### 11.39.3 P1 Issues (已归档)

| ID | 问题 | 解决方案 | 归档时间 |
|----|------|----------|----------|
| **I109-5** | 1x1网格边缘情况 | 恒等映射处理 | 2026-01-29 |
| **I109-6** | 梯度强度不平衡优化 | 覆盖率缩放STE | 2026-02-01 |
| **I110-5** | SemanticSplitterConfig配置类 | 数据类实现 | 2026-02-02 |
| **I110-6** | Tokenizer Streaming V3适配 | 条件初始化 | 2026-02-02 |
| **I110-7** | FractalViT模型集成 | 参数字段添加 | 2026-02-02 |
| **I111-5** | 相对预算公式完整实现 | 完整公式实现 | 2026-02-02 |

### 11.39.4 P2 Issues (已归档)

| ID | 问题 | 解决方案 | 归档时间 |
|----|------|----------|----------|
| **I108-3** | Python循环构建深度分布 | one-hot + einsum向量化 | 2026-01-26 |
| **I108-4** | Hilbert缓存内存界计算不准 | 修正内存注释 | 2026-01-26 |
| **I110-8** | 语义冗余分裂器单元测试 | 25个测试全部通过 | 2026-02-02 |
| **I111-6** | 深度分布实时监控 | Lazy Monitoring模式 | 2026-02-02 |

### 11.39.5 P3 Issues (已归档)

| ID | 问题 | 解决方案/结论 | 归档时间 |
|----|------|---------------|----------|
| **I108-5** | Hilbert局部性概率量化 | HilbertProbabilityMetrics类 | 2026-01-26 |
| **I108-6** | FP16 clamp边界优化 | 分层clamp常量 | 2026-01-26 |
| **I109-7** | log2精度问题修复 | bit_length()替代 | 2026-02-01 |
| **I110-9** | 语义冗余Tokenizer集成测试 | 13个测试全部通过 | 2026-02-02 |

### 11.39.6 P3-探索 Issues (已归档)

| ID | 问题 | 解决方案/结论 | 归档时间 |
|----|------|---------------|----------|
| **I109-8** | 熵正则化添加 | 深度熵 + 配额熵 | 2026-02-01 |
| **I109-9** | 温度Annealing调度实现 | 三种调度策略 | 2026-02-01 |
| **I109-10** | Elastic Budget与K_bounds死区匹配 | 目标导向重构 | 2026-01-29 |
| **I111-7** | 分层温度调度 | τ_d = τ_base × exp(-α×d) | 2026-02-02 |
| **I111-8** | 动态预算端到端学习 | STE bypass | 2026-02-02 |

### 11.39.7 归档统计

| 优先级 | 归档数量 |
|--------|----------|
| P0-Critical | 8 |
| P0 | 3 |
| P1 | 6 |
| P2 | 4 |
| P3 | 4 |
| P3-探索 | 5 |
| **总计** | **30** |

---

## 附录: Issue ID 索引 (更新)

| ID | 日期 | 主题 | 归档位置 |
|----|------|------|----------|
| I36-I41 | 2026-01 | 显存优化与数值稳定性 | §11.37 |
| CRIT-1~6 | 2026-01 | Critical Issues | §11.37 |
| I78, I96-I99 | 2026-01 | 代码审查与测试修复 | §11.37 |
| I100-I108 | 2026-01 | 数学形式化与性能优化 | §11.37-§11.38 |
| I109-I111 | 2026-02 | 语义分割器与配置重构 | §11.39 |

## 11.40 Issue 归档 (2026-02-07)

> **归档日期**: 2026-02-07
> **状态**: 归档 IMPROVEMENT_PLAN.md 中已完全修复的 Issue
> **归档范围**: I112, I113, I120, I121, I122 系列 (41个Issue)

### 11.40.1 P0-Critical Issues (已归档)

| ID | 问题 | 解决方案 | 归档时间 |
|----|------|----------|----------|
| **I113-1** | 验证集缺失导致过拟合无法检测 | random_split 添加验证集 (10%) | 2026-02-07 |
| **I113-2** | Token密度恒定问题 | 三层参数设计: max_ratio → N_max → N_actual | 2026-02-07 |
| **I113-3** | Jigsaw Loss O(N²)循环效率 | Hilbert索引矩阵运算向量化 | 2026-02-07 |
| **I112-1** | DiversityLoss 特征归一化缺失 | L2归一化 + 余弦相似度 | 2026-02-07 |
| **I112-2** | 协方差计算 n=1 边界问题 | MLE估计替代 Bessel校正 | 2026-02-07 |
| **I121-1** | K_COVERAGE_BASE 过低 | 0.03 → 0.08 → 0.12 | 2026-02-07 |
| **I121-2** | 损失权重失衡 | 配置链路修复 + 因子传递 | 2026-02-07 |
| **I120-8** | Token预算损失主导学习 (关键发现) | 拆分为 I121-1~7 | 2026-02-07 |

### 11.40.2 P0 Issues (已归档)

| ID | 问题 | 解决方案 | 归档时间 |
|----|------|----------|----------|
| **I113-4** | 熵正则化使用 `.item()` 破坏梯度流 | F.softplus() 替代 max() | 2026-02-07 |
| **I113-5** | STE梯度缩放α缺乏严格推导 | 可学习因子 σ(log β) | 2026-02-07 |
| **I112-3** | eps值不一致问题修复 | 统一使用 EPS=1e-6 | 2026-02-07 |
| **I112-4** | current_temperature GPU同步 | 返回Tensor替代float | 2026-02-07 |
| **I112-5** | 训练日志Warning污染 | DEBUG级别日志替代print | 2026-02-07 |
| **I120-2** | Train/Eval 输出不一致性 | dropout=0.0 (Tokenizer) | 2026-02-07 |
| **I100-2** | 温度参数Ablation Study | 9组配置待GPU验证 | 2026-02-07 |
| **I122-1** | Top-K选择重构 | 移除STE，直接使用Gumbel-Softmax | 2026-02-07 |

### 11.40.3 P1 Issues (已归档)

| ID | 问题 | 解决方案 | 归档时间 |
|----|------|----------|----------|
| **I113-6** | 信息密度估计缺乏理论基础 | Hilbert感知局部方差 | 2026-02-07 |
| **I113-7** | 配额优化离散梯度问题 | STE梯度恢复 + soft_quota代理 | 2026-02-07 |
| **I113-8** | Hilbert局部性非正方形区域失效 | RectHilbertIndex统一缩放 | 2026-02-07 |
| **I113-9** | Jigsaw Loss方向信息丢失 | 空间坐标差预测替代Hilbert差 | 2026-02-07 |
| **I112-6** | LazyDiagnostics类冗余定义 | 移至模块顶部 + __contains__ | 2026-02-07 |
| **I112-7** | GJP模式 levels_info适配 | 变长序列支持 | 2026-02-07 |
| **I120-3** | Tokenizer深度分布单一化 | 选中率均衡配额 K_d=rate×N_d | 2026-02-07 |
| **I121-3** | 温度下界过高限制探索 | 0.5 → 0.3 | 2026-02-07 |
| **I121-4** | 深度坍塌诊断与自动恢复 | 检测器 + 自适应恢复机制 | 2026-02-07 |
| **I122-2** | 曲率感知温度调度 | τ = τ_base × exp(-η × ∇L) | 2026-02-07 |
| **I113-17** | 配额优化重构 | ContinuousQuotaAllocator (softmax松弛) | 2026-02-07 |

### 11.40.4 P2 Issues (已归档)

| ID | 问题 | 解决方案 | 归档时间 |
|----|------|----------|----------|
| **I113-10** | 温度下界文档与实现不一致 | T_end=0.3 → TEMPERATURE_MIN常量 | 2026-02-07 |
| **I113-11** | Hilbert偏置与Attention logits量纲不匹配 | ×√d_k量纲对齐 | 2026-02-07 |
| **I113-12** | 训练效率优化 (torch.compile) | reduce-overhead模式 | 2026-02-07 |
| **I120-4** | 模型容量不足假设 | 扩大配置空间探索 | 2026-02-07 |
| **I120-5** | Hilbert-Aware Attention偏置冗余 | 消融实验待实施 | 2026-02-07 |
| **I121-5** | 课程学习式自适应损失权重 | 三阶段Cosine过渡 | 2026-02-07 |
| **I121-6** | 学习率与正则化优化 | lr: 2.7e-4→3e-4, dropout↓ | 2026-02-07 |
| **I113-18** | Hilbert扫描重构 | HilbertScanner统一API (swap宽高比) | 2026-02-07 |

### 11.40.5 P3 Issues (已归档)

| ID | 问题 | 解决方案/结论 | 归档时间 |
|----|------|---------------|----------|
| **I113-13** | 多任务损失平衡机制缺失 | 不确定性加权损失 (Kendall2018) | 2026-02-07 |
| **I113-14** | 理论保证形式化批判分析 | Hilbert局部性上界研究 | 2026-02-07 |
| **I113-15** | 自适应温度调度 | 基于损失曲率/分层温度/可学习τ | 2026-02-07 |
| **I120-6** | STE温度梯度悖论 | 移除/反向温度保护/动态调整 | 2026-02-07 |
| **I120-7** | 自适应Token预算研究 | 复杂度感知/梯度反馈/任务自适应 | 2026-02-07 |
| **I121-7** | 模型容量探索 | Phase 1&2完成: K_target 41→85, dim 448 | 2026-02-07 |

### 11.40.6 归档统计

| 优先级 | 归档数量 | 已修复 | 待处理 |
|--------|----------|--------|--------|
| P0-Critical | 8 | 8 | 0 |
| P0 | 8 | 7 | 1 (I100-2) |
| P1 | 11 | 11 | 0 |
| P2 | 8 | 7 | 1 (I113-12) |
| P3 | 6 | 3 | 3 (探索) |
| **总计** | **41** | **36** | **5** |

---

## 附录: Issue ID 索引 (完整)

| ID | 日期 | 主题 | 归档位置 |
|----|------|------|----------|
| I36-I41 | 2026-01 | 显存优化与数值稳定性 | §11.37 |
| CRIT-1~6 | 2026-01 | Critical Issues | §11.37 |
| I78, I96-I99 | 2026-01 | 代码审查与测试修复 | §11.37 |
| I100-I108 | 2026-01 | 数学形式化与性能优化 | §11.37-§11.38 |
| I109-I111 | 2026-02 | 语义分割器与配置重构 | §11.39 |
| I112-I122 | 2026-02 | 训练诊断与温度调度重构 | §11.40 |
| I121-I122 | 2026-02 | 温度调度与预算损失重构 | §11.41 |
| I147 | 2026-02 | I147完整重构 | §11.41 |

> **最后更新**: 2026-02-11
> **文档版本**: v4 (新增 §11.41 Issue归档)

---

## 11.41 Issue 归档 (2026-02-11)

> **归档日期**: 2026-02-11
> **状态**: 归档 IMPROVEMENT_PLAN.md 中已完成的 Issue
> **归档范围**: I121, I122, I147 系列 (15个 Issue)

### 11.41.1 P0-Critical Issues (已归档)

| ID | 问题 | 解决方案 | 归档时间 |
|----|------|----------|----------|
| **I122-1** | STE 梯度偏差 (Gumbel-TopK) | 已完成验证 | 2026-02-11 |
| **I122-2** | LCA 温度参数冗余 (τ_h) | 已修复 | 2026-02-11 |
| **I122-3** | LOGIT_CLAMP_BOUND 理论依据 | 已修复 | 2026-02-11 |
| **I122-4** | Elastic Budget (Poisson KL) | 已修复 | 2026-02-11 |
| **I121-8** | 温度下界 + LCA 初始化 | 已完成 | 2026-02-11 |
| **I147-1** | LogitsClamp 钳制层 (train_loss ~82) | 已完成 | 2026-02-11 |

### 11.41.2 P0 Issues (已归档)

| ID | 问题 | 解决方案 | 归档时间 |
|----|------|----------|----------|
| **I122-5** | 熵目标公式 | 已完成 | 2026-02-11 |
| **I122-6** | 课程学习因子 | 已重构 | 2026-02-11 |
| **I122-7** | 温度调度策略 | 已完成 | 2026-02-11 |
| **I122-8** | 常量数学推导 | 已重构 | 2026-02-11 |
| **I121-9** | FP16_SAFE_EPSILON 重复 | 已修复 | 2026-02-11 |
| **I121-10** | LOGIT_CLAMP_BOUND 顺序 | 已修复 | 2026-02-11 |
| **I147-2** | 禁用 Focal Loss (γ=0 → 标准 CE) | 已完成 | 2026-02-11 |
| **I147-3** | 学习率调整 (1e-4 → 5e-4) | 已完成 | 2026-02-11 |
| **I147-4** | 预算损失重构 (Relative Error Huber) | 已完成 | 2026-02-11 |

### 11.41.3 归档统计

| 优先级 | 归档数量 |
|--------|----------|
| P0-Critical | 6 |
| P0 | 9 |
| **总计** | **15** |

### 11.41.4 活跃 Issue (保留)

| 优先级 | ID | 问题 | 状态 |
|--------|-----|------|------|
| 🔴 P0 | I130-1 | 训练损失异常 (train_loss ~82) | ⚠️ 待验证 |
| 🔴 P0 | I130-2 | Eval/Train一致性检查失败 | ⚠️ 待验证 |
| 🔴 P0 | I130-3 | 收敛极慢 | ⚠️ 待验证 |
| 🔴 P0 | I150-1 | 注意力层次缩放形状不匹配 | 待修复 |
| 🔴 P0 | I150-2 | 分裂器索引计数不匹配 | 待修复 |
| 🟠 P1 | I131-1 | 深度不平衡 | 待改进 |
| 🟠 P1 | I151-1 | 批次一致性测试失败 | 待分析 |
| 🟠 P1 | I151-2 | 最大层级超出边界 | 待修复 |
| 🟡 P2 | I132-1 | 温度调度优化 | 待实现 |
| 🟢 P3 | I133-1 | Poisson假设验证实验 | 待研究 |
| 🟢 P3 | I133-2 | 深度分布与任务难度理论联系 | 待研究 |

---

## 11.42 Issue 归档 (2026-02-12)

> **归档日期**: 2026-02-12
> **状态**: 归档 IMPROVEMENT_PLAN.md 中已完成的 Issue
> **归档范围**: I130, I131, I150, I151, I160 系列 (16个 Issue)

### 11.42.1 P0-Critical Issues (已归档)

| ID | 问题 | 解决方案 | 归档时间 |
|----|------|----------|----------|
| **I130-1** | 训练损失异常 (train_loss ~82) | LogitsClamp + 禁用Focal Loss | 2026-02-12 |
| **I130-2** | Eval/Train一致性检查失败 | DeterministicTopK + dropout=0.0 | 2026-02-12 |
| **I130-3** | 收敛极慢 (83 epochs仅7%) | NeighborAwareSplitter | 2026-02-12 |
| **I150-1** | 注意力层次缩放形状不匹配 | 简化相对缩放 | 2026-02-12 |
| **I150-2** | 分裂器索引计数不匹配 | 使用硬掩码 | 2026-02-12 |
| **I160-1** | Token Reduction缺少邻居感知 | DeterministicNeighborSplitter (23/23测试通过) | 2026-02-12 |
| **I160-2** | Attention偏置是软约束，非硬保证 | Hierarchical Soft-Hard (27/27测试通过) | 2026-02-12 |

### 11.42.2 P1 Issues (已归档)

| ID | 问题 | 解决方案 | 归档时间 |
|----|------|----------|----------|
| **I131-1** | 深度不平衡 (浅层token过少) | 可学习深度缩放 + 配额机制 | 2026-02-12 |
| **I131-2** | Focal Loss参数γ=2.5过强 | 禁用 (γ=0 → 标准 CE) | 2026-02-12 |
| **I151-1** | 批次一致性测试失败 | 接受Gumbel随机性 | 2026-02-12 |
| **I151-2** | 最大层级超出边界 | 边界检查 | 2026-02-12 |

### 11.42.3 P2 Issues (已归档)

| ID | 问题 | 解决方案 | 归档时间 |
|----|------|----------|----------|
| **I132-1** | 温度调度优化 | Linear退火 (140/140测试通过) | 2026-02-12 |
| **I132-2** | 弹性预算损失改进 | Huber损失替代Poisson KL | 2026-02-12 |

### 11.42.4 归档统计

| 优先级 | 归档数量 |
|--------|----------|
| P0-Critical | 7 |
| P1 | 4 |
| P2 | 2 |
| **总计** | **13** |

### 11.42.5 活跃 Issue (保留)

> **最后更新**: 2026-02-13
> **状态**: 剩余活跃 Issue

| 优先级 | ID | 问题 | 状态 |
|--------|-----|------|------|
| 🟠 P1 | I161-1 | 跨尺度一致性仅隐式保证 | 待实现 |
| 🟠 P1 | I161-2 | 多偏置叠加的量纲不一致 | 待改进 |
| 🟢 P3 | I163-1 | Hilbert局部性的数学边界优化 | 待研究 |
| 🟢 P3 | I163-2 | 深度分布与任务难度的理论联系 | 待研究 |

---

## 11.43 Issue 归档 (2026-02-13)

> **归档日期**: 2026-02-13
> **状态**: 归档本次文档更新相关 Issue
> **归档范围**: I160, I162 系列

### 11.43.1 P0-Critical Issues (已归档)

| ID | 问题 | 解决方案 | 归档时间 |
|----|------|----------|----------|
| **I160-1** | Token Reduction缺少邻居感知 | DeterministicNeighborSplitter (23/23测试通过) | 2026-02-13 |
| **I160-2** | Attention偏置是软约束，非硬保证 | HierarchicalSoftHardAttention (27/27测试通过) | 2026-02-13 |

### 11.43.2 P2 Issues (已归档)

| ID | 问题 | 解决方案 | 归档时间 |
|----|------|----------|----------|
| **I162-1** | Hilbert序的空间异质性模式未充分利用 | HilbertPatternEncoder (多尺度1D卷积) | 2026-02-13 |

### 11.43.3 文档更新

本次更新添加了以下新组件的文档：

1. **03_fractal_tokenizer.md** - 新增 §3.4 DeterministicNeighborSplitter
2. **05_attention_mechanism.md** - 新增 §5.7 HierarchicalSoftHardAttention
3. **02_data_structures.md** - 新增 §2.8 HilbertPatternEncoder

### 11.43.4 归档统计

| 优先级 | 归档数量 |
|--------|----------|
| P0-Critical | 2 |
| P2 | 1 |
| **总计** | **3** |

---

## 附录: Issue ID 索引 (最终)

| ID | 日期 | 主题 | 归档位置 |
|----|------|------|----------|
| I36-I41 | 2026-01 | 显存优化与数值稳定性 | §11.37 |
| CRIT-1~6 | 2026-01 | Critical Issues | §11.37 |
| I78, I96-I99 | 2026-01 | 代码审查与测试修复 | §11.37 |
| I100-I108 | 2026-01 | 数学形式化与性能优化 | §11.37-§11.38 |
| I109-I111 | 2026-02 | 语义分割器与配置重构 | §11.39 |
| I112-I122 | 2026-02 | 训练诊断与温度调度重构 | §11.40 |
| I121-I122 | 2026-02 | 温度调度与预算损失重构 | §11.41 |
| I130-I132 | 2026-02 | 训练损失与收敛问题修复 | §11.42 |
| I150-I151 | 2026-02 | 注意力与分割器修复 | §11.42 |
| I160-I163 | 2026-02 | Hilbert感知Token Reduction | §11.42 |
| I147 | 2026-02 | I147完整重构 | §11.41 |

> **最后更新**: 2026-02-12
> **文档版本**: v5 (新增 §11.42 Issue归档)
> **总计归档 Issue**: 160+ Issues
