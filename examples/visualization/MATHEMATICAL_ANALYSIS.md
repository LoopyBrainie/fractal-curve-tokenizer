# Mathematical Analysis of Fractal Curve Vision Transformer

## Executive Summary

This document provides a comprehensive mathematical critique and analysis of the Fractal Curve Vision Transformer (Fractal ViT) from a formal mathematical perspective, examining the theoretical foundations, algorithmic implementations, and architectural innovations.

**Key Findings:**
- ✅ **Hilbert Curve Properties**: Correctly implements locality-preserving space-filling curve
- ✅ **LCA Bias Mechanism**: Elegant O(log N) parameterization vs O(N²) in standard ViT
- ✅ **Adaptive Tokenization**: Mathematically sound complexity measure with learnable thresholds
- ⚠️ **Training Stability**: Requires careful temperature annealing and auxiliary losses

---

## 1. Hilbert Curve Formalization

### 1.1 Mathematical Definition

The Hilbert curve is a continuous fractal mapping:

$$H: [0, n^2) \leftrightarrow [0, n) \times [0, n), \quad n = 2^k$$

**Key Properties:**

1. **Bijection**: Every point in the grid is visited exactly once
2. **Continuity**: The curve is continuous in the limit
3. **Locality Preservation**: 
   $$\|p_1 - p_2\|_2 \leq C \cdot |H^{-1}(p_1) - H^{-1}(p_2)|^{1/2}$$
   where $C$ is a dimension-dependent constant (typically $C \approx 2$)

### 1.2 Implementation Analysis (curve_hilbert.py)

**Correctness**: The implementation uses the standard bit-interleaving algorithm with rotation transformations.

**Key Functions:**

```python
def xy_to_d(n: int, x: int, y: int) -> int
```
- **Algorithm**: Gray code-based encoding with quadrant rotations
- **Complexity**: $O(\log n)$ time, $O(1)$ space
- **Caching**: LRU cache with 4096 entries (effective for repeated queries)

**Mathematical Verification:**
- ✅ Bijection property holds (verified by `d_to_xy ∘ xy_to_d = id`)
- ✅ Locality bound satisfied (empirically tested with `compute_locality()`)

### 1.3 Pseudo-Hilbert Extension

For non-power-of-2 dimensions, the code implements a **Pseudo-Hilbert** curve:

**Hybrid Strategy:**
- If $\frac{n_{\text{padded}}^2}{H \times W} < \frac{4}{3}$: Use standard Hilbert + filter
- Else: Use recursive subdivision

**Mathematical Justification:**
- Padding overhead: $\rho = \frac{n^2}{HW}$
- Threshold $\rho^* = \frac{4}{3}$ derived from locality loss analysis
- Empirical locality: $\approx 1.49$ (vs $\sqrt{2} \approx 1.414$ for perfect Hilbert)

**Critique:** This is a reasonable engineering trade-off, though the locality bound degrades slightly.

---

## 2. Adaptive Quadtree Tokenization

### 2.1 Complexity Function

The splitting decision is based on a **learnable complexity** measure:

$$C_\theta(R) = \sigma\left(\text{MLP}\left(\text{ROI-Align}(F, R)\right)\right)$$

where:
- $F \in \mathbb{R}^{D \times H' \times W'}$: Shared feature map
- $\text{ROI-Align}$: Differentiable spatial pooling
- $\text{MLP}$: Multi-layer perceptron (no saturation issues)
- $\sigma$: Sigmoid activation, $C_\theta(R) \in [0, 1]$

### 2.2 Splitting Criterion

**Mathematical Formulation:**

$$\text{Split}(R) \iff C_\theta(R) > \tau_d \land d < d_{\max} \land \text{size}(R) \geq \text{min\_size}$$

where:
- $\tau_d$: Learnable depth-dependent threshold
- $\tau_d = \tau_{\text{base},d} + \delta_d$, with $\delta_d$ learned
- $d$: Current depth in quadtree

### 2.3 Gumbel-Softmax Relaxation

For differentiability during training:

$$z = \text{GumbelSoftmax}\left(\log p_{\text{split}}, \tau(t)\right)$$

with temperature annealing:

$$\tau(t) = \tau_{\max} \cdot \left(\frac{\tau_{\min}}{\tau_{\max}}\right)^{t/T}$$

**Recommended Parameters** (from code analysis):
- $\tau_{\max} = 1.0$: Initial exploration
- $\tau_{\min} = 0.3$: Final exploitation (raised from 0.1 to prevent gradient vanishing)

**Mathematical Critique:**

✅ **Advantages:**
- End-to-end differentiable
- No saturation issues (unlike variance-based complexity)
- Learns domain-specific splitting patterns

⚠️ **Challenges:**
- Requires careful temperature scheduling
- Needs auxiliary losses for stability (see Section 4)

### 2.4 Comparison with Fixed Schemes

The code previously supported:
- **Scheme B** (Balanced Greedy): Priority queue with 2:1 balance
- **Scheme C** (Fixed Budget DP): Dynamic programming

These were **removed** because:
1. **No Gradients**: Cannot be trained end-to-end
2. **Saturation**: Variance-based $C(R)$ saturates at high complexity
3. **Inferior Performance**: Empirical results favored learnable approach

**Mathematical Insight:** The learnable splitter is a **strictly more expressive** family, as it can approximate any fixed scheme through appropriate MLP weights.

---

## 3. LCA-Based Attention Bias

### 3.1 Mathematical Formulation

The attention mechanism is:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} \cdot \sigma_{\text{scale}} + B_{\text{hilbert}} + B_{\text{level}}\right) V$$

where:

$$B_{\text{hilbert}}[i,j] = \tau_h \cdot \text{LCAEmbed}(\text{LCA}(i,j))$$

### 3.2 LCA Definition

For quadtree-encoded tokens:

$$\text{LCA}(i, j) = \max\{k : \text{path}_i[0:k] = \text{path}_j[0:k]\}$$

**Geometric Interpretation:**

$$\text{LCA}(i, j) = \ell \implies \|pos_i - pos_j\|_\infty \leq \frac{N}{2^\ell}$$

### 3.3 Parameter Efficiency

**Standard ViT:** Learnable position bias $B \in \mathbb{R}^{N \times N}$ requires $O(N^2)$ parameters.

**Fractal ViT:** LCA embedding table:
- Parameters: $\text{LCAEmbed} \in \mathbb{R}^{(d_{\max}+1) \times H}$
- Complexity: $O(d_{\max} \cdot H) \approx O(\log N \cdot H)$
- Reduction: $\approx 99.8\%$ for typical N

**Temperature Scaling:**

$$\tau_h = \text{softplus}(\gamma_h)$$

Initialized such that $\tau_h \approx 1.5$, providing $1.5\sigma$ spatial prior.

### 3.4 Vectorized LCA Computation

The code implements vectorized path comparison:

```python
def compute_common_ancestor_depth(paths: Tensor) -> Tensor:
    """
    Args:
        paths: [B, N, D] quadtree paths
    Returns:
        lca_depths: [B, N, N]
    """
```

**Algorithm:**
```
matches[i,j,k] = (path_i[k] == path_j[k])
cumulative_match[i,j,k] = ∏_{k'≤k} matches[i,j,k']
lca_depths[i,j] = Σ_k cumulative_match[i,j,k]
```

**Complexity:** $O(B \cdot N^2 \cdot D)$ but fully parallelized on GPU.

**Caching:** Uses WeakRef to avoid data_ptr collisions across batches.

### 3.5 Mathematical Critique

✅ **Strengths:**
- Explicit geometric meaning
- Extremely parameter-efficient
- Encodes hierarchical structure

⚠️ **Limitations:**
- Assumes quadtree structure (not general graphs)
- LCA correctness depends on accurate path computation (P11-3 fix required)

---

## 4. Auxiliary Training Losses

Training the learnable splitter requires multiple auxiliary objectives to prevent collapse and maintain diversity.

### 4.1 Threshold Barrier Loss

**Purpose:** Constrain thresholds to valid logit space.

$$L_{\text{barrier}} = \lambda \cdot \sum_{d=0}^{D} \left[\max(0, \tau_{\min} - \tau_d)^2 + \max(0, \tau_d - \tau_{\max})^2\right]$$

**Parameters:**
- $\tau_{\min} = -3.0$, $\tau_{\max} = 3.0$ (logit space, post P11-10)
- $\lambda = 5.0$

**Mathematical Properties:**
- **Dead Zone:** When $\tau \in [-3, 3]$, $L = 0$ (no gradient interference)
- **Quadratic Penalty:** Steeper penalization for larger violations

### 4.2 Elastic Budget Loss (P10-9)

**Purpose:** Control token count range while allowing flexibility.

$$L_{\text{elastic}} = \lambda_{\text{over}} \cdot \phi(N - N_{\max}) + \lambda_{\text{under}} \cdot \psi(N_{\min} - N)$$

where:
- $\phi(x) = \frac{\text{ReLU}(x)^2}{N_{\max}}$: Quadratic penalty for exceeding budget
- $\psi(x) = \frac{\text{ReLU}(x)}{N_{\max}}$: Linear soft constraint for minimum

**Soft Token Count:**

$$N_{\text{soft}} = \sum_{d=0}^{D} R_d \cdot (1 - p_d)$$

with recursion:
$$R_0 = B, \quad R_{d+1} = R_d \cdot p_d \cdot 4$$

**Mathematical Insight:** This is the **expected number of leaf nodes** in the probabilistic quadtree.

**Gradient Flow:**
$$\frac{\partial N_{\text{soft}}}{\partial p_d} = -R_d + 4 \cdot \sum_{k>d} \frac{\partial R_k}{\partial p_d} \cdot (1 - p_k)$$

Both direct (fewer leaves at depth $d$) and indirect (more regions at deeper levels) effects contribute.

### 4.3 Soft Entropy Loss (P10-4/P10-5/P10-13)

**Purpose:** Encourage multi-scale diversity, prevent collapse to single depth.

**Improved Formulation (P10-13):**

$$L_{\text{entropy}} = \text{KL}(p || u) + w \cdot \text{ReLU}(p_{\max} - \theta)^2$$

where:
- $\text{KL}(p || u) = \log(D+1) - H(p)$: KL divergence to uniform
- $H(p) = -\sum_d p(d) \log p(d)$: Shannon entropy
- $p_{\max}$: Probability of most dominant depth
- $\theta = 0.8$: Collapse threshold (reduced from 0.9)
- $w = 1.0$: Anti-collapse weight

**Soft Depth Distribution:**

$$p(d) = \frac{L_d}{\sum_k L_k}$$

where $L_d$ is the expected number of leaves at depth $d$ (from §4.2).

**Mathematical Advantages:**
1. ✅ Always non-negative: $L \in [0, \log(D+1)]$
2. ✅ Minimum at uniform distribution: $L = 0$ when $p(d) = \frac{1}{D+1}$
3. ✅ Gradient direction unchanged from $-H(p)$ (only offset by constant)

### 4.4 Soft 2:1 Balance Loss (P-BAL-1)

**Purpose:** Enforce Hilbert curve 2:1 size constraint differentiably.

$$L_{\text{balance}} = \sum_{(i,j) \in \text{Adj}} \max(0, |\tilde{d}_i - \tilde{d}_j| - 1)^2$$

where $\tilde{d}_i$ is the **soft depth** (expected depth):

$$\tilde{d}_i = \sum_{d=0}^{D} d \cdot P(\text{depth}_i = d)$$

**Probability of Stopping at Depth $d$:**

$$P(\text{depth}_i = d) = \begin{cases}
\left[\prod_{k<d} p_k(R_i)\right] \cdot (1 - p_d(R_i)) & d < D \\
\prod_{k<D} p_k(R_i) & d = D
\end{cases}$$

**Gradient Analysis:**

$$\frac{\partial \tilde{d}_i}{\partial p_k} = \sum_{d>k} d \cdot \frac{\partial P(\text{depth}_i = d)}{\partial p_k}$$

This is **fully differentiable** through the probability chain.

### 4.5 Unified Loss Function

$$L_{\text{total}} = L_{\text{task}} + \alpha_1 L_{\text{barrier}} + \alpha_2 L_{\text{elastic}} + \alpha_3 L_{\text{entropy}} + \alpha_4 L_{\text{balance}}$$

**Recommended Weights:**
- $\alpha_1 = 5.0$: Barrier (high to enforce bounds)
- $\alpha_2 = 0.1$: Elastic (moderate, depends on task)
- $\alpha_3 = 0.1$: Entropy (moderate)
- $\alpha_4 = 0.01$: Balance (weak, often satisfied naturally)

---

## 5. Positional Encoding

### 5.1 Fractal Path Embedding

The model embeds both depth and quadtree path:

$$E_{\text{pos}}(t_i) = E_{\text{depth}}(d_i) + \sum_{k=1}^{d_i} E_{\text{path}}(q_k) \cdot w_k$$

where:
- $d_i$: Depth of token $i$
- $q_k \in \{0,1,2,3\}$: Quadrant at level $k$
- $w_k$: Learnable level weights (decay with depth)

### 5.2 Mathematical Properties

**Hierarchical Structure:** Tokens sharing longer path prefixes have more similar embeddings.

**Inductive Bias:** Encodes multi-scale spatial relationships explicitly.

---

## 6. Comparison with Standard ViT

| Property | Standard ViT | Fractal ViT |
|----------|-------------|-------------|
| **Tokenization** | Fixed $16\times 16$ grid | Adaptive quadtree |
| **Token Count** | $N = \frac{HW}{P^2}$ (constant) | $N \in [N_{\min}, N_{\max}]$ (variable) |
| **Patch Ordering** | Raster scan (row-major) | Hilbert curve |
| **Position Encoding** | Learnable $N^2$ bias | LCA: $(d_{\max}+1) \times H$ |
| **Inductive Bias** | None | Spatial locality + multi-scale |
| **Parameter Efficiency (Pos)** | $O(N^2)$ | $O(\log N \cdot H)$ |

### 6.1 Theoretical Advantages

1. **Content-Adaptive:** Allocates more tokens to complex regions
2. **Locality-Preserving:** Hilbert order maintains 2D spatial coherence
3. **Parameter-Efficient:** LCA bias uses $\log N$ instead of $N^2$ parameters
4. **Multi-Scale:** Naturally represents objects at multiple resolutions

### 6.2 Computational Complexity

**Tokenization:**
- Standard ViT: $O(HW)$ (fixed split)
- Fractal ViT: $O(HW \cdot D)$ (BFS tree traversal)

**Attention:**
- Standard ViT: $O(N^2 \cdot D)$
- Fractal ViT: $O(N^2 \cdot D)$ (same, but $N$ can be smaller)

**Overall:** Similar complexity, but Fractal ViT can use fewer tokens for the same perceptual quality.

---

## 7. Mathematical Soundness Assessment

### 7.1 Correctness

✅ **Hilbert Curve:** Correctly implements standard algorithm with proper caching  
✅ **LCA Computation:** Vectorized and efficient, with WeakRef caching  
✅ **Quadtree Structure:** Maintains 2:1 balance constraint differentiably  
✅ **Attention Mechanism:** Standard multi-head attention with valid geometric bias  

### 7.2 Numerical Stability

✅ **Softmax Clamping:** Logits clamped to $[-20, 20]$ to prevent overflow  
✅ **Epsilon Terms:** $\epsilon = 10^{-8}$ added to prevent log(0)  
✅ **Softplus Activations:** Ensure positive scales: $\sigma_{\text{scale}} = \text{softplus}(x) > 0$  
⚠️ **Mixed Precision:** FP16 can cause issues; code forces FP32 for critical computations  

### 7.3 Convergence Properties

⚠️ **Temperature Annealing:** Critical for convergence; low final temperature ($T < 0.3$) causes gradient vanishing  
⚠️ **Auxiliary Loss Balance:** Requires tuning; imbalance leads to collapse or budget violation  
✅ **Dead Zones:** Barrier and elastic losses have dead zones, preventing gradient interference during healthy training  

---

## 8. Open Mathematical Questions

1. **Optimal Temperature Schedule:** Is exponential annealing provably optimal for Gumbel-Softmax in this setting?

2. **LCA Bias Learnability:** Under what conditions does the LCA bias improve upon learnable position encodings?

3. **Multi-Scale Capacity:** What is the VC dimension or Rademacher complexity of the adaptive tokenization?

4. **Convergence Rate:** Can we derive convergence guarantees for the learnable splitter with auxiliary losses?

5. **Hilbert vs Other Curves:** Are Z-order or Peano curves viable alternatives with similar theoretical properties?

---

## 9. Recommendations for Practitioners

### 9.1 Hyperparameters

**Tokenization:**
- `max_depth`: 4 (images 224×224) or 5 (images 512×512)
- `min_region_size`: 7 (minimum for texture features)

**Temperature:**
- `T_start`: 1.0 (exploration)
- `T_end`: 0.3 (safe lower bound, per P11-11 analysis)
- `schedule`: 'exponential' (smooth annealing)

**Loss Weights:**
- `barrier_lambda`: 5.0
- `elastic_lambda_over`: 0.1
- `elastic_lambda_under`: 0.05 (raised in I14-1 D1)
- `entropy_weight`: 0.1
- `balance_weight`: 0.01

### 9.2 Training Tips

1. **Warm-up:** Use fixed (non-learnable) splitting for first few epochs
2. **Monitoring:** Track depth distribution entropy (target: $> 0.5 \times \log(D+1)$)
3. **Budget:** Set $N_{\max} = 4^{d_{\max}}$ (theoretical maximum)
4. **Initialization:** Initialize thresholds near 0 in logit space

### 9.3 Debugging

**Symptom:** Collapse to single depth  
**Diagnosis:** Check `depth_distribution_stats()`, dominant_prob > 0.8  
**Solution:** Increase `entropy_weight`, decrease `T_end`

**Symptom:** Too many tokens (exceeds budget)  
**Diagnosis:** `actual_token_count > N_max`  
**Solution:** Increase `elastic_lambda_over`, check threshold initialization

**Symptom:** Gradient vanishing  
**Diagnosis:** Loss plateaus, MLP gradients near zero  
**Solution:** Raise `T_end` to 0.3-0.5, check for FP16 overflow

---

## 10. Conclusion

The Fractal Curve Vision Transformer presents a mathematically rigorous approach to adaptive tokenization with strong theoretical foundations:

**Strengths:**
- ✅ Locality-preserving Hilbert curve with proven spatial properties
- ✅ Efficient LCA-based attention bias with $O(\log N)$ parameterization
- ✅ Fully differentiable learnable splitting via Gumbel-Softmax
- ✅ Comprehensive auxiliary losses for training stability

**Areas for Improvement:**
- ⚠️ Requires careful hyperparameter tuning (especially temperature)
- ⚠️ Additional computational cost during tokenization (BFS tree)
- ⚠️ Mixed precision training needs attention to numerical issues

**Overall Assessment:** The mathematical framework is sound and represents a significant advancement over fixed tokenization. The code implementation is mature, with numerous fixes addressing numerical stability and gradient flow. Recommended for applications requiring adaptive multi-scale representation.

---

## References

1. Hilbert, D. (1891). "Über die stetige Abbildung einer Linie auf ein Flächenstück"
2. Zhang, J., & Kamata, S. (2007). "A generalized Pseudo-Hilbert scan using arbitrary rectangular scanning grids"
3. Dosovitskiy, A., et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
4. Jang, E., Gu, S., & Poole, B. (2017). "Categorical Reparameterization with Gumbel-Softmax"

---

**Document Version:** 1.0  
**Last Updated:** 2026-01-05  
**Author:** AI-Assisted Mathematical Analysis
