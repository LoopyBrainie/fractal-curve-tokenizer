# Fractal Curve ViT: Comprehensive Visualization Suite

This directory contains a comprehensive suite of visualization tools for understanding the mathematical properties and architectural innovations of the **Fractal Curve Vision Transformer**.

## Overview

The visualization suite provides multiple approaches to explore Fractal ViT:

1. **Interactive Dashboard (Gradio)**: Real-time exploration with your own images
2. **Static Visualizations (Seaborn)**: Publication-quality analytical plots
3. **Animated Explanations (Manim)**: Educational animations of core concepts
4. **Mathematical Analysis**: Formal critique and theoretical foundations

## Files

### Interactive Visualization
- `app.py`: Main Gradio application for interactive exploration
- `visualizer.py`: Original matplotlib-based visualizer (legacy)
- `utils.py`: Helper functions for model loading and data generation

### Static Analysis (NEW)
- `seaborn_viz.py`: **Publication-quality visualizations using Seaborn**
  - Depth distribution analysis with entropy metrics
  - LCA bias matrix heatmaps
  - Attention pattern analysis
  - Complexity-depth correlation studies
  - Comparative Fractal ViT vs Standard ViT analysis

### Animated Explanations (NEW)
- `manim_viz.py`: **Educational animations using ManimGL**
  - Hilbert curve generation (orders 1-5)
  - Quadtree splitting animation
  - Token ordering along Hilbert curve
  - Attention flow with LCA bias visualization

### Documentation (NEW)
- `MATHEMATICAL_ANALYSIS.md`: **Comprehensive mathematical critique**
  - Formal analysis of all mathematical components
  - Correctness proofs and complexity analysis
  - Training stability recommendations
  - Comparison with Standard ViT
  - Open research questions

## Usage

### Interactive Dashboard (Gradio)

```bash
# Run the interactive dashboard
uv run examples/visualization/app.py
```

Then open the displayed URL (usually `http://localhost:7860`) in your browser.

**Features:**
- Upload your own images or use synthetic test images
- **Tokenization Tab**: Quadtree decomposition with Hilbert path overlay
- **Attention Tab**: Compare learned attention vs theoretical LCA bias
- **Depth Tab**: Analyze token depth distribution and entropy

### Static Visualizations (Seaborn)

```python
from examples.visualization.seaborn_viz import SeabornVisualizer
import numpy as np

# Initialize visualizer
viz = SeabornVisualizer(figsize=(12, 8), dpi=150)

# Depth distribution analysis
depths = np.array([0, 1, 1, 2, 2, 2, 3, 3, 3, 3])
fig = viz.plot_depth_distribution(
    depths, 
    title="Token Depth Distribution",
    show_entropy=True,
    save_path="depth_dist.png"
)

# LCA bias matrix
lca_matrix = compute_lca_depths_from_paths(paths)
fig = viz.plot_lca_bias_matrix(
    lca_matrix,
    title="LCA Structural Bias",
    save_path="lca_bias.png"
)

# Complexity analysis
fig = viz.plot_complexity_analysis(
    complexities,
    depths,
    title="Complexity vs Depth Correlation",
    save_path="complexity_analysis.png"
)

# Comparative analysis
fractal_stats = {'num_tokens': 48, 'entropy': 1.2, 'flops': 2.3e9}
standard_stats = {'num_tokens': 196, 'entropy': 0.0, 'flops': 3.1e9}
fig = viz.plot_comparative_analysis(
    fractal_stats,
    standard_stats,
    metrics=['num_tokens', 'entropy', 'flops'],
    save_path="comparison.png"
)
```

### Animated Visualizations (Manim)

```bash
# Render Hilbert curve animation
manimgl examples/visualization/manim_viz.py HilbertCurveAnimation

# Render quadtree splitting animation
manimgl examples/visualization/manim_viz.py QuadtreeSplittingAnimation

# Render attention flow animation
manimgl examples/visualization/manim_viz.py AttentionFlowAnimation
```

Or use the Python API:

```python
from examples.visualization.manim_viz import (
    render_hilbert_animation,
    render_quadtree_animation,
    render_attention_animation
)

# Render animations
render_hilbert_animation("hilbert.mp4")
render_quadtree_animation("quadtree.mp4")
render_attention_animation("attention.mp4")
```

## Mathematical Foundations

Each visualization is grounded in rigorous mathematical formulations:

### 1. Hilbert Curve Properties
- **Bijection**: $H: [0, n^2) \leftrightarrow [0, n) \times [0, n)$
- **Locality**: $\|p_1 - p_2\|_2 \leq C \cdot |H^{-1}(p_1) - H^{-1}(p_2)|^{1/2}$
- **Self-Similarity**: Each quadrant contains rotated copy

### 2. Adaptive Tokenization
- **Complexity**: $C(R) = \alpha \cdot C_{\text{var}}(R) + (1-\alpha) \cdot C_{\text{grad}}(R)$
- **Splitting**: $\text{Split}(R) \iff C(R) > \tau_d$
- **Learnable**: End-to-end differentiable via Gumbel-Softmax

### 3. LCA Attention Bias
- **Definition**: $\text{LCA}(i,j) = \max\{k : \text{path}_i[0:k] = \text{path}_j[0:k]\}$
- **Bias**: $B[i,j] = \tau_h \cdot \text{Embed}(\text{LCA}(i,j))$
- **Efficiency**: $O(\log N)$ parameters vs $O(N^2)$ in Standard ViT

### 4. Depth Distribution Entropy
- **Entropy**: $H = -\sum_d p(d) \log p(d)$
- **Maximum**: $H_{\max} = \log(D+1)$ for uniform distribution
- **Target**: $H \approx 0.5 \times H_{\max}$ for balanced multi-scale

## For Experts: Key Differences from Traditional ViT

| Aspect | Standard ViT | Fractal ViT |
|--------|-------------|-------------|
| **Tokenization** | Fixed 16×16 grid | Adaptive quadtree (4×4 to 32×32) |
| **Token Count** | Constant $N = HW/P^2$ | Variable $N \in [N_{\min}, N_{\max}]$ |
| **Ordering** | Raster scan | Hilbert curve (locality-preserving) |
| **Position Encoding** | Learnable $N^2$ bias | LCA: $(d_{\max}+1) \times H \approx \log N$ |
| **Inductive Bias** | None | Spatial locality + multi-scale structure |

### Mathematical Advantages

1. **Content-Adaptive**: Allocates tokens based on local complexity
2. **Parameter-Efficient**: 99.8% reduction in position bias parameters
3. **Locality-Preserving**: Hilbert order maintains 2D spatial coherence
4. **Multi-Scale**: Natural representation at multiple resolutions

### When to Use Fractal ViT

✅ **Good for:**
- Images with varying complexity (natural scenes, medical images)
- Limited compute budget (adaptive token allocation)
- Applications requiring interpretability (explicit spatial structure)
- Multi-scale feature learning

⚠️ **Consider Standard ViT when:**
- Uniform images (documents, textures)
- Maximum throughput required (fixed computation)
- Simpler implementation preferred

## Citation

If you use these visualizations in your research, please cite:

```bibtex
@software{fractal_vit_viz,
  title = {Fractal Curve Vision Transformer: Comprehensive Visualization Suite},
  author = {[Author Names]},
  year = {2026},
  url = {https://github.com/LoopyBrainie/fractal-curve-tokenizer}
}
```

## References

1. Dosovitskiy, A., et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
2. Hilbert, D. (1891). "Über die stetige Abbildung einer Linie auf ein Flächenstück"
3. Jang, E., Gu, S., & Poole, B. (2017). "Categorical Reparameterization with Gumbel-Softmax"
