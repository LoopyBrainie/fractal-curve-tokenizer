# Fractal ViT Visualization Suite

This directory contains visualization tools to demonstrate the mathematical principles of the Fractal Curve Tokenizer and Fractal ViT.

## Overview

The visualizations are divided into two categories:
1.  **Static Analysis (`static_analysis/`)**: Uses `seaborn` and `matplotlib` to generate plots showing token density, depth distribution, and comparisons with standard ViT.
2.  **Dynamic Animations (`manim_animations/`)**: Uses `manim` to create animations of the adaptive quadtree splitting, Hilbert curve traversal, and LCA-based attention.

## Mathematical Concepts

### 1. Adaptive Quadtree Tokenization
Unlike standard ViT which uses a fixed grid (e.g., 16x16 patches), Fractal ViT recursively splits the image based on a complexity function $C(R)$:

$$ C(R) = \alpha \cdot C_{var}(R) + (1-\alpha) \cdot C_{grad}(R) $$

Where $C_{var}$ is the normalized variance and $C_{grad}$ is the normalized gradient energy. A region $R$ is split if $C(R) > \tau_d$, where $\tau_d$ is a depth-dependent threshold.

### 2. Hilbert Curve Traversal
The 2D quadtree leaves are mapped to a 1D sequence using a recursive Hilbert curve. This preserves locality better than raster scan order.

### 3. LCA-Based Attention
The model uses a relative positional encoding based on the Lowest Common Ancestor (LCA) in the quadtree. The attention bias is a function of the tree distance between two tokens.

## Usage

### Prerequisites
*   Python 3.8+
*   `numpy`, `matplotlib`, `seaborn`, `opencv-python`
*   `manim` (for animations)

### Running Static Analysis
Run the driver script from the project root:
```bash
python examples/visualization/run_all.py
```
This will generate:
*   `token_density.png`: Heatmap of token density.
*   `depth_distribution.png`: Histogram of token depths.
*   `vit_comparison.png`: Visual comparison between fixed grid and adaptive quadtree.

### Running Animations
To generate the animations, use the `manim` command line tool:

**Tokenization Process:**
```bash
manim -pql examples/visualization/manim_animations/tokenization.py AdaptiveTokenization
```

**Attention Mechanism:**
```bash
manim -pql examples/visualization/manim_animations/attention.py LCAAttention
```
