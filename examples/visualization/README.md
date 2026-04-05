# Fractal Curve ViT: Interactive Visualization Dashboard

This directory contains an interactive Gradio application designed to demonstrate the mathematical properties and architectural innovations of the **Fractal Curve ViT**.

## Overview

The dashboard provides a hands-on way to explore:
1.  **Adaptive Tokenization**: See how the model splits images into variable-sized patches based on complexity.
2.  **Hilbert Ordering**: Visualize the space-filling curve that orders these patches.
3.  **Attention Mechanisms**: Inspect attention maps to see the effect of the Hilbert Bias ($B_{LCA}$).

## Files

- `app.py`: The main Gradio application entry point.
- `visualizer.py`: Contains the `FractalVisualizer` class responsible for generating matplotlib figures.
- `utils.py`: Helper functions for model loading and synthetic data generation.

## Usage

Ensure you have the project dependencies installed (including `gradio`).

Run the application:

```bash
uv run examples/visualization/app.py
```

Then open the displayed URL (usually `http://localhost:7860`) in your browser.

## Features

- **Input**: Upload your own image or use the generated synthetic image (which contains gradients, edges, and noise to test the splitter).
- **Tokenization Tab**: Displays the quadtree grid and the Hilbert path.
- **Attention Tab**: Use the slider to select a query token. The visualization shows:
    - **Learned Attention**: The actual attention weights from the model.
    - **LCA Bias**: The theoretical structural bias derived from the quadtree hierarchy.
- **Depth Tab**: Analyzes the distribution of patch sizes (depths).
