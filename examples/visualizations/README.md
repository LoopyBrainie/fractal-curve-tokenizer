# Visualizations

This directory contains visualization scripts for the Fractal Curve Tokenizer project.

## Scripts

1.  **Path Trace Visualization** (`visualize_path_trace.py`)
    *   Visualizes the difference between Raster scan and Hilbert curve scan paths.
    *   Run: `python visualize_path_trace.py`
    *   Output: `path_trace.png`

2.  **Splitter Decision Heatmap** (`visualize_splitter_heatmap.py`)
    *   Visualizes the quadtree splitting decisions made by the tokenizer on a synthetic image.
    *   Run: `python visualize_splitter_heatmap.py`
    *   Output: `splitter_heatmap.png`

3.  **Attention Locality Analysis** (`visualize_attention_locality.py`)
    *   Visualizes the self-attention matrix of the Fractal ViT to demonstrate locality preservation (diagonal clustering).
    *   Run: `python visualize_attention_locality.py`
    *   Output: `attention_locality.png`

4.  **Interactive Dashboard** (`dashboard.py`)
    *   A Gradio-based interactive dashboard to experiment with tokenizer parameters.
    *   Requires `gradio`: `pip install gradio`
    *   Run: `python dashboard.py`
