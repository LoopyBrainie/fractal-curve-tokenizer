import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import torch

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from vit_pytorch.curve_hilbert import HilbertCurve

def visualize_path_trace(n=16):
    """
    Visualizes Raster vs Hilbert scan paths.
    """
    print(f"Generating path trace for grid size {n}x{n}...")
    
    # Raster Path
    raster_x = []
    raster_y = []
    for y in range(n):
        for x in range(n):
            raster_x.append(x)
            raster_y.append(y)

    # Hilbert Path
    hilbert_x = []
    hilbert_y = []
    num_points = n * n
    for d in range(num_points):
        x, y = HilbertCurve.d_to_xy(n, d)
        hilbert_x.append(x)
        hilbert_y.append(y)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot Raster
    axes[0].plot(raster_x, raster_y, marker='.', linestyle='-', alpha=0.6, markersize=2)
    axes[0].set_title(f"Raster Scan (n={n})")
    axes[0].invert_yaxis() # Image coordinates
    axes[0].set_aspect('equal')
    axes[0].grid(True, linestyle='--', alpha=0.3)

    # Plot Hilbert
    axes[1].plot(hilbert_x, hilbert_y, marker='.', linestyle='-', alpha=0.6, color='orange', markersize=2)
    axes[1].set_title(f"Hilbert Curve Scan (n={n})")
    axes[1].invert_yaxis()
    axes[1].set_aspect('equal')
    axes[1].grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), 'path_trace.png')
    plt.savefig(output_path)
    print(f"Saved path trace visualization to {output_path}")

if __name__ == "__main__":
    visualize_path_trace()
