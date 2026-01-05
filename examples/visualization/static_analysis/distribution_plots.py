import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from typing import List

import sys
import os

# Add project root to sys.path to allow imports from src and examples
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from examples.visualization.utils.quadtree_utils import QuadtreeVisualizer, Patch

# Define output directory
OUTPUT_DIR = os.path.join(project_root, "workspace", "visualizations", "static")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_synthetic_image(size=512):
    """Generate an image with varying complexity."""
    img = np.zeros((size, size), dtype=np.uint8)
    
    # Background gradient
    for i in range(size):
        img[i, :] = i * 255 // size
        
    # High frequency noise in top-left
    noise = np.random.randint(0, 255, (size//2, size//2), dtype=np.uint8)
    img[0:size//2, 0:size//2] = noise
    
    # Simple shape in bottom-right
    cv2.circle(img, (3*size//4, 3*size//4), size//8, 255, -1)
    
    return img

def plot_token_density(patches: List[Patch], image_size: int):
    """Plot a heatmap of token density."""
    density_map = np.zeros((image_size, image_size))
    
    for p in patches:
        # Each patch is 1 token. Density = 1 / area
        # Or simply count tokens per pixel? 
        # Let's visualize "Tokens per unit area". 
        # Smaller patches have higher density.
        val = 1.0 / (p.size * p.size)
        density_map[p.y:p.y+p.size, p.x:p.x+p.size] = val
        
    plt.figure(figsize=(10, 8))
    sns.heatmap(density_map, cmap="viridis", cbar_kws={'label': 'Token Density (1/pixel²)'})
    plt.title("Adaptive Token Density Map")
    plt.axis('off')
    output_path = os.path.join(OUTPUT_DIR, "token_density.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")

def plot_depth_distribution(patches: List[Patch]):
    """Plot histogram of patch depths."""
    depths = [p.depth for p in patches]
    
    plt.figure(figsize=(8, 6))
    sns.histplot(depths, discrete=True, shrink=0.8)
    plt.title("Token Depth Distribution")
    plt.xlabel("Quadtree Depth")
    plt.ylabel("Count")
    output_path = os.path.join(OUTPUT_DIR, "depth_distribution.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")

def main():
    img = generate_synthetic_image()
    viz = QuadtreeVisualizer(max_depth=6)
    patches = viz.split(img)
    
    print(f"Generated {len(patches)} tokens from adaptive splitting.")
    
    plot_token_density(patches, img.shape[0])
    plot_depth_distribution(patches)
    
    # Also save the visualization of the grid
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for p in patches:
        cv2.rectangle(img_bgr, (p.x, p.y), (p.x+p.size, p.y+p.size), (0, 255, 0), 1)
    output_path = os.path.join(OUTPUT_DIR, "quadtree_grid.png")
    cv2.imwrite(output_path, img_bgr)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    main()
