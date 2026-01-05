import numpy as np
import matplotlib.pyplot as plt
import cv2

import sys
import os

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from examples.visualization.utils.quadtree_utils import QuadtreeVisualizer

# Define output directory
OUTPUT_DIR = os.path.join(project_root, "workspace", "visualizations", "static")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_synthetic_image(size=512):
    img = np.zeros((size, size), dtype=np.uint8)
    # Complex texture
    noise = np.random.randint(0, 255, (size, size), dtype=np.uint8)
    # Smooth background
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (size//2, size//2), size//3, 1, -1)
    
    img = np.where(mask==1, 0, noise) # Center is smooth, outside is noise
    return img

def main():
    size = 512
    img = generate_synthetic_image(size)
    
    # Standard ViT: Fixed 16x16 patches
    patch_size = 32
    fixed_grid_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    num_fixed_tokens = 0
    for y in range(0, size, patch_size):
        for x in range(0, size, patch_size):
            cv2.rectangle(fixed_grid_img, (x, y), (x+patch_size, y+patch_size), (0, 0, 255), 1)
            num_fixed_tokens += 1
            
    # Fractal ViT: Adaptive
    viz = QuadtreeVisualizer(max_depth=6, min_size=4)
    patches = viz.split(img)
    adaptive_grid_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for p in patches:
        cv2.rectangle(adaptive_grid_img, (p.x, p.y), (p.x+p.size, p.y+p.size), (0, 255, 0), 1)
        
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(fixed_grid_img)
    axes[0].set_title(f"Standard ViT (Fixed {patch_size}x{patch_size})\n{num_fixed_tokens} Tokens")
    axes[0].axis('off')
    
    axes[1].imshow(adaptive_grid_img)
    axes[1].set_title(f"Fractal ViT (Adaptive)\n{len(patches)} Tokens")
    axes[1].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "vit_comparison.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")

if __name__ == "__main__":
    main()
