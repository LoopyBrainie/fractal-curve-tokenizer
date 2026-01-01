import sys
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from vit_pytorch.tokenizer_streaming import StreamingFractalTokenizerV3
from vit_pytorch.split_adaptive import TensorSplitResult

def create_synthetic_image(size=224):
    """Creates a synthetic image with varying complexity."""
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    
    # Simple pattern: Circle in the center, high frequency noise in corner
    R = np.sqrt(X**2 + Y**2)
    img = np.zeros((size, size, 3))
    
    # Circle
    mask = (R < 0.5)
    img[mask] = [0.8, 0.2, 0.2] # Red circle
    
    # High freq noise
    noise_mask = (X > 0.5) & (Y > 0.5)
    img[noise_mask] = np.random.rand(np.sum(noise_mask), 3)
    
    # Background
    img[~mask & ~noise_mask] = [0.1, 0.1, 0.3] # Blue background
    
    return (img * 255).astype(np.uint8)

def visualize_splitter_heatmap():
    print("Initializing Tokenizer...")
    tokenizer = StreamingFractalTokenizerV3(
        image_size=224,
        channels=3,
        d_model=64,
        max_depth=4,
        learnable_temperature=1.0
    )
    
    # Create synthetic image
    img_np = create_synthetic_image()
    img_tensor = torch.from_numpy(img_np).float().permute(2, 0, 1).unsqueeze(0) / 255.0 # [1, 3, 224, 224]
    
    print("Running Tokenizer...")
    # We need to access the splitter result. 
    # The tokenizer.tokenize method calls splitter internally.
    # But we can also call splitter directly if we extract features first.
    
    with torch.no_grad():
        features = tokenizer.shared_conv(img_tensor)
        tensor_result = tokenizer.splitter(features, image_size=(224, 224), hard=True)
        
    regions = tensor_result.regions.cpu().numpy() # [N, 4] (x1, y1, x2, y2)
    depths = tensor_result.depths.cpu().numpy()   # [N]
    
    print(f"Generated {len(regions)} tokens.")
    
    # Visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img_np)
    
    # Color map for depths
    # Depth 0: Blue (Coarse) -> Depth 4: Red (Fine)
    cmap = plt.get_cmap('jet')
    max_depth = tokenizer.max_depth
    
    for region, depth in zip(regions, depths):
        x1, y1, x2, y2 = region
        width = x2 - x1
        height = y2 - y1
        
        color = cmap(depth / max_depth)
        
        # Create a Rectangle patch
        rect = patches.Rectangle(
            (x1, y1), width, height, 
            linewidth=1, 
            edgecolor=color, 
            facecolor=(*color[:3], 0.1) # Transparent fill
        )
        ax.add_patch(rect)
        
    # Add legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=cmap(d/max_depth), lw=4) for d in range(max_depth + 1)]
    ax.legend(custom_lines, [f'Depth {d}' for d in range(max_depth + 1)])
    
    ax.set_title("Splitter Decision Heatmap")
    ax.axis('off')
    
    output_path = os.path.join(os.path.dirname(__file__), 'splitter_heatmap.png')
    plt.savefig(output_path)
    print(f"Saved splitter heatmap to {output_path}")

if __name__ == "__main__":
    visualize_splitter_heatmap()
