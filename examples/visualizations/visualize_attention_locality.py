import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from vit_pytorch.model_fractal_vit import FractalCurveViT
from vit_pytorch.tokenizer_streaming import StreamingFractalTokenizerV3

def visualize_attention_locality():
    print("Initializing Model...")
    model = FractalCurveViT(
        image_size=224,
        num_classes=1000,
        dim=64,
        depth=2,
        heads=4,
        mlp_dim=128,
        tokenizer=StreamingFractalTokenizerV3(
            image_size=224,
            d_model=64,
            max_depth=4
        )
    )
    model.eval()
    
    # Hook to capture attention weights
    attention_weights = []
    def hook_fn(module, input, output):
        # output is the result of softmax(dots), shape [B, H, N, N]
        attention_weights.append(output.detach().cpu())

    # Register hook on the first layer's attention softmax
    # Structure: model.transformer.layers[0].attention.attend
    try:
        target_layer = model.transformer.layers[0].attention.attend
        handle = target_layer.register_forward_hook(hook_fn)
    except AttributeError as e:
        print(f"Error finding target layer: {e}")
        # Fallback to inspect structure
        print(model)
        return
    
    print("Running Forward Pass...")
    
    # Create synthetic image to ensure non-trivial tokenization
    x = np.linspace(-1, 1, 224)
    y = np.linspace(-1, 1, 224)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    img_np = np.zeros((224, 224, 3))
    mask = (R < 0.5)
    img_np[mask] = [0.8, 0.2, 0.2]
    noise_mask = (X > 0.5) & (Y > 0.5)
    img_np[noise_mask] = np.random.rand(np.sum(noise_mask), 3)
    img_np[~mask & ~noise_mask] = [0.1, 0.1, 0.3]
    img = torch.from_numpy(img_np).float().permute(2, 0, 1).unsqueeze(0) # [1, 3, 224, 224]
    
    with torch.no_grad():
        model(img)
        
    handle.remove()
    
    if not attention_weights:
        print("Error: No attention weights captured. Check model structure.")
        return

    attn = attention_weights[0][0] # First batch, [H, N, N]
    avg_attn = attn.mean(dim=0).numpy() # Average over heads, [N, N]
    
    print(f"Captured attention matrix of shape {avg_attn.shape}")
    
    # Visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    im = ax.imshow(avg_attn, cmap='viridis', interpolation='nearest')
    plt.colorbar(im, ax=ax)
    
    ax.set_title("Attention Matrix (Avg over Heads)")
    ax.set_xlabel("Key Token Index")
    ax.set_ylabel("Query Token Index")
    
    output_path = os.path.join(os.path.dirname(__file__), 'attention_locality.png')
    plt.savefig(output_path)
    print(f"Saved attention locality visualization to {output_path}")

if __name__ == "__main__":
    visualize_attention_locality()
