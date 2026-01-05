import sys
import os
from pathlib import Path
import torch
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.vit_pytorch.model_fractal_vit import FractalCurveViT
from examples.visualization.utils import create_synthetic_image, get_device
from examples.visualization.visualizer import FractalVisualizer

# Global state
model = None
device = None
visualizer = None
current_tokens = None
current_levels = None
current_attn = None
current_img_tensor = None
current_img_np = None

def initialize_model():
    global model, device, visualizer
    if model is None:
        device = get_device()
        print(f"Initializing model on {device}...")
        model = FractalCurveViT(
            image_size=256,
            num_classes=1000,
            dim=512,
            depth=6,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1,
            tokenizer_type='streaming_v3'
        ).to(device)
        model.eval()
        
        # Hook attention
        def hook_fn(module, input, output):
            global current_attn
            # output is (B, H, N, N)
            # Average over heads for visualization: (B, N, N)
            current_attn = output.mean(dim=1).detach()
            
        # Register hook on the last block
        model.transformer.layers[-1].attention.attend.register_forward_hook(hook_fn)
        
        visualizer = FractalVisualizer(model, device)
        print("Model initialized.")

def process_image(input_img):
    """
    Process the input image (numpy array from Gradio) through the model.
    """
    global current_tokens, current_levels, current_img_tensor, current_img_np
    
    initialize_model()
    
    # Preprocess image
    # Input is (H, W, 3) uint8 or float
    if input_img is None:
        # Use synthetic image
        img_tensor = create_synthetic_image(size=256).to(device)
        img_np = img_tensor[0].permute(1, 2, 0).cpu().numpy()
        # Normalize for display if needed, but create_synthetic_image returns [0,1]
    else:
        # Resize to 256x256
        from PIL import Image
        # input_img is numpy array, likely uint8
        pil_img = Image.fromarray(input_img)
        pil_img = pil_img.resize((256, 256), Image.Resampling.BILINEAR)
        img_np = np.array(pil_img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
    
    current_img_np = img_np
    current_img_tensor = img_tensor
    
    with torch.no_grad():
        # Tokenize
        token_output = model.tokenizer(img_tensor)
        seq = token_output.sequences[0]
        current_tokens = seq.tokens
        current_levels = seq.get_levels()
        
        # Run forward pass to trigger attention hook
        _ = model(img_tensor)
        
    # Generate Tokenization Plot
    fig_tok = visualizer.plot_tokenization(current_img_np, current_tokens, current_levels)
    
    # Generate Depth Plot
    fig_depth = visualizer.plot_depth_distribution(current_levels)
    
    # Update slider max
    num_tokens = current_tokens.shape[0]
    
    return fig_tok, fig_depth, gr.update(maximum=num_tokens-1, value=num_tokens//2, label=f"Select Token Index (0-{num_tokens-1})")

def update_attention(token_idx):
    global current_attn, current_tokens, current_levels, current_img_np
    
    if current_attn is None or current_tokens is None:
        return None, None
        
    fig_attn = visualizer.plot_attention(
        current_img_np, 
        current_tokens, 
        current_levels, 
        current_attn[0], # (N, N)
        int(token_idx)
    )
    
    fig_lca = visualizer.plot_lca_bias(
        current_img_np,
        current_tokens,
        current_levels,
        int(token_idx)
    )
    
    return fig_attn, fig_lca

# --- Gradio Interface ---

with gr.Blocks(title="Fractal Curve ViT Explorer") as demo:
    gr.Markdown(
        """
        # Fractal Curve ViT: Interactive Analysis Dashboard
        
        This dashboard visualizes the internal mechanisms of the **Fractal Curve Vision Transformer**, designed for Computer Vision experts.
        
        ## Mathematical Foundations
        
        1.  **Adaptive Tokenization**: $I \\xrightarrow{\\text{Split}} \\{R_i\\}_{i=1}^N$ via learnable complexity $C_\\theta(R)$.
        2.  **Hilbert Ordering**: Tokens are ordered by a space-filling curve $\\pi$, preserving 2D locality in 1D.
        3.  **Structured Attention**: $A_{ij} = \\text{softmax}(QK^T/\\sqrt{d} + B_{\\text{LCA}}(i,j))$, where $B_{\\text{LCA}}$ encodes quadtree distance.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Input Image", type="numpy")
            btn_process = gr.Button("Process / Generate Synthetic", variant="primary")
            
            gr.Markdown("### Analysis Controls")
            token_slider = gr.Slider(minimum=0, maximum=100, step=1, label="Select Token Index", value=0)
            
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("Tokenization & Hilbert Curve"):
                    plot_tok = gr.Plot(label="Quadtree Decomposition")
                    gr.Markdown(
                        """
                        **Visualization**: The image is recursively split into variable-sized patches based on content complexity.
                        - **Boxes**: Represent tokens $t_i$. Smaller boxes = higher complexity regions.
                        - **Color**: Indicates quadtree depth $d$.
                        - **Red Line**: The Hilbert curve path $\\pi$ connecting the tokens.
                        """
                    )
                    
                with gr.TabItem("Attention Mechanism"):
                    with gr.Row():
                        plot_attn = gr.Plot(label="Learned Attention Heatmap")
                        plot_lca = gr.Plot(label="Theoretical LCA Bias")
                    gr.Markdown(
                        """
                        **Comparison**:
                        - **Left**: Actual learned attention weights $A_{i,j}$.
                        - **Right**: Theoretical structural bias $B_{\\text{LCA}}(i,j)$ derived from the quadtree.
                        - **Insight**: Observe how the learned attention often aligns with the structural bias, validating the effectiveness of the LCA prior.
                        """
                    )
                    
                with gr.TabItem("Depth Distribution"):
                    plot_depth = gr.Plot(label="Depth Histogram")
                    gr.Markdown(
                        """
                        **Analysis**: Distribution of token depths.
                        - A peak at higher depths indicates complex textures.
                        - A peak at lower depths indicates smooth background regions.
                        """
                    )

    # Event Wiring
    btn_process.click(
        fn=process_image,
        inputs=[input_image],
        outputs=[plot_tok, plot_depth, token_slider]
    )
    
    token_slider.change(
        fn=update_attention,
        inputs=[token_slider],
        outputs=[plot_attn, plot_lca]
    )

if __name__ == "__main__":
    print("Starting Gradio App...")
    # Allow Gradio to find an available port by not specifying server_port
    demo.launch(server_name="0.0.0.0", share=False)
