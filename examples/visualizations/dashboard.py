import sys
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import math

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from vit_pytorch.tokenizer_streaming import StreamingFractalTokenizerV3
from vit_pytorch.curve_hilbert import PseudoHilbertCurve
from vit_pytorch.attn_hilbert_bias import LCAHilbertBias

try:
    import gradio as gr
except ImportError:
    print("Gradio is not installed. Please install it with `pip install gradio` to run the dashboard.")
    gr = None

def resize_keep_aspect(image, max_size=224):
    """Resizes image to fit within max_size while maintaining aspect ratio."""
    if image is None:
        return None
        
    h, w = image.shape[:2]
    scale = max_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    img_pil = Image.fromarray(image).resize((new_w, new_h), Image.Resampling.LANCZOS)
    return img_pil

def get_dense_complexity_map(tokenizer, features, h, w):
    """Computes a dense complexity map by evaluating the MLP on the finest grid."""
    splitter = tokenizer.splitter
    p = tokenizer.base_patch_size
    
    # Create a grid of boxes for the finest level
    grid_h = h // p
    grid_w = w // p
    
    if grid_h == 0 or grid_w == 0:
        return np.zeros((h, w))

    # Create boxes for all patches
    boxes = []
    for y in range(grid_h):
        for x in range(grid_w):
            x1 = x * p
            y1 = y * p
            x2 = x1 + p
            y2 = y1 + p
            # Batch index 0
            boxes.append([0, x1, y1, x2, y2])
            
    boxes_tensor = torch.tensor(boxes, device=features.device, dtype=features.dtype)
    
    # Scale boxes to feature map dimensions (1/p scale)
    scale = 1.0 / p
    boxes_tensor[:, 1:] *= scale
    
    # ROI Align
    from torchvision.ops import roi_align
    pooled = roi_align(
        features,
        boxes_tensor,
        output_size=(splitter.pool_size, splitter.pool_size),
        spatial_scale=1.0,
        aligned=True
    )
    
    # MLP
    flat = pooled.flatten(start_dim=1)
    with torch.no_grad():
        complexities = splitter.complexity_mlp(flat) # Logits
        probs = torch.sigmoid(complexities) # 0-1
        
    # Reshape to grid
    probs_grid = probs.view(grid_h, grid_w).cpu().numpy()
    
    # Resize to original image size for visualization
    map_img = Image.fromarray((probs_grid * 255).astype(np.uint8)).resize((w, h), Image.Resampling.NEAREST)
    return np.array(map_img)

def visualize_attention_bias(tokenizer, levels_info, num_tokens):
    """Visualizes the LCA Attention Bias matrix."""
    if num_tokens > 1024:
        return None, "Too many tokens to visualize attention bias (>1024)"
        
    # Instantiate LCA Bias
    lca_bias = LCAHilbertBias(
        max_depth=tokenizer.max_depth,
        heads=1,
        lca_temperature=1.5
    )
    
    # levels_info is [B, N, D+1]
    levels_info = levels_info.to(torch.device('cpu')) 
    
    with torch.no_grad():
        # Forward returns [B, H, N, N] or [H, N, N]
        bias = lca_bias(levels_info)
        
    if bias is None:
        return None, "Failed to compute bias"
        
    # bias is [1, 1, N, N] -> [N, N]
    bias_matrix = bias[0, 0].numpy()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(bias_matrix, cmap='viridis')
    plt.colorbar(im, ax=ax, label='Attention Bias (Log Scale)')
    ax.set_title(f"LCA Hilbert Bias ({num_tokens}x{num_tokens})")
    ax.set_xlabel("Key Token Index")
    ax.set_ylabel("Query Token Index")
    
    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return img, "Success"

def process_image(image, temperature, split_bias, show_path, max_depth, enforce_balance):
    if gr is None:
        return None, None, None, "Gradio not installed."

    if image is None:
        return None, None, None, "Please upload an image."
        
    img_pil = resize_keep_aspect(image, 384) # Increased resolution
    w, h = img_pil.size
    
    tokenizer = StreamingFractalTokenizerV3(
        image_size=(h, w),
        channels=3,
        d_model=64,
        max_depth=int(max_depth),
        learnable_temperature=temperature,
        enforce_balance=enforce_balance
    )
    
    img_tensor = torch.from_numpy(np.array(img_pil)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    
    # Apply Bias
    # Ensure offsets match max_depth
    if tokenizer.splitter.threshold_offsets.shape[0] != max_depth + 1:
         tokenizer.splitter.threshold_offsets = torch.nn.Parameter(torch.zeros(max_depth + 1))
         
    original_offsets = tokenizer.splitter.threshold_offsets.data.clone()
    tokenizer.splitter.threshold_offsets.data -= split_bias
    
    try:
        with torch.no_grad():
            features = tokenizer.shared_conv(img_tensor)
            tensor_result = tokenizer.splitter(features, image_size=(h, w), hard=True)
            
            # Get levels info for attention bias by running patch_embed
            # Use internal method for TensorSplitResult
            tokens, levels_info, _ = tokenizer._embed_with_tensor_result(features, tensor_result)
            
        regions = tensor_result.regions.cpu().numpy()
        depths = tensor_result.depths.cpu().numpy()
        num_tokens = len(regions)
        
        # --- 1. Main Visualization (Fractal Split) ---
        # Sort by Pseudo-Hilbert
        # Optimize: Create rank map once instead of calling xy_to_d repeatedly
        scan_path = PseudoHilbertCurve.scan(h, w)
        rank_map = np.zeros((h, w), dtype=np.int32)
        for d, (x, y) in enumerate(scan_path):
            rank_map[y, x] = d
            
        sorted_indices = []
        for i, region in enumerate(regions):
            x1, y1, x2, y2 = region
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cx = max(0, min(w - 1, cx))
            cy = max(0, min(h - 1, cy))
            
            d = rank_map[cy, cx]
            sorted_indices.append((d, i))
            
        sorted_indices.sort()
        perm = [i for d, i in sorted_indices]
        regions = regions[perm]
        depths = depths[perm]
        
        # Reorder levels_info to match visualization
        levels_info = levels_info[:, perm, :]
        
        fig_h = 8
        fig_w = 8 * (w / h)
        fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
        ax.imshow(img_pil)
        
        cmap = plt.get_cmap('jet')
        max_depth = tokenizer.max_depth
        
        for region, depth in zip(regions, depths):
            x1, y1, x2, y2 = region
            width = x2 - x1
            height = y2 - y1
            color = cmap(depth / max_depth)
            rect = patches.Rectangle(
                (x1, y1), width, height, 
                linewidth=1, 
                edgecolor=color, 
                facecolor=(*color[:3], 0.1)
            )
            ax.add_patch(rect)
            
        if show_path:
            centers_x = []
            centers_y = []
            for region in regions:
                x1, y1, x2, y2 = region
                centers_x.append((x1 + x2) / 2)
                centers_y.append((y1 + y2) / 2)
            ax.plot(centers_x, centers_y, color='white', linewidth=1.5, alpha=0.8, linestyle='-')
            if centers_x:
                ax.plot(centers_x[0], centers_y[0], 'go', markersize=5)
                ax.plot(centers_x[-1], centers_y[-1], 'ro', markersize=5)
            
        ax.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        viz_img = Image.open(buf)
        plt.close(fig)
        
        # --- 2. Complexity Map ---
        comp_map = get_dense_complexity_map(tokenizer, features, h, w)
        fig_c, ax_c = plt.subplots(figsize=(8, 8 * (h/w)))
        im_c = ax_c.imshow(comp_map, cmap='magma')
        plt.colorbar(im_c, ax=ax_c, label='Predicted Complexity')
        ax_c.axis('off')
        ax_c.set_title("Latent Complexity Map")
        
        buf_c = io.BytesIO()
        plt.savefig(buf_c, format='png', bbox_inches='tight', pad_inches=0)
        buf_c.seek(0)
        comp_img = Image.open(buf_c)
        plt.close(fig_c)
        
        # --- 3. Attention Bias ---
        attn_img, attn_msg = visualize_attention_bias(tokenizer, levels_info, num_tokens)
        if attn_img is None:
            fig_err, ax_err = plt.subplots()
            ax_err.text(0.5, 0.5, attn_msg, ha='center', va='center')
            ax_err.axis('off')
            buf_err = io.BytesIO()
            plt.savefig(buf_err, format='png')
            buf_err.seek(0)
            attn_img = Image.open(buf_err)
            plt.close(fig_err)

        stats = (
            f"Tokens: {num_tokens}\n"
            f"Grid Size: {w//4}x{h//4} ({ (w//4)*(h//4) } max)\n"
            f"Compression: {100 * (1 - num_tokens/((w//4)*(h//4))):.1f}%\n"
            f"Method: Pseudo-Hilbert Scan"
        )
        
        return viz_img, comp_img, attn_img, stats
        
    finally:
        tokenizer.splitter.threshold_offsets.data = original_offsets

if __name__ == "__main__":
    if gr:
        with gr.Blocks(title="分形曲线分词器仪表盘 (Fractal Curve Tokenizer Dashboard)") as demo:
            gr.Markdown("# 分形曲线分词器仪表盘 (Fractal Curve Tokenizer Dashboard)")
            gr.Markdown("探索分形 ViT 分词器的内部机制 (Explore the internal mechanics of the Fractal ViT Tokenizer).")
            
            with gr.Row():
                with gr.Column(scale=1):
                    input_img = gr.Image(type="numpy", label="输入图像 (Input Image)")
                    temp_slider = gr.Slider(0.1, 5.0, value=1.0, label="温度 (Temperature) - 控制分割的软硬程度")
                    bias_slider = gr.Slider(-5.0, 5.0, value=0.0, label="层级偏置 (Level Bias) - 控制分割的精细度")
                    max_depth_slider = gr.Slider(2, 6, value=4, step=1, label="最大深度 (Max Depth) - 四叉树的最大层数")
                    balance_check = gr.Checkbox(value=True, label="强制平衡 (Enforce Balance) - 限制相邻区域大小差异")
                    path_check = gr.Checkbox(value=True, label="显示伪希尔伯特路径 (Show Pseudo-Hilbert Path)")
                    btn = gr.Button("执行分词 (Tokenize)", variant="primary")
                    stats_box = gr.Textbox(label="统计信息 (Statistics)")
                
                with gr.Column(scale=2):
                    with gr.Tabs():
                        with gr.TabItem("分形分割 (Fractal Split)"):
                            out_viz = gr.Image(type="pil", label="分词可视化 (Tokenization Visualization)")
                        with gr.TabItem("复杂度热力图 (Complexity Map)"):
                            out_comp = gr.Image(type="pil", label="潜在复杂度 (Latent Complexity)")
                            gr.Markdown("此热力图显示了 LearnableSplitter 对每个区域预测的'复杂度'分数。高复杂度（较亮）区域会触发更细的分割。\n\n(This heatmap shows the 'complexity' score predicted by the LearnableSplitter for each region. High complexity (brighter) areas trigger finer splits.)")
                        with gr.TabItem("注意力偏置 (Attention Bias)"):
                            out_attn = gr.Image(type="pil", label="LCA 注意力偏置 (LCA Attention Bias)")
                            gr.Markdown("基于四叉树上最近公共祖先 (LCA) 距离的空间偏置矩阵可视化。颜色越深 = 树中距离越近 = 注意力偏置越高。\n\n(Visualization of the spatial bias matrix derived from the Lowest Common Ancestor (LCA) distance on the quadtree. Darker = Closer in tree = Higher attention bias.)")

            btn.click(
                process_image,
                inputs=[input_img, temp_slider, bias_slider, path_check, max_depth_slider, balance_check],
                outputs=[out_viz, out_comp, out_attn, stats_box]
            )
            
        demo.launch()
