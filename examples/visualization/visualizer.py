import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
import io
from typing import Tuple, List, Optional

def decode_box(depth: int, path: List[int], image_size: int = 256) -> Tuple[float, float, float, float]:
    """
    Decodes a quadtree path into a bounding box (x, y, w, h).
    Path is a list of quadrant indices (0=TL, 1=TR, 2=BL, 3=BR).
    """
    x, y = 0.0, 0.0
    w, h = float(image_size), float(image_size)
    
    for i in range(depth):
        w /= 2
        h /= 2
        q = path[i]
        
        if q == 0: # TL
            pass
        elif q == 1: # TR
            x += w
        elif q == 2: # BL
            y += h
        elif q == 3: # BR
            x += w
            y += h
            
    return x, y, w, h

def compute_lca_depth(path1: List[int], path2: List[int]) -> int:
    """
    Computes the depth of the Lowest Common Ancestor (LCA) of two paths.
    LCA depth is the length of the longest common prefix.
    """
    min_len = min(len(path1), len(path2))
    lca = 0
    for k in range(min_len):
        if path1[k] == path2[k]:
            lca += 1
        else:
            break
    return lca

class FractalVisualizer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.image_size = model.image_size if isinstance(model.image_size, int) else model.image_size[0]

    def plot_tokenization(self, image: np.ndarray, tokens: torch.Tensor, levels: torch.Tensor) -> Figure:
        """
        Visualizes the quadtree decomposition and Hilbert curve.
        Mathematical Concept: I -> {R_i} via C_theta(R)
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(image)
        
        centers_x = []
        centers_y = []
        
        cmap = plt.get_cmap('viridis')
        max_depth = levels[:, 0].max().item()
        if max_depth == 0: max_depth = 1
        
        N = tokens.shape[0]
        
        for i in range(N):
            depth = levels[i, 0].item()
            path = levels[i, 1:depth+1].tolist()
            x, y, w, h = decode_box(depth, path, self.image_size)
            
            centers_x.append(x + w/2)
            centers_y.append(y + h/2)
            
            rect = patches.Rectangle(
                (x, y), w, h, 
                linewidth=1, 
                edgecolor=cmap(depth / max_depth), 
                facecolor='none',
                alpha=0.7
            )
            ax.add_patch(rect)
            
            # Annotate first few tokens to show order
            if i < 5:
                ax.text(x + w/2, y + h/2, str(i), color='white', fontsize=10, ha='center', va='center', fontweight='bold')

        # Plot Hilbert Path
        # Use raw string for latex to avoid syntax warning
        ax.plot(centers_x, centers_y, color='red', linewidth=1.5, alpha=0.6, label=r'Hilbert Path $\pi$')
        
        ax.set_title(f"Fractal Tokenization ($N={N}$)\nColor=Depth $d$, Red Line=Ordering $\pi$")
        ax.axis('off')
        
        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max_depth))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r'Quadtree Depth $d$')
        
        plt.tight_layout()
        return fig

    def plot_attention(self, image: np.ndarray, tokens: torch.Tensor, levels: torch.Tensor, 
                      attn_map: torch.Tensor, query_idx: int) -> Figure:
        """
        Visualizes attention weights for a specific query token.
        Mathematical Concept: A_ij = softmax(QK^T + B_LCA)
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(image, alpha=0.5)
        
        N = tokens.shape[0]
        if query_idx >= N: query_idx = N - 1
        
        # Normalize attention
        attn_row = attn_map[query_idx, :].cpu().numpy()
        attn_min, attn_max = attn_row.min(), attn_row.max()
        if attn_max - attn_min > 1e-6:
            attn_norm = (attn_row - attn_min) / (attn_max - attn_min)
        else:
            attn_norm = np.zeros_like(attn_row)
            
        cmap = plt.get_cmap('plasma')
        
        for i in range(N):
            depth = levels[i, 0].item()
            path = levels[i, 1:depth+1].tolist()
            x, y, w, h = decode_box(depth, path, self.image_size)
            
            weight = attn_norm[i]
            
            # Draw box filled with attention color
            rect = patches.Rectangle(
                (x, y), w, h, 
                linewidth=0, 
                facecolor=cmap(weight),
                alpha=0.7 * weight
            )
            ax.add_patch(rect)
            
            # Highlight query
            if i == query_idx:
                rect_q = patches.Rectangle(
                    (x, y), w, h, 
                    linewidth=3, 
                    edgecolor='#00FF00',
                    facecolor='none'
                )
                ax.add_patch(rect_q)
                
        ax.set_title(f"Attention Map for Token $t_{{{query_idx}}}$\n$A_{{i,j}}$ Visualization")
        ax.axis('off')
        plt.tight_layout()
        return fig

    def plot_lca_bias(self, image: np.ndarray, tokens: torch.Tensor, levels: torch.Tensor, query_idx: int) -> Figure:
        """
        Visualizes the theoretical LCA bias for a specific query token.
        Mathematical Concept: B_LCA(i, j) propto LCA_Depth(i, j)
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(image, alpha=0.5)
        
        N = tokens.shape[0]
        if query_idx >= N: query_idx = N - 1
        
        query_depth = levels[query_idx, 0].item()
        query_path = levels[query_idx, 1:query_depth+1].tolist()
        
        lca_depths = []
        for i in range(N):
            depth = levels[i, 0].item()
            path = levels[i, 1:depth+1].tolist()
            lca = compute_lca_depth(query_path, path)
            lca_depths.append(lca)
            
        lca_depths = np.array(lca_depths)
        max_lca = lca_depths.max() if lca_depths.max() > 0 else 1
        
        cmap = plt.get_cmap('magma') # Different cmap for bias
        
        for i in range(N):
            depth = levels[i, 0].item()
            path = levels[i, 1:depth+1].tolist()
            x, y, w, h = decode_box(depth, path, self.image_size)
            
            # Normalize LCA depth for visualization
            # Higher LCA depth = closer in tree = higher bias
            weight = lca_depths[i] / max_lca
            
            rect = patches.Rectangle(
                (x, y), w, h, 
                linewidth=0, 
                facecolor=cmap(weight),
                alpha=0.7 * weight
            )
            ax.add_patch(rect)
            
            if i == query_idx:
                rect_q = patches.Rectangle(
                    (x, y), w, h, 
                    linewidth=3, 
                    edgecolor='#00FF00',
                    facecolor='none'
                )
                ax.add_patch(rect_q)
                
        ax.set_title(f"LCA Structural Bias for Token $t_{{{query_idx}}}$\n$B_{{LCA}}(i,j) \propto \text{{Depth}}(\text{{LCA}}(R_i, R_j))$")
        ax.axis('off')
        
        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max_lca))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('LCA Depth (Structural Proximity)')
        
        plt.tight_layout()
        return fig

    def plot_depth_distribution(self, levels: torch.Tensor) -> Figure:
        """
        Visualizes the distribution of token depths.
        Mathematical Concept: Distribution of C_theta(R)
        """
        depths = levels[:, 0].cpu().numpy()
        max_depth = int(depths.max())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        counts, bins, patches = ax.hist(depths, bins=range(max_depth + 2), align='left', rwidth=0.8, color='teal', alpha=0.7)
        
        ax.set_title("Token Depth Distribution")
        ax.set_xlabel(r"Depth Level $d$")
        ax.set_ylabel(r"Count $N_d$")
        ax.set_xticks(range(max_depth + 1))
        ax.grid(axis='y', alpha=0.3)
        
        # Add text annotations
        for i, count in enumerate(counts):
            if count > 0:
                ax.text(bins[i], count, str(int(count)), ha='center', va='bottom')
                
        plt.tight_layout()
        return fig
