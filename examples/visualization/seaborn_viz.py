# -*- coding: utf-8 -*-
"""
Seaborn-based Static Visualizations for Fractal Curve ViT

Mathematical Focus:
==================
This module provides publication-quality static visualizations using Seaborn
to analyze the mathematical properties of the Fractal Curve Vision Transformer.

Key Visualizations:
1. Token depth distributions (histogram + KDE)
2. Attention heatmaps with LCA structure
3. Complexity metrics across image regions
4. Hilbert curve ordering analysis
5. Comparative analysis: Fractal ViT vs Standard ViT

Mathematical Formulations:
=========================
- Depth Distribution: P(d) = N_d / Σ_k N_k
- LCA Distance: LCA(i,j) = len(common_prefix(path_i, path_j))
- Complexity: C(R) = α·C_var(R) + (1-α)·C_grad(R)
- Hilbert Locality: ||p₁-p₂||₂ ≤ C·|H⁻¹(p₁)-H⁻¹(p₂)|^(1/2)
"""

import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path
import pandas as pd

# Set Seaborn style for publication-quality figures
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
sns.set_palette("husl")


class SeabornVisualizer:
    """
    Seaborn-based visualizer for mathematical analysis of Fractal ViT.
    
    Mathematical Properties Analyzed:
    ---------------------------------
    1. **Depth Distribution Entropy**: H = -Σ p(d)·log p(d)
       Measures multi-scale representation diversity
       
    2. **LCA Bias Matrix Structure**: B[i,j] = τ_h · Embed(LCA(i,j))
       Encodes hierarchical spatial relationships
       
    3. **Attention Pattern Locality**: A[i,j] = softmax(QK^T/√d + B_LCA)
       Shows influence of geometric prior on learned attention
       
    4. **Complexity-Depth Correlation**: Pearson ρ(C, d)
       Validates adaptive splitting hypothesis
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 150):
        """
        Initialize Seaborn visualizer.
        
        Args:
            figsize: Default figure size in inches
            dpi: Resolution for saved figures
        """
        self.figsize = figsize
        self.dpi = dpi
        self.colors = sns.color_palette("husl", 10)
    
    def plot_depth_distribution(
        self,
        depths: np.ndarray,
        title: str = "Token Depth Distribution",
        max_depth: int = 4,
        show_entropy: bool = True,
        save_path: Optional[Path] = None
    ) -> Figure:
        """
        Visualize depth distribution with KDE and statistical annotations.
        
        Mathematical Analysis:
        ---------------------
        - Histogram: Empirical P(d) = N_d / N
        - KDE: Kernel density estimate with Gaussian kernel
        - Entropy: H = -Σ p(d)·log₂ p(d) bits
        - Uniformity Test: χ² statistic vs uniform distribution
        
        Args:
            depths: Array of token depths [N]
            title: Plot title
            max_depth: Maximum depth for x-axis
            show_entropy: Whether to compute and display entropy
            save_path: Path to save figure (optional)
            
        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram with KDE overlay
        sns.histplot(
            depths, bins=range(max_depth + 2), 
            kde=True, stat='density', 
            alpha=0.6, ax=ax1, color=self.colors[0]
        )
        ax1.set_xlabel('Depth Level $d$', fontsize=12)
        ax1.set_ylabel('Density $P(d)$', fontsize=12)
        ax1.set_title(f'{title}\nHistogram + KDE', fontsize=13, fontweight='bold')
        ax1.set_xticks(range(max_depth + 1))
        ax1.grid(axis='y', alpha=0.3)
        
        # Compute statistics
        if show_entropy:
            unique, counts = np.unique(depths, return_counts=True)
            probs = counts / counts.sum()
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            max_entropy = np.log2(max_depth + 1)
            entropy_ratio = entropy / max_entropy
            
            # Add text box with statistics
            stats_text = (
                f'N = {len(depths)}\n'
                f'Mean depth: {depths.mean():.2f}\n'
                f'Std depth: {depths.std():.2f}\n'
                f'Entropy: {entropy:.3f} bits\n'
                f'Max entropy: {max_entropy:.3f} bits\n'
                f'Ratio: {entropy_ratio:.2%}'
            )
            ax1.text(
                0.02, 0.98, stats_text,
                transform=ax1.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10
            )
        
        # Count plot
        depth_counts = pd.DataFrame({
            'Depth': depths,
            'Count': np.ones_like(depths)
        })
        depth_summary = depth_counts.groupby('Depth').count().reset_index()
        
        sns.barplot(
            data=depth_summary, x='Depth', y='Count',
            ax=ax2, palette='viridis'
        )
        ax2.set_xlabel('Depth Level $d$', fontsize=12)
        ax2.set_ylabel('Token Count $N_d$', fontsize=12)
        ax2.set_title('Token Count by Depth', fontsize=13, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add count labels on bars
        for i, row in depth_summary.iterrows():
            ax2.text(
                i, row['Count'], f"{int(row['Count'])}",
                ha='center', va='bottom', fontsize=10
            )
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_lca_bias_matrix(
        self,
        lca_depths: np.ndarray,
        title: str = "LCA Bias Matrix",
        cmap: str = "YlOrRd",
        save_path: Optional[Path] = None
    ) -> Figure:
        """
        Visualize LCA depth matrix as a heatmap.
        
        Mathematical Interpretation:
        ---------------------------
        LCA(i,j) = Depth of Lowest Common Ancestor in quadtree
        
        Properties:
        - Diagonal: LCA(i,i) = d_i (self-distance)
        - Symmetric: LCA(i,j) = LCA(j,i)
        - Block Structure: Siblings share high LCA values
        
        This matrix is embedded into attention bias:
        B[i,j] = τ_h · LCAEmbed(LCA[i,j])
        
        Args:
            lca_depths: LCA depth matrix [N, N]
            title: Plot title
            cmap: Colormap name
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(
            lca_depths,
            cmap=cmap,
            square=True,
            cbar_kws={'label': 'LCA Depth'},
            ax=ax,
            linewidths=0.5,
            linecolor='gray',
            alpha=0.9
        )
        
        ax.set_xlabel('Token Index $j$', fontsize=12)
        ax.set_ylabel('Token Index $i$', fontsize=12)
        ax.set_title(
            f'{title}\n$LCA(i,j)$ = Depth of Lowest Common Ancestor',
            fontsize=13, fontweight='bold'
        )
        
        # Add mathematical formula
        formula_text = (
            r'$B[i,j] = \tau_h \cdot \mathrm{Embed}(\mathrm{LCA}(i,j))$' + '\n'
            r'Hierarchical spatial prior for attention'
        )
        ax.text(
            0.5, -0.1, formula_text,
            transform=ax.transAxes,
            ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7)
        )
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_attention_heatmap(
        self,
        attention_weights: np.ndarray,
        lca_depths: Optional[np.ndarray] = None,
        title: str = "Attention Weights",
        save_path: Optional[Path] = None
    ) -> Figure:
        """
        Visualize attention weights with optional LCA comparison.
        
        Mathematical Formulation:
        ------------------------
        Attention: A[i,j] = softmax(Q_i·K_j^T/√d + B_LCA[i,j])
        
        where:
        - Q_i·K_j^T: Content-based similarity
        - B_LCA[i,j]: Geometric prior from tree structure
        
        This visualization shows how learned attention correlates
        with the structural bias from the Hilbert quadtree.
        
        Args:
            attention_weights: Attention matrix [N, N]
            lca_depths: Optional LCA matrix for comparison
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure object
        """
        if lca_depths is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
        
        # Attention weights heatmap
        sns.heatmap(
            attention_weights,
            cmap='Blues',
            square=True,
            cbar_kws={'label': 'Attention Weight'},
            ax=ax1,
            vmin=0,
            vmax=attention_weights.max(),
            linewidths=0,
            alpha=0.9
        )
        
        ax1.set_xlabel('Key Token Index $j$', fontsize=12)
        ax1.set_ylabel('Query Token Index $i$', fontsize=12)
        ax1.set_title(
            f'{title}\n' + r'$A[i,j] = \mathrm{softmax}(QK^T/\sqrt{d} + B_{LCA})$',
            fontsize=13, fontweight='bold'
        )
        
        # LCA comparison if provided
        if lca_depths is not None:
            # Normalize LCA for comparison
            lca_norm = lca_depths / (lca_depths.max() + 1e-10)
            
            sns.heatmap(
                lca_norm,
                cmap='Reds',
                square=True,
                cbar_kws={'label': 'Normalized LCA Depth'},
                ax=ax2,
                vmin=0,
                vmax=1,
                linewidths=0,
                alpha=0.9
            )
            
            ax2.set_xlabel('Token Index $j$', fontsize=12)
            ax2.set_ylabel('Token Index $i$', fontsize=12)
            ax2.set_title(
                'LCA Structural Bias (Normalized)\n' + r'$B_{LCA}[i,j] \propto \mathrm{LCA}(i,j)$',
                fontsize=13, fontweight='bold'
            )
            
            # Compute correlation
            corr = np.corrcoef(attention_weights.flatten(), lca_norm.flatten())[0, 1]
            fig.text(
                0.5, 0.02,
                f'Pearson Correlation (Attention vs LCA): ρ = {corr:.3f}',
                ha='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5)
            )
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_complexity_analysis(
        self,
        complexities: np.ndarray,
        depths: np.ndarray,
        regions: Optional[np.ndarray] = None,
        title: str = "Complexity vs Depth Analysis",
        save_path: Optional[Path] = None
    ) -> Figure:
        """
        Analyze relationship between complexity and splitting depth.
        
        Mathematical Hypothesis:
        -----------------------
        H₀: Depth is positively correlated with region complexity
        C(R) = α·Var(R)/(Var(R)+σ₀²) + (1-α)·G(R)/(G(R)+g₀²)
        
        Validation:
        - Scatter plot: C vs d
        - Regression line: d = β₀ + β₁·C
        - Correlation: Pearson ρ and Spearman ρ_s
        
        Args:
            complexities: Complexity scores [N]
            depths: Corresponding depths [N]
            regions: Optional region areas [N]
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure object
        """
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(2, 2, figure=fig)
        
        # Scatter plot: Complexity vs Depth
        ax1 = fig.add_subplot(gs[0, 0])
        sns.scatterplot(
            x=complexities, y=depths,
            alpha=0.6, s=50, ax=ax1,
            color=self.colors[2]
        )
        sns.regplot(
            x=complexities, y=depths,
            scatter=False, ax=ax1,
            color=self.colors[3], line_kws={'linewidth': 2}
        )
        ax1.set_xlabel('Complexity $C(R)$', fontsize=12)
        ax1.set_ylabel('Depth $d$', fontsize=12)
        ax1.set_title('Complexity-Depth Relationship', fontsize=13, fontweight='bold')
        ax1.grid(alpha=0.3)
        
        # Compute statistics
        from scipy.stats import pearsonr, spearmanr
        pearson_r, pearson_p = pearsonr(complexities, depths)
        spearman_r, spearman_p = spearmanr(complexities, depths)
        
        stats_text = (
            f'Pearson ρ = {pearson_r:.3f} (p={pearson_p:.2e})\n'
            f'Spearman ρ_s = {spearman_r:.3f} (p={spearman_p:.2e})'
        )
        ax1.text(
            0.02, 0.98, stats_text,
            transform=ax1.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
            fontsize=10
        )
        
        # Box plot: Complexity distribution per depth
        ax2 = fig.add_subplot(gs[0, 1])
        data_df = pd.DataFrame({'Complexity': complexities, 'Depth': depths})
        sns.boxplot(data=data_df, x='Depth', y='Complexity', ax=ax2, palette='Set2')
        ax2.set_xlabel('Depth Level $d$', fontsize=12)
        ax2.set_ylabel('Complexity $C(R)$', fontsize=12)
        ax2.set_title('Complexity Distribution by Depth', fontsize=13, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # Violin plot
        ax3 = fig.add_subplot(gs[1, 0])
        sns.violinplot(data=data_df, x='Depth', y='Complexity', ax=ax3, palette='muted')
        ax3.set_xlabel('Depth Level $d$', fontsize=12)
        ax3.set_ylabel('Complexity $C(R)$', fontsize=12)
        ax3.set_title('Complexity Distribution (Violin)', fontsize=13, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        # Joint distribution
        ax4 = fig.add_subplot(gs[1, 1])
        sns.kdeplot(
            x=complexities, y=depths,
            fill=True, cmap='viridis', ax=ax4,
            levels=5, alpha=0.6
        )
        ax4.set_xlabel('Complexity $C(R)$', fontsize=12)
        ax4.set_ylabel('Depth $d$', fontsize=12)
        ax4.set_title('Joint Density (KDE)', fontsize=13, fontweight='bold')
        
        fig.suptitle(title, fontsize=15, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_comparative_analysis(
        self,
        fractal_stats: Dict[str, Any],
        standard_stats: Dict[str, Any],
        metrics: List[str] = ['num_tokens', 'entropy', 'flops'],
        title: str = "Fractal ViT vs Standard ViT",
        save_path: Optional[Path] = None
    ) -> Figure:
        """
        Compare Fractal ViT with Standard ViT on key metrics.
        
        Comparison Metrics:
        ------------------
        1. **Token Count**: N_fractal vs N_standard
        2. **Depth Entropy**: H_fractal vs H_standard (uniform=0)
        3. **FLOPs**: Computational complexity
        4. **Attention Sparsity**: Non-zero attention weights
        
        Args:
            fractal_stats: Dictionary of Fractal ViT statistics
            standard_stats: Dictionary of Standard ViT statistics
            metrics: List of metrics to compare
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
        if len(metrics) == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, metrics):
            values = [fractal_stats.get(metric, 0), standard_stats.get(metric, 0)]
            labels = ['Fractal ViT', 'Standard ViT']
            colors_pair = [self.colors[0], self.colors[5]]
            
            bars = ax.bar(labels, values, color=colors_pair, alpha=0.8, edgecolor='black')
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison', fontsize=13, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2., height,
                    f'{val:.2f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold'
                )
            
            # Calculate improvement
            if values[1] > 0:
                improvement = (values[0] - values[1]) / values[1] * 100
                ax.text(
                    0.5, 0.95, f'Change: {improvement:+.1f}%',
                    transform=ax.transAxes,
                    ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
                    fontsize=10
                )
        
        fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig


# Utility functions for data preparation
def compute_lca_depths_from_paths(paths: torch.Tensor) -> np.ndarray:
    """
    Compute LCA depth matrix from quadtree paths.
    
    Mathematical Definition:
    -----------------------
    LCA(i,j) = max{k : path_i[0:k] == path_j[0:k]}
    
    Args:
        paths: Quadtree paths [N, max_depth]
        
    Returns:
        LCA depth matrix [N, N]
    """
    N, D = paths.shape
    paths_np = paths.cpu().numpy()
    
    lca_matrix = np.zeros((N, N), dtype=np.int32)
    
    for i in range(N):
        for j in range(N):
            # Find length of common prefix
            lca = 0
            for k in range(D):
                if paths_np[i, k] == paths_np[j, k]:
                    lca += 1
                else:
                    break
            lca_matrix[i, j] = lca
    
    return lca_matrix


def extract_statistics(result, levels_info: torch.Tensor) -> Dict[str, Any]:
    """
    Extract comprehensive statistics from tokenization result.
    
    Args:
        result: Tokenization result
        levels_info: Level information tensor
        
    Returns:
        Dictionary of statistics
    """
    depths = levels_info[:, 0].cpu().numpy()
    
    # Compute entropy
    unique, counts = np.unique(depths, return_counts=True)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    
    return {
        'num_tokens': len(depths),
        'mean_depth': float(depths.mean()),
        'std_depth': float(depths.std()),
        'entropy': float(entropy),
        'depth_distribution': dict(zip(unique.tolist(), counts.tolist()))
    }
