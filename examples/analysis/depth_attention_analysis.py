# -*- coding: utf-8 -*-
"""
P6-3: Depth Attention 可视化分析工具

数学背景
========
验证模型是否学习了深度感知的注意力模式。

给定:
- Attention weights A ∈ R^{H×N×N}
- Token 深度 d_i ∈ {0, 1, ..., D}

构造深度对聚合矩阵 M ∈ R^{(D+1)×(D+1)}:
    M[d_q, d_k] = mean_{i,j: d_i=d_q, d_j=d_k} A[i,j]

分析指标
--------
1. 对角线主导性 (Diagonal Dominance):
   DiagRatio = Σ_d M[d,d] / Σ_{d_q,d_k} M[d_q,d_k]
   - 值 > 1/(D+1) 表示同深度 token 间注意力更强

2. 跨尺度注意力衰减:
   DecayRate = M[d,d] / M[d,d±1]
   - 值 > 1 表示深度越远注意力越弱

3. 深度对称性:
   Symmetry = ‖M - M^T‖_F
   - 值接近 0 表示注意力对称

使用方法
========
```python
from examples.analysis.depth_attention_analysis import DepthAttentionAnalyzer

# 创建分析器
analyzer = DepthAttentionAnalyzer(max_depth=4)

# 分析单个样本
metrics = analyzer.analyze(
    attention_weights,  # (H, N, N) or (B, H, N, N)
    depths              # (N,) or (B, N)
)

# 可视化
analyzer.plot_depth_attention_matrix(metrics)
analyzer.plot_attention_heatmap_by_depth(metrics)
```
"""

import sys
import math
from typing import Dict, Optional, Tuple, List, Any
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

# 可选依赖
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


class DepthAttentionAnalyzer:
    """深度感知注意力分析器。
    
    P6-3 实现: 验证模型是否学习了有意义的深度-注意力关系。
    
    Attributes:
        max_depth: 最大深度值
        depth_labels: 深度标签列表 ["d=0", "d=1", ...]
    """
    
    def __init__(self, max_depth: int = 4):
        """初始化分析器。
        
        Args:
            max_depth: 最大深度值 (包含)
        """
        self.max_depth = max_depth
        self.num_depths = max_depth + 1
        self.depth_labels = [f"d={d}" for d in range(self.num_depths)]
    
    def compute_depth_attention_matrix(
        self,
        attention: torch.Tensor,
        depths: torch.Tensor,
    ) -> torch.Tensor:
        """计算深度对聚合注意力矩阵。
        
        数学公式:
            M[d_q, d_k] = mean_{i,j: d_i=d_q, d_j=d_k} A[i,j]
        
        Args:
            attention: (H, N, N) 或 (N, N) 注意力权重
            depths: (N,) token 深度
            
        Returns:
            (D+1, D+1) 深度对聚合矩阵
        """
        if attention.dim() == 3:
            # (H, N, N) -> (N, N) 按 head 平均
            attention = attention.mean(dim=0)
        
        N = attention.shape[0]
        D = self.num_depths
        
        # 初始化累加矩阵和计数矩阵
        sum_matrix = torch.zeros(D, D, device=attention.device)
        count_matrix = torch.zeros(D, D, device=attention.device)
        
        # 确保 depths 是整数
        depths = depths.long().clamp(0, self.max_depth)
        
        # 向量化计算: 构建深度对索引
        # depth_i[i] = d_i, depth_j[j] = d_j
        depth_i = depths.unsqueeze(1).expand(N, N)  # (N, N)
        depth_j = depths.unsqueeze(0).expand(N, N)  # (N, N)
        
        # 使用 scatter_add 进行分组累加
        flat_idx = depth_i * D + depth_j  # (N, N), 范围 [0, D²-1]
        flat_sum = torch.zeros(D * D, device=attention.device)
        flat_count = torch.zeros(D * D, device=attention.device)
        
        flat_sum.scatter_add_(0, flat_idx.flatten(), attention.flatten())
        flat_count.scatter_add_(0, flat_idx.flatten(), torch.ones_like(attention).flatten())
        
        sum_matrix = flat_sum.view(D, D)
        count_matrix = flat_count.view(D, D)
        
        # 计算平均，避免除零
        avg_matrix = sum_matrix / (count_matrix + 1e-8)
        
        return avg_matrix
    
    def analyze(
        self,
        attention: torch.Tensor,
        depths: torch.Tensor,
        per_head: bool = False,
    ) -> Dict[str, Any]:
        """执行完整的深度注意力分析。
        
        Args:
            attention: (H, N, N) 或 (B, H, N, N) 注意力权重
            depths: (N,) 或 (B, N) token 深度
            per_head: 是否返回每个 head 的独立分析
            
        Returns:
            分析结果字典，包含:
            - depth_matrix: (D+1, D+1) 深度对聚合矩阵
            - diagonal_dominance: 对角线主导性分数
            - decay_rate: 跨尺度衰减率
            - symmetry_score: 对称性分数
            - per_head_matrices: (可选) 每个 head 的矩阵
        """
        # 处理 batch 维度
        if attention.dim() == 4:
            # (B, H, N, N) -> (H, N, N) 按 batch 平均
            attention = attention.mean(dim=0)
        if depths.dim() == 2:
            # (B, N) -> (N,) 取第一个样本的深度 (假设同 batch 深度一致)
            depths = depths[0]
        
        results = {}
        
        # 1. 计算整体深度矩阵
        depth_matrix = self.compute_depth_attention_matrix(attention, depths)
        results['depth_matrix'] = depth_matrix.cpu().numpy()
        
        # 2. 计算对角线主导性
        diag_sum = depth_matrix.diag().sum().item()
        total_sum = depth_matrix.sum().item()
        results['diagonal_dominance'] = diag_sum / (total_sum + 1e-8)
        results['expected_diagonal'] = 1.0 / self.num_depths  # 随机情况下的期望值
        
        # 3. 计算跨尺度衰减率
        decay_rates = []
        for d in range(self.num_depths):
            center_val = depth_matrix[d, d].item()
            neighbor_vals = []
            if d > 0:
                neighbor_vals.append(depth_matrix[d, d-1].item())
            if d < self.max_depth:
                neighbor_vals.append(depth_matrix[d, d+1].item())
            if neighbor_vals:
                avg_neighbor = sum(neighbor_vals) / len(neighbor_vals)
                if avg_neighbor > 1e-8:
                    decay_rates.append(center_val / avg_neighbor)
        results['decay_rate'] = np.mean(decay_rates) if decay_rates else 1.0
        
        # 4. 计算对称性
        symmetry = torch.norm(depth_matrix - depth_matrix.T, p='fro').item()
        results['symmetry_score'] = symmetry
        
        # 5. Per-head 分析 (可选)
        if per_head and attention.dim() >= 3:
            H = attention.shape[0]
            per_head_matrices = []
            per_head_diag = []
            for h in range(H):
                head_matrix = self.compute_depth_attention_matrix(
                    attention[h], depths
                )
                per_head_matrices.append(head_matrix.cpu().numpy())
                per_head_diag.append(head_matrix.diag().sum().item() / (head_matrix.sum().item() + 1e-8))
            results['per_head_matrices'] = per_head_matrices
            results['per_head_diagonal_dominance'] = per_head_diag
        
        return results
    
    def print_report(self, results: Dict[str, Any]) -> None:
        """打印分析报告。
        
        Args:
            results: analyze() 返回的结果字典
        """
        print("\n" + "=" * 60)
        print(" Depth Attention Analysis Report")
        print("=" * 60)
        
        # 深度矩阵
        print("\n📊 Depth-Pair Attention Matrix:")
        print("-" * 40)
        matrix = results['depth_matrix']
        
        # 打印表头
        header = "     " + "  ".join([f"{label:>6}" for label in self.depth_labels])
        print(header)
        
        # 打印矩阵
        for i, row_label in enumerate(self.depth_labels):
            row_str = f"{row_label:>4} " + "  ".join([f"{matrix[i, j]:>6.3f}" for j in range(self.num_depths)])
            print(row_str)
        
        # 指标
        print("\n📈 Analysis Metrics:")
        print("-" * 40)
        
        diag_dom = results['diagonal_dominance']
        expected = results['expected_diagonal']
        diag_ratio = diag_dom / expected
        
        print(f"  Diagonal Dominance: {diag_dom:.4f}")
        print(f"  Expected (random):  {expected:.4f}")
        print(f"  Ratio:              {diag_ratio:.2f}x {'✅ (depth-aware)' if diag_ratio > 1.2 else '⚠️ (weak signal)'}")
        
        print(f"\n  Decay Rate:         {results['decay_rate']:.4f}")
        print(f"  Interpretation:     {'Same-depth > neighbors' if results['decay_rate'] > 1 else 'Cross-depth dominant'}")
        
        print(f"\n  Symmetry Score:     {results['symmetry_score']:.4f}")
        print(f"  Interpretation:     {'Symmetric' if results['symmetry_score'] < 0.1 else 'Asymmetric'}")
        
        # Per-head 分析
        if 'per_head_diagonal_dominance' in results:
            print("\n📊 Per-Head Diagonal Dominance:")
            print("-" * 40)
            for h, dd in enumerate(results['per_head_diagonal_dominance']):
                bar_len = int(dd * 40 / max(results['per_head_diagonal_dominance']))
                bar = "█" * bar_len
                print(f"  Head {h}: {dd:.4f} {bar}")
        
        print("\n" + "=" * 60)
    
    def plot_depth_attention_matrix(
        self,
        results: Dict[str, Any],
        title: str = "Depth-Pair Attention Matrix",
        save_path: Optional[str] = None,
    ) -> None:
        """绘制深度对注意力热图。
        
        Args:
            results: analyze() 返回的结果字典
            title: 图标题
            save_path: 保存路径 (可选)
        """
        if not HAS_MATPLOTLIB:
            print("Warning: matplotlib not installed, skipping plot")
            return
        
        matrix = results['depth_matrix']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if HAS_SEABORN:
            sns.heatmap(
                matrix,
                annot=True,
                fmt='.3f',
                cmap='YlOrRd',
                xticklabels=self.depth_labels,
                yticklabels=self.depth_labels,
                ax=ax
            )
        else:
            im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
            ax.set_xticks(range(self.num_depths))
            ax.set_yticks(range(self.num_depths))
            ax.set_xticklabels(self.depth_labels)
            ax.set_yticklabels(self.depth_labels)
            plt.colorbar(im, ax=ax)
            
            # 添加数值标注
            for i in range(self.num_depths):
                for j in range(self.num_depths):
                    ax.text(j, i, f'{matrix[i, j]:.3f}',
                           ha='center', va='center', fontsize=9)
        
        ax.set_xlabel('Key Depth (d_k)')
        ax.set_ylabel('Query Depth (d_q)')
        ax.set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.show()
    
    def plot_per_head_comparison(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None,
    ) -> None:
        """绘制每个 head 的深度矩阵对比图。
        
        Args:
            results: analyze() 返回的结果字典 (需要 per_head=True)
            save_path: 保存路径 (可选)
        """
        if not HAS_MATPLOTLIB:
            print("Warning: matplotlib not installed, skipping plot")
            return
        
        if 'per_head_matrices' not in results:
            print("Warning: per_head analysis not available, call analyze(per_head=True)")
            return
        
        matrices = results['per_head_matrices']
        n_heads = len(matrices)
        
        # 计算子图网格
        n_cols = min(4, n_heads)
        n_rows = (n_heads + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
        axes = np.atleast_2d(axes).flatten()
        
        vmin = min(m.min() for m in matrices)
        vmax = max(m.max() for m in matrices)
        
        for h, (matrix, ax) in enumerate(zip(matrices, axes)):
            im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=vmin, vmax=vmax)
            ax.set_xticks(range(self.num_depths))
            ax.set_yticks(range(self.num_depths))
            ax.set_xticklabels(self.depth_labels, fontsize=8)
            ax.set_yticklabels(self.depth_labels, fontsize=8)
            ax.set_title(f'Head {h}', fontsize=10)
            
            # 对角线主导性
            diag_dom = results['per_head_diagonal_dominance'][h]
            ax.text(0.02, 0.98, f'DD={diag_dom:.2f}', 
                   transform=ax.transAxes, fontsize=8,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 隐藏多余的子图
        for ax in axes[n_heads:]:
            ax.axis('off')
        
        fig.colorbar(im, ax=axes.tolist(), shrink=0.6, label='Attention')
        fig.suptitle('Per-Head Depth Attention Patterns', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.show()


def demo():
    """演示分析器的使用方法。"""
    print("\n" + "=" * 60)
    print(" P6-3 Demo: Depth Attention Analyzer")
    print("=" * 60)
    
    # 创建模拟数据
    torch.manual_seed(42)
    
    max_depth = 4
    num_heads = 4
    seq_len = 64
    
    # 模拟深度分布
    depths = torch.randint(0, max_depth + 1, (seq_len,))
    
    # 模拟注意力权重 (加入深度感知偏置)
    attention = torch.rand(num_heads, seq_len, seq_len)
    
    # 添加同深度偏置: 同深度的 token 间注意力更强
    for i in range(seq_len):
        for j in range(seq_len):
            if depths[i] == depths[j]:
                attention[:, i, j] += 0.3  # 同深度偏置
    
    # Softmax 归一化 (模拟真实 attention)
    attention = torch.softmax(attention, dim=-1)
    
    # 创建分析器
    analyzer = DepthAttentionAnalyzer(max_depth=max_depth)
    
    # 执行分析
    results = analyzer.analyze(attention, depths, per_head=True)
    
    # 打印报告
    analyzer.print_report(results)
    
    # 绘制图表 (如果有 matplotlib)
    if HAS_MATPLOTLIB:
        print("\n📊 Generating plots...")
        analyzer.plot_depth_attention_matrix(results)
        analyzer.plot_per_head_comparison(results)
    else:
        print("\n⚠️ matplotlib not installed, skipping visualization")
    
    return results


if __name__ == "__main__":
    demo()
