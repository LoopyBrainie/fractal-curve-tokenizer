# -*- coding: utf-8 -*-
"""
P6-4: Depth Embedding 分析工具

数学背景
========
验证 depth_embed 是否有效分离不同深度的表示。

给定 depth embeddings E ∈ R^{(D+1)×dim}，分析:

1. 余弦相似度矩阵:
   CosSim[i,j] = (e_i · e_j) / (‖e_i‖ ‖e_j‖)

2. 分离度 (Separation Score):
   Sep = (diag_mean - offdiag_mean) / offdiag_mean
   - 值 > 0 表示自相似性 > 跨深度相似性

3. 层次性 (Hierarchy Score):
   检验: CosSim[d, d+1] > CosSim[d, d+2]?
   - 相邻深度应比远距离深度更相似

4. 方差分析:
   - Intra-cluster: 同深度 token 嵌入的方差
   - Inter-cluster: 不同深度中心的方差

使用方法
========
```python
from examples.analysis.depth_embedding_analysis import DepthEmbeddingAnalyzer

# 从模型提取 depth embedding
depth_embed = model.fractal_tokenizer.depth_embed.weight.detach()

# 分析
analyzer = DepthEmbeddingAnalyzer()
results = analyzer.analyze(depth_embed)
analyzer.print_report(results)
analyzer.plot_similarity_matrix(results)
```
"""

import sys
from typing import Dict, Optional, Any, List
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

# 可选依赖
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


class DepthEmbeddingAnalyzer:
    """Depth Embedding 分析器。
    
    P6-4 实现: 验证深度嵌入是否有效分离不同深度。
    """
    
    def __init__(self):
        """初始化分析器。"""
        pass
    
    def compute_cosine_similarity_matrix(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """计算余弦相似度矩阵。
        
        Args:
            embeddings: (D+1, dim) 深度嵌入
            
        Returns:
            (D+1, D+1) 余弦相似度矩阵
        """
        # 归一化
        normed = F.normalize(embeddings, p=2, dim=-1)
        # 点积 = 余弦相似度
        sim_matrix = torch.mm(normed, normed.T)
        return sim_matrix
    
    def compute_euclidean_distance_matrix(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """计算欧几里得距离矩阵。
        
        Args:
            embeddings: (D+1, dim) 深度嵌入
            
        Returns:
            (D+1, D+1) 欧几里得距离矩阵
        """
        # 使用广播计算成对距离
        diff = embeddings.unsqueeze(0) - embeddings.unsqueeze(1)  # (D+1, D+1, dim)
        dist_matrix = torch.norm(diff, p=2, dim=-1)  # (D+1, D+1)
        return dist_matrix
    
    def analyze(
        self,
        embeddings: torch.Tensor,
        depth_scale: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """执行完整的深度嵌入分析。
        
        Args:
            embeddings: (D+1, dim) 深度嵌入权重
            depth_scale: (D+1,) 可选的 depth_scale 参数
            
        Returns:
            分析结果字典
        """
        embeddings = embeddings.detach().float()
        num_depths = embeddings.shape[0]
        dim = embeddings.shape[1]
        
        results = {
            'num_depths': num_depths,
            'embedding_dim': dim,
        }
        
        # 1. 余弦相似度矩阵
        cos_sim = self.compute_cosine_similarity_matrix(embeddings)
        results['cosine_similarity'] = cos_sim.cpu().numpy()
        
        # 2. 欧几里得距离矩阵
        euc_dist = self.compute_euclidean_distance_matrix(embeddings)
        results['euclidean_distance'] = euc_dist.cpu().numpy()
        
        # 3. 分离度 (Separation Score)
        # 对于余弦相似度: 对角线 = 1 (自相似)
        # 我们关心的是非对角线元素
        mask_diag = torch.eye(num_depths, dtype=torch.bool, device=cos_sim.device)
        mask_offdiag = ~mask_diag
        
        offdiag_mean = cos_sim[mask_offdiag].mean().item()
        offdiag_std = cos_sim[mask_offdiag].std().item()
        
        # 分离度定义: 1 - 平均非对角相似度 (越高越好)
        results['separation_score'] = 1.0 - offdiag_mean
        results['offdiag_similarity_mean'] = offdiag_mean
        results['offdiag_similarity_std'] = offdiag_std
        
        # 4. 层次性分析 (Hierarchy)
        # 检查: 相邻深度是否比远距离深度更相似?
        adjacent_sims = []
        distant_sims = []
        for d in range(num_depths):
            for delta in range(1, num_depths):
                if d + delta < num_depths:
                    sim = cos_sim[d, d + delta].item()
                    if delta == 1:
                        adjacent_sims.append(sim)
                    else:
                        distant_sims.append(sim)
        
        results['adjacent_similarity_mean'] = np.mean(adjacent_sims) if adjacent_sims else 0
        results['distant_similarity_mean'] = np.mean(distant_sims) if distant_sims else 0
        results['hierarchy_score'] = (
            results['adjacent_similarity_mean'] - results['distant_similarity_mean']
        )
        
        # 5. 嵌入范数分析
        norms = torch.norm(embeddings, p=2, dim=-1)
        results['embedding_norms'] = norms.cpu().numpy()
        results['norm_mean'] = norms.mean().item()
        results['norm_std'] = norms.std().item()
        
        # 6. 主成分分析 (PCA)
        # 计算协方差矩阵的特征值
        centered = embeddings - embeddings.mean(dim=0, keepdim=True)
        cov = torch.mm(centered.T, centered) / (num_depths - 1)
        eigenvalues = torch.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues.flip(0)[:min(10, num_depths)]  # Top-10
        
        # 解释方差比
        total_var = eigenvalues.sum().item()
        explained_var = (eigenvalues / total_var).cpu().numpy()
        results['pca_explained_variance'] = explained_var
        results['pca_cumulative_variance'] = np.cumsum(explained_var)
        
        # 7. 如果有 depth_scale，分析其与嵌入的关系
        if depth_scale is not None:
            depth_scale = depth_scale.detach().float()
            results['depth_scale'] = depth_scale.cpu().numpy()
            
            # depth_scale 与嵌入范数的相关性
            correlation = torch.corrcoef(torch.stack([
                norms, depth_scale
            ]))[0, 1].item()
            results['scale_norm_correlation'] = correlation
        
        return results
    
    def print_report(self, results: Dict[str, Any]) -> None:
        """打印分析报告。
        
        Args:
            results: analyze() 返回的结果字典
        """
        print("\n" + "=" * 60)
        print(" Depth Embedding Analysis Report")
        print("=" * 60)
        
        print(f"\n📊 Basic Info:")
        print(f"  Num Depths: {results['num_depths']}")
        print(f"  Embedding Dim: {results['embedding_dim']}")
        
        # 余弦相似度矩阵
        print("\n📊 Cosine Similarity Matrix:")
        print("-" * 40)
        cos_sim = results['cosine_similarity']
        num_depths = results['num_depths']
        
        # 表头
        header = "     " + "  ".join([f"d={d:>2}" for d in range(num_depths)])
        print(header)
        
        # 矩阵
        for i in range(num_depths):
            row = f"d={i:>2} " + "  ".join([f"{cos_sim[i, j]:>5.2f}" for j in range(num_depths)])
            print(row)
        
        # 指标
        print("\n📈 Analysis Metrics:")
        print("-" * 40)
        
        sep = results['separation_score']
        print(f"  Separation Score:     {sep:.4f}")
        print(f"    (Higher = better depth separation)")
        
        hier = results['hierarchy_score']
        print(f"\n  Hierarchy Score:      {hier:.4f}")
        print(f"    Adjacent sim mean:  {results['adjacent_similarity_mean']:.4f}")
        print(f"    Distant sim mean:   {results['distant_similarity_mean']:.4f}")
        status = "✅ (hierarchy preserved)" if hier > 0 else "⚠️ (no clear hierarchy)"
        print(f"    Interpretation:     {status}")
        
        print(f"\n  Embedding Norms:")
        norms = results['embedding_norms']
        print(f"    Mean: {results['norm_mean']:.4f}, Std: {results['norm_std']:.4f}")
        print(f"    Per-depth: {[f'{n:.2f}' for n in norms]}")
        
        print(f"\n  PCA Explained Variance (top-5):")
        pca_var = results['pca_explained_variance'][:5]
        cum_var = results['pca_cumulative_variance'][:5]
        for i, (v, c) in enumerate(zip(pca_var, cum_var)):
            bar = "█" * int(v * 40)
            print(f"    PC{i+1}: {v:.2%} (cum: {c:.2%}) {bar}")
        
        if 'scale_norm_correlation' in results:
            corr = results['scale_norm_correlation']
            print(f"\n  depth_scale vs norm correlation: {corr:.4f}")
        
        print("\n" + "=" * 60)
    
    def plot_similarity_matrix(
        self,
        results: Dict[str, Any],
        metric: str = 'cosine',
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """绘制相似度/距离矩阵热图。
        
        Args:
            results: analyze() 返回的结果字典
            metric: 'cosine' 或 'euclidean'
            title: 图标题
            save_path: 保存路径
        """
        if not HAS_MATPLOTLIB:
            print("Warning: matplotlib not installed")
            return
        
        if metric == 'cosine':
            matrix = results['cosine_similarity']
            cmap = 'RdYlBu_r'  # 红蓝色阶，高相似度为红
            label = 'Cosine Similarity'
            default_title = 'Depth Embedding Cosine Similarity'
        else:
            matrix = results['euclidean_distance']
            cmap = 'YlOrRd'  # 黄红色阶，大距离为红
            label = 'Euclidean Distance'
            default_title = 'Depth Embedding Euclidean Distance'
        
        num_depths = results['num_depths']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if HAS_SEABORN:
            sns.heatmap(
                matrix,
                annot=True,
                fmt='.2f',
                cmap=cmap,
                xticklabels=[f'd={d}' for d in range(num_depths)],
                yticklabels=[f'd={d}' for d in range(num_depths)],
                ax=ax,
                square=True,
            )
        else:
            im = ax.imshow(matrix, cmap=cmap, aspect='equal')
            ax.set_xticks(range(num_depths))
            ax.set_yticks(range(num_depths))
            ax.set_xticklabels([f'd={d}' for d in range(num_depths)])
            ax.set_yticklabels([f'd={d}' for d in range(num_depths)])
            plt.colorbar(im, ax=ax, label=label)
            
            for i in range(num_depths):
                for j in range(num_depths):
                    ax.text(j, i, f'{matrix[i, j]:.2f}',
                           ha='center', va='center', fontsize=9,
                           color='white' if metric == 'cosine' and matrix[i, j] > 0.5 else 'black')
        
        ax.set_xlabel('Depth')
        ax.set_ylabel('Depth')
        ax.set_title(title or default_title)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.show()
    
    def plot_embedding_2d(
        self,
        embeddings: torch.Tensor,
        title: str = 'Depth Embeddings (PCA)',
        save_path: Optional[str] = None,
    ) -> None:
        """将嵌入投影到 2D 并可视化。
        
        Args:
            embeddings: (D+1, dim) 深度嵌入
            title: 图标题
            save_path: 保存路径
        """
        if not HAS_MATPLOTLIB:
            print("Warning: matplotlib not installed")
            return
        
        embeddings = embeddings.detach().float()
        num_depths = embeddings.shape[0]
        
        # PCA 降维到 2D
        centered = embeddings - embeddings.mean(dim=0, keepdim=True)
        U, S, V = torch.linalg.svd(centered, full_matrices=False)
        coords_2d = (centered @ V[:2, :].T).cpu().numpy()
        
        # 绘制
        fig, ax = plt.subplots(figsize=(8, 6))
        
        colors = plt.cm.viridis(np.linspace(0, 1, num_depths))
        
        for d in range(num_depths):
            ax.scatter(coords_2d[d, 0], coords_2d[d, 1], 
                      c=[colors[d]], s=200, label=f'd={d}', 
                      edgecolors='black', linewidth=1.5, zorder=3)
            ax.annotate(f'd={d}', (coords_2d[d, 0], coords_2d[d, 1]),
                       textcoords='offset points', xytext=(8, 8),
                       fontsize=10, fontweight='bold')
        
        # 连接相邻深度
        for d in range(num_depths - 1):
            ax.plot([coords_2d[d, 0], coords_2d[d+1, 0]],
                   [coords_2d[d, 1], coords_2d[d+1, 1]],
                   'k--', alpha=0.3, linewidth=1, zorder=1)
        
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.show()


def demo():
    """演示分析器的使用方法。"""
    print("\n" + "=" * 60)
    print(" P6-4 Demo: Depth Embedding Analyzer")
    print("=" * 60)
    
    # 模拟深度嵌入
    torch.manual_seed(42)
    
    num_depths = 5
    dim = 64
    
    # 创建有结构的嵌入 (相邻深度更相似)
    base = torch.randn(dim)
    embeddings = torch.zeros(num_depths, dim)
    for d in range(num_depths):
        # 加入深度相关的偏移
        offset = torch.randn(dim) * 0.3
        embeddings[d] = base + d * 0.5 * torch.randn(dim) + offset
    
    # 创建 depth_scale (模拟 P6-1 的输出)
    depth_scale = torch.linspace(1.0, 1.2, num_depths)
    
    # 分析
    analyzer = DepthEmbeddingAnalyzer()
    results = analyzer.analyze(embeddings, depth_scale)
    
    # 打印报告
    analyzer.print_report(results)
    
    # 绘图
    if HAS_MATPLOTLIB:
        print("\n📊 Generating plots...")
        analyzer.plot_similarity_matrix(results, metric='cosine')
        analyzer.plot_embedding_2d(embeddings)
    else:
        print("\n⚠️ matplotlib not installed, skipping visualization")
    
    return results


if __name__ == "__main__":
    demo()
