r"""
分割器分析可视化模块
====================

提供 Learnable Splitter 的可视化分析工具。

数学形式化
==========

**Token Distribution**:
    经验分布: $\hat{P}(N=n) = \frac{\text{count}(N=n)}{B}$
    
    期望值: $\mathbb{E}[N] = \sum_n n \cdot \hat{P}(N=n)$
    
    健康指标: 分布应在 $[N_{min}, N_{max}]$ 范围内集中

**Depth Distribution**:
    深度占比: $p_d = N_d / \sum_{d'} N_{d'}$
    
    熵: $H = -\sum_d p_d \log p_d$
    
    最大熵 (均匀分布): $H_{max} = \log(D+1)$，其中 $D$ 是最大深度
    
    熵比例: $r = H / H_{max}$ (越高表示多尺度利用越均衡)

**Health Timeline**:
    健康评分: $\text{health}(t) = \min(1, s_{token}) \cdot \min(1, s_{entropy})$
    
    其中:
    - $s_{token} = N_{avg} / N_{min}$ (token 数量是否达到最小值)
    - $s_{entropy} = r$ (熵比例)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any, TYPE_CHECKING
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from .config import VisualizationConfig


def _ensure_seaborn():
    """确保 seaborn 可用"""
    try:
        import seaborn as sns
        return sns
    except ImportError:
        raise ImportError(
            "seaborn is required for visualization. "
            "Install with: pip install seaborn"
        )


def plot_token_distribution(
    token_counts: List[int],
    n_min: Optional[int] = None,
    n_max: Optional[int] = None,
    figsize: tuple = (12, 6),
    config: Optional['VisualizationConfig'] = None,
    bins: int = 50,
) -> Figure:
    r"""绘制 Token 数量分布直方图
    
    数学形式化:
        $\hat{P}(N=n) = \frac{\text{count}(N=n)}{B}$
        
        可视化 Elastic Budget 边界 $[N_{min}, N_{max}]$
    
    Args:
        token_counts: 每个样本的 token 数量列表
        n_min: Elastic Budget 最小值
        n_max: Elastic Budget 最大值
        figsize: 图像尺寸
        config: 可视化配置
        bins: 直方图 bin 数量
        
    Returns:
        matplotlib Figure 对象
    """
    sns = _ensure_seaborn()
    
    token_counts = np.array(token_counts)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 左图: 直方图
    ax1 = axes[0]
    
    # 自动调整 bins
    actual_bins = min(bins, len(np.unique(token_counts)))
    
    sns.histplot(token_counts, bins=actual_bins, kde=True, ax=ax1, color='steelblue')
    
    # 添加 Elastic Budget 边界
    if n_min is not None:
        ax1.axvline(x=n_min, color='red', linestyle='--', linewidth=2, 
                    label=f'N_min = {n_min}')
    if n_max is not None:
        ax1.axvline(x=n_max, color='orange', linestyle='--', linewidth=2, 
                    label=f'N_max = {n_max}')
    
    # 添加均值和中位数
    mean_tokens = np.mean(token_counts)
    median_tokens = np.median(token_counts)
    ax1.axvline(x=mean_tokens, color='green', linestyle='-', linewidth=2, 
                label=f'Mean = {mean_tokens:.1f}')
    ax1.axvline(x=median_tokens, color='purple', linestyle=':', linewidth=2, 
                label=f'Median = {median_tokens:.1f}')
    
    ax1.set_xlabel('Token Count per Image', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Token Count Distribution', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    
    # 右图: 箱线图 + 统计信息
    ax2 = axes[1]
    
    bp = ax2.boxplot(token_counts, orientation='vertical', patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    
    ax2.set_ylabel('Token Count', fontsize=12)
    ax2.set_title('Token Count Statistics', fontsize=14, fontweight='bold')
    
    # 添加统计信息文本
    stats_text = (
        f"Samples: {len(token_counts)}\n"
        f"Mean: {mean_tokens:.2f}\n"
        f"Median: {median_tokens:.2f}\n"
        f"Std: {np.std(token_counts):.2f}\n"
        f"Min: {np.min(token_counts)}\n"
        f"Max: {np.max(token_counts)}\n"
    )
    
    # 检查崩溃状态
    collapse_count = np.sum(token_counts <= 2)
    if collapse_count > 0:
        stats_text += f"\n⚠️ Collapse: {collapse_count} ({100*collapse_count/len(token_counts):.1f}%)"
    
    ax2.text(
        1.3, 0.5, stats_text,
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment='center',
        horizontalalignment='left',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
    )
    
    # 如果有边界，标注违规比例
    if n_min is not None or n_max is not None:
        under_count = np.sum(token_counts < n_min) if n_min else 0
        over_count = np.sum(token_counts > n_max) if n_max else 0
        total = len(token_counts)
        
        boundary_text = f"Under N_min: {under_count} ({100*under_count/total:.1f}%)\n"
        boundary_text += f"Over N_max: {over_count} ({100*over_count/total:.1f}%)"
        
        ax2.text(
            1.3, 0.1, boundary_text,
            transform=ax2.transAxes,
            fontsize=9,
            verticalalignment='bottom',
            horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
        )
    
    plt.tight_layout()
    
    return fig


def plot_depth_distribution(
    depth_counts: Dict[int, int],
    max_depth: Optional[int] = None,
    figsize: tuple = (10, 6),
    config: Optional['VisualizationConfig'] = None,
) -> Figure:
    r"""绘制深度分布柱状图
    
    数学形式化:
        深度占比: $p_d = N_d / \sum_{d'} N_{d'}$
        
        熵: $H = -\sum_d p_d \log p_d$
        
        理想情况: 熵高表示多尺度利用均衡
    
    Args:
        depth_counts: {depth: count} 字典
        max_depth: 最大深度 (用于填充缺失深度)
        figsize: 图像尺寸
        config: 可视化配置
        
    Returns:
        matplotlib Figure 对象
    """
    sns = _ensure_seaborn()
    
    # 确定最大深度
    if max_depth is None:
        max_depth = max(depth_counts.keys()) if depth_counts else 4
    
    # 填充缺失深度
    depths = list(range(max_depth + 1))
    counts = [depth_counts.get(d, 0) for d in depths]
    total = sum(counts)
    
    if total == 0:
        total = 1  # 避免除零
    
    # 计算比例
    ratios = [c / total for c in counts]
    
    # 计算熵
    entropy = -sum(r * math.log(r + 1e-10) for r in ratios if r > 0)
    max_entropy = math.log(max_depth + 1)
    entropy_ratio = entropy / max_entropy if max_entropy > 0 else 0
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 左图: 柱状图
    ax1 = axes[0]
    
    colors = sns.color_palette("viridis", len(depths))
    bars = ax1.bar(depths, counts, color=colors, edgecolor='black', alpha=0.8)
    
    ax1.set_xlabel('Depth', fontsize=12)
    ax1.set_ylabel('Token Count', fontsize=12)
    ax1.set_title('Depth Distribution (Count)', fontsize=14, fontweight='bold')
    ax1.set_xticks(depths)
    
    # 在柱子上显示数值
    for bar, count in zip(bars, counts):
        if count > 0:
            ax1.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.01,
                str(count),
                ha='center', va='bottom', fontsize=9,
            )
    
    # 右图: 比例饼图
    ax2 = axes[1]
    
    # 过滤掉零值
    non_zero_depths = [d for d, r in zip(depths, ratios) if r > 0.01]
    non_zero_ratios = [r for r in ratios if r > 0.01]
    
    if non_zero_ratios:
        wedges, texts, autotexts = ax2.pie(
            non_zero_ratios,
            labels=[f'd={d}' for d in non_zero_depths],
            autopct='%1.1f%%',
            colors=sns.color_palette("viridis", len(non_zero_depths)),
            startangle=90,
        )
        ax2.set_title('Depth Distribution (Ratio)', fontsize=14, fontweight='bold')
    
    # 添加熵信息
    entropy_text = (
        f"Entropy: {entropy:.3f}\n"
        f"Max Entropy: {max_entropy:.3f}\n"
        f"Entropy Ratio: {entropy_ratio:.1%}\n"
        f"\n"
        f"Total Tokens: {total}"
    )
    
    # 熵评估
    if entropy_ratio < 0.3:
        entropy_text += "\n\n⚠️ Low entropy: Single-scale dominated"
    elif entropy_ratio > 0.7:
        entropy_text += "\n\n✅ High entropy: Multi-scale balanced"
    
    fig.text(
        0.5, -0.05, entropy_text,
        transform=fig.transFigure,
        fontsize=10,
        horizontalalignment='center',
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
    )
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    
    return fig


def plot_splitter_health_timeline(
    health_timeline: List[Dict[str, float]],
    figsize: tuple = (14, 8),
    config: Optional['VisualizationConfig'] = None,
) -> Figure:
    r"""绘制分割器健康状态时间线
    
    数学形式化:
        健康评分: $\text{health}(t) \in [0, 1]$
        
        崩溃检测: $\text{collapse}(t) = \mathbb{1}[\text{avg\_tokens} < 2]$
    
    Args:
        health_timeline: 健康状态列表，每个元素为 dict
            {health_score, avg_tokens, entropy, temperature, ...}
        figsize: 图像尺寸
        config: 可视化配置
        
    Returns:
        matplotlib Figure 对象
    """
    sns = _ensure_seaborn()
    
    if not health_timeline:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No health data available', ha='center', va='center')
        return fig
    
    epochs = list(range(len(health_timeline)))
    
    # 提取指标
    health_scores = [h.get('health_score', 0) for h in health_timeline]
    avg_tokens = [h.get('avg_tokens', 0) for h in health_timeline]
    entropies = [h.get('entropy', 0) for h in health_timeline]
    temperatures = [h.get('temperature', 1.0) for h in health_timeline]
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 左上: 健康评分
    ax1 = axes[0, 0]
    ax1.plot(epochs, health_scores, 'b-', linewidth=2, marker='o', markersize=3)
    ax1.fill_between(epochs, health_scores, alpha=0.3)
    ax1.axhline(y=0.5, color='orange', linestyle='--', label='Warning threshold')
    ax1.axhline(y=0.3, color='red', linestyle='--', label='Critical threshold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Health Score')
    ax1.set_title('Splitter Health Score', fontweight='bold')
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc='lower left')
    
    # 右上: 平均 Token 数
    ax2 = axes[0, 1]
    ax2.plot(epochs, avg_tokens, 'g-', linewidth=2, marker='s', markersize=3)
    ax2.axhline(y=2, color='red', linestyle='--', label='Collapse threshold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Avg Tokens')
    ax2.set_title('Average Tokens per Image', fontweight='bold')
    ax2.legend(loc='upper right')
    
    # 检测崩溃点
    collapse_epochs = [e for e, t in zip(epochs, avg_tokens) if t < 2]
    if collapse_epochs:
        ax2.scatter(collapse_epochs, [avg_tokens[e] for e in collapse_epochs], 
                    c='red', s=100, marker='x', zorder=5, label='Collapse')
    
    # 左下: 熵
    ax3 = axes[1, 0]
    ax3.plot(epochs, entropies, 'm-', linewidth=2, marker='^', markersize=3)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Depth Entropy')
    ax3.set_title('Depth Distribution Entropy', fontweight='bold')
    
    # 右下: 温度
    ax4 = axes[1, 1]
    ax4.plot(epochs, temperatures, 'c-', linewidth=2, marker='d', markersize=3)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Temperature')
    ax4.set_title('Gumbel-Softmax Temperature', fontweight='bold')
    
    # 总标题
    final_health = health_scores[-1] if health_scores else 0
    status = "✅ Healthy" if final_health > 0.5 else "⚠️ Warning" if final_health > 0.3 else "❌ Critical"
    fig.suptitle(
        f'Splitter Health Timeline - Final Status: {status} ({final_health:.2f})',
        fontsize=14, fontweight='bold', y=1.02,
    )
    
    plt.tight_layout()
    
    return fig


def plot_token_depth_heatmap(
    token_depth_matrix: np.ndarray,
    figsize: tuple = (12, 8),
    config: Optional['VisualizationConfig'] = None,
) -> Figure:
    """绘制 Token-Depth 关系热图
    
    Args:
        token_depth_matrix: [num_samples, max_depth+1] 每个样本各深度的 token 数
        figsize: 图像尺寸
        config: 可视化配置
        
    Returns:
        matplotlib Figure 对象
    """
    sns = _ensure_seaborn()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 如果样本太多，聚合显示
    if len(token_depth_matrix) > 100:
        # 按 token 总数分组
        total_tokens = token_depth_matrix.sum(axis=1)
        sorted_indices = np.argsort(total_tokens)
        token_depth_matrix = token_depth_matrix[sorted_indices]
        
        # 采样显示
        step = len(token_depth_matrix) // 100
        token_depth_matrix = token_depth_matrix[::max(1, step)]
    
    cmap = config.cmap_sequential if config else "YlOrRd"
    
    im = ax.imshow(token_depth_matrix.T, aspect='auto', cmap=cmap)
    
    ax.set_xlabel('Sample Index (sorted by total tokens)', fontsize=12)
    ax.set_ylabel('Depth', fontsize=12)
    ax.set_title('Token-Depth Distribution Heatmap', fontsize=14, fontweight='bold')
    
    # 添加颜色条
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Token Count', fontsize=10)

    return fig


def plot_complexity_depth_correlation(
    complexity_depth_pairs: List[Dict[str, float]],
    figsize: tuple = (14, 5),
    config: Optional['VisualizationConfig'] = None,
) -> Figure:
    r"""绘制 I133-2 验证实验结果：图像复杂度与深度分布的相关性

    数学形式化:
        假设 H1: 简单图像 → 深层 Token 占主导 (d 较小)
        假设 H2: 复杂图像 → 浅层 Token 更活跃 (d 较大)

        相关性分析:
        - corr(complexity, avg_depth): 复杂度与平均深度的皮尔逊相关系数
        - corr(complexity, shallow_ratio): 复杂度与浅层比例的相关性
        - corr(complexity, deep_ratio): 复杂度与深层比例的相关性

    Args:
        complexity_depth_pairs: [{'complexity': ..., 'avg_depth': ..., 'shallow_ratio': ..., 'deep_ratio': ...}, ...]
        figsize: 图像尺寸
        config: 可视化配置

    Returns:
        matplotlib Figure 对象
    """
    sns = _ensure_seaborn()

    # 提取数据
    complexities = np.array([p['complexity'] for p in complexity_depth_pairs])
    avg_depths = np.array([p['avg_depth'] for p in complexity_depth_pairs])
    shallow_ratios = np.array([p['shallow_ratio'] for p in complexity_depth_pairs])
    deep_ratios = np.array([p['deep_ratio'] for p in complexity_depth_pairs])

    # 计算相关性
    corr_depth = np.corrcoef(complexities, avg_depths)[0, 1] if np.std(complexities) > 1e-6 and np.std(avg_depths) > 1e-6 else 0.0
    corr_shallow = np.corrcoef(complexities, shallow_ratios)[0, 1] if np.std(complexities) > 1e-6 and np.std(shallow_ratios) > 1e-6 else 0.0
    corr_deep = np.corrcoef(complexities, deep_ratios)[0, 1] if np.std(complexities) > 1e-6 and np.std(deep_ratios) > 1e-6 else 0.0

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 子图1: 复杂度 vs 平均深度
    ax1 = axes[0]
    ax1.scatter(complexities, avg_depths, alpha=0.3, s=10)
    z = np.polyfit(complexities, avg_depths, 1)
    p = np.poly1d(z)
    x_line = np.linspace(complexities.min(), complexities.max(), 100)
    ax1.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'r = {corr_depth:.3f}')
    ax1.set_xlabel('Image Complexity (Edge Density)', fontsize=11)
    ax1.set_ylabel('Average Depth', fontsize=11)
    ax1.set_title('Complexity vs Avg Depth', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 子图2: 复杂度 vs 浅层比例
    ax2 = axes[1]
    ax2.scatter(complexities, shallow_ratios, alpha=0.3, s=10)
    z = np.polyfit(complexities, shallow_ratios, 1)
    p = np.poly1d(z)
    ax2.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'r = {corr_shallow:.3f}')
    ax2.set_xlabel('Image Complexity (Edge Density)', fontsize=11)
    ax2.set_ylabel('Shallow Token Ratio (d≤1)', fontsize=11)
    ax2.set_title('Complexity vs Shallow Ratio', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 子图3: 复杂度 vs 深层比例
    ax3 = axes[2]
    ax3.scatter(complexities, deep_ratios, alpha=0.3, s=10)
    z = np.polyfit(complexities, deep_ratios, 1)
    p = np.poly1d(z)
    ax3.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'r = {corr_deep:.3f}')
    ax3.set_xlabel('Image Complexity (Edge Density)', fontsize=11)
    ax3.set_ylabel('Deep Token Ratio (d≥3)', fontsize=11)
    ax3.set_title('Complexity vs Deep Ratio', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.suptitle('I133-2: Complexity-Depth Correlation Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig
