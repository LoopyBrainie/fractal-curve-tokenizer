# -*- coding: utf-8 -*-
"""
绘图工具函数
"""

import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# =============================================================================
# 样式设置
# =============================================================================

def set_style(style: str = 'seaborn-v0_8'):
    """
    设置绘图样式

    Args:
        style: 样式名称
    """
    try:
        plt.style.use(style)
    except Exception:
        print(f"Warning: Style '{style}' not found, using default")


def apply_publication_style():
    """
    应用发表级别的样式设置
    """
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.linewidth': 1.2,
        'axes.edgecolor': 'black',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 2,
        'lines.markersize': 6,
    })


# =============================================================================
# 颜色映射
# =============================================================================

def create_color_map(
    name: str = 'viridis',
    n_colors: int = 256
) -> plt.cm.colors.Colormap:
    """
    创建颜色映射

    Args:
        name: 颜色映射名称
        n_colors: 颜色数量

    Returns:
        颜色映射对象
    """
    return plt.cm.get_cmap(name, n_colors)


def get_depth_colors(n_depths: int):
    """
    获取深度对应的颜色

    Args:
        n_depths: 深度数量

    Returns:
        颜色列表
    """
    return plt.cm.viridis(np.linspace(0, 1, n_depths))


def get_attention_colors():
    """
    获取注意力热图颜色

    Returns:
        颜色映射
    """
    return LinearSegmentedColormap.from_list(
        'attention',
        ['#FFFFFF', '#FFFF00', '#FFA500', '#FF0000'],
        N=256
    )


# =============================================================================
# 图形保存
# =============================================================================

def save_figure(
    fig: plt.Figure,
    path: str,
    formats: tuple = ('png', 'pdf'),
    dpi: int = 300
) -> None:
    """
    保存图形到文件

    Args:
        fig: matplotlib 图形
        path: 保存路径
        formats: 输出格式列表
        dpi: 分辨率
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        save_path = path.with_suffix(f'.{fmt}')
        fig.savefig(save_path, dpi=dpi, format=fmt)
        print(f"Saved figure to {save_path}")


def save_html_report(
    html_content: str,
    path: str
) -> None:
    """
    保存 HTML 报告

    Args:
        html_content: HTML 内容
        path: 保存路径
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"Saved HTML report to {path}")


# =============================================================================
# 图形创建辅助函数
# =============================================================================

def create_figure(
    n_rows: int = 1,
    n_cols: int = 1,
    figsize: tuple = (10, 6),
    title: Optional[str] = None,
    style: Optional[str] = None
) -> tuple:
    """
    创建图形

    Args:
        n_rows: 行数
        n_cols: 列数
        figsize: 图像大小
        title: 标题
        style: 样式

    Returns:
        (fig, axes)
    """
    if style:
        set_style(style)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])

    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold')

    return fig, axes


def add_colorbar(
    ax: plt.Axes,
    mappable: plt.cm.ScalarMappable,
    label: str = '',
    shrink: float = 0.8
) -> 'matplotlib.colorbar.Colorbar':  # noqa: F821
    """
    添加颜色条

    Args:
        ax: 坐标轴
        mappable: 可视化对象
        label: 标签
        shrink: 缩放比例

    Returns:
        颜色条对象
    """
    return plt.colorbar(mappable, ax=ax, shrink=shrink, label=label)


def add_annotation(
    ax: plt.Axes,
    text: str,
    xy: tuple,
    xytext: tuple = (10, 10),
    arrowprops: dict = None
) -> None:
    """
    添加标注

    Args:
        ax: 坐标轴
        text: 文本
        xy: 目标位置
        xytext: 文本位置
        arrowprops: 箭头属性
    """
    if arrowprops is None:
        arrowprops = dict(arrowstyle='->', color='black')

    ax.annotate(text, xy=xy, xytext=xytext,
               textcoords='offset points',
               arrowprops=arrowprops,
               fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


# =============================================================================
# 统计可视化辅助
# =============================================================================

def plot_error_bars(
    ax: plt.Axes,
    x: list,
    means: list,
    stds: list,
    label: str = '',
    color: str = 'blue',
    marker: str = 'o'
) -> None:
    """
    绘制带误差条的折线图

    Args:
        ax: 坐标轴
        x: x 值
        means: 平均值
        stds: 标准差
        label: 标签
        color: 颜色
        marker: 标记
    """
    ax.errorbar(x, means, yerr=stds,
               label=label, color=color, marker=marker,
               capsize=5, capthick=2, linewidth=2)


def plot_confusion_matrix(
    ax: plt.Axes,
    cm: np.ndarray,
    labels: list = None,
    normalize: bool = True
) -> None:
    """
    绘制混淆矩阵

    Args:
        ax: 坐标轴
        cm: 混淆矩阵
        labels: 类别标签
        normalize: 是否归一化
    """
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)

    ax.imshow(cm, cmap='Blues')

    if labels is not None:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)

        # 添加数值
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f'{cm[i, j]:.2f}',
                              ha='center', va='center',
                              color='white' if cm[i, j] > 0.5 else 'black',
                              fontsize=8)


# =============================================================================
# 图形美化
# =============================================================================

def remove_axis_spines(ax: plt.Axes, which: str = 'all') -> None:
    """
    移除坐标轴边框

    Args:
        ax: 坐标轴
        which: 哪条边 ('all', 'top', 'bottom', 'left', 'right')
    """
    if which == 'all':
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    else:
        ax.spines[which].set_visible(False)


def add_border(
    ax: plt.Axes,
    color: str = 'black',
    linewidth: float = 2
) -> None:
    """
    添加边框

    Args:
        ax: 坐标轴
        color: 颜色
        linewidth: 线宽
    """
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(color)
        spine.set_linewidth(linewidth)


# 预定义样式
PRETTY_STYLES = {
    'presentation': {
        'font.size': 14,
        'axes.titlesize': 18,
        'figure.dpi': 100,
    },
    'paper': {
        'font.size': 10,
        'axes.titlesize': 12,
        'figure.dpi': 300,
        'lines.linewidth': 1.5,
    },
    'poster': {
        'font.size': 18,
        'axes.titlesize': 24,
        'figure.dpi': 150,
        'lines.linewidth': 3,
    }
}
