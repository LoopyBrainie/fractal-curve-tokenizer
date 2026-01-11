#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化基础模块 (Visualization Base Module)

数学形式化
============

可视化层定义：
    V_ℓ: R_ℓ → {Figure_i}_{i=1}^{n_ℓ}

其中：
    R_ℓ = 第 ℓ 层的评估结果 (来自 evaluation_layers.py)
    n_ℓ = 该层产生的图表数量

设计原则：
1. 单一职责：每个可视化层仅处理对应评估层的结果
2. 数据驱动：可视化内容与实验结果强相关，非架构展示
3. 复用优先：复用 evaluation_layers.py 的数据结构，避免重复计算

Author: GitHub Copilot
Date: 2026-01-11
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import numpy as np


# ============================================================================
# 配置数据类
# ============================================================================

@dataclass
class FigureConfig:
    """图表配置
    
    数学描述：
        配置空间 C = (dpi, size, style, format, ...)
        图表生成 G: R × C → Figure
    
    Attributes:
        dpi: 图像分辨率 (dots per inch)
        figsize: 图像尺寸 (width, height) in inches
        style: matplotlib 样式名称
        save_format: 保存格式 ("png", "pdf", "svg")
        font_family: 字体族
        title_fontsize: 标题字体大小
        label_fontsize: 标签字体大小
        tick_fontsize: 刻度字体大小
        legend_fontsize: 图例字体大小
        colormap: 默认色图
        auto_save: 是否自动保存
        show: 是否显示图表
    """
    dpi: int = 150
    figsize: Tuple[float, float] = (10, 8)
    style: str = "seaborn-v0_8-whitegrid"
    save_format: str = "png"
    font_family: str = "sans-serif"
    title_fontsize: int = 14
    label_fontsize: int = 11
    tick_fontsize: int = 10
    legend_fontsize: int = 10
    colormap: str = "viridis"
    auto_save: bool = True
    show: bool = False
    
    # 中文支持
    chinese_font: str = "SimHei"
    use_chinese: bool = True
    
    def apply(self) -> None:
        """应用配置到 matplotlib"""
        try:
            plt.style.use(self.style)
        except OSError:
            # 样式不可用时使用默认
            pass
        
        plt.rcParams['figure.dpi'] = self.dpi
        plt.rcParams['figure.figsize'] = self.figsize
        plt.rcParams['font.family'] = self.font_family
        plt.rcParams['font.size'] = self.label_fontsize
        plt.rcParams['axes.titlesize'] = self.title_fontsize
        plt.rcParams['axes.labelsize'] = self.label_fontsize
        plt.rcParams['xtick.labelsize'] = self.tick_fontsize
        plt.rcParams['ytick.labelsize'] = self.tick_fontsize
        plt.rcParams['legend.fontsize'] = self.legend_fontsize
        
        if self.use_chinese:
            plt.rcParams['font.sans-serif'] = [self.chinese_font, 'DejaVu Sans', 'Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False


@dataclass
class VisualizationResult:
    """可视化结果容器
    
    Attributes:
        figures: 生成的图表列表
        names: 每个图表的名称（用于保存）
        descriptions: 每个图表的描述
        layer_name: 所属评估层名称
    """
    figures: List[Figure] = field(default_factory=list)
    names: List[str] = field(default_factory=list)
    descriptions: List[str] = field(default_factory=list)
    layer_name: str = ""
    
    def __len__(self) -> int:
        return len(self.figures)
    
    def __iter__(self):
        return iter(zip(self.figures, self.names, self.descriptions))
    
    def save_all(self, output_dir: Path, config: FigureConfig) -> List[Path]:
        """保存所有图表
        
        Args:
            output_dir: 输出目录
            config: 图表配置
            
        Returns:
            保存的文件路径列表
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        for fig, name, _ in self:
            filename = f"{name}.{config.save_format}"
            filepath = output_dir / filename
            
            # 先尝试用 bbox_inches='tight'，检查结果尺寸
            # 如果尺寸异常（宽或高超过 4000 像素），则回退到普通保存
            try:
                from io import BytesIO
                buf = BytesIO()
                fig.savefig(buf, dpi=config.dpi, bbox_inches='tight', format='png')
                buf.seek(0)
                # 读取 PNG 头获取尺寸
                buf.read(16)  # Skip signature + IHDR type
                import struct
                width = struct.unpack('>I', buf.read(4))[0]
                height = struct.unpack('>I', buf.read(4))[0]
                buf.close()
                
                # 检查尺寸是否合理
                max_dimension = 4000  # 最大允许尺寸
                if width > max_dimension or height > max_dimension:
                    # 尺寸异常，不使用 bbox_inches='tight'
                    fig.savefig(filepath, dpi=config.dpi)
                else:
                    # 尺寸正常，使用 bbox_inches='tight'
                    fig.savefig(filepath, dpi=config.dpi, bbox_inches='tight')
            except Exception:
                # 出错时使用默认保存
                fig.savefig(filepath, dpi=config.dpi)
            
            saved_paths.append(filepath)
            plt.close(fig)
        
        return saved_paths
    
    def show_all(self) -> None:
        """显示所有图表"""
        for fig, name, desc in self:
            print(f"\n[{name}] {desc}")
            plt.figure(fig.number)
            plt.show()


# ============================================================================
# 可视化层基类
# ============================================================================

class VisualizationLayer(ABC):
    """可视化层抽象基类
    
    数学形式化：
        V_ℓ: R_ℓ → VisualizationResult
        
    其中 R_ℓ 是对应评估层的结果 dataclass。
    
    子类实现：
        - L1VisualizationLayer: 分类结果可视化
        - L2VisualizationLayer: Tokenizer 行为可视化
        - L3VisualizationLayer: 注意力机制可视化
        - L4VisualizationLayer: 特征表示可视化
        - L5VisualizationLayer: 资源效率可视化
        - L6VisualizationLayer: 训练稳定性可视化
    """
    
    # 子类必须定义
    LAYER_NAME: str = ""
    LAYER_DESCRIPTION: str = ""
    
    def __init__(self, config: Optional[FigureConfig] = None):
        """初始化可视化层
        
        Args:
            config: 图表配置，None 则使用默认
        """
        self.config = config or FigureConfig()
        self.config.apply()
    
    @abstractmethod
    def visualize(
        self,
        metrics: Any,
        class_names: Optional[List[str]] = None,
        **kwargs,
    ) -> VisualizationResult:
        """生成可视化图表
        
        Args:
            metrics: 对应层的评估指标 dataclass
            class_names: 类别名称列表（可选）
            **kwargs: 额外参数
            
        Returns:
            VisualizationResult 包含所有生成的图表
        """
        pass
    
    def _create_figure(
        self,
        nrows: int = 1,
        ncols: int = 1,
        figsize: Optional[Tuple[float, float]] = None,
        **kwargs,
    ) -> Tuple[Figure, Any]:
        """创建图表的辅助方法
        
        Args:
            nrows: 子图行数
            ncols: 子图列数
            figsize: 图表尺寸，None 则使用配置默认值
            **kwargs: 传递给 plt.subplots 的额外参数
            
        Returns:
            (Figure, Axes) 元组
        """
        if figsize is None:
            # 根据子图数量调整尺寸
            base_w, base_h = self.config.figsize
            figsize = (base_w * ncols / max(1, ncols - 0.5), 
                      base_h * nrows / max(1, nrows - 0.5))
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
        return fig, axes
    
    def _add_watermark(self, fig: Figure, text: str = "FractalCurveViT") -> None:
        """添加水印
        
        Args:
            fig: matplotlib Figure
            text: 水印文本
        """
        fig.text(
            0.99, 0.01, text,
            ha='right', va='bottom',
            fontsize=8, color='gray', alpha=0.5,
            transform=fig.transFigure
        )
    
    def _format_percentage(self, value: float, decimals: int = 1) -> str:
        """格式化百分比
        
        Args:
            value: 0-1 范围的值
            decimals: 小数位数
            
        Returns:
            格式化的百分比字符串
        """
        return f"{value * 100:.{decimals}f}%"
    
    def _get_color_palette(self, n: int, cmap_name: Optional[str] = None) -> np.ndarray:
        """获取颜色调色板
        
        Args:
            n: 需要的颜色数量
            cmap_name: 色图名称，None 则使用配置默认值
            
        Returns:
            形状为 (n, 4) 的 RGBA 颜色数组
        """
        cmap_name = cmap_name or self.config.colormap
        cmap = plt.get_cmap(cmap_name)
        return cmap(np.linspace(0, 0.9, n))


# ============================================================================
# 工具函数
# ============================================================================

def create_gridspec_layout(
    fig: Figure,
    layout: List[List[int]],
    wspace: float = 0.3,
    hspace: float = 0.3,
) -> Dict[int, plt.Axes]:
    """创建复杂网格布局
    
    Args:
        fig: matplotlib Figure
        layout: 二维布局数组，每个元素是子图 ID
                例如 [[1, 1, 2], [3, 3, 2]] 表示：
                - 子图1 占第一行左侧2/3
                - 子图2 占右侧全部
                - 子图3 占第二行左侧2/3
        wspace: 水平间距
        hspace: 垂直间距
        
    Returns:
        字典 {subplot_id: Axes}
    """
    layout = np.array(layout)
    nrows, ncols = layout.shape
    
    gs = GridSpec(nrows, ncols, figure=fig, wspace=wspace, hspace=hspace)
    
    axes_dict = {}
    processed = set()
    
    for subplot_id in np.unique(layout):
        if subplot_id in processed:
            continue
        
        # 找到该 subplot_id 占据的区域
        rows, cols = np.where(layout == subplot_id)
        row_start, row_end = rows.min(), rows.max() + 1
        col_start, col_end = cols.min(), cols.max() + 1
        
        ax = fig.add_subplot(gs[row_start:row_end, col_start:col_end])
        axes_dict[subplot_id] = ax
        processed.add(subplot_id)
    
    return axes_dict


def truncate_labels(labels: List[str], max_len: int = 15) -> List[str]:
    """截断过长的标签
    
    Args:
        labels: 原始标签列表
        max_len: 最大长度
        
    Returns:
        截断后的标签列表
    """
    truncated = []
    for label in labels:
        if len(label) > max_len:
            truncated.append(label[:max_len-2] + "..")
        else:
            truncated.append(label)
    return truncated


def compute_entropy(probs: np.ndarray, eps: float = 1e-10) -> float:
    """计算熵
    
    数学形式：
        H(p) = -Σᵢ pᵢ log(pᵢ)
        
    Args:
        probs: 概率分布数组
        eps: 防止 log(0) 的小值
        
    Returns:
        熵值
    """
    probs = np.asarray(probs)
    probs = probs / (probs.sum() + eps)  # 归一化
    probs = np.clip(probs, eps, 1.0)
    return -np.sum(probs * np.log(probs))


def format_large_number(n: int) -> str:
    """格式化大数字
    
    Args:
        n: 数字
        
    Returns:
        格式化字符串，如 "1.2M", "345K"
    """
    if n >= 1e9:
        return f"{n/1e9:.1f}B"
    elif n >= 1e6:
        return f"{n/1e6:.1f}M"
    elif n >= 1e3:
        return f"{n/1e3:.1f}K"
    else:
        return str(n)
