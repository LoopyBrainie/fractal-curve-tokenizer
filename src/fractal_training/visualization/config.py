r"""
可视化配置与实验可视化器
========================

提供统一的可视化配置和自动化实验可视化功能。

数学形式化
==========

保存路径生成:
    $\text{path} = \text{base\_dir} / \text{prefix}\_\text{timestamp}.\text{format}$

时间戳格式:
    $\text{timestamp} = \text{YYYYMMDD\_HHMMSS}$

DPI 与分辨率:
    $\text{resolution} = \text{figsize} \times \text{dpi}$
    
    例: (12, 8) × 150 = 1800 × 1200 像素
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, TYPE_CHECKING
import warnings

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.figure import Figure


@dataclass
class VisualizationConfig:
    """可视化配置
    
    Attributes:
        save_dir: 保存目录，默认为 None (使用 experiments/{run_id}/visualizations/)
        format: 图像格式 (png, pdf, svg)
        dpi: 分辨率
        figsize_default: 默认图像尺寸
        style: matplotlib 样式
        palette: seaborn 调色板
        font_scale: 字体缩放比例
        auto_save: 是否自动保存
        show_plot: 是否显示图像
        tight_layout: 是否使用紧凑布局
        transparent: 背景是否透明
    """
    
    # 保存配置
    save_dir: Optional[str] = None
    format: str = "png"
    dpi: int = 150
    
    # 图像配置
    figsize_default: tuple = (12, 8)
    style: str = "whitegrid"  # seaborn style
    palette: str = "husl"     # seaborn palette
    font_scale: float = 1.1
    
    # 行为配置
    auto_save: bool = True
    show_plot: bool = False
    tight_layout: bool = True
    transparent: bool = False
    
    # 颜色配置
    cmap_diverging: str = "RdYlGn"   # 用于准确率热图
    cmap_sequential: str = "Blues"   # 用于混淆矩阵
    cmap_categorical: str = "husl"   # 用于类别
    
    def get_save_path(
        self,
        prefix: str,
        save_dir: Optional[str] = None,
        timestamp: bool = True,
    ) -> Path:
        """生成保存路径
        
        Args:
            prefix: 文件名前缀
            save_dir: 覆盖默认保存目录
            timestamp: 是否添加时间戳
            
        Returns:
            完整保存路径
        """
        base_dir = Path(save_dir or self.save_dir or ".")
        base_dir.mkdir(parents=True, exist_ok=True)
        
        if timestamp:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_{ts}.{self.format}"
        else:
            filename = f"{prefix}.{self.format}"
        
        return base_dir / filename


def auto_save_figure(
    fig: Figure,
    prefix: str,
    config: Optional[VisualizationConfig] = None,
    save_dir: Optional[str] = None,
    close_after: bool = True,
) -> Optional[Path]:
    """自动保存图像
    
    Args:
        fig: matplotlib Figure 对象
        prefix: 文件名前缀
        config: 可视化配置
        save_dir: 覆盖保存目录
        close_after: 保存后是否关闭图像
        
    Returns:
        保存路径 (如果保存成功)
    """
    config = config or VisualizationConfig()
    
    if not config.auto_save:
        return None
    
    save_path = config.get_save_path(prefix, save_dir=save_dir)
    
    try:
        if config.tight_layout:
            fig.tight_layout()
        
        fig.savefig(
            save_path,
            dpi=config.dpi,
            format=config.format,
            transparent=config.transparent,
            bbox_inches="tight",
        )
        
        print(f"[Visualization] Saved: {save_path}")
        
        if close_after:
            plt.close(fig)
        
        return save_path
        
    except Exception as e:
        warnings.warn(f"[Visualization] Failed to save figure: {e}")
        return None


def setup_style(config: Optional[VisualizationConfig] = None) -> None:
    """设置 seaborn/matplotlib 样式
    
    Args:
        config: 可视化配置
    """
    config = config or VisualizationConfig()
    
    try:
        import seaborn as sns
        
        sns.set_theme(
            style=config.style,
            palette=config.palette,
            font_scale=config.font_scale,
        )
    except ImportError:
        # 如果 seaborn 不可用，使用 matplotlib 默认样式
        plt.style.use("ggplot")


class ExperimentVisualizer:
    r"""实验可视化器
    
    提供统一的实验可视化接口，自动管理保存路径和配置。
    
    数学形式化:
        保存路径层次:
        $\text{experiments}/\text{run\_id}/\text{visualizations}/\text{type}/\text{name}.\text{format}$
    
    使用示例:
    ```python
    visualizer = ExperimentVisualizer(
        run_id="fractal_vit_20260105_120000",
        base_dir="experiments",
    )
    
    # 生成所有可视化
    visualizer.plot_classification_results(metrics_result)
    visualizer.plot_splitter_analysis(resource_result, token_counts)
    visualizer.plot_training_curves(history)
    
    # 或一次性生成
    visualizer.plot_all(metrics_result, resource_result, token_counts, history)
    ```
    """
    
    def __init__(
        self,
        run_id: Optional[str] = None,
        base_dir: str = "experiments",
        config: Optional[VisualizationConfig] = None,
    ):
        """
        Args:
            run_id: 实验 ID (用于生成保存路径)
            base_dir: 基础目录
            config: 可视化配置
        """
        self.run_id = run_id or f"run_{datetime.now():%Y%m%d_%H%M%S}"
        self.base_dir = Path(base_dir)
        self.config = config or VisualizationConfig()
        
        # 设置保存目录
        self.vis_dir = self.base_dir / self.run_id / "visualizations"
        self.config.save_dir = str(self.vis_dir)
        
        # 设置样式
        setup_style(self.config)
    
    def _get_subdir(self, category: str) -> Path:
        """获取子目录"""
        subdir = self.vis_dir / category
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir
    
    def plot_classification_results(
        self,
        per_class_accuracy: Dict[int, float],
        confusion_matrix: Optional[Any] = None,
        class_names: Optional[List[str]] = None,
        class_counts: Optional[Dict[int, int]] = None,
    ) -> List[Path]:
        """绘制分类结果可视化
        
        Args:
            per_class_accuracy: 每个类别的准确率
            confusion_matrix: 混淆矩阵 (numpy array)
            class_names: 类别名称列表
            class_counts: 每个类别的样本数
            
        Returns:
            保存的文件路径列表
        """
        from .class_metrics import (
            plot_per_class_accuracy,
            plot_confusion_matrix,
            plot_class_distribution,
            plot_head_tail_comparison,
        )
        
        saved_paths = []
        save_dir = str(self._get_subdir("classification"))
        
        # 1. Per-class accuracy heatmap
        fig = plot_per_class_accuracy(
            per_class_accuracy,
            class_names=class_names,
            config=self.config,
        )
        path = auto_save_figure(
            fig, "per_class_accuracy",
            config=self.config, save_dir=save_dir,
        )
        if path:
            saved_paths.append(path)
        
        # 2. Confusion matrix (如果提供)
        if confusion_matrix is not None:
            fig = plot_confusion_matrix(
                confusion_matrix,
                class_names=class_names,
                config=self.config,
            )
            path = auto_save_figure(
                fig, "confusion_matrix",
                config=self.config, save_dir=save_dir,
            )
            if path:
                saved_paths.append(path)
        
        # 3. Class distribution (如果提供样本数)
        if class_counts is not None:
            fig = plot_class_distribution(
                class_counts,
                per_class_accuracy,
                class_names=class_names,
                config=self.config,
            )
            path = auto_save_figure(
                fig, "class_distribution",
                config=self.config, save_dir=save_dir,
            )
            if path:
                saved_paths.append(path)
            
            # 4. Head/Tail comparison
            fig = plot_head_tail_comparison(
                per_class_accuracy,
                class_counts,
                config=self.config,
            )
            path = auto_save_figure(
                fig, "head_tail_comparison",
                config=self.config, save_dir=save_dir,
            )
            if path:
                saved_paths.append(path)
        
        return saved_paths
    
    def plot_splitter_analysis(
        self,
        token_counts: List[int],
        depth_counts: Optional[Dict[int, int]] = None,
        health_timeline: Optional[List[Dict[str, float]]] = None,
        n_min: Optional[int] = None,
        n_max: Optional[int] = None,
    ) -> List[Path]:
        """绘制分割器分析可视化
        
        Args:
            token_counts: 每个样本的 token 数量
            depth_counts: 每个深度的 token 数量
            health_timeline: 健康状态时间线
            n_min: Elastic Budget 最小值
            n_max: Elastic Budget 最大值
            
        Returns:
            保存的文件路径列表
        """
        from .splitter_analysis import (
            plot_token_distribution,
            plot_depth_distribution,
            plot_splitter_health_timeline,
        )
        
        saved_paths = []
        save_dir = str(self._get_subdir("splitter"))
        
        # 1. Token distribution
        fig = plot_token_distribution(
            token_counts,
            n_min=n_min,
            n_max=n_max,
            config=self.config,
        )
        path = auto_save_figure(
            fig, "token_distribution",
            config=self.config, save_dir=save_dir,
        )
        if path:
            saved_paths.append(path)
        
        # 2. Depth distribution (如果提供)
        if depth_counts is not None:
            fig = plot_depth_distribution(
                depth_counts,
                config=self.config,
            )
            path = auto_save_figure(
                fig, "depth_distribution",
                config=self.config, save_dir=save_dir,
            )
            if path:
                saved_paths.append(path)
        
        # 3. Health timeline (如果提供)
        if health_timeline is not None:
            fig = plot_splitter_health_timeline(
                health_timeline,
                config=self.config,
            )
            path = auto_save_figure(
                fig, "health_timeline",
                config=self.config, save_dir=save_dir,
            )
            if path:
                saved_paths.append(path)
        
        return saved_paths
    
    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        flops_budget: Optional[float] = None,
        memory_budget: Optional[float] = None,
    ) -> List[Path]:
        """绘制训练曲线可视化
        
        Args:
            history: 训练历史 (key: 指标名, value: 值列表)
            flops_budget: FLOPS 预算
            memory_budget: 内存预算 (MB)
            
        Returns:
            保存的文件路径列表
        """
        from .resource_curves import (
            plot_flops_curve,
            plot_memory_curve,
            plot_training_curves,
        )
        
        saved_paths = []
        save_dir = str(self._get_subdir("training"))
        
        # 1. Training curves (loss, accuracy)
        fig = plot_training_curves(
            history,
            config=self.config,
        )
        path = auto_save_figure(
            fig, "training_curves",
            config=self.config, save_dir=save_dir,
        )
        if path:
            saved_paths.append(path)
        
        # 2. FLOPS curve (如果提供)
        if "flops" in history or "resource_flops" in history:
            flops_values = history.get("flops") or history.get("resource_flops", [])
            if flops_values:
                fig = plot_flops_curve(
                    flops_values,
                    budget=flops_budget,
                    config=self.config,
                )
                path = auto_save_figure(
                    fig, "flops_curve",
                    config=self.config, save_dir=save_dir,
                )
                if path:
                    saved_paths.append(path)
        
        # 3. Memory curve (如果提供)
        if "memory_mb" in history or "resource_memory" in history:
            memory_values = history.get("memory_mb") or history.get("resource_memory", [])
            if memory_values:
                fig = plot_memory_curve(
                    memory_values,
                    budget=memory_budget,
                    config=self.config,
                )
                path = auto_save_figure(
                    fig, "memory_curve",
                    config=self.config, save_dir=save_dir,
                )
                if path:
                    saved_paths.append(path)
        
        return saved_paths
    
    def plot_all(
        self,
        per_class_accuracy: Optional[Dict[int, float]] = None,
        confusion_matrix: Optional[Any] = None,
        token_counts: Optional[List[int]] = None,
        depth_counts: Optional[Dict[int, int]] = None,
        history: Optional[Dict[str, List[float]]] = None,
        class_names: Optional[List[str]] = None,
        class_counts: Optional[Dict[int, int]] = None,
        n_min: Optional[int] = None,
        n_max: Optional[int] = None,
        flops_budget: Optional[float] = None,
    ) -> List[Path]:
        """生成所有可视化
        
        Returns:
            保存的文件路径列表
        """
        all_paths = []
        
        # 分类结果
        if per_class_accuracy is not None:
            paths = self.plot_classification_results(
                per_class_accuracy,
                confusion_matrix=confusion_matrix,
                class_names=class_names,
                class_counts=class_counts,
            )
            all_paths.extend(paths)
        
        # 分割器分析
        if token_counts is not None:
            paths = self.plot_splitter_analysis(
                token_counts,
                depth_counts=depth_counts,
                n_min=n_min,
                n_max=n_max,
            )
            all_paths.extend(paths)
        
        # 训练曲线
        if history is not None:
            paths = self.plot_training_curves(
                history,
                flops_budget=flops_budget,
            )
            all_paths.extend(paths)
        
        print(f"\n[ExperimentVisualizer] Generated {len(all_paths)} visualizations")
        print(f"[ExperimentVisualizer] Saved to: {self.vis_dir}")
        
        return all_paths
