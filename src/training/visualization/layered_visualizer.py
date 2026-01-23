#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分层可视化器 (Layered Visualizer)

数学形式化
============

本模块提供与 LayeredEvaluator 对齐的统一可视化入口：

    V: LayeredEvaluationReport → {Figure_i}_{i=1}^N

分层可视化映射：
    V₁: L1ClassificationMetrics → 分类结果图表
    V₂: L2TokenizerMetrics → Tokenizer 行为图表
    V₃: L3AttentionMetrics → 注意力机制图表
    V₄: L4RepresentationMetrics → 特征表示图表
    V₅: L5EfficiencyMetrics → 资源效率图表
    V₆: L6StabilityMetrics → 训练稳定性图表

使用示例：
    from evaluation_layers import LayeredEvaluator
    from visualization.layered_visualizer import LayeredVisualizer
    
    # 评估
    evaluator = LayeredEvaluator(model, device)
    report = evaluator.evaluate(test_loader)
    
    # 可视化
    visualizer = LayeredVisualizer()
    results = visualizer.visualize_all(report)
    results.save_all(output_dir)

Author: GitHub Copilot
Date: 2026-01-11
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from .base import FigureConfig, VisualizationResult

# 导入各层可视化器
from .layer_classification import L1ClassificationVisualizer
from .layer_tokenizer import L2TokenizerVisualizer
from .layer_attention import L3AttentionVisualizer
from .layer_representation import L4RepresentationVisualizer
from .layer_efficiency import L5EfficiencyVisualizer
from .layer_stability import L6StabilityVisualizer
from .layer_splitter import L7SplitterVisualizer
from .layer_gradient import L8GradientFlowVisualizer


@dataclass
class LayeredVisualizationReport:
    """分层可视化报告
    
    Attributes:
        results: 各层可视化结果字典
        total_figures: 总图表数
        layer_order: 层级顺序
    """
    results: Dict[str, VisualizationResult] = field(default_factory=dict)
    layer_order: List[str] = field(default_factory=lambda: [
        "L1_Classification",
        "L2_Tokenizer", 
        "L3_Attention",
        "L4_Representation",
        "L5_Efficiency",
        "L6_Stability",
        "L7_Splitter",
        "L8_GradientFlow",
    ])
    
    @property
    def total_figures(self) -> int:
        return sum(len(r) for r in self.results.values())
    
    def save_all(
        self,
        output_dir: Union[str, Path],
        config: Optional[FigureConfig] = None,
    ) -> Dict[str, List[Path]]:
        """保存所有图表
        
        Args:
            output_dir: 输出目录
            config: 图表配置
            
        Returns:
            {layer_name: [saved_paths]}
        """
        output_dir = Path(output_dir)
        config = config or FigureConfig()
        
        saved = {}
        for layer_name in self.layer_order:
            if layer_name in self.results:
                layer_dir = output_dir / layer_name.lower()
                paths = self.results[layer_name].save_all(layer_dir, config)
                saved[layer_name] = paths
                print(f"[{layer_name}] Saved {len(paths)} figures to {layer_dir}")
        
        print(f"\n[Total] {self.total_figures} figures saved to {output_dir}")
        return saved
    
    def get_figures(self, layer: Optional[str] = None) -> List[plt.Figure]:
        """获取图表列表
        
        Args:
            layer: 层名称，None 则返回所有层
            
        Returns:
            Figure 列表
        """
        if layer:
            return self.results.get(layer, VisualizationResult()).figures
        
        all_figs = []
        for name in self.layer_order:
            if name in self.results:
                all_figs.extend(self.results[name].figures)
        return all_figs


class LayeredVisualizer:
    """分层可视化器
    
    整合所有层级可视化器，提供统一的可视化入口。
    
    数学形式化：
        V = (V₁, V₂, V₃, V₄, V₅, V₆)
        V: LayeredEvaluationReport → LayeredVisualizationReport
    """
    
    def __init__(self, config: Optional[FigureConfig] = None):
        """初始化分层可视化器
        
        Args:
            config: 图表配置
        """
        self.config = config or FigureConfig()
        
        # 初始化各层可视化器
        self.L1 = L1ClassificationVisualizer(self.config)
        self.L2 = L2TokenizerVisualizer(self.config)
        self.L3 = L3AttentionVisualizer(self.config)
        self.L4 = L4RepresentationVisualizer(self.config)
        self.L5 = L5EfficiencyVisualizer(self.config)
        self.L6 = L6StabilityVisualizer(self.config)
        self.L7 = L7SplitterVisualizer(self.config)
        self.L8 = L8GradientFlowVisualizer(self.config)
        
        self._layer_map = {
            "L1_Classification": self.L1,
            "L2_Tokenizer": self.L2,
            "L3_Attention": self.L3,
            "L4_Representation": self.L4,
            "L5_Efficiency": self.L5,
            "L6_Stability": self.L6,
            "L7_Splitter": self.L7,
            "L8_GradientFlow": self.L8,
        }
    
    def visualize_all(
        self,
        report: Any,  # LayeredEvaluationReport
        class_names: Optional[List[str]] = None,
        # 可选的额外数据
        confusion_matrix: Optional[np.ndarray] = None,
        token_counts: Optional[List[int]] = None,
        features: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        training_history: Optional[List[Dict]] = None,
        **kwargs,
    ) -> LayeredVisualizationReport:
        """生成所有层的可视化
        
        Args:
            report: LayeredEvaluationReport 评估报告
            class_names: 类别名称列表
            confusion_matrix: 混淆矩阵（用于 L1）
            token_counts: Token 数量列表（用于 L2）
            features: 特征数组（用于 L4 t-SNE）
            labels: 标签数组（用于 L4 t-SNE）
            training_history: 训练历史（用于 L6）
            **kwargs: 传递给各层的额外参数
            
        Returns:
            LayeredVisualizationReport
        """
        vis_report = LayeredVisualizationReport()
        
        print("\n" + "=" * 60)
        print("Generating Layered Visualizations")
        print("=" * 60)
        
        # L1: 分类结果
        print("\n[L1] Classification Results...")
        try:
            vis_report.results["L1_Classification"] = self.L1.visualize(
                report.L1_classification,
                class_names=class_names,
                confusion_matrix=confusion_matrix,
                **kwargs.get("L1", {}),
            )
            print(f"    Generated {len(vis_report.results['L1_Classification'])} figures")
        except Exception as e:
            print(f"    [WARN] L1 visualization failed: {e}")
        
        # L2: Tokenizer 行为
        print("\n[L2] Tokenizer Behavior...")
        try:
            vis_report.results["L2_Tokenizer"] = self.L2.visualize(
                report.L2_tokenizer,
                class_names=class_names,
                token_counts=token_counts,
                **kwargs.get("L2", {}),
            )
            print(f"    Generated {len(vis_report.results['L2_Tokenizer'])} figures")
        except Exception as e:
            print(f"    [WARN] L2 visualization failed: {e}")
        
        # L3: 注意力机制
        print("\n[L3] Attention Mechanism...")
        try:
            vis_report.results["L3_Attention"] = self.L3.visualize(
                report.L3_attention,
                **kwargs.get("L3", {}),
            )
            print(f"    Generated {len(vis_report.results['L3_Attention'])} figures")
        except Exception as e:
            print(f"    [WARN] L3 visualization failed: {e}")
        
        # L4: 特征表示
        print("\n[L4] Feature Representation...")
        try:
            vis_report.results["L4_Representation"] = self.L4.visualize(
                report.L4_representation,
                class_names=class_names,
                features=features,
                labels=labels,
                **kwargs.get("L4", {}),
            )
            print(f"    Generated {len(vis_report.results['L4_Representation'])} figures")
        except Exception as e:
            print(f"    [WARN] L4 visualization failed: {e}")
        
        # L5: 资源效率
        print("\n[L5] Resource Efficiency...")
        try:
            vis_report.results["L5_Efficiency"] = self.L5.visualize(
                report.L5_efficiency,
                accuracy=report.L1_classification.top1_accuracy,
                **kwargs.get("L5", {}),
            )
            print(f"    Generated {len(vis_report.results['L5_Efficiency'])} figures")
        except Exception as e:
            print(f"    [WARN] L5 visualization failed: {e}")
        
        # L6: 训练稳定性
        print("\n[L6] Training Stability...")
        try:
            vis_report.results["L6_Stability"] = self.L6.visualize(
                report.L6_stability,
                training_history=training_history,
                **kwargs.get("L6", {}),
            )
            print(f"    Generated {len(vis_report.results['L6_Stability'])} figures")
        except Exception as e:
            print(f"    [WARN] L6 visualization failed: {e}")
        
        # L7: Splitter 可视化（如果有）
        if hasattr(report, 'L7_splitter') and report.L7_splitter is not None:
            print("\n[L7] Splitter Metrics...")
            try:
                vis_report.results["L7_Splitter"] = self.L7.visualize(
                    report.L7_splitter,
                    **kwargs.get("L7", {}),
                )
                print(f"    Generated {len(vis_report.results['L7_Splitter'])} figures")
            except Exception as e:
                print(f"    [WARN] L7 visualization failed: {e}")
        
        # L8: 梯度流可视化（如果有）
        if hasattr(report, 'L8_gradient_flow') and report.L8_gradient_flow is not None:
            print("\n[L8] Gradient Flow...")
            try:
                vis_report.results["L8_GradientFlow"] = self.L8.visualize(
                    report.L8_gradient_flow,
                    **kwargs.get("L8", {}),
                )
                print(f"    Generated {len(vis_report.results['L8_GradientFlow'])} figures")
            except Exception as e:
                print(f"    [WARN] L8 visualization failed: {e}")
        
        print("\n" + "=" * 60)
        print(f"Total: {vis_report.total_figures} figures generated")
        print("=" * 60 + "\n")
        
        return vis_report
    
    def visualize_layer(
        self,
        layer: str,
        metrics: Any,
        **kwargs,
    ) -> VisualizationResult:
        """生成单层可视化
        
        Args:
            layer: 层名称 ("L1_Classification", "L2_Tokenizer", etc.)
            metrics: 对应层的评估指标
            **kwargs: 额外参数
            
        Returns:
            VisualizationResult
        """
        if layer not in self._layer_map:
            raise ValueError(f"Unknown layer: {layer}. "
                           f"Available: {list(self._layer_map.keys())}")
        
        visualizer = self._layer_map[layer]
        return visualizer.visualize(metrics, **kwargs)
    
    def generate_html_report(
        self,
        vis_report: LayeredVisualizationReport,
        output_path: Union[str, Path],
        title: str = "Fractal ViT Evaluation Report",
    ) -> Path:
        """生成 HTML 报告
        
        Args:
            vis_report: 可视化报告
            output_path: 输出目录或 HTML 文件路径
            title: 报告标题
            
        Returns:
            HTML 文件路径
        """
        output_path = Path(output_path)
        
        # 判断是否为目录：已存在且是目录，或者没有 .html 后缀
        if output_path.exists() and output_path.is_dir():
            html_file = output_path / "evaluation_report.html"
        elif output_path.suffix.lower() == '.html':
            output_path.parent.mkdir(parents=True, exist_ok=True)
            html_file = output_path
        else:
            # 假设是目录路径
            output_path.mkdir(parents=True, exist_ok=True)
            html_file = output_path / "evaluation_report.html"
        
        # HTML 模板
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 40px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        .layer-section {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .figure-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
            gap: 20px;
        }}
        .figure-item {{
            background: #fafafa;
            border-radius: 4px;
            padding: 10px;
            text-align: center;
        }}
        .figure-item img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }}
        .figure-caption {{
            font-size: 14px;
            color: #666;
            margin-top: 8px;
        }}
        .summary {{
            background: #e8f6f3;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 30px;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="summary">
        <p><strong>Total Figures:</strong> {vis_report.total_figures}</p>
        <p><strong>Layers:</strong> {', '.join(vis_report.layer_order)}</p>
    </div>
"""
        
        for layer_name in vis_report.layer_order:
            if layer_name not in vis_report.results:
                continue
            
            result = vis_report.results[layer_name]
            html_content += f"""
    <div class="layer-section">
        <h2>{layer_name.replace('_', ' ')}</h2>
        <div class="figure-grid">
"""
            
            for fig, name, desc in result:
                # 保存图片并获取相对路径
                img_path = html_file.parent / "images" / f"{name}.png"
                img_path.parent.mkdir(parents=True, exist_ok=True)
                
                # 先尝试用 bbox_inches='tight'，检查结果尺寸
                try:
                    from io import BytesIO
                    import struct as _struct
                    buf = BytesIO()
                    fig.savefig(buf, dpi=150, bbox_inches='tight', format='png')
                    buf.seek(0)
                    buf.read(16)  # Skip signature + IHDR type
                    width = _struct.unpack('>I', buf.read(4))[0]
                    height = _struct.unpack('>I', buf.read(4))[0]
                    buf.close()
                    
                    # 检查尺寸是否合理
                    max_dimension = 4000
                    if width > max_dimension or height > max_dimension:
                        fig.savefig(img_path, dpi=150)
                    else:
                        fig.savefig(img_path, dpi=150, bbox_inches='tight')
                except Exception:
                    fig.savefig(img_path, dpi=150)
                
                plt.close(fig)
                
                rel_path = f"images/{name}.png"
                html_content += f"""
            <div class="figure-item">
                <img src="{rel_path}" alt="{name}">
                <div class="figure-caption">{desc}</div>
            </div>
"""
            
            html_content += """
        </div>
    </div>
"""
        
        html_content += """
</body>
</html>
"""
        
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"[HTML Report] Generated: {html_file}")
        return html_file
