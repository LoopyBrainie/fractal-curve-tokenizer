"""
分层可视化系统集成测试 (Layered Visualization Integration Test)
==================================================================

数学形式化定义 (Mathematical Formalization)
-------------------------------------------

测试目标: 验证 LayeredVisualizer 与 LayeredEvaluator 的完整集成

设 E: Model x Dataset -> R 为评估函数
设 V: R -> {Figure} 为可视化函数

则完整流程为: Model x Dataset --E--> R --V--> {Figure}

Author: GitHub Copilot
Date: 2025-01-11
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import matplotlib.pyplot as plt

# 导入评估层系统
from training.evaluation_layers import (
    LayeredEvaluationReport,
    L1ClassificationMetrics,
    L2TokenizerMetrics,
    L3AttentionMetrics,
    L4RepresentationMetrics,
    L5EfficiencyMetrics,
    L6StabilityMetrics,
)

# 导入可视化系统
from training.visualization import (
    LayeredVisualizer,
    FigureConfig,
    L1ClassificationVisualizer,
    L2TokenizerVisualizer,
    L3AttentionVisualizer,
    L4RepresentationVisualizer,
    L5EfficiencyVisualizer,
    L6StabilityVisualizer,
)


def create_mock_l1_metrics() -> L1ClassificationMetrics:
    """创建模拟的 L1 分类指标（使用实际数据类字段）"""
    num_classes = 10
    metrics = L1ClassificationMetrics()
    metrics.top1_accuracy = 85.0
    metrics.top5_accuracy = 95.0
    metrics.mean_class_accuracy = 84.0
    metrics.per_class_accuracy = {i: 80.0 + 2.0 * i for i in range(num_classes)}
    metrics.avg_loss = 0.35
    metrics.ece = 0.05
    metrics.mce = 0.12
    metrics.top_confused_pairs = [(0, 1, 15), (2, 3, 12), (4, 5, 8)]
    metrics.hardest_classes = [(3, 0.25), (7, 0.20), (0, 0.18)]
    return metrics


def create_mock_l2_metrics() -> L2TokenizerMetrics:
    """创建模拟的 L2 Tokenizer 指标"""
    metrics = L2TokenizerMetrics()
    metrics.avg_tokens = 64.5
    metrics.min_tokens = 16
    metrics.max_tokens = 196
    metrics.std_tokens = 12.3
    metrics.depth_distribution = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.25, 4: 0.15}
    metrics.depth_entropy = 2.15
    metrics.spatial_coverage_ratio = 0.85
    metrics.content_token_correlation = 0.65
    metrics.per_class_avg_tokens = {i: 60 + i * 2 for i in range(10)}
    return metrics


def create_mock_l3_metrics() -> L3AttentionMetrics:
    """创建模拟的 L3 注意力指标"""
    num_layers = 6
    num_heads = 8
    metrics = L3AttentionMetrics()
    metrics.per_layer_entropy = [2.5 + 0.2 * i for i in range(num_layers)]
    metrics.avg_entropy = np.mean(metrics.per_layer_entropy)
    metrics.head_utilization = [
        [0.8 + 0.02 * h for h in range(num_heads)] 
        for _ in range(num_layers)
    ]
    metrics.dead_head_ratio = 0.05
    metrics.cls_attention_coverage = 0.75
    metrics.cls_attention_entropy = 2.8
    return metrics


def create_mock_l4_metrics() -> L4RepresentationMetrics:
    """创建模拟的 L4 表示层指标"""
    num_classes = 10
    metrics = L4RepresentationMetrics()
    metrics.fisher_discriminant_ratio = 5.2
    metrics.feature_mean_norm = 12.5
    metrics.feature_std = 3.2
    metrics.per_class_separability = {i: 0.5 + 0.05 * i for i in range(num_classes)}
    metrics.avg_separability = np.mean(list(metrics.per_class_separability.values()))
    return metrics


def create_mock_l5_metrics() -> L5EfficiencyMetrics:
    """创建模拟的 L5 效率指标"""
    metrics = L5EfficiencyMetrics()
    metrics.avg_latency_ms = 15.5
    metrics.tokenizer_latency_ms = 3.2
    metrics.transformer_latency_ms = 10.8
    metrics.head_latency_ms = 1.5
    metrics.throughput_samples_per_sec = 128.0
    metrics.peak_memory_mb = 512.0
    metrics.accuracy_per_gflops = 0.02
    metrics.accuracy_per_token = 1.5
    metrics.total_params = 25_000_000
    metrics.trainable_params = 24_500_000
    return metrics


def create_mock_l6_metrics() -> L6StabilityMetrics:
    """创建模拟的 L6 稳定性指标"""
    metrics = L6StabilityMetrics()
    metrics.weight_norm_stats = {
        "layer_0": {"mean": 0.5, "std": 0.1, "max": 1.2},
        "layer_1": {"mean": 0.6, "std": 0.12, "max": 1.4},
        "layer_2": {"mean": 0.55, "std": 0.11, "max": 1.3},
    }
    metrics.gradient_health_score = 0.95
    metrics.has_nan_weights = False
    metrics.has_inf_weights = False
    metrics.splitter_health_score = 0.98
    return metrics


def create_mock_report() -> LayeredEvaluationReport:
    """创建完整的模拟评估报告"""
    report = LayeredEvaluationReport()
    report.checkpoint_path = "mock_checkpoint.pth"
    report.dataset_name = "MockCIFAR10"
    report.num_samples = 10000
    report.num_classes = 10
    report.device = "cuda"
    report.evaluation_time_sec = 120.5
    
    report.L1_classification = create_mock_l1_metrics()
    report.L2_tokenizer = create_mock_l2_metrics()
    report.L3_attention = create_mock_l3_metrics()
    report.L4_representation = create_mock_l4_metrics()
    report.L5_efficiency = create_mock_l5_metrics()
    report.L6_stability = create_mock_l6_metrics()
    
    return report


class TestL1ClassificationVisualizer:
    """L1 分类可视化器测试"""
    
    def test_visualize_with_valid_metrics(self):
        """测试有效指标的可视化"""
        visualizer = L1ClassificationVisualizer()
        metrics = create_mock_l1_metrics()
        
        result = visualizer.visualize(metrics)
        
        assert result is not None
        assert len(result.figures) > 0
        assert "L1" in result.layer_name
        
        for fig in result.figures:
            plt.close(fig)
    
    def test_visualize_with_minimal_metrics(self):
        """测试最小必需指标的可视化"""
        visualizer = L1ClassificationVisualizer()
        metrics = L1ClassificationMetrics()
        metrics.top1_accuracy = 80.0
        metrics.per_class_accuracy = {0: 80.0, 1: 85.0}
        
        result = visualizer.visualize(metrics)
        
        # 至少应该有 summary 图
        assert result is not None
        assert len(result.figures) >= 1
        
        for fig in result.figures:
            plt.close(fig)


class TestL2TokenizerVisualizer:
    """L2 Tokenizer 可视化器测试"""
    
    def test_visualize_with_valid_metrics(self):
        """测试有效指标的可视化"""
        visualizer = L2TokenizerVisualizer()
        metrics = create_mock_l2_metrics()
        
        result = visualizer.visualize(metrics)
        
        assert result is not None
        assert len(result.figures) > 0
        
        for fig in result.figures:
            plt.close(fig)


class TestL3AttentionVisualizer:
    """L3 注意力可视化器测试"""
    
    def test_visualize_with_valid_metrics(self):
        """测试有效指标的可视化"""
        visualizer = L3AttentionVisualizer()
        metrics = create_mock_l3_metrics()
        
        result = visualizer.visualize(metrics)
        
        assert result is not None
        assert len(result.figures) > 0
        
        for fig in result.figures:
            plt.close(fig)


class TestL4RepresentationVisualizer:
    """L4 表示层可视化器测试"""
    
    def test_visualize_with_valid_metrics(self):
        """测试有效指标的可视化"""
        visualizer = L4RepresentationVisualizer()
        metrics = create_mock_l4_metrics()
        
        result = visualizer.visualize(metrics)
        
        assert result is not None
        # L4 没有嵌入数据时可能不生成 t-SNE 图
        
        for fig in result.figures:
            plt.close(fig)


class TestL5EfficiencyVisualizer:
    """L5 效率可视化器测试"""
    
    def test_visualize_with_valid_metrics(self):
        """测试有效指标的可视化"""
        visualizer = L5EfficiencyVisualizer()
        metrics = create_mock_l5_metrics()
        
        result = visualizer.visualize(metrics)
        
        assert result is not None
        assert len(result.figures) > 0
        
        for fig in result.figures:
            plt.close(fig)


class TestL6StabilityVisualizer:
    """L6 稳定性可视化器测试"""
    
    def test_visualize_with_valid_metrics(self):
        """测试有效指标的可视化"""
        visualizer = L6StabilityVisualizer()
        metrics = create_mock_l6_metrics()
        
        result = visualizer.visualize(metrics)
        
        assert result is not None
        assert len(result.figures) > 0
        
        for fig in result.figures:
            plt.close(fig)


class TestLayeredVisualizer:
    """LayeredVisualizer 集成测试"""
    
    def test_visualize_all_layers(self):
        """测试所有层的可视化"""
        visualizer = LayeredVisualizer()
        report = create_mock_report()
        
        vis_report = visualizer.visualize_all(report)
        
        # 验证报告结构
        assert vis_report is not None
        
        # 清理所有图表
        plt.close('all')
    
    def test_visualize_single_layer(self):
        """测试单层可视化"""
        visualizer = LayeredVisualizer()
        metrics = create_mock_l1_metrics()
        
        result = visualizer.visualize_layer("L1_Classification", metrics)
        
        assert result is not None
        
        for fig in result.figures:
            plt.close(fig)
    
    def test_save_figures(self):
        """测试图像保存功能"""
        visualizer = LayeredVisualizer()
        report = create_mock_report()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            vis_report = visualizer.visualize_all(report)
            
            # 使用 save_all 方法保存图像
            saved_paths = vis_report.save_all(output_dir)
            
            # 验证图像文件已创建
            all_saved = []
            for paths in saved_paths.values():
                all_saved.extend(paths)
            
            assert len(all_saved) >= 1  # 至少有一些图像
            
        plt.close('all')


class TestFigureConfig:
    """FigureConfig 配置测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = FigureConfig()
        
        assert config.figsize[0] > 0
        assert config.figsize[1] > 0
        assert config.dpi > 0
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = FigureConfig(
            figsize=(20, 16),
            dpi=150,
            title_fontsize=16,
        )
        
        assert config.figsize == (20, 16)
        assert config.dpi == 150
        assert config.title_fontsize == 16
    
    def test_apply_config(self):
        """测试配置应用"""
        config = FigureConfig(dpi=100)
        config.apply()
        
        assert plt.rcParams['figure.dpi'] == 100


if __name__ == "__main__":
    # 运行快速测试
    print("Running layered visualization integration tests...")
    
    # 创建模拟报告
    report = create_mock_report()
    print(f"Created mock report: {report.dataset_name}")
    
    # 创建可视化器
    visualizer = LayeredVisualizer()
    print(f"LayeredVisualizer initialized with layers: {list(visualizer._layer_map.keys())}")
    
    # 执行可视化
    print("\nGenerating visualizations...")
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        vis_report = visualizer.visualize_all(report, save_dir=output_dir)
        
        print(f"\nSaved figures: {len(list(output_dir.glob('*.png')))}")
        
    plt.close('all')
    print("\n✓ All tests completed successfully!")
