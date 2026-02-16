#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FractalCurveViT 统一评估与可视化入口 (Unified Evaluation & Visualization Entry)
================================================================================

本脚本整合分层评估系统 (LayeredEvaluator) 与分层可视化系统 (LayeredVisualizer)，
提供一站式的模型评估与可视化报告生成。

数学形式化
============

设完整评估-可视化流程为：

    Φ: (M, D) → (R, V)

其中：
    M = 训练好的 FractalCurveViT 模型 (checkpoint)
    D = 评估数据集
    R = LayeredEvaluationReport = (L1, L2, L3, L4, L5, L6, L7, L8, L9) 分层评估报告
    V = LayeredVisualizationReport = {Figure_i} 可视化图表集合

分层结构：
    L1: 分类性能 (Classification) - 准确率、ECE、混淆矩阵
    L2: Tokenizer 行为 (Tokenizer) - Token 数、深度分布、空间覆盖
    L3: 注意力机制 (Attention) - 注意力熵、Head 利用率
    L4: 特征表示 (Representation) - Fisher 判别比、类别可分性
    L5: 资源效率 (Efficiency) - 延迟、吞吐量、内存
    L6: 训练稳定性 (Stability) - 权重范数、数值稳定性
    L7: 分割器专项 (Splitter) - 决策分析、配额、温度
    L8: 梯度流分析 (Gradient Flow) - 各组件梯度、STE 有效性
    L9: 细粒度分类 (Fine-grained) - CUB-200 专用评估（Center Loss、MCA、混淆分析）

使用方式
=========

命令行::

    # 完整评估与可视化（支持所有数据集）
    python evaluate_and_visualize.py --checkpoint path/to/best.pth

    # 指定数据集（CUB-200 自动启用细粒度评估）
    python evaluate_and_visualize.py --checkpoint best.pth --dataset cub200 --output ./report

    # Tiny-ImageNet 评估
    python evaluate_and_visualize.py --checkpoint best.pth --dataset tiny-imagenet

    # CUB-200 细粒度评估（自动触发 L9）
    python evaluate_and_visualize.py --checkpoint best.pth --dataset cub200

    # 跳过某些层
    python evaluate_and_visualize.py --checkpoint best.pth --skip-layers L3 L5

    # 仅评估，不生成可视化
    python evaluate_and_visualize.py --checkpoint best.pth --eval-only

    # 仅可视化（使用已有的 JSON 报告）
    python evaluate_and_visualize.py --report evaluation_report.json --vis-only

编程接口::

    from evaluate_and_visualize import run_evaluation_and_visualization

    # 通用评估
    report, vis_report = run_evaluation_and_visualization(
        checkpoint_path="path/to/checkpoint.pth",
        dataset_name="cifar10",
        output_dir="./output",
    )

    # CUB-200 细粒度评估（自动包含 L9 分析）
    report, vis_report = run_evaluation_and_visualization(
        checkpoint_path="path/to/checkpoint.pth",
        dataset_name="cub200",
        output_dir="./cub200_report",
    )

Author: GitHub Copilot
Date: 2026-01-11
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch

# 项目路径设置
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
EXAMPLES_PATH = PROJECT_ROOT / "examples"
TRAINING_PATH = EXAMPLES_PATH / "training"

for path in [PROJECT_ROOT, SRC_PATH, EXAMPLES_PATH, TRAINING_PATH]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# 导入分层评估系统 (使用绝对导入)
from training.layered_evaluator import LayeredEvaluator
from training.evaluation_layers import LayeredEvaluationReport

# 导入分层可视化系统
from training.visualization import (
    LayeredVisualizer,
    LayeredVisualizationReport,
    FigureConfig,
)

# 导入 CUB-200 鸟类类别名称
try:
    from training.trainer.cub200_trainer import CUB200_BIRD_CLASSES
except ImportError:
    CUB200_BIRD_CLASSES = None


# ============================================================================
# 实验配置加载函数
# ============================================================================

def load_config_from_experiment(checkpoint_path: str) -> Dict[str, Any]:
    """从实验目录自动加载 config.json

    查找路径优先级:
        1. checkpoints/best.pth → ../logs/config.json
        2. checkpoints/best.pth → ../../config.json

    数学形式化:
        config_path = argmax_{p ∈ candidates} exists(p)

    Returns:
        Dict 包含训练配置和 CUB200 配置
    """
    checkpoint = Path(checkpoint_path)

    # 候选路径
    candidates = [
        checkpoint.parent.parent / "logs" / "config.json",  # experiments/exp/logs/config.json
        checkpoint.parent.parent / "config.json",            # experiments/exp/config.json
    ]

    for config_path in candidates:
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                print(f"[INFO] Loaded config from: {config_path}")
                return config
            except Exception as e:
                print(f"[WARN] Failed to load config from {config_path}: {e}")

    print(f"[INFO] No config.json found for checkpoint: {checkpoint_path}")
    return {}


def merge_evaluation_config(
    base_config: Dict[str, Any],
    exp_config: Dict[str, Any],
    dataset_name: str,
) -> Dict[str, Any]:
    """合并评估配置（命令行参数优先）

    数学形式化:
        final_config = argmax_{source} priority(source)
        where priority(cli) > priority(config.json) > priority(default)

    Args:
        base_config: 命令行基础配置
        exp_config: 从 config.json 加载的配置
        dataset_name: 数据集名称

    Returns:
        合并后的配置
    """
    merged = base_config.copy()

    # 从实验配置中提取对应数据集的配置
    if dataset_name == 'cub200' and 'cub200_config' in exp_config:
        cub200_cfg = exp_config['cub200_config']
        # 仅覆盖未在命令行指定的参数
        for key, value in cub200_cfg.items():
            if key and key not in merged or (key in merged and merged[key] is None):
                merged[key] = value

    # 提取训练配置（通用参数）
    if 'training_config' in exp_config:
        training_cfg = exp_config['training_config']
        for key, value in training_cfg.items():
            if key:  # 确保 key 不是 None
                # 映射旧配置名称到新名称
                key_mapping = {
                    'drop_path': 'drop_path_rate',
                }
                mapped_key = key_mapping.get(key, key)
                if mapped_key not in merged or merged[mapped_key] is None:
                    merged[mapped_key] = value

    return merged


# ============================================================================
# 核心函数
# ============================================================================

def run_evaluation_and_visualization(
    checkpoint_path: str,
    dataset_name: str = "cifar10",
    output_dir: Optional[str] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    device: Optional[str] = None,
    skip_layers: Optional[List[str]] = None,
    eval_only: bool = False,
    vis_only: bool = False,
    report_path: Optional[str] = None,
    save_json: bool = True,
    save_figures: bool = True,
    generate_html: bool = True,
    figure_config: Optional[FigureConfig] = None,
    # 新增参数
    evaluate_train: bool = False,       # 是否评估训练集（CUB-200 专用）
    use_config_json: bool = True,       # 是否自动从 config.json 加载参数
    exp_config: Optional[Dict[str, Any]] = None,  # 预加载的实验配置
    # I140: Splitter 配置参数
    splitter_feature_dim: Optional[int] = None,
    splitter_pool_size: Optional[int] = None,
    splitter_hidden_dim: Optional[int] = None,
) -> Tuple[LayeredEvaluationReport, Optional[LayeredVisualizationReport]]:
    """执行完整的分层评估与可视化

    数学形式化：
        Φ: (checkpoint, dataset) → (EvalReport, VisReport)

    其中：
        EvalReport = (L1_metrics, L2_metrics, ..., L6_metrics)
        VisReport = {figures} ∪ {html_report}

    Parameters
    ----------
    checkpoint_path : str
        模型 checkpoint 文件路径
    dataset_name : str
        数据集名称 (cifar10, cifar100, mnist, tiny-imagenet, cub200)
    output_dir : str, optional
        输出目录，默认为 checkpoint 同级目录下的 layered_report/
    batch_size : int
        评估批次大小
    num_workers : int
        数据加载线程数
    device : str, optional
        设备 (cuda/cpu)，默认自动检测
    skip_layers : list of str, optional
        跳过的层，例如 ['L3', 'L5']
    eval_only : bool
        仅评估，不生成可视化
    vis_only : bool
        仅可视化（需提供 report_path）
    report_path : str, optional
        已有的评估报告 JSON 路径（用于 vis_only 模式）
    save_json : bool
        是否保存 JSON 评估报告
    save_figures : bool
        是否保存可视化图像
    generate_html : bool
        是否生成 HTML 报告
    figure_config : FigureConfig, optional
        可视化配置
    evaluate_train : bool
        是否评估训练集（CUB-200 专用，用于过拟合诊断）
    use_config_json : bool
        是否自动从 config.json 加载参数
    exp_config : dict, optional
        预加载的实验配置（与 use_config_json 配合使用）
    splitter_feature_dim : int, optional
        Splitter 特征维度（必须与检查点匹配，解决配置不兼容问题）
    splitter_pool_size : int, optional
        Splitter 池化大小（必须与检查点匹配）
    splitter_hidden_dim : int, optional
        Splitter 复杂度 MLP 隐藏层维度（必须与检查点匹配）

    Returns
    -------
    Tuple[LayeredEvaluationReport, LayeredVisualizationReport]
        (评估报告, 可视化报告)
    """
    start_time = time.time()
    checkpoint_path = Path(checkpoint_path)

    # 从 config.json 加载实验配置
    loaded_exp_config: Dict[str, Any] = {}
    if use_config_json and exp_config is None:
        loaded_exp_config = load_config_from_experiment(str(checkpoint_path))
        exp_config = loaded_exp_config
    elif exp_config is None:
        exp_config = {}

    # 设置输出目录
    # 默认在实验目录下创建 evaluation/ 文件夹
    # 例如: experiments/fractal_vit_xxx/checkpoints/best.pth -> experiments/fractal_vit_xxx/evaluation/
    if output_dir is None:
        exp_dir = checkpoint_path.parent.parent  # 从 checkpoints/ 上移到实验目录
        output_dir = exp_dir / "evaluation"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("FractalCurveViT Unified Evaluation & Visualization")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {dataset_name}")
    print(f"Evaluate Train: {evaluate_train}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    # ========================================================================
    # 阶段 1: 分层评估
    # ========================================================================
    eval_report: LayeredEvaluationReport

    if vis_only and report_path:
        # 仅可视化模式：从 JSON 加载已有报告
        print("\n[Phase 1] Loading existing evaluation report...")
        eval_report = _load_report_from_json(report_path)
        print(f"  Loaded: {report_path}")
    else:
        # 正常评估模式
        print("\n[Phase 1] Running Layered Evaluation...")

        evaluator = LayeredEvaluator(
            checkpoint_path=str(checkpoint_path),
            dataset_name=dataset_name,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            exp_config=exp_config,  # 传递实验配置
            # I140: Splitter 配置参数
            splitter_feature_dim=splitter_feature_dim,
            splitter_pool_size=splitter_pool_size,
            splitter_hidden_dim=splitter_hidden_dim,
        )

        eval_report = evaluator.run_full_evaluation(
            skip_layers=skip_layers,
            evaluate_train=evaluate_train,  # 传递训练集评估参数
        )

        # 保存 JSON 报告
        if save_json:
            json_path = output_dir / "evaluation_report.json"
            _save_report_to_json(eval_report, json_path)
            print(f"\n[OK] Evaluation report saved: {json_path}")
    
    # ========================================================================
    # 阶段 2: 分层可视化
    # ========================================================================
    vis_report: Optional[LayeredVisualizationReport] = None
    
    if not eval_only:
        print("\n[Phase 2] Generating Layered Visualizations...")
        
        config = figure_config or FigureConfig()
        visualizer = LayeredVisualizer(config=config)
        
        # 获取类别名称（如果可用）
        class_names = _get_class_names(dataset_name)
        
        # 生成可视化
        vis_report = visualizer.visualize_all(
            eval_report,
            class_names=class_names,
        )
        
        # 保存图像
        if save_figures:
            figures_dir = output_dir / "figures"
            figures_dir.mkdir(exist_ok=True)
            saved_paths = vis_report.save_all(figures_dir, config)
            print(f"\n[OK] Figures saved to: {figures_dir}")
        
        # 生成 HTML 报告
        if generate_html:
            html_path = visualizer.generate_html_report(vis_report, output_dir)
            print(f"[OK] HTML report saved: {html_path}")
    
    # ========================================================================
    # 完成
    # ========================================================================
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("Evaluation & Visualization Complete")
    print("=" * 70)
    print(f"Total time: {elapsed:.1f}s")
    print(f"Output directory: {output_dir}")
    
    # 打印摘要
    _print_summary(eval_report)
    
    plt.close('all')  # 清理 matplotlib
    
    return eval_report, vis_report


# ============================================================================
# 辅助函数
# ============================================================================

def _get_class_names(dataset_name: str) -> Optional[List[str]]:
    """获取数据集的类别名称"""
    # 导入 CUB-200 鸟类类别名称
    global CUB200_BIRD_CLASSES

    DATASET_CLASSES = {
        "cifar10": [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ],
        "cifar100": None,  # 100 类，太长不显示
        "mnist": [str(i) for i in range(10)],
        "tiny-imagenet": None,  # 200 类
        "cub200": CUB200_BIRD_CLASSES,  # 200 类鸟类（使用真实名称）
    }
    return DATASET_CLASSES.get(dataset_name.lower())


def _save_report_to_json(report: LayeredEvaluationReport, path: Path) -> None:
    """保存评估报告到 JSON"""
    data = report.to_dict()
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def _load_report_from_json(path: str) -> LayeredEvaluationReport:
    """从 JSON 加载评估报告"""
    from evaluation_layers import (
        L1ClassificationMetrics,
        L2TokenizerMetrics,
        L3AttentionMetrics,
        L4RepresentationMetrics,
        L5EfficiencyMetrics,
        L6StabilityMetrics,
    )
    from layered_evaluator import L9FinegrainedMetrics

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    report = LayeredEvaluationReport()

    # 元信息
    meta = data.get('meta', {})
    report.checkpoint_path = meta.get('checkpoint_path', '')
    report.dataset_name = meta.get('dataset_name', '')
    report.num_samples = meta.get('num_samples', 0)
    report.num_classes = meta.get('num_classes', 0)
    report.device = meta.get('device', '')
    report.evaluation_time_sec = meta.get('evaluation_time_sec', 0.0)

    # L1
    if 'L1_classification' in data:
        l1_data = data['L1_classification']
        report.L1_classification = L1ClassificationMetrics(
            top1_accuracy=l1_data.get('top1_accuracy', 0.0),
            top5_accuracy=l1_data.get('top5_accuracy', 0.0),
            mean_class_accuracy=l1_data.get('mean_class_accuracy', 0.0),
            per_class_accuracy=l1_data.get('per_class_accuracy', {}),
            avg_loss=l1_data.get('avg_loss', 0.0),
            ece=l1_data.get('ece', 0.0),
            mce=l1_data.get('mce', 0.0),
            top_confused_pairs=[tuple(x) for x in l1_data.get('top_confused_pairs', [])],
            hardest_classes=[tuple(x) for x in l1_data.get('hardest_classes', [])],
        )

    # L2
    if 'L2_tokenizer' in data:
        l2_data = data['L2_tokenizer']
        report.L2_tokenizer = L2TokenizerMetrics(
            avg_tokens=l2_data.get('avg_tokens', 0.0),
            min_tokens=l2_data.get('min_tokens', 0),
            max_tokens=l2_data.get('max_tokens', 0),
            std_tokens=l2_data.get('std_tokens', 0.0),
            depth_distribution=l2_data.get('depth_distribution', {}),
            depth_entropy=l2_data.get('depth_entropy', 0.0),
            spatial_coverage_ratio=l2_data.get('spatial_coverage_ratio', 0.0),
            content_token_correlation=l2_data.get('content_token_correlation', 0.0),
            per_class_avg_tokens=l2_data.get('per_class_avg_tokens', {}),
        )

    # L3
    if 'L3_attention' in data:
        l3_data = data['L3_attention']
        report.L3_attention = L3AttentionMetrics(
            per_layer_entropy=l3_data.get('per_layer_entropy', []),
            avg_entropy=l3_data.get('avg_entropy', 0.0),
            head_utilization=l3_data.get('head_utilization', []),
            dead_head_ratio=l3_data.get('dead_head_ratio', 0.0),
            cls_attention_coverage=l3_data.get('cls_attention_coverage', 0.0),
            cls_attention_entropy=l3_data.get('cls_attention_entropy', 0.0),
        )

    # L4
    if 'L4_representation' in data:
        l4_data = data['L4_representation']
        report.L4_representation = L4RepresentationMetrics(
            fisher_discriminant_ratio=l4_data.get('fisher_discriminant_ratio', 0.0),
            feature_mean_norm=l4_data.get('feature_mean_norm', 0.0),
            feature_std=l4_data.get('feature_std', 0.0),
            per_class_separability=l4_data.get('per_class_separability', {}),
            avg_separability=l4_data.get('avg_separability', 0.0),
        )

    # L5
    if 'L5_efficiency' in data:
        l5_data = data['L5_efficiency']
        report.L5_efficiency = L5EfficiencyMetrics(
            avg_latency_ms=l5_data.get('avg_latency_ms', 0.0),
            tokenizer_latency_ms=l5_data.get('tokenizer_latency_ms', 0.0),
            transformer_latency_ms=l5_data.get('transformer_latency_ms', 0.0),
            head_latency_ms=l5_data.get('head_latency_ms', 0.0),
            throughput_samples_per_sec=l5_data.get('throughput_samples_per_sec', 0.0),
            peak_memory_mb=l5_data.get('peak_memory_mb', 0.0),
            accuracy_per_gflops=l5_data.get('accuracy_per_gflops', 0.0),
            accuracy_per_token=l5_data.get('accuracy_per_token', 0.0),
            total_params=l5_data.get('total_params', 0),
            trainable_params=l5_data.get('trainable_params', 0),
        )

    # L6
    if 'L6_stability' in data:
        l6_data = data['L6_stability']
        report.L6_stability = L6StabilityMetrics(
            weight_norm_stats=l6_data.get('weight_norm_stats', {}),
            gradient_health_score=l6_data.get('gradient_health_score', 1.0),
            has_nan_weights=l6_data.get('has_nan_weights', False),
            has_inf_weights=l6_data.get('has_inf_weights', False),
            splitter_health_score=l6_data.get('splitter_health_score', 1.0),
        )

    # L9: CUB-200 Fine-grained
    if 'L9_finegrained' in data:
        l9_data = data['L9_finegrained']
        report.L9_finegrained = L9FinegrainedMetrics(
            top1_accuracy=l9_data.get('top1_accuracy', 0.0),
            top5_accuracy=l9_data.get('top5_accuracy', 0.0),
            mean_class_accuracy=l9_data.get('mean_class_accuracy', 0.0),
            loss=l9_data.get('loss', 0.0),
            center_loss=l9_data.get('center_loss'),
            avg_center_distance=l9_data.get('avg_center_distance'),
            intra_inter_ratio=l9_data.get('intra_inter_ratio'),
            confusion_entropy=l9_data.get('confusion_entropy'),
            missing_classes=l9_data.get('missing_classes', []),
            easy_classes=[tuple(x) for x in l9_data.get('easy_classes', [])],
            hard_classes=[tuple(x) for x in l9_data.get('hard_classes', [])],
            most_confused_pairs=[tuple(x) for x in l9_data.get('most_confused_pairs', [])],
        )

    return report


def _print_summary(report: LayeredEvaluationReport) -> None:
    """打印评估摘要"""
    print("\n" + "-" * 50)
    print("EVALUATION SUMMARY")
    print("-" * 50)

    # L1: 分类性能
    l1 = report.L1_classification
    print(f"\n[L1] Classification:")
    print(f"     Top-1 Accuracy: {l1.top1_accuracy:.2f}%")
    print(f"     Top-5 Accuracy: {l1.top5_accuracy:.2f}%")
    print(f"     ECE: {l1.ece:.2f}%")

    # L2: Tokenizer
    l2 = report.L2_tokenizer
    print(f"\n[L2] Tokenizer:")
    print(f"     Avg Tokens: {l2.avg_tokens:.1f}")
    print(f"     Token Range: [{l2.min_tokens}, {l2.max_tokens}]")
    print(f"     Depth Entropy: {l2.depth_entropy:.3f}")

    # L3: Attention
    l3 = report.L3_attention
    print(f"\n[L3] Attention:")
    print(f"     Avg Entropy: {l3.avg_entropy:.3f}")
    print(f"     Dead Head Ratio: {l3.dead_head_ratio:.2%}")

    # L4: Representation
    l4 = report.L4_representation
    print(f"\n[L4] Representation:")
    print(f"     Fisher Ratio: {l4.fisher_discriminant_ratio:.2f}")
    print(f"     Avg Separability: {l4.avg_separability:.3f}")

    # L5: Efficiency
    l5 = report.L5_efficiency
    print(f"\n[L5] Efficiency:")
    print(f"     Latency: {l5.avg_latency_ms:.2f} ms")
    print(f"     Throughput: {l5.throughput_samples_per_sec:.1f} samples/s")
    print(f"     Parameters: {l5.total_params:,}")

    # L6: Stability
    l6 = report.L6_stability
    health_status = "✓ Healthy" if not (l6.has_nan_weights or l6.has_inf_weights) else "✗ Issues"
    print(f"\n[L6] Stability:")
    print(f"     Health: {health_status}")
    print(f"     Splitter Score: {l6.splitter_health_score:.2f}")

    # L7: Splitter
    if hasattr(report, 'L7_splitter') and report.L7_splitter is not None:
        l7 = report.L7_splitter
        print(f"\n[L7] Splitter:")

        # 静态配置指标
        if l7.temperature > 0:
            print(f"     Temperature: {l7.temperature:.3f}")
        print(f"     Quota Entropy: {l7.quota_entropy:.3f}")
        if l7.quotas:
            quotas_str = ", ".join([f"d{d}:{v:.2f}" for d, v in sorted(l7.quotas.items())])
            print(f"     Target Quotas: {quotas_str}")

        # 行为指标 (实际运行结果)
        if l7.actual_num_tokens_mean > 0:
            print(f"     Actual Tokens: mean={l7.actual_num_tokens_mean:.1f}, "
                  f"std={l7.actual_num_tokens_std:.1f}, "
                  f"range=[{l7.actual_num_tokens_min}-{l7.actual_num_tokens_max}]")
        if l7.actual_depth_distribution:
            depth_str = ", ".join([f"d{d}:{p:.1%}" for d, p in sorted(l7.actual_depth_distribution.items())])
            print(f"     Actual Depth Dist: {depth_str}")
        if l7.actual_splitter_entropy_mean > 0:
            print(f"     Splitter Entropy: mean={l7.actual_splitter_entropy_mean:.3f}, "
                  f"std={l7.actual_splitter_entropy_std:.3f}")
        if l7.quota_usage_ratio:
            ratio_str = ", ".join([f"d{d}:{r:.2f}x" for d, r in sorted(l7.quota_usage_ratio.items())])
            print(f"     Quota Usage Ratio: {ratio_str}")

    # L8: Gradient Flow
    if hasattr(report, 'L8_gradient_flow') and report.L8_gradient_flow is not None:
        l8 = report.L8_gradient_flow
        print(f"\n[L8] Gradient Flow:")
        print(f"     Total Grad Norm: {l8.total_grad_norm:.4f}")
        print(f"     Vanishing: {len(l8.vanishing_gradients)}, Exploding: {len(l8.exploding_gradients)}")

    # L9: CUB-200 Fine-grained (仅当数据集为 cub200 时显示)
    if hasattr(report, 'L9_finegrained') and report.L9_finegrained is not None:
        l9 = report.L9_finegrained
        print(f"\n[L9] CUB-200 Fine-grained Classification:")
        print(f"     Top-1 Accuracy: {l9.top1_accuracy:.2f}%")
        print(f"     Mean Class Acc (MCA): {l9.mean_class_accuracy:.2f}%")
        if l9.center_loss is not None:
            print(f"     Center Loss: {l9.center_loss:.4f}")
        if l9.intra_inter_ratio is not None:
            print(f"     Intra/Inter Ratio: {l9.intra_inter_ratio:.2f}")
        if l9.missing_classes:
            print(f"     Missing Classes: {len(l9.missing_classes)}")
        if l9.hard_classes:
            print(f"     Hardest Class: {l9.hard_classes[0][0]} ({l9.hard_classes[0][1]:.1f}%)")

    print("-" * 50)


# ============================================================================
# 命令行接口
# ============================================================================

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="FractalCurveViT Unified Evaluation & Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full evaluation and visualization
  python evaluate_and_visualize.py --checkpoint experiments/run_1/best.pth
  
  # Specify dataset and output directory
  python evaluate_and_visualize.py --checkpoint best.pth --dataset tiny-imagenet --output-dir ./report
  
  # Skip certain layers
  python evaluate_and_visualize.py --checkpoint best.pth --skip-layers L3 L5
  
  # Evaluation only (no visualization)
  python evaluate_and_visualize.py --checkpoint best.pth --eval-only
  
  # Visualization only (from existing JSON report)
  python evaluate_and_visualize.py --report evaluation_report.json --vis-only
        """,
    )
    
    # 主要参数
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        help="Path to model checkpoint (.pth/.pt file)",
    )
    parser.add_argument(
        "--report", "-r",
        type=str,
        help="Path to existing evaluation report JSON (for --vis-only mode)",
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100", "mnist", "tiny-imagenet", "cub200"],
        help="Dataset name (default: cifar10)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        dest="output",
        help="Output directory (default: checkpoint_dir/evaluation/)",
    )
    
    # 运行模式
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run evaluation only, skip visualization",
    )
    parser.add_argument(
        "--vis-only",
        action="store_true",
        help="Run visualization only (requires --report)",
    )
    parser.add_argument(
        "--skip-layers",
        nargs="+",
        type=str,
        default=None,
        help="Layers to skip (e.g., L3 L5)",
    )
    
    # 评估参数
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=64,
        help="Batch size for evaluation (default: 64)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu, default: auto)",
    )
    
    # 输出选项
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Do not save JSON evaluation report",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Do not save visualization figures",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Do not generate HTML report",
    )
    
    # 可视化配置
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI (default: 150)",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[12, 10],
        help="Figure size (width height, default: 12 10)",
    )

    # CUB-200 专用参数
    parser.add_argument(
        "--evaluate-train",
        action="store_true",
        help="Also evaluate training set (CUB-200 only, for overfitting diagnosis)",
    )
    parser.add_argument(
        "--no-config-json",
        action="store_true",
        help="Do not auto-load config.json from experiment directory",
    )

    # I140: Splitter 配置参数（解决模型配置不兼容问题）
    parser.add_argument(
        "--splitter-feature-dim",
        type=int,
        default=None,
        dest="splitter_feature_dim",
        help="Splitter feature_dim (must match checkpoint, e.g., 48, 192, 256)",
    )
    parser.add_argument(
        "--splitter-pool-size",
        type=int,
        default=None,
        dest="splitter_pool_size",
        help="Splitter pool_size (must match checkpoint, e.g., 4, 8)",
    )
    parser.add_argument(
        "--splitter-hidden-dim",
        type=int,
        default=None,
        dest="splitter_hidden_dim",
        help="Splitter hidden_dim for complexity MLP (must match checkpoint)",
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 验证参数
    if args.vis_only:
        if not args.report:
            print("[ERROR] --vis-only requires --report")
            sys.exit(1)
        # vis-only 模式需要一个虚拟的 checkpoint 路径来确定输出目录
        if not args.checkpoint:
            args.checkpoint = Path(args.report).parent / "dummy.pth"
    else:
        if not args.checkpoint:
            print("[ERROR] --checkpoint is required")
            sys.exit(1)
        if not Path(args.checkpoint).exists():
            print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
            sys.exit(1)

    # 创建图表配置
    figure_config = FigureConfig(
        dpi=args.dpi,
        figsize=tuple(args.figsize),
    )

    # 从 config.json 加载实验配置（如果启用且可用）
    exp_config = None
    if not args.no_config_json and not args.vis_only:
        exp_config = load_config_from_experiment(str(args.checkpoint))

    # 运行评估与可视化
    try:
        run_evaluation_and_visualization(
            checkpoint_path=str(args.checkpoint),
            dataset_name=args.dataset,
            output_dir=args.output,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            skip_layers=args.skip_layers,
            eval_only=args.eval_only,
            vis_only=args.vis_only,
            report_path=args.report,
            save_json=not args.no_json,
            save_figures=not args.no_figures,
            generate_html=not args.no_html,
            figure_config=figure_config,
            # 新增参数
            evaluate_train=args.evaluate_train,
            use_config_json=not args.no_config_json,
            exp_config=exp_config,
            # I140: Splitter 配置参数
            splitter_feature_dim=args.splitter_feature_dim,
            splitter_pool_size=args.splitter_pool_size,
            splitter_hidden_dim=args.splitter_hidden_dim,
        )

        # 深度分布与任务难度关系分析 - 集成在评估流程中
        print("\n" + "=" * 60)
        print("Running Depth-Difficulty Analysis")
        print("=" * 60)

        # 导入分析函数
        from examples.analysis.depth_difficulty_analysis import run_evaluation as run_depth_diff_analysis

        # 运行分析，结果保存到与评估报告相同的目录
        run_depth_diff_analysis(
            checkpoint_path=str(args.checkpoint),
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            max_samples=1000,
            output_dir=str(args.output),
        )

    except KeyboardInterrupt:
        print("\n[!] Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
