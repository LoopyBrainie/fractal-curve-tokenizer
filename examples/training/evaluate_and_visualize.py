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
    R = LayeredEvaluationReport = (L1, L2, L3, L4, L5, L6) 分层评估报告
    V = LayeredVisualizationReport = {Figure_i} 可视化图表集合

分层结构：
    L1: 分类性能 (Classification) - 准确率、ECE、混淆矩阵
    L2: Tokenizer 行为 (Tokenizer) - Token 数、深度分布、空间覆盖
    L3: 注意力机制 (Attention) - 注意力熵、Head 利用率
    L4: 特征表示 (Representation) - Fisher 判别比、类别可分性
    L5: 资源效率 (Efficiency) - 延迟、吞吐量、内存
    L6: 训练稳定性 (Stability) - 权重范数、数值稳定性

使用方式
=========

命令行::

    # 完整评估与可视化
    python evaluate_and_visualize.py --checkpoint path/to/best.pth
    
    # 指定数据集和输出目录
    python evaluate_and_visualize.py --checkpoint best.pth --dataset cifar10 --output ./report
    
    # 跳过某些层
    python evaluate_and_visualize.py --checkpoint best.pth --skip-layers L3 L5
    
    # 仅评估，不生成可视化
    python evaluate_and_visualize.py --checkpoint best.pth --eval-only
    
    # 仅可视化（使用已有的 JSON 报告）
    python evaluate_and_visualize.py --report evaluation_report.json --vis-only

编程接口::

    from evaluate_and_visualize import run_evaluation_and_visualization
    
    report, vis_report = run_evaluation_and_visualization(
        checkpoint_path="path/to/checkpoint.pth",
        dataset_name="cifar10",
        output_dir="./output",
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

# 导入分层评估系统
from layered_evaluator import LayeredEvaluator
from evaluation_layers import LayeredEvaluationReport

# 导入分层可视化系统
from visualization import (
    LayeredVisualizer,
    LayeredVisualizationReport,
    FigureConfig,
)


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
        数据集名称 (cifar10, cifar100, mnist, tiny-imagenet)
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
        
    Returns
    -------
    Tuple[LayeredEvaluationReport, LayeredVisualizationReport]
        (评估报告, 可视化报告)
    """
    start_time = time.time()
    checkpoint_path = Path(checkpoint_path)
    
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
        )
        
        eval_report = evaluator.run_full_evaluation(skip_layers=skip_layers)
        
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
    DATASET_CLASSES = {
        "cifar10": [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ],
        "cifar100": None,  # 100 类，太长不显示
        "mnist": [str(i) for i in range(10)],
        "tiny-imagenet": None,  # 200 类
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
        choices=["cifar10", "cifar100", "mnist", "tiny-imagenet"],
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
