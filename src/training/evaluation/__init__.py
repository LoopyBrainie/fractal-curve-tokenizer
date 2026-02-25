# -*- coding: utf-8 -*-
"""Training Evaluation - 分层评估模块

本模块实现 FractalCurveViT 的完整分层评估系统。

评估架构:
- L1: ClassificationEvaluator (准确率、ECE、混淆矩阵)
- L2: TokenizerEvaluator (Token 数、深度分布)
- L3: AttentionEvaluator (注意力熵、Head 利用率)
- L4: RepresentationEvaluator (Fisher 比、可分性)
- L5: EfficiencyEvaluator (延迟、吞吐量、内存)
- L6: StabilityEvaluator (权重健康、数值稳定性)
- L7: SplitterEvaluator (分割器健康)
- L8: GradientFlowEvaluator (梯度流)

主要组件:
- LayeredEvaluator: 分层评估器主类
- evaluation_layers.py: 各层评估指标定义
- evaluator.py: 评估器实现
- visualize.py: 评估结果可视化
"""

from .evaluation_layers import (
    L1ClassificationMetrics,
    L2TokenizerMetrics,
    L3AttentionMetrics,
    L4RepresentationMetrics,
    L5EfficiencyMetrics,
    L6StabilityMetrics,
    L7SplitterMetrics,
    L8GradientFlowMetrics,
    LayeredEvaluationReport,
    ClassificationEvaluator,
    TokenizerEvaluator,
    AttentionEvaluator,
    RepresentationEvaluator,
    EfficiencyEvaluator,
    StabilityEvaluator,
    SplitterEvaluator,
    GradientFlowEvaluator,
)

from .layered_evaluator import LayeredEvaluator, create_evaluator

__all__ = [
    # Evaluation layers
    "L1ClassificationMetrics",
    "L2TokenizerMetrics",
    "L3AttentionMetrics",
    "L4RepresentationMetrics",
    "L5EfficiencyMetrics",
    "L6StabilityMetrics",
    "L7SplitterMetrics",
    "L8GradientFlowMetrics",
    "LayeredEvaluationReport",
    # Evaluators
    "ClassificationEvaluator",
    "TokenizerEvaluator",
    "AttentionEvaluator",
    "RepresentationEvaluator",
    "EfficiencyEvaluator",
    "StabilityEvaluator",
    "SplitterEvaluator",
    "GradientFlowEvaluator",
    "LayeredEvaluator",
    "create_evaluator",
]
