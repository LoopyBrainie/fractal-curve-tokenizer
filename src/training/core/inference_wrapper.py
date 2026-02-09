#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一推理接口模块

提供函数式接口，确保 Trainer's validate() 和独立 eval.py 使用相同的推理逻辑。

核心函数
========
- inference(): 单次推理，支持带标签计算 loss
- evaluate(): 完整评估流程，返回 EvalResult
- extract_logits(): 统一的 logits 提取工具

Usage
=====
>>> from training.core.inference_wrapper import inference, evaluate
>>>
>>> # Trainer 的 validate 步骤
>>> loss, logits, stats = inference(model, imgs, labels, device=device, use_amp=True)
>>>
>>> # 独立 eval 脚本
>>> result = evaluate(model, loader, device=device)
>>> print(f"Accuracy: {result.accuracy:.2f}%")

Author: Claude
Date: 2026-01-27
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


# ============================================================================
# 数据类定义
# ============================================================================

@dataclass
class InferenceStats:
    """推理统计信息（统一格式）

    保持与 model_fractal_vit.py 中 TrainingStats 的兼容性
    """
    logits: torch.Tensor                    # [B, num_classes]
    # I141: 添加 torch.Tensor 支持，与 TrainingStats 保持一致
    num_tokens: Union[int, List[int], torch.Tensor]  # Token 数量
    depth_used: int                         # 使用的深度
    depth_distribution: Dict[int, float]    # 深度分布
    features: torch.Tensor                  # [B, dim] 池化特征
    # I147: 添加 transformer_tokens 字段，与 TrainingStats 保持一致
    transformer_tokens: Optional[torch.Tensor] = None  # [B, N, dim] Transformer 输出

    # I145: 添加 shared_features 字段，与 TrainingStats 保持一致
    shared_features: Optional[torch.Tensor] = None  # [B, d_model, H/p, W/p]

    # I99-1: 添加 ema_stats 字段，与 TrainingStats 保持一致
    ema_stats: Optional[torch.Tensor] = None

    # 可选字段
    splitter_entropy: float = 0.0
    temperature: float = 1.0

    # 向后兼容字段 (I139 修复: 统一默认值与 TrainingStats)
    aux_infos: Optional[List[Dict[str, Any]]] = None
    split_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """评估结果"""
    accuracy: float           # Top-1 准确率 (%)
    avg_loss: float           # 平均 loss
    num_samples: int          # 样本数
    top5_accuracy: float = 0.0  # Top-5 准确率
    per_class_accuracy: Optional[Dict[int, float]] = None  # 逐类别准确率
    ece: float = 0.0          # Expected Calibration Error (%)


# ============================================================================
# 核心函数
# ============================================================================

def inference(
    model: nn.Module,
    imgs: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    *,
    device: torch.device,
    use_amp: bool = False,
) -> Union[Tuple[torch.Tensor, torch.Tensor, InferenceStats], InferenceStats]:
    """
    统一推理函数

    确保 Trainer 的 validate() 和独立 eval.py 使用完全相同的推理逻辑。

    Args:
        model: FractalCurveViT 模型
        imgs: 输入图像 [B, C, H, W]
        labels: 可选标签，用于计算 loss
        device: 计算设备
        use_amp: 是否使用混合精度

    Returns:
        有标签时: (loss, logits, stats)
        无标签时: InferenceStats 或 logits（保持原有接口）

    关键点
    ======
    1. Data Preprocessing 在函数内部完成 (imgs.to(device))
    2. 统一的 TrainingStats 处理（向后兼容 TrainingStats 和 Tensor）
    3. NaN/Inf 检查
    4. Post-processing 标准化
    """
    # Data Preprocessing
    imgs = imgs.to(device)

    # Forward
    with get_amp_context(device, use_amp):
        stats = model(imgs)

    # 统一处理 TrainingStats vs Tensor
    logits = extract_logits(stats)

    # NaN/Inf 检查
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        raise ValueError("Model output contains NaN/Inf")

    # 包装为 InferenceStats
    inference_stats = wrap_stats(stats, logits)

    # Post-processing
    if labels is not None:
        labels = labels.to(device)
        loss = F.cross_entropy(logits, labels)
        return loss, logits, inference_stats
    else:
        return inference_stats


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    use_amp: bool = False,
    num_classes: Optional[int] = None,
    return_per_class: bool = False,
) -> EvalResult:
    """
    完整评估流程

    使用 inference() 函数执行推理，汇总评估指标。

    Args:
        model: 模型
        loader: 数据加载器
        device: 计算设备
        use_amp: 是否使用混合精度
        num_classes: 类别数（可选，用于 Top-5 计算）
        return_per_class: 是否返回逐类别准确率

    Returns:
        EvalResult 包含所有评估指标
    """
    model.eval()

    all_preds: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    all_probs: List[torch.Tensor] = []
    total_loss = 0.0
    total_samples = 0

    for batch in tqdm(loader, desc="Evaluating"):
        imgs, batch_labels = batch
        loss, logits, _ = inference(model, imgs, batch_labels, device=device, use_amp=use_amp)

        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        all_preds.append(preds.cpu())
        all_labels.append(batch_labels.cpu())
        all_probs.append(probs.cpu())

        total_loss += loss.item() * len(batch_labels)
        total_samples += len(batch_labels)

    # 汇总结果
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)

    # 计算 Top-1 准确率
    correct = (all_preds == all_labels).sum().item()
    accuracy = correct / total_samples * 100

    result = EvalResult(
        accuracy=accuracy,
        avg_loss=total_loss / total_samples,
        num_samples=total_samples,
    )

    # Top-5 准确率
    if num_classes and num_classes >= 5:
        top5_preds = all_probs.topk(5, dim=1)[1]
        top5_correct = (top5_preds == all_labels.unsqueeze(1)).any(dim=1)
        result.top5_accuracy = top5_correct.float().mean().item() * 100

    # 逐类别准确率
    if return_per_class and num_classes:
        result.per_class_accuracy = compute_per_class_accuracy(
            all_preds, all_labels, num_classes
        )

    # ECE
    result.ece = compute_ece(all_probs, all_labels)

    return result


# ============================================================================
# 工具函数
# ============================================================================

def extract_logits(stats: Any) -> torch.Tensor:
    """
    统一提取 logits

    兼容 TrainingStats 对象和旧版 tuple/Tensor 返回格式。

    Args:
        stats: 模型输出（TrainingStats, tuple, 或 Tensor）

    Returns:
        logits: [B, num_classes]
    """
    # TrainingStats 对象 (preferred)
    if hasattr(stats, 'logits'):
        return stats.logits
    # 旧版返回格式 (tuple)
    elif isinstance(stats, tuple):
        return stats[0]
    # 直接是 Tensor
    elif isinstance(stats, torch.Tensor):
        return stats
    else:
        raise TypeError(f"Unknown model output type: {type(stats)}")


def wrap_stats(stats: Any, logits: torch.Tensor) -> InferenceStats:
    """
    包装模型输出为 InferenceStats

    保持向后兼容，从 TrainingStats 提取诊断信息。

    Args:
        stats: 模型输出
        logits: 已提取的 logits

    Returns:
        InferenceStats 对象
    """
    # 如果已经是 InferenceStats，直接返回
    if isinstance(stats, InferenceStats):
        return stats

    # 提取可选字段
    num_tokens = getattr(stats, 'num_tokens', len(logits))
    depth_distribution = getattr(stats, 'depth_distribution', {})

    # I147: 提取 transformer_tokens 字段
    transformer_tokens = getattr(stats, 'transformer_tokens', None)

    # I145: 提取 shared_features 字段
    shared_features = getattr(stats, 'shared_features', None)

    # I99-1: 提取 ema_stats 字段
    ema_stats = getattr(stats, 'ema_stats', None)

    # 构造 InferenceStats
    return InferenceStats(
        logits=logits,
        num_tokens=num_tokens,
        depth_used=getattr(stats, 'depth_used', 0),
        depth_distribution=depth_distribution if depth_distribution else {},
        features=getattr(stats, 'features', logits.new_zeros(logits.size(0), logits.size(1))),
        transformer_tokens=transformer_tokens,  # I147: 新增字段
        shared_features=shared_features,  # I145: 新增字段
        ema_stats=ema_stats,  # I99-1: 新增字段
        splitter_entropy=getattr(stats, 'splitter_entropy', 0.0),
        temperature=getattr(stats, 'temperature', 1.0),
        aux_infos=getattr(stats, 'aux_infos', None),
        split_info=getattr(stats, 'split_info', {}),
    )


def compute_per_class_accuracy(
    preds: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> Dict[int, float]:
    """计算逐类别准确率"""
    per_class_correct = {i: 0 for i in range(num_classes)}
    per_class_total = {i: 0 for i in range(num_classes)}

    for pred, label in zip(preds.numpy(), labels.numpy()):
        per_class_total[label] += 1
        if pred == label:
            per_class_correct[label] += 1

    return {
        c: per_class_correct[c] / per_class_total[c] * 100
        for c in range(num_classes)
        if per_class_total[c] > 0
    }


def compute_ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """计算 Expected Calibration Error

    ECE = Σ_{m=1}^M (|Bₘ|/N) |acc(Bₘ) - conf(Bₘ)|
    """
    confidences = probs.max(dim=1)[0]
    preds = probs.argmax(dim=1)
    accuracies = (preds == labels).float()

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].float().mean()
            ece += prop_in_bin * abs(avg_accuracy - avg_confidence)

    return ece.item() * 100  # 转为百分比


# ============================================================================
# 内部工具
# ============================================================================

class AmpContext:
    """AMP 上下文管理器 (I146: 更新使用新版 torch.amp.autocast)

    封装 torch.amp.autocast，提供统一的上下文接口。
    """

    def __init__(self, device: torch.device, use_amp: bool):
        self.device = device
        self.use_amp = use_amp
        self.ctx = None

    def __enter__(self):
        if self.use_amp and self.device.type == 'cuda':
            # I146: 使用新版 torch.amp.autocast 替代弃用的 torch.cuda.amp.autocast
            self.ctx = torch.amp.autocast('cuda')
            self.ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.ctx is not None:
            self.ctx.__exit__(exc_type, exc_val, exc_tb)


def get_amp_context(device: torch.device, use_amp: bool) -> AmpContext:
    """获取 AMP 上下文"""
    return AmpContext(device, use_amp)


# ============================================================================
# 便捷别名（保持向后兼容）
# ============================================================================

# 为了保持向后兼容，提供旧的函数名作为别名
InferenceWrapper = None  # 保留作为占位符，表示使用函数式接口


def create_inference_wrapper(
    model: nn.Module,
    device: torch.device,
    use_amp: bool = False,
) -> "InferenceWrapperHelper":
    """创建推理包装器（函数式接口的替代方式）

    保留此函数用于向后兼容，实际推荐直接使用 inference() 函数。
    """
    return InferenceWrapperHelper(model, device, use_amp)


class InferenceWrapperHelper:
    """推理包装器辅助类（保持向后兼容）

    内部使用，推荐直接调用 inference() 函数。
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        use_amp: bool = False,
    ):
        self.model = model
        self.device = device
        self.use_amp = use_amp
        self.model.eval()

    def __call__(
        self,
        imgs: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, InferenceStats], InferenceStats]:
        """执行推理"""
        return inference(
            self.model, imgs, labels,
            device=self.device,
            use_amp=self.use_amp,
        )

    def evaluate(
        self,
        loader: DataLoader,
        return_per_class: bool = False,
    ) -> EvalResult:
        """完整评估流程"""
        return evaluate(
            self.model, loader,
            device=self.device,
            use_amp=self.use_amp,
            return_per_class=return_per_class,
        )
