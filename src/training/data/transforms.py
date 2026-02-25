# -*- coding: utf-8 -*-
"""Data Transforms - 数据增强和数据集定义模块

本模块包含数据集规格定义和 Mixup/CutMix 数据增强实现。

主要组件:
- DatasetSpec: 数据集规格数据类
- MixupCutmix: Mixup 和 CutMix 混合增强
- mixup_criterion: Mixup 损失计算
- DATASETS: 支持的数据集配置字典

数学形式化
==========
Mixup:
    x̃ = λ · x_i + (1 - λ) · x_j
    ỹ = λ · y_i + (1 - λ) · y_j
    其中 λ ~ Beta(α, α)

CutMix:
    在图像上随机裁剪一个矩形区域并交换
    ỹ = λ · y_i + (1 - λ) · y_j
    其中 λ = 1 - (裁剪区域面积) / (总面积)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from vit_pytorch.core.constants import EPS


# ============================================================================
# 数据集规格
# ============================================================================

@dataclass
class DatasetSpec:
    """数据集规格"""
    name: str
    num_classes: int
    image_size: int
    channels: int
    mean: Tuple[float, ...]
    std: Tuple[float, ...]


# ============================================================================
# 数据集配置
# ============================================================================

DATASETS = {
    "cifar10": DatasetSpec("CIFAR10", 10, 32, 3, (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": DatasetSpec("CIFAR100", 100, 32, 3, (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    "mnist": DatasetSpec("MNIST", 10, 28, 1, (0.1307,), (0.3081,)),
    "tiny-imagenet": DatasetSpec("TinyImageNet", 200, 64, 3, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    # CUB-200-2011: 细粒度鸟类分类数据集
    # 图像尺寸混合，动态获取每张图像的实际尺寸
    "cub200": DatasetSpec("CUB200", 200, None, 3, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}


# ============================================================================
# Mixup/CutMix 实现
# ============================================================================

class MixupCutmix:
    """Mixup 和 CutMix 数据增强

    参考:
    - Mixup: https://arxiv.org/abs/1710.09412
    - CutMix: https://arxiv.org/abs/1905.04899
    """

    def __init__(
        self,
        mixup_alpha: float = 0.8,
        cutmix_alpha: float = 1.0,
        prob: float = 0.5,
        num_classes: int = 10,
        label_smoothing: float = 0.0,
    ):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing

    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """应用 Mixup 或 CutMix

        Args:
            images: [B, C, H, W] 图像张量
            labels: [B] 标签张量

        Returns:
            mixed_images: 混合后的图像
            mixed_labels: 混合后的 one-hot 标签 [B, num_classes]
        """
        batch_size = images.size(0)
        device = images.device

        # 检查 label 范围，防止越界
        if not torch.all(labels >= 0):
            raise ValueError(f"Label 包含负值")
        if torch.all(labels >= self.num_classes):
            raise ValueError(f"Label 越界")

        # 转换为 one-hot 并应用 label smoothing
        labels_one_hot = F.one_hot(labels, self.num_classes).float()
        if self.label_smoothing > 0:
            labels_one_hot = labels_one_hot * (1 - self.label_smoothing) + self.label_smoothing / self.num_classes

        # 随机决定是否应用增强
        if random.random() > self.prob:
            return images, labels_one_hot

        # 随机选择 Mixup 或 CutMix
        use_cutmix = random.random() > 0.5 and self.cutmix_alpha > 0

        if use_cutmix:
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        else:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha) if self.mixup_alpha > 0 else 1.0

        # 随机打乱索引
        index = torch.randperm(batch_size, device=device)

        if use_cutmix:
            # CutMix: 随机裁剪区域
            _, _, H, W = images.shape
            cut_h = int(H * np.sqrt(1 - lam))
            cut_w = int(W * np.sqrt(1 - lam))

            cx = random.randint(0, W)
            cy = random.randint(0, H)

            x1 = max(0, cx - cut_w // 2)
            x2 = min(W, cx + cut_w // 2)
            y1 = max(0, cy - cut_h // 2)
            y2 = min(H, cy + cut_h // 2)

            mixed_images = images.clone()
            mixed_images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]

            # 重新计算 lambda 基于实际裁剪区域
            lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
        else:
            # Mixup: 线性混合
            lam_t = torch.tensor(lam, dtype=images.dtype, device=device)
            mixed_images = lam_t * images + (1 - lam_t) * images[index]

        # 混合标签 (始终使用 float32 以保持精度)
        mixed_labels = lam * labels_one_hot + (1 - lam) * labels_one_hot[index]

        return mixed_images, mixed_labels


def mixup_criterion(
    outputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """计算 Mixup/CutMix 的交叉熵损失

    Args:
        outputs: [B, C] 模型输出 logits
        targets: [B, C] one-hot 或 soft 标签

    Returns:
        损失标量
    """
    # 检查 logits 是否包含 NaN/Inf
    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
        raise ValueError(f"Logits 包含 NaN/Inf: nan={torch.isnan(outputs).sum()}, inf={torch.isinf(outputs).sum()}")

    # 数值稳定的 log_softmax
    log_probs = F.log_softmax(outputs, dim=1)

    # 确保 targets 归一化且非负
    targets = targets.clamp(min=0)
    targets = targets / (targets.sum(dim=1, keepdim=True) + EPS)

    loss = -(targets * log_probs).sum(dim=1).mean()

    # 检查 loss 是否为 NaN
    if torch.isnan(loss):
        raise ValueError("Loss 为 NaN，可能是 logits 过大或标签问题")

    return loss


__all__ = [
    "DatasetSpec",
    "DATASETS",
    "MixupCutmix",
    "mixup_criterion",
]
