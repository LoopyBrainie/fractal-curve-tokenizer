# ============================================================================
# 细粒度分类损失函数模块
# Fine-grained Classification Loss Functions
# ============================================================================
"""
细粒度分类（如 CUB-200-2011 鸟类分类）的专用损失函数。

数学形式化：
    L_total = L_ce + λ_center · L_center + λ_triplet · L_triplet + λ_entropy · L_entropy

其中：
    - L_ce: 交叉熵损失 (with label smoothing)
    - L_center: 类中心损失，增强类内紧凑性
    - L_triplet: 三元组损失，增强类间可分性
    - L_entropy: 注意力熵正则化，鼓励聚焦判别性区域
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from vit_pytorch.constants import EPS  # I112-3: 统一数值稳定性常量


@dataclass
class FinegrainedLossConfig:
    """细粒度损失配置
    
    Attributes:
        num_classes: 类别数
        feat_dim: 特征维度
        label_smoothing: 标签平滑系数
        use_center_loss: 是否使用 Center Loss
        center_loss_weight: Center Loss 权重 λ_center
        use_attention_entropy: 是否使用注意力熵正则化
        attention_entropy_weight: 注意力熵权重 λ_entropy
        use_focal: 是否使用 Focal Loss
        focal_gamma: Focal Loss γ 参数
    """
    num_classes: int = 200
    feat_dim: int = 256
    label_smoothing: float = 0.1
    use_center_loss: bool = True
    center_loss_weight: float = 0.01
    use_attention_entropy: bool = False
    attention_entropy_weight: float = 0.05
    use_focal: bool = False
    focal_gamma: float = 2.5  # I28-1: 从 2.0 提升到 2.5 (难/易样本比 243x)


class CenterLoss(nn.Module):
    """
    Center Loss: 增强类内紧凑性
    
    数学公式:
        L_center = (1/2m) Σ_i ||f_i - c_{y_i}||²
        
    其中:
        - f_i: 样本 i 的特征向量
        - c_y: 类别 y 的特征中心（可学习参数）
        - m: batch size
    
    梯度更新:
        ∂L/∂f_i = f_i - c_{y_i}
        c_y ← c_y - α · Σ_{i:y_i=y}(c_y - f_i) / (1 + n_y)
        
    复杂度:
        时间: O(B × D)
        空间: O(C × D) 用于存储类中心
        
    Reference:
        Wen et al., "A Discriminative Feature Learning Approach for Deep Face Recognition", ECCV 2016
    """
    
    def __init__(self, num_classes: int, feat_dim: int, center_momentum: float = 0.9):
        """
        Args:
            num_classes: 类别数 C
            feat_dim: 特征维度 D
            center_momentum: 中心更新动量（用于 EMA 更新）
        """
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.center_momentum = center_momentum
        
        # 可学习的类中心 [C, D]
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim) * 0.01)
        
        # 注册 buffer 用于统计每个类的样本数（用于归一化）
        self.register_buffer('class_counts', torch.zeros(num_classes))
    
    def forward(
        self, 
        features: torch.Tensor, 
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算 Center Loss
        
        Args:
            features: 特征向量 [B, D]
            labels: 类别标签 [B]
            
        Returns:
            loss: Center Loss 标量
            stats: 统计信息 {center_loss, avg_distance}
        """
        batch_size = features.size(0)
        
        # 获取对应类别的中心 [B, D]
        centers_batch = self.centers[labels]
        
        # 计算欧氏距离平方 ||f - c||²
        diff = features - centers_batch  # [B, D]
        distances = (diff ** 2).sum(dim=1)  # [B]
        
        # Center Loss = (1/2m) Σ ||f - c||²
        loss = distances.mean() * 0.5
        
        # 统计信息
        stats = {
            'center_loss': loss.item(),
            'avg_center_distance': distances.mean().sqrt().item(),
        }
        
        return loss, stats
    
    def update_centers(self, features: torch.Tensor, labels: torch.Tensor, lr: float = 0.5):
        """
        使用批次数据更新类中心（可选，用于非梯度更新模式）
        
        更新公式:
            Δc_j = Σ_{i:y_i=j}(c_j - f_i) / (1 + n_j)
            c_j ← c_j - α · Δc_j
        """
        with torch.no_grad():
            for j in range(self.num_classes):
                mask = labels == j
                if mask.sum() > 0:
                    class_features = features[mask]  # [n_j, D]
                    center_j = self.centers[j]  # [D]
                    
                    # 计算增量
                    delta = (center_j.unsqueeze(0) - class_features).mean(dim=0)
                    
                    # 更新中心
                    self.centers[j] = center_j - lr * delta


class AttentionEntropyLoss(nn.Module):
    """
    注意力熵正则化: 鼓励注意力聚焦于判别性区域
    
    数学公式:
        H(A) = -Σ_j a_j log(a_j)  (注意力分布的熵)
        L_entropy = -H(A)  (最小化熵 = 最大化聚焦)
        
    或者使用目标熵模式:
        L_entropy = |H(A) - H_target|
        
    低熵表示注意力集中在少数 token 上（判别性区域）
    高熵表示注意力均匀分布（缺乏聚焦）
    
    对于细粒度分类，我们希望低熵（聚焦喙、眼睛等判别性区域）
    """
    
    def __init__(
        self,
        mode: str = 'minimize',  # 'minimize' or 'target'
        target_entropy: Optional[float] = None,
        eps: float = EPS,  # I112-3: 使用统一 EPS (1e-6)
    ):
        """
        Args:
            mode: 'minimize' (最小化熵，最大化聚焦) 或 'target' (匹配目标熵)
            target_entropy: 目标熵值（仅在 mode='target' 时使用）
            eps: 数值稳定性常数
        """
        super().__init__()
        self.mode = mode
        self.target_entropy = target_entropy
        self.eps = eps
    
    def forward(
        self, 
        attention_weights: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算注意力熵损失
        
        Args:
            attention_weights: 注意力权重 [B, H, N, N] 或 [B, N]
                              假设已经 softmax 归一化
            
        Returns:
            loss: 熵损失标量
            stats: 统计信息
        """
        # 处理不同输入形状
        if attention_weights.dim() == 4:
            # [B, H, N, N] -> 取 CLS token 的注意力 [B, H, N]
            # 或者对所有 head 平均后取均值
            attn = attention_weights.mean(dim=1)  # [B, N, N]
            attn = attn[:, 0, :]  # CLS -> other tokens [B, N]
        elif attention_weights.dim() == 3:
            attn = attention_weights[:, 0, :]  # [B, N]
        else:
            attn = attention_weights  # [B, N]
        
        # 确保归一化
        attn = attn / (attn.sum(dim=-1, keepdim=True) + self.eps)
        
        # 计算熵 H = -Σ p log p (I145: 使用 clamp 保护 log 边界)
        entropy = -(attn * torch.clamp(attn, min=self.eps).log()).sum(dim=-1)  # [B]
        mean_entropy = entropy.mean()
        
        # 计算损失
        if self.mode == 'minimize':
            # 最小化熵 = 最大化聚焦
            loss = mean_entropy
        elif self.mode == 'target' and self.target_entropy is not None:
            # 匹配目标熵
            loss = (mean_entropy - self.target_entropy).abs()
        else:
            loss = mean_entropy
        
        stats = {
            'attention_entropy': mean_entropy.item(),
            'max_attention': attn.max(dim=-1)[0].mean().item(),
        }
        
        return loss, stats


class FinegrainedLoss(nn.Module):
    """
    细粒度分类组合损失
    
    数学公式:
        L_total = L_ce + λ_center · L_center + λ_entropy · L_entropy
        
    其中:
        - L_ce: 带 label smoothing 的交叉熵（可选 Focal Loss）
        - L_center: Center Loss（类内紧凑性）
        - L_entropy: 注意力熵正则化（判别性聚焦）
    """
    
    def __init__(self, config: FinegrainedLossConfig):
        super().__init__()
        self.config = config
        
        # 主分类损失
        if config.use_focal:
            from .focal_loss import FocalLoss
            self.ce_loss = FocalLoss(
                gamma=config.focal_gamma,
                label_smoothing=config.label_smoothing,
            )
        else:
            self.ce_loss = nn.CrossEntropyLoss(
                label_smoothing=config.label_smoothing
            )
        
        # Center Loss
        self.center_loss = None
        if config.use_center_loss:
            self.center_loss = CenterLoss(
                num_classes=config.num_classes,
                feat_dim=config.feat_dim,
            )
        
        # 注意力熵正则化
        self.attention_entropy_loss = None
        if config.use_attention_entropy:
            self.attention_entropy_loss = AttentionEntropyLoss(mode='minimize')
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        features: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算组合损失

        Args:
            logits: 分类 logits [B, C]
            labels: 标签 [B]
            features: 特征向量 [B, D] 或 List[Tensor]（每个样本一个tensor）
            attention_weights: 注意力权重（用于熵正则化）

        Returns:
            total_loss: 总损失
            stats: 各损失分量统计
        """
        stats = {}

        # 主分类损失
        ce_loss = self.ce_loss(logits, labels)
        stats['ce_loss'] = ce_loss.item()
        total_loss = ce_loss

        # Center Loss
        if self.center_loss is not None and features is not None:
            # I35: 处理 List[Tensor] 或 Tensor 输入
            if isinstance(features, list):
                features_tensor = torch.stack(features)  # List -> [B, D]
            else:
                features_tensor = features
            center_loss, center_stats = self.center_loss(features_tensor, labels)
            total_loss = total_loss + self.config.center_loss_weight * center_loss
            stats.update(center_stats)
        
        # 注意力熵正则化
        if self.attention_entropy_loss is not None and attention_weights is not None:
            entropy_loss, entropy_stats = self.attention_entropy_loss(attention_weights)
            total_loss = total_loss + self.config.attention_entropy_weight * entropy_loss
            stats.update(entropy_stats)
        
        stats['total_loss'] = total_loss.item()
        
        return total_loss, stats


def create_finegrained_loss(
    num_classes: int = 200,
    feat_dim: int = 256,
    label_smoothing: float = 0.1,
    use_center_loss: bool = True,
    center_loss_weight: float = 0.01,
    use_focal: bool = False,
    focal_gamma: float = 2.0,
) -> FinegrainedLoss:
    """
    创建细粒度分类损失函数的便捷工厂函数
    
    Args:
        num_classes: 类别数（CUB-200 为 200）
        feat_dim: 特征维度
        label_smoothing: 标签平滑
        use_center_loss: 是否使用 Center Loss
        center_loss_weight: Center Loss 权重
        use_focal: 是否使用 Focal Loss
        focal_gamma: Focal Loss gamma
        
    Returns:
        FinegrainedLoss 实例
    """
    config = FinegrainedLossConfig(
        num_classes=num_classes,
        feat_dim=feat_dim,
        label_smoothing=label_smoothing,
        use_center_loss=use_center_loss,
        center_loss_weight=center_loss_weight,
        use_focal=use_focal,
        focal_gamma=focal_gamma,
    )
    return FinegrainedLoss(config)
