# -*- coding: utf-8 -*-
r"""CLS Token 注意力追踪器模块

用于追踪 CLS Token 对不同深度（Depth 0-4）Token 的注意力权重分布，
检测全局信息链路是否被切断，并提供 Global Context Anchor 机制。

使用方式:
    from vit_pytorch.core.cls_attention_tracker import CLSAttentionTracker

    tracker = CLSAttentionTracker(enabled=True, depth0_threshold=0.2)

    # 在每层注意力计算后调用
    result = tracker.compute_cls_depth_attention(
        attention_weights,  # [B, H, N, N] 或 [B, N, N]
        token_depths,       # [B, N] 各 token 的深度
    )

    print(f"Depth 0 注意力: {result.depth0_attention:.2%}")
    print(f"是否链路断开: {result.link_broken}")
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CLSDepthAttentionResult:
    """CLS 深度注意力分析结果"""
    # 各深度注意力占比
    attention_by_depth: Dict[int, float] = field(default_factory=dict)

    # 关键指标
    depth0_attention: float = 0.0  # CLS 对 Depth 0 的注意力
    depth1_attention: float = 0.0
    depth2_attention: float = 0.0
    depth3_attention: float = 0.0
    depth4_attention: float = 0.0

    # 阈值检测
    depth0_below_threshold: bool = False
    link_broken: bool = False  # 全局信息链路是否被切断

    # 统计信息
    num_layers_analyzed: int = 0
    avg_depth0_attention: float = 0.0  # 多层平均


class CLSAttentionTracker:
    """CLS Token 注意力追踪器

    追踪 CLS Token 对不同深度 Token 的注意力权重分布，
    用于检测全局信息聚合是否正常工作。

    数学定义:
    - CLS 对深度 d 的注意力: A_depth(d) = Σ_{j: depth(j)=d} A[0, j]
    - 链路断开阈值: A_depth(0) < threshold (默认 20%)
    """

    # 默认阈值
    DEFAULT_DEPTH0_THRESHOLD = 0.20  # 20%

    def __init__(
        self,
        enabled: bool = True,
        depth0_threshold: float = DEFAULT_DEPTH0_THRESHOLD,
        record_history: bool = True,
        max_depth: int = 4,
    ):
        """
        Args:
            enabled: 是否启用追踪
            depth0_threshold: Depth 0 注意力阈值，低于此值视为链路断开
            record_history: 是否记录历史
            max_depth: 最大深度
        """
        self.enabled = enabled
        self.depth0_threshold = depth0_threshold
        self.record_history = record_history
        self.max_depth = max_depth

        # 历史记录
        self.depth0_history: List[float] = []
        self.all_depths_history: List[Dict[int, float]] = []
        self.warning_history: List[bool] = []
        self.layer_counter = 0

    @torch._dynamo.disable  # 🌟 I-OOM FIX: 禁用 Dynamo 追踪，防止图断裂导致 backward 泄漏
    def compute_cls_depth_attention(
        self,
        attention_weights: torch.Tensor,
        token_depths: torch.Tensor,
    ) -> Optional[CLSDepthAttentionResult]:
        """
        计算 CLS 对各深度 Token 的注意力

        Args:
            attention_weights: 注意力权重 [B, H, N, N] 或 [B, N, N]
            token_depths: 各 token 的深度 [B, N]

        Returns:
            CLSDepthAttentionResult 或 None（如果禁用）
        """
        if not self.enabled:
            return None

        # 处理不同维度的注意力权重
        if attention_weights.dim() == 4:
            # [B, H, N, N] -> [B, N, N] 取平均
            attention_weights = attention_weights.mean(dim=1)
        elif attention_weights.dim() == 3:
            # [B, N, N] 保持不变
            pass
        else:
            return None

        B, N, _ = attention_weights.shape
        _, num_tokens = token_depths.shape

        # 限制 token 数量匹配
        N = min(N, num_tokens)

        # CLS 位于位置 0，提取 CLS 对所有 token 的注意力
        # attention_weights[:, 0, :] 是 CLS 对各 token 的注意力
        cls_attn = attention_weights[:, 0, :N]  # [B, N]

        # 计算各深度的注意力
        attention_by_depth: Dict[int, float] = {}
        total_attention = cls_attn.sum(dim=-1).mean().item()  # 归一化基数

        for depth in range(self.max_depth + 1):
            # 找出该深度的所有 token
            depth_mask = token_depths[:, :N] == depth  # [B, N]
            if depth_mask.any():
                # CLS 对该深度 token 的注意力总和
                depth_attn = (cls_attn * depth_mask.float()).sum(dim=-1).mean()
                attention_ratio = (depth_attn.item() / (total_attention + 1e-8))
                attention_by_depth[depth] = attention_ratio

        # 提取关键指标
        depth0_attention = attention_by_depth.get(0, 0.0)
        depth1_attention = attention_by_depth.get(1, 0.0)
        depth2_attention = attention_by_depth.get(2, 0.0)
        depth3_attention = attention_by_depth.get(3, 0.0)
        depth4_attention = attention_by_depth.get(4, 0.0)

        # 检测链路是否断开
        depth0_below_threshold = depth0_attention < self.depth0_threshold
        link_broken = depth0_below_threshold

        result = CLSDepthAttentionResult(
            attention_by_depth=attention_by_depth,
            depth0_attention=depth0_attention,
            depth1_attention=depth1_attention,
            depth2_attention=depth2_attention,
            depth3_attention=depth3_attention,
            depth4_attention=depth4_attention,
            depth0_below_threshold=depth0_below_threshold,
            link_broken=link_broken,
        )

        # 记录历史
        if self.record_history:
            self.depth0_history.append(depth0_attention)
            self.all_depths_history.append(attention_by_depth.copy())
            self.warning_history.append(link_broken)

        self.layer_counter += 1

        return result

    def compute_layer_average(self) -> Optional[CLSDepthAttentionResult]:
        """计算多层的平均结果"""
        if not self.enabled or not self.all_depths_history:
            return None

        # 计算平均 Depth 0 注意力
        avg_depth0 = sum(self.depth0_history) / len(self.depth0_history)

        # 计算各深度的平均注意力
        avg_by_depth: Dict[int, float] = {}
        for depth in range(self.max_depth + 1):
            depths_values = [
                h.get(depth, 0.0)
                for h in self.all_depths_history
            ]
            avg_by_depth[depth] = sum(depths_values) / len(depths_values) if depths_values else 0.0

        return CLSDepthAttentionResult(
            attention_by_depth=avg_by_depth,
            depth0_attention=avg_by_depth.get(0, 0.0),
            depth1_attention=avg_by_depth.get(1, 0.0),
            depth2_attention=avg_by_depth.get(2, 0.0),
            depth3_attention=avg_by_depth.get(3, 0.0),
            depth4_attention=avg_by_depth.get(4, 0.0),
            depth0_below_threshold=avg_depth0 < self.depth0_threshold,
            link_broken=avg_depth0 < self.depth0_threshold,
            num_layers_analyzed=len(self.depth0_history),
            avg_depth0_attention=avg_depth0,
        )

    def get_diagnostics_summary(self) -> Dict:
        """获取诊断摘要"""
        if not self.depth0_history:
            return {"status": "no_data"}

        avg_depth0 = sum(self.depth0_history) / len(self.depth0_history)
        warnings = sum(self.warning_history)

        return {
            "status": "analyzing",
            "layers_analyzed": len(self.depth0_history),
            "avg_depth0_attention": avg_depth0,
            "min_depth0_attention": min(self.depth0_history),
            "max_depth0_attention": max(self.depth0_history),
            "depth0_below_threshold_count": warnings,
            "global_link_intact": avg_depth0 >= self.depth0_threshold,
        }

    def reset(self):
        """重置历史记录"""
        self.depth0_history.clear()
        self.all_depths_history.clear()
        self.warning_history.clear()
        self.layer_counter = 0


class GlobalContextAnchor(nn.Module):
    """Global Context Anchor 机制

    强制 CLS 在前几层关注 Depth 0 的全局 Token，
    确保全局信息链路畅通。

    数学定义:
        Anchor_bias[i, j] = {
            α  if i == CLS 且 depth(j) == 0 且 layer < anchor_layers
            0  otherwise
        }

    其中 α = exp(log_anchor_strength) 是可学习参数。
    """

    def __init__(
        self,
        anchor_layers: int = 3,
        initial_strength: float = 2.0,
    ):
        """
        Args:
            anchor_layers: 前几层启用 anchor
            initial_strength: 初始 anchor 强度（对数尺度）
        """
        super().__init__()
        self.anchor_layers = anchor_layers

        # 可学习的 anchor 强度（对数尺度，确保为正）
        self.log_anchor_strength = nn.Parameter(torch.tensor(initial_strength))

    def forward(
        self,
        batch_size: int,
        num_tokens: int,
        token_depths: torch.Tensor,
        layer_idx: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """
        计算 Global Context Anchor 偏置

        Args:
            batch_size: 批次大小
            num_tokens: token 数量
            token_depths: 各 token 的深度 [B, N]
            layer_idx: 当前层索引
            device: 设备

        Returns:
            Anchor 偏置 [B, 1, 1, N] 或 None（如果当前层不需要 anchor）
        """
        if layer_idx >= self.anchor_layers:
            return None

        # 计算 anchor 强度
        anchor_strength = self.log_anchor_strength.exp()  # 确保为正

        # 创建 anchor 偏置
        # 只对 CLS (位置 0) 和 Depth 0 的 token 添加偏置
        depth0_mask = (token_depths == 0).float()  # [B, N]
        depth0_mask = depth0_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]

        anchor_bias = depth0_mask * anchor_strength

        return anchor_bias

    def get_current_strength(self) -> float:
        """获取当前 anchor 强度"""
        return self.log_anchor_strength.exp().item()


def quick_analyze_cls_attention(
    attention_weights: torch.Tensor,
    token_depths: torch.Tensor,
    threshold: float = 0.20,
) -> Dict:
    """
    便捷函数：快速分析 CLS 注意力

    Args:
        attention_weights: [B, H, N, N] 或 [B, N, N]
        token_depths: [B, N]
        threshold: Depth 0 阈值

    Returns:
        包含关键指标的字典
    """
    tracker = CLSAttentionTracker(enabled=True, depth0_threshold=threshold)
    result = tracker.compute_cls_depth_attention(attention_weights, token_depths)

    if result is None:
        return {"enabled": False}

    return {
        "enabled": True,
        "depth0_attention": result.depth0_attention,
        "depth1_attention": result.depth1_attention,
        "depth2_attention": result.depth2_attention,
        "depth3_attention": result.depth3_attention,
        "depth4_attention": result.depth4_attention,
        "link_broken": result.link_broken,
        "attention_by_depth": result.attention_by_depth,
    }
