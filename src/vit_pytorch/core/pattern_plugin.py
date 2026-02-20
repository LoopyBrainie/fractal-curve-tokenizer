# -*- coding: utf-8 -*-
"""
双路径插件模块 (I162-1)

将 DualPathFractalViT 的双路径功能以插件化方式整合到 FractalCurveViT 中。

数学形式化
============

核心设计：Dual-Path Plugin

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    HilbertPatternPlugin                                 │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   Input: Tokens + PosEmb + CLS                                          │
    │                 │                                                        │
    │                 ▼                                                        │
    │   ┌─────────────────────┐      ┌─────────────────────────┐            │
    │   │  Path 1: Direct    │      │  Path 2: Pattern        │            │
    │   │                     │      │                         │            │
    │   │  Transformer       │      │  Hilbert Sort           │            │
    │   │  (原始语义)         │      │  → 1D Conv (多尺度)     │            │
    │   │                     │      │  → Pattern Features     │            │
    │   │                     │      │           ↓             │            │
    │   │                     │      │  Transformer           │            │
    │   └──────────┬──────────┘      └───────────┬────────────┘            │
    │              │                              │                          │
    │              └──────────────┬───────────────┘                          │
    │                          ↓                                              │
    │               ┌─────────────────────┐                                  │
    │               │   Feature Fusion   │                                  │
    │               │ cat([D, P], dim=-1)│  [B, N, 2D]                     │
    │               │ LayerNorm → FFN   │                                  │
    │               └─────────┬───────────┘                                  │
    │                         ▼                                              │
    │               ┌─────────────────────┐                                  │
    │               │ CLS → Pooled        │  [B, 2D]                       │
    │               │ → MLP Head          │                                  │
    │               └─────────────────────┘                                  │
    └─────────────────────────────────────────────────────────────────────────┘

与 DualPathFractalViT 的关系
--------------------------
    DualPathFractalViT = FractalCurveViT(..., use_pattern_plugin=True)

    插件模式复用 DualPathFractalViT 的核心逻辑:
    - 双路径并行处理
    - 特征拼接融合
    - mlp_head 输入维度翻倍

向后兼容性
----------
    use_pattern_plugin=False 时:
    - 行为与原始 FractalCurveViT 完全一致
    - 无额外计算开销
    - 可加载旧模型权重 (strict=False)
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from vit_pytorch.core.pattern_encoder import (
    HilbertPatternEncoder,
    HilbertPatternEncoderLight,
)


class HilbertPatternPlugin(nn.Module):
    """双路径插件：封装模式编码与特征融合逻辑

    将 DualPathFractalViT 的双路径并行处理逻辑封装为可插拔的插件组件。

    数学形式:
        direct_out = Transformer(Tokens + PosEmb)      # [B, N+1, D]
        pattern_out = Transformer(PatternFeats)        # [B, N+1, D]
        fused = cat([direct[:, 1:], pattern[:, 1:]], dim=-1)  # [B, N, 2D]
        pooled = cls + fused[0]  # [B, 2D]

    Args:
        dim: 基础维度
        pattern_dim: 模式特征维度 (默认等于 dim)
        pattern_scale: 模式特征缩放因子
        pattern_encoder: HilbertPatternEncoder 实例
    """

    def __init__(
        self,
        dim: int,
        pattern_dim: Optional[int] = None,
        pattern_scale: float = 1.0,
        pattern_encoder: Optional[HilbertPatternEncoder] = None,
    ):
        super().__init__()
        self.dim = dim
        self.pattern_dim = pattern_dim or dim
        self.pattern_scale = pattern_scale
        self.pattern_encoder = pattern_encoder

    def forward(
        self,
        direct_tokens: torch.Tensor,      # [B, N+1, D] 含 CLS
        levels_info: Any,
        geometry_emb: torch.Tensor,       # [B, N+1, D]
        transformer: nn.Module,
        hilbert_order: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """双路径前向传播

        Args:
            direct_tokens: 直接分支的输入 tokens [B, N+1, D]，包含 CLS
            levels_info: 层级信息
            geometry_emb: 几何嵌入 [B, N+1, D]
            transformer: Transformer 模块
            hilbert_order: Hilbert 排序索引 [N]

        Returns:
            fused: [B, N, 2D] 融合后的 token 特征 (不含 CLS)
            pooled: [B, 2D] 用于分类的 CLS 特征
        """
        B = direct_tokens.shape[0]

        # ============================================================
        # Path 1: 直接分支
        # ============================================================
        direct_out = transformer(
            direct_tokens,
            levels_info=levels_info,
            geometry_emb=geometry_emb,
        )  # [B, N+1, D]

        # ============================================================
        # Path 2: 模式分支
        # ============================================================
        if self.pattern_encoder is not None and hilbert_order is not None:
            # 提取 token (不含 CLS)
            tokens = direct_tokens[:, 1:, :]  # [B, N, D]

            # 模式编码
            pattern_features = self.pattern_encoder(tokens, hilbert_order)  # [B, N, D]

            # 添加 CLS 位置 (填充零)
            pattern_with_cls = torch.cat([
                torch.zeros(B, 1, self.pattern_dim, device=tokens.device),
                pattern_features * self.pattern_scale
            ], dim=1)  # [B, N+1, D]

            # 模式分支 Transformer
            pattern_out = transformer(
                pattern_with_cls,
                levels_info=levels_info,
                geometry_emb=geometry_emb,
            )  # [B, N+1, D]
        else:
            # 无模式编码器时，退化为单路径
            pattern_out = direct_out

        # ============================================================
        # 特征融合
        # ============================================================
        # 从两个分支获取特征 (不含 CLS)
        direct_token_feats = direct_out[:, 1:, :]   # [B, N, D]
        pattern_token_feats = pattern_out[:, 1:, :]  # [B, N, D]

        # 拼接两个分支
        fused = torch.cat([direct_token_feats, pattern_token_feats], dim=-1)  # [B, N, 2D]

        # ============================================================
        # CLS 特征构建
        # ============================================================
        # 使用直接分支的 CLS 特征，扩展到 2D 维度
        cls_feat = direct_out[:, 0:1, :]  # [B, 1, D]
        cls_feat_expanded = torch.cat([cls_feat, cls_feat], dim=-1)  # [B, 1, 2D]

        # 构建带 CLS 的完整特征
        fused_with_cls = torch.cat([cls_feat_expanded, fused], dim=1)  # [B, N+1, 2D]

        # 池化 CLS 特征用于分类
        pooled = fused_with_cls[:, 0, :]  # [B, 2D]

        return fused, pooled

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, "
            f"pattern_dim={self.pattern_dim}, "
            f"pattern_scale={self.pattern_scale}, "
            f"has_encoder={self.pattern_encoder is not None}"
        )


class HilbertPatternPluginFactory:
    """插件工厂：创建 HilbertPatternPlugin 实例

    用于从配置字典创建插件，保持与 FractalCurveViT 的解耦。
    """

    @staticmethod
    def create(
        dim: int,
        config: Optional[dict] = None,
        pattern_encoder: Optional[HilbertPatternEncoder] = None,
    ) -> Optional[HilbertPatternPlugin]:
        """创建插件实例

        Args:
            dim: 基础维度
            config: 插件配置字典，支持字段:
                - enabled: bool (默认 False)
                - pattern_dim: int (默认等于 dim)
                - pattern_scale: float (默认 1.0)
            pattern_encoder: 已有的 HilbertPatternEncoder 实例

        Returns:
            HilbertPatternPlugin 实例 或 None (如果未启用)
        """
        if config is None:
            config = {}

        enabled = config.get("enabled", False)
        if not enabled:
            return None

        return HilbertPatternPlugin(
            dim=dim,
            pattern_dim=config.get("pattern_dim", dim),
            pattern_scale=config.get("pattern_scale", 1.0),
            pattern_encoder=pattern_encoder,
        )


def create_hilbert_pattern_plugin(
    dim: int,
    enabled: bool = False,
    pattern_dim: Optional[int] = None,
    pattern_scale: float = 1.0,
    pattern_encoder: Optional[HilbertPatternEncoder] = None,
) -> Optional[HilbertPatternPlugin]:
    """工厂函数：创建 HilbertPatternPlugin 实例

    Args:
        dim: 基础维度
        enabled: 是否启用双路径插件
        pattern_dim: 模式特征维度
        pattern_scale: 模式特征缩放因子
        pattern_encoder: HilbertPatternEncoder 实例

    Returns:
        HilbertPatternPlugin 实例 或 None
    """
    if not enabled:
        return None

    return HilbertPatternPlugin(
        dim=dim,
        pattern_dim=pattern_dim,
        pattern_scale=pattern_scale,
        pattern_encoder=pattern_encoder,
    )
