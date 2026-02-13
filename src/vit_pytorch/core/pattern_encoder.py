# -*- coding: utf-8 -*-
"""
Hilbert序模式编码器模块 (I162-1)

数学形式化
============

核心洞见 (Wang et al., 2021):
    Hilbert序本身就是有效的特征提取序列

    Hilbert曲线 H: [0, N²) → [0, N) × [0, N)

    关键性质:
    1. 局部性: ∀i, ||pos_i - pos_{i+1}||_2 ≤ √2
    2. 层次性: Hilbert曲线是分形结构

    Wang方法:
        I → Hilbert序展开 → CNN → 特征

    本模块:
        Tokens → Hilbert序重排 → 1D卷积 → 模式特征

    数学等价:
        Conv1D(k, σ_H) on Tokens ≈ 2D窗口卷积 on 图像

        其中 σ_H 是Hilbert排序，保证相邻索引对应空间邻域

与Wang论文的对齐:
    ┌─────────────────────────────────────────────────────────────┐
    │  Wang et al.: 图像 → Hilbert序 → 2D-CNN                   │
    │                                                               │
    │  本模块:     Token → Hilbert序重排 → 1D卷积               │
    │                                                               │
    │  等价性:     两者都在Hilbert序上做滑动窗口特征提取         │
    └─────────────────────────────────────────────────────────────┘

多尺度设计:
    ┌─────────────────────────────────────────────────────────────┐
    │  kernel_size     感受野 (等价2D窗口)     适用场景         │
    ├─────────────────────────────────────────────────────────────┤
    │  k=3             ~3×3                   细粒度纹理         │
    │  k=7             ~7×7                   中等模式            │
    │  k=15            ~15×15                 区域结构           │
    └─────────────────────────────────────────────────────────────┘

复杂度分析:
    时间: O(B × N × Σk_i)  (k_i为各分支卷积核大小)
    空间: O(B × N × D × m)  (m为分支数)
    vs 2D卷积: O(B × H × W × Σk_i²)

    对于N=H×W的Token序列:
    - 1D: O(N × Σk_i) = O(HW × Σk_i)
    - 2D: O(HW × Σk_i²)

    当k_i << min(H,W)时，1D卷积更高效
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HilbertPatternEncoder(nn.Module):
    """Hilbert序模式编码器 (I162-1 核心实现)

    将Token按Hilbert序重排，然后应用多尺度1D卷积捕捉空间异质性模式。

    数学形式化
    ----------
        t̃ = Tokens[σ_H]           # Hilbert序重排
        f^{(w)} = Conv1D_w(t̃)    # 窗口大小为w的卷积
        f = Fusion([f^{(w1)}, f^{(w2)}, ...])

    其中 σ_H 是Hilbert排序，满足:
        ||pos_i - pos_{i+1}||_2 ≤ √2

    这保证1D卷积的相邻感受野对应2D空间邻域。

    Args:
        dim: Token特征维度
        window_sizes: 多尺度卷积核大小元组
        out_dim: 输出维度（默认等于dim）
        dropout: Dropout比率
    """

    def __init__(
        self,
        dim: int,
        window_sizes: Tuple[int, ...] = (3, 7, 15),
        out_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.window_sizes = window_sizes
        self.out_dim = out_dim or dim

        # 多尺度卷积分支
        # 使用 groups=dim 实现深度可分离卷积（通道独立）
        self.convs = nn.ModuleList([
            nn.Sequential(
                # depthwise卷积: groups=dim, 每个通道独立卷积
                nn.Conv1d(dim, dim, w, padding=w // 2, groups=dim),
                nn.GELU(),
                # pointwise卷积: 1x1卷积进行通道间交互
                nn.Conv1d(dim, dim, 1),
                nn.GELU(),
            )
            for w in window_sizes
        ])

        # 特征融合层
        fusion_dim = dim * len(window_sizes)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, self.out_dim),
        )

        # 可学习的缩放因子（初始为1）
        self.scale = nn.Parameter(torch.ones(1))

    def forward(
        self,
        tokens: torch.Tensor,
        hilbert_order: torch.Tensor
    ) -> torch.Tensor:
        """前向传播

        Args:
            tokens: [B, N, D] 原始Token特征
            hilbert_order: [N] Hilbert排序索引

        Returns:
            pattern_features: [B, N, out_dim] 模式特征
        """
        B, N, D = tokens.shape

        # 验证输入
        if hilbert_order.shape[0] != N:
            raise ValueError(
                f"hilbert_order长度 ({hilbert_order.shape[0]}) "
                f"不等于token数量 ({N})"
            )

        # 1. Hilbert序重排
        # 确保hilbert_order是整数类型
        if hilbert_order.dtype != torch.long:
            hilbert_order = hilbert_order.long()

        sorted_tokens = tokens[:, hilbert_order]  # [B, N, D]

        # 2. 转换为Conv1d格式 [B, D, N]
        x = sorted_tokens.transpose(1, 2)

        # 3. 多尺度卷积
        multi_scale_features = []
        for conv in self.convs:
            feat = conv(x)  # [B, D, N]
            multi_scale_features.append(feat.transpose(1, 2))  # [B, N, D]

        # 4. 特征拼接
        concatenated = torch.cat(multi_scale_features, dim=-1)  # [B, N, D * len(window_sizes)]

        # 5. 融合为输出维度
        pattern_features = self.fusion(concatenated)  # [B, N, out_dim]

        # 6. 应用缩放
        pattern_features = pattern_features * self.scale

        return pattern_features

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, "
            f"window_sizes={self.window_sizes}, "
            f"out_dim={self.out_dim}"
        )


class HilbertPatternEncoderLight(HilbertPatternEncoder):
    """轻量级Hilbert模式编码器 (单尺度版本)

    适用于计算资源受限的场景。
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        out_dim: Optional[int] = None,
    ):
        super().__init__(
            dim=dim,
            window_sizes=(kernel_size,),
            out_dim=out_dim,
            dropout=0.0,  # 轻量版无dropout
        )

    def extra_repr(self) -> str:
        return f"dim={self.dim}, kernel_size={self.window_sizes[0]}, out_dim={self.out_dim}"


class MultiHeadHilbertPatternEncoder(nn.Module):
    """多头Hilbert模式编码器 (I162-1 探索版本)

    与Multi-Head Attention类似，将特征分成多个头独立处理。
    每个头捕捉不同类型的空间模式。

    数学形式化
    ----------
        Q_h = Tokens @ W_Qh    # 头h的查询
        K_h = Tokens @ W_Kh    # 头h的键
        V_h = Tokens @ W_Vh    # 头h的值

        Pattern_h = Conv1D(Q_h, K_h, V_h)  # 头h的模式

        Output = Concat([Pattern_h for h in heads]) @ W_O

    优势:
        - 多头并行捕捉不同尺度的模式
        - 与Transformer架构一致，易于集成
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        window_sizes: Tuple[int, ...] = (3, 7),
        head_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim or (dim // num_heads)
        self.out_dim = out_dim or dim

        if self.head_dim * num_heads != dim:
            raise ValueError(
                f"dim ({dim}) 必须能被 num_heads ({num_heads}) 整除，"
                f"或者指定 head_dim 使其乘积等于 dim"
            )

        # 投影层
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim)
        self.k_proj = nn.Linear(dim, num_heads * self.head_dim)
        self.v_proj = nn.Linear(dim, num_heads * self.head_dim)

        # 模式编码器（每头独立）
        self.pattern_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(self.head_dim, self.head_dim, w, padding=w // 2, groups=self.head_dim),
                nn.GELU(),
                nn.Conv1d(self.head_dim, self.head_dim, 1),
            )
            for w in window_sizes
        ])

        # 输出融合
        fusion_dim = self.head_dim * num_heads * len(window_sizes)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, self.out_dim),
        )

        self.scale = nn.Parameter(torch.ones(1))

    def forward(
        self,
        tokens: torch.Tensor,
        hilbert_order: torch.Tensor
    ) -> torch.Tensor:
        B, N, D = tokens.shape

        # 1. 多头投影
        Q = self.q_proj(tokens).view(B, N, self.num_heads, self.head_dim)
        K = self.k_proj(tokens).view(B, N, self.num_heads, self.head_dim)
        V = self.v_proj(tokens).view(B, N, self.num_heads, self.head_dim)

        # [B, num_heads, N, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # 2. Hilbert序重排
        if hilbert_order.dtype != torch.long:
            hilbert_order = hilbert_order.long()
        sorted_order = hilbert_order.unsqueeze(0).unsqueeze(-1)  # [1, 1, N, 1]

        Q_sorted = Q.gather(2, sorted_order.expand(-1, -1, -1, self.head_dim))
        K_sorted = K.gather(2, sorted_order.expand(-1, -1, -1, self.head_dim))
        V_sorted = V.gather(2, sorted_order.expand(-1, -1, -1, self.head_dim))

        # 3. 多尺度模式编码
        multi_scale_out = []
        for encoder in self.pattern_encoders:
            # 合并heads和head_dim用于卷积
            x = Q_sorted.reshape(B * self.num_heads, self.head_dim, N)

            feat = encoder(x)  # [B*heads, head_dim, N]
            feat = feat.view(B, self.num_heads, self.head_dim, N)

            # 与V_sorted交互
            # Pattern = Attention(Q_sorted, K_sorted, V_sorted) + Conv_feat
            attn = torch.matmul(
                Q_sorted / (self.head_dim ** 0.5),
                K_sorted.transpose(-2, -1)
            )
            attn = F.softmax(attn, dim=-1)

            attended = torch.matmul(attn, V_sorted)  # [B, heads, N, head_dim]

            combined = attended + feat.transpose(1, 2)  # [B, N, heads, head_dim]
            multi_scale_out.append(combined)

        # 4. 拼接并融合
        concatenated = torch.cat(multi_scale_out, dim=-1)  # [B, N, heads*head_dim*len(window_sizes)]
        output = self.fusion(concatenated)  # [B, N, out_dim]

        return output * self.scale


def create_hilbert_pattern_encoder(
    dim: int,
    mode: str = "light",
    **kwargs
) -> HilbertPatternEncoder:
    """工厂函数：创建合适的Hilbert模式编码器

    Args:
        dim: Token特征维度
        mode: 模式选择
            - "light": 轻量级，单卷积核
            - "standard": 标准，多尺度卷积
            - "multihead": 多头版本
        **kwargs: 其他参数

    Returns:
        HilbertPatternEncoder实例
    """
    if mode == "light":
        return HilbertPatternEncoderLight(dim=dim, **kwargs)
    elif mode == "standard":
        return HilbertPatternEncoder(dim=dim, **kwargs)
    elif mode == "multihead":
        return MultiHeadHilbertPatternEncoder(dim=dim, **kwargs)
    else:
        raise ValueError(f"未知模式: {mode}")
