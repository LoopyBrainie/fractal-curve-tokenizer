# -*- coding: utf-8 -*-
"""
分形位置编码模块

数学形式化
============

分形位置编码结合深度和路径信息:

    E_pos(i) = Fusion(E_depth(d_i) + E_path(i))

深度编码 (Depth Embedding):
    E_depth: Z → R^D
    E_depth(d) = Embedding(d), d ∈ {0, 1, ..., L_max}

路径编码 (Path Embedding):
    对 token i，其从根到叶的路径为 (q_i^(1), ..., q_i^(d_i))
    其中 q_i^(j) ∈ {0, 1, 2, 3} 是第 j 层的象限索引
    
    E_path(i) = Σ_{j=1}^{d_i} QuadrantEmb(j, q_i^(j))
    
    QuadrantEmb: [L_max × 4, D] 的可学习嵌入表

融合网络:
    Fusion(x) = Linear(LayerNorm(x))

注意力偏置:
    B_level[i,j] = LevelAttnBias[d_i, d_j]
    可学习的 [L_max+1, L_max+1] 偏置矩阵

复杂度分析
----------
forward (位置编码):
    时间: O(N · L_max · D)
          ├─ 深度编码:  O(N)            — Embedding lookup
          ├─ 路径编码:  O(N · L_max · D) — 路径求和 + 归一化
          └─ 融合网络:  O(N · D · D)     — MLP
    空间: O(L_max · 4 · D)  — quadrant_embedding 表

其中: N=seq_len, D=dim, L_max=max_level

P11-5 修复: 删除了未使用的 level_attention_bias 参数和 get_attention_bias 方法。
注意力偏置功能已由 LCAHilbertBias (attn_hilbert_bias.py) 统一提供。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from vit_pytorch.core.levels_info import LevelsInfo  # I98-4
from vit_pytorch.layers.embeddings.fractal_path import OrientationExtractor  # Scheme C


class GeometryField(nn.Module):
    """几何流形场编码器 (Scheme C 核心组件)

    数学形式化
    ===========

    核心思想: 将位置编码从"一次性注入"转为"每层注入的结构化偏置"

    流形场定义:
        M(i, j) = f(area_i, area_j, LCA_depth(i,j), orientation_i, orientation_j)

    其中:
        - area_i, area_j: 区域面积 (指数衰减 4^{-d})
        - LCA_depth: 共同祖先深度
        - orientation: 旋转状态

    注入机制:
        每层 Transformer Block 的 QK 计算时:
            Attention(Q, K) = softmax(QK^T / √d + M)

    优势:
        1. 每层注入，解决 Signal Washout (深层稀释)
        2. 流形场保持尺度-位置耦合
        3. 旋转感知保持 Hilbert 几何
        4. 完全向量化，无 Python 循环
    """

    def __init__(
        self,
        dim: int,
        max_level: int = 8,
        heads: int = 8,
        # I-PHASE4: Low-Rank 优化
        rank: int = 16,
        # P1: 依赖注入 - 外部传入的 OrientationExtractor
        orientation_extractor: Optional['OrientationExtractor'] = None,
    ):
        """初始化几何流形场

        Args:
            dim: 嵌入维度
            max_level: 最大四叉树深度
            heads: 注意力头数
            rank: Low-Rank 分解的秩 (默认 16，远小于 dim=256)
            orientation_extractor: 依赖注入的旋转感知提取器 (P1)
                - 如果提供，则使用提供的实例 (共享状态)
                - 如果为 None，则内部创建新实例
        """
        super().__init__()
        self.dim = dim
        self.max_level = max_level
        self.heads = heads
        self.rank = rank

        # 1. 面积编码器: 深度 → 面积 (指数衰减)
        self.area_embedding = nn.Embedding(max_level + 1, dim)

        # 2. 旋转感知编码器 (P1: 依赖注入)
        if orientation_extractor is not None:
            # 使用外部提供的实例 (Splitter 和 GeometryField 共享)
            self.orientation_extractor = orientation_extractor
        else:
            # 内部创建 (向后兼容)
            self.orientation_extractor = OrientationExtractor(
                max_level=max_level,
                embedding_dim=dim
            )

        # I-PHASE4: Low-Rank 流形场融合网络
        # 原始: O(dim * 3 * dim) 参数
        # Low-Rank: O(dim * 3 * rank + rank * dim) 参数
        # 压缩比: ~3*dim / (3*rank + rank) = ~dim/rank
        # 当 dim=256, rank=16 时，压缩约 16x
        #
        # 注意: 移除 LayerNorm(rank) 因为它会将 manifold 信息归一化到单位球面，
        # 导致 ||manifold_emb|| ≈ 恒定 (约 0.99)，使 Poincaré 距离失去诊断意义。
        # 改用直接 GELU 激活，保留manifold嵌入的原始尺度变化。
        input_dim = dim * 3
        self.manifold_fusion_lowrank = nn.Sequential(
            nn.Linear(input_dim, rank),  # 压缩到低秩
            nn.GELU(),                    # 直接激活，不归一化
            nn.Linear(rank, dim),        # 解压回原始维度
        )

        # 原始融合网络 (当 rank=dim 时退化为完整版本)
        self.manifold_fusion_full = nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )

        # 4. 偏置缩放参数
        self.area_scale = nn.Parameter(torch.zeros(1))
        self.orientation_scale = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.area_embedding.weight, std=0.02)
        # P1 FIX: 启用面积编码的梯度流动
        nn.init.ones_(self.area_scale)  # 初始为1，允许面积编码参与训练
        nn.init.zeros_(self.orientation_scale)

    def compute_area_encoding(
        self,
        depths: torch.Tensor,
    ) -> torch.Tensor:
        """计算区域面积编码

        数学形式:
            area[d] = 4^{-d} (原始硬编码)
            area[d] = γ^{-d} (P1 FIX: 可学习衰减底数)

        P1 FIX: 移除硬编码的 4^{-d} 衰减，使用可学习的深度嵌入
        原始问题: 当 d=8 时，4^{-8} ≈ 0.000015，导致 manifold_bias_mean ≈ 0.0026
        解决方案: 让模型通过 area_embedding 学习每个深度的最优表示，
                 area_scale 动态调整整体强度

        Args:
            depths: [B, N] 深度

        Returns:
            area_emb: [B, N, dim] 面积编码
        """
        depths_clamped = depths.clamp(0, self.max_level)

        # P1 FIX: 移除硬编码的 4^{-d} 衰减，直接使用可学习的嵌入
        # area_scale (初始为0) 控制整体强度，area_embedding 学习每层最优表示
        area_emb = self.area_embedding(depths_clamped)

        # 可选：添加深度感知的缩放（如果需要保持一些深度感知）
        # 使用 tanh 限制深度范围，避免指数级差异
        depth_factor = torch.tanh((depths_clamped.float() - self.max_level / 2) / (self.max_level / 2))
        area_emb = area_emb * (1.0 + 0.1 * depth_factor.unsqueeze(-1))

        return area_emb

    def forward(
        self,
        levels_info: 'LevelsInfo',
    ) -> torch.Tensor:
        """计算几何流形场偏置

        Args:
            levels_info: LevelsInfo 实例

        Returns:
            manifold_bias: [B, dim] 流形场编码
        """

        # 提取深度和路径
        depths = levels_info.depths.clamp(0, self.max_level)
        paths = levels_info.paths

        # 1. 面积编码
        area_emb = self.compute_area_encoding(depths)

        # 2. 旋转感知编码
        rot_emb, dir_emb = self.orientation_extractor(paths, depths)

        # 3. 融合流形场
        manifold_input = torch.cat([
            area_emb,
            rot_emb,
            dir_emb,
        ], dim=-1)  # [B, N, dim*3]

        # I-PHASE4: 选择 Low-Rank 或完整融合网络
        if self.rank < self.dim:
            manifold_emb = self.manifold_fusion_lowrank(manifold_input)
        else:
            manifold_emb = self.manifold_fusion_full(manifold_input)

        return manifold_emb

    def get_layer_bias(
        self,
        levels_info: 'LevelsInfo',
        layer_idx: int,
        num_layers: int,
    ) -> torch.Tensor:
        """获取每层的几何偏置 (带温度衰减)

        Args:
            levels_info: LevelsInfo 实例
            layer_idx: 当前层索引
            num_layers: 总层数

        Returns:
            layer_bias: [B, N, dim] 当前层的几何偏置
        """
        # 基础流形场
        manifold_emb = self.forward(levels_info)

        # 温度衰减: 深层偏置逐渐减弱 (防止过度结构化)
        temperature = 1.0 - (layer_idx / num_layers) * 0.5

        # 缩放
        layer_bias = manifold_emb * temperature

        return layer_bias

    @property
    @torch._dynamo.disable  # 🌟 修复：禁止 Dynamo 追踪此属性
    def embed_output(self) -> Dict[str, Any]:
        """GeometryField 诊断输出

        命名空间:
            embed/params/*: 可学习尺度参数

        I-OOM FIX: 使用 Disable & Flush 模式
        """

        output: Dict[str, Any] = {}

        # embed/params/* - 几何缩放参数
        if hasattr(self, 'area_scale') and self.area_scale is not None:
            output["params/area_scale"] = float(self.area_scale.item())

        if hasattr(self, 'orientation_scale') and self.orientation_scale is not None:
            output["params/orientation_scale"] = float(self.orientation_scale.item())

        return output
