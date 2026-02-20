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

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from vit_pytorch.core.constants import EMBEDDING_INIT_STD, HILBERT_BIAS_SCALE
# AreaEncoder is imported lazily in the __init__ method to avoid circular imports
from vit_pytorch.core.config import AreaEncoderConfig  # I98-3: 协议驱动配置
from vit_pytorch.core.levels_info import LevelsInfo  # I98-4


class FractalPositionEmbedding(nn.Module):
    """
    高级分形位置编码，完全对齐增强tokenizer
    支持动态层级、Hilbert路径编码和多尺度空间感知
    
    P11-2 修复: max_level 参数现在应传入与 tokenizer.max_level 一致的值，
    确保 Embedding 表大小与实际使用的深度范围匹配，减少约 90% 的参数浪费。
    
    I27-2 修复: 添加 dropout 参数，允许统一控制正则化强度。
    
    数学依据 (Dropout in Position Embedding)
    =========================================
    Position Embedding 是信息瓶颈，需要保持信号完整性。
    
    推荐配置:
        dropout ∈ [0.05, 0.15]
        
    分析:
        - 过高 dropout (>0.2): 位置信息丢失 → 模型无法学习空间关系
        - 过低 dropout (<0.05): 过拟合到特定位置模式
        
    经验公式:
        p_pos ≈ 0.5 × p_transformer  (位置编码应比主干更保守)
    """

    def __init__(
        self,
        dim: int,
        max_level: int = 8,  # P0 修复: 统一使用 max_level
        max_seq_len: int = 10000,
        use_hilbert_encoding: bool = True,
        use_spatial_encoding: bool = True,
        # I27-2: 可配置 Dropout (默认 0.1，约为 transformer dropout 的一半)
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.max_level = max_level  # P0 修复: 统一使用 max_level
        self.max_seq_len = max_seq_len
        self.use_hilbert_encoding = use_hilbert_encoding
        self.use_spatial_encoding = use_spatial_encoding
        self.dropout_rate = dropout  # I27-2: 保存用于调试

        # 1. 深度编码 (Depth Embedding)
        self.depth_embedding = nn.Embedding(max_level + 1, dim)

        # 2. 层级路径编码 (Hierarchical Path Embedding)
        # 替代原有的 LSTM 和 2D 绝对位置编码
        # 每个层级有 4 个象限 (0, 1, 2, 3)
        # 总共 max_level * 4 个唯一的层级-象限组合
        self.quadrant_embedding = nn.Embedding(max_level * 4, dim)

        # I106-2: 残差路径编码 - 防止深层梯度消失
        # 逐层残差连接: x_k = LayerNorm(x_{k-1} + QuadrantEmb(k, q^k))
        self.path_layer_norm = nn.LayerNorm(dim)

        # 3. 融合网络 (简化版)
        # 输入: Depth Emb + Path Emb
        # I27-2: 使用可配置 dropout 替代硬编码
        self.fusion_network = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),  # I27-2: 使用传入的 dropout 参数
        )

        # P11-5: 删除了 level_attention_bias，注意力偏置由 LCAHilbertBias 统一提供

        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.normal_(self.depth_embedding.weight, std=EMBEDDING_INIT_STD)
        nn.init.normal_(self.quadrant_embedding.weight, std=EMBEDDING_INIT_STD)

    def forward(
        self,
        levels_info: LevelsInfo,
        sequence_positions: Optional[torch.Tensor] = None,
        # I31-3: 额外参数用于与 AreaEnhancedPositionEmbedding 接口兼容
        regions: Optional[torch.Tensor] = None,
        image_size: Optional[int] = None,
    ) -> torch.Tensor:
        # I98-4: 兼容 raw tensor 和 LevelsInfo 对象
        if isinstance(levels_info, torch.Tensor):
            # 转换为 LevelsInfo，确保数据类型为 Long
            if levels_info.dtype != torch.long:
                levels_info = levels_info.long()

            # 从数据形状推断 max_level: info_dim = max_level + 1
            info_dim = levels_info.shape[-1]
            inferred_max_level = info_dim - 1
            levels_info = LevelsInfo(data=levels_info, max_level=inferred_max_level)

        if levels_info.data.numel() == 0:
            return torch.zeros(0, self.dim, device=levels_info.data.device, dtype=torch.float32)

        device = levels_info.data.device

        # 从 LevelsInfo 提取 depths 和 paths
        depths = levels_info.depths.clamp(0, self.max_level).long()
        paths = levels_info.paths  # (..., path_len)

        # STAB-7 修复: 确保索引张量为连续格式
        if depths.dim() > 1:
            depths = depths.contiguous()

        # 1. 深度编码
        depth_emb = self.depth_embedding(depths)

        # 2. 层级路径编码
        # 计算每个路径节点的全局索引: level_index * 4 + quadrant_index
        # paths 的第 j 列对应第 j+1 层 (因为 col 0 是 depth)
        
        path_len = paths.shape[-1]
        
        # 生成层级偏移量: [0, 4, 8, ..., (path_len-1)*4]
        level_offsets = torch.arange(path_len, device=device) * 4
        
        # 广播相加: (..., path_len) + (path_len,) -> (..., path_len)
        flat_indices = paths + level_offsets
        
        # 安全截断，防止越界 (虽然理论上不应该发生)
        flat_indices = flat_indices.clamp(0, self.max_level * 4 - 1)

        # STAB-7 修复: 确保索引张量为连续格式
        # channels-last 格式与 Embedding 层不兼容，必须转换为 contiguous
        if flat_indices.dim() > 1:
            flat_indices = flat_indices.contiguous()
        else:
            flat_indices = flat_indices.contiguous()

        # 查找 Embedding: (..., path_len, dim)
        path_embs = self.quadrant_embedding(flat_indices)

        # I106-2: 残差路径编码 - 防止深层梯度消失
        # 逐层残差计算: x_k = LayerNorm(x_{k-1} + QuadrantEmb(k, q^k))
        # x_0 = depth_emb (已在前面计算)
        x = depth_emb  # [B, N, dim]

        # 创建层级索引用于有效判断
        seq_indices = torch.arange(path_len, device=device)

        # 逐层残差连接
        for k in range(path_len):
            # 获取第 k 层的象限嵌入
            layer_emb = path_embs[..., k, :]  # [B, N, dim]

            # 创建有效掩码: 只在有效层级时残差连接
            valid_mask = (seq_indices[k] < depths).unsqueeze(-1).float()  # [B, N, 1]

            # 残差连接: x = x + layer_emb (仅有效位置)
            x = x + layer_emb * valid_mask

            # LayerNorm 归一化
            x = self.path_layer_norm(x)

        path_final = x

        # 3. 融合
        # path_final 已经包含了 depth_emb (作为 x_0)，直接返回
        return self.fusion_network(path_final)

    def check_scale_consistency(self, levels_info: LevelsInfo, epsilon: float = 0.1) -> None:
        """I106-4: 验证尺度一致性约束 C3

        检查 Level-0（全图）Embedding 与四个子象限 Level-1 Embedding 均值的距离

        数学形式:
            dist(E_{level-0}, (1/4) * Σ_{q=0}^{3} E_{level-1}^q) < ε

        Args:
            levels_info: LevelsInfo 实例
            epsilon: 距离阈值（默认 0.1）

        Raises:
            AssertionError: 当距离超过阈值时
        """
        if not self.training:
            return

        device = levels_info.data.device

        # 计算 Level-0 嵌入 (depth=0)
        level_0_mask = levels_info.depths == 0
        if not level_0_mask.any():
            return  # 无 Level-0 token，跳过检查

        # Level-0 嵌入
        pos_emb_0 = self.forward(levels_info)  # [B, N, dim]
        level_0_emb = pos_emb_0[level_0_mask].mean(dim=0)  # [dim]

        # 计算 Level-1 嵌入 (depth=1，四个象限)
        level_1_mask = levels_info.depths == 1
        if not level_1_mask.any():
            return  # 无 Level-1 token，跳过检查

        pos_emb_1 = self.forward(levels_info)  # [B, N, dim]

        # 按象限分组计算均值
        quadrant_means = []
        paths = levels_info.paths  # [B, N, max_level]

        for q in range(4):
            # 找 depth=1 且 quadrant=q 的 token
            q_mask = (levels_info.depths == 1) & (paths[..., 0] == q)
            if q_mask.any():
                quadrant_means.append(pos_emb_1[q_mask].mean(dim=0))

        if len(quadrant_means) < 4:
            return  # 象限不完整，跳过检查

        # 计算 Level-1 均值
        level_1_mean = torch.stack(quadrant_means).mean(dim=0)  # [dim]

        # 计算余弦距离
        cos_dist = 1 - torch.nn.functional.cosine_similarity(
            level_0_emb.unsqueeze(0),
            level_1_mean.unsqueeze(0)
        ).abs()

        # 断言检查
        assert cos_dist.item() < epsilon, (
            f"Scale consistency check failed: "
            f"cos_dist={cos_dist.item():.4f} >= epsilon={epsilon}"
        )


from vit_pytorch.core.constants import EMBEDDING_INIT_STD, HILBERT_BIAS_SCALE


class AreaEnhancedPositionEmbedding(nn.Module):
    """面积增强的位置编码 (I31-3, I98-3: 协议驱动配置)

    数学形式化
    ==========

    面积增强位置编码:
        E_pos = Fusion(E_depth + E_path + λ · E_area)

    其中:
        E_area = AreaEncoder(s)     # 面积嵌入
        λ = area_scale (零初始化)   # 可学习权重

    I98-3: 协议驱动配置
        使用 AreaEncoderConfig 替代硬编码参数。

    属性
    ----
    base_embedding : FractalPositionEmbedding
        基础位置编码
    area_encoder : AreaEncoder
        面积编码器 (从 attn_hilbert_bias.py 导入)
    area_scale : nn.Parameter
        可学习权重 (零初始化)
    config : AreaEncoderConfig
        编码器配置 (I98-3)
    """

    def __init__(
        self,
        dim: int,
        max_level: int = 8,
        fourier_levels: int = 4,
        use_hilbert_encoding: bool = True,
        use_spatial_encoding: bool = True,
        dropout: float = 0.1,
        area_config: Optional[AreaEncoderConfig] = None,
    ):
        """初始化面积增强位置编码 (I98-3 协议驱动版本).

        参数
        ----
        dim : int
            嵌入维度
        max_level : int, optional
            最大层级，默认 8
        fourier_levels : int, optional
            傅里叶频率级别数，默认 4
        use_hilbert_encoding : bool, optional
            是否使用 Hilbert 编码，默认 True
        use_spatial_encoding : bool, optional
            是否使用空间编码，默认 True
        dropout : float, optional
            Dropout 比率，默认 0.1
        area_config : AreaEncoderConfig, optional
            面积编码器配置 (I98-3)
        """
        super().__init__()
        self.dim = dim
        self.max_level = max_level

        # I98-3: 使用配置类
        if area_config is None:
            area_config = AreaEncoderConfig(fourier_levels=fourier_levels)
        self.config = area_config

        # 基础位置编码 (深度 + 路径)
        self.base_embedding = FractalPositionEmbedding(
            dim=dim,
            max_level=max_level,
            use_hilbert_encoding=use_hilbert_encoding,
            use_spatial_encoding=use_spatial_encoding,
            dropout=dropout,
        )

        # 面积编码器 (I31-3, I98-3: 使用配置类)
        from vit_pytorch.layers.attention.hilbert_bias import AreaEncoder
        self.area_encoder = AreaEncoder(
            dim=dim,
            config=area_config
        )

        # 可学习权重 (零初始化)
        self.area_scale = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        """初始化权重。"""
        # Area encoder 权重由其内部初始化
        pass

    def forward(
        self,
        levels_info: LevelsInfo,
        regions: Optional[torch.Tensor] = None,
        image_size: Optional[int] = None,
    ) -> torch.Tensor:
        """计算面积增强位置编码。

        参数
        ----
        levels_info : LevelsInfo
            LevelsInfo 实例
        regions : torch.Tensor, optional
            区域边界张量，形状 [B, N, 4]
        image_size : int or tuple, optional
            图像尺寸，可以是整数或 (W, H) 元组

        返回
        ----
        torch.Tensor
            位置编码，形状 [B, N, dim]
        """
        # I98-4: 兼容 raw tensor 和 LevelsInfo 对象
        if isinstance(levels_info, torch.Tensor):
            # 转换为 LevelsInfo，确保数据类型为 Long
            if levels_info.dtype != torch.long:
                levels_info = levels_info.long()

            # 从数据形状推断 max_level: info_dim = max_level + 1
            info_dim = levels_info.shape[-1]
            inferred_max_level = info_dim - 1
            levels_info = LevelsInfo(data=levels_info, max_level=inferred_max_level)

        # 1. 基础位置编码
        pos_emb = self.base_embedding(levels_info)

        # 2. 面积编码 (直接计算，无需零检查，数学等价)
        if regions is not None and image_size is not None:
            # 处理 image_size 格式：支持 int 或 (W, H) 元组
            if isinstance(image_size, int):
                image_size_tuple = (image_size, image_size)
            else:
                image_size_tuple = image_size

            area_emb = self.area_encoder(regions, image_size_tuple)

            # 处理 CLS token: regions 包含 CLS (全零区域)，但 levels_info 的第一个是 CLS
            # area_emb 的形状是 [B, N, dim]，需要与 pos_emb 对齐
            if area_emb.shape[1] == pos_emb.shape[1] + 1:
                # 跳过 CLS 对应的第一个区域
                area_emb = area_emb[:, 1:, :]

            # 残差注入
            pos_emb = pos_emb + self.area_scale * area_emb

        return pos_emb


# P11-5: 删除了 get_attention_bias 方法
# 注意力偏置功能已由 LCAHilbertBias (attn_hilbert_bias.py) 统一提供
# 该方法基于 LCA 深度计算偏置，语义更精确 (编码空间距离而非尺度组合)


# =============================================================================
# Scheme C: Structured Manifold Bias - Geometry Field
# =============================================================================

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
    ):
        """初始化几何流形场

        Args:
            dim: 嵌入维度
            max_level: 最大四叉树深度
            heads: 注意力头数
        """
        super().__init__()
        self.dim = dim
        self.max_level = max_level
        self.heads = heads

        # 1. 面积编码器: 深度 → 面积 (指数衰减)
        self.area_embedding = nn.Embedding(max_level + 1, dim)

        # 2. 旋转感知编码器
        from vit_pytorch.layers.embeddings.fractal_path import OrientationExtractor
        self.orientation_extractor = OrientationExtractor(
            max_level=max_level,
            embedding_dim=dim
        )

        # 3. 流形场融合网络
        # 输入: area_emb + orientation_emb + lca_emb
        self.manifold_fusion = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )

        # 4. 偏置缩放参数
        self.area_scale = nn.Parameter(torch.zeros(1))
        self.orientation_scale = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.area_embedding.weight, std=0.02)
        # area_scale 零初始化: 初始禁用面积编码
        nn.init.zeros_(self.area_scale)
        nn.init.zeros_(self.orientation_scale)

    def compute_area_encoding(
        self,
        depths: torch.Tensor,
    ) -> torch.Tensor:
        """计算区域面积编码

        数学形式:
            area[d] = 4^{-d}

        Args:
            depths: [B, N] 深度

        Returns:
            area_emb: [B, N, dim] 面积编码
        """
        # 指数衰减: 4^{-d}
        depths_clamped = depths.clamp(0, self.max_level)
        area_weights = 4.0 ** (-depths_clamped.float())

        # 查找嵌入
        area_emb = self.area_embedding(depths_clamped)

        # 应用面积权重
        area_emb = area_emb * area_weights.unsqueeze(-1)

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
        from vit_pytorch.core.levels_info import LevelsInfo

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

        manifold_emb = self.manifold_fusion(manifold_input)

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


class MultiLayerGeometryField(nn.Module):
    """多层几何流形场 (解决 Signal Washout)

    核心思想: 在每一层 Transformer Block 注入几何偏置，
    而非仅在输入层一次性注入

    数学形式:
        For layer l in [0, L-1]:
            bias_l = GeometryField(levels_info, layer_idx=l)
            output_l = Attention(Q_l, K_l, V_l, bias=bias_l)
    """

    def __init__(
        self,
        dim: int,
        max_level: int = 8,
        heads: int = 8,
        num_layers: int = 12,
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers

        # 共享的几何流形场
        self.geometry_field = GeometryField(
            dim=dim,
            max_level=max_level,
            heads=heads,
        )

        # 每层的可学习缩放参数
        self.layer_scales = nn.Parameter(torch.ones(num_layers))

    def forward(
        self,
        levels_info: 'LevelsInfo',
    ) -> List[torch.Tensor]:
        """计算所有层 的几何偏置

        Args:
            levels_info: LevelsInfo 实例

        Returns:
            layer_biases: [num_layers, B, N, dim] 所有层的几何偏置
        """
        # 共享流形场
        base_manifold = self.geometry_field.forward(levels_info)

        # 每层应用不同的缩放
        layer_biases = []
        for layer_idx in range(self.num_layers):
            scale = self.layer_scales[layer_idx]
            layer_bias = base_manifold * scale
            layer_biases.append(layer_bias)

        return layer_biases


# 类型别名用于前向引用
from typing import List
