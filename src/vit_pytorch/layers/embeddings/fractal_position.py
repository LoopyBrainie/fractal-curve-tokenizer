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
import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from vit_pytorch.core.constants import EMBEDDING_INIT_STD, EPS
from vit_pytorch.core.config import AreaEncoderConfig  # I98-3: 协议驱动配置
from vit_pytorch.core.levels_info import LevelsInfo  # I98-4
from vit_pytorch.core.depth_utils import compute_normalized_area
from vit_pytorch.layers.embeddings.fractal_path import OrientationExtractor  # Scheme C


# =============================================================================
# AreaEncoder (moved from hilbert_bias.py)
# =============================================================================


class AreaEncoder(nn.Module):
    """面积编码器 (I31-3, I32-7, I98-3: 协议驱动配置化)"""

    def __init__(
        self,
        dim: int,
        config: Optional[AreaEncoderConfig] = None,
    ):
        super().__init__()

        if config is None:
            config = AreaEncoderConfig()

        self.config = config
        self.dim = dim
        self.fourier_levels = config.fourier_levels
        self.freq_base = config.freq_base
        self.fourier_dim = config.fourier_levels * 2  # sin + cos
        self.cutoff_ratio = config.cutoff_ratio

        if config.fourier_levels > 0:
            self.fourier_proj = nn.Linear(self.fourier_dim, config.hidden_dim)
            self.mlp = nn.Sequential(
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, dim)
            )
        else:
            self.fourier_proj = None
            self.mlp = None

        self.area_weight = nn.Parameter(torch.zeros(1))
        self._init_weights()

    def get_config(self) -> AreaEncoderConfig:
        return AreaEncoderConfig(
            fourier_levels=self.fourier_levels,
            freq_base=self.freq_base,
            hidden_dim=self.config.hidden_dim,
            output_dim=self.dim,
            cutoff_ratio=self.cutoff_ratio,
        )

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _compute_nyquist_normalized_size(
        self,
        regions: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> torch.Tensor:
        B, N, _ = regions.shape
        W, H = image_size

        widths = torch.abs(regions[..., 2] - regions[..., 0])
        heights = torch.abs(regions[..., 3] - regions[..., 1])

        L_patch = torch.sqrt(widths * heights + EPS)
        L_image = math.sqrt(W * H)
        L_norm = L_patch / L_image

        return L_norm

    def _compute_dynamic_fourier_features(
        self,
        area_scores: torch.Tensor,
        L_norm: torch.Tensor,
    ) -> torch.Tensor:
        B, N = area_scores.shape

        omega_nyquist = (math.pi / (L_norm + EPS)).unsqueeze(-1)
        omega_cutoff = omega_nyquist * self.cutoff_ratio

        base_freqs = torch.tensor(
            [math.pi * (self.freq_base ** k) for k in range(self.fourier_levels)],
            device=area_scores.device,
            dtype=torch.float32,  # D4-AUDIT FIX: Force FP32 for numerical stability in trig ops
        ).view(1, 1, self.fourier_levels)

        denom = omega_nyquist - omega_cutoff
        denom_safe = denom.clamp(min=1e-6)

        gate_full = torch.where(
            base_freqs <= omega_cutoff,
            torch.ones(1, device=area_scores.device, dtype=area_scores.dtype),
            torch.where(
                base_freqs < omega_nyquist,
                0.5 * (1 + torch.cos(
                    math.pi * (base_freqs - omega_cutoff) / denom_safe
                )),
                torch.zeros(1, device=area_scores.device, dtype=area_scores.dtype)
            )
        )

        area_expanded = area_scores.unsqueeze(-1)
        freq_times_area = base_freqs * area_expanded

        sin_all = torch.sin(freq_times_area.clamp(-100, 100)) * gate_full  # D4-AUDIT FIX: 输入 clamp 防止数值不稳定
        cos_all = torch.cos(freq_times_area.clamp(-100, 100)) * gate_full

        gamma = torch.stack([sin_all, cos_all], dim=-1).view(B, N, 2 * self.fourier_levels)

        return gamma

    def forward(
        self,
        regions: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> torch.Tensor:
        B, N, _ = regions.shape

        if self.fourier_levels == 0:
            return torch.zeros(B, N, self.dim, device=regions.device)

        if isinstance(image_size, int):
            image_size_tuple = (image_size, image_size)
        else:
            image_size_tuple = image_size
        W, H = image_size_tuple

        area_scores = compute_normalized_area(regions, image_size, epsilon=EPS)
        L_norm = self._compute_nyquist_normalized_size(regions, image_size_tuple)
        gamma = self._compute_dynamic_fourier_features(area_scores, L_norm)
        area_emb = self.mlp(self.fourier_proj(gamma))
        area_emb = area_emb * self.area_weight
        area_emb = F.normalize(area_emb, p=2, dim=-1)

        return area_emb


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

        # I-NAN: 统一诊断缓存
        self._diagnostic_cache: Dict[str, Any] = {}

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
        # Fix-12: 使用统一的护城河工厂方法
        levels_info = LevelsInfo.ensure(levels_info, default_max_level=self.max_level)

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
        level_offsets = torch.arange(path_len, device=device) << 2  # D4-AUDIT FIX: *4 → <<2
        
        # 广播相加: (..., path_len) + (path_len,) -> (..., path_len)
        flat_indices = paths + level_offsets
        
        # 安全截断，防止越界 (虽然理论上不应该发生)
        flat_indices = flat_indices.clamp(0, self.max_level * 4 - 1)

        # STAB-7 修复: 确保索引张量为连续格式
        # channels-last 格式与 Embedding 层不兼容，必须转换为 contiguous
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

    def check_scale_consistency(
        self, levels_info: LevelsInfo, epsilon: float = 0.1
    ) -> Optional[torch.Tensor]:
        """I106-4: 验证尺度一致性约束 C3

        检查 Level-0（全图）Embedding 与四个子象限 Level-1 Embedding 均值的距离

        数学形式:
            dist(E_{level-0}, (1/4) * Σ_{q=0}^{3} E_{level-1}^q) < ε

        Args:
            levels_info: LevelsInfo 实例
            epsilon: 距离阈值（默认 0.1）

        Returns:
            cos_dist: 余弦距离，供日志系统记录。None 表示跳过检查。
        """
        if not self.training:
            return None


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

                # I184: 记录但不中断训练（尺度一致性作为日志指标）
        if cos_dist > epsilon:
            warnings.warn(
                f"Scale consistency violation: {float(cos_dist):.4f} > {epsilon}"
            )

        # I-NAN: 缓存结果供 embed_output 使用
        self._diagnostic_cache["health/scale_consistency_dist"] = float(cos_dist)

        return cos_dist  # 返回供日志系统记录

@property
    @torch._dynamo.disable  # 🌟 修复：禁止 Dynamo 追踪此属性，防止 Guard 失败导致重编译泄漏
    def embed_output(self) -> Dict[str, Any]:
        """FractalPositionEmbedding 诊断输出

        命名空间:
            embed/params/*: 可学习参数统计
            embed/health/*: 数值健康度

        I-OOM FIX: 使用 Disable & Flush 模式：
        - @torch._dynamo.disable 屏蔽追踪
        - 读取后立即 .cpu().item() 迁移到 CPU
        - 读取后立即清空缓存斩断计算图引用
        """
        output: Dict[str, Any] = {}

        # embed/params/* - 嵌入权重统计
        if hasattr(self, 'depth_embedding') and self.depth_embedding is not None:
            w = self.depth_embedding.weight
            output["params/depth_emb_norm"] = float(w.norm().item())
            output["params/depth_emb_mean"] = float(w.mean().item())
            output["params/depth_emb_std"] = float(w.std().item())

        if hasattr(self, 'quadrant_embedding') and self.quadrant_embedding is not None:
            w = self.quadrant_embedding.weight
            output["params/quadrant_emb_norm"] = float(w.norm().item())
            output["params/quadrant_emb_mean"] = float(w.mean().item())
            output["params/quadrant_emb_std"] = float(w.std().item())

        # I-NAN: 从统一诊断缓存合并 forward 中计算的指标
        # I-OOM FIX: 访问后清空缓存，防止累积导致显存泄漏
        if self._diagnostic_cache:
            for k, v in self._diagnostic_cache.items():
                if isinstance(v, torch.Tensor):
                    output[k] = v.detach().cpu().item()
                else:
                    output[k] = v
            self._diagnostic_cache.clear()  # 🌟 立即清空缓存释放计算图

        # embed/health/* - nan_grad_hooks 注册数
        if hasattr(self, '_nan_grad_hooks') and self._nan_grad_hooks:
            output["health/nan_grad_hooks_registered"] = len(self._nan_grad_hooks)

        return output


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
        # Fix-12: 使用统一的护城河工厂方法
        levels_info = LevelsInfo.ensure(levels_info, default_max_level=self.max_level)

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

    @property
    @torch._dynamo.disable  # 🌟 修复：禁止 Dynamo 追踪此属性
    def embed_output(self) -> Dict[str, Any]:
        """AreaEnhancedPositionEmbedding 诊断输出

        命名空间:
            embed/params/*: 可学习参数统计

        I-OOM FIX: 使用 Disable & Flush 模式
        """
        output: Dict[str, Any] = {}

        # embed/params/* - 面积增强尺度参数
        if hasattr(self, 'area_scale') and self.area_scale is not None:
            output["params/area_scale"] = float(self.area_scale.item())

        # 合并 base_embedding 的诊断输出
        if hasattr(self, 'base_embedding') and self.base_embedding is not None:
            base_output = self.base_embedding.embed_output
            output.update(base_output)

        return output


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
        # I-PHASE4: Low-Rank 优化
        rank: int = 16,
    ):
        """初始化几何流形场

        Args:
            dim: 嵌入维度
            max_level: 最大四叉树深度
            heads: 注意力头数
            rank: Low-Rank 分解的秩 (默认 16，远小于 dim=256)
        """
        super().__init__()
        self.dim = dim
        self.max_level = max_level
        self.heads = heads
        self.rank = rank

        # 1. 面积编码器: 深度 → 面积 (指数衰减)
        self.area_embedding = nn.Embedding(max_level + 1, dim)

        # 2. 旋转感知编码器 (OrientationExtractor 已从顶部导入)
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
        area_weights = torch.exp2(-depths_clamped.float() * 2.0)  # D4-AUDIT FIX: 4.**x → exp2(x*2)

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

    @property
    @torch._dynamo.disable  # 🌟 修复：禁止 Dynamo 追踪此属性
    def embed_output(self) -> Dict[str, Any]:
        """MultiLayerGeometryField 诊断输出

        命名空间:
            embed/params/*: 每层的几何缩放参数

        I-OOM FIX: 使用 Disable & Flush 模式
        """

        output: Dict[str, Any] = {}

        # embed/params/* - 层缩放参数
        if hasattr(self, 'layer_scales') and self.layer_scales is not None:
            ls = self.layer_scales  # [num_layers]
            for d in range(ls.numel()):
                output[f"params/layer_scale_lvl_{d}"] = float(ls[d].item())
            output["params/layer_scale_mean"] = float(ls.mean().item())
            output["params/layer_scale_std"] = float(ls.std().item())

        return output
