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

from typing import Optional

import torch
import torch.nn as nn

from .constants import EMBEDDING_INIT_STD, HILBERT_BIAS_SCALE


class FractalPositionEmbedding(nn.Module):
    """
    高级分形位置编码，完全对齐增强tokenizer
    支持动态层级、Hilbert路径编码和多尺度空间感知
    
    P11-2 修复: max_level 参数现在应传入与 tokenizer.max_depth 一致的值，
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
        max_level: int = 8,  # P11-2: 默认改为 8，应由上层传入实际 max_depth
        max_seq_len: int = 10000,
        use_hilbert_encoding: bool = True,
        use_spatial_encoding: bool = True,
        # I27-2: 可配置 Dropout (默认 0.1，约为 transformer dropout 的一半)
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.max_level = max_level
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
        levels_info: torch.Tensor,
        sequence_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if levels_info.numel() == 0:
            return torch.zeros(0, self.dim, device=levels_info.device, dtype=torch.float32)

        device = levels_info.device
        
        # levels_info: (..., max_info_len)
        # col 0: depth
        # col 1..: path indices (0-3)
        
        depths = levels_info[..., 0].clamp(0, self.max_level).long()
        paths = levels_info[..., 1:].long() # (..., path_len)
        
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
        
        # 查找 Embedding: (..., path_len, dim)
        path_embs = self.quadrant_embedding(flat_indices)
        
        # 创建掩码: 只保留有效层级的路径节点
        # mask[i, j] = 1 if j < depths[i] else 0
        seq_indices = torch.arange(path_len, device=device)
        mask = seq_indices < depths.unsqueeze(-1) # (..., path_len)
        
        # STAB-4 修复: 按深度归一化，防止深层 token 的 ||E_path|| ∝ √d 导致范数失衡
        # 原公式: E_path = Σ E_j → ||E_path|| ∝ √d (深层 token 范数过大)
        # 修复后: E_path = (Σ E_j) / √d → ||E_path|| ≈ const (范数一致)
        path_count = mask.sum(dim=-1, keepdim=True).clamp(min=1).float()
        path_final = (path_embs * mask.unsqueeze(-1)).sum(dim=-2) / torch.sqrt(path_count)
        
        # 3. 融合
        # 直接相加，保留层级和位置信息
        combined_emb = depth_emb + path_final
        
        return self.fusion_network(combined_emb)

    # P11-5: 删除了 get_attention_bias 方法
    # 注意力偏置功能已由 LCAHilbertBias (attn_hilbert_bias.py) 统一提供
    # 该方法基于 LCA 深度计算偏置，语义更精确 (编码空间距离而非尺度组合)
