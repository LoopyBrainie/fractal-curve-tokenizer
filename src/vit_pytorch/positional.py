from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class AdvancedFractalPositionEmbedding(nn.Module):
    """
    高级分形位置编码，完全对齐增强tokenizer
    支持动态层级、Hilbert路径编码和多尺度空间感知
    """

    def __init__(
        self,
        dim: int,
        max_level: int = 50,
        max_seq_len: int = 10000,
        use_hilbert_encoding: bool = True,
        use_spatial_encoding: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.max_level = max_level
        self.max_seq_len = max_seq_len
        self.use_hilbert_encoding = use_hilbert_encoding
        self.use_spatial_encoding = use_spatial_encoding

        # 1. 深度编码 (Depth Embedding)
        self.depth_embedding = nn.Embedding(max_level + 1, dim)

        # 2. 层级路径编码 (Hierarchical Path Embedding)
        # 替代原有的 LSTM 和 2D 绝对位置编码
        # 每个层级有 4 个象限 (0, 1, 2, 3)
        # 总共 max_level * 4 个唯一的层级-象限组合
        self.quadrant_embedding = nn.Embedding(max_level * 4, dim)

        # 3. 融合网络 (简化版)
        # 输入: Depth Emb + Path Emb
        self.fusion_network = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.level_attention_bias = nn.Parameter(torch.zeros(max_level + 1, max_level + 1))

        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.normal_(self.depth_embedding.weight, std=0.02)
        nn.init.normal_(self.quadrant_embedding.weight, std=0.02)
        nn.init.uniform_(self.level_attention_bias, -0.1, 0.1)

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
        
        # 应用掩码并求和: (..., dim)
        # 这实现了 "Level 1 Emb + Level 2 Emb + ..." 的逻辑
        path_final = (path_embs * mask.unsqueeze(-1)).sum(dim=-2)
        
        # 3. 融合
        # 直接相加，保留层级和位置信息
        combined_emb = depth_emb + path_final
        
        return self.fusion_network(combined_emb)

    def get_attention_bias(self, depths: torch.Tensor) -> torch.Tensor:
        """
        计算基于层级的注意力偏置矩阵
        
        Args:
            depths: (N,) 每个 token 的深度值
            
        Returns:
            (N, N) 注意力偏置矩阵
        """
        # 向量化实现，避免双重循环
        depths_clamped = depths.clamp(0, self.max_level).long()  # (N,)
        
        # 使用高级索引一次性获取所有偏置
        # bias[i, j] = level_attention_bias[depths[i], depths[j]]
        bias = self.level_attention_bias[depths_clamped.unsqueeze(1), depths_clamped.unsqueeze(0)]
        
        return bias
