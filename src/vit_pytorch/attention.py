# -*- coding: utf-8 -*-
"""Hilbert-aware multi-scale attention module.

This module implements HilbertAwareMultiScaleAttention, which encodes
hierarchical depth and Hilbert path relationships to modulate attention weights.

Supports three bias computation modes:
- 'original': Full neural network computation (high memory, accurate)
- 'low_rank': Low-rank factorization (recommended, memory efficient)
- 'hierarchical': Layer-wise computation (interpretable, moderate memory)
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .constants import HILBERT_BIAS_SCALE, LEVEL_BIAS_SCALE
from .utils import extract_depths

BiasMode = Literal['original', 'low_rank', 'hierarchical']


class LowRankHilbertBias(nn.Module):
    """低秩分解的 Hilbert Bias 实现。
    
    使用两个独立的路径编码器，将 O(S²) 的偏置矩阵分解为 O(S×r) 的低秩形式。
    
    数学原理:
        B[i,j] = φ(path_i)^T · ψ(path_j)
        其中 φ, ψ: R^d → R^r 是可学习的编码器
    
    复杂度:
        计算: O(S·r·H) vs 原始 O(S²·64·H)
        显存: O(S·r·H) vs 原始 O(S²·H)
    
    注意：实际的 levels_info 路径维度取决于图像大小和 max_level 的组合，
    forward 时会对输入进行动态截断或填充以匹配模型的 path_dim。
    """
    
    def __init__(self, path_dim: int, rank: int, heads: int) -> None:
        """初始化低秩 Hilbert Bias。
        
        Args:
            path_dim: 路径维度（建议设置足够大，如 max_level + 16）
            rank: 秩参数，控制近似精度（推荐 32-64）
            heads: 注意力头数
        """
        super().__init__()
        self.rank = rank
        self.heads = heads
        # 保存 path_dim 用于动态调整输入
        self.path_dim = path_dim
        
        # Query 路径编码器
        self.path_encoder_q = nn.Sequential(
            nn.Linear(path_dim, 64),
            nn.ReLU(),
            nn.Linear(64, rank * heads),
        )
        
        # Key 路径编码器
        self.path_encoder_k = nn.Sequential(
            nn.Linear(path_dim, 64),
            nn.ReLU(),
            nn.Linear(64, rank * heads),
        )
    
    def _adjust_path_dim(self, paths: torch.Tensor) -> torch.Tensor:
        """调整路径维度以匹配模型期望的 path_dim。
        
        Args:
            paths: 输入路径 (..., actual_path_len)
            
        Returns:
            调整后的路径 (..., path_dim)
        """
        actual_dim = paths.shape[-1]
        if actual_dim == self.path_dim:
            return paths
        elif actual_dim < self.path_dim:
            # 填充零
            padding = torch.zeros(*paths.shape[:-1], self.path_dim - actual_dim, 
                                  device=paths.device, dtype=paths.dtype)
            return torch.cat([paths, padding], dim=-1)
        else:
            # 截断（保留前 path_dim 个元素）
            return paths[..., :self.path_dim]
    
    def forward(self, levels_info: torch.Tensor) -> Optional[torch.Tensor]:
        """计算低秩 Hilbert Bias。
        
        Args:
            levels_info: (B, S, Info) 层级信息
            
        Returns:
            (B, H, S, S) 偏置矩阵，若无效则返回 None
        """
        if levels_info.numel() == 0:
            return None
        
        if levels_info.dim() == 2:
            # 旧格式: (S, Info)
            if levels_info.shape[1] <= 1:
                return None
            paths = levels_info[:, 1:].float()  # (S, Path)
            # 调整路径维度以匹配模型
            paths = self._adjust_path_dim(paths)
            
            # 编码路径
            phi = self.path_encoder_q(paths)  # (S, rank*H)
            psi = self.path_encoder_k(paths)  # (S, rank*H)
            
            phi = phi.view(-1, self.heads, self.rank)  # (S, H, r)
            psi = psi.view(-1, self.heads, self.rank)  # (S, H, r)
            
            # 低秩矩阵乘法: bias[i,j] = φ[i] · ψ[j]^T
            bias = torch.einsum('ihr,jhr->hij', phi, psi)  # (H, S, S)
            return bias
        else:
            # 新格式: (B, S, Info)
            batch_size, seq_len, info_dim = levels_info.shape
            if info_dim <= 1:
                return None
            
            paths = levels_info[:, :, 1:].float()  # (B, S, Path)
            # 调整路径维度以匹配模型
            paths = self._adjust_path_dim(paths)
            
            # 编码路径
            phi = self.path_encoder_q(paths)  # (B, S, rank*H)
            psi = self.path_encoder_k(paths)  # (B, S, rank*H)
            
            phi = phi.view(batch_size, seq_len, self.heads, self.rank)  # (B, S, H, r)
            psi = psi.view(batch_size, seq_len, self.heads, self.rank)  # (B, S, H, r)
            
            # 低秩矩阵乘法
            bias = torch.einsum('bihr,bjhr->bhij', phi, psi)  # (B, H, S, S)
            return bias


class HierarchicalHilbertBias(nn.Module):
    """分层计算的 Hilbert Bias 实现。
    
    利用四叉树的层级结构，将偏置分解为各层的贡献之和。
    
    数学原理:
        B[i,j] = Σ_{ℓ=1}^L b^(ℓ)(q_i^(ℓ), q_j^(ℓ), context)
        其中 q^(ℓ) 是第 ℓ 层的象限索引
    
    优势:
        - 可解释性强（可视化各层贡献）
        - 参数共享（泛化能力好）
        - 可并行计算各层
    """
    
    def __init__(self, max_depth: int, heads: int) -> None:
        """初始化分层 Hilbert Bias。
        
        Args:
            max_depth: 最大四叉树深度
            heads: 注意力头数
        """
        super().__init__()
        self.max_depth = max_depth
        self.heads = heads
        
        # 每层独立的偏置网络
        # 输入特征: [same_quad, quad_diff, q_i, q_j] (4维)
        self.layer_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(4, 16),
                nn.ReLU(),
                nn.Linear(16, heads),
            )
            for _ in range(max_depth)
        ])
    
    def forward(self, levels_info: torch.Tensor) -> Optional[torch.Tensor]:
        """计算分层 Hilbert Bias。
        
        Args:
            levels_info: (B, S, Info) 层级信息
            
        Returns:
            (B, H, S, S) 偏置矩阵，若无效则返回 None
        """
        if levels_info.numel() == 0:
            return None
        
        if levels_info.dim() == 2:
            # 旧格式: (S, Info)
            if levels_info.shape[1] <= 1:
                return None
            paths = levels_info[:, 1:].long()  # (S, Path)
            seq_len, path_len = paths.shape
            
            bias = torch.zeros(self.heads, seq_len, seq_len, device=paths.device)
            
            # 逐层累加偏置
            for level in range(min(path_len, len(self.layer_nets))):
                q = paths[:, level]  # (S,)
                
                # 计算特征
                same_quad = (q.unsqueeze(0) == q.unsqueeze(1)).float()  # (S, S)
                quad_diff = (q.unsqueeze(0) - q.unsqueeze(1)).abs().float()  # (S, S)
                q_i = q.unsqueeze(1).expand(seq_len, seq_len).float()  # (S, S)
                q_j = q.unsqueeze(0).expand(seq_len, seq_len).float()  # (S, S)
                
                context = torch.stack([same_quad, quad_diff, q_i, q_j], dim=-1)  # (S, S, 4)
                
                # 通过第 level 层网络
                layer_bias = self.layer_nets[level](context)  # (S, S, H)
                bias += layer_bias.permute(2, 0, 1)  # (H, S, S)
            
            return bias
        else:
            # 新格式: (B, S, Info)
            batch_size, seq_len, info_dim = levels_info.shape
            if info_dim <= 1:
                return None
            
            paths = levels_info[:, :, 1:].long()  # (B, S, Path)
            path_len = paths.shape[-1]
            
            bias = torch.zeros(batch_size, self.heads, seq_len, seq_len, device=paths.device)
            
            # 逐层累加偏置
            for level in range(min(path_len, len(self.layer_nets))):
                q = paths[:, :, level]  # (B, S)
                
                # 计算特征 (向量化)
                same_quad = (q.unsqueeze(2) == q.unsqueeze(1)).float()  # (B, S, S)
                quad_diff = (q.unsqueeze(2) - q.unsqueeze(1)).abs().float()  # (B, S, S)
                q_i = q.unsqueeze(2).expand(batch_size, seq_len, seq_len).float()  # (B, S, S)
                q_j = q.unsqueeze(1).expand(batch_size, seq_len, seq_len).float()  # (B, S, S)
                
                context = torch.stack([same_quad, quad_diff, q_i, q_j], dim=-1)  # (B, S, S, 4)
                
                # 通过第 level 层网络
                layer_bias = self.layer_nets[level](context)  # (B, S, S, H)
                bias += layer_bias.permute(0, 3, 1, 2)  # (B, H, S, S)
            
            return bias


class HilbertAwareMultiScaleAttention(nn.Module):
    """Hilbert 曲线感知的多尺度注意力机制。

    通过编码层级深度和 Hilbert 路径关系来调制注意力权重。
    
    Attributes:
        heads: 注意力头数
        dim_head: 每个头的维度
        max_level: 最大层级
        use_hilbert_bias: 是否使用 Hilbert 偏置
        use_level_scaling: 是否使用层级缩放
        scale: 注意力缩放因子
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        max_level: int = 50,
        use_hilbert_bias: bool = True,
        use_level_scaling: bool = True,
        bias_mode: BiasMode = 'low_rank',
        low_rank_r: int = 32,
    ) -> None:
        """初始化 HilbertAwareMultiScaleAttention。
        
        Args:
            dim: 输入维度
            heads: 注意力头数
            dim_head: 每个头的维度
            dropout: Dropout 比率
            max_level: 最大层级
            use_hilbert_bias: 是否使用 Hilbert 路径偏置
            use_level_scaling: 是否使用层级缩放
            bias_mode: Hilbert Bias 计算模式
                - 'original': 原始全连接网络（高显存，精确）
                - 'low_rank': 低秩分解（推荐，显存友好）
                - 'hierarchical': 分层计算（可解释性强）
            low_rank_r: 低秩分解的秩参数（仅当 bias_mode='low_rank' 时有效）
        """
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.max_level = max_level
        self.use_hilbert_bias = use_hilbert_bias
        self.use_level_scaling = use_level_scaling
        self.bias_mode = bias_mode

        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # 根据 bias_mode 初始化对应的实现
        if use_hilbert_bias:
            if bias_mode == 'original':
                self.hilbert_bias_network: Optional[nn.Sequential] = nn.Sequential(
                    nn.Linear(2, 64),
                    nn.ReLU(),
                    nn.Linear(64, heads),
                    nn.Tanh(),
                )
                self.hilbert_bias_impl: Optional[nn.Module] = None
            elif bias_mode == 'low_rank':
                self.hilbert_bias_network = None
                # 路径维度需要足够大以容纳实际的 levels_info
                # 实际路径维度 = max_info_len - 1 ≈ max_level + log2(image_size/min_patch) + 3
                # 使用 max_level + 16 作为安全的默认值
                estimated_path_dim = max_level + 16
                self.hilbert_bias_impl = LowRankHilbertBias(
                    path_dim=estimated_path_dim,
                    rank=low_rank_r,
                    heads=heads,
                )
            elif bias_mode == 'hierarchical':
                self.hilbert_bias_network = None
                self.hilbert_bias_impl = HierarchicalHilbertBias(
                    max_depth=max_level,
                    heads=heads,
                )
            else:
                raise ValueError(f"Unknown bias_mode: {bias_mode}")
        else:
            self.hilbert_bias_network = None
            self.hilbert_bias_impl = None

        if use_level_scaling:
            self.level_scale_embedding: Optional[nn.Embedding] = nn.Embedding(max_level + 1, heads)
            nn.init.constant_(self.level_scale_embedding.weight, 1.0)
            nn.init.normal_(self.level_scale_embedding.weight, std=0.1)
        else:
            self.level_scale_embedding = None

        self.scale_weights = nn.Parameter(torch.ones(heads))
        self.relative_pos_embedding = nn.Embedding(2 * max_level + 1, heads)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def _compute_hilbert_bias(self, levels_info: torch.Tensor) -> Optional[torch.Tensor]:
        """计算基于 Hilbert 路径的注意力偏置。
        
        根据 bias_mode 调用不同的实现：
        - 'original': 使用原始的全连接网络
        - 'low_rank': 使用低秩分解
        - 'hierarchical': 使用分层计算
        
        Args:
            levels_info: 层级信息张量，形状为 (Seq, Info) 或 (Batch, Seq, Info)
            
        Returns:
            Hilbert 偏置张量，形状为 (H, S, S) 或 (B, H, S, S)，若无效则返回 None
        """
        if not self.use_hilbert_bias or levels_info.numel() == 0:
            return None

        # 使用新的优化实现（low_rank 或 hierarchical）
        if self.hilbert_bias_impl is not None:
            return self.hilbert_bias_impl(levels_info)
        
        # 原始实现（original mode）
        assert self.hilbert_bias_network is not None

        device = levels_info.device

        if levels_info.dim() == 2:
            # Old behavior: (Seq, Info)
            seq_len = levels_info.shape[0]
            if levels_info.shape[1] <= 1:
                return None
            paths = levels_info[:, 1:].float() # (S, Path)
            path_i = paths.unsqueeze(1) # (S, 1, Path)
            path_j = paths.unsqueeze(0) # (1, S, Path)
            
            path_dist = torch.norm(path_i - path_j, dim=2)
            path_sim = F.cosine_similarity(path_i, path_j, dim=2, eps=1e-6)
            path_features = torch.stack([path_dist, path_sim], dim=-1) # (S, S, 2)
            
            bias = self.hilbert_bias_network(path_features) # (S, S, H)
            return bias.permute(2, 0, 1) # (H, S, S)
        else:
            # New behavior: (Batch, Seq, Info)
            batch_size, seq_len, info_dim = levels_info.shape
            if info_dim <= 1:
                return None
            
            paths = levels_info[:, :, 1:].float() # (B, S, Path)
            path_i = paths.unsqueeze(2) # (B, S, 1, Path)
            path_j = paths.unsqueeze(1) # (B, 1, S, Path)
            
            path_dist = torch.norm(path_i - path_j, dim=3) # (B, S, S)
            path_sim = F.cosine_similarity(path_i, path_j, dim=3, eps=1e-6) # (B, S, S)
            
            path_features = torch.stack([path_dist, path_sim], dim=-1) # (B, S, S, 2)
            
            bias = self.hilbert_bias_network(path_features) # (B, S, S, H)
            return bias.permute(0, 3, 1, 2) # (B, H, S, S)

    def _compute_level_bias(self, levels_info: torch.Tensor) -> Optional[torch.Tensor]:
        """计算基于层级差异的相对位置偏置。
        
        Args:
            levels_info: 层级信息张量，形状为 (Seq, Info) 或 (Batch, Seq, Info)
            
        Returns:
            层级偏置张量，形状为 (H, S, S) 或 (B, H, S, S)，若无效则返回 None
        """
        if levels_info.numel() == 0:
            return None

        if levels_info.dim() == 2:
            # Old behavior: (Seq, Info)
            depths = extract_depths(levels_info, self.max_level)
            level_diff = depths.unsqueeze(0) - depths.unsqueeze(1)
            level_diff = level_diff.clamp(-self.max_level, self.max_level) + self.max_level
            rel_pos_bias = self.relative_pos_embedding(level_diff)
            return rel_pos_bias.permute(2, 0, 1) # (H, S, S)
        else:
            # New behavior: (Batch, Seq, Info)
            depths = extract_depths(levels_info, self.max_level) # (B, S)
            level_diff = depths.unsqueeze(2) - depths.unsqueeze(1) # (B, S, S)
            level_diff = level_diff.clamp(-self.max_level, self.max_level) + self.max_level
            rel_pos_bias = self.relative_pos_embedding(level_diff) # (B, S, S, H)
            return rel_pos_bias.permute(0, 3, 1, 2) # (B, H, S, S)

    def forward(
        self,
        x: torch.Tensor,
        levels_info: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """前向传播。
        
        Args:
            x: 输入张量，形状为 [B, N, D]
            levels_info: 层级信息（可选）
            attention_mask: 注意力掩码（可选）
            
        Returns:
            输出张量，形状为 [B, N, D]
        """
        batch, _, _ = x.shape

        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        dots = dots * self.scale_weights.view(1, -1, 1, 1)

        if self.use_level_scaling and levels_info is not None and levels_info.numel() > 0:
            # Type guard: guaranteed non-None when use_level_scaling is True
            assert self.level_scale_embedding is not None
            
            depths = extract_depths(levels_info, self.max_level)
            if levels_info.dim() == 2:
                level_scales = self.level_scale_embedding(depths)
                level_scales = level_scales.transpose(0, 1).unsqueeze(0).unsqueeze(-1)
            else:
                level_scales = self.level_scale_embedding(depths) # (B, S, H)
                level_scales = level_scales.permute(0, 2, 1).unsqueeze(-1) # (B, H, S, 1)
            
            dots = dots * level_scales

        if levels_info is not None:
            hilbert_bias = self._compute_hilbert_bias(levels_info)
            if hilbert_bias is not None:
                # hilbert_bias: (H, S, S) or (B, H, S, S)
                if hilbert_bias.dim() == 3:
                    dots = dots + hilbert_bias.unsqueeze(0) * HILBERT_BIAS_SCALE
                else:
                    dots = dots + hilbert_bias * HILBERT_BIAS_SCALE

            level_bias = self._compute_level_bias(levels_info)
            if level_bias is not None:
                # level_bias: (H, S, S) or (B, H, S, S)
                if level_bias.dim() == 3:
                    dots = dots + level_bias.unsqueeze(0) * LEVEL_BIAS_SCALE
                else:
                    dots = dots + level_bias * LEVEL_BIAS_SCALE

        if attention_mask is not None:
            mask_value = -torch.finfo(dots.dtype).max
            dots.masked_fill_(~attention_mask.bool(), mask_value)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)
