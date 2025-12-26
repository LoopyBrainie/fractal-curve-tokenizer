# -*- coding: utf-8 -*-
"""
向量化四叉树路径编码模块

数学形式化
============

四叉树路径编码将 2D 坐标转换为递归分割路径:

    (x, y) → (q₁, q₂, ..., q_d)

其中 q_l ∈ {0, 1, 2, 3} 表示第 l 层的象限:
    0 = 左上, 1 = 右上, 2 = 左下, 3 = 右下

计算公式 (向量化位运算):
    q_l = bit(x, d-l) + 2 × bit(y, d-l)
    
其中 bit(v, k) = (v >> k) & 1 是第 k 位

复杂度:
    - 传统 Python 循环: O(N × D) 串行
    - 向量化位运算: O(D) 并行，GPU 利用率 >90%

层级关系:
    共同祖先深度 = 最长公共前缀长度
    → 用于层级注意力偏置
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional, Tuple

import torch
import torch._dynamo
import torch.nn as nn

from .config_fractal import FractalConfig
from .curve_hilbert import HilbertCurve


# 创建兼容 torch.compile 的缓存装饰器
def _dynamo_safe_lru_cache(maxsize: int = 128):
    """LRU 缓存装饰器，兼容 torch.compile."""
    def decorator(func):
        cached = lru_cache(maxsize=maxsize)(func)
        return torch._dynamo.disable(cached)
    return decorator


class VectorizedPathEncoder:
    """向量化的四叉树路径编码器.
    
    核心优化:
    1. 预计算 Hilbert 坐标映射 (初始化时一次性)
    2. 向量化位运算计算路径 (无 Python 循环)
    3. 批量处理所有 token
    """
    
    def __init__(self, config: FractalConfig):
        """初始化路径编码器.
        
        Args:
            config: FractalConfig 实例
        """
        self.config = config
        self._coord_cache: dict = {}
    
    @_dynamo_safe_lru_cache(maxsize=16)
    def _get_hilbert_coords(self, grid_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取 Hilbert 坐标映射 (带缓存).
        
        Args:
            grid_size: 网格边长
            
        Returns:
            (x_coords, y_coords): 各 [grid_size²] 的坐标张量
        """
        # 找到最接近的 2 的幂
        n = 1
        while n < grid_size:
            n *= 2
        
        num_points = grid_size * grid_size
        x_coords = []
        y_coords = []
        
        for d in range(n * n):
            x, y = HilbertCurve.d_to_xy(n, d)
            if x < grid_size and y < grid_size:
                x_coords.append(x)
                y_coords.append(y)
                if len(x_coords) >= num_points:
                    break
        
        return (
            torch.tensor(x_coords, dtype=torch.long),
            torch.tensor(y_coords, dtype=torch.long),
        )
    
    def get_coords_on_device(
        self,
        grid_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取指定设备上的坐标张量."""
        x, y = self._get_hilbert_coords(grid_size)
        return x.to(device), y.to(device)
    
    @staticmethod
    def compute_quadrant_paths(
        x: torch.Tensor,
        y: torch.Tensor,
        max_depth: int,
    ) -> torch.Tensor:
        """向量化计算四叉树路径.
        
        数学: q_l = bit(x, max_depth-l) + 2 × bit(y, max_depth-l)
        
        Args:
            x: [N] x 坐标
            y: [N] y 坐标
            max_depth: 最大深度
            
        Returns:
            paths: [N, max_depth] 四叉树路径，每个元素 ∈ {0, 1, 2, 3}
        """
        N = x.shape[0]
        device = x.device
        
        if max_depth == 0:
            return torch.zeros(N, 1, dtype=torch.long, device=device)
        
        # 创建深度索引 [0, 1, ..., max_depth-1]
        depths = torch.arange(max_depth, device=device)
        
        # 计算每个深度的位移量 [max_depth-1, max_depth-2, ..., 0]
        shifts = max_depth - 1 - depths  # [D]
        
        # 扩展维度以支持广播
        x_exp = x.unsqueeze(1)  # [N, 1]
        y_exp = y.unsqueeze(1)  # [N, 1]
        shifts_exp = shifts.unsqueeze(0)  # [1, D]
        
        # 向量化位运算
        qx = (x_exp >> shifts_exp) & 1  # [N, D]
        qy = (y_exp >> shifts_exp) & 1  # [N, D]
        
        # 组合象限: 0=左上, 1=右上, 2=左下, 3=右下
        paths = qx + 2 * qy  # [N, D]
        
        return paths.long()
    
    @staticmethod
    def compute_common_ancestor_depth(
        paths: torch.Tensor,
        chunk_size: int = 64,
    ) -> torch.Tensor:
        """向量化计算所有 token 对的共同祖先深度.
        
        共同祖先深度 = 最长公共前缀长度
        
        支持 2D 和 3D 输入:
        - 2D: [N, D] → [N, N]
        - 3D: [B, N, D] → [B, N, N] (分块向量化，降低内存)
        
        数学定义:
            LCA(i, j) = max{k : p_i[1:k] = p_j[1:k]}
                      = sum_{d=1}^{D} prod_{k=1}^{d} 1[p_i[k] = p_j[k]]
        
        Args:
            paths: [N, D] 或 [B, N, D] 四叉树路径
            chunk_size: 3D 输入的分块大小，用于控制内存使用
                        默认 64，将峰值内存从 O(B×N²×D) 降至 O(B×chunk²×D)
            
        Returns:
            common_depth: [N, N] 或 [B, N, N] 共同祖先深度矩阵
            
        Note:
            P1-4 优化: 对 3D 输入使用分块计算，内存降低 ~16x 且速度更快
            （得益于更好的缓存局部性）
        """
        if paths.dim() == 2:
            # 2D: [N, D] → [N, N] (小规模，直接计算)
            N, D = paths.shape
            paths_i = paths.unsqueeze(1)  # [N, 1, D]
            paths_j = paths.unsqueeze(0)  # [1, N, D]
            match = (paths_i == paths_j)  # [N, N, D]
            cumulative_match = match.cumprod(dim=-1)  # [N, N, D]
            return cumulative_match.sum(dim=-1)  # [N, N]
        else:
            # 3D: [B, N, D] → [B, N, N] (P1-4 优化: 分块计算)
            B, N, D = paths.shape
            result = torch.zeros(B, N, N, dtype=torch.long, device=paths.device)
            
            for i in range(0, N, chunk_size):
                for j in range(0, N, chunk_size):
                    i_end = min(i + chunk_size, N)
                    j_end = min(j + chunk_size, N)
                    
                    # 只计算当前块，内存 O(B × chunk² × D)
                    paths_i = paths[:, i:i_end, :].unsqueeze(2)  # [B, chunk, 1, D]
                    paths_j = paths[:, j:j_end, :].unsqueeze(1)  # [B, 1, chunk, D]
                    match = (paths_i == paths_j)  # [B, chunk, chunk, D]
                    cumulative_match = match.cumprod(dim=-1)
                    result[:, i:i_end, j:j_end] = cumulative_match.sum(dim=-1)
            
            return result


class FractalPathEmbedding(nn.Module):
    """基于四叉树路径的位置编码.
    
    结合深度编码和路径编码:
        E_pos = Fusion(E_depth(scale) + E_path(x, y, depth))
    
    其中:
        E_depth: 尺度级别编码
        E_path: 四叉树路径的聚合编码
    """
    
    def __init__(
        self,
        dim: int,
        config: FractalConfig,
    ):
        """初始化路径位置编码.
        
        Args:
            dim: 嵌入维度
            config: FractalConfig 实例
        """
        super().__init__()
        self.dim = dim
        self.config = config
        self.max_depth = config.max_depth
        
        # 路径编码器
        self.path_encoder = VectorizedPathEncoder(config)
        
        # 尺度编码 (scale_idx → embedding)
        self.scale_embedding = nn.Embedding(config.num_scales, dim)
        
        # 四叉树路径编码
        # 每层 4 个象限，共 max_depth 层
        self.quadrant_embedding = nn.Embedding(4 * max(config.max_depth, 1), dim)
        
        # 融合网络
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )
        
        # 预计算坐标 (注册为 buffer 以便自动迁移设备)
        x_coords, y_coords = self.path_encoder._get_hilbert_coords(config.grid_size)
        self.register_buffer('x_coords', x_coords)
        self.register_buffer('y_coords', y_coords)
        
        # 预计算最细网格的路径
        paths = VectorizedPathEncoder.compute_quadrant_paths(
            x_coords, y_coords, config.max_depth
        )
        self.register_buffer('base_paths', paths)
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        nn.init.normal_(self.scale_embedding.weight, std=0.02)
        nn.init.normal_(self.quadrant_embedding.weight, std=0.02)
    
    def _encode_paths(
        self,
        paths: torch.Tensor,
        depths: torch.Tensor,
    ) -> torch.Tensor:
        """编码四叉树路径.
        
        Args:
            paths: [B, N, max_depth] 四叉树路径
            depths: [B, N] 每个 token 的有效深度
            
        Returns:
            path_emb: [B, N, dim]
        """
        B, N, D = paths.shape
        device = paths.device
        
        # 层级偏移: level * 4 + quadrant
        level_offsets = torch.arange(D, device=device) * 4  # [D]
        indices = paths + level_offsets  # [B, N, D]
        
        # 查找 embedding
        path_embs = self.quadrant_embedding(indices)  # [B, N, D, dim]
        
        # 掩码: 只保留有效深度的路径
        # depths[b, n] 表示 token (b, n) 的有效深度
        level_indices = torch.arange(D, device=device)  # [D]
        mask = level_indices.unsqueeze(0).unsqueeze(0) < depths.unsqueeze(-1)  # [B, N, D]
        
        # 应用掩码并求和
        path_embs = path_embs * mask.unsqueeze(-1)  # [B, N, D, dim]
        path_final = path_embs.sum(dim=2)  # [B, N, dim]
        
        return path_final
    
    def forward(
        self,
        scale_indices: torch.Tensor,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """计算位置编码.
        
        Args:
            scale_indices: [B, N] 每个 token 选择的尺度索引
            batch_size: 批次大小 (若 scale_indices 未提供 B 维度)
            
        Returns:
            pos_emb: [B, N, dim] 位置编码
        """
        if scale_indices.dim() == 1:
            scale_indices = scale_indices.unsqueeze(0)
            if batch_size is not None:
                scale_indices = scale_indices.expand(batch_size, -1)
        
        B, N = scale_indices.shape
        device = scale_indices.device
        
        # 1. 尺度编码
        scale_emb = self.scale_embedding(scale_indices)  # [B, N, dim]
        
        # 2. 路径编码
        # 获取预计算的路径并扩展到 batch
        paths = self.base_paths[:N].unsqueeze(0).expand(B, -1, -1)  # [B, N, max_depth]
        paths = paths.to(device)
        
        # 计算每个 token 的有效深度
        # scale_idx=0 (最细) → depth=max_depth
        # scale_idx=num_scales-1 (最粗) → depth=1
        depths = self.max_depth - scale_indices + 1  # [B, N]
        depths = depths.clamp(min=1, max=self.max_depth)
        
        path_emb = self._encode_paths(paths, depths)  # [B, N, dim]
        
        # 3. 融合
        combined = torch.cat([scale_emb, path_emb], dim=-1)  # [B, N, dim*2]
        return self.fusion(combined)


class HierarchicalAttentionBias(nn.Module):
    """基于四叉树层级关系的注意力偏置.
    
    核心思想: 共同祖先越近，注意力偏置越大
    
        bias(i, j) = f(common_ancestor_depth(i, j))
    
    数学性质:
        - 自反性: bias(i, i) = max_bias (共同祖先 = 自身)
        - 对称性: bias(i, j) = bias(j, i)
        - 层级性: 若 ancestor(i,j) 更近，则 bias 更大
    """
    
    def __init__(
        self,
        config: FractalConfig,
        heads: int,
    ):
        """初始化层级注意力偏置.
        
        Args:
            config: FractalConfig 实例
            heads: 注意力头数
        """
        super().__init__()
        self.config = config
        self.heads = heads
        self.max_depth = config.max_depth
        
        # 共同祖先深度 → 偏置值
        # depth 0 = 无共同祖先 (除根)
        # depth max_depth = 完全相同的路径
        self.ancestor_bias = nn.Embedding(config.max_depth + 2, heads)
        
        # 预计算共同祖先深度矩阵
        path_encoder = VectorizedPathEncoder(config)
        x, y = path_encoder._get_hilbert_coords(config.grid_size)
        paths = VectorizedPathEncoder.compute_quadrant_paths(x, y, config.max_depth)
        common_depth = VectorizedPathEncoder.compute_common_ancestor_depth(paths)
        self.register_buffer('common_depth_matrix', common_depth)
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        # 初始化: 共同祖先越近，偏置越正
        with torch.no_grad():
            for i in range(self.max_depth + 2):
                # 线性增长: depth 0 → -0.1, depth max → +0.1
                value = (i / (self.max_depth + 1) - 0.5) * 0.2
                self.ancestor_bias.weight[i].fill_(value)
    
    def forward(
        self,
        seq_len: int,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """计算层级注意力偏置.
        
        Args:
            seq_len: 序列长度
            batch_size: 批次大小
            
        Returns:
            bias: [B, H, N, N] 注意力偏置
        """
        # 截取到实际序列长度
        common_depth = self.common_depth_matrix[:seq_len, :seq_len]  # [N, N]
        
        # 查找偏置
        bias = self.ancestor_bias(common_depth)  # [N, N, H]
        
        # 调整维度并扩展 batch
        bias = bias.permute(2, 0, 1)  # [H, N, N]
        bias = bias.unsqueeze(0).expand(batch_size, -1, -1, -1)  # [B, H, N, N]
        
        return bias
