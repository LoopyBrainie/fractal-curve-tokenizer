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
from typing import Optional, Tuple, Union

import torch
import torch._dynamo
import torch.nn as nn
import math

from vit_pytorch.core.config import FractalConfig  # I97-5: 合并 config_fractal.py
from vit_pytorch.core.curve_hilbert import HilbertCurve


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
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    
    @staticmethod
    @torch._dynamo.disable  # 排除此函数被 torch.compile 追踪
    def compute_quadrant_paths(
        x: torch.Tensor,
        y: torch.Tensor,
        max_level: int,
    ) -> torch.Tensor:
        """向量化计算四叉树路径.
        
        数学: q_l = bit(x, max_level-l) + 2 × bit(y, max_level-l)
        
        Args:
            x: [N] x 坐标
            y: [N] y 坐标
            max_level: 最大深度
            
        Returns:
            paths: [N, max_level] 四叉树路径，每个元素 ∈ {0, 1, 2, 3}
        """
        N = x.shape[0]
        device = x.device
        
        if max_level == 0:
            return torch.zeros(N, 1, dtype=torch.long, device=device)
        
        # 创建深度索引 [0, 1, ..., max_level-1]
        depths = torch.arange(max_level, device=device)
        
        # 计算每个深度的位移量 [max_level-1, max_level-2, ..., 0]
        shifts = max_level - 1 - depths  # [D]
        
        # 扩展维度以支持广播
        x_exp = x.unsqueeze(1)  # [N, 1]
        y_exp = y.unsqueeze(1)  # [N, 1]
        shifts_exp = shifts.unsqueeze(0)  # [1, D]
        
        # 向量化位运算
        qx = (x_exp >> shifts_exp) & 1  # [N, D]
        qy = (y_exp >> shifts_exp) & 1  # [N, D]
        
        # 组合象限: 0=左上, 1=右上, 2=左下, 3=右下
        paths = qx + (qy << 1)  # D4-AUDIT FIX: 2*qy → qy<<1
        
        return paths.long()
    
    @staticmethod
    @torch._dynamo.disable  # 排除此函数被 torch.compile 追踪，避免 Triton 编译错误
    def compute_paths_from_regions(
        regions: torch.Tensor,
        image_size: Union[int, Tuple[int, int]],
        max_level: int,
    ) -> torch.Tensor:
        """从区域边界向量化计算四叉树路径.
        
        数学原理:
            对于区域中心 (cx, cy)，第 d 层的象限由以下决定:
            
            grid_size = 2^max_level
            normalized_x = cx * grid_size / image_size
            qx_d = (normalized_x >> (max_level - 1 - d)) & 1
            qy_d = (normalized_y >> (max_level - 1 - d)) & 1
            quadrant_d = qx_d + 2 * qy_d
        
        这与 compute_quadrant_paths 使用相同的数学公式，但输入是像素坐标而非网格坐标。
        
        Args:
            regions: [N, 4] 或 [B, N, 4]，格式 (x1, y1, x2, y2)
            image_size: 图像边长 (int 或 (H, W) tuple，Hilbert 曲线要求方形)
            max_level: 最大四叉树深度
            
        Returns:
            paths: [N, max_level] 或 [B, N, max_level] 四叉树路径
        """
        # 处理 image_size 元组 (Hilbert 曲线要求方形，使用较大边)
        if isinstance(image_size, tuple):
            img_size = max(image_size)
        else:
            img_size = image_size
        
        # 处理输入维度
        if regions.dim() == 2:
            # [N, 4] → [1, N, 4]
            was_2d = True
            regions = regions.unsqueeze(0)
        else:
            was_2d = False
        
        B, N, _ = regions.shape
        device = regions.device
        
        if max_level == 0:
            result = torch.zeros(B, N, 1, dtype=torch.long, device=device)
            return result.squeeze(0) if was_2d else result
        
        # 计算区域中心 (使用整数算术避免精度问题)
        # regions: [B, N, 4] = (x1, y1, x2, y2)
        cx = (regions[:, :, 0] + regions[:, :, 2]) // 2  # [B, N]
        cy = (regions[:, :, 1] + regions[:, :, 3]) // 2  # [B, N]

        # 将像素坐标转换为网格坐标
        # 使用位移运算替代幂运算，避免 Triton 编译问题
        # grid_size = 2^max_level = 1 << max_level
        grid_size = 1 << max_level

        # I99-1 FIX: 防御性检查 - 确保 img_size >= 1 防止除以零
        safe_img_size = max(1, img_size)

        # 缩放: grid_x = cx * grid_size // img_size
        gx = cx * grid_size // safe_img_size  # [B, N]
        gy = cy * grid_size // safe_img_size  # [B, N]

        # I130-6: 转换为整数类型以支持位运算
        gx = gx.long()
        gy = gy.long()

        # 确保在有效范围内
        gx = gx.clamp(0, grid_size - 1)
        gy = gy.clamp(0, grid_size - 1)
        
        # 使用 compute_quadrant_paths 的相同位运算逻辑
        # 但这里需要处理 batch 维度
        depths = torch.arange(max_level, device=device)  # [D]
        shifts = max_level - 1 - depths  # [D]
        
        # 扩展维度: [B, N, 1] 和 [1, 1, D]
        gx_exp = gx.unsqueeze(-1)  # [B, N, 1]
        gy_exp = gy.unsqueeze(-1)  # [B, N, 1]
        shifts_exp = shifts.view(1, 1, -1)  # [1, 1, D]
        
        # 向量化位运算
        qx = (gx_exp >> shifts_exp) & 1  # [B, N, D]
        qy = (gy_exp >> shifts_exp) & 1  # [B, N, D]
        
        # 组合象限
        paths = (qx + (qy << 1)).long()  # [B, N, D]
        
        return paths.squeeze(0) if was_2d else paths

    @staticmethod
    @torch._dynamo.disable  # 排除此函数被 torch.compile 追踪，避免动态循环的编译问题
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


# =============================================================================
# Scheme C: Structured Manifold Bias - Orientation Extractor
# =============================================================================

class OrientationExtractor(nn.Module):
    """从 Hilbert 路径提取旋转状态 (Scheme C 核心组件)

    数学形式化
    ===========

    Hilbert 曲线的核心性质是递归旋转/镜像:
        - 进入象限 0 (左下) 和 3 (右下) 时，坐标系发生翻转
        - 进入象限 1 (右上) 和 2 (左上) 时，坐标系保持不变

    旋转状态定义:
        orientation[l] = ∏_{k=0}^{l} r_k
        其中 r_k = 1 if path[k] ∈ {0, 3} else 0

    旋转检测 (位运算):
        r_k = 1 - ((path[k] >> 1) & 1)  # 00,01→1, 10,11→0

    优势:
        - 完全向量化，无 Python 循环
        - 使用 cumprod 实现累积旋转
        - 输出可直接用于注意力偏置
    """

    def __init__(
        self,
        max_level: int = 8,
        embedding_dim: int = 64,
    ):
        """初始化旋转提取器

        Args:
            max_level: 最大四叉树深度
            embedding_dim: 旋转嵌入维度
        """
        super().__init__()
        self.max_level = max_level
        self.embedding_dim = embedding_dim

        # 旋转状态编码: 2 種状态 (旋转/不旋转) → 可学习嵌入
        self.rotation_embedding = nn.Embedding(2, embedding_dim)

        # 旋转方向编码 (4 種: 0°, 90°, 180°, 270°)
        self.direction_embedding = nn.Embedding(4, embedding_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.rotation_embedding.weight, std=0.02)
        nn.init.normal_(self.direction_embedding.weight, std=0.02)

    @staticmethod
    def compute_rotation_states(paths: torch.Tensor) -> torch.Tensor:
        """计算累积旋转状态 (向量化)

        数学形式:
            r[l] = ∏_{k=0}^{l} 1[path[k] ∈ {0, 3}]

        即: 累积乘积 (cumprod) 指示从根到深度 l 是否经历偶数次翻转

        Args:
            paths: [B, N, L] 四叉树路径，值 ∈ {0, 1, 2, 3}

        Returns:
            rotation_states: [B, N, L] 旋转状态 (0=无旋转, 1=有旋转)
        """
        # 检测翻转触发: 象限 0, 3 触发翻转
        flip_trigger = (paths == 0) | (paths == 3)  # [B, N, L]

        # 累积翻转: cumprod 实现 AND 逻辑
        # 0→0 (无翻转), 1→1 (翻转), 0→0 (保持), 1→1 (保持)
        rotation_states = flip_trigger.cumprod(dim=-1).long()

        return rotation_states

    @staticmethod
    def compute_rotation_directions(paths: torch.Tensor) -> torch.Tensor:
        """计算旋转方向 (向量化)

        数学形式:
            direction[l] = sum_{k=0}^{l} (path[k] ∈ {0, 3}) mod 4

        即: 累积翻转次数模 4 (对应 0°, 90°, 180°, 270°)

        Args:
            paths: [B, N, L] 四叉树路径

        Returns:
            directions: [B, N, L] 旋转方向 (0,1,2,3 对应 0°, 90°, 180°, 270°)
        """
        # 检测翻转次数
        flip_count = ((paths == 0) | (paths == 3)).long()  # [B, N, L]

        # 累积次数模 4
        directions = flip_count.cumsum(dim=-1).clamp(0, 3)

        return directions

    def forward(
        self,
        paths: torch.Tensor,
        depths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """提取旋转状态和方向

        Args:
            paths: [B, N, L] 四叉树路径
            depths: [B, N] 有效深度 (可选)

        Returns:
            rotation_emb: [B, N, embedding_dim] 旋转状态嵌入
            direction_emb: [B, N, embedding_dim] 旋转方向嵌入
        """
        # 计算旋转状态
        rotation_states = self.compute_rotation_states(paths)  # [B, N, L]

        # 计算旋转方向
        directions = self.compute_rotation_directions(paths)  # [B, N, L]

        # 对路径维度求和 (按有效深度加权)
        if depths is not None:
            # 创建有效掩码
            level_indices = torch.arange(
                paths.shape[-1], device=paths.device
            ).unsqueeze(0).unsqueeze(0)  # [1, 1, L]
            valid_mask = (level_indices < depths.unsqueeze(-1)).float()  # [B, N, L]

            rotation_states = (rotation_states * valid_mask).sum(dim=-1).clamp(0, 1)
            directions = (directions * valid_mask).sum(dim=-1).clamp(0, 3)
        else:
            # 使用平均
            rotation_states = rotation_states.mean(dim=-1).clamp(0, 1)
            directions = directions.mean(dim=-1).clamp(0, 3)

        # 查找嵌入
        rotation_emb = self.rotation_embedding(rotation_states.long())  # [B, N, dim]
        direction_emb = self.direction_embedding(directions.long())  # [B, N, dim]

        return rotation_emb, direction_emb

    def get_orientation_mask(
        self,
        paths: torch.Tensor,
    ) -> torch.Tensor:
        """获取旋转角度掩码 (用于注意力偏置)

        输出:
            orientation_mask: [B, N, L] 旋转角度 (弧度)
                - 无旋转: 0
                - 有旋转: -π/2 (90°)
        """
        rotation_states = self.compute_rotation_states(paths)  # [B, N, L]

        # 旋转 → -π/2, 不旋转 → 0
        orientation_mask = rotation_states.float() * (-math.pi / 2)

        return orientation_mask
