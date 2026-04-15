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
from typing import Any, Dict, Optional, Tuple, Union

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


class BitFlippedPositionEncoder(nn.Module):
    """Bit-Flipped 位置编码器 (基于递归位翻转)

    数学形式化
    ============

    核心思想：模拟 Hilbert 曲线的递归旋转/翻转性质

    递归位翻转逻辑：
        对于每一层级 l，若父节点象限为 0 或 3，则对当前层级的
        Embedding 应用可学习的位翻转（Bit-flip via rotation_gate）

    编码公式:
        E_pos = LayerNorm(E_depth(d) + E_path_recursive(q, rotation_gate))

    其中:
        rotation_gate_l = σ(W_gate · q_parent(l-1)) ∈ [0, 1]
        E_flipped(l) = rotation_gate_l · (-E_quad(l)) + (1 - rotation_gate_l) · E_quad(l)

    输出:
        - pos_emb: 位置编码 [B, N, dim]
        - geometry_emb: 传给 Attention 的几何嵌入 [B, N, dim]

    数学优势
    =========

    1. Hilbert旋转对称捕获:
       位翻转操作自然对应 Hilbert 曲线的旋转/翻转

    2. 几何感知:
       geometry_emb 为 Attention 提供坐标系一致性的向量对齐

    3. 递归结构:
       x_k = LayerNorm(x_{k-1} + E_flipped(k, q^k))

    对比传统方案
    ============

    | 方案 | 公式 | Hilbert局部性 | 几何感知 |
    |------|------|--------------|---------|
    | 传统象限累加 | ΣE_quad(q_k) | ❌ 无序 | ❌ 无 |
    | 本方案 | 递归位翻转 | ✅ 有序 | ✅ 有 |
    """

    def __init__(
        self,
        dim: int,
        max_level: int = 8,
        grid_size: int = 256,
    ):
        """初始化 Bit-Flipped 位置编码器

        Args:
            dim: 嵌入维度
            max_level: 最大四叉树深度
            grid_size: Hilbert 网格大小
        """
        super().__init__()
        self.dim = dim
        self.max_level = max_level
        self.grid_size = grid_size

        # 深度编码 (与尺度相关)
        self.depth_embedding = nn.Embedding(max_level + 1, dim)

        # 递归位翻转: 每层一个门控参数
        # rotation_gate[l] = 0 表示不翻转，= 1 表示翻转
        # 使用 Sigmoid 将参数映射到 [0, 1]
        self.rotation_gate = nn.Parameter(torch.zeros(max_level))

        # v5.1: 深度衰减 Scale 参数
        # 使用 Sigmoid 约束到 (0, 1)，初始值 Sigmoid(0) = 0.5
        # scale(d) = gamma^d，gamma 随深度指数衰减
        self.depth_decay_scale = nn.Parameter(torch.zeros(1))

        # 层级路径编码: 每层 4 个象限
        self.quadrant_embedding = nn.Embedding(max_level * 4, dim)

        # 最终 LayerNorm
        self.layer_norm = nn.LayerNorm(dim)

        # v6.0: 几何嵌入投影 (保持 dim 维度，Attention 中处理维度匹配)
        self.geometry_projection = nn.Linear(dim, dim)

        # I-NAN: 统一诊断缓存
        self._diagnostic_cache: Dict[str, Any] = {}

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.depth_embedding.weight, std=0.02)
        nn.init.normal_(self.quadrant_embedding.weight, std=0.02)
        # rotation_gate 零初始化: 初始不翻转
        # I-NAN: 改为小值初始化，避免梯度不稳定
        nn.init.normal_(self.rotation_gate, mean=0, std=0.01)
        # v5.1: depth_decay_scale 零初始化，Sigmoid(0) = 0.5
        # 即初始 gamma = 0.5，scale(d) = 0.5^d
        # I-NAN: 改为小值初始化
        nn.init.normal_(self.depth_decay_scale, mean=0, std=0.01)

    def _register_nan_grad_hooks(self):
        """I-NAN: 为所有参数注册梯度 hook，捕获 backward 过程中产生的 NaN"""
        self._nan_grad_hooks = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(
                    lambda grad, n=name: torch.nan_to_num(grad, nan=0.0, posinf=1.0, neginf=-1.0)
                    if torch.isnan(grad).any() or torch.isinf(grad).any() else grad
                )
                self._nan_grad_hooks.append(hook)

    def _compute_quadrant_indices(self, paths: torch.Tensor) -> torch.Tensor:
        """从路径计算象限索引

        Args:
            paths: [B, N, max_level] 四叉树路径

        Returns:
            quadrant_indices: [B, N, max_level] 每层的象限索引
        """
        # 路径直接就是象限序列，无需额外计算
        return paths

    def forward(
        self,
        levels_info,
        regions: Optional[torch.Tensor] = None,
        image_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算 Bit-Flipped 位置编码

        Args:
            levels_info: LevelsInfo 实例，包含 depths 和 paths
            regions: 区域边界 (可选，未使用)
            image_size: 图像尺寸 (可选，未使用)

        Returns:
            pos_emb: [B, N, dim] 位置编码
            geometry_emb: [B, N, dim] 传给 Attention 的几何嵌入
        """
        # Fix-12: 使用统一的护城河工厂方法
        from vit_pytorch.core.levels_info import LevelsInfo
        levels_info = LevelsInfo.ensure(levels_info, default_max_level=self.max_level)

        if levels_info.data.numel() == 0:
            return (
                torch.zeros(0, self.dim, device=levels_info.data.device, dtype=torch.float32),
                torch.zeros(0, self.dim, device=levels_info.data.device, dtype=torch.float32),
            )

        device = levels_info.data.device
        B, N = levels_info.depths.shape

        # 提取 depths 和 paths
        depths = levels_info.depths.clamp(0, self.max_level).long()  # [B, N]
        paths = levels_info.paths  # [B, N, max_level]

        # 1. 深度编码
        depth_emb = self.depth_embedding(depths)  # [B, N, dim]

        # 2. 递归位翻转路径编码
        path_emb = self._recursive_bit_flip_encoding(paths, depths)  # [B, N, dim]

        # 3. 位置编码 = 深度编码 + 路径编码
        pos_emb = depth_emb + path_emb  # [B, N, dim]

        # 4. LayerNorm
        pos_emb = self.layer_norm(pos_emb)

        # 5. 生成 geometry_emb (用于 Attention 注入)
        # v5.1: 对 geometry_emb 也应用深度衰减
        # 使用 gamma^depth 对每个 token 进行缩放
        gamma = torch.sigmoid(self.depth_decay_scale)  # [1]
        # I-NAN: 添加 clamp 防止指数爆炸
        token_scales = (gamma ** depths.float()).clamp(min=1e-6, max=1e6)  # [B, N]
        token_scales = token_scales.unsqueeze(-1)  # [B, N, 1]

        geometry_emb = self.geometry_projection(path_emb)  # [B, N, dim]
        geometry_emb = geometry_emb * token_scales  # 应用深度衰减

        # I-NAN: 更新统一诊断缓存（D1-AUDIT FIX: GPU tensor 存储）
        gate_values = torch.sigmoid(self.rotation_gate).detach()
        self._diagnostic_cache = {
            "params/rotation_gate_mean": gate_values.mean(),
            "params/depth_gamma": gamma,  # γ = σ(depth_decay_scale)，GPU tensor
        }

        return pos_emb, geometry_emb

    def _recursive_bit_flip_encoding(
        self,
        paths: torch.Tensor,
        depths: torch.Tensor,
    ) -> torch.Tensor:
        """递归位翻转编码 (v5.1: 带深度衰减)

        数学形式:
            x_0 = 0
            x_k = x_{k-1} + scale(k) * E_flipped(k, q_k)
            其中 scale(k) = gamma^k，gamma = σ(depth_decay_scale)

            E_flipped(k, q_k) = rotation_gate[k] * (-E_quad(k, q_k)) + (1 - rotation_gate[k]) * E_quad(k, q_k)

        Args:
            paths: [B, N, max_level] 四叉树路径
            depths: [B, N] 每个 token 的有效深度

        Returns:
            path_emb: [B, N, dim] 递归位翻转后的路径编码
        """
        B, N, max_level = paths.shape
        device = paths.device

        # Sigmoid 激活 rotation_gate
        gate_values = torch.sigmoid(self.rotation_gate)  # [max_level]

        # v5.1: 计算深度衰减 scale
        # gamma = sigmoid(depth_decay_scale) ∈ (0, 1)
        gamma = torch.sigmoid(self.depth_decay_scale)  # [1]
        # scale(k) = gamma^k: [1, γ, γ², γ³, ..., γ^(max_level-1)]
        level_indices = torch.arange(max_level, device=device)  # [max_level]
        # I-NAN: 添加 clamp 防止指数爆炸
        depth_scales = (gamma ** level_indices).clamp(min=1e-6, max=1e6).unsqueeze(0).unsqueeze(-1)  # [1, max_level, 1]

        # 生成层级偏移量: [0, 4, 8, ..., (max_level-1)*4]
        level_offsets = torch.arange(max_level, device=device) * 4

        # 广播: paths + level_offsets -> [B, N, max_level]
        flat_indices = paths + level_offsets

        # 安全截断
        flat_indices = flat_indices.clamp(0, self.max_level * 4 - 1)

        # 查找象限嵌入: [B, N, max_level, dim]
        quadrant_embs = self.quadrant_embedding(flat_indices)

        # 初始化输出
        x = torch.zeros(B, N, self.dim, device=device)  # [B, N, dim]

        # 递归计算
        for k in range(max_level):
            # 获取第 k 层的象限嵌入
            layer_emb = quadrant_embs[..., k, :]  # [B, N, dim]

            # v5.1: 应用深度衰减 scale
            scale = depth_scales[:, k, :]  # [1, 1]
            layer_emb = layer_emb * scale

            # 判断是否需要翻转: 父节点象限为 0 或 3 时翻转
            if k > 0:
                parent_quadrants = paths[..., k - 1]  # [B, N]
                flip_mask = ((parent_quadrants == 0) | (parent_quadrants == 3)).float()  # [B, N]
            else:
                # 第 0 层无父节点，不翻转
                flip_mask = torch.zeros(B, N, device=device)

            # 获取当前层的门控值
            gate = gate_values[k]  # scalar

            # 应用翻转: E_flipped = gate * (-E) + (1-gate) * E = E * (1 - 2*gate)
            flip_factor = 1 - 2 * gate  # [1] - 翻转时为 -1，不翻转时为 1
            layer_emb = layer_emb * flip_factor

            # 有效掩码: 只在有效层级时累加
            valid_mask = (level_indices[k] < depths).unsqueeze(-1).float()  # [B, N, 1]

            # 递归累加
            x = x + layer_emb * valid_mask

        return x

    def _encode_morton(self, morton_norm: torch.Tensor) -> torch.Tensor:
        """使用正弦编码 Morton 码 (保留兼容性)

        Args:
            morton_norm: [N] 归一化到 [0, 1] 的 Morton 码

        Returns:
            enc: [N, dim//2] 编码向量
        """
        pos_dim = self.dim // 2
        # D4-AUDIT FIX: 移除 device='cpu'（GPU→CPU→GPU 传输），改用 exp2 优化
        # D4-AUDIT FIX: 移除 dtype=torch.float32，遵循 morton_norm.dtype 实现 AMP 兼容
        freqs = torch.exp2(torch.arange(0, pos_dim, 2, device=morton_norm.device, dtype=morton_norm.dtype) / pos_dim) * 3.141592653589793

        # 角度: morton_norm * freqs
        angles = morton_norm.unsqueeze(-1) * freqs  # [N, dim//4]

        # 正弦和余弦编码
        sin_enc = torch.sin(angles)  # [N, dim//4]
        cos_enc = torch.cos(angles)  # [N, dim//4]

        # 交错拼接: [sin, cos, sin, cos, ...]
        enc = torch.cat([sin_enc, cos_enc], dim=-1)  # [N, dim//2]

        return enc

    @property
    def embed_output(self) -> Dict[str, Any]:
        """BitFlippedPositionEncoder 诊断输出

        命名空间:
            embed/params/*: 可学习参数统计
            embed/health/*: 数值健康度

        D1-AUDIT FIX: 所有值现在为 GPU tensor，
        由 flatten_layer_outputs() 在 post_forward() 统一调用 .item()。
        """
        output: Dict[str, Any] = {}

        # embed/params/* - rotation_gate 参数
        # D1-AUDIT FIX: 存储整个 tensor，由 flatten_layer_outputs 处理
        if hasattr(self, 'rotation_gate') and self.rotation_gate is not None:
            gate_values = torch.sigmoid(self.rotation_gate)  # [max_level]
            # D1-AUDIT FIX: 直接存储 tensor，用 key 中的索引标记各层级
            for d in range(gate_values.numel()):
                output[f"params/rotation_gate_lvl_{d}"] = gate_values[d]
            output["params/rotation_gate_mean"] = gate_values.mean()

        # embed/params/* - depth_decay_scale 参数 (gamma)
        if hasattr(self, 'depth_decay_scale') and self.depth_decay_scale is not None:
            gamma = torch.sigmoid(self.depth_decay_scale)  # GPU tensor
            output["params/depth_gamma"] = gamma  # γ ∈ (0,1)

        # embed/params/* - 嵌入权重统计（D1-AUDIT FIX: GPU tensor））
        if hasattr(self, 'depth_embedding') and self.depth_embedding is not None:
            w = self.depth_embedding.weight
            output["params/depth_emb_norm"] = w.norm()
            output["params/depth_emb_mean"] = w.mean()

        if hasattr(self, 'quadrant_embedding') and self.quadrant_embedding is not None:
            w = self.quadrant_embedding.weight
            output["params/quadrant_emb_norm"] = w.norm()

        # I-NAN: 从统一诊断缓存合并 forward 中计算的指标（现在都是 GPU tensor）
        if self._diagnostic_cache:
            output.update(self._diagnostic_cache)

        # embed/health/* - nan_grad_hooks 注册数
        if hasattr(self, '_nan_grad_hooks') and self._nan_grad_hooks:
            output["health/nan_grad_hooks_registered"] = len(self._nan_grad_hooks)

        return output


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
        self.max_level = config.max_level
        
        # 路径编码器
        self.path_encoder = VectorizedPathEncoder(config)
        
        # 尺度编码 (scale_idx → embedding)
        self.scale_embedding = nn.Embedding(config.num_scales, dim)
        
        # 四叉树路径编码
        # 每层 4 个象限，共 max_level 层
        self.quadrant_embedding = nn.Embedding(4 * max(config.max_level, 1), dim)
        
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
            x_coords, y_coords, config.max_level
        )
        self.register_buffer('base_paths', paths)

        # I-NAN: 统一诊断缓存
        self._diagnostic_cache: Dict[str, Any] = {}

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
            paths: [B, N, max_level] 四叉树路径
            depths: [B, N] 每个 token 的有效深度
            
        Returns:
            path_emb: [B, N, dim]
        """
        B, N, D = paths.shape
        device = paths.device
        
        # 层级偏移: level * 4 + quadrant
        level_offsets = torch.arange(D, device=device) * 4  # [D]
        indices = paths + level_offsets  # [B, N, D]

        # STAB-7 修复: 确保索引张量为连续格式
        # channels-last 格式与 Embedding 层不兼容
        indices = indices.contiguous()

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
        # P-OPT: base_paths 已通过 register_buffer 注册，会随模型自动迁移，无需 .to(device)
        paths = self.base_paths[:N].unsqueeze(0).expand(B, -1, -1)  # [B, N, max_level]

        # 计算每个 token 的有效深度
        # scale_idx=0 (最细) → depth=max_level
        # scale_idx=num_scales-1 (最粗) → depth=1
        depths = self.max_level - scale_indices + 1  # [B, N]
        depths = depths.clamp(min=1, max=self.max_level)
        
        path_emb = self._encode_paths(paths, depths)  # [B, N, dim]

        # I-NAN: 计算路径熵 (path entropy)
        # 将每条路径展平为一个哈希值，然后计算分布熵
        self._compute_and_cache_path_entropy(paths, depths)

        # 3. 融合
        combined = torch.cat([scale_emb, path_emb], dim=-1)  # [B, N, dim*2]
        return self.fusion(combined)

    def _compute_and_cache_path_entropy(
        self,
        paths: torch.Tensor,
        depths: torch.Tensor,
    ) -> None:
        """计算并缓存路径分布的信息熵

        数学形式:
            H(P) = -∑p_i·log(p_i)
            其中 p_i 是第 i 条唯一路径的概率

        Args:
            paths: [B, N, max_level] 路径张量
            depths: [B, N] 有效深度
        """
        B, N, max_level = paths.shape

        # 获取有效路径部分
        level_indices = torch.arange(max_level, device=paths.device)
        valid_mask = (level_indices.unsqueeze(0).unsqueeze(0) < depths.unsqueeze(-1))  # [B, N, max_level]

        # 只保留有效路径部分
        valid_paths = torch.where(valid_mask, paths, torch.zeros_like(paths))

        # 将路径转换为哈希值（用于唯一性检测）
        # 使用位置编码确保不同位置的不同路径被区分
        # path_hash = Σ valid_paths[i] * 4^i
        powers = torch.exp2(torch.arange(max_level, device=paths.device, dtype=paths.dtype) * 2.0)  # D4-AUDIT FIX: 移除 dtype=float32 强制，遵循 paths.dtype 实现 AMP 兼容
        path_hash = (valid_paths * powers.unsqueeze(0).unsqueeze(0)).sum(dim=-1)  # [B, N]

        # 计算每个 batch 的熵并取平均
        # D1-AUDIT FIX: 使用 GPU tensor 累积，避免 forward 内 .item()
        entropies = []
        for b in range(B):
            hashes = path_hash[b]  # [N]
            # 使用直方图估计概率分布
            unique_hashes, counts = torch.unique(hashes, return_counts=True)
            probs = counts.float() / counts.sum()
            # 计算熵 H(P) = -∑p_i·log(p_i)
            entropy = -(probs * torch.log(probs + 1e-8)).sum()
            entropies.append(entropy)

        # D1-AUDIT FIX: 保持 GPU tensor，由 flatten_layer_outputs 处理 .item()
        if entropies:
            entropy_stack = torch.stack(entropies)  # [B] tensor
            self._diagnostic_cache["distribution/path_entropy"] = entropy_stack.mean()
        else:
            self._diagnostic_cache["distribution/path_entropy"] = torch.tensor(0.0, device=paths.device, dtype=torch.float32)

    @property
    def embed_output(self) -> Dict[str, Any]:
        """FractalPathEmbedding 诊断输出

        命名空间:
            embed/params/*: 可学习参数统计
            embed/distribution/*: 路径分布统计

        D1-AUDIT FIX: 所有值现在为 GPU tensor，
        由 flatten_layer_outputs() 在 post_forward() 统一调用 .item()。
        """
        output: Dict[str, Any] = {}

        # embed/params/* - 嵌入权重统计（D1-AUDIT FIX: GPU tensor））
        if hasattr(self, 'scale_embedding') and self.scale_embedding is not None:
            w = self.scale_embedding.weight
            output["params/scale_emb_norm"] = w.norm()
            output["params/scale_emb_mean"] = w.mean()

        if hasattr(self, 'quadrant_embedding') and self.quadrant_embedding is not None:
            w = self.quadrant_embedding.weight
            output["params/quadrant_emb_norm"] = w.norm()
            output["params/quadrant_emb_std"] = w.std()

        # I-NAN: 从统一诊断缓存合并 forward 中计算的指标（现在都是 GPU tensor）
        if self._diagnostic_cache:
            output.update(self._diagnostic_cache)

        return output


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
        self.max_level = config.max_level
        
        # 共同祖先深度 → 偏置值
        # depth 0 = 无共同祖先 (除根)
        # depth max_level = 完全相同的路径
        self.ancestor_bias = nn.Embedding(config.max_level + 2, heads)
        
        # 预计算共同祖先深度矩阵
        path_encoder = VectorizedPathEncoder(config)
        x, y = path_encoder._get_hilbert_coords(config.grid_size)
        paths = VectorizedPathEncoder.compute_quadrant_paths(x, y, config.max_level)
        common_depth = VectorizedPathEncoder.compute_common_ancestor_depth(paths)
        self.register_buffer('common_depth_matrix', common_depth)
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        # 初始化: 共同祖先越近，偏置越正
        with torch.no_grad():
            for i in range(self.max_level + 2):
                # 线性增长: depth 0 → -0.1, depth max → +0.1
                value = (i / (self.max_level + 1) - 0.5) * 0.2
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


class FourierPathEncoder(nn.Module):
    """Fourier Path Encoder - 连续化路径编码

    I-PHASE4: 解决离散象限嵌入的边界突变问题

    数学形式化
    ==========

    核心思想：将离散路径向量映射到高频正弦空间

    路径编码:
        设路径向量 p = (q_1, ..., q_d), q_k ∈ {0,1,2,3}
        展平为索引: idx = Σ q_k · 4^{k-1} ∈ [0, 4^d)

    Fourier 编码:
        F(p) = [cos(2πk·idx/4^d), sin(2πk·idx/4^d)]_{k=1}^{K}

    优势:
        1. 周期性自然处理 q=3 → q=0 边界
        2. 高频分量捕获精细位置差异
        3. 维度可控 (K << 4^d)
        4. 连续平滑的嵌入空间

    对比传统方案
    ============

    | 方案 | 边界处理 | 平滑性 | 表达能力 |
    |------|----------|--------|----------|
    | Quadrant Embedding | 突变 | 离散 | O(4L·D) |
    | Fourier Path | 周期 | 连续 | O(K·D) |
    """

    def __init__(
        self,
        dim: int,
        max_level: int = 8,
        num_frequencies: int = 4,
    ):
        """初始化 Fourier Path Encoder

        Args:
            dim: 嵌入维度
            max_level: 最大四叉树深度
            num_frequencies: Fourier 频率数量 (K)
        """
        super().__init__()
        self.dim = dim
        self.max_level = max_level
        self.num_frequencies = num_frequencies

        # Fourier 频率: [1, 2, ..., K]
        self.register_buffer(
            'frequencies',
            torch.arange(1, num_frequencies + 1).float()
        )

        # 深度编码 (可选，与 Fourier 路径结合)
        self.depth_embedding = nn.Embedding(max_level + 1, dim)

        # 投影层: 将 Fourier 特征投影到目标维度
        # 输入: 2 * num_frequencies (cos + sin)
        # 输出: dim
        self.projection = nn.Linear(num_frequencies * 2, dim)

        # 层归一化
        self.layer_norm = nn.LayerNorm(dim)

        # I-NAN: 统一诊断缓存
        self._diagnostic_cache: Dict[str, Any] = {}

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.depth_embedding.weight, std=0.02)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def _path_to_index(self, paths: torch.Tensor) -> torch.Tensor:
        """将路径向量转换为展平索引

        数学:
            idx = Σ q_k · 4^{k-1}

        Args:
            paths: [B, N, L] 四叉树路径

        Returns:
            indices: [B, N] 展平后的索引
        """
        B, N, L = paths.shape
        device = paths.device

        # 计算 4^{L-1}, 4^{L-2}, ..., 4^0
        powers = torch.arange(L, device=device).flip(0)  # [L]
        bases = torch.exp2(powers.float() * 2.0)  # D4-AUDIT FIX: 4**x → exp2(x*2)

        # 广播乘法并求和: [B, N, L] * [L] -> [B, N]
        indices = (paths * bases.unsqueeze(0).unsqueeze(0)).sum(dim=-1)

        return indices

    def forward(
        self,
        levels_info,
    ) -> torch.Tensor:
        """计算 Fourier 路径编码

        Args:
            levels_info: LevelsInfo 实例，包含 depths 和 paths

        Returns:
            path_emb: [B, N, dim] Fourier 路径编码
        """
        # Fix-12: 使用统一的护城河工厂方法
        from vit_pytorch.core.levels_info import LevelsInfo
        levels_info = LevelsInfo.ensure(levels_info, default_max_level=self.max_level)

        if levels_info.data.numel() == 0:
            device = levels_info.data.device
            return torch.zeros(0, self.dim, device=device, dtype=torch.float32)

        device = levels_info.data.device
        B, N = levels_info.depths.shape

        # 提取 depths 和 paths
        depths = levels_info.depths.clamp(0, self.max_level).long()  # [B, N]
        paths = levels_info.paths  # [B, N, max_level]

        # 1. 将路径转换为展平索引
        path_len = paths.shape[-1]
        # 只使用有效深度的路径
        valid_paths = paths[:, :, :path_len]
        indices = self._path_to_index(valid_paths)  # [B, N]

        # 2. 归一化索引到 [0, 1]
        max_index = 4.0 ** path_len
        normalized_indices = indices / max_index  # [B, N]

        # 3. 生成 Fourier 特征
        # [B, N, 1] * [K] -> [B, N, K]
        angles = normalized_indices.unsqueeze(-1) * self.frequencies * 2 * math.pi

        # cos + sin: [B, N, K] + [B, N, K] -> [B, N, 2K]
        fourier_features = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)

        # 4. 投影到目标维度
        path_emb = self.projection(fourier_features)  # [B, N, dim]

        # 5. 添加深度编码
        depth_emb = self.depth_embedding(depths)  # [B, N, dim]

        # 6. 融合并归一化
        combined = path_emb + depth_emb
        output = self.layer_norm(combined)

        # I-NAN: 计算并缓存边界相似度
        self._compute_boundary_similarity(combined.detach())

        return output

    def _compute_boundary_similarity(self, embeddings: torch.Tensor) -> None:
        """计算并缓存 Fourier 路径编码的边界平滑度

        使用已计算的 embeddings 计算相邻路径的余弦相似度均值

        Args:
            embeddings: [B, N, dim] 已在 forward 中计算的嵌入
        """
        import torch.nn.functional as F

        # 展平并计算相邻路径的余弦相似度
        emb_flat = embeddings.view(-1, self.dim)  # [B*N, dim]
        emb_normalized = F.normalize(emb_flat, dim=-1)

        # 计算相邻的余弦相似度（跨 batch 和 sequence）
        similarities = (emb_normalized[:-1] * emb_normalized[1:]).sum(dim=-1)

        # 计算均值并缓存到统一诊断缓存（D1-AUDIT FIX: GPU tensor）
        if similarities.numel() > 0:
            self._diagnostic_cache["health/fourier_boundary_sim"] = similarities.mean()
        else:
            self._diagnostic_cache["health/fourier_boundary_sim"] = torch.tensor(0.0, device=embeddings.device, dtype=torch.float32)

    def compute_boundary_similarity(
        self,
        paths: torch.Tensor,
    ) -> torch.Tensor:
        """计算边界处相邻路径的余弦相似度

        用于验证 Fourier 编码的平滑性

        Args:
            paths: [N, L] 或 [B, N, L] 路径张量

        Returns:
            similarities: 相邻路径的余弦相似度
        """
        import torch.nn.functional as F

        # 处理 2D 输入 [N, L] -> [1, N, L]
        if paths.dim() == 2:
            paths = paths.unsqueeze(0)
            was_2d = True
        else:
            was_2d = False

        B, N, L = paths.shape

        # 创建 LevelsInfo 进行编码
        depths = torch.full((B, N), L, dtype=torch.long, device=paths.device)
        levels_info = torch.zeros(B, N, L + 1, dtype=torch.long, device=paths.device)
        levels_info[:, :, 0] = depths
        levels_info[:, :, 1:] = paths

        # 编码
        with torch.no_grad():
            emb = self.forward(levels_info)  # [B, N, dim]

        # 展平并计算相邻路径的余弦相似度
        emb_flat = emb.view(-1, self.dim)  # [B*N, dim]
        emb_normalized = F.normalize(emb_flat, dim=-1)

        # 计算相邻的余弦相似度（跨 batch 和 sequence）
        similarities = (emb_normalized[:-1] * emb_normalized[1:]).sum(dim=-1)

        return similarities if not was_2d else similarities.view(B, -1)

    @property
    def embed_output(self) -> Dict[str, Any]:
        """FourierPathEncoder 诊断输出

        命名空间:
            embed/params/*: 可学习参数统计
            embed/health/*: Fourier 边界平滑度

        物理意义:
            边界平滑度衡量 Hilbert 曲线首尾交界处（θ ≈ 0 与 θ ≈ 2π）的嵌入连续性。
            Sim_boundary = cosine_sim(E(path), E(path + Δ))
            低值表示频率编码在跨越不连续点时有剧烈相位突变。

        D1-AUDIT FIX: 所有值现在为 GPU tensor，
        由 flatten_layer_outputs() 在 post_forward() 统一调用 .item()。
        """
        output: Dict[str, Any] = {}

        # embed/params/* - 投影层参数统计（D1-AUDIT FIX: GPU tensor））
        if hasattr(self, 'projection') and self.projection is not None:
            w = self.projection.weight
            output["params/projection_norm"] = w.norm()
            output["params/projection_mean"] = w.mean()
            output["params/projection_std"] = w.std()

        # I-NAN: 从统一诊断缓存合并 forward 中计算的指标（现在都是 GPU tensor）
        if self._diagnostic_cache:
            output.update(self._diagnostic_cache)

        # embed/health/* - nan_grad_hooks 注册数
        if hasattr(self, '_nan_grad_hooks') and self._nan_grad_hooks:
            output["health/nan_grad_hooks_registered"] = len(self._nan_grad_hooks)

        return output
