# -*- coding: utf-8 -*-
"""
Fractal 配置模块 - 统一参数推导

数学形式化
============

从两个基础参数推导所有 Fractal ViT 配置:

    (image_size, min_patch_size) → FractalConfig

推导链:
    max_depth = ⌈log₂(image_size / min_patch_size)⌉
    num_scales = max_depth + 1
    patch_sizes = (min_patch_size × 2^i)_{i=0}^{max_depth}
    grid_size = image_size / min_patch_size
    num_tokens = grid_size²

约束条件:
    1. image_size % min_patch_size == 0 (可整除)
    2. grid_size = image_size / min_patch_size 可以是任意正整数
       (使用 Pseudo-Hilbert 曲线支持非 2^k 尺寸)
    3. image_size 可以是任意正整数

Hilbert 曲线策略 (自动选择):
    - grid_size = 2^k: 使用标准 Hilbert 曲线 (最优局部性)
    - padding_ratio < 4/3: 使用 Hilbert + Padding
    - padding_ratio ≥ 4/3: 使用 Pseudo-Hilbert 递归细分

Gumbel-Softmax 温度退火:
    τ(t) = τ_max - (τ_max - τ_min) × (t / T)           # linear
    τ(t) = τ_max × (τ_min / τ_max)^{t/T}               # exponential
    τ(t) = τ_min + 0.5(τ_max - τ_min)(1 + cos(πt/T))   # cosine

Hilbert Bias 模式:
    - 'lca': LCA 嵌入 (~36 参数，推荐)
    - 'low_rank': 低秩分解 (~50K 参数)
    - 'hierarchical': 分层计算

示例:
    # 标准 2^k 配置
    config = FractalConfig(64, 4)
    → grid_size=16 ✓ (2^4)
    
    # 非 2^k 配置 (使用 Pseudo-Hilbert)
    config = FractalConfig(60, 5)
    → grid_size=12 (自动使用 Pseudo-Hilbert)
    
    # 使用配置
    tokenizer = StreamingFractalTokenizerV2.from_config(config)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Tuple


# 类型别名
BiasMode = Literal['original', 'low_rank', 'hierarchical', 'lca']
AnnealSchedule = Literal['linear', 'exponential', 'cosine']


@dataclass
class FractalConfig:
    """Fractal ViT 的统一配置.
    
    所有几何参数从 (image_size, min_patch_size) 自动推导。
    训练相关参数可灵活配置。
    
    几何参数 (自动推导):
        max_depth: 最大四叉树深度 = log₂(image_size / min_patch_size)
        num_scales: 尺度数量 = max_depth + 1
        patch_sizes: 所有 patch 大小的元组，从小到大
        grid_size: 最细网格的边长 = image_size / min_patch_size
        num_tokens: 最细网格的 token 数量 = grid_size²
        
    Gumbel-Softmax 配置:
        gumbel_tau_init: 初始温度，控制探索程度
        gumbel_tau_min: 温度下界，过低会导致梯度消失
        gumbel_tau_max: 温度上界
        gumbel_anneal_schedule: 退火策略 ('linear', 'exponential', 'cosine')
        
    Hilbert Bias 配置:
        hilbert_bias_mode: 偏置计算模式
        low_rank_r: 低秩分解的秩 (仅 bias_mode='low_rank' 时有效)
        
    Tokenizer 配置:
        variable_tokens: 是否启用可变 token 数量
        use_soft_weights: 是否使用软权重 (实验性)
        
    Hilbert 策略 (自动推导):
        uses_pseudo_hilbert: 是否使用 Pseudo-Hilbert 曲线
    """
    
    # ========== 基础参数 (必需) ==========
    image_size: int
    min_patch_size: int = 4
    
    # ========== Gumbel-Softmax 配置 ==========
    gumbel_tau_init: float = 2.0
    gumbel_tau_min: float = 0.5
    gumbel_tau_max: float = 5.0
    gumbel_anneal_schedule: AnnealSchedule = 'cosine'
    
    # ========== Hilbert Bias 配置 ==========
    hilbert_bias_mode: BiasMode = 'lca'
    low_rank_r: int = 32
    
    # ========== Tokenizer 配置 ==========
    variable_tokens: bool = True  # 默认启用可变 token 模式 (Patch=Token)
    use_soft_weights: bool = False
    
    # ========== 推导参数 (自动计算) ==========
    max_depth: int = field(init=False)
    num_scales: int = field(init=False)
    patch_sizes: Tuple[int, ...] = field(init=False)
    grid_size: int = field(init=False)
    num_tokens: int = field(init=False)
    uses_pseudo_hilbert: bool = field(init=False)
    
    # 混合策略阈值: ρ* = 4/3
    PADDING_RATIO_THRESHOLD: float = 4 / 3
    
    def __post_init__(self) -> None:
        """从基础参数推导所有配置."""
        # 验证约束
        if self.image_size <= 0 or self.min_patch_size <= 0:
            raise ValueError(
                f"image_size ({self.image_size}) 和 min_patch_size ({self.min_patch_size}) 必须为正数"
            )
        
        if self.image_size % self.min_patch_size != 0:
            raise ValueError(
                f"image_size ({self.image_size}) 必须能被 min_patch_size ({self.min_patch_size}) 整除"
            )
        
        # 计算 grid_size (现在支持任意正整数)
        ratio = self.image_size // self.min_patch_size
        if ratio <= 0:
            raise ValueError(f"grid_size = {ratio} 必须为正数")
        
        # 验证 Gumbel 参数
        if self.gumbel_tau_min >= self.gumbel_tau_max:
            raise ValueError(
                f"gumbel_tau_min ({self.gumbel_tau_min}) 必须小于 gumbel_tau_max ({self.gumbel_tau_max})"
            )
        
        if not (self.gumbel_tau_min <= self.gumbel_tau_init <= self.gumbel_tau_max):
            raise ValueError(
                f"gumbel_tau_init ({self.gumbel_tau_init}) 必须在 [{self.gumbel_tau_min}, {self.gumbel_tau_max}] 范围内"
            )
        
        # 计算推导参数
        # max_depth 使用 ceil(log2) 以支持非 2^k
        if ratio > 0:
            max_depth = math.ceil(math.log2(ratio)) if ratio > 1 else 0
        else:
            max_depth = 0
        
        object.__setattr__(self, 'max_depth', max_depth)
        object.__setattr__(self, 'num_scales', max_depth + 1)
        object.__setattr__(self, 'patch_sizes', tuple(
            self.min_patch_size * (2 ** i) for i in range(max_depth + 1)
        ))
        object.__setattr__(self, 'grid_size', ratio)
        object.__setattr__(self, 'num_tokens', ratio * ratio)
        
        # 确定 Hilbert 策略
        # 如果 grid_size 不是 2 的幂，则根据 padding_ratio 决定
        is_power_of_2 = ratio > 0 and (ratio & (ratio - 1) == 0)
        if is_power_of_2:
            uses_pseudo = False
        else:
            # 计算扩展到 2^k 的 padding ratio
            n = 1
            while n < ratio:
                n *= 2
            padding_ratio = (n * n) / (ratio * ratio)
            uses_pseudo = padding_ratio >= self.PADDING_RATIO_THRESHOLD
        
        object.__setattr__(self, 'uses_pseudo_hilbert', uses_pseudo)
    
    def scale_to_depth(self, scale_idx: int) -> int:
        """将尺度索引转换为四叉树深度.
        
        scale_idx=0 (最细) → depth=max_depth
        scale_idx=max_depth (最粗) → depth=0
        """
        return self.max_depth - scale_idx
    
    def depth_to_scale(self, depth: int) -> int:
        """将四叉树深度转换为尺度索引."""
        return self.max_depth - depth
    
    def patch_size_at_scale(self, scale_idx: int) -> int:
        """获取指定尺度的 patch 大小."""
        return self.patch_sizes[scale_idx]
    
    def grid_size_at_scale(self, scale_idx: int) -> int:
        """获取指定尺度的网格边长."""
        return self.image_size // self.patch_sizes[scale_idx]
    
    def compute_temperature(self, current_epoch: int, total_epochs: int) -> float:
        """根据退火策略计算当前温度.
        
        Args:
            current_epoch: 当前 epoch (0-indexed)
            total_epochs: 总 epoch 数
            
        Returns:
            当前温度值
        """
        if total_epochs <= 1:
            return self.gumbel_tau_init
        
        progress = min(current_epoch / (total_epochs - 1), 1.0)
        
        if self.gumbel_anneal_schedule == 'linear':
            return self.gumbel_tau_max - (self.gumbel_tau_max - self.gumbel_tau_min) * progress
        elif self.gumbel_anneal_schedule == 'exponential':
            return self.gumbel_tau_max * (self.gumbel_tau_min / self.gumbel_tau_max) ** progress
        else:  # cosine
            return self.gumbel_tau_min + 0.5 * (self.gumbel_tau_max - self.gumbel_tau_min) * (
                1 + math.cos(math.pi * progress)
            )
    
    def __repr__(self) -> str:
        hilbert_strategy = "Pseudo-Hilbert" if self.uses_pseudo_hilbert else "Standard Hilbert"
        is_power_of_2 = self.grid_size > 0 and (self.grid_size & (self.grid_size - 1) == 0)
        grid_note = "" if is_power_of_2 else f" (非 2^k, 使用 {hilbert_strategy})"
        
        return (
            f"FractalConfig(\n"
            f"  # Geometry\n"
            f"  image_size={self.image_size}, min_patch_size={self.min_patch_size}\n"
            f"  max_depth={self.max_depth}, num_scales={self.num_scales}\n"
            f"  patch_sizes={self.patch_sizes}\n"
            f"  grid_size={self.grid_size}{grid_note}, num_tokens={self.num_tokens}\n"
            f"  # Hilbert Strategy\n"
            f"  uses_pseudo_hilbert={self.uses_pseudo_hilbert}\n"
            f"  # Gumbel-Softmax\n"
            f"  tau=[{self.gumbel_tau_min}, {self.gumbel_tau_init}, {self.gumbel_tau_max}], "
            f"schedule='{self.gumbel_anneal_schedule}'\n"
            f"  # Hilbert Bias\n"
            f"  bias_mode='{self.hilbert_bias_mode}', low_rank_r={self.low_rank_r}\n"
            f"  # Tokenizer\n"
            f"  variable_tokens={self.variable_tokens}, use_soft_weights={self.use_soft_weights}\n"
            f")"
        )


def create_fractal_config(
    image_size: int,
    min_patch_size: int = 4,
    **kwargs,
) -> FractalConfig:
    """便捷函数：创建 FractalConfig.
    
    Args:
        image_size: 输入图像边长
        min_patch_size: 最小 patch 大小，默认 4
        **kwargs: 其他可选配置 (gumbel_*, hilbert_*, variable_tokens 等)
        
    Returns:
        FractalConfig 实例
        
    Examples:
        # 基础配置
        config = create_fractal_config(64, 4)
        
        # 自定义 Gumbel 温度
        config = create_fractal_config(
            64, 4,
            gumbel_tau_init=1.0,
            gumbel_tau_min=0.3,
            gumbel_anneal_schedule='exponential'
        )
        
        # 启用可变 token
        config = create_fractal_config(64, 4, variable_tokens=True)
    """
    return FractalConfig(image_size, min_patch_size, **kwargs)
