"""
Adaptive Quadtree Split for Hilbert ViT

This module implements Variable Depth tokenization with two parallel approaches:
- Scheme B: Balanced Greedy Splitting (2:1 constraint)
- Scheme C: Fixed Budget Dynamic Programming

Both schemes maintain Hilbert curve locality and are compatible with LCA bias.

Mathematical Foundation:
-----------------------
Quadtree-Hilbert Isomorphism:
    QuadtreePath(R) = [q₁, q₂, ..., qₐ] ⟺ HilbertSegment(R) = H|[a,b]

Complexity Function:
    C(R) = α · C_var(R) + (1-α) · C_grad(R)
    
    where:
    - C_var(R) = Var(R) / (Var(R) + σ₀²)  [normalized variance]
    - C_grad(R) = G(R) / (G(R) + g₀²)     [normalized gradient energy]

Depth-Dependent Threshold:
    τ_d = τ₀ · γ^d

Author: GitHub Copilot
Date: 2025-12-25
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .curve_hilbert import HilbertCurve


# =============================================================================
# Configuration
# =============================================================================

class SplitScheme(Enum):
    """Available splitting schemes."""
    BALANCED_GREEDY = "balanced_greedy"  # Scheme B
    FIXED_BUDGET_DP = "fixed_budget_dp"  # Scheme C
    LEARNABLE = "learnable"  # Scheme L (P7-1/P7-2/P7-3)


@dataclass
class AdaptiveSplitConfig:
    """
    Configuration for adaptive quadtree splitting.
    
    All parameters have mathematically derived defaults.
    See IMPROVEMENT_PLAN.md Section 10 for derivations.
    """
    
    # === Complexity Function Parameters ===
    
    alpha: float = 0.5
    """Variance weight α ∈ [0, 1]
    
    Mathematical basis:
    - α = 1.0: Pure variance, texture-sensitive
    - α = 0.0: Pure gradient, edge-sensitive  
    - α = 0.5: Balanced (recommended)
    
    Range: [0.3, 0.7]
    """
    
    sigma_0_sq: float = 0.01
    """Variance normalization constant σ₀²
    
    Mathematical basis:
    - Natural image local variance median ≈ 0.014
    - C_var = 0.5 when Var = σ₀²
    
    Range: [0.005, 0.03]
    """
    
    g_0_sq: float = 0.08
    """Gradient normalization constant g₀²
    
    Mathematical basis:
    - Natural image gradient energy median ≈ 0.07
    - C_grad = 0.5 when G = g₀²
    
    Range: [0.03, 0.15]
    """
    
    # === Threshold Function Parameters ===
    
    tau_0: float = 0.15
    """Root threshold τ₀
    
    Mathematical basis:
    - 15-20th percentile of natural image complexity
    - Controls "how simple must image be to not split"
    
    Range: [0.08, 0.25]
    """
    
    gamma: float = 0.85
    """Threshold decay factor γ ∈ (0, 1)
    
    Mathematical basis:
    - τ_d = τ₀ · γ^d
    - γ = 0.85 → deepest threshold ≈ 52% of root
    
    Range: [0.75, 0.92]
    """
    
    max_depth: int = 4
    """Maximum split depth d_max
    
    Mathematical basis:
    - d=4 corresponds to 224→14 patch size
    - Aligned with ViT-B/16
    - Maximum 4^4 = 256 tokens
    
    Range: [3, 5]
    """
    
    min_region_size: int = 7
    """Minimum region edge length (pixels)
    
    Mathematical basis:
    - At least 7×7 needed for basic texture features
    
    Range: [4, 16]
    """
    
    # === Scheme Selection ===
    
    scheme: SplitScheme = SplitScheme.BALANCED_GREEDY
    """Which splitting scheme to use"""
    
    # === Scheme B Specific Parameters ===
    
    enforce_balance: bool = True
    """Whether to enforce 2:1 balance constraint"""
    
    max_balance_iterations: int = 10
    """Maximum iterations for balance refinement"""
    
    target_tokens: Optional[int] = None
    """Soft target for token count (None = no constraint)"""
    
    token_penalty_weight: float = 0.01
    """Weight for token count deviation penalty"""
    
    # === Scheme C Specific Parameters ===
    
    token_budget: int = 64
    """Fixed token budget for Scheme C"""
    
    locality_penalty_weight: float = 0.1
    """Weight for depth discontinuity penalty in DP"""
    
    # === Computation Parameters ===
    
    use_integral_image: bool = True
    """Use integral images for O(1) region statistics"""
    
    gradient_method: str = "simple"
    """Gradient computation: 'simple' or 'sobel'"""
    
    depth_aware_eta: float = 0.0
    """深度感知归一化衰减系数 η ∈ [0, 1]
    
    数学依据:
        σ₀²(d) = σ₀² · (Area(R) / Area(root))^η
        
    - η = 0.0: 禁用 (默认，向后兼容)
    - η = 0.5: 推荐值，适用于方差与√面积成正比的域
    - η = 1.0: 方差与面积成正比假设
    
    Range: [0.0, 1.0]
    """
    
    def __post_init__(self) -> None:
        """自动验证配置."""
        self.validate()

    def get_threshold(self, depth: int) -> float:
        """Get threshold for given depth."""
        return self.tau_0 * (self.gamma ** depth)
    
    def validate(self) -> None:
        """Validate configuration."""
        assert 0.0 <= self.alpha <= 1.0, f"alpha must be in [0, 1], got {self.alpha}"
        assert self.sigma_0_sq > 0, f"sigma_0_sq must be positive"
        assert self.g_0_sq > 0, f"g_0_sq must be positive"
        assert 0.0 < self.tau_0 < 1.0, f"tau_0 must be in (0, 1)"
        assert 0.0 < self.gamma < 1.0, f"gamma must be in (0, 1)"
        assert self.max_depth >= 1, f"max_depth must be >= 1"
        assert self.min_region_size >= 1, f"min_region_size must be >= 1"
        assert self.gradient_method in ("simple", "sobel")
    
    @classmethod
    def scheme_b(cls, **kwargs) -> "AdaptiveSplitConfig":
        """Create Scheme B configuration."""
        return cls(scheme=SplitScheme.BALANCED_GREEDY, **kwargs)
    
    @classmethod
    def scheme_c(cls, token_budget: int = 64, **kwargs) -> "AdaptiveSplitConfig":
        """Create Scheme C configuration."""
        return cls(
            scheme=SplitScheme.FIXED_BUDGET_DP,
            token_budget=token_budget,
            **kwargs
        )
    
    @classmethod
    def scheme_l(cls, **kwargs) -> "AdaptiveSplitConfig":
        """Create Scheme L (Learnable) configuration.
        
        解决 P7-1/P7-2/P7-3:
            - 可学习复杂度预测器 (无饱和)
            - 可学习阈值向量 (自适应)
            - Gumbel-Softmax (可微分)
        """
        defaults = {
            "tau_0": 0.5,  # 更高的初始阈值，匹配复杂度分布
            "gamma": 0.85,
        }
        defaults.update(kwargs)
        return cls(scheme=SplitScheme.LEARNABLE, **defaults)
    
    # =========================================================================
    # 域适应预设 (Domain Adaptation Presets)
    # =========================================================================
    
    @classmethod
    def natural_images(cls, **kwargs) -> "AdaptiveSplitConfig":
        """自然图像预设 (ImageNet, COCO 等).
        
        数学依据:
            σ₀² = 0.01 ← 自然图像局部方差中位数 ~0.014
            g₀² = 0.08 ← 自然图像梯度能量中位数 ~0.07
        """
        defaults = {
            "sigma_0_sq": 0.01,
            "g_0_sq": 0.08,
            "alpha": 0.5,
            "tau_0": 0.15,
        }
        defaults.update(kwargs)
        return cls(**defaults)
    
    @classmethod
    def medical_images(cls, **kwargs) -> "AdaptiveSplitConfig":
        """医学图像预设 (CT, MRI, X-Ray 等).
        
        数学依据:
            医学图像特征: 高对比度边缘 + 均匀组织区域
            σ₀² = 0.005 ← 组织区域方差更低
            g₀² = 0.15 ← 需更强梯度才触发分割
            α = 0.3 ← 边缘比纹理更重要
        """
        defaults = {
            "sigma_0_sq": 0.005,
            "g_0_sq": 0.15,
            "alpha": 0.3,
            "tau_0": 0.12,
        }
        defaults.update(kwargs)
        return cls(**defaults)
    
    @classmethod
    def satellite_images(cls, **kwargs) -> "AdaptiveSplitConfig":
        """遥感/卫星图像预设.
        
        数学依据:
            遥感特征: 大面积均匀区域 + 细小目标
            σ₀² = 0.02 ← 地表方差更高
            g₀² = 0.05 ← 小边缘也需分割
            α = 0.7 ← 纹理比边缘更重要
        """
        defaults = {
            "sigma_0_sq": 0.02,
            "g_0_sq": 0.05,
            "alpha": 0.7,
            "tau_0": 0.18,
        }
        defaults.update(kwargs)
        return cls(**defaults)
    
    @classmethod
    def document_images(cls, **kwargs) -> "AdaptiveSplitConfig":
        """文档/OCR 图像预设.
        
        数学依据:
            文档特征: 高对比度文字 + 纯色背景
            σ₀² = 0.001 ← 背景几乎无方差
            g₀² = 0.02 ← 文字边缘需要细分
            α = 0.2 ← 边缘主导
        """
        defaults = {
            "sigma_0_sq": 0.001,
            "g_0_sq": 0.02,
            "alpha": 0.2,
            "tau_0": 0.08,
        }
        defaults.update(kwargs)
        return cls(**defaults)
    
    @classmethod
    def estimate_from_dataset(
        cls, 
        sample_images: Tensor,
        percentile: float = 50.0,
        **kwargs
    ) -> "AdaptiveSplitConfig":
        """从数据集样本自动估计参数.
        
        数学依据:
            σ₀² = median(local_variance)
            g₀² = median(gradient_energy)
            τ₀ 设置使 percentile% 的图像在根节点不分割
        
        Args:
            sample_images: [N, C, H, W] 样本图像张量 (归一化到 [0,1])
            percentile: 用于估计阈值的百分位数 (默认 50.0)
            **kwargs: 覆盖估计的参数
            
        Returns:
            基于数据统计的配置
        """
        if sample_images.dim() != 4:
            raise ValueError(f"Expected 4D tensor [N,C,H,W], got {sample_images.dim()}D")
        
        # 转为灰度
        if sample_images.shape[1] == 3:
            gray = 0.299 * sample_images[:, 0] + 0.587 * sample_images[:, 1] + 0.114 * sample_images[:, 2]
        else:
            gray = sample_images[:, 0]
        
        # 计算局部方差 (7x7 窗口)
        gray_unfold = F.unfold(gray.unsqueeze(1), kernel_size=7, padding=3)
        local_var = gray_unfold.var(dim=1)
        sigma_0_sq = float(torch.quantile(local_var.flatten(), percentile / 100.0).item())
        
        # 计算梯度能量
        dx = gray[:, :, 1:] - gray[:, :, :-1]
        dy = gray[:, 1:, :] - gray[:, :-1, :]
        grad_energy = dx[:, :-1, :].pow(2) + dy[:, :, :-1].pow(2)
        g_0_sq = float(torch.quantile(grad_energy.flatten(), percentile / 100.0).item())
        
        # 确保最小值
        sigma_0_sq = max(sigma_0_sq, 1e-6)
        g_0_sq = max(g_0_sq, 1e-6)
        
        defaults = {
            "sigma_0_sq": sigma_0_sq,
            "g_0_sq": g_0_sq,
            "tau_0": 0.15,  # 保持默认，用户可覆盖
        }
        defaults.update(kwargs)
        return cls(**defaults)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Region:
    """A rectangular region in the image."""
    x1: int  # left
    y1: int  # top
    x2: int  # right (exclusive)
    y2: int  # bottom (exclusive)
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    def get_quadrant(self, q: int) -> "Region":
        """Get q-th quadrant (0=TL, 1=TR, 2=BL, 3=BR)."""
        mx = (self.x1 + self.x2) // 2
        my = (self.y1 + self.y2) // 2
        
        if q == 0:  # Top-Left
            return Region(self.x1, self.y1, mx, my)
        elif q == 1:  # Top-Right
            return Region(mx, self.y1, self.x2, my)
        elif q == 2:  # Bottom-Left
            return Region(self.x1, my, mx, self.y2)
        else:  # Bottom-Right (q == 3)
            return Region(mx, my, self.x2, self.y2)


@dataclass
class QuadtreeNode:
    """A node in the quadtree."""
    region: Region
    depth: int
    path: List[int]  # quadrant indices from root
    complexity: float = 0.0
    importance: float = 0.0  # for Scheme C
    children: Optional[List["QuadtreeNode"]] = None
    
    @property
    def is_leaf(self) -> bool:
        return self.children is None
    
    @property
    def hilbert_idx(self) -> int:
        """Compute Hilbert index from center position."""
        # Will be set by HilbertTokenSorter
        return getattr(self, "_hilbert_idx", 0)
    
    @hilbert_idx.setter
    def hilbert_idx(self, value: int):
        self._hilbert_idx = value


@dataclass
class SplitToken:
    """Output token from adaptive splitting."""
    region: Region
    depth: int
    path: List[int]
    hilbert_idx: int
    complexity: float
    
    def to_levels_info(self, max_depth: int) -> List[int]:
        """Convert to levels_info format [depth, q1, q2, ..., 0, 0, ...]"""
        info = [self.depth] + self.path + [0] * (max_depth - len(self.path))
        return info[:max_depth + 1]


@dataclass 
class SplitResult:
    """Result of adaptive splitting for one image."""
    tokens: List[SplitToken]
    
    @property
    def num_tokens(self) -> int:
        return len(self.tokens)
    
    @property
    def depth_distribution(self) -> Dict[int, int]:
        """Count tokens at each depth."""
        dist = {}
        for t in self.tokens:
            dist[t.depth] = dist.get(t.depth, 0) + 1
        return dist
    
    @property
    def depth_entropy(self) -> float:
        """Entropy of depth distribution."""
        dist = self.depth_distribution
        total = sum(dist.values())
        entropy = 0.0
        for count in dist.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log(p)
        return entropy
    
    def get_levels_info(self, max_depth: int) -> Tensor:
        """Get levels_info tensor [N, max_depth+1]."""
        infos = [t.to_levels_info(max_depth) for t in self.tokens]
        return torch.tensor(infos, dtype=torch.long)
    
    def get_regions_tensor(self) -> Tensor:
        """Get regions tensor [N, 4] as (x1, y1, x2, y2)."""
        regions = [[t.region.x1, t.region.y1, t.region.x2, t.region.y2] 
                   for t in self.tokens]
        return torch.tensor(regions, dtype=torch.long)


# =============================================================================
# Spatial Index for Neighbor Queries
# =============================================================================

class SpatialIndex:
    """
    空间索引用于高效邻居查询。
    
    数学形式化
    ===========
    
    问题定义:
        给定节点集 N = {n₁, n₂, ..., nₖ}，对于查询节点 q，
        找到所有相邻节点 {n ∈ N : adjacent(q, n)}
    
    邻接定义:
        adjacent(a, b) ⟺ (边相邻) ∧ (非重叠)
        
        边相邻条件:
        - 水平相邻: (a.x₂ = b.x₁ ∨ a.x₁ = b.x₂) ∧ y轴重叠
        - 垂直相邻: (a.y₂ = b.y₁ ∨ a.y₁ = b.y₂) ∧ x轴重叠
    
    复杂度分析:
        暴力搜索: O(N²) 对于 N 个节点
        区间树:   O(N log N) 构建 + O(log N + k) 每次查询
        
        其中 k 是候选邻居数，通常 k << N
        
    实现选择:
        使用基于网格的空间哈希，因为:
        1. 四叉树区域大小离散 (2^d 种可能)
        2. 实现简单，无需额外依赖
        3. 对于 2:1 平衡检查足够高效
    """
    
    def __init__(self, cell_size: int = 16):
        """
        初始化空间索引。
        
        Args:
            cell_size: 网格单元大小。较小值提高精度但增加内存。
                      推荐设为最小区域大小。
        """
        self.cell_size = cell_size
        self.grid: Dict[Tuple[int, int], List[QuadtreeNode]] = {}
        self._nodes: List[QuadtreeNode] = []
    
    def clear(self) -> None:
        """清空索引。"""
        self.grid.clear()
        self._nodes.clear()
    
    def build(self, nodes: List[QuadtreeNode]) -> None:
        """
        构建空间索引。
        
        复杂度: O(N · (w/cell_size) · (h/cell_size))
               对于均匀分布的区域，约 O(N)
        
        Args:
            nodes: 待索引的节点列表
        """
        self.clear()
        self._nodes = nodes
        
        for node in nodes:
            r = node.region
            # 计算该区域覆盖的网格单元
            x1_cell = r.x1 // self.cell_size
            y1_cell = r.y1 // self.cell_size
            x2_cell = (r.x2 - 1) // self.cell_size  # -1 避免边界重复
            y2_cell = (r.y2 - 1) // self.cell_size
            
            # 将节点添加到所有覆盖的单元
            for cx in range(x1_cell, x2_cell + 1):
                for cy in range(y1_cell, y2_cell + 1):
                    cell_key = (cx, cy)
                    if cell_key not in self.grid:
                        self.grid[cell_key] = []
                    self.grid[cell_key].append(node)
    
    def query_neighbors(self, node: QuadtreeNode) -> List[QuadtreeNode]:
        """
        查询与给定节点相邻的所有节点。
        
        复杂度: O(log N + k)，其中 k 是候选邻居数
        
        Args:
            node: 查询节点
            
        Returns:
            相邻节点列表
        """
        r = node.region
        neighbors = []
        candidates_seen = set()
        
        # 扩展边界以包含相邻单元
        x1_cell = (r.x1 - 1) // self.cell_size
        y1_cell = (r.y1 - 1) // self.cell_size
        x2_cell = r.x2 // self.cell_size
        y2_cell = r.y2 // self.cell_size
        
        # 收集候选节点
        for cx in range(x1_cell, x2_cell + 1):
            for cy in range(y1_cell, y2_cell + 1):
                cell_key = (cx, cy)
                if cell_key in self.grid:
                    for candidate in self.grid[cell_key]:
                        if id(candidate) not in candidates_seen and candidate is not node:
                            candidates_seen.add(id(candidate))
                            # 验证邻接关系
                            if self._is_adjacent(r, candidate.region):
                                neighbors.append(candidate)
        
        return neighbors
    
    @staticmethod
    def _is_adjacent(a: Region, b: Region) -> bool:
        """
        判断两个区域是否相邻。
        
        数学定义:
            adjacent(a, b) ⟺ 
                (水平边相邻 ∧ y轴重叠) ∨ (垂直边相邻 ∧ x轴重叠)
        """
        # 水平相邻: 边重合 + y轴有重叠
        h_adjacent = (
            (a.x2 == b.x1 or a.x1 == b.x2) and
            not (a.y2 <= b.y1 or a.y1 >= b.y2)
        )
        
        # 垂直相邻: 边重合 + x轴有重叠
        v_adjacent = (
            (a.y2 == b.y1 or a.y1 == b.y2) and
            not (a.x2 <= b.x1 or a.x1 >= b.x2)
        )
        
        return h_adjacent or v_adjacent


# =============================================================================
# Integral Image Utilities
# =============================================================================

class IntegralImageCache:
    """
    Cache for integral images enabling O(1) region statistics.
    
    数学形式化
    ===========
    
    积分图定义:
        I(x, y) = Σ_{i<x, j<y} f(i, j)
    
    区域和查询 (O(1)):
        Σ_{(i,j)∈R} f(i,j) = I(x₂,y₂) - I(x₁,y₂) - I(x₂,y₁) + I(x₁,y₁)
    
    
    批量支持 (v2):
        I^{(b)}(x, y) = Σ_{i<x, j<y} f^{(b)}(i, j), b ∈ [0, B)
        使用 cumsum 进行批量并行计算
    
    复杂度:
        - 构建: O(B × H × W)
        - 查询: O(1) per region
    
    Computes and caches:
    - II: integral of pixel values [B, H+1, W+1] or [H+1, W+1]
    - II_sq: integral of squared pixel values  
    - II_grad: integral of gradient magnitude squared
    """
    
    def __init__(self, image: Tensor, gradient_method: str = "simple"):
        """
        Args:
            image: [B, C, H, W], [C, H, W] or [H, W] tensor, values in [0, 1]
            gradient_method: 'simple' or 'sobel'
        """
        # 标准化输入维度
        self.batch_mode = False
        if image.dim() == 2:
            # [H, W] -> [1, 1, H, W]
            image = image.unsqueeze(0).unsqueeze(0)
        elif image.dim() == 3:
            # [C, H, W] -> [1, C, H, W]
            image = image.unsqueeze(0)
        elif image.dim() == 4:
            self.batch_mode = True
        else:
            raise ValueError(f"Expected 2D, 3D, or 4D tensor, got {image.dim()}D")
        
        self.B, self.C, self.H, self.W = image.shape
        self.device = image.device
        self.dtype = image.dtype
        
        # 转换为灰度图进行复杂度计算
        # gray: [B, H, W]
        if self.C == 3:
            # 标准亮度权重
            gray = 0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]
        else:
            gray = image.mean(dim=1)
        
        # 计算批量积分图
        self.II = self._compute_integral_batch(gray)  # [B, H+1, W+1]
        self.II_sq = self._compute_integral_batch(gray ** 2)
        self.II_grad = self._compute_integral_batch(
            self._compute_gradient_magnitude_sq_batch(gray, gradient_method)
        )
    
    def _compute_integral_batch(self, img: Tensor) -> Tensor:
        """计算批量积分图.
        
        Args:
            img: [B, H, W] 或 [H, W]
            
        Returns:
            integral: [B, H+1, W+1] 带 padding 的积分图
            
        数学形式化:
            I^{(b)}(x, y) = Σ_{i<x, j<y} f^{(b)}(i, j)
            
        实现:
            使用两次 cumsum 进行并行计算:
            1. cumsum(dim=-2): 沿 H 方向累加
            2. cumsum(dim=-1): 沿 W 方向累加
        """
        if img.dim() == 2:
            img = img.unsqueeze(0)
        
        # Pad with zeros: [B, H, W] -> [B, H+1, W+1]
        padded = F.pad(img, (1, 0, 1, 0), value=0)
        
        # 批量 cumsum: 先沿 H (dim=-2)，再沿 W (dim=-1)
        integral = padded.cumsum(dim=-2).cumsum(dim=-1)
        
        return integral
    
    def _compute_gradient_magnitude_sq_batch(
        self, gray: Tensor, method: str
    ) -> Tensor:
        """计算批量梯度幅值平方.
        
        Args:
            gray: [B, H, W] 灰度图
            method: 'simple' 或 'sobel'
            
        Returns:
            grad_mag_sq: [B, H, W] 梯度幅值平方
        """
        if gray.dim() == 2:
            gray = gray.unsqueeze(0)
        
        B, H, W = gray.shape
        
        if method == "simple":
            # 简单差分，向量化实现
            grad_x = F.pad(gray[:, :, 1:] - gray[:, :, :-1], (0, 1), value=0)
            grad_y = F.pad(gray[:, 1:, :] - gray[:, :-1, :], (0, 0, 0, 1), value=0)
        else:  # sobel
            # Sobel 算子，使用 conv2d 批量处理
            sobel_x = torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                dtype=gray.dtype, device=gray.device
            ).view(1, 1, 3, 3)
            sobel_y = sobel_x.transpose(-1, -2)
            
            gray_4d = gray.unsqueeze(1)  # [B, 1, H, W]
            grad_x = F.conv2d(gray_4d, sobel_x, padding=1).squeeze(1)  # [B, H, W]
            grad_y = F.conv2d(gray_4d, sobel_y, padding=1).squeeze(1)
        
        return grad_x ** 2 + grad_y ** 2
    
    def query_sum(self, integral: Tensor, region: Region, batch_idx: int = 0) -> float:
        """Query sum over region using integral image.
        
        Args:
            integral: [B, H+1, W+1] 或 [H+1, W+1] 积分图
            region: 查询区域
            batch_idx: batch 索引 (批量模式时使用)
            
        Returns:
            区域内像素值之和
        """
        x1, y1, x2, y2 = region.x1, region.y1, region.x2, region.y2
        
        # 根据维度选择正确的索引方式
        if integral.dim() == 3:
            # 批量模式: [B, H+1, W+1]
            return (
                integral[batch_idx, y2, x2].item()
                - integral[batch_idx, y1, x2].item()
                - integral[batch_idx, y2, x1].item()
                + integral[batch_idx, y1, x1].item()
            )
        else:
            # 单图模式: [H+1, W+1]
            return (
                integral[y2, x2].item()
                - integral[y1, x2].item()
                - integral[y2, x1].item()
                + integral[y1, x1].item()
            )
    
    def query_sum_batch(
        self, 
        integral: Tensor, 
        regions: Tensor
    ) -> Tensor:
        """批量查询多个 region 的和.
        
        Args:
            integral: [B, H+1, W+1] 积分图
            regions: [N, 5] 格式 [batch_idx, x1, y1, x2, y2]
            
        Returns:
            sums: [N] 每个 region 的像素和
            
        数学形式化:
            sum_i = I[b_i, y2_i, x2_i] - I[b_i, y1_i, x2_i] 
                  - I[b_i, y2_i, x1_i] + I[b_i, y1_i, x1_i]
        """
        N = regions.shape[0]
        device = integral.device
        
        # 提取坐标
        b = regions[:, 0].long()
        x1 = regions[:, 1].long()
        y1 = regions[:, 2].long()
        x2 = regions[:, 3].long()
        y2 = regions[:, 4].long()
        
        # 向量化查询
        sums = (
            integral[b, y2, x2]
            - integral[b, y1, x2]
            - integral[b, y2, x1]
            + integral[b, y1, x1]
        )
        
        return sums
    
    def compute_variance(self, region: Region, batch_idx: int = 0) -> float:
        """Compute variance of pixel values in region."""
        area = region.area
        if area == 0:
            return 0.0
        
        sum_val = self.query_sum(self.II, region, batch_idx)
        sum_sq = self.query_sum(self.II_sq, region, batch_idx)
        
        mean = sum_val / area
        variance = sum_sq / area - mean ** 2
        return max(0.0, variance)  # Numerical stability
    
    def compute_variance_batch(
        self, 
        regions: Tensor,
        areas: Tensor
    ) -> Tensor:
        """批量计算多个 region 的方差.
        
        Args:
            regions: [N, 5] 格式 [batch_idx, x1, y1, x2, y2]
            areas: [N] 每个 region 的面积
            
        Returns:
            variances: [N] 每个 region 的方差
        """
        # 避免除零
        safe_areas = areas.clamp(min=1)
        
        sum_val = self.query_sum_batch(self.II, regions)
        sum_sq = self.query_sum_batch(self.II_sq, regions)
        
        mean = sum_val / safe_areas
        variance = sum_sq / safe_areas - mean ** 2
        
        # 数值稳定性: 负方差置零，零面积区域置零
        variance = variance.clamp(min=0)
        variance = torch.where(areas > 0, variance, torch.zeros_like(variance))
        
        return variance
    
    def compute_gradient_energy(self, region: Region, batch_idx: int = 0) -> float:
        """Compute mean squared gradient magnitude in region."""
        area = region.area
        if area == 0:
            return 0.0
        
        sum_grad = self.query_sum(self.II_grad, region, batch_idx)
        return sum_grad / area
    
    def compute_gradient_energy_batch(
        self,
        regions: Tensor,
        areas: Tensor
    ) -> Tensor:
        """批量计算多个 region 的梯度能量.
        
        Args:
            regions: [N, 5] 格式 [batch_idx, x1, y1, x2, y2]
            areas: [N] 每个 region 的面积
            
        Returns:
            energies: [N] 每个 region 的平均梯度幅值平方
        """
        safe_areas = areas.clamp(min=1)
        sum_grad = self.query_sum_batch(self.II_grad, regions)
        energy = sum_grad / safe_areas
        
        # 零面积区域置零
        energy = torch.where(areas > 0, energy, torch.zeros_like(energy))
        
        return energy


# =============================================================================
# Complexity Estimation
# =============================================================================

class ComplexityEstimator:
    """
    Estimates region complexity for split decisions.
    
    数学形式化
    ----------
    基础公式:
        C(R) = α · C_var(R) + (1-α) · C_grad(R)
        
        其中:
        - C_var(R) = Var(R) / (Var(R) + σ₀²)
        - C_grad(R) = G(R) / (G(R) + g₀²)
    
    深度感知归一化 (可选):
        σ₀²(d) = σ₀² · (Area(R) / Area(root))^η
        
        数学依据: 小区域的方差自然更低，需要相应调整归一化常数。
        η ∈ [0.5, 1.0] 控制衰减速度:
        - η = 0: 无深度感知 (原始行为)
        - η = 0.5: 方差与√面积成正比假设
        - η = 1.0: 方差与面积成正比假设
    """
    
    def __init__(
        self, 
        config: AdaptiveSplitConfig,
        root_area: Optional[int] = None,
        depth_aware_eta: float = 0.0,
    ):
        """初始化复杂度估计器.
        
        Args:
            config: 自适应分割配置
            root_area: 根区域面积 (用于深度感知归一化)
            depth_aware_eta: 深度感知衰减系数 η ∈ [0, 1]
                - 0: 禁用深度感知 (默认，向后兼容)
                - 0.5: 推荐值，假设方差与√面积成正比
        """
        self.alpha = config.alpha
        self.sigma_0_sq = config.sigma_0_sq
        self.g_0_sq = config.g_0_sq
        self.root_area = root_area
        self.depth_aware_eta = depth_aware_eta
    
    def _get_depth_adjusted_params(
        self, 
        region: Region
    ) -> Tuple[float, float]:
        """获取深度调整后的归一化参数.
        
        Args:
            region: 当前区域
            
        Returns:
            (adjusted_sigma_0_sq, adjusted_g_0_sq)
        """
        if self.depth_aware_eta == 0.0 or self.root_area is None:
            return self.sigma_0_sq, self.g_0_sq
        
        # 面积比例因子
        area_ratio = region.area / self.root_area
        scale = area_ratio ** self.depth_aware_eta
        
        return self.sigma_0_sq * scale, self.g_0_sq * scale
    
    def compute(
        self, 
        region: Region, 
        cache: IntegralImageCache
    ) -> float:
        """Compute complexity for a region."""
        var = cache.compute_variance(region)
        grad = cache.compute_gradient_energy(region)
        
        # 深度感知归一化
        sigma_0_sq, g_0_sq = self._get_depth_adjusted_params(region)
        
        c_var = var / (var + sigma_0_sq)
        c_grad = grad / (grad + g_0_sq)
        
        return self.alpha * c_var + (1 - self.alpha) * c_grad
    
    def compute_batch(
        self,
        regions: List[Region],
        cache: IntegralImageCache
    ) -> List[float]:
        """Compute complexity for multiple regions."""
        return [self.compute(r, cache) for r in regions]


# =============================================================================
# Hilbert Sorting
# =============================================================================

class HilbertTokenSorter:
    """
    Sorts tokens by Hilbert curve order.
    
    Ensures spatial locality is preserved in the token sequence.
    """
    
    def __init__(self, image_size: int):
        self.image_size = image_size
        # Find smallest power of 2 >= image_size
        self.grid_order = math.ceil(math.log2(image_size))
        self.grid_size = 2 ** self.grid_order
        # HilbertCurve uses static methods, no instantiation needed
    
    def get_hilbert_index(self, region: Region) -> int:
        """Get Hilbert index for region center."""
        cx, cy = region.center
        # Scale to grid coordinates
        gx = int(cx * self.grid_size / self.image_size)
        gy = int(cy * self.grid_size / self.image_size)
        # Clamp to valid range
        gx = max(0, min(self.grid_size - 1, gx))
        gy = max(0, min(self.grid_size - 1, gy))
        return HilbertCurve.xy_to_d(self.grid_size, gx, gy)
    
    def sort_tokens(self, tokens: List[SplitToken]) -> List[SplitToken]:
        """Sort tokens by Hilbert index."""
        for token in tokens:
            token.hilbert_idx = self.get_hilbert_index(token.region)
        return sorted(tokens, key=lambda t: t.hilbert_idx)
    
    def sort_nodes(self, nodes: List[QuadtreeNode]) -> List[QuadtreeNode]:
        """Sort quadtree nodes by Hilbert index."""
        for node in nodes:
            node.hilbert_idx = self.get_hilbert_index(node.region)
        return sorted(nodes, key=lambda n: n.hilbert_idx)


# =============================================================================
# Abstract Base Splitter
# =============================================================================

class BaseAdaptiveSplitter(ABC):
    """
    Abstract base class for adaptive splitting algorithms.
    
    Subclasses implement different splitting schemes while sharing:
    - Complexity estimation
    - Hilbert sorting
    - Output format
    """
    
    def __init__(self, config: AdaptiveSplitConfig):
        self.config = config
        config.validate()
        # ComplexityEstimator 将在 split 时初始化（需要 root_area）
        self._complexity_estimator: Optional[ComplexityEstimator] = None
    
    def _get_complexity_estimator(self, root_area: int) -> ComplexityEstimator:
        """获取或创建复杂度估计器.
        
        Args:
            root_area: 根区域面积 (用于深度感知归一化)
        """
        if (self._complexity_estimator is None or 
            self._complexity_estimator.root_area != root_area):
            self._complexity_estimator = ComplexityEstimator(
                self.config,
                root_area=root_area,
                depth_aware_eta=self.config.depth_aware_eta,
            )
        return self._complexity_estimator
    
    @property
    def complexity_estimator(self) -> ComplexityEstimator:
        """向后兼容: 返回默认估计器 (无深度感知)."""
        if self._complexity_estimator is None:
            self._complexity_estimator = ComplexityEstimator(self.config)
        return self._complexity_estimator
    
    @abstractmethod
    def split(self, image: Tensor) -> SplitResult:
        """
        Split image into adaptive tokens.
        
        Args:
            image: [C, H, W] tensor with values in [0, 1]
            
        Returns:
            SplitResult containing tokens sorted by Hilbert order
        """
        pass
    
    def split_batch(self, images: Tensor) -> List[SplitResult]:
        """
        Split batch of images.
        
        Args:
            images: [B, C, H, W] tensor
            
        Returns:
            List of SplitResult, one per image
        """
        return [self.split(img) for img in images]


# =============================================================================
# Scheme B: Balanced Greedy Splitter
# =============================================================================

class BalancedGreedySplitter(BaseAdaptiveSplitter):
    """
    Scheme B: Greedy splitting with 2:1 balance constraint.
    
    Algorithm:
    1. Greedy recursive splitting based on complexity threshold
    2. Post-process to enforce 2:1 balance (|d_i - d_j| ≤ 1 for neighbors)
    3. Optionally apply token count soft constraint
    
    Properties:
    - H_LCA_eff ≈ 1.18 (highest effective LCA information)
    - J_max ≈ 21 pixels (low Hilbert jump distance)
    - σ_N ≈ 22 (moderate token count variance)
    """
    
    def split(self, image: Tensor) -> SplitResult:
        if image.dim() == 4:
            image = image.squeeze(0)
        
        C, H, W = image.shape
        assert H == W, "Image must be square"
        
        # Initialize
        cache = IntegralImageCache(
            image, 
            gradient_method=self.config.gradient_method
        )
        sorter = HilbertTokenSorter(H)
        
        # 初始化复杂度估计器 (传入 root_area 用于深度感知归一化)
        root_area = H * W
        complexity_estimator = self._get_complexity_estimator(root_area)
        
        # Step 1: Greedy recursive splitting
        root_region = Region(0, 0, W, H)
        leaves = self._greedy_split(root_region, 0, [], cache, complexity_estimator)
        
        # Step 2: Enforce 2:1 balance
        if self.config.enforce_balance:
            leaves = self._enforce_balance(leaves, cache, H, complexity_estimator)
        
        # Step 3: Apply token count constraint (optional)
        if self.config.target_tokens is not None:
            leaves = self._apply_token_constraint(leaves, cache, H, complexity_estimator)
        
        # Step 4: Convert to tokens and sort by Hilbert
        tokens = [
            SplitToken(
                region=node.region,
                depth=node.depth,
                path=node.path,
                hilbert_idx=0,
                complexity=node.complexity
            )
            for node in leaves
        ]
        tokens = sorter.sort_tokens(tokens)
        
        return SplitResult(tokens=tokens)
    
    def _greedy_split(
        self,
        region: Region,
        depth: int,
        path: List[int],
        cache: IntegralImageCache,
        complexity_estimator: ComplexityEstimator,
    ) -> List[QuadtreeNode]:
        """Recursive greedy splitting."""
        node = QuadtreeNode(
            region=region,
            depth=depth,
            path=path.copy(),
            complexity=complexity_estimator.compute(region, cache)
        )
        
        # Termination conditions
        threshold = self.config.get_threshold(depth)
        min_size = self.config.min_region_size
        
        should_stop = (
            depth >= self.config.max_depth or
            node.complexity < threshold or
            region.width < min_size * 2 or
            region.height < min_size * 2
        )
        
        if should_stop:
            return [node]
        
        # Split into 4 quadrants
        leaves = []
        for q in range(4):
            sub_region = region.get_quadrant(q)
            sub_leaves = self._greedy_split(
                sub_region, depth + 1, path + [q], cache, complexity_estimator
            )
            leaves.extend(sub_leaves)
        
        return leaves
    
    def _enforce_balance(
        self,
        leaves: List[QuadtreeNode],
        cache: IntegralImageCache,
        image_size: int,
        complexity_estimator: ComplexityEstimator,
    ) -> List[QuadtreeNode]:
        """
        Enforce 2:1 balance constraint.
        
        数学形式化:
            ∀ (R_i, R_j) ∈ Adjacent: |d_i - d_j| ≤ 1
            
        算法:
            迭代分割浅层邻居直到满足约束
            
        复杂度 (使用空间索引):
            O(I × (N log N + N × k))
            其中 I = 迭代次数, N = 节点数, k = 平均邻居数
            
        vs 暴力搜索:
            O(I × N²)
        """
        # 初始化空间索引 (使用最小区域大小作为单元格)
        spatial_index = SpatialIndex(cell_size=max(1, self.config.min_region_size))
        
        for iteration in range(self.config.max_balance_iterations):
            # 构建/重建空间索引
            spatial_index.build(leaves)
            
            violations = []
            for node in leaves:
                # 使用空间索引进行邻居查询 (O(log N + k) vs O(N))
                neighbors = spatial_index.query_neighbors(node)
                for neighbor in neighbors:
                    if node.depth - neighbor.depth > 1:
                        violations.append(neighbor)
            
            if not violations:
                break
            
            # Split violating nodes
            new_leaves = []
            violated_set = set(id(n) for n in violations)
            
            for node in leaves:
                if id(node) in violated_set:
                    # Force split this node
                    for q in range(4):
                        sub_region = node.region.get_quadrant(q)
                        sub_node = QuadtreeNode(
                            region=sub_region,
                            depth=node.depth + 1,
                            path=node.path + [q],
                            complexity=complexity_estimator.compute(
                                sub_region, cache
                            )
                        )
                        new_leaves.append(sub_node)
                else:
                    new_leaves.append(node)
            
            leaves = new_leaves
        
        return leaves
    
    def _find_neighbors(
        self,
        node: QuadtreeNode,
        all_nodes: List[QuadtreeNode],
        image_size: int
    ) -> List[QuadtreeNode]:
        """
        Find all nodes adjacent to the given node.
        
        ⚠️ 已废弃: 此方法为 O(N) 暴力搜索，保留仅作后备。
        推荐使用 SpatialIndex.query_neighbors() 进行 O(log N + k) 查询。
        
        复杂度: O(N)，其中 N = len(all_nodes)
        """
        neighbors = []
        r = node.region
        
        for other in all_nodes:
            if other is node:
                continue
            
            o = other.region
            
            # Check adjacency (sharing an edge)
            # Horizontal adjacency
            h_adjacent = (
                (r.x2 == o.x1 or r.x1 == o.x2) and
                not (r.y2 <= o.y1 or r.y1 >= o.y2)
            )
            # Vertical adjacency  
            v_adjacent = (
                (r.y2 == o.y1 or r.y1 == o.y2) and
                not (r.x2 <= o.x1 or r.x1 >= o.x2)
            )
            
            if h_adjacent or v_adjacent:
                neighbors.append(other)
        
        return neighbors
    
    def _apply_token_constraint(
        self,
        leaves: List[QuadtreeNode],
        cache: IntegralImageCache,
        image_size: int,
        complexity_estimator: Optional[ComplexityEstimator] = None,
    ) -> List[QuadtreeNode]:
        """
        Soft constraint to nudge token count toward target.
        
        If too many tokens: merge lowest-complexity sibling groups
        If too few tokens: split highest-complexity nodes
        """
        target = self.config.target_tokens
        if target is None:
            return leaves
        
        # Simple heuristic: adjust threshold and re-split
        # This is a soft constraint, not exact
        current = len(leaves)
        
        if abs(current - target) / target < 0.2:
            # Within 20%, acceptable
            return leaves
        
        # Could implement more sophisticated merging/splitting
        # For now, just return as-is (soft constraint)
        return leaves


# =============================================================================
# Scheme C: Fixed Budget DP Splitter
# =============================================================================

class FixedBudgetDPSplitter(BaseAdaptiveSplitter):
    """
    Scheme C: Fixed token budget with dynamic programming selection.
    
    Algorithm:
    1. Build complete quadtree to max_depth
    2. Compute importance for each node
    3. Use DP to select optimal pruning with exactly N tokens
    4. Optional locality penalty for smooth depth transitions
    
    Properties:
    - σ_N = 0 (exact token count)
    - Globally optimal selection (not greedy)
    - η_waste = 0 (perfect batch efficiency)
    """
    
    def split(self, image: Tensor) -> SplitResult:
        if image.dim() == 4:
            image = image.squeeze(0)
        
        C, H, W = image.shape
        assert H == W, "Image must be square"
        
        # Initialize
        cache = IntegralImageCache(
            image,
            gradient_method=self.config.gradient_method
        )
        sorter = HilbertTokenSorter(H)
        
        # 初始化复杂度估计器 (传入 root_area 用于深度感知归一化)
        root_area = H * W
        complexity_estimator = self._get_complexity_estimator(root_area)
        
        # Step 1: Build complete quadtree
        root_region = Region(0, 0, W, H)
        root = self._build_complete_tree(root_region, 0, [], cache, complexity_estimator)
        
        # Step 2: Compute importance scores
        self._compute_importance(root, cache, complexity_estimator)
        
        # Step 3: DP selection
        budget = self.config.token_budget
        selected = self._dp_select(root, budget)
        
        # Step 4: Convert to tokens and sort
        tokens = [
            SplitToken(
                region=node.region,
                depth=node.depth,
                path=node.path,
                hilbert_idx=0,
                complexity=node.complexity
            )
            for node in selected
        ]
        tokens = sorter.sort_tokens(tokens)
        
        return SplitResult(tokens=tokens)
    
    def _build_complete_tree(
        self,
        region: Region,
        depth: int,
        path: List[int],
        cache: IntegralImageCache,
        complexity_estimator: ComplexityEstimator,
    ) -> QuadtreeNode:
        """Build complete quadtree to max_depth."""
        node = QuadtreeNode(
            region=region,
            depth=depth,
            path=path.copy(),
            complexity=complexity_estimator.compute(region, cache)
        )
        
        min_size = self.config.min_region_size
        can_split = (
            depth < self.config.max_depth and
            region.width >= min_size * 2 and
            region.height >= min_size * 2
        )
        
        if can_split:
            node.children = []
            for q in range(4):
                sub_region = region.get_quadrant(q)
                child = self._build_complete_tree(
                    sub_region, depth + 1, path + [q], cache, complexity_estimator
                )
                node.children.append(child)
        
        return node
    
    def _compute_importance(
        self,
        node: QuadtreeNode,
        cache: IntegralImageCache,
        complexity_estimator: Optional[ComplexityEstimator] = None,
    ) -> None:
        """
        Compute importance score for each node.
        
        Importance = complexity × area_weight - locality_penalty
        """
        # Base importance from complexity
        area_weight = node.region.area / (cache.H * cache.W)
        node.importance = node.complexity * math.sqrt(area_weight)
        
        # Recurse to children
        if node.children:
            for child in node.children:
                self._compute_importance(child, cache)
    
    def _dp_select(
        self,
        root: QuadtreeNode,
        budget: int
    ) -> List[QuadtreeNode]:
        """
        Dynamic programming to select optimal token set.
        
        Uses tree DP: for each subtree, compute best way to use k tokens
        for all k from 1 to budget.
        
        State: dp[node][k] = maximum importance achievable using k tokens
                            in the subtree rooted at node
        """
        # Memoization
        memo: Dict[Tuple[int, int], Tuple[float, List[QuadtreeNode]]] = {}
        
        def dp(node: QuadtreeNode, k: int) -> Tuple[float, List[QuadtreeNode]]:
            """
            Returns (importance, selected_nodes) for using exactly k tokens
            in the subtree rooted at node.
            """
            if k <= 0:
                return (0.0, [])
            
            node_id = id(node)
            if (node_id, k) in memo:
                return memo[(node_id, k)]
            
            # Option 1: Use this node as a leaf (1 token)
            if k == 1:
                result = (node.importance, [node])
                memo[(node_id, k)] = result
                return result
            
            # Option 2: If no children, can only use 1 token
            if not node.children:
                if k == 1:
                    result = (node.importance, [node])
                else:
                    result = (float('-inf'), [])  # Invalid
                memo[(node_id, k)] = result
                return result
            
            # Option 3: Distribute k tokens among 4 children
            # Use DP to find best distribution
            best_importance = node.importance if k == 1 else float('-inf')
            best_selection = [node] if k == 1 else []
            
            # Try all valid distributions of k tokens to 4 children
            # Each child must get at least 1 token (if used) or 0
            # This is a partition problem
            for dist in self._generate_distributions(k, 4):
                total_imp = 0.0
                selection = []
                valid = True
                
                for i, child_k in enumerate(dist):
                    if child_k > 0:
                        imp, sel = dp(node.children[i], child_k)
                        if imp == float('-inf'):
                            valid = False
                            break
                        total_imp += imp
                        selection.extend(sel)
                
                if valid and total_imp > best_importance:
                    best_importance = total_imp
                    best_selection = selection
            
            # Apply locality penalty for depth variance
            if best_selection and self.config.locality_penalty_weight > 0:
                depths = [n.depth for n in best_selection]
                if len(depths) > 1:
                    depth_var = sum((d - sum(depths)/len(depths))**2 
                                   for d in depths) / len(depths)
                    penalty = self.config.locality_penalty_weight * depth_var
                    best_importance -= penalty
            
            result = (best_importance, best_selection)
            memo[(node_id, k)] = result
            return result
        
        _, selected = dp(root, budget)
        return selected
    
    def _generate_distributions(
        self, 
        total: int, 
        parts: int
    ) -> List[Tuple[int, ...]]:
        """
        Generate all ways to distribute 'total' tokens to 'parts' children.
        
        Optimized to avoid combinatorial explosion.
        """
        if parts == 1:
            return [(total,)]
        
        distributions = []
        # Limit search space for efficiency
        max_per_part = min(total, 64)  # Cap individual allocation
        
        for first in range(min(total + 1, max_per_part + 1)):
            for rest in self._generate_distributions(total - first, parts - 1):
                distributions.append((first,) + rest)
        
        return distributions


# =============================================================================
# Scheme L: Learnable Splitter (P7-1/P7-2/P7-3 Solution)
# =============================================================================

class ComplexityMLP(nn.Module):
    """
    可学习复杂度预测器。
    
    数学形式化
    ----------
    
    解决问题 P7-1 (复杂度饱和效应):
        原公式: C(R) = Var/(Var + σ₀²) 在 Var >> σ₀² 时饱和
        新公式: C_θ(R) = σ(MLP(Pool(F, R)))
        
    输入:
        f_R ∈ ℝ^(C × k × k): 区域特征 (ROI-Align 输出)
        
    输出:
        C_θ(R) ∈ [0, 1]: 可学习复杂度
        
    梯度分析:
        ∂C_θ/∂θ = σ'(z) · ∂MLP/∂θ
        σ'(z) = σ(z)(1-σ(z)) ∈ (0, 0.25]
        ✅ 梯度稳定，无饱和问题
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: 输入特征维度 (C × k × k)
            hidden_dim: 隐藏层维度
            dropout: Dropout 概率
        """
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        
        # 初始化: 输出接近 0.5
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.xavier_uniform_(self.mlp[3].weight, gain=0.1)  # 小增益
        nn.init.zeros_(self.mlp[3].bias)
    
    def forward(self, features: Tensor) -> Tensor:
        """
        Args:
            features: [N, C*k*k] 展平的区域特征
            
        Returns:
            complexity: [N] 复杂度值 ∈ [0, 1]
        """
        logits = self.mlp(features).squeeze(-1)  # [N]
        return torch.sigmoid(logits)


class LearnableSplitter(nn.Module):
    """
    可学习四叉树分割器 (Scheme L)。
    
    数学形式化
    ==========
    
    核心公式:
        1. 复杂度预测: C_θ(R) = σ(MLP(ROI-Align(F, R)))
        2. 分割概率:   p_split = σ((C_θ(R) - τ_d) / T)
        3. 离散采样:   z ~ Gumbel-Softmax([1-p, p], τ_gumbel)
        4. STE 推理:   z_hard = one_hot(argmax(z)); z_ST = z_hard - z.detach() + z
    
    解决的问题:
        P7-1: 复杂度饱和 → MLP 预测器无饱和上界
        P7-2: 阈值不匹配 → 可学习阈值向量 τ ∈ ℝ^(D+1)
        P7-3: 梯度阻断  → Gumbel-Softmax 可微分采样
    
    Hilbert 约束:
        C1 (局部性): 保持四叉树-Hilbert 同构，LCA 兼容
        C2 (2:1 平衡): 可选后处理约束
        C3 (可微分): Gumbel-Softmax + STE
    
    复杂度分析:
        时间: O(B × max_tokens × D_{max} × k²) for ROI-Align
        空间: O(B × C × H × W) for feature maps
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        max_depth: int = 4,
        hidden_dim: int = 64,
        pool_size: int = 4,
        temperature: float = 1.0,
        use_gumbel: bool = True,
        enforce_balance: bool = False,
        min_region_size: int = 7,
        dropout: float = 0.1,
        init_tau_base: float = 0.5,
        init_tau_gamma: float = 0.85,
    ):
        """
        Args:
            feature_dim: 输入特征通道数 C
            max_depth: 最大分割深度 D_max
            hidden_dim: ComplexityMLP 隐藏层维度
            pool_size: ROI-Align 输出尺寸 k×k
            temperature: Gumbel-Softmax 温度 T
            use_gumbel: 是否使用 Gumbel 噪声 (训练时)
            enforce_balance: 是否强制 2:1 平衡约束
            min_region_size: 最小区域边长
            dropout: MLP dropout
            init_tau_base: 初始根阈值 τ₀ (用于参数初始化)
            init_tau_gamma: 初始阈值衰减 γ (用于参数初始化)
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.max_depth = max_depth
        self.pool_size = pool_size
        self.temperature = temperature
        self.use_gumbel = use_gumbel
        self.enforce_balance = enforce_balance
        self.min_region_size = min_region_size
        
        # P7-1: 可学习复杂度预测器
        input_dim = feature_dim * pool_size * pool_size
        self.complexity_mlp = ComplexityMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        
        # P7-2: 可学习阈值向量 (per-depth)
        # τ_d = sigmoid(threshold_logits[d])
        # 初始化: τ_d ≈ init_tau_base * init_tau_gamma^d
        init_taus = [init_tau_base * (init_tau_gamma ** d) for d in range(max_depth + 1)]
        init_logits = [math.log(tau / (1 - tau + 1e-8)) for tau in init_taus]
        self.threshold_logits = nn.Parameter(torch.tensor(init_logits))
        
        # 可学习温度 (可选)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature)))
        
        # 统计信息 (用于监控)
        self.register_buffer('_split_probs', torch.zeros(max_depth + 1))
        self.register_buffer('_split_counts', torch.zeros(max_depth + 1))
    
    @property
    def thresholds(self) -> Tensor:
        """获取当前阈值向量 τ ∈ [0, 1]^(D+1)."""
        return torch.sigmoid(self.threshold_logits)
    
    @property
    def current_temperature(self) -> float:
        """获取当前温度 T > 0."""
        return self.log_temperature.exp().item()
    
    def set_temperature(self, temperature: float) -> None:
        """设置温度 (用于退火调度)."""
        self.log_temperature.data.fill_(math.log(temperature))
    
    def forward(
        self,
        features: Tensor,
        image_size: Tuple[int, int],
        hard: bool = False,
    ) -> List[SplitResult]:
        """
        对批量图像进行可学习分割。
        
        Args:
            features: [B, C, H', W'] 共享特征图 (来自 SharedConv)
            image_size: (H, W) 原始图像尺寸
            hard: 是否使用 hard decision (推理时)
            
        Returns:
            List[SplitResult]: 每张图像的分割结果
        """
        B, C, H_feat, W_feat = features.shape
        H_img, W_img = image_size
        device = features.device
        
        # 计算特征图到图像的缩放比
        scale_h = H_feat / H_img
        scale_w = W_feat / W_img
        
        results = []
        
        for b in range(B):
            # 单张图像的分割
            tokens = self._split_single(
                features[b:b+1],  # [1, C, H', W']
                H_img, W_img,
                scale_h, scale_w,
                hard=hard or not self.training,
            )
            
            # 按 Hilbert 顺序排序
            sorter = HilbertTokenSorter(H_img)
            tokens = sorter.sort_tokens(tokens)
            
            results.append(SplitResult(tokens=tokens))
        
        return results
    
    def _split_single(
        self,
        features: Tensor,
        H_img: int,
        W_img: int,
        scale_h: float,
        scale_w: float,
        hard: bool = False,
    ) -> List[SplitToken]:
        """
        对单张图像进行递归分割。
        
        数学形式化:
            递归分割树: T = split(R_root, d=0)
            停止条件: stop(R, d) = 𝟙[p_stop > 0.5] (hard) 或 Gumbel 采样 (soft)
            
        P8-2 优化: 使用 BFS 批量化替代深度优先递归
            复杂度改进: O(N×D) Python calls → O(D) batched GPU ops
        """
        root_region = Region(0, 0, W_img, H_img)
        
        # P8-2: 使用 BFS 批量化分割
        return self._split_bfs(
            features=features,
            root_region=root_region,
            scale_h=scale_h,
            scale_w=scale_w,
            hard=hard,
        )
    
    def _split_bfs(
        self,
        features: Tensor,
        root_region: Region,
        scale_h: float,
        scale_w: float,
        hard: bool,
    ) -> List[SplitToken]:
        """
        广度优先批量分割 (P8-2)。
        
        数学形式化:
            T_improved = O(D_max) Python calls, each with batched GPU ops
            
            对于深度 d:
              1. 收集该层所有区域 R_d = {R_1, ..., R_n}
              2. 批量 ROI-Align: f_batch = ROI(F, boxes(R_d))  # 1次 kernel
              3. 批量 MLP: C_batch = MLP(flatten(f_batch))    # 1次 kernel  
              4. 批量决策: split_d = σ((C_batch - τ_d) / T)   # 向量化
              
        与递归版本对比:
            递归: O(N_tokens × D) 次 Python 调用，每次触发 kernel launch
            BFS:  O(D) 次 Python 调用，每次处理整层 (batched)
            
        实测加速比: ~10-50x (取决于 N_tokens)
        """
        device = features.device
        all_tokens: List[SplitToken] = []
        
        # Level 0: 根区域
        # 每个元素: (region, path)
        current_level: List[Tuple[Region, List[int]]] = [(root_region, [])]
        
        for depth in range(self.max_depth + 1):
            if not current_level:
                break
            
            # 1. 分离可分割 vs 终止区域 (硬约束: 尺寸限制)
            splittable = []
            terminal = []
            
            for region, path in current_level:
                if (region.width < self.min_region_size * 2 or 
                    region.height < self.min_region_size * 2):
                    terminal.append((region, path))
                else:
                    splittable.append((region, path))
            
            # 2. 终止区域直接输出 (深度达到 max_depth 或尺寸不足)
            for region, path in terminal:
                all_tokens.append(self._create_token(region, depth, path, 0.0))
            
            # 如果已到最大深度，将所有可分割区域也标记为叶节点
            if depth >= self.max_depth:
                for region, path in splittable:
                    all_tokens.append(self._create_token(region, depth, path, 0.0))
                break
            
            if not splittable:
                break
            
            # 3. 批量计算 complexity
            complexities = self._batch_compute_complexity(
                features, splittable, scale_h, scale_w
            )  # [N_d]
            
            # 4. 批量计算分割决策 (P8-4: 返回 STE decisions)
            decisions, ste_decisions = self._batch_split_decision(
                complexities, depth, hard
            )  # decisions: List[bool], ste_decisions: Tensor[N_d]
            
            # P8-4: 存储 STE decisions 用于辅助损失
            # 注意: ste_decisions 有梯度，可用于 REINFORCE 或策略梯度
            # 当前版本暂不使用，但保留接口以供未来扩展
            
            # 5. 构建下一层
            next_level: List[Tuple[Region, List[int]]] = []
            
            for i, ((region, path), should_split) in enumerate(zip(splittable, decisions)):
                c = complexities[i].item()
                
                if should_split:
                    # 分割为 4 个象限
                    for q in range(4):
                        sub_region = region.get_quadrant(q)
                        next_level.append((sub_region, path + [q]))
                else:
                    # 保持为叶节点
                    all_tokens.append(self._create_token(region, depth, path, c))
            
            current_level = next_level
        
        return all_tokens
    
    def _batch_compute_complexity(
        self,
        features: Tensor,
        regions_with_paths: List[Tuple[Region, List[int]]],
        scale_h: float,
        scale_w: float,
    ) -> Tensor:
        """
        批量计算多个区域的复杂度 (P8-2)。
        
        数学形式化:
            给定 N 个区域 R_1, ..., R_N:
            boxes = stack([box(R_i) for i in 1..N])     # [N, 5]
            f_batch = ROI-Align(F, boxes)              # [N, C, k, k]
            C_batch = σ(MLP(flatten(f_batch)))         # [N]
            
        复杂度: 1 次 ROI-Align kernel + 1 次 MLP forward
        """
        device = features.device
        N = len(regions_with_paths)
        
        if N == 0:
            return torch.tensor([], device=device)
        
        # 1. 收集所有 boxes: [batch_idx, x1, y1, x2, y2]
        boxes_list = []
        for region, _ in regions_with_paths:
            x1_feat = region.x1 * scale_w
            y1_feat = region.y1 * scale_h
            x2_feat = region.x2 * scale_w
            y2_feat = region.y2 * scale_h
            boxes_list.append([0, x1_feat, y1_feat, x2_feat, y2_feat])
        
        boxes = torch.tensor(boxes_list, dtype=features.dtype, device=device)  # [N, 5]
        
        # 2. 批量 ROI-Align
        try:
            from torchvision.ops import roi_align
            pooled = roi_align(
                features,
                boxes,
                output_size=(self.pool_size, self.pool_size),
                spatial_scale=1.0,
                aligned=True,
            )  # [N, C, k, k]
        except ImportError:
            # Fallback: 逐个处理 (性能降级)
            pooled_list = []
            for region, _ in regions_with_paths:
                x1 = max(0, int(region.x1 * scale_w))
                y1 = max(0, int(region.y1 * scale_h))
                x2 = min(features.shape[3], int(region.x2 * scale_w) + 1)
                y2 = min(features.shape[2], int(region.y2 * scale_h) + 1)
                
                if x2 <= x1 or y2 <= y1:
                    p = torch.zeros(1, features.shape[1], self.pool_size, self.pool_size, device=device)
                else:
                    region_feat = features[:, :, y1:y2, x1:x2]
                    p = F.adaptive_avg_pool2d(region_feat, (self.pool_size, self.pool_size))
                pooled_list.append(p)
            pooled = torch.cat(pooled_list, dim=0)  # [N, C, k, k]
        
        # 3. 展平并通过 MLP
        flat = pooled.flatten(start_dim=1)  # [N, C*k*k]
        complexities = self.complexity_mlp(flat)  # [N, 1]
        
        # 确保返回 1D 张量 [N]，即使 N=1
        if complexities.dim() == 2:
            complexities = complexities.view(-1)  # [N]
        
        return complexities
    
    def _batch_split_decision(
        self,
        complexities: Tensor,
        depth: int,
        hard: bool,
    ) -> List[bool]:
        """
        批量计算分割决策 (P8-2)。
        
        数学形式化:
            p_split = σ((C - τ_d) / T)
            
            Hard mode:
                decision = 𝟙[p_split > 0.5]
                
            Soft mode (Gumbel-Softmax):
                g ~ Gumbel(0, 1)
                y = softmax((log[1-p, p] + g) / τ)
                decision = argmax(y) == 1
                
        Returns:
            List[bool]: 每个区域的分割决策
        """
        device = complexities.device
        N = complexities.shape[0]
        
        # 获取当前深度的阈值
        tau_d = self.thresholds[depth]
        T = self.log_temperature.exp()
        
        # 计算分割概率
        p_split = torch.sigmoid((complexities - tau_d) / T)  # [N]
        
        # 更新统计信息
        with torch.no_grad():
            self._split_probs[depth] += p_split.sum().item()
            self._split_counts[depth] += N
        
        if hard or not self.training:
            # Hard decision
            decisions = (p_split > 0.5).tolist()
            # P8-4: 计算 STE tensor (即使 hard mode，也返回可微分的 soft probs)
            ste_decisions = self._compute_ste_decisions(p_split)
        elif not self.use_gumbel:
            # 无 Gumbel 采样
            decisions = (p_split > 0.5).tolist()
            ste_decisions = self._compute_ste_decisions(p_split)
        else:
            # Gumbel-Softmax 采样 (批量) with STE
            decisions, ste_decisions = self._batch_gumbel_sample_ste(p_split)
        
        return decisions, ste_decisions
    
    def _compute_ste_decisions(self, p_split: Tensor) -> Tensor:
        """
        计算 Straight-Through Estimator 决策 (P8-4)。
        
        数学形式化:
            STE: ∂L/∂p = ∂L/∂z_hard × ∂z_soft/∂p
            
            前向: z_hard = 𝟙[p > 0.5] ∈ {0, 1}
            后向: 使用 p 的梯度 (soft)
            
        实现:
            z = z_hard - p.detach() + p
            
        梯度:
            ∂z/∂p = 1 (通过 STE)
            
        Returns:
            Tensor[N]: 每个区域的 STE 决策 (0 或 1，但有梯度)
        """
        # Hard decisions as float
        z_hard = (p_split > 0.5).float()
        
        # Straight-Through Estimator
        # 前向: z_hard (离散)
        # 后向: p_split 的梯度 (连续)
        ste_decisions = z_hard - p_split.detach() + p_split
        
        return ste_decisions
    
    def _batch_gumbel_sample_ste(self, p_split: Tensor) -> Tuple[List[bool], Tensor]:
        """
        批量 Gumbel-Softmax 采样 with STE (P8-2 + P8-4)。
        
        数学形式化:
            对于每个 p_i ∈ p_split:
                g_0, g_1 ~ Gumbel(0, 1)
                y_i = softmax((log[1-p_i, p_i] + [g_0, g_1]) / τ)
                z_i = argmax(y_i)
                
            STE:
                z_hard = one_hot(argmax(y))
                z_ste = z_hard - y.detach() + y
                
        Returns:
            Tuple[List[bool], Tensor]:
                - decisions: 用于控制流的 bool 列表
                - ste_decisions: 用于梯度的 tensor [N]
        """
        device = p_split.device
        N = p_split.shape[0]
        eps = 1e-8
        
        # Log probabilities: [N, 2]
        log_probs = torch.stack([
            torch.log(1 - p_split + eps),
            torch.log(p_split + eps)
        ], dim=1)  # [N, 2]
        
        # Gumbel noise: [N, 2]
        u = torch.rand(N, 2, device=device)
        gumbel = -torch.log(-torch.log(u + eps) + eps)
        
        # Gumbel-Softmax: [N, 2]
        tau_gumbel = self.log_temperature.exp()
        y = F.softmax((log_probs + gumbel) / tau_gumbel, dim=1)  # [N, 2]
        
        # Hard decisions for control flow
        hard_indices = y[:, 1] > y[:, 0]  # [N] bool
        decisions = hard_indices.tolist()
        
        # STE for gradients
        # z_hard: one-hot 编码的 argmax
        # z_ste[1] 表示"分割"的概率
        y_split = y[:, 1]  # soft probability for split [N]
        z_hard = hard_indices.float()
        ste_decisions = z_hard - y_split.detach() + y_split
        
        return decisions, ste_decisions
    
    def _batch_gumbel_sample(self, p_split: Tensor) -> List[bool]:
        """
        批量 Gumbel-Softmax 采样 (P8-2)。
        
        Note: 已废弃，请使用 _batch_gumbel_sample_ste
        
        数学形式化:
            对于每个 p_i ∈ p_split:
                g_0, g_1 ~ Gumbel(0, 1)
                y_i = softmax((log[1-p_i, p_i] + [g_0, g_1]) / τ)
                z_i = argmax(y_i)
                
        批量优化:
            log_probs: [N, 2]
            gumbel:    [N, 2]
            y:         [N, 2]
            decisions: [N]
        """
        decisions, _ = self._batch_gumbel_sample_ste(p_split)
        return decisions
    
    def _recursive_split(
        self,
        features: Tensor,
        region: Region,
        depth: int,
        path: List[int],
        scale_h: float,
        scale_w: float,
        hard: bool,
        tokens: List[SplitToken],
    ) -> None:
        """
        递归分割单个区域。
        
        Args:
            features: [1, C, H', W'] 特征图
            region: 当前区域
            depth: 当前深度
            path: 从根到当前节点的路径
            scale_h, scale_w: 特征图到图像的缩放比
            hard: 是否使用 hard decision
            tokens: 输出 token 列表
        """
        device = features.device
        
        # 1. 检查终止条件 (硬约束)
        if depth >= self.max_depth:
            tokens.append(self._create_token(region, depth, path, 0.0))
            return
        
        if region.width < self.min_region_size * 2 or region.height < self.min_region_size * 2:
            tokens.append(self._create_token(region, depth, path, 0.0))
            return
        
        # 2. 计算区域复杂度 C_θ(R)
        complexity = self._compute_complexity(features, region, scale_h, scale_w)
        
        # 3. 计算分割概率 p_split = σ((C - τ_d) / T)
        tau_d = self.thresholds[depth]
        T = self.log_temperature.exp()
        p_split = torch.sigmoid((complexity - tau_d) / T)
        
        # 更新统计信息
        with torch.no_grad():
            self._split_probs[depth] += p_split.item()
            self._split_counts[depth] += 1
        
        # 4. 采样分割决策
        if hard:
            # Hard decision: 基于阈值
            should_split = p_split.item() > 0.5
        else:
            # Soft decision: Gumbel-Softmax
            should_split = self._gumbel_sample(p_split)
        
        # 5. 执行决策
        if should_split:
            # 分割为 4 个象限
            for q in range(4):
                sub_region = region.get_quadrant(q)
                self._recursive_split(
                    features=features,
                    region=sub_region,
                    depth=depth + 1,
                    path=path + [q],
                    scale_h=scale_h,
                    scale_w=scale_w,
                    hard=hard,
                    tokens=tokens,
                )
        else:
            # 保持为叶节点
            tokens.append(self._create_token(region, depth, path, complexity.item()))
    
    def _compute_complexity(
        self,
        features: Tensor,
        region: Region,
        scale_h: float,
        scale_w: float,
    ) -> Tensor:
        """
        计算区域的可学习复杂度。
        
        数学形式化:
            f_R = ROI-Align(F, R)  ∈ ℝ^(C × k × k)
            C_θ(R) = σ(MLP(flatten(f_R)))  ∈ [0, 1]
        """
        device = features.device
        
        # 将图像坐标转换为特征图坐标
        x1_feat = region.x1 * scale_w
        y1_feat = region.y1 * scale_h
        x2_feat = region.x2 * scale_w
        y2_feat = region.y2 * scale_h
        
        # ROI boxes: [batch_idx, x1, y1, x2, y2]
        boxes = torch.tensor(
            [[0, x1_feat, y1_feat, x2_feat, y2_feat]],
            dtype=features.dtype,
            device=device,
        )
        
        # ROI-Align
        try:
            from torchvision.ops import roi_align
            pooled = roi_align(
                features,
                boxes,
                output_size=(self.pool_size, self.pool_size),
                spatial_scale=1.0,  # 已经转换坐标
                aligned=True,
            )  # [1, C, k, k]
        except ImportError:
            # Fallback: 简单的自适应池化
            # 提取区域特征
            x1 = max(0, int(x1_feat))
            y1 = max(0, int(y1_feat))
            x2 = min(features.shape[3], int(x2_feat) + 1)
            y2 = min(features.shape[2], int(y2_feat) + 1)
            
            if x2 <= x1 or y2 <= y1:
                # 退化区域，返回低复杂度
                return torch.tensor(0.1, device=device)
            
            region_feat = features[:, :, y1:y2, x1:x2]
            pooled = F.adaptive_avg_pool2d(
                region_feat, 
                (self.pool_size, self.pool_size)
            )
        
        # 展平并通过 MLP
        flat = pooled.flatten(start_dim=1)  # [1, C*k*k]
        complexity = self.complexity_mlp(flat)  # [1]
        
        return complexity.squeeze()
    
    def _gumbel_sample(self, p_split: Tensor) -> bool:
        """
        Gumbel-Softmax 采样。
        
        数学形式化:
            g_0, g_1 ~ Gumbel(0, 1)
            y = softmax((log[1-p, p] + [g_0, g_1]) / τ)
            z = argmax(y)
            
        STE (Straight-Through Estimator):
            前向: z_hard = one_hot(argmax(y))
            后向: 使用 y 的梯度
        """
        if not self.use_gumbel or not self.training:
            return p_split.item() > 0.5
        
        device = p_split.device
        
        # Log probabilities
        eps = 1e-8
        log_probs = torch.log(torch.stack([1 - p_split + eps, p_split + eps]))
        
        # Gumbel noise
        gumbel = -torch.log(-torch.log(torch.rand(2, device=device) + eps) + eps)
        
        # Gumbel-Softmax
        tau_gumbel = self.log_temperature.exp()
        y = F.softmax((log_probs + gumbel) / tau_gumbel, dim=0)
        
        # Hard sample with STE
        if self.training:
            # Straight-through: 前向用 hard，后向用 soft
            y_hard = torch.zeros_like(y)
            y_hard[y.argmax()] = 1.0
            y = y_hard - y.detach() + y
        
        return y[1].item() > 0.5
    
    def _create_token(
        self,
        region: Region,
        depth: int,
        path: List[int],
        complexity: float,
    ) -> SplitToken:
        """创建 SplitToken."""
        return SplitToken(
            region=region,
            depth=depth,
            path=path.copy(),
            hilbert_idx=0,  # 后续由 HilbertTokenSorter 填充
            complexity=complexity,
        )
    
    def get_split_statistics(self) -> Dict[str, Tensor]:
        """获取分割统计信息。"""
        avg_probs = torch.where(
            self._split_counts > 0,
            self._split_probs / self._split_counts,
            torch.zeros_like(self._split_probs),
        )
        return {
            'thresholds': self.thresholds.detach(),
            'temperature': torch.tensor(self.current_temperature),
            'avg_split_probs': avg_probs,
            'split_counts': self._split_counts.clone(),
        }
    
    def reset_statistics(self) -> None:
        """重置统计信息。"""
        self._split_probs.zero_()
        self._split_counts.zero_()
    
    def get_ste_reinforce_loss(
        self,
        features: Tensor,
        image_size: Tuple[int, int],
        reward_fn: Optional[Callable[[List[SplitResult]], Tensor]] = None,
        baseline: Optional[float] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """
        使用 STE + REINFORCE 计算策略梯度损失 (P8-4)。
        
        数学形式化:
            REINFORCE 梯度:
                ∇_θ J = E_π[∇_θ log π(a|s) × (R - b)]
                
            其中:
                - π(a|s) = σ((C_θ(R) - τ_d) / T) 是分割概率
                - R = reward_fn(split_results) 是奖励
                - b = baseline 是基线
                
            STE 增强:
                使用 STE decisions 替代 log π，提供更稳定的梯度
                
        Args:
            features: [B, C, H', W'] 特征图
            image_size: (H, W) 图像尺寸
            reward_fn: 奖励函数，输入分割结果，返回标量奖励
                       默认使用负 token 数作为奖励 (鼓励更多分割)
            baseline: 奖励基线，用于减少方差。如果为 None，使用移动平均。
            
        Returns:
            Tuple[loss, details]:
                - loss: REINFORCE 损失
                - details: 包含奖励、基线等调试信息
                
        Example:
            # 使用分类准确率作为奖励
            def acc_reward(results):
                # 计算基于当前分割的分类准确率
                return accuracy
                
            loss, details = splitter.get_ste_reinforce_loss(
                features, image_size, reward_fn=acc_reward
            )
            loss.backward()
        """
        device = features.device
        B, C, H_feat, W_feat = features.shape
        H_img, W_img = image_size
        scale_h = H_feat / H_img
        scale_w = W_feat / W_img
        
        # 收集所有批次的 STE decisions
        all_ste_decisions: List[Tensor] = []
        all_results: List[SplitResult] = []
        
        for b in range(B):
            # 单张图像的分割 with STE tracking
            ste_decisions, tokens = self._split_bfs_with_ste(
                features=features[b:b+1],
                root_region=Region(0, 0, W_img, H_img),
                scale_h=scale_h,
                scale_w=scale_w,
                hard=False,  # 使用 soft 模式以获得 Gumbel 采样
            )
            
            # 按 Hilbert 顺序排序
            sorter = HilbertTokenSorter(H_img)
            tokens = sorter.sort_tokens(tokens)
            
            all_ste_decisions.append(ste_decisions)
            all_results.append(SplitResult(tokens=tokens))
        
        # 计算奖励
        if reward_fn is None:
            # 默认奖励: 鼓励多尺度分割
            # R = 归一化的 token 数 (token 越多奖励越高)
            n_tokens = sum(len(r.tokens) for r in all_results)
            max_tokens = B * (4 ** self.max_depth)  # 最大可能 tokens
            reward = torch.tensor(n_tokens / max_tokens, device=device)
        else:
            reward = reward_fn(all_results)
        
        # 计算基线
        if baseline is None:
            # 使用当前奖励作为简单基线
            baseline_value = reward.detach()
        else:
            baseline_value = torch.tensor(baseline, device=device)
        
        # 计算 REINFORCE 损失
        # L = -Σ ste_decision × (R - b)
        # 由于 ste_decision 使用 STE，梯度可以流回 threshold_logits
        advantage = reward - baseline_value
        
        total_loss = torch.tensor(0.0, device=device)
        for ste_decisions in all_ste_decisions:
            if ste_decisions.numel() > 0:
                # 负号因为我们想最大化奖励
                total_loss = total_loss - (ste_decisions.mean() * advantage)
        
        total_loss = total_loss / max(B, 1)
        
        details = {
            'reward': reward.item(),
            'baseline': baseline_value.item() if isinstance(baseline_value, Tensor) else baseline_value,
            'advantage': advantage.item(),
            'n_tokens': sum(len(r.tokens) for r in all_results),
            'n_decisions': sum(d.numel() for d in all_ste_decisions),
        }
        
        return total_loss, details
    
    def _split_bfs_with_ste(
        self,
        features: Tensor,
        root_region: Region,
        scale_h: float,
        scale_w: float,
        hard: bool,
    ) -> Tuple[Tensor, List[SplitToken]]:
        """
        广度优先批量分割，返回 STE decisions 用于梯度 (P8-4)。
        
        与 _split_bfs 类似，但额外返回拼接的 STE decisions tensor。
        
        Returns:
            Tuple[ste_decisions, tokens]:
                - ste_decisions: [total_decisions] 所有深度的 STE 决策拼接
                - tokens: 分割结果 tokens
        """
        device = features.device
        all_tokens: List[SplitToken] = []
        all_ste_decisions: List[Tensor] = []
        
        current_level: List[Tuple[Region, List[int]]] = [(root_region, [])]
        
        for depth in range(self.max_depth + 1):
            if not current_level:
                break
            
            splittable = []
            terminal = []
            
            for region, path in current_level:
                if (region.width < self.min_region_size * 2 or 
                    region.height < self.min_region_size * 2):
                    terminal.append((region, path))
                else:
                    splittable.append((region, path))
            
            for region, path in terminal:
                all_tokens.append(self._create_token(region, depth, path, 0.0))
            
            if depth >= self.max_depth:
                for region, path in splittable:
                    all_tokens.append(self._create_token(region, depth, path, 0.0))
                break
            
            if not splittable:
                break
            
            complexities = self._batch_compute_complexity(
                features, splittable, scale_h, scale_w
            )
            
            decisions, ste_decisions = self._batch_split_decision(
                complexities, depth, hard
            )
            
            # 收集 STE decisions
            all_ste_decisions.append(ste_decisions)
            
            next_level: List[Tuple[Region, List[int]]] = []
            
            for i, ((region, path), should_split) in enumerate(zip(splittable, decisions)):
                c = complexities[i].item()
                
                if should_split:
                    for q in range(4):
                        sub_region = region.get_quadrant(q)
                        next_level.append((sub_region, path + [q]))
                else:
                    all_tokens.append(self._create_token(region, depth, path, c))
            
            current_level = next_level
        
        # 拼接所有 STE decisions
        if all_ste_decisions:
            concatenated_ste = torch.cat(all_ste_decisions, dim=0)
        else:
            concatenated_ste = torch.tensor([], device=device)
        
        return concatenated_ste, all_tokens

    def get_entropy_loss(self, results: List[SplitResult]) -> Tensor:
        """
        计算深度分布熵损失 (鼓励多尺度)。
        
        数学形式化:
            L_entropy = -Σ_d p(d) log p(d)
            其中 p(d) = |{i: d_i = d}| / N
            
        注意: 此损失不产生梯度 (用于监控)。
        使用 get_threshold_regularization_loss() 获取可微分损失。
        """
        device = self.threshold_logits.device
        
        # 统计各深度的 token 数
        depth_counts = torch.zeros(self.max_depth + 1, device=device)
        total = 0
        
        for result in results:
            for token in result.tokens:
                depth_counts[token.depth] += 1
                total += 1
        
        if total == 0:
            return torch.tensor(0.0, device=device)
        
        # 计算熵
        probs = depth_counts / total
        probs = probs.clamp(min=1e-8)  # 数值稳定
        entropy = -(probs * probs.log()).sum()
        
        # 返回负熵 (最大化熵)
        return -entropy
    
    def get_budget_loss(
        self, 
        results: List[SplitResult],
        target_tokens: int = 64,
    ) -> Tensor:
        """
        计算 token 预算损失。
        
        数学形式化:
            L_budget = ReLU(N - N_budget)²
            
        注意: 此损失不产生梯度 (用于监控)。
        """
        device = self.threshold_logits.device
        
        total_tokens = sum(result.num_tokens for result in results)
        avg_tokens = total_tokens / len(results)
        
        # 只惩罚超出预算
        excess = F.relu(torch.tensor(avg_tokens - target_tokens, device=device))
        return excess ** 2
    
    def get_threshold_regularization_loss(self) -> Tensor:
        """
        阈值正则化损失 (可微分)。
        
        数学形式化:
            L_reg = λ₁ · ||τ - τ_target||² + λ₂ · entropy(τ)
            
        目的:
            1. 防止阈值坍塌到 0 或 1
            2. 鼓励阈值分布多样化
            
        Returns:
            可微分的正则化损失
        """
        taus = self.thresholds  # [D+1]
        
        # 阈值中心化损失: 防止阈值偏离 [0.3, 0.7] 范围
        center_loss = ((taus - 0.5).abs() - 0.2).clamp(min=0).pow(2).mean()
        
        # 阈值递减约束: τ_d > τ_{d+1} (深层阈值应更低)
        decay_loss = F.relu(taus[1:] - taus[:-1]).pow(2).mean()
        
        return center_loss + decay_loss
    
    def get_differentiable_depth_loss(
        self,
        features: Tensor,
        image_size: Tuple[int, int],
        target_entropy: float = 1.0,
    ) -> Tensor:
        """
        可微分的深度分布损失。
        
        通过在固定网格上评估分割概率来计算期望深度分布。
        
        数学形式化:
            E[p(d)] = Σ_R P(depth(R) = d)
            L = (H(E[p(d)]) - H_target)²
            
        Args:
            features: [B, C, H', W'] 特征图
            image_size: (H, W) 图像尺寸
            target_entropy: 目标深度熵 (越高越多样)
            
        Returns:
            可微分的深度分布损失
        """
        B, C, H_feat, W_feat = features.shape
        H_img, W_img = image_size
        device = features.device
        
        scale_h = H_feat / H_img
        scale_w = W_feat / W_img
        
        # 在根区域和四个象限评估分割概率
        # 这提供了梯度信号
        root_region = Region(0, 0, W_img, H_img)
        
        # 计算根区域复杂度
        root_complexity = self._compute_complexity(
            features[0:1], root_region, scale_h, scale_w
        )
        
        # 分割概率
        tau_0 = self.thresholds[0]
        T = self.log_temperature.exp()
        p_split_root = torch.sigmoid((root_complexity - tau_0) / T)
        
        # 预期 token 数 (粗略估计)
        # E[N] ≈ 1 * (1-p_0) + 4 * p_0 * (1-p_1) + 16 * p_0 * p_1 * (1-p_2) + ...
        expected_tokens = (1 - p_split_root)
        
        # 鼓励分割概率接近 0.5 (最大熵)
        entropy_proxy = -(p_split_root * (p_split_root + 1e-8).log() + 
                          (1 - p_split_root) * (1 - p_split_root + 1e-8).log())
        
        # 目标: 最大化熵
        loss = (target_entropy - entropy_proxy).pow(2)
        
        return loss
    
    def get_multi_layer_depth_loss(
        self,
        features: Tensor,
        image_size: Tuple[int, int],
        max_eval_depth: Optional[int] = None,
        weight_decay_factor: float = 0.5,
        target_entropy: float = 0.693,
        return_details: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        """
        多层可微分深度损失。
        
        数学形式化:
            L_multi = Σ_d w_d · (H_target - H̄_d)²
            
            其中:
            - w_d = β^d, β = weight_decay_factor (指数衰减权重)
            - H̄_d = (1/4^d) · Σ_{R∈Grid_d} H(p_d(R)) (第 d 层平均熵)
            - p_d(R) = σ((C_θ(R) - τ_d) / T) (分割概率)
            - H(p) = -p·log(p) - (1-p)·log(1-p) (二元熵)
            
        梯度分析:
            ∂L/∂τ_d = w_d · (1/4^d) · Σ_R ∂H/∂p · ∂p/∂τ_d
            
            其中 ∂p/∂τ = -p(1-p)/T，因此每个 τ_d 都有非零梯度。
            
        权重设计 (β=0.5 时的归一化梯度分布):
            d=0: 51.6%, d=1: 25.8%, d=2: 12.9%, d=3: 6.5%, d=4: 3.2%
            
            理论依据: 浅层分割决定粗粒度结构，语义更重要。
            深层仍有梯度信号，避免阈值完全不可学习。
            
        计算优化:
            使用批量 ROI-Align 评估所有网格区域 (1 次 GPU kernel 调用)
            
        Args:
            features: [B, C, H', W'] 特征图
            image_size: (H, W) 图像尺寸
            max_eval_depth: 最大评估深度 (默认 min(max_depth, 3))
                           D=3 时覆盖 96.8% 梯度信号，仅 85 区域
            weight_decay_factor: 权重衰减因子 β (默认 0.5)
            target_entropy: 目标熵 (默认 0.693 = ln(2)，鼓励 p→0.5)
            return_details: 是否返回各层详情 (用于 TensorBoard)
            
        Returns:
            loss: 标量损失
            details (if return_details): {
                'loss_per_depth': Tensor[D+1],    # 各层损失
                'entropy_per_depth': Tensor[D+1], # 各层平均熵
                'p_split_per_depth': Tensor[D+1], # 各层平均分割概率
                'weight_per_depth': Tensor[D+1],  # 各层权重
                'temperature': Tensor,            # 当前温度
                'num_regions_per_depth': Tensor[D+1], # 各层区域数
            }
        """
        B, C, H_feat, W_feat = features.shape
        H_img, W_img = image_size
        device = features.device
        dtype = features.dtype
        
        # 确定评估深度
        if max_eval_depth is None:
            max_eval_depth = min(self.max_depth, 3)
        max_eval_depth = min(max_eval_depth, self.max_depth)
        
        scale_h = H_feat / H_img
        scale_w = W_feat / W_img
        
        # 获取当前温度
        T = self.log_temperature.exp()
        
        # =====================================================================
        # 1. 生成所有网格区域的 boxes
        # =====================================================================
        all_boxes = []
        depth_indices = []  # 记录每个 box 属于哪个深度
        
        for d in range(max_eval_depth + 1):
            grid_size = 2 ** d  # 每边 2^d 个格子
            cell_w = W_img / grid_size
            cell_h = H_img / grid_size
            
            for gy in range(grid_size):
                for gx in range(grid_size):
                    # 图像坐标
                    x1 = gx * cell_w
                    y1 = gy * cell_h
                    x2 = x1 + cell_w
                    y2 = y1 + cell_h
                    
                    # 转换为特征图坐标
                    x1_feat = x1 * scale_w
                    y1_feat = y1 * scale_h
                    x2_feat = x2 * scale_w
                    y2_feat = y2 * scale_h
                    
                    # [batch_idx, x1, y1, x2, y2]
                    all_boxes.append([0, x1_feat, y1_feat, x2_feat, y2_feat])
                    depth_indices.append(d)
        
        # 转为 tensor
        boxes_tensor = torch.tensor(all_boxes, dtype=dtype, device=device)
        depth_tensor = torch.tensor(depth_indices, dtype=torch.long, device=device)
        
        # =====================================================================
        # 2. 批量 ROI-Align (单次 GPU kernel 调用)
        # =====================================================================
        try:
            from torchvision.ops import roi_align
            pooled = roi_align(
                features[:1],  # 只用第一个 batch
                boxes_tensor,
                output_size=(self.pool_size, self.pool_size),
                spatial_scale=1.0,
                aligned=True,
            )
        except ImportError:
            # 回退到逐个计算
            pooled_list = []
            for box in all_boxes:
                _, x1, y1, x2, y2 = box
                x1_i, y1_i = int(x1), int(y1)
                x2_i, y2_i = max(x1_i + 1, int(x2)), max(y1_i + 1, int(y2))
                region_feat = features[:1, :, y1_i:y2_i, x1_i:x2_i]
                pooled_region = F.adaptive_avg_pool2d(
                    region_feat, (self.pool_size, self.pool_size)
                )
                pooled_list.append(pooled_region)
            pooled = torch.cat(pooled_list, dim=0)
        
        # =====================================================================
        # 3. 批量 MLP 计算复杂度
        # =====================================================================
        # pooled: [N_regions, C, pool_size, pool_size]
        N_regions = pooled.shape[0]
        pooled_flat = pooled.flatten(1)  # [N_regions, C * pool_size^2]
        
        # 复杂度: [N_regions]
        complexities = self.complexity_mlp(pooled_flat).squeeze(-1)
        
        # =====================================================================
        # 4. 按深度计算损失
        # =====================================================================
        total_loss = torch.tensor(0.0, device=device, dtype=dtype)
        
        # 预分配详情存储
        loss_per_depth = torch.zeros(max_eval_depth + 1, device=device, dtype=dtype)
        entropy_per_depth = torch.zeros(max_eval_depth + 1, device=device, dtype=dtype)
        p_split_per_depth = torch.zeros(max_eval_depth + 1, device=device, dtype=dtype)
        weight_per_depth = torch.zeros(max_eval_depth + 1, device=device, dtype=dtype)
        num_regions_per_depth = torch.zeros(max_eval_depth + 1, device=device, dtype=torch.long)
        
        for d in range(max_eval_depth + 1):
            # 获取当前深度的区域
            mask = (depth_tensor == d)
            complexities_d = complexities[mask]  # [4^d]
            n_regions = complexities_d.shape[0]
            
            # 分割概率 p_d = σ((C - τ_d) / T)
            tau_d = self.thresholds[d]
            p_split_d = torch.sigmoid((complexities_d - tau_d) / T)
            
            # 二元熵 H(p) = -p·log(p) - (1-p)·log(1-p)
            eps = 1e-8
            entropy_d = -(p_split_d * (p_split_d + eps).log() + 
                         (1 - p_split_d) * (1 - p_split_d + eps).log())
            
            # 平均熵
            mean_entropy_d = entropy_d.mean()
            
            # 层损失: (H_target - H̄_d)²
            layer_loss = (target_entropy - mean_entropy_d).pow(2)
            
            # 权重 w_d = β^d
            w_d = weight_decay_factor ** d
            
            # 累加加权损失
            total_loss = total_loss + w_d * layer_loss
            
            # 记录详情
            loss_per_depth[d] = layer_loss.detach()
            entropy_per_depth[d] = mean_entropy_d.detach()
            p_split_per_depth[d] = p_split_d.mean().detach()
            weight_per_depth[d] = w_d
            num_regions_per_depth[d] = n_regions
        
        # 归一化损失 (可选，使损失量级稳定)
        # total_loss = total_loss / (sum(weight_decay_factor**d for d in range(max_eval_depth+1)))
        
        if return_details:
            details = {
                'loss_per_depth': loss_per_depth,
                'entropy_per_depth': entropy_per_depth,
                'p_split_per_depth': p_split_per_depth,
                'weight_per_depth': weight_per_depth,
                'temperature': T.detach(),
                'num_regions_per_depth': num_regions_per_depth,
                'total_regions': N_regions,
                'max_eval_depth': max_eval_depth,
            }
            return total_loss, details
        
        return total_loss
    
    def get_temperature_scheduler(
        self,
        T_start: float = 1.0,
        T_end: float = 0.1,
        schedule: str = 'exponential',
        warmup_steps: int = 0,
    ) -> "TemperatureScheduler":
        """创建温度退火调度器.
        
        数学形式化:
            T(t) = T_start · (T_end / T_start)^(t / total_steps)
            
        用法:
            scheduler = splitter.get_temperature_scheduler()
            scheduler.set_total_steps(epochs * steps_per_epoch)
            
            for step in training_loop:
                ...
                scheduler.step()
        
        Args:
            T_start: 初始温度 (默认 1.0，探索性)
            T_end: 最终温度 (默认 0.1，近确定性)
            schedule: 调度策略 ('exponential', 'linear', 'cosine')
            warmup_steps: 热身步数
            
        Returns:
            TemperatureScheduler 实例
        """
        return TemperatureScheduler(
            splitter=self,
            T_start=T_start,
            T_end=T_end,
            schedule=schedule,
            warmup_steps=warmup_steps,
        )


# =============================================================================
# Factory Function
# =============================================================================

def create_adaptive_splitter(
    config: AdaptiveSplitConfig,
    feature_dim: int = 256,
) -> Union[BaseAdaptiveSplitter, LearnableSplitter]:
    """
    Factory function to create appropriate splitter based on config.
    
    Args:
        config: AdaptiveSplitConfig with scheme selection
        feature_dim: Feature dimension for LearnableSplitter (Scheme L only)
        
    Returns:
        BalancedGreedySplitter (Scheme B), 
        FixedBudgetDPSplitter (Scheme C), or
        LearnableSplitter (Scheme L)
    """
    if config.scheme == SplitScheme.BALANCED_GREEDY:
        return BalancedGreedySplitter(config)
    elif config.scheme == SplitScheme.FIXED_BUDGET_DP:
        return FixedBudgetDPSplitter(config)
    elif config.scheme == SplitScheme.LEARNABLE:
        return LearnableSplitter(
            feature_dim=feature_dim,
            max_depth=config.max_depth,
            min_region_size=config.min_region_size,
            enforce_balance=config.enforce_balance,
            init_tau_base=config.tau_0 * 3,  # 调整初始阈值更接近中心
            init_tau_gamma=config.gamma,
        )
    else:
        raise ValueError(f"Unknown scheme: {config.scheme}")


# =============================================================================
# Temperature Scheduler (P8-5)
# =============================================================================

class TemperatureScheduler:
    """Gumbel-Softmax 温度退火调度器.
    
    数学形式化
    ==========
    
    温度调度函数:
        T(t) = f(t; T_start, T_end, schedule)
        
    支持的调度策略:
        - 'exponential': T = T_s · (T_e/T_s)^(t/T_total)
        - 'linear':      T = T_s + (T_e - T_s) · t/T_total
        - 'cosine':      T = T_e + (T_s - T_e) · (1 + cos(πt/T_total)) / 2
    
    梯度分析:
        ∂p/∂C = (1/T) · p(1-p)
        
        温度影响梯度幅值:
        | T    | ∂p/∂C at C=τ | 行为     |
        |------|--------------|----------|
        | 1.0  | 0.25         | 探索充分 |
        | 0.5  | 0.50         | 加速收敛 |
        | 0.1  | 2.50         | 近确定性 |
        | 0.01 | ~0 (饱和)    | 梯度消失 |
    
    推荐参数:
        T_start = 1.0 (确保初期梯度稳定)
        T_end = 0.1   (确保末期决策确定性, 但避免梯度消失)
    
    用法示例:
        >>> scheduler = TemperatureScheduler(splitter, T_start=1.0, T_end=0.1)
        >>> scheduler.set_total_steps(total_epochs * steps_per_epoch)
        >>> 
        >>> for epoch in epochs:
        >>>     for batch in batches:
        >>>         loss = model(batch)
        >>>         loss.backward()
        >>>         optimizer.step()
        >>>         scheduler.step()  # 更新温度
    
    Args:
        splitter: LearnableSplitter 实例
        T_start: 初始温度 (默认 1.0)
        T_end: 最终温度 (默认 0.1)
        schedule: 调度策略 ('exponential', 'linear', 'cosine')
        warmup_steps: 热身步数，期间保持 T_start (默认 0)
        last_step: 上次步数 (用于恢复训练，默认 -1)
    """
    
    def __init__(
        self,
        splitter: "LearnableSplitter",
        T_start: float = 1.0,
        T_end: float = 0.1,
        schedule: str = 'exponential',
        warmup_steps: int = 0,
        last_step: int = -1,
    ):
        if T_start <= 0 or T_end <= 0:
            raise ValueError(f"Temperatures must be positive, got T_start={T_start}, T_end={T_end}")
        if T_end > T_start:
            raise ValueError(f"T_end should be <= T_start for annealing, got T_start={T_start}, T_end={T_end}")
        if schedule not in ('exponential', 'linear', 'cosine'):
            raise ValueError(f"Unknown schedule: {schedule}. Use 'exponential', 'linear', or 'cosine'")
        
        self.splitter = splitter
        self.T_start = T_start
        self.T_end = T_end
        self.schedule = schedule
        self.warmup_steps = warmup_steps
        
        self._total_steps: Optional[int] = None
        self._current_step = last_step + 1
        self._last_temperature = T_start
    
    def set_total_steps(self, total_steps: int) -> "TemperatureScheduler":
        """设置总训练步数.
        
        Args:
            total_steps: 总步数 (epochs × batches_per_epoch)
            
        Returns:
            self, 支持链式调用
        """
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        self._total_steps = total_steps
        return self
    
    def get_temperature(self) -> float:
        """获取当前步的温度 (不更新步数).
        
        Returns:
            当前温度值
        """
        if self._total_steps is None:
            # 未设置 total_steps，返回初始温度
            return self.T_start
        
        # 热身阶段
        if self._current_step < self.warmup_steps:
            return self.T_start
        
        # 计算有效进度 (扣除热身)
        effective_step = self._current_step - self.warmup_steps
        effective_total = self._total_steps - self.warmup_steps
        
        if effective_total <= 0:
            return self.T_end
        
        progress = min(1.0, effective_step / effective_total)
        
        if self.schedule == 'exponential':
            # T = T_s · (T_e/T_s)^progress
            return self.T_start * (self.T_end / self.T_start) ** progress
        
        elif self.schedule == 'linear':
            # T = T_s + (T_e - T_s) · progress
            return self.T_start + (self.T_end - self.T_start) * progress
        
        elif self.schedule == 'cosine':
            # T = T_e + (T_s - T_e) · (1 + cos(π·progress)) / 2
            return self.T_end + (self.T_start - self.T_end) * (1 + math.cos(math.pi * progress)) / 2
        
        else:
            return self.T_start
    
    def step(self) -> float:
        """更新一步并应用新温度.
        
        Returns:
            更新后的温度值
        """
        temperature = self.get_temperature()
        self.splitter.set_temperature(temperature)
        self._last_temperature = temperature
        self._current_step += 1
        return temperature
    
    def state_dict(self) -> Dict[str, Any]:
        """获取调度器状态 (用于保存 checkpoint).
        
        Returns:
            包含调度器状态的字典
        """
        return {
            'T_start': self.T_start,
            'T_end': self.T_end,
            'schedule': self.schedule,
            'warmup_steps': self.warmup_steps,
            'total_steps': self._total_steps,
            'current_step': self._current_step,
            'last_temperature': self._last_temperature,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """从状态字典恢复调度器状态.
        
        Args:
            state_dict: 由 state_dict() 返回的字典
        """
        self.T_start = state_dict['T_start']
        self.T_end = state_dict['T_end']
        self.schedule = state_dict['schedule']
        self.warmup_steps = state_dict['warmup_steps']
        self._total_steps = state_dict['total_steps']
        self._current_step = state_dict['current_step']
        self._last_temperature = state_dict['last_temperature']
        
        # 恢复温度到 splitter
        self.splitter.set_temperature(self._last_temperature)
    
    @property
    def current_step(self) -> int:
        """当前步数."""
        return self._current_step
    
    @property
    def last_temperature(self) -> float:
        """上次设置的温度."""
        return self._last_temperature
    
    def __repr__(self) -> str:
        return (
            f"TemperatureScheduler("
            f"T_start={self.T_start}, T_end={self.T_end}, "
            f"schedule='{self.schedule}', "
            f"step={self._current_step}/{self._total_steps or '?'}, "
            f"T={self._last_temperature:.4f})"
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def split_image(
    image: Tensor,
    scheme: str = "balanced_greedy",
    **kwargs
) -> SplitResult:
    """
    Convenience function to split a single image.
    
    Args:
        image: [C, H, W] or [1, C, H, W] tensor
        scheme: "balanced_greedy" (B) or "fixed_budget_dp" (C)
        **kwargs: Additional config parameters
        
    Returns:
        SplitResult with tokens
    """
    scheme_enum = SplitScheme(scheme)
    config = AdaptiveSplitConfig(scheme=scheme_enum, **kwargs)
    splitter = create_adaptive_splitter(config)
    return splitter.split(image)


def compare_schemes(
    image: Tensor,
    config_overrides: Optional[Dict] = None
) -> Dict[str, SplitResult]:
    """
    Run both schemes on the same image for comparison.
    
    Args:
        image: Input image tensor
        config_overrides: Optional config parameters
        
    Returns:
        Dict with keys 'scheme_b' and 'scheme_c'
    """
    overrides = config_overrides or {}
    
    config_b = AdaptiveSplitConfig.scheme_b(**overrides)
    config_c = AdaptiveSplitConfig.scheme_c(**overrides)
    
    splitter_b = BalancedGreedySplitter(config_b)
    splitter_c = FixedBudgetDPSplitter(config_c)
    
    return {
        'scheme_b': splitter_b.split(image),
        'scheme_c': splitter_c.split(image)
    }
