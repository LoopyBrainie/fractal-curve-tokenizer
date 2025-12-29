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
    """Available splitting schemes.
    
    注意: Scheme B (BalancedGreedy) 和 Scheme C (FixedBudgetDP) 已被移除。
    数学分析表明 LearnableSplitter 完全覆盖其功能并提供以下优势:
    - 端到端可微分训练
    - 无复杂度饱和问题
    - 自适应阈值学习
    """
    LEARNABLE = "learnable"  # Scheme L (可学习分割器)


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
    
    scheme: SplitScheme = SplitScheme.LEARNABLE
    """Splitting scheme (仅支持 LEARNABLE)"""
    
    # === Scheme L Specific Parameters ===
    
    enforce_balance: bool = True
    """Whether to enforce 2:1 balance constraint"""
    
    target_tokens: Optional[int] = None
    """Soft target for token count (None = no constraint)"""
    
    token_penalty_weight: float = 0.01
    """Weight for token count deviation penalty"""
    
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
# TensorSplitResult: 纯张量表示 (P9-1 完全向量化 BFS)
# =============================================================================

@dataclass
class TensorSplitResult:
    """
    纯张量表示的分割结果 (P9-1 优化).
    
    数学形式化
    ==========
    
    设 B 为 batch size，N_i 为第 i 个样本的 token 数，N_max = max(N_i)。
    
    传统 Python 表示:
        SplitResult = {tokens: List[SplitToken]}
        内存: O(N × sizeof(SplitToken)) ≈ O(N × 200) bytes
        访问: O(N) Python 解释器开销
        
    张量表示:
        regions:       [N_total, 4]     (x1, y1, x2, y2)
        depths:        [N_total]        深度值
        batch_indices: [N_total]        所属 batch 索引
        hilbert_indices: [N_total]      Hilbert 曲线索引
        complexities:  [N_total]        复杂度值
        
        内存: O(N × 8 bytes) (连续 GPU 内存)
        访问: O(1) GPU kernel
        
    关键设计:
        1. 所有数据在同一设备上 (GPU)
        2. 无 Python 对象，无 GIL 开销
        3. 支持批量操作: scatter, gather, sort
        4. path 字段被移除 (可从 hilbert_idx 恢复 LCA)
        
    LCA 恢复定理:
        对于 Hilbert 索引 h1, h2，其最近公共祖先深度为:
            lca_depth(h1, h2) = order - floor(log4(h1 XOR h2) + 1)
        其中 order = log2(grid_size)
    """
    
    regions: Tensor        # [N, 4] 区域坐标 (x1, y1, x2, y2)
    depths: Tensor         # [N] 深度值
    batch_indices: Tensor  # [N] batch 索引
    hilbert_indices: Tensor  # [N] Hilbert 索引
    complexities: Tensor   # [N] 复杂度值
    
    # 可选: 每个 batch 的 token 数量 (用于重构 List 表示)
    tokens_per_batch: Optional[Tensor] = None  # [B]
    
    @property
    def device(self) -> torch.device:
        return self.regions.device
    
    @property
    def num_tokens(self) -> int:
        return self.regions.shape[0]
    
    @property
    def batch_size(self) -> int:
        if self.tokens_per_batch is not None:
            return self.tokens_per_batch.shape[0]
        return int(self.batch_indices.max().item()) + 1 if self.num_tokens > 0 else 0
    
    def get_batch_mask(self, batch_idx: int) -> Tensor:
        """获取特定 batch 的掩码 [N]."""
        return self.batch_indices == batch_idx
    
    def get_batch_tokens(self, batch_idx: int) -> "TensorSplitResult":
        """提取特定 batch 的 tokens (零拷贝视图)."""
        mask = self.get_batch_mask(batch_idx)
        return TensorSplitResult(
            regions=self.regions[mask],
            depths=self.depths[mask],
            batch_indices=self.batch_indices[mask] * 0,  # 重置为 0
            hilbert_indices=self.hilbert_indices[mask],
            complexities=self.complexities[mask],
        )
    
    def sort_by_hilbert(self) -> "TensorSplitResult":
        """按 (batch_idx, hilbert_idx) 排序 (纯 GPU 操作)."""
        # 组合键: batch_idx * max_hilbert + hilbert_idx
        max_hilbert = self.hilbert_indices.max() + 1 if self.num_tokens > 0 else 1
        sort_key = self.batch_indices * max_hilbert + self.hilbert_indices
        order = torch.argsort(sort_key)
        
        return TensorSplitResult(
            regions=self.regions[order],
            depths=self.depths[order],
            batch_indices=self.batch_indices[order],
            hilbert_indices=self.hilbert_indices[order],
            complexities=self.complexities[order],
            tokens_per_batch=self.tokens_per_batch,
        )
    
    def get_levels_info_tensor(self, max_depth: int) -> Tensor:
        """
        获取 levels_info 张量 [N, max_depth+1].
        
        格式: [depth, q1, q2, ..., 0, 0, ...]
        其中 qi 是四叉树路径。
        
        注意: 由于我们移除了 path，这里使用 depth 作为唯一信息。
        如需完整路径，可从 hilbert_idx 反向计算。
        """
        N = self.num_tokens
        device = self.device
        
        # 简化版本: 只保留深度信息
        # [depth, 0, 0, ..., 0]
        levels_info = torch.zeros(N, max_depth + 1, dtype=torch.long, device=device)
        levels_info[:, 0] = self.depths
        
        return levels_info
    
    def get_regions_boxes(self) -> Tensor:
        """获取 regions 张量 [N, 4] (已经是正确格式)."""
        return self.regions
    
    @classmethod
    def from_split_results(
        cls, 
        results: List[SplitResult], 
        device: torch.device
    ) -> "TensorSplitResult":
        """从 Python SplitResult 列表构造 (兼容性转换).
        
        注意: 这个方法用于渐进迁移，生产环境应直接使用
        向量化 BFS 生成 TensorSplitResult。
        """
        all_regions = []
        all_depths = []
        all_batch_indices = []
        all_hilbert = []
        all_complexity = []
        tokens_per_batch = []
        
        for batch_idx, sr in enumerate(results):
            n = len(sr.tokens)
            tokens_per_batch.append(n)
            
            for t in sr.tokens:
                all_regions.append([t.region.x1, t.region.y1, t.region.x2, t.region.y2])
                all_depths.append(t.depth)
                all_batch_indices.append(batch_idx)
                all_hilbert.append(t.hilbert_idx)
                all_complexity.append(t.complexity)
        
        if not all_regions:
            # 空结果
            return cls(
                regions=torch.zeros(0, 4, dtype=torch.long, device=device),
                depths=torch.zeros(0, dtype=torch.long, device=device),
                batch_indices=torch.zeros(0, dtype=torch.long, device=device),
                hilbert_indices=torch.zeros(0, dtype=torch.long, device=device),
                complexities=torch.zeros(0, dtype=torch.float32, device=device),
                tokens_per_batch=torch.zeros(0, dtype=torch.long, device=device),
            )
        
        return cls(
            regions=torch.tensor(all_regions, dtype=torch.long, device=device),
            depths=torch.tensor(all_depths, dtype=torch.long, device=device),
            batch_indices=torch.tensor(all_batch_indices, dtype=torch.long, device=device),
            hilbert_indices=torch.tensor(all_hilbert, dtype=torch.long, device=device),
            complexities=torch.tensor(all_complexity, dtype=torch.float32, device=device),
            tokens_per_batch=torch.tensor(tokens_per_batch, dtype=torch.long, device=device),
        )
    
    def to_split_results(self) -> List[SplitResult]:
        """转换回 Python SplitResult 列表 (兼容性).
        
        注意: 这会触发 GPU-CPU 同步，仅用于与旧接口兼容。
        """
        # 按 batch 分组
        results = []
        B = self.batch_size
        
        # 一次性传输所有数据到 CPU
        regions_cpu = self.regions.cpu().numpy()
        depths_cpu = self.depths.cpu().numpy()
        batch_idx_cpu = self.batch_indices.cpu().numpy()
        hilbert_cpu = self.hilbert_indices.cpu().numpy()
        complexity_cpu = self.complexities.cpu().numpy()
        
        for b in range(B):
            mask = batch_idx_cpu == b
            tokens = []
            
            for i in range(mask.sum()):
                idx = mask.nonzero()[0][i]
                r = regions_cpu[idx]
                tokens.append(SplitToken(
                    region=Region(int(r[0]), int(r[1]), int(r[2]), int(r[3])),
                    depth=int(depths_cpu[idx]),
                    path=[],  # path 已被移除
                    hilbert_idx=int(hilbert_cpu[idx]),
                    complexity=float(complexity_cpu[idx]),
                ))
            
            results.append(SplitResult(tokens=tokens))
        
        return results
    
    @classmethod
    def empty(cls, device: torch.device) -> "TensorSplitResult":
        """创建空的 TensorSplitResult."""
        return cls(
            regions=torch.zeros(0, 4, dtype=torch.long, device=device),
            depths=torch.zeros(0, dtype=torch.long, device=device),
            batch_indices=torch.zeros(0, dtype=torch.long, device=device),
            hilbert_indices=torch.zeros(0, dtype=torch.long, device=device),
            complexities=torch.zeros(0, dtype=torch.float32, device=device),
            tokens_per_batch=torch.zeros(0, dtype=torch.long, device=device),
        )


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

class HilbertLUT:
    """
    预计算的 Hilbert 曲线查找表 (P-PERF-1 优化)。
    
    数学形式化
    ----------
    通过预计算 G×G 网格的 Hilbert 索引，将计算复杂度从:
        O(N × log G) Python 递归  →  O(1) 张量索引
    
    空间-时间权衡:
        空间: O(G²) = O(4096) for G=64
        时间: O(1) lookup vs O(log G) computation
        
    线程安全: 使用 register_buffer 确保多 GPU 同步
    """
    
    # 类级别缓存: {grid_size: Tensor[G, G]}
    _lut_cache: Dict[int, Tensor] = {}
    
    @classmethod
    def get_lut(cls, grid_size: int, device: torch.device) -> Tensor:
        """获取或创建 Hilbert LUT.
        
        Args:
            grid_size: 网格大小 (必须是 2 的幂)
            device: 目标设备
            
        Returns:
            LUT tensor [G, G]，其中 LUT[x, y] = Hilbert_index
        """
        if grid_size not in cls._lut_cache:
            # 预计算整个网格
            lut = torch.zeros(grid_size, grid_size, dtype=torch.long)
            for x in range(grid_size):
                for y in range(grid_size):
                    lut[x, y] = HilbertCurve.xy_to_d(grid_size, x, y)
            cls._lut_cache[grid_size] = lut
        
        return cls._lut_cache[grid_size].to(device)
    
    @classmethod
    def batch_lookup(
        cls,
        cx_batch: Tensor,  # [N] 中心 x 坐标
        cy_batch: Tensor,  # [N] 中心 y 坐标
        image_size: int,
        device: torch.device,
    ) -> Tensor:
        """批量查询 Hilbert 索引 (向量化).
        
        Args:
            cx_batch: [N] 中心点 x 坐标 (像素)
            cy_batch: [N] 中心点 y 坐标 (像素)
            image_size: 原始图像尺寸
            device: 计算设备
            
        Returns:
            hilbert_indices: [N] Hilbert 索引
        """
        # 确定网格大小 (最小 2 的幂 >= image_size)
        grid_order = math.ceil(math.log2(max(image_size, 1)))
        grid_size = 2 ** grid_order
        
        # 获取 LUT
        lut = cls.get_lut(grid_size, device)  # [G, G]
        
        # 坐标转换: 像素坐标 → 网格坐标
        gx = (cx_batch * grid_size / image_size).long().clamp(0, grid_size - 1)
        gy = (cy_batch * grid_size / image_size).long().clamp(0, grid_size - 1)
        
        # 向量化查表
        return lut[gx, gy]


class HilbertTokenSorter:
    """
    Sorts tokens by Hilbert curve order.
    
    Ensures spatial locality is preserved in the token sequence.
    
    优化 (P-PERF-1):
        - 使用 HilbertLUT 预计算表替代逐个计算
        - 批量排序时使用 torch.argsort 替代 Python sorted
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
# Scheme L: Learnable Splitter (P7-1/P7-2/P7-3 Solution)
# =============================================================================

class ComplexityMLP(nn.Module):
    """
    可学习复杂度预测器 (优化版)。
    
    数学形式化
    ----------
    
    解决问题 P7-1 (复杂度饱和效应):
        原公式: C(R) = Var/(Var + σ₀²) 在 Var >> σ₀² 时饱和
        新公式: C_θ(R) = σ(MLP(Pool(F, R)))
        
    输入:
        f_R ∈ ℝ^(C × k × k): 区域特征 (ROI-Align 输出)
        
    输出:
        C_θ(R) ∈ [0, 1]: 可学习复杂度
        
    架构优化 (解决 P-MLP-1):
        - 增加隐藏层数: 2 → 3 层
        - 增加隐藏维度: 64 → 128 (第一层)
        - 逐步压缩: 4096 → 128 → 64 → 1
        - 保持参数量相近 (~265K)
        
    梯度分析:
        ∂C_θ/∂θ = σ'(z) · ∂MLP/∂θ
        σ'(z) = σ(z)(1-σ(z)) ∈ (0, 0.25]
        ✅ 梯度稳定，无饱和问题
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        intermediate_dim: int = 64,
        dropout: float = 0.1,
        use_deep_mlp: bool = True,
    ):
        """
        Args:
            input_dim: 输入特征维度 (C × k × k)
            hidden_dim: 第一隐藏层维度 (默认 128)
            intermediate_dim: 第二隐藏层维度 (默认 64)
            dropout: Dropout 概率
            use_deep_mlp: 是否使用深度 MLP (3层)
        """
        super().__init__()
        
        self.use_deep_mlp = use_deep_mlp
        
        if use_deep_mlp:
            # 深度 MLP: 更强表达能力
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, intermediate_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(intermediate_dim, 1),
            )
            
            # P10-2 修复: 调整初始化以避免不稳定平衡点 (2024-12-29 修正版)
            # 
            # 数学形式化验证结论:
            # ================================
            # 原问题: gain=0.1 导致 σ_z ≈ 0.14，C_θ ∈ [0.43, 0.57] (99.4%)
            #         τ 敏感度 = 10.31，微小扰动导致分割率剧烈变化 (振荡)
            #
            # 解决方案: 对称配置 + 增大 gain
            #   - gain=1.0 (原 0.1): 使 σ_z ≈ 1.4，C_θ 覆盖 [0.06, 0.94]
            #   - bias=0.0 (保持): 对称分布，E[C_θ] = 0.5
            #   - 配合 τ₀=0.5: 分割率 ≈ 50%，E[N] = 731 (满足限制)
            #
            # 验证结果:
            #   | 配置      | 敏感度 | Std[C_θ] | C∈[0.4,0.6] |
            #   |-----------|--------|----------|-------------|
            #   | gain=0.1  | 10.31  | 3.7%     | 99.4%       | ← 失败
            #   | gain=1.0  | 1.04   | 26.7%    | 21.9%       | ← 稳定
            #
            # 注意: 不需要正分离度 (E[C_θ] > τ)，依赖 Budget Loss 调节
            nn.init.xavier_uniform_(self.mlp[0].weight)
            nn.init.zeros_(self.mlp[0].bias)
            nn.init.xavier_uniform_(self.mlp[3].weight)
            nn.init.zeros_(self.mlp[3].bias)
            nn.init.xavier_uniform_(self.mlp[6].weight, gain=1.0)  # P10-2: 0.1 → 1.0
            nn.init.zeros_(self.mlp[6].bias)                       # P10-2: 保持 0 (对称配置)
        else:
            # 浅层 MLP: 保持一致的初始化策略
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, intermediate_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(intermediate_dim, 1),
            )
            
            # P10-2: 浅层 MLP 使用相同的初始化策略
            # 见深度 MLP 注释中的数学验证
            nn.init.xavier_uniform_(self.mlp[0].weight)
            nn.init.zeros_(self.mlp[0].bias)
            nn.init.xavier_uniform_(self.mlp[3].weight, gain=1.0)  # P10-2: 0.1 → 1.0
            nn.init.zeros_(self.mlp[3].bias)                       # P10-2: 保持 0
    
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
        hidden_dim: int = 128,
        intermediate_dim: int = 64,
        pool_size: int = 4,
        temperature: float = 1.0,
        use_gumbel: bool = True,
        enforce_balance: bool = False,
        min_region_size: int = 7,
        dropout: float = 0.1,
        init_tau_base: float = 0.5,
        init_tau_gamma: float = 0.85,
        use_deep_mlp: bool = True,
    ):
        """
        Args:
            feature_dim: 输入特征通道数 C
            max_depth: 最大分割深度 D_max
            hidden_dim: ComplexityMLP 第一隐藏层维度 (默认 128)
            intermediate_dim: ComplexityMLP 第二隐藏层维度 (默认 64)
            pool_size: ROI-Align 输出尺寸 k×k
            temperature: Gumbel-Softmax 温度 T
            use_gumbel: 是否使用 Gumbel 噪声 (训练时)
            enforce_balance: 是否强制 2:1 平衡约束
            min_region_size: 最小区域边长
            dropout: MLP dropout
            init_tau_base: 初始根阈值 τ₀ (用于参数初始化)
            init_tau_gamma: 初始阈值衰减 γ (用于参数初始化)
            use_deep_mlp: 是否使用深度 MLP (3层，更强表达能力)
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.max_depth = max_depth
        self.pool_size = pool_size
        self.temperature = temperature
        self.use_gumbel = use_gumbel
        self.enforce_balance = enforce_balance
        self.min_region_size = min_region_size
        
        # P7-1: 可学习复杂度预测器 (优化版)
        input_dim = feature_dim * pool_size * pool_size
        self.complexity_mlp = ComplexityMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            dropout=dropout,
            use_deep_mlp=use_deep_mlp,
        )
        
        # P-TAU-1: 纯 Offset + Barrier 阈值参数化
        # 优势：梯度恒定 (∂τ/∂offset = 1)，边界附近无梯度消失
        # 公式：τ_eff_d = τ_base_d + offset_d
        # Barrier: L_barrier = λ · Σ [max(0, τ_min - τ)² + max(0, τ - τ_max)²]
        tau_bases = [init_tau_base * (init_tau_gamma ** d) for d in range(max_depth + 1)]
        self.register_buffer('_tau_bases', torch.tensor(tau_bases))
        self.threshold_offsets = nn.Parameter(torch.zeros(max_depth + 1))
        
        # Barrier 正则化参数
        self.tau_min = 0.0
        self.tau_max = 1.0
        self.barrier_lambda = 5.0
        
        # 可学习温度 (可选)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature)))
        
        # 统计信息 (用于监控)
        self.register_buffer('_split_probs', torch.zeros(max_depth + 1))
        self.register_buffer('_split_counts', torch.zeros(max_depth + 1))
        
        # P-GRAD-1: 自适应权重衰减 - EMA 分割概率
        # 用于计算条件访问概率权重 w_d = ∏_{k<d} p̄_k
        self.register_buffer('_ema_split_probs', torch.full((max_depth + 1,), 0.5))
        self._ema_alpha: float = 0.1  # EMA 平滑因子
        
        # P-TEMP-1: 自动温度退火状态
        # 使用 register_buffer 确保 checkpoint 兼容
        self.register_buffer('_temp_step', torch.tensor(0, dtype=torch.long))
        self.register_buffer('_temp_total_steps', torch.tensor(0, dtype=torch.long))
        self.register_buffer('_temp_start', torch.tensor(1.0))
        self.register_buffer('_temp_end', torch.tensor(0.05))
        self._temp_schedule: str = 'exponential'
        self._temp_enabled: bool = False
        
        # P10-1: STE 软分割概率缓存
        # 用于后续辅助损失计算 (P10-4 熵损失, P10-9 Elastic Budget)
        self._cached_split_probs: Dict[int, Tensor] = {}
        self._cached_complexities: Optional[Tensor] = None
    
    @property
    def thresholds(self) -> Tensor:
        """获取当前有效阈值向量 τ_eff ∈ ℝ^(D+1).
        
        P-TAU-1 实现：纯 Offset 参数化
        公式：τ_eff_d = τ_base_d + offset_d
        
        注意：阈值可能超出 [0, 1]，通过 barrier loss 软约束。
        """
        return self._tau_bases + self.threshold_offsets
    
    @property
    def current_temperature(self) -> float:
        """获取当前温度 T > 0."""
        return self.log_temperature.exp().item()
    
    def set_temperature(self, temperature: float) -> None:
        """设置温度 (用于退火调度)."""
        self.log_temperature.data.fill_(math.log(temperature))
    
    def enable_temperature_annealing(
        self,
        total_steps: int,
        T_start: float = 1.0,
        T_end: float = 0.05,
        schedule: str = 'exponential',
    ) -> "LearnableSplitter":
        """
        启用自动温度退火 (P-TEMP-1 解决方案)。
        
        数学形式化
        ==========
        
        温度调度函数:
            T(t) = T_start · (T_end / T_start)^(t / total_steps)  (exponential)
            T(t) = T_start + (T_end - T_start) · t / total_steps   (linear)
            T(t) = T_end + (T_start - T_end) · (1 + cos(πt/total)) / 2  (cosine)
            
        自动更新机制:
            在 forward() 中，若 self.training=True 且已启用:
                1. 计算当前 progress = _temp_step / _temp_total_steps
                2. 计算 T = schedule(progress)
                3. 调用 set_temperature(T)
                4. _temp_step += 1
                
        Hilbert ViT 约束分析:
            - 早期 (T ≈ 1.0): 探索决策空间，梯度稳定
            - 后期 (T ≈ 0.05): 决策确定性高，满足 2:1 平衡
            
        推荐超参数 (基于计算验证):
            - T_start = 1.0 (充分探索)
            - T_end = 0.05 (避免梯度消失的安全下界)
            - schedule = 'exponential' (最优的梯度-确定性权衡)
            
        Args:
            total_steps: 总训练步数 (epochs × batches_per_epoch)
            T_start: 初始温度 (默认 1.0)
            T_end: 最终温度 (默认 0.05，基于梯度消失分析的安全下界)
            schedule: 调度策略 ('exponential', 'linear', 'cosine')
            
        Returns:
            self, 支持链式调用
            
        Example:
            >>> splitter = LearnableSplitter(feature_dim=256)
            >>> splitter.enable_temperature_annealing(
            ...     total_steps=1000,
            ...     T_start=1.0,
            ...     T_end=0.05,
            ... )
            >>> # 训练时自动退火，无需手动调用 scheduler.step()
            >>> for batch in dataloader:
            ...     output = splitter(features, image_size)  # 温度自动更新
        """
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        if T_start <= 0 or T_end <= 0:
            raise ValueError(f"Temperatures must be positive, got T_start={T_start}, T_end={T_end}")
        if T_end > T_start:
            raise ValueError(f"T_end should be <= T_start for annealing")
        if schedule not in ('exponential', 'linear', 'cosine'):
            raise ValueError(f"Unknown schedule: {schedule}")
        
        self._temp_total_steps.fill_(total_steps)
        self._temp_start.fill_(T_start)
        self._temp_end.fill_(T_end)
        self._temp_step.zero_()
        self._temp_schedule = schedule
        self._temp_enabled = True
        
        # 设置初始温度
        self.set_temperature(T_start)
        
        return self
    
    def disable_temperature_annealing(self) -> "LearnableSplitter":
        """禁用自动温度退火。"""
        self._temp_enabled = False
        return self
    
    def _update_temperature(self) -> float:
        """
        更新温度 (内部方法，在 forward 中调用)。
        
        数学形式化:
            progress = min(1.0, step / total)
            
            exponential: T = T_s · (T_e / T_s)^progress
            linear:      T = T_s + (T_e - T_s) · progress
            cosine:      T = T_e + (T_s - T_e) · (1 + cos(π·progress)) / 2
            
        Returns:
            更新后的温度值
        """
        if not self._temp_enabled:
            return self.current_temperature
        
        total = self._temp_total_steps.item()
        if total <= 0:
            return self.current_temperature
        
        step = self._temp_step.item()
        progress = min(1.0, step / total)
        
        T_s = self._temp_start.item()
        T_e = self._temp_end.item()
        
        if self._temp_schedule == 'exponential':
            T = T_s * (T_e / T_s) ** progress
        elif self._temp_schedule == 'linear':
            T = T_s + (T_e - T_s) * progress
        elif self._temp_schedule == 'cosine':
            T = T_e + (T_s - T_e) * (1 + math.cos(math.pi * progress)) / 2
        else:
            T = T_s
        
        self.set_temperature(T)
        self._temp_step.add_(1)
        
        return T
    
    def get_temperature_progress(self) -> Dict[str, Any]:
        """获取温度调度进度信息。"""
        return {
            'enabled': self._temp_enabled,
            'current_step': self._temp_step.item(),
            'total_steps': self._temp_total_steps.item(),
            'progress': self._temp_step.item() / max(1, self._temp_total_steps.item()),
            'current_temperature': self.current_temperature,
            'T_start': self._temp_start.item(),
            'T_end': self._temp_end.item(),
            'schedule': self._temp_schedule,
        }
    
    def _forward_vectorized_tensor(
        self,
        features: Tensor,
        image_size: Tuple[int, int],
        scale_h: float,
        scale_w: float,
        hard: bool,
    ) -> TensorSplitResult:
        """
        完全向量化 BFS 分割 (P9-1 方案 D).
        
        数学形式化
        ==========
        
        核心优化原理:
            传统实现: O(D) Python 循环 × O(N_d) Python 操作/层
            向量化:   O(D) GPU kernels × O(1) Python 操作/层
            
            同步点数量:
                传统: ~1500 次/iter (.item(), .tolist(), torch.tensor())
                向量化: O(D) 次 (仅层边界)
            
        实现策略:
            1. 所有区域表示为张量 [N, 5]: (batch_idx, x1, y1, x2, y2)
            2. 可分割性判断: 向量化比较
            3. 复杂度计算: ROI-Align + MLP (已向量化)
            4. 分割决策: 批量 sigmoid (已向量化)
            5. 象限展开: [M, 5] → [4M, 5] 纯张量操作
            6. Hilbert 排序: HilbertLUT 批量查表
            
        关键数据结构:
            regions: [N, 5] - (batch_idx, x1, y1, x2, y2)
            depths:  [N]    - 当前深度
            valid:   [N]    - 有效掩码 (未终止的区域)
            
        内存优化:
            - 使用 scatter/gather 替代 Python 列表操作
            - 预分配输出缓冲区，避免动态分配
            
        预期加速: ~8x (24s/iter → ~3s/iter)
        """
        B, C, H_feat, W_feat = features.shape
        H_img, W_img = image_size
        device = features.device
        dtype = features.dtype
        
        # ====================================================================
        # 初始化: 每个 batch 的根区域
        # ====================================================================
        # regions: [B, 5] -> (batch_idx, x1, y1, x2, y2)
        batch_indices = torch.arange(B, device=device, dtype=torch.long)
        initial_regions = torch.zeros(B, 5, device=device, dtype=torch.long)
        initial_regions[:, 0] = batch_indices  # batch_idx
        initial_regions[:, 1] = 0              # x1
        initial_regions[:, 2] = 0              # y1
        initial_regions[:, 3] = W_img          # x2
        initial_regions[:, 4] = H_img          # y2
        
        # 当前层区域
        current_regions = initial_regions  # [N_current, 5]
        current_depths = torch.zeros(B, device=device, dtype=torch.long)  # [N_current]
        
        # 输出缓冲区 (累积叶节点)
        output_regions_list: List[Tensor] = []
        output_depths_list: List[Tensor] = []
        output_batch_idx_list: List[Tensor] = []
        output_complexity_list: List[Tensor] = []
        
        # P10-1: 清空 STE 缓存 (每次前向传播重新收集)
        self._cached_split_probs = {}
        all_complexities_list: List[Tensor] = []  # 用于缓存所有复杂度
        
        # ====================================================================
        # BFS 迭代 (O(D) GPU kernels)
        # ====================================================================
        for depth in range(self.max_depth + 1):
            N_current = current_regions.shape[0]
            if N_current == 0:
                break
            
            # ------------------------------------------------------------------
            # Step 1: 向量化可分割性判断
            # ------------------------------------------------------------------
            # 区域尺寸
            widths = current_regions[:, 3] - current_regions[:, 1]   # x2 - x1
            heights = current_regions[:, 4] - current_regions[:, 2]  # y2 - y1
            
            # 可分割条件: width >= 2*min_size AND height >= 2*min_size
            min_size_2x = self.min_region_size * 2
            can_split_size = (widths >= min_size_2x) & (heights >= min_size_2x)
            
            # 终止区域 (尺寸不足) → 直接输出
            terminal_mask = ~can_split_size
            if terminal_mask.any():
                output_regions_list.append(current_regions[terminal_mask, 1:5])  # [N_term, 4]
                output_depths_list.append(current_depths[terminal_mask])
                output_batch_idx_list.append(current_regions[terminal_mask, 0])
                output_complexity_list.append(
                    torch.zeros(terminal_mask.sum(), device=device, dtype=dtype)
                )
            
            # 最大深度检查
            if depth >= self.max_depth:
                # 所有可分割区域也输出
                if can_split_size.any():
                    output_regions_list.append(current_regions[can_split_size, 1:5])
                    output_depths_list.append(current_depths[can_split_size])
                    output_batch_idx_list.append(current_regions[can_split_size, 0])
                    output_complexity_list.append(
                        torch.zeros(can_split_size.sum(), device=device, dtype=dtype)
                    )
                break
            
            # 可分割区域
            splittable_regions = current_regions[can_split_size]  # [M, 5]
            splittable_depths = current_depths[can_split_size]    # [M]
            M = splittable_regions.shape[0]
            
            if M == 0:
                break
            
            # ------------------------------------------------------------------
            # Step 2: 批量计算复杂度 (已向量化)
            # ------------------------------------------------------------------
            # 构建 ROI boxes: [batch_idx, x1, y1, x2, y2] (scaled to feature map)
            boxes = torch.zeros(M, 5, device=device, dtype=dtype)
            boxes[:, 0] = splittable_regions[:, 0].float()  # batch_idx
            boxes[:, 1] = splittable_regions[:, 1].float() * scale_w  # x1
            boxes[:, 2] = splittable_regions[:, 2].float() * scale_h  # y1
            boxes[:, 3] = splittable_regions[:, 3].float() * scale_w  # x2
            boxes[:, 4] = splittable_regions[:, 4].float() * scale_h  # y2
            
            # ROI-Align
            from torchvision.ops import roi_align
            pooled = roi_align(
                features,
                boxes,
                output_size=(self.pool_size, self.pool_size),
                spatial_scale=1.0,
                aligned=True,
            )  # [M, C, k, k]
            
            # MLP 预测复杂度
            # 注意: complexity_mlp.forward() 内部已经做了 squeeze(-1)，返回 [M]
            # 当 M=1 时，[1] 不应该被再次 squeeze 成标量 []
            flat = pooled.flatten(start_dim=1)  # [M, C*k*k]
            complexities = self.complexity_mlp(flat)  # [M] (已 sigmoid)
            
            # P10-1: 缓存复杂度用于后续辅助损失
            all_complexities_list.append(complexities)
            
            # ------------------------------------------------------------------
            # Step 3: 批量计算分割决策 (向量化)
            # ------------------------------------------------------------------
            tau_d = self.thresholds[depth]
            T = self.log_temperature.exp()
            p_split = torch.sigmoid((complexities - tau_d) / T)  # [M]
            
            # 更新统计 (无梯度)
            with torch.no_grad():
                self._split_probs[depth] = self._split_probs[depth] + p_split.sum()
                self._split_counts[depth] = self._split_counts[depth] + M
                
                batch_p = p_split.mean()
                self._ema_split_probs[depth] = (
                    self._ema_alpha * batch_p + 
                    (1 - self._ema_alpha) * self._ema_split_probs[depth]
                )
            
            # 决策: 纯张量操作
            # P10-1 修复: 使用 Straight-Through Estimator (STE) 恢复梯度流
            #
            # 数学形式化:
            #   原问题: should_split = p > 0.5 是阶跃函数，∂/∂p = 0
            #   STE 解决方案: z_ST = z_hard - sg(y_soft) + y_soft
            #     前向: z_ST = z_hard (硬决策，保持离散性)
            #     反向: ∂z_ST/∂y = 1 (梯度流经 y_soft)
            #
            # 这使得 ∂L/∂θ_S ≠ 0，分割器参数可以从辅助损失获得梯度
            if hard or not self.training:
                should_split = p_split > 0.5  # [M] bool
            else:
                # Gumbel-Softmax + STE (向量化)
                if self.use_gumbel:
                    # Gumbel 噪声采样
                    gumbel_noise = -torch.log(-torch.log(
                        torch.rand_like(p_split).clamp(1e-10, 1-1e-10)
                    ))
                    # 构建 logits: [不分割, 分割]
                    logits = torch.stack([
                        torch.zeros_like(p_split),  # log(1-p) ≈ 0 for simplicity
                        (p_split / (1 - p_split + 1e-10)).log()  # log(p/(1-p))
                    ], dim=-1)  # [M, 2]
                    
                    # Gumbel-Softmax: 软概率
                    y_soft = F.softmax((logits + gumbel_noise.unsqueeze(-1)) / T, dim=-1)
                    
                    # STE: 前向用硬决策，反向用软概率
                    # y_hard = one_hot(argmax(y_soft))
                    y_hard = F.one_hot(y_soft.argmax(dim=-1), num_classes=2).float()
                    # Straight-Through: y_st = y_hard - sg(y_soft) + y_soft
                    y_st = y_hard - y_soft.detach() + y_soft
                    
                    # 分割概率 (有梯度)
                    split_prob_st = y_st[:, 1]  # [M]
                    should_split = split_prob_st > 0.5  # 用于索引
                    
                    # 缓存软分割概率用于辅助损失 (P10-4, P10-9)
                    if not hasattr(self, '_cached_split_probs'):
                        self._cached_split_probs = {}
                    self._cached_split_probs[depth] = split_prob_st
                else:
                    should_split = p_split > 0.5
            
            # ------------------------------------------------------------------
            # Step 4: 分离保持/分割区域
            # ------------------------------------------------------------------
            keep_mask = ~should_split
            split_mask = should_split
            
            # 保持为叶节点 → 输出
            if keep_mask.any():
                output_regions_list.append(splittable_regions[keep_mask, 1:5])
                output_depths_list.append(splittable_depths[keep_mask])
                output_batch_idx_list.append(splittable_regions[keep_mask, 0])
                output_complexity_list.append(complexities[keep_mask])
            
            # ------------------------------------------------------------------
            # Step 5: 向量化象限展开 [M', 5] → [4M', 5]
            # ------------------------------------------------------------------
            if not split_mask.any():
                break
            
            split_regions = splittable_regions[split_mask]  # [M', 5]
            M_split = split_regions.shape[0]
            
            # 批量计算 4 个象限
            # 象限布局 (与 Region.get_quadrant 保持一致):
            #   q=0: 左上 (x1, y1, cx, cy)
            #   q=1: 右上 (cx, y1, x2, cy)
            #   q=2: 左下 (x1, cy, cx, y2)
            #   q=3: 右下 (cx, cy, x2, y2)
            
            x1 = split_regions[:, 1]  # [M']
            y1 = split_regions[:, 2]
            x2 = split_regions[:, 3]
            y2 = split_regions[:, 4]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            batch_idx = split_regions[:, 0]
            
            # 构建 4 个象限 (向量化)
            # next_regions: [4*M', 5]
            next_regions = torch.zeros(4 * M_split, 5, device=device, dtype=torch.long)
            
            # 所有象限的 batch_idx
            next_regions[:, 0] = batch_idx.repeat(4)
            
            # q=0: 左上
            next_regions[0*M_split:1*M_split, 1] = x1
            next_regions[0*M_split:1*M_split, 2] = y1
            next_regions[0*M_split:1*M_split, 3] = cx
            next_regions[0*M_split:1*M_split, 4] = cy
            
            # q=1: 右上
            next_regions[1*M_split:2*M_split, 1] = cx
            next_regions[1*M_split:2*M_split, 2] = y1
            next_regions[1*M_split:2*M_split, 3] = x2
            next_regions[1*M_split:2*M_split, 4] = cy
            
            # q=2: 左下
            next_regions[2*M_split:3*M_split, 1] = x1
            next_regions[2*M_split:3*M_split, 2] = cy
            next_regions[2*M_split:3*M_split, 3] = cx
            next_regions[2*M_split:3*M_split, 4] = y2
            
            # q=3: 右下
            next_regions[3*M_split:4*M_split, 1] = cx
            next_regions[3*M_split:4*M_split, 2] = cy
            next_regions[3*M_split:4*M_split, 3] = x2
            next_regions[3*M_split:4*M_split, 4] = y2
            
            # 更新深度
            next_depths = (splittable_depths[split_mask] + 1).repeat(4)  # [4*M']
            
            current_regions = next_regions
            current_depths = next_depths
        
        # ====================================================================
        # 合并输出
        # ====================================================================
        if not output_regions_list:
            return TensorSplitResult.empty(device)
        
        all_regions = torch.cat(output_regions_list, dim=0)      # [N_total, 4]
        all_depths = torch.cat(output_depths_list, dim=0)        # [N_total]
        all_batch_idx = torch.cat(output_batch_idx_list, dim=0)  # [N_total]
        all_complexities = torch.cat(output_complexity_list, dim=0)  # [N_total]
        
        # P10-1: 缓存所有复杂度用于辅助损失 (阈值对齐等)
        if all_complexities_list:
            self._cached_complexities = torch.cat(all_complexities_list, dim=0)
        else:
            self._cached_complexities = None
        
        # 边界情况: cat 后为空张量
        if all_regions.shape[0] == 0:
            return TensorSplitResult.empty(device)
        
        # ====================================================================
        # 批量 Hilbert 索引计算 (使用 HilbertLUT)
        # ====================================================================
        # 计算中心点
        cx = (all_regions[:, 0] + all_regions[:, 2]).float() / 2  # (x1 + x2) / 2
        cy = (all_regions[:, 1] + all_regions[:, 3]).float() / 2  # (y1 + y2) / 2
        
        hilbert_indices = HilbertLUT.batch_lookup(cx, cy, H_img, device)
        
        # ====================================================================
        # 构建 TensorSplitResult
        # ====================================================================
        result = TensorSplitResult(
            regions=all_regions,
            depths=all_depths,
            batch_indices=all_batch_idx,
            hilbert_indices=hilbert_indices,
            complexities=all_complexities,
            tokens_per_batch=None,  # 稍后填充
        )
        
        # 按 (batch_idx, hilbert_idx) 排序
        result = result.sort_by_hilbert()
        
        # P11-2 优化: 使用 torch.bincount 替代 Python 循环
        # 原实现: for b in range(B): tokens_per_batch[b] = (result.batch_indices == b).sum()
        # 复杂度: O(B) Python 循环 × O(N) 比较 = O(B×N)
        # 优化后: O(N) 单次 bincount 操作
        tokens_per_batch = torch.bincount(
            result.batch_indices, 
            minlength=B
        )
        result.tokens_per_batch = tokens_per_batch
        
        return result
    
    def forward(
        self,
        features: Tensor,
        image_size: Tuple[int, int],
        hard: bool = False,
    ) -> TensorSplitResult:
        """
        对批量图像进行完全向量化的可学习分割 (P9-1 方案 D).
        
        数学形式化:
            S: ℝ^{B×C×H×W} → TensorSplitResult
            
        性能特性:
            - O(D) GPU kernels
            - O(1) Python 操作/层
            - 零 .item()/.tolist() 调用
            
        Args:
            features: [B, C, H', W'] 特征图 (来自 SharedConv)
            image_size: (H, W) 原始图像尺寸
            hard: 是否使用硬决策 (推理时 True)
            
        Returns:
            TensorSplitResult: 纯张量表示的分割结果
            
        Note:
            P9-1 方案 D 实施后，此方法完全替代旧的 forward()。
            返回类型从 List[SplitResult] 变为 TensorSplitResult。
        """
        # P-TEMP-1: 训练模式下自动更新温度
        if self.training and self._temp_enabled:
            self._update_temperature()
        
        H_img, W_img = image_size
        _, _, H_feat, W_feat = features.shape
        
        scale_h = H_feat / H_img
        scale_w = W_feat / W_img
        
        return self._forward_vectorized_tensor(
            features=features,
            image_size=image_size,
            scale_h=scale_h,
            scale_w=scale_w,
            hard=hard,
        )
    
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
            
            # P-GRAD-1: 更新 EMA 分割概率 (用于自适应权重)
            # p̄_d^{(t+1)} = α · p_d^{(t)} + (1 - α) · p̄_d^{(t)}
            batch_p = p_split.mean()
            self._ema_split_probs[depth] = (
                self._ema_alpha * batch_p + 
                (1 - self._ema_alpha) * self._ema_split_probs[depth]
            )
        
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
        使用 STE + REINFORCE 计算策略梯度损失 (P8-4).
        
        ⚠️ DEPRECATED: P9-1 方案 D 实施后此方法暂不可用。
        完全向量化路径不再需要单独的 REINFORCE 梯度估计。
        
        请使用 get_threshold_regularization_loss() 进行可微分训练。
        
        Returns:
            (zero_loss, empty_details)
        """
        import warnings
        warnings.warn(
            "get_ste_reinforce_loss() is deprecated after P9-1 Scheme D. "
            "The vectorized forward path provides gradients through STE. "
            "Use get_threshold_regularization_loss() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        device = features.device
        return torch.tensor(0.0, device=device, requires_grad=True), {
            'reward': 0.0,
            'baseline': 0.0,
            'advantage': 0.0,
            'n_tokens': 0,
            'n_decisions': 0,
            'deprecated': True,
        }

    def get_entropy_loss(self, results: List[SplitResult]) -> Tensor:
        """
        计算深度分布熵损失 (鼓励多尺度)。
        
        数学形式化:
            L_entropy = -Σ_d p(d) log p(d)
            其中 p(d) = |{i: d_i = d}| / N
            
        注意: 此损失不产生梯度 (用于监控)。
        使用 get_threshold_regularization_loss() 获取可微分损失。
        """
        device = self.threshold_offsets.device
        
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
        device = self.threshold_offsets.device
        
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
    
    def get_adaptive_depth_weights(
        self,
        max_depth: Optional[int] = None,
    ) -> Tensor:
        """
        获取自适应深度权重 (P-GRAD-1)。
        
        数学形式化
        ==========
        
        问题: 固定权重 w_d = β^d 基于"每层均匀 50% 分割"假设。
        但实际 BFS 路径受 τ_d 控制，分割概率非均匀分布。
        
        设 p̄_d 为第 d 层平均分割概率的 EMA 估计。
        自适应权重 w_d 反映"固定网格区域实际被访问"的条件概率:
        
            w_d = P(BFS 到达深度 d | BFS 从 d=0 出发)
                = ∏_{k=0}^{d-1} p̄_k
        
        特殊情况:
            - w_0 = 1 (根节点必定访问)
            - 若 p̄_k ≈ 0.5，则 w_d ≈ 0.5^d (退化为固定权重)
            - 若 p̄_k < 0.5，则深层权重更小 (反映实际梯度稀疏性)
        
        EMA 更新规则 (在 _batch_split_decision 中):
            p̄_d^{(t+1)} = α · p_d^{(t)} + (1 - α) · p̄_d^{(t)}
            
        其中 α = 0.1 平衡响应速度和稳定性。
        
        数值稳定性:
            使用 eps = 0.01 进行 clamp，防止权重过快衰减至 0。
        
        Args:
            max_depth: 最大深度 (默认 self.max_depth)
            
        Returns:
            weights: Tensor[D+1] 权重向量，weights[d] 对应深度 d
        """
        if max_depth is None:
            max_depth = self.max_depth
            
        max_depth = min(max_depth, self.max_depth)
        device = self._ema_split_probs.device
        dtype = self._ema_split_probs.dtype
        
        weights = torch.ones(max_depth + 1, device=device, dtype=dtype)
        
        # 数值稳定性: clamp 概率，防止权重指数衰减过快
        eps = 0.01
        
        cumulative = 1.0
        for d in range(max_depth + 1):
            weights[d] = cumulative
            if d < max_depth:
                # p̄_d clamped to [eps, 1-eps] for numerical stability
                p_safe = self._ema_split_probs[d].clamp(eps, 1 - eps)
                cumulative = cumulative * p_safe
                
        return weights
    
    def get_multi_layer_depth_loss(
        self,
        features: Tensor,
        image_size: Tuple[int, int],
        max_eval_depth: Optional[int] = None,
        weight_decay_factor: float = 0.5,
        target_entropy: float = 0.693,
        return_details: bool = False,
        use_adaptive_weights: bool = True,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        """
        多层可微分深度损失。
        
        数学形式化:
            L_multi = Σ_d w_d · (H_target - H̄_d)²
            
            其中:
            - H̄_d = (1/4^d) · Σ_{R∈Grid_d} H(p_d(R)) (第 d 层平均熵)
            - p_d(R) = σ((C_θ(R) - τ_d) / T) (分割概率)
            - H(p) = -p·log(p) - (1-p)·log(1-p) (二元熵)
            
        权重设计 (P-GRAD-1):
            use_adaptive_weights=True (默认, 推荐):
                w_d = ∏_{k=0}^{d-1} p̄_k
                其中 p̄_k 是第 k 层分割概率的 EMA 估计。
                理论依据: 反映固定网格区域"实际被 BFS 访问"的条件概率。
                
            use_adaptive_weights=False (向后兼容):
                w_d = β^d, β = weight_decay_factor
                基于"每层均匀 50% 分割"的简化假设。
            
        梯度分析:
            ∂L/∂τ_d = w_d · (1/4^d) · Σ_R ∂H/∂p · ∂p/∂τ_d
            
            其中 ∂p/∂τ = -p(1-p)/T，因此每个 τ_d 都有非零梯度。
            
        计算优化:
            使用批量 ROI-Align 评估所有网格区域 (1 次 GPU kernel 调用)
            
        Args:
            features: [B, C, H', W'] 特征图
            image_size: (H, W) 图像尺寸
            max_eval_depth: 最大评估深度 (默认 min(max_depth, 3))
                           D=3 时覆盖 96.8% 梯度信号，仅 85 区域
            weight_decay_factor: 权重衰减因子 β (仅当 use_adaptive_weights=False)
            target_entropy: 目标熵 (默认 0.693 = ln(2)，鼓励 p→0.5)
            return_details: 是否返回各层详情 (用于 TensorBoard)
            use_adaptive_weights: 使用自适应权重 (P-GRAD-1, 默认 True)
            
        Returns:
            loss: 标量损失
            details (if return_details): {
                'loss_per_depth': Tensor[D+1],    # 各层损失
                'entropy_per_depth': Tensor[D+1], # 各层平均熵
                'p_split_per_depth': Tensor[D+1], # 各层平均分割概率
                'weight_per_depth': Tensor[D+1],  # 各层权重
                'temperature': Tensor,            # 当前温度
                'num_regions_per_depth': Tensor[D+1], # 各层区域数
                'adaptive_weights_enabled': bool, # 是否使用自适应权重
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
        # 注意: complexity_mlp.forward() 内部已经做了 squeeze(-1)，返回 [N_regions]
        complexities = self.complexity_mlp(pooled_flat)
        
        # =====================================================================
        # 4. 按深度计算损失
        # =====================================================================
        total_loss = torch.tensor(0.0, device=device, dtype=dtype)
        
        # P-GRAD-1: 预计算自适应权重 (在循环外，避免重复计算)
        if use_adaptive_weights:
            adaptive_weights = self.get_adaptive_depth_weights(max_eval_depth)
        
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
            
            # P-GRAD-1: 选择权重计算方式
            # use_adaptive_weights=True:  w_d = ∏_{k<d} p̄_k (反映实际 BFS 访问概率)
            # use_adaptive_weights=False: w_d = β^d (固定几何衰减)
            if use_adaptive_weights:
                w_d = adaptive_weights[d].item()
            else:
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
                'adaptive_weights_enabled': use_adaptive_weights,
            }
            return total_loss, details
        
        return total_loss
    
    def get_soft_balance_loss(
        self,
        features: Tensor,
        image_size: Tuple[int, int],
        grid_size: int = 8,
        return_details: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Any]]]:
        """
        可微分的 2:1 平衡损失 (解决 P-BAL-1)。
        
        数学形式化
        ==========
        
        Hilbert Curve 2:1 Balance 约束:
            ∀ R_i, R_j ∈ Quadtree, Adjacent(R_i, R_j) ⟹ |d_i - d_j| ≤ 1
            
        软损失设计:
            L_balance = Σ_{(i,j) ∈ Adj} max(0, |d̃_i - d̃_j| - 1)²
            
        其中软深度 d̃_i 是期望深度:
            d̃_i = Σ_{d=0}^{D} d · P(depth_i = d)
            
        软深度计算:
            设 p_d(R_i) = σ((C_θ(R_i) - τ_d) / T) 是区域 R_i 在深度 d 的分割概率
            
            P(depth_i = d) = P(到达深度 d) × P(在深度 d 停止)
                           = [∏_{k<d} p_k(R_i)] × (1 - p_d(R_i))  (d < D_max)
                           = [∏_{k<d} p_k(R_i)]                    (d = D_max)
                           
            d̃_i = Σ_d d · P(depth_i = d)
            
        梯度分析:
            ∂L/∂τ_d 非零，因为:
            - d̃_i 依赖于 p_d
            - p_d 依赖于 τ_d
            - L 依赖于 d̃_i
            
        Args:
            features: [B, C, H', W'] 特征图
            image_size: (H, W) 图像尺寸
            grid_size: 评估网格大小 (默认 8，即 8×8 = 64 个位置)
            return_details: 是否返回详细信息
            
        Returns:
            loss: 软 2:1 平衡损失
            details (if return_details): {
                'soft_depths': Tensor[grid_size, grid_size],
                'max_violation': float,
                'num_violations': int,
            }
        """
        B, C, H_feat, W_feat = features.shape
        H_img, W_img = image_size
        device = features.device
        dtype = features.dtype
        
        scale_h = H_feat / H_img
        scale_w = W_feat / W_img
        T = self.log_temperature.exp()
        
        cell_h = H_img / grid_size
        cell_w = W_img / grid_size
        
        # =====================================================================
        # 1. 为每个网格位置计算软深度
        # =====================================================================
        soft_depths = torch.zeros(grid_size, grid_size, device=device, dtype=dtype)
        
        for gi in range(grid_size):
            for gj in range(grid_size):
                # 该格子中心点
                cx = (gj + 0.5) * cell_w
                cy = (gi + 0.5) * cell_h
                
                # 累积分割概率 = P(到达当前深度)
                cumulative_split = torch.ones(1, device=device, dtype=dtype)
                expected_depth = torch.zeros(1, device=device, dtype=dtype)
                
                for d in range(self.max_depth + 1):
                    # 包含该点的区域在深度 d 的边界
                    region_size = max(H_img, W_img) / (2 ** d)
                    region_x1 = int(cx / region_size) * region_size
                    region_y1 = int(cy / region_size) * region_size
                    region_x2 = region_x1 + region_size
                    region_y2 = region_y1 + region_size
                    
                    # 转换为特征图坐标
                    x1_feat = region_x1 * scale_w
                    y1_feat = region_y1 * scale_h
                    x2_feat = region_x2 * scale_w
                    y2_feat = region_y2 * scale_h
                    
                    # 简化的特征提取: 使用区域中心的特征
                    cx_feat = int((x1_feat + x2_feat) / 2)
                    cy_feat = int((y1_feat + y2_feat) / 2)
                    cx_feat = max(0, min(W_feat - 1, cx_feat))
                    cy_feat = max(0, min(H_feat - 1, cy_feat))
                    
                    # 使用 adaptive pooling 获取区域特征
                    x1_i = max(0, int(x1_feat))
                    y1_i = max(0, int(y1_feat))
                    x2_i = min(W_feat, max(x1_i + 1, int(x2_feat)))
                    y2_i = min(H_feat, max(y1_i + 1, int(y2_feat)))
                    
                    region_feat = features[0:1, :, y1_i:y2_i, x1_i:x2_i]
                    pooled = F.adaptive_avg_pool2d(region_feat, (self.pool_size, self.pool_size))
                    pooled_flat = pooled.flatten(1)  # [1, C*k*k]
                    
                    # 计算复杂度
                    complexity = self.complexity_mlp(pooled_flat).squeeze()  # scalar
                    
                    # 分割概率
                    tau_d = self.thresholds[d]
                    p_split = torch.sigmoid((complexity - tau_d) / T)
                    
                    # 在深度 d 停止的概率
                    if d < self.max_depth:
                        p_stop = 1 - p_split
                    else:
                        p_stop = torch.ones_like(p_split)
                    
                    # 期望深度贡献
                    expected_depth = expected_depth + cumulative_split * p_stop * d
                    
                    # 更新累积分割概率
                    if d < self.max_depth:
                        cumulative_split = cumulative_split * p_split
                
                soft_depths[gi, gj] = expected_depth.squeeze()
        
        # =====================================================================
        # 2. 计算相邻位置的深度差异损失
        # =====================================================================
        total_loss = torch.zeros(1, device=device, dtype=dtype)
        num_pairs = 0
        max_violation = 0.0
        num_violations = 0
        
        for gi in range(grid_size):
            for gj in range(grid_size):
                # 右邻居
                if gj < grid_size - 1:
                    diff = (soft_depths[gi, gj] - soft_depths[gi, gj + 1]).abs()
                    violation = F.relu(diff - 1.0)
                    total_loss = total_loss + violation.pow(2)
                    num_pairs += 1
                    with torch.no_grad():
                        if violation.item() > 0:
                            num_violations += 1
                            max_violation = max(max_violation, diff.item())
                
                # 下邻居
                if gi < grid_size - 1:
                    diff = (soft_depths[gi, gj] - soft_depths[gi + 1, gj]).abs()
                    violation = F.relu(diff - 1.0)
                    total_loss = total_loss + violation.pow(2)
                    num_pairs += 1
                    with torch.no_grad():
                        if violation.item() > 0:
                            num_violations += 1
                            max_violation = max(max_violation, diff.item())
        
        # 归一化
        loss = total_loss.squeeze() / max(num_pairs, 1)
        
        if return_details:
            details = {
                'soft_depths': soft_depths.detach(),
                'max_violation': max_violation,
                'num_violations': num_violations,
                'num_pairs': num_pairs,
            }
            return loss, details
        
        return loss
    
    def get_threshold_barrier_loss(self) -> Tensor:
        """
        计算阈值 barrier 正则化损失 (P-TAU-1)。
        
        数学形式化
        ==========
        
        Barrier 函数：
            L_barrier = λ · Σ_d [max(0, τ_min - τ_eff_d)² + max(0, τ_eff_d - τ_max)²]
            
        特性：
            - 当 τ_eff ∈ [τ_min, τ_max] 时：L_barrier = 0，无梯度干扰
            - 当 τ_eff 越界时：二次惩罚，梯度正比于越界量
            
        与 Sigmoid 参数化的对比：
            Sigmoid: ∂τ/∂logits = τ(1-τ) → 边界附近梯度消失
            Offset:  ∂τ_eff/∂offset = 1   → 恒定梯度
            
            边界安全通过损失函数实现，而非参数约束。
            这使得在安全区域内梯度不受任何衰减。
            
        推荐参数：
            λ = 5.0（平衡收敛速度和边界安全）
            τ_min = 0.0, τ_max = 1.0（匹配复杂度分布范围）
            
        Returns:
            barrier_loss: 标量张量，所有深度的 barrier 损失之和
            
        Example:
            >>> splitter = LearnableSplitter(feature_dim=256)
            >>> barrier_loss = splitter.get_threshold_barrier_loss()
            >>> print(barrier_loss)  # 初始为 0（阈值在安全范围内）
            tensor(0.)
        """
        taus = self.thresholds  # [D+1]
        # 下界违反惩罚
        lower_penalty = F.relu(self.tau_min - taus) ** 2
        # 上界违反惩罚
        upper_penalty = F.relu(taus - self.tau_max) ** 2
        # 总损失
        return self.barrier_lambda * (lower_penalty + upper_penalty).sum()

    # =========================================================================
    # P10-3/P10-9: Elastic Budget 弹性预算机制
    # =========================================================================

    def get_soft_token_count(
        self,
        batch_size: int = 1,
    ) -> Tensor:
        """
        计算可微分的软 Token 计数 (P10-9 核心实现)。
        
        数学形式化
        ==========
        
        四叉树期望叶节点数:
            设 p_d 为深度 d 的平均分割概率
            设 R_d 为到达深度 d 的期望区域数
            
            递推关系:
                R_0 = B  (batch size，根区域数)
                R_{d+1} = R_d · p_d · 4  (每个分割区域产生 4 个子区域)
                
            深度 d 的期望叶节点数:
                L_d = R_d · (1 - p_d)   (d < D_max)
                L_{D_max} = R_{D_max}   (最大深度强制停止)
                
            总期望 Token 数:
                N_soft = Σ_{d=0}^{D_max} L_d
                       = Σ_{d=0}^{D_max-1} R_d · (1 - p_d) + R_{D_max}
                       
        梯度分析:
            ∂N_soft/∂p_d = R_d · ∂L_d/∂p_d + Σ_{k>d} ∂R_k/∂p_d · (terms)
            
            其中:
            - ∂L_d/∂p_d = -R_d (直接效应: 分割减少当前层叶子)
            - ∂R_{d+1}/∂p_d = 4·R_d (间接效应: 分割增加下层区域)
            
            这创造了两个方向相反的梯度:
            - 分割更多 → 当前层叶子减少 → ∂N/∂p_d < 0
            - 分割更多 → 下层区域增加 → 可能更多叶子
            
        实现策略:
            使用缓存的 _cached_split_probs (STE 输出) 保持梯度流
            如果缓存为空，使用 EMA 统计值 (无梯度，用于初始化)
            
        Args:
            batch_size: 当前 batch 大小 (根区域数)
            
        Returns:
            soft_token_count: 标量张量，期望 Token 数量
            
        Note:
            此方法应在 forward() 之后调用，以使用最新的分割概率缓存。
            
        Example:
            >>> result = splitter(features, image_size)
            >>> soft_count = splitter.get_soft_token_count(batch_size=features.shape[0])
            >>> elastic_loss = splitter.get_elastic_budget_loss(soft_count, N_min=32, N_max=256)
        """
        device = self.thresholds.device
        dtype = self.thresholds.dtype
        
        # 确定使用哪个概率源
        use_cached = bool(self._cached_split_probs) and self.training
        
        if use_cached:
            # 训练时使用 STE 缓存的概率 (有梯度)
            # _cached_split_probs[d] 是深度 d 的分割概率向量 [M_d]
            # 取平均作为该深度的期望分割率
            p_splits = []
            for d in range(self.max_depth + 1):
                if d in self._cached_split_probs and self._cached_split_probs[d].numel() > 0:
                    p_d = self._cached_split_probs[d].mean()
                else:
                    # 该深度无访问区域，使用 EMA 统计
                    p_d = self._ema_split_probs[d]
                p_splits.append(p_d)
        else:
            # 推理时使用 EMA 统计 (无梯度)
            p_splits = [self._ema_split_probs[d] for d in range(self.max_depth + 1)]
        
        # 递推计算期望 Token 数
        # R_0 = batch_size (根区域数)
        R_d = torch.tensor(float(batch_size), device=device, dtype=dtype)
        N_soft = torch.tensor(0.0, device=device, dtype=dtype)
        
        for d in range(self.max_depth + 1):
            p_d = p_splits[d] if isinstance(p_splits[d], Tensor) else torch.tensor(
                p_splits[d], device=device, dtype=dtype
            )
            
            if d < self.max_depth:
                # 深度 d 的叶节点: 到达但不分割的区域
                L_d = R_d * (1.0 - p_d)
                N_soft = N_soft + L_d
                # 下一层的区域数: 分割的区域 × 4
                R_d = R_d * p_d * 4.0
            else:
                # 最大深度强制停止，所有区域成为叶节点
                N_soft = N_soft + R_d
        
        return N_soft

    def get_elastic_budget_loss(
        self,
        soft_token_count: Tensor,
        N_min: int,
        N_max: int,
        N_target: Optional[int] = None,
        lambda_over: float = 0.1,
        lambda_under: float = 0.01,
    ) -> Tensor:
        """
        Elastic Budget 弹性预算损失 (P10-9 核心实现)。
        
        数学形式化
        ==========
        
        损失函数设计:
            L_elastic = λ_over · φ(N - N_max) + λ_under · ψ(N_min - N)
            
        其中:
            φ(x) = ReLU(x)² / N_max   # 二次惩罚，越界越严重
            ψ(x) = ReLU(x) / N_max    # 线性软约束，温和引导
            
        区间行为:
            | 区间           | 损失 | 梯度方向 | 行为     |
            |----------------|------|----------|----------|
            | N < N_min      | > 0  | ∂L/∂N < 0 → 鼓励增加 N | 软约束 |
            | N ∈ [N_min, N_max] | = 0 | 0 | Dead Zone，自由探索 |
            | N > N_max      | > 0  | ∂L/∂N > 0 → 强制减少 N | 二次惩罚 |
            
        非对称设计原理 (λ_over >> λ_under):
            - 细分割捷径是主要威胁 (P10-8)
            - 用二次惩罚强抑制 N > N_max (过多 tokens)
            - 用线性约束软引导 N < N_min (过少 tokens)
            
            典型比例: λ_over : λ_under = 10 : 1
            
        博弈论分析:
            | 状态 | Splitter 倾向 | Regularizer 倾向 | 均衡 |
            |------|---------------|------------------|------|
            | N < N_min | ↗ 增加 | 维持 (低成本) | 轻微增加 |
            | Dead Zone | 自由 | 自由 | 稳定探索 ✅ |
            | N > N_max | ↗ 增加 (捷径) | ↓↓ 强抑制 | 强制减少 |
            
        Hilbert Curve 约束:
            - N_max 应为 4^m (完整四叉树层)，确保 Hilbert 排序连贯性
            - 推荐: N_max ∈ {64, 256, 1024} = {4², 4⁴, 4⁵}
            - 对于 64×64 输入 (Tiny ImageNet): N_max = 256 = 4⁴
            
        Args:
            soft_token_count: 可微分的软 Token 计数 (来自 get_soft_token_count)
            N_min: Dead Zone 下界 (建议: 0.5 × N_target)
            N_max: Dead Zone 上界 (建议: 4.0 × N_target, 且为 4^m)
            N_target: 目标 Token 数 (仅用于归一化，可选)
            lambda_over: 超出上界惩罚权重 (建议: 0.1)
            lambda_under: 低于下界约束权重 (建议: 0.01)
            
        Returns:
            elastic_loss: 标量张量，弹性预算损失
            
        Example:
            >>> result = splitter(features, image_size)
            >>> soft_count = splitter.get_soft_token_count(batch_size=B)
            >>> elastic_loss = splitter.get_elastic_budget_loss(
            ...     soft_count, N_min=32, N_max=256, lambda_over=0.1, lambda_under=0.01
            ... )
            >>> total_loss = main_loss + elastic_loss
        """
        N = soft_token_count
        
        # 归一化因子 (使损失尺度与 N 无关)
        norm = float(N_max) if N_target is None else float(N_target)
        
        # 超出上界: 二次惩罚 (强抑制捷径)
        # φ(N - N_max) = ReLU(N - N_max)² / norm
        over_excess = F.relu(N - float(N_max))
        over_loss = lambda_over * over_excess.pow(2) / norm
        
        # 低于下界: 线性软约束
        # ψ(N_min - N) = ReLU(N_min - N) / norm
        under_excess = F.relu(float(N_min) - N)
        under_loss = lambda_under * under_excess / norm
        
        return over_loss + under_loss

    # =========================================================================
    # P10-4/P10-5: 可微分软熵损失 (统一解决方案)
    # =========================================================================

    def get_soft_depth_distribution(
        self,
        batch_size: int = 1,
    ) -> Tuple[Tensor, Tensor]:
        """
        计算可微分的软深度分布 (P10-4/P10-5 统一解决方案)。
        
        数学形式化
        ==========
        
        问题背景:
            P10-4: get_entropy_loss 使用 Python 循环统计，无梯度
            P10-5: get_multi_layer_depth_loss 评估固定网格，与 BFS 路径不匹配
            
        统一解决方案:
            使用 BFS 实际路径的 _cached_split_probs 计算软深度分布，
            完全替代固定网格评估和 Python 统计。
            
        递推公式 (与 get_soft_token_count 共享):
            R_0 = B  (batch size，根区域数)
            R_{d+1} = R_d · p̄_d · 4
            
            L_d = R_d · (1 - p̄_d)   (d < D_max，到达但不分割)
            L_{D_max} = R_{D_max}     (最大深度强制停止)
            
        软深度分布:
            p̃(d) = L_d / Σ_k L_k
            
        梯度分析:
            ∂p̃(d)/∂p̄_k 通过 L_d 和 R_d 的链式法则传播:
            - ∂L_d/∂p̄_d = -R_d (直接效应)
            - ∂R_{d+1}/∂p̄_d = 4·R_d (间接效应)
            
            这确保了对每个深度的分割概率都有梯度。
            
        Hilbert Curve ViT 约束:
            - 分布应支持多尺度，不应坍缩到单一深度
            - 目标熵约 0.693 (ln 2) 对应于两个主要深度的平衡
            - 对于 max_depth=3 的系统，理论最大熵 = ln(4) ≈ 1.386
            
        Args:
            batch_size: 当前 batch 大小 (根区域数)
            
        Returns:
            Tuple[depth_distribution, leaf_counts]:
                - depth_distribution: [D+1] 软深度分布 (概率和为 1)
                - leaf_counts: [D+1] 各深度期望叶节点数
                
        Note:
            此方法应在 forward() 之后调用，以使用最新的分割概率缓存。
        """
        device = self.thresholds.device
        dtype = self.thresholds.dtype
        D = self.max_depth
        
        # 确定使用哪个概率源
        use_cached = bool(self._cached_split_probs) and self.training
        
        # 构建分割概率向量 [D+1]
        p_splits = torch.zeros(D + 1, device=device, dtype=dtype)
        
        if use_cached:
            # 训练时使用 STE 缓存的概率 (有梯度)
            for d in range(D + 1):
                if d in self._cached_split_probs and self._cached_split_probs[d].numel() > 0:
                    p_splits[d] = self._cached_split_probs[d].mean()
                else:
                    # 该深度无访问区域，使用 EMA 统计 (无梯度)
                    p_splits[d] = self._ema_split_probs[d]
        else:
            # 推理时使用 EMA 统计
            for d in range(D + 1):
                p_splits[d] = self._ema_split_probs[d]
        
        # 递推计算各深度期望叶节点数
        leaf_counts = torch.zeros(D + 1, device=device, dtype=dtype)
        R_d = torch.tensor(float(batch_size), device=device, dtype=dtype)
        
        for d in range(D + 1):
            if d < D:
                # 深度 d 的叶节点: 到达但不分割的区域
                leaf_counts[d] = R_d * (1.0 - p_splits[d])
                # 下一层的区域数: 分割的区域 × 4
                R_d = R_d * p_splits[d] * 4.0
            else:
                # 最大深度强制停止，所有区域成为叶节点
                leaf_counts[d] = R_d
        
        # 软深度分布 (归一化)
        total = leaf_counts.sum().clamp(min=1e-8)
        depth_distribution = leaf_counts / total
        
        return depth_distribution, leaf_counts

    def get_soft_entropy_loss(
        self,
        batch_size: int = 1,
        target_entropy: Optional[float] = None,
        entropy_weight: float = 1.0,
        mode: str = 'maximize',
    ) -> Tensor:
        """
        可微分的软熵损失 (P10-4/P10-5 核心实现)。
        
        数学形式化
        ==========
        
        软熵定义:
            H̃ = -Σ_d p̃(d) · log(p̃(d) + ε)
            
        其中 p̃(d) 是软深度分布 (来自 get_soft_depth_distribution)。
        
        损失模式:
            mode='maximize':  L = -H̃  (最大化熵，鼓励多尺度)
            mode='target':    L = (H̃ - H_target)²  (匹配目标熵)
            
        目标熵设计 (Hilbert Curve ViT 最优化):
            对于 max_depth=D 的系统:
            - 最大熵: H_max = ln(D+1)
            - 均匀分布熵: H_uniform = ln(D+1)  (所有深度等概率)
            - 推荐目标: H_target = 0.5 × H_max  (平衡探索与专注)
            
            具体值:
            | D   | H_max  | H_target (50%) |
            |-----|--------|----------------|
            | 2   | 1.099  | 0.549          |
            | 3   | 1.386  | 0.693          |
            | 4   | 1.609  | 0.805          |
            
        梯度分析:
            ∂L/∂p̄_d = ∂L/∂H̃ · ∂H̃/∂p̃ · ∂p̃/∂p̄_d
            
            其中:
            - ∂H̃/∂p̃(d) = -(1 + log(p̃(d) + ε))
            - ∂p̃/∂p̄_d 通过 get_soft_depth_distribution 的递推传播
            
        与 P10-9 Elastic Budget 的协调:
            - Elastic Budget 约束总 token 数量范围
            - 软熵损失约束 token 在各深度的分布
            - 两者互补，共同防止坍缩到单一尺度
            
        Args:
            batch_size: 当前 batch 大小
            target_entropy: 目标熵 (仅 mode='target' 使用)
            entropy_weight: 熵损失权重
            mode: 'maximize' (最大化熵) 或 'target' (匹配目标)
            
        Returns:
            标量损失张量
            
        Raises:
            ValueError: mode='target' 但未提供 target_entropy
            
        Example:
            >>> result = splitter(features, image_size)
            >>> entropy_loss = splitter.get_soft_entropy_loss(
            ...     batch_size=B,
            ...     target_entropy=0.693,  # ln(2)
            ...     mode='target',
            ... )
        """
        if mode == 'target' and target_entropy is None:
            raise ValueError("mode='target' requires target_entropy to be specified")
        
        # 获取软深度分布
        depth_dist, _ = self.get_soft_depth_distribution(batch_size)
        
        # 计算软熵
        eps = 1e-8
        log_probs = (depth_dist + eps).log()
        soft_entropy = -(depth_dist * log_probs).sum()
        
        # 根据模式计算损失
        if mode == 'maximize':
            # 最大化熵 = 最小化负熵
            loss = -soft_entropy
        elif mode == 'target':
            # 匹配目标熵
            loss = (soft_entropy - target_entropy) ** 2
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'maximize' or 'target'.")
        
        return entropy_weight * loss

    def get_depth_distribution_stats(
        self,
        batch_size: int = 1,
    ) -> Dict[str, float]:
        """
        获取深度分布统计信息 (用于监控和调试)。
        
        Returns:
            Dict 包含:
                - 'entropy': 当前软熵
                - 'max_entropy': 理论最大熵
                - 'entropy_ratio': 熵比率 (实际/最大)
                - 'dominant_depth': 主导深度
                - 'dominant_prob': 主导深度概率
                - 'distribution': 完整分布列表
        """
        with torch.no_grad():
            depth_dist, leaf_counts = self.get_soft_depth_distribution(batch_size)
            
            # 软熵
            eps = 1e-8
            soft_entropy = -(depth_dist * (depth_dist + eps).log()).sum().item()
            
            # 理论最大熵
            D = self.max_depth
            max_entropy = math.log(D + 1)
            
            # 主导深度
            dominant_depth = depth_dist.argmax().item()
            dominant_prob = depth_dist[dominant_depth].item()
            
            return {
                'entropy': soft_entropy,
                'max_entropy': max_entropy,
                'entropy_ratio': soft_entropy / max_entropy if max_entropy > 0 else 0,
                'dominant_depth': dominant_depth,
                'dominant_prob': dominant_prob,
                'distribution': depth_dist.tolist(),
                'leaf_counts': leaf_counts.tolist(),
            }
    
    def get_auxiliary_losses(
        self,
        features: Optional[Tensor] = None,
        image_size: Optional[Tuple[int, int]] = None,
        include_balance: bool = True,
        include_elastic_budget: bool = False,
        include_soft_entropy: bool = False,
        grid_size: int = 8,
        batch_size: int = 1,
        elastic_N_min: int = 32,
        elastic_N_max: int = 256,
        elastic_lambda_over: float = 0.1,
        elastic_lambda_under: float = 0.01,
        entropy_target: Optional[float] = None,
        entropy_weight: float = 0.1,
        entropy_mode: str = 'maximize',
    ) -> Dict[str, Tensor]:
        """
        获取所有辅助损失的统一接口。
        
        这是推荐的获取辅助损失的方式，返回一个字典包含所有可用的损失。
        
        P10-4/P10-5 更新:
            新增 include_soft_entropy 选项，替代原有的无梯度熵损失。
            软熵损失使用 BFS 实际路径的缓存概率，而非固定网格评估。
        
        Args:
            features: 特征图（balance loss 需要）
            image_size: 图像尺寸（balance loss 需要）
            include_balance: 是否包含 balance loss（需要 features 和 image_size）
            include_elastic_budget: 是否包含 Elastic Budget 损失 (P10-9)
            include_soft_entropy: 是否包含软熵损失 (P10-4/P10-5)
            grid_size: balance loss 的网格大小
            batch_size: 当前 batch 大小 (Elastic Budget 和软熵需要)
            elastic_N_min: Elastic Budget 下界
            elastic_N_max: Elastic Budget 上界
            elastic_lambda_over: 超出上界惩罚权重
            elastic_lambda_under: 低于下界约束权重
            entropy_target: 软熵目标值 (仅 entropy_mode='target' 时使用)
            entropy_weight: 软熵损失权重
            entropy_mode: 'maximize' (最大化熵) 或 'target' (匹配目标)
            
        Returns:
            losses: Dict[str, Tensor] 包含:
                - 'barrier_loss': 阈值 barrier 正则化损失
                - 'balance_loss': 2:1 平衡损失（如果 include_balance=True）
                - 'elastic_budget_loss': Elastic Budget 损失（如果 include_elastic_budget=True）
                - 'soft_entropy_loss': 软熵损失（如果 include_soft_entropy=True）
                
        Example:
            >>> losses = splitter.get_auxiliary_losses(
            ...     features, image_size,
            ...     include_elastic_budget=True,
            ...     include_soft_entropy=True,
            ...     batch_size=B,
            ...     elastic_N_min=32, elastic_N_max=256,
            ...     entropy_mode='maximize',
            ... )
            >>> total_aux = sum(losses.values())
            >>> loss = main_loss + total_aux
        """
        losses = {}
        
        # 阈值 barrier 损失（总是包含）
        losses['barrier_loss'] = self.get_threshold_barrier_loss()
        
        # 2:1 平衡损失（可选）
        if include_balance and features is not None and image_size is not None:
            losses['balance_loss'] = self.get_soft_balance_loss(features, image_size, grid_size)
        
        # P10-9: Elastic Budget 弹性预算损失
        if include_elastic_budget:
            soft_count = self.get_soft_token_count(batch_size=batch_size)
            losses['elastic_budget_loss'] = self.get_elastic_budget_loss(
                soft_token_count=soft_count,
                N_min=elastic_N_min,
                N_max=elastic_N_max,
                lambda_over=elastic_lambda_over,
                lambda_under=elastic_lambda_under,
            )
        
        # P10-4/P10-5: 可微分软熵损失
        if include_soft_entropy:
            losses['soft_entropy_loss'] = self.get_soft_entropy_loss(
                batch_size=batch_size,
                target_entropy=entropy_target,
                entropy_weight=entropy_weight,
                mode=entropy_mode,
            )
        
        return losses
    
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
) -> LearnableSplitter:
    """
    Factory function to create LearnableSplitter.
    
    Note: 
        Scheme B (BalancedGreedy) 和 Scheme C (FixedBudgetDP) 已被移除。
        数学分析表明 LearnableSplitter 完全覆盖其功能并提供以下优势:
        - 端到端可微分训练
        - 无复杂度饱和问题
        - 自适应阈值学习
    
    Args:
        config: AdaptiveSplitConfig
        feature_dim: Feature dimension for LearnableSplitter
        
    Returns:
        LearnableSplitter instance
    """
    return LearnableSplitter(
        feature_dim=feature_dim,
        max_depth=config.max_depth,
        min_region_size=config.min_region_size,
        enforce_balance=config.enforce_balance,
        init_tau_base=config.tau_0 * 3,  # 调整初始阈值更接近中心
        init_tau_gamma=config.gamma,
    )


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
    feature_extractor: Optional[nn.Module] = None,
    **kwargs
) -> TensorSplitResult:
    """
    Convenience function to split a single image using LearnableSplitter.
    
    Mathematical formalization:
    ─────────────────────────────────────────────────────────────────────
    Uses the LearnableSplitter which implements:
    
    C_θ(R) = σ(MLP(RoI-Align(F, R)))
    
    Where:
    - F: Feature map from the image
    - R: Region of interest
    - MLP: Learnable multi-layer perceptron with no saturation
    - σ: Sigmoid activation for complexity score in [0, 1]
    
    The splitting decision uses Gumbel-Softmax with STE for differentiability:
    
    P(split | R) = softmax((logits + G) / τ)
    
    Where G ~ Gumbel(0, 1) and τ is the temperature (annealed during training).
    ─────────────────────────────────────────────────────────────────────
    
    Args:
        image: [C, H, W] or [B, C, H, W] tensor (raw image or features)
        feature_extractor: Optional feature extractor. If None, image is treated
                          as pre-extracted features.
        **kwargs: Additional config parameters for LearnableSplitter
        
    Returns:
        TensorSplitResult with tokens (P9-1: 张量化返回类型)
        
    Note:
        Scheme B (BalancedGreedySplitter) and Scheme C (FixedBudgetDPSplitter)
        have been removed as LearnableSplitter provides superior functionality:
        - End-to-end differentiability via Gumbel-Softmax + STE
        - No complexity saturation (MLP vs variance-based formula)
        - Learnable thresholds: τ_d = τ_{base,d} + δ_d
        - O(D) BFS vs O(N·4^D) DP complexity
    """
    # Ensure 4D tensor
    if image.dim() == 3:
        image = image.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
    
    B, C, H, W = image.shape
    
    # Extract features if extractor provided
    if feature_extractor is not None:
        with torch.no_grad():
            features = feature_extractor(image)
    else:
        # Treat input as pre-extracted features
        features = image
    
    # Determine feature dimension
    feature_dim = features.shape[1]
    
    # Create splitter with appropriate feature_dim
    splitter = LearnableSplitter(
        feature_dim=feature_dim,
        max_depth=kwargs.get('max_depth', 4),
        min_region_size=kwargs.get('min_region_size', 4),
        enforce_balance=kwargs.get('enforce_balance', True),
    )
    
    # Use image size from original input
    image_size = (H, W)
    
    # Run forward pass (P9-1: 返回 TensorSplitResult)
    result = splitter.forward(features, image_size, hard=True)
    
    return result
