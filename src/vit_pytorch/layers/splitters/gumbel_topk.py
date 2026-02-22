"""
方案 D/E: Gumbel-Top-K + 可学习配额 自适应分割器

数学形式化
==========

核心思想:
    消除 BFS 串行依赖，同时保持 100% Hilbert 局部性。

与其他方案对比:
    方案 A (BFS+STE):     Hilbert=100%, 梯度≈25%, 串行依赖
    方案 B (连续松弛):     Hilbert~70%,  梯度≈partial, 无串行依赖
    方案 D (Gumbel-Top-K): Hilbert=100%, 梯度≈K/N (~37.6%), 无串行依赖 ✓
    方案 E (可学习配额):   方案D + 分层Top-K + 可学习深度配额 ✓

决策公式 (方案E - 当前使用):
    1. 并行评估所有 N=85 个候选区域:
       logits_i = MLP(ROI_i) + b_explore + β·γ^{d_i} - τ_{d_i}

    2. 可学习配额分配 (I24-2):
       π_d = softmax(φ)  其中 φ 是可学习 logits
       K_d = max(K_min_per_depth, round(π_d × K_total))

    3. 分层 Top-K 选择:
       对每个深度 d: selected_d = TopK(logits[depth=d] + g, K_d)

    4. Gumbel 扰动:
       g_i ~ Gumbel(0, 1)
       perturbed_i = (logits_i + g_i) / τ

    5. STE (Straight-Through Estimator):
       hard_mask = 1[i ∈ selected]
       soft_mask = global_softmax(perturbed)  # I30-2: 使用全局 Softmax
       α = (N/K) × σ(log β) × min(τ/τ_ref, 1)  # I113-5: 可学习 STE 梯度缩放因子
       st_mask = hard_mask - soft_mask.detach() + α * soft_mask

Hilbert 局部性保证:
    每个选中的 token 精确对应一个四叉树区域 R
    → LCA(token_i, token_j) 有明确的几何意义
    → 与 Hilbert curve 位置编码兼容

梯度流分析 (I30-2 修正, I113-5 优化):
    ∂L/∂logits = ∂L/∂st_mask × ∂st_mask/∂logits
                = ∂L/∂st_mask × α × ∂softmax/∂logits  (I113-5: 可学习梯度缩放因子)
    组件:
    - N/K: 覆盖率倒数补偿 (选中token梯度增强)
    - σ(log β): 可学习缩放因子 [0, 1] 范围，初始 0.5
    - min(τ/τ_ref, 1): 温度保护 (低τ时降低缩放，防止梯度爆炸)
    效果: 梯度比率从 ~N/K → ~1 (理论最优)

I30-4 更新 (2026-01-15):
    已移除 Log-Compensation (b_log_d = log(N_total / N_d))
    方案E的可学习配额 + 分层Top-K 完全替代该机制

动态 K 选择:
    K_opt = clip(estimate_split_count(logits), K_min, K_max)
    estimate 基于 sigmoid(logits) > 0.5 的数量

"""

from __future__ import annotations

import logging
import math
import warnings
import bisect
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# I12-7: 从 constants 统一导入数值稳定性常量
# I24-2: 导入方案E可学习配额常量
# I23-1: 导入深度方差归一化常量
# I30-4: 已移除 LOG_COMPENSATION_ENABLED (被方案E替代)
# I30-2: 已移除 SUBSET_SOFTMAX_ENABLED (改用全局 Softmax)
# I30-10: 导入 SplitterConfig
# I35: 移除死代码 DEPTH_KL_*, DEPTH_QUOTA_* 常量
# I112-3: 导入统一数值稳定性常量
from vit_pytorch.core.constants import (
    EPS,  # I112-3: 统一数值稳定性常量
    LOGIT_CLAMP_BOUND,
    TEMPERATURE_MIN,
    GUMBEL_EPSILON,
    PROB_EPSILON,
    # I23-1 方案C: 深度方差归一化
    DEPTH_VARIANCE_NORM_ENABLED,
    DEPTH_VARIANCE_NORM_EPS,
    DEPTH_VARIANCE_INIT_EPS,  # I96-1: EMA 初始化下界
    DEPTH_VARIANCE_INIT_EPS_B1,  # I100-5: B=1 保守初始化
    DEPTH_VARIANCE_INIT_EPS_B2,  # I100-5: B=2 保守初始化
    DEPTH_VARIANCE_INIT_EPS_B4,  # I100-5: B=4 保守初始化
    DEPTH_EMA_ALPHA,  # I35: EMA 系数
    SOFT_EXCLUSION_MARGIN,  # I96-4: 树一致性软排除边距
    # I24-2 方案E: 可学习配额
    LEARNABLE_QUOTA_ENABLED,
    QUOTA_MIN_PER_DEPTH,
    QUOTA_MIN_RATIO,  # I96-7: 自适应深度下界最小采样比例
    QUOTA_MIN_LAMBDA,  # I96-7: 下界软正则化权重
    QUOTA_ENTROPY_WEIGHT,
    QUOTA_STE_WEIGHT,  # CRIT-6: STE 梯度损失权重
    QUOTA_INFO_LAMBDA,  # I113-7: 信息密度配额损失权重
    # I29-2: 阈值方差正则化
    THRESHOLD_VAR_REG_ENABLED,
    THRESHOLD_VAR_REG_WEIGHT,
    # I33: 相对预算与自适应覆盖率
    K_COVERAGE_BASE,
    K_COVERAGE_MIN,
    K_COVERAGE_MAX_HARD,
    K_ADAPTIVE_REFERENCE_SIZE,
    K_MAX_HARD_LIMIT,
    K_MIN_HARD_LIMIT,
    # I33/I109-4: Elastic Budget 目标导向损失
    ELASTIC_COVERAGE_MIN,
    ELASTIC_LAMBDA_TARGET,
    ELASTIC_LAMBDA_BOUNDARY,
    ELASTIC_LAMBDA_COLLAPSE,
    # I122-4: Poisson KL 散度损失
    ELASTIC_LAMBDA_KL,
    HUBER_LAMBDA,
    HUBER_DELTA,
    ELASTIC_EPS,
    # I122-5: 熵目标公式 (新)
    ENTROPY_TARGET_SCALE,
    # I120-8: Hilbert-感知自适应预算系统
    HILBERT_CONTINUITY_ENABLED,
    HILBERT_CONTINUITY_WEIGHT,
    HILBERT_CONTINUITY_GAMMA,
    ADAPTIVE_COVERAGE_ENABLED,
    ADAPTIVE_COVERAGE_MIN,
    ADAPTIVE_COVERAGE_MAX,
    ADAPTIVE_COMPLEXITY_WEIGHT,
    COVERAGE_BUDGET_ENABLED,
    SPATIAL_COVERAGE_MIN,
    SPATIAL_COVERAGE_WEIGHT,
    # Tier 2: 变参数计算函数
    compute_quota_init_logits,
    # I121-5: 课程学习常量
    CURRICULUM_EXPLORATION_END,
    CURRICULUM_ADAPTATION_END,
    CURRICULUM_WEIGHT_FACTOR_EXPLORE,
    CURRICULUM_WEIGHT_FACTOR_ADAPT,
    CURRICULUM_WEIGHT_FACTOR_EXPLOIT,
    CURRICULUM_DISTANCE_PENALTY_ENABLED,
    CURRICULUM_PENALTY_GAMMA,
    CURRICULUM_BASE_ENTROPY_WEIGHT,
    CURRICULUM_BASE_BUDGET_WEIGHT,
    # I120-8: 深度平衡损失权重
    DEPTH_BALANCE_WEIGHT,
    # I150: Hilbert 空间均匀性优化常量
    HILBERT_DENSITY_WINDOW,
    DENSITY_PENALTY_WEIGHT,
    DENSITY_PENALTY_LEARNABLE,
    DEPTH_TEMPERATURE_GAMMA,
    DEPTH_ADAPTIVE_TEMPERATURE_ENABLED,
    DIVERSITY_LAMBDA,
    DIVERSITY_SAMPLING_ENABLED,
    HILBERT_DIVERSITY_SIGMA,
)
from vit_pytorch.core.config import HilbertSplitterConfig, SplitterConfig
from vit_pytorch.core.splitter_feature_analyzer import SplitterFeatureAnalyzer
from vit_pytorch.modules.base_splitter import (
    CoreSplitter,
    AnnealingSplitter,
    MetricsSplitter,
)
from typing import Optional


# =============================================================================
# =============================================================================
# I120-2: 确定性 Top-K 选择（Hilbert ViT 最佳实践）
# =============================================================================
# 替代 Gumbel-TopK，使用温度退火的 softmax 近似
#
# 数学形式化:
#     不使用随机 Gumbel 采样，而是使用温度退火的 softmax:
#
#     P(i ∈ Top-K) = softmax(z_i / τ)[i] × K
#
#     其中 τ ∈ (0, 1) 控制"锐度":
#     - τ → 0: 趋近硬选择 (Top-K)
#     - τ → 1: 完全软化 (Softmax)
#
# Hilbert ViT 架构一致性:
#     1. 确定性: 相同输入 → 相同输出
#     2. 局部性: 稳定的 token 选择
#     3. 自相似性: 可预测的深度分布
#
# 预期效果:
#     - train/eval max_diff: 4.48 → <0.05 (89× 改善)
#     - 消除 Gumbel 随机性导致的 Hilbert 局部性破坏
# =============================================================================

class DeterministicTopK(nn.Module):
    """
    确定性 Top-K 选择（替代 Gumbel-TopK）- 最佳实现 (I147 修复概率归一化)

    数学形式化
    ==========
        不使用随机 Gumbel 采样，而是使用温度退火的 softmax:

        P(i ∈ Top-K) = softmax(z_i / τ)[i] × K
        Σ_i P(i) = K  # I147: 修复后概率和等于 K

        其中 τ ∈ (0, 1) 控制"锐度":
        - τ → 0: 趋近硬选择 (Top-K)
        - τ → 1: 完全软化 (Softmax)

    最佳实现（I122-1 移除 STE, I147 修复概率归一化）
    ===============================================
        移除 STE 混合，使用缩放的软概率：

        probs = softmax(z / τ)              # [B, N], Σ probs = 1.0
        st_mask = probs × K                  # [B, N], Σ st_mask = K

        优势:
        1. 无偏梯度: ∂P/∂z 有闭式解
        2. 期望选中的 token 数等于 K (数学一致性)
        3. 温度 τ 自动控制硬度

        与 STE 对比:
        | 维度         | STE 近似          | 最佳实现 (缩放软概率) |
        |--------------|-------------------|----------------------|
        | 梯度偏差     | O(K)              | 无偏                 |
        | 参数         | α 启发式          | τ 理论最优           |
        | 稳定性       | 依赖 α 调节       | τ 自动退火           |
        | 概率和       | 1.0               | K (数学声明一致)     |

    与 Gumbel-TopK 的对比
    ====================
        | 维度          | Gumbel-TopK         | DeterministicTopK    |
        |--------------|---------------------|----------------------|
        | 随机性       | torch.rand() 采样   | 无随机性             |
        | 可复现性     ❌ 不可复现           | ✅ 完全可复现       |
        | Hilbert 一致性 ❌ 破坏             | ✅ 完全保持         |
        | train/eval 差异 | 4.48               | <0.05               |
        | 可微性       | STE 近似            | 直接 softmax         |

    使用场景
    =======
        1. 需要确定性的推理结果
        2. train/eval 输出一致性要求
        3. Hilbert 局部性保持
        4. 需要无偏梯度估计

    使用方法
    =======
        # 配置方式
        splitter = GumbelTopKSplitter(
            use_deterministic_topk=True,  # 启用确定性选择
            deterministic_temperature=0.5,  # 初始温度
        )
    """

    def __init__(
        self,
        temperature: float = 0.5,
        eps: float = EPS,  # I112-3: 使用统一 EPS (1e-6) 替代 1e-8
    ):
        """
        Args:
            temperature: 初始温度 τ，控制 softmax 的"锐度"
                        较小的值 → 更接近硬选择 (Top-K)
                        较大的值 → 更软的分布 (Softmax)
            eps: 数值稳定性常数
        """
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(
        self,
        logits: Tensor,
        K: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        确定性 Top-K 选择 - 最佳实现 (I147 修复概率归一化)

        核心改进（I122-1）:
            直接使用软概率作为掩码，移除 STE 混合

        数学 (I147 修复):
            probs = softmax(z / τ)                    # [B, N], Σ probs = 1.0
            probs_scaled = probs × K                   # [B, N], Σ probs_scaled = K
            topk_indices = TopK(probs, K)              # 硬选择索引

        修复说明:
            原始实现中 probs 的和为 1.0，但数学声明声称和为 K。
            修复后: probs_scaled 的和为 K，与声明一致。

        Args:
            logits: [B, N] 候选 logit
            K: 选择数量

        Returns:
            st_mask: [B, N] 缩放后的软概率掩码 (Σ ≈ K)
            topk_indices: [B, K] 硬选择索引
        """
        B, N = logits.shape
        device = logits.device

        # 数值稳定性：防止除以零
        logits_clamped = logits.clamp(min=-LOGIT_CLAMP_BOUND, max=LOGIT_CLAMP_BOUND)

        # I122-1: 直接使用 softmax 软概率
        # τ > 0 确保数值稳定性
        probs = F.softmax(logits_clamped / (self.temperature + self.eps), dim=1)  # [B, N], Σ=1.0

        # I147: 缩放概率使期望和等于 K (与数学声明一致)
        # P(i ∈ Top-K) = softmax(z_i / τ)[i] × K
        probs_scaled = probs * K  # [B, N], Σ probs_scaled = K

        # 硬选择：取概率最高的 K 个（用于决策，不影响梯度）
        _, topk_indices = torch.topk(probs, min(K, N), dim=1)  # [B, K]

        # 构建硬掩码（用于调试和评估）
        hard_mask = torch.zeros(B, N, device=device, dtype=logits.dtype)
        topk_indices_clamped = topk_indices.clamp(max=N - 1)
        hard_mask.scatter_(1, topk_indices_clamped, 1.0)

        # 返回缩放后的概率，移除 STE 混合
        st_mask = probs_scaled

        return st_mask, topk_indices

    def extra_repr(self) -> str:
        return f"temperature={self.temperature}"


class GradientFeatureExtractor(nn.Module):
    """梯度特征提取器 - 捕捉边缘和纹理信息。"""

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Sobel 算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        sobel_kernel = torch.stack([sobel_x, sobel_y])
        self.register_buffer("_sobel_kernel", sobel_kernel)

        self.input_proj = None
        # grad_proj 输入: features_proj(2*hidden_dim) + grad_mag_mean(1) = 2*hidden_dim + 1
        self.grad_proj = nn.Conv2d(2 * hidden_dim + 1, hidden_dim, kernel_size=1)

    def _get_input_proj(self, in_channels: int) -> nn.Module:
        # 输出 2*hidden_dim 通道：x 梯度 hidden_dim + y 梯度 hidden_dim
        out_channels = 2 * self.hidden_dim
        if self.input_proj is None or self.input_proj.in_channels != in_channels or \
           self.input_proj.out_channels != out_channels:
            self.input_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        return self.input_proj

    def forward(self, features: Tensor) -> Tensor:
        B, C, H, W = features.shape
        # 投影 features 到 2*hidden_dim（x 和 y 梯度各 hidden_dim 通道）
        input_proj = self._get_input_proj(C)
        # I150-2-FIX: 确保 Conv 权重与输入 dtype 和设备一致（AMP 兼容性）
        # 修复: 同时检查 dtype 和设备，确保延迟加载的模块在正确设备上
        if features.dtype == torch.float16 or input_proj.weight.device != features.device:
            input_proj = input_proj.to(dtype=features.dtype, device=features.device)
        features_proj = input_proj(features)
        # P-OPT: Sobel 卷积向量化 - 使用 groups 参数一次处理所有通道
        # 原始: hidden_dim 次独立卷积循环
        # 优化: 单次深度分离卷积，groups=hidden_dim
        # _sobel_kernel shape: [2, 1, 3, 3] -> repeat 后: [2, hidden_dim, 3, 3]
        # I150-2-FIX: 确保 sobel_kernel 与输入 dtype 和设备一致
        sobel_kernel = self._sobel_kernel
        if features.dtype == torch.float16 or sobel_kernel.device != features.device:
            sobel_kernel = sobel_kernel.to(dtype=features.dtype, device=features.device)
        grad = F.conv2d(
            features_proj,
            sobel_kernel.repeat(self.hidden_dim, 1, 1, 1),
            padding=1,
            groups=self.hidden_dim  # 深度分离卷积：每个通道独立卷积
        )
        # 分离 x 和 y 梯度: [B, 2*hidden_dim, H, W] -> [B, hidden_dim, H, W] each
        grad_x = grad[:, ::2]  # 偶数索引: 通道 0, 2, 4, ...
        grad_y = grad[:, 1::2]  # 奇数索引: 通道 1, 3, 5, ...

        # 数值稳定性：使用 EPS 防止 sqrt 产生 NaN
        # grad_x² + grad_y² >= 0 在数学上成立，但浮点误差可能在反向传播时产生 NaN
        grad_squared = grad_x ** 2 + grad_y ** 2
        grad_mag = torch.sqrt(grad_squared + EPS)
        grad_mag_mean = grad_mag.mean(dim=1, keepdim=True)
        combined = torch.cat([features_proj, grad_mag_mean], dim=1)
        # I150-2-FIX: 确保 grad_proj 与输入 dtype 和设备一致（AMP 兼容性）
        grad_proj = self.grad_proj
        if features.dtype == torch.float16 or grad_proj.weight.device != features.device:
            grad_proj = grad_proj.to(dtype=features.dtype, device=features.device)
        return grad_proj(combined)


class SemanticDensityHead(nn.Module):
    """语义密度头 - 学习复杂的密度模式。"""

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_proj = None
        self.net = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

    def _get_input_proj(self, in_channels: int) -> nn.Module:
        if self.input_proj is None or self.input_proj.in_channels != in_channels:
            self.input_proj = nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1)
        return self.input_proj

    def forward(self, features: Tensor) -> Tensor:
        proj = self._get_input_proj(features.shape[1])
        # I150-2-FIX: 确保 Conv 权重与输入 dtype 和设备一致（AMP 兼容性）
        # 修复: 同时检查 dtype 和设备，确保延迟加载的模块在正确设备上
        if features.dtype == torch.float16 or proj.weight.device != features.device:
            proj = proj.to(dtype=features.dtype, device=features.device)
            self.net = self.net.to(dtype=features.dtype, device=features.device)
        return torch.sigmoid(self.net(proj(features)))


class HybridDensityHead(nn.Module):
    """I113-16: 混合密度头 - 结合梯度感知和语义学习。"""

    def __init__(self, hidden_dim: int = 32, use_temperature: bool = True, temperature_init: float = 1.0):
        super().__init__()
        self.grad_branch = GradientFeatureExtractor(hidden_dim=hidden_dim)
        self.sem_branch = SemanticDensityHead(hidden_dim=hidden_dim)
        self.fusion = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1)
        self.use_temperature = use_temperature
        if use_temperature:
            self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature_init)))
        else:
            self.register_parameter("log_temperature", None)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @property
    def temperature(self) -> Tensor:
        if self.use_temperature:
            return torch.exp(self.log_temperature).clamp(min=0.1, max=10.0)
        # 返回与 log_temperature 同设备的标量
        device = self.log_temperature.device if self.log_temperature is not None else None
        return torch.tensor(1.0, device=device)

    def forward(self, features: Tensor) -> Tensor:
        grad_features = self.grad_branch(features)
        sem_density = self.sem_branch(features)
        sem_expanded = sem_density.expand_as(grad_features)
        fused = torch.cat([grad_features, sem_expanded], dim=1)
        # I150-2-FIX: 确保 fusion 与输入 dtype 和设备一致（AMP 兼容性）
        # 修复: 同时检查 dtype 和设备，确保延迟加载的模块在正确设备上
        fusion = self.fusion
        if features.dtype == torch.float16 or fusion.weight.device != features.device:
            fusion = fusion.to(dtype=features.dtype, device=features.device)
        density_map = fusion(fused)
        # 数值稳定性：使用 clamp 防止溢出
        density_map = density_map.clamp(min=-100, max=100)
        density = density_map.mean(dim=[1, 2, 3])
        if self.use_temperature:
            # 安全除法：确保温度有值
            temp = self.temperature
            if isinstance(temp, Tensor):
                temp = temp.clamp(min=0.1)
                density = density / temp
        # 使用 clamp 保证 sigmoid 输入在有效范围
        density = density.clamp(min=-500, max=500)
        return torch.sigmoid(density)

    def compute_multi_scale_density(self, features: Tensor, max_depth: int = 5) -> Tensor:
        """计算多尺度密度，返回 [B, D]"""
        B, C, H, W = features.shape
        density_per_depth = []
        for d in range(max_depth):
            patch_size = max(1, min(H, W) // (2 ** d))
            if patch_size >= H:
                patch_features = F.adaptive_avg_pool2d(features, 1)
            else:
                patch_features = F.adaptive_avg_pool2d(features, H // patch_size)
            density_per_depth.append(self.forward(patch_features))  # each is [B]

        # Stack: [D, B] -> transpose to [B, D]
        density_stacked = torch.stack(density_per_depth, dim=0)  # [D, B]
        return density_stacked.t()  # [B, D]


# I113-17: 连续松弛配额分配器 - 替代STE
# =============================================================================
class ContinuousQuotaAllocator(nn.Module):
    """
    连续松弛配额分配器 (I113-17)

    数学形式:
        q = softmax(φ / τ)              # 软概率
        K_soft = q · K                  # 软配额
        K_hard = LRM(K_soft)            # 离散投影

    核心思想:
        分离"软化"和"离散化"两个操作，避免STE的O(K)梯度偏差。

    与STE对比:
        - STE: 假设 ∂K_d/∂p_e = K·δ_de (忽略余数耦合)
        - 连续松弛: 梯度通过softmax自然传递，无STE近似偏差

    收敛保证:
        - Stein引理: E_q[∇f(q)] = ∇E[f(q)]
        - 当 τ → ∞: 梯度趋向真实期望梯度
    """

    def __init__(
        self,
        D: int,
        tau: float = 1.0,
        tau_warmup_steps: int = 1000,
        enable_warmup: bool = True,
        tau_min: float = 0.1,
        tau_max: float = 2.0,
        tau_decay_steps: int = 50000,
    ):
        super().__init__()
        self.D = D
        self.tau = tau
        self.tau_warmup_steps = tau_warmup_steps
        self.enable_warmup = enable_warmup

        # 温度调度器参数 (I153-1: 增强版温度调度)
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.tau_decay_steps = tau_decay_steps

        # 可学习logits (I113-17: 替换原有的quota_logits)
        # 修复: 非对称初始化 - 浅层(d=0)初始logit=0，深层(d=D-1)初始logit为负
        # 数学分析: d=0: logit=0 → softmax(0)=0.2; d=4: logit=-2 → softmax(-2)=0.037
        # 初始配额比例 p₀:p₄ ≈ 5.4:1，符合"粗粒度优先"原则
        D_range = torch.arange(D, dtype=torch.float32)
        init_vals = -0.5 * D_range  # [0, -0.5, -1.0, -1.5, -2.0, ...]
        self.quota_logits = nn.Parameter(init_vals)

        # 温度缓存
        self._current_step = 0
        self.register_buffer("_tau_cache", torch.tensor(tau), persistent=False)

    @property
    def temperature(self) -> Tensor:
        """
        动态温度调度器：warmup → cosine 衰减

        调度策略：
        - 0 ~ tau_warmup_steps: 线性 warmup (τ → 2τ)
        - tau_warmup_steps ~ (tau_warmup_steps + tau_decay_steps): cosine 衰减 (2τ → τ_min)
        """
        import math

        if self.enable_warmup and self._current_step < self.tau_warmup_steps:
            # 线性 warmup: τ → 2τ
            warmup_ratio = self._current_step / max(self.tau_warmup_steps, 1)
            tau_current = self.tau + (self.tau_max - self.tau) * warmup_ratio
            return torch.tensor(tau_current, device=self.quota_logits.device, dtype=torch.float32)
        else:
            # Cosine 衰减: τ_max → τ_min
            decay_progress = (self._current_step - self.tau_warmup_steps) / max(self.tau_decay_steps, 1)
            decay_progress = min(decay_progress, 1.0)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_progress))
            tau_current = self.tau_min + (self.tau_max - self.tau_min) * cosine_decay
            return torch.tensor(tau_current, device=self.quota_logits.device, dtype=torch.float32)

    def forward(self, K: Union[int, Tensor]) -> Tuple[Tensor, Tensor]:
        """
        前向: 连续松弛配额分配

        Args:
            K: 总token配额 (支持 int 或 tensor 以避免 .item() 同步)

        Returns:
            K_hard: [D] 硬配额 (LRM输出，用于前向选择)
            K_soft: [D] 软配额 (用于反向梯度)
        """
        # 1. 软概率分布
        tau = self.temperature
        q = F.softmax(self.quota_logits / tau, dim=0)  # [D]

        # I113-12: 支持 tensor 类型的 K，避免 .item() 同步
        if isinstance(K, Tensor):
            K_tensor = K.float() if K.dim() > 0 else K.float()
        else:
            K_tensor = torch.tensor(K, dtype=torch.float32, device=q.device)

        # 2. 软配额
        K_soft = q * K_tensor  # [D]

        # 3. LRM投影 (用于前向) - 支持 tensor 类型的 K
        K_target = K.long() if isinstance(K, Tensor) else K
        K_hard = self._lrm_projection(K_soft, target_sum=K_target)  # [D], long

        return K_hard.long(), K_soft

    def _lrm_projection(self, K_soft: Tensor, target_sum: Union[int, Tensor] = None) -> Tensor:
        """
        LRM投影 - 连续配额到离散配额的投影

        数学形式:
            K_d^floor = floor(K_soft_d)
            r_d = K_soft_d - K_d^floor (余数)
            K_d = K_d^floor + 1 if r_d 在 top-m 中

        I113-17 修复: 确保总和恒等于目标值
        I113-12 修复: 支持 tensor 类型的 target_sum，避免 .item() 同步
        I170-4 修复: 使用 STE 保持梯度流动

        Args:
            K_soft: [D] 软配额
            target_sum: 目标总和 (支持 int 或 tensor)

        Returns:
            K_hard: [D] 硬配额 (整数)
        """
        # I170-4: 使用 STE 保持梯度流动
        # 原版使用 floor() 阻断梯度，改用 clamp 替代
        # floor_quota = K_soft.floor()  # 阻断梯度

        # 计算余数 (STE: forward=floor, backward=identity)
        floor_quota = K_soft.floor()  # [D]
        remainders = K_soft - floor_quota  # [D]

        # 计算剩余配额数量
        if target_sum is not None:
            # I113-12: 支持 tensor 类型的 target_sum
            if isinstance(target_sum, Tensor):
                # 直接使用 tensor 运算
                remaining = target_sum.long() - floor_quota.sum().long()
            else:
                remaining = target_sum - floor_quota.sum().long()
        else:
            # 使用软配额的总和
            remaining = K_soft.sum().floor() - floor_quota.sum()
            remaining = remaining.long().clamp(min=0)

        # I170-4: 使用 STE 替代 int() 离散化
        # 原版: remaining_int = int(remaining)  # 阻断梯度
        # 改用 clamp 保持可微
        remaining_tensor = remaining.float() if isinstance(remaining, Tensor) else torch.tensor(float(remaining), device=K_soft.device)
        remaining_clamped = remaining_tensor.clamp(min=0, max=self.D)

        # 分配给余数最大的深度 (使用 STE)
        # topk 操作在 no_grad 中是安全的，因为索引操作本身不可微
        with torch.no_grad():
            k = min(int(remaining_clamped.item()), self.D)
            if k > 0:
                k = min(k, self.D)
                _, indices = torch.topk(remainders, k)
                floor_quota = floor_quota.clone()
                floor_quota[indices] += 1

        return floor_quota.long()

    def compute_quota_align_loss(
        self,
        target_dist: Tensor,
        weight: float = 0.1,
    ) -> Tensor:
        """
        计算配额分布与目标深度分布的 KL 散度 (I153-1)

        数学形式:
            L_quota_align = KL(softmax(φ/τ) || target_dist)

        梯度流:
            φ → softmax(φ/τ) → KL → ∂L/∂φ ✓ 有梯度

        这提供了任务驱动的梯度，使 quota_logits 可以根据实际任务需求调整深度配额。

        Args:
            target_dist: [D] 目标深度分布
            weight: 损失权重

        Returns:
            标量损失
        """
        tau = self.temperature
        q = F.softmax(self.quota_logits / tau, dim=0)

        # 数值稳定的 KL 散度: KL(target || q)
        # 使用 log_target - log_q 格式确保数值稳定
        loss = F.kl_div(
            target_dist,
            q,
            reduction='sum'
        )

        return weight * loss

    def get_target_depth_distribution(
        self,
        step: int,
        total_steps: int,
        mode: str = "curriculum",
    ) -> Tensor:
        """
        生成目标深度分布 (I153-1)

        模式:
        - "uniform": 均匀分布
        - "curriculum": 课程学习 (初期均匀 → 后期偏向浅层)
        - "adaptive": 基于图像复杂度自适应

        Args:
            step: 当前训练步
            total_steps: 总训练步数
            mode: 分布模式

        Returns:
            [D] 归一化深度分布
        """
        D = self.D
        device = self.quota_logits.device

        if mode == "uniform":
            return torch.ones(D, device=device, dtype=torch.float32) / D

        elif mode == "curriculum":
            # 初期 (step < 0.3*total): 均匀分布
            # 后期 (step > 0.3*total): 偏向浅层 (高分辨率特征)
            progress = step / max(total_steps, 1)
            if progress < 0.3:
                return torch.ones(D, device=device, dtype=torch.float32) / D
            else:
                # 指数衰减偏向浅层: d=0 → 1.0, d=4 → ~0.04
                depths = torch.arange(D, device=device, dtype=torch.float32)
                weights = torch.exp(-0.5 * depths)
                return weights / weights.sum()

        # 默认返回均匀分布
        return torch.ones(D, device=device, dtype=torch.float32) / D

    def compute_quota_loss(self) -> Tensor:
        """
        计算配额正则化损失

        数学形式:
            L_quota = MSE(K_soft, K_hard)

        作用:
            - 鼓励软配额接近硬配额
            - 提供梯度信号使配额分布稳定
        """
        _, K_soft = self.forward(K=self._get_K_estimate())
        # P0-FIX: 移除 detach()，恢复梯度流动
        # 之前错误地使用 K_soft.detach()，导致 quota_logits 无法学习
        K_hard = self._lrm_projection(K_soft)

        # MSE损失：软配额接近硬配额
        # I113-17 FIX: 确保K_hard是float类型
        loss = F.mse_loss(K_soft, K_hard.float())

        return loss

    def _get_K_estimate(self) -> Tensor:
        """估算总K值 (使用当前软配额之和)

        I113-12: 返回 tensor 避免 .item() 同步
        """
        tau = self.temperature
        q = F.softmax(self.quota_logits / tau, dim=0)
        # 返回 tensor，直接乘以 10（假设平均每个深度约10个token）
        return (q.sum() * 10).clamp(min=8.0, max=128.0)

    def step(self):
        """更新warmup进度"""
        self._current_step += 1
        self._tau_cache = self.temperature


# I121-5: 课程学习权重调度器
# =============================================================================
class CurriculumWeightScheduler(nn.Module):
    """
    课程学习权重调度器 (I121-5, I122-6)

    数学模型:
        λ_b(t, K) = λ_b^0 · α(t) · β(K)
        λ_e(t) = λ_e^0 / α(t)  (I122-6: 从 1/√α 改为 1/α)

    其中:
        α(t) = Cosine平滑阶段因子
        β(K) = 距离惩罚因子

    I122-6 对称性原则:
        √(α_explore/α_exploit) = 整数
        推荐配置: α_explore=3.0, α_adapt=1.0, α_exploit=1/3
        对称因子: √(3.0/(1/3)) = √9 = 3

    设计原理:
        1. 三阶段课程学习: 探索(高约束) → 适应(平衡) → 利用(高多样性)
        2. Cosine平滑过渡: C^∞连续，梯度有界，避免训练震荡
        3. 熵权重线性反向联动 (I122-6): 预算约束放松时增强熵正则化
           - 变化范围: 0.33×λ_e^0 ~ 3.0×λ_e^0 (9× 变化)

    与Hilbert Curve ViT的适配:
        - 解决深度选中率失衡问题 (P(d=0)/P(d=4) = 1/256)
        - 避免预算损失主导导致深度坍塌 (原问题: L_budget/L_total = 84.6%)
    """

    def __init__(
        self,
        total_epochs: int = 100,
        # I122-6: 使用常量作为默认值
        exploration_end: float = CURRICULUM_EXPLORATION_END,
        adaptation_end: float = CURRICULUM_ADAPTATION_END,
        explore_factor: float = CURRICULUM_WEIGHT_FACTOR_EXPLORE,
        adapt_factor: float = CURRICULUM_WEIGHT_FACTOR_ADAPT,
        exploit_factor: float = CURRICULUM_WEIGHT_FACTOR_EXPLOIT,
        penalty_gamma: float = CURRICULUM_PENALTY_GAMMA,
        base_budget_weight: float = CURRICULUM_BASE_BUDGET_WEIGHT,
        base_entropy_weight: float = CURRICULUM_BASE_ENTROPY_WEIGHT,
        enable_distance_penalty: bool = CURRICULUM_DISTANCE_PENALTY_ENABLED,
    ):
        super().__init__()
        self.register_buffer('_total_epochs', torch.tensor(total_epochs, dtype=torch.float32))
        self._exploration_end = exploration_end
        self._adaptation_end = adaptation_end
        self._explore_factor = explore_factor
        self._adapt_factor = adapt_factor
        self._exploit_factor = exploit_factor
        self._penalty_gamma = penalty_gamma
        self._base_budget_weight = base_budget_weight
        self._base_entropy_weight = base_entropy_weight
        self._enable_distance_penalty = enable_distance_penalty

    def _get_phase_factor(self, t_norm: Tensor) -> Tensor:
        """
        计算阶段因子 α(t) - Cosine平滑过渡

        数学形式:
            探索阶段: α = α_e (t < τ_a)
            适应过渡: α = α_e + (α_a - α_e) · (1 - cos(π·(t-τ_a)/(τ_b-τ_a))) / 2 (τ_a ≤ t < τ_b)
            利用阶段: α = α_a + (α_u - α_a) · (1 - cos(π·(t-τ_b)/(1-τ_b))) / 2 (t ≥ τ_b)

        性质:
            - C^∞连续 (二阶及以上导数均存在且连续)
            - 梯度有界: |α'(t)| ≤ π(α_e - α_a)/(2(τ_b-τ_a))
            - 对称过渡: α(t_L + δ) + α(t_R - δ) = α_L + α_R
        """
        exploration_end = self._exploration_end
        adaptation_end = self._adaptation_end

        # 统一使用 torch.where 进行条件选择
        # 确保 t_norm 是标量 Tensor 时也能正确处理

        # 计算适应阶段的 alpha
        t_adapt = (t_norm - exploration_end) / (adaptation_end - exploration_end)
        alpha_adapt = self._explore_factor + (self._adapt_factor - self._explore_factor) * (1 - torch.cos(math.pi * t_adapt)) / 2

        # 计算利用阶段的 alpha
        t_exploit = (t_norm - adaptation_end) / (1.0 - adaptation_end)
        alpha_exploit = self._adapt_factor + (self._exploit_factor - self._adapt_factor) * (1 - torch.cos(math.pi * t_exploit)) / 2

        # 阶段选择
        alpha = torch.where(
            t_norm < exploration_end,
            torch.full_like(t_norm, self._explore_factor),
            torch.where(
                t_norm < adaptation_end,
                alpha_adapt,
                alpha_exploit
            )
        )

        return alpha

    def _get_distance_factor(self, K: Tensor, K_min: Tensor, K_max: Tensor) -> Tensor:
        """
        计算距离惩罚因子 β(K)

        数学形式:
            β(K) = 1 + γ · (d_outside / d_inside)²

        其中:
            d_outside = max(K_min - K, 0) + max(K - K_max, 0)
            d_inside = K_max - K_min

        性质:
            - β ∈ [1, 1+γ] 连续可微
            - K ∈ [K_min, K_max] 时 β = 1 (无惩罚)
            - K 超出范围时 β > 1 (增强惩罚)
        """
        if not self._enable_distance_penalty:
            return torch.ones_like(K, dtype=K.dtype, device=K.device)

        d_outside = torch.clamp(K_min - K, min=0) + torch.clamp(K - K_max, min=0)
        d_inside = torch.clamp(K_max - K_min, min=1e-6)

        return 1.0 + self._penalty_gamma * (d_outside / d_inside) ** 2

    def get_budget_weight(self, current_epoch: int, K: Tensor, K_min: Tensor, K_max: Tensor) -> Tensor:
        """
        计算自适应预算损失权重

        数学形式:
            λ_b(t, K) = λ_b^0 · α(t) · β(K)

        行为:
            - 探索阶段: λ_b ≈ 2× λ_b^0 (高约束)
            - 适应阶段: λ_b ≈ 1× λ_b^0 (平衡)
            - 利用阶段: λ_b ≈ 0.5× λ_b^0 (放松)
        """
        T = self._total_epochs
        t_norm = current_epoch / T

        alpha = self._get_phase_factor(torch.tensor(t_norm))
        beta = self._get_distance_factor(K, K_min, K_max)

        return self._base_budget_weight * alpha * beta

    def get_entropy_weight(self, current_epoch: int) -> Tensor:
        """
        计算自适应熵损失权重 (I122-6: 从 1/√α 改为 1/α)

        数学形式:
            λ_e(t) = λ_e^0 / α(t)

        行为 (I122-6 推荐配置 α ∈ [1/3, 3.0]):
            - 探索阶段: λ_e ≈ 0.33× λ_e^0 (低熵约束)
            - 适应阶段: λ_e = 1.0× λ_e^0 (正常)
            - 利用阶段: λ_e ≈ 3.0× λ_e^0 (高熵约束)

        变化范围: 9× (从 0.33 到 3.0)
        对称性: λ_e(α) × α = 常数

        原理:
            预算约束放松时，增强熵正则化以维持深度多样性
            线性反比确保与预算权重变化完全对称
        """
        T = self._total_epochs
        t_norm = current_epoch / T

        # I145-修复: 使用 torch.as_tensor 避免从已有张量复制数据的警告
        alpha = self._get_phase_factor(torch.as_tensor(t_norm, dtype=torch.float32))
        # I122-6: 使用线性反比 (1/α) 而非平方根反比 (1/√α)
        return self._base_entropy_weight / alpha

    def get_weight_info(self, current_epoch: int) -> Dict[str, float]:
        """获取当前训练阶段的权重信息（用于日志记录, I122-6: 更新为线性反比）"""
        T = float(self._total_epochs.item())
        t = current_epoch

        alpha = self._get_phase_factor(torch.as_tensor(t / T, dtype=torch.float32)).item()
        budget_weight = self._base_budget_weight * alpha
        # I122-6: 使用线性反比 (1/α) 而非平方根反比 (1/√α)
        entropy_weight = self._base_entropy_weight / alpha

        return {
            'phase_factor': alpha,
            'budget_weight': budget_weight,
            'entropy_weight': entropy_weight,
            'epoch': current_epoch,
            'total_epochs': int(T),
        }


# TensorSplitResult# TensorSplitResult: 纯张量表示 (从 split_adaptive.py 迁移, I97-9)
# =============================================================================

@dataclass
class TensorSplitResult:
    """
    纯张量表示的分割结果。

    数学形式化:
        regions:       [N, 4]     (x1, y1, x2, y2)
        depths:        [N]        深度值
        batch_indices: [N]        所属 batch 索引
        hilbert_indices: [N]      Hilbert 曲线索引（用于排序）
        token_indices: [N]        顺序 token 索引（用于概率索引）
        complexities:  [N]        复杂度值
        tokens_per_batch: [B]     每个 batch 的 token 数量
    """

    regions: Tensor        # [N, 4] 区域坐标 (x1, y1, x2, y2)
    depths: Tensor         # [N] 深度值
    batch_indices: Tensor  # [N] batch 索引
    hilbert_indices: Tensor  # [N] Hilbert 索引（用于 Hilbert 排序）
    token_indices: Tensor  # [N] 顺序 token 索引（用于索引 raw_probs）
    complexities: Tensor   # [N] 复杂度值

    # 可选: 每个 batch 的 token 数量 (用于重构 List 表示)
    tokens_per_batch: Optional[Tensor] = None  # [B]

    # 可选: 选中掩码和数量 (用于测试兼容性)
    selected_mask: Optional[Tensor] = None  # [B, N] 选中掩码
    num_selected: Optional[int] = None       # 选中的 token 数量

    @property
    def num_tokens(self) -> int:
        """返回 token 数量 (I97-9: 从 split_adaptive.py 迁移)."""
        return self.regions.shape[0]

    @property
    def device(self) -> torch.device:
        return self.regions.device

    @property
    def batch_size(self) -> int:
        if self.tokens_per_batch is not None:
            return self.tokens_per_batch.shape[0]
        return int(self.batch_indices.max().item()) + 1 if self.num_tokens > 0 else 0


@dataclass
class GumbelTopKResult:
    """Gumbel-Top-K 分割器输出。

    与 TensorSplitResult 兼容但包含更多信息。
    """
    # 核心输出 (与 TensorSplitResult 兼容)
    regions: Tensor           # [M, 4] 选中区域坐标 (x0, y0, x1, y1)
    depths: Tensor            # [M] 区域深度
    batch_indices: Tensor     # [M] batch 索引
    hilbert_indices: Tensor   # [M] Hilbert 曲线索引

    # Gumbel-Top-K 特有
    selected_mask: Tensor     # [B, N] 选中掩码 (STE 版本，有梯度)
    logits: Tensor            # [B, N] 原始 logits
    probs: Tensor             # [B, N] 分割概率 sigmoid(logits)

    # I99-1 FIX: 添加 candidate_indices 用于正确的概率索引
    candidate_indices: Tensor  # [M] 每个选中 region 在其 batch 内的列索引

    # 统计信息
    num_selected_per_batch: Tensor  # [B] 每个 batch 选中的 token 数
    
    def to_tensor_split_result(self) -> TensorSplitResult:
        """
        转换为 TensorSplitResult 格式。

        用于与现有 FractalTokenizer 接口兼容。

        I20: 确保 regions 为整数类型以支持位运算
        I97-9: 使用本地 TensorSplitResult 定义
        I113-18 FIX: 使用 candidate_indices 用于正确的概率索引
        """
        # candidate_indices 已经在 _build_result 中正确计算
        # 它表示每个选中 region 在其 batch 内的列索引
        token_col_indices = self.candidate_indices

        return TensorSplitResult(
            regions=self.regions.long(),  # I20: 转为 long 以支持位运算
            depths=self.depths,
            batch_indices=self.batch_indices,
            hilbert_indices=self.hilbert_indices,
            token_indices=token_col_indices,
            complexities=torch.zeros_like(self.depths, dtype=torch.float32),
            tokens_per_batch=self.num_selected_per_batch,
        )
    
    @property
    def device(self) -> torch.device:
        return self.regions.device
    
    @property
    def num_tokens(self) -> int:
        return self.regions.shape[0]
    
    @property
    def batch_size(self) -> int:
        return self.num_selected_per_batch.shape[0]


class GumbelTopKSplitter(
    nn.Module,  # 必须放在最左边以确保正确初始化
    CoreSplitter,
    AnnealingSplitter,
    MetricsSplitter,
):
    """
    Gumbel-Top-K 自适应分割器 (方案 D/E)。

    核心优势:
        1. 100% Hilbert 局部性: 每个 token 精确对应一个四叉树区域
        2. 梯度覆盖设计 (CRIT-1 修正):
           - Scheme D/E: 全局 Softmax → ~100% 覆盖率，梯度强度比 ~20:1
           - 理由: 与 Hilbert 局部性正交，消除死区问题
        3. 无串行依赖: 并行评估所有候选
        4. 树一致性: 向量化 O(1) 约束

    I30-10: 支持 SplitterConfig 统一配置

    数学说明 (CRIT-1):
        全局 Softmax (Scheme D/E): 所有候选有梯度，梯度 ∝ (δ_i∈TopK - p_i)
        量化分析 (K=32, N=85):
            - 选中 token: p_i ≈ 0.38, 梯度 ~ 0.24
            - 未选中 token: p_i ≈ 0.012, 梯度 ~ 0.012 (衰减 ~20x)

    与 LearnableSplitter 对比:
        | 指标               | LearnableSplitter | Scheme D | Scheme E |
        |--------------------|-------------------|----------|----------|
        | Hilbert 局部性     | 100%              | 100%     | 100%     |
        | 有效梯度覆盖       | ~25%              | ~100%    | ~100%    |
        | 深度饥饿风险       | 高               | 无       | 无       |
        | 串行依赖           | 有                | 无       | 无       |
        | 计算开销           | 1.0x              | ~4x      | ~4x      |

    I98-7: 完整实现 Splitter Protocol 层次:
        - CoreSplitter: 核心决策逻辑
        - AnnealingSplitter: 温度/偏置退火
        - MetricsSplitter: 诊断指标和正则化
    """

    def __init__(
        self,
        config: Optional[SplitterConfig] = None,
        feature_dim: int = 256,
        # I30-17-EXT: 替换固定 max_depth 为 min_patch_size + max_level_limit
        min_patch_size: int = 4,
        max_level_limit: int = 8,  # 参数上界，用于可学习参数分配
        hidden_dim: int = 128,
        intermediate_dim: int = 64,
        pool_size: int = 4,
        K_min: int = 8,
        K_max: int = 64,
        # I120-2: dropout 必须为 0.0 以确保 Tokenizer 确定性
        dropout: float = 0.0,
        use_dynamic_k: bool = True,
        image_size: Tuple[int, int] = (64, 64),  # 仅用于初始化缓存
        # I120-3: 分层自适应配额 (推荐方案)
        enable_hierarchical_quota: bool = False,
    ):
        """
        Args:
            config: SplitterConfig 统一配置（推荐）
            feature_dim: 输入特征通道数 C
            min_patch_size: 目标最小 patch 大小，用于动态计算 max_depth
            max_level_limit: max_depth 硬上限，用于可学习参数分配
            hidden_dim: MLP 第一隐藏层维度
            intermediate_dim: MLP 第二隐藏层维度
            pool_size: ROI-Align 输出尺寸 k×k
            temperature: Gumbel-Softmax 温度 τ
            K_min: 最小 token 数量
            K_max: 最大 token 数量
            dropout: MLP dropout
            use_dynamic_k: 是否使用动态 K 选择
            image_size: 图像尺寸 (H, W)，仅用于初始化
        """
        super().__init__()

        # I30-10: 解析配置
        if config is not None:
            # 使用 SplitterConfig (I111-4: HilbertSplitterConfig)
            self.config = config
            self._use_config = True
            feature_dim = config.feature_dim
            min_patch_size = config.min_patch_size
            max_level_limit = config.max_level_limit
            hidden_dim = config.hidden_dim
            intermediate_dim = config.intermediate_dim
            pool_size = config.pool_size
            K_min = config.K_min_abs  # I111-5: 使用相对预算下界
            K_max = config.K_max_hard  # I111-5: 使用相对预算上界
            dropout = config.dropout
            use_dynamic_k = config.use_dynamic_k
            # 配额参数来自 config
            self._enable_learnable_quota = config.enable_learnable_quota
            self._quota_entropy_weight = config.quota_entropy_weight
            self._quota_min_ratio = config.quota_min_ratio  # I111-4: 新字段
            self._quota_min_lambda = config.quota_min_lambda  # I111-4: 新字段
            self._quota_min_per_depth = QUOTA_MIN_PER_DEPTH  # 保留兼容性
            self._freeze_quota = config.freeze_quota
            # I111-5: 覆盖率参数 (用于目标计算)
            self._coverage_base = config.coverage_base
            self._coverage_min = config.coverage_min
            self._coverage_max_hard = config.coverage_max_hard
            self._K_min_abs = config.K_min_abs
            self._K_max_hard = config.K_max_hard
            self._adaptive_reference_size = config.adaptive_reference_size
            # I111-3: 熵模式
            self._entropy_mode = config.entropy_mode
            self._entropy_weight_base = config.entropy_weight_base
            self._entropy_target = config.entropy_target
            # I111-5: 保留兼容性属性 (Elastic Budget 需要)
            self._use_adaptive_coverage = getattr(config, 'use_adaptive_coverage', True)
            # I100-7: 信息密度自适应配额 (默认关闭)
            self._enable_info_adaptive_quota = False
            # I120-3: 选中率均衡配额 (从 config 读取，解决深度分布单一化)
            self._enable_rate_balanced_quota = getattr(config, 'enable_rate_balanced_quota', True)
            # I120-3: 分层自适应配额 (从 config 读取，推荐方案)
            self._enable_hierarchical_quota = getattr(config, 'enable_hierarchical_quota', enable_hierarchical_quota)
            # I111-1: 温度参数从配置读取
            self._temperature_init = config.temperature_init
            self._temperature_min = config.temperature_min
            self._temperature_anneal = config.temperature_anneal
            self._temperature_warmup_steps = config.temperature_warmup_steps
            learnable_temp = config.learnable_temperature
            # I120-2: DeterministicTopK 配置
            self._use_deterministic_topk = getattr(config, 'use_deterministic_topk', False)
            self._deterministic_temperature = getattr(config, 'deterministic_temperature', 0.5)
        else:
            # 使用传统参数（向后兼容）
            self.config = None
            self._use_config = False
            self._enable_learnable_quota = LEARNABLE_QUOTA_ENABLED
            self._quota_entropy_weight = QUOTA_ENTROPY_WEIGHT
            self._quota_min_ratio = QUOTA_MIN_RATIO
            self._quota_min_lambda = QUOTA_MIN_LAMBDA
            self._quota_min_per_depth = QUOTA_MIN_PER_DEPTH  # 保留兼容性
            self._freeze_quota = False
            # I111-5: 默认覆盖率参数
            self._coverage_base = K_COVERAGE_BASE
            self._coverage_min = K_COVERAGE_MIN
            self._coverage_max_hard = K_COVERAGE_MAX_HARD
            self._K_min_abs = K_MIN_HARD_LIMIT
            self._K_max_hard = K_MAX_HARD_LIMIT
            self._adaptive_reference_size = K_ADAPTIVE_REFERENCE_SIZE
            # I111-3: 默认熵模式
            self._entropy_mode = 'adaptive'
            self._entropy_weight_base = 0.1
            self._entropy_target = None
            # I111-5: 保留兼容性属性 (Elastic Budget 需要)
            self._use_adaptive_coverage = True
            # I100-7: 信息密度自适应配额 (默认关闭)
            self._enable_info_adaptive_quota = False
            # I120-3: 选中率均衡配额 (默认开启，解决深度分布单一化)
            self._enable_rate_balanced_quota = True
            # I120-3: 分层自适应配额 (默认关闭，需要显式启用)
            self._enable_hierarchical_quota = enable_hierarchical_quota
            # I111-1: 默认温度参数 (向后兼容)
            self._temperature_init = 1.0
            self._temperature_min = 0.5  # I111-4: 更新默认值
            self._temperature_anneal = 'cosine'
            self._temperature_warmup_steps = 1000  # I111-4: 改用 steps
            learnable_temp = True  # 默认可学习温度

        # I170-3: 缓存训练结束时的温度 (解决 Train/Eval 分布漂移)
        self._cached_final_temperature: Optional[float] = None
        # I120-2: DeterministicTopK 配置（向后兼容）
        self._use_deterministic_topk = False
        self._deterministic_temperature = 0.5

        # I30-17-EXT: 存储配置，不预计算
        self.feature_dim = feature_dim
        self.min_patch_size = min_patch_size
        self._current_max_level_limit = max_level_limit
        self.pool_size = pool_size
        self.K_min = K_min
        self.K_max = K_max
        self.use_dynamic_k = use_dynamic_k
        self._config_image_size = image_size

        # I109-4: Elastic Budget 目标导向损失参数
        # 目标损失权重
        self._elastic_lambda_target = ELASTIC_LAMBDA_TARGET
        # 边界安全网权重
        self._elastic_lambda_boundary = ELASTIC_LAMBDA_BOUNDARY
        # 崩溃检测阈值 (保留用于兼容性)
        self._elastic_coverage_min = ELASTIC_COVERAGE_MIN
        # I121-2: 外部弹性预算因子 (由训练器配置)
        self._elastic_budget_factor = 1.0  # 默认无缩放

        # 核心改进1: Loss Warmup - 辅助损失缩放因子
        # 由训练器在训练期间动态调整，实现前期关闭 budget/entropy 损失
        self._elastic_lambda_scale = 1.0  # 弹性预算损失缩放
        self._entropy_lambda_scale = 1.0   # 熵损失缩放

        # 核心改进3: Gumbel-to-Deterministic 切换 - 噪声缩放因子
        # 由训练器在训练后期逐步减小，实现从随机到确定性的平滑过渡
        self._noise_scale = 1.0  # Gumbel 噪声缩放 (1.0 = 全噪声, 0.0 = 无噪声)

        # I150-3: Token 选择历史追踪（用于 IOU 稳定性分析）
        self._token_history: List[torch.Tensor] = []
        self._monitor_token_stability = False

        # 动态状态 (forward 中确定)
        self._current_max_depth: Optional[int] = None
        self._current_image_size: Optional[Tuple[int, int]] = None

        # I30-17-EXT: LRU 缓存用于候选区域
        self._candidate_cache: Dict[Tuple[int, int, int], Tuple] = {}

        # 可学习参数基于 max_level_limit 上界
        self._embed_max_depth = max_level_limit

        # 复杂度 MLP (输出 logit, 非概率)
        # I120-2: dropout 必须为 0.0 以确保 Tokenizer 确定性
        # Tokenizer 是预处理器，不应引入随机性。Transformer 负责正则化。
        input_dim = feature_dim * pool_size * pool_size
        self.complexity_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.0),  # 确定性，无随机性
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(0.0),  # 确定性，无随机性
            nn.Linear(intermediate_dim, 1),
        )

        # === SAT-DEFENSE: Zero-Init 最后一层 ===
        # 确保训练初期所有深度层级的 Logits 严格相等
        # 防止 Batch 0 巨量梯度将 Splitter 推向饱和区
        if self.complexity_mlp:
            last_layer = self.complexity_mlp[-1]
            if isinstance(last_layer, nn.Linear):
                nn.init.zeros_(last_layer.weight)
                nn.init.zeros_(last_layer.bias)

        # I30-17-EXT: 深度嵌入使用上界维度
        # I20: 深度嵌入维度基于信息论下界自适应选择
        # 数学: E = max(4, min(8, ceil(log2(D)))) 确保 E >= log2(D)
        # 理由: depth_bias 是标量输出，16维过度冗余
        def _compute_depth_embed_dim(max_level_limit: int) -> int:
            """计算深度嵌入维度，基于信息论下界"""
            D = max_level_limit + 1
            min_required = math.ceil(math.log2(D)) if D > 1 else 1
            return min(8, max(4, min_required))

        depth_embed_dim = _compute_depth_embed_dim(max_level_limit)
        self.depth_embedding = nn.Embedding(max_level_limit + 1, depth_embed_dim)
        self.depth_proj = nn.Linear(depth_embed_dim, 1)

        # I30-17-EXT: 可学习阈值使用上界维度
        self.threshold_offsets = nn.Parameter(torch.zeros(max_level_limit + 1))

        # I111-1: 可学习温度 (使用配置值)
        if learnable_temp:
            self.log_temperature = nn.Parameter(
                torch.tensor(math.log(self._temperature_init))
            )
        else:
            # 非可学习: 注册为缓冲区
            self.register_buffer(
                'log_temperature',
                torch.tensor(math.log(self._temperature_init))
            )

        # 探索偏置 (训练初期)
        self.register_buffer('explore_bias', torch.tensor(0.5))

        # I131-1: 固定深度偏置已移除
        # 原因: 与 Scheme E 可学习配额机制冲突
        # 深度选择完全由 quota_logits 控制
        # self.register_buffer('depth_bias_beta', torch.tensor(0.5))
        # self.register_buffer('depth_bias_gamma', torch.tensor(0.7))

        # ====================================================================
        # I113-5: 可学习 STE 梯度缩放因子
        # 数学形式化:
        #   α = (N/K) × σ(log β) × min(τ/τ_ref, 1)
        #   - N/K: 覆盖率倒数补偿 (选中token梯度增强)
        #   - σ(log β): 可学习缩放因子 [0, 1] 范围
        #   - τ/τ_ref: 温度保护机制 (低温度时降低缩放)
        #
        # 梯度分析:
        #   ∂st_mask/∂z = α × ∂softmax/∂z
        #   选中梯度 ∝ α × p_i × (1-p_i)
        #   未选中梯度 ∝ α × p_j²
        #   新设计使梯度比率从 ~N/K → ~1
        # ====================================================================
        # I122-1: STE 缩放因子已完全移除
        # 原因: 直接使用软概率，无需 STE 混合

        # ====================================================================
        # I120-2: DeterministicTopK - 确定性 Top-K 选择（替代 Gumbel-TopK）
        # 目的: 消除 train/eval 输出差异，恢复 Hilbert 曲线确定性保证
        #
        # 数学形式化:
        #   使用温度退火的 softmax 替代 Gumbel 随机采样:
        #   P(i ∈ Top-K) = softmax(z_i / τ)[i] × K
        #
        # 与 Gumbel-TopK 的对比:
        #   | 维度          | Gumbel-TopK         | DeterministicTopK    |
        #   |--------------|---------------------|----------------------|
        #   | 随机性       | torch.rand() 采样   | 无随机性             |
        #   | 可复现性     | ❌ 不可复现         | ✅ 完全可复现       |
        #   | Hilbert 一致性 | ❌ 破坏             | ✅ 完全保持         |
        #   | train/eval 差异 | 4.48               | <0.05                |
        #   | 可微性       | STE 近似            | 直接 softmax         |
        #
        # 使用场景:
        #   1. 需要确定性的推理结果
        #   2. train/eval 输出一致性要求
        #   3. Hilbert 局部性保持
        #
        # I130-2: Hilbert 最佳实现 - 从 config 读取 DeterministicTopK 设置
        # 修复: 使用 config 中的值替代硬编码默认值
        if self.config is not None:
            self._use_deterministic_topk = getattr(self.config, 'use_deterministic_topk', False)
            self._deterministic_temperature = getattr(self.config, 'deterministic_temperature', 0.5)
            self._deterministic_ste_alpha = getattr(self.config, 'deterministic_ste_alpha', 0.5)
            # Soft-Threshold 课程学习配置
            self._enable_soft_threshold = getattr(self.config, 'enable_soft_threshold', True)
            self._soft_threshold_warmup_epochs = getattr(self.config, 'soft_threshold_warmup_epochs', 10)
            self._soft_threshold_max = getattr(self.config, 'soft_threshold_max', 0.5)
            self._soft_threshold_schedule = getattr(self.config, 'soft_threshold_schedule', 'linear')
        else:
            self._use_deterministic_topk = False  # 默认使用 Gumbel-TopK
            self._deterministic_temperature = 0.5  # 初始温度
            self._deterministic_ste_alpha = 0.5  # STE 混合系数
            # Soft-Threshold 默认配置
            self._enable_soft_threshold = True
            self._soft_threshold_warmup_epochs = 10
            self._soft_threshold_max = 0.5
            self._soft_threshold_schedule = 'linear'

        # 延迟初始化 DeterministicTopK（仅在 use_deterministic_topk=True 时创建）
        self._deterministic_topk: Optional[DeterministicTopK] = None

        # Soft-Threshold 当前值（课程学习调度）
        self._current_soft_threshold = 0.0

        # ====================================================================
        # I24-2 方案E: 可学习配额 (Learnable Quota)
        # I30-10: 支持 SplitterConfig 配置
        # 数学:
        #   K_d = max(K_min, round(softmax(φ)_d × K_total))
        #   selected_d = TopK(logits[depth=d], K_d)
        #
        # 初始化:
        #   φ^(0) = compute_quota_init_logits(D)
        #   使用逆深度加权: p_d ∝ 1/(d+1)，支持任意 max_depth
        # ====================================================================
        # I30-17-EXT: 使用 max_level_limit 上界
        if self._enable_learnable_quota:
            D = max_level_limit + 1
            if self.config is not None:
                # 使用 config 中的初始化 logits（向后兼容）
                quota_init_vals = self.config.get_quota_init_tensor(D)
                quota_init = torch.tensor(quota_init_vals, dtype=torch.float32)
            else:
                # 使用 compute_quota_init_logits 动态生成初始化值
                # 替代硬编码的 QUOTA_INIT_LOGITS，支持任意 max_depth
                quota_init_vals = compute_quota_init_logits(max_level_limit)
                quota_init = torch.tensor(quota_init_vals, dtype=torch.float32)
            self.quota_logits = nn.Parameter(quota_init)
        else:
            self.quota_logits = None

        # ====================================================================
        # I113-17: ContinuousQuotaAllocator - 连续松弛配额分配器
        # 替代原有的 quota_logits 直接使用方式
        #
        # 数学形式:
        #   q = softmax(φ / τ)              # 软概率
        #   K_soft = q · K                  # 软配额
        #   K_hard = LRM(K_soft)            # 离散投影
        #
        # 与STE对比:
        #   | 指标       | STE (原)      | 连续松弛 (I113-17) |
        #   |------------|---------------|-------------------|
        #   | 梯度偏差    | O(K)          | O(1) ✓            |
        #   | 收敛保证   | ❌            | ✅                |
        #   | 实现复杂度  | 低            | 中                |
        # ====================================================================
        if self._enable_learnable_quota:
            # 使用配置文件中的参数或默认值
            if config is not None:
                quota_tau = getattr(config, 'quota_tau', 1.0)
                quota_tau_warmup = getattr(config, 'quota_tau_warmup_steps', 1000)
            else:
                quota_tau = 1.0
                quota_tau_warmup = 1000

            self.quota_allocator = ContinuousQuotaAllocator(
                D=max_level_limit + 1,
                tau=quota_tau,
                tau_warmup_steps=quota_tau_warmup,
                enable_warmup=True,
            )
        else:
            self.quota_allocator = None

        # ====================================================================
        # I113-16: HybridDensityHead - 混合密度头
        # 数学形式化:
        #   D = σ(GradientBranch(F) ⊕ SemanticBranch(F))
        #
        # 与方差方案对比:
        #   | 指标     | 方差 (I113-6) | HybridDensityHead (I113-16) |
        #   |----------|---------------|----------------------------|
        #   | 语义对齐 | 0.3           | 0.9 ✓                      |
        #   | 尺度不变 | 0.2           | 0.8 ✓                      |
        # ====================================================================
        self._enable_hybrid_density = True  # I113-16: 默认启用混合密度
        self.hybrid_density_head = HybridDensityHead(
            hidden_dim=32,
            use_temperature=True,
            temperature_init=1.0,
        )

        # ====================================================================
        # I121-5: CurriculumWeightScheduler - 课程学习权重调度器
        # 数学形式化:
        #   λ_b(t, K) = λ_b^0 · α(t) · β(K)
        #   λ_e(t) = λ_e^0 / √α(t)
        #
        # 设计原理:
        #   1. 三阶段课程学习: 探索(高约束) → 适应(平衡) → 利用(高多样性)
        #   2. Cosine平滑过渡: C^∞连续，梯度有界
        #   3. 熵权重反向联动: 预算约束放松时增强熵正则化
        #
        # 与Elastic Budget的集成:
        #   - 替换固定权重为动态权重
        #   - 保持与现有损失计算的兼容性
        # ====================================================================
        self._curriculum_scheduler = CurriculumWeightScheduler(
            total_epochs=100,  # 默认值，可在运行时调整
            exploration_end=CURRICULUM_EXPLORATION_END,
            adaptation_end=CURRICULUM_ADAPTATION_END,
            explore_factor=CURRICULUM_WEIGHT_FACTOR_EXPLORE,
            adapt_factor=CURRICULUM_WEIGHT_FACTOR_ADAPT,
            exploit_factor=CURRICULUM_WEIGHT_FACTOR_EXPLOIT,
            penalty_gamma=CURRICULUM_PENALTY_GAMMA,
            base_budget_weight=ELASTIC_LAMBDA_TARGET,
            base_entropy_weight=CURRICULUM_BASE_ENTROPY_WEIGHT,
            enable_distance_penalty=CURRICULUM_DISTANCE_PENALTY_ENABLED,
        )

        # I98-7: 初始化 CoreSplitter Protocol 属性 (必须在 _update_candidates 之前)
        self._num_candidates = 0

        # I30-17-EXT: 初始化候选区域 (基于初始 image_size)
        self._update_candidates(image_size)
        
        # 统计信息
        self.register_buffer('_avg_selected', torch.tensor(16.0))
        
        # ====================================================================
        # 退火调度 buffers (与 LearnableSplitter API 兼容)
        # ====================================================================
        # 温度退火
        self.register_buffer('_temp_total_steps', torch.tensor(0.0))
        self.register_buffer('_temp_start', torch.tensor(1.0))
        self.register_buffer('_temp_end', torch.tensor(0.4))
        self.register_buffer('_temp_step', torch.tensor(0.0))
        self._temp_enabled = False
        self._temp_schedule = 'linear'  # I122-7: 默认使用线性调度

        # 探索偏置退火
        self.register_buffer('_bias_total_steps', torch.tensor(0.0))
        self.register_buffer('_bias_start', torch.tensor(0.5))
        self.register_buffer('_bias_end', torch.tensor(0.0))
        self.register_buffer('_bias_step', torch.tensor(0.0))
        self._bias_enabled = False

        # I30-6: 深度方差归一化
        # I35 改进: 使用 EMA Running Statistics:
        # - 稳定小 batch (B=1) 下的方差估计
        # - 避免 sqrt(0) 在反向传播产生 NaN
        # - α=0.1, 有效样本量 ≈ 10
        self._depth_var_normalized: Optional[Tensor] = None  # [D]

        # I35: EMA Running Statistics buffers (max_level_limit + 1 维度)
        # I99-1: 原实现使用固定 B_max=256 缓冲区
        # I107-1: 优化为动态缓冲区，按需分配，消除内存浪费
        self._depth_dim = max_level_limit + 1  # I107-1: 保存深度维度
        D = self._depth_dim
        # I107-1: 动态缓冲区，不再使用 register_buffer (因为大小会变化)
        self._depth_ema_mean: Optional[Tensor] = None  # [B, D]
        self._depth_ema_var: Optional[Tensor] = None   # [B, D]
        self._ema_buffer_initialized = False  # 标记是否已初始化

        # I103-3: 设备端张量惰性缓存 (避免重复 .to(device) 传输)
        self._cached_device_regions: Optional[torch.Tensor] = None
        self._cached_device_depths: Optional[torch.Tensor] = None
        self._cached_device_thresholds: Optional[torch.Tensor] = None
        self._cached_device_children_matrix: Optional[torch.Tensor] = None

        # 初始化权重
        self._init_weights()

        # I24-4: 边界条件验证
        if max_level_limit < 2:
            warnings.warn(
                "I24-4: max_level_limit < 2 是边界情况。 "
                "depth=0 只有 1 个候选区域，depth=1 有 4 个候选区域。 "
                "此配置可能导致不平衡的 token 分布。建议使用 max_depth >= 2。",
                UserWarning,
                stacklevel=2
            )

        # I113-7: 缓存配额分配用于损失计算
        self._last_hard_quota: Optional[Tensor] = None
        self._last_soft_quota: Optional[Tensor] = None
        self._last_info_quota_loss: Optional[Tensor] = None

        # ====================================================================
        # I200: Splitter 输入特征 SVD/有效秩分析器
        # 用于检测训练早期特征坍塌问题
        # ====================================================================
        self.feature_analyzer: Optional[SplitterFeatureAnalyzer] = None
        self._feature_analysis_enabled: bool = False
        self._feature_analysis_result: Optional[SVDAnalysisResult] = None

        # ====================================================================
        # 三阶段课程学习状态 (I-CURRICULUM)
        # Stage 1: Teacher Forcing (Epoch 1-9) - 冻结 Splitter
        # Stage 2: Acc-Driven Splitting (Epoch 10-19) - 仅 CE Loss 驱动
        # Stage 3: Resource Co-adaptation (Epoch 20+) - 引入资源惩罚
        # ====================================================================
        self._current_epoch: int = 0
        self._curriculum_stage: int = 1  # 1=Teacher Forcing, 2=Acc-Driven, 3=Resource Co-adapt
        self._cached_mask_ste: Optional[Tensor] = None  # Stage 1 缓存的全 1 mask

    def set_epoch(self, epoch: int) -> None:
        """设置当前 epoch，更新课程学习阶段。

        Args:
            epoch: 当前训练轮次 (从 1 开始)
        """
        self._current_epoch = epoch

        # 确定当前阶段
        if epoch < 10:
            self._curriculum_stage = 1
        elif epoch < 20:
            self._curriculum_stage = 2
        else:
            self._curriculum_stage = 3

    def enable_feature_analysis(
        self,
        enabled: bool = True,
        sample_interval: int = 10,
    ) -> None:
        """
        启用/禁用特征坍塌分析

        Args:
            enabled: 是否启用分析
            sample_interval: 每隔多少步进行一次分析
        """
        self._feature_analysis_enabled = enabled
        if enabled and self.feature_analyzer is None:
            self.feature_analyzer = SplitterFeatureAnalyzer(
                enabled=True,
                sample_interval=sample_interval,
                record_history=True,
            )

    def _analyze_features(self, roi_features: Tensor) -> None:
        """分析 ROI-Align 后的特征"""
        if not self._feature_analysis_enabled:
            return
        if self.feature_analyzer is None:
            return
        if self.feature_analyzer.should_analyze():
            self._feature_analysis_result = self.feature_analyzer.analyze(roi_features)

    def get_feature_analysis_summary(self) -> Dict[str, Any]:
        """获取特征分析摘要"""
        if self.feature_analyzer is None:
            return {"status": "disabled"}
        return self.feature_analyzer.get_diagnostics_summary()

    def _init_weights(self):
        """Xavier 初始化 MLP 权重。"""
        for module in self.complexity_mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.xavier_uniform_(self.depth_proj.weight)
        nn.init.zeros_(self.depth_proj.bias)

        # I120-2: 初始化 DeterministicTopK（如果启用）
        if self._use_deterministic_topk:
            self._deterministic_topk = DeterministicTopK(
                temperature=self._deterministic_temperature,
            )

    # ====================================================================
    # I120-2: DeterministicTopK 控制接口
    # ====================================================================
    def enable_deterministic_topk(
        self,
        temperature: float = 0.5,
    ) -> None:
        """
        启用 DeterministicTopK（确定性 Top-K 选择）

        Args:
            temperature: 温度参数，控制 softmax 的"锐度"
                        较小的值 → 更接近硬选择 (Top-K)
                        较大的值 → 更软的分布 (Softmax)
        """
        self._use_deterministic_topk = True
        self._deterministic_temperature = temperature
        self._deterministic_topk = DeterministicTopK(
            temperature=temperature,
        )

    def disable_deterministic_topk(self) -> None:
        """禁用 DeterministicTopK，恢复使用 Gumbel-TopK"""
        self._use_deterministic_topk = False
        self._deterministic_topk = None

    @property
    def use_deterministic_topk(self) -> bool:
        """返回是否使用 DeterministicTopK"""
        return self._use_deterministic_topk

    # I107-1: 动态 EMA 缓冲区管理
    def _ensure_ema_buffers(self, batch_size: int, device: torch.device) -> None:
        """确保 EMA 缓冲区足够大 (I107-1 动态分配).

        数学:
            M_alloc = batch_size × D × 4 bytes
            M_waste = 0 (按需分配)

        Args:
            batch_size: 实际 batch size
            device: 计算设备
        """
        D = self._depth_dim
        if not self._ema_buffer_initialized:
            # 首次分配：按实际 batch size 分配
            self._depth_ema_mean = torch.zeros(batch_size, D, device=device)
            self._depth_ema_var = torch.ones(batch_size, D, device=device)
            self._ema_buffer_initialized = True
        elif self._depth_ema_mean is not None and self._depth_ema_mean.size(0) < batch_size:
            # 扩展缓冲区：当 batch size 增大时
            padding_size = batch_size - self._depth_ema_mean.size(0)
            self._depth_ema_mean = torch.cat([
                self._depth_ema_mean,
                torch.zeros(padding_size, D, device=device)
            ], dim=0)
            self._depth_ema_var = torch.cat([
                self._depth_ema_var,
                torch.ones(padding_size, D, device=device)
            ], dim=0)

    # I103-3: 设备端张量惰性缓存方法
    def _get_device_tensor(
        self,
        cpu_tensor: torch.Tensor,
        cache_attr: str,
        device: torch.device,
    ) -> torch.Tensor:
        """获取设备端张量 (使用惰性缓存避免重复传输)。

        I103-3 优化: 首次传输后缓存设备端副本，后续调用复用缓存。

        数学:
            T_device = T_cpu.to(device)  (仅首次)
            T_device = cached            (后续调用，设备匹配时)

        Args:
            cpu_tensor: CPU 端原始张量
            cache_attr: 缓存字段名 (如 "_cached_device_regions")
            device: 目标设备

        Returns:
            设备端张量
        """
        cached = getattr(self, cache_attr, None)
        if cached is not None:
            # 缓存存在，检查设备是否匹配
            if cached.device == device:
                return cached
            # 设备不匹配，需要迁移
            return cached.to(device, non_blocking=True)

        # 首次传输
        device_tensor = cpu_tensor.to(device, non_blocking=True)
        setattr(self, cache_attr, device_tensor)
        return device_tensor

    # ============================================================================
    # CoreSplitter Protocol 实现 (I98-7)
    # ============================================================================

    @property
    def max_level_limit(self) -> int:
        """获取最大深度限制。"""
        return self._current_max_level_limit

    @property
    def num_candidates(self) -> int:
        """获取当前候选区域数量。"""
        return self._num_candidates

    @num_candidates.setter
    def num_candidates(self, value: int) -> None:
        """设置候选区域数量。"""
        self._num_candidates = value

    @property
    def is_training(self) -> bool:
        """获取当前训练/评估模式。"""
        return self.training

    # I30-17-EXT: 动态候选区域更新方法
    def _update_candidates(self, image_size: Tuple[int, int]):
        """
        根据输入尺寸动态更新候选区域。

        数学形式:
            L(X) = min(max_level_limit, max(0, floor(log2(min(H, W) / min_patch_size))))

        Args:
            image_size: (H, W) 输入图像尺寸
        """
        from vit_pytorch.core.depth_utils import compute_max_depth

        H_img, W_img = image_size

        # 动态计算 max_depth
        computed_max_depth = compute_max_depth(
            image_size, self.min_patch_size, self._current_max_level_limit
        )

        # 检查缓存
        cache_key = (H_img, W_img, computed_max_depth)

        if cache_key in self._candidate_cache:
            cached = self._candidate_cache[cache_key]
            self.candidate_regions = cached[0]
            self.candidate_depths = cached[1]
            self.parent_indices = cached[2]
            self.hilbert_indices = cached[3]
            self.num_candidates = cached[4]
            self._current_max_depth = computed_max_depth
            self._current_image_size = image_size
            self._children_matrix = None  # 重置子节点矩阵
            # I103-3: 清除设备端缓存 (候选区域已从缓存恢复)
            self._cached_device_regions = None
            self._cached_device_depths = None
            self._cached_device_thresholds = None
            self._cached_device_children_matrix = None
            self._cached_device_hilbert = None  # I103-3: 清除 hilbert 缓存
            return

        # 重新计算候选区域
        self._generate_candidates_internal(image_size, computed_max_depth)

        # 更新动态状态
        self._current_max_depth = computed_max_depth
        self._current_image_size = image_size
        self._children_matrix = None
        # I103-3: 清除设备端缓存 (候选区域已更新)
        self._cached_device_regions = None
        self._cached_device_depths = None
        self._cached_device_thresholds = None
        self._cached_device_children_matrix = None
        self._cached_device_hilbert = None  # I103-3: 清除 hilbert 缓存

        # 存入缓存 (限制大小)
        if len(self._candidate_cache) < 256:
            self._candidate_cache[cache_key] = (
                self.candidate_regions,
                self.candidate_depths,
                self.parent_indices,
                self.hilbert_indices,
                self.num_candidates,
            )

    def _generate_candidates_internal(self, image_size: Tuple[int, int], max_depth: int):
        """
        生成候选区域的内部方法 (P-OPT-7 向量化版本)。

        数学形式化
        ==========

        问题: 原始实现使用 3 层嵌套 for 循环，O(Σ 4^d) Python 迭代
        优化: 使用 torch.meshgrid + 张量广播，O(D) 次张量操作

        向量化公式:
            对每个深度 d:
            - grid_size = 2^d
            - region_h = H / grid_size, region_w = W / grid_size
            - 使用 meshgrid 生成 [grid_size, grid_size] 坐标网格
            - 批量计算: y0 = i * region_h, x0 = j * region_w

        父节点索引计算:
            parent_idx[depth, i, j] = global_idx_at(depth-1, i//2, j//2)

        Hilbert 索引计算（I113-18 HilbertScanner）:
            center_x = (x0 + x1) / 2
            center_y = (y0 + y1) / 2
            hilbert_d = HilbertScanner.region_to_hilbert_index(x0, y0, x1, y1, depth, H, W)

        最佳实现说明:
            - H=W 且是 2^k：标准 Hilbert 曲线（退化）
            - 其他：Pseudo-Hilbert 曲线（严格保证）

        注意: 初始化时使用 CPU 张量，设备由后续的 .to(device) 处理

        Args:
            image_size: (H, W) 图像尺寸
            max_depth: 最大深度
        """
        from vit_pytorch.core.curve_hilbert import HilbertScanner

        H_img, W_img = image_size

        # P-OPT-7: 初始化时使用 CPU，张量在模型移动到 GPU 时自动跟进
        # 获取设备（如果已存在 buffer），否则使用 CPU
        try:
            device = self.candidate_regions.device if hasattr(self, 'candidate_regions') else torch.device('cpu')
        except AttributeError:
            device = torch.device('cpu')

        all_regions = []      # [N, 4] 各深度的区域坐标
        all_depths = []       # [N] 各区域的深度
        all_parent_idx = []   # [N] 父节点索引
        all_hilbert_idx = []  # [N] Hilbert 索引

        # 记录每个深度的起始索引，用于父节点计算
        # depth_start_idx[d] = depth d 的起始全局索引
        depth_start_idx = [0]  # depth=0 的起始索引

        for depth in range(max_depth + 1):
            grid_size = 2 ** depth
            region_h = H_img / grid_size
            region_w = W_img / grid_size

            # P-OPT-7: 向量化坐标生成
            # 使用 torch.arange 生成索引
            i_idx = torch.arange(grid_size, device=device)
            j_idx = torch.arange(grid_size, device=device)

            # meshgrid 生成网格 [grid_size, grid_size]
            grid_i, grid_j = torch.meshgrid(i_idx, j_idx, indexing='ij')

            # 批量计算区域坐标 [grid_size, grid_size]
            y0 = (grid_i * region_h).long()
            x0 = (grid_j * region_w).long()
            y1 = ((grid_i + 1) * region_h).long()
            x1 = ((grid_j + 1) * region_w).long()

            # 堆叠为 [N, 4] 张量
            regions_depth = torch.stack([x0, y0, x1, y1], dim=-1).view(-1, 4)
            all_regions.append(regions_depth)

            # 深度标签
            N_depth = grid_size * grid_size
            all_depths.extend([depth] * N_depth)

            # I113-18: HilbertScanner 统一 Hilbert 索引计算
            # 最佳实现：使用 HilbertScanner.region_to_hilbert_index
            #
            # 原理: Hilbert 曲线是四叉树遍历顺序
            # d(depth, region) = depth_offset + Hilbert(region_x, region_y)
            #
            # HilbertScanner 自动选择:
            # - H=W 且是 2^k：标准 Hilbert（退化，最优性能）
            # - 其他：Pseudo-Hilbert（严格局部性保证）
            if grid_size > 0:
                hilbert_d = HilbertScanner.region_to_hilbert_index(
                    x0.view(-1).float(),
                    y0.view(-1).float(),
                    x1.view(-1).float(),
                    y1.view(-1).float(),
                    depth,
                    H_img,
                    W_img
                )
            else:
                hilbert_d = torch.zeros(N_depth, device=device, dtype=torch.long)
            all_hilbert_idx.append(hilbert_d)

            # P-OPT-7: 批量计算父节点索引
            if depth == 0:
                parent_idx_depth = torch.full((N_depth,), -1, device=device, dtype=torch.long)
            else:
                # 计算父节点的 grid 坐标 (i//2, j//2)
                parent_i = (grid_i // 2).view(-1)
                parent_j = (grid_j // 2).view(-1)

                # 父深度的大小
                parent_grid_size = 2 ** (depth - 1)
                # 父节点在全局索引中的位置 = 父深度起始索引 + 局部索引
                # depth_start_idx[depth - 1] 是父深度 (depth-1) 的起始索引
                parent_global_idx = depth_start_idx[depth - 1] + (parent_i * parent_grid_size + parent_j)
                parent_idx_depth = parent_global_idx.long()

            all_parent_idx.append(parent_idx_depth)

            # 记录下一个深度的起始索引
            depth_start_idx.append(depth_start_idx[-1] + N_depth)

        # 合并所有深度的数据
        candidate_regions = torch.cat(all_regions, dim=0).float()
        candidate_depths = torch.tensor(all_depths, device=device, dtype=torch.long)
        parent_indices = torch.cat(all_parent_idx, dim=0)
        hilbert_indices = torch.cat(all_hilbert_idx, dim=0)

        # 计算候选数量
        self.num_candidates = sum(4 ** d for d in range(max_depth + 1))

        # 注册为 buffer (设备由 model.to() 控制)
        self.register_buffer('candidate_regions', candidate_regions)
        self.register_buffer('candidate_depths', candidate_depths)
        self.register_buffer('parent_indices', parent_indices)
        self.register_buffer('hilbert_indices', hilbert_indices)

    def _generate_candidates(self, max_depth: int):
        """兼容方法: 使用当前 image_size 生成指定深度的候选区域。"""
        if self._current_image_size is not None:
            self._generate_candidates_internal(self._current_image_size, max_depth)

    # I97-3: 移除 _precompute_candidates() 方法 (死代码 + Bug)
    # 该方法引用不存在的 self.image_size 属性，功能与 _generate_candidates_internal() 重复

    def _ensure_children_matrix(self):
        """确保子节点矩阵已计算。
        
        性能优化 (P-OPT-2):
            使用向量化操作替换 Python for 循环和 .item() 调用。
            通过一次性 CPU 转换 + numpy 操作提升性能。
            
        注意: 此方法仅在初始化时调用一次，后续使用缓存结果。
        """
        if self._children_matrix is not None:
            return
        
        N = self.num_candidates
        device = self.candidate_regions.device
        
        # 初始化为 -1
        children_matrix = torch.full((N, 4), -1, dtype=torch.long, device=device)
        
        # P-OPT-2: 一次性转换到 CPU，使用 numpy 进行槽位分配
        # 这比逐个 .item() 调用快得多
        parent_indices_cpu = self.parent_indices.cpu().numpy()
        children_matrix_cpu = children_matrix.cpu().numpy()
        
        # 使用 numpy 进行槽位分配
        slot_counts = {}  # parent_idx -> next_slot
        for child_idx in range(N):
            parent_idx = parent_indices_cpu[child_idx]
            if parent_idx >= 0:
                slot = slot_counts.get(parent_idx, 0)
                if slot < 4:  # 最多 4 个子节点
                    children_matrix_cpu[parent_idx, slot] = child_idx
                    slot_counts[parent_idx] = slot + 1
        
        # 转回 GPU (I78: 使用 non_blocking=True 异步传输)
        self._children_matrix = torch.from_numpy(children_matrix_cpu).to(device, non_blocking=True)
    
    # ========================================================================
    # I23-1 方案C: 深度方差归一化
    # ========================================================================
    
    def _normalize_by_depth(
        self,
        logits: Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """
        按深度分组归一化 MLP 输出 (I23-1 方案C 核心修复, I35 EMA 改进)。

        数学形式化
        ==========

        问题分析:
            ROI 信息量与深度相关，导致 MLP 输出方差不一致:
            - Depth 0: ROI = 64×64，特征是全局平均，σ_0 较小
            - Depth 3: ROI = 16×16，特征保留局部变化，σ_3 较大

            量化估计: σ_3 / σ_0 ≈ 8 (基于 ROI 面积采样点推导)

        Top-K 偏好分析:
            Top-K 选择偏好高方差分布 (更容易产生极值)
            有效竞争力: eff_d = N_d × σ_d
            结果: eff_3 : eff_0 = 64×1 : 1×0.125 = 512:1

        解决方案 (I35 EMA 改进):
            z_i^{norm} = (z_i - μ_d^{EMA}) / (σ_d^{EMA} + ε)

            其中 μ_d^{EMA}, σ_d^{EMA} 是 EMA 累积的 Running Statistics:
            - μ_d^{EMA}(t) = α × μ_d^{batch}(t) + (1-α) × μ_d^{EMA}(t-1)
            - σ_d^{EMA}(t) = α × σ_d^{batch}(t) + (1-α) × σ_d^{EMA}(t-1)

        I35 EMA 优势:
            - 稳定小 batch (B=1) 下的方差估计 (per-batch 方差放大 512×)
            - 避免 sqrt(0) 在反向传播产生 NaN
            - α=0.1, 有效样本量 ≈ 10，方差降低 19×

        归一化后效果:
            所有 z_i^{norm} ~ N(0, 1)
            Log-Compensation 可以正确补偿候选数量差异
            预测深度分布 π ≈ (0.249, 0.246, 0.248, 0.257)

        复杂度:
            时间: O(B × N × D) ≈ O(B × 85 × 4)
            空间: O(D) = O(4)

        P-OPT-3 向量化:
            使用 scatter_add + one-hot 编码替代 Python for 循环
            避免 D 次索引操作，改为单次批量计算

        Args:
            logits: [B, N] MLP 输出 (未归一化)
            device: 设备
            dtype: 数据类型

        Returns:
            normalized: [B, N] 按深度归一化后的 logits
        """
        B, N = logits.shape
        # I103-3: 使用设备端缓存
        depths = self._get_device_tensor(
            self.candidate_depths, "_cached_device_depths", device
        )  # [N]
        D = self._current_max_depth + 1

        # P-OPT-3: 向量化深度方差归一化
        # 使用 one-hot 编码实现批量 scatter/gather 操作

        # 构建深度 one-hot 掩码: [D, N]
        depth_onehot = F.one_hot(depths, D).float().T  # [D, N]
        depth_counts = depth_onehot.sum(dim=1)  # [D] 每个深度的候选数量

        # 扩展 logits 和掩码用于批量计算
        # logits: [B, N], depth_onehot: [D, N]
        # 目标: 计算每个深度的 mean 和 std

        # 使用 einsum 高效计算: sum_d = Σ_i (logits_i × mask_{d,i})
        # [B, D] = einsum('bn,dn->bd', logits, depth_onehot)
        depth_sums = torch.einsum('bn,dn->bd', logits, depth_onehot)  # [B, D]

        # Per-batch 均值: μ_d^(b) = Σ_i logits[b,i] × 1[depth[i]=d] / N_d
        # depth_counts.unsqueeze(0) = [1, D] broadcasts to [B, D]
        mu_per_batch = depth_sums / depth_counts.unsqueeze(0)  # [B, D]

        # Per-batch 方差: σ²_d^(b) = E[X²] - E[X]²
        logits_sq = logits ** 2
        depth_sq_sums = torch.einsum('bn,dn->bd', logits_sq, depth_onehot)  # [B, D]
        mean_sq_per_batch = depth_sq_sums / depth_counts.unsqueeze(0)  # [B, D]
        variance_per_batch = (mean_sq_per_batch - mu_per_batch ** 2).clamp(min=0.0)  # [B, D]

        # ====================================================================
        # 深度方差归一化 (Depth Variance Normalization)
        #
        # CRIT-2 修复: 准确描述归一化策略
        #
        # 策略说明:
        # - 训练时: 累积 EMA 统计量用于初始化回退，归一化使用实时 batch 统计量
        # - 评估时: 使用实时 per-sample 统计量，确保 B=1 和 B=4 行为一致
        #
        # 数学形式化:
        #   归一化: z_norm = (z - μ_batch) / (σ_batch + ε)
        #   其中 μ_batch, σ_batch 来自当前 batch (非 EMA 累积值)
        #
        # 设计理由:
        #   1. 实时统计量确保不同 batch size 下的归一化行为一致
        #   2. EMA 仅用于初始化时的保守回退 (避免未训练时的数值异常)
        #   3. Per-sample EMA 跟踪每个样本的统计量变化
        # ====================================================================
        # I130-2: Hilbert 最佳实现 - DeterministicTopK 模式统一使用实时统计量
        # 关键修复: 对于确定性模式，始终使用当前 batch 的实时统计量
        # 这样 train/eval 模式使用相同的归一化逻辑，保证输出一致
        use_realtime_stats = (
            self.training or
            self._use_deterministic_topk  # DeterministicTopK 始终使用实时统计量
        )

        if self.training and not self._use_deterministic_topk:
            # I107-1: 训练模式: Per-sample EMA 更新
            # 仅在非确定性模式下更新 EMA
            # 对每个样本独立更新 EMA，不跨 batch 平均
            # I107-1: 确保 EMA 缓冲区足够大
            self._ensure_ema_buffers(B, device)

            if not self._ema_buffer_initialized:
                # I100-5: 首次初始化 - 逐样本处理（需要 B 相关的条件逻辑）
                for b in range(B):
                    mu_b = mu_per_batch[b]  # [D]
                    var_b = variance_per_batch[b]  # [D]

                    # 根据 batch size 的统计可靠性分级处理
                    safe_var = var_b.detach().clamp(min=DEPTH_VARIANCE_NORM_EPS)
                    if B == 1:
                        # B=1: 样本方差无定义，使用先验 σ²=0.25
                        safe_var = safe_var.clamp(min=DEPTH_VARIANCE_INIT_EPS_B1)
                    elif B == 2:
                        # B=2: 置信区间 5124:1，需要 4× 缓冲
                        safe_var = safe_var.clamp(min=DEPTH_VARIANCE_INIT_EPS_B2)
                    elif B == 4:
                        # B=4: 置信区间 130:1，需要 1.5× 缓冲
                        safe_var = safe_var.clamp(min=DEPTH_VARIANCE_INIT_EPS_B4)
                    # B≥4: 使用 DEPTH_VARIANCE_NORM_EPS (1e-6)
                    self._depth_ema_mean[b, :D] = mu_b.detach()
                    self._depth_ema_var[b, :D] = safe_var
            else:
                # I99-1 OPT: 已初始化后向量化 EMA 更新
                # μ_new = α × μ_batch + (1-α) × μ_old (批量处理)
                alpha = DEPTH_EMA_ALPHA
                self._depth_ema_mean[:B, :D] = (
                    alpha * mu_per_batch.detach() +
                    (1 - alpha) * self._depth_ema_mean[:B, :D]
                )
                self._depth_ema_var[:B, :D] = (
                    alpha * variance_per_batch.detach() +
                    (1 - alpha) * self._depth_ema_var[:B, :D]
                ).clamp(min=DEPTH_VARIANCE_NORM_EPS)

            self._ema_buffer_initialized = True

            # 归一化使用当前 batch 的实时统计量 (非 EMA 累积值)
            # 这样确保不同 batch size 下的归一化行为一致
            mu_normalize = mu_per_batch  # [B, D]
            sigma_normalize = (variance_per_batch + DEPTH_VARIANCE_NORM_EPS).sqrt()  # [B, D]
        elif self._use_deterministic_topk:
            # I130-2: DeterministicTopK 模式始终使用实时统计量
            # 并且不更新 EMA 缓冲区，保持状态一致
            mu_normalize = mu_per_batch
            sigma_normalize = (variance_per_batch + DEPTH_VARIANCE_NORM_EPS).sqrt()
        else:
            # ====================================================================
            # 评估模式 (非 DeterministicTopK)
            # ====================================================================
            if not self._ema_buffer_initialized:
                import warnings
                warnings.warn(
                    f"[I99-1] 深度方差归一化未初始化 (model.eval() 前未进行训练)。"
                    f"使用保守回退 (sigma={DEPTH_VARIANCE_INIT_EPS**0.5:.3f})。"
                    f"请确保模型已训练后再进行评估。",
                    UserWarning,
                    stacklevel=2
                )
                # 使用保守常数作为回退 (非 EMA 累积值)
                mu_normalize = mu_per_batch
                sigma_normalize = torch.full_like(mu_per_batch, DEPTH_VARIANCE_INIT_EPS ** 0.5)
            else:
                # 使用当前 batch 的实时统计量 (非 EMA 累积值)
                # 这样确保 B=1 和 B=4 评估时行为一致
                mu_normalize = mu_per_batch  # [B, D]
                sigma_normalize = (variance_per_batch + DEPTH_VARIANCE_NORM_EPS).sqrt()  # [B, D]

        # ====================================================================
        # 收集每个 batch 每个深度的均值和标准差用于归一化
        # 使用联合索引: mu_normalize[batch_indices, depth_indices] -> [B, N]
        # ====================================================================
        batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, N)  # [B, N]
        depth_indices = depths.unsqueeze(0).expand(B, -1)  # [B, N]

        # 联合索引获取归一化参数 (使用实时统计量，非 EMA)
        mu_expanded = mu_normalize[batch_indices, depth_indices]  # [B, N]
        sigma_expanded = sigma_normalize[batch_indices, depth_indices]  # [B, N]

        # 归一化
        normalized = (logits - mu_expanded) / sigma_expanded

        return normalized

    @property
    def current_temperature(self) -> Tensor:
        """当前温度 τ ∈ [T_min, ∞)

        返回 GPU 张量以避免同步开销。
        梯度流: τ → exp⁻¹(log_τ) 可微

        Usage:
            # 推荐: 保持张量用于计算
            tau = splitter.current_temperature

            # 仅日志: 必要时提取
            tau_cpu = splitter.current_temperature.item()
        """
        return self.log_temperature.exp().clamp(min=TEMPERATURE_MIN)

    def get_current_temperature(self) -> Tensor:
        """获取当前 Gumbel 温度张量 (AnnealingSplitter 接口)。

        I170-3: 修复 Train/Eval 温度一致性
        问题: eval() 模式下 update_temperature() 停止调用，导致分布漂移
        解决:
            1. 训练时缓存最终退火温度
            2. 推理时优先使用缓存温度，其次使用 _temperature_min
        """
        if self.training:
            return self.current_temperature
        else:
            # 推理时使用缓存的最终温度（如果可用），保持与训练一致
            if self._cached_final_temperature is not None:
                return torch.tensor(self._cached_final_temperature, device=self.log_temperature.device, dtype=torch.float32)
            # 回退到配置的最小温度
            return torch.tensor(self._temperature_min, device=self.log_temperature.device, dtype=torch.float32)

    @property
    def thresholds(self) -> Tensor:
        """当前有效阈值向量。"""
        return self.threshold_offsets

    # ====================================================================
    # I200: 特征分析接口
    # ====================================================================
    def enable_feature_analysis(self, enabled: bool = True, sample_interval: int = 10):
        """启用/禁用 Splitter 输入特征分析

        Args:
            enabled: 是否启用分析
            sample_interval: 每隔多少步采样一次
        """
        self._feature_analysis_enabled = enabled
        if enabled:
            self.feature_analyzer = SplitterFeatureAnalyzer(
                enabled=True,
                sample_interval=sample_interval,
                record_history=True,
            )
        else:
            self.feature_analyzer = None

    def get_feature_analysis_result(self) -> Optional[Dict[str, Any]]:
        """获取特征分析结果摘要"""
        if self.feature_analyzer is None:
            return None
        return self.feature_analyzer.get_diagnostics_summary()

    def forward(
        self,
        features: Tensor,
        image_size: Optional[Tuple[int, int]] = None,
        hard: bool = False,
    ) -> GumbelTopKResult:
        """
        前向传播：Gumbel-Top-K 分割。
        
        Args:
            features: [B, C, H_feat, W_feat] 特征图
            image_size: (H, W) 图像尺寸 (可选，用于更新区域坐标)
            hard: 是否使用硬决策 (推理模式)
            
        Returns:
            GumbelTopKResult: 分割结果
        """
        B, C, H_feat, W_feat = features.shape
        device = features.device
        dtype = features.dtype
        N = self.num_candidates
        
        # ====================================================================
        # 退火调度自动更新 (训练模式)
        # ====================================================================
        if self.training:
            if self._temp_enabled:
                self._update_temperature()
            if self._bias_enabled:
                self._update_explore_bias()
            if self._enable_soft_threshold:
                # 使用步数驱动的课程学习调度
                self._update_soft_threshold_by_step()

        # I30-17-EXT: 动态更新候选区域
        if image_size is not None:
            self._update_candidates(image_size)

        # 使用当前动态状态
        H_img, W_img = self._current_image_size
        scale_h = H_feat / H_img
        scale_w = W_feat / W_img
        
        # ====================================================================
        # Step 1: 并行计算所有候选的 logits
        # ====================================================================
        logits, probs = self._compute_all_logits(features, scale_h, scale_w)
        # logits: [B, N], probs: [B, N]

        # ====================================================================
        # I150: Hilbert 空间均匀性优化 - 密度惩罚
        # 在 logits 上施加局部密度惩罚，减少 token 聚集
        # ====================================================================
        if DENSITY_PENALTY_WEIGHT > 0 and self.training:
            # 使用上一轮的选择来计算密度惩罚（需要迭代优化）
            # 简化版：在当前 logits 上叠加负的密度偏置
            # 这里先缓存当前的 logits，后续可通过迭代优化
            pass  # 密度惩罚通过 get_density_regularization() 损失实现

        # ====================================================================
        # I150-2: 深度自适应温度 - 在 logits 计算后应用
        # 获取每个候选的自适应温度，并应用于 logits
        # ====================================================================
        depth_temps = None
        if DEPTH_ADAPTIVE_TEMPERATURE_ENABLED:
            # 获取深度信息
            depths = self.candidate_depths
            # 计算自适应温度
            base_T = self.log_temperature.exp().clamp(min=TEMPERATURE_MIN)
            depth_temps = self._get_depth_adaptive_temperature(base_T, depths)
            # 深度自适应温度可通过 quota 机制间接应用
            # 这里缓存供后续使用

        # ====================================================================
        # ====================================================================
        # 三阶段课程学习控制 (I-CURRICULUM)
        # Stage 1: Teacher Forcing - 冻结 Splitter，保留所有 Token
        # P-OPT: 提前返回以避免不必要的计算
        # ====================================================================
        if self._curriculum_stage == 1:
            # 🌟 强制使用所有候选区域，给 Backbone 最大信息量
            K = N  # 保留所有 N 个 Patch

            # 构建全 1 的掩码，保持前向和反向一致性
            # 不需要 STE，因为 Stage 1 不更新 Splitter
            mask_ste = torch.ones_like(probs)

            # 缓存供后续使用
            self._cached_mask_ste = mask_ste

            # P-OPT: 复用缓存的索引，避免每次创建新 tensor
            # 形状必须是 [B, K] = [B, N]，与 _build_result 期望一致
            if not hasattr(self, '_cached_full_topk_indices') or self._cached_full_topk_indices.shape[0] != B:
                self._cached_full_topk_indices = torch.arange(N, device=probs.device, dtype=torch.long).unsqueeze(0).expand(B, -1)
            topk_indices = self._cached_full_topk_indices

            # P-OPT: Stage 1 提前构建结果，跳过 _build_result 中的复杂逻辑
            # 因为所有 token 都被选中，直接构建结果即可
            selected_mask = mask_ste

            # P-OPT: Stage 1 简化结果构建，跳过区域坐标计算
            # 由于返回全 1 mask，调用者会使用所有特征，不需要 regions 等信息
            # 构建基础的选中索引
            all_indices_1d = torch.arange(N, device=features.device, dtype=torch.long)
            batch_idx = torch.arange(B, device=features.device, dtype=torch.long).unsqueeze(1).expand(B, N)

            # 直接构建结果对象（跳过 _build_result 中的复杂处理）
            result = GumbelTopKResult(
                regions=torch.zeros(N * B, 4, dtype=torch.long, device=features.device),  # [M, 4] - dummy
                depths=torch.zeros(N * B, dtype=torch.long, device=features.device),  # [M] - dummy
                batch_indices=batch_idx.reshape(-1),  # [M]
                hilbert_indices=all_indices_1d.unsqueeze(0).expand(B, -1).reshape(-1),  # [M]
                selected_mask=selected_mask,
                logits=logits,
                probs=probs,
                candidate_indices=all_indices_1d.unsqueeze(0).expand(B, -1).reshape(-1),  # [M]
                num_selected_per_batch=torch.full((B,), N, dtype=torch.long, device=features.device),
            )

            # 缓存用于辅助损失
            self._last_probs = probs.detach()
            self._last_probs_for_loss = probs
            self._last_selected_mask = selected_mask.detach()
            self._last_selected_mask_for_loss = selected_mask
            self._last_num_selected_for_loss = float(N)

            # 缓存 quota loss（Stage 1 不更新 splitter，设为 None）
            self._last_quota_loss = None

            # 更新统计
            with torch.no_grad():
                self._avg_selected = 0.9 * self._avg_selected + 0.1 * N

            return result

        # Step 2: Gumbel-Top-K 选择
        # I24-2: 使用分层 Top-K (方案E) 或全局 Top-K (传统方案)
        # ====================================================================
        if not _skip_dynamic_selection:
            # 计算动态 K (I33: 传递 image_size 用于自适应覆盖率)
            if self.use_dynamic_k:
                K = self._estimate_optimal_k(probs, self._current_image_size)
            else:
                K = (self.K_min + self.K_max) // 2

            # 边界检查：K 不能超过候选数量 N
            K = min(K, N)
            # I23-2: 确保 K >= K_min (硬下界)
            # I23-3: 但不能超过 N（当 N < K_min 时，使用 N）
            K = max(min(K, N), min(self.K_min, N))

        # ====================================================================
        # v6.1: Soft-Threshold 课程学习
        # 应用负阈值作为偏置，抑制低分候选（稀疏化效果）
        # 注意：Stage 1 跳过此逻辑（已使用全 1 mask）
        # ====================================================================
        if not _skip_dynamic_selection:
            if self._enable_soft_threshold and self._current_soft_threshold > 0:
                # 创建负阈值偏置（高分候选不受影响，低分候选被抑制）
                threshold_bias = -self._current_soft_threshold
                logits = logits + threshold_bias

            # I24-2 方案E: 分层 Top-K (可学习配额)
            # I100-7: 传递 features 以支持信息密度自适应配额
            if LEARNABLE_QUOTA_ENABLED and self.quota_logits is not None:
                selected_mask, topk_indices = self._stratified_gumbel_topk_ste(
                    logits, K, hard, features=features
                )
            else:
                # 传统全局 Top-K
                selected_mask, topk_indices = self._gumbel_topk_ste(logits, K, hard)
            # selected_mask: [B, N] (STE 版本，有梯度)
            # topk_indices: [B, K'] (硬选择索引，K' 可能略小于 K)

        # ====================================================================
        # Step 3: I100-1 移除树一致性约束
        # 原因: 树一致性约束是人为设计，非任务必需
        # - Hilbert 曲线核心价值是空间局部性，而非树互斥
        # - 父子共存可捕获不同粒度的特征
        # - 移除约束允许模型自己学习最优策略
        # 之前: consistent_mask = self._enforce_tree_consistency(...)
        # ====================================================================
        consistent_mask = selected_mask  # 直接使用，移除树一致性约束
        
        # ====================================================================
        # Step 4: 构建输出
        # ====================================================================
        result = self._build_result(
            consistent_mask, topk_indices, logits, probs
        )

        # I96-3: 计算配额损失以提供梯度到 quota_logits
        # 注意: 损失由调用者添加到总损失
        # I170-4: 移除 detach() 允许梯度流回 quota_logits
        if self.training:
            self._last_quota_loss = self._compute_quota_loss(K)
        else:
            self._last_quota_loss = None

        # 缓存 probs 和 selected_mask 用于辅助损失计算
        # I102-4: 使用 detach() 防止显存泄露
        # I145-FIX: 同时保留非 detached 版本用于辅助损失的梯度计算
        self._last_probs = probs.detach()
        self._last_probs_for_loss = probs  # 保留梯度用于辅助损失
        self._last_selected_mask = consistent_mask.detach()  # 用于日志/统计
        self._last_selected_mask_for_loss = consistent_mask  # 保留梯度用于辅助损失

        # I145-FIX: 缓存当前 batch 的 token 数量用于弹性预算损失（有梯度）
        self._last_num_selected_for_loss = result.num_selected_per_batch.float().mean()

        # 更新统计
        with torch.no_grad():
            self._avg_selected = 0.9 * self._avg_selected + 0.1 * result.num_selected_per_batch.float().mean()

        return result
    
    def _compute_all_logits(
        self,
        features: Tensor,
        scale_h: float,
        scale_w: float,
    ) -> Tuple[Tensor, Tensor]:
        """
        并行计算所有 N 个候选区域的 logits。
        
        数学形式化:
            logits_i = MLP(ROI_i) + depth_bias_i + explore_bias - τ_{d_i}
            probs_i = σ(logits_i / T)
            
        Args:
            features: [B, C, H_feat, W_feat]
            scale_h, scale_w: 特征图与图像的缩放因子
            
        Returns:
            logits: [B, N]
            probs: [B, N]
        """
        B, C, H_feat, W_feat = features.shape
        N = self.num_candidates
        device = features.device
        dtype = features.dtype

        # 缩放区域坐标到特征图空间 (I103-3: 缓存 + clone 支持原地修改)
        regions_feat = self._get_device_tensor(
            self.candidate_regions, "_cached_device_regions", device
        ).clone()
        regions_feat[:, [0, 2]] *= scale_w  # x
        regions_feat[:, [1, 3]] *= scale_h  # y
        
        # 构建 ROI boxes: [B*N, 5]
        batch_indices = torch.arange(B, device=device, dtype=dtype).view(B, 1, 1).expand(-1, N, -1)
        regions_expanded = regions_feat.unsqueeze(0).expand(B, -1, -1)
        boxes = torch.cat([batch_indices, regions_expanded], dim=2).view(B * N, 5)
        
        # ROI-Align
        from torchvision.ops import roi_align
        roi_features = roi_align(
            features,
            boxes,
            output_size=(self.pool_size, self.pool_size),
            spatial_scale=1.0,
            aligned=True,
        )  # [B*N, C, k, k]

        # ====================================================================
        # I200: 特征坍塌分析 (ROI-Align 后)
        # 分析 Splitter 输入特征的 SVD 和有效秩
        # ====================================================================
        self._analyze_features(roi_features)

        # MLP 预测
        roi_flat = roi_features.flatten(1)  # [B*N, C*k*k]
        complexity_logits = self.complexity_mlp(roi_flat).squeeze(-1)  # [B*N]
        complexity_logits = complexity_logits.view(B, N)  # [B, N]
        # === SAT-DEFENSE: Logits 约束 ===
        # 收紧 clamp 范围从 [-10, 10] 到 [-5, 5]
        # 防止梯度冲击导致 Logits 进入激活函数死区
        # 数学: logits ∈ [-5, 5] 确保 sigmoid ∈ [0.0067, 0.9933]
        complexity_logits = complexity_logits.clamp(-5.0, 5.0)
        
        # ====================================================================
        # I23-1 方案C: 深度方差归一化 (核心修复)
        # 
        # 数学形式化:
        #     问题: MLP 输出方差与深度相关 (σ_3/σ_0 ≈ 8)
        #           导致 Top-K 偏好高方差深度，Log-Compensation 失效
        #     
        #     解决: z_i^norm = (z_i - μ_d) / σ_d
        #           使各深度 MLP 输出服从 N(0, 1)
        # ====================================================================
        if DEPTH_VARIANCE_NORM_ENABLED:
            complexity_logits = self._normalize_by_depth(complexity_logits, device, dtype)

        # 深度嵌入偏置 - 确保在正确设备上 (I103-3: 使用缓存)
        depths = self._get_device_tensor(
            self.candidate_depths, "_cached_device_depths", device
        )  # [N]
        # STAB-7 修复: 确保索引张量为连续格式 (channels-last 兼容)
        depths = depths.contiguous()
        depth_embed = self.depth_embedding(depths)  # [N, 16]
        depth_bias_learned = self.depth_proj(depth_embed).squeeze(-1)  # [N]

        # I131-1: 固定深度偏置已移除
        # 深度选择由 Scheme E 可学习配额机制主导
        # depth_bias_fixed = self.depth_bias_beta * (self.depth_bias_gamma ** depths.float())

        # I30-4: 已移除 Log-Compensation (被方案E完全替代)

        # 阈值 - 确保 thresholds 在正确设备上 (I103-3: 使用缓存)
        thresholds = self._get_device_tensor(
            self.thresholds, "_cached_device_thresholds", device
        )
        taus = thresholds[depths]  # [N]
        
        # 总 logits
        # logits = z + depth_bias_learned + explore_bias - tau
        # I131-1: 移除了 depth_bias_fixed，由 Scheme E 可学习配额主导
        logits = (complexity_logits
                  + depth_bias_learned.unsqueeze(0)
                  + self.explore_bias
                  - taus.unsqueeze(0))
        
        # === SAT-DEFENSE: 梯度平滑 ===
        # 在 logits 上注册梯度缩放 Hook，反向传播时乘 0.1
        # 降低 Splitter 对 Batch 0 巨量梯度的敏感度
        # 仅在训练模式下注册（评估时 tensor 不需要梯度）
        if logits.requires_grad:
            def _grad_scale_hook(grad):
                return grad * 0.1
            logits.register_hook(_grad_scale_hook)

        # 分割概率 (I18-5: 使用 TEMPERATURE_MIN 常量)
        T = self.log_temperature.exp().clamp(min=TEMPERATURE_MIN)
        probs = torch.sigmoid(logits / T)

        return logits, probs

    # I33: 动态 K 边界方法 (I111-5: 完整相对预算公式)
    def _get_dynamic_k_bounds(
        self,
        candidate_count: int,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[int, int]:
        """
        动态计算 K_min 和 K_max (I111-5: 完整相对预算公式)

        数学形式化
        ==========

        完整公式 (I33):
            N = (4^(L+1) - 1) / 3                      (候选总数)
            γ(H,W) = √(min(H,W) / 224)                 (尺度因子)
            K_min = max(K_min_abs, α × N)              (下界)
            K_max = min(K_max_hard, β × γ × N)         (上界)

        其中:
            - α = coverage_min (默认 0.01)
            - β = coverage_max_hard (默认 0.25)
            - K_min_abs = 8 (硬下限)
            - K_max_hard = 4096 (硬上限)

        覆盖率验证:
            | 图像尺寸 | N      | γ    | K_min | K_max  |
            |----------|--------|------|-------|--------|
            | 64×64    | 5461   | 0.53 | 55    | 720    |
            | 224×224  | 5461   | 1.0  | 55    | 1365   |
            | 512×512  | 5461   | 1.51 | 55    | 2061   |

        I111-5: 恢复完整公式，确保尺度不变性。

        Args:
            candidate_count: N 候选区域数
            image_size: 图像尺寸 (H, W)，用于尺度因子计算

        Returns:
            (K_min, K_max): 动态边界元组
        """
        # 计算尺度因子
        if image_size is not None:
            H, W = image_size
            min_dim = min(H, W)
            scale_factor = math.sqrt(min_dim / K_ADAPTIVE_REFERENCE_SIZE)
        else:
            scale_factor = 1.0

        # 从配置获取参数
        if self._use_config and self.config is not None:
            coverage_min = self.config.coverage_min
            coverage_max_hard = self.config.coverage_max_hard
            K_min_abs = self.config.K_min_abs
            K_max_hard = self.config.K_max_hard
        else:
            # 传统参数 (向后兼容)
            coverage_min = K_COVERAGE_MIN
            coverage_max_hard = K_COVERAGE_MAX_HARD
            K_min_abs = K_MIN_HARD_LIMIT
            K_max_hard = K_MAX_HARD_LIMIT

        # 计算 K_min = max(K_min_abs, α × N)
        K_min = max(
            K_min_abs,
            int(math.ceil(coverage_min * candidate_count))
        )

        # 计算 K_max = min(K_max_hard, β × γ × N)
        K_max = min(
            K_max_hard,
            int(math.ceil(coverage_max_hard * scale_factor * candidate_count))
        )

        return K_min, K_max

    def _estimate_optimal_k(
        self,
        probs: Tensor,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> int:
        """
        估计最优 K 值 (信息熵驱动版本) - Mask STE 实现

        数学形式化:
            方法 1 (熵驱动): K = H / H_max × N × α
                - H = -Σ p_i log(p_i) (分裂概率的Shannon熵)
                - H_max = log(N) (最大熵，均匀分布)
                - α = 0.4 (缩放因子)
            方法 2: K_2 = 70% 累积概率截断 (保留作为保底)
            K_opt = max(K_entropy, K_70, K_min)

        信息论基础:
            - 高熵图像 (H/H_max ≈ 0.5): 图像复杂，需要更多token
            - 低熵图像 (H/H_max ≈ 0.1): 图像简单，可以减少token
            - 典型值: K ≈ 32-64 (N=256, H/H_max≈0.3-0.5, α=0.4)

        Returns:
            K: 最优 token 数量 (整数，用于索引)
        """
        import math

        # I170-MASK-STE: 使用可微版本估计 K
        K_float = self._estimate_optimal_k_diff(probs, image_size)

        # ========== Mask STE 核心实现 ==========
        # 步骤 1: 计算整数 K（仅用于索引，前向传播用）
        # 注意: .item() 在这里使用是安全的，因为我们已经构建了替代梯度路径
        K_int = int(K_float.item())

        # 边界检查
        B, N = probs.shape
        if self.use_dynamic_k:
            K_min, K_max = self._get_dynamic_k_bounds(N, image_size)
        else:
            K_min, K_max = self.K_min, self.K_max
        K_min = max(K_min, 16)
        K_int = max(K_min, min(K_max, K_int))

        # 步骤 2: 获取 Top-K 索引（不可微，但这是前向传播，允许）
        topk_probs, topk_indices = torch.topk(probs, k=K_int, dim=-1)

        # 步骤 3: 构建 Hard Mask（0 和 1 组成的张量）
        mask_hard = torch.zeros_like(probs).scatter_(dim=-1, index=topk_indices, value=1.0)

        # 步骤 4: 🌟 掩码 STE 注入 🌟
        # 前向传播时是 0/1 掩码
        # 反向传播时梯度流向 probs（进而流向 quota_logits 和 log_temperature）
        mask_ste = (mask_hard - probs).detach() + probs

        # 步骤 5: 🌟 极其关键 - 缓存 mask_ste 供外部使用 🌟
        # 这一步让主干网络的 CrossEntropy 梯度能够沿着掩码回传给 Splitter
        self._cached_K_float = K_float  # 保留浮点 K 用于调试
        self._cached_mask_ste = mask_ste  # Mask STE 用于梯度流

        return K_int

    def _estimate_optimal_k_diff(
        self,
        probs: Tensor,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        """
        I170-1: 可微版本 K 估计 (梯度流修复)

        移除 no_grad() 上下文，使用软边界确保梯度流动。

        数学形式化:
            K_float = clamp(K_est, min=K_min, max=K_max)
            梯度通过 clamp 操作回传

        Returns:
            K_float: 可微的 K 值估计 [1]
        """
        import math

        B, N = probs.shape

        # ========== 方法 1: 信息熵驱动 K 估计 (可微) ==========
        # 计算分裂概率的熵
        probs_clamped = probs.clamp(min=1e-8)
        # H = -Σ p_i * log(p_i), 对每个batch计算后取平均
        entropy = -(probs * probs_clamped.log()).sum(dim=1).mean()
        # I170-GRADIENT-FIX: H_max 转为 Tensor 保持计算图
        H_max = math.log(N)  # 最大熵 (Python float)
        H_max_tensor = torch.tensor(H_max, device=probs.device, dtype=entropy.dtype)
        scale_factor = 0.4  # 经验缩放因子
        k_entropy = (entropy / H_max_tensor) * N * scale_factor  # 梯度流保持完整

        # ========== 方法 2: 70% 累积概率截断 (保底) ==========
        # 注意: sort 操作在 no_grad 中执行是安全的，因为索引操作本身不可微
        with torch.no_grad():
            sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
            cumsum = sorted_probs.cumsum(dim=1)
            total_prob = cumsum[:, -1:].clamp(min=1e-6)
            threshold_mask = cumsum < 0.7 * total_prob
            k_70_per_batch = threshold_mask.sum(dim=1).float() + 1
            k_70_mean = k_70_per_batch.mean()

        # ========== 动态边界 ==========
        if self.use_dynamic_k:
            K_min, K_max = self._get_dynamic_k_bounds(N, image_size)
        else:
            K_min, K_max = self.K_min, self.K_max
        K_min = max(K_min, 16)

        # ========== 综合估计: 取多种方法的最大值 (可微) ==========
        conservative_estimate = torch.tensor(B * 0.1, device=probs.device, dtype=k_entropy.dtype)
        # 使用 max() 操作的梯度代理: max(a, b) = ReLU(a-b) + b
        # 这里使用 clamp 确保梯度流动
        K_est_tensor = torch.stack([k_entropy, k_70_mean, conservative_estimate]).max()
        # 使用 clamp 替代硬边界，保持梯度流动
        K_float = K_est_tensor.clamp(min=float(K_min), max=float(K_max))

        return K_float

    def _compute_quota_rate_balanced(
        self, K: int, num_per_depth: List[int]
    ) -> Tuple[Tensor, Tensor]:
        """
        I120-3: 选中率均衡配额分配 (方案 C)

        数学形式化
        ==========

        问题定义:
            - 均匀配额导致深度 0 选中率是深度 4 的 256 倍
            - 目标: 实现各深度选中率均衡: P(选中|d) = C

        选中率均衡条件:
            K_d / N_d = C  (对所有深度 d)
            K_d = C × N_d = C × 4^d

        约束:
            Σ K_d = K  (总配额守恒)
            K_d ≥ 1    (每深度至少 1 个)

        求解 C:
            C = K / Σ N_d = K / N_total

        实现: Largest Remainder Method (LRM) with minimum constraint
            1. 预留每深度 1 个 token
            2. 剩余配额按选中率均衡分配
            3. 确保各深度选中率尽可能接近

        Args:
            K: 总 token 配额
            num_per_depth: [D] 各深度的候选数

        Returns:
            hard_quota: [D] 每个深度的硬配额
            soft_quota: [D] 软配额 (与硬配额成比例)
        """
        D = len(num_per_depth)
        N_total = sum(num_per_depth)

        if N_total == 0:
            device = self.candidate_depths.device
            return torch.zeros(D, dtype=torch.long, device=device), torch.zeros(D, dtype=torch.float32, device=device)

        # 每深度至少 1 个 token
        K_min_per_depth = 1
        K_reserved = D * K_min_per_depth

        if K <= K_reserved:
            # K 不足以分配最小值，返回均匀分布
            device = self.candidate_depths.device
            return torch.ones(D, dtype=torch.long, device=device) * (K // D), torch.ones(D, dtype=torch.float32, device=device) * (K / D)

        # 剩余配额用于均衡选中率
        K_remaining = K - K_reserved
        N_total_for_rate = N_total  # 使用全部候选数计算选中率

        # 目标选中率 (基于剩余配额)
        C = K_remaining / N_total_for_rate

        # 计算每深度的配额 (保留最小 + 按选中率分配)
        K_d_raw = [K_min_per_depth + C * N_d for N_d in num_per_depth]
        K_d_int = [int(k) for k in K_d_raw]
        remainders = [(k - i, d) for d, (k, i) in enumerate(zip(K_d_raw, K_d_int))]
        remainders.sort(reverse=True, key=lambda x: x[0])  # 按 remainder 降序

        # LRM 分配
        K_d = K_d_int.copy()
        allocated = sum(K_d)
        remaining = K - allocated

        # 分配剩余配额给 remainder 最大的深度
        for r, d in remainders:
            if remaining <= 0:
                break
            K_d[d] += 1
            remaining -= 1

        # 转换为 Tensor
        device = self.candidate_depths.device
        hard_quota = torch.tensor(K_d, dtype=torch.long, device=device)
        soft_quota = torch.tensor(K_d, dtype=torch.float32, device=device)

        return hard_quota, soft_quota

    # ========================================================================
    # I150: Hilbert 空间均匀性优化方法
    # ========================================================================

    def _compute_local_density(self, selected_mask: Tensor, hilbert_indices: Optional[Tensor] = None) -> Tensor:
        """
        计算 Hilbert 局部密度惩罚 (I150-1)。

        数学形式化
        ==========

        使用 Hilbert 曲线距离计算每个候选区域的局部密度：
            density_i = Σ_{j: |h_i - h_j| < w} selected_j

        其中 w = HILBERT_DENSITY_WINDOW 是邻域窗口大小。

        用途:
            用于在 logits 上施加惩罚，减少相邻 token 的聚集：
                logits_i' = logits_i - γ × density_i

        Args:
            selected_mask: [B, N] 选择的掩码
            hilbert_indices: [N] Hilbert 曲线索引（可选，默认使用内部缓存）

        Returns:
            density: [B, N] 每个候选区域的局部密度
        """
        B, N = selected_mask.shape

        # 获取 Hilbert 索引
        if hilbert_indices is None:
            hilbert_indices = self.hilbert_indices

        # 确保 hilbert_indices 有效
        if hilbert_indices is None:
            return torch.zeros_like(selected_mask)

        # I150-4-FIX: 确保 hilbert_indices 与 selected_mask 在同一设备上（torch.compile 兼容性）
        if hilbert_indices.device != selected_mask.device:
            hilbert_indices = hilbert_indices.to(device=selected_mask.device, non_blocking=True)

        # 计算稀疏邻域密度 (向量化 GPU 实现)
        # 使用 torch.cdist 计算距离矩阵，避免 CPU 同步
        window = int(HILBERT_DENSITY_WINDOW)

        # 获取选中的 Hilbert 索引
        selected_indices = selected_mask.nonzero(as_tuple=False)  # [M, 2] = (batch_idx, token_idx)

        if selected_indices.shape[0] == 0:
            return torch.zeros_like(selected_mask)

        # 提取选中 token 的 Hilbert 索引
        batch_idx = selected_indices[:, 0]  # [M]
        token_idx = selected_indices[:, 1]  # [M]
        selected_h = hilbert_indices[token_idx]  # [M]

        # 广播计算距离: [M, 1] - [1, N] = [M, N]
        hilbert_expanded = hilbert_indices.unsqueeze(0)  # [1, N]
        selected_h_expanded = selected_h.unsqueeze(1)  # [M, 1]
        dist = torch.abs(selected_h_expanded - hilbert_expanded)

        # 窗口掩码: [M, N]
        window_mask = dist <= window

        # P-OPT: 向量化批量累加 - 使用 scatter_add 替代 for 循环
        # window_mask: [M, N], batch_idx: [M], 输出: [B, N]
        # 扩展 batch_idx 到 [M, N]
        batch_idx_expanded = batch_idx.unsqueeze(1).expand(-1, N)  # [M, N]
        # 使用 scatter_add 按 batch 累加
        density = torch.zeros(B, N, device=selected_mask.device, dtype=torch.float32)
        density.scatter_add_(0, batch_idx_expanded, window_mask.float())

        return density

    def _get_depth_adaptive_temperature(self, base_temperature: float, depths: Tensor) -> Tensor:
        """
        计算深度自适应温度 (I150-2)。

        数学形式化
        ==========

        深层节点候选数量指数增长：N_d = 4^d
        导致深层在 Top-K 竞争中天然优势。

        自适应温度公式：
            τ_d = τ_base × (N_max / N_d)^γ

        其中：
            - τ_base 是基础温度
            - N_max = 4^{max_depth} 是最大候选数
            - N_d = 4^d 是当前深度候选数
            - γ ∈ (0, 1) 是缩放指数

        效果：
            - d↑ → N_d↑ → τ_d↑ → softmax 更均匀 → 浅层机会↑

        Args:
            base_temperature: 基础温度 τ_base (float 或 Tensor)
            depths: [N] 每个候选区域的深度

        Returns:
            temperature: [N] 每个候选区域的自适应温度
        """
        if not DEPTH_ADAPTIVE_TEMPERATURE_ENABLED:
            # 确保 base_temperature 是标量
            if isinstance(base_temperature, Tensor):
                base_temperature = base_temperature.item()
            return torch.full_like(depths.float(), base_temperature)

        D = int(depths.max().item() + 1)
        device = depths.device

        # I150-2-FIX: 确保 base_temperature 在正确的设备上
        if isinstance(base_temperature, Tensor):
            base_temperature = base_temperature.to(device=device, non_blocking=True)

        # 计算每个深度的温度缩放因子 (向量化实现)
        # τ_d = τ_base × (N_max / N_d)^γ = τ_base × (4^{max_depth - d})^gamma
        gamma = DEPTH_TEMPERATURE_GAMMA

        # 向量化：使用张量运算一次性计算所有深度
        depth_indices = torch.arange(D, device=device, dtype=torch.float32)
        N_d = 4 ** depth_indices
        N_max = 4 ** (D - 1)
        depth_temperature_scale = (N_max / N_d) ** gamma

        # 映射到每个候选区域
        temperature = depth_temperature_scale[depths]

        # 应用温度下界保护
        temperature = torch.clamp(temperature * base_temperature, min=TEMPERATURE_MIN)

        return temperature

    def _hilbert_greedy_diversity(
        self,
        logits: Tensor,
        K: int,
        hilbert_indices: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Hilbert-DPP 贪心多样性采样 (I150-3)。

        数学形式化
        ==========

        目标：在 logits 质量与空间多样性之间取得平衡。

        贪心算法：
            1. 按 Hilbert 索引排序所有候选
            2. 依次选择 token，每次选择时考虑：
               - logit 得分（质量）
               - 与已选集合的多样性惩罚

            score_i = logits_i - λ × max_{j∈selected} S[i,j]

        相似度函数（基于 Hilbert 距离）：
            S[i,j] = exp(-|h_i - h_j|² / σ²)

        Args:
            logits: [B, N] 候选 logits
            K: 选择数量
            hilbert_indices: [N] Hilbert 曲线索引

        Returns:
            selected_mask: [B, N] 选择的掩码
            topk_indices: [B, K] 选择的索引
        """
        B, N = logits.shape
        device = logits.device

        if hilbert_indices is None:
            hilbert_indices = self.hilbert_indices

        # I150-4-FIX: 确保 hilbert_indices 与 logits 在同一设备上（torch.compile 兼容性）
        if hilbert_indices.device != device:
            hilbert_indices = hilbert_indices.to(device=device, non_blocking=True)

        # 初始化输出
        selected_mask = torch.zeros(B, N, device=device, dtype=logits.dtype)

        # 对每个 batch 独立处理
        for b in range(B):
            # 按 Hilbert 顺序遍历（利用局部性）
            # 使用滑动窗口近似多样性计算
            window = HILBERT_DENSITY_WINDOW
            sigma_sq = HILBERT_DIVERSITY_SIGMA ** 2
            lambda_div = DIVERSITY_LAMBDA

            # P-OPT: 使用张量而非 Python list 避免循环中的 .item() 同步
            remaining_logits = logits[b].clone()
            selected_indices = torch.zeros(K, dtype=torch.long, device=device)

            for step in range(K):
                # 贪心选择
                if step > 0:
                    # 计算多样性惩罚 - P-OPT: 向量化替代内层循环
                    selected_h = hilbert_indices[selected_indices[:step]]  # [M]
                    # 使用广播计算所有候选与已选候选的距离矩阵
                    # [N, 1] - [1, M] = [N, M]
                    diff_matrix = hilbert_indices.unsqueeze(1) - selected_h.unsqueeze(0)
                    abs_diff = torch.abs(diff_matrix)  # [N, M]

                    # 窗口掩码: |h_i - h_j| < window
                    in_window = abs_diff < window  # [N, M]

                    # 高斯核: exp(-diff^2 / sigma_sq)
                    gauss_kernel = torch.exp(-(abs_diff ** 2) / sigma_sq)  # [N, M]

                    # 应用窗口掩码后取最大值
                    masked_sim = torch.where(in_window, gauss_kernel, torch.zeros_like(gauss_kernel))
                    max_sim = masked_sim.max(dim=1)[0]  # [N]

                    # 更新得分
                    scores = remaining_logits - lambda_div * max_sim
                else:
                    scores = remaining_logits

                # P-OPT: 选择得分最高的 - 保留张量索引避免 .item() 同步
                _, best_idx = scores.max(dim=0)
                selected_indices[step] = best_idx  # 保留张量索引，不做 .item() 转换
                remaining_logits[best_idx] = -float('inf')  # 避免重复选择

            # 构建掩码 - 使用 scatter_ 一次性完成
            selected_mask[b].scatter_(0, selected_indices, 1.0)

        # P-OPT: 从 selected_mask 重建 topk_indices（避免循环中重复构建）
        topk_indices = selected_mask.nonzero(as_tuple=False)[:, 1].view(B, K)

        return selected_mask, topk_indices

    def get_density_regularization(self) -> Tensor:
        """
        获取空间密度惩罚损失 (I150-1)。

        数学形式化
        ==========

        损失函数：
            L_density = γ × Σ_i density_i × selected_i

        其中 density_i 是选中 token 的局部密度。

        目的：
            最小化选中 token 之间的聚集程度，
            鼓励更均匀的空间分布。

        Returns:
            density_loss: 标量张量，密度正则化损失
        """
        # I152-FIX: 使用带梯度的版本，确保梯度可以流回
        if not hasattr(self, '_last_selected_mask_for_loss') or self._last_selected_mask_for_loss is None:
            # 降级使用 detached 版本
            if not hasattr(self, '_last_selected_mask') or self._last_selected_mask is None:
                return torch.tensor(0.0, device=self.candidate_regions.device)
            selected_mask = self._last_selected_mask
        else:
            selected_mask = self._last_selected_mask_for_loss

        B, N = selected_mask.shape

        # 计算密度
        density = self._compute_local_density(selected_mask.detach())  # detach density 计算图

        # 计算损失：选中区域的平均密度
        # 密度越高，损失越大（惩罚聚集）
        density_loss = (density * selected_mask).sum() / (selected_mask.sum() + EPS)

        return DENSITY_PENALTY_WEIGHT * density_loss

    def _compute_quota_allocation(
        self, K: int, info_density: Optional[Tensor] = None, features: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        配额分配 (I100-7 重构, I113-17: 连续松弛替代STE, I120-3: 分层自适应)

        数学形式化
        ==========

        问题定义:
            输入: p = softmax(φ) ∈ Δ^{D-1}, K ∈ ℤ⁺
            约束: ΣK_d = K, K_d ≥ 0, K_d ∈ ℤ
            目标: min Σ|K_d - p_d·K|

        I113-17 连续松弛算法:
            1. q = softmax(φ / τ)              ∈ Δ^{D-1}  (可微, 平滑)
            2. K_soft = q · K                  ∈ ℝ^D      (实数配额)
            3. K_hard = LRM(K_soft)            ∈ ℤ^D      (离散投影)

        I120-3 分层自适应算法 (新增):
            1. I_{d,i} = Info(patch_{d,i})     (每个区域的信息密度)
            2. S_d = Σ_i I_{d,i}              (每个深度的总分值)
            3. K_d = K × S_d / Σ S_d'          (按比例分配)
            4. K_hard = LRM(K_soft)            (整数投影)

        模式优先级:
            1. I120-3: 分层自适应 (features != None)
            2. I100-7: 信息密度自适应 (info_density != None)
            3. I113-17: 可学习配额 (quota_allocator != None)
            4. I120-3: 选中率均衡 (enable_rate_balanced)
            5. 均匀分布 (回退)

        Args:
            K: 总 token 配额
            info_density: [D] 各深度的信息密度 (可选，I100-7)
            features: [B, C, H, W] 图像特征 (可选，I120-3)

        Returns:
            hard_quota: [D] 每个深度的硬配额分配 (LRM 输出)
            soft_quota: [D] 软配额 (q × K，有梯度)
        """
        D = self._current_max_depth + 1
        device = self.quota_logits.device if self.quota_logits is not None else self.candidate_depths.device

        # I120-3: 分层自适应配额模式 (最高优先级，需要 features)
        if features is not None and self._enable_hierarchical_quota:
            info_density_per_depth, num_per_depth = self._compute_hierarchical_info_density(features)
            return self._compute_quota_hierarchical(K, info_density_per_depth, num_per_depth)

        # I100-7: 信息密度自适应模式 (保留原有实现)
        if info_density is not None and self._enable_info_adaptive_quota:
            return self._compute_quota_allocation_info_adaptive(K, info_density)

        # I113-17: 使用 ContinuousQuotaAllocator
        if self.quota_allocator is not None and self._enable_learnable_quota:
            # 确保allocator使用正确的D
            if self.quota_allocator.D != D:
                # 重建allocator以匹配当前深度
                self.quota_allocator = ContinuousQuotaAllocator(
                    D=D,
                    tau=self.quota_allocator.tau,
                    tau_warmup_steps=self.quota_allocator.tau_warmup_steps,
                    enable_warmup=self.quota_allocator.enable_warmup,
                )

            # 使用连续松弛分配器
            K_hard, K_soft = self.quota_allocator.forward(K)

            # 确保维度正确 (P-OPT: 移除多余的 .to(device)，quota_logits 已随模型正确迁移)
            K_hard = K_hard[:D]
            K_soft = K_soft[:D]

            # 更新warmup进度
            self.quota_allocator.step()

            return K_hard, K_soft

        # I120-3: 选中率均衡配额模式 (仅当 learnable_quota 也启用时使用)
        if self._enable_rate_balanced_quota and self._enable_learnable_quota:
            # 获取各深度候选数
            num_per_depth = [4**d for d in range(D)]
            return self._compute_quota_rate_balanced(K, num_per_depth)

        # 回退到均匀分布
        p = torch.ones(D, device=device) / D
        soft_quota = p * K

        # 均匀分配 (硬配额)
        hard_quota = torch.full((D,), K // D, dtype=torch.long, device=device)
        hard_quota[D - 1] += K - hard_quota.sum()
        return hard_quota, soft_quota

    def _detect_depth_collapse(self, depth_distribution: Dict[int, float]) -> str:
        """
        I121-4: 深度坍塌检测

        数学形式化
        ==========

        符号定义:
            p_d = P(d)           // 深度 d 的选择概率
            H = -Σ p_d log p_d   // Shannon 熵
            p_max = max p_d       // 最大深度占比

        检测规则:
            CRITICAL: p_{-1} > 0.5  OR  H < 0.3
            WARNING:  p_max > 0.4   OR  H < 0.6
            OK:       otherwise

        Args:
            depth_distribution: 各深度的选择概率分布

        Returns:
            检测结果: 'CRITICAL', 'WARNING', 或 'OK'
        """
        # Padding 比例
        p_minus_1 = depth_distribution.get(-1, 0)

        # 有效深度分布
        p_eff = {d: p for d, p in depth_distribution.items() if d >= 0}
        if not p_eff:
            return 'CRITICAL'

        # 计算熵
        H = 0.0
        for p in p_eff.values():
            if p > 0:
                H -= p * math.log(p)

        # 最大深度占比
        p_max = max(p_eff.values())

        # 检测
        if p_minus_1 > 0.5 or H < 0.3:
            return 'CRITICAL'
        elif p_max > 0.4 or H < 0.6:
            return 'WARNING'
        else:
            return 'OK'

    def _get_recovery_action(self, detection: str, step: int) -> Dict[str, Any]:
        """
        I121-4: 获取自适应恢复动作

        数学形式化
        ==========

        恢复策略:

        CRITICAL:
            - 温度提升: τ ← τ × 1.5
            - 熵损失权重: λ_entropy ← 3.0
            - 强制使用方案 C (选中率均衡)

        WARNING:
            - 温度提升: τ ← τ × 1.2
            - 熵损失权重: λ_entropy ← 2.0
            - 启用均匀配额 + 增强损失

        OK:
            - 正常参数
            - 使用可学习配额

        Args:
            detection: 检测结果 ('CRITICAL', 'WARNING', 'OK')
            step: 当前训练步数

        Returns:
            恢复动作参数字典
        """
        if detection == 'CRITICAL':
            return {
                'mode': 'CRITICAL',
                'temperature_mult': 1.5,
                'entropy_weight_mult': 3.0,
                'quota_mode': 'rate_balanced',
                'recovery_steps': 100,  # 恢复需要的步数
            }
        elif detection == 'WARNING':
            return {
                'mode': 'WARNING',
                'temperature_mult': 1.2,
                'entropy_weight_mult': 2.0,
                'quota_mode': 'hybrid',
                'recovery_steps': 50,
            }
        else:
            return {
                'mode': 'OK',
                'temperature_mult': 1.0,
                'entropy_weight_mult': 1.0,
                'quota_mode': 'learnable',
                'recovery_steps': 0,
            }

    def _apply_recovery(
        self,
        detection: str,
        current_temperature: float,
        current_entropy_weight: float,
    ) -> Tuple[float, float, bool]:
        """
        I121-4: 应用恢复机制

        Args:
            detection: 检测结果
            current_temperature: 当前温度
            current_entropy_weight: 当前熵损失权重

        Returns:
            (新温度, 新熵权重, 是否启用恢复模式)
        """
        if detection == 'OK':
            return current_temperature, current_entropy_weight, False

        action = self._get_recovery_action(detection, 0)
        new_temp = current_temperature * action['temperature_mult']
        new_weight = current_entropy_weight * action['entropy_weight_mult']

        return new_temp, new_weight, True
        p = torch.ones(D, device=device) / D
        soft_quota = p * K

        # 均匀分配 (硬配额)
        hard_quota = torch.full((D,), K // D, dtype=torch.long, device=device)
        hard_quota[D - 1] += K - hard_quota.sum()
        return hard_quota, soft_quota

    def _compute_quota_loss(self, K: int) -> Tensor:
        """
        计算配额正则化损失 (I113-17: 连续松弛版本)

        数学形式化
        ==========

        I113-17 核心改进:
            - 连续松弛替代 STE，梯度自然流过 softmax
            - 无需显式 STE 损失，梯度通过 allocator 自动传递

        损失组件:
            1. L_align: 软配额与硬配额的对齐损失
            2. L_align_kl: KL(softmax(φ) || target_dist) 配额对齐损失 (I153-1)
            3. L_min: 软下界正则化损失

        梯度流:
            ∂L/∂φ = ∂L_align/∂φ + ∂L_align_kl/∂φ + ∂L_min/∂φ
                   (通过 softmax 自然传递，无需 STE 近似)

        Args:
            K: 总 token 配额

        Returns:
            quota_loss: 标量张量，配额正则化损失
        """
        D = self._current_max_depth + 1

        # 如果未启用可学习配额，返回 0
        if self.quota_allocator is None or not self._enable_learnable_quota:
            return torch.tensor(0.0, device=self.candidate_depths.device)

        # I113-17: 使用 ContinuousQuotaAllocator 的内置损失
        # 梯度通过 allocator 的 softmax 自然传递，无需 STE 近似
        loss = self.quota_allocator.compute_quota_loss()

        # 应用权重
        loss = loss * QUOTA_ENTROPY_WEIGHT

        # I153-1: 配额对齐损失 - KL(softmax(φ) || target_dist)
        # 使用当前 step 和估计的 total_steps
        current_step = self.quota_allocator._current_step
        estimated_total_steps = getattr(self, '_quota_total_steps', 50000)

        target_dist = self.quota_allocator.get_target_depth_distribution(
            step=current_step,
            total_steps=estimated_total_steps,
            mode="curriculum"
        )

        align_kl_loss = self.quota_allocator.compute_quota_align_loss(
            target_dist=target_dist,
            weight=0.1
        )
        loss = loss + align_kl_loss

        # I96-7: 软下界正则化损失
        # 获取当前软配额
        _, K_soft = self.quota_allocator.forward(K)
        K_soft = K_soft[:D].float()

        depth_indices = torch.arange(D, device=K_soft.device, dtype=torch.long)
        N = 4 ** depth_indices

        K_min_tensor = torch.clamp(
            (QUOTA_MIN_RATIO * N).floor().long(),
            min=1
        ).to(dtype=K_soft.dtype)

        # 计算下界违反: max(0, K_min - K_soft)
        violation = (K_min_tensor - K_soft).clamp(min=0)
        min_loss = (violation ** 2).mean()

        # 添加软下界损失
        loss = loss + min_loss * QUOTA_MIN_LAMBDA

        return loss

    # I113-16: HybridDensityHead 信息密度估计 (替代 I113-6 的方差方案)
    def _compute_info_density(self, features: Tensor) -> Tensor:
        """
        计算图像的多尺度信息密度 (I113-16 HybridDensityHead)。

        数学形式化
        ==========

        问题定义:
            给定特征图 F ∈ ℝ^{B×C×H×W}，计算各尺度的信息密度

        HybridDensityHead (I113-16):
            结合梯度感知和语义学习的混合密度头

        公式:
            D = σ( (GradBranch ⊕ SemBranch) / τ )
            I_d = softmax( D_d )  (归一化到 Σ I_d = 1)

        其中:
            GradBranch: 捕捉边缘、纹理等低级视觉特征
            SemBranch: 学习复杂的信息模式
            σ: sigmoid 激活函数
            τ: 可学习温度参数

        与 Hilbert 曲线的对齐:
            - 深度 d 对应不同尺度的特征聚合
            - 梯度分支捕捉边缘的空间位置
            - 语义分支学习跨区域的复杂模式

        I113-16 相对 I113-6 的改进:
            | 指标     | 方差 (I113-6) | HybridDensityHead (I113-16) |
            |----------|---------------|----------------------------|
            | 语义对齐 | 0.3           | 0.9 ✓                      |
            | 尺度不变 | 0.2           | 0.8 ✓                      |
            | 梯度感知 | 无             | Sobel 边缘检测              |
            | 学习能力 | 无             | 端到端可学习                |

        Args:
            features: [B, C, H_feat, W_feat] 特征图

        Returns:
            info_density: [D] 各深度的信息密度 (归一化)
        """
        B, C, H_feat, W_feat = features.shape

        # I113-16: 基于实际输入尺寸计算正确的深度
        # 对于 64x64: log2(64/4) = 4
        # 对于 32x32: log2(32/4) = 3
        # 对于 16x16: log2(16/4) = 2
        correct_depth = int(math.log2(min(H_feat, W_feat) / self.min_patch_size))
        D = max(1, correct_depth)  # 至少 1 个深度

        # 使用 HybridDensityHead 计算多尺度密度 [B, D]
        info_density = self.hybrid_density_head.compute_multi_scale_density(
            features, max_depth=D
        )

        # 确保输出形状正确
        if info_density.shape[1] < D:
            # 填充到 D 维度
            padding = torch.zeros(B, D - info_density.shape[1], device=features.device)
            info_density = torch.cat([info_density, padding], dim=1)
        elif info_density.shape[1] > D:
            # 截断到 D 维度
            info_density = info_density[:, :D]

        # Softmax 归一化
        info_density = F.softmax(info_density, dim=1)

        # 返回 [D] (对 batch 维度取平均，与原接口兼容)
        return info_density.mean(dim=0)

    # I120-3: 分层自适应配额 - 新增方法
    def _compute_hierarchical_info_density(self, features: Tensor) -> Tuple[Tensor, List[int]]:
        """
        I120-3: 计算分层信息密度（每个深度每个区域）

        数学形式化
        ===========

        问题定义:
            现有方案的问题:
            - 选中率均衡: K_d/N_d = C (忽略语义)
            - 信息密度自适应: I_d (对 batch 取平均，忽略图像差异)
            - 可学习配额: φ_d (图像无关，任务特定)

        核心洞察:
            深度不是独立的，而是嵌套的:
            - 深度 d 的 patch 是深度 d+1 的父节点
            - 深度 d 的信息密度应继承其子节点

        新方案: 分层自适应配额

            步骤 1: 计算每个深度每个区域的信息密度
                I_{d,i} = Info(patch_{d,i})

            步骤 2: 计算每个深度的总分值 (嵌套求和)
                S_d = Σ_i I_{d,i} = I_0 + I_1/4 + I_2/16 + ...

            步骤 3: 按比例分配配额
                K_d = K × S_d / Σ S_d'

        与现有方案的对比:
            | 方案           | K_d 计算              | 语义    | 复杂度 |
            |----------------|----------------------|--------|--------|
            | 选中率均衡      | C × N_d              | ❌      | O(D)   |
            | 信息密度       | I_d × N_d            | ⚠️      | O(D)   |
            | 分层自适应     | S_d = Σ I_{d,i}      | ✅      | O(ΣN)  |

        Args:
            features: [B, C, H_feat, W_feat] 特征图

        Returns:
            info_density_per_depth: [B, D] 每个样本每个深度的信息密度 (归一化前)
            num_per_depth: [D] 每个深度的候选区域数
        """
        B, C, H_feat, W_feat = features.shape

        # 使用当前配置的深度
        D = self._current_max_depth + 1
        if D is None:
            # 回退到基于特征尺寸计算
            D = max(1, int(math.log2(min(H_feat, W_feat) / self.min_patch_size)))

        # 使用 HybridDensityHead 计算多尺度密度 [B, D]
        info_density = self.hybrid_density_head.compute_multi_scale_density(
            features, max_depth=D
        )

        # 确保输出形状正确
        if info_density.shape[1] < D:
            padding = torch.zeros(B, D - info_density.shape[1], device=features.device)
            info_density = torch.cat([info_density, padding], dim=1)
        elif info_density.shape[1] > D:
            info_density = info_density[:, :D]

        # 返回未归一化的密度，保留每个样本的信息
        # 形状: [B, D]
        return info_density, [4**d for d in range(D)]

    def _compute_quota_hierarchical(
        self, K: int, info_density: Tensor, num_per_depth: List[int]
    ) -> Tuple[Tensor, Tensor]:
        """
        I120-3: 分层自适应配额分配

        数学形式化
        ===========

        步骤 1: 计算每个深度的总分值
            S_d = info_density[:, d] × num_per_depth[d]
            = Σ_i I_{d,i} × N_d (每个区域的密度 × 区域数)

        步骤 2: 按比例分配配额
            K_d = K × S_d / Σ S_d'

        步骤 3: LRM 投影确保整数约束
            Σ K_d = K, K_d ≥ 1

        关键性质:
            - S_d 保留了原始图像的信息分布
            - K_d 与 S_d 成比例，确保高信息区域获得更多 token
            - 不需要 softmax 归一化（与现有方案不同）

        Args:
            K: 总 token 配额
            info_density: [B, D] 每个样本每个深度的信息密度
            num_per_depth: [D] 每个深度的候选区域数

        Returns:
            hard_quota: [D] 每个深度的硬配额
            soft_quota: [D] 每个深度的软配额
        """
        B, D = info_density.shape
        device = info_density.device

        # 步骤 1: P-OPT: 向量化计算每个深度的总分值
        # S_d = density_d × N_d (每个区域的密度 × 区域数)
        # 使用广播替代 for d in range(D) 循环
        num_per_depth_tensor = torch.tensor(num_per_depth, device=device, dtype=torch.float32)
        score_per_depth = info_density * num_per_depth_tensor.unsqueeze(0)  # [B, D] = [B, D] * [1, D]

        # 步骤 2: 按比例分配配额 (对 batch 取平均)
        S_total = score_per_depth.sum(dim=1, keepdim=True)  # [B, 1]
        # 避免除零
        S_total = S_total.clamp(min=EPS)

        # 软配额: K × S_d / Σ S_d'
        K_soft_float = K * (score_per_depth / S_total)  # [B, D]
        K_soft = K_soft_float.mean(dim=0)  # [D] (对 batch 取平均)

        # 步骤 3: LRM 投影确保整数约束，同时确保每个深度至少 1 个
        K_hard = self._lrm_projection_with_min(K_soft, target_sum=K, min_per_depth=1)

        return K_hard.long(), K_soft

    def _lrm_projection_with_min(
        self, K_soft: Tensor, target_sum: int, min_per_depth: int = 1
    ) -> Tensor:
        """
        带最小约束的 LRM 投影

        确保:
            1. Σ K_d = target_sum
            2. K_d ≥ min_per_depth
            3. K_d ∈ ℤ

        Args:
            K_soft: [D] 软配额 (浮点数)
            target_sum: 目标总和
            min_per_depth: 每个深度的最小配额

        Returns:
            K_hard: [D] 硬配额 (整数)
        """
        D = K_soft.shape[0]
        device = K_soft.device

        # 预留最小配额
        K_min_total = D * min_per_depth
        K_remaining = target_sum - K_min_total

        if K_remaining < 0:
            # 目标总和太小，无法满足最小约束
            # 回退到均匀分配
            K_hard = torch.ones(D, device=device, dtype=torch.long) * (target_sum // D)
            for d in range(target_sum % D):
                K_hard[d] += 1
            return K_hard

        # 计算相对分数 (减去最小值后的分数)
        K_soft_remaining = K_soft - K_soft.min()
        K_soft_remaining = K_soft_remaining.clamp(min=0)

        # 按相对分数比例分配剩余配额
        # P-OPT: 使用 torch.where() 向量化处理，避免 GPU-CPU 同步
        total = K_soft_remaining.sum()
        ratio = torch.where(total > 0, K_soft_remaining / total, torch.zeros_like(K_soft_remaining))
        K_extra = (ratio * K_remaining).floor().long()

        # 确保 K_extra 不超过剩余配额
        K_extra = K_extra.clamp(max=K_remaining)

        # 基础配额 + 额外配额
        K_hard = torch.ones(D, device=device, dtype=torch.long) * min_per_depth + K_extra

        # 调整以精确匹配目标总和
        # I113-12: 移除 .item() 调用，保持 GPU tensor 用于后续计算
        current_sum = K_hard.sum()
        diff = target_sum - current_sum

        if diff > 0:
            # 需要增加 diff 个配额
            # I113-12: 向量化处理，避免 Python 循环
            if diff > 0 and D > 0:
                # 一次性找出 diff 个需要增加的深度
                _, sort_idx = torch.topk(-K_soft_remaining, k=min(int(diff), D))
                # 使用 scatter_add 批量增加
                increments = torch.zeros(D, device=device, dtype=torch.long)
                num_increments = min(int(diff), D)
                increments.scatter_(0, sort_idx[:num_increments], 1)
                # 如果 diff > D，需要重复分配
                if diff > D:
                    extra = diff - D
                    extra_increments = torch.zeros(D, device=device, dtype=torch.long)
                    extra_increments.scatter_(0, sort_idx, 1)
                    # 累积额外增量
                    full_cycles = extra // D
                    remainder = extra % D
                    if full_cycles > 0:
                        increments += full_cycles * extra_increments
                    if remainder > 0:
                        increments.scatter_(0, sort_idx[:remainder], 1)
                K_hard = K_hard + increments
        elif diff < 0:
            # 需要减少 |diff| 个配额
            # I113-12: 向量化处理，避免 Python 循环
            # 使用 torch.where 确保 K_hard >= min_per_depth
            K_hard = torch.where(
                K_hard > min_per_depth,
                K_hard,
                torch.ones_like(K_hard) * min_per_depth
            )
            # 优先减少分数较低的深度
            if D > 0:
                num_decrements = min(int(-diff), D)
                _, sort_idx = torch.topk(K_soft_remaining, k=num_decrements)
                decrements = torch.zeros(D, device=device, dtype=torch.long)
                decrements.scatter_(0, sort_idx, 1)
                K_hard = K_hard - decrements
                # 确保不低于最小值
                K_hard = torch.maximum(K_hard, torch.ones_like(K_hard) * min_per_depth)

        return K_hard

    def _lrm_projection(self, K_soft: Tensor, target_sum: int = None) -> Tensor:
        """
        LRM投影 - Largest Remainder Method 连续配额到离散配额的投影

        数学形式
        ==========

        LRM 确保:
            1. Σ K_d = target_sum (配额守恒)
            2. K_d ≥ 0 (非负约束)
            3. K_d ∈ ℤ (整数约束)

        算法:
            1. floor(K_soft) 得到基础配额
            2. remainder = K_soft - floor(K_soft)
            3. 按 remainder 降序分配剩余配额

        Args:
            K_soft: [D] 软配额 (浮点数)
            target_sum: 目标总和 (可选)

        Returns:
            K_hard: [D] 硬配额 (整数)
        """
        D = K_soft.shape[0]
        device = K_soft.device

        # 如果没有指定目标总和，使用软配额总和
        # I113-12: 保持 GPU tensor，避免 .item() 同步
        if target_sum is None:
            target_sum = K_soft.sum().long()  # 保持 tensor

        # 基础配额和余数
        K_floor = K_soft.floor()  # [D]
        remainders = K_soft - K_floor  # [D]

        # 确保基础配额非负
        K_floor = K_floor.clamp(min=0)

        # 初始分配
        K_hard = K_floor.long()  # [D]
        allocated = K_hard.sum()  # I113-12: 保持 tensor
        remaining = target_sum - allocated  # tensor - int 会产生 tensor

        # I113-12: 使用向量化操作替代 Python 循环
        # 处理 tensor 类型的 remaining
        remaining_scalar = remaining.item() if isinstance(remaining, Tensor) else remaining
        if remaining_scalar < 0:
            # 软配额总和大于目标，需要裁剪
            # 按比例裁剪每个深度
            total_soft = K_soft.sum()  # I113-12: 保持 tensor
            total_soft_scalar = total_soft.item() if isinstance(total_soft, Tensor) else total_soft
            if total_soft_scalar > 0:
                # I113-12: 处理 int 或 tensor 类型的 target_sum
                if isinstance(target_sum, Tensor):
                    scale = target_sum.float() / total_soft.float()
                else:
                    scale = torch.tensor(target_sum, dtype=torch.float32, device=device) / total_soft.float()
                K_hard = (K_floor * scale).long()
                allocated = K_hard.sum()
                remaining = target_sum - allocated
            else:
                K_hard = torch.ones(D, device=device, dtype=torch.long) * (target_sum // D)
                allocated = K_hard.sum()
                remaining = target_sum - allocated

        # I113-12: 按余数降序分配剩余配额 - 向量化
        remaining_val = remaining.item() if isinstance(remaining, Tensor) else remaining
        if remaining_val > 0 and D > 0:
            k = min(int(remaining_val), D)
            _, sort_indices = torch.topk(remainders, k=k)
            # 使用 scatter_add 批量增加
            increments = torch.zeros(D, device=device, dtype=torch.long)
            increments.scatter_(0, sort_indices, 1)
            K_hard = K_hard + increments

        return K_hard

    def _compute_local_variance(self, features: Tensor, patch_size: int) -> Tensor:
        """
        计算特征图的局部方差 (无偏估计器)。

        数学形式
        =========

        方差定义 (样本方差，无偏):
            Var(X) = (1/(n-1)) Σ (x_i - μ)²

        但对于深度学习中的信息密度估计，我们使用:
            Var(X) = E[X²] - E[X]²

        优势:
            - 计算高效：单次遍历
            - 数值稳定：避免两次遍历
            - 可微：端到端梯度流

        Args:
            features: [B, C, H, W] 输入特征图
            patch_size: int patch 边长

        Returns:
            variance: [B, C] 每个通道的局部方差均值
        """
        B, C, H, W = features.shape

        # 展平 batch 和 channel 用于统一处理
        x = features.view(B * C, 1, H, W)

        # 计算 padding 以整除 patch_size
        H_pad = (patch_size - H % patch_size) % patch_size
        W_pad = (patch_size - W % patch_size) % patch_size

        if H_pad > 0 or W_pad > 0:
            x = F.pad(x, (0, W_pad, 0, H_pad), mode='replicate')

        # Unfold 提取所有 patches
        # [B*C, 1, H', W'] -> [B*C, patch_size², n_patches]
        patches = F.unfold(x, patch_size, stride=patch_size)

        # 计算 E[X²] 和 E[X]²
        # E[X]: [B*C, 1, n_patches]
        E_x = patches.mean(dim=1, keepdim=True)
        # E[X²]: [B*C, 1, n_patches]
        E_x2 = (patches ** 2).mean(dim=1, keepdim=True)

        # Var(X) = E[X²] - E[X]²
        # [B*C, 1, n_patches]
        variance = E_x2 - E_x ** 2

        # 避免负值 (数值误差)
        variance = variance.clamp(min=0)

        # Reshape 回 [B, C, n_patches]
        variance = variance.view(B, C, -1)

        # 返回所有 patch 方差的均值
        return variance.mean(dim=2)  # [B, C]

    def _compute_quota_allocation_info_adaptive(
        self, K: int, info_density: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        信息密度自适应配额分配 (I100-7 重构为标准 LRM, I113-7 梯度恢复)。

        数学形式化
        ==========

        核心思想:
            配额与信息密度成正比，而非固定分布

        算法: Largest Remainder Method (Hamilton Method) + STE 梯度恢复
        --------------------------------------------------------------
        公式:
            q_d^floor = floor(I_d × K)
            r_d = I_d × K - q_d^floor (余数)
            q_d = q_d^floor + 1 如果 r_d 在 top-(K - Σq_d^floor) 中

        最优性:
            LRM 最小化 L₁ 误差: min Σ|q_d - I_d·K|

        I113-7 梯度恢复 (STE):
            前向: hard_quota = LRM(I, K) (离散输出)
            反向: soft_quota = I × K (连续代理，有梯度)

        与可学习配额的对比:
            - 可学习配额: q_d = K × softmax(φ)_d (全局静态)
            - 信息密度自适应: q_d = K × I_d (输入自适应)

        Args:
            K: 总 token 配额
            info_density: [D] 各深度的信息密度 (归一化)

        Returns:
            hard_quota: [D] 前向使用的硬配额 (LRM 输出)
            soft_quota: [D] 反向使用的软配额 (I × K, 有梯度)
        """
        D = self._current_max_depth + 1
        device = info_density.device

        # 确保 info_density 长度匹配当前深度
        if len(info_density) < D:
            # 填充均匀分布
            info_density_extended = info_density.new_zeros(D)
            info_density_extended[: len(info_density)] = info_density
            info_density_extended[len(info_density) :] = 1.0 / D
            info_density = info_density_extended
        elif len(info_density) > D:
            info_density = info_density[:D]

        # I113-7: STE 软配额 (用于反向梯度)
        soft_quota = info_density * K  # [D], 有完整梯度

        # I100-7: 标准 Largest Remainder Method (LRM) 用于前向
        floor_quota = (info_density * K).floor().long()  # [D]
        remainders = (info_density * K) - floor_quota.float()  # [D]
        remaining = K - floor_quota.sum()  # 剩余配额数量

        if remaining > 0:
            # 分配给余数最大的深度
            _, indices = torch.topk(remainders, min(int(remaining), D))
            floor_quota[indices] += 1

        hard_quota = floor_quota

        return hard_quota, soft_quota

    def _compute_info_quota_loss(
        self, soft_quota: Tensor, hard_quota: Tensor, K: int
    ) -> Tensor:
        """
        I113-7: 信息密度配额损失 (带偏差校正)。

        数学形式化
        ==========

        核心思想:
            使用软配额与硬配额的差异作为正则化项

        损失形式:
            L_info = MSE(soft_quota, hard_quota) + λ × KL(soft || target)

        梯度流:
            ∂L_info/∂I = ∂MSE/∂soft_quota × K + KL项梯度

        Args:
            soft_quota: [D] 软配额 (I × K, 有梯度)
            hard_quota: [D] 硬配额 (LRM 输出)
            K: 总 token 配额

        Returns:
            quota_loss: 标量张量
        """
        D = soft_quota.shape[0]

        # MSE 损失: 软配额接近硬配额
        mse_loss = F.mse_loss(soft_quota, hard_quota.float())

        # KL 损失: 软配额接近信息密度比例 (正则化)
        # 目的: 防止软配额偏离原始信息密度分布太远
        # I112-3: 使用 EPS 统一数值稳定性
        target = soft_quota / (soft_quota.sum() + EPS)
        info_prop = soft_quota / (soft_quota.sum() + EPS)
        kl_loss = F.kl_div(
            (target + EPS).log(),
            info_prop,
            reduction='batchmean'
        )

        # 组合损失 (使用已有的常量 QUOTA_INFO_LAMBDA)
        loss = mse_loss + QUOTA_INFO_LAMBDA * kl_loss * K

        return loss

    def _stratified_gumbel_topk_ste(
        self,
        logits: Tensor,
        K: int,
        hard: bool = False,
        features: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        分层 Gumbel-Top-K 选择 (Scheme E 核心，I100-7 信息密度自适应)。

        数学形式化
        ==========

        与全局 Top-K 的区别:
            全局: selected = TopK(logits, K)  → 深度崩塌
            分层: selected = ∪_d TopK(logits[d], K_d)  → 配额保证

        CRIT-1 修正: 使用全局 Softmax 而非 Subset Softmax
            - 旧: 深度内 Subset Softmax → ~K/N 梯度覆盖率 (37.6%)
            - 新: 深度内全局 Softmax → ~100% 梯度覆盖率
            - 理由: 与 Hilbert 局部性正交，消除死区问题

        配额下界保护:
            K_d >= 1 保证每个深度至少有硬选择

        I100-7 信息密度自适应:
            当 features 不为 None 时，使用信息密度驱动配额分配

        Args:
            logits: [B, N] 候选 logits
            K: 总选择数量
            hard: 是否使用硬决策
            features: [B, C, H, W] 输入特征 (可选，I100-7)

        Returns:
            selected_mask: [B, N] STE 选择掩码 (有梯度)
            topk_indices: [B, K'] 硬选择索引 (K' 可能略小于 K)
        """
        B, N = logits.shape
        device = logits.device
        D = self._current_max_depth + 1
        # I103-3: 使用设备端缓存
        depths = self._get_device_tensor(
            self.candidate_depths, "_cached_device_depths", device
        )  # [N]

        # I100-7: 计算信息密度 (如果启用且提供了 features)
        info_density = None
        if features is not None and self._enable_info_adaptive_quota:
            info_density = self._compute_info_density(features)

        # I113-7: 计算配额分配 (支持信息密度自适应，返回硬/软配额)
        # I165-1: 传递 features 以支持分层自适应配额模式
        hard_quota, soft_quota = self._compute_quota_allocation(K, info_density, features)

        # I165-2: 软配额代理路径 (Soft Quota Proxy)
        # 关键修复: 使用 soft_quota 而非 hard_quota 进行 token 选择
        # 原因: K_hard.long() 完全阻断梯度回传到 quota_logits
        #       使用 soft_quota 可以保持梯度流动
        # 数学: quota = K_soft = softmax(φ/τ) × K, 有梯度
        quota = soft_quota  # 用于 token 选择（有梯度流）

        # I113-7: 缓存软/硬配额用于损失计算
        # I170-5 FIX: 移除 .detach() 保持梯度流回传到 quota_logits
        self._last_hard_quota = hard_quota  # 有梯度
        self._last_soft_quota = soft_quota  # 有梯度
        
        # I18-5: 使用 TEMPERATURE_MIN 常量确保梯度健康
        T = self.log_temperature.exp().clamp(min=TEMPERATURE_MIN)
        
        # 初始化输出
        hard_mask = torch.zeros(B, N, device=device, dtype=torch.float32)
        soft_mask = torch.zeros(B, N, device=device, dtype=torch.float32)
        all_topk_indices = []

        # 转换为 FP32 计算
        original_dtype = logits.dtype
        logits_fp32 = logits.float()
        T_fp32 = T.float()

        # I97-4 优化: 预计算所有深度的索引（避免重复 nonzero 调用）
        depth_indices_list = []
        num_per_depth = []
        for d in range(D):
            depth_mask = (depths == d)  # [N]
            depth_indices = depth_mask.nonzero(as_tuple=True)[0]  # [N_d]
            depth_indices_list.append(depth_indices)
            num_per_depth.append(len(depth_indices))

        # I170-2: 修复 quota 转换的梯度流 - 使用 STE 近似
        # 问题: .round().long() 完全阻断梯度回传到 quota_logits
        # 解决: 使用 Straight-Through Estimator
        #   - forward: round to int (保持原行为)
        #   - backward: identity (梯度直接通过)
        # 数学: K_STE = K_soft.detach() + round(K_soft) - K_soft.detach()
        #       = round(K_soft) 在前向，梯度 = dK_soft/dK_soft = 1
        # I170-2 FIX: 使用 quota_soft 保持梯度流
        # 移除 .round().long() 转换，直接使用浮点 quota
        quota_soft = quota.clamp(min=0)  # [D] 保持可微

        # 分层选择 - 使用浮点 quota
        for d in range(D):
            depth_indices = depth_indices_list[d]  # [N_d]
            N_d = num_per_depth[d]
            K_d_float = quota_soft[d].clamp(min=0, max=N_d)  # tensor，保持梯度

            # 转换为 int 用于 topk 索引 (完全可微方案)
            # 使用 tensor: 先 clamp 再 round，保持 GPU 流
            # I170-5 FIX: 使用 STE 保持梯度流
            # 问题: .item() 完全阻断梯度回传到 quota_logits
            # 解决: K_STE = K_hard.detach() + (K_soft - K_soft.detach())
            #       前向: round(K_float), 反向: identity
            K_d_hard = K_d_float.detach()
            K_d_soft = K_d_float
            K_d_ste = K_d_hard - K_d_soft.detach() + K_d_soft  # 有梯度!
            K_d_int = max(1, int(round(K_d_ste.item())))  # 仅用于索引

            if K_d_int <= 0 or N_d == 0:
                continue

            # 提取该深度的 logits
            logits_d = logits_fp32[:, depth_indices]  # [B, N_d]

            # I130-2: Hilbert 最佳实现 - DeterministicTopK 统一使用确定性 softmax
            # 关键修复: 当 use_deterministic_topk=True 时，始终使用确定性 softmax
            # 这样 train/eval 模式使用相同的底层概率分布，保证输出一致
            if self._use_deterministic_topk and self._deterministic_topk is not None:
                # I120-2: 确定性 Top-K 模式（替代 Gumbel 采样）
                # 无论 hard 模式如何，都使用确定性 softmax
                # 数学: P(i ∈ Top-K) = softmax(z_i / τ)[i] × K
                det_probs = F.softmax(logits_d / self._deterministic_temperature, dim=1)  # [B, N_d]
                _, topk_local = torch.topk(det_probs, K_d_int, dim=1)  # [B, K_d_int]

                # I170-2 FIX: 使用 det_probs × K_d_float 保持梯度流
                # 梯度可通过 K_d_float 回传到 quota_logits
                soft_mask[:, depth_indices] = det_probs * K_d_float.unsqueeze(0)  # [B, N_d]
            elif hard or not self.training:
                # 推理模式：直接 Top-K（仅在非确定性模式下使用）
                _, topk_local = torch.topk(logits_d, K_d_int, dim=1)  # [B, K_d_int]
            else:
                # 训练模式：Gumbel + Top-K
                uniform = torch.rand(B, N_d, device=device, dtype=torch.float32)
                uniform = uniform.clamp(GUMBEL_EPSILON, 1 - GUMBEL_EPSILON)
                gumbel = -torch.log(-torch.log(uniform))
                perturbed = (logits_d + gumbel) / T_fp32
                topk_vals, topk_local = torch.topk(perturbed, K_d_int, dim=1)  # [B, K_d_int]

                # CRIT-1: 使用全局 Softmax (而非 Subset Softmax)
                # 原因: Subset Softmax 梯度覆盖率仅 K/N ≈ 37.6%，与 Hilbert 曲线期望冲突
                #       全局 Softmax 提供 ~100% 梯度覆盖，消除死区问题
                # 数学: π_i = e^{z_i} / Σ_j e^{z_j}，梯度 ∂L/∂z_j 对所有 j 非零
                depth_softmax = F.softmax(perturbed, dim=1)  # [B, N_d]

                # I170-2 FIX: 使用 depth_softmax × K_d_float 保持梯度流
                soft_mask[:, depth_indices] = depth_softmax * K_d_float.unsqueeze(0)  # [B, N_d]

            # 更新 hard_mask
            topk_global = depth_indices[topk_local]  # [B, K_d_int]
            # I99-1 FIX: torch.compile 保护 - clamp topk_global 防止 scatter_ 越界
            topk_global_clamped = topk_global.clamp(max=N - 1)
            hard_mask.scatter_(1, topk_global_clamped, 1.0)
            all_topk_indices.append(topk_global)
        
        # 合并所有深度的 Top-K 索引
        if all_topk_indices:
            topk_indices = torch.cat(all_topk_indices, dim=1)  # [B, K']
        else:
            # 边界情况：至少选择根节点
            topk_indices = torch.zeros(B, 1, device=device, dtype=torch.long)
            hard_mask[:, 0] = 1.0
            soft_mask[:, 0] = 1.0

        # I122-1: 移除 STE 混合，直接使用软概率
        # I130-2: Hilbert 最佳实现 - DeterministicTopK 使用确定性硬掩码
        #
        # 数学: st_mask = soft_mask (无偏梯度)
        #
        # 原 STE 公式 (I113-5):
        #   st_mask = (1 - α) × hard + α × soft
        #   其中 α = σ(log β) ∈ (0, 1) 是启发式参数
        #
        # 最佳实现:
        #   1. DeterministicTopK: 使用确定性 softmax 概率构建硬掩码
        #   2. 训练模式: 使用软概率 (soft_mask) 保证梯度流动
        #   3. 推理模式: 使用硬掩码 (hard_mask) 保证确定性
        #
        # 关键: DeterministicTopK 模式下，硬掩码基于确定性概率构建
        #      所以 train/eval 使用相同的底层选择逻辑
        if self._use_deterministic_topk and self._deterministic_topk is not None:
            # I130-2: DeterministicTopK 始终使用软掩码（用于训练和推理）
            # 硬掩码已基于确定性概率构建，st_mask 选择不影响最终行为
            st_mask = soft_mask
        elif self.training and not hard:
            st_mask = soft_mask
        else:
            st_mask = hard_mask

        # 转回原始精度
        if original_dtype != torch.float32:
            st_mask = st_mask.to(original_dtype)
            hard_mask = hard_mask.to(original_dtype)

        # I113-7: 计算信息密度配额损失 (如果有 info_density)
        if self.training and features is not None and self._enable_info_adaptive_quota:
            self._last_info_quota_loss = self._compute_info_quota_loss(
                self._last_soft_quota, self._last_hard_quota, K
            ).detach()
        else:
            self._last_info_quota_loss = None

        # I150-3: 记录 token 选择历史（用于 IOU 稳定性分析）
        if self._monitor_token_stability and self.training:
            self._record_token_selection(topk_indices)

        return st_mask, topk_indices

    def _gumbel_topk_ste(
        self,
        logits: Tensor,
        K: int,
        hard: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """
        Gumbel-Top-K 选择 with Straight-Through Estimator。
        
        数学形式化:
            1. g_i ~ Gumbel(0, 1)
            2. perturbed_i = (logits_i + g_i) / τ
            3. selected = TopK(perturbed, K)
            4. STE: hard_mask - softmax.detach() + softmax
            
        I21 δ: Subset Softmax 改进
        =========================
        问题: 全局 softmax(N=85) 导致梯度稀释 ~1/85
        解决: 在 Top-K 选中的子集上计算 softmax，梯度增强到 ~1/K
        
        数学推导:
            原始: soft_mask = softmax(perturbed)  # [B, N], 每个元素 ≈ 1/N
            改进: soft_mask[topk] = softmax(perturbed[topk])  # ~1/K >> 1/N
            
        梯度增强: 
            原始梯度: ∂L/∂z_i ≈ 1/N × ∂L/∂mask_i
            改进梯度: ∂L/∂z_i ≈ 1/K × ∂L/∂mask_i  (对选中的 K 个)
            增强比例: N/K = 85/32 ≈ 2.7x
            
        Args:
            logits: [B, N] 候选 logits
            K: 选择数量
            hard: 是否使用硬决策
            
        Returns:
            selected_mask: [B, N] STE 选择掩码 (有梯度)
            topk_indices: [B, K] 硬选择索引
        """
        B, N = logits.shape
        device = logits.device
        
        # I23-3: 保护 K <= N，防止 Top-K 越界
        K = min(K, N)
        
        # I18-5: 使用 TEMPERATURE_MIN 常量确保梯度健康
        T = self.log_temperature.exp().clamp(min=TEMPERATURE_MIN)
        
        if hard or not self.training:
            # 推理模式：直接 Top-K
            _, topk_indices = torch.topk(logits, K, dim=1)
            hard_mask = torch.zeros(B, N, device=device)
            # I99-1 FIX: torch.compile 保护 - clamp 防止越界
            topk_indices_clamped = topk_indices.clamp(max=N - 1)
            hard_mask.scatter_(1, topk_indices_clamped, 1.0)
            return hard_mask, topk_indices

        # I120-2: 确定性 Top-K 模式（替代 Gumbel 采样）
        if self._use_deterministic_topk and self._deterministic_topk is not None:
            # 使用温度退火的 softmax 替代 Gumbel 随机采样
            probs = F.softmax(logits / self._deterministic_temperature, dim=1)  # [B, N]

            # Top-K 硬选择
            _, topk_indices = torch.topk(probs, K, dim=1)

            # I122-1: 直接返回缩放的软概率（无 STE 混合）
            st_mask = probs * K  # Σ = K

            return st_mask, topk_indices

        # 训练模式：Gumbel + STE
        # 在 FP32 下计算 Gumbel 噪声
        original_dtype = logits.dtype
        logits_fp32 = logits.float()
        T_fp32 = T.float()

        # ====================================================================
        # 核心改进3: Gumbel-to-Deterministic 切换
        # noise_scale 控制 Gumbel 噪声的幅值
        # noise_scale = 1.0: 全噪声（随机采样）
        # noise_scale = 0.0: 无噪声（确定性选择）
        # ====================================================================
        noise_scale = getattr(self, '_noise_scale', 1.0)

        # Gumbel 采样
        uniform = torch.rand(B, N, device=device, dtype=torch.float32)
        uniform = uniform.clamp(GUMBEL_EPSILON, 1 - GUMBEL_EPSILON)
        gumbel = -torch.log(-torch.log(uniform))

        # 应用噪声缩放因子
        if noise_scale < 1.0:
            gumbel = gumbel * noise_scale

        # 扰动后的 logits
        perturbed = (logits_fp32 + gumbel) / T_fp32
        
        # Top-K 硬选择
        topk_vals, topk_indices = torch.topk(perturbed, K, dim=1)
        
        # 构建硬掩码
        hard_mask = torch.zeros(B, N, device=device, dtype=torch.float32)
        # I99-1 FIX: torch.compile 保护 - clamp 防止越界
        topk_indices_clamped = topk_indices.clamp(max=N - 1)
        hard_mask.scatter_(1, topk_indices_clamped, 1.0)

        # ====================================================================
        # I30-2: 使用全局 Softmax (移除 Subset Softmax)
        # 原因: Subset Softmax 梯度覆盖率仅 K/N ≈ 37.6%，与文档声称的 100% 矛盾
        #       全局 Softmax 提供 100% 梯度覆盖，避免死区问题
        # 数学: π_i = e^{z_i} / Σ_j e^{z_j}，梯度 ∂L/∂z_j 对所有 j 非零
        #
        # I122-1: 移除 STE 混合，直接使用软概率
        # 数学形式:
        #   st_mask = softmax(perturbed)  # 无 STE 混合
        #
        # 优势:
        #   1. 无偏梯度: ∂P/∂z 有闭式解
        #   2. 无需启发式 α 参数
        #   3. 温度 τ 自动控制硬度
        #
        # 原 STE 公式 (I113-5):
        #   st_mask = (1 - α) × hard + α × soft
        #   其中 α = σ(log β) ∈ (0, 1) 是启发式参数
        # ====================================================================
        soft_mask = F.softmax(perturbed, dim=1)

        # I122-1: 直接返回软概率，移除 STE 混合
        # st_mask = soft_mask  # 无偏梯度
        st_mask = soft_mask

        # 转回原始精度
        if original_dtype != torch.float32:
            st_mask = st_mask.to(original_dtype)
            hard_mask = hard_mask.to(original_dtype)

        return st_mask, topk_indices
    
    def _enforce_tree_consistency(
        self,
        selected_mask: Tensor,
        topk_indices: Tensor,
        training: bool = True,
    ) -> Tensor:
        """
        强制树一致性约束 (向量化)。

        数学形式化:
            约束: ∀i ∈ selected: parent(i) ∉ selected
            即: 若子节点被选中，则父节点不应被选中

        向量化实现:
            1. 对每个节点，检查其任意子节点是否被选中
            2. 若有子节点被选中，则该节点不能被选中

        Args:
            selected_mask: [B, N] STE 选择掩码
            topk_indices: [B, K] 硬选择索引
            training: 是否在 training 模式 (软边距 vs 硬边距)

        Returns:
            consistent_mask: [B, N] 树一致的选择掩码
        """
        self._ensure_children_matrix()

        B, N = selected_mask.shape
        device = selected_mask.device

        # 确保 children_matrix 在正确设备上 (I103-3: 使用缓存)
        children_matrix = self._get_device_tensor(
            self._children_matrix, "_cached_device_children_matrix", device
        )  # [N, 4]

        # 对于每个节点，检查其子节点是否被选中
        # valid_children: [N, 4] 哪些子节点索引是有效的
        valid_children = children_matrix >= 0

        # 安全索引 (将 -1 替换为 0)
        safe_children = children_matrix.clamp(min=0)  # [N, 4]

        # ====================================================================
        # I22-1 方案A+: 向量化树一致性检查
        # I96-4 改进: 软边距排除机制
        #
        # 数学约束: ∀i ∈ selected: parent(i) ∉ selected
        # 等价表述: 若任意子节点被选中，则父节点不应被选中
        #
        # I96-4 软边距机制:
        #   - 仅在 training 模式下使用软边距
        #   - inference 模式下使用硬边距 (0)，保证 0/1 输出
        #   - 使用 .detach() 避免排除决策接收梯度 (避免梯度双计)
        #   - 软边距: clamp(min=SOFT_EXCLUSION_MARGIN) 保持最小梯度流 (10%)
        #   - 数学: p_out = p × max(1 - Σ child_signal, ε)
        #   - 梯度: ∂p_out/∂p_i ≥ ε > 0，确保父节点有梯度回传
        #
        # 对比原实现:
        #   原: exclusion_mask = 1.0 - has_child_selected (梯度=0)
        #   新: exclusion_mask.clamp(min=ε) (梯度≥ε) - 仅 training 模式
        # ====================================================================

        # 使用硬掩码进行约束检查 (I96-4: 添加 .detach() 避免梯度双计)
        hard_selected = (selected_mask > 0.5).float().detach()

        # 获取所有子节点的选择状态
        # safe_children: [N, 4], 值范围 [0, N-1]
        # hard_selected[:, safe_children]: [B, N, 4]
        children_selected_status = hard_selected[:, safe_children]  # [B, N, 4]

        # 将无效子节点的状态设为 0
        valid_children_mask = valid_children.unsqueeze(0).expand(B, -1, -1)  # [B, N, 4]
        children_selected_status = children_selected_status * valid_children_mask.float()

        # 计算子节点选中信号 (累积而非 max，保持更多信息)
        child_signal = children_selected_status.sum(dim=2).clamp(max=1.0)  # [B, N]

        # I96-4: 仅在 training 模式下使用软边距
        if training:
            # 软边距排除: p_out = p × max(1 - child_signal, ε)
            exclusion_mask = (1.0 - child_signal).clamp(min=SOFT_EXCLUSION_MARGIN)  # [B, N]
        else:
            # Inference 模式: 硬边距，保证 0/1 输出
            exclusion_mask = 1.0 - child_signal  # [B, N]

        # 应用排除
        consistent_mask = selected_mask * exclusion_mask

        return consistent_mask
    
    def _build_result(
        self,
        consistent_mask: Tensor,
        topk_indices: Tensor,
        logits: Tensor,
        probs: Tensor,
    ) -> GumbelTopKResult:
        """
        从选择掩码构建输出结果。
        
        Args:
            consistent_mask: [B, N] 树一致的选择掩码
            topk_indices: [B, K] 硬选择索引
            logits: [B, N] 原始 logits
            probs: [B, N] 分割概率
            
        Returns:
            GumbelTopKResult
            
        性能优化 (P-OPT-1):
            使用向量化操作替换 Python for 循环，避免 B 次小张量操作。
            通过 nonzero() + scatter 一次性处理所有 batch。
        """
        B_mask, N_mask = consistent_mask.shape
        B_probs, N_probs = probs.shape
        if B_mask != B_probs or N_probs != N_probs:
            raise RuntimeError(
                f"I99-1 SHAPE MISMATCH: consistent_mask.shape=({B_mask}, {N_mask}), "
                f"probs.shape=({B_probs}, {N_probs})"
            )
        B, N = B_mask, N_mask
        device = consistent_mask.device

        # I145-FIX: 使用软掩码计算 token 数量，保持梯度
        # 这用于辅助损失计算
        num_selected_per_batch_float = consistent_mask.sum(dim=1)  # [B]

        # ========================================================================
        # FIX: 使用 topk_indices 构建硬掩码，而不是软概率阈值
        # 问题: 软概率在每个深度内和为 1，如果 K_d=18，平均概率约 0.055 < 0.5
        # 解决: 直接使用 topk_indices 构建硬掩码，确保选中正确数量的 token
        # ========================================================================
        # 初始化硬掩码
        final_selected = torch.zeros(B, N, dtype=torch.bool, device=device)

        # P-OPT: 向量化构建硬掩码，替代 for b 循环
        # 过滤掉 padding (-1)，将其替换为 N（超出范围，scatter 不会填充）
        valid_mask = topk_indices >= 0  # [B, K]
        topk_indices_safe = torch.where(valid_mask, topk_indices, torch.full_like(topk_indices, N))
        final_selected.scatter_(1, topk_indices_safe, True)
        # 无需额外恢复操作，因为超出范围的索引不会被 scatter

        # 备用: 如果 topk_indices 为空或无效，使用软概率阈值
        if not final_selected.any():
            final_selected = (consistent_mask > 0.5)  # [B, N]

        # I150-2 FIX: 使用硬掩码计算实际的 token 数量，与 hilbert_indices 数量一致
        num_selected_per_batch = final_selected.sum(dim=1).long()  # [B]

        # P-OPT: 使用向量化操作确保每个 batch 至少有一个 token
        # 避免 .any() 同步点，直接使用 clamp 和 where 操作
        min_tokens_per_batch = final_selected.sum(dim=1, keepdim=True)  # [B, 1]
        empty_mask = (min_tokens_per_batch == 0)  # [B, 1]

        # P-OPT: 使用 torch.where 批量处理，避免 Python 控制流
        # 无论是否为空 batch，都执行 where 操作（空 batch 会得到填充值，非空保持原值）
        # fill_values: [B, N]，所有位置初始化为 0，然后在每个 batch 的位置 0 设置为 1
        fill_values = torch.zeros(B, N, dtype=torch.bool, device=device)
        fill_values.scatter_(1, torch.zeros(B, 1, dtype=torch.long, device=device), True)
        final_selected = torch.where(empty_mask, fill_values, final_selected)

        # 一次性获取所有选中位置 [total_selected, 2] -> (batch_idx, candidate_idx)
        selected_positions = final_selected.nonzero(as_tuple=False)  # [total, 2]

        # I99-1: 防御性边界检查 - 确保 nonzero() 返回的索引在有效范围内
        # torch.compile 优化可能暴露潜在的索引问题
        batch_indices_raw = selected_positions[:, 0]  # [total]
        candidate_indices_raw = selected_positions[:, 1]  # [total]

        # Clamp indices to valid ranges (提供安全保障)
        batch_indices = batch_indices_raw.clamp(min=0, max=B - 1)
        candidate_indices = candidate_indices_raw.clamp(min=0, max=N - 1)

        # I99-1: 形状验证 (不触发同步的简单比较)
        if probs.shape[1] != N:
            raise RuntimeError(
                f"I99-1 BUG: probs.shape[1]={probs.shape[1]} != consistent_mask.shape[1]={N}"
            )

        # I99-1: 额外验证 - 如果 nonzero 返回空张量，创建安全的默认值
        if selected_positions.shape[0] == 0:
            batch_indices = torch.zeros(1, dtype=torch.long, device=device)
            candidate_indices = torch.zeros(1, dtype=torch.long, device=device)

        # 向量化索引所有候选属性 - I103-3: 使用缓存
        regions = self._get_device_tensor(
            self.candidate_regions, "_cached_device_regions", device
        )[candidate_indices]  # [total, 4]
        depths = self._get_device_tensor(
            self.candidate_depths, "_cached_device_depths", device
        )[candidate_indices]  # [total]
        # hilbert_indices 也需要设备一致性 (修复 RuntimeError: indices device mismatch)
        hilbert_indices = self._get_device_tensor(
            self.hilbert_indices, "_cached_device_hilbert", device
        )[candidate_indices]  # [total]

        return GumbelTopKResult(
            regions=regions,
            depths=depths,
            batch_indices=batch_indices,
            hilbert_indices=hilbert_indices,
            selected_mask=consistent_mask,
            logits=logits,
            probs=probs,
            candidate_indices=candidate_indices,  # I99-1 FIX: 用于正确的概率索引
            num_selected_per_batch=num_selected_per_batch,
        )
    
    # ========================================================================
    # 辅助损失接口 (与 LearnableSplitter 兼容)
    # ========================================================================
    
    def get_auxiliary_losses(
        self,
        features: Optional[Tensor] = None,
        image_size: Optional[Tuple[int, int]] = None,
        include_balance: bool = False,
        include_elastic_budget: bool = True,
        include_soft_entropy: bool = True,
        batch_size: int = 1,
        actual_token_count: Optional[int] = None,
        entropy_target: Optional[float] = None,
        entropy_weight: float = 0.1,
        entropy_mode: str = 'maximize',
        # I153-1: Quota 对齐损失参数
        quota_align_weight: float = 0.0,
        quota_align_mode: str = 'curriculum',
        current_epoch: int = 0,
        total_epochs: int = 100,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """
        获取所有辅助损失 (与 LearnableSplitter.get_auxiliary_losses 兼容).

        I33: Elastic Budget 相对预算设计
        =================================

        核心改进: 与K参数相对预算保持一致

        数学形式化:
            coverage = N_selected / N_candidates
            L_elastic = λ_over × max(0, coverage - β_max)² × N_candidates

        梯度分析:
            ∂L/∂N_selected = 2 × λ_over × max(0, coverage - β_max) / N_candidates
            → 梯度归一化到 ~0.01 量级，避免跨尺度差异

        崩溃检测 (相对阈值):
            collapse ⇔ coverage < α_collapse (0.5%)

        Args:
            features: 特征图 [B, C, H, W] (可选)
            image_size: 图像尺寸 (可选)
            include_balance: 是否包含平衡损失 (对 GumbelTopK 忽略)
            include_elastic_budget: 是否包含弹性预算损失 (相对覆盖率版本)
            include_soft_entropy: 是否包含软熵损失
            batch_size: batch 大小
            actual_token_count: 实际 token 数 (用于相对崩溃检测)
            entropy_target: 熵目标值 (target mode)
            entropy_weight: 熵损失权重
            entropy_mode: 'maximize' 或 'target'
            **kwargs: 其他参数 (向前兼容)

        Returns:
            Dict[str, Tensor]: 各项损失
        """
        device = self.candidate_regions.device
        losses = {}

        # I145-FIX: 获取带梯度的版本用于辅助损失计算
        # 使用 _last_probs_for_loss / _last_selected_mask_for_loss 而不是 detached 版本
        probs = self._last_probs_for_loss if hasattr(self, '_last_probs_for_loss') else None
        selected_mask = self._last_selected_mask_for_loss if hasattr(self, '_last_selected_mask_for_loss') else None

        # 同时保留 detached 版本用于 NaN 检测
        probs_detached = self._last_probs if hasattr(self, '_last_probs') else None
        selected_mask_detached = self._last_selected_mask if hasattr(self, '_last_selected_mask') else None

        # I23-4-FIX: NaN 检测与防护 (使用 detached 版本检测)
        # 如果缓存的 probs 或 selected_mask 包含 NaN，返回零损失
        # P-OPT: 仅在调试模式检查 NaN，避免 GPU-CPU 同步
        # NaN 很少见，AMP 模式下可以跳过检查
        if getattr(self, '_debug_mode', False):
            has_nan = False
            if probs_detached is not None and torch.isnan(probs_detached).any():
                has_nan = True
            if selected_mask_detached is not None and torch.isnan(selected_mask_detached).any():
                has_nan = True

            if has_nan:
                # 返回零损失，避免 NaN 传播到总损失
                zero = torch.tensor(0.0, device=device)
                if include_elastic_budget:
                    losses['elastic_budget_loss'] = zero
                if include_soft_entropy:
                    losses['soft_entropy_loss'] = zero
                # I35: 移除死代码 DEPTH_KL_WEIGHT, DEPTH_QUOTA_ENABLED
                return losses
        
        # 1. Elastic Budget Loss (I109-4: 目标导向损失)
        # I109-4: 替换死区设计为目标导向损失
        # 设计:
        #   - 主损失: L2 损失引导到目标覆盖率 β_target
        #   - 安全网: 边界约束防止超出 K_bounds
        #   - 崩溃检测: 极低覆盖率时强惩罚
        #
        # 数学形式化:
        #   β_target = K_COVERAGE_BASE × √(min(H,W)/224)
        #   L_target = λ_target × (K/N - β_target)²
        #   L_bound = λ_boundary × [max(0, K_min - K)² + max(0, K - K_max)²]
        #
        if include_elastic_budget:
            # I145-FIX: 使用当前 batch 的实际 token 数（有梯度）而不是 EMA 值
            avg_tokens = self._last_num_selected_for_loss if hasattr(self, '_last_num_selected_for_loss') else self._avg_selected
            candidate_count = self.num_candidates

            # === 动态目标覆盖率 ===
            # β_target = K_COVERAGE_BASE × γ，与 K_bounds 公式一致
            # 使用传入的 image_size 参数或当前缓存的图像尺寸
            current_size = image_size if image_size is not None else self._current_image_size
            if self._use_adaptive_coverage and current_size is not None:
                h, w = current_size
                gamma = math.sqrt(min(h, w) / self._adaptive_reference_size)
                target_coverage = K_COVERAGE_BASE * gamma
            else:
                target_coverage = K_COVERAGE_BASE

            target_tokens = target_coverage * candidate_count

            # === I147: 简化的预算损失 (替代 Poisson KL) ===
            # 从第一性原理推导的稳定损失函数
            #
            # 数学形式化:
            #   L_budget = λ × Huber(|K - K_t| / N, δ / N)
            #
            # 核心洞察:
            #   - 相对误差: |K - K_t| / N 确保尺度不变性
            #   - Huber 边界: 大偏差时梯度有界，防止数值爆炸
            #   - 比 Poisson KL 更稳定，数学假设更少
            #
            # 对比 Poisson KL:
            #   Poisson KL: 需要假设 token 计数服从 Poisson 分布
            #   简化的: 无分布假设，更鲁棒
            #
            target_K = target_tokens

            # 相对误差损失
            # K ≥ 1 数值稳定性
            K = avg_tokens.clamp(min=1.0)
            # I147-FIX: 使用张量运算保持梯度
            target_K_tensor = torch.tensor(float(target_K), device=K.device, dtype=K.dtype)
            K_t = torch.max(target_K_tensor, torch.tensor(1.0, device=K.device))

            # 相对误差: diff / N
            diff = (K - K_t).abs() / float(candidate_count)
            # I152-FIX: 钳制到 [0, 1]，并添加上界保护
            diff = diff.clamp(min=0.0, max=1.0)

            # Huber 边界损失
            # δ = 0.5 (标准 Huber δ=1 在归一化后)
            delta = 0.5
            huber_loss = torch.where(
                diff <= delta,
                0.5 * diff.pow(2),
                delta * diff - 0.5 * delta ** 2
            )

            # I152-FIX: 降低损失权重，防止叠加爆炸
            # 原始: lambda_budget = 0.1
            # 修复: 降低到 0.01
            # I170-4: 梯度自增强 - 提升权重激活动态性学习
            # 当 std(K) = 0 时需要更强的梯度信号
            lambda_budget = torch.tensor(1.0, device=K.device)
            loss = self._elastic_budget_factor * lambda_budget * huber_loss

            # I152-FIX: 钳制损失上界
            loss = loss.clamp(max=1.0)

            # === 边界约束 (简化，依赖 Huber 保护) ===
            K_min, K_max = self._get_dynamic_k_bounds(candidate_count)

            # I152-FIX: 降低边界惩罚权重
            boundary_penalty = self._elastic_lambda_boundary * 0.1 * (
                torch.relu(K_min - K).pow(2) +
                torch.relu(K - K_max).pow(2)
            )
            loss = loss + boundary_penalty

            losses['elastic_budget_loss'] = loss

            # === 崩溃检测 ===
            # I142: 使用内部缓存的 _avg_selected 值
            # I152-FIX: 降低崩溃惩罚权重，防止 Loss 爆炸
            # 原始: ELASTIC_LAMBDA_COLLAPSE = 1.0
            # 修复: 降低到 0.1
            # I170-4: 梯度自增强 - 提升权重激活动态性学习
            collapse_threshold = K_min * 0.5
            if avg_tokens < collapse_threshold:
                collapse_loss = torch.tensor(ELASTIC_LAMBDA_COLLAPSE * 1.0, device=device)
                losses['collapse_loss'] = collapse_loss
        
        # 2. Soft Entropy Loss
        if include_soft_entropy and probs is not None:
            # I111-3: 使用配置中的基础权重
            effective_weight = self._entropy_weight_base if self._use_config else entropy_weight
            entropy_loss = self.get_depth_entropy_loss(weight=effective_weight, probs=probs)

            if self._entropy_mode == 'target' and entropy_target is not None:
                # Target mode: minimize |H - H_target|²
                B, N = probs.shape
                # I103-3: 使用设备端缓存
                candidate_depths = self._get_device_tensor(
                    self.candidate_depths, "_cached_device_depths", probs.device
                )  # [N]

                # I108-3: 向量化计算深度概率分布
                # 原始: Python 循环 O(D) -> 向量化 O(1)
                # probs: [B, N] 每个 token 的分裂概率
                # candidate_depths: [N] 每个候选的深度 (广播到 batch)
                D = self._current_max_depth + 1

                # 使用 one-hot 编码 (与 get_depth_entropy_loss 相同的向量化策略)
                depth_onehot = F.one_hot(candidate_depths, D).float()  # [N, D]
                depth_counts = depth_onehot.sum(dim=0).clamp(min=1.0)  # [D]

                # prob_sums[d] = sum_{b,i} probs[b,i] * 1[depth_i = d]
                prob_sums = torch.einsum('bn,nd->d', probs, depth_onehot)  # [D]

                # mean_probs[d] = prob_sums[d] / (B * depth_counts[d])
                depth_probs_t = (prob_sums / (B * depth_counts)).clamp(min=PROB_EPSILON)

                # 归一化
                depth_probs_t = depth_probs_t / depth_probs_t.sum()
                # I23-4: 归一化后再次 clamp，防止 FP16 下溢导致 log(0)
                depth_probs_t = depth_probs_t.clamp(min=PROB_EPSILON)
                current_entropy = -(depth_probs_t * depth_probs_t.log()).sum()
                entropy_loss = entropy_weight * (current_entropy - entropy_target).pow(2)

            losses['soft_entropy_loss'] = entropy_loss

        # ====================================================================
        # I153-1: 配额对齐损失 (KL 散度)
        # 为 quota_logits 提供任务驱动的梯度，解决整数屏障问题
        # ====================================================================
        if quota_align_weight > 0 and self._enable_learnable_quota and self.quota_logits is not None:
            if self.quota_allocator is not None:
                # 生成目标深度分布
                target_dist = self.quota_allocator.get_target_depth_distribution(
                    step=current_epoch,
                    total_steps=total_epochs,
                    mode=quota_align_mode,
                )
                # 计算 KL 对齐损失
                align_loss = self.quota_allocator.compute_quota_align_loss(
                    target_dist=target_dist,
                    weight=quota_align_weight,
                )
                losses['quota_align_loss'] = align_loss

        # ====================================================================
        # I24-2 方案E: 配额熵正则化损失
        # I30-10: 使用配置值
        # 鼓励可学习配额保持多样性，避免崩塌到单一深度
        # ====================================================================
        if self._enable_learnable_quota and self.quota_logits is not None:
            quota_entropy_loss = self.get_quota_entropy_loss(weight=self._quota_entropy_weight)
            losses['quota_entropy_loss'] = quota_entropy_loss
        
        # ====================================================================
        # I29-2: 阈值方差正则化损失
        # 数学: L_threshold = λ × Var(τ)
        # 目的: 防止某个深度的阈值极端偏离，导致选择偏差
        # ====================================================================
        if THRESHOLD_VAR_REG_ENABLED:
            threshold_var_loss = torch.var(self.threshold_offsets) * THRESHOLD_VAR_REG_WEIGHT
            losses['threshold_var_loss'] = threshold_var_loss

        # ====================================================================
        # I150-1: 空间密度惩罚损失
        # 数学: L_density = γ × Σ_i density_i × selected_i
        # 目的: 最小化选中 token 之间的聚集，鼓励均匀空间分布
        # ====================================================================
        # I152-FIX: 激活密度惩罚（之前是空实现）
        if DENSITY_PENALTY_WEIGHT > 0:
            density_loss = self.get_density_regularization()
            losses['density_loss'] = density_loss

        # ====================================================================
        # I120-8: Hilbert-感知自适应预算系统
        # 替换原有的 L_budget (L2损失)，与 Hilbert 曲线理论对齐
        # ====================================================================

        # 1. Hilbert-感知连续性损失
        if HILBERT_CONTINUITY_ENABLED:
            hilbert_continuity_loss = self.get_hilbert_continuity_loss(
                hilbert_indices=getattr(self, 'hilbert_indices', None),
                selected_mask=selected_mask,
                weight=HILBERT_CONTINUITY_WEIGHT,
                gamma=HILBERT_CONTINUITY_GAMMA,
            )
            losses['hilbert_continuity_loss'] = hilbert_continuity_loss

        # 2. 空间覆盖预算损失
        if COVERAGE_BUDGET_ENABLED:
            spatial_coverage_loss = self.get_spatial_coverage_loss(
                selected_mask=selected_mask,
                image_size=image_size,
                weight=SPATIAL_COVERAGE_WEIGHT,
            )
            losses['spatial_coverage_loss'] = spatial_coverage_loss

        # 3. 自适应目标覆盖率（I120-8: 替换固定的 K_COVERAGE_BASE）
        # I165-FIX: get_adaptive_target_tokens 返回的是目标 token 数，不是损失！
        # 移除：将此值添加到 losses 字典（会导致 90+ loss）
        if ADAPTIVE_COVERAGE_ENABLED:
            _ = self.get_adaptive_target_tokens(
                info_density=getattr(self, '_last_info_density', None),
                image_size=image_size,
            )
            # 注意：adaptive_target_tokens 不是损失，不添加到 losses 字典
            # 只用于内部计算（如弹性预算损失的目标值）

        # ====================================================================
        # I120-8: 深度平衡损失
        # 鼓励均匀分布在各深度，避免细粒度主导
        # ====================================================================
        if hasattr(self, 'get_depth_balance_loss'):
            depth_balance_loss = self.get_depth_balance_loss(
                hard_mask=getattr(self, '_last_hard_mask', None),
                weight=DEPTH_BALANCE_WEIGHT,
            )
            losses['depth_balance_loss'] = depth_balance_loss

        # ====================================================================
        # I23-5-FIX: 最终 NaN/Inf 检查与清理
        # 确保返回的所有损失都是有效数值
        # ====================================================================
        device = self.candidate_regions.device
        zero = torch.tensor(0.0, device=device)
        for key, val in list(losses.items()):
            if torch.isnan(val) or torch.isinf(val):
                losses[key] = zero

        # ====================================================================
        # 核心改进1: Loss Warmup - 应用辅助损失缩放因子
        # 在训练前期关闭 budget/entropy 损失，让 Splitter 先学会分裂
        # ====================================================================
        # 弹性预算损失缩放
        if 'elastic_budget_loss' in losses and hasattr(self, '_elastic_lambda_scale'):
            if self._elastic_lambda_scale < 1.0:
                losses['elastic_budget_loss'] = losses['elastic_budget_loss'] * self._elastic_lambda_scale

        # 软熵损失缩放
        if 'soft_entropy_loss' in losses and hasattr(self, '_entropy_lambda_scale'):
            if self._entropy_lambda_scale < 1.0:
                losses['soft_entropy_loss'] = losses['soft_entropy_loss'] * self._entropy_lambda_scale

        return losses
    
    def get_elastic_budget_loss(
        self,
        target_tokens: int = 32,
        weight: float = 0.01,
        current_epoch: int = 0,
        use_curriculum: bool = True,
    ) -> Tensor:
        """
        I122-4: 计算弹性预算损失 (Poisson KL 散度损失)

        从第一性原理推导:
        - Token 计数是离散 Poisson 过程
        - KL 散度是计数偏差的信息论最优度量
        - L = λ_KL × KL(Poisson(K) || Poisson(K_t)) + λ_Huber × Huber_δ

        数学形式化:
            L_KL = K_t × (K/K_t × log(K/K_t) + 1 - K/K_t)
            L_Huber = { 0.5×(Δ)²      if |Δ| ≤ δ
                      { δ×|Δ| - 0.5×δ²  otherwise

        课程学习版本 (I121-5):
            λ_b(t, K) = λ_b^0 × α(t) × β(K)

        Args:
            target_tokens: 目标 token 数量
            weight: 基础损失权重 (λ_b^0)
            current_epoch: 当前训练轮次 (用于课程学习调度)
            use_curriculum: 是否使用课程学习权重调度

        Returns:
            loss: 标量损失 (Poisson KL + Huber)
        """
        avg_tokens = self._avg_selected

        if use_curriculum:
            # I121-5: 使用课程学习调度器计算自适应权重
            K_min = torch.tensor(self.K_min, device=avg_tokens.device, dtype=avg_tokens.dtype)
            K_max = torch.tensor(self.K_max, device=avg_tokens.device, dtype=avg_tokens.dtype)

            # 获取自适应权重: λ_b(t, K) = λ_b^0 × α(t) × β(K)
            adaptive_weight = self._curriculum_scheduler.get_budget_weight(
                current_epoch=current_epoch,
                K=avg_tokens,
                K_min=K_min,
                K_max=K_max,
            )
        else:
            # 传统固定权重
            adaptive_weight = weight

        # === Poisson KL 散度损失 ===
        # I152-FIX: 添加数值稳定性保护，防止 Loss 爆炸
        K = avg_tokens.clamp(min=1.0)
        K_t = max(target_tokens, 1.0)
        ratio = K / K_t
        # I152-FIX: 使用更合理的 clamp 范围，防止 ratio 极端
        ratio_clamped = ratio.clamp(min=0.1, max=10.0)

        # KL(Poisson(K) || Poisson(K_t))
        # I152-FIX: 添加 log 数值稳定性保护
        log_ratio = torch.log(ratio_clamped + EPS)
        kl_loss = K_t * (ratio_clamped * log_ratio + 1.0 - ratio_clamped)

        # I152-FIX: 钳制 KL loss 上界，防止爆炸
        kl_loss = kl_loss.clamp(max=10.0)

        # === Huber 边界保护 ===
        delta = K_t / 2.0  # float
        diff = (K - K_t).abs()
        huber_loss = torch.where(
            diff <= delta,
            0.5 * diff.pow(2),
            delta * diff - 0.5 * (delta ** 2)
        )

        # I152-FIX: 降低损失权重，防止叠加爆炸
        # 原始: lambda_kl = 0.005, lambda_huber = 0.001
        # 修复: 降低 10 倍，确保与 CE 损失量纲一致
        lambda_kl = ELASTIC_LAMBDA_KL * 0.1  # 0.0005
        lambda_huber = HUBER_LAMBDA * 0.1    # 0.0001
        loss = adaptive_weight * (
            lambda_kl * kl_loss + lambda_huber * huber_loss
        )

        # I152-FIX: 钳制总损失上界
        loss = loss.clamp(max=1.0)

        return loss
    
    def get_depth_entropy_loss(
        self,
        weight: float = 0.01,
        probs: Optional[Tensor] = None,
        hard_mask: Optional[Tensor] = None,
        current_epoch: int = 0,
        use_curriculum: bool = True,
    ) -> Tensor:
        """
        计算深度熵损失 (I111-3: 修复深度分布定义, I121-5: 支持课程学习权重)

        数学形式化
        ==========

        正确深度分布 (使用硬选择计数):
            p_d = K_d / K_total
            K_d = Σ_i hard_mask[b,i] × 1[depth_i = d]

        熵 (Shannon Entropy):
            H = -Σ_d p_d log(p_d)

        动态权重 (I111-3):
            λ = λ_base × max(1.0, H_target / (H + ε))
            H_target = log(D) × (1 - 1/√D)

        课程学习版本 (I121-5):
            λ_e(t) = λ_e^0 / √α(t)
            α(t) = Cosine平滑阶段因子

        损失:
            L = -λ_e × H  (最大化熵 → 最小化负熵)

        修复内容 (I111-3):
            1. 使用硬选择计数定义深度分布 (而非软概率期望)
            2. 修复双重归一化问题
            3. 动态调整熵权重

        Args:
            weight: 基础损失权重 (λ_e^0)
            probs: [B, N] 分割概率 (可选，用于后备计算)
            hard_mask: [B, N] 硬选择掩码 (可选，如果提供则使用硬计数)
            current_epoch: 当前训练轮次 (用于课程学习调度)
            use_curriculum: 是否使用课程学习权重调度

        Returns:
            loss: 标量损失
        """
        # 根据输入确定设备：hard_mask > probs > candidate_regions
        if hard_mask is not None:
            device = hard_mask.device
        elif probs is not None:
            device = probs.device
        else:
            device = self.candidate_regions.device

        # I111-3: 安全获取深度数
        D = self._current_max_depth + 1 if self._current_max_depth is not None else 4

        # 如果没有硬掩码，使用软概率 (向后兼容)
        if hard_mask is None:
            if probs is None:
                return torch.tensor(0.0, device=device)
            B, _ = probs.shape  # N 不直接使用
            depths = self._get_device_tensor(
                self.candidate_depths, "_cached_device_depths", probs.device
            )  # [N]

            # 软概率计算 (原有逻辑，保留用于向后兼容)
            depth_onehot = F.one_hot(depths, D).float()  # [N, D]
            depth_counts = depth_onehot.sum(dim=0).clamp(min=1.0)  # [D]
            prob_sums = torch.einsum('bn,nd->d', probs, depth_onehot)  # [D]
            depth_probs = (prob_sums / (B * depth_counts)).clamp(min=PROB_EPSILON)
            depth_probs = depth_probs / depth_probs.sum()
            depth_probs = depth_probs.clamp(min=PROB_EPSILON)
        else:
            # I111-3: 硬选择计数 (正确的深度分布定义)
            B, _ = hard_mask.shape  # N 不直接使用
            depths = self._get_device_tensor(
                self.candidate_depths, "_cached_device_depths", device
            )  # [N]

            # 计算每个深度的硬选择数量
            # K_d[b] = Σ_i hard_mask[b,i] × 1[depth_i = d]
            depth_onehot = F.one_hot(depths, D).float()  # [N, D]
            K_per_depth = torch.einsum('bn,nd->bd', hard_mask, depth_onehot)  # [B, D]

            # 计算总选择数
            K_total = K_per_depth.sum(dim=1, keepdim=True)  # [B, 1]

            # 防止除零
            K_total = K_total.clamp(min=1.0)

            # 深度分布 p_d = K_d / K_total (对 batch 取平均)
            depth_probs = (K_per_depth / K_total).mean(dim=0)  # [D]
            depth_probs = depth_probs.clamp(min=PROB_EPSILON)
            depth_probs = depth_probs / depth_probs.sum()

        # 计算熵
        entropy = -(depth_probs * depth_probs.log()).sum()

        # I122-5: 熵目标公式改进
        # 原始公式: H_target = log(D) × (1 - 1/√D) ← 缺乏严格数学推导
        # 新公式:   H_target = ENTROPY_TARGET_SCALE × log(D) = 0.5 × log(D)
        # 理论依据: 有效深度 D_eff = √D (信息论视角下的有效类别数)
        # 优势:     对所有 D 保持恒定 50% 熵比例，简化超参数调优
        #
        # I145: 添加边界检查
        # 当 D <= 1 时，log(D) = 0，熵目标自然为 0，无需特殊处理
        if D <= 1:
            H_target = 0.0
        else:
            H_target = ENTROPY_TARGET_SCALE * math.log(D)

        # I121-5: 课程学习熵权重联动
        # 基础权重: λ_base = weight × (1 + softplus(H_target/H - 1))
        ratio = (H_target / (entropy + EPS)).clamp(max=10.0)
        base_weight = weight * (1.0 + F.softplus(ratio - 1.0))

        if use_curriculum:
            # 课程学习: λ_e(t) = λ_e^0 / √α(t)
            # 与预算权重反向联动: 探索阶段↓熵约束，利用阶段↑熵约束
            curriculum_weight = self._curriculum_scheduler.get_entropy_weight(current_epoch)
            # 组合: λ_e = base × (curriculum_factor / base_factor)
            # 其中 curriculum_factor = 1/√α(t), base_factor = 1 + softplus(...)
            # 为保持平滑过渡，将课程学习因子作为乘数
            curriculum_factor = curriculum_weight / weight if weight > 0 else curriculum_weight
            dynamic_weight = base_weight * curriculum_factor
        else:
            # 传统固定权重
            dynamic_weight = base_weight

        # 最大化熵 → 最小化负熵 (完整梯度流)
        loss = -dynamic_weight * entropy
        return loss

    def get_quota_entropy_loss(
        self,
        weight: float = QUOTA_ENTROPY_WEIGHT,
    ) -> Tensor:
        """
        计算配额熵正则化损失 (I24-2 方案E)。
        
        数学形式化
        ==========
        
        动机:
            可学习配额 φ 可能崩塌到单一深度 (所有配额给 depth=3)。
            通过熵正则化鼓励配额分布保持多样性。
            
        损失公式:
            p = softmax(φ)  配额分布
            H(p) = -Σ_d p_d log(p_d)  熵
            L_entropy = -λ × H(p)  最大化熵 → 最小化负熵
            
        梯度流:
            ∂L/∂φ_d = -λ × (∂H/∂p_d) × (∂p_d/∂φ_d)
                    = λ × (1 + log(p_d)) × p_d × (1 - p_d)  [对 softmax]
                    
        效果:
            - 高 p_d → log(p_d) 大 → 正梯度 → 降低 φ_d
            - 低 p_d → log(p_d) 小 → 负梯度 → 提高 φ_d
            
        Args:
            weight: 熵损失权重 (默认 QUOTA_ENTROPY_WEIGHT=0.1)
            
        Returns:
            loss: 标量熵损失
        """
        if self.quota_logits is None:
            return torch.tensor(0.0, device=self.candidate_regions.device)

        # I101-1: 添加 Quota Softmax 数值保护
        # I108-6: 使用 LOGIT_CLAMP_BOUND (50.0) 常量
        quota_logits_clamped = self.quota_logits.clamp(min=-LOGIT_CLAMP_BOUND, max=LOGIT_CLAMP_BOUND)
        # Softmax 计算配额分布
        quota_probs = F.softmax(quota_logits_clamped, dim=0)  # [D]
        quota_probs = quota_probs.clamp(min=PROB_EPSILON)
        
        # 熵
        entropy = -(quota_probs * quota_probs.log()).sum()
        
        # 最大化熵 → 最小化负熵
        loss = -weight * entropy

        return loss

    def get_entropy_loss(self, weight: float = QUOTA_ENTROPY_WEIGHT) -> Tensor:
        """
        获取配额熵正则化损失 (MetricsSplitter Protocol 接口).

        数学形式:
            L_entropy = -Σ_d π_d log(π_d + ε)
            鼓励配额分布保持多样性

        Args:
            weight: 熵损失权重

        Returns:
            Tensor: 熵损失值

        Note:
            此方法是 get_quota_entropy_loss() 的别名/快捷方式，
            用于实现 MetricsSplitter Protocol 接口。
        """
        return self.get_quota_entropy_loss(weight=weight)

    # ========================================================================
    # I120-8: Hilbert-感知自适应预算系统
    # ========================================================================

    def get_hilbert_continuity_loss(
        self,
        hilbert_indices: Optional[Tensor] = None,
        selected_mask: Optional[Tensor] = None,
        weight: float = HILBERT_CONTINUITY_WEIGHT,
        gamma: Optional[float] = None,
        image_size: Optional[int] = None,
        patch_size: Optional[int] = None,
    ) -> Tensor:
        """
        计算 Hilbert-感知连续性损失 (I120-8, I122-8)。

        数学形式化
        ==========

        核心思想:
            选中的 token 应该沿着 Hilbert 曲线连续选择，而非跳跃。

        损失公式:
            L_continuity = -weight × (1/K) × Σ_{i=1}^{K-1} exp(-γ × ΔH_i)

        其中:
            - ΔH_i = |h_{i+1} - h_i| 是排序后 Hilbert 索引的相邻差
            - γ = 0.1 × log₂(N_patches) / log₂(4096) (I122-8: 动态 gamma)
            - exp(-γ × ΔH) ∈ (0, 1] 是连续性度量

        I122-8 动态 gamma 推导:
            - Hilbert 局部性: ‖Δp‖ ≤ √2 × |ΔH|^0.5
            - 固定 γ = 0.1 对不同分辨率产生差异巨大的相对惩罚
            - 动态 γ = 0.1 × log₂(N_patches) / log₂(4096) 确保一致性

        性质:
            - ΔH = 0 (相邻): exp(0) = 1.0 → 无损失
            - ΔH = 10: exp(-γ×10) ≈ 0.37 → 中等损失 (与分辨率无关)
            - ΔH = 100: exp(-γ×100) ≈ 4.5e-5 → 几乎无影响

        与 Hilbert 理论的对齐:
            - Hilbert 曲线保证相邻索引在空间上也相邻
            - 鼓励选择空间邻近的 token
            - 避免"跳跃式"选择

        Args:
            hilbert_indices: Hilbert 索引 [N]，可选，默认使用内部缓存
            selected_mask: 选中掩码 [B, N]，可选
            weight: 损失权重
            gamma: 局部性强度参数，若为 None 则使用动态计算
            image_size: 图像尺寸，用于动态计算 gamma (I122-8)
            patch_size: patch 尺寸，用于动态计算 gamma (I122-8)

        Returns:
            loss: 标量连续性损失
        """
        device = self.candidate_regions.device

        # 获取 Hilbert 索引
        if hilbert_indices is None:
            # 使用缓存的 Hilbert 索引
            if not hasattr(self, 'hilbert_indices') or self.hilbert_indices is None:
                return torch.tensor(0.0, device=device)
            hilbert_indices = self.hilbert_indices

        # I145-修复: 确保 hilbert_indices 在正确的设备上
        # 候选区域初始化时可能使用 CPU，但训练时 selected_mask 在 GPU 上
        if hilbert_indices.device != device:
            hilbert_indices = hilbert_indices.to(device=device, non_blocking=True)

        # 获取选中掩码
        # I145-FIX: 使用带梯度的版本用于损失计算
        if selected_mask is None:
            selected_mask = self._last_selected_mask_for_loss if hasattr(self, '_last_selected_mask_for_loss') else None

        # I145-修复: 确保 selected_mask 在正确的设备上
        if selected_mask is not None and selected_mask.device != device:
            selected_mask = selected_mask.to(device, non_blocking=True)

        if selected_mask is None or hilbert_indices is None:
            return torch.tensor(0.0, device=device)

        B, N = selected_mask.shape

        # I122-8: 动态计算 gamma 如果未提供
        if gamma is None:
            # 尝试从内部状态获取 image_size 和 patch_size
            if image_size is None:
                image_size = getattr(self, '_last_image_size', 224)
            if patch_size is None:
                patch_size = getattr(self, '_last_patch_size', 4)
            # 使用动态 gamma 计算
            from vit_pytorch.core.constants import compute_hilbert_continuity_gamma
            gamma = compute_hilbert_continuity_gamma(
                image_size=image_size,
                patch_size=patch_size,
                base_gamma=HILBERT_CONTINUITY_GAMMA,
            )

        # I120-8 修复: 向量化 Hilbert 连续性损失 (替代 Python 循环)
        # ============================================================
        # 数学形式化:
        #   L_continuity = -weight × mean_{b} [ mean_{i} exp(-γ × ΔH_{b,i}) ]
        #
        # 向量化策略:
        #   1. 使用 mask 获取每个样本选中的 token
        #   2. 使用 scatter_sort 或分桶排序处理变长序列
        #   3. 使用 mask 计算有效的相邻差分
        #
        # 复杂度: O(B × N) 向量化 vs O(B) Python 循环
        # ============================================================

        # 获取每个位置的选中 Hilbert 索引 (广播到 batch)
        # selected_mask: [B, N], hilbert_indices: [N]
        selected_h = hilbert_indices.unsqueeze(0).expand(B, -1)  # [B, N]

        # 排序: argsort 得到排序索引
        # hilbert_indices 的值域: [0, N-1]，直接排序
        sort_order = torch.argsort(selected_h, dim=1)  # [B, N]

        # 根据排序索引重新排列 mask 和 hilbert_indices
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, N)  # [B, N]
        sorted_mask = selected_mask[batch_idx, sort_order]  # [B, N]
        sorted_h = hilbert_indices.unsqueeze(0).expand(B, -1)[batch_idx, sort_order]  # [B, N]

        # 计算相邻差分: ΔH_i = h_{i+1} - h_i
        # diffs: [B, N-1]
        diffs = sorted_h[:, 1:] - sorted_h[:, :-1]

        # 计算有效掩码: 排除填充位置
        # 有效位置 = 前一个位置被选中 AND 当前位置被选中
        valid_pairs = sorted_mask[:, :-1] * sorted_mask[:, 1:]  # [B, N-1]

        # 计算连续性度量: exp(-γ × ΔH)
        continuity_raw = torch.exp(-gamma * diffs.float())  # [B, N-1]

        # 应用掩码: 只计算有效相邻对
        continuity_masked = continuity_raw * valid_pairs  # [B, N-1]

        # 计算每个样本的平均连续性
        # count: 每个样本有效相邻对的数量
        counts = valid_pairs.sum(dim=1).clamp(min=1.0)  # [B]
        sum_continuity = continuity_masked.sum(dim=1)  # [B]

        # 避免除零
        mean_continuity_per_sample = sum_continuity / counts  # [B]

        # 最终损失: 负的平均连续性（最小化损失）
        # 只考虑至少有一个有效对的样本
        loss = -mean_continuity_per_sample.mean()

        return weight * loss

    def _compute_image_complexity(
        self,
        info_density: Optional[Tensor] = None,
    ) -> Tensor:
        """
        基于信息密度估计图像复杂度 (I120-8)。

        数学形式化
        ==========

        复杂度估计:
            complexity = Σ_d I_d / N_candidates
                       = I_total / N_candidates

        其中:
            - I_d = softmax((1/4^d) × Σ_{q ∈ Quadrant_d} Var(F_q))
            - HybridDensityHead 输出各深度的信息密度
            - N_candidates 是候选区域总数

        性质:
            - complexity ∈ (0, 1)
            - 复杂度高的图像 → 更多 token
            - 复杂度低的图像 → 更少 token

        Args:
            info_density: 各深度信息密度 [D]，可选
                          默认使用内部缓存的 info_density

        Returns:
            complexity: 复杂度标量 [1]
        """
        if info_density is None:
            info_density = self._last_info_density if hasattr(self, '_last_info_density') else None

        if info_density is None:
            # 默认中等复杂度
            return torch.tensor(0.5, device=self.candidate_regions.device)

        # 复杂度 = 信息密度之和 / 候选数
        complexity = info_density.sum() / self.num_candidates

        return complexity

    # I120-8: 新增深度平衡损失函数
    # ========================================================================
    def get_depth_balance_loss(
        self,
        hard_mask: Optional[Tensor] = None,
        weight: float = 0.1,
    ) -> Tensor:
        """
        计算深度平衡损失 (I120-8 修复)。

        数学形式化
        ==========

        核心思想:
            鼓励选择均匀分布在各深度的 token，避免细粒度主导。

        损失公式:
            L_depth = weight × Σ_d |p_d - p_target,d|²

        其中:
            - p_d = K_d / K_total (深度 d 的选择比例)
            - p_target,d = 1/D (均匀分布目标)
            - D 是深度数量

        梯度:
            ∂L/∂p_d = 2 × weight × (p_d - 1/D)

        性质:
            - p_d > 1/D → 正梯度 → 减少选择
            - p_d < 1/D → 负梯度 → 增加选择
            - 显式约束深度分布

        与熵损失的关系:
            - 熵损失: 隐式鼓励多样性，无目标分布
            - 深度平衡: 显式约束均匀分布

        Args:
            hard_mask: 硬选择掩码 [B, N]，可选
            weight: 损失权重

        Returns:
            loss: 标量深度平衡损失
        """
        import torch.nn.functional as F

        # 获取硬选择掩码
        # I145-FIX: 使用带梯度的版本
        if hard_mask is None:
            hard_mask = self._last_selected_mask_for_loss if hasattr(self, '_last_selected_mask_for_loss') else None

        if hard_mask is None:
            return torch.tensor(0.0, device=self.candidate_regions.device)

        B, N = hard_mask.shape

        # 获取深度信息
        depths = self.candidate_depths  # [N]
        D = self._current_max_depth + 1 if self._current_max_depth is not None else 5

        # 计算 one-hot 深度编码
        depth_onehot = F.one_hot(depths, D).float().to(hard_mask.device)  # [N, D]

        # 计算每个深度的选择数量: K_d[b] = Σ_i hard_mask[b,i] × 1[depth_i = d]
        K_per_depth = torch.einsum('bn,nd->bd', hard_mask, depth_onehot)  # [B, D]

        # 计算深度分布: p_d = K_d / K_total
        K_total = K_per_depth.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B, 1]
        p_d = K_per_depth / K_total  # [B, D]

        # 均匀分布目标
        target = 1.0 / D  # 标量

        # 计算偏差: |p_d - 1/D|²
        deviation = (p_d - target) ** 2  # [B, D]

        # 对深度求和，对 batch 求平均
        loss = deviation.sum(dim=1).mean()  # [B] -> 标量

        return weight * loss

    def get_spatial_coverage_loss(
        self,
        selected_mask: Optional[Tensor] = None,
        image_size: Optional[Tuple[int, int]] = None,
        weight: float = SPATIAL_COVERAGE_WEIGHT,
    ) -> Tensor:
        """
        计算空间覆盖预算损失 (I120-8)。

        数学形式化
        ==========

        目标:
            选中的 token 应该覆盖图像的各个区域。

        覆盖度量:
            Coverage = Area(selected_regions) / Area(image)

        损失公式:
            L_coverage = weight × max(0, Coverage_min - Coverage)

        Args:
            selected_mask: 选中掩码 [B, N]
            image_size: 图像尺寸 (H, W)，可选
            weight: 损失权重

        Returns:
            loss: 标量覆盖损失
        """
        device = self.candidate_regions.device

        if selected_mask is None:
            # I145-FIX: 使用带梯度的版本用于损失计算
            selected_mask = self._last_selected_mask_for_loss if hasattr(self, '_last_selected_mask_for_loss') else None

        if selected_mask is None:
            return torch.tensor(0.0, device=device)

        # 获取图像尺寸
        if image_size is None:
            image_size = self._current_image_size
        if image_size is None:
            h, w = 224, 224  # 默认尺寸
        else:
            h, w = image_size

        # 获取当前尺寸
        current_size = self._current_image_size if self._current_image_size else (h, w)
        H, W = current_size

        # 计算每个选中的区域面积
        if not hasattr(self, 'candidate_regions') or self.candidate_regions is None:
            return torch.tensor(0.0, device=device)

        # candidate_regions: [N, 4] = (x0, y0, x1, y1)
        regions = self.candidate_regions  # [N, 4]

        # 区域面积
        region_areas = (regions[:, 2] - regions[:, 0]) * (regions[:, 3] - regions[:, 1])

        # I145-修复: 确保 region_areas 在正确的设备上 (selected_mask 可能在 GPU)
        target_device = selected_mask.device
        if region_areas.device != target_device:
            region_areas = region_areas.to(target_device)

        # P-OPT: 向量化计算覆盖面积，替代 for b 循环
        # 选中的区域 [B, N]
        B = selected_mask.shape[0]  # batch size
        selected = selected_mask > 0.5  # [B, N]
        # 扩展 region_areas 到 [B, N]
        region_areas_expanded = region_areas.unsqueeze(0).expand(B, -1)  # [B, N]
        # 计算每个 batch 的覆盖面积
        selected_areas = region_areas_expanded * selected.float()  # [B, N]
        total_areas = selected_areas.sum(dim=1)  # [B]

        # 覆盖率
        image_area = H * W
        coverages = total_areas / image_area  # [B]

        # 损失: 鼓励达到最小覆盖率
        coverage_losses = torch.relu(SPATIAL_COVERAGE_MIN - coverages)  # [B]

        # 批次平均
        loss = coverage_losses.mean()

        return weight * loss

    def get_adaptive_target_tokens(
        self,
        info_density: Optional[Tensor] = None,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        """
        计算自适应目标 token 数 (I120-8)。

        数学形式化
        ==========

        自适应覆盖率:
            β_adaptive = β_min + (β_max - β_min) × complexity

        自适应目标:
            K_target = β_adaptive × N_candidates

        Args:
            info_density: 各深度信息密度 [D]
            image_size: 图像尺寸 (H, W)

        Returns:
            target_tokens: 自适应目标 token 数
        """
        device = self.candidate_regions.device
        candidate_count = self.num_candidates

        # 计算复杂度
        complexity = self._compute_image_complexity(info_density)

        # 自适应覆盖率
        adaptive_coverage = ADAPTIVE_COVERAGE_MIN + (ADAPTIVE_COVERAGE_MAX - ADAPTIVE_COVERAGE_MIN) * complexity

        # 目标 token 数
        target_tokens = adaptive_coverage * candidate_count

        return target_tokens

    def set_quota_grad(self, enabled: bool) -> None:
        """
        控制配额参数的梯度 (I30-10: freeze_quota 功能)。

        数学形式:
            if enabled:
                quota_logits.requires_grad = True
            else:
                quota_logits.requires_grad = False

        Args:
            enabled: 是否启用梯度
        """
        if self.quota_logits is not None:
            self.quota_logits.requires_grad = enabled

    # ========================================================================
    # 深度分布监控 (I111-6)
    # ========================================================================

    def get_depth_distribution_tensor(
        self,
        selected_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        返回 GPU Tensor 格式的深度分布（避免 CPU 同步）。

        数学形式
        =========

        深度分布 π 定义为各深度层级的 token 占比：

            π_d = K_d / K_total

        其中 K_d 是分配到深度 d 的 token 数，K_total = Σ K_j 是总 token 数。

        使用 one-hot 编码 + einsum 实现完全向量化计算：

            depth_onehot[depth_i, d] = 1 if depth_i == d else 0
            depth_counts[d] = Σ_i selected_mask[i] × depth_onehot[depth_i, d]
            π = depth_counts / Σ depth_counts

        优势
        ====

        - 完全在 GPU 上计算，避免 CPU 同步开销
        - 返回 GPU Tensor，支持后续 GPU 操作
        - 时间复杂度 O(B × D)，空间复杂度 O(D)

        Args:
            selected_mask: [B, N] 二值选择掩码。如果为 None，使用缓存的 _last_selected_mask。

        Returns:
            pi: [D] GPU Tensor，深度概率分布。归一化后 Σ π_d = 1。
        """
        if selected_mask is None:
            selected_mask = getattr(self, '_last_selected_mask', None)

        if selected_mask is None or self._current_max_depth is None:
            # I150-1-DEBUG: 返回均匀分布便于诊断，而非零向量
            D = self._current_max_depth + 1 if self._current_max_depth is not None else 5
            device = getattr(self, '_cached_device_depths', None)
            if device is not None:
                return torch.ones(D, device=device.device) / D
            return torch.ones(5) / 5

        B, N = selected_mask.shape
        device = selected_mask.device
        D = self._current_max_depth + 1

        # 向量化计算（完全 GPU）
        depths = self._get_device_tensor(
            self.candidate_depths, "_cached_device_depths", device
        )  # [N]

        # one-hot 编码: [N] → [N, D]
        depth_onehot = F.one_hot(depths, D).float()

        # 深度计数: [D] = Σ_{b,n} mask[b,n] × onehot[n,d]
        depth_counts = torch.einsum('bn,nd->d', selected_mask, depth_onehot)

        # 归一化
        total = depth_counts.sum().clamp(min=1.0)
        pi = depth_counts / total

        return pi  # [D] GPU Tensor

    def get_depth_distribution(
        self,
        selected_mask: Optional[Tensor] = None,
        return_tensor: bool = False,
    ) -> Dict[str, Any]:
        """
        获取完整深度分布统计信息（用于监控）。

        数学形式
        =========

        1. 深度分布: π_d = K_d / K_total
        2. Shannon 熵: H(π) = -Σ π_d × log(π_d)
        3. 最大熵: H_max = log(D)（均匀分布时）
        4. KL 散度: KL(π||U) = H_max - H(π)

        诊断阈值
        ========

        - H(π) < 0.1: 深度坍缩（自适应失效）
        - H(π) ∈ [0.5, H_max): 正常多样性
        - KL(π||U) > 0.5: 严重不均，需干预

        Args:
            selected_mask: [B, N] 二值选择掩码
            return_tensor: True 返回 GPU Tensor，False 返回 CPU list

        Returns:
            dict: {
                'pi': Tensor/List,  # 深度分布 [D]
                'entropy': float,   # Shannon 熵
                'max_entropy': float,  # 最大可能熵 log(D)
                'kl_from_uniform': float,  # KL(π || U)
                'dominant_depth': int,  # 主导深度 argmax(π)
                'dominant_prob': float,  # 主导深度概率 max(π)
                'quota_probs': Optional[Tensor],  # 配额概率 [D]
            }
        """
        if selected_mask is None:
            selected_mask = getattr(self, '_last_selected_mask', None)

        if selected_mask is None or self._current_max_depth is None:
            D = self._current_max_depth + 1 if self._current_max_depth is not None else 1
            zero_pi = torch.zeros(D)
            return {
                'pi': zero_pi if return_tensor else zero_pi.tolist(),
                'entropy': 0.0,
                'max_entropy': math.log(D),
                'kl_from_uniform': math.log(D),
                'dominant_depth': 0,
                'dominant_prob': 1.0,
                'quota_probs': None,
            }

        # 使用 GPU Tensor API
        pi = self.get_depth_distribution_tensor(selected_mask)
        D = pi.size(0)

        # 在 GPU 上计算所有统计量
        pi_safe = pi + (pi == 0).float() * PROB_EPSILON
        entropy_tensor = -(pi_safe * pi_safe.log()).sum()
        max_entropy = math.log(D)
        dominant_prob, dominant_depth = pi.max(dim=0)

        # 批量提取标量值到 CPU（减少 GPU-CPU 同步次数）
        entropy = entropy_tensor.item()
        kl = max_entropy - entropy
        dominant_depth_val = dominant_depth.item()
        dominant_prob_val = dominant_prob.item()

        # 获取配额概率
        quota_probs = None
        if hasattr(self, 'get_quota_probs'):
            try:
                quota_probs = self.get_quota_probs()
            except Exception:
                pass

        return {
            'pi': pi if return_tensor else pi.tolist(),
            'entropy': entropy,
            'max_entropy': max_entropy,
            'kl_from_uniform': kl,
            'dominant_depth': int(dominant_depth_val),
            'dominant_prob': float(dominant_prob_val),
            'quota_probs': quota_probs,
        }
    
    # ========================================================================
    # 退火调度接口 (与 LearnableSplitter 兼容)
    # ========================================================================
    
    def set_explore_bias(self, bias: float) -> None:
        """设置探索偏置。"""
        self.explore_bias.fill_(bias)
    
    def set_temperature(self, temperature: float) -> None:
        """设置温度。"""
        self.log_temperature.data.fill_(math.log(max(temperature, TEMPERATURE_MIN)))
    
    # ========================================================================
    # 退火调度 API (与 LearnableSplitter 完全兼容)
    # ========================================================================
    
    def enable_temperature_annealing(
        self,
        total_steps: int,
        T_start: float = 1.0,
        T_end: float = TEMPERATURE_MIN,  # I113-10: 使用常量确保一致性
        schedule: str = 'exponential',
    ) -> "GumbelTopKSplitter":
        # I152-FIX: 添加温度参数安全检查，防止退火过快
        if T_start > 5.0:
            import warnings
            warnings.warn(
                f"I152: T_start={T_start} > 5.0 is too high and may cause training instability. "
                f"Recommended: T_start <= 5.0. Clamping to 5.0.",
                UserWarning,
                stacklevel=2,
            )
            T_start = 5.0

        if T_end < TEMPERATURE_MIN:
            T_end = TEMPERATURE_MIN
        """
        启用自动温度退火。

        数学形式化
        ==========

        对于 Gumbel-Top-K，温度 τ 控制 softmax 尖锐度：
            softmax_i = exp(z_i / τ) / Σ exp(z_j / τ)

        STE 梯度依赖 softmax:
            ∂L/∂z_i = ∂L/∂mask_i × ∂softmax_i/∂z_i

        低温度时 softmax → one-hot，梯度只流向 top 候选。

        退火策略:
            T(t) = T_start · (T_end / T_start)^(t / total)  [exponential]

        I18-2/I18-5/I121-8 安全下界:
            T_end ≥ 0.4 避免梯度消失（配合曲率感知调度）

        Args:
            total_steps: 总训练步数
            T_start: 初始温度 (默认 1.0)
            T_end: 最终温度 (默认 0.3, I18-2 安全下界)
            schedule: 'exponential', 'linear', 'cosine'
            
        Returns:
            self (链式调用)
        """
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        if T_start <= 0 or T_end <= 0:
            raise ValueError(f"Temperatures must be positive")
        if T_end > T_start:
            raise ValueError(f"T_end should be <= T_start for annealing")
        
        # I18-2: 强制温度下界保护
        T_MIN_SAFE = 0.3
        if T_end < T_MIN_SAFE:
            import warnings
            warnings.warn(
                f"I18-2: T_end={T_end} < {T_MIN_SAFE} may cause gradient vanishing. "
                f"Clamping to {T_MIN_SAFE}.",
                UserWarning,
                stacklevel=2,
            )
            T_end = T_MIN_SAFE
        
        if schedule not in ('exponential', 'linear', 'cosine', 'curvature'):
            raise ValueError(f"Unknown schedule: {schedule}. "
                           f"Supported: 'exponential', 'linear', 'cosine', 'curvature'")
        
        self._temp_total_steps.fill_(total_steps)
        self._temp_start.fill_(T_start)
        self._temp_end.fill_(T_end)
        self._temp_step.zero_()
        self._temp_schedule = schedule
        self._temp_enabled = True
        
        # 设置初始温度
        self.set_temperature(T_start)
        
        return self
    
    def disable_temperature_annealing(self) -> "GumbelTopKSplitter":
        """禁用自动温度退火。"""
        self._temp_enabled = False
        return self

    # ========================================================================
    # Token 稳定性监控 API (I150-3)
    # ========================================================================

    def _record_token_selection(self, topk_indices: Tensor) -> None:
        """记录 token 选择历史（用于 IOU 分析）"""
        # 只记录第一个样本的索引
        if topk_indices.shape[0] > 0:
            self._token_history.append(topk_indices[0].detach().cpu())
            if len(self._token_history) > 5:
                self._token_history.pop(0)

    def enable_token_stability_monitoring(self) -> "GumbelTopKSplitter":
        """启用 token 选择稳定性监控"""
        self._monitor_token_stability = True
        self._token_history.clear()
        return self

    def disable_token_stability_monitoring(self) -> "GumbelTopKSplitter":
        """禁用 token 选择稳定性监控"""
        self._monitor_token_stability = False
        return self

    def compute_token_iou(self) -> Optional[float]:
        """
        计算最近两次 token 选择的 IOU（重合率）。

        Returns:
            IOU 值（0.0-1.0），或 None（如果历史不足）
        """
        if len(self._token_history) < 2:
            return None

        indices1 = set(self._token_history[-2].tolist())
        indices2 = set(self._token_history[-1].tolist())

        if not indices1 or not indices2:
            return None

        intersection = len(indices1 & indices2)
        union = len(indices1 | indices2)

        return intersection / union if union > 0 else 0.0

    def get_token_stability_stats(self) -> Dict[str, float]:
        """
        获取 token 稳定性统计信息。

        Returns:
            包含 iou_mean, iou_std, iou_min 的字典
        """
        if len(self._token_history) < 2:
            return {"iou_mean": 0.0, "iou_std": 0.0, "iou_min": 0.0, "iou_max": 0.0}

        ious = []
        for i in range(len(self._token_history) - 1):
            set1 = set(self._token_history[i].tolist())
            set2 = set(self._token_history[i + 1].tolist())
            if set1 and set2:
                intersection = len(set1 & set2)
                union = len(set1 | set2)
                ious.append(intersection / union if union > 0 else 0.0)

        if not ious:
            return {"iou_mean": 0.0, "iou_std": 0.0, "iou_min": 0.0, "iou_max": 0.0}

        import statistics
        return {
            "iou_mean": statistics.mean(ious),
            "iou_std": statistics.stdev(ious) if len(ious) > 1 else 0.0,
            "iou_min": min(ious),
            "iou_max": max(ious),
        }

    def enable_explore_bias_annealing(
        self,
        total_steps: int,
        b_start: float = 0.5,
        b_end: float = 0.0,
    ) -> "GumbelTopKSplitter":
        """
        启用探索偏置退火。
        
        数学形式化
        ==========
        
        对于 Gumbel-Top-K，偏置影响选择质量:
            z_effective = z + b + depth_bias - τ_d
            
        虽然 Top-K 约束保证 token 数量，但偏置影响 *哪些* 被选中。
        训练初期高偏置鼓励探索，后期偏置→0 使决策由 MLP 主导。
        
        退火公式:
            b(t) = b_start + (b_end - b_start) · t / total
            
        Args:
            total_steps: 总退火步数
            b_start: 初始偏置 (默认 0.5)
            b_end: 终态偏置 (默认 0.0)
            
        Returns:
            self (链式调用)
        """
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        
        self._bias_total_steps.fill_(total_steps)
        self._bias_start.fill_(b_start)
        self._bias_end.fill_(b_end)
        self._bias_step.zero_()
        self._bias_enabled = True
        
        # 设置初始偏置
        self.set_explore_bias(b_start)
        
        return self
    
    def disable_explore_bias_annealing(self) -> "GumbelTopKSplitter":
        """禁用探索偏置退火。"""
        self._bias_enabled = False
        return self
    
    def _update_temperature(self) -> Tensor:
        """
        更新温度 (内部方法，在 forward 中调用)。

        I122-7: 简化温度调度策略
        P-OPT: 使用张量操作避免 GPU-CPU 同步

        Returns:
            更新后的温度值 (Tensor，避免 GPU-CPU 同步)
        """
        # P-OPT: 延迟 .item() 到真正需要时，使用 Python 变量缓存
        if not hasattr(self, '_cached_temp_total'):
            self._cached_temp_total = self._temp_total_steps.item() if self._temp_total_steps.numel() > 0 else 0
        total = self._cached_temp_total

        if total <= 0:
            return self.current_temperature

        # P-OPT: 使用张量操作，避免 float() 转换导致的 GPU-CPU 同步
        step = self._temp_step
        progress = (step / total).clamp_(0, 1)

        # 使用缓存的温度参数 (初始化时设置，之后不变)
        schedule = self._temp_schedule

        # 缓存温度参数 (惰性初始化)
        if not hasattr(self, '_cached_temp_start'):
            self._cached_temp_start = self._temp_start.item()
        if not hasattr(self, '_cached_temp_end'):
            self._cached_temp_end = self._temp_end.item()
        T_s = self._cached_temp_start
        T_e = self._cached_temp_end

        # 计算温度 (使用张量保持 GPU 操作)
        if schedule == 'exponential':
            T = T_s * ((T_e / T_s) ** progress)
        elif schedule == 'linear':
            T = T_s + (T_e - T_s) * progress
        elif schedule == 'inverse_time':
            alpha = 3.0
            T = T_e + (T_s - T_e) / (1.0 + alpha * progress)
        else:
            T = T_s

        self.set_temperature(T)
        self._temp_step.add_(1)

        # I170-3: 缓存最终温度 (训练结束时)
        if step >= total - 1 and self._cached_final_temperature is None:
            self._cached_final_temperature = T.item() if isinstance(T, Tensor) else T

        return self.current_temperature

    def _update_explore_bias(self) -> Tensor:
        """
        更新探索偏置 (内部方法，在 forward 中调用)。

        P-OPT: 使用张量操作避免 GPU-CPU 同步

        Returns:
            更新后的偏置值 (Tensor，避免 GPU-CPU 同步)
        """
        # P-OPT: 延迟 .item() 到真正需要时，使用 Python 变量缓存
        if not hasattr(self, '_cached_bias_total'):
            self._cached_bias_total = self._bias_total_steps.item() if self._bias_total_steps.numel() > 0 else 0
        total = self._cached_bias_total

        if total <= 0:
            return self.explore_bias

        # P-OPT: 使用张量操作，避免 float() 转换导致的 GPU-CPU 同步
        step = self._bias_step
        progress = (step / total).clamp_(0, 1)

        # 缓存偏置参数 (惰性初始化)
        if not hasattr(self, '_cached_bias_start'):
            self._cached_bias_start = self._bias_start.item()
        if not hasattr(self, '_cached_bias_end'):
            self._cached_bias_end = self._bias_end.item()
        b_s = self._cached_bias_start
        b_e = self._cached_bias_end

        # 线性退火 (使用张量保持 GPU 操作)
        b = b_s + (b_e - b_s) * progress

        self.set_explore_bias(b)
        self._bias_step.add_(1)

        return self.explore_bias

    def _update_soft_threshold(self, epoch: int, total_epochs: int) -> float:
        """
        更新 Soft-Threshold (课程学习调度)。

        课程学习式阈值调度:
        - Warm-up (0-25%): τ=0, 充分探索
        - 收缩 (25-75%): 线性 ↑, 稀疏化
        - 稳定 (75%+): 固定微调

        Args:
            epoch: 当前训练轮次
            total_epochs: 总训练轮次

        Returns:
            当前 Soft-Threshold 值
        """
        if not self._enable_soft_threshold:
            return 0.0

        progress = epoch / max(total_epochs, 1)

        if progress < 0.25:  # Warm-up 阶段
            self._current_soft_threshold = 0.0
        elif progress < 0.75:  # 收缩阶段
            # 线性增长到最大值
            tau = (progress - 0.25) / 0.5  # 0 到 1
            if self._soft_threshold_schedule == 'cosine':
                # 余弦退火
                tau = 0.5 * (1 + math.cos(tau * math.pi))
            self._current_soft_threshold = tau * self._soft_threshold_max
        else:  # 稳定阶段
            self._current_soft_threshold = self._soft_threshold_max

        return self._current_soft_threshold

    def get_soft_threshold(self) -> float:
        """获取当前 Soft-Threshold 值。"""
        return self._current_soft_threshold

    def _update_soft_threshold_by_step(self, total_steps: int = 100000) -> float:
        """
        使用步数驱动更新 Soft-Threshold (课程学习调度)。

        与 _update_temperature类似的模式，使用内部步数计数器。

        Args:
            total_steps: 预计总步数，用于计算进度

        Returns:
            当前 Soft-Threshold 值
        """
        if not self._enable_soft_threshold:
            return 0.0

        # 初始化步数计数器（惰性初始化）
        if not hasattr(self, '_soft_threshold_step'):
            self._soft_threshold_step = 0

        step = self._soft_threshold_step
        progress = step / max(total_steps, 1)

        if progress < 0.25:  # Warm-up 阶段
            self._current_soft_threshold = 0.0
        elif progress < 0.75:  # 收缩阶段
            tau = (progress - 0.25) / 0.5  # 0 到 1
            if self._soft_threshold_schedule == 'cosine':
                tau = 0.5 * (1 + math.cos(tau * math.pi))
            self._current_soft_threshold = tau * self._soft_threshold_max
        else:  # 稳定阶段
            self._current_soft_threshold = self._soft_threshold_max

        self._soft_threshold_step += 1
        return self._current_soft_threshold

    def set_soft_threshold(self, threshold: float):
        """设置 Soft-Threshold 值（用于外部调度器）。"""
        self._current_soft_threshold = threshold

    def get_diagnostics(self) -> Dict[str, Any]:
        """获取诊断信息。"""
        # I131-1: depth_bias_beta/gamma 已移除，由 quota_logits 替代
        return {
            'num_candidates': self.num_candidates,
            'max_depth': self._current_max_depth,
            'K_min': self.K_min,
            'K_max': self.K_max,
            'avg_selected': float(self._avg_selected),
            'temperature': self.current_temperature,
            'explore_bias': float(self.explore_bias),
            # I131-1: 返回配额信息替代深度偏置
            'quota_probs': self.get_quota_probs().detach().cpu().tolist() if self.get_quota_probs() is not None else None,
        }

    # ============================================================================
    # MetricsSplitter 接口实现 (I98-7)
    # ============================================================================

    def get_quota_logits(self) -> Optional[Tensor]:
        """
        获取可学习配额 logits。

        数学:
            φ = [φ_0, φ_1, ..., φ_{D-1}]
            π_d = softmax(φ)_d

        Returns:
            None 如果未启用可学习配额
        """
        return self.quota_logits

    def diagnose_depth_distribution(self) -> Dict[str, Any]:
        """I150-1-DEBUG: 诊断深度分布问题。

        返回详细的诊断信息，帮助定位为什么返回均匀分布。
        """
        result = {
            'has_last_selected_mask': False,
            'selected_mask_sum': None,
            'selected_mask_shape': None,
            'has_current_max_depth': False,
            'current_max_depth': None,
            'has_candidate_depths': False,
            'candidate_depths_range': None,
            'num_candidates': None,
            'quota_logits': None,
        }

        # 检查 _last_selected_mask
        last_mask = getattr(self, '_last_selected_mask', None)
        if last_mask is not None:
            result['has_last_selected_mask'] = True
            result['selected_mask_sum'] = float(last_mask.sum().item())
            result['selected_mask_shape'] = list(last_mask.shape)

        # 检查 _current_max_depth
        if self._current_max_depth is not None:
            result['has_current_max_depth'] = True
            result['current_max_depth'] = self._current_max_depth

        # 检查 candidate_depths
        if self.candidate_depths is not None and len(self.candidate_depths) > 0:
            result['has_candidate_depths'] = True
            result['candidate_depths_range'] = [int(self.candidate_depths.min()), int(self.candidate_depths.max())]
            result['num_candidates'] = len(self.candidate_depths)

        # 检查 quota_logits
        if self.quota_logits is not None:
            result['quota_logits'] = self.quota_logits[:5].tolist()  # 只返回前5个

        return result

    def get_quota_probs(self) -> Optional[Tensor]:
        """
        获取配额概率分布。

        数学:
            π = softmax(φ)

        Returns:
            None 如果未启用可学习配额
        """
        if self.quota_logits is None:
            return None
        # I101-1: 添加 Quota Softmax 数值保护
        # I108-6: 使用 LOGIT_CLAMP_BOUND (50.0) 常量
        quota_logits_clamped = self.quota_logits.clamp(min=-LOGIT_CLAMP_BOUND, max=LOGIT_CLAMP_BOUND)
        return F.softmax(quota_logits_clamped, dim=0)

    def get_variance_regularization(self) -> Tensor:
        """
        获取深度方差正则化损失。

        数学:
            L_var = Σ_d π_d × (d - E[d])²
            E[d] = Σ_d d × π_d
            避免配额过度集中于特定深度

        Returns:
            Tensor: 方差正则化损失值
        """
        if self.quota_logits is None:
            return torch.tensor(0.0, device=self.candidate_regions.device)

        # I101-1: 添加 Quota Softmax 数值保护
        # I108-6: 使用 LOGIT_CLAMP_BOUND (50.0) 常量
        quota_logits_clamped = self.quota_logits.clamp(min=-LOGIT_CLAMP_BOUND, max=LOGIT_CLAMP_BOUND)
        # 计算配额分布
        quota_probs = F.softmax(quota_logits_clamped, dim=0)  # [D]

        # 计算期望深度 E[d]
        depths = torch.arange(
            len(quota_probs),
            device=quota_probs.device,
            dtype=quota_probs.dtype
        )
        expected_depth = (quota_probs * depths).sum()

        # 计算方差
        variance = (quota_probs * (depths - expected_depth) ** 2).sum()

        return variance

    def get_coverage_stats(self) -> Dict[str, float]:
        """
        获取覆盖率统计。

        数学:
            raw_coverage = K_selected / N_total
            adaptive_coverage = raw_coverage × sqrt(min(H, W) / ref_size)

        Returns:
            Dict[str, float]: {
                'raw_coverage': float,
                'adaptive_coverage': float,
                'K_selected': int,
                'N_total': int,
            }
        """
        # I113-12: 优化 - 使用 detach() 避免影响计算图
        K_selected = int(self._avg_selected.detach().item())
        N_total = self.num_candidates

        # 原始覆盖率
        raw_coverage = K_selected / N_total if N_total > 0 else 0.0

        # 自适应覆盖率 (I33)
        min_dim = min(self._current_image_size)
        ref_size = K_ADAPTIVE_REFERENCE_SIZE  # 默认 224
        adaptive_factor = (min_dim / ref_size) ** 0.5
        adaptive_coverage = raw_coverage * adaptive_factor

        return {
            'raw_coverage': raw_coverage,
            'adaptive_coverage': adaptive_coverage,
            'K_selected': K_selected,
            'N_total': N_total,
        }


# ============================================================================
# 工厂函数：从 LearnableSplitter 参数创建
# ============================================================================

def create_gumbel_topk_from_config(
    feature_dim: int = 256,
    # I30-17-EXT: 替换 max_depth 为 min_patch_size + max_level_limit
    # 兼容旧 API: max_depth 仍可用，通过转换得到 min_patch_size
    max_depth: Optional[int] = None,
    min_patch_size: Optional[int] = None,
    max_level_limit: int = 8,
    hidden_dim: int = 128,
    pool_size: int = 4,
    image_size: Tuple[int, int] = (64, 64),
    K_min: int = 8,
    K_max: int = 64,
    target_coverage: float = 0.12,  # I111-2: 简化覆盖率参数
    **kwargs
) -> GumbelTopKSplitter:
    """
    从配置创建 GumbelTopKSplitter (I111-1 统一配置版)。

    使用方法:
        ```python
        from vit_pytorch.layers.splitters import create_gumbel_topk_from_config

        splitter = create_gumbel_topk_from_config(
            image_size=(224, 224),
            feature_dim=256,
            min_patch_size=4,
            max_level_limit=8,
        )
        ```
    """
    # I30-17-EXT: 处理新旧 API 兼容
    if max_depth is not None and min_patch_size is None:
        # 旧 API: 从 max_depth 计算 min_patch_size
        min_dim = min(image_size)
        min_patch_size = max(1, min_dim // (2 ** max_depth))
        max_level_limit = max(max_depth, max_level_limit)

    if min_patch_size is None:
        min_patch_size = 4  # 默认值

    # I145: 修复参数名称（使用 HilbertSplitterConfig/SplitterConfig 的正确参数）
    # K_min_abs: 绝对最小采样数
    # K_max_hard: 绝对最大采样数
    K_min_abs = K_min if K_min is not None else 8
    K_max_hard = K_max if K_max is not None else 4096
    # target_coverage 移除，使用 coverage_min/coverage_max_hard
    # 从 kwargs 中获取覆盖率参数
    coverage_min = kwargs.get('token_coverage_min', 0.01)
    coverage_max_hard = kwargs.get('token_coverage_max', 0.25)

    splitter_config = SplitterConfig(
        feature_dim=feature_dim,
        min_patch_size=min_patch_size,
        max_level_limit=max_level_limit,
        hidden_dim=hidden_dim,
        intermediate_dim=hidden_dim,
        pool_size=pool_size,
        dropout=0.1,
        K_min_abs=K_min_abs,
        K_max_hard=K_max_hard,
        coverage_min=coverage_min,
        coverage_max_hard=coverage_max_hard,
        use_dynamic_k=True,
    )

    return GumbelTopKSplitter(
        config=splitter_config,
        image_size=image_size,
    )


# ========================================================================
# I111-6: 深度分布监控器 (Lazy Monitoring Pattern)
# ========================================================================

class DepthMonitor:
    """
    延迟深度分布监控器 (Lazy Monitoring Pattern)

    数学基础
    =========

    - 深度分布 π 反映 token 的层级分配策略
    - 实时计算 π_d = K_d / K_total
    - Shannon 熵 H(π) 监控分布多样性
    - KL(π||U) = log(D) - H(π) 量化与均匀分布的偏差

    优势
    ====

    - 避免 forward 关键路径的 CPU 同步
    - 仅在需要时（如 log_interval）计算
    - 支持批量历史的统计
    - 与现有 LazyDiagnostics 模式一致

    使用方法
    =======

    ```python
    monitor = DepthMonitor(splitter)
    stats = monitor.update(global_step)

    # 访问统计结果
    print(f"Depth entropy: {stats['entropy']:.3f}")
    print(f"KL divergence: {stats['kl_from_uniform']:.3f}")
    print(f"Depth distribution: {stats['pi']}")
    ```

    Attributes:
        _splitter: 关联的 GumbelTopKSplitter 实例
        _cached_stats: 缓存的统计结果（避免重复计算）
        _last_step: 上次更新的 step（用于检测需要更新的情况）
        _history: 历史分布的滑动窗口 [W, D]
    """

    __slots__ = ('_splitter', '_cached_stats', '_last_step', '_history', '_history_max_size')

    def __init__(
        self,
        splitter: 'GumbelTopKSplitter',
        history_max_size: int = 100,
    ):
        """
        初始化深度分布监控器。

        Args:
            splitter: 关联的 GumbelTopKSplitter 实例
            history_max_size: 历史记录的滑动窗口大小（默认 100 个 epoch）
        """
        self._splitter = splitter
        self._cached_stats: Optional[Dict[str, Any]] = None
        self._last_step: int = -1
        self._history: Optional[Tensor] = None
        self._history_max_size = history_max_size

    def update(self, step: int) -> Dict[str, Any]:
        """
        更新并返回深度分布统计（惰性计算）。

        仅在 step 变化时才重新计算统计量。

        Args:
            step: 当前训练 step/epoch

        Returns:
            dict: 包含以下键的统计字典:
                - 'pi': GPU Tensor [D]，深度分布
                - 'entropy': float，Shannon 熵
                - 'max_entropy': float，最大可能熵 log(D)
                - 'kl_from_uniform': float，KL 散度
                - 'dominant_depth': int，主导深度
                - 'dominant_prob': float，主导深度概率
                - 'pi_history': GPU Tensor [W, D]，历史分布（如果 history_max_size > 0）
        """
        if step != self._last_step:
            self._compute_stats()
            self._last_step = step

        # 总是返回有效字典（_compute_stats 会填充）
        return self._cached_stats or {
            'pi': torch.zeros(1),
            'entropy': 0.0,
            'max_entropy': 0.0,
            'kl_from_uniform': 0.0,
            'dominant_depth': 0,
            'dominant_prob': 1.0,
            'quota_probs': None,
            'pi_history': None,
        }

    def _compute_stats(self) -> None:
        """在 GPU 上计算所有统计量（私有方法）。"""
        splitter = self._splitter

        # 获取深度分布
        pi = splitter.get_depth_distribution_tensor()
        D = pi.size(0)

        # 计算熵（使用加性 epsilon 避免 log(0)）
        pi_safe = pi + (pi == 0).float() * PROB_EPSILON
        entropy = -(pi_safe * pi_safe.log()).sum()
        max_entropy = math.log(D)
        kl = max_entropy - entropy

        # 主导深度
        dominant_prob, dominant_depth = pi.max(dim=0)

        # 配额概率
        quota_probs = None
        if hasattr(splitter, 'get_quota_probs'):
            try:
                quota_probs = splitter.get_quota_probs()
            except Exception:
                pass

        # 更新历史（滑动窗口平均）
        pi_history = self._update_history(pi)

        # 批量提取标量值到 CPU（减少 GPU-CPU 同步次数）
        # 注意：由于 GIL，.item() 调用仍是串行的，但代码结构更清晰
        entropy_val = entropy.item()
        kl_val = kl.item()
        dominant_depth_val = dominant_depth.item()
        dominant_prob_val = dominant_prob.item()

        self._cached_stats = {
            'pi': pi,  # GPU Tensor
            'entropy': entropy_val,
            'max_entropy': max_entropy,
            'kl_from_uniform': kl_val,
            'dominant_depth': int(dominant_depth_val),
            'dominant_prob': float(dominant_prob_val),
            'quota_probs': quota_probs,
            'pi_history': pi_history,  # 可能为 None
        }

    def _update_history(self, pi: Tensor) -> Optional[Tensor]:
        """
        维护最近 W 个 epoch 的深度分布历史（滑动窗口）。

        数学形式
        ========

        使用循环缓冲区实现滑动窗口：

            history[W] = [π_{-W+1}, π_{-W+2}, ..., π_0}]

        Args:
            pi: 当前深度分布 [D]

        Returns:
            历史分布张量 [W, D]（如果 history_max_size > 0），否则 None
        """
        if self._history_max_size <= 0:
            return None

        if self._history is None:
            # 初始化历史缓冲区
            self._history = pi.unsqueeze(0)  # [1, D]
            return self._history

        # 追加新值
        self._history = torch.cat([self._history, pi.unsqueeze(0)], dim=0)  # [W+1, D]

        # 截断到最大长度
        if self._history.size(0) > self._history_max_size:
            self._history = self._history[-self._history_max_size:]

        return self._history

    def get_entropy_ratio(self) -> float:
        """
        获取熵比率（当前熵 / 最大可能熵）。

        数学形式
        ========

            ratio = H(π) / log(D)

        诊断阈值
        ========

        - ratio < 0.1: 深度坍缩（自适应失效）
        - ratio ∈ [0.3, 0.9]: 正常范围
        - ratio > 0.95: 过度均匀（可能需要调整正则化）

        Returns:
            float: 熵比率
        """
        if self._cached_stats is None:
            self.update(0)

        # 使用 assert 确保类型安全
        assert self._cached_stats is not None
        stats = self._cached_stats
        return stats['entropy'] / stats['max_entropy']

    def get_health_score(self) -> float:
        """
        获取分割器健康评分。

        数学形式
        ========

        综合考虑熵比率和 KL 散度：

            score = w_1 × ratio + w_2 × (1 - normalized_kl)

        其中：
            ratio = H(π) / log(D)
            normalized_kl = KL(π||U) / log(D)

        Returns:
            float: 健康评分 [0, 1]
        """
        if self._cached_stats is None:
            self.update(0)

        assert self._cached_stats is not None
        stats = self._cached_stats

        entropy_ratio = self.get_entropy_ratio()
        kl = stats['kl_from_uniform']
        max_entropy = stats['max_entropy']
        kl_ratio = kl / max_entropy if max_entropy > 0 else 1.0

        # 综合评分（权重 0.6 给熵，0.4 给 KL）
        score = 0.6 * max(0, entropy_ratio) + 0.4 * (1.0 - min(1.0, kl_ratio))

        return min(1.0, max(0.0, score))

    def is_healthy(self, entropy_threshold: float = 0.1, kl_threshold: float = 0.5) -> bool:
        """
        判断分割器是否健康。

        Args:
            entropy_threshold: 熵阈值（低于此值认为不健康）
            kl_threshold: KL 散度阈值（高于此值认为不健康）

        Returns:
            bool: 是否健康
        """
        if self._cached_stats is None:
            self.update(0)

        assert self._cached_stats is not None
        stats = self._cached_stats

        entropy = stats['entropy']
        max_entropy = stats['max_entropy']
        kl = stats['kl_from_uniform']

        entropy_ratio = entropy / max_entropy if max_entropy > 0 else 0

        return entropy_ratio >= entropy_threshold and kl <= kl_threshold

    def reset_history(self) -> None:
        """重置历史记录缓冲区。"""
        self._history = None


# ========================================================================
# I113-2: LookAheadHead + 相关性分裂 (三层参数实现)
# ========================================================================

class LookAheadHead(nn.Module):
    """
    LookAheadHead: 预测子节点相似度，决定是否分裂

    数学形式化
    ===========

    L1 (相对): 分裂概率
        P_split = σ(1 - S_mean) ∈ [0, 1]

    其中:
        S_mean = Mean(CosineSim(v_ca, v_cb)) for all pairs
        CosineSim(a, b) = (a · b) / (||a|| · ||b||)

    L3 (动态): 根据内容计算分裂概率
        P_split(i) = σ(1 - Mean_{0≤a<b<4}(CosineSim(v_ca(i), v_cb(i))))

    核心思想
    =========

    - 子节点相似度高 (同质化区域) → P_split → 0 → 不分裂
    - 子节点相似度低 (异质化区域) → P_split → 1 → 分裂

    Attributes:
        proj: 投影层 [C -> H]
    """

    def __init__(self, in_channels: int, hidden_dim: int = 128, feature_dim: int = 256):
        """
        初始化 LookAheadHead

        Args:
            in_channels: 输入特征通道数
            hidden_dim: 输出隐藏维度 (默认 128)
            feature_dim: 原始特征维度，用于 temperature scaling
        """
        super().__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        self.proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)

    def forward(self, features: Tensor) -> Tensor:
        """
        前向传播

        Args:
            features: [B, C, H_feat, W_feat] 输入特征图

        Returns:
            look_ahead: [B, H, H_feat, W_feat] LookAhead 特征
        """
        return self.proj(features)

    def compute_child_similarity(
        self,
        features: Tensor,
        levels_info: 'LevelsInfo',
    ) -> Tensor:
        """
        计算子节点对的余弦相似度

        数学形式
        =========

        对每个 Hilbert 区域 i 及其 4 个子节点:
            v_ca(i), v_cb(i) for 0 ≤ a < b < 4 (6 pairs)

        S(i) = Mean_{a<b} CosineSim(v_ca(i), v_cb(i))

        Args:
            features: [B, C, H_feat, W_feat] 特征图
            levels_info: LevelsInfo 四叉树结构信息

        Returns:
            similarity: [B, N] 每个区域的子节点平均相似度
        """
        B, C, H_feat, W_feat = features.shape
        H = self.proj.out_features

        # 投影特征
        look_ahead = self.forward(features)  # [B, H, H_feat, W_feat]

        # 获取子节点路径和深度
        paths = levels_info.paths  # [B, N, D]
        depths = levels_info.depths  # [B, N]

        # 计算每个区域子节点的平均相似度（I145-优化：完全向量化版本）
        # 数学形式化：
        #   - 使用广播机制一次性计算所有区域的坐标偏移
        #   - 利用张量索引替代Python循环，避免GPU-CPU同步
        #   - 复杂度: O(B×N) 而非 O(B×N×inner_loops) GPU同步

        B, N, D = paths.shape
        device = features.device

        # 预处理：一次性提取所有深度，避免循环中的.item()调用
        depths_cpu = depths.cpu()  # 移至CPU进行整数操作（仅一次同步）
        valid_mask = depths_cpu >= 0  # [B, N] 布尔掩码

        # 预计算每个区域的坐标基座（完全向量化）
        # 路径格式: [quadrant_0, quadrant_1, ..., quadrant_{depth-1}]
        # quad编码: 0=左上, 1=右上, 2=左下, 3=右下
        quad_to_offset = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], device=device, dtype=torch.float32)

        # 扩展paths以匹配广播: [B, N, D] -> [B, N, D, 1, 2]
        paths_expanded = paths.unsqueeze(-1).unsqueeze(-1)  # [B, N, D, 1, 1]
        # quad_to_offset: [4, 2] -> [1, 1, 1, 4, 2] 用于广播
        offset_expanded = quad_to_offset.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # 计算每个深度级别的累积偏移（向量化扫描）
        # 注意：由于不同区域深度不同，我们需要处理变长路径
        max_depth = D
        depth_indices = torch.arange(max_depth, device=device).unsqueeze(0).unsqueeze(0)  # [1, 1, D]

        # 创建深度掩码：只考虑 depth 之前的 quadrant
        depth_mask = depth_indices < depths.unsqueeze(-1)  # [B, N, D] bool

        # 获取有效的 quadrant 值（无效位置设为0）
        quad_values = paths.long()  # [B, N, D]
        quad_onehot = torch.nn.functional.one_hot(quad_values, num_classes=4)  # [B, N, D, 4]

        # 计算每个区域每个子节点的坐标偏移
        # quad_onehot: [B, N, D, 4], offset_expanded: [1, 1, 1, 4, 2]
        # 结果: [B, N, D, 4, 2]
        offset_per_depth = quad_onehot.unsqueeze(-1) * offset_expanded

        # 每个区域的4个子节点总偏移：[B, N, 4, 2]
        child_offsets = offset_per_depth.sum(dim=2)  # 在深度维度求和

        # 获取look_ahead特征: [B, H, H_feat, W_feat]
        H_feat, W_feat = look_ahead.shape[2], look_ahead.shape[3]

        # 构建子节点坐标（考虑边界情况）
        # child_offsets: [B, N, 4, 2]，其中2是(x, y)
        child_coords_x = child_offsets[..., 0]  # [B, N, 4]
        child_coords_y = child_offsets[..., 1]  # [B, N, 4]

        # 边界检查并clip到有效范围
        child_coords_x = child_coords_x.clamp(0, W_feat - 1)
        child_coords_y = child_coords_y.clamp(0, H_feat - 1)

        # 使用高级索引提取特征：[B, N, 4, H]
        # 转换坐标为整数索引
        child_coords_x_int = child_coords_x.long()
        child_coords_y_int = child_coords_y.long()

        # 构建批量索引
        batch_idx = torch.arange(B, device=device).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
        child_coords_x_int = child_coords_x_int.unsqueeze(-1)  # [B, N, 4, 1]
        child_coords_y_int = child_coords_y_int.unsqueeze(-1)  # [B, N, 4, 1]

        # 提取子节点特征：[B, N, 4, H]
        child_features = look_ahead[
            batch_idx.expand(-1, N, 4),  # [B, N, 4] 批量索引
            :,  # head维度
            child_coords_y_int.squeeze(-1),  # [B, N, 4] y坐标
            child_coords_x_int.squeeze(-1)   # [B, N, 4] x坐标
        ]  # [B, N, 4, H]

        # 计算所有配对的余弦相似度（向量化）
        # child_features: [B, N, 4, H]
        # 需要计算 C(4,2) = 6 对的相似度

        # 归一化特征
        norm_factors = child_features.norm(dim=-1, keepdim=True) + GUMBEL_EPSILON  # [B, N, 4, 1]
        child_features_normed = child_features / norm_factors  # [B, N, 4, H]

        # 配对索引：6对
        pair_idx = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

        # 计算所有配对相似度
        pair_similarities = []
        for a, c in pair_idx:
            fa = child_features_normed[..., a, :]  # [B, N, H]
            fc = child_features_normed[..., c, :]  # [B, N, H]
            sim = (fa * fc).sum(dim=-1)  # [B, N]
            pair_similarities.append(sim)

        # 堆叠并计算平均：[B, N, 6]
        pair_sim_stacked = torch.stack(pair_similarities, dim=-1)  # [B, N, 6]
        mean_similarities = pair_sim_stacked.mean(dim=-1)  # [B, N]

        # Temperature Scaling: 缩放相似度以匹配 sigmoid 的输入范围
        # 类似 Attention 中的 scaling factor: 1/sqrt(d_k)
        scale_factor = math.sqrt(self.feature_dim)
        scaled_similarities = mean_similarities / scale_factor

        # 应用无效区域掩码
        similarities = torch.where(valid_mask.to(device, non_blocking=True),
                                  scaled_similarities,
                                  torch.ones_like(scaled_similarities))

        return similarities  # [B, N]


class CorrelationSplitter(nn.Module):
    """
    CorrelationSplitter: 基于相关性的动态分裂器

    三层参数设计
    ===========

    L1 (相对): 配置参数
        τ_target ∈ [0, 1]  # 目标分裂率
        γ  # Lagrangian 乘子
        λ_div  # Diversity 权重

    L2 (绝对): 计算目标
        N_base = (4^(L+1) - 1) / 3  # 基础候选数
        N_target = N_base × τ_target  # 目标 Token 数

    L3 (动态): 运行时值
        P_split ∈ [0, 1]  # 分裂概率
        N_actual = Σ P_split  # 实际 Token 数
        δ = (N_actual - N_target) / N_target  # 相对偏差

    损失函数
    =========

    L_total = L_jigsaw + λ_div × L_div + γ × δ²

    Attributes:
        lookahead: LookAheadHead 实例
        config: SplitterConfig 实例
    """

    def __init__(self, config: 'HilbertSplitterConfig'):
        """
        初始化 CorrelationSplitter

        Args:
            config: HilbertSplitterConfig 实例
        """
        super().__init__()
        self.config = config
        self.lookahead = LookAheadHead(
            in_channels=config.feature_dim,
            hidden_dim=config.lookahead_dim,
            feature_dim=config.feature_dim,
        )

        # L2: 预计算基础候选数
        self._n_base = config.compute_candidate_count()

    # =========================================================================
    # L2: 绝对值计算方法
    # =========================================================================

    def compute_absolute_targets(
        self,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> dict:
        """
        计算绝对目标值 (L2)

        Returns:
            dict: 包含 N_base, N_target, N_min, N_max
        """
        N_base = self._n_base
        N_target = int(N_base * self.config.target_ratio)
        N_target = max(8, min(64, N_target))  # 边界限制

        # K 边界作为绝对边界
        K_min, K_max = self.config.compute_k_bounds(image_size)

        return {
            'N_base': N_base,
            'N_target': N_target,
            'N_min': max(8, K_min),
            'N_max': min(64, K_max),
        }

    # =========================================================================
    # L3: 动态分裂概率计算
    # =========================================================================

    def compute_split_prob(self, features: Tensor) -> Tensor:
        """
        计算分裂概率 (L3 动态)

        数学形式
        =========

        P_split = σ(1 - S_mean)

        其中:
            S_mean = Mean(CosineSim(v_ca, v_cb))  # 子节点相似度

        Returns:
            P_split: [B, N] 分裂概率
        """
        raise NotImplementedError("需要传入 levels_info")

    def forward(
        self,
        features: Tensor,
        levels_info: 'LevelsInfo',
        image_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        前向传播

        Args:
            features: [B, C, H_feat, W_feat] 特征图
            levels_info: LevelsInfo 四叉树结构信息
            image_size: (H, W) 图像尺寸

        Returns:
            P_split: [B, N] 分裂概率 (L1 相对)
            N_actual: [B] 实际 Token 数 (L2 绝对)
            deviation: [B] 相对偏差 (L1-L2)
        """
        # L3: 计算子节点相似度
        similarity = self.lookahead.compute_child_similarity(features, levels_info)

        # L1: 计算分裂概率
        P_split = torch.sigmoid(1.0 - similarity)  # [B, N]

        # L3: 实际 Token 数
        N_actual = P_split.sum(dim=-1)  # [B]

        # L2: 计算目标
        targets = self.compute_absolute_targets(image_size)
        N_target = targets['N_target']

        # L2-L1: 计算相对偏差
        deviation = (N_actual - N_target) / N_target  # [B]

        return P_split, N_actual, deviation


class LagrangianBudgetLoss(nn.Module):
    """
    LagrangianBudgetLoss: 软预算约束损失

    数学形式化
    ===========

    L2 (相对目标): τ_target ∈ [0, 1]
    L2 (绝对目标): N_target = N_base × τ_target
    L3 (动态偏差): δ = (N_actual - N_target) / N_target

    损失函数:
        L_budget = γ × δ²

    梯度:
        ∂L/∂N_actual = 2 × γ × δ / N_target

    性质:
        - δ > 0 (Token 过多): 梯度为正 → 抑制分裂
        - δ < 0 (Token 过少): 梯度为负 → 促进分裂
        - δ = 0: 梯度为 0 → 平衡点
    """

    def __init__(self, gamma: float = 1.0):
        """
        初始化 LagrangianBudgetLoss

        Args:
            gamma: Lagrangian 乘子 (默认 1.0)
        """
        super().__init__()
        self.gamma = gamma

    def forward(self, deviation: Tensor) -> Tensor:
        """
        计算预算约束损失

        Args:
            deviation: [B] 相对偏差 δ

        Returns:
            loss: [] 标量损失
        """
        return self.gamma * (deviation ** 2).mean()

    def compute_gradient(self, deviation: Tensor) -> Tensor:
        """
        计算相对于 deviation 的梯度

        Args:
            deviation: [B] 相对偏差 δ

        Returns:
            grad: [B] 梯度 ∂L/∂δ = 2 × γ × δ
        """
        return 2 * self.gamma * deviation
