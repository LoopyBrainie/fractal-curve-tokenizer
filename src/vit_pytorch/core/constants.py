# -*- coding: utf-8 -*-
"""
全局常量定义模块

数学形式化
============

本模块定义项目中所有超参数的默认值。

层级常量:
    L_max ∈ {10, 50}     最大递归深度
    L_extra = 5          深度安全余量

缩放因子:
    λ_hilbert = 0.1      Hilbert 偏置: B_h → λ_hilbert · B_h
    λ_level = 0.05       层级偏置: B_l → λ_level · B_l

初始化:
    σ_emb = 0.02         嵌入层初始化标准差

Logits 裁剪:
    logits = clamp(logits, -10, 10)  防止数值溢出

LearnableSplitter 默认参数 (P11-10/P11-11 修复后):
    T_start = 1.0        Gumbel-Softmax 起始温度
    T_end = 0.5          Gumbel-Softmax 终止温度 (I24-7: 保持探索能力)
    τ_base = 0.0         初始基础阈值 (P11-10: 对称于 z 分布)
    γ = 0.85             阈值衰减因子 (每层)
"""

from __future__ import annotations

# ==================== 初始化相关常量 ====================

#: 嵌入层权重初始化标准差
EMBEDDING_INIT_STD: float = 0.02

# ==================== 注意力缩放常量 ====================

#: Hilbert 路径偏置的缩放因子
#: I113-11: 基础缩放值，有效值需配合 attn_hilbert_bias.py 中的 √d_k 量纲对齐
#: 当前实现: B_hilbert_eff = B_hilbert * HILBERT_BIAS_SCALE * √d_k
#: 初始化: HILBERT_BIAS_SCALE = 1.0 (在 config.py 中可配置)
HILBERT_BIAS_SCALE: float = 1.0

#: 层级偏置的缩放因子
#: I113-11: 与 Hilbert 偏置保持一致的对齐策略
#: 当前实现: B_level_eff = B_level * LEVEL_BIAS_SCALE * √d_k
LEVEL_BIAS_SCALE: float = 1.0

# ==================== Splitter 温度常量 ====================

#: Gumbel-Softmax 起始温度 T_start
SPLITTER_TEMP_START: float = 1.0

#: Gumbel-Softmax 终止温度 T_end
#: I121-3 修复: 从 0.5 降低到 0.3，提升梯度强度和深度多样性
#: I121-8 修复: 保持 0.3 (与 TEMPERATURE_MIN 和 splitter 实现一致)
#: 数学: τ=0.3 时梯度强度 ≈ 3.3 (有效梯度)
#:       配合曲率感知调度，防止低曲率时温度过低导致梯度饱和
#: I113-10 验证: τ ∈ [0.3, 1.0] 确保梯度有效且训练稳定
SPLITTER_TEMP_END: float = 0.3

#: 温度退火调度策略
#: 可选值: 'exponential', 'linear'
#: I122-7 修复: 统一使用 'linear' (cosine 已移除，缺乏理论依据)
#: I122-7 理由: linear 调度具有恒定变化率，行为可预测
#: 数学: T(t) = T_end + (T_start - T_end) × (1 - t/T)
SPLITTER_TEMP_SCHEDULE: str = 'linear'

# ==================== 数值稳定性常量 (I12-7/I112-3) ====================
# 数学分析见: workspace/numerical_constants_analysis.py

# I112-3: 统一的数值稳定性常量
# 数学依据: 1e-6 是 FP16 的安全下界
#   - FP16 最小正规数: ~6e-8
#   - 1e-6 提供 16× 安全余量
#   - 相对于 FP16 精度 (~1e-3)，提供 1000× 余量
# 使用场景: 所有需要 eps 的场景统一使用此常量
EPS: float = 1e-6

#: Gumbel 采样 uniform clamp (FP32 计算中安全边界)
#: 数学: g = -log(-log(u)), u ∈ [ε, 1-ε]
#: I112-3: 更新为 EPS (1e-6)，与 FP16 统一
GUMBEL_EPSILON: float = EPS

#: Log 计算安全 epsilon
#: 用途: log(p + ε) 防止 log(0)
#: I112-3: 更新为 EPS (1e-6)，消除 FP16 下溢风险
LOG_EPSILON: float = EPS

#: 除法安全 epsilon
#: 用途: x / (sum + ε) 防止除零
#: I112-3: 更新为 EPS (1e-6)，统一数值边界
DIVISION_EPSILON: float = EPS

#: 概率下界 (避免 0 概率参与计算)
#: 用途: prob.clamp(min=PROB_EPSILON)
#: I112-3: 更新为 EPS (1e-6)，与 FP16 安全对齐
PROB_EPSILON: float = EPS

#: I102-4: 形状归一化 epsilon (FP16 安全下界)
#: 用途: norm = ||f|| + ε 防止除零
#: I112-3: 更新为 EPS，统一使用
SHAPE_NORM_EPSILON: float = EPS

#: FP16 安全 epsilon (替代原有的 1e-8)
#: 数学依据: FP16 最小正规数 ~6e-5，精度 ~1e-3
#: 使用 1e-6 作为安全下界，避免 FP16 下溢
FP16_SAFE_EPSILON: float = 1e-6  # I122-? 修复: 统一使用 1e-6

#: 温度参数下界 (Gumbel-Softmax/Top-K)
#: 数学分析: T < 0.1 时 softmax 梯度趋近于 0
#: I35 改进: 从 0.1 提升到 0.3，保持更健康的梯度流
#: I121-8 修复: 保持 0.3 (与 hilbert_optimal_splitter.py 实际值一致)
#: I113-10 验证: T=0.3 时 softmax 梯度 ≈ 1/τ = 3.3 (有效梯度)
#: 注意: SPLITTER_TEMP_END 也应保持 0.3 以与 TEMPERATURE_MIN 一致
TEMPERATURE_MIN: float = 0.3

# ==================== I108-6: FP16 Clamp 边界常量 ====================
# 数学分析见: workspace/fp16_clamp_analysis.md

#: Logits clamp 边界 (Attention / Gumbel logits)
#: I147 修复: 从 50.0 降低到 10.0，保持有效梯度
#: 数学依据: softmax(x > 10) ≈ 0.99995（梯度 ≈ 5e-5，仍有效）
#: 问题: softmax(50) ≈ 1.0（梯度 ≈ 1e-18，梯度消失）
#: FP16 安全: 10 << 65504 (~6500× 安全余量）
#: 验证: clamp(-10, 10) 确保数值稳定且保留有效梯度
LOGIT_CLAMP_BOUND: float = 10.0

#: Gradient clamp 边界 (FP16)
#: 数学依据: P(|grad| > 20) ≈ 10^-6 << 4.5% (原 clamp(-10, 10) 的裁剪率)
#: FP16 安全: 20 << 65504 (~3200× 安全余量)
#: 梯度流: 几乎无损 (原 ~4.5% 梯度裁剪 → ~0.0001%)
GRAD_CLAMP_BOUND: float = 20.0

#: Scale clamp 边界 (softplus 输出)
#: 数学依据: softplus(15) ≈ 1.0e-6 (引理 4.2)
#: 验证: 覆盖 99.99% 的 softplus 输出范围
SCALE_CLAMP_BOUND: float = 15.0

# ==================== I121-6: 动态梯度裁剪常量 ====================
# 数学分析: 梯度范数与学习率成正比，动态调整确保训练稳定性
# 公式: clip_norm = GRAD_CLIP_BASE_NORM × (current_lr / base_lr)

#: 动态梯度裁剪基准学习率
#: 数学: 以 3e-4 为基准，对应 GRAD_CLIP_BASE_NORM = 1.0
GRAD_CLIP_BASE_LR: float = 3e-4

#: 动态梯度裁剪基准范数
#: 数学: 对应 base_lr 的标准裁剪阈值
GRAD_CLIP_BASE_NORM: float = 1.0

#: 动态梯度裁剪范数下限 (防止裁剪过松)
GRAD_CLIP_MIN_NORM: float = 0.5

#: 动态梯度裁剪范数上限 (防止裁剪过紧)
GRAD_CLIP_MAX_NORM: float = 2.0

# ==================== 方案 E: 可学习配额常量 (I24-2) ====================
# 数学分析: 解决 Log-Compensation 对 Top-K 理论无效的问题
# 原理: 先按可学习配额分配各深度 K_d，再在深度内 Top-K 选择
# 优势: 
#   1. 配额硬约束保证深度分布
#   2. 可学习比例允许任务自适应
#   3. 深度内竞争保持 token 质量优化

#: 是否启用可学习配额方案 (替代 Log-Compensation)
LEARNABLE_QUOTA_ENABLED: bool = True

QUOTA_ENTROPY_WEIGHT: float = 0.1
# ==================== I33: 相对预算与自适应覆盖率常量 ====================
# 数学分析: 动态深度下 K_max 与 K_min 应随图像尺寸自适应
# 原理:
#   1. 覆盖率 = K / N_candidates 应保持尺度不变性
#   2. β(H, W) = β_0 × sqrt(min(H, W) / 224) 实现线性尺度调整
#   3. 硬上限约束防止大分辨率下的显存溢出

#: 基准覆盖率 (224×224 图像的目标覆盖率)
#: 数学: β_0 = 0.25 表示目标采样 25% 的候选区域
#: I121-1 修复: 原 0.03 → 0.08 保守提升
#: I121-4 修复: 0.08 → 0.12 提升选中率均衡效果
#: I121-7 修复: 0.12 → 0.25 实现 Hilbert 曲线完整遍历
#: K_target = 0.25 × 341 = 85，确保各深度有足够配额
K_COVERAGE_BASE: float = 0.25

#: 最小覆盖率 (防止欠采样)
#: 数学: α = 0.01 保证最小 1% 覆盖率
K_COVERAGE_MIN: float = 0.01

#: 最大覆盖率硬上限 (防止大分辨率下的显存溢出)
#: 数学: β_max = 0.50 防止 K 增长过快
#: I121-1 修复: 0.25 → 0.30 适应 K_COVERAGE_BASE = 0.08
#: I121-7 修复: 0.30 → 0.50 支持完整 Hilbert 遍历
K_COVERAGE_MAX_HARD: float = 0.50

#: 覆盖率自适应参考尺寸
#: 数学: γ = sqrt(min(H, W) / 224) 缩放因子
K_ADAPTIVE_REFERENCE_SIZE: int = 224

#: K_min 绝对下限 (保证最小表达能力)
#: I145-修复: 从 4 提升到 8，与 ModelGene 和训练脚本默认值一致
#: 数学依据: K=8 确保足够的 token 数量以维持 Hilbert 曲线局部性
K_MIN_HARD_LIMIT: int = 8

#: K_max 显存硬上限 (防止 OOM)
#: 数学: K=8192 时 attention 矩阵 ≈ 256MB (batch=8)，可接受
#: I121-7 修复: 4096 → 8192 适应高覆盖率配置
K_MAX_HARD_LIMIT: int = 8192


# ============================================================================
# TIER 2: 变参数计算函数 (Variables)
# ============================================================================
# 这些函数根据传入的参数动态计算变参数。
# 设计原则：计算在模型架构内部进行，而非由训练器/评估器计算。

import math  # noqa: E402
from typing import Optional, Tuple  # noqa: E402


def compute_max_level(image_size: int, min_patch_size: int) -> int:
    """计算四叉树最大深度。

    .. deprecated::
        I145: 此函数已废弃。请使用 `depth_utils.compute_max_depth()`，
        它支持元组形式的 image_size 和 hard_limit 参数。

    数学形式化:
        max_depth = ceil(log2(min(H, W) / min_patch_size))

    推导:
        - 深度 d 的 patch 尺寸 = min_patch_size × 2^d
        - 目标: min_patch_size × 2^max_depth ≈ min(H, W)
        - 解: max_depth ≈ log2(min(H, W) / min_patch_size)

    示例:
        224×224 图像, min_patch_size=4
        -> 224/4 = 56
        -> log2(56) ≈ 5.81
        -> ceil = 6

    Args:
        image_size: 输入图像尺寸（最小边长）
        min_patch_size: 最小 patch 尺寸

    Returns:
        四叉树分区的最大递归深度

    Raises:
        ValueError: 当 image_size 或 min_patch_size 非正数时
    """
    import warnings
    warnings.warn(
        "constants.compute_max_level() is deprecated. Use depth_utils.compute_max_depth() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    if image_size <= 0:
        raise ValueError(f"image_size 必须为正数, 得到 {image_size}")
    if min_patch_size <= 0:
        raise ValueError(f"min_patch_size 必须为正数, 得到 {min_patch_size}")

    ratio = image_size // min_patch_size
    if ratio <= 1:
        return 0

    return math.ceil(math.log2(ratio))


def compute_num_candidates(max_level: int) -> int:
    """计算四叉树候选节点总数。

    .. deprecated::
        I145: 此函数已废弃。请使用 `depth_utils.compute_total_candidates()`。

    数学形式化:
        N_candidates = Σ(4^d), d=0..max_level = (4^(max_level+1) - 1) / 3

    示例:
        max_level=0: N=1
        max_level=1: N=1+4=5
        max_level=2: N=1+4+16=21
        max_level=6: N=5461

    Args:
        max_level: 四叉树最大深度

    Returns:
        候选节点总数
    """
    import warnings
    warnings.warn(
        "constants.compute_num_candidates() is deprecated. Use depth_utils.compute_total_candidates() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    if max_level < 0:
        raise ValueError(f"max_level 必须非负, 得到 {max_level}")
    return (4 ** (max_level + 1) - 1) // 3


def compute_k_bounds(
    max_level: int,
    token_coverage_min: float,
    token_coverage_max: Optional[float],
    image_size: Optional[int] = None,
    target_ratio: float = 0.5,  # I113-2: L1 相对参数
) -> Tuple[int, int]:
    """从覆盖率参数计算 K_min 和 K_max。

    数学形式化:
        N = Σ(4^d), d=0..max_level
        scale = sqrt(min(H, W) / 224)  # 分辨率自适应
        K_min = max(K_MIN_HARD, ceil(N × coverage_min))
        K_max = min(K_MAX_HARD, ceil(N × coverage_max × scale))

    I113-2 三层参数设计:
        L1 相对: target_ratio ∈ [0, 1]
        L2 绝对: N_base = Σ(4^d), N_target = N_base × target_ratio
        L3 动态: 实际 K 值根据内容动态调整

    设计原理:
        - 覆盖率保证跨分辨率的尺度不变性
        - 硬上限约束防止显存溢出
        - scale 因子适应不同分辨率的上下文需求
        - target_ratio 提供归一化的分裂目标

    Args:
        max_level: 四叉树最大深度
        token_coverage_min: 最小覆盖率 α
        token_coverage_max: 最大覆盖率 β (已废弃，使用 target_ratio 替代)
        image_size: 输入图像尺寸（用于 scale 计算）
        target_ratio: L1 相对目标分裂率 [0, 1]

    Returns:
        (K_min, K_max) 元组
    """
    # I113-2: 处理废弃的 token_coverage_max
    if token_coverage_max is None:
        # 使用 target_ratio 计算 K_max
        N = compute_num_candidates(max_level)
        N_target = int(N * target_ratio)
        K_max = max(K_MIN_HARD_LIMIT, min(K_MAX_HARD_LIMIT, N_target))
        K_min = max(K_MIN_HARD_LIMIT, int(math.ceil(N * token_coverage_min)))
        return (K_min, K_max)

    # 计算候选节点总数
    N = compute_num_candidates(max_level)

    # 计算分辨率自适应 scale 因子
    if image_size is not None:
        scale = math.sqrt(image_size / K_ADAPTIVE_REFERENCE_SIZE)
    else:
        scale = 1.0

    # 计算 K_min（确保最小表达能力）
    K_min = max(
        K_MIN_HARD_LIMIT,
        int(math.ceil(N * token_coverage_min))
    )

    # 计算 K_max（确保不超显存）
    K_max = min(
        K_MAX_HARD_LIMIT,
        int(math.ceil(N * token_coverage_max * scale))
    )

    return K_min, K_max


# 向后兼容别名
compute_max_depth = compute_max_level


def clamp_temperature(temperature: float, min_val: float = TEMPERATURE_MIN) -> float:
    """钳制温度参数到安全范围。

    数学形式化:
        T_safe = max(T, TEMPERATURE_MIN)
        softmax 梯度: ∂p/∂z ≈ p × (1-p) / T
        T >= 0.4 确保梯度流健康 (I121-8: τ=0.4 梯度强度 ≈ 2.5)

    Args:
        temperature: 原始温度值
        min_val: 最小安全温度（默认 TEMPERATURE_MIN = 0.4）

    Returns:
        钳制后的安全温度值
    """
    return max(temperature, min_val)
