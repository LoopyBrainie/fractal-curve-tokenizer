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
#: I24-7 分析: 0.3 过低会过早停止探索
#: 改进: 提高到 0.5，保持更长的探索能力
#: 数学: T=0.5 时 softmax 梯度仍有效 (∂p/∂z ≈ 1)
SPLITTER_TEMP_END: float = 0.5

#: 温度退火调度策略
#: 可选值: 'exponential', 'linear', 'cosine'
#: I24-7 改进: 使用 cosine 退火
#: 优势: 开始慢降(保持探索) → 中期快降(高效收敛) → 末期平稳(稳定决策)
#: 数学: T(t) = T_end + (T_start - T_end) * (1 + cos(πt)) / 2
SPLITTER_TEMP_SCHEDULE: str = 'cosine'

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
#: I112-3: 更新为 EPS，统一使用
FP16_SAFE_EPSILON: float = EPS

#: 温度参数下界 (Gumbel-Softmax/Top-K)
#: 数学分析: T < 0.1 时 softmax 梯度趋近于 0
#: I35 改进: 从 0.1 提升到 0.3，保持更健康的梯度流
#: 验证: T=0.3 时 softmax 梯度仍有效 (∂p/∂z ≈ 1/τ)
TEMPERATURE_MIN: float = 0.3

# ==================== I113-5: STE 梯度缩放常量 ====================
# 数学分析: STE 缩放因子 α = (N/K) × σ(log β) × min(τ/τ_ref, 1)

#: STE 缩放参考温度 (用于温度保护机制)
#: 用途: min(τ/τ_ref, 1) 防止低温度时梯度爆炸
#: 数学: τ_ref = 1.0 作为参考点，与 Gumbel 扰动归一化一致
STE_SCALE_TEMP_REF: float = 1.0

#: STE 可学习缩放下界 (防止过度缩放)
#: 用途: σ(log β) ∈ (0, 1)，但通过 clamp 限制有效范围
#: 数学: 防止学习到极端值导致梯度不稳定
STE_SCALE_MIN: float = 0.1

#: STE 可学习缩放上界
STE_SCALE_MAX: float = 2.0

# ==================== I108-6: FP16 Clamp 边界常量 ====================
# 数学分析见: workspace/fp16_clamp_analysis.md

#: Logits clamp 边界 (Attention / Gumbel logits)
#: 数学依据: softmax(x > 50) ≈ one-hot (引理 2.1)
#: FP16 安全: 50 << 65504 (FP16 最大值 ~1300× 安全余量)
#: 验证: clamp(-50, 50) 确保数值稳定且不丢失有效信息
LOGIT_CLAMP_BOUND: float = 50.0

#: Gradient clamp 边界 (FP16)
#: 数学依据: P(|grad| > 20) ≈ 10^-6 << 4.5% (原 clamp(-10, 10) 的裁剪率)
#: FP16 安全: 20 << 65504 (~3200× 安全余量)
#: 梯度流: 几乎无损 (原 ~4.5% 梯度裁剪 → ~0.0001%)
GRAD_CLAMP_BOUND: float = 20.0

#: Scale clamp 边界 (softplus 输出)
#: 数学依据: softplus(15) ≈ 1.0e-6 (引理 4.2)
#: 验证: 覆盖 99.99% 的 softplus 输出范围
SCALE_CLAMP_BOUND: float = 15.0

#: FP16 安全 epsilon (替代原有的 1e-8)
#: 数学依据: FP16 最小正规数 ~6e-5，精度 ~1e-3
#: 使用 1e-6 作为安全下界，避免 FP16 下溢
FP16_SAFE_EPSILON: float = 1e-6

# ==================== 深度平衡常量 (I24-2 方案E) ====================
# 数学分析: 解决深度分布崩溃问题
# 问题: 候选数量不平衡 (d=0:1, d=1:4, d=2:16, d=3:64) 导致 Top-K 偏向 depth=3
# 解决: 方案E (可学习配额 + 分层 Top-K) 通过配额保证深度分布
# I30-4: 已移除 Log-Compensation (被方案E完全替代)

#: I23-1 方案C: 深度方差归一化是否启用
#: 数学: z_i^norm = (z_i - μ_d) / σ_d，使各深度 MLP 输出服从 N(0,1)
#: I24-5 批判分析: 原实现 Batch 统计量在小 batch (B=1) 下方差放大 512×，单样本深度 (depth=0) 完全失效
#: I30-6 修复: 使用 EMA Running Statistics (α=0.1)，有效样本量 10，方差降低 19×
#: 结论: 启用 EMA 归一化，解决小 batch 稳定性问题，与方案E 配额机制协同保证深度平衡
DEPTH_VARIANCE_NORM_ENABLED: bool = True  # I30-6: EMA 方案启用

#: 方差归一化的稳定性 epsilon (更新下界)
#: 用途: EMA 更新时的最小方差下界，允许统计量收敛到真实方差
DEPTH_VARIANCE_NORM_EPS: float = 1e-6

#: I96-1: EMA 初始化下界 (小 batch 保守初始化)
#: 用途: B≤2 时单批次方差估计不可靠，使用保守下界避免极端值
#: 数学: 5个数量级差异 (0.1 vs 1e-6) 确保初始化bias不会持续存在
#: I100-5 批判: B=1 (undefined) 和 B=2 (5124:1 置信区间) 问题严重程度不同，需要分层处理
DEPTH_VARIANCE_INIT_EPS: float = 0.1   # B≥4 时使用 (保留向后兼容)

#: I100-5: 分层小 batch 保守初始化常量
#: 依据卡方分布置信区间分析:
#: - B=1: 样本方差无定义，使用先验 σ²=0.25 (σ=0.5)
#: - B=2: 置信区间 5124:1，需要 4× 缓冲 (0.4)
#: - B=4: 置信区间 130:1，需要 1.5× 缓冲 (0.15)
DEPTH_VARIANCE_INIT_EPS_B1: float = 0.25   # B=1: σ² = 0.25 (σ = 0.5)
DEPTH_VARIANCE_INIT_EPS_B2: float = 0.4    # B=2: 4×0.1 = 0.4
DEPTH_VARIANCE_INIT_EPS_B4: float = 0.15   # B=4: 1.5×0.1 = 0.15

#: EMA 归一化的平滑系数 (I35: 新增)
#: 数学: α = 0.1, 有效样本量 ≈ 10, 方差降低 19× vs per-batch
#: 效果: 稳定小 batch (B=1) 下的方差估计，避免 sqrt(0) NaN
DEPTH_EMA_ALPHA: float = 0.1

#: I96-4: 树一致性软排除边距
#: 用途: 保持最小梯度流 (10%)，避免父节点梯度被完全切断
#: 数学: p_out = p × max(1 - Σ child_signal, ε)，ε = 0.1
#: 梯度分析: ∂p_out/∂p_i ≥ ε > 0，确保父节点有梯度回传
SOFT_EXCLUSION_MARGIN: float = 0.1

# ==================== 方案 E: 可学习配额常量 (I24-2) ====================
# 数学分析: 解决 Log-Compensation 对 Top-K 理论无效的问题
# 原理: 先按可学习配额分配各深度 K_d，再在深度内 Top-K 选择
# 优势: 
#   1. 配额硬约束保证深度分布
#   2. 可学习比例允许任务自适应
#   3. 深度内竞争保持 token 质量优化

#: 是否启用可学习配额方案 (替代 Log-Compensation)
LEARNABLE_QUOTA_ENABLED: bool = True

#: 每个深度的最小配额 (内部使用，I96-7: 推荐使用 QUOTA_MIN_RATIO)
QUOTA_MIN_PER_DEPTH: int = 2

#: I96-7: 最小采样比例 (自适应深度下界软目标)
#: 数学: K_d^min = max(1, ceil(α × N_d))，其中 N_d = 4^d
#: 注意: 仅作为软目标，不硬性约束
#: 优势: 解决深度 0 问题 (N_0=1, K_0^min=1)，跨深度采样比例恒定
#: 量化 (α=0.02): 深度 1 采样 25%，深度 4 采样 ~2.3%
QUOTA_MIN_RATIO: float = 0.02

#: I96-7: 下界软正则化权重
#: 数学: L_min = λ × Σ max(0, K_d^min - K_d)²
#: 用途: 鼓励但不强制深度下界，软约束允许模型学习最优分布
#: 优势: 无约束满足问题，梯度完整，保持 Hilbert 曲线局部性
QUOTA_MIN_LAMBDA: float = 0.1

#: 配额初始化 (对数空间，softmax 后 = DEPTH_QUOTA_TARGET)
#: 计算: φ_d = log(p_d) - mean(log(p))
#: 验证: softmax(-0.495, -0.207, 0.016, 0.486) ≈ (0.15, 0.20, 0.25, 0.40)
QUOTA_INIT_LOGITS: tuple = (-0.495, -0.207, 0.016, 0.486)

#: 配额熵正则化权重 (鼓励分布多样性)
#: 数学: L_entropy = -λ × H(K/K_total)
QUOTA_ENTROPY_WEIGHT: float = 0.1

#: CRIT-6 修复: 配额 STE 梯度损失权重
#: 数学: L_ste = λ × MSE(K_soft, K_hard)
#: 用途: 直接为 quota_logits 提供梯度，修复 floor().long() 断裂的梯度
#: 原理: K_soft = softmax(φ) × K 有完整梯度，K_hard = K_soft.detach().round().long() 无梯度
#:       MSE 损失使 ∂L/∂φ_d = 2λ × (K_soft_d - K_hard_d) × K × ∂p_d/∂φ_d
QUOTA_STE_WEIGHT: float = 0.5

#: I113-7: 信息密度配额损失权重
#: 数学: L_info = MSE(soft_quota, hard_quota) + λ × KL(soft || target)
#: 用途: 为 info_density → quota 分配提供梯度，恢复端到端学习能力
QUOTA_INFO_LAMBDA: float = 0.1

# ==================== I33: 相对预算与自适应覆盖率常量 ====================
# 数学分析: 动态深度下 K_max 与 K_min 应随图像尺寸自适应
# 原理:
#   1. 覆盖率 = K / N_candidates 应保持尺度不变性
#   2. β(H, W) = β_0 × sqrt(min(H, W) / 224) 实现线性尺度调整
#   3. 硬上限约束防止大分辨率下的显存溢出

#: 基准覆盖率 (224×224 图像的目标覆盖率)
#: 数学: β_0 = 0.03 表示目标采样 3% 的候选区域
#: I109-10 修复: 原 0.12 导致 K_target > K_max (10486 > 4096)
#: 修正后 β_0 = 0.03 使 K_target ∈ [K_min, K_max]
K_COVERAGE_BASE: float = 0.03

#: 最小覆盖率 (防止欠采样)
#: 数学: α = 0.01 保证最小 1% 覆盖率
K_COVERAGE_MIN: float = 0.01

#: 最大覆盖率硬上限 (防止大分辨率下的显存溢出)
#: 数学: β_max = 0.25 防止 K 增长过快 (原 0.08 → 0.25)
#: I36 优化: 小图像需更高上限以支持更多 tokens
K_COVERAGE_MAX_HARD: float = 0.25

#: 覆盖率自适应参考尺寸
#: 数学: γ = sqrt(min(H, W) / 224) 缩放因子
K_ADAPTIVE_REFERENCE_SIZE: int = 224

#: K_min 绝对下限 (保证最小表达能力)
K_MIN_HARD_LIMIT: int = 8

#: K_max 显存硬上限 (防止 OOM)
#: 数学: K=4096 时 attention 矩阵 ≈ 64MB (batch=8)，可接受
K_MAX_HARD_LIMIT: int = 4096

#: K_max 采样比例 (K_max = ceil(K_MAX_SAMPLE_RATIO × N))
#: 数学: α = 0.25 表示最多采样 25% 的候选区域 (原 0.05 → 0.25)
#: I36 优化: 提高以支持 64×64 小图像更多 tokens
K_MAX_SAMPLE_RATIO: float = 0.25

#: K_min 采样比例 (K_min = ceil(K_MIN_SAMPLE_RATIO × N))
#: 数学: β = 0.03 表示最少采样 3% 的候选区域 (原 0.005 → 0.03)
#: I36 优化: 提高以保证小图像最小 tokens
K_MIN_SAMPLE_RATIO: float = 0.03

# ==================== I109-4: Elastic Budget 目标导向损失 ====================
# 设计原则: 目标导向损失 + 边界安全网
# 数学形式化:
#   1. 主损失: L = λ_target × (K/N - β_target)²
#   2. 安全网: 边界约束防止超出 K_bounds
#   3. 目标覆盖率 β_target = K_COVERAGE_BASE × √(min(H,W)/224)

#: 崩溃检测阈值 (低于 K_min 一半时触发)
#: 数学: coverage_min = 0.03 低于此值触发崩溃检测
ELASTIC_COVERAGE_MIN: float = 0.03

#: Elastic Budget 目标损失权重
#: 数学: λ = 0.1 使损失量级与其他辅助损失匹配
#: 引导 token 数量趋向最优覆盖率 (β_target = 12% @ 224×224)
ELASTIC_LAMBDA_TARGET: float = 0.1

#: Elastic Budget 边界安全网权重
#: 数学: λ = 0.1 超出 K_bounds 时额外惩罚
#: 与目标损失权重相等，实现对称保护
ELASTIC_LAMBDA_BOUNDARY: float = 0.1

#: 崩溃检测损失权重
#: 数学: λ_collapse = 1.0 确保崩溃时强惩罚
ELASTIC_LAMBDA_COLLAPSE: float = 1.0

# ==================== I24-8: 阈值正则化常量 ====================

#: 阈值方差正则化是否启用
#: 数学: L_threshold = λ × Var(τ)
#: 目的: 防止某个深度阈值极端偏离，导致选择偏差
THRESHOLD_VAR_REG_ENABLED: bool = True

#: 阈值方差正则化权重
#: 设计: 0.3 使梯度量级与其他辅助损失匹配
THRESHOLD_VAR_REG_WEIGHT: float = 0.3

# ==================== I26-1: 重叠惩罚常量 ====================

#: 是否启用父子重叠惩罚 (默认关闭)
#: 数学: L_overlap = λ × Σ(parent_selected × child_selected) / K
#: 用途: 当父子共存导致冗余时启用
OVERLAP_PENALTY_ENABLED: bool = False

#: 重叠惩罚权重
#: 推导: 若典型重叠率 ρ ≈ 0.1，weight = 0.1 使 L ≈ 0.01 (轻量正则)
OVERLAP_PENALTY_WEIGHT: float = 0.1

# ==================== 信息长度相关常量 ====================

#: levels_info 的最小长度
MIN_INFO_LEN: int = 16


# ============================================================================
# TIER 2: 变参数计算函数 (Variables)
# ============================================================================
# 这些函数根据传入的参数动态计算变参数。
# 设计原则：计算在模型架构内部进行，而非由训练器/评估器计算。

import math
from typing import Optional, Tuple


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


def compute_quota_init_logits(max_level: int) -> Tuple[float, ...]:
    """计算配额初始化的 logits 值。

    数学形式化:
        使用逆深度加权: p_d ∝ 1/(d+1)
        logits[d] = log(p_d) - mean(log(p))

    设计原理:
        - 逆深度权重使较浅深度获得略高配额（粗粒度特征）
        - 转换为 log space 确保 softmax 初始化的合理性
        - 支持任意 max_level，不再硬编码 max_level=4

    示例:
        max_level=4:
        -> weights = [1/1, 1/2, 1/3, 1/4, 1/5] = [1.0, 0.5, 0.333, 0.25, 0.2]
        -> logits 调整使 softmax 后分布更均匀

    Args:
        max_level: 四叉树最大深度

    Returns:
        初始化的 logits 元组，长度为 max_level + 1

    Raises:
        ValueError: 当 max_level < 0 时
    """
    if max_level < 0:
        raise ValueError(f"max_level 必须非负, 得到 {max_level}")
    if max_level == 0:
        return (0.0,)

    import math

    # 逆深度权重: p_d ∝ 1/(d+1)
    raw_weights = [1.0 / (d + 1) for d in range(max_level + 1)]

    # 转换为 log space
    # I112-3: 使用 EPS 统一数值稳定性
    log_weights = [math.log(w + EPS) for w in raw_weights]
    mean_log = sum(log_weights) / len(log_weights)

    # 归一化使均值为 0
    logits = tuple(w - mean_log for w in log_weights)

    return logits


# 向后兼容别名
compute_max_depth = compute_max_level


def clamp_temperature(temperature: float, min_val: float = TEMPERATURE_MIN) -> float:
    """钳制温度参数到安全范围。

    数学形式化:
        T_safe = max(T, TEMPERATURE_MIN)
        softmax 梯度: ∂p/∂z ≈ p × (1-p) / T
        T >= 0.3 确保梯度流健康

    Args:
        temperature: 原始温度值
        min_val: 最小安全温度（默认 TEMPERATURE_MIN = 0.3）

    Returns:
        钳制后的安全温度值
    """
    return max(temperature, min_val)
