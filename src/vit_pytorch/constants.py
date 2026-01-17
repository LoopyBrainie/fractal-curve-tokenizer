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
    T_end = 0.3          Gumbel-Softmax 终止温度 (P11-11: 安全下界)
    τ_base = 0.0         初始基础阈值 (P11-10: 对称于 z 分布)
    γ = 0.85             阈值衰减因子 (每层)
"""

from __future__ import annotations

# ==================== 层级相关常量 ====================

#: 默认最大递归层级（当未指定 max_level 时使用）
#: P11-2/I12-5: 与代码中的默认值保持一致
DEFAULT_MAX_LEVEL: int = 8

#: 系统支持的最大层级上限
MAX_SUPPORTED_LEVEL: int = 50

#: 递归深度安全余量（在 depth_limit 之上额外允许的层级）
EXTRA_DEPTH_CAP: int = 5

# ==================== 初始化相关常量 ====================

#: 嵌入层权重初始化标准差
EMBEDDING_INIT_STD: float = 0.02

# ==================== 注意力缩放常量 ====================

#: Hilbert 路径偏置的缩放因子
HILBERT_BIAS_SCALE: float = 0.1

#: 层级偏置的缩放因子
LEVEL_BIAS_SCALE: float = 0.05

# ==================== LearnableSplitter 默认参数 ====================

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

#: 阈值衰减因子 γ ∈ (0, 1)，每层深度阈值为 τ_d = τ_base · γ^d
SPLIT_GAMMA: float = 0.85

#: LearnableSplitter 初始基础阈值 (用于 MLP 参数初始化)
#: P11-10 修复: 使用 logit 空间初始化，τ₀ = 0.0 (对称初始化)
#: 注意: split_adaptive.py 中硬编码为 0.0，此常量仅为文档目的
LEARNABLE_INIT_TAU_BASE: float = 0.0

#: LearnableSplitter 初始阈值衰减 (用于 MLP 参数初始化)
LEARNABLE_INIT_TAU_GAMMA: float = 0.85

# ==================== 数值稳定性常量 ====================

#: Logits 截断的最小值（防止数值不稳定）
LOGITS_CLAMP_MIN: float = -10.0

#: Logits 截断的最大值（防止数值不稳定）
LOGITS_CLAMP_MAX: float = 10.0

# ==================== 数值稳定性常量 (I12-7) ====================
# 数学分析见: workspace/numerical_constants_analysis.py

#: Gumbel 采样 uniform clamp (FP32 计算中安全边界)
#: 数学: g = -log(-log(u)), u ∈ [ε, 1-ε]
#: FP32 安全边界: ε >= 1e-10 (FP32 最小 ~1e-38)
#: FP16 兼容边界: ε >= 1e-8 (FP16 最小正规数 ~6e-8)
#: 验证: ε=1e-8 → g ∈ [-18.4, 18.4], Gumbel 分布覆盖 99.99%
#: 注意: gumbel_topk_splitter.py 中已显式使用 FP32 计算
GUMBEL_EPSILON: float = 1e-8  # I30-8: FP16 安全阈值

#: Log 计算安全 epsilon
#: 用途: log(p + ε) 防止 log(0)
#: FP16 安全: ε >= 1e-8
#: 熵误差: < 1e-8 (可忽略)
#: I30-8: 提升到 1e-8 与 GUMBEL_EPSILON 统一
LOG_EPSILON: float = 1e-8

#: 除法安全 epsilon
#: 用途: x / (sum + ε) 防止除零
DIVISION_EPSILON: float = 1e-8

#: 概率下界 (避免 0 概率参与计算)
#: 用途: prob.clamp(min=PROB_EPSILON)
PROB_EPSILON: float = 1e-8

#: 温度参数下界 (Gumbel-Softmax/Top-K)
#: 数学分析: T < 0.1 时 softmax 梯度趋近于 0
#: 验证见: workspace/ste_gradient_analysis.py
TEMPERATURE_MIN: float = 0.1

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

#: 方差归一化的稳定性 epsilon
DEPTH_VARIANCE_NORM_EPS: float = 1e-6

#: Depth KL 正则化损失权重
#: 数学: L_depth = λ × D_KL(π_depth || Uniform)
#: 目标: 鼓励选中 token 的深度分布趋向均匀
#: I23-1 方案A: 0.1 → 0.5 (梯度量级分析表明 0.1 太弱，被主任务梯度淹没)
DEPTH_KL_WEIGHT: float = 0.5

#: I23-1 方案D: 软配额正则化是否启用
#: 数学: L_quota = Σ_d ReLU(|π_d - π_d^target| - ε)^2
#: 效果: 惩罚极端偏离目标分布，允许任务微调
DEPTH_QUOTA_ENABLED: bool = True

#: 软配额目标分布 (基于信息论分析)
#: 说明: depth 3 略高是因为高频纹理信息对分类有额外贡献
DEPTH_QUOTA_TARGET: tuple = (0.15, 0.20, 0.25, 0.40)

#: 软配额容忍带 ε (±5%)
DEPTH_QUOTA_TOLERANCE: float = 0.05

#: 软配额损失权重
DEPTH_QUOTA_WEIGHT: float = 0.2

# ==================== 方案 E: 可学习配额常量 (I24-2) ====================
# 数学分析: 解决 Log-Compensation 对 Top-K 理论无效的问题
# 原理: 先按可学习配额分配各深度 K_d，再在深度内 Top-K 选择
# 优势: 
#   1. 配额硬约束保证深度分布
#   2. 可学习比例允许任务自适应
#   3. 深度内竞争保持 token 质量优化

#: 是否启用可学习配额方案 (替代 Log-Compensation)
LEARNABLE_QUOTA_ENABLED: bool = True

#: 每个深度的最小配额 (防止死区)
#: 数学: K_d >= K_MIN_PER_DEPTH 保证梯度流
#: I26-1: 从 1 增加到 2，防止 tree consistency 后完全清空
QUOTA_MIN_PER_DEPTH: int = 2

#: 配额初始化 (对数空间，softmax 后 = DEPTH_QUOTA_TARGET)
#: 计算: φ_d = log(p_d) - mean(log(p))
#: 验证: softmax(-0.495, -0.207, 0.016, 0.486) ≈ (0.15, 0.20, 0.25, 0.40)
QUOTA_INIT_LOGITS: tuple = (-0.495, -0.207, 0.016, 0.486)

#: 配额熵正则化权重 (鼓励分布多样性)
#: 数学: L_entropy = -λ × H(K/K_total)
QUOTA_ENTROPY_WEIGHT: float = 0.1

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
