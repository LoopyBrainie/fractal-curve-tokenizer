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
#: P10-11 修复: 0.1 → 0.3 (梯度放大从10x降至3.3x, 数值稳定)
#: 数学推导见 scripts/verify_p10_11_temperature_analysis.py
SPLITTER_TEMP_END: float = 0.3

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

#: Gumbel 采样 uniform clamp (FP32)
#: 数学: g = -log(-log(u)), u ∈ [ε, 1-ε]
#: 验证: ε=1e-10 → g ∈ [-23, 23], 足够表达随机性
GUMBEL_EPSILON: float = 1e-10

#: Log 计算安全 epsilon
#: 用途: log(p + ε) 防止 log(0)
#: 验证: 对熵计算误差 < 1e-10
LOG_EPSILON: float = 1e-10

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

# ==================== 深度平衡常量 (I21) ====================
# 数学分析: 解决深度分布崩溃问题
# 问题: 候选数量不平衡 (d=0:1, d=1:4, d=2:16, d=3:64) 导致 Top-K 偏向 depth=3

#: Log-Compensation 是否启用
#: 数学: b_d = log(N_total / N_d) 实现期望均衡
LOG_COMPENSATION_ENABLED: bool = True

#: Depth KL 正则化损失权重
#: 数学: L_depth = λ × D_KL(π_depth || Uniform)
#: 目标: 鼓励选中 token 的深度分布趋向均匀
DEPTH_KL_WEIGHT: float = 0.1

#: Subset Softmax 是否启用
#: 数学: 将 STE softmax 从 N=85 缩小到 K=32，梯度增强 ~2.7x
SUBSET_SOFTMAX_ENABLED: bool = True

# ==================== 信息长度相关常量 ====================

#: levels_info 的最小长度
MIN_INFO_LEN: int = 16
