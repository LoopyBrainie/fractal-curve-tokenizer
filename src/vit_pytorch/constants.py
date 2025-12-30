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
DEFAULT_MAX_LEVEL: int = 10

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

#: Gumbel-Softmax 终止温度 T_end (0.1 接近 hard sampling)
SPLITTER_TEMP_END: float = 0.1

#: 阈值衰减因子 γ ∈ (0, 1)，每层深度阈值为 τ_d = τ_base · γ^d
SPLIT_GAMMA: float = 0.85

#: LearnableSplitter 初始基础阈值 (用于 MLP 参数初始化)
#: 注意: 这不同于已废弃的规则分割器的 tau_0=0.15
#: LearnableSplitter 使用 sigmoid 输出 C_θ(R) ∈ [0,1]，所以 0.5 是中心初始化
LEARNABLE_INIT_TAU_BASE: float = 0.5

#: LearnableSplitter 初始阈值衰减 (用于 MLP 参数初始化)
LEARNABLE_INIT_TAU_GAMMA: float = 0.85

# ==================== 数值稳定性常量 ====================

#: Logits 截断的最小值（防止数值不稳定）
LOGITS_CLAMP_MIN: float = -10.0

#: Logits 截断的最大值（防止数值不稳定）
LOGITS_CLAMP_MAX: float = 10.0

# ==================== 信息长度相关常量 ====================

#: levels_info 的最小长度
MIN_INFO_LEN: int = 16
