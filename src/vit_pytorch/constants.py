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
    λ_global = 0.1       全局上下文: ctx → λ_global · ctx

初始化:
    σ_emb = 0.02         嵌入层初始化标准差

Logits 裁剪:
    logits = clamp(logits, -10, 10)  防止数值溢出
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

#: 全局上下文的缩放因子 [已废弃: ARCH-R1 删除了 global_context_attn]
GLOBAL_CONTEXT_SCALE: float = 0.1

# ==================== 数值稳定性常量 ====================

#: Logits 截断的最小值（防止数值不稳定）
LOGITS_CLAMP_MIN: float = -10.0

#: Logits 截断的最大值（防止数值不稳定）
LOGITS_CLAMP_MAX: float = 10.0

# ==================== 信息长度相关常量 ====================

#: levels_info 的最小长度
MIN_INFO_LEN: int = 16
