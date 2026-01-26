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

# ==================== 初始化相关常量 ====================

#: 嵌入层权重初始化标准差
EMBEDDING_INIT_STD: float = 0.02

# ==================== 注意力缩放常量 ====================

#: Hilbert 路径偏置的缩放因子
HILBERT_BIAS_SCALE: float = 0.1

#: 层级偏置的缩放因子
LEVEL_BIAS_SCALE: float = 0.05

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

#: I102-4: 形状归一化 epsilon (FP16 安全下界)
#: 用途: norm = ||f|| + ε 防止除零
#: FP16 安全: ε >= 1e-6 (FP16 最小正规数 ~6e-8)
#: 验证: 原 ε=1e-8 位于 FP16 边界，可能导致下溢
SHAPE_NORM_EPSILON: float = 1e-6

#: 温度参数下界 (Gumbel-Softmax/Top-K)
#: 数学分析: T < 0.1 时 softmax 梯度趋近于 0
#: I35 改进: 从 0.1 提升到 0.3，保持更健康的梯度流
#: 验证: T=0.3 时 softmax 梯度仍有效 (∂p/∂z ≈ 1/τ)
TEMPERATURE_MIN: float = 0.3

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

# ==================== I33: 相对预算与自适应覆盖率常量 ====================
# 数学分析: 动态深度下 K_max 与 K_min 应随图像尺寸自适应
# 原理:
#   1. 覆盖率 = K / N_candidates 应保持尺度不变性
#   2. β(H, W) = β_0 × sqrt(min(H, W) / 224) 实现线性尺度调整
#   3. 硬上限约束防止大分辨率下的显存溢出

#: 基准覆盖率 (224×224 图像的目标覆盖率)
#: 数学: β_0 = 0.12 表示目标采样 12% 的候选区域
#: I36 优化: 64×64 小图像需更高覆盖率，原 0.05 → 0.12
K_COVERAGE_BASE: float = 0.12

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

# ==================== I33: Elastic Budget 相对预算常量 ====================
# 设计原则: 与K参数相对预算保持一致
# 数学分析:
#   1. 相对覆盖率保证尺度不变性
#   2. 与 _get_dynamic_k_bounds() 统一设计

#: Elastic Budget 相对覆盖率硬上限
#: 数学: coverage_max = 0.25 防止过度采样 (原 0.08 → 0.25)
#: I36 优化: 与 K_COVERAGE_MAX_HARD 保持一致
ELASTIC_COVERAGE_MAX: float = 0.25

#: Elastic Budget 相对覆盖率下界 (用于崩溃检测)
#: 数学: coverage_min = 0.03 低于此值触发崩溃检测 (原 0.005 → 0.03)
#: I36 优化: 与 K_MIN_SAMPLE_RATIO 保持一致
ELASTIC_COVERAGE_MIN: float = 0.03

#: Elastic Budget 损失权重
#: 数学: λ = 0.1 使损失量级与其他辅助损失匹配
ELASTIC_LAMBDA_OVER: float = 0.1

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
