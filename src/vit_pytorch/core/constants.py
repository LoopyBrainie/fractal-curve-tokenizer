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
#: I120-8 修复: 降低至 0.01，避免 L2 损失主导学习
#: 数学: λ = 0.01 使梯度规模与熵损失匹配 (O(K) vs O(log D))
#: 引导 token 数量趋向最优覆盖率 (β_target = 12% @ 224×224)
ELASTIC_LAMBDA_TARGET: float = 0.01

#: Elastic Budget 边界安全网权重
#: 数学: λ = 0.1 超出 K_bounds 时额外惩罚
#: 与目标损失权重相等，实现对称保护
ELASTIC_LAMBDA_BOUNDARY: float = 0.1

#: 崩溃检测损失权重
#: 数学: λ_collapse = 1.0 确保崩溃时强惩罚
ELASTIC_LAMBDA_COLLAPSE: float = 1.0

# ==================== I122-4: Poisson KL 散度损失 (新) ====================
# 设计原则: 从第一性原理推导的最优损失函数
# 数学形式化:
#   L_KL = K_t × KL(Poisson(K) || Poisson(K_t))
#   L_Huber = Huber_δ(K, K_t) with δ = K_t / 2
#   L = λ_KL × L_KL + λ_Huber × L_Huber
#
# 核心洞察:
#   - Token 计数是离散 Poisson 过程
#   - KL 散度是计数偏差的信息论最优度量
#   - L_KL ≥ 0 且 L_KL = 0 当且仅当 K = K_t
#
# 梯度分析:
#   - ∂L_KL/∂K = λ_KL × log(K/K_t)
#   - K > K_t 时梯度 > 0 (减少 K)
#   - K < K_t 时梯度 < 0 (增加 K)
#   - 大偏差时梯度温和 (K=256 vs K_t=32: 梯度 ≈ 0.44)

#: Poisson KL 散度损失权重
#: 数学: 与熵损失量级匹配，λ_KL = 0.005
#: 推导: 目标梯度 ≈ 0.1-0.5 (与熵损失量级匹配)
#:       当 |log(K/K_t)| = 1 (e≈2.7x偏差) 时，梯度 = λ_KL
#:       设目标梯度 = 0.2: λ_KL = 0.2
#:       保守使用 λ_KL = 0.005
ELASTIC_LAMBDA_KL: float = 0.005

#: Huber 边界保护权重
#: 数学: 边界惩罚弱于主项，λ_Huber = 0.001 ≈ λ_KL / 5
#: 用途: δ = K_t / 2 范围内使用 L2，范围外使用线性惩罚
HUBER_LAMBDA: float = 0.001

#: Huber delta 参数 (K_t / 2)
#: 数学: δ = K_t / 2，确保合理边界范围
#: 含义: |K - K_t| ≤ δ 时使用 L2，否则使用线性惩罚
HUBER_DELTA: float = 16.0

#: 数值稳定性 epsilon (Poisson KL 计算用)
#: 防止 log(0) 和除零
ELASTIC_EPS: float = 1e-8

# ==================== I122-5: 熵目标公式 (新) ====================
# 设计原则: 基于信息论的有效深度概念
# 数学形式化:
#   H_target = log(D_eff) = ENTROPY_TARGET_SCALE × log(D)
#   其中 D_eff = ENTROPY_TARGET_SCALE⁻¹ × D (有效深度)
#
# 核心洞察:
#   - 原始公式 H = log(D) × (1 - 1/√D) 缺乏严格数学推导
#   - √D 项可与 Hilbert 局部性维度建立联系 (||H(d1)-H(d2)|| ≤ √2 × |d1-d2|^(1/2))
#   - 推荐简化: H_target = 0.5 × log(D)，保持恒定 50% 熵比例
#
# 理论依据:
#   - 有效深度 D_eff = √D (信息论视角下的有效类别数)
#   - H_target = log(D_eff) = 0.5 × log(D)
#   - 对所有 D 保持恒定 50% 熵比例，简化超参数调优

#: 熵目标缩放因子
#: 数学: H_target = ENTROPY_TARGET_SCALE × log(D)
#:       ENTROPY_TARGET_SCALE = 0.5 表示目标熵为最大熵的 50%
#:       等价于 D_eff = √D (有效深度 = √D)
#: 验证:
#   - D=4: H_target = 0.5 × 1.386 = 0.693 = log(2) ✓
#   - D=8: H_target = 0.5 × 2.079 = 1.039 = log(2.83) ≈ log(√8)
ENTROPY_TARGET_SCALE: float = 0.5

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

# ==================== I120-8: Hilbert-感知自适应预算系统 ====================
# 设计原则: 移除 L_budget (L2损失)，替换为 Hilbert-感知损失
# 数学形式化:
#   1. L_continuity = -Σ exp(-γ × ΔH_i) 鼓励 Hilbert 连续选择
#   2. 自适应覆盖率 β = β_min + (β_max - β_min) × complexity

#: Hilbert-感知连续性损失是否启用 (I120-8)
#: 替代 L_budget，与 Hilbert 曲线局部性保证对齐
HILBERT_CONTINUITY_ENABLED: bool = True

#: Hilbert-感知连续性损失权重
#: I120-8 修复: 提高至 0.2，增强 Hilbert 连续性约束
#: 数学: λ_continuity = 0.2 与降低后的预算损失保持平衡
HILBERT_CONTINUITY_WEIGHT: float = 0.2

#: Hilbert-感知连续性损失 gamma 参数 (I122-8: 保留基准值，添加动态计算函数)
#: 数学: exp(-γ × ΔH) 控制连续性强度
#: γ = 0.1 时，ΔH = 10 → exp(-1) ≈ 0.37 (显著惩罚)
#: 注意: 对于不同分辨率，γ 应动态调整为 γ = 0.1 × log₂(N_patches)
#:       见 compute_hilbert_continuity_gamma() 函数
HILBERT_CONTINUITY_GAMMA: float = 0.1

#: 任务自适应覆盖率是否启用 (I120-8)
#: 根据图像复杂度动态调整目标覆盖率
ADAPTIVE_COVERAGE_ENABLED: bool = True

#: 自适应覆盖率最小值 (简单图像)
#: 数学: β_min = 0.05 保证最小覆盖率
ADAPTIVE_COVERAGE_MIN: float = 0.05

#: 自适应覆盖率最大值 (复杂图像)
#: 数学: β_max = 0.40 允许复杂图像更多 tokens
ADAPTIVE_COVERAGE_MAX: float = 0.40

#: 复杂度估计权重 (HybridDensityHead 输出)
#: 数学: complexity = Σ I_d / N_candidates
#: I_d = softmax((1/4^d) × Σ Var(F_q))
ADAPTIVE_COMPLEXITY_WEIGHT: float = 1.0

#: 空间覆盖预算是否启用 (I120-8)
#: 确保选中的 token 覆盖图像的各个区域
COVERAGE_BUDGET_ENABLED: bool = True

#: 空间覆盖预算最小值
#: 数学: coverage_min = 0.3 保证 30% 区域覆盖
SPATIAL_COVERAGE_MIN: float = 0.3

#: 空间覆盖预算损失权重
#: 数学: λ_coverage = 0.05 轻量正则化
SPATIAL_COVERAGE_WEIGHT: float = 0.05

# ==================== I121-5, I122-6: 课程学习常量 ====================
# 设计原则: 根据训练阶段动态调整损失权重，平衡预算约束与深度多样性
# 数学形式化:
#   1. λ_b(t, K) = λ_b^0 · α(t) · β(K)
#      α(t) = Cosine平滑阶段因子
#      β(K) = 距离惩罚因子
#   2. λ_e(t) = λ_e^0 / α(t)  (I122-6: 从 1/√α 改为 1/α)
#      熵权重与预算权重线性反向联动
#      变化范围: 0.33×λ_e^0 ~ 3.0×λ_e^0 (9× 变化)
#
# 对称性原则 (I122-6):
#   √(α_explore/α_exploit) = 整数
#   推荐: √(3.0/0.33) = √9 = 3

#: 探索阶段结束 (归一化训练进度 0-1)
#: 数学: t ∈ [0, 0.25T] 为探索阶段，高权重强化预算约束
CURRICULUM_EXPLORATION_END: float = 0.25

#: 适应阶段结束 (归一化训练进度 0-1)
#: 数学: t ∈ [0.25T, 0.75T] 为适应阶段，平滑过渡
CURRICULUM_ADAPTATION_END: float = 0.75

#: 探索阶段权重因子 (I122-6: 从 2.0 改为 3.0)
#: 数学: α = 3.0 探索阶段增强预算约束
#: 对称因子: √(3.0/0.33) = √9 = 3
CURRICULUM_WEIGHT_FACTOR_EXPLORE: float = 3.0

#: 适应阶段权重因子
#: 数学: α = 1.0 适应阶段恢复正常权重
CURRICULUM_WEIGHT_FACTOR_ADAPT: float = 1.0

#: 利用阶段权重因子 (I122-6: 从 0.5 改为 0.33)
#: 数学: α = 1/3 利用阶段放松预算约束，允许深度多样性
CURRICULUM_WEIGHT_FACTOR_EXPLOIT: float = 1.0 / 3.0

#: 是否启用距离惩罚因子
#: 数学: β(K) > 1 当 K 超出 [K_min, K_max] 时增强惩罚
CURRICULUM_DISTANCE_PENALTY_ENABLED: bool = True

#: 距离惩罚强度系数 (I122-6: 从 1.0 改为 0.5)
#: 数学: β(K) = 1 + γ · (d_outside / d_inside)²
#: 验证: γ = 0.5 时，β ∈ [1.0, 1.5] (50% 最大增强)
CURRICULUM_PENALTY_GAMMA: float = 0.5

#: 课程学习熵权重基础值
#: I120-8 修复: 提高至 0.5，增强深度多样性约束
#: 数学: λ_e^0 = 0.5 与降低后的预算权重保持平衡
CURRICULUM_BASE_ENTROPY_WEIGHT: float = 0.5

#: 课程学习预算权重基础值
#: I120-8 修复: 降低至 0.01，避免 L2 损失主导
#: 数学: λ_b^0 = 0.01 基础权重 (会被配置覆盖)
CURRICULUM_BASE_BUDGET_WEIGHT: float = 0.01

# I120-8: 深度平衡损失权重
#: 深度平衡损失权重
#: 数学: λ_depth = 0.1 与其他辅助损失量级匹配
DEPTH_BALANCE_WEIGHT: float = 0.1

# ============================================================================
# Hilbert 空间均匀性优化常量 (I150 新增)
# ============================================================================
# 解决 GumbelTopK 采样空间分布不均问题的优化方案

# I150-1: 几何密度惩罚 (Hilbert Space Density Bias)
#: Hilbert 邻近度窗口大小
#: 数学: w 决定了局部邻域的范围，w 越大惩罚范围越广
#: 选择: w=64 覆盖约 1/4 的 Hilbert 曲线，适合中等尺度均匀性
HILBERT_DENSITY_WINDOW: int = 64

#: 几何密度惩罚权重
#: 数学: L_density = γ × Σ density_i，γ 控制均匀性强度
#: 选择: γ=0.1 与 quota 正则化权重相当
DENSITY_PENALTY_WEIGHT: float = 0.1

#: 密度惩罚学习率系数 (可选，让模型学习最优 γ)
#: 如果为 None，则使用固定权重
DENSITY_PENALTY_LEARNABLE: bool = False

# I150-2: 深度自适应温度 (Adaptive Temperature per Depth)
#: 深度温度缩放指数
#: 数学: τ_d = τ_base × (N_max / N_d)^γ，其中 N_d = 4^d
#: γ ∈ (0, 1) 控制缩放强度，γ=0.5 提供温和的平衡
DEPTH_TEMPERATURE_GAMMA: float = 0.5

#: 是否启用深度自适应温度
#: 推荐: True，与 quota 机制协同工作
DEPTH_ADAPTIVE_TEMPERATURE_ENABLED: bool = True

# I150-3: Hilbert-DPP 贪心采样
#: DPP 贪心多样性权重
#: 数学: score_i = logits_i - λ × max_{j∈selected} S[i,j]
#: λ 控制多样性-质量权衡，λ 越大多样性越高
DIVERSITY_LAMBDA: float = 0.5

#: 是否启用 Hilbert-DPP 贪心采样
#: 注意: 目前与分层 Top-K 不兼容
DIVERSITY_SAMPLING_ENABLED: bool = False

#: Hilbert 距离带宽参数 (用于相似度计算)
#: 数学: S[i,j] = exp(-|h_i - h_j|² / σ²)
#: σ 决定相似度衰减速度，σ 越大相似度衰减越慢
HILBERT_DIVERSITY_SIGMA: float = 128.0


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


# ==================== I122-8: 动态常量计算函数 ====================

def compute_hilbert_continuity_gamma(
    image_size: int,
    patch_size: int,
    base_gamma: float = HILBERT_CONTINUITY_GAMMA,
    base_n_patches: int = 4096,
) -> float:
    """计算分辨率自适应的 Hilbert 连续性 gamma 参数 (I122-8)。

    数学形式化:
        γ(image_size, patch_size) = base_γ × log₂(N_patches) / log₂(N_base)

    其中:
        N_patches = (image_size / patch_size)²
        N_base = 4096 (224×224, patch=4 的基准)

    设计原理:
        1. Hilbert 曲线的局部性保证: ‖Δp‖ ≤ √2 × |ΔH|^0.5
        2. 固定 γ = 0.1 对不同分辨率产生差异巨大的相对惩罚
        3. 动态 γ 确保 exp(-γ × ΔH) 的衰减与图像大小无关

    验证:
        | ΔH | = 1 → ‖Δp‖ ≤ √2 (1 个 patch)
        | ΔH | = 9 → ‖Δp‖ ≤ 3√2 (约 3 个 patches)
        ΔH = 10 隐含允许 3 个 patch 跳跃

    示例:
        64×64, patch=4: N=256, log₂=8, γ = 0.1 × 8/12 = 0.67
        224×224, patch=4: N=3136, log₂≈11.6, γ = 0.1 × 11.6/12 = 0.97
        224×224, patch=16: N=196, log₂≈7.6, γ = 0.1 × 7.6/12 = 0.63

    Args:
        image_size: 输入图像尺寸（正方形边长）
        patch_size: Patch 尺寸
        base_gamma: 基准 gamma 值 (默认 HILBERT_CONTINUITY_GAMMA = 0.1)
        base_n_patches: 基准 patches 数量 (默认 4096)

    Returns:
        分辨率自适应的 gamma 值
    """
    if image_size <= 0:
        raise ValueError(f"image_size 必须为正数, 得到 {image_size}")
    if patch_size <= 0:
        raise ValueError(f"patch_size 必须为正数, 得到 {patch_size}")
    if patch_size > image_size:
        raise ValueError(f"patch_size ({patch_size}) 不能大于 image_size ({image_size})")

    # 计算 patches 数量
    n_patches = (image_size // patch_size) ** 2

    # 计算 log₂ 比例
    log_ratio = math.log2(n_patches) / math.log2(base_n_patches)

    # 动态 gamma
    gamma = base_gamma * log_ratio

    return gamma


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
        T >= 0.4 确保梯度流健康 (I121-8: τ=0.4 梯度强度 ≈ 2.5)

    Args:
        temperature: 原始温度值
        min_val: 最小安全温度（默认 TEMPERATURE_MIN = 0.4）

    Returns:
        钳制后的安全温度值
    """
    return max(temperature, min_val)
