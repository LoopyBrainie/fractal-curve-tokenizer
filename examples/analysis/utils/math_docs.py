# -*- coding: utf-8 -*-
"""
Mathematical Documentation - 数学公式文档

提供 Fractal Curve ViT 核心数学概念的参考文档。

使用方法
========
    from examples.analysis.utils.math_docs import HILBERT_LOCALITY_BOUND
    print(HILBERT_LOCALITY_BOUND)

目录
=====
1. Hilbert 曲线局部性界
2. LCA-距离等价性定理
3. Gumbel-Softmax 梯度分析
4. SwiGLU 门控机制
5. Focal Loss 梯度分析
"""

# =============================================================================
# 1. Hilbert 曲线局部性界
# =============================================================================

HILBERT_LOCALITY_BOUND = r"""
**Hilbert 曲线局部性界**

对于 n × n 网格上的 Hilbert 曲线 H_n: [0, n²) → [0, n) × [0, n)，有:

‖H_n(d₁) - H_n(d₂)‖_2 ≤ √(2 × |d₁ - d₂|)

其中:
- ‖·‖_2 是欧几里得距离
- 右侧的 √2 是因为对角线移动是最大步长
- |d₁ - d₂| 是 1D 索引差

**证明要点**:
1. Hilbert 曲线是分形曲线，每次递归将正方形分为 4 个子正方形
2. 每次递归，索引差最多增加 4 倍，但空间距离最多增加 2 倍
3. 归纳可得上述界

**对比栅格顺序**:
- 栅格顺序中相邻索引 (i, j) 和 (i+1, j) 距离为 1
- 但对角线跳跃 (n-1, 0) 到 (0, n-1) 距离为 √2 × n

**代码验证**:
    from examples.analysis.utils.hilbert_utils import generate_hilbert_curve
    coords = generate_hilbert_curve(4)
    # 验证局部性界
    for i in range(len(coords)):
        for j in range(len(coords)):
            d_ij = abs(i - j)
            dist = ((coords[i][0] - coords[j][0])**2 + (coords[i][1] - coords[j][1])**2)**0.5
            assert dist <= (2 * d_ij)**0.5 + 1e-6, f"Locality bound violated at ({i}, {j})"
"""

# =============================================================================
# 2. LCA-距离等价性定理
# =============================================================================

LCA_DISTANCE_EQUIVALENCE = r"""
**LCA-距离等价性定理**

设 pos(d) 为深度 d 区域的 Hilbert 中心坐标，则:

LCA(i, j) = ℓ ⟹ ‖pos_i - pos_j‖_∞ ≤ N / 2^ℓ

其中:
- LCA(i, j) 是两个 token 在四叉树中的最近公共祖先深度
- N = 2^max_depth 是网格边长
- ‖·‖_∞ 是 Chebyshev 距离 (最大坐标差)

**意义**:
- ℓ 越小 (越接近根)，区域越大，距离上界越大
- ℓ 越大 (越接近叶子)，区域越小，距离上界越小
- 可用于构建层级感知的位置偏置

**应用**:
    from examples.analysis.visualization.attn_hilbert_bias import LCAHilbertBias

    # 创建偏置: 深度 ℓ 处的 tokens 有较小的注意力距离
    bias = LCAHilbertBias(dim=64, max_depth=4)
    # 输出偏置形状: [1, num_heads, N, N]
"""

# =============================================================================
# 3. Gumbel-Softmax 梯度分析
# =============================================================================

GUMBEL_SOFTMAX_GRADIENT = r"""
**Gumbel-Softmax 梯度分析**

给定 logits z，Softmax 概率 p_i = exp(z_i) / Σ exp(z_j)

温度 T 下的梯度:
∂p_i/∂z_j = (δ_ij - p_j) × p_i / T

其中 δ_ij 是 Kronecker delta。

**Top-K 选择梯度覆盖**:
- 选中的 token: 梯度 ∝ p_i(1-p_i) ≈ 1/T (当 p_i ≈ 1)
- 未选中的 token: 梯度 ∝ p_i² ≈ 0
- 梯度比 ≈ (1/T) / ε = K/N

当 T = 0.5, K = 32, N = 85 时:
梯度覆盖率 ≈ 32/85 ≈ 37.6%

**代码实现**:
    from examples.analysis.visualization.gumbel_topk_splitter import GumbelTopKSplitter

    splitter = GumbelTopKSplitter(max_depth=4, temperature=0.5)
    selected, info = splitter(logits)  # 返回选中的 tokens 和辅助信息
"""

# =============================================================================
# 4. SwiGLU 门控机制
# =============================================================================

SWIGLU_GATING = r"""
**SwiGLU 门控机制**

SwiGLU(x) = (W_g x ⊙ W_v x) × σ(W_g x)

其中:
- ⊙ 是 Hadamard 积 (元素级乘法)
- σ(x) = x / (1 + exp(-x)) 是 Swish 激活函数

**梯度流**:
∂SwiGLU/∂x = W_out × [σ(x) ⊙ (1 - σ(x)) ⊙ W_v x + σ(x) ⊙ W_v]

**与 ReLU 对比**:
- ReLU: ∂/∂x = W_v × 1[x > 0] (稀疏梯度)
- SwiGLU: 平滑梯度，更稳定的训练

**代码实现**:
    from vit_pytorch.ffn_swiglu import SwiGLUFFN

    ffn = SwiGLUFFN(dim=512, hidden_dim=2048, max_depth=4)
    output = ffn(x, depth_ids)  # x: [B, N, D], depth_ids: [B, N]
"""

# =============================================================================
# 5. Focal Loss 梯度分析
# =============================================================================

FOCAL_LOSS_GRADIENT = r"""
**Focal Loss 梯度分析**

FL(p_t) = -α_t (1 - p_t)^γ log(p_t)

对数it z_t 的梯度:
∂FL/∂z_t = α_t [(1-p_t)^γ - γ p_t (1-p_t)^{γ-1}] × (p_t - y_t)

**类别不平衡处理**:
- α_t: 类别权重，平衡正负样本
- (1-p_t)^γ: 降低易分类样本的权重
- γ > 0: 聚焦困难样本

**当 p_t → 1** (易分类):
(1-p_t)^γ → 0，梯度趋近 0

**当 p_t ≈ 0.5** (困难样本):
梯度较大，促进学习

**代码实现**:
    from training.losses.focal_loss import FocalLoss

    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    loss = criterion(logits, targets)
"""

# =============================================================================
# 6. 四叉树分割数学
# =============================================================================

QUADTREE_SPLIT = r"""
**四叉树分割数学**

对于图像 I，大小为 N × N，深度为 d 的分割:

区域数量: N_d = 4^d
区域大小: S_d = N / 2^d × N / 2^d
总区域数 (深度 0 到 D): Σ_{d=0}^D 4^d = (4^{D+1} - 1) / 3

**Hilbert 排序**:
每个区域 R 分配 Hilbert 索引 h(R) ∈ [0, N_d)

**分形位置编码**:
P(R) = [HilbertEncode(h(R)); depth(R); area(R)]

其中:
- HilbertEncode: Hilbert 索引的嵌入向量
- depth: 区域深度 (0 到 D)
- area: 区域面积归一化值
"""

# =============================================================================
# 7. Token 预算约束
# =============================================================================

TOKEN_BUDGET = r"""
**Token 预算约束**

目标: 限制计算 FLOPs 在预算 B 内

Token FLOPs 估计:
FLOPs(N) = 12 × N × D² + 2 × N² × D

其中:
- N: token 数量
- D: 特征维度
- 12ND²: Q, K, V, O 投影 + FFN (每个 2ND²)
- 2N²D: 注意力分数计算

预算约束:
N ≤ N_max where FLOPs(N_max) ≤ B

**代码验证**:
    from examples.analysis.visualization.architecture_design import plot_token_count_range

    # 可视化不同预算下的 token 范围
    plot_token_count_range(budgets=[1e9, 5e9, 1e10])
"""

# =============================================================================
# 8. 深度感知注意力偏置
# =============================================================================

DEPTH_AWARE_ATTENTION = r"""
**深度感知注意力偏置**

Fractal Curve ViT 的注意力偏置:

B[i, j] = B_hilbert[i, j] + B_level[i, j] + B_affine[i, j]

其中:
1. **Hilbert 偏置**:
   B_hilbert[i, j] = τ_h × Embedding(LCA(i, j))

2. **Level 偏置**:
   B_level[i, j] = Embedding(depth_i - depth_j + L)

3. **Affine 偏置**:
   B_affine[i, j] = α_i × β_j (可学习的缩放)

**参数效率**:
- 标准 ViT: O(N²) 位置参数
- Fractal ViT: O(D×H + D×L + 2D) 位置参数

其中 H 是最大深度，L 是最大深度差。
"""


if __name__ == "__main__":
    # 打印所有数学文档的标题
    docs = {
        "Hilbert 局部性界": HILBERT_LOCALITY_BOUND,
        "LCA-距离等价性": LCA_DISTANCE_EQUIVALENCE,
        "Gumbel-Softmax": GUMBEL_SOFTMAX_GRADIENT,
        "SwiGLU": SWIGLU_GATING,
        "Focal Loss": FOCAL_LOSS_GRADIENT,
        "四叉树分割": QUADTREE_SPLIT,
        "Token 预算": TOKEN_BUDGET,
        "深度感知注意力": DEPTH_AWARE_ATTENTION,
    }

    for name, doc in docs.items():
        print(f"\n{'=' * 60}")
        print(f"文档: {name}")
        print('=' * 60)
        # 只打印前 200 个字符
        print(doc[:200] + "...")
