# 第三章：分形分词器

## 3.1 概述

`StreamingFractalTokenizerV3` 通过自适应四叉树分割和 Hilbert 曲线重排序实现**变深度分词**。推荐的分割器是 **H1SS (Hilbert Splitter with Stable Selection)**，使用 Entmax 稀疏激活，提供完整梯度流和稳定的训练/推理一致性。

**效率说明**：~40× 的计算减少来自于 token 数量减少（$N_{V3} \approx 32$ vs $N_{ViT} \approx 307K$），而非渐近复杂度变化。注意力保持 $O(N^2 \cdot D)$，但 $N$ 减少了 ~40×。

**架构说明**：Token 深度由 H1SS 选择机制和 Entmax 稀疏激活决定。Transformer 有效深度固定为 `num_layers // 2`。

---

## 3.2 数学形式化

### 3.2.1 分词管道

$$I \xrightarrow{\text{Splitter}} \text{split\_result} \xrightarrow{\text{Embedding}} \{T_i, L_i\} \xrightarrow{\text{HilbertSort}} \{T_i', L_i'\}$$

其中：
- $I \in \mathbb{R}^{B \times C \times H \times W}$：输入批次
- $\text{split\_result}$：来自 H1SS/H-entmax 的选中区域
- $T_i$：Token 嵌入
- $L_i$：级别（深度）信息

### 3.2.2 H1SS: Hilbert 稳定选择分割器

H1SS 基于六个公理（A1-A6）实现最优 token 选择：

| 公理 | 描述 | 实际表现 |
|:------|:------------|:---------|
| A1 | 1D Hilbert 流形卷积 | DistanceDecay Conv1D 固定衰减权重 |
| A2 | 无 Gumbel 扰动 | ✓ 确定性 |
| A3 | Entmax 稀疏激活 | **矛盾**：稀疏性 ↔ 梯度流 |
| A4 | 树一致性软约束 | 课程学习 λ，效果存疑 |
| A5 | 单次 Entmax 投影 | TopK after Entmax，非真正 Entmax 选择 K |
| A6 | < 10K 参数 | ✓ 约 10K |

**⚠️ 核心矛盾：稀疏性 ↔ 梯度流**

| α 值 | Entmax 行为 | 梯度覆盖率 | 稀疏性 |
|:------|:------------|:-----------|:-------|
| α → 1.0 | Softmax | **100%** | 无 |
| α = 1.2 | 弱 Entmax | **100%** | 低 |
| α = 1.49 | 近似 sparsemax | ~60-80% | 中 |
| α = 1.5 | sparsemax | **~0%** ❌ | 高 |

**关键发现（I107）**：α=1.5 的 sparsemax 导致 **100% 梯度消失**！当某元素概率为 0 时，其对应梯度也为 0。

**α-Entmax 定义**：

$$\text{entmax}_\alpha(z) = \arg\max_{p \in \Delta^{n-1}} (p^T z + H_\alpha(p))$$

其中 $H_\alpha(p) = \frac{1}{1-\alpha} \log \sum_i p_i^\alpha$ 是 Rényi 熵。

**性质分析**：

| α 值 | 行为 | 梯度性质 |
|:-----|:-----|:---------|
| α → 1 | softmax | 完全可微，稠密输出 |
| α = 1.5 | sparsemax | 稀疏输出，**梯度可能为零** |
| α → ∞ | argmax | 极度稀疏，**梯度消失** |

**这是一个不可能三角**：

```
            稀疏性
              ▲
             / \
            /   \
           /  ✗  \
          /       \
        梯度 ◁────▷ 树一致性
```

任何方案只能同时优化两个目标。

**Alpha 预热策略（I107）**：

H1SS 使用 alpha 预热调度以防止早期训练中的硬截断：

$$\alpha(t) = \begin{cases} 1.2 + 0.3 \cdot \frac{t}{T_{warmup}} & t < T_{warmup} \\ \min(1.49, 1.2 + 0.29 \cdot \frac{t - T_{warmup}}{T_{schedule} - T_{warmup}}) & t \geq T_{warmup} \end{cases}$$

默认调度：$T_{warmup} = 10$，$T_{schedule} = 25$

**⚠️ 重要更新**：α 永远不会达到 1.5，最大为 1.49，以避免梯度消失。

**选择（I122-1 更新）**：

原始 STE 公式：

$$\text{st\_mask} = \text{hard\_mask} - \text{soft\_mask.detach()} + \alpha \cdot \text{soft\_mask}$$

已被**entmax 稀疏选择**取代以提高稳定性：

$$\text{selected\_mask} = \text{entmax}_\alpha(z) \cdot K$$

其中 $K$ 是目标 token 数量，$\alpha$ 是 entmax 参数。对于 $\alpha < 1.9$，entmax 退化为 softmax，提供直接梯度流无需 STE 近似。

### 3.2.3 H-entmax: Hilbert 有序 Entmax（替代方案）

**Hilbert 有序 Entmax 分割器**使用 α-Entmax 实现精确稀疏性和完整梯度流：

| 特性 | 描述 |
|:--------|:------------|
| **并行性** | 100% |
| **梯度流** | **100%**（完整，无 STE） |
| **Hilbert 局部性** | 依赖 Conv1D 特征提取 |
| **稀疏性** | 使用 α=1.5（⚠️ 梯度消失风险） |

**H-entmax 的局限性**：

1. **无深度控制**：深度阈值是简单的按层缩放
2. **无树约束**：可能选择子节点而不选父节点
3. **无 K 估计**：需要外部指定 K

### 3.2.4 树一致性约束

树一致性约束防止同时选择父节点及其子节点：

$$\text{consistent}(i) \iff i \in \text{Selected} \land \forall c \in \text{children}(i), c \notin \text{Selected}$$

**H1SS 树约束（软约束）**：

$$z_{parent} \leftarrow z_{parent} - \lambda \cdot \max_{c \in children(parent)} z_c$$

**⚠️ 问题分析**：

设 $z_{parent} = 5$, $\max(z_{children}) = 10$, $\lambda = 0.3$

结果：$z_{parent}^{new} = 5 - 0.3 \times 10 = 2$

但如果其他父节点 $z_{other} = 1.5$，而 TopK 选择 K=10 个：
- $z_{parent}^{new} = 2$ 仍可能被选中
- $z_{other} = 1.5$ 可能被淘汰

**结论**：树约束**不能保证**树一致性，只能**鼓励**树一致性。

**树约束损失**（H1SS）：

$$\mathcal{L}_{tree} = \sum_{i} \sum_{c \in \text{children}(i)} \max(0, p_i - p_c + \epsilon)$$

**跳跃损失**（Hilbert 连续性）：

跳跃损失通过惩罚大的 Hilbert 距离跳跃来鼓励选择空间连续的区域：

$$\mathcal{L}_{jump} = \mathbb{E}[(\Delta h - 1)^2_+]$$

其中：
- $\Delta h = |h_{i+1} - h_i|$：相邻选中 token 之间 Hilbert 索引的绝对差异
- $(\cdot)_+ = \max(0, \cdot)$：ReLU 激活

**为什么用 ReLU 而非指数（I107）**：

| 形式 | $\Delta h = 1$ | $\Delta h \to \infty$ | 行为 |
|:-----|:----------------|:---------------------|:---------|
| $\exp(-\gamma \Delta h)$ | $0.37$（高惩罚） | $0$（无惩罚） | 错误：大跳跃不受惩罚 |
| $(\Delta h - 1)^2_+$ | $0$（无惩罚） | $\infty$（二次增长） | 正确：按比例惩罚大跳跃 |

ReLU 形式正确实现了直觉：小 Hilbert 跳跃（$\Delta h \leq 1$，对应相邻区域）应该被允许，而大跳跃应该受到二次惩罚。

## 3.3 分割器深度对比分析

### 3.3.1 梯度覆盖率对比

| 方法 | 梯度覆盖率 | 原因 |
|:-----|:-----------|:-----|
| Gumbel-TopK | ~37% | STE 只在选中 token 上传递梯度 |
| H1SS (α≈1.2) | **100%** | softmax 完全可微 |
| H1SS (α≈1.49) | ~60-80% | 稀疏 Entmax 仍有部分梯度 |
| H-entmax (α=1.5) | **~0%** ❌ | 硬截断导致梯度消失 |

### 3.3.2 局部性保持对比

**测量方法**：

$$J(S) = \frac{1}{K-1} \sum_{i=1}^{K-1} |h(s_i) - h(s_{i+1})|$$

**两种实现的局部性来源**：

| 组件 | H1SS | H-entmax |
|:-----|:-----|:---------|
| 序列排序 | Hilbert | Hilbert |
| 特征混合 | DistanceDecay Conv1D | Standard Conv1D |
| 选择机制 | Entmax TopK | Entmax TopK |
| 局部性强化 | SDS 正则化（可选） | 无 |

**⚠️ 批判**：H1SS 的 `HilbertDistanceDecayConv1D` 使用**固定衰减权重** $w_d = 1/(|d|+1)$，没有利用梯度学习最优邻域权重。

### 3.3.3 综合对比

| 指标 | H1SS | H-entmax | 备注 |
|:-----|:-----|:---------|:-----|
| **可学习参数** | ~10K | ~2K | H-entmax 更轻量 |
| **配置参数** | ~20 | ~4 | H1SS 更复杂 |
| **梯度覆盖率** | ~100%（早期）/ 60-80%（晚期） | ~0%（α=1.5 时） | H1SS 更稳定 |
| **稀疏性** | 可调 | 固定 | H1SS 更灵活 |
| **树约束** | 有（软约束） | 无 | H1SS 有额外约束 |
| **α 调度** | 5+ 组件 | 无 | H1SS 复杂 |
| **维护性** | ⚠️ 复杂 | ✓ 简洁 | H-entmax 更易维护 |

## 3.4 分割器比较总结

| 分割器 | 梯度 | 稀疏性 | 参数 | 状态 |
|:---------|:---------|:---------|:-----------|:-------|
| GumbelTopK | ~37% (STE) | 软 | 中等 | 已废弃 |
| DeterministicNeighbor | 100% | 软 | 中等 | 已废弃 |
| **H1SS (HilbertOptimal)** | **~100%（早期）/ 60-80%（晚期）** | **动态** | **< 10K** | ✓ **推荐（需简化）** |
| **H-entmax** | **~0%（α=1.5）⚠️** | **固定** | **~2K** | ⚠️ **慎用** |

**建议**：
- **短期**：采用 H1SS，但简化调度器（移除 α 课程学习）
- **长期**：重新设计 H1SS，用硬约束替代软约束

### H1SS 参数

```python
HilbertOptimalSplitter(
    feature_dim=256,
    hidden_dim=64,
    max_level_limit=8,
    K_min=8,
    K_max=64,
    entmax_alpha=1.2,  # α-Entmax alpha
    tree_constraint_weight=0.1,
    temperature_init=1.0,
    temperature_min=0.3,
    jump_loss_weight=0.1,
    density_field_hidden_dim=32,
)
```

### H-entmax 参数

H-entmax 功能由 `HilbertOptimalSplitter` 使用 α-Entmax 激活提供：

```python
# 使用 HilbertOptimalSplitter 的 entmax_alpha 实现 H-entmax 行为
HilbertOptimalSplitter(
    feature_dim=256,
    hidden_dim=64,
    max_level_limit=8,
    K_min=8,
    K_max=64,
    entmax_alpha=1.5,  # α-Entmax（完全稀疏选择）
    tree_constraint_weight=0.1,
    temperature_init=1.0,
    temperature_min=0.3,
)
```

---

## 3.4 DeterministicNeighborSplitter（已废弃）

> **注意**：此分割器**已废弃**。请使用 **H1SS (HilbertOptimalSplitter)** 代替。

### 概述

`DeterministicNeighborSplitter` 由于其确定性行为和 100% 梯度覆盖而先前被推荐，但已被 H1SS 取代，后者提供更好的参数效率（< 10K 参数）和稀疏激活。

### 比较

| 维度 | GumbelTopK | DeterministicNeighbor | H1SS |
|:---------|:------------|:---------------------|:-----|
| **随机性来源** | $g \sim Gumbel(0,1)$ | 无 | 无 |
| **梯度覆盖** | ~37% (STE) | 100%（直接） | **100%** |
| **参数** | 中等 | 中等 | **< 10K** |
| **稀疏性** | 软 | 软 | **稀疏** |
| **状态** | 已废弃 | 已废弃 | ✓ **推荐** |

---

## 3.5 Token 嵌入

### 3.5.1 HilbertNativePatchEmbed

满足四个约束的区域到 token 嵌入：

| 约束 | 描述 | 实现 |
|:-----------|:------------|:---------------|
| **C1** | 尺度等变性 | 共享卷积 + 自适应池化 |
| **C2** | 深度感知 | 可学习深度调制 |
| **C3** | Hilbert 兼容性 | 保持四叉树路径 |
| **C4** | 可微性 | ROI-Align 实现平滑梯度 |

**数学形式**：

$$t_i = \text{Pool}(F[R_i]) \cdot \sigma_d + E_d$$

其中：
- $F$：共享卷积特征
- $\text{Pool}$：池化到固定大小
- $\sigma_d$：深度相关尺度（可学习）
- $E_d$：深度嵌入

### 3.5.2 ROI-Align

为跨区域边界的平滑梯度流：

$$\text{ROIAlign}(F, R) = \text{BilinearInterpolate}(F, \text{SamplePoints}(R))$$

这避免了整数舍入的量化伪影。

### 3.5.3 形状-尺度增强（I31-2）

$$t_i' = t_i + E_{shape}(R_i)$$

其中 $E_{shape}$ 由形状-尺度编码器计算。

---

## 3.5 Hilbert 重排序

分割和嵌入后，token 按其 Hilbert 曲线索引排序：

$$\pi(i) = \text{argsort}(H^{-1}(\text{center}(R_i)))$$

### 同构

$$\text{QuadtreePath}(R) = [q_1, \ldots, q_d] \iff \text{HilbertSegment}(R) = H|_{[a,b]}$$

这确保了：
1. 空间上相邻的区域在序列中有接近的位置
2. LCA 关系被保留用于注意力偏置

### P11-3：基于区域的 LCA 计算

为准确的注意力偏置，LCA 直接从区域边界计算：

$$\text{Path}(R) = \text{bit}(cx, D-d) + 2 \cdot \text{bit}(cy, D-d)$$
$$\text{LCA}(i, j) = \text{Length}(\text{CommonPrefix}(\text{Path}(i), \text{Path}(j)))$$

### I161-1：深度根归一化 Hilbert 索引

原始 Hilbert 索引 $H \in [0, 4^d)$ 跨深度变化，使得直接比较存在问题。**深度根归一化**解决了这个问题：

$$H_{norm} = \left(\frac{H}{4^d}\right)^{\frac{1}{d}}$$

**属性**：

- $H_{norm} \in [0, 1]$ 适用于所有深度 $d$
- 深度 $d$ 的 $H_{norm}$ 自然与深度 $d-1$（父级别）的 $H_{norm}$ 对齐
- 保持 Hilbert 曲线的自相似性

**为什么这很重要**：

没有归一化，深度-0 且 $H=100$ 的 token 会显得比深度-3 且 $H=10$ 的 token "更远"，尽管深度-3 的 token 覆盖了更小、更具体的区域。归一化确保尺度不变比较，同时保持空间局部性排序。

## 3.6 实现

### 类：StreamingFractalTokenizerV3

```python
class StreamingFractalTokenizerV3(BaseTokenizer):
    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]] = 224,
        channels: int = 3,
        d_model: int = 256,
        base_patch_size: int = 4,
        min_patch_size: Union[int, Tuple[int, int]] = 4,
        max_level: Optional[int] = None,
        use_hilbert_order: bool = True,
        depth_scale_range: Optional[Tuple[float, float]] = (0.5, 2.0),
        use_interpolated_pooling: bool = False,
        use_dynamic_weight: bool = False,
    ):
        """
        参数:
            image_size: 输入图像大小（int 或 (W, H) 元组）
            channels: 输入通道数
            d_model: 模型维度
            base_patch_size: 级别 0 的基础 patch 大小
            min_patch_size: 目标最小 patch 大小
            max_level: 最大四叉树级别（如果为 None 则自动计算，I30-17）
            use_hilbert_order: 启用 Hilbert 曲线排序
            depth_scale_range: P6-1 sigmoid 参数化的深度尺度范围
            use_interpolated_pooling: 启用插值池化（I-PHASE4）
            use_dynamic_weight: 启用动态权重（C3 尺度等变性）
        """
```

### 核心逻辑：HilbertOptimalSplitter (H1SS)

```python
class HilbertOptimalSplitter(nn.Module):
    def forward(self, features: Tensor) -> TensorSplitResult:
        """
        返回:
            TensorSplitResult，包含：
            - regions: [M, 4] 选中的区域坐标
            - depths: [M] 区域深度
            - batch_indices: [M] 批次索引
            - hilbert_indices: [M] Hilbert 曲线索引
            - selected_mask: [B, N] Entmax 稀疏选择掩码
            - logits: [B, N] 原始 logits
            - probs: [B, N] 分割概率
            - num_selected_per_batch: [B] 每样本的 token 数量
        """
```

### 诊断指标

- **尺度熵**：测量选中深度的多样性
- **Top-K 重叠**：（调试）父节点和子节点同时在 Top-K 中的频率（一致性检查前）
- **配额分布**：$\pi_d$ 来自方案 E 可学习配额的数值

---

## 3.7 使用示例

### 基本用法（H1SS 推荐）

```python
from vit_pytorch import StreamingFractalTokenizerV3

# H1SS (Hilbert Splitter with Stable Selection)
# 注意：分割器现在是单独组件传递给 tokenize()
tokenizer = StreamingFractalTokenizerV3(
    image_size=224,
    d_model=256,
    base_patch_size=4,
    depth_scale_range=(0.5, 2.0),  # P6-1: sigmoid 参数化
)

# 返回带有对齐 token 和级别信息的 TokenizerOutput
output = tokenizer.tokenize(images)

# 访问结果
tokens = output.tokens          # [B, N, D]
levels_info = output.levels     # [B, N, max_level+1]
regions = output.regions        # [B, N, 4] - 区域边界
split_probs = output.split_probs  # [B, N] - 选择置信度
```

### 直接分割器用法

```python
from vit_pytorch.layers.splitters.hilbert_optimal_splitter import HilbertOptimalSplitter

splitter = HilbertOptimalSplitter(
    feature_dim=256,
    hidden_dim=64,
    max_level_limit=8,
    K_min=8,
    K_max=64,
    entmax_alpha=1.2,
    tree_constraint_weight=0.1,
)
```

---

## 3.8 温度退火

温度退火控制 H1SS 和 H-entmax 的探索与利用之间的权衡：

$$\tau(t) = \tau_{start} \cdot \left(\frac{\tau_{end}}{\tau_{start}}\right)^{t / T_{total}}$$

**默认调度（H1SS）**：

| 参数 | 值 |
|:----------|:------|
| $\tau_{start}$ | 1.0 |
| $\tau_{end}$ | 0.3 |
| 调度 | 指数衰减 |

---

## 3.9 常量参考

所有数值稳定性常量都集中在 `constants.py` 中：

| 常量 | 值 | 数学基础 |
|:---------|:------|:-------------------|
| `EPS` | $1e-6$ | 通用 FP 稳定性 |
| `FP16_SAFE_EPSILON` | $1e-6$ | FP16 安全下界 |
| `TEMPERATURE_MIN` | $0.3$ | 防止梯度饱和（I113-10: τ=0.3 时梯度强度 ≈ 3.3） |
| `LOGIT_CLAMP_BOUND` | $10.0$ | $\text{softmax}(x > 10) \approx 0.99995$ |
| `GRAD_CLAMP_BOUND` | $20.0$ | 梯度幅度边界 |
| `SCALE_CLAMP_BOUND` | $15.0$ | 深度尺度钳位边界 |
| `SPLITTER_TEMP_START` | $1.0$ | 初始温度 |
| `SPLITTER_TEMP_END` | $0.3$ | 最终温度 |
| `LEARNABLE_QUOTA_ENABLED` | `True` | 启用方案 E 配额 |
| `QUOTA_MIN_PER_DEPTH` | $2$ | 每深度最小 token 数 |
| `QUOTA_MIN_RATIO` | $0.02$ | 最小配额比例 |
| `K_COVERAGE_BASE` | $0.25$ | 基础 token 覆盖率 |
| `K_MIN_HARD_LIMIT` | $8$ | 最小绝对 K |

### 关键常量的数学形式

**级别聚合器初始化**：

$$\gamma = \log(e^{1.3133} - 1) \approx 0.26$$

这使得 $\text{softplus}(\gamma) \approx 1.0$，确保初始尺度为中性。

**Gumbel 温度边界**：

$$\tau \geq 0.3$$

低于此阈值，softmax 梯度呈指数消失。该边界源自：

$$\frac{\partial \text{softmax}(x/\tau)}{\partial x} = \frac{\text{softmax}(x/\tau)(1 - \text{softmax}(x/\tau))}{\tau}$$

对于 $\tau < 0.3$，梯度幅度低于 $10^{-5}$。

> **下一章**: [04_positional_embedding.md](04_positional_embedding.md) - 位置编码
