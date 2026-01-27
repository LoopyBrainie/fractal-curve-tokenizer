# 第三章：分形分词器

## 3.1 概述

`StreamingFractalTokenizerV3` 通过自适应四叉树分割和 Hilbert 曲线重排序实现**变深度分词**。它采用 **Gumbel-Top-K（方案 D）** 机制和 **可学习配额分配（方案 E）**，实现 ~K/N 梯度覆盖率（约 37.6%）和并行执行，取代了早期的基于 BFS 的方法。

**效率说明**：~40× 的计算减少来自于 token 数量减少（$N_{V3} \approx 32$ vs $N_{ViT} \approx 307K$），而非渐近复杂度变化。注意力保持 $O(N^2 \cdot D)$，但 $N$ 减少了 ~40×。

**并行约束**：每个 token 的完整动态深度与 GPU SIMT 并行性不兼容。当前实现为提高并行效率使用批次级固定深度。

---

## 3.2 数学形式化

### 3.2.1 分词管道

$$I \xrightarrow{\text{SharedConv}} F \xrightarrow{\text{ParallelEval}} \{S_i, \text{logits}_i\}_{i=1}^{N_{cand}} \xrightarrow{\text{TopK}} \{R_j\}_{j=1}^{K} \xrightarrow{\text{Consist}} \{T_k\} \xrightarrow{\text{Sort}}$$

其中：
- $I \in \mathbb{R}^{C \times H \times W}$：输入图像
- $N_{cand}$：候选四叉树区域总数（$\sum_{d=0}^{D} 4^d$）（例如 D=3 时为 85）
- $R_j$：选中的区域
- $T_k$：最终一致的 token 集合（无重叠）

### 3.2.2 Gumbel-Top-K 决策（方案 D）

我们不逐区域进行阈值决策，而是并行评估所有候选并选择得分最高的 Top-K。

**决策 Logits**（I30-4 更新）：

$$\text{logits}_i = \text{MLP}(\text{ROI}(F, R_i)) + b_{explore} + \beta \cdot \gamma^{d_i} - \tau_{d_i}$$

其中：
- $\text{MLP}(\text{ROI}(F, R_i))$：可学习复杂度得分（I30-4）
- $\tau_{d_i}$：每深度可学习阈值
- $b_{explore}$：退火探索偏置
- $\gamma^{d_i}$：深度惩罚项（可选）

> **注意 (I30-4)**：对数补偿偏置 $b_{log}(d_i) = \log(N_{total} / N_{d_i})$ 已被移除，转而使用可学习配额机制（方案 E）。

**随机选择**：

$$z_i = \text{logits}_i + g_i, \quad g_i \sim \text{Gumbel}(0, 1)$$
$$\text{selected\_indices} = \text{TopK}(\{z_i/\tau\}_{i=1}^{N_{cand}}, K)$$

该形式通过直通估计器（STE）为**选中的**候选提供梯度，未选中的候选收到衰减的梯度（约 20× 减少）。

### 3.2.3 可学习配额分配（方案 E）

当 `LEARNABLE_QUOTA_ENABLED=True` 时，模型学习跨深度的最优 token 配额分布。

**配额计算**：

$$\pi_d = \text{softmax}(\phi_d), \quad \phi \in \mathbb{R}^{D+1}$$
$$K_d = \text{round}(\pi_d \cdot K_{total})$$
$$K_d = \max(K_d, K_{min})$$

其中：
- $\phi_d$：深度 $d$ 的可学习 logit
- $\pi_d$：深度的学习概率分布
- $K_d$：分配给深度 $d$ 的配额
- $K_{min}$：每深度的最小配额（默认：2）

**层次化 Top-K 选择**：

对于每个深度 $d$，从该深度的区域中选择恰好 $K_d$ 个 token，然后合并各深度的结果。

### 3.2.4 树一致性

原始 Top-K 选择可能违反树结构（例如，同时选择父节点和其子节点）。我们通过向量化操作强制执行一致性：

$$\text{consistent}(i) \iff i \in \text{TopK} \land \forall c \in \text{children}(i), c \notin \text{TopK}$$

这确保如果选择了父节点，则忽略其子节点，保持有效的划分（或其子集）。

### 3.2.5 形状-尺度编码器（I31）

为增强 token 表示，形状-尺度编码器捕获区域几何：

**长宽比**：

$$r = \log(w/h) \quad \text{（对数变换以保证对称性）}$$

**归一化面积**：

$$s = \frac{w \cdot W_{patch}}{W_{total} \cdot H_{total}}$$

**门控组合**：

$$g = \sigma(\text{MLP}([r; s]))$$
$$E_{shape}(R) = \text{MLP}([r \cdot g; s \cdot (1-g)])$$

---

## 3.3 分割方案

### 3.3.1 方案 D：Gumbel-Top-K（基础）

| 特性 | 描述 |
|:--------|:------------|
| **并行性** | 100%（所有区域在一个批次中评估） |
| **梯度流** | ~37.6%（K/N）- STE 为选中的 token 提供梯度，未选中的衰减 |
| **Hilbert 局部性** | 100%（严格遵守 Hilbert 曲线排序） |
| **复杂度** | $O(N_{cand})$ 并行评估 |

### 3.3.2 方案 E：可学习配额（扩展）

| 特性 | 描述 |
|:--------|:------------|
| **自适应预算** | 学习跨深度的最优 token 分布 |
| **深度平衡** | 防止过度分配到浅层或深层 token |
| **可解释性** | $\pi_d$ 揭示模型偏好的深度分布 |

### 3.3.3 方案比较

| 方案 | 并行性 | 梯度 | 配额 | 状态 |
|:-------|:------------|:---------|:------|:-------|
| A (BFS) | ~30% | 部分 | 固定 | 已废弃 |
| B (松弛) | ~60% | 部分 | 固定 | 已废弃 |
| C (固定预算) | 100% | STE | 固定 | 已废弃 |
| **D (Gumbel-Top-K)** | **100%** | **STE** | **固定** | **当前** |
| **E (可学习配额)** | **100%** | **STE** | **学习** | **推荐** |

---

## 3.4 Token 嵌入

### 3.4.1 HilbertNativePatchEmbed

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

### 3.4.2 ROI-Align

为跨区域边界的平滑梯度流：

$$\text{ROIAlign}(F, R) = \text{BilinearInterpolate}(F, \text{SamplePoints}(R))$$

这避免了整数舍入的量化伪影。

### 3.4.3 形状-尺度增强（I31-2）

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

---

## 3.6 实现

### 类：StreamingFractalTokenizerV3

```python
class StreamingFractalTokenizerV3(BaseTokenizer):
    def __init__(
        self,
        image_size: int = 224,
        d_model: int = 384,
        base_patch_size: int = 4,
        min_patch_size: int = 4,
        max_depth: Optional[int] = None,
        K_min: int = 8,
        K_max: int = 64,
        splitter_dropout: float = 0.15,
    ):
        """
        参数:
            image_size: 输入图像大小
            d_model: 模型维度
            base_patch_size: 级别 0 的基础 patch 大小
            min_patch_size: 目标最小 patch 大小
            max_depth: 最大四叉树深度（如果为 None 则自动计算）
            K_min: 最小 token 数量（I23-2）
            K_max: 最大 token 数量
            splitter_dropout: 分隔器 MLP 的 dropout
        """
        ...
```

### 核心逻辑：GumbelTopKSplitter

```python
class GumbelTopKSplitter(nn.Module):
    def forward(self, features: Tensor) -> GumbelTopKResult:
        """
        返回:
            GumbelTopKResult 包含：
            - regions: [M, 4] 选中的区域坐标
            - depths: [M] 区域深度
            - batch_indices: [M] 批次索引
            - hilbert_indices: [M] Hilbert 曲线索引
            - selected_mask: [B, N] STE 梯度掩码
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

```python
from vit_pytorch import StreamingFractalTokenizerV3

tokenizer = StreamingFractalTokenizerV3(
    image_size=224,
    d_model=384,
    base_patch_size=4,
    K_min=8,
    K_max=64,
)

# 返回带有对齐 token 和深度信息的 TokenizerOutput
output = tokenizer.tokenize(images)

# 访问结果
tokens = output.tokens          # [B, N, D]
levels_info = output.levels     # [B, N, max_depth+1]
regions = output.regions        # [B, N, 4] - 区域边界
split_probs = output.split_probs  # [B, N] - 选择置信度
```

### 高级：带可学习配额的方案 E

```python
from vit_pytorch import GumbelTopKSplitter

splitter = GumbelTopKSplitter(
    dim=384,
    max_depth=6,
    K_total=32,
    learnable_quota=True,    # 启用方案 E
    quota_init_logits=None,  # 自动初始化
)

# 训练后，检查学习到的配额分布
quota_probs = F.softmax(splitter.quota_logits, dim=0)
# quota_probs[d] = 将 token 分配给深度 d 的概率
```

---

## 3.8 温度退火

Gumbel-Softmax 温度控制探索与利用之间的权衡：

$$\tau(t) = \tau_{start} \cdot \left(\frac{\tau_{end}}{\tau_{start}}\right)^{t / T_{total}}$$

**默认调度**：

| 参数 | 值 |
|:----------|:------|
| $\tau_{start}$ | 1.0 |
| $\tau_{end}$ | 0.5 |
| 调度 | 指数衰减 |

---

## 3.9 常量参考

所有数值稳定性常量都集中在 `constants.py` 中：

| 常量 | 值 | 数学基础 |
|:---------|:------|:-------------------|
| `GUMBEL_EPSILON` | $1e-8$ | FP16 安全下界 |
| `LOG_EPSILON` | $1e-8$ | 对数稳定性 |
| `DIVISION_EPSILON` | $1e-8$ | 除法稳定性 |
| `PROB_EPSILON` | $1e-5$ | 概率钳位 |
| `LOGIT_CLAMP_BOUND` | $50.0$ | $\text{softmax}(x > 50) \approx \text{one-hot}$ |
| `GRAD_CLAMP_BOUND` | $20.0$ | $P(\|grad\| > 20) \approx 10^{-6}$ |
| `TEMPERATURE_MIN` | $0.3$ | 防止梯度饱和 |

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
