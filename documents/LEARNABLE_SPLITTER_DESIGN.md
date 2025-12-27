# 可学习 Splitter 形式化设计与分析

> **文档版本**: v1.0  
> **创建日期**: 2025-12-27  
> **目标**: 从数学形式化角度推演可学习分割方案，契合 Hilbert Curve ViT 架构

---

## 目录

1. [问题形式化定义](#一问题形式化定义)
2. [当前规则方法的数学缺陷](#二当前规则方法的数学缺陷)
3. [可学习方案推演](#三可学习方案推演)
4. [方案对比验证](#四方案对比验证)
5. [最优实现设计](#五最优实现设计)
6. [实现规范](#六实现规范)

---

## 一、问题形式化定义

### 1.1 Hilbert Curve ViT 的核心约束

**定义 1.1 (Variable Depth Tokenization)**:
给定图像 $I \in \mathbb{R}^{C \times H \times W}$，Variable Depth Tokenization 是一个映射：
$$\mathcal{T}: I \mapsto \{(R_i, d_i)\}_{i=1}^{N}$$

其中:
- $R_i \subset [0,H) \times [0,W)$ 是互不重叠的矩形区域
- $d_i \in \{0, 1, ..., D_{max}\}$ 是四叉树深度
- $\bigcup_i R_i = [0,H) \times [0,W)$ (完全覆盖)
- $N \in [1, 4^{D_{max}}]$ 是自适应 token 数

**定义 1.2 (Hilbert-Quadtree 同构)**:
对于深度 $d$ 的区域 $R$，存在唯一的四叉树路径 $\pi(R) = [q_1, q_2, ..., q_d]$，其中 $q_i \in \{0,1,2,3\}$。

该路径满足 Hilbert 同构:
$$\text{HilbertIdx}(\text{center}(R)) \equiv f(\pi(R)) \mod 4^d$$

**约束 C1 (Hilbert 局部性保持)**:
分割决策必须保持四叉树结构，以确保 LCA (Lowest Common Ancestor) 偏置的有效性：
$$\text{LCA}(R_i, R_j) = \text{CommonPrefix}(\pi(R_i), \pi(R_j))$$

**约束 C2 (2:1 平衡约束)**:
相邻区域的深度差最大为 1：
$$\forall R_i, R_j \text{ adjacent}: |d_i - d_j| \leq 1$$

**约束 C3 (可微分性)**:
对于端到端训练，分割决策必须可微分，允许梯度从分类损失回传到分割参数。

### 1.2 目标函数

**最优分割定义**:
$$\mathcal{T}^* = \arg\min_{\mathcal{T}} \mathbb{E}_{(I, y) \sim \mathcal{D}} \left[ \mathcal{L}_{cls}(f_\theta(\mathcal{T}(I)), y) \right]$$

其中 $f_\theta$ 是 ViT 分类器。

**辅助约束**:
1. **Token 预算**: $\mathbb{E}[N] \leq N_{budget}$
2. **尺度多样性**: $H(\{d_i\}) \geq H_{min}$ (深度分布熵)
3. **计算效率**: $\text{FLOPs}(\mathcal{T}) = O(HW)$

---

## 二、当前规则方法的数学缺陷

### 2.1 复杂度函数分析

**当前复杂度公式**:
$$C(R) = \alpha \cdot \underbrace{\frac{\text{Var}(R)}{\text{Var}(R) + \sigma_0^2}}_{C_{var}} + (1-\alpha) \cdot \underbrace{\frac{G(R)}{G(R) + g_0^2}}_{C_{grad}}$$

**问题 2.1 (饱和效应)**:

设 $\text{Var}(R) = v$，则：
$$\frac{\partial C_{var}}{\partial v} = \frac{\sigma_0^2}{(v + \sigma_0^2)^2}$$

当 $v >> \sigma_0^2$ 时，梯度趋近于 0，导致**区分度丧失**。

**数值验证**:

| $\text{Var}(R)$ | $\sigma_0^2$ | $C_{var}$ | $\frac{\partial C_{var}}{\partial v}$ |
|-----------------|--------------|-----------|---------------------------------------|
| 0.001 | 0.01 | 0.091 | 8.26 |
| 0.01 | 0.01 | 0.500 | 2.50 |
| 0.08 | 0.01 | 0.889 | 0.12 |
| 0.20 | 0.01 | 0.952 | 0.02 |

**结论**: 对于典型图像 ($\text{Var} \approx 0.08$)，$C_{var} \approx 0.89$，远超 $\tau_0 = 0.15$。

### 2.2 阈值-复杂度不匹配定理

**定理 2.1 (不匹配定理)**:
设图像集 $\mathcal{I}$ 的复杂度分布为 $C \sim P_C$。若 $\tau_0 < \text{percentile}(P_C, 5\%)$，则几乎所有区域都会分割到最大深度。

**证明**:
$$P(\text{split at depth } d) = P(C(R^{(d)}) > \tau_0 \cdot \gamma^d)$$

当 $\tau_0 = 0.15$，$\gamma = 0.85$，$\text{median}(C) > 0.8$ 时：
$$P(\text{split}) > 0.95 \quad \forall d \leq D_{max}$$

**推论 2.1**:
当前参数配置下，深度分布熵趋近于 0：
$$H(\{d_i\}) \approx 0 \quad \text{(所有 token 在最大深度)}$$

### 2.3 梯度阻断问题

**问题 2.2 (离散决策不可微)**:

当前分割使用 hard threshold:
$$\text{split}(R, d) = \mathbb{1}[C(R) > \tau_d]$$

指示函数 $\mathbb{1}[\cdot]$ 的梯度为 0 (almost everywhere)，阻断了端到端训练。

---

## 三、可学习方案推演

### 方案 A: 可学习阈值 (Learnable Threshold)

**形式化**:
$$\tau_d^{(l)} = \sigma(\theta_\tau^{(d)}) \quad \text{where } \theta_\tau \in \mathbb{R}^{D_{max}+1}$$

**问题**:
- 仅调整阈值，不改变复杂度计算
- 复杂度饱和问题仍然存在
- 不满足可微分约束 C3

**适用性**: ❌ 不推荐

---

### 方案 B: 可学习归一化参数 (Learnable Normalization)

**形式化**:
$$C_{var}^{(l)} = \frac{\text{Var}(R)}{\text{Var}(R) + \sigma_\theta^2}, \quad \sigma_\theta = \text{softplus}(\theta_\sigma)$$

**分析**:
- 学习 $\sigma_\theta^2$ 可自适应调整归一化尺度
- 但仍然是**单参数**对所有区域，缺乏内容自适应性
- 不满足可微分约束 C3

**改进**:
$$\sigma_\theta^2(R) = \text{MLP}(\text{AvgPool}(R))$$

使归一化常数依赖于区域内容。

**适用性**: ⚠️ 部分有效，但不完整

---

### 方案 C: 可学习复杂度预测器 (Learnable Complexity Predictor)

**形式化**:
$$C_\theta(R) = \sigma(\text{MLP}(\text{Pool}(F, R)))$$

其中:
- $F = \text{Conv}(I)$ 是共享特征图
- $\text{Pool}(F, R)$ 是 ROI-Align 池化
- $\sigma$ 是 sigmoid，确保 $C \in [0, 1]$

**优势**:
1. 端到端学习：从分类损失学习最优复杂度定义
2. 内容自适应：每个区域独立预测
3. 特征共享：与 PatchEmbed 共用 Conv

**可微分决策**:

使用 **Gumbel-Softmax** 实现可微分离散决策：
$$z = \text{GumbelSoftmax}([1-p_{split}, p_{split}], \tau)$$
$$p_{split} = \sigma\left(\frac{C_\theta(R) - \tau_d}{T}\right)$$

**适用性**: ✅ 推荐

---

### 方案 D: 端到端可微分四叉树 (Differentiable Quadtree)

**形式化**:

定义软四叉树结构：
$$\hat{T}(R) = \sum_{d=0}^{D_{max}} w_d(R) \cdot T^{(d)}(R)$$

其中:
- $T^{(d)}(R)$ 是深度 $d$ 的 token 表示
- $w_d(R) = \text{softmax}(\text{MLP}(F[R]))[d]$ 是软深度权重

**问题**:
- 计算量大：需要所有可能深度的表示
- 不满足 Hilbert 局部性约束 C1
- 破坏四叉树结构

**适用性**: ❌ 与 Hilbert Curve ViT 不兼容

---

### 方案 E: 强化学习分割策略 (RL-based Splitting)

**形式化**:
- 状态: $s = (I, \{R_i^{(t)}\}_{i=1}^{N_t})$ (当前区域集合)
- 动作: $a = \{split/keep\}^{N_t}$ 
- 奖励: $r = -\mathcal{L}_{cls}$ (负分类损失)

$$\pi_\theta(a|s) = \prod_i P_\theta(\text{split}_i | R_i, F)$$

**问题**:
- 高方差梯度估计
- 训练不稳定
- 收敛慢

**适用性**: ⚠️ 理论可行但实践困难

---

### 方案 F: 递归可微分分割 (Recursive Differentiable Split)

**形式化**:

定义递归分割函数：
$$\mathcal{S}_\theta(R, d) = \begin{cases}
\{(R, d)\} & \text{if } d = D_{max} \text{ or } p_{stop}(R, d) > 0.5 \\
\bigcup_{q=0}^{3} \mathcal{S}_\theta(R_q, d+1) & \text{otherwise}
\end{cases}$$

其中:
$$p_{stop}(R, d) = \sigma\left(\frac{\tau_d - C_\theta(R)}{T}\right)$$

**Straight-Through Estimator (STE)**:

前向传播使用 hard decision，反向传播使用 soft gradient：
$$z_{hard} = \mathbb{1}[p_{stop} > 0.5]$$
$$\frac{\partial \mathcal{L}}{\partial \theta} = \frac{\partial \mathcal{L}}{\partial z_{hard}} \cdot \frac{\partial p_{stop}}{\partial \theta}$$

**适用性**: ✅ **最契合 Hilbert Curve ViT**

---

## 四、方案对比验证

### 4.1 计算复杂度分析

| 方案 | 训练时间 | 推理时间 | 内存 | 与 ViT 集成 |
|------|----------|----------|------|-------------|
| A: 可学习阈值 | O(HW) | O(HW) | O(1) | 简单 |
| B: 可学习归一化 | O(HW) | O(HW) | O(D) | 中等 |
| **C: 复杂度预测器** | O(HW + ND) | O(HW + ND) | O(ND) | 优秀 |
| D: 软四叉树 | O(HW · 4^D) | O(HW · 4^D) | O(N · D · d) | 差 |
| E: RL 策略 | O(HW · T) | O(HW) | O(ND) | 复杂 |
| **F: 递归可微分** | O(HW + ND) | O(HW + ND) | O(ND) | 优秀 |

### 4.2 Hilbert 约束兼容性

| 方案 | C1: 局部性 | C2: 2:1 平衡 | C3: 可微分 |
|------|------------|--------------|------------|
| A | ✅ | ✅ | ❌ |
| B | ✅ | ✅ | ❌ |
| **C** | ✅ | ✅ | ✅ (Gumbel) |
| D | ❌ | ❌ | ✅ |
| E | ✅ | ✅ | ⚠️ (高方差) |
| **F** | ✅ | ✅ | ✅ (STE) |

### 4.3 理论收敛性

**定理 4.1 (方案 C/F 收敛性)**:

设 $C_\theta$ 是 Lipschitz 连续的复杂度预测器，温度 $T > 0$。则梯度估计的方差有界：
$$\text{Var}[\nabla_\theta \mathcal{L}] \leq \frac{L^2}{T^2} \cdot \text{Var}[C_\theta]$$

**证明略**。关键点：温度 $T$ 控制 soft decision 的锐度，更低的 $T$ 提供更精确的离散近似但更高的方差。

**推荐策略**: 温度退火 $T(t) = T_0 \cdot \exp(-\beta t)$

---

## 五、最优实现设计

基于上述分析，推荐采用 **方案 F (递归可微分分割)** 与 **方案 C (复杂度预测器)** 的结合。

### 5.1 架构概览

```
                    ┌──────────────────────────────────────────┐
                    │           LearnableSplitter              │
                    └──────────────────────────────────────────┘
                                        │
            ┌───────────────────────────┼───────────────────────────┐
            │                           │                           │
            ▼                           ▼                           ▼
    ┌───────────────┐           ┌───────────────┐           ┌───────────────┐
    │ SharedFeature │           │ ComplexityNet │           │  SplitPolicy  │
    │   Extractor   │           │   (per-ROI)   │           │  (recursive)  │
    └───────────────┘           └───────────────┘           └───────────────┘
            │                           │                           │
            │ F ∈ R^{B×D×H'×W'}        │ C_θ(R) ∈ [0,1]            │ {(R_i, d_i)}
            │                           │                           │
            └───────────────────────────┴───────────────────────────┘
                                        │
                                        ▼
                              ┌───────────────────┐
                              │  HilbertNative    │
                              │  PatchEmbed       │
                              │  (ROI-Align)      │
                              └───────────────────┘
                                        │
                                        ▼
                              Tokens ∈ R^{B×N×D}
```

### 5.2 数学形式化

**特征提取**:
$$F = \text{SharedConv}(I) \in \mathbb{R}^{D \times H' \times W'}$$

与 `HilbertNativePatchEmbed` 共享权重。

**复杂度预测**:
$$C_\theta(R) = \sigma\left(\text{MLP}\left(\text{ROI-Pool}(F, R)\right)\right)$$

其中 MLP: $\mathbb{R}^D \to \mathbb{R}^1$，使用 2 层结构。

**递归分割**:
```python
def split(R, d):
    if d == D_max or should_stop(R, d):
        return [(R, d)]
    else:
        return [split(R_q, d+1) for q in range(4)].flatten()

def should_stop(R, d):
    p_stop = sigmoid((tau[d] - C_theta(R)) / T)
    if training:
        return sample_gumbel_bernoulli(p_stop)
    else:
        return p_stop > 0.5
```

**可学习参数**:
- $\theta_{conv}$: 共享卷积权重 (与 PatchEmbed 共享)
- $\theta_{mlp}$: 复杂度 MLP 权重
- $\tau = [\tau_0, \tau_1, ..., \tau_{D_{max}}]$: 可学习阈值

### 5.3 训练目标

$$\mathcal{L} = \mathcal{L}_{cls} + \lambda_1 \mathcal{L}_{entropy} + \lambda_2 \mathcal{L}_{budget}$$

**分类损失**:
$$\mathcal{L}_{cls} = \text{CrossEntropy}(f_\theta(\mathcal{T}(I)), y)$$

**熵正则化** (鼓励尺度多样性):
$$\mathcal{L}_{entropy} = -\sum_{d=0}^{D_{max}} p(d) \log p(d)$$
$$p(d) = \frac{|\{i: d_i = d\}|}{N}$$

**预算约束**:
$$\mathcal{L}_{budget} = \text{ReLU}(N - N_{budget})^2$$

### 5.4 Gumbel-Softmax 实现细节

**前向传播**:
$$g_0, g_1 \sim \text{Gumbel}(0, 1)$$
$$\text{logits} = [\log(1 - p_{stop}), \log(p_{stop})]$$
$$y = \text{softmax}((\text{logits} + [g_0, g_1]) / \tau)$$

**Straight-Through (推理时)**:
$$z_{hard} = \text{one\_hot}(\arg\max(y))$$
$$z_{ST} = z_{hard} - y.\text{detach}() + y$$

这确保前向使用 hard decision，反向使用 soft gradient。

### 5.5 温度调度

$$T(t) = T_{start} \cdot \left(\frac{T_{end}}{T_{start}}\right)^{t/T_{total}}$$

推荐值:
- $T_{start} = 1.0$ (开始时较软)
- $T_{end} = 0.1$ (结束时趋近 hard)
- 预热阶段固定 $T = T_{start}$

---

## 六、实现规范

### 6.1 类设计

```python
class LearnableSplitter(nn.Module):
    """
    可学习四叉树分割器。
    
    数学形式化:
        C_θ(R) = σ(MLP(Pool(F, R)))
        p_stop(R, d) = σ((τ_d - C_θ(R)) / T)
        
    与 HilbertNativePatchEmbed 集成:
        1. 共享特征提取器 (SharedConv)
        2. 复杂度网络使用相同特征图
        3. 分割结果直接传给 PatchEmbed
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        max_depth: int = 4,
        hidden_dim: int = 64,
        num_thresholds: int = 5,  # max_depth + 1
        temperature: float = 1.0,
        use_gumbel: bool = True,
    ):
        ...
    
    def forward(
        self,
        features: Tensor,  # [B, D, H', W'] from SharedConv
        image_size: Tuple[int, int],
    ) -> List[SplitResult]:
        """
        可微分分割。
        
        Returns:
            List[SplitResult]: 每个图像的分割结果
        """
        ...
    
    def _recursive_split(
        self,
        region: Region,
        depth: int,
        features: Tensor,
        batch_idx: int,
    ) -> List[SplitToken]:
        """递归分割单个区域。"""
        ...
    
    def _compute_complexity(
        self,
        features: Tensor,
        region: Region,
    ) -> Tensor:
        """
        计算区域复杂度。
        
        Returns:
            Tensor: scalar in [0, 1]
        """
        ...
    
    def _should_stop(
        self,
        complexity: Tensor,
        depth: int,
    ) -> Tuple[bool, Tensor]:
        """
        决定是否停止分割。
        
        Returns:
            (decision, probability) 用于 loss 计算
        """
        ...
```

### 6.2 与 Tokenizer 集成

```python
class StreamingFractalTokenizerV4(BaseTokenizer):
    """支持可学习分割的 Tokenizer V4。"""
    
    def __init__(
        self,
        image_size: int = 224,
        d_model: int = 256,
        max_depth: int = 4,
        learnable_split: bool = True,  # 新参数
        ...
    ):
        super().__init__()
        
        # 共享特征提取
        self.patch_embed = HilbertNativePatchEmbed(...)
        
        # 分割器选择
        if learnable_split:
            self.splitter = LearnableSplitter(
                feature_dim=d_model,
                max_depth=max_depth,
            )
        else:
            self.splitter = BalancedGreedySplitter(...)
    
    def forward(self, images: Tensor) -> TokenizerOutput:
        # 1. 提取共享特征
        features = self.patch_embed.shared_conv(images)
        
        # 2. 分割 (可学习或规则)
        if isinstance(self.splitter, LearnableSplitter):
            split_results = self.splitter(features, images.shape[-2:])
            # 收集分割概率用于 loss
            self._split_probs = self.splitter.get_split_probs()
        else:
            split_results = self.splitter.split_batch(images)
        
        # 3. Patch Embedding
        tokens, levels_info = self.patch_embed(images, split_results)
        
        return TokenizerOutput(tokens=tokens, levels_info=levels_info)
```

### 6.3 训练循环修改

```python
# 在 train_step 中
def train_step(model, images, labels, config):
    outputs = model(images)
    
    # 主分类损失
    loss_cls = F.cross_entropy(outputs.logits, labels)
    
    # 可学习分割损失
    if hasattr(model.tokenizer, '_split_probs'):
        split_probs = model.tokenizer._split_probs
        
        # 熵损失 (鼓励多尺度)
        loss_entropy = compute_entropy_loss(split_probs)
        
        # 预算损失
        n_tokens = outputs.tokens.shape[1]
        loss_budget = F.relu(n_tokens - config.token_budget) ** 2
        
        loss = (
            loss_cls 
            + config.lambda_entropy * loss_entropy
            + config.lambda_budget * loss_budget
        )
    else:
        loss = loss_cls
    
    return loss
```

### 6.4 超参数推荐

| 参数 | 推荐值 | 范围 | 说明 |
|------|--------|------|------|
| `hidden_dim` | 64 | [32, 128] | 复杂度 MLP 隐层维度 |
| `temperature_start` | 1.0 | [0.5, 2.0] | 初始温度 |
| `temperature_end` | 0.1 | [0.05, 0.2] | 最终温度 |
| `warmup_epochs` | 5 | [3, 10] | 规则分割预热 |
| `lambda_entropy` | 0.1 | [0.01, 0.5] | 熵正则化权重 |
| `lambda_budget` | 0.01 | [0.001, 0.1] | 预算约束权重 |
| `token_budget` | 64 | [32, 128] | 目标 token 数 |

---

## 附录: 验证计划

### A.1 消融实验

1. **规则 vs 可学习**: 固定参数对比准确率
2. **温度调度**: 不同调度策略的收敛速度
3. **损失权重**: $\lambda_1, \lambda_2$ 敏感性分析
4. **特征共享**: 是否共享特征对精度的影响

### A.2 可视化

1. 学习到的复杂度分布 $C_\theta(R)$
2. 分割决策可视化 (按深度着色)
3. 注意力模式对比 (规则 vs 可学习)
4. 收敛曲线 (分类损失 + 熵损失 + 预算损失)

### A.3 预期改进

| 指标 | 规则方法 | 可学习方法 | 预期提升 |
|------|----------|------------|----------|
| Top-1 准确率 | 49.41% | ~52% | +2.6% |
| 深度分布熵 | ~0 | ~1.0 | +∞ |
| Token 方差 | ~0 | ~15 | 可控 |

---

*文档结束。*
