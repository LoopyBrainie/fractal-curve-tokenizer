# 形式分析：Hilbert Splitter 对比研究

---

## 零、从形式化到计算验证的批判性分析

### 0.1 分析框架

本文对以下两种splitter方案进行批判性对比分析：

| 方案 | 实现 | 代码位置 |
|-----|-----|---------|
| **H1SS** | HilbertOptimalSplitter | `hilbert_optimal_splitter.py` |
| **H-Entmax** | HilbertOrderedEntmaxSplitter | `hilbert_entmax.py` |

**分析维度**:
1. **形式正确性**: 公理一致性、数学承诺兑现
2. **计算验证**: 梯度覆盖率、局部性保持、树一致性
3. **工程权衡**: 参数效率、复杂度、可维护性
4. **进化方向**: 未来发展的最优路径

---

## 一、形式正确性批判

### 1.1 H1SS 公理体系分析

**A1 (Locality)**: Conv1D Hilbert 流形卷积
```
数学承诺: J(S) = mean(|h(s_i) - h(s_{i+1})|) 理论最优
实现: HilbertDistanceDecayConv1D (I167-1)
问题: 固定距离衰减权重 w_d = 1/(|d|+1) 是否是最优的局部性先验?
```

**验证方法**: 
- Locality Efficiency = Selection Locality / Oracle Locality
- 当前实现返回该比值，理想情况应接近 1.0

**批判**:
- A1 声称"强制邻域交互"，但固定衰减权重缺乏适应性
- 实际局部性依赖 Entmax 的稀疏选择，而非卷积本身

---

**A2 (Determinism)**: 移除 Gumbel，纯 softmax/entmax
```
数学承诺: 无随机性，相同输入→相同输出
实现: entmax_bisect + STE
问题: Entmax 本身是否完全确定性?
```

**验证方法**:
```python
# Determinism Score = IOU(train_selections, eval_selections)
compute_determinism_score(train_selected, eval_selected)
```

**批判**:
- Entmax 是确定性的 (给定相同输入)，✓
- 但训练过程中的 dropout/batchnorm 仍引入不确定性
- **真正的问题**: Entmax 的 α 调度在训练/评估模式下一致吗?

---

**A3 (Gradient)**: Entmax 稀疏激活
```
数学承诺: 选中/非选中梯度差异显著
实现: alpha ∈ [1.2, 1.5] 动态调度
问题: α=1.5 真的产生稀疏梯度吗?
```

**关键发现 (I107)**:
```python
# 测试结果: alpha=1.5 产生 0% 非零输出
# 这导致梯度无法回传!
# 解决: α 从 1.2 预热到 1.49 (永远不达到 1.5)
```

**批判**:
- 这暴露了 H1SS 的核心矛盾: **稀疏性 ↔ 梯度流**
- α=1.5 的"数学最优稀疏"在实际中不可用
- 必须牺牲理论最优以保证训练稳定

---

**A4 (Tree Consistency)**: 软约束 z_parent -= λ × max(z_children)
```
数学承诺: 父节点未被选中时，惩罚其高分
实现: 课程学习 λ + 可学习 log_lambda 残差
问题: 这个约束真的保证树一致性吗?
```

**验证方法**:
```python
# Tree Consistency = 1 - violations / total_parents
compute_tree_consistency(selected_mask, parent_indices, children_matrix)
```

**批判**:
- 这是一个**软约束**，不是硬约束
- 如果子节点分数始终高于父节点，约束永远无法生效
- **潜在问题**: λ 课程学习可能导致训练早期树一致性很差

---

**A5 (Consistency)**: 单次 Entmax 投影
```
数学承诺: E[|S|] = K, Var → 0
实现: Entmax + TopK
问题: Entmax 的期望真的精确等于 K 吗?
```

**计算验证**:
```python
# α-Entmax 性质: 不保证精确的 K 选择
# 实际: probs.sum() ≈ K，但可能有偏差
```

**批判**:
- H1SS 使用 **TopK after Entmax**，不是真正的"Entmax 选择 K"
- Entmax 提供稀疏概率，然后 TopK 选择 K 个
- 这与 A5 声明的数学形式**不完全一致**

---

### 1.2 H-Entmax 形式分析

**核心设计**:
```python
scores = entmax_1.5(MLP(features) + complexity)
selected = TopK(scores, k)
```

**优点**:
1. **数学清晰**: 明确的稀疏性保证
2. **梯度透明**: 100% 覆盖率 (vs H1SS 的 ~37% 早期)
3. **实现简洁**: ~10 个参数 vs H1SS 的 100+ 参数

**缺点**:
1. **无深度控制**: 深度阈值是简单的按层缩放
2. **无树约束**: 可能选择子节点而不选父节点
3. **无 K 估计**: 需要外部指定 K

---

## 二、计算验证对比

### 2.1 梯度覆盖率 (Gradient Coverage)

| 方法 | 梯度覆盖率 | 原因 |
|-----|----------|-----|
| Gumbel-TopK | ~37% | STE 只在选中token上传递梯度 |
| H-Entmax (α=1.5) | **100%** | Entmax 是完全可微的 |
| H1SS (早期 α=1.2) | **100%** | softmax 完全可微 |
| H1SS (晚期 α=1.49) | ~60-80% | 稀疏 Entmax 仍有部分梯度 |

**关键问题**: H1SS 声称"A3 梯度优势"，但实际上：
- 早期用 softmax → 无稀疏性
- 晚期用 entmax → 有稀疏性但梯度减少

**这是一个根本矛盾**: 要稀疏性就要牺牲梯度，要梯度就要放弃稀疏性。

---

### 2.2 局部性保持 (Locality Preservation)

**测量方法**:
```python
J(S) = mean(|h(s_i) - h(s_{i+1})|)
```

**两种实现的局部性来源**:

| 组件 | H1SS | H-Entmax |
|-----|-----|---------|
| 序列排序 | Hilbert | Hilbert |
| 特征混合 | DistanceDecay Conv1D | Standard Conv1D |
| 选择机制 | Entmax TopK | Entmax TopK |
| 局部性强化 | SDS 正则化 (可选) | 无 |

**批判**:
- H1SS 的 `HilbertDistanceDecayConv1D` 使用**固定衰减**，没有利用梯度学习最优邻域权重
- H-Entmax 的 `HilbertLocalComplexity` 也是**固定 Conv1D 权重**
- **两者都没有真正学习到数据依赖的局部性模式**

---

### 2.3 树一致性 (Tree Consistency)

**H1SS 的树约束**:
```python
logits = logits - λ * max_child_logits  # z_parent -= λ × max(z_children)
```

**问题分析**:
1. **时序问题**: λ 课程从 0.05 增长到 0.3，在训练早期树一致性很差
2. **幅度问题**: λ=0.3 足够大吗? 如果 max(z_children) = 10 而 z_parent = 5，约束后 z_parent = 2，仍可能被 TopK 选中
3. **梯度冲突**: 树约束与 Entmax 选择可能冲突

**H-Entmax**: 完全无树约束，可能选择深度=4 的子节点而不选深度=2 的父节点

---

## 三、工程权衡批判

### 3.1 参数效率

| 指标 | H1SS | H-Entmax | 比率 |
|-----|-----|---------|-----|
| 可学习参数 | ~10K | ~2K | 5x |
| 配置参数 | ~20 | ~4 | 5x |
| 辅助损失数 | 3+ | 0 | - |
| 调度器组件 | 5+ | 0 | - |

**H1SS 的复杂性来源**:
```python
# 1. Entmax alpha 调度器
entmax_alpha_init = 1.2
entmax_alpha_warmup = 1.49  # V4
entmax_schedule_epochs = 25

# 2. K 课程学习
_current_K = K_min  # 随 epoch 增长

# 3. λ 课程学习
_log_lambda = nn.Parameter(torch.tensor(0.0))
_lambda_schedule_epochs = 20

# 4. 温度调度
temperature_init = 1.0
temperature_min = 0.3

# 5. SDS 正则化 (可选)
use_sds_regularization = False

# 6. 距离衰减卷积选择
use_distance_decay_conv = True
```

**批判**: H1SS 是一个"超参数艺术作品"，而非"数学公理系统"

---

### 3.2 可维护性与可复现性

**问题 1**: I107 修复了什么?
```python
# H1SS 代码注释:
# I107: 从 1.5 改为 1.2，防止 alpha=1.5 导致 Entmax 硬截断
# 测试结果: alpha=1.5 产生 0% 非零输出
```
这暴露了:**α=1.5 的数学承诺与实际行为不符**

**问题 2**: 多个"I"标记表明什么?
- I164-1: 动态λ调整
- I167-1: 距离衰减卷积
- I167-4: SDS 正则化
- I-NAN: 密度场初始化

**每一个"I"标记都代表一次"修复"或"增强"，暗示原始设计存在缺陷**

---

## 四、核心矛盾总结

### 矛盾 1: 稀疏性 ↔ 梯度流

| 阶段 | H1SS 选择 | 结果 |
|-----|---------|-----|
| 早期 (α=1.2) | Softmax | 100% 梯度，无稀疏性 |
| 中期 (α→1.49) | 过渡态 | 梯度减少，稀疏增加 |
| 晚期 (α≈1.49) | Entmax | 稀疏最大化，梯度最小化 |

**这是一个不可能三角**:
```
        稀疏性
          ▲
         / \
        /   \
       /  ✗  \
      /       \
    梯度 ◁────▷ 树一致性
```

**任何方案只能同时优化两个**

---

### 矛盾 2: 复杂调度 ↔ 理论优雅

H1SS 声称基于"6条数学公理"，但实际上：
- 6 个调度器管理超参数
- 每个调度器有自己的课程学习策略
- 调度器之间可能冲突

**这与"公理系统"的概念相悖**: 真正的公理系统应该是稳定、可预测的

---

## 五、进化方向建议

### 方向 A: 简化 H1SS → H-Entmax++

**保留**:
- Hilbert 排序的局部性优势 ✓
- Entmax 的稀疏性 ✓

**移除**:
- 树约束 (效果存疑)
- SDS 正则化 (I167-4)
- 距离衰减卷积 (固定权重无学习能力)
- K 课程学习 (可用 Gumbel 替代)

**增加**:
- 可学习的邻域权重 (而非固定衰减)
- 数据依赖的深度控制

---

### 方向 B: 统一 H1SS 与 H-Entmax

**观察**: H1SS 的 `HilbertDistanceDecayConv1D` 与 H-Entmax 的 `HilbertLocalComplexity` 本质相同

**统一路径**:
```python
class UnifiedHilbertSplitter(nn.Module):
    def __init__(self, ...):
        # 使用 H-Entmax 的 Conv1D 架构
        self.complexity_extractor = HilbertLocalComplexity(...)
        
        # 使用 H1SS 的树约束 (但简化)
        self.tree_constraint_weight = 0.1
        
        # 使用 H1SS 的 Entmax 调度 (但简化)
        self.alpha_schedule = [1.2, 1.5]  # 两阶段
```

---

### 方向 C: 理论验证优先

**建议**: 在继续工程优化之前，先回答这些问题:

1. **Locality Efficiency 实际测量值是多少?**
   - H1SS vs H-Entmax vs 随机选择
   - 如果差异不大，为什么需要复杂的局部性机制?

2. **树约束真的有效吗?**
   - 对比: 有/无树约束的 Tree Consistency
   - 如果 λ 课程学习效果不佳，为什么不直接用硬约束?

3. **α 调度的敏感性分析**
   - 如果 α=1.49 和 α=1.5 效果差异很小，为什么需要精确调度?

---

## 六、结论

### H1SS 的价值:
- 完整的理论框架 (6 公理)
- 全面的诊断指标
- 多层次调度器

### H1SS 的问题:
- 数学声明与实际行为不符 (α=1.5 的硬截断)
- 过度工程化 (20+ 可调组件)
- 调度器相互干扰

### H-Entmax 的价值:
- 简洁可证明
- 100% 梯度覆盖率
- 易于分析

### H-Entmax 的问题:
- 无深度控制
- 无树约束
- 无 K 估计

### 建议:

**短期 (V1)**: 采用 H-Entmax 作为基础，添加:
- 可学习的邻域卷积权重
- 简单的深度控制机制

**长期 (V2)**: 重新设计 H1SS:
- 移除所有调度器
- 用硬约束替代软约束
- 保持理论框架，简化实现

---

## 附录 A: 数学推导补充

### A.1 Entmax 的稀疏性分析

**α-Entmax 定义**:
$$
\text{entmax}_\alpha(z) = \arg\max_{p \in \Delta^{n-1}} (p^T z + H_\alpha(p))
$$

其中 $H_\alpha(p) = \frac{1}{1-\alpha} \log \sum_i p_i^\alpha$ 是 Rényi 散度

**性质分析**:

| α 值 | 行为 | 数学性质 |
|-----|-----|---------|
| α → 1 | softmax | 完全可微，稠密输出 |
| α = 1.5 | sparsemax | 稀疏输出，顶层梯度可能为零 |
| α → ∞ | argmax | 极度稀疏，梯度消失 |

**关键发现**: 当 α > 1 时，entmax 的解是**稀疏的**：如果某个元素的概率为0，其对应的梯度也为0。

**这解释了 I107 的发现**: α=1.5 产生0%非零输出 → 100%梯度消失

---

### A.2 树约束的数学分析

**H1SS 树约束**:
$$
z_{parent} \leftarrow z_{parent} - \lambda \cdot \max_{c \in children(parent)} z_c
$$

**问题**: 这是一个**软约束**，不是硬约束

**设**: $z_{parent} = 5$, $\max(z_{children}) = 10$, $\lambda = 0.3$

**结果**: $z_{parent}^{new} = 5 - 0.3 \times 10 = 2$

**但**: 如果其他父节点 $z_{other} = 1.5$，而 TopK 选择 K=10 个：
- $z_{parent}^{new} = 2$ 仍可能被选中
- $z_{other} = 1.5$ 可能被淘汰

**结论**: 树约束**不能保证**树一致性，只能**鼓励**树一致性

---

### A.3 局部性效率的形式化

**Selection Locality**:
$$
J(S) = \frac{1}{K-1} \sum_{i=1}^{K-1} |h(s_i) - h(s_{i+1})|
$$

其中 $h(s_i)$ 是 Hilbert 曲线上的位置索引

**Oracle Locality** (理论最优):
$$
J_{oracle} = 1.0
$$

**Locality Efficiency**:
$$
\eta_{locality} = \frac{J(S)}{J_{oracle}} \in [0, 1]
$$

**问题**: 固定衰减权重 $w_d = 1/(|d|+1)$ 是否是最优的?

**反例**: 如果特征显示应该捕获长程依赖 (如全局上下文)，固定衰减会阻止这一点

---

## 附录 B: 计算验证框架

### B.1 验证指标体系

```python
class SplitterMetrics:
    """Splitter 评估指标"""
    
    def __init__(self, splitter, dataset):
        self.splitter = splitter
        self.dataset = dataset
        
    def compute_all(self):
        return {
            # 梯度指标
            'gradient_coverage': self.gradient_coverage(),
            'gradient_magnitude_mean': self.gradient_magnitude_mean(),
            'gradient_magnitude_std': self.gradient_magnitude_std(),
            
            # 局部性指标
            'locality_score': self.locality_score(),
            'locality_efficiency': self.locality_efficiency(),
            'oracle_gap': self.oracle_gap(),
            
            # 树一致性指标
            'tree_consistency': self.tree_consistency(),
            'tree_violations_count': self.tree_violations_count(),
            'depth_distribution': self.depth_distribution(),
            
            # 选择稳定性指标
            'selection_iou_mean': self.selection_iou_mean(),
            'selection_iou_std': self.selection_iou_std(),
            'selection_stability': self.selection_stability(),
            
            # 效率指标
            'active_ratio': self.active_ratio(),
            'entropy_per_token': self.entropy_per_token(),
        }
```

---

### B.2 梯度覆盖率验证

```python
def gradient_coverage(model, features, image_size):
    """
    计算梯度覆盖率
    
    定义: coverage = |{i : |∂L/∂z_i| > τ}| / N
    
    τ: 数值稳定性阈值 (通常取 1e-6)
    """
    features.requires_grad = True
    
    # 前向传播
    result = model(features, image_size, hard=False)
    
    # 计算损失
    loss = result.probs.sum()
    
    # 反向传播
    loss.backward()
    
    # 计算覆盖率
    grad = features.grad
    coverage = (grad.abs() > 1e-6).float().mean()
    
    return coverage.item()


def gradient_magnitude_distribution(model, features, image_size):
    """
    计算梯度幅度分布
    
    用于检测:
    1. 梯度消失: 大部分梯度接近0
    2. 梯度爆炸: 少数梯度异常大
    3. 梯度不均衡: selected vs unselected 差异
    """
    features.requires_grad = True
    result = model(features, image_size, hard=False)
    
    loss = result.probs.sum()
    loss.backward()
    
    grad = features.grad.abs()
    
    return {
        'grad_mean': grad.mean().item(),
        'grad_std': grad.std().item(),
        'grad_min': grad.min().item(),
        'grad_max': grad.max().item(),
        'grad_percentiles': {
            'p25': grad.quantile(0.25).item(),
            'p50': grad.quantile(0.50).item(),
            'p75': grad.quantile(0.75).item(),
            'p95': grad.quantile(0.95).item(),
            'p99': grad.quantile(0.99).item(),
        }
    }
```

---

### B.3 局部性验证

```python
def locality_score(hilbert_indices, selected_mask):
    """
    计算 Selection Locality Score
    
    J(S) = mean(|h(s_i) - h(s_{i+1})|)
    """
    # 处理批量掩码
    if selected_mask.dim() == 2:
        scores = []
        for b in range(selected_mask.shape[0]):
            mask = selected_mask[b] > 0.5
            h_sel = hilbert_indices[mask].float().sort()[0]
            if len(h_sel) < 2:
                scores.append(0.0)
                continue
            jumps = (h_sel[1:] - h_sel[:-1]).abs()
            scores.append(jumps.mean().item())
        return np.mean(scores)
    
    # 单样本
    h_sel = hilbert_indices[selected_mask > 0.5].float().sort()[0]
    if len(h_sel) < 2:
        return 0.0
    jumps = (h_sel[1:] - h_sel[:-1]).abs()
    return jumps.mean().item()


def locality_efficiency(hilbert_indices, selected_mask):
    """
    计算局部性效率
    
    η = J(S) / J_oracle = J(S) / 1.0
    """
    J_S = locality_score(hilbert_indices, selected_mask)
    J_oracle = 1.0  # 连续采样的理想跳距
    return J_S / J_oracle


def compare_with_random_baseline(hilbert_indices, n_candidates, n_selected, n_trials=1000):
    """
    对比随机选择基线
    
    如果 H1SS/H-Entmax 的局部性效率与随机选择无显著差异，
    说明局部性机制没有发挥作用
    """
    random_efficiencies = []
    
    for _ in range(n_trials):
        # 随机选择
        mask = torch.zeros(n_candidates, dtype=torch.bool)
        chosen = torch.randperm(n_candidates)[:n_selected]
        mask[chosen] = True
        
        eff = locality_efficiency(hilbert_indices, mask)
        random_efficiencies.append(eff)
    
    return {
        'random_mean': np.mean(random_efficiencies),
        'random_std': np.std(random_efficiencies),
        'random_ci95': (
            np.mean(random_efficiencies) - 1.96 * np.std(random_efficiencies),
            np.mean(random_efficiencies) + 1.96 * np.std(random_efficiencies)
        )
    }
```

---

### B.4 树一致性验证

```python
def tree_consistency_violations(selected_mask, parent_indices, children_matrix):
    """
    统计树一致性违反
    
    违反定义: 子节点被选中，但其父节点未被选中
    """
    N = selected_mask.shape[0]
    violations = []
    
    for i in range(N):
        if parent_indices[i] < 0:  # 根节点
            continue
            
        parent_selected = selected_mask[parent_indices[i]]
        children = children_matrix[i]
        children_valid = children[children >= 0]
        
        if not parent_selected and selected_mask[i]:
            # 当前节点被选中但其父节点未选中
            violations.append({
                'child_idx': i,
                'parent_idx': parent_indices[i],
                'parent_logit': None,  # 需要 logits
                'child_logit': None,
            })
    
    return violations


def depth_distribution_analysis(result, splitter):
    """
    分析深度分布
    
    关键问题: 
    1. 深度是否过度集中?
    2. 浅层 (d=0,1) vs 深层 (d=3,4) 的比例是否合理?
    """
    depths = result.depths.cpu().numpy()
    
    from collections import Counter
    depth_counts = Counter(depths)
    total = len(depths)
    
    distribution = {
        f'd_{d}': depth_counts.get(d, 0) / total
        for d in range(splitter.max_level_limit + 1)
    }
    
    # 熵: 衡量分布的均匀程度
    probs = np.array(list(distribution.values()))
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    
    return {
        'distribution': distribution,
        'entropy': entropy,
        'dominant_depth': max(distribution, key=distribution.get),
        'shallow_ratio': sum(distribution[f'd_{d}'] for d in [0, 1]),
        'deep_ratio': sum(distribution[f'd_{d}'] for d in [3, 4]),
    }
```

---

### B.5 选择稳定性验证

```python
def selection_iou(set1, set2):
    """计算两个选择集合的 IOU"""
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def selection_stability(splitter, features, image_size, n_trials=10):
    """
    测量选择的稳定性
    
    方法:
    1. 固定特征，多次推理，测量 IOU 变异系数
    2. 添加微小噪声，测量 IOU 衰减
    """
    ious = []
    
    for _ in range(n_trials):
        result1 = splitter(features, image_size, hard=True)
        result2 = splitter(features, image_size, hard=True)
        
        set1 = set(result1.selected_mask[0].nonzero().tolist())
        set2 = set(result2.selected_mask[0].nonzero().tolist())
        
        ious.append(selection_iou(set1, set2))
    
    return {
        'mean_iou': np.mean(ious),
        'std_iou': np.std(ious),
        'min_iou': np.min(ious),
        'stability_score': np.mean(ious) / (np.std(ious) + 1e-6)
    }


def robustness_to_noise(splitter, features, image_size, noise_scale=0.01):
    """
    测量对噪声的鲁棒性
    
    关键问题: 小的输入扰动会导致选择大幅变化吗?
    """
    # 干净选择
    result_clean = splitter(features, image_size, hard=True)
    set_clean = set(result_clean.selected_mask[0].nonzero().tolist())
    
    # 噪声选择
    noise = torch.randn_like(features) * noise_scale
    features_noisy = features + noise
    result_noisy = splitter(features_noisy, image_size, hard=True)
    set_noisy = set(result_noisy.selected_mask[0].nonzero().tolist())
    
    return {
        'clean_vs_noisy_iou': selection_iou(set_clean, set_noisy),
        'noise_scale': noise_scale,
        'relative_change': 1 - selection_iou(set_clean, set_noisy)
    }
```

---

## 附录 C: 关键问题清单

在继续优化之前，必须回答这些问题：

### C.1 局部性验证
- [ ] H1SS 的实际 Locality Efficiency 测量值是多少?
- [ ] H-Entmax 的 Locality Efficiency 与 H1SS 有显著差异吗?
- [ ] 如果差异 < 5%，为什么需要 DistanceDecay Conv?

### C.2 树约束验证
- [ ] 树约束真的减少了树一致性违反吗?
- [ ] λ 课程学习的最佳值是多少?
- [ ] 如果 λ=0.1 和 λ=0.3 效果相同，为什么要课程学习?

### C.3 梯度验证
- [ ] H1SS 早期 (α=1.2) 的梯度覆盖率真的 100% 吗?
- [ ] H1SS 晚期 (α=1.49) 的梯度覆盖率是多少?
- [ ] 梯度覆盖率与最终精度有相关性吗?

### C.4 选择稳定性验证
- [ ] 相同输入的选择 IOU 是多少?
- [ ] 选择的不稳定性会影响训练收敛吗?
- [ ] Dropout/batchnorm 如何影响确定性?

### C.5 深度分布验证
- [ ] 实际深度分布是什么?
- [ ] 是否过度集中于某个深度?
- [ ] 深度分布与精度有相关性吗?

---

## 附录 D: 实验设计方案

### D.1 消融实验设计

| 实验 | H1SS 变体 | 预期结果 | 如何验证 |
|-----|----------|---------|---------|
| A | 移除树约束 | Locality↑? Tree↓? | Tree Consistency Score |
| B | 移除 SDS | Locality↓? 性能? | Locality Efficiency |
| C | 固定 α=1.5 | 梯度消失? 性能? | Gradient Coverage |
| D | 移除 K 课程 | 早期不稳定? | Selection IOU |
| E | Conv1D vs DistanceDecay | 性能差异? | End-to-end Accuracy |

### D.2 对比实验设计

| 对比 | H1SS vs H-Entmax | 测量指标 |
|-----|-------------------|---------|
| 1 | Locality Efficiency | Locality Score / Oracle |
| 2 | Tree Consistency | Violations / Total |
| 3 | Gradient Coverage | % params with grad > τ |
| 4 | Selection Stability | IOU std across trials |
| 5 | Depth Distribution | Entropy, KL-divergence |
| 6 | End-to-end Accuracy | ImageNet top-1 |

---

## 附录 E: 行动路线图

基于上述分析，提出以下优先级排序的行动建议：

### E.1 第一优先级：验证核心假设

**问题 1**: DistanceDecay Conv 是否有效?
```
如果 Locality Efficiency 差异 < 5%，则 DistanceDecay 是无效复杂性
验证方法: Ablation E (Conv1D vs DistanceDecay)
决策点: 差异小 → 简化为标准 Conv1D
```

**问题 2**: 树约束是否必要?
```
如果移除树约束后 Tree Consistency 不下降，则树约束是无效复杂性
验证方法: Ablation A (移除树约束)
决策点: 差异小 → 移除树约束
```

**问题 3**: α 调度是否有效?
```
如果固定 α=1.49 与调度版本效果相同，则调度是无效复杂性
验证方法: Ablation C (固定 α=1.5)
决策点: 差异小 → 简化为固定调度
```

### E.2 第二优先级：统一架构

**统一假设**: H1SS 和 H-Entmax 的本质区别仅在于:
1. 邻域复杂度提取方式 (DistanceDecay Conv1D vs Standard Conv1D)
2. 树约束机制
3. α 调度策略

**统一架构设计**:
```
HilbertSplitterV2:
├── HilbertLocalComplexity (H-Entmax)      # 可学习的 Conv1D
├── (可选) TreeConstraint: simple λ=0.1   # 硬编码，非课程
├── EntmaxAlpha: 固定 1.5 或 1.49          # 无调度
└── K 选择: TopK after Entmax
```

**预期收益**:
- 参数: 10K → 2K (5x 简化)
- 调度器: 5+ → 0 (完全移除)
- 可维护性: 大幅提升

### E.3 第三优先级：探索新方向

基于 NAP/MAT 论文的 O(N) 复杂度优势:

**方向 1**: 适应性窗口大小
```
当前: 固定 kernel_size=5
改进: 根据局部信息密度动态调整窗口
参考: A-ViT 的 halting score 机制
```

**方向 2**: 可学习的距离衰减
```
当前: w_d = 1/(|d|+1) 固定
改进: w_d = softmax(gap, temperature)
其中 gap 是数据驱动的距离嵌入
```

**方向 3**: 两阶段选择
```
Stage 1: 粗筛 (Entmax α=1.2，保留 50% tokens)
Stage 2: 精筛 (Entmax α=1.5，保留 K tokens)
参考: CF-ViT 的粗到细策略
```

### E.4 补充：分形尺度交叉注意力 (Fractal-Scale Cross-Attention)

**核心思想**: 跨尺度交互 — Depth=n 的 Token 显式观察 Depth=n-1 父 Token

```
┌─────────────────────────────────────────────────┐
│  Attention Matrix 增强                          │
│                                                 │
│  正常交互: Token(i) ← Hilbert 邻域             │
│  额外交互: Token(i) ← Parent(Token(i))         │
│                                                 │
│  意义: 共享父分形的远距离 Token 可高效交换信息  │
└─────────────────────────────────────────────────┘
```

**数学形式**:
```
A_enhanced(i,j) = A(i,j) + β * A_parent(i, parent(j))
其中 β 是可学习的跨尺度权重
```

### E.5 补充：分形位置偏置 (Fractal Locality Bias)

**核心思想**: 用 Hilbert 序列距离替代笛卡尔坐标偏移

```
当前: RelativeBias = f(|x_i - x_j|, |y_i - y_j|)
改进: RelativeBias = f(|H(i) - H(j)|)
其中 H(i) 是 Hilbert 曲线索引
```

**多尺度自适应**: 引入深度缩放因子 τ_depth
```
f(|H(i) - H(j)|) = MLP( |H(i) - H(j)| / τ_depth )
深层 (大 depth): τ 大 → 细节局部注意力
浅层 (小 depth): τ 小 → 结构全局注意力
```

**优势**: 分形路径距离 vs 欧几里得距离 — 更能捕捉自相似结构中的相对位置

### E.6 补充：分形阶数跳变特征对齐

**问题**: 区域 A 用 Order 4，区域 B 用 Order 2，Token 密度不同，特征如何对齐？

**解决**: SViT Super Token + Implicit Upsampling
```
低阶 Token (Order 2) ─→ Implicit Up-sampling ─→ 高阶特征空间
                         (轻量级卷积映射)
                         统一物理尺度 (Physical Scale)
```

**意义**: Transformer 计算时所有 Token 具有统一的物理尺度参考系

### E.7 补充：Halting Module 训练稳定性

**预警**: 离散不可微决策 → 容易陷入局部最优 (所有 Token 选择最浅深度)

**Peer Advice 方案**:
```
训练早期: Soft-Halting
    halt_weight(d) = softmax(halt_score, temperature=T)
    所有深度加权平均 → 梯度平稳回传

训练中后期: Hard-Halting
    选择 halt_score 最高的深度 → 离散决策
    切换时机: validation loss 开始收敛时
```

**关键检查点**:
- `budget_loss` 梯度稳定性监控
- 分形深度分布熵 (不应过早塌缩到单一深度)

### E.8 补充：Fused Hilbert Tokenizer 硬件优化

**问题**: Hilbert 索引计算 + ROIAlign 频繁调用 → Python 层 Dispatch 开销

**解决**: Triton/CUDA Fused Kernel
```
原子操作: Hilbert 坐标计算 + Indexing + Feature Gathering
收益: 大幅降低 Python 层开销，提升吞吐量
目标: 让 FractalCurveViT 在速度上超越传统 ViT
```

**GPU 缓存友好性考虑**:
- Hilbert 曲线虽保持局部性，但内存访问可能非连续
- 需在 Kernel 层面做内存访问模式优化

### E.9 决策矩阵 (更新)

| 验证结果 | 行动 |
|---------|-----|
| Locality Efficiency 差异 < 5% | 移除 DistanceDecay Conv1D |
| Tree Consistency 差异 ≈ 0 | 移除树约束 |
| α=1.49 固定 = 调度版本 | 移除 α 调度 |
| 所有上述成立 | 采用 HilbertSplitterV2 简化架构 |
| 深度分布熵过早塌缩 | 启用 Soft-Halting 策略 |
| 跨阶特征不对齐 | 引入 Super Token Up-sampling |

### E.10 风险评估 (更新)

| 风险 | 影响 | 缓解措施 |
|-----|-----|---------|
| 简化后精度下降 | 高 | 保留 Ablation 作为回退选项 |
| 树一致性违反增加 | 中 | 添加简单的硬约束 (非课程学习) |
| 选择稳定性下降 | 中 | 使用温度退火而非完全确定性 |
| Halting 陷入局部最优 | 高 | Soft-Halting 早期 → Hard-Halting 中后期 |
| 分形阶数跳变特征不连续 | 高 | SViT Super Token + Implicit Up-sampling |
| Python Dispatch 开销大 | 中 | Triton/CUDA Fused Kernel |

---

<!-- 分割线: 以下为原始调研内容 -->

# 分形结构在视觉领域的研究调研

## 研究背景

本项目 (fractal-curve-tokenizer) 使用 Hilbert 曲线（一种空间填充曲线）进行视觉 Transformer 的 token 化。以下是对相关领域论文的研究分析。

---

## 零、核心理论基础（赵等 2022 + 李徐 2025）

### 0.1 Zigzag 到 Hilbert 的范式转换

**参考论文**: Zhao 等 (2022) - Rethinking the Zigzag Flattening

**核心洞察**:
| 扫描方式 | 问题 | Hilbert 优势 |
|---------|-----|------------|
| Zigzag/Raster | 跨行时产生"跳跃"，相邻像素在序列中相距甚远 | 序列中相邻 = 2D空间相邻 |
| 分形优势 | - | 自相似性，最大程度保持空间局部性 |

**对本项目的意义**:
> 验证了 FractalCurveViT 使用 Hilbert 曲线作为分形 Tokenizer 的**合法性**。这不是美学选择，而是数学保证——在多尺度特征提取时保持特征点的相对位置不变。

### 0.2 邻域感知 Token 减枝与合并

**参考论文**: Li 和 Xu (2025) - Neighbor-Aware Token Reduction

**两个关键算子**:

#### NAP (Neighbor-Aware Pruning) 邻域感知剪枝
```
原理: 利用"相邻即近邻"特性，直接在 1D Hilbert 序列上做窗口化显著性检测
优势: 如果一个 Token 在序列邻域内贡献度低 → 可安全移除
复杂度: O(N) 而非 O(N²)
```

#### MAT (Merging by Adjacent Token Similarity) 邻域相似度合并
```
传统方法: 需要计算全局相似度矩阵 → O(N²)
Hilbert方法: 只需计算 1D 序列中相邻 Token 的相似度 → O(N)
```

**对本项目的意义**:
> 可实现 **O(N) 复杂度**的 Token 融合，而不是昂贵的 Attention-based 合并

### 0.3 分形扫描与长序列建模

**参考论文**: RainMamba (2024) & Hilbert Mamba (2025)

**Hilbert Selective Scan (HilbertSS)**:
```
场景: 视频/高分辨率图像处理
问题: 常规扫描破坏时空相关性
解决: Hilbert 路径作为扫描顺序
效果: 显著提升对细节（雨丝、病灶边缘）的感知
```

**局部-全局互惠 (Local-Global Reciprocal)**:
```
通过 Hilbert 路径连接"瓶颈查询" (Bottleneck Queries)
在不损失局部偏置 (Locality Bias) 的前提下捕获全局上下文
```

---

## 一、相关论文研究

### 1. 自适应 Token 采样系列

#### ATS: Adaptive Token Sampling (ECCV 2022)
- **核心思想**: 提出无参数可微分模块 ATS，根据自注意力矩阵对 token 打分，自适应采样重要 token
- **论文**: [2111.15667] Adaptive Token Sampling For Efficient Vision Transformers
- **与本项目关联**: 分形分割本身就是一种自适应选择机制，可借鉴其"动态 token 数量"思想

#### A-ViT: Adaptive Tokens for Vision Transformer (CVPR 2022 Oral)
- **核心思想**: 引入"halting score"机制，动态决定每个 token 的计算终止时间
- **论文**: [2112.07658] A-ViT: Adaptive Tokens for Efficient Vision Transformer
- **创新点**:
  - ponder loss 鼓励 token 适时停止
  - 分布先验正则化稳定训练
  - 吞吐量提升 62% (DeiT-Tiny) / 38% (DeiT-Small)
- **与本项目关联**: 本项目的深度选择机制与 A-ViT 的多级停止机制有相似之处

### 2. 分层 Token 聚合系列

#### QuadTree Attention (ICLR 2022)
- **核心思想**: 从粗到细建立注意力金字塔，复杂度降为线性
- **论文**: QuadTree Attention for Vision Transformers
- **关键机制**:
  - 构建 token pyramids
  - 快速跳过不相关区域
  - 选择 top-K patch 进入下一层
- **与本项目关联**: 
  - **最相关** - 分形分割天然具有四叉树结构
  - 可借鉴其"粗细结合"的注意力计算策略

#### CF-ViT: Coarse-to-Fine Vision Transformer
- **核心思想**: 两阶段推理：粗粒度分类 → 细粒度重分割
- **关键观察**:
  - 粗粒度 patch split 可定位信息区域
  - 大多数图像可通过短 token 序列识别
- **与本项目关联**: 可探索"先用浅层分形快速定位，再深层细化"的两阶段策略

### 3. Super Token 系列

#### SViT: Super Token Vision Transformer (2025)
- **核心思想**: 超 token 采样 + 稀疏关联学习
- **三组件**:
  - Convolutional Position Embedding (CPE)
  - Super Token Attention (STA)
  - Convolutional FFN (ConvFFN)
- **与本项目关联**: 超 token 概念与分形聚合的"父 token"概念相似

### 4. 局部注意力与层次化结构

#### Swin Transformer (ICCV 2021)
- **核心思想**: Window attention + Shifted window 引入 cross-window 关系
- **关键贡献**: 计算复杂度与图像大小成线性关系
- **与本项目关联**: 分形的空间局部性可增强 window attention 的局部感受野

#### NAT: Neighborhood Attention Transformer
- **核心思想**: 邻域注意力，将感受野扩展到最近邻像素
- **优点**: 比 Swin 受约束更少，包含局部归纳偏差
- **与本项目关联**: Hilbert 曲线的局部性可增强邻域聚合效果

### 5. 其他高效 Transformer

#### Tokenformer (2024)
- **核心思想**: 参数 token 化，使用交叉注意力管理交互
- **创新点**: 完全基于注意力的架构，token-parameter 交互也用注意力
- **与本项目关联**: 可探索将分形系数作为可学习 token 的可能性

#### FaSA: Factorization Self-Attention
- **核心思想**: 将传统注意力矩阵分解为稀疏子注意力
- **优势**: 同时具备局部窗口计算效率和长程依赖建模能力

---

## 二、对本项目的启示

### 1. 短期可借鉴方向

| 论文/技术 | 可借鉴点 | 实现复杂度 |
|---------|---------|----------|
| QuadTree Attention | 层次化注意力计算 | 中 |
| ATS token scoring | 动态重要性评估 | 低 |
| A-ViT halting | 多级停止机制 | 中 |
| 两阶段推理 (CF-ViT) | 粗→细策略 | 低 |

### 2. 长期探索方向

1. **稀疏注意力与分形结合**
   - 利用 Hilbert 曲线的局部性设计稀疏注意力模式
   - 避免全注意力 O(N²) 复杂度

2. **可学习分形深度**
   - 借鉴 A-ViT 的 halting score
   - 为不同图像区域动态选择分形深度

3. **多尺度特征融合**
   - 参考 QuadTree 的金字塔结构
   - 在不同分形深度捕获多尺度特征

4. **分形感知位置编码**
   - Hilbert 索引天然包含空间信息
   - 可探索作为隐式位置编码

---

## 三、关键论文列表

| 论文 | 年份 | 会议/期刊 | 核心贡献 |
|-----|-----|---------|---------|
| ATS | 2022 | arXiv | 自适应 token 采样 |
| A-ViT | 2022 | CVPR | 自适应计算时间 |
| QuadTree Attention | 2022 | ICLR | 线性复杂度的层次注意力 |
| Swin Transformer | 2021 | ICCV | Window attention |
| CF-ViT | 2022 | arXiv | 粗到细两阶段推理 |
| NAT | 2022 | arXiv | 邻域注意力 |
| Tokenformer | 2024 | arXiv | 参数 token 化 |
| SViT | 2025 | - | 超 token 采样 |
| FaSA | 2023 | arXiv | 分解注意力 |
| Vision Transformers Need Registers | 2023 | arXiv | 寄存器 token |

---

## 四、技术路线图（四大方向融合）

### 方向1: QuadTree 层次注意力 (核心)

**核心思想**: 将分形分割视为天然的四叉树，构建层次化注意力

**实现架构**:
```
Layer 0: 原始 H×W patches
    ↓ 分形分割 (2×2)
Layer 1: H/2 × W/2 super-patches (每个聚合4个token)
    ↓ 分形分割 (2×2)
Layer 2: H/4 × W/4 super-super-patches
    ↓ ...
```

**关键技术点**:
- **Token Pyramid**: 在每层构建 token 金字塔
- **Bottom-up Selection**: 从粗层选择 top-K 相关的细粒度区域
- **Top-down Refinement**: 选中的区域在下一层进行更精细的分形

**类比参考**:
- 本项目的 `HilbertOptimalSplitter` 已有 depth 选择能力
- 可增强为"层级注意力路由器"

### 方向2: 自适应深度选择 (A-ViT 启发)

**核心思想**: 借鉴 halting score，动态决定每个区域的计算深度

**实现方案**:
```python
class AdaptiveFractalDepth:
    def __init__(self, max_depth=4):
        self.halting_score = {}  # 每个区域的停止分数
        self.threshold = 0.5

    def should_refine(self, region, depth):
        # 基于信息密度/熵/注意力分数决定是否继续分形
        info_density = self.compute_info_density(region)
        self.halting_score[region] = info_density
        return info_density > self.threshold and depth < self.max_depth
```

**与本项目融合**:
- 当前: 固定的深度分配或启发式深度选择
- 增强: 学习型 halting 模块，数据驱动决定深度

### 方向3: 两阶段推理 (CF-ViT 启发)

**核心思想**: 粗粒度快速定位 → 细粒度精确分类

**推理流程**:
```
Stage 1 (Coarse):
  - 输入: 图像 → 16×16 patches
  - 分形深度: 2层 (获取粗粒度语义)
  - 快速分类 + 置信度检测
  - if 置信度高: 直接输出
  - else: 进入 Stage 2

Stage 2 (Fine):
  - 定位 Stage 1 置信度低的区域
  - 对这些区域进行更深的分形 (深度 3-4)
  - 细粒度分类 + 重新评估
```

**优势**:
- 简单图像: 只需浅层分形 → 计算量大幅降低
- 复杂图像: 自动启用深层分形 → 保证精度

### 方向4: 稀疏注意力模式

**核心思想**: 利用 Hilbert 曲线的局部性，设计稀疏但高效的注意力

**稀疏策略**:
```
传统 Full Attention: O(N²) - 所有 token 两两交互
     ↓
Window Attention: O(N) - 仅窗口内交互 (Swin)
     ↓
Fractal Sparse Attention: O(N^1.5) - 基于分形的层次稀疏
     ↓
  - 近距离 token: 细粒度交互
  - 远距离 token: 粗粒度/稀疏交互
```

**实现要点**:
- Hilbert 距离 < threshold: 细粒度注意力
- Hilbert 距离 > threshold: 通过父节点间接交互

---

## 五、与本项目现有架构的融合

### 当前架构 (fractal-curve-tokenizer)
```
image → PatchEmbed → Fractal Split (depth=0)
    → Attention Block × N
    → Classifier
```

### 增强后的目标架构
```
image → PatchEmbed
    → [Fractal Splitter + Adaptive Depth] 
    → [Layer 0: 细粒度 tokens]
    → [Layer 1: 聚合 tokens + 层级注意力]
    → [Layer 2+: 超聚合 tokens + 稀疏注意力]
    → [Halting Controller: 决定何时停止]
    → Classifier
```

### 关键模块增强

| 现有模块 | 增强方向 | 新增能力 |
|---------|---------|---------|
| `HilbertOptimalSplitter` | 增加层级路由 | 自适应子区域选择 |
| `manifold_attention.py` | 层次化 | 四叉树式注意力 |
| `tokenizer.py` | 动态 token 数量 | 两阶段推理 |
| 新增 | Halting Module | 自适应深度选择 |

---

## 六、实施建议

### Phase 1: 基础增强 (2-4周)
- [ ] 实现层级注意力 (参考 QuadTree)
- [ ] 添加 token importance scoring (参考 ATS)
- [ ] 集成到现有 FractalViT

### Phase 2: 自适应机制 (4-8周)
- [ ] 实现 halting score 模块
- [ ] 添加动态深度选择
- [ ] 验证精度与效率提升

### Phase 3: 两阶段推理 (8-12周)
- [ ] 粗粒度快速推理路径
- [ ] 置信度检测机制
- [ ] 细粒度重分割逻辑

### Phase 4: 稀疏注意力 (12-16周)
- [ ] 基于 Hilbert 距离的稀疏模式
- [ ] 混合细粒度/粗粒度注意力
- [ ] 性能优化与部署

---

## 七、针对本项目的三个核心研究思路

### 思路1: 动态分辨率分级采样 (Adaptive Fractal Sampling)

**理论基础**: Zhao 等人的 Zigzag → Hilbert 范式转换

**实现思路**:
```
在图像显著区域（由第一层输出判断）:
  → 使用更高阶 Hilbert 曲线（更高密度 Token）

在背景区域:
  → 使用低阶 Hilbert 曲线（更低密度 Token）
```

**数学优势**:
> Hilbert 曲线在不同阶数间具有**嵌套性**，这种多分辨率融合比传统 Patch 重组更优雅

**伪代码**:
```python
class AdaptiveFractalSampling:
    def __init__(self, max_order=6):
        self.max_order = max_order

    def sample(self, image, saliency_map):
        # saliency_map: 第一层输出的显著性分数
        regions = self.divide_regions(image)

        for region in regions:
            if saliency_map[region] > THRESHOLD_HIGH:
                yield self.hilbert_curve(region, order=self.max_order)
            elif saliency_map[region] > THRESHOLD_LOW:
                yield self.hilbert_curve(region, order=self.max_order-1)
            else:
                yield self.hilbert_curve(region, order=self.max_order-2)
```

---

### 思路2: O(N) 复杂度的分形池化 (Fractal Pooling)

**理论基础**: Li 和 Xu 的 MAT 策略

**实现思路**:
```
位置: Transformer 层之间

操作: 直接对 Hilbert 序列中相邻的 2^n 个 Token 进行池化

优势:
  1. 天然保持 2D 空间结构（因为 Hilbert 邻接 ≈ 空间邻接）
  2. 无需重新计算位置编码（RoPE）
  3. Hilbert 编码本身就带有分形深度信息
```

**复杂度对比**:
| 方法 | 复杂度 | 位置保持 |
|-----|-------|---------|
| 传统 Token Merging | O(N²) | 否 |
| Attention-based Merge | O(N²) | 部分 |
| **Fractal Pooling** | **O(N)** | **是** |

---

### 思路3: 分形位置偏置 (Fractal Locality Bias)

**理论基础**: Hilbert 距离天然编码空间关系

**实现思路**:
```
在 Attention 计算时，引入基于 Hilbert 距离的偏置矩阵

公式: RelativeBias = f(|H(i) - H(j)|)

其中 H(i) 是第 i 个 Token 在 Hilbert 曲线上的序数
```

**与 RoPE 的区别**:
| 方法 | 编码方式 | 分形友好度 |
|-----|---------|----------|
| RoPE | x, y 坐标偏移 | 低（需要从 Hilbert 索引反算） |
| **Fractal Locality Bias** | **Hilbert 序列距离** | **高** |

**优势**:
> 比传统的 x, y 坐标偏移更能体现分形结构的局部联系——因为 Hilbert 距离直接量化了"在分形路径上的接近程度"

---

## 八、关键论文列表

| 论文 | 年份 | 会议/期刊 | 核心贡献 |
|-----|-----|---------|---------|
| Zhao et al (Zigzag→Hilbert) | 2022 | - | 范式转换理论 |
| Li & Xu (NAP/MAT) | 2025 | - | O(N) 邻域操作 |
| RainMamba/HilbertMamba | 2024-2025 | - | Hilbert 选择性扫描 |
| ATS | 2022 | arXiv | 自适应 token 采样 |
| A-ViT | 2022 | CVPR | 自适应计算时间 |
| QuadTree Attention | 2022 | ICLR | 线性复杂度的层次注意力 |
| Swin Transformer | 2021 | ICCV | Window attention |
| CF-ViT | 2022 | arXiv | 粗到细两阶段推理 |
| NAT | 2022 | arXiv | 邻域注意力 |
| Tokenformer | 2024 | arXiv | 参数 token 化 |
| SViT | 2025 | - | 超 token 采样 |
| FaSA | 2023 | arXiv | 分解注意力 |
| Vision Transformers Need Registers | 2023 | arXiv | 寄存器 token |

---

## 九、实施建议

### Phase 0: 理论验证 (1-2周)
- [ ] 验证 Hilbert 局部性理论（与 Zigzag 对比实验）
- [ ] 测量本项目中的 Token 冗余度
- [ ] 确定 NAP/MAT 的适用场景
- [ ] **新增**: 消融实验验证 DistanceDecay Conv 有效性 (附录 E.1)

### Phase 1: 基础增强 (2-4周)
- [ ] 实现层级注意力 (参考 QuadTree)
- [ ] 添加 token importance scoring (参考 ATS)
- [ ] 集成到现有 FractalViT
- [ ] **新增**: Fractal-Scale Cross-Attention (附录 E.4)

### Phase 2: 自适应机制 (4-8周)
- [ ] 实现 halting score 模块
- [ ] 添加动态深度选择
- [ ] 验证精度与效率提升
- [ ] **更新**: Soft-Halting 早期 → Hard-Halting 中后期 (附录 E.7)
- [ ] **新增**: Fractal Locality Bias with τ_depth (附录 E.5)

### Phase 3: 两阶段推理 (8-12周)
- [ ] 粗粒度快速推理路径
- [ ] 置信度检测机制
- [ ] 细粒度重分割逻辑
- [ ] **新增**: Super Token Up-sampling for 阶数跳变对齐 (附录 E.6)

### Phase 4: 稀疏注意力 + 硬件优化 (12-16周)
- [ ] 基于 Hilbert 距离的稀疏模式
- [ ] Fractal Pooling 实现
- [ ] 分形位置偏置
- [ ] **新增**: Triton/CUDA Fused Hilbert Tokenizer (附录 E.8)
- [ ] 性能优化与部署

---

## 十、相关资源

### 核心论文
- Zhao et al - Rethinking the Zigzag Flattening (2022)
- Li & Xu - Neighbor-Aware Token Reduction (2025)
- RainMamba / Hilbert Mamba (2024-2025)
- ATS: https://arxiv.org/abs/2111.15667
- A-ViT: https://arxiv.org/abs/2112.07658
- QuadTree: ICLR 2022
- CF-ViT: https://arxiv.org/abs/2203.03821
- Swin Transformer: https://github.com/microsoft/Swin-Transformer
- NAT: https://github.com/SHI-Labs/Neighborhood-Attention-Transformer

- ATS 论文: https://arxiv.org/abs/2111.15667
- A-ViT 论文: https://arxiv.org/abs/2112.07658
- QuadTree 论文: ICLR 2022
- CF-ViT 论文: https://arxiv.org/abs/2203.03821
- Swin Transformer: https://github.com/microsoft/Swin-Transformer
- NAT: https://github.com/SHI-Labs/Neighborhood-Attention-Transformer

- 论文列表: https://github.com/adaptivetokensampling/ATS
- QuadTree 代码: 需自行搜索
- Swin Transformer: https://github.com/microsoft/Swin-Transformer
