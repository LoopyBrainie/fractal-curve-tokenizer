# Hilbert Curve ViT 训练系统架构设计

> **版本**: v2.0 (完全重构)  
> **日期**: 2026-01-04  
> **设计原则**: 形式化分析 + 计算验证 + 组件解耦

---

## 📐 一、设计原则与架构边界

### 1.1 核心原则

```
1. 形式化优先 (Formalization First)
   - 每个组件都有明确的数学定义
   - 通过计算验证确保正确性
   
2. 解耦设计 (Decoupling)
   - 训练器 ⊥ 模型架构
   - 每个组件可独立测试和替换
   
3. Hilbert ViT 特定 (Domain-Specific)
   - 针对分形 tokenization 特性设计
   - 空间局部性、深度分布、资源监控
   
4. 无向后兼容 (No Backward Compatibility)
   - 彻底重构，追求最佳实现
   - 清晰的接口定义
```

### 1.2 架构分层

```
┌─────────────────────────────────────────────────────────┐
│                Training Orchestrator                     │
│  - 协调各组件                                             │
│  - 不包含业务逻辑                                         │
└─────────────────────────────────────────────────────────┘
                          ↓
    ┌─────────────────────┬─────────────────────┬─────────────────────┐
    ↓                     ↓                     ↓                     ↓
┌─────────┐      ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│ Sampler │      │    Loss     │      │   Metrics   │      │  Callbacks  │
│         │      │  Functions  │      │             │      │             │
│ - 类别  │      │  - 任务损失 │      │  - 准确率   │      │  - 资源监控 │
│   平衡  │      │  - 资源惩罚 │      │  - 深度分布 │      │  - 检查点   │
│ - 渐进  │      │  - 正则化   │      │  - Token 统计│     │  - 可视化   │
│   采样  │      │             │      │             │      │             │
└─────────┘      └─────────────┘      └─────────────┘      └─────────────┘
                          ↓
                ┌─────────────────────┐
                │   Model Interface   │
                │  (只读统计接口)       │
                └─────────────────────┘
                          ↓
                ┌─────────────────────┐
                │  FractalCurveViT    │
                │  (模型架构层)         │
                └─────────────────────┘
```

**依赖规则**:
1. **单向依赖**: 训练器 → 模型 (只读接口)
2. **无侵入**: 训练器不修改模型内部状态
3. **插件式**: 各组件通过接口通信，可独立替换

---

## 🔬 二、Hilbert Curve ViT 的数学特性

### 2.1 空间局部性 (Locality Preservation)

**Hilbert 曲线的 Hölder 连续性**:

$$\|H(d_1) - H(d_2)\|_2 \leq C \cdot |d_1 - d_2|^{1/2}$$

**含义**: 序列中相邻的 token 在空间上也趋向相邻。

**训练含义**:
1. **相邻 token 应有相似特征** → 梯度局部性
2. **Attention 模式应体现空间连续性** → 局部注意力头分析
3. **错误应在空间上聚集** → 错误热图分析

### 2.2 深度分布 (Depth Distribution)

**Variable Depth Tokenization**:

$$N_{total} = \sum_{d=0}^{D_{max}} N_d$$

其中 $N_d$ 是深度 $d$ 的 token 数量。

**分形特性**:
- 深层 token (小 patch) → 高频细节
- 浅层 token (大 patch) → 低频结构

**训练含义**:
1. **深度熵**: $H = -\sum_d p_d \log p_d$ 应保持多样性
2. **深度-难度关联**: 困难样本可能需要更多深层 token
3. **资源分配**: 深层 token 计算成本更高

### 2.3 分形自相似性 (Fractal Self-Similarity)

**四叉树递归结构**:

$$\text{Region}(d+1) = \bigcup_{i=1}^4 \text{Quadrant}_i(\text{Region}(d))$$

**LCA (Lowest Common Ancestor) 距离**:

$$d_{LCA}(t_i, t_j) = \max\{d : \text{Path}(t_i)[0:d] = \text{Path}(t_j)[0:d]\}$$

**训练含义**:
1. **层级关系**: LCA 深度 = 空间接近度
2. **Attention bias 验证**: 检查 $\text{Attn}(t_i, t_j) \propto f(d_{LCA})$
3. **位置编码效果**: LCA 编码 vs 绝对位置编码

---

## 📊 三、组件形式化设计

### 3.1 资源统计接口 (Resource Stats Interface)

**接口定义** (`core/resource_stats.py`):

```python
@dataclass
class ModelResourceStats:
    """模型资源使用统计（只读）"""
    
    # Token 统计
    avg_tokens_per_image: float
    token_depth_distribution: List[float]  # [N_0, N_1, ..., N_D]
    
    # FLOPS 统计
    total_flops: float
    flops_breakdown: Dict[str, float]
    
    # 深度统计
    avg_depth: float
    depth_entropy: float  # H = -Σ p_d log p_d
```

**数学性质**:
1. **只读性**: 训练器读取但不修改
2. **无状态**: 每次前向传播重新计算
3. **批次聚合**: 提供 batch 平均和逐样本统计

**FLOPS 计算公式**:

```
FLOPS_total = FLOPS_tokenizer + L · (FLOPS_attention + FLOPS_ffn)

其中:
    FLOPS_tokenizer = N_regions · C · (k² + M_mlp)
    FLOPS_attention = 4ND² + N²D  (Q,K,V proj + attn + out proj)
    FLOPS_ffn = 8ND²  (up + down projection)
```

**深度加权 Token 数**:

$$N_{weighted} = \sum_{d=0}^{D_{max}} N_d \cdot e^{\alpha d}$$

参数 $\alpha$ 控制深度惩罚强度：
- $\alpha = 0$: 无权重 ($N_{weighted} = N_{total}$)
- $\alpha = 0.1$: 温和惩罚深层 token
- $\alpha = 0.5$: 强烈惩罚深层 token

### 3.2 资源感知损失 (Resource-Aware Loss)

**损失函数设计** (`losses/resource_loss.py`):

$$\mathcal{L}_{resource} = \lambda_{flops} \mathcal{L}_{flops} + \lambda_{token} \mathcal{L}_{token} + \lambda_{entropy} \mathcal{L}_{entropy}$$

#### 3.2.1 FLOPS 约束

$$\mathcal{L}_{flops} = \text{ReLU}\left(\frac{\text{FLOPS}_{actual}}{\text{FLOPS}_{budget}} - 1\right)^2$$

**设计原理**:
- 仅惩罚超预算情况 (ReLU)
- 二次惩罚确保严格约束
- FLOPS_budget 根据硬件设定 (RTX 4070: ~5 GFLOPS)

#### 3.2.2 Token 数量约束

$$\mathcal{L}_{token} = \text{ReLU}(N_{weighted} - N_{budget})^2$$

**为什么使用加权数量?**
- 深层 token (depth=4) 比浅层 token (depth=0) 消耗更多计算
- 权重 $w(d) = e^{0.1d}$: depth=0 → 1.0, depth=4 → 1.49
- 鼓励模型使用浅层 token

#### 3.2.3 深度熵正则

$$\mathcal{L}_{entropy} = (H_{actual} - H_{target})^2$$

其中目标熵:

$$H_{target} = \log(D_{max} + 1)$$

**设计原理**:
- 防止深度坍缩 (所有 token 集中在某一深度)
- 鼓励多样化深度分布
- $H_{target}$ 是均匀分布的熵 (最大熵)

**计算验证**:

```python
# 验证：深度熵在合理范围
D_max = 4
H_uniform = np.log(D_max + 1)  # ≈ 1.609
H_collapsed = 0.0  # 所有 token 在同一深度
H_typical = 1.0 - 1.3  # 实际训练中的熵

assert 0 <= H_typical <= H_uniform
```

### 3.3 Hilbert 特定评估指标 (Hilbert-Specific Metrics)

#### 3.3.1 局部性保持度量

**定义**: 测量序列相邻性 vs 空间相邻性的一致性

$$\text{Locality}_{\text{preservation}} = \frac{1}{N} \sum_{i=1}^{N-1} \mathbb{1}[\|pos(t_i) - pos(t_{i+1})\|_2 \leq \theta]$$

其中:
- $pos(t_i)$ 是 token $i$ 的空间中心坐标
- $\theta$ 是空间相邻阈值 (如 1.5 个 patch 宽度)

**计算验证**:

```python
def compute_locality_preservation(
    token_positions: np.ndarray,  # [N, 2] (x, y)
    threshold: float = 1.5,
) -> float:
    """
    计算局部性保持率
    
    参数:
        token_positions: Token 在 2D 空间中的位置 [N, 2]
        threshold: 空间相邻阈值（单位：patch 宽度）
    
    返回:
        preservation_rate: [0, 1] 局部性保持率
    """
    distances = np.linalg.norm(
        token_positions[1:] - token_positions[:-1],
        axis=1
    )
    preservation = (distances <= threshold).mean()
    return float(preservation)

# 理论验证：Hilbert 曲线应优于光栅扫描
# - Hilbert: 保持率 ~0.8-0.9
# - Raster: 保持率 ~0.5-0.6
```

#### 3.3.2 深度-复杂度关联度

**假设**: 高频细节区域应使用更多深层 token

$$\text{Corr}(\text{Depth}, \text{LocalVar}) \in [0, 1]$$

**计算方法**:

```python
def compute_depth_complexity_correlation(
    depths: np.ndarray,  # [N] token 深度
    local_variances: np.ndarray,  # [N] 局部方差
) -> float:
    """
    计算深度与局部复杂度的 Pearson 相关系数
    
    数学形式:
        ρ = Cov(D, V) / (σ_D · σ_V)
    
    预期:
        - 理想情况: ρ > 0.5 (强正相关)
        - 崩溃情况: ρ ≈ 0 (无关联，Splitter 失效)
    """
    correlation = np.corrcoef(depths, local_variances)[0, 1]
    return float(correlation)
```

#### 3.3.3 LCA 距离与 Attention 权重的一致性

**假设**: Attention 权重应随 LCA 距离衰减

$$\mathbb{E}[\text{Attn}(t_i, t_j) | d_{LCA} = k] \propto f(k)$$

**测量方法**:

```python
def compute_lca_attention_alignment(
    attention_weights: np.ndarray,  # [N, N] 注意力矩阵
    lca_distances: np.ndarray,  # [N, N] LCA 深度矩阵
    max_lca: int = 4,
) -> Dict[int, float]:
    """
    计算 LCA 距离分组的平均注意力权重
    
    数学形式:
        μ_k = E[Attn(i,j) | d_LCA(i,j) = k]
    
    预期衰减模式:
        μ_0 > μ_1 > μ_2 > μ_3 > μ_4
        
    返回:
        {lca_depth: avg_attention_weight}
    """
    lca_attention_map = {}
    for lca in range(max_lca + 1):
        mask = (lca_distances == lca)
        if mask.sum() > 0:
            avg_attn = attention_weights[mask].mean()
            lca_attention_map[lca] = float(avg_attn)
    
    return lca_attention_map
```

### 3.4 回调系统 (Callback System)

**设计原则**: 无侵入、可组合、事件驱动

```python
class TrainingCallback(ABC):
    """训练回调基类"""
    
    @abstractmethod
    def on_epoch_start(self, trainer, epoch: int):
        pass
    
    @abstractmethod
    def on_batch_end(self, trainer, batch_idx: int, logs: Dict):
        pass
    
    @abstractmethod
    def on_epoch_end(self, trainer, epoch: int, logs: Dict):
        pass
```

#### 3.4.1 资源监控回调

```python
class ResourceMonitorCallback(TrainingCallback):
    """
    监控模型资源使用情况
    
    统计信息:
        - 每 epoch 平均 FLOPS
        - Token 数量分布
        - 深度分布熵
        - GPU 显存使用
    
    触发条件:
        - FLOPS 超预算 → 警告
        - 深度熵 < 阈值 → Splitter 坍缩警告
    """
```

#### 3.4.2 检查点回调

```python
class CheckpointCallback(TrainingCallback):
    """
    智能检查点保存
    
    策略:
        - 保存最佳验证准确率模型
        - 每 N epoch 保存一次
        - 保存完整统计信息（资源 + 性能）
    """
```

#### 3.4.3 可视化回调

```python
class VisualizationCallback(TrainingCallback):
    """
    Hilbert ViT 特定可视化
    
    输出:
        - 深度分布热图
        - Token 数量趋势
        - Attention 模式分析
        - 局部性保持曲线
    """
```

---

## 🛠️ 四、模块化训练器设计

### 4.1 训练器接口

```python
class HilbertViTTrainer:
    """
    Hilbert Curve ViT 专用训练器
    
    设计原则:
        1. 组件可插拔
        2. 无模型架构依赖
        3. 完整的实验追踪
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        loss_fn: nn.Module,
        resource_loss: Optional[ResourceAwareLoss] = None,
        callbacks: List[TrainingCallback] = None,
        device: str = "cuda",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.resource_loss = resource_loss
        self.callbacks = callbacks or []
        self.device = device
        
        # 统计信息
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "resource_stats": [],
        }
    
    def train_epoch(self, epoch: int) -> Dict:
        """训练一个 epoch"""
        pass
    
    def validate(self) -> Dict:
        """验证当前模型"""
        pass
    
    def fit(self, num_epochs: int):
        """完整训练循环"""
        for epoch in range(num_epochs):
            # 触发回调
            for callback in self.callbacks:
                callback.on_epoch_start(self, epoch)
            
            # 训练
            train_logs = self.train_epoch(epoch)
            
            # 验证
            val_logs = self.validate()
            
            # 记录历史
            self._update_history(train_logs, val_logs)
            
            # 触发回调
            for callback in self.callbacks:
                callback.on_epoch_end(self, epoch, {**train_logs, **val_logs})
```

### 4.2 配置系统

```python
@dataclass
class TrainingConfig:
    """训练配置（数据类）"""
    
    # 基础配置
    num_epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 0.05
    
    # 资源约束
    flops_budget: float = 5e9  # 5 GFLOPS (RTX 4070)
    token_budget: int = 128
    depth_weight_alpha: float = 0.1
    
    # 损失权重
    resource_loss_weight: float = 0.01
    
    # 采样策略
    use_class_balanced_sampling: bool = True
    progressive_sampling_start: float = 0.0
    progressive_sampling_end: float = 0.5
    
    # 回调
    save_checkpoint_every: int = 10
    visualize_every: int = 5
    
    # 设备
    device: str = "cuda"
    mixed_precision: bool = True
    
    def validate(self):
        """验证配置合法性"""
        assert self.num_epochs > 0
        assert self.batch_size > 0
        assert 0 <= self.progressive_sampling_end <= 1.0
        # ... 更多验证
```

---

## 🧪 五、计算验证框架

### 5.1 单元测试策略

```python
class TestResourceStats(unittest.TestCase):
    """资源统计模块测试"""
    
    def test_weighted_token_count(self):
        """验证深度加权公式"""
        stats = ModelResourceStats(
            token_depth_distribution=[50, 30, 20]
        )
        
        # 手动计算
        expected = 50 * np.exp(0) + 30 * np.exp(0.1) + 20 * np.exp(0.2)
        actual = stats.get_weighted_token_count(alpha=0.1)
        
        self.assertAlmostEqual(actual, expected, places=2)
    
    def test_flops_computation(self):
        """验证 FLOPS 计算"""
        # 已知参数：N=100, D=384, L=10
        N, D, L = 100, 384, 10
        
        # 理论值
        flops_attn = L * (4 * N * D * D + N * N * D)
        flops_ffn = L * (8 * N * D * D)
        
        # 实际计算
        stats = ModelResourceStats._compute_flops_breakdown(
            avg_tokens=N,
            model_config={"dim": D, "depth": L},
            batch_size=1
        )
        
        self.assertAlmostEqual(
            stats["attention"],
            flops_attn,
            delta=flops_attn * 0.01  # 1% 误差
        )
```

### 5.2 积分测试

```python
class TestEndToEndTraining(unittest.TestCase):
    """端到端训练测试"""
    
    def test_overfitting_small_dataset(self):
        """测试在小数据集上过拟合能力"""
        # 10 个样本，应能达到 100% 准确率
        pass
    
    def test_resource_constraint_respected(self):
        """验证资源约束是否生效"""
        # 训练后 FLOPS 应接近预算
        pass
    
    def test_depth_distribution_diversity(self):
        """验证深度分布的多样性"""
        # 深度熵应 > 阈值
        pass
```

---

## 📦 六、目录结构

```
examples/training/
├── core/
│   ├── __init__.py
│   ├── resource_stats.py       # 资源统计接口
│   ├── trainer.py              # 训练器基类
│   └── config.py               # 配置数据类
│
├── losses/
│   ├── __init__.py
│   ├── focal_loss.py           # Focal Loss
│   ├── balanced_ce.py          # Class-Balanced CE
│   └── resource_loss.py        # 资源感知损失 (新)
│
├── metrics/
│   ├── __init__.py
│   ├── hilbert_metrics.py      # Hilbert 特定指标
│   └── per_class_metrics.py   # 逐类别指标
│
├── samplers/
│   ├── __init__.py
│   ├── balanced_sampler.py     # 类别平衡采样
│   └── progressive_sampler.py  # 渐进式采样
│
├── callbacks/
│   ├── __init__.py
│   ├── resource_monitor.py     # 资源监控
│   ├── checkpoint.py           # 检查点保存
│   ├── visualization.py        # 可视化
│   └── early_stopping.py       # 早停
│
├── tests/
│   ├── test_resource_stats.py  # 资源统计测试
│   ├── test_resource_loss.py   # 资源损失测试
│   ├── test_metrics.py         # 指标测试
│   └── test_end_to_end.py      # 端到端测试
│
├── utils/
│   ├── __init__.py
│   ├── flops_counter.py        # FLOPS 计数器
│   └── data_loading.py         # 数据加载工具
│
├── ARCHITECTURE_DESIGN.md      # 本文档
├── train_fractal_vit_v2.py     # 训练脚本 (新)
└── README.md                   # 快速开始
```

---

## 🎯 七、实施路线图

### Phase 1: 基础设施 (1-2 天)

- [x] `core/resource_stats.py` - 资源统计接口
- [ ] `losses/resource_loss.py` - 资源感知损失
- [ ] `tests/test_resource_stats.py` - 单元测试

### Phase 2: Hilbert 特定指标 (1 天)

- [ ] `metrics/hilbert_metrics.py` - 局部性、LCA 一致性
- [ ] `tests/test_metrics.py` - 指标验证

### Phase 3: 回调系统 (1 天)

- [ ] `callbacks/resource_monitor.py` - 资源监控
- [ ] `callbacks/visualization.py` - 可视化
- [ ] `callbacks/checkpoint.py` - 检查点

### Phase 4: 训练器集成 (1-2 天)

- [ ] `core/trainer.py` - 模块化训练器
- [ ] `core/config.py` - 配置系统
- [ ] `tests/test_end_to_end.py` - 端到端测试

### Phase 5: 文档与示例 (0.5 天)

- [ ] `README.md` - 快速开始指南
- [ ] `train_fractal_vit_v2.py` - 完整训练脚本

**总计**: 约 5-6 天完成完整重构

---

## 📚 八、参考文献与数学依据

### 8.1 Hilbert 曲线理论

1. **Hölder 连续性**:
   - Sagan, H. (1994). *Space-Filling Curves*. Springer.
   - 证明: $\|H(d_1) - H(d_2)\|_2 = O(|d_1 - d_2|^{1/2})$

2. **局部性保持**:
   - Moon, B., et al. (2001). "Analysis of the clustering properties of Hilbert space-filling curve"
   - 实验验证: Hilbert 优于 Z-curve 和光栅扫描

### 8.2 资源感知训练

1. **FLOPS 约束**:
   - Howard, A., et al. (2019). "Searching for MobileNetV3"
   - 使用 ReLU 软约束避免梯度消失

2. **深度熵正则**:
   - Pereyra, G., et al. (2017). "Regularizing Neural Networks by Penalizing Confident Output Distributions"
   - 最大熵原理防止模式崩溃

### 8.3 类别平衡

1. **Effective Number**:
   - Cui, Y., et al. (2019). "Class-Balanced Loss Based on Effective Number of Samples"
   - $E_n = (1 - \beta^n) / (1 - \beta)$

2. **Focal Loss**:
   - Lin, T.-Y., et al. (2017). "Focal Loss for Dense Object Detection"
   - $\mathcal{L} = -\alpha_t (1 - p_t)^\gamma \log p_t$

---

## ✅ 九、设计验证清单

- [x] 每个组件有明确的数学定义
- [ ] 所有公式都有单元测试验证
- [x] 接口设计遵循单一职责原则
- [x] 组件之间通过接口解耦
- [ ] 包含计算验证示例
- [ ] 提供完整的测试套件
- [x] 文档包含设计原理和数学推导

---

## 📝 十、附录

### A. 符号表

| 符号 | 含义 | 维度 |
|------|------|------|
| $I$ | 输入图像 | $\mathbb{R}^{B \times C \times H \times W}$ |
| $T$ | Token 序列 | $\mathbb{R}^{B \times N \times D}$ |
| $N_d$ | 深度 $d$ 的 token 数 | $\mathbb{Z}^+$ |
| $H$ | Hilbert 曲线映射 | $[0, n^2) \leftrightarrow [0, n)^2$ |
| $d_{LCA}$ | LCA 深度 | $\mathbb{Z} \in [0, D_{max}]$ |
| $\mathcal{L}_{resource}$ | 资源感知损失 | $\mathbb{R}^+$ |

### B. 实现细节

**PyTorch 混合精度训练**:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:
    with autocast():
        output = model(batch)
        loss = loss_fn(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**分布式训练支持** (未来扩展):
```python
from torch.nn.parallel import DistributedDataParallel as DDP

model = DDP(model, device_ids=[rank])
```

---

**文档结束**
