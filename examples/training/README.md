# Hilbert Curve ViT 训练系统 (v2.0)

> **完全重构版本** | 基于形式化分析 + 计算验证 + 组件解耦

---

## 📐 设计哲学

本训练系统专为 **Hilbert Curve ViT** 设计，遵循以下原则：

1. **形式化优先**: 每个组件都有明确的数学定义和推导
2. **解耦设计**: 训练器与模型架构完全分离，组件可独立替换
3. **Hilbert 特定**: 针对分形 tokenization 的特殊性质设计
4. **计算验证**: 所有数学公式都有单元测试验证正确性

---

## 🎯 已完成功能

### ✅ Phase 1: 类别平衡 (I15-1, I15-2)

**问题**: 类别准确率极度不均（头部类别 80%+，尾类别 0%）

**解决方案**:
- **ClassBalancedSampler**: 逆频率加权采样
- **ProgressiveSampler**: 从均匀到类别平衡的渐进采样
- **FocalLoss**: 聚焦困难样本
- **ClassBalancedCrossEntropy**: 基于有效样本数的加权 CE

**状态**: ✅ 完成 + 9/9 单元测试通过

### ✅ Phase 2: 资源感知约束 (新)

**问题**: 
- FLOPS 无约束，计算成本不可控
- Learnable Splitter 缺乏资源感知
- 深度分布可能坍缩（所有 token 同一深度）

**解决方案**:

#### 1. 资源统计接口 (`core/resource_stats.py`)

**只读接口**，模型暴露统计信息：

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
- **深度加权 Token 数**: $N_{weighted} = \sum_d N_d \cdot e^{\alpha d}$
- **FLOPS 分解**: Tokenizer + Attention + FFN + Head

#### 2. 资源感知损失 (`losses/resource_loss.py`)

**三个损失项的加权组合**:

$$\mathcal{L}_{resource} = \lambda_{flops} \mathcal{L}_{flops} + \lambda_{token} \mathcal{L}_{token} + \lambda_{entropy} \mathcal{L}_{entropy}$$

##### (a) FLOPS 约束损失

$$\mathcal{L}_{flops} = \text{ReLU}\left(\frac{\text{FLOPS}_{actual}}{\text{FLOPS}_{budget}} - 1\right)^2$$

**设计原理**:
- 仅惩罚超预算情况 (ReLU)
- 二次惩罚确保严格约束
- RTX 4070 Laptop: FLOPS_budget = 5 GFLOPS

##### (b) Token 数量约束损失

$$\mathcal{L}_{token} = \text{ReLU}(N_{weighted} - N_{budget})^2$$

其中深度权重:

$$N_{weighted} = \sum_{d=0}^{D_{max}} N_d \cdot e^{\alpha d}$$

**深度权重示例** ($\alpha=0.1$):
- depth=0: w=1.00 (基准)
- depth=2: w=1.22 (+22%)
- depth=4: w=1.49 (+49%)

**效果**: 深层 token (小 patch) 成本更高，鼓励模型使用浅层 token

##### (c) 深度熵正则损失

$$\mathcal{L}_{entropy} = (H_{actual} - H_{target})^2$$

其中:

$$H_{actual} = -\sum_d p_d \log p_d, \quad H_{target} = \log(D_{max} + 1)$$

**设计原理**:
- 防止深度坍缩（所有 token 集中在某一深度）
- 鼓励多样化深度分布
- $H_{target}$ 是均匀分布的最大熵

**状态**: ✅ 完成 + 25/25 单元测试通过

---

## 🧪 计算验证结果

### 测试 1: FLOPS 约束

| FLOPS  | 描述 | 损失 $\mathcal{L}_{flops}$ |
|--------|------|-------------------------|
| 4.0G   | 低于预算 | 0.0000 ✓ |
| 5.0G   | 恰好预算 | 0.0000 ✓ |
| 6.0G   | 超出 20% | 0.0400 |
| 7.5G   | 超出 50% | 0.2500 |

**验证**: 二次惩罚关系 $(1.5/1.2)^2 = 6.25$ ✓

### 测试 2: Token 约束 (预算=100, $\alpha=0.1$)

| 深度分布 | 描述 | 加权数 | 损失 $\mathcal{L}_{token}$ |
|----------|------|--------|--------------------------|
| [50, 30, 20] | 浅层为主 | 107.6 | 57.50 |
| [20, 30, 50] | 深层为主 | 114.2 | 202.36 |
| [33, 33, 34] | 均匀分布 | 111.0 | 120.96 |

**验证**: 深层分布受更大惩罚 ✓

### 测试 3: 深度熵正则

| 深度分布 | 描述 | 熵 $H$ | 最大熵 | 损失 $\mathcal{L}_{entropy}$ |
|----------|------|--------|--------|---------------------------|
| [100, 0, 0, 0, 0] | 完全坍缩 | 0.000 | 1.609 | 2.590 |
| [20, 20, 20, 20, 20] | 完全均匀 | 1.609 | 1.609 | 0.000 ✓ |
| [40, 30, 20, 10, 0] | 渐变分布 | 1.280 | 1.609 | 0.109 |

**验证**: 均匀分布达到零损失 ✓

### 测试 4: 深度权重验证

**分布**: [20, 30, 50] (深层为主)

| $\alpha$ | 加权 Token 数 $N_{weighted}$ | 增幅 |
|----------|--------------------------|------|
| 0.00 | 100.0 | 0% (无权重) |
| 0.05 | 106.8 | +6.8% |
| 0.10 | 114.2 | +14.2% |
| 0.20 | 131.2 | +31.2% |

**验证**: 权重呈指数增长 ✓

---

## 📦 目录结构

```
examples/training/
├── core/
│   ├── __init__.py
│   ├── resource_stats.py       ✅ 资源统计接口
│   ├── trainer.py              ⏳ 训练器基类 (TODO)
│   └── config.py               ⏳ 配置数据类 (TODO)
│
├── losses/
│   ├── __init__.py
│   ├── focal_loss.py           ✅ Focal Loss
│   ├── balanced_ce.py          ✅ Class-Balanced CE
│   └── resource_loss.py        ✅ 资源感知损失
│
├── samplers/
│   ├── __init__.py
│   ├── balanced_sampler.py     ✅ 类别平衡采样
│   └── test_samplers.py        ✅ 采样器测试
│
├── tests/
│   ├── test_resource_loss.py   ✅ 25/25 通过
│   └── test_end_to_end.py      ⏳ 端到端测试 (TODO)
│
├── metrics/                    ⏳ Hilbert 特定指标 (TODO)
├── callbacks/                  ⏳ 回调系统 (TODO)
│
├── ARCHITECTURE_DESIGN.md      ✅ 完整架构设计文档
├── train_fractal_vit_v2.py     ⏳ 新训练脚本 (TODO)
└── README.md                   ✅ 本文档
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
uv pip install torch torchvision numpy
```

### 2. 运行计算验证

```bash
cd examples/training
uv run python losses/resource_loss.py
```

**输出**:
```
============================================================
资源损失函数计算验证
============================================================

[测试 1] FLOPS 约束
------------------------------------------------------------
  低于预算: FLOPS=4.0G, L_flops=0.0000
  恰好预算: FLOPS=5.0G, L_flops=0.0000
  超出 20%: FLOPS=6.0G, L_flops=0.0400
  超出 50%: FLOPS=7.5G, L_flops=0.2500
...
```

### 3. 运行单元测试

```bash
cd examples/training
uv run pytest tests/test_resource_loss.py -v
```

**输出**:
```
========================================= test session starts =========================================
...
tests\test_resource_loss.py::TestDepthEntropy::test_uniform_distribution_maximum_entropy PASSED  [ 16%]
tests\test_resource_loss.py::TestFLOPSLoss::test_over_budget_nonzero_loss PASSED                [ 44%]
tests\test_resource_loss.py::TestCombinedLoss::test_all_components PASSED                       [ 72%]
...
========================================== 25 passed in 2.08s =========================================
```

### 4. 使用资源感知损失

```python
from examples.training.core.resource_stats import ModelResourceStats
from examples.training.losses.resource_loss import ResourceAwareLoss

# 创建损失函数
resource_loss_fn = ResourceAwareLoss(
    flops_budget=5e9,      # RTX 4070 Laptop
    token_budget=128,      # 目标 token 数
    depth_weight_alpha=0.1,  # 深度惩罚系数
    lambda_flops=0.1,      # FLOPS 损失权重
    lambda_token=0.01,     # Token 损失权重
    lambda_entropy=0.05,   # 熵损失权重
)

# 前向传播后获取统计
stats = ModelResourceStats(
    total_flops=model_flops,
    token_depth_distribution=model_depth_dist,
    depth_entropy=model_entropy,
)

# 计算资源损失
resource_loss = resource_loss_fn(stats)

# 总损失
total_loss = task_loss + resource_loss

# 反向传播
total_loss.backward()
```

---

## 📊 设计亮点

### 1. 形式化分析

**所有数学公式都经过严格推导**:

- FLOPS 计算公式基于 Transformer 标准复杂度分析
- 深度熵基于 Shannon 信息论
- 深度权重基于指数衰减模型

### 2. 计算验证

**25 个单元测试覆盖所有关键场景**:

- 数学公式正确性（手动计算对比）
- 边界情况（零值、极端值、空数据）
- 梯度流验证（确保可微分）
- 组件集成测试

### 3. 组件解耦

**训练器与模型完全分离**:

```
┌─────────────────┐
│  Training Layer │  ← ResourceAwareLoss
├─────────────────┤
│  Interface      │  ← ModelResourceStats (只读)
├─────────────────┤
│  Model Layer    │  ← FractalCurveViT
└─────────────────┘
```

**优点**:
- 训练策略变更不影响模型
- 模型升级不影响训练器
- 组件可独立测试和优化

### 4. Hilbert 特定设计

**深度加权反映分形特性**:

- 深层 token = 高频细节 → 计算成本高
- 浅层 token = 低频结构 → 计算成本低
- 权重 $w(d) = e^{\alpha d}$ 体现指数关系

**深度熵保证多样性**:

- 防止 Splitter 坍缩（所有 token 同一深度）
- 鼓励多尺度表示（分形自相似性）

---

## 🔬 数学形式化摘要

### 资源损失总体形式

$$\mathcal{L}_{resource} = \lambda_{flops} \underbrace{\left[\text{ReLU}\left(\frac{F}{F_0} - 1\right)\right]^2}_{\mathcal{L}_{flops}} + \lambda_{token} \underbrace{\left[\text{ReLU}(N_w - N_0)\right]^2}_{\mathcal{L}_{token}} + \lambda_{entropy} \underbrace{(H - H_0)^2}_{\mathcal{L}_{entropy}}$$

其中:
- $F$: 实际 FLOPS, $F_0$: FLOPS 预算
- $N_w = \sum_d N_d e^{\alpha d}$: 深度加权 token 数
- $H = -\sum_d p_d \log p_d$: 深度熵, $H_0 = \log(D+1)$: 目标熵

### Hilbert Curve 特性

$$\|H(d_1) - H(d_2)\|_2 \leq C \cdot |d_1 - d_2|^{1/2}$$

**含义**: 序列相邻的 token 在空间上也趋向相邻（局部性保持）

---

## 📈 下一步计划

### Phase 3: Hilbert 特定指标 (1 天)

- [ ] 局部性保持度量
- [ ] 深度-复杂度关联度
- [ ] LCA 距离与 Attention 一致性

### Phase 4: 回调系统 (1 天)

- [ ] 资源监控回调
- [ ] 检查点保存回调
- [ ] 可视化回调

### Phase 5: 模块化训练器 (1-2 天)

- [ ] `HilbertViTTrainer` 基类
- [ ] 配置系统 `TrainingConfig`
- [ ] 端到端训练脚本

### Phase 6: 完整测试 (0.5 天)

- [ ] 端到端集成测试
- [ ] 过拟合能力测试
- [ ] 资源约束效果测试

**总计**: 约 3-4 天完成完整重构

---

## 📚 参考文献

### Hilbert 曲线理论

1. Sagan, H. (1994). *Space-Filling Curves*. Springer.
2. Moon, B., et al. (2001). "Analysis of the clustering properties of Hilbert space-filling curve"

### 资源感知训练

1. Howard, A., et al. (2019). "Searching for MobileNetV3"
2. Pereyra, G., et al. (2017). "Regularizing Neural Networks by Penalizing Confident Output Distributions"

### 类别平衡

1. Cui, Y., et al. (2019). "Class-Balanced Loss Based on Effective Number of Samples"
2. Lin, T.-Y., et al. (2017). "Focal Loss for Dense Object Detection"

---

## ✅ 完成进度

- [x] Phase 1: 类别平衡 (I15-1, I15-2)
- [x] Phase 2a: 资源统计接口
- [x] Phase 2b: 资源感知损失
- [x] Phase 2c: 单元测试 (25/25 通过)
- [ ] Phase 3: Hilbert 特定指标
- [ ] Phase 4: 回调系统
- [ ] Phase 5: 模块化训练器
- [ ] Phase 6: 完整测试

**当前进度**: 40% (2/5 phases 完成)

---

**文档版本**: v2.0  
**最后更新**: 2026-01-04  
**测试覆盖**: 34/34 单元测试通过 (9 samplers + 25 resource loss)  
**代码行数**: ~1500 行 (核心组件 + 测试)
