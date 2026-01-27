# 第九章：训练系统

## 9.1 概述

本章描述了 `FractalCurveViT` 的模块化训练基础设施，包括提供类别平衡采样、Focal Loss、FLOPS 预算和完全解耦的训练器架构的新 `training` 模块。

### 关键设计原则

1. **模型无关**：训练器不依赖特定模型实现
2. **可插拔组件**：采样器、损失、指标、回调可独立替换
3. **配置驱动**：所有超参数通过数据类配置和 YAML 文件
4. **数学可验证**：所有组件都有正式定义和单元测试

### 模块结构

```
examples/training/
├── samplers/      # ClassBalancedSampler, ProgressiveSampler
├── losses/        # FocalLoss, ClassBalancedCE, CompositeLoss
├── metrics/       # ClassificationMetrics (MCA, per-class, head/tail)
├── schedulers/    # FLOPSBudgetLoss, BudgetScheduler
├── trainer/       # ModularTrainer, Callbacks
└── config/        # ExperimentConfig, ConfigLoader
```

---

## 9.2 数据集支持

### 支持的数据集

| 数据集 | 类别数 | 大小 | 自动下载 |
|:--------|:--------|:-----|:--------------|
| MNIST | 10 | 28×28 | ✓ |
| CIFAR-10 | 10 | 32×32 | ✓ |
| CIFAR-100 | 100 | 32×32 | ✓ |
| Tiny-ImageNet | 200 | 64×64 | ✓ |
| ImageNet | 1000 | 224×224 | ✗ |

### 类别平衡采样

对于不平衡数据集（例如 Tiny-ImageNet），使用 `ClassBalancedSampler`：

```python
from training import ClassBalancedSampler

# 从数据集获取标签
labels = [label for _, label in train_dataset]

# 创建采样器，β=0.9（接近逆频率）
sampler = ClassBalancedSampler(labels, beta=0.9)

# 在 DataLoader 中使用
train_loader = DataLoader(train_dataset, batch_size=128, sampler=sampler)
```

**数学形式**：

$$P(\text{sample } i) = \frac{w_i}{\sum_j w_j}, \quad w_i = \frac{1}{n_{c_i}^\beta}$$

其中：
- $n_{c_i}$：类别 $c_i$ 的样本数
- $\beta \in [0, 1]$：平衡强度（0=均匀，1=逆频率）

---

## 9.3 损失函数

### Focal Loss

用于困难样本挖掘：

```python
from training import FocalLoss

loss_fn = FocalLoss(gamma=2.0, alpha=None)  # alpha=None 表示自动计算
```

**数学形式**：

$$\mathcal{L}_{focal} = -\alpha_c (1 - p_c)^\gamma \log(p_c)$$

### 类别平衡交叉熵

基于样本有效数量（Cui et al., CVPR 2019）：

```python
from training import ClassBalancedCE

loss_fn = ClassBalancedCE(class_counts=class_counts, beta=0.9999)
```

**数学形式**：

$$E_c = \frac{1 - \beta^{n_c}}{1 - \beta}, \quad w_c = \frac{1}{E_c}$$

### 组合 Focal + 类别平衡

```python
from training import FocalClassBalancedLoss

loss_fn = FocalClassBalancedLoss(
    num_classes=200,
    class_counts=class_counts,
    gamma=2.0,
    cb_beta=0.9999,
)
```

---

## 9.4 模型配置

### 推荐超参数

| 参数 | CIFAR-10 | ImageNet | 备注 |
|:----------|:---------|:---------|:------|
| `dim` | 192 | 512 | 嵌入维度 |
| `depth` | 9 | 12 | Transformer 层数 |
| `heads` | 6 | 8 | 注意力头数 |
| `mlp_dim` | 384 | 2048 | FFN 隐藏维度 |
| `patch_size` | 4 | 16 | 基础 patch 大小 |
| `dropout` | 0.1 | 0.1 | dropout 率 |
| `drop_path` | 0.1 | 0.1 | DropPath 率 |

---

## 9.5 模块化训练器

### 基本用法

```python
from training import (
    ModularTrainer,
    TrainerConfig,
    FocalLoss,
    ClassificationMetrics,
    EarlyStoppingCallback,
    CheckpointCallback,
)

# 创建训练器
trainer = ModularTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    loss_fn=FocalLoss(gamma=2.0),
    metrics=ClassificationMetrics(num_classes=200),
    callbacks=[
        EarlyStoppingCallback(patience=20),
        CheckpointCallback(checkpoint_dir="./checkpoints"),
    ],
    config=TrainerConfig(
        num_epochs=100,
        gradient_clip_norm=1.0,
        use_amp=True,
    ),
)

# 开始训练
history = trainer.fit()
```

### 训练循环

`ModularTrainer` 实现：

```python
def train_epoch(self):
    self.model.train()
    for batch_idx, (inputs, targets) in enumerate(self.train_loader):
        # 带 AMP 的前向传播
        with torch.amp.autocast('cuda', enabled=self.config.use_amp):
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)

        # 带梯度裁剪的反向传播
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # 回调
        self.callbacks.on_batch_end(self, ctx)
```

---

## 9.6 FLOPS 预算约束

### FLOPS 计算

```python
from training import FLOPSConfig, compute_transformer_flops

config = FLOPSConfig(embed_dim=384, num_layers=12, num_heads=8)
flops = compute_transformer_flops(n_tokens=197, config=config)
# 4.54 GFLOPS
```

**数学形式**：

$$\text{FLOPS} = L \cdot (12ND^2 + 2N^2D)$$

其中：
- $L$：层数
- $N$：Token 数量
- $D$：嵌入维度

### FLOPS 预算损失

```python
from training import FLOPSBudgetLoss

budget_loss = FLOPSBudgetLoss(
    budget=5e9,  # 5 GFLOPS
    lambda_weight=0.1,
)

loss = budget_loss(actual_flops)
```

**数学形式**：

$$\mathcal{L}_{FLOPS} = \lambda \cdot \text{ReLU}\left(\frac{\text{FLOPS}_{actual}}{\text{Budget}} - 1\right)^2$$

### 预算调度器

动态预算退火（余弦或线性）：

```python
from training import BudgetScheduler

scheduler = BudgetScheduler(
    budget_max=7.5e9,  # 初始宽松
    budget_min=5.0e9,  # 最终严格
    total_epochs=100,
    schedule="cosine",
)

for epoch in range(100):
    current_budget = scheduler.step(epoch)
```

---

## 9.7 学习率调度

### 预热 + 余弦退火

```python
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

# 前 5 个 epoch 预热
warmup = LinearLR(optimizer, start_factor=0.1, total_iters=5)

# 剩余 epoch 的余弦退火
cosine = CosineAnnealingLR(optimizer, T_max=epochs - 5)

# 组合调度器
scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[5])
```

### 调度可视化

```
LR
 │
 │     ╱‾‾‾‾‾‾‾‾╲
 │    ╱          ╲
 │   ╱            ╲
 │  ╱              ╲
 │ ╱                ╲
 │╱                  ╲
 └────────────────────── Epoch
   5    预热   余弦
```

---

## 9.8 YAML 配置系统

### 配置文件

```yaml
# configs/base.yaml
name: base
data:
  batch_size: 128
  sampler_type: default
model:
  embed_dim: 384
  depth: 10
loss:
  type: cross_entropy
budget:
  enabled: false
optimizer:
  type: adamw
  lr: 0.0001
training:
  num_epochs: 100
```

### 配置继承

```yaml
# configs/tiny_imagenet_balanced.yaml
base: base.yaml  # 从 base 继承

name: tiny_imagenet_class_balanced
data:
  sampler_type: class_balanced
  sampler_beta: 0.9
loss:
  type: focal_cb
  focal_gamma: 2.0
budget:
  enabled: true
  flops_budget: 5000000000
```

### 加载配置

```python
from training import ConfigLoader

loader = ConfigLoader(config_dir='configs')
config = loader.load('tiny_imagenet_balanced.yaml')

# 命令行覆盖
config = loader.load('base.yaml', overrides={
    'training.num_epochs': 50,
    'optimizer.lr': 5e-5,
})
```

---

## 9.9 指标

### 分类指标

```python
from training import ClassificationMetrics

metrics = ClassificationMetrics(num_classes=200, topk=(1, 5))

# 训练期间
for outputs, targets in dataloader:
    metrics.update(outputs, targets)

# 计算结果
result = metrics.compute()
print(f"Top-1: {result.top1_accuracy:.2%}")
print(f"MCA: {result.mean_class_accuracy:.2%}")
print(f"Head Acc: {result.head_accuracy:.2%}")
print(f"Tail Acc: {result.tail_accuracy:.2%}")
```

**平均类别准确率 (MCA)**：

$$\text{MCA} = \frac{1}{C} \sum_{c=1}^C \frac{\text{TP}_c}{n_c}$$

---

## 9.10 V3 训练特性

### 变深度 Tokens

V3 使用内容自适应四叉树分割，无需温度退火：

```python
for epoch in range(epochs):
    for images, labels in dataloader:
        output = model(images)
        loss = criterion(output, labels)

        # 可选：监控分割统计
        if epoch % 10 == 0:
            stats = model.tokenizer.get_split_stats()
            print(f"Mean tokens: {np.mean(stats['num_tokens']):.1f}")
            print(f"Depth entropy: {model.tokenizer.get_scale_entropy():.3f}")

        loss.backward()
        optimizer.step()
```

### 诊断

```python
# 分割统计
stats = tokenizer.get_split_stats()
# {
#     'num_tokens': [48, 52, 64, ...],      # 每张图像的 token 数
#     'depth_distributions': [
#         {0: 1, 1: 4, 2: 16, 3: 27},       # 每图像深度计数
#         ...
#     ],
#     'mean_complexity': 0.42,
# }

# 深度多样性（越高越好）
entropy = tokenizer.get_scale_entropy()  # 目标: > 1.5
```

---

## 9.11 实验管理

### 目录结构

```
experiments/
└── fractal_vit_simple_20251214_123456/
    ├── checkpoints/
    │   ├── best.pth
    │   └── last.pth
    ├── logs/
    │   └── training.log
    ├── visualizations/
    │   ├── training_curves.png
    │   └── attention_maps.png
    └── training_history.json
```

### 检查点格式

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_accuracy': best_acc,
    'config': config.__dict__,
}
torch.save(checkpoint, 'best.pth')
```

### 加载检查点

```python
checkpoint = torch.load('best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

---

## 9.12 训练脚本

### 命令行界面

```bash
python examples/training/train_fractal_vit.py \
    --dataset cifar10 \
    --image-size 32 \
    --dim 192 \
    --depth 9 \
    --heads 6 \
    --mlp-dim 384 \
    --tokenizer-type streaming_v3 \
    --bias-mode lca \
    --ffn-type swiglu_level \
    --epochs 100 \
    --batch-size 128 \
    --lr 5e-4 \
    --weight-decay 0.03
```

### 关键参数

| 参数 | 默认值 | 描述 |
|:---------|:--------|:------------|
| `--tokenizer-type` | `streaming_v3` | 分词器类型（仅 V3） |
| `--bias-mode` | `lca` | Hilbert 偏置模式 |
| `--ffn-type` | `swiglu_level` | FFN 类型 |
| `--dim` | 384 | 模型维度 |
| `--depth` | 6 | Transformer 层数 |
| `--heads` | 6 | 注意力头数 |
| `--lr` | 5e-4 | 学习率 |
| `--weight-decay` | 0.03 | 权重衰减 |

---

## 9.13 最佳实践

### 内存优化

1. **清除分词器缓存**：每批次后调用 `model.clear_tokenizer_cache()`
2. **混合精度**：使用 `torch.cuda.amp` 实现 2× 内存减少
3. **梯度检查点**：为大型模型启用

### 训练稳定性

1. **梯度裁剪**：`max_norm=1.0`
2. **学习率预热**：5-10 个 epoch
3. **权重衰减**：大多数配置使用 0.03

### 类别不平衡

1. **使用 ClassBalancedSampler**：`beta=0.9` 实现强平衡
2. **使用 FocalLoss**：`gamma=2.0` 聚焦困难样本
3. **监控 MCA**：不仅仅是 Top-1 准确率

### 监控

1. **深度分布熵**：应 > 1.5
2. **Token 数量变化**：一些变化是健康的
3. **梯度范数**：观察爆发
4. **每类准确率**：检查头部 vs 尾部类别

---

## 9.14 实现状态

| 组件 | 状态 | 位置 |
|:----------|:-------|:---------|
| ClassBalancedSampler | ✅ 完成 | `training.samplers` |
| ProgressiveSampler | ✅ 完成 | `training.samplers` |
| FocalLoss | ✅ 完成 | `training.losses` |
| ClassBalancedCE | ✅ 完成 | `training.losses` |
| ClassificationMetrics | ✅ 完成 | `training.metrics` |
| FLOPSBudgetLoss | ✅ 完成 | `training.schedulers` |
| BudgetScheduler | ✅ 完成 | `training.schedulers` |
| ModularTrainer | ✅ 完成 | `training.trainer` |
| ConfigLoader | ✅ 完成 | `training.config` |
| W&B 集成 | ⏳ 待定 | - |
| 可视化面板 | ⏳ 待定 | - |

**测试**：31/31 通过

> **下一章**: [10_testing_qa.md](10_testing_qa.md) - 测试与质量保证
