# 第九章：训练系统

## 9.1 概述

本章描述了 `FractalCurveViT` 的模块化训练基础设施，包括梯度监控、数值稳定性防御、混合精度训练和完全解耦的训练器架构。

### 关键设计原则

1. **模型无关**：训练器函数不依赖特定模型实现
2. **数值安全优先**：全面的 NaN/Inf 检测和梯度验证
3. **配置驱动**：所有超参数通过数据类配置
4. **数学可验证**：所有组件都有正式定义和单元测试

### 模块结构

```
src/training/
├── config.py           # Config, TrainingHyperparams, NumericalConfig 等
├── train_fractal_vit.py # 主训练入口
├── trainer/
│   ├── epoch_train.py  # train_one_epoch, train_one_epoch_simple
│   ├── epoch_eval.py   # evaluate, evaluate_simple
│   ├── loss.py         # MixupCutmixLoss, compute_loss
│   └── state.py        # TrainingState, EpochMetrics
├── scheduler/
│   └── lr_scheduler.py # WarmupCosineScheduler, create_scheduler
├── monitor/
│   ├── gradient_monitor.py   # GradientMonitor, GradientStatisticsTracker
│   ├── loss_monitor.py       # LossMonitor, LossTracker
│   └── numerical_defense.py   # NumericalDefender, GradientValidator
├── checkpoint/
│   ├── saver.py        # save_checkpoint, save_epoch_stats
│   └── loader.py       # load_checkpoint, find_latest_checkpoint
└── training_logs/
    ├── epoch_logger.py  # EpochLogger
    └── metrics.py       # MetricsTracker, compute_* 函数
```

---

## 9.2 训练入口

### 主脚本

```bash
python -m src.training.train_fractal_vit [arguments]
```

或直接：

```bash
python src/training/train_fractal_vit.py [arguments]
```

### 快速测试

```bash
python src/training/train_fractal_vit.py --quick-test --use-amp
```

### 关键 CLI 参数

| 参数 | 默认值 | 描述 |
|:---------|:--------|:------------|
| `--splitter-type` | `gumbel_topk` | 分割器类型：`hilbert_optimal` (H1SS)、`hilbert_entmax`、`gumbel_topk` |
| `--dim` | 384 | 模型嵌入维度 |
| `--num-layers` | 8 | Transformer 层数 |
| `--heads` | 6 | 注意力头数 |
| `--mlp-dim` | 1536 | FFN 隐藏维度 |
| `--lr` | 8e-5 | 学习率 |
| `--weight-decay` | 0.1 | 权重衰减 |
| `--batch-size` | 128 | 批次大小 |
| `--num-epochs` | 100 | 训练轮数 |
| `--use-amp` | False | 使用自动混合精度 |
| `--compile` | False | 使用 torch.compile |
| `--dataset` | `cub200` | 数据集名称 |

> **注意**：使用 `--splitter-type hilbert_optimal` 获取 H1SS（推荐）或 `hilbert_entmax` 获取 H-entmax。

---

## 9.3 配置系统

### Config 数据类

```python
from src.training import (
    Config,
    TrainingHyperparams,
    NumericalConfig,
    MixedPrecisionConfig,
    create_config,
)

# 创建默认配置
config = create_config(
    num_epochs=100,
    batch_size=128,
    base_lr=8e-5,
    weight_decay=0.1,
)

# 访问超参数
print(config.hyperparams.num_epochs)
print(config.numerical.detect_anomaly)
```

### TrainingHyperparams

核心训练超参数：

| 参数 | 默认值 | 描述 |
|:----------|:--------|:------------|
| `num_epochs` | 100 | 总训练轮数 |
| `batch_size` | 128 | 每迭代批次大小 |
| `accumulation_steps` | 1 | 梯度累积步数 |
| `gradient_clip_norm` | 1.0 | 梯度裁剪阈值 |
| `base_lr` | 5e-4 | 基础学习率 |
| `warmup_epochs` | 5 | LR 预热轮数 |
| `min_lr` | 1e-6 | 最小学习率 |
| `warmup_start_lr` | 1e-7 | 预热期间起始 LR |
| `weight_decay` | 0.05 | 权重衰减系数 |
| `label_smoothing` | 0.0 | 标签平滑因子 |
| `mixup_alpha` | 0.8 | Mixup alpha 参数 |
| `cutmix_alpha` | 1.0 | CutMix alpha 参数 |
| `mixup_cutmix_prob` | 0.5 | 应用 Mixup/Cutmix 的概率 |
| `log_interval` | 50 | 日志记录间隔 |
| `eval_interval` | 1 | 评估间隔 |
| `checkpoint_interval` | 10 | 检查点保存间隔 |

### NumericalConfig

数值稳定性配置：

| 参数 | 默认值 | 描述 |
|:----------|:--------|:------------|
| `detect_anomaly` | False | 启用 PyTorch 异常检测 |
| `check_gradients` | True | 检查梯度的 NaN/Inf |
| `skip_on_nan_grad` | True | 跳过 NaN/Inf 梯度的优化器步骤 |
| `record_grad_norms` | True | 记录梯度范数 |
| `record_layer_grad_norms` | True | 记录每层梯度范数 |
| `record_loss_components` | True | 记录损失组件分解 |

---

## 9.4 训练函数

### 基本训练循环

```python
from src.training import (
    train_one_epoch,
    evaluate,
    TrainingState,
    create_scheduler,
    save_checkpoint,
)

# 创建模型、优化器、数据加载器
model = FractalCurveViT(...)
optimizer = torch.optim.AdamW(model.parameters(), lr=8e-5)
scheduler = create_scheduler(optimizer, config)
scaler = torch.cuda.amp.GradScaler()

# 训练状态
state = TrainingState(
    epoch=0,
    global_step=0,
    best_metric=0.0,
)

# 训练一个 epoch
for epoch in range(config.hyperparams.num_epochs):
    state.epoch = epoch

    # 训练
    metrics = train_one_epoch(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        state=state,
        config=config,
        device=device,
    )

    # 评估
    eval_metrics = evaluate(
        model=model,
        val_loader=val_loader,
        device=device,
    )

    # 保存检查点
    if epoch % 10 == 0:
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            metrics=metrics,
            epoch=epoch,
            is_best=(eval_metrics['top1'] > state.best_metric),
        )
```

### TrainingState

```python
from src.training import TrainingState

state = TrainingState(
    epoch=0,
    global_step=0,
    best_metric=0.0,
    is_best=False,
    optimizer_state=None,
    scheduler_state=None,
    scaler_state=None,
    sampler_state=None,
    metrics_history={},
    warning_count=0,
    nan_skip_count=0,
)
```

---

## 9.5 损失函数

### MixupCutmixLoss

结合 Mixup 和 CutMix 数据增强：

```python
from src.training.trainer.loss import MixupCutmixLoss

loss_fn = MixupCutmixLoss(
    mixup_alpha=0.8,
    cutmix_alpha=1.0,
    mixup_prob=0.5,  # 应用数据增强的概率
    num_classes=200,
)

# 在训练循环中
for images, labels in train_loader:
    # 自动应用 Mixup/Cutmix
    mixed_images, mixed_labels = loss_fn(images, labels)

    outputs = model(mixed_images)
    loss = compute_loss(outputs, mixed_labels, loss_fn)
```

**Mixup 数学形式**：

$$x̃ = \lambda \cdot x_i + (1 - \lambda) \cdot x_j$$
$$ỹ = \lambda \cdot y_i + (1 - \lambda) \cdot y_j$$
$$\lambda \sim \text{Beta}(\alpha, \alpha)$$

**CutMix 数学形式**：

$$x̃ = \text{Mix}(x_i, x_j, \text{region})$$
$$\lambda = 1 - \frac{\text{region\_area}}{\text{total\_area}}$$

### compute_loss

```python
from src.training.trainer.loss import compute_loss

# 对于整数标签
loss = compute_loss(logits, labels, None)

# 对于软标签（Mixup/Cutmix）
loss = compute_loss(logits, mixed_labels, loss_fn)
```

---

## 9.6 数值防御系统

### GradientValidator

检查梯度的数值问题：

```python
from src.training.monitor import GradientValidator

validator = GradientValidator(
    model=model,
    skip_on_issue=True,
    log_warnings=True,
)

# 反向传播后
should_skip = validator.check_gradients()
if not should_skip:
    optimizer.step()
```

### NumericalDefender

全面的数值稳定性保护：

```python
from src.training.monitor import NumericalDefender

defender = NumericalDefender(
    model=model,
    skip_on_nan=True,
    detect_anomaly=False,
)

# 在训练循环中
with defender:
    outputs = model(images)
    loss = criterion(outputs, labels)
    scaler.scale(loss).backward()
```

### GradientMonitor

每层梯度监控：

```python
from src.training.monitor import GradientMonitor

monitor = GradientMonitor(
    model=model,
    log_interval=100,
)

# 在训练循环中
monitor.record_gradients()
if state.global_step % 100 == 0:
    stats = monitor.get_statistics()
    print(stats)
```

---

## 9.7 学习率调度

### WarmupCosineScheduler

```python
from src.training.scheduler import create_scheduler

scheduler = create_scheduler(
    optimizer,
    schedule_type='warmup_cosine',
    num_epochs=100,
    warmup_epochs=15,
    base_lr=8e-5,
    min_lr=1e-6,
)

# 在训练循环中
scheduler.step()
```

**数学形式**：

**预热** ($t \leq T_{warmup}$)：
$$lr(t) = lr_{base} \cdot \frac{t}{T_{warmup}}$$

**余弦退火** ($t > T_{warmup}$)：
$$lr(t) = lr_{min} + (lr_{base} - lr_{min}) \cdot \frac{1}{2}(1 + \cos(\pi \cdot \frac{t - T_{warmup}}{T_{total} - T_{warmup}}))$$

---

## 9.8 检查点管理

### 保存检查点

```python
from src.training.checkpoint import save_checkpoint

save_checkpoint(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    scaler=scaler,
    metrics=metrics,
    epoch=epoch,
    checkpoint_dir='./checkpoints',
    is_best=True,
)
```

### 加载检查点

```python
from src.training.checkpoint import load_checkpoint

checkpoint = load_checkpoint(
    checkpoint_path='./checkpoints/best.pth',
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    scaler=scaler,
)

start_epoch = checkpoint['epoch'] + 1
```

### 查找最新/最佳检查点

```python
from src.training.checkpoint import find_latest_checkpoint, find_best_checkpoint

latest = find_latest_checkpoint('./checkpoints')
best = find_best_checkpoint('./checkpoints')
```

---

## 9.9 指标和日志

### MetricsTracker

```python
from src.training.training_logs import MetricsTracker, compute_accuracy

tracker = MetricsTracker()

# 训练期间
for outputs, targets in dataloader:
    tracker.update(outputs=outputs, targets=targets)

# 计算指标
metrics = tracker.compute()
# {
#     'top1_accuracy': 0.85,
#     'top5_accuracy': 0.98,
#     'num_samples': 1000,
# }
```

### compute_accuracy

```python
from src.training.training_logs import compute_accuracy

top1, top5 = compute_accuracy(logits, targets, topk=(1, 5))
```

### EpochLogger

```python
from src.training.training_logs import EpochLogger

logger = EpochLogger(
    log_dir='./logs',
    experiment_name='fractal_vit_exp',
)

logger.log_epoch(epoch, metrics, lr=scheduler.get_last_lr()[0])
```

---

## 9.10 H1SS 训练特性

### Hilbert Splitter with Stable Selection (H1SS)

H1SS 是推荐的分割器类型，基于六个公理：

| 公理 | 描述 |
|:------|:------------|
| A1 | 1D Hilbert 流形卷积 |
| A2 | 无 Gumbel 扰动 |
| A3 | Entmax 稀疏激活 |
| A4 | 树一致性软约束 |
| A5 | 单次 Entmax 投影 |
| A6 | < 10K 参数 |

### 使用 H1SS 训练

```bash
python -m src.training.train_fractal_vit \
    --splitter-type hilbert_optimal \
    --dataset cub200 \
    --image-size 224 \
    --dim 384 \
    --num-layers 8 \
    --heads 6 \
    --use-amp \
    --compile
```

### 诊断

```python
# 获取模型诊断信息
if hasattr(model, 'get_diagnostics'):
    diagnostics = model.get_diagnostics()
    print(f"Splitter type: {diagnostics.get('splitter_type')}")
    print(f"Gradient coverage: {diagnostics.get('gradient_coverage')}")
```

---

## 9.11 混合精度训练

### 启用 AMP

```bash
python -m src.training.train_fractal_vit --use-amp
```

### 手动 AMP

```python
scaler = torch.cuda.amp.GradScaler()

for images, labels in train_loader:
    with torch.cuda.amp.autocast():
        outputs = model(images)
        loss = criterion(outputs, labels)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

---

## 9.12 最佳实践

### 内存优化

1. **清除分词器缓存**：每批次后调用 `model.clear_tokenizer_cache()`
2. **混合精度**：使用 `--use-amp` 实现 2× 内存减少
3. **梯度检查点**：为大型模型使用 `--gradient-checkpoint`
4. **通道最后**：使用 `--channels-last` 实现更快的卷积

### 训练稳定性

1. **梯度裁剪**：`gradient_clip_norm=1.0`
2. **学习率预热**：15 个 epoch
3. **权重衰减**：大多数配置使用 0.1
4. **数值防御**：启用 `skip_on_nan=True` 处理 NaN 梯度

### 监控

1. **梯度范数**：使用 `GradientMonitor` 观察爆发
2. **损失趋势**：使用 `LossMonitor` 跟踪损失组件
3. **Token 数量变化**：一些变化有利于自适应分词

---

## 9.13 实现状态

| 组件 | 状态 | 位置 |
|:----------|:-------|:---------|
| `train_one_epoch` | ✅ 完成 | `src/training/trainer/epoch_train.py` |
| `evaluate` | ✅ 完成 | `src/training/trainer/epoch_eval.py` |
| `MixupCutmixLoss` | ✅ 完成 | `src/training/trainer/loss.py` |
| `TrainingState` | ✅ 完成 | `src/training/trainer/state.py` |
| `NumericalDefender` | ✅ 完成 | `src/training/monitor/numerical_defense.py` |
| `GradientMonitor` | ✅ 完成 | `src/training/monitor/gradient_monitor.py` |
| `LossMonitor` | ✅ 完成 | `src/training/monitor/loss_monitor.py` |
| `WarmupCosineScheduler` | ✅ 完成 | `src/training/scheduler/lr_scheduler.py` |
| `save_checkpoint` | ✅ 完成 | `src/training/checkpoint/saver.py` |
| `load_checkpoint` | ✅ 完成 | `src/training/checkpoint/loader.py` |
| `EpochLogger` | ✅ 完成 | `src/training/logging/epoch_logger.py` |
| `MetricsTracker` | ✅ 完成 | `src/training/logging/metrics.py` |

---

> **下一章**: [10_testing_qa.md](10_testing_qa.md) - 测试与质量保证
