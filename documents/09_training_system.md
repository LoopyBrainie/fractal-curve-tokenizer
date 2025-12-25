# 第九章：训练系统 (train_fractal_vit.py)

本项目包含一个功能完备的训练脚本，支持多种数据集和训练策略。

## 9.1 数据集支持

脚本通过 `DatasetSpec` dataclass 统一管理不同数据集的配置。

| 数据集          | 支持  | 自动下载 | 说明           |
|:------------ |:--- |:---- |:------------ |
| CIFAR10      | ✅   | ✅    | 10 类，32×32   |
| CIFAR100     | ✅   | ✅    | 100 类，32×32  |
| MNIST        | ✅   | ✅    | 10 类，28×28   |
| TinyImageNet | ✅   | ✅    | 200 类，64×64  |
| ImageNet     | ✅   | ❌    | 1000 类，需手动下载 |
| COCO         | ✅   | ❌    | 分类模式         |
| Caltech256   | ✅   | ❌    | 256 类        |

### 数据增强策略

| 数据集           | 增强策略                                            |
|:------------- |:----------------------------------------------- |
| MNIST         | Resize, RandomRotation, Normalize               |
| CIFAR10/100   | AutoAugment (CIFAR10 policy)                    |
| ImageNet/COCO | AutoAugment (ImageNet policy)                   |
| 通用            | RandomCrop, RandomHorizontalFlip, RandomErasing |

---

## 9.2 模型构建与配置

### build_model() 函数

根据参数实例化 `FractalCurveViT` 模型。

### 关键命令行参数

| 参数                 | 默认值            | 说明                         |
|:------------------ |:-------------- |:-------------------------- |
| `--tokenizer-type` | `streaming_v3` | Tokenizer 类型 (✅ V3 推荐)     |
| `--bias-mode`      | `lca`          | Hilbert Bias 模式 (推荐)       |
| `--ffn-type`       | `swiglu_level` | FFN 类型                     |
| `--rank`           | 32             | Low-Rank 秩 (仅 low_rank 模式) |
| `--dim`            | 384            | 模型维度                       |
| `--depth`          | 6              | Transformer 层数             |
| `--heads`          | 6              | 注意力头数                      |

---

## 9.3 训练循环

### train_one_epoch()

```python
def train_one_epoch(model, dataloader, optimizer, criterion):
    for images, labels in dataloader:
        # 1. 前向传播
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        # 2. 反向传播
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 3. 清理缓存
        model.clear_tokenizer_cache()
```

### 混合精度训练

使用 `torch.cuda.amp` (或 `torch.amp`) 进行加速：

- `autocast()`: 自动选择精度
- `GradScaler`: 梯度缩放

### 梯度裁剪

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 9.4 学习率调度

采用组合调度策略：

1. **Warmup**: `LinearLR`，前 5 个 Epoch 线性预热
2. **Cosine Annealing**: `CosineAnnealingLR`，余弦退火衰减
3. **SequentialLR**: 串联上述两个调度器

```python
warmup = LinearLR(optimizer, start_factor=0.1, total_iters=5)
cosine = CosineAnnealingLR(optimizer, T_max=epochs-5)
scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[5])
```

---

## 9.5 V3 训练特性

### Variable Depth Tokens 架构

V3 采用 **Variable Depth Tokens** 架构，基于内容自适应四叉树分割，无需温度退火或深度偏置调度：

```python
# V3 训练循环 (简化)
for epoch in range(epochs):
    for images, labels in dataloader:
        output = model(images)
        loss = criterion(output, labels)
        
        # 获取分割统计
        stats = model.tokenizer.get_training_stats()
        print(f"Depth entropy: {stats.get('depth_entropy', 0):.3f}")
        
        loss.backward()
        optimizer.step()
```

### 诊断方法

```python
# 获取分割统计
stats = tokenizer.get_split_stats()
# {
#     'num_tokens': [48, 52, ...],  # 每图像 token 数
#     'depth_distributions': [{0: 4, 1: 16, 2: 28}, ...],  # 深度分布
# }

# 深度分布熵 (多样性指标)
entropy = tokenizer.get_scale_entropy()
```

---

## 9.6 历史：温度退火与 Depth Bias (已废弃)

> **重要**: 以下内容仅作历史参考。V2 (Gumbel-Softmax) 已从代码库完全移除，V3 也已从 Cross-Scale Attention 重构为 Variable Depth Tokens 架构。

### V2 温度调度 (已删除)

V2 使用 Gumbel-Softmax 需要温度退火：$\tau(t) = \tau_{max} \cdot \left(\frac{\tau_{min}}{\tau_{max}}\right)^{t/T}$

### Depth Bias 预热 (已删除)

V2 的深度偏置调度：$\text{logits}'_{i,j,s} = \text{logits}_{i,j,s} + \beta(t) \cdot e^{-\lambda s}$
- warmup = 0.2 (训练前 20% 使用偏置)

---

## 9.7 实验管理

### ExperimentPaths

自动生成带时间戳的实验目录结构：

```
experiments/
└── fractal_vit_simple_20251214_123456/
    ├── checkpoints/
    │   └── best.pth
    ├── logs/
    │   └── training.log
    ├── visualizations/
    │   └── training_curves.png
    └── training_history.json
```

### Checkpoint 保存

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_accuracy': best_acc,
}
torch.save(checkpoint, 'best.pth')
```

---

## 9.8 P0 训练配置修复 (2025-12-23)

> **重要**: 以下参数默认值已针对 Tiny-ImageNet 优化，预期提升准确率 5-8%。

| 参数 | 旧默认值 | 新默认值 | 影响 |
|------|----------|----------|------|
| `mlp_dim` | `dim * 2` | `dim * 4` | FFN 容量提升 +3-5% |
| `dropout` | 0.3 | 0.1 | 减轻过正则化 +2-3% |
| `drop_path` | 0.2 | 0.1 | 减轻过正则化 +1-2% |
| `mixup_alpha` | 0.8 | 0.4 | 适配小图像 +1-2% |
| `weight_decay` | 0.05 | 0.03 | 适配模型规模 +0.5-1% |
| `variable_tokens` | True | False | 训练稳定性 |

## 9.9 使用示例

```bash
# 使用推荐配置训练 (V3 + LCA 模式)
python examples/training/train_fractal_vit.py \
    --dataset tiny-imagenet \
    --tokenizer-type streaming_v3 \
    --bias-mode lca \
    --ffn-type swiglu_level \
    --epochs 100 \
    --batch-size 128 \
    --lr 1e-4 \
    --use-amp

# 使用 Low-Rank 模式 (大模型推荐)
python examples/training/train_fractal_vit.py \
    --dataset imagenet \
    --tokenizer-type streaming_v3 \
    --bias-mode low_rank \
    --rank 32 \
    --dim 768 \
    --epochs 300

# 使用 V1 Tokenizer (固定尺度)
python examples/training/train_fractal_vit.py \
    --dataset cifar10 \
    --tokenizer-type streaming \
```
