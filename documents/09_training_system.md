# 第九章：训练系统 (train_fractal_vit.py)

本项目包含一个功能完备的训练脚本，支持多种数据集和训练策略。

## 9.1 数据集支持

脚本通过 `DatasetSpec` dataclass 统一管理不同数据集的配置。

| 数据集 | 支持 | 自动下载 | 说明 |
| :--- | :--- | :--- | :--- |
| CIFAR10 | ✅ | ✅ | 10 类，32×32 |
| CIFAR100 | ✅ | ✅ | 100 类，32×32 |
| MNIST | ✅ | ✅ | 10 类，28×28 |
| TinyImageNet | ✅ | ✅ | 200 类，64×64 |
| ImageNet | ✅ | ❌ | 1000 类，需手动下载 |
| COCO | ✅ | ❌ | 分类模式 |
| Caltech256 | ✅ | ❌ | 256 类 |

### 数据增强策略

| 数据集 | 增强策略 |
| :--- | :--- |
| MNIST | Resize, RandomRotation, Normalize |
| CIFAR10/100 | AutoAugment (CIFAR10 policy) |
| ImageNet/COCO | AutoAugment (ImageNet policy) |
| 通用 | RandomCrop, RandomHorizontalFlip, RandomErasing |

---

## 9.2 模型构建与配置

### build_model() 函数

根据参数实例化模型：
- GPU: `NextGenerationFractalViT`
- CPU: `SimpleFractalViT` (自动减小规模)

### 关键命令行参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `--tokenizer-type` | `streaming_v2` | Tokenizer 类型 |
| `--bias-mode` | `low_rank` | Hilbert Bias 模式 |
| `--ffn-type` | `swiglu_level` | FFN 类型 |
| `--rank` | 32 | Low-Rank 秩 |
| `--dim` | 384 | 模型维度 |
| `--depth` | 6 | Transformer 层数 |
| `--heads` | 6 | 注意力头数 |

---

## 9.3 训练循环

### train_one_epoch()

```python
def train_one_epoch(model, dataloader, optimizer, criterion):
    for images, labels in dataloader:
        # 1. 前向传播
        with autocast():
            outputs = model(images)
            loss_cls = criterion(outputs, labels)
        
        # 2. 辅助损失 (仅 Legacy Tokenizer)
        if model.tokenizer_type == 'legacy':
            reward = -loss_cls.detach()
            loss_aux = model.get_tokenizer_loss(reward)
            total_loss = loss_cls + loss_aux
        else:
            total_loss = loss_cls
        
        # 3. 反向传播
        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # 4. 清理缓存
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

## 9.5 实验管理

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

## 9.6 REINFORCE 训练 (仅 Legacy Tokenizer)

### 策略梯度损失

$$\mathcal{L}_{policy} = -\frac{1}{N} \sum_{i=1}^N (R_i - b) \cdot \log \pi(a_i|s_i)$$

其中：
- $R_i$: 奖励信号 (Reward) = $-\mathcal{L}_{CE}$
- $b$: 基线 (Baseline) = EMA of rewards
- $\pi(a|s)$: 策略网络输出

### 熵正则化

$$\mathcal{L}_{entropy} = -\sum \pi(a|s) \log \pi(a|s)$$

鼓励策略保持随机性，防止过早收敛。

### 总辅助损失

$$\mathcal{L}_{aux} = \mathcal{L}_{policy} - \lambda \cdot \mathcal{L}_{entropy}$$

---

## 9.7 使用示例

```bash
# 使用推荐配置训练
python examples/training/train_fractal_vit.py \
    --dataset cifar10 \
    --tokenizer-type streaming_v2 \
    --bias-mode low_rank \
    --ffn-type swiglu_level \
    --epochs 100 \
    --batch-size 64 \
    --lr 5e-4

# 使用 Legacy Tokenizer (不推荐)
python examples/training/train_fractal_vit.py \
    --dataset cifar10 \
    --tokenizer-type legacy \
    --epochs 100
```
