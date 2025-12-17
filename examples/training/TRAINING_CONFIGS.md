# Training Configuration Examples

## 🚀 核心优化：Streaming Tokenizer

**当前架构 (v0.5.0+)**：使用 `StreamingFractalTokenizerV2`，具有以下特性：

| 特性 | 说明 |
|------|------|
| **Gumbel-Softmax** | 可微分尺度选择，端到端训练 |
| **全 GPU 执行** | 消除 CPU 瓶颈 |
| **自适应多尺度** | 根据区域复杂度选择 patch 大小 |
| **温度退火** | 训练过程中自动调整 Gumbel τ |
| **深度探索优先 (v2.2)** | 训练初期引导模型探索细粒度特征 |

### 深度探索优先 Warmup (v2.2 新增)

在训练初期，模型会对小尺度 patch（更高分辨率）添加正偏置，引导模型优先学习细粒度特征：

```
数学形式:
  logits' = logits + depth_bias × scale_weights
  
其中:
  - depth_bias: 偏置强度，从 2.0 衰减到 0
  - scale_weights: [1.0, 0.67, 0.33, 0.0] (小尺度权重大)
  
调度策略:
  - Warmup 期 (前 20%): 保持最大偏置，强制探索深层级
  - 衰减期 (后 80%): 快速衰减 (指数 2.0)，让模型自主决策
```

**效果**：
- 训练初期：更多使用 4×4 patch，学习边缘、纹理等细节
- 训练后期：偏置消失，模型根据内容自适应选择尺度

---

## RTX 4070 Laptop - Tiny ImageNet (6 hours)

**Hardware Specs:**
- GPU: RTX 4070 Laptop (8GB VRAM, Ampere architecture)
- CPU Workers: 自动检测

**Dataset Setup:**
Tiny ImageNet 将在首次运行时 **自动下载** (~237 MB)。

**推荐配置 (生产训练):**

```bash
uv run python examples/training/train_fractal_vit.py \
    --dataset tiny-imagenet \
    --epochs 100 \
    --batch-size 64 \
    --num-workers 8 \
    --warmup-epochs 10 \
    --lr 8e-4 \
    --weight-decay 0.05 \
    --dropout 0.3 \
    --drop-path 0.2 \
    --label-smoothing 0.1 \
    --gradient-clip 1.0 \
    --use-amp \
    --ffn-type swiglu_level \
    --dim 224 \
    --depth 9 \
    --heads 8 \
    --dim-head 28 \
    --max-level 4 \
    --num-scales 3 \
    --pool cls \
    --patience 15
```

**Expected Performance:**
- Training Time: ~4-6 hours
- Throughput: ~150-250 images/sec
- Peak VRAM: ~6.5GB (safe for 8GB card)
- Parameters: ~5.8M
- Expected Accuracy: 35-45% (top-1)

**Key Optimizations:**
- ✅ **TF32 Auto-enabled**: ~8× matrix multiplication speedup
- ✅ **Mixed Precision (AMP)**: 30-40% faster training
- ✅ **Optimal Batch Size**: 64 maximizes GPU utilization
- ✅ **8 Workers**: Good balance for shared memory
- ✅ **SwiGLU FFN**: -5.8% parameters vs GELU
- ✅ **Early Stopping**: patience=15 防止过拟合
- ✅ **Label Smoothing**: 0.1 正则化

---

## Alternative Configurations

### 🔬 快速验证 (Quick Test)

```bash
uv run python examples/training/train_fractal_vit.py \
    --quick-test \
    --dataset cifar10 \
    --use-amp
```

**Expected:**
- Time: ~2-5 minutes
- Accuracy: 验证模型正常学习

### CIFAR-10 标准训练 (30 min)

```bash
uv run python examples/training/train_fractal_vit.py \
    --dataset cifar10 \
    --epochs 100 \
    --batch-size 128 \
    --num-workers 8 \
    --warmup-epochs 10 \
    --lr 5e-4 \
    --dropout 0.2 \
    --drop-path 0.1 \
    --use-amp \
    --ffn-type swiglu_level \
    --dim 192 \
    --depth 8 \
    --patience 10
```

**Expected:** 85-90% accuracy in ~30 minutes

### CIFAR-100 训练 (1 hour)

```bash
uv run python examples/training/train_fractal_vit.py \
    --dataset cifar100 \
    --epochs 150 \
    --batch-size 128 \
    --num-workers 8 \
    --warmup-epochs 15 \
    --lr 5e-4 \
    --dropout 0.3 \
    --drop-path 0.2 \
    --label-smoothing 0.1 \
    --use-amp \
    --ffn-type swiglu_level \
    --patience 20
```

**Expected:** 70-75% accuracy

### Memory-Constrained (4-5GB VRAM)

```bash
uv run python examples/training/train_fractal_vit.py \
    --dataset tiny-imagenet \
    --epochs 100 \
    --batch-size 32 \
    --accum-steps 2 \
    --num-workers 6 \
    --use-amp \
    --gradient-checkpoint \
    --ffn-type swiglu_level \
    --dim 160 \
    --depth 8 \
    --warmup-epochs 10
```

### Ultra-Low Memory (2-3GB VRAM)

```bash
uv run python examples/training/train_fractal_vit.py \
    --dataset cifar100 \
    --epochs 100 \
    --batch-size 16 \
    --accum-steps 4 \
    --num-workers 4 \
    --use-amp \
    --gradient-checkpoint \
    --ffn-type swiglu \
    --dim 128 \
    --depth 6 \
    --max-level 2
```

### High-End GPU (12-16GB VRAM)

```bash
uv run python examples/training/train_fractal_vit.py \
    --dataset tiny-imagenet \
    --epochs 150 \
    --batch-size 128 \
    --num-workers 12 \
    --warmup-epochs 15 \
    --lr 6e-4 \
    --weight-decay 0.08 \
    --dropout 0.3 \
    --drop-path 0.25 \
    --use-amp \
    --ffn-type swiglu_level \
    --dim 320 \
    --depth 12 \
    --heads 10 \
    --max-level 4 \
    --num-scales 4
```

---

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dataset` | 数据集 (cifar10, cifar100, tiny-imagenet) | cifar10 |
| `--quick-test` | 快速测试模式（3 epochs, 256 samples） | False |
| `--num-workers` | DataLoader workers（留空则自动检测） | 自动 |
| `--gradient-checkpoint` | 梯度检查点，节省 VRAM | False |
| `--ffn-type` | FFN 类型 (gelu, swiglu, swiglu_level) | swiglu_level |
| `--num-scales` | 多尺度层数 | 3 |
| `--patience` | 早停耐心值 | 10 |
| `--drop-path` | Stochastic Depth rate | 0.2 |
| `--label-smoothing` | 标签平滑 | 0.1 |
| `--mixup-alpha` | Mixup alpha (0 禁用) | 0.8 |
| `--cutmix-alpha` | CutMix alpha (0 禁用) | 1.0 |

---

## Tips for RTX 4070 Laptop

1. **Power Management**: 确保接入电源，设置为高性能模式
2. **Cooling**: 良好散热对持续性能至关重要
3. **TF32**: Ampere 架构自动启用，提供 ~8× matmul 加速
4. **监控**: 使用 `nvidia-smi -l 1` 监控 GPU 使用率

**Check GPU Before Training:**
```bash
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'Compute: {torch.cuda.get_device_capability(0)}')"
```

---

## Troubleshooting

### Out of Memory (OOM)

**Solutions (按优先级):**

```bash
# 1. 减小 batch size + 梯度累积
--batch-size 32 --accum-steps 2

# 2. 减小模型尺寸
--dim 192 --depth 8 --max-level 3

# 3. 启用梯度检查点
--gradient-checkpoint

# 4. 极低内存配置
--batch-size 16 --accum-steps 4 --dim 128 --depth 6 --max-level 2
```

### Tiny ImageNet 下载问题

```bash
# 删除损坏文件后重新运行
rm -f data/tiny-imagenet-200.zip
rm -rf data/tiny-imagenet-200
uv run python examples/training/train_fractal_vit.py --dataset tiny-imagenet ...
```

---

## 性能监控

训练第一个 epoch 会输出性能分解：

```
Epoch 1/100:
  Train: loss=2.3591, acc=7.79%
  Val:   loss=2.3301, acc=16.00%
  Time:  10.3s, Throughput: 245.3 samples/s
  Perf:  data=0.8%, fwd=50.2%, mem=5.82GB
```

| 指标 | 说明 | 健康范围 |
|------|------|----------|
| data | 数据加载占比 | <5% |
| fwd | 前向传播占比 | 40-60% |
| mem | VRAM 峰值 | <7GB (8GB 卡) |
| Throughput | 样本吞吐量 | >150 |

**Expected throughput for RTX 4070 Laptop:**

| Config | Batch Size | Throughput | VRAM |
|--------|------------|------------|------|
| dim=192, depth=8 | 64 | ~250 samples/s | ~5GB |
| dim=224, depth=9 | 64 | ~180 samples/s | ~6GB |
| dim=256, depth=10 | 32 | ~100 samples/s | ~7GB |
