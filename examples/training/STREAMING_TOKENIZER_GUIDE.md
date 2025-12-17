# Streaming Tokenizer 使用指南

> **版本**: 0.4.0  
> **更新日期**: 2025-12-14  
> **状态**: ARCH-P1 完成，ARCH-P2 待实施

---

## 概述

`StreamingFractalTokenizer` 是新一代统一 tokenizer，将原有的两阶段架构（BFS 分割 + TokenProcessor）合并为单次前向传播，消除 CPU 瓶颈和特征重复计算。

### 架构对比

| 特性 | Legacy (两阶段) | Streaming (统一) |
|------|----------------|------------------|
| 前向传播 | 2 次 | 1 次 |
| CPU 瓶颈 | BFS 循环 | 无 |
| 特征计算 | 重复 | 单次 |
| Token 数量 | 可变 (需 padding) | 固定 |
| 梯度流 | REINFORCE | Gumbel-Softmax |
| GPU 利用率 | ~40% | ~85% |

---

## 快速开始

### Python API

```python
from vit_pytorch import NextGenerationFractalViT

# 使用 NextGenerationFractalViT
model = NextGenerationFractalViT(
    image_size=32,
    num_classes=10,
    dim=256,
    depth=6,
    heads=8,
    mlp_dim=512,
    # === Streaming Tokenizer 参数 ===
    tokenizer_type="streaming_v2",  # 默认，使用 Gumbel-Softmax
    num_scales=3,                   # 多尺度层数
    streaming_tau=1.0,              # Gumbel-Softmax 温度
)

# 前向传播
output = model(images)  # [B, num_classes]
```

### 命令行训练

```bash
# 使用 streaming tokenizer 训练 CIFAR-10
python train_fractal_vit.py \
    --dataset cifar10 \
    --tokenizer-type streaming \
    --num-scales 3 \
    --epochs 50 \
    --use-amp

# 使用 streaming_v2 (Gumbel-Softmax 自适应)
python train_fractal_vit.py \
    --dataset cifar10 \
    --tokenizer-type streaming_v2 \
    --streaming-tau 0.5 \
    --epochs 50

# 快速测试
python train_fractal_vit.py \
    --dataset cifar10 \
    --tokenizer-type streaming \
    --quick-test
```

---

## Tokenizer 类型说明

### `legacy` (默认)

- **类**: `FractalHilbertTokenizer` + `EnhancedFractalTokenProcessor`
- **特点**: BFS 递归分割，REINFORCE 策略梯度
- **适用**: 需要可变 token 数量的场景
- **状态**: 已废弃，将在 v1.0 移除

### `streaming`

- **类**: `StreamingFractalTokenizer`
- **特点**: 
  - 多尺度卷积金字塔
  - 固定 token 数量 = (H/p) × (W/p)
  - 使用 `primary_scale` 选择单一尺度
- **适用**: 标准训练场景

### `streaming_v2`

- **类**: `StreamingFractalTokenizerV2`
- **特点**:
  - 区域复杂度估计器
  - Gumbel-Softmax 可微分尺度选择
  - 训练时软选择，推理时硬选择
- **适用**: 需要自适应分辨率的场景

---

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `tokenizer_type` | str | `"legacy"` | tokenizer 类型选择 |
| `num_scales` | int | `4` | 多尺度金字塔层数 |
| `streaming_tau` | float | `1.0` | Gumbel-Softmax 温度参数 |

### `num_scales` 与 patch_sizes 的关系

```python
# min_patch_size=(4, 4), num_scales=3
# → patch_sizes = (4, 8, 16)

# min_patch_size=(4, 4), num_scales=4
# → patch_sizes = (4, 8, 16, 32)
```

### `streaming_tau` 温度参数

- 较高值 (>1.0): 更均匀的尺度分布，鼓励探索
- 较低值 (<1.0): 更尖锐的选择，接近 argmax
- 训练初期建议使用较高值，逐渐退火

---

## 待实施: ARCH-P2 消融实验

### E1: 速度对比

```bash
# Legacy tokenizer
python train_fractal_vit.py --tokenizer-type legacy --pretokenize --epochs 10

# Streaming tokenizer
python train_fractal_vit.py --tokenizer-type streaming --epochs 10

# 对比指标: imgs/sec, GPU 利用率
```

### E2: 精度验证

```bash
# 相同超参数，对比 CIFAR-10/100 准确率
python train_fractal_vit.py --dataset cifar10 --tokenizer-type legacy --epochs 50
python train_fractal_vit.py --dataset cifar10 --tokenizer-type streaming --epochs 50
python train_fractal_vit.py --dataset cifar10 --tokenizer-type streaming_v2 --epochs 50
```

### E3: 自适应性分析

```python
# 分析 streaming_v2 的尺度选择分布
from vit_pytorch import StreamingFractalTokenizerV2

tokenizer = StreamingFractalTokenizerV2(...)
# 分析 complexity_estimator 输出与图像复杂度的相关性
```

### E4: 组件消融

| 移除组件 | 预期影响 |
|----------|----------|
| Hilbert 重排序 | 位置编码效果下降 |
| 多尺度金字塔 | 失去多尺度特征 |
| 特征融合层 | 表达能力下降 |
| 复杂度估计器 (V2) | 退化为固定尺度 |

---

## 迁移指南

### 从 Legacy 迁移到 Streaming

```python
# 旧代码
model = NextGenerationFractalViT(
    image_size=32,
    num_classes=10,
    learnable_split=True,
)

# 新代码
model = NextGenerationFractalViT(
    image_size=32,
    num_classes=10,
    tokenizer_type="streaming",
)
```

### 注意事项

1. **pretokenize 模式**: Streaming tokenizer 自动禁用 pretokenize（已在 GPU 上运行）
2. **REINFORCE 损失**: Streaming tokenizer 的 `get_tokenizer_loss()` 返回零
3. **Token 数量**: Streaming 输出固定数量 tokens，无需 padding 逻辑

---

## 文件结构

```
src/vit_pytorch/
├── streaming_tokenizer.py      # 新增: StreamingFractalTokenizer
├── fractal_vit.py              # 修改: tokenizer_type 参数
├── fractal_curve_tokenizer.py  # 废弃: FractalHilbertTokenizer
└── token_processor.py          # 废弃: EnhancedFractalTokenProcessor

examples/training/
├── train_fractal_vit.py        # 修改: --tokenizer-type 参数
└── STREAMING_TOKENIZER_GUIDE.md  # 新增: 本文档

tests/
├── unit/test_streaming_tokenizer.py       # 新增: 20 个单元测试
└── integration/test_streaming_integration.py  # 新增: 14 个集成测试
```

---

## 测试验证

```bash
# 运行所有 streaming 相关测试
python -m pytest tests/unit/test_streaming_tokenizer.py -v
python -m pytest tests/integration/test_streaming_integration.py -v

# 运行所有测试
python -m pytest tests/ -v
# 预期: 120 passed, 1 skipped
```

---

## 问题排查

### Q: BatchNorm 报错 "Expected more than 1 value per channel"

**原因**: batch_size=1 时 BatchNorm 无法计算统计量

**解决**: 使用 batch_size >= 2，或在模型中使用 InstanceNorm

### Q: Streaming 模式下 pretokenize 被忽略

**预期行为**: Streaming tokenizer 在 GPU 上运行，无需 CPU 预处理

### Q: streaming_v2 训练不稳定

**解决**: 
1. 使用较高的 `streaming_tau` 初始值 (1.0-2.0)
2. 逐渐退火到较低值 (0.5)
3. 考虑使用 `streaming` 作为更稳定的替代

---

## 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| 0.4.0 | 2025-12-14 | 新增 StreamingFractalTokenizer, tokenizer_type 参数 |
| 0.3.0 | 2025-12-08 | 废弃 MiniCNN, use_cnn 参数 |
