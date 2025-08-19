# Fractal ViT 训练脚本说明

本目录包含了专门用于训练Fractal ViT模型的训练脚本。

## 📁 文件说明

### 1. `train_fractal_vit.py` - 简化版训练脚本 ⭐ **新增**
**推荐用于快速测试和原型开发**

```bash
# 基础用法
python train_fractal_vit.py

# 快速测试（5轮训练，1000样本）
python train_fractal_vit.py --quick-test

# 自定义参数
python train_fractal_vit.py --epochs 20 --batch-size 32 --lr 5e-4
```

**特点:**
- 🚀 简单易用，开箱即用
- ⚡ 支持快速测试模式
- 📊 自动生成训练曲线图
- 💾 自动保存最佳模型
- 🎯 专注核心功能，代码简洁

### 2. `train_fractal_vit_complete.py` - 完整版训练脚本 ⭐ **新增**
**推荐用于正式实验和研究**

```bash
# 完整训练
python train_fractal_vit_complete.py

# CIFAR-100训练
python train_fractal_vit_complete.py --dataset CIFAR100 --epochs 200

# 自定义模型配置
python train_fractal_vit_complete.py --dim 384 --depth 12 --heads 6 --max-level 4
```

**特点:**
- 🔧 完整的配置选项
- 📈 TensorBoard支持
- 💾 完善的检查点系统
- 📊 多数据集支持(CIFAR-10/100, MNIST)
- 📋 详细的实验报告生成
- 🔄 支持训练恢复和早停
- 📱 混合精度训练支持

### 3. 其他现有脚本

- `train_enhanced_fractal_vit.py` - 增强版演示脚本
- `train_fractal_cifar10.py` - CIFAR-10专用训练
- `train_standard_vit_baseline.py` - 标准ViT基线

## 🚀 快速开始

### 方案1: 简单训练（推荐新手）
```bash
cd tests/training_scripts
python train_fractal_vit.py --quick-test
```
这将运行5轮快速训练，大约需要5-10分钟。

### 方案2: 完整训练（推荐研究）
```bash
cd tests/training_scripts
python train_fractal_vit_complete.py --epochs 50 --dataset CIFAR10
```
这将运行完整的训练流程，包含详细的日志和结果分析。

## ⚙️ 参数说明

### 简化版参数
- `--epochs`: 训练轮次 (默认: 50)
- `--batch-size`: 批次大小 (默认: 64)
- `--lr`: 学习率 (默认: 1e-3)
- `--quick-test`: 启用快速测试模式
- `--seed`: 随机种子 (默认: 42)

### 完整版参数
#### 数据集参数
- `--dataset`: 选择数据集 (CIFAR10/CIFAR100/MNIST)
- `--subset-size`: 使用数据子集大小

#### 模型参数
- `--image-size`: 输入图像尺寸 (默认: 32)
- `--dim`: 嵌入维度 (默认: 256)
- `--depth`: Transformer深度 (默认: 6)
- `--heads`: 注意力头数 (默认: 8)
- `--min-patch-size`: 最小patch尺寸 (默认: [4, 4])
- `--max-level`: 最大分形层级 (默认: 3)
- `--no-learnable-split`: 禁用可学习分割

#### 训练参数
- `--epochs`: 训练轮次 (默认: 100)
- `--batch-size`: 批次大小 (默认: 64)
- `--lr`: 学习率 (默认: 3e-4)
- `--optimizer`: 优化器选择 (adamw/adam)
- `--scheduler`: 学习率调度器 (cosine/linear)

#### 系统参数
- `--no-amp`: 禁用混合精度训练
- `--no-tensorboard`: 禁用TensorBoard
- `--resume`: 从检查点恢复训练
- `--early-stopping`: 早停耐心值

## 📊 输出文件

### 简化版输出
- `best_fractal_vit.pth`: 最佳模型权重
- `fractal_vit_training_curves.png`: 训练曲线图

### 完整版输出
```
fractal_vit_experiments/exp_YYYYMMDD_HHMMSS/
├── checkpoints/
│   ├── best.pth           # 最佳模型检查点
│   ├── latest.pth         # 最新检查点
│   └── epoch_*.pth        # 定期检查点
├── logs/
│   ├── training.log       # 训练日志
│   └── tensorboard/       # TensorBoard日志
├── results/
│   ├── training_curves.png    # 训练曲线图
│   ├── training_report.json   # JSON格式报告
│   └── training_report.md     # Markdown格式报告
└── config.json            # 实验配置
```

## 💡 使用建议

### 🧪 快速测试新想法
```bash
python train_fractal_vit.py --quick-test --epochs 3
```

### 🔬 正式实验
```bash
python train_fractal_vit_complete.py \
    --dataset CIFAR10 \
    --epochs 100 \
    --dim 384 \
    --depth 12 \
    --max-level 4 \
    --learnable-split
```

### 📱 资源受限环境
```bash
python train_fractal_vit.py \
    --epochs 20 \
    --batch-size 32 \
    --subset-size 5000
```

### 🏆 追求最佳性能
```bash
python train_fractal_vit_complete.py \
    --dataset CIFAR10 \
    --epochs 200 \
    --dim 512 \
    --depth 16 \
    --heads 8 \
    --batch-size 128 \
    --lr 1e-4 \
    --early-stopping 20
```

## 🔧 环境要求

- Python 3.7+
- PyTorch 1.8+
- torchvision
- numpy
- tqdm
- matplotlib (可选，用于绘图)
- tensorboard (可选，用于完整版)

## 📚 相关文档

- [项目主README](../../README.md)
- [实验指南](../../EXPERIMENT_GUIDE.md)
- [模型架构说明](../../vit_pytorch/)

---

**提示**: 如果遇到GPU内存不足，请减小batch_size或使用--subset-size参数限制数据大小。
