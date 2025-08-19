# 完整版 Fractal ViT 训练脚本使用说明

## 📋 脚本特性

### 🎯 基于快速训练结果的优化
根据快速训练的结果分析（最佳准确率25%，训练时间338.44秒），我们对完整版脚本进行了以下优化：

#### 🔧 模型参数优化
- **维度降低**: 从256降至192，提高训练效率
- **深度增加**: 从6层增至8层，提高表现力
- **头数优化**: 从8个降至6个，平衡复杂度
- **学习率调整**: 从3e-4调至5e-4，基于训练曲线优化

#### 🛡️ 训练稳定性改进
- **早停机制**: 默认启用15轮早停，防止过拟合
- **权重衰减**: 从0.1降至0.05，减少过度正则化
- **Worker数量**: 从4降至2，避免内存问题

### 📁 文件管理规范
严格遵循项目文件管理规范：

```
项目结构:
├── experiments/
│   └── fractal_vit_complete_YYYYMMDD_HHMMSS/    # 实验数据
│       ├── checkpoints/                          # 模型检查点
│       ├── logs/                                # 训练日志
│       └── results/                             # 实验结果
└── workspace/
    ├── models/fractal_vit/                      # 最佳模型文件
    ├── results/                                 # 结果数据
    ├── visualizations/                          # 训练曲线图
    └── reports/                                 # 实验报告
```

### 🚀 使用方式

#### 基础训练
```bash
python train_fractal_vit_complete.py
```

#### 快速测试模式
```bash
python train_fractal_vit_complete.py --subset-size 1000 --epochs 10
```

#### 高性能训练
```bash
python train_fractal_vit_complete.py \
    --epochs 100 \
    --dim 256 \
    --depth 12 \
    --lr 3e-4 \
    --batch-size 128
```

#### CIFAR-100训练
```bash
python train_fractal_vit_complete.py \
    --dataset CIFAR100 \
    --epochs 150 \
    --early-stopping 20
```

### 📊 输出文件说明

#### 实验目录 (`experiments/fractal_vit_complete_YYYYMMDD_HHMMSS/`)
- **checkpoints/**
  - `best.pth`: 最佳验证准确率的完整检查点
  - `latest.pth`: 最新的训练检查点  
  - `epoch_*.pth`: 定期保存的检查点(每10轮)

- **logs/**
  - `training.log`: 详细的训练日志
  - `tensorboard/`: TensorBoard日志文件

- **results/**
  - `training_curves.png/pdf`: 训练曲线可视化
  - `training_report.json`: JSON格式的详细报告
  - `training_report.md`: Markdown格式的报告

#### Workspace目录
- **models/fractal_vit/**
  - `fractal_vit_best_YYYYMMDD_HHMMSS.pth`: 最佳模型完整检查点
  - `fractal_vit_model_YYYYMMDD_HHMMSS.pth`: 仅模型状态字典(体积更小)

- **results/**
  - `fractal_vit_report_YYYYMMDD_HHMMSS.json`: 实验结果数据
  - `config_YYYYMMDD_HHMMSS.json`: 实验配置备份

- **visualizations/**
  - `fractal_vit_training_curves_YYYYMMDD_HHMMSS.png`: 训练曲线图

- **reports/**
  - `fractal_vit_report_YYYYMMDD_HHMMSS.md`: 详细实验报告

### ⚙️ 参数配置

#### 数据集参数
- `--dataset`: 数据集选择 (CIFAR10/CIFAR100/MNIST)
- `--subset-size`: 使用数据子集大小(用于快速测试)

#### 模型参数  
- `--image-size`: 输入图像尺寸 (默认32)
- `--dim`: 嵌入维度 (默认192，已优化)
- `--depth`: Transformer深度 (默认8，已优化)
- `--heads`: 注意力头数 (默认6，已优化)
- `--min-patch-size`: 最小patch尺寸 (默认[4,4])
- `--max-level`: 最大分形层级 (默认3)
- `--no-learnable-split`: 禁用可学习分割

#### 训练参数
- `--epochs`: 训练轮次 (默认50，已优化)
- `--batch-size`: 批次大小 (默认64)
- `--lr`: 学习率 (默认5e-4，已优化)
- `--optimizer`: 优化器 (adamw/adam)
- `--scheduler`: 学习率调度器 (cosine/linear)

#### 系统参数
- `--no-amp`: 禁用混合精度训练
- `--no-tensorboard`: 禁用TensorBoard日志
- `--resume`: 从检查点恢复训练
- `--early-stopping`: 早停耐心值 (默认15，已启用)

### 📈 性能期望

基于快速训练结果分析和优化：

#### 预期改进
- **收敛速度**: 优化学习率应提高收敛速度
- **最终准确率**: 增加深度和早停应提高最终准确率
- **训练稳定性**: 减少权重衰减应提高训练稳定性
- **内存使用**: 减少worker和维度应降低内存使用

#### 训练时间估计
- **CPU训练**: 约2-4小时 (50轮)
- **GPU训练**: 约30-60分钟 (50轮)
- **快速测试**: 约5-10分钟 (10轮，1000样本)

### 🔧 故障排除

#### 内存不足
```bash
python train_fractal_vit_complete.py --batch-size 32 --no-amp
```

#### 训练太慢
```bash
python train_fractal_vit_complete.py --subset-size 5000 --epochs 20
```

#### GPU内存不足
```bash
python train_fractal_vit_complete.py --batch-size 16 --dim 128 --depth 6
```

### 💡 最佳实践

1. **首次使用**: 先运行快速测试模式验证环境
2. **正式训练**: 使用默认参数开始，根据结果调整
3. **性能优化**: 监控训练曲线，适时调整学习率
4. **资源管理**: 根据硬件配置调整批次大小和模型维度
5. **实验跟踪**: 使用TensorBoard监控训练过程

### 📚 相关文档
- [快速训练脚本](./train_fractal_vit.py)
- [训练脚本说明](./TRAINING_README.md)
- [项目主文档](../../README.md)
- [实验指南](../../EXPERIMENT_GUIDE.md)

---

**更新日期**: 2025年8月19日
**版本**: v2.0 (基于快速训练结果优化)
