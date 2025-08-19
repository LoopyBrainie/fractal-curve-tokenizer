# 优化版 Fractal ViT 训练脚本使用指南

## 🚀 快速开始

### 基础训练
```bash
# 默认配置，使用CIFAR-10
python train_fractal_vit.py

# 快速测试（5个epoch，1000样本）
python train_fractal_vit.py --quick-test

# 使用不同数据集
python train_fractal_vit.py --dataset cifar100 --epochs 100
python train_fractal_vit.py --dataset mnist --epochs 30
```

### 进阶配置
```bash
# 启用早停机制
python train_fractal_vit.py --early-stopping --patience 15

# 调整模型参数
python train_fractal_vit.py --dim 256 --depth 12 --heads 12

# 调整训练参数
python train_fractal_vit.py --lr 1e-4 --batch-size 128 --weight-decay 0.05
```

## 📊 参数说明

### 训练参数
- `--epochs`: 训练轮次 (默认: 50)
- `--batch-size`: 批次大小 (默认: 64)
- `--lr`: 初始学习率 (默认: 5e-4，基于训练分析优化)
- `--weight-decay`: 权重衰减 (默认: 0.01)

### 数据参数
- `--dataset`: 数据集选择 (`cifar10`, `cifar100`, `mnist`)
- `--val-split`: 验证集比例 (默认: 0.1)

### 模型参数
- `--dim`: 模型维度 (默认: 192，基于性能分析优化)
- `--depth`: Transformer深度 (默认: 8，相比之前增加)
- `--heads`: 注意力头数 (默认: 8)
- `--dropout`: Dropout率 (默认: 0.1)
- `--max-level`: 分形最大层级 (默认: 3)

### 训练策略
- `--early-stopping`: 启用早停机制
- `--patience`: 早停耐心值 (默认: 10)
- `--quick-test`: 快速测试模式

## 🎯 优化特性

### 1. 基于训练分析的参数优化
- **学习率**: 从1e-3优化到5e-4，基于之前25%准确率的快速测试分析
- **模型深度**: 从6层增加到8层，平衡性能和计算效率
- **调度器**: 使用余弦退火重启，更好的收敛特性

### 2. 多数据集支持
```python
# 自动适配不同数据集的配置
- CIFAR-10: 10类，32x32，RGB
- CIFAR-100: 100类，32x32，RGB  
- MNIST: 10类，28x28，灰度（自动调整到32x32）
```

### 3. 智能文件管理
```
experiments/
  fractal_vit_simple_20250819_120000/
    ├── config.json              # 实验配置
    ├── training_history.json    # 训练历史
    ├── checkpoints/
    │   └── best_model.pth      # 最佳模型
    ├── logs/                    # 训练日志
    └── visualizations/
        └── training_curves.png  # 训练曲线

workspace/
  ├── models/fractal_vit/        # 模型备份
  └── visualizations/            # 可视化备份
```

### 4. 增强的训练监控
- 实时显示学习率变化
- 梯度裁剪防止梯度爆炸
- 过拟合监控图表
- 详细的训练进度显示

### 5. 早停机制
```bash
# 自动在验证损失不改善时停止训练
python train_fractal_vit.py --early-stopping --patience 10
```

## 📈 训练结果分析

### 可视化输出
脚本会自动生成4个子图：
1. **训练损失曲线**: 训练vs验证损失
2. **准确率曲线**: 训练vs验证准确率，标注最佳验证准确率
3. **过拟合指标**: 验证损失-训练损失差异
4. **准确率差异**: 训练-验证准确率差异

### 训练历史记录
所有训练数据保存在`training_history.json`：
```json
{
  "train_losses": [...],
  "train_accs": [...],
  "val_losses": [...], 
  "val_accs": [...],
  "best_val_acc": 85.2,
  "best_epoch": 23,
  "test_acc": 84.1,
  "training_time": 1234.5
}
```

## 🛠️ 故障排除

### 常见问题
1. **CUDA内存不足**: 减少`--batch-size`
2. **训练太慢**: 使用`--quick-test`进行快速验证
3. **过拟合**: 启用`--early-stopping`，增加`--dropout`
4. **欠拟合**: 增加`--depth`，减少`--dropout`

### 性能优化建议
- 使用GPU训练可获得显著加速
- 根据GPU内存调整批次大小
- 对于CIFAR-100，建议增加epochs到100+
- 对于快速实验，使用MNIST数据集

## 🔬 实验设计建议

### 超参数搜索
```bash
# 不同学习率
for lr in 1e-4 3e-4 5e-4 1e-3; do
    python train_fractal_vit.py --lr $lr --quick-test
done

# 不同模型大小
for dim in 128 192 256; do
    python train_fractal_vit.py --dim $dim --quick-test
done
```

### 消融研究
```bash
# 测试分形层级影响
python train_fractal_vit.py --max-level 2 --quick-test
python train_fractal_vit.py --max-level 3 --quick-test
python train_fractal_vit.py --max-level 4 --quick-test
```

---

*这个优化版脚本基于之前的训练分析，针对25%的基线准确率进行了参数和架构优化，应该能获得显著更好的性能。*
