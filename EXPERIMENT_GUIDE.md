# Fractal ViT vs Standard ViT 对比实验

这是一个完整的控制变量实验，用于比较Fractal ViT和标准ViT在CIFAR-10数据集上的性能。

## 🚀 快速开始

### 方法1: 使用启动脚本（推荐）
```bash
python run_experiments.py
```

### 方法2: 直接运行主程序
```bash
# 标准实验（100轮，批次大小128）
python train_fractal_vs_standard_cifar10.py

# 快速测试（5轮）
python train_fractal_vs_standard_cifar10.py --epochs 5 --batch-size 64

# 自定义参数
python train_fractal_vs_standard_cifar10.py --epochs 50 --batch-size 256 --lr 5e-4 --seed 2024
```

## 📊 实验设计

### 控制变量
为了确保公平比较，以下参数在两个模型中保持完全一致：

- **数据集**: CIFAR-10 (50,000训练 + 10,000测试)
- **图像尺寸**: 32x32 pixels
- **批次大小**: 128 (可调整)
- **训练轮次**: 100 (可调整)
- **优化器**: AdamW (lr=3e-4, weight_decay=0.1)
- **学习率调度**: Cosine退火 + Warmup (10轮)
- **数据增强**: RandomHorizontalFlip, RandomRotation, ColorJitter
- **模型参数**:
  - 嵌入维度 (dim): 384
  - Transformer深度: 12层
  - 注意力头数: 6
  - MLP隐藏层维度: 1536
  - Dropout: 0.1

### 唯一差异 (实验变量)
- **Standard ViT**: 使用固定4x4 patch size的标准tokenizer
- **Fractal ViT**: 使用自适应分形Hilbert tokenizer
  - 最小patch尺寸: (4, 4)
  - 最大递归层级: 3
  - 可学习分割决策: 启用

## 🔧 环境要求

### 硬件要求
- **GPU**: 推荐NVIDIA GPU (8GB+ VRAM)
- **CPU**: 4核心以上
- **内存**: 16GB+
- **存储**: 5GB可用空间

### 软件依赖
```bash
# 核心依赖
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
tqdm>=4.60.0
einops>=0.6.0

# 可选依赖
wandb  # 实验追踪（可选）
```

## 📈 实验输出

每次实验会创建一个时间戳目录 `experiments/cifar10_experiment_YYYYMMDD_HHMMSS/`，包含：

### 结果文件
- `experiment_report.md` - 详细的实验报告
- `experiment_report.json` - 机器可读的结果数据
- `config.json` - 实验配置备份
- `training_curves_comparison.png/pdf` - 训练曲线对比图

### 模型检查点
- `best_standard_vit.pth` - 最佳性能的标准ViT模型
- `best_fractal_vit.pth` - 最佳性能的Fractal ViT模型
- `checkpoint_*_epoch_*.pth` - 定期保存的训练检查点

## 📋 实验配置选项

### 基础配置
```bash
--epochs N          # 训练轮次 (默认: 100)
--batch-size N       # 批次大小 (默认: 128)
--lr FLOAT          # 学习率 (默认: 3e-4)
--seed N            # 随机种子 (默认: 42)
```

### 预定义实验配置

1. **quick_test**: 5轮快速测试，用于验证代码
2. **standard_experiment**: 100轮标准实验
3. **large_batch**: 大批次实验 (batch_size=256)
4. **small_batch**: 小批次实验 (batch_size=64)
5. **different_seed**: 不同随机种子的重复实验

## 🎯 期望结果

### 性能指标
- **准确率**: 两模型在CIFAR-10上的分类准确率
- **收敛速度**: 达到最佳性能所需的轮次
- **训练稳定性**: 训练过程中的波动情况
- **参数效率**: 模型参数数量对比

### 预期假设
- Fractal ViT可能在以下方面表现更好：
  - 捕捉多尺度特征
  - 处理细节信息
  - 适应性tokenization
- Standard ViT可能在以下方面表现更好：
  - 训练效率
  - 参数利用率
  - 计算复杂度

## 🔍 结果分析

### 关键指标
1. **最佳测试准确率**: 模型达到的最高准确率
2. **最终测试准确率**: 训练结束时的准确率
3. **训练收敛性**: 训练曲线的平滑程度
4. **泛化能力**: 训练集与测试集性能差距

### 统计显著性
- 建议运行多次实验（不同随机种子）
- 计算均值和标准差
- 进行统计检验

## ⚠️ 注意事项

### 训练时间
- **CPU训练**: ~8-12小时 (100轮)
- **GPU训练**: ~2-4小时 (100轮)
- **快速测试**: ~10-15分钟 (5轮)

### 内存使用
- **GPU内存**: ~4-6GB (batch_size=128)
- **系统内存**: ~8-12GB
- **存储空间**: 每个实验~1-2GB

### 常见问题
1. **CUDA内存不足**: 减小batch_size
2. **训练太慢**: 使用GPU，减少epochs进行测试
3. **精度不收敛**: 检查学习率，增加训练轮次

## 📞 技术支持

如果遇到问题，请检查：
1. GPU驱动和CUDA版本兼容性
2. PyTorch和torchvision版本匹配
3. 数据集下载完整性
4. 足够的磁盘空间

## 📚 扩展实验

### 其他数据集
可以修改代码支持其他数据集：
- CIFAR-100
- ImageNet-1K (需要更多计算资源)
- 自定义数据集

### 模型变体
可以尝试的模型配置：
- 不同的patch尺寸
- 不同的模型深度
- 不同的注意力头数
- 混合精度训练优化

---
*这个实验框架为Fractal ViT的研究提供了严格的对比基准。*
