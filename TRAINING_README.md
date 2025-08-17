# Fractal ViT vs Standard ViT - CIFAR-10 训练对比

完整的控制变量实验，比较Fractal ViT和标准ViT在CIFAR-10数据集上的性能。

## 快速开始

### 1. 环境检查
```bash
python check_environment.py
```

### 2. 运行实验
```bash
# 方式1: 交互式启动（推荐新手）
python run_experiments.py

# 方式2: 直接运行
python train_fractal_vs_standard_cifar10.py

# 方式3: 快速测试（5轮）
python train_fractal_vs_standard_cifar10.py --epochs 5
```

## 实验设计

**控制变量**: 两个模型使用完全相同的训练配置（数据、优化器、学习率等）

**唯一差异**: 
- Standard ViT: 固定4x4 patch tokenizer
- Fractal ViT: 自适应分形Hilbert tokenizer

## 预期输出

每次实验会生成：
- 📊 训练曲线对比图
- 📄 详细实验报告  
- 💾 最佳模型检查点
- 📈 性能对比数据

## 硬件要求

- **推荐**: NVIDIA GPU (8GB+ VRAM)
- **最低**: CPU + 16GB RAM（训练会较慢）

## 训练时间

- **GPU**: ~2-4小时 (100轮)
- **CPU**: ~8-12小时 (100轮)  
- **测试**: ~10-15分钟 (5轮)

---

更多详细信息请查看 `EXPERIMENT_GUIDE.md`
