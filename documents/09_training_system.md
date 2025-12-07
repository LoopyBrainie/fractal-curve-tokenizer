# 第九章：训练系统 (train_fractal_vit.py)

本项目包含一个功能完备的训练脚本，支持多种数据集和训练策略。

## 9.1 数据集支持

脚本通过 `DatasetSpec` dataclass 统一管理不同数据集的配置。

*   **支持列表**：CIFAR10, CIFAR100, MNIST, ImageNet, COCO (分类模式), Caltech256, TinyImageNet。
*   **特殊处理**：
    *   `CocoClassificationWrapper`: 将 COCO 目标检测数据集转换为分类数据集（取最大物体类别）。
    *   `TinyImageNetVal`: 自定义加载器处理 TinyImageNet 的验证集标注文件。
*   **数据增强**：
    *   `AutoAugment`: 自动增强策略 (CIFAR10/ImageNet 策略)。
    *   `RandomErasing`: 随机擦除。
    *   `RandomCrop`, `RandomHorizontalFlip`: 基础增强。

## 9.2 模型构建与配置

*   `build_model()`: 根据参数实例化 `NextGenerationFractalViT` 或 `SimpleFractalViT`。
*   **自适应策略**：
    *   如果检测到 CPU 环境，自动切换到 `SimpleFractalViT` 并减小模型规模，以允许在无 GPU 环境下调试。
    *   `--disable-hilbert-bias`: 在 CPU 上自动禁用 Hilbert Bias 计算以提升速度。

## 9.3 训练循环

### train_one_epoch()
*   **混合精度 (AMP)**: 使用 `torch.cuda.amp` (或 `torch.amp`) 进行加速。
*   **REINFORCE 实现**:
    *   计算分类损失 `ce_loss`。
    *   计算奖励 `reward = -ce_loss`。
    *   维护基线 `baseline_ema` (指数移动平均) 以减少方差。
    *   调用 `model.get_tokenizer_loss()` 获取策略梯度损失 `aux_loss`。
    *   总损失 `loss = ce_loss + aux_loss`。
    *   调用 `model.clear_tokenizer_cache()`。
*   **梯度裁剪**: 防止梯度爆炸。

### evaluate()
*   标准评估循环，计算 Loss 和 Accuracy。

## 9.4 学习率调度
采用了组合调度策略：
1.  **Warmup**: `LinearLR`，前 5 个 Epoch 线性预热。
2.  **Cosine Annealing**: `CosineAnnealingLR`，之后进行余弦退火衰减。
3.  **SequentialLR**: 串联上述两个调度器。

## 9.5 实验管理

*   **ExperimentPaths**: 自动生成带时间戳的实验目录结构。
    *   `experiments/fractal_vit_simple_YYYYMMDD_HHMMSS/`
*   **Checkpoint**: 保存最佳模型 (`best.pth`)，包含模型权重、优化器状态、Scheduler 状态。
*   **可视化**:
    *   保存 `training_history.json`。
    *   使用 Matplotlib 绘制 Loss 和 Accuracy 曲线并保存为 PNG。
