# Fractal ViT 完整训练报告

## 实验信息
- **实验ID**: fractal_vit_complete_20250819_113315
- **时间戳**: 2025-08-19T11:41:35.684981
- **设备**: cpu
- **数据集**: CIFAR10

## 模型配置
- **图像尺寸**: 32
- **嵌入维度**: 256
- **Transformer深度**: 6
- **注意力头数**: 8
- **最小patch尺寸**: [4, 4]
- **最大层级**: 3
- **可学习分割**: True

## 训练配置
- **训练轮次**: 5
- **批次大小**: 64
- **学习率**: 0.0003
- **优化器**: adamw
- **调度器**: cosine

## 训练结果
- **最佳验证准确率**: 23.00%
- **最佳epoch**: 1
- **最终训练准确率**: 20.80%
- **最终验证准确率**: 16.00%
- **总epoch数**: 5

## 模型信息
- **总参数数量**: 10,099,417
- **可训练参数数量**: 10,099,417

## 文件位置
- **实验目录**: `D:\myProject\fractal-curve-tokenizer\experiments\fractal_vit_complete_20250819_113315`
- **最佳模型**: `D:\myProject\fractal-curve-tokenizer\workspace\models\fractal_vit\fractal_vit_best_20250819_113315.pth`
- **训练曲线**: `D:\myProject\fractal-curve-tokenizer\workspace\visualizations\fractal_vit_training_curves_20250819_113315.png`

## 文件管理规范
本实验遵循项目文件管理规范：
- 📁 **实验数据**: 保存在 `experiments/fractal_vit_complete_20250819_113315/`
- 📁 **模型文件**: 保存在 `workspace/models/fractal_vit/`
- 📁 **结果数据**: 保存在 `workspace/results/`
- 📁 **可视化文件**: 保存在 `workspace/visualizations/`
- 📁 **报告文档**: 保存在 `workspace/reports/`

## 生成文件
### 实验目录文件
- `checkpoints/best.pth`: 最佳模型检查点
- `checkpoints/latest.pth`: 最新检查点
- `logs/training.log`: 详细训练日志
- `results/training_curves.png/pdf`: 训练曲线图
- `results/training_report.json`: JSON格式报告

### Workspace文件
- `models/fractal_vit/fractal_vit_best_20250819_113315.pth`: 最佳模型
- `results/fractal_vit_report_20250819_113315.json`: 实验报告
- `visualizations/fractal_vit_training_curves_20250819_113315.png`: 训练曲线
- `reports/fractal_vit_report_20250819_113315.md`: 本报告
