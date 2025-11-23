# Fractal Curve Tokenizer (分形曲线分词器)

[English](README.md) | [中文](README_zh.md)

一种下一代 Vision Transformer (ViT) 架构，利用分形几何和希尔伯特曲线实现自适应、多尺度的图像分词。

## 概览 (Overview)

Fractal Curve Tokenizer 为 Vision Transformers 引入了一种新颖的图像分词方法。它不再将图像划分为固定的网格块，而是使用基于图像复杂性（方差、边缘、纹理）的递归自适应分割策略。这使得模型能够为复杂区域分配更多 token，为均匀区域分配更少 token，从而优化计算效率并捕获多尺度特征。

关键技术包括：

* **自适应分形分词 (Adaptive Fractal Tokenization)**：根据内容复杂性递归分割图像块。
* **希尔伯特曲线遍历 (Hilbert Curve Traversal)**：使用真正的递归希尔伯特曲线将 token 展平为 1D 序列时，保留 2D 空间局部性。
* **高级位置嵌入 (Advanced Positional Embedding)**：编码层级深度和路径历史以保持结构上下文。
* **带填充的批处理 (Batch Processing with Padding)**：高效处理批次内的可变长度 token 序列。

## 特性 (Features)

* **自适应分辨率 (Adaptive Resolution)**：根据图像内容自动调整 token 密度。
* **空间局部性保持 (Spatial Locality Preservation)**：使用希尔伯特曲线比光栅扫描顺序更好地保持空间关系。
* **多尺度特征提取 (Multi-Scale Feature Extraction)**：同时捕获不同尺度的特征。
* **高效批次训练 (Efficient Batch Training)**：优化的 `pad_sequence` 和掩码实现，用于高速训练。
* **灵活架构 (Flexible Architecture)**：支持 `NextGenerationFractalViT`（全功能集）和 `SimpleFractalViT`（轻量级，向后兼容）。

## 安装 (Installation)

确保已安装 uv 和 PyTorch。

```bash
# 克隆仓库
git clone https://github.com/LoopyBrainie/fractal-curve-tokenizer.git
cd fractal-curve-tokenizer

# 安装依赖
pip install -r requirements.txt
# 或者如果使用 uv/poetry（推荐）
uv sync
```

## 使用方法 (Usage)

### 基础推理 (Basic Inference)

```python
import torch
from vit_pytorch.fractal_vit import NextGenerationFractalViT

# 初始化模型
model = NextGenerationFractalViT(
    image_size=256,
    num_classes=1000,
    dim=512,
    depth=6,
    heads=8,
    mlp_dim=1024,
    min_patch_size=(4, 4),
    max_level=5
)

# 前向传播
img = torch.randn(1, 3, 256, 256)
logits = model(img) # (1, 1000)
```

### 训练 (Training)

本项目包含一个位于 `examples/training/train_fractal_vit.py` 的健壮训练脚本。该脚本支持在 CIFAR-10, CIFAR-100 和 MNIST 上进行训练。

#### 基础训练命令

```bash
python examples/training/train_fractal_vit.py --dataset cifar10 --epochs 50 --batch-size 64
```

#### 关键参数

* `--dataset`: 选择数据集 (`cifar10`, `cifar100`, `mnist`)。默认值: `cifar10`。
* `--epochs`: 训练轮数。默认值: `50`。
* `--batch-size`: 批次大小。默认值: `64`。
* `--lr`: 学习率。默认值: `5e-4`。
* `--quick-test`: 在一小部分数据上运行简短的 5 轮训练以验证流程。

    ```bash
    python examples/training/train_fractal_vit.py --quick-test
    ```

* `--use-simple`: 强制使用 `SimpleFractalViT`（更轻量的模型）而不是完整的 `NextGenerationFractalViT`。推荐用于 CPU 训练或基准比较。
* `--device`: 手动指定设备 (`cpu`, `cuda`, `auto`)。默认值: `auto`。
* `--num-workers`: 数据加载工作线程数。默认值: `2`。

#### 输出 (Output)

训练产物保存在 `experiments/` 目录下，按时间戳组织：

* `checkpoints/`: 保存的模型权重 (`best.pth`)。
* `logs/`: 训练日志。
* `visualizations/`: 损失和准确率曲线。
* `training_history.json`: 每个 epoch 的详细指标。

## 测试 (Testing)

本项目使用 `pytest` 进行测试。测试套件已重组为单元测试和集成测试。

```bash
# 运行所有测试
uv run pytest

# 仅运行单元测试
uv run pytest tests/unit

# 仅运行集成测试
uv run pytest tests/integration
```

## 项目结构 (Project Structure)

```text
fractal-curve-tokenizer/
├── src/
│   └── vit_pytorch/
│       ├── fractal_vit.py          # 主模型定义
│       ├── fractal_curve_tokenizer.py # 自适应分词器逻辑
│       ├── positional.py           # 高级位置嵌入
│       ├── transformer.py          # 支持掩码的 Transformer 块
│       └── ...
├── examples/
│   └── training/
│       └── train_fractal_vit.py    # 主训练脚本
├── tests/
│   ├── unit/                       # 组件单元测试
│   └── integration/                # 工作流集成测试
├── experiments/                    # 训练输出 (git 忽略)
└── workspace/                      # 本地数据和模型 (git 忽略)
```

## 许可证 (License)

本项目基于 MIT 许可证开源 - 详见 [LICENSE](LICENSE) 文件。
