# Fractal Curve Tokenizer (分形曲线分词器)

[English](README.md) | [中文](README_zh.md)

一种下一代 Vision Transformer (ViT) 架构，利用分形几何和希尔伯特曲线实现自适应、多尺度的图像分词。

## 概览 (Overview)

Fractal Curve Tokenizer 为 Vision Transformers 引入了一种新颖的图像分词方法。它不再将图像划分为固定的网格块，而是使用基于图像复杂性（方差、边缘、纹理）的递归自适应分割策略。这使得模型能够为复杂区域分配更多 token，为均匀区域分配更少 token，从而优化计算效率并捕获多尺度特征。

关键技术包括：

* **可学习的分形分词 (Learnable Fractal Tokenization)**：利用 **MiniCNN** 和 **REINFORCE** (结合 Gumbel-Softmax) 在训练过程中动态学习最优的图像分割策略。
* **希尔伯特曲线遍历 (Hilbert Curve Traversal)**：使用真正的递归希尔伯特曲线将 token 展平为 1D 序列时，保留 2D 空间局部性。
* **高级位置嵌入 (Advanced Positional Embedding)**：编码层级深度和路径历史以保持结构上下文。
* **带填充的批处理 (Batch Processing with Padding)**：高效处理批次内的可变长度 token 序列。

## 特性 (Features)

* **自适应分辨率 (Adaptive Resolution)**：根据图像内容复杂性自动调整 token 密度。
* **可微分分词器 (Differentiable Tokenizer)**：分割决策是完全可微分的，并进行端到端优化。
* **空间局部性保持 (Spatial Locality Preservation)**：使用希尔伯特曲线比光栅扫描顺序更好地保持空间关系。
* **强健的正则化 (Robust Regularization)**：集成 **DropPath** (随机深度) 和熵正则化以防止过拟合。
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
# CIFAR-10（自动下载）
python examples/training/train_fractal_vit.py --dataset cifar10 --epochs 50 --batch-size 64

# Tiny ImageNet（自动下载 ~237 MB）
python examples/training/train_fractal_vit.py --dataset tiny-imagenet --quick-test
```

**注意：** CIFAR-10、CIFAR-100、MNIST 和 Tiny ImageNet 等数据集在首次使用时会自动下载到 `workspace/data/`。大型数据集（ImageNet、COCO）需要手动下载。

#### 完整参数列表 (Full Argument List)

| 参数 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `--dataset` | str | `cifar10` | 使用的数据集: `cifar10`, `cifar100`, `mnist`, `imagenet`, `coco`, `caltech256`, `tiny-imagenet`。 |
| `--data-root` | str | `workspace/data` | 数据集根目录路径。支持的数据集会自动创建此目录。 |
| `--epochs` | int | `50` | 训练轮数。 |
| `--batch-size` | int | `64` | 训练批次大小。 |
| `--lr` | float | `5e-4` | 初始学习率。 |
| `--weight-decay` | float | `0.01` | 优化器权重衰减。 |
| `--val-split` | float | `0.1` | 用于验证的训练数据比例。 |
| `--subset-size` | int | `None` | 限制训练样本数量（用于调试）。 |
| `--dim` | int | `192` | 模型嵌入维度。 |
| `--depth` | int | `8` | Transformer 深度。 |
| `--heads` | int | `8` | 注意力头数。 |
| `--dim-head` | int | `32` | 每个注意力头的维度。 |
| `--dropout` | float | `0.1` | Dropout 比率。 |
| `--emb-dropout` | float | `0.1` | 嵌入层 Dropout 比率。 |
| `--max-level` | int | `4` | 分形分词的最大递归层级。 |
| `--pool` | str | `cls` | 池化方法: `cls` 或 `mean`。 |
| `--use-simple` | flag | `False` | 使用 `SimpleFractalViT` 而不是 `NextGenerationFractalViT`。 |
| `--no-learnable-split` | flag | `False` | 禁用可学习的分割决策网络（使用启发式规则）。 |
| `--quick-test` | flag | `False` | 在小数据集上运行快速的 5 轮测试。 |
| `--use-amp` | flag | `False` | 启用自动混合精度 (AMP) 训练。 |
| `--gradient-clip` | float | `1.0` | 梯度裁剪阈值。 |
| `--num-workers` | int | `2` | 数据加载工作线程数。 |
| `--seed` | int | `42` | 随机种子。 |
| `--disable-hilbert-bias` | flag | `False` | 禁用希尔伯特路径注意力偏置（CPU 上更快）。 |
| `--force-next-gen` | flag | `False` | 即使在 CPU 上也强制使用 `NextGenerationFractalViT`。 |
| `--device` | str | `auto` | 使用的设备: `auto`, `cpu`, `cuda`。 |

#### 输出 (Output)

训练产物保存在 `experiments/` 目录下，按时间戳组织：

* `checkpoints/`: 保存的模型权重 (`best.pth`)。
* `logs/`: 训练日志。
* `visualizations/`: 损失和准确率曲线。
* `training_history.json`: 每个 epoch 的详细指标。

## 基准测试 (Benchmarking)

本项目包含全面的基准测试工具，用于评估模型性能、分词效率、收敛行为以及与标准 ViT 的对比。

### 1. 模型性能基准测试

测试前向/反向传播时间和分词分析：

```bash
python -m tests.benchmarks.benchmark_fractal_vit
```

**关键指标：**
- 前向/反向传播时间（毫秒）
- 分词时间和 token 数量分布
- 模型参数量和内存使用
- 吞吐量（图像/秒）

**结果：** 保存到 `benchmark_results/`，包含 JSON 指标和可选可视化。

### 2. 收敛性分析

在合成分类任务上分析训练收敛性：

```bash
python -m tests.benchmarks.check_convergence \
  --output-dir benchmark_results \
  --image-size 32 \
  --num-epochs 20 \
  --num-runs 3
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--output-dir` | str | `benchmark_results` | 结果输出目录 |
| `--image-size` | int | `32` | 合成数据的图像尺寸 |
| `--num-classes` | int | `4` | 分类类别数 |
| `--num-epochs` | int | `30` | 每次运行的训练轮数 |
| `--num-runs` | int | `3` | 重复运行次数 |

**测试场景：**
- 颜色分类（简单任务）
- 图案分类（中等难度）
- 复杂度分类（困难任务）

**输出指标：**
- 收敛速度（达到阈值所需轮数）
- 最终训练/验证准确率和损失
- 过拟合检测
- 损失方差和梯度统计

### 3. 标准 ViT 对比

将 FractalViT 各版本与标准基于 patch 的 ViT 进行对比：

```bash
python -m tests.benchmarks.compare_fractal_vs_standard \
  --output-dir benchmark_results \
  --image-size 64 \
  --num-epochs 20 \
  --batch-size 8
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--output-dir` | str | `benchmark_results` | 输出目录 |
| `--image-size` | int | `64` | 输入图像尺寸 |
| `--num-classes` | int | `10` | 分类数量 |
| `--batch-size` | int | `8` | 批次大小 |
| `--num-epochs` | int | `20` | 训练轮数 |
| `--plot` | flag | `False` | 生成对比图表 |

**对比指标：**
- 分词效率（自适应 vs 固定 patch）
- 计算性能（前向/反向速度）
- 内存使用
- 训练收敛和最终准确率
- 参数数量

### 4. 预训练模型评估

评估训练好的 `.pth` 检查点，生成全面的指标和可视化：

```bash
python -m tests.benchmarks.evaluate_pretrained \
  --checkpoint experiments/fractal_vit_simple_20251208/checkpoints/best.pth \
  --visualize \
  --num-samples 500 \
  --output-dir benchmark_results/pretrained_eval
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--checkpoint` | str | **必需** | `.pth` 检查点文件路径 |
| `--output-dir` | str | `{checkpoint_dir}/evaluation` | 结果输出目录 |
| `--visualize` | flag | `False` | 生成可视化图表 |
| `--num-samples` | int | `None` | 测试样本数量（None = 全部）|
| `--batch-size` | int | `32` | 评估批次大小 |

**评估指标：**
- 测试准确率（top-1 和 top-5）
- 吞吐量（图像/秒）
- 分词分析（token 数量分布）
- 图像复杂度与 token 数量的相关性

**生成的可视化（如果使用 `--visualize`）：**
- `tokenization_analysis.png`：示例图像及其 token 分布
- `complexity_vs_tokens.png`：复杂度 vs token 数量散点图
- `performance_summary.png`：准确率和吞吐量图表
- `confusion_matrix.png`：混淆矩阵（小样本量时）
- `evaluation_results.json`：JSON 格式的完整指标

**多检查点评估示例：**

```bash
# 评估最佳模型
python -m tests.benchmarks.evaluate_pretrained \
  -c experiments/fractal_vit_simple_20251208_222817/checkpoints/best.pth \
  --visualize

# 在子集上快速评估
python -m tests.benchmarks.evaluate_pretrained \
  -c workspace/models/fractal_vit/fractal_vit_simple_best.pth \
  --num-samples 100 \
  -o quick_eval
```

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
│   ├── benchmarks/                 # 全面的基准测试套件
│   │   ├── benchmark_fractal_vit.py    # 性能基准测试
│   │   ├── check_convergence.py        # 收敛性分析
│   │   ├── compare_fractal_vs_standard.py # ViT 对比
│   │   ├── evaluate_pretrained.py      # 预训练模型评估
│   │   └── benchmark_metrics.py        # 核心指标定义
│   ├── unit/                       # 组件单元测试
│   └── integration/                # 工作流集成测试
├── experiments/                    # 训练输出 (git 忽略)
├── benchmark_results/              # 基准测试输出和可视化
└── workspace/                      # 本地数据和模型 (git 忽略)
```

## 许可证 (License)

本项目基于 MIT 许可证开源 - 详见 [LICENSE](LICENSE) 文件。
