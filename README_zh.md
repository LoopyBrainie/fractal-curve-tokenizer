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

项目包含功能完善的训练脚本 `examples/training/train_fractal_vit.py`，支持多种数据集和高级训练特性。

#### 快速开始

```bash
# CIFAR-10 默认设置
uv run python examples/training/train_fractal_vit.py --dataset cifar10 --epochs 50

# Tiny ImageNet 使用 SwiGLU FFN（推荐）
uv run python examples/training/train_fractal_vit.py \
    --dataset tiny-imagenet \
    --ffn-type swiglu_level \
    --epochs 100 \
    --warmup-epochs 10

# 快速测试（5轮，512样本）
uv run python examples/training/train_fractal_vit.py --quick-test --use-amp
```

#### FFN 架构选择

训练脚本支持三种前馈网络变体：

| FFN 类型 | 参数量 | 速度 | 推荐场景 |
|----------|--------|------|----------|
| `gelu` | 379K | 基准 | 仅用于基准对比 |
| `swiglu` | 334K (-11.9%) | 快 | 轻量级任务 |
| `swiglu_level` | 357K (-5.8%) | 中等 | **默认，最佳平衡** |

**示例：**
```bash
# 使用 SwiGLU + Level Adaptation（默认）
uv run python examples/training/train_fractal_vit.py --ffn-type swiglu_level

# 使用轻量级 SwiGLU
uv run python examples/training/train_fractal_vit.py --ffn-type swiglu
```

#### 支持的数据集

| 数据集 | 自动下载 | 类别数 | 图像尺寸 | 备注 |
|--------|---------|--------|---------|------|
| `cifar10` | ✅ | 10 | 32×32 | 默认数据集 |
| `cifar100` | ✅ | 100 | 32×32 | 更具挑战性 |
| `mnist` | ✅ | 10 | 28×28 | 灰度数字 |
| `tiny-imagenet` | ❌ 手动 | 200 | 64×64 | 需要下载 |

**Tiny ImageNet 设置：**
```bash
# 手动下载和解压
cd data
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
# 预期目录结构：
# data/tiny-imagenet-200/train/n01443537/images/*.JPEG
# data/tiny-imagenet-200/val/images/*.JPEG
```

#### 关键训练参数

| 参数 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `--dataset` | str | `cifar10` | 数据集: `cifar10`, `cifar100`, `mnist`, `tiny-imagenet` |
| `--epochs` | int | `50` | 训练轮数 |
| `--batch-size` | int | `64` | 训练批次大小 |
| `--lr` | float | `5e-4` | 初始学习率 |
| `--warmup-epochs` | int | `10` | **[新]** Warmup 轮数 |
| `--weight-decay` | float | `0.01` | 权重衰减（L2正则化） |
| `--val-split` | float | `0.1` | 验证集划分比例 |
| `--dim` | int | `192` | 模型嵌入维度 |
| `--depth` | int | `8` | Transformer 深度（层数） |
| `--heads` | int | `8` | 注意力头数量 |
| `--dim-head` | int | `32` | 每个注意力头的维度 |
| `--max-level` | int | `4` | 最大分形递归层级 |
| `--ffn-type` | str | `swiglu_level` | **[新]** FFN类型: `gelu`, `swiglu`, `swiglu_level` |
| `--pool` | str | `cls` | 池化方法: `cls` 或 `mean` |
| `--dropout` | float | `0.1` | Dropout 比率 |
| `--gradient-clip` | float | `1.0` | 梯度裁剪值 |
| `--use-amp` | flag | `False` | 启用混合精度训练 |
| `--accum-steps` | int | `1` | 梯度累积步数 |
| `--num-workers` | int | `4` | 数据加载工作进程数 |
| `--no-learnable-split` | flag | `False` | 禁用可学习分词 |
| `--quick-test` | flag | `False` | 快速5轮测试，512样本 |

#### 性能优化

训练脚本包含自动性能增强：

| 优化项 | 自动启用 | 预期加速 | 要求 |
|--------|---------|----------|------|
| TF32 加速 | ✅ CUDA | ~8× 矩阵运算 | Ampere+ GPU (RTX 30/40) |
| cuDNN Benchmark | ✅ CUDA | 5-15% | 固定输入尺寸 |
| Spawn 多进程 | ✅ 始终 | 稳定性 | CUDA 兼容性 |
| 持久化 Workers | ✅ 始终 | 更快的 epoch | num_workers > 0 |
| 混合精度 (AMP) | ⚙️ `--use-amp` | 20-40% | 现代 GPU |
| 梯度累积 | ⚙️ `--accum-steps` | 更大批次 | 有限显存 |

**示例：最大性能**
```bash
# RTX 3090/4090 最优设置
uv run python examples/training/train_fractal_vit.py \
    --dataset cifar100 \
    --ffn-type swiglu_level \
    --batch-size 128 \
    --use-amp \
    --num-workers 4 \
    --warmup-epochs 10
```

**示例：有限显存**
```bash
# 用 8GB 显存模拟 batch size 256
uv run python examples/training/train_fractal_vit.py \
    --batch-size 64 \
    --accum-steps 4 \
    --use-amp
```

#### 训练输出

训练产物自动保存在 `experiments/` 目录下，按时间戳组织：

```
experiments/
└── fractal_vit_20251211_142110/
    ├── checkpoints/
    │   └── best.pth              # 最佳模型权重
    ├── logs/
    │   ├── config.json           # 训练配置
    │   ├── metrics.json          # 每轮指标
    │   └── final.json            # 最终结果 + tokenization 分析
    └── visualizations/           # (可选) 可视化图表
```

**详细日志包含：**
* 每轮的训练/验证损失和准确率
* 学习率变化
* Token 数量统计和自适应性分析
* Variance-token 相关性评估
* GPU 信息和训练时间
* `visualizations/`: 损失和准确率曲线。
* `training_history.json`: 每个 epoch 的详细指标。

## 基准测试 (Benchmarking)

本项目包含全面的基准测试工具，用于评估模型性能、分词效率、收敛行为以及与标准 ViT 的对比。

### 1. 模型性能基准测试

测试前向/反向传播时间和分词分析：

```bash
uv run python -m tests.benchmarks.benchmark_fractal_vit
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
uv run python -m tests.benchmarks.check_convergence \
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
uv run python -m tests.benchmarks.evaluate_pretrained \
  -c experiments/fractal_vit_simple_20251208_222817/checkpoints/best.pth \
  --visualize

# 在子集上快速评估
uv run python -m tests.benchmarks.evaluate_pretrained \
  -c workspace/models/fractal_vit/fractal_vit_simple_best.pth \
  --num-samples 100 \
  -o quick_eval
```

### 5. CNN 消融实验 (CNN Ablation Study)

评估使用 CNN 特征进行分割决策与仅使用手工特征相比的影响。

```bash
uv run python tests/benchmarks/benchmark_cnn_ablation.py --dataset cifar10 --epochs 20
```

**参数:**
- `--dataset`: 使用的数据集 (`cifar10`, `tiny-imagenet`)。
- `--epochs`: 训练轮数。
- `--batch-size`: 批次大小 (默认: 64)。
- `--use-cnn`: 启用 CNN 特征 (基准测试中默认为 False 以测试基线)。

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
