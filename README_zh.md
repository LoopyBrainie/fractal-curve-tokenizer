# Fractal Curve Tokenizer (分形曲线分词器)

[English](README.md) | [中文](README_zh.md)

一种下一代 Vision Transformer (ViT) 架构，利用分形几何和 Hilbert 曲线实现自适应、多尺度的图像分词。

## 概览

Fractal Curve Tokenizer 为 Vision Transformers 引入了一种新颖的图像分词方法。它不再将图像划分为固定的网格块，而是使用 **Hilbert 曲线遍历** 在将 token 展平为 1D 序列时保留 2D 空间局部性。

**核心创新：**

* **流式分形 Tokenizer**: 使用多尺度卷积金字塔和 Gumbel-Softmax 尺度选择实现端到端可微分词（V2）。
* **Hilbert 曲线遍历**: 使用具有良好局部性的空间填充曲线保持 2D 空间邻近关系。
* **低秩 Hilbert 注意力偏置**: 高效的 $O(N \cdot r)$ 注意力偏置计算，替代 $O(N^2)$。
* **SwiGLU 前馈网络**: 现代化的门控线性单元架构（LLaMA/PaLM 风格）。
* **高级位置编码**: 编码层级深度和象限路径历史。

## 架构

```
输入图像 (B, C, H, W)
        │
        ▼
┌─────────────────────────────┐
│ StreamingFractalTokenizerV2 │  ← 多尺度卷积金字塔 + Gumbel-Softmax
│ - MultiScalePatchEncoder    │
│ - Hilbert 重排序            │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│ AdvancedFractalPosition     │  ← 深度 + 路径编码 + 融合
│ Embedding                   │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│ EnhancedFractalTransformer  │  ← 层级感知 LayerNorm
│ - HilbertAwareAttention     │  ← 低秩 Hilbert 偏置
│ - SwiGLU FFN                │  ← Gate * Value 投影
│ - DropPath                  │
└─────────────────────────────┘
        │
        ▼
    MLP Head → Logits
```

## 可视化

以下可视化展示了 Hilbert 曲线分词的核心概念。

### Hilbert 曲线阶数对比

不同阶数 Hilbert 曲线的对比，展示空间填充模式如何扩展：

![Hilbert 阶数对比](workspace/visualizations/fractal_curves/hilbert_order_comparison.png)

### 局部性保持

展示 Hilbert 曲线如何在 1D 序列中保持 2D 空间局部性：

![局部性保持](workspace/visualizations/fractal_curves/hilbert_locality.png)

### 四叉树结构

递归四叉树分解与 Hilbert 遍历顺序的可视化：

![四叉树结构](workspace/visualizations/fractal_curves/quadtree_structure.png)

### 2D 到 1D 映射

2D 网格位置如何通过 Hilbert 曲线映射到 1D token 序列：

![2D 到 1D 映射](workspace/visualizations/fractal_curves/2d_to_1d_mapping.png)

### 多尺度层级

不同分辨率下的多尺度 patch 提取可视化：

![多尺度层级](workspace/visualizations/fractal_curves/multiscale_hierarchy.png)

### 混合尺度自适应分词

展示模型如何根据区域复杂度自适应选择不同的 patch 大小：

![混合尺度分割](workspace/visualizations/fractal_curves/mixed_level_segmentation.png)

### Hilbert 曲线生长动画

![Hilbert 生长](workspace/visualizations/fractal_curves/hilbert_growth.gif)

## 特性

* **端到端可微分**: 无需 REINFORCE - 使用 Gumbel-Softmax 完全可微。
* **空间局部性保持**: Hilbert 曲线在 1D 序列中保持 2D 邻域关系。
* **高效注意力**: 低秩 Hilbert 偏置将显存从 $O(N^2)$ 降低到 $O(N \cdot r)$。
* **现代 FFN**: SwiGLU 带可选的层级自适应处理。
* **强健正则化**: 集成 DropPath（随机深度）和熵正则化。
* **灵活架构**: 支持 `NextGenerationFractalViT`（全功能）和 `SimpleFractalViT`（轻量级）。

## 安装

确保已安装 uv 和 PyTorch。

```bash
# 克隆仓库
git clone https://github.com/LoopyBrainie/fractal-curve-tokenizer.git
cd fractal-curve-tokenizer

# 安装依赖（推荐：uv）
uv sync

# 或使用 pip
pip install -e .
```

## 使用方法

### 基础推理

```python
import torch
from vit_pytorch import NextGenerationFractalViT

# 使用推荐设置初始化模型
model = NextGenerationFractalViT(
    image_size=224,
    num_classes=1000,
    dim=384,
    depth=6,
    heads=6,
    mlp_dim=768,
    tokenizer_type='streaming_v2',  # 推荐: Gumbel-Softmax
    bias_mode='low_rank',           # 推荐: 内存高效
    ffn_type='swiglu_level',        # 推荐: SwiGLU + 层级自适应
)

# 前向传播
img = torch.randn(1, 3, 224, 224)
logits = model(img)  # (1, 1000)
```

### Tokenizer 类型

| 类型 | 类 | 描述 | 状态 |
|------|-----|------|------|
| `streaming_v2` | `StreamingFractalTokenizerV2` | Gumbel-Softmax 自适应尺度 | ✅ **推荐** |
| `streaming` | `StreamingFractalTokenizer` | 固定多尺度卷积 | ✅ 稳定 |
| `legacy` | `FractalHilbertTokenizer` | BFS + REINFORCE | ⚠️ 废弃 |

### 独立使用 Tokenizer

```python
from vit_pytorch import StreamingFractalTokenizerV2

tokenizer = StreamingFractalTokenizerV2(
    image_size=224,
    dim=384,
    scales=[4, 8, 16],
    temperature=1.0,
)

images = torch.randn(2, 3, 224, 224)
output = tokenizer.tokenize(images)

# 访问 tokens 和 levels
tokens = output.sequences[0].tokens       # (N, 384)
levels = output.sequences[0].get_levels() # (N, info_len)
```

### 训练

项目包含完整的训练脚本 `examples/training/train_fractal_vit.py`。

#### 快速开始

```bash
# CIFAR-10 使用推荐设置
uv run python examples/training/train_fractal_vit.py \
    --dataset cifar10 \
    --tokenizer-type streaming_v2 \
    --bias-mode low_rank \
    --ffn-type swiglu_level \
    --epochs 50

# 快速测试（5轮，512样本）
uv run python examples/training/train_fractal_vit.py --quick-test --use-amp
```

#### 关键训练参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `--tokenizer-type` | `streaming_v2` | Tokenizer: `streaming_v2`, `streaming`, `legacy` |
| `--bias-mode` | `low_rank` | Hilbert 偏置: `low_rank`, `hierarchical`, `original` |
| `--ffn-type` | `swiglu_level` | FFN 类型: `swiglu_level`, `swiglu`, `gelu` |
| `--dataset` | `cifar10` | 数据集: `cifar10`, `cifar100`, `mnist`, `tiny-imagenet` |
| `--epochs` | `50` | 训练轮数 |
| `--batch-size` | `64` | 训练批次大小 |
| `--lr` | `5e-4` | 初始学习率 |
| `--dim` | `192` | 模型嵌入维度 |
| `--depth` | `8` | Transformer 深度（层数） |
| `--heads` | `8` | 注意力头数量 |
| `--use-amp` | `False` | 启用混合精度训练 |

#### FFN 架构选择

| FFN 类型 | 参数量 | 速度 | 使用场景 |
|----------|--------|------|----------|
| `gelu` | 基准 | 1.0x | 基准对比 |
| `swiglu` | -12% | 1.15x | 轻量级推理 |
| `swiglu_level` | -6% | 1.10x | **默认，最佳准确率** |

#### 支持的数据集

| 数据集 | 自动下载 | 类别数 | 图像尺寸 |
|--------|---------|--------|---------|
| `cifar10` | ✅ | 10 | 32×32 |
| `cifar100` | ✅ | 100 | 32×32 |
| `mnist` | ✅ | 10 | 28×28 |
| `tiny-imagenet` | ❌ 手动 | 200 | 64×64 |

#### 训练输出

```
experiments/
└── fractal_vit_20251214_142110/
    ├── checkpoints/
    │   └── best.pth              # 最佳模型权重
    ├── logs/
    │   ├── config.json           # 训练配置
    │   └── metrics.json          # 每轮指标
    └── visualizations/           # 训练曲线
```

## 基准测试

### 模型性能

```bash
uv run python -m tests.benchmarks.benchmark_fractal_vit
```

### 与标准 ViT 对比

```bash
uv run python -m tests.benchmarks.compare_fractal_vs_standard \
    --output-dir benchmark_results \
    --image-size 64 \
    --num-epochs 20
```

### 评估预训练模型

```bash
uv run python -m tests.benchmarks.evaluate_pretrained \
    --checkpoint experiments/.../checkpoints/best.pth \
    --visualize
```

## 测试

```bash
# 运行所有测试
uv run pytest

# 仅单元测试
uv run pytest tests/unit

# 仅集成测试
uv run pytest tests/integration
```

**测试状态**: 120 通过, 1 跳过

## 项目结构

```text
fractal-curve-tokenizer/
├── src/
│   └── vit_pytorch/
│       ├── __init__.py             # 包入口，模块导出
│       ├── fractal_vit.py          # 主模型定义
│       ├── streaming_tokenizer.py  # StreamingFractalTokenizer V1/V2
│       ├── transformer.py          # 带层级感知归一化的 Transformer
│       ├── attention.py            # Hilbert 感知注意力 + 低秩偏置
│       ├── feedforward.py          # SwiGLU FFN + 层级自适应
│       ├── positional.py           # 深度 + 路径位置编码
│       ├── hilbert.py              # Hilbert 曲线 d ↔ (x,y) 映射
│       ├── tokenization.py         # 基类和数据结构
│       ├── constants.py            # 超参数默认值
│       ├── features.py             # Token 特征计算
│       ├── utils.py                # 工具函数
│       └── _deprecated/            # 废弃模块 (v1.0 移除)
│           ├── fractal_curve_tokenizer.py  # 旧版 BFS + REINFORCE
│           └── token_processor.py          # 旧版 token 处理器
├── examples/
│   └── training/
│       └── train_fractal_vit.py    # 训练脚本
├── tests/
│   ├── unit/                       # 单元测试
│   ├── integration/                # 集成测试
│   └── benchmarks/                 # 性能基准测试
├── documents/                      # 项目文档
├── experiments/                    # 训练输出 (gitignored)
└── workspace/                      # 本地数据/模型 (gitignored)
```

## 模块层级

| 层级 | 模块 | 描述 |
|------|------|------|
| **L4** 应用层 | `fractal_vit.py` | `NextGenerationFractalViT`, `SimpleFractalViT` |
| **L3** 管道层 | `streaming_tokenizer.py` | 多尺度分词 + Hilbert 重排序 |
| | `transformer.py` | 层级感知 Transformer 块 |
| **L2** 组件层 | `attention.py` | Hilbert 偏置 + 低秩分解 |
| | `feedforward.py` | SwiGLU + 层级自适应 |
| | `positional.py` | 深度 + 路径编码 |
| **L1** 基础层 | `hilbert.py` | 空间填充曲线算法 |
| | `tokenization.py` | 基类, `TokenizerOutput` |
| | `constants.py` | 默认超参数 |

## 数学形式化

**核心流程:**
$$I \xrightarrow{T} (T, L) \xrightarrow{E_{pos}} T' \xrightarrow{\text{Transformer}} X' \xrightarrow{\text{Pool}} z \xrightarrow{\text{MLP}} \hat{y}$$

**Hilbert 曲线映射:**
$$H: [0, n^2) \leftrightarrow [0, n)^2$$

**低秩 Hilbert 偏置:**
$$B_{hilbert}[i,j] = \phi(p_i)^T \cdot \psi(p_j), \quad \phi, \psi: \mathbb{R}^d \to \mathbb{R}^r$$

**SwiGLU FFN:**
$$\text{SwiGLU}(x) = W_{out} \cdot (\text{Swish}(W_{gate} \cdot x) \odot W_{value} \cdot x)$$

## 许可证

本项目基于 MIT 许可证开源 - 详见 [LICENSE](LICENSE) 文件。
