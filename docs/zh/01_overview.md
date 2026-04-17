# 第一章：系统架构

## 1.1 概述

**Fractal Curve Tokenizer** 项目引入了一种 Vision Transformer (ViT) 架构，用**内容自适应四叉树分词**取代传统固定网格分块，并使用 **Hilbert 空间填充曲线**进行排序。通过利用 Hilbert 曲线的局部性保持特性，模型动态向复杂图像区域分配更多 tokens，同时为 transformer 保持空间连贯的序列。

---

## 1.2 动机与关键创新

标准 ViT 存在两个根本限制：

1. **局部性盲区**：光栅扫描序列化破坏了 2D 空间邻近性——垂直相邻的分块在序列中变得遥远
2. **尺度不变性违反**：均匀 16×16 分块在均匀区域（天空、墙壁）浪费计算，同时在复杂区域（边缘、纹理）分辨率不足

本项目通过以下方式解决这些问题：

### Hilbert 局部性保持

Hilbert 排序确保序列中相邻的 tokens 在 2D 空间中是空间邻近的，提供强大的归纳偏置：

$$\|p_1 - p_2\|_2 \leq C \cdot |H^{-1}(p_1) - H^{-1}(p_2)|^{1/2}$$

### 自适应四叉树分割

`StreamingFractalTokenizerV3` 使用可微分分割器基于局部复杂度将图像分解为变大小分块：

$$\text{Split}(R) \iff C(R) > \tau_d$$

### 基于 LCA 的注意力

通过利用四叉树层次中的**最近公共祖先（LCA）**，模型实现了几何注意力偏置，参数约 100 个（vs 标准 ViT 的 $O(N^2)$）：

$$B[i,j] = \text{LCAEmbed}(\text{LCA}(i, j))$$

### 效率

自适应分词相比标准 ViT-16 实现了注意力矩阵大小约 **40× 减少**（224×224 图像）：

| 指标 | 标准 ViT-16 | Fractal ViT |
|:-------|:----------------|:------------|
| Token 数量 | ~196 (14×14 分块) | ~32-64 |
| 注意力矩阵 | $N^2 \approx 38K$ | $N^2 \approx 1-4K$ |
| **减少** | - | **~40×** |

> **注意**：复杂度保持 $O(N^2 \cdot D)$。效率增益来自于 token 数量减少，而非渐近复杂度变化。

---

## 1.3 系统架构管道

**Fractal ViT 数据流**

```
Image (B, C, H, W)
        │
        ▼
┌───────────────────────────────────────────────┐
│         StreamingFractalTokenizerV3            │
│  ┌─────────────────────────────────────────┐ │
│  │ SharedConv (Feature Extraction)          │ │
│  │ HilbertOptimalSplitter (Decision)        │ │
│  │ ROI-Align (Region Pooling)              │ │
│  │ HilbertSort (Curve Ordering)             │ │
│  └─────────────────────────────────────────┘ │
└───────────────────────────────────────────────┘
        │
        ▼
   (tokens, levels_info)
        │
        ▼
┌───────────────────────────────────────────────┐
│       FractalPositionEmbedding                 │
│       (Depth + Path Encoding)                 │
└───────────────────────────────────────────────┘
        │
        ▼
   Level-Aware LayerNorm
        │
        ▼
┌───────────────────────────────────────────────┐
│     FractalTransformerBlock × L                │
│  ┌─────────────────────────────────────────┐ │
│  │ ManifoldNativeAttention (LCA Bias)      │ │
│  │ Level-Aware LayerNorm                    │ │
│  │ AdaptiveFractalFeedForward (SwiGLU)      │ │
│  └─────────────────────────────────────────┘ │
└───────────────────────────────────────────────┘
        │
        ▼
   Global Pooling (CLS / Mean)
        │
        ▼
┌───────────────────────────────────────────────┐
│       MLP Head + LogitsClamp                  │
└───────────────────────────────────────────────┘
        │
        ▼
   Class Logits (B, num_classes)
```

---

## 1.4 架构层次

代码库遵循严格的**4 层层次结构**以实现模块化和依赖管理：

| 层次 | 名称 | 用途 | 关键实体 |
|:------|:-----|:--------|:--------------|
| **L4** | **应用** | 高级模型组装 | `FractalCurveViT` |
| **L3** | **管道** | 分词和 Transformer 阶段 | `StreamingFractalTokenizer`, `FractalTransformer` |
| **L2** | **组件** | 专用层和逻辑 | `HilbertOptimalSplitter`, `ManifoldNativeAttention`, `SwiGLUFFN` |
| **L1** | **基础** | 数学原语和配置 | `HilbertCurve`, `LevelsInfo`, `FractalConfig`, `SplitterProtocol` |

### 导入规则（L1→L2→L3→L4）

```python
# L1 导入：无（基础）
# L2 导入：仅 L1
from vit_pytorch.core.curve_hilbert import HilbertCurve
from vit_pytorch.core.levels_info import LevelsInfo

# L3 导入：L1, L2
from vit_pytorch.modules.tokenizer import StreamingFractalTokenizerV3

# L4 导入：所有
from vit_pytorch import FractalCurveViT
```

> **错误**：`from vit_pytorch.modules.base_splitter import CoreSplitter`
> **正确**：`from vit_pytorch.core.splitter_protocol import CoreSplitter`

---

## 1.5 代码实体关系

```
                    Natural Language Space
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
         "L4 uses L3"          "L3 uses L2 Protocol"
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
              Implementation      ┌───────┴───────┐
                                 ▼               ▼
                          FractalCurveViT   StreamingFractalTokenizer
                          +forward()          +tokenize()
                          +analyze()          +split()
                                               │
                              ┌────────────────┼────────────────┐
                              ▼                ▼                ▼
                       HilbertOptimalSplitter ManifoldNativeAttention SwiGLUFFN
                       +forward()           +forward()         +forward()
                       +update_candidates()
```

---

## 1.6 模型-训练器接口

**原则**：模型定义能力，训练器决定使用方式。

`forward()` 返回 `TrainingStats`：

```python
@dataclass
class TrainingStats:
    logits: Tensor              # 分类 logits
    num_tokens: Tensor         # 每样本 token 数量
    depth_used: Tensor         # 实际使用的最大深度
    depth_distribution: Tensor  # 每深度 token 数量
    features: Optional[Tensor]  # 最终层特征
    transformer_tokens: Tensor  # Transformer 后的 tokens
    # 从各层收集的辅助输出：
    # - splitter_output (entropy, budget_loss, etc.)
    # - attention_outputs[i] (geometric_bias_mean, etc.)
    # - ffn_outputs[i] (level_mixing_mean, etc.)
```

---

## 1.7 文档结构

| 章节 | 内容 |
|:--------|:--------|
| [00_introduction](00_introduction.md) | 项目背景与动机 |
| [01_overview](01_overview.md) | 系统架构（本章） |
| [02_data_structures](02_data_structures.md) | 核心数学基础 |
| [03_fractal_tokenizer](03_fractal_tokenizer.md) | 分词管道 |
| [04_positional_embedding](04_positional_embedding.md) | 位置编码 |
| [05_attention_mechanism](05_attention_mechanism.md) | 流形本地注意力 |
| [06_feedforward_network](06_feedforward_network.md) | 前馈网络 |
| [07_transformer_encoder](07_transformer_encoder.md) | Transformer 块 |
| [08_fractal_vit_model](08_fractal_vit_model.md) | 完整模型 |
| [09_training_system](09_training_system.md) | 训练基础设施 |
| [10_testing_qa](10_testing_qa.md) | 测试和基准测试 |
| [appendix](appendix.md) | 附录 |

---

## 1.8 快速开始

### 安装

```bash
# 使用 uv（推荐）
uv sync

# 或 pip
pip install -e .
```

### 基本用法

```python
import torch
from vit_pytorch import FractalCurveViT

model = FractalCurveViT(
    image_size=224,
    num_classes=1000,
    dim=384,
    num_layers=12,
    heads=6,
    mlp_dim=768,
    tokenizer_type='streaming_v3',
    bias_mode='lca',
    ffn_type='swiglu_level',
)

images = torch.randn(4, 3, 224, 224)
stats = model(images)
print(f"Logits shape: {stats.logits.shape}")  # (4, 1000)
print(f"Token count: {stats.num_tokens.mean().item():.1f}")
```

### 训练

```bash
# CUB-200（动态分辨率）
uv run python src/training/train_fractal_vit.py \
    --dataset cub200 \
    --image-size None \
    --use-amp \
    --compile
```

---

## 1.9 项目状态

| 指标 | 值 |
|:-------|:------|
| 版本 | 0.8.x |
| 测试覆盖率 | 295+ 测试通过 |
| 分词器 | V3（变深度 Tokens） |
| 注意力偏置 | LCA（推荐） |
| FFN | SwiGLU + 级别自适应 |

> **下一章**: [00_introduction.md](00_introduction.md) - 项目背景与动机
