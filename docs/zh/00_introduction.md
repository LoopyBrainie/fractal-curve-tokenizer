# 引言

> **Vision Transformer 的分形曲线分词器**

## 背景与动机

传统 Vision Transformer (ViT) 将图像分割为固定大小的 patches（例如 16×16），忽略了视觉内容的内在多尺度结构。本研究探索了一种替代方案：**由 Hilbert 空间填充曲线引导的内容自适应分词**。

### 问题陈述

标准 ViT 分词存在两个根本性限制：

1. **尺度不变性违反**：均匀 patches 无法高效表示细粒度纹理和粗粒度语义
2. **局部性盲区**：patches 的线性序列不编码空间邻近性

### 假设

**Hilbert 曲线的局部性保持特性**为 Vision Transformer 提供了自然的归纳偏置，使得：

- 空间连贯的 token 序列成为可能
- 通过 LCA（最近公共祖先）关系实现层次化注意力模式
- 通过自适应四叉树分割实现多尺度表示

---

## 数学基础

### Hilbert 曲线

**Hilbert 曲线**是一种连续的分形映射，在保持局部性的同时填充二维空间：

$$H: [0, n^2) \leftrightarrow [0, n) \times [0, n)$$

**局部性边界**：对于二维网格中的任意两点 $p_1, p_2$：

$$\|p_1 - p_2\|_2 \leq C \cdot |H^{-1}(p_1) - H^{-1}(p_2)|^{1/2}$$

其中 $C$ 是依赖于维度的常数。这确保了 Hilbert 序列中相邻的位置对应于空间上接近的位置。

### 四叉树-Hilbert 同构

本研究的一个基本性质：四叉树路径与 Hilbert 曲线段之间的双射：

$$\text{QuadtreePath}(R) = [q_1, q_2, \ldots, q_d] \iff \text{HilbertSegment}(R) = H|_{[a,b]}$$

这使得自适应四叉树分割能够保持 Hilbert 有序的 token 序列。

---

## 架构概览

| 组件 | 实现 | 描述 |
|:----------|:---------------|:------------|
| 分词器 | `StreamingFractalTokenizerV3` | 自适应四叉树 + Hilbert 重排序 |
| 位置编码 | `FractalPositionEmbedding` | 深度 + 路径编码 |
| 注意力偏置 | `LCAHilbertBias` | ~100 个参数，明确的几何意义 |
| FFN | `SwiGLUFFN` | 门控激活 + 深度自适应 |

---

## 快速开始

### 安装

```bash
# 使用 uv（推荐）
uv sync

# 或使用 pip
pip install -e .
```

### 基本用法

```python
from vit_pytorch import FractalCurveViT

model = FractalCurveViT(
    image_size=224,
    num_classes=1000,
    dim=384,
    depth=6,
    heads=6,
    mlp_dim=768,
    tokenizer_type='streaming_v3',  # 变深度 Tokens
    hilbert_bias_mode='lca',        # LCA Hilbert 偏置
)

images = torch.randn(4, 3, 224, 224)
logits = model(images)  # (4, 1000)
```

---

## 文档结构

| 章节 | 内容 |
|:--------|:--------|
| [01_overview](01_overview.md) | 系统架构与数据流 |
| [02_data_structures](02_data_structures.md) | 核心数据结构 |
| [03_fractal_tokenizer](03_fractal_tokenizer.md) | 分词管道 |
| [04_positional_embedding](04_positional_embedding.md) | 位置编码 |
| [05_attention_mechanism](05_attention_mechanism.md) | Hilbert 感知注意力 |
| [06_feedforward_network](06_feedforward_network.md) | 前馈网络 |
| [07_transformer_encoder](07_transformer_encoder.md) | Transformer 编码器 |
| [08_fractal_vit_model](08_fractal_vit_model.md) | 完整模型 |
| [09_training_system](09_training_system.md) | 训练系统 |
| [10_testing_qa](10_testing_qa.md) | 测试与质量保证 |
| [11_issues_roadmap](11_issues_roadmap.md) | 开发历史 |
| [appendix](appendix.md) | 附录 |

---

## 项目状态

| 指标 | 值 |
|:-------|:------|
| 版本 | 0.8.x |
| 测试覆盖率 | 295+ 测试通过 |
| 分词器 | V3（变深度 Tokens） |
| Hilbert 偏置 | LCA（推荐） |
| FFN | SwiGLU + 深度自适应 |

> **下一章**: [01_overview.md](01_overview.md) - 系统架构
