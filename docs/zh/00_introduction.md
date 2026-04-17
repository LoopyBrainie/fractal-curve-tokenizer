# 第零章：项目背景与动机

## 0.1 问题陈述：固定分块 ViT 的局限性

传统 Vision Transformer 将图像分割为固定大小的网格（例如 16×16 分块）。这种方法存在两个主要低效问题：

### 0.1.1 尺度不变性违反

均匀网格对高熵区域（边缘、纹理）和低熵区域（天空、均匀背景）使用相同分辨率。这导致：
- **冗余计算**：简单区域浪费 tokens（均匀区域上的 tokens）
- **信息丢失**：复杂区域分辨率不足（细粒度细节）

### 0.1.2 局部性盲区

标准的光栅扫描序列化破坏了 2D 空间邻近性。图像中垂直相邻的分块，在线性序列中变得遥远，迫使 Transformer 从头学习空间关系，没有任何几何归纳偏置。

---

## 0.2 数学基础

### 0.2.1 Hilbert 局部性边界

Hilbert 曲线 ($H$) 是一个连续分形映射：

$$H: [0, n^2) \leftrightarrow [0, n) \times [0, n)$$

它具有严格的**局部性边界**：

$$\|p_1 - p_2\|_2 \leq C \cdot |H^{-1}(p_1) - H^{-1}(p_2)|^{1/2}$$

这确保了**Hilbert 序列中相近的点在 2D 空间中也保证是空间邻近的**。指数 1/2 反映了 Hilbert 曲线的分形维度。

### 0.2.2 四叉树-Hilbert 同构

一个核心的技术洞察：四叉树路径和 Hilbert 曲线段之间存在**双射**。由四叉树路径定义的任何区域 $R$ 可以精确映射到 Hilbert 曲线的连续段：

$$\text{QuadtreePath}(R) = [q_1, q_2, \ldots, q_d] \iff \text{HilbertSegment}(R) = H|_{[a,b]}$$

这种同构使得：
- 变大小的四叉树区域可以映射到连续的 Hilbert 段
- 空间关系通过 token 排序保持
- 层次结构编码在序列中

### 0.2.3 基于 LCA 的几何

四叉树的层次性质允许计算任意两个 token 之间的**最近公共祖先（LCA）**：

$$\text{LCA}(i, j) = \text{Length}(\text{CommonPrefix}(\text{Path}(i), \text{Path}(j)))$$

LCA 的深度作为空间和结构距离的代理：
- **LCA 深度高** → token 在附近象限 → 更强的注意力偏置
- **LCA 深度低** → token 在远处区域 → 更弱的注意力偏置

这提供了**最小参数的几何注意力偏置**（~100 参数 vs 标准 ViT 的 $O(N^2)$）。

---

## 0.3 架构概述

### 当前实现：V3 变深度 Tokens

系统使用**内容自适应四叉树分割**，每个区域有独立的分割决策：

$$\text{Split}(R) \iff C(R) > \tau_d$$

其中 $C(R)$ 是可学习的复杂度度量，$\tau_d$ 是深度相关阈值。每个区域独立做出决策，不与其他尺度竞争。

### 关键组件

| 组件 | 用途 |
|:----------|:--------|
| `StreamingFractalTokenizerV3` | 自适应 token 生成 |
| `HilbertOptimalSplitter` | 区域分割决策 |
| `ManifoldNativeAttention` | 基于 LCA 的几何注意力 |
| `SwiGLUFFN` | 级别自适应前馈网络 |

---

## 0.4 系统映射：概念到代码

### 分词管道数据流

```
Image (B, C, H, W)
        │
        ▼
┌───────────────────────────────────────────────┐
│       StreamingFractalTokenizerV3             │
│  ┌─────────────────────────────────────────┐ │
│  │ SharedConv (Feature Extraction)         │ │
│  │ HilbertOptimalSplitter (Decision)       │ │
│  │ ROI-Align (Region Pooling)             │ │
│  │ HilbertSort (Curve Ordering)             │ │
│  └─────────────────────────────────────────┘ │
└───────────────────────────────────────────────┘
        │
        ▼
   TokenizerOutput (tokens, levels_info)
        │
        ▼
┌───────────────────────────────────────────────┐
│       FractalPositionEmbedding                │
│       (Depth + Path Encoding)                 │
└───────────────────────────────────────────────┘
        │
        ▼
   FractalTransformerBlock × L
        │
        ▼
   Class Logits
```

### 架构组件层次

```python
from vit_pytorch import FractalCurveViT

model = FractalCurveViT(
    image_size=224,
    num_classes=1000,
    dim=384,
    num_layers=12,
    heads=6,
    mlp_dim=768,
)

images = torch.randn(4, 3, 224, 224)
logits = model(images)  # (4, 1000)
```

```
FractalCurveViT (L4)
├── tokenizer: StreamingFractalTokenizerV3
│   ├── patch_embed: HilbertNativePatchEmbed
│   └── splitter: CoreSplitter → HilbertOptimalSplitter
├── transformer: FractalTransformer
│   └── layers: FractalTransformerBlock[]
│       ├── attention: ManifoldNativeAttention
│       │   └── bias: LCAHilbertBias
│       └── ffn: AdaptiveFractalFeedForward
└── pos_drop: nn.Dropout
```

---

## 0.5 效率增益

通过使用变深度 tokens，模型显著减少了 Transformer 处理的序列长度 $N$：

| 指标 | 标准 ViT-16 | Fractal ViT |
|:-------|:----------------|:------------|
| 图像大小 | 224×224 | 224×224 |
| 分块大小 | 16×16 | 变大小 (4×4 到 64×64) |
| Token 数量 ($N$) | ~196 (14×14) | ~32-64 |
| 注意力矩阵 ($N^2$) | ~38K | ~1-4K |
| **减少因子** | - | **~40×** |

> **注意**：注意力复杂度保持 $O(N^2 \cdot D)$。~40× 效率增益来自于 token 数量减少（$N \approx 32-64$ vs $196$），而非渐近复杂度变化。

### 为什么变 Token 数量有效

- 标准 ViT，16×16 分块，224×224 图像：$N = (224/16)^2 = 196$ tokens
- Fractal ViT，自适应分割：$N \approx 32-64$ tokens（取决于图像复杂度）
- 模型学习向复杂区域（边缘、纹理）分配更多 tokens，向简单区域（背景）分配更少 tokens

---

## 0.6 关键创新总结

| 创新 | 描述 | 收益 |
|:-----------|:------------|:--------|
| **Hilbert 局部性** | 空间填充曲线在 1D 序列中保持 2D 邻近性 | 强大的几何归纳偏置 |
| **自适应四叉树** | 基于复杂度的内容依赖分割 | 高效的 token 分配 |
| **LCA 注意力偏置** | 来自四叉树层次的几何偏置 | ~100 参数而非 $O(N^2)$ |
| **尺度感知残差** | 父到子的信息流 | 防止信息冲刷 |

---

## 0.7 文档导航

| 章节 | 内容 |
|:--------|:--------|
| [00_introduction](00_introduction.md) | 项目背景与动机（本章） |
| [01_overview](01_overview.md) | 系统架构概述 |
| [02_data_structures](02_data_structures.md) | 核心数学基础 |
| [03_fractal_tokenizer](03_fractal_tokenizer.md) | 分词管道 |

> **下一章**: [01_overview.md](01_overview.md) - 系统架构
