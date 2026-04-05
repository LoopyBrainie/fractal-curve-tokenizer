# 第一章：系统架构

## 1.1 概述

本章描述了 Fractal Curve ViT 架构的完整数据流和模块结构。

### 高级管道

$$I \xrightarrow{\text{Tokenize}} (T, L) \xrightarrow{E_{pos}} T' \xrightarrow{\text{CLS}} [c; T'] \xrightarrow{\text{Transformer}} X' \xrightarrow{\text{Pool}} z \xrightarrow{\text{MLP}} \hat{y}$$

其中：
- $I \in \mathbb{R}^{B \times C \times H \times W}$：输入图像批次
- $T \in \mathbb{R}^{B \times N \times D}$：Token 嵌入
- $L \in \mathbb{Z}^{B \times N \times \text{Info}}$：深度信息（四叉树路径）
- $X' \in \mathbb{R}^{B \times (N+1) \times D}$：编码序列（带 CLS token）
- $\hat{y} \in \mathbb{R}^{B \times C_{out}}$：分类 logits

---

## 1.2 数据流图

```
输入图像 (B, C, H, W)
        │
        ▼
┌───────────────────────────────────────────────┐
│       StreamingFractalTokenizerV3             │
│  ┌─────────────────────────────────────────┐  │
│  │ 1. 可学习复杂度: C_theta(R)              │  │
│  │ 2. 可微四叉树分割                        │  │
│  │ 3. 通过 ROI-Align 的区域池化             │  │
│  │ 4. Hilbert 曲线重排序                    │  │
│  └─────────────────────────────────────────┘  │
└───────────────────────────────────────────────┘
        │
        ▼
   (tokens, levels_info)
        │
        ▼
┌───────────────────────────────────────────────┐
│       FractalPositionEmbedding                │
│  E_pos(i) = Fusion(E_depth(d_i) + E_path(i))  │
└───────────────────────────────────────────────┘
        │
        ▼
   T' = T + E_pos
        │
        ▼
┌───────────────────────────────────────────────┐
│            添加 CLS Token                      │
│       [CLS; T'] → (B, N+1, D)                 │
└───────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────┐
│         FractalTransformer × L                │
│  ┌─────────────────────────────────────────┐  │
│  │ 深度感知 LayerNorm                       │  │
│  │ HilbertAwareMultiScaleAttention         │  │
│  │   + LCA Hilbert 偏置                     │  │
│  │ DropPath + 残差连接                       │  │
│  │ 深度感知 LayerNorm                       │  │
│  │ AdaptiveFractalFeedForward (SwiGLU)     │  │
│  │ DropPath + 残差连接                       │  │
│  └─────────────────────────────────────────┘  │
└───────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────┐
│     池化: CLS 或 Mean                          │
└───────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────┐
│     MLP 头: LN → Linear → GELU → Linear       │
└───────────────────────────────────────────────┘
        │
        ▼
   Logits (B, num_classes)
```

---

## 1.3 模块层次结构

| 层级 | 模块 | 文件 | 核心功能 |
|:------|:-------|:-----|:-------------------|
| **L4** | 应用层 | `model_fractal_vit.py` | `FractalCurveViT` |
| **L3** | 管道层 | `tokenizer_streaming.py` | `StreamingFractalTokenizerV3` |
|        |          | `block_transformer.py` | `FractalTransformer` |
| **L2** | 组件层 | `gumbel_topk_splitter.py` | `GumbelTopKSplitter` (方案 D/E) |
|        |            | `attn_hilbert_bias.py` | `HilbertAwareMultiScaleAttention`, `LCAHilbertBias` |
|        |            | `ffn_swiglu.py` | `SwiGLUFFN`, `AdaptiveFractalFeedForward` |
|        |            | `embed_fractal_position.py` | `FractalPositionEmbedding` |
| **L1** | 基础层 | `curve_hilbert.py` | `HilbertCurve`, `PseudoHilbertCurve` |

---

## 1.4 关键创新

### 1.4.1 变深度 Tokens (V3)

与固定网格分词不同，V3 执行**内容自适应四叉树分割**：

$$\text{Split}(R) \iff C(R) > \tau_d$$

其中：
- $C(R) = \alpha \cdot \frac{\text{Var}(R)}{\text{Var}(R) + \sigma_0^2} + (1-\alpha) \cdot \frac{G(R)}{G(R) + g_0^2}$
- $\tau_d = \tau_0 \cdot \gamma^d$（深度相关阈值）

### 1.4.2 LCA Hilbert 偏置

从四叉树 LCA（最近公共祖先）深度派生的注意力偏置：

$$B[i,j] = \text{LCAEmbed}(\text{LCA}(i, j))$$

这提供了明确的几何意义，仅需 ~100 个可学习参数。

### 1.4.3 带深度自适应的 SwiGLU FFN

$$\text{SwiGLU}(x) = W_{out} \cdot (\text{Swish}(W_{gate} \cdot x) \odot W_{value} \cdot x)$$

扩展为深度自适应残差：

$$\text{Output} = (1 - \alpha_d) \cdot \text{FFN}(x) + \alpha_d \cdot \text{Adapter}([x; E_{level}(d)])$$

---

## 1.5 配置

### FractalConfig

```python
from vit_pytorch import FractalConfig

config = FractalConfig(
    image_size=224,
    min_patch_size=4,
    tokenizer_type='streaming_v3',
)
```

### 模型实例化

```python
from vit_pytorch import FractalCurveViT

model = FractalCurveViT(
    image_size=224,
    num_classes=1000,
    dim=384,
    num_layers=12,
    heads=6,
    mlp_dim=768,
    ffn_type='swiglu_level',
)
```

---

## 1.6 复杂度分析

> **关键说明**："40×  reduction" 指的是 **token 数量减少**，而非渐近复杂度。Token 数量从 ~307K（224×224 图像的标准 ViT 16×16 patches）减少到 ~32（V3 变深度 tokens）。

| 操作 | 复杂度 | 备注 |
|:----------|:-----------|:------|
| 分词 | $O(N_{cand} \cdot D)$ | $N_{cand} = \sum_{d=0}^{D} 4^d$（例如 D=3 时为 85） |
| Hilbert 重排序 | $O(N \log N)$ | 按 Hilbert 索引排序 |
| LCA 计算 | $O(N^2)$ | 缓存后摊销为 $O(1)$ |
| 注意力 | $O(N^2 \cdot D)$ | 标准 transformer（N ≈ 32） |
| **有效计算量** | ~40× 减少 | $N_{V3} \approx 32$ vs $N_{ViT} \approx 307K$ |

> **注意**：由于与 GPU SIMT 并行性的根本不兼容性，完整的动态深度自适应已推迟。当前实现使用固定的 `depth // 2` 以提高并行效率。

> **下一章**: [02_data_structures.md](02_data_structures.md) - 核心数据结构
