# Fractal Curve Tokenizer（分形曲线分词器）

[English](README.md) | [中文](README_zh.md)

一个采用 **Hilbert 曲线分词** 和 **自适应多尺度 patch 选择** 的视觉 Transformer。Fractal ViT 根据图像内容复杂度动态分配 token——复杂区域分配更多 token，均匀背景分配更少 token。

## Fractal ViT vs. 标准 ViT

|              | 标准 ViT        | Fractal ViT            |
| ------------ | ------------- | ---------------------- |
| **分词方式**     | 固定 16×16 网格   | 自适应四叉树 (4×4 ~ 32×32)   |
| **Token 数量** | 固定: N = HW/P² | 可变: N ∈ [N_min, N_max] |
| **Patch 排序** | 光栅扫描 (行优先)    | Hilbert 曲线 (保局部性)      |
| **位置编码**     | 可学习 N² 偏置     | 基于 LCA 的层级偏置 (~100 参数) |
| **归纳偏置**     | 无 (数据驱动)      | 空间局部性 + 多尺度结构          |

### 固定网格的问题

标准 ViT 对所有图像区域一视同仁：

```
标准 ViT: 4 tokens (均匀 16×16)           Fractal ViT: 5 tokens (自适应)
┌────────────┬────────────┐                ┌──────────────────────────┐
│            │            │                │                          │
│   16×16    │   16×16    │                │          32×32           │
│   (天空)    │   (天空)   │                │       (均匀天空)          │
├────────────┼────────────┤       →        ├──────┬──────┬────────────┤
│            │            │                │ 4×4  │ 4×4  │            │
│   16×16    │   16×16    │                |(眼睛)│ (眼睛)│    8×8     │
│   (人脸)    │   (人脸)   │                └──────┴──────┴────────────┘
└────────────┴────────────┘
```

Fractal ViT 为复杂区域（眼睛、边缘）分配**细粒度 token**，为均匀区域（天空、背景）分配**粗粒度 token**。

## 核心技术贡献

### 1. Hilbert 曲线排序

Patch 沿 **Hilbert 空间填充曲线** 排序，而非行优先光栅扫描：

$$H: [0, n^2) \leftrightarrow [0, n) \times [0, n)$$

**为什么用 Hilbert？** 该曲线在 1D 序列中保持 2D 局部性：
$$\|p_1 - p_2\|_2 \leq C \cdot |H^{-1}(p_1) - H^{-1}(p_2)|^{1/2}$$

这意味着空间相邻的 patch 在注意力序列中也保持接近，提升局部注意力模式的有效性。

![Hilbert 曲线阶数](workspace/visualizations/hilbert_curve.png)

*架构工具直接生成的 Hilbert 曲线（示例阶数 4）。*

### 2. 自适应四叉树分词

我们使用**内容感知四叉树**替代固定 patch 大小，根据局部复杂度分割区域：

$$C(R) = \alpha \cdot \frac{\text{Var}(R)}{\text{Var}(R) + \sigma_0^2} + (1-\alpha) \cdot \frac{G(R)}{G(R) + g_0^2}$$

- **Var(R)**：局部像素方差（纹理）
- **G(R)**：梯度能量（边缘）
- **分割条件**：$C(R) > \tau_0 \cdot \gamma^d$（深度相关阈值）

![四叉树结构](workspace/visualizations/quadtree.png)

*由模型分割启发式驱动的合成自适应四叉树（无需训练数据）。*

### 3. 基于 LCA 的注意力偏置

我们用**最低公共祖先 (LCA)** 公式替换标准 N² 可学习位置偏置：

$$B[i,j] = \tau_h \cdot \text{Embed}(\text{LCA}(i,j))$$

其中 LCA(i,j) 是 token i 和 j 的路径首次分叉的树深度。这提供了：

- **O(log N) 参数**，相比标准 ViT 的 O(N²)
- **明确的几何意义**：相邻 patch 共享更深的祖先
- **每头可学习温度** τ_h，用于自适应缩放

![LCA 偏置矩阵](workspace/visualizations/lca_bias.png)

*仅由四叉树路径推导的 LCA 深度矩阵（架构级偏置）。*

## 架构

```
图像 (B, C, H, W)
       │
       ▼
┌─────────────────────────────────────┐
│  StreamingFractalTokenizerV3        │
│  ├─ SharedConv: 特征提取            │
│  ├─ AdaptiveSplit: 四叉树分割       │
│  ├─ ROI-Align: 变尺寸池化           │
│  └─ HilbertSort: 曲线排序           │
└─────────────────────────────────────┘
       │
       ▼ (T ∈ ℝ^{B×N×D}, L ∈ ℤ^{B×N×(1+depth)})
┌─────────────────────────────────────┐
│  FractalPositionEmbedding           │
│  └─ 深度 + 象限路径编码             │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  FractalTransformer (×L 层)         │
│  ├─ HilbertAttention + LCA 偏置     │
│  └─ SwiGLU FFN + 层级自适应         │
└─────────────────────────────────────┘
       │
       ▼
   [CLS] 池化 → MLP Head → Logits
```

## 快速开始

```python
import torch
from vit_pytorch import FractalCurveViT

model = FractalCurveViT(
    image_size=224,
    num_classes=1000,
    dim=384,
    depth=6,
    heads=6,
    tokenizer_type="streaming_v3",
    hilbert_bias_mode="lca",
)

img = torch.randn(1, 3, 224, 224)
logits = model(img)  # (1, 1000)
```

## 配置选项

### 分割方案

| 方案                | 说明                    |
| ----------------- | --------------------- |
| `balanced_greedy` | 快速优先队列分割，带 2:1 平衡约束   |
| `fixed_budget_dp` | 动态规划，严格 token 预算      |
| `learnable`       | 端到端可微（Gumbel-Softmax） |

### 注意力偏置模式

| 模式             | 参数量  | 说明             |
| -------------- | ---- | -------------- |
| `lca`          | ~100 | **推荐** LCA 嵌入表 |
| `low_rank`     | ~50K | 分解偏置: B = ΦΨᵀ  |
| `hierarchical` | ~5K  | 逐层可学习偏置        |

## 安装

```bash
git clone https://github.com/LoopyBrainie/fractal-curve-tokenizer.git
cd fractal-curve-tokenizer
pip install -e .  # 或: uv sync
```

## 训练

```bash
# CIFAR-10
python examples/training/train_fractal_vit.py --dataset cifar10 --epochs 50

# 使用可学习分割
python examples/training/train_fractal_vit.py \
    --dataset tiny-imagenet \
    --split-scheme learnable \
    --epochs 100
```

## 测试

```bash
pytest tests/  # 295 个测试用例
```

## 相关工作

- **ViT** (Dosovitskiy et al., 2020)：固定 16×16 patch，光栅排序
- **Swin Transformer**：移动窗口，但仍是均匀网格
- **Quadtree Transformer**：类似四叉树思想，不同注意力机制
- **Fractal ViT (本项目)**：Hilbert 排序 + LCA 偏置 + 自适应分割

## 许可证

MIT 许可证 - 详见 [LICENSE](LICENSE)。
