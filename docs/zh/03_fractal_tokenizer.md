# 第三章：分形分词管道

## 3.1 概述

**分形分词管道**将原始输入图像转换为自适应的、变长序列的分形 tokens。与使用固定大小分块刚性网格的标准 Vision Transformers (ViTs) 不同，该管道基于局部复杂度使用四叉树结构化方法映射到 Hilbert 曲线来动态分割图像空间。

这确保了：
- **高熵区域**（边缘、纹理）获得更高的 token 密度
- **低熵区域**（背景）用更少、更大的 tokens 表示

---

## 3.2 管道数据流

```
Image (B, C, H, W)
        │
        ▼
┌───────────────────────────────────────────────┐
│         L2 组件                               │
│  ┌─────────────────────────────────────────┐ │
│  │ SharedConv (特征提取)                     │ │
│  │ HilbertSplitter (最优/Entmax)           │ │
│  │ Vectorized ROI-Align Pooling (向量化 ROI对齐池化) │ │
│  │ HilbertSort (1D 排序)                    │ │
│  └─────────────────────────────────────────┘ │
└───────────────────────────────────────────────┘
        │
        ▼
   Token 序列 (B, N, D)
        │
        ▼
   LevelsInfo (元数据)
```

---

## 3.3 StreamingFractalTokenizerV3

`StreamingFractalTokenizerV3` 是分词引擎的统一实现。

### 3.3.1 关键职责

| 组件 | 用途 |
|:----------|:--------|
| **SharedConv** | 初始特征提取以避免冗余计算 |
| **向量化管道** | P9-1 优化用于并行处理变深度 tokens |
| **HilbertSort** | 将四叉树节点映射到保持空间局部性的 1D 序列 |

### 3.3.2 SplitResult 契约

```python
@dataclass
class SplitResult:
    tokens: Tensor              # [B, N, D] Token 嵌入
    paths: Tensor              # [B, N, max_level] 四叉树路径
    depths: Tensor             # [B, N] Token 深度
    indices: Tensor           # [B, N] 用于排序的 Hilbert 索引
    levels_info: LevelsInfo     # 层次元数据
    num_tokens: Tensor         # [B] 每样本 token 数量
    auxiliary_outputs: Dict[str, float]  # 熵、预算损失等
```

### 3.3.3 向量化处理 (P9-1)

分词器使用**向量化操作**以兼容 `torch.compile`：

```python
def tokenize(self, x: Tensor) -> SplitResult:
    # 步骤 1: 共享特征提取
    features = self.shared_conv(x)  # [B, D, H', W']

    # 步骤 2: 递归分割（向量化）
    candidates = self._generate_candidates(features)

    # 步骤 3: 并行 ROI 池化
    tokens = self.roi_align(features, regions)

    # 步骤 4: Hilbert 排序
    indices = self.hilbert_sorter(paths, depths)

    return SplitResult(...)
```

**源码**：`src/vit_pytorch/modules/tokenizer.py`

---

## 3.4 Token 分割器

分割器是管道的**"大脑"**，决定分形表示的粒度。

### 3.4.1 分割器比较

| 分割器类 | 机制 | 关键特性 |
|:---------------|:----------|:-----------|
| `HilbertOptimalSplitter` | H1SS（6 公理） | **推荐默认**；最优区域选择 |
| `HilbertDistanceDecayConv` | 空间衰减 | 偏向于中心/高细节锚点的分割 |

### 3.4.2 HilbertOptimalSplitter (H1SS)

实现**6 公理**以保证树一致性的推荐默认分割器：

```python
HilbertOptimalSplitter(
    feature_dim=256,
    hidden_dim=64,
    max_level_limit=8,
    K_min=8,              # 最小 token 数量
    K_max=64,             # 最大 token 数量
    entmax_alpha=1.2,     # Entmax 参数
    tree_constraint_weight=0.1,  # 树一致性损失权重
    temperature_init=1.0,
    temperature_min=0.3,
    jump_loss_weight=0.1,  # 鼓励深度跳跃
    density_field_hidden_dim=32,
    use_distance_decay_conv=True,   # I167-1
    use_sds_regularization=False,   # I167-4
    sds_lambda=0.1,
)
```

**公理 (H1SS)**：

| 公理 | 描述 |
|:------|:------------|
| **A1** | 覆盖：根覆盖整个图像 |
| **A2** | 包含：子区域 ⊆ 父区域 |
| **A3** | 不相交：兄弟区域不重叠 |
| **A4** | 最大深度：≤ $D_{max}$ |
| **A5** | 单调性：父区域被选中 ⇒ ≥1 子区域被选中 |
| **A6** | 连续性：选中区域形成连通子图 |

### 3.4.3 HilbertDistanceDecayConv (I167-1)

用于空间混合的**解耦 Hilbert 距离衰减卷积**：

```python
z = Pointwise(Depthwise(x, w_decay))
w_decay[k] = 1 / (|k - center| + 1)
```

**关键属性**：
- **Depthwise**：固定距离衰减权重（0 参数）
- **Pointwise**：可学习的通道混合（D 参数）

### 3.4.4 辅助损失

分割器提供用于训练稳定性的辅助损失：

| 损失 | 用途 | 代码 |
|:-----|:--------|:-----|
| **熵损失** | 防止崩溃到单一尺度 | `splitter.entropy` |
| **预算损失** | 强制执行 token 数量约束 $K_{min}, K_{max}$ | `splitter.budget_loss` |
| **树约束** | 父-子四叉树逻辑的软约束 | `tree_constraint_weight` |

---

## 3.5 Hilbert 模式编码器 (I162-1)

`HilbertPatternEncoder` 对 Hilbert 排序的 tokens 执行**多尺度 1D 卷积**。

### 3.5.1 架构

```
Hilbert 排序的 Tokens [B, N, D]
        │
        ▼
┌─────────────────────────────────────────────┐
│   多尺度 Depthwise Conv1D                   │
│   ┌─────────────────────────────────────┐ │
│   │ kernel_size=3  → 局部模式           │ │
│   │ kernel_size=5  → 区域上下文          │ │
│   │ kernel_size=7  → 广泛结构            │ │
│   └─────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
        │
        ▼
   模式特征 [B, N, D]
```

### 3.5.2 数学公式

$$f_{out} = \text{Concat}\left( \text{Conv1D}_3(x), \text{Conv1D}_5(x), \text{Conv1D}_7(x) \right)$$

**源码**：`src/vit_pytorch/core/pattern_encoder.py`

---

## 3.6 SDS 正则化 (I167-4)

**SDS（空间不连续性分数）**正则化分割器以优先选择空间连贯的区域：

$$\text{SDS} = \frac{1}{N^2} \sum_{i,j} \left| \|p_i - p_j\|_2 - \frac{|h_i - h_j|}{H_{\max}} \right|$$

其中：
- $p_i, p_j$：2D 物理坐标
- $h_i, h_j$：Hilbert 索引
- $H_{\max}$：最大 Hilbert 索引

**用法**：

```python
HilbertOptimalSplitter(
    use_sds_regularization=True,
    sds_lambda=0.1,  # 正则化权重
)
```

---

## 3.7 分块嵌入和位置编码

### 3.7.1 FractalPositionEmbedding

结合深度嵌入与象限路径编码：

$$E_{pos}(i) = \text{Fusion}(E_{depth}(d_i) + E_{path}(i))$$

### 3.7.2 层级感知组件

| 组件 | 用途 |
|:----------|:--------|
| `FractalPathEmbedding` | 编码象限路径序列 |
| `LevelEmbedding` | 编码四叉树深度 |
| `AreaEnhancedEncoding` | 根据分块面积调整幅度 |

### 3.7.3 面积增强编码

根据物理分块大小调整特征幅度：

```python
# 防止大面积分块的信号冲刷
scale = sqrt(area_patch / area_total)
enhanced = tokens * scale
```

---

## 3.8 系统集成

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FractalCurveViT                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Image (B, C, H, W)                                                 │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────┐     │
│  │            StreamingFractalTokenizerV3                       │     │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌────────────┐   │     │
│  │  │   SharedConv    │→ │ HilbertOptimal  │→ │ HilbertSort │   │     │
│  │  │ (特征提取)      │  │ Splitter (H1SS)  │  │            │   │     │
│  │  └─────────────────┘  └─────────────────┘  └────────────┘   │     │
│  └─────────────────────────────────────────────────────────────┘     │
│       │                                                              │
│       ▼                                                              │
│  SplitResult (tokens, paths, depths, levels_info)                   │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────┐     │
│  │            FractalPositionEmbedding                          │     │
│  │  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐    │     │
│  │  │ LevelEmbed  │→ │ FractalPath  │→ │ AreaEnhanced  │    │     │
│  │  │ (深度)       │  │ (象限)        │  │ (幅度)        │    │     │
│  │  └──────────────┘  └──────────────┘  └───────────────┘    │     │
│  └─────────────────────────────────────────────────────────────┘     │
│       │                                                              │
│       ▼                                                              │
│  FractalTransformer × L                                              │
│       │                                                              │
│       ▼                                                              │
│  Class Logits                                                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3.9 关键创新 (I162-1, I167-1, I167-2, I167-4)

| 问题 | 解决方案 | 参考 |
|:------|:---------|:----------|
| **I162-1** | Hilbert 模式编码器用于多尺度特征 | `pattern_encoder.py` |
| **I167-1** | 距离衰减卷积用于空间混合 | `hilbert_distance_decay_conv.py` |
| **I167-2** | SDSMetric 用于 Hilbert 局部性验证 | `curve_hilbert.py` |
| **I167-4** | 分割器中的 SDS 正则化 | `HilbertOptimalSplitter` |

---

## 3.10 文档导航

| 章节 | 内容 |
|:--------|:--------|
| [03_fractal_tokenizer](03_fractal_tokenizer.md) | 分形分词管道（本章） |
| [04_positional_embedding](04_positional_embedding.md) | 位置编码 |
| [05_attention_mechanism](05_attention_mechanism.md) | 流形本地注意力 |

> **下一章**: [04_positional_embedding.md](04_positional_embedding.md) - 位置编码
