# 第二章：核心数据结构

## 2.1 概述

本章定义了流经 Fractal Curve ViT 管道的基本数据结构。

---

## 2.2 TokenizerOutput

所有分词器的统一输出结构。

### 定义

```python
@dataclass
class TokenizerOutput:
    sequences: List[TokenSequence]  # 每张图像的 token 序列
```

### TokenSequence

```python
@dataclass
class TokenSequence:
    tokens: Tensor           # (N, D) - token 嵌入
    attention_mask: Tensor   # (N,) - 有效 token 掩码
    metadata: Dict[str, Any] # 附加信息

    def get_levels(self) -> Tensor:
        """从 metadata 中提取 levels_info。"""
        return self.metadata.get('levels', None)
```

### 访问模式

```python
# 按图像访问
for seq in output.sequences:
    tokens = seq.tokens          # (N_i, D)
    levels = seq.get_levels()    # (N_i, max_depth+1)

# 批次访问（带填充）
tokens, levels, mask = output.to_batch()  # (B, N_max, D), (B, N_max, Info), (B, N_max)
```

---

## 2.3 levels_info 张量

每个 token 的层次位置编码。

### 形状

$$L \in \mathbb{Z}^{B \times N \times (d_{max} + 1)}$$

### 结构

| 索引 | 内容 | 范围 | 描述 |
|:------|:--------|:------|:------------|
| `[:, :, 0]` | 深度 | $[0, d_{max}]$ | Token 的四叉树深度 |
| `[:, :, 1:]` | 路径 | $[0, 3]^{d_{max}}$ | 四叉树路径（象限索引） |

### 象限编码

```
象限索引（Hilbert 兼容）：

    ┌─────┬─────┐
    │  1  │  2  │
    ├─────┼─────┤
    │  0  │  3  │
    └─────┴─────┘
```

### 数学解释

对于深度为 $d$、路径为 $[q_1, q_2, \ldots, q_d]$ 的 token：

$$\text{position}(t) = \sum_{i=1}^{d} q_i \cdot 4^{d-i}$$

这与 Hilbert 曲线段双射对应。

### 示例

```python
# 深度为 2、路径为 [1, 3] 的 token（右下 → 右上）
levels_info[b, t] = [2, 1, 3, 0, 0, 0]
#                    ^  ^  ^  ^^^^^^^
#                    |  |  |  填充（未使用）
#                    |  |  └── q_2 = 3
#                    |  └───── q_1 = 1
#                    └──────── 深度 = 2
```

---

## 2.4 FractalConfig

整个系统的统一配置数据类。

### 定义

```python
@dataclass
class FractalConfig:
    # 模型维度
    d_model: int = 384
    num_heads: int = 6

    # 分词器配置
    image_size: int = 224
    min_patch_size: int = 4
    max_depth: int = 4

    # Hilbert 偏置配置
    hilbert_bias_mode: str = 'lca'  # 'lca', 'low_rank', 'hierarchical'
    low_rank_r: int = 32

    # 注意力参数
    lca_temperature: float = 1.5
    learnable_temperature: bool = True

    # FFN 配置
    ffn_type: str = 'swiglu_level'
```

### 派生属性

```python
@property
def num_scales(self) -> int:
    """四叉树中的尺度数量。"""
    return self.max_depth + 1

@property
def patch_sizes(self) -> Tuple[int, ...]:
    """从细到粗的可用 patch 大小。"""
    return tuple(self.min_patch_size * (2 ** i) for i in range(self.num_scales))
```

---

## 2.5 AdaptiveSplitConfig

内容自适应四叉树分割的配置。

### 复杂度函数参数

| 参数 | 符号 | 默认值 | 描述 |
|:----------|:-------|:--------|:------------|
| `alpha` | $\alpha$ | 0.5 | 方差权重，取值范围 $[0, 1]$ |
| `sigma_0_sq` | $\sigma_0^2$ | 0.01 | 方差归一化常数 |
| `g_0_sq` | $g_0^2$ | 0.08 | 梯度归一化常数 |

### 阈值函数参数

| 参数 | 符号 | 默认值 | 描述 |
|:----------|:-------|:--------|:------------|
| `tau_0` | $\tau_0$ | 0.15 | 根阈值 |
| `gamma` | $\gamma$ | 0.85 | 阈值衰减因子 |
| `max_depth` | $d_{max}$ | 4 | 最大分割深度 |

### 分割方案

```python
class SplitScheme(Enum):
    BALANCED_GREEDY = "balanced_greedy"  # 方案 B：带 2:1 平衡的贪心算法
    FIXED_BUDGET_DP = "fixed_budget_dp"  # 方案 C：带 token 预算的动态规划
    LEARNABLE = "learnable"              # 方案 L：端到端可学习
```

---

## 2.6 QuadtreeNode

分割过程中四叉树节点的内部表示。

### 定义

```python
@dataclass
class QuadtreeNode:
    x: int              # 左上角 x 坐标
    y: int              # 左上角 y 坐标
    size: int           # 区域大小（像素）
    depth: int          # 四叉树深度
    path: List[int]     # 从根出发的象限路径
    complexity: float   | 计算的复杂度 C(R)

    @property
    def region(self) -> Tuple[int, int, int, int]:
        """返回 (x, y, x+size, y+size) 边界框。"""
        return (self.x, self.y, self.x + self.size, self.y + self.size)
```

### 不变式

1. **大小约束**：$\text{size} = \text{image\_size} / 2^{\text{depth}}$
2. **路径长度**：$\text{len(path)} = \text{depth}$
3. **对齐**：$(x, y)$ 对齐到 $\text{size}$ 像素网格

---

## 2.7 HilbertIndex

二维坐标与 Hilbert 曲线位置之间的映射。

### 数学定义

$$H: [0, n^2) \leftrightarrow [0, n) \times [0, n)$$

### 实现

```python
class HilbertCurve:
    def __init__(self, order: int):
        """初始化给定阶数的 Hilbert 曲线。

        参数:
            order: 网格大小的对数（例如 order=4 → 16×16 网格）
        """
        self.order = order
        self.n = 2 ** order

    def d2xy(self, d: int) -> Tuple[int, int]:
        """将 Hilbert 索引转换为 (x, y) 坐标。"""
        ...

    def xy2d(self, x: int, y: int) -> int:
        """将 (x, y) 坐标转换为 Hilbert 索引。"""
        ...
```

### 局部性性质

对于任意两点 $p_1, p_2$：

$$\|p_1 - p_2\|_2 \leq C \cdot |H^{-1}(p_1) - H^{-1}(p_2)|^{1/2}$$

---

## 2.8 张量形状约定

### 输入/输出形状

| 张量 | 形状 | 描述 |
|:-------|:------|:------------|
| 图像 | $(B, C, H, W)$ | 输入图像批次 |
| Tokens | $(B, N, D)$ | Token 嵌入 |
| Levels | $(B, N, d_{max}+1)$ | 深度信息 |
| 注意力掩码 | $(B, 1, 1, N)$ | 广播兼容掩码 |
| Hilbert 偏置 | $(B, H, N, N)$ | 每头注意力偏置 |
| Logits | $(B, C_{out})$ | 分类输出 |

### 维度符号

| 符号 | 含义 | 典型值 |
|:-------|:--------|:--------------|
| $B$ | 批次大小 | 32 |
| $C$ | 图像通道数 | 3 |
| $H, W$ | 图像高度/宽度 | 224 |
| $N$ | Token 数量 | 16-196 |
| $D$ | 模型维度 | 384 |
| $H$ | 注意力头数 | 6 |
| $d_{max}$ | 最大深度 | 4 |

> **下一章**: [03_fractal_tokenizer.md](03_fractal_tokenizer.md) - 分词管道
