# 第五章：注意力机制 (attention.py)

本章详细解析 `HilbertAwareMultiScaleAttention`，这是模型理解分形结构和空间关系的核心组件。

## 5.1 数学形式化

### 标准多头注意力

$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) \cdot V$

### Hilbert 感知注意力

$\text{HilbertAttn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} \cdot \sigma_{scale} + B_{hilbert} + B_{level}\right) \cdot V$

---

## 5.2 偏置项详解

### 1. LCA Hilbert Bias (LCA 嵌入，✅ 默认推荐)

**数学定义**:
$B_{hilbert}[i,j] = \text{LCAEmbed}(\text{LCA}(i, j))$

其中 $\text{LCA}(i, j)$ 是 token $i$ 和 $j$ 在四叉树上的最近公共祖先深度。

**定理 (LCA-距离等价性)**:
对于四叉树编码的两个 token $i, j$:
$$\text{LCA}(i, j) = \ell \implies \|pos_i - pos_j\|_\infty \le N / 2^\ell$$

其中 $N$ 是网格边长。这意味着 LCA 深度直接编码了空间距离的上界。

**初始化** (对数衰减):
$\text{embed}[d] \propto \frac{\log(1 + d)}{\log(1 + D_{max})}$

深层（邻近）token 获得更高的初始偏置。

**优势**:
| 指标 | LCA | 值 |
| :--- | :--- | :--- |
| 参数量 | $(D_{max} + 1) \times H$ | ~100 (max_depth=8, heads=8) |
| 计算复杂度 | $O(N^2 \times D)$ | 可预计算缓存 |
| 可解释性 | ⭐⭐⭐⭐⭐ | LCA 深度 ⟺ 空间距离 |
| 几何意义 | 显式 | 无需学习距离关系 |

**实现类**: `LCAHilbertBias`

```python
class LCAHilbertBias(nn.Module):
    def __init__(self, max_depth, heads):
        self.lca_embedding = nn.Embedding(max_depth + 1, heads)
        # 对数衰减初始化
```

### 2. Low-Rank Hilbert Bias (低秩分解)

**数学定义**:
$B_{hilbert}[i,j] = \phi(p_i)^T \cdot \psi(p_j)$

其中 $\phi, \psi: \mathbb{R}^d \to \mathbb{R}^r$ 是可学习线性投影。

**复杂度对比**:
| 指标 | Original | Low-Rank (r=32) |
| :--- | :--- | :--- |
| 计算 | $O(S^2 \cdot 64)$ | $O(S \cdot r)$ |
| 显存 | $O(S^2)$ | $O(S \cdot r)$ |
| 参数量 | - | ~50K |

**实现类**: `LowRankHilbertBias`

```python
class LowRankHilbertBias(nn.Module):
    def __init__(self, path_dim, rank, heads):
        self.path_encoder_q = nn.Sequential(...)  # φ
        self.path_encoder_k = nn.Sequential(...)  # ψ
```

### 3. Hierarchical Hilbert Bias (分层计算)

**数学定义**:
$B_{hilbert}[i,j] = \sum_{\ell=1}^{L} b^{(\ell)}(q_i^{(\ell)}, q_j^{(\ell)})$

利用四叉树层级结构，各层独立计算。

**特点**:

- 可解释性强
- 强调层级结构

### 4. Level Bias (相对层级偏置)

**数学定义**:
$B_{level}[i,j] = \text{Embedding}(\text{clamp}(d_i - d_j + L, 0, 2L))$

**作用**: 编码跨层级关系（如父节点关注子节点）

---

## 5.3 类：HilbertAwareMultiScaleAttention

### 初始化参数

| 参数                 | 类型    | 默认值        | 说明                                   |
|:------------------ |:----- |:---------- |:------------------------------------ |
| `dim`              | int   | -          | 输入维度                                 |
| `heads`            | int   | 8          | 注意力头数                                |
| `dim_head`         | int   | 64         | 每头维度                                 |
| `dropout`          | float | 0.0        | Dropout 比率                           |
| `use_hilbert_bias` | bool  | True       | 是否使用 Hilbert 偏置                      |
| `bias_mode`        | str   | 'low_rank' | 偏置模式 (构造器默认，FractalConfig 默认为 'lca') |
| `low_rank_r`       | int   | 32         | Low-Rank 秩                           |
| `max_level`        | int   | 50         | 最大层级                                 |

> **注意**: `FractalConfig.hilbert_bias_mode` 默认为 `'lca'`，这是推荐值。
> `HilbertAwareMultiScaleAttention` 构造器默认为 `'low_rank'` 是为了向后兼容。

### forward(x, levels_info, attention_mask)

**Step 1: QKV 投影**

```python
x = self.norm(x)
qkv = self.to_qkv(x)  # (B, S, 3 * Inner_Dim)
q, k, v = qkv.chunk(3, dim=-1)
q, k, v = map(rearrange, [q, k, v], ['b s (h d) -> b h s d'] * 3)
```

**Step 2: 点积注意力**

```python
dots = torch.matmul(q, k.transpose(-1, -2))  # (B, H, S, S)
dots = dots * self.scale  # 1/√d_k
```

**Step 3: 层级缩放**

```python
if self.use_level_scaling:
    depths = extract_depths(levels_info, self.max_level)
    scale_factors = self.level_scale_embedding(depths)
    dots = dots * scale_factors
```

**Step 4: Hilbert 偏置注入**

```python
if self.use_hilbert_bias:
    hilbert_bias = self._compute_hilbert_bias(levels_info)
    dots = dots + hilbert_bias * HILBERT_BIAS_SCALE
```

**Step 5: 层级偏置**

```python
level_bias = self._compute_level_bias(levels_info)
dots = dots + level_bias * LEVEL_BIAS_SCALE
```

**Step 6: 掩码**

```python
dots.masked_fill_(~attention_mask, float('-inf'))
```

**Step 7: Softmax 与输出**

```python
attn = F.softmax(dots, dim=-1)
attn = self.dropout(attn)
out = torch.matmul(attn, v)
out = rearrange(out, 'b h s d -> b s (h d)')
return self.to_out(out)
```

---

## 5.4 bias_mode 选项对比

| 模式             | 参数量  | 复杂度              | 可解释性  | 推荐度        |
|:-------------- |:---- |:---------------- |:----- |:---------- |
| `lca`          | ~100 | $O(N^2 \cdot D)$ | ⭐⭐⭐⭐⭐ | ✅ **默认推荐** |
| `low_rank`     | ~50K | $O(N \cdot r)$   | ⭐⭐    | 大模型可选      |
| `hierarchical` | ~5K  | $O(N \cdot L)$   | ⭐⭐⭐⭐  | 可解释性场景     |

> **注意**: `original` 模式已在 P3-4 中移除，以简化代码库。

**推荐**: 对于大多数场景，使用 `lca` 模式（FractalConfig 默认值）。

---

## 5.5 使用示例

```python
from vit_pytorch import HilbertAwareMultiScaleAttention

# 推荐: 使用 LCA 模式
attn = HilbertAwareMultiScaleAttention(
    dim=384,
    heads=6,
    dim_head=64,
    bias_mode='lca',  # 推荐，参数量 ~100
)

# 备选: 使用 Low-Rank 模式
attn_lr = HilbertAwareMultiScaleAttention(
    dim=384,
    heads=6,
    dim_head=64,
    bias_mode='low_rank',
    low_rank_r=32,
)

x = torch.randn(2, 100, 384)  # (B, S, D)
levels_info = torch.zeros(2, 100, 10, dtype=torch.long)  # (B, S, Info)
mask = torch.ones(2, 1, 1, 100, dtype=torch.bool)

out = attn(x, levels_info, mask)  # (2, 100, 384)
```
