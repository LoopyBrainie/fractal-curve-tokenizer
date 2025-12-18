# 第四章：位置编码 (positional.py)

本章解析 `FractalPositionEmbedding`，它解决了在分形网格中定义位置的问题。

## 4.1 数学形式化

分形位置编码结合深度和路径信息：

$$E_{pos}(i) = \text{Fusion}(E_{depth}(d_i) + E_{path}(i))$$

### 深度编码 (Depth Embedding)

$$E_{depth}: \mathbb{Z} \to \mathbb{R}^D$$
$$E_{depth}(d) = \text{Embedding}(d), \quad d \in \{0, 1, \ldots, L_{max}\}$$

### 路径编码 (Path Embedding)

#### 1. 路径生成 (Vectorized Path Generation)

将 2D 坐标 $(x, y)$ 转换为四叉树路径 $(q_1, \ldots, q_d)$。
利用向量化位运算实现高效计算：

$q_\ell = \text{bit}(x, d-\ell) + 2 \times \text{bit}(y, d-\ell)$

其中 $\text{bit}(v, k) = (v \gg k) \& 1$ 表示取第 $k$ 位。

- $q_\ell \in \{0, 1, 2, 3\}$: 0=左上, 1=右上, 2=左下, 3=右下
- 计算复杂度: $O(D)$ (并行)，优于传统循环的 $O(N \times D)$

#### 2. 嵌入查找

对 token $i$，其路径为 $(q_i^{(1)}, \ldots, q_i^{(d_i)})$。

$E_{path}(i) = \sum_{j=1}^{d_i} \text{QuadrantEmb}(j, q_i^{(j)})$

其中 `QuadrantEmb` 是 $[L_{max} \times 4, D]$ 的可学习嵌入表。

### 融合网络

$\text{Fusion}(x) = \text{Dropout}(\text{GELU}(\text{LayerNorm}(\text{Linear}(x))))$

---

## 4.2 类：FractalPositionEmbedding

### 初始化参数

| 参数                     | 类型   | 默认值   | 说明              |
|:---------------------- |:---- |:----- |:--------------- |
| `dim`                  | int  | -     | 嵌入维度            |
| `max_level`            | int  | 50    | 最大递归深度          |
| `max_seq_len`          | int  | 10000 | 最大序列长度          |
| `use_hilbert_encoding` | bool | True  | 是否使用 Hilbert 编码 |
| `use_spatial_encoding` | bool | True  | 是否使用空间编码        |

### forward(levels_info, ...)

**输入**: 

- `levels_info` 张量
  - 形状: `(N, Info_Len)` (单序列) 或 `(B, N, Info_Len)` (Batch)
  - 格式: `[depth, q_1, q_2, ..., q_depth, 0, ...]`

**流程**:

**1. 深度编码**

```python
depths = levels_info[..., 0]  # 提取深度
depth_emb = self.depth_embedding(depths)  # (B, N, D)
```

**2. 路径编码**

```python
paths = levels_info[..., 1:]  # 提取路径部分

# 坐标扁平化：区分不同层级的同一象限
offsets = torch.arange(path_len) * 4
flat_indices = paths + offsets

# 查表
path_embs = self.quadrant_embedding(flat_indices)  # (B, N, Path_Len, D)

# 掩码聚合
mask = index < depth  # 只聚合有效路径
path_final = (path_embs * mask).sum(dim=-2)  # (B, N, D)
```

**3. 特征融合**

```python
combined = depth_emb + path_final
result = self.fusion_network(combined)
```

**输出**: `result` 形状 `(B, N, D)`

---

## 4.3 辅助方法：get_attention_bias(depths)

**功能**: 计算基于层级差的注意力偏置矩阵。

**数学定义**:
$B_{level}[i,j] = \text{LevelAttnBias}[d_i, d_j]$

**逻辑**:

- 输入深度张量 `depths` `(N,)`
- **向量化实现**: 使用广播机制一次性查表
- `bias[i, j] = table[d[i], d[j]]`

**用途**: 可选地加到 Attention Logits 中，增强层级感知能力。

---

## 4.4 初始化策略

```python
def _init_parameters(self):
    nn.init.normal_(self.depth_embedding.weight, std=EMBEDDING_INIT_STD)
    nn.init.normal_(self.quadrant_embedding.weight, std=EMBEDDING_INIT_STD)
    nn.init.uniform_(self.level_attention_bias, -HILBERT_BIAS_SCALE, HILBERT_BIAS_SCALE)
```

- 嵌入使用正态分布初始化，$\sigma = 0.02$
- 注意力偏置使用均匀分布初始化

---

## 4.5 与标准位置编码的对比

| 特性       | 标准 ViT     | 分形位置编码              |
|:-------- |:---------- |:------------------- |
| **编码类型** | 1D/2D 绝对位置 | 深度 + 路径             |
| **层级感知** | 无          | 有 (depth_embedding) |
| **空间结构** | 网格坐标       | 四叉树路径               |
| **可学习性** | 部分/全部      | 全部可学习               |
| **参数量**  | O(N×D)     | O(L×4×D) + O(L×D)   |
