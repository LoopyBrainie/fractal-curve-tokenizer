# 第四章：位置编码

## 4.1 概述

`FractalPositionEmbedding` 使用**深度**和**四叉树路径**信息对 token 位置进行编码，提供针对变深度分词量身定制的层次位置编码。它支持**面积增强**编码（I31-3）以实现面积感知的位置表示。

---

## 4.2 数学形式化

### 4.2.1 位置编码定义

$$E_{pos}(i) = \text{Fusion}(E_{depth}(d_i) + E_{path}(p_i))$$

其中：
- $d_i \in [0, d_{max}]$：token $i$ 的深度
- $p_i = [q_1, \ldots, q_d]$：token $i$ 的四叉树路径
- $E_{depth}: \mathbb{Z} \to \mathbb{R}^D$：深度嵌入
- $E_{path}: [0,3]^{d_{max}} \to \mathbb{R}^D$：路径嵌入
- $\text{Fusion}: \mathbb{R}^D \to \mathbb{R}^D$：融合网络

### 4.2.2 深度嵌入

按深度索引的可学习嵌入：

$$E_{depth}(d) = W_{depth}[d], \quad W_{depth} \in \mathbb{R}^{(d_{max}+1) \times D}$$

### 4.2.3 路径嵌入（STAB-4 修复）

沿路径聚合象限嵌入，并进行深度归一化：

**原始（不稳定）**：

$$E_{path}(p) = \sum_{i=1}^{d} W_{quad}[L_i \cdot 4 + q_i]$$

这导致 $\|E_{path}\| \propto \sqrt{d}$，造成跨深度的方差失衡。

**归一化（STAB-4）**：

$$E_{path}(p) = \frac{1}{\sqrt{d}} \sum_{i=1}^{d} W_{quad}[L_i \cdot 4 + q_i]$$

其中：
- $W_{quad} \in \mathbb{R}^{(d_{max} \cdot 4) \times D}$：扁平化的级别-象限嵌入
- $L_i$：第 $i$ 步的级别索引
- $q_i$：第 $i$ 步的象限索引
- $\frac{1}{\sqrt{d}}$：保持跨深度恒定方差的归一化因子

**可变深度的掩码**：

只有有效的路径步骤贡献到总和：

$$\text{mask}[j] = \begin{cases} 1 & \text{if } j < d_i \\ 0 & \text{otherwise} \end{cases}$$
$$E_{path}(p) = \frac{1}{\sqrt{\sum_j \text{mask}[j]}} \sum_j \text{mask}[j] \cdot W_{quad}[j]$$

### 4.2.4 融合网络

带残差连接的双层 MLP：

$$\text{Fusion}(x) = x + \text{MLP}(x)$$

其中 $\text{MLP}(x) = W_2 \cdot \text{GELU}(W_1 \cdot x)$。

---

## 4.3 面积增强位置编码（I31-3）

`AreaEnhancedPositionEmbedding` 扩展了基础编码，添加了面积信息。

### 4.3.1 数学形式

$$E_{pos}(i) = \text{Fusion}(E_{depth}(d_i) + E_{path}(p_i) + \lambda \cdot E_{area}(R_i))$$

其中：
- $E_{area}(R_i)$：来自 `AreaEncoder` 的面积嵌入
- $\lambda$：可学习的缩放参数（初始化为零）

### 4.3.2 AreaEncoder（与注意力共享）

相同的 `AreaEncoder` 用于位置编码和注意力偏置：

**面积分数**：

$$f_{area} = \frac{\log(s_{patch} + 1)}{\log(S_{total} + 1)}$$

**傅里叶特征**：

$$\gamma(f) = [\sin(2^k \pi f), \cos(2^k \pi f)]_{k=0}^{L-1}$$

**MLP 投影**：

$$E_{area}(R) = \text{MLP}(\gamma(f_{area}))$$

### 4.3.3 残差注入

面积嵌入通过残差连接注入：

$$E_{pos} = E_{base} + \lambda \cdot E_{area}$$

$\lambda$ 初始化为 0，模型可以逐渐学习使用面积信息。

---

## 4.4 几何解释

### 4.4.1 象限编码

象限索引编码每个级别内的空间位置：

```
级别 0（根）:     级别 1:              级别 2:
┌─────────────┐     ┌──────┬──────┐       ┌───┬───┬───┬───┐
│             │     │  2   │  3   │       │ 2 │ 3 │ 2 │ 3 │
│      0      │  →  ├──────┼──────┤   →   ├───┼───┼───┼───┤
│             │     │  0   │  1   │       │ 0 │ 1 │ 0 │ 1 │
└─────────────┘     └──────┴──────┘       ├───┼───┼───┼───┤
                                          │ 2 │ 3 │ 2 │ 3 │
                                          ├───┼───┼───┼───┤
                                          │ 0 │ 1 │ 0 │ 1 │
                                          └───┴───┴───┴───┘
```

### 4.4.2 路径唯一性

每个四叉树路径唯一标识一个空间区域：

$$\text{Region}([q_1, \ldots, q_d]) = \bigcap_{i=1}^{d} \text{Quadrant}(q_i, i)$$

### 4.4.3 Hilbert 兼容性

路径编码保持 Hilbert 曲线局部性：

$$|H^{-1}(p_i) - H^{-1}(p_j)| \propto \|E_{path}(p_i) - E_{path}(p_j)\|_2$$

---

## 4.5 实现

### 类：FractalPositionEmbedding

```python
class FractalPositionEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_level: int = 8,
        max_seq_len: int = 10000,
        use_hilbert_encoding: bool = True,
        use_spatial_encoding: bool = True,
        dropout: float = 0.1,
    ):
        """
        参数:
            dim: 嵌入维度
            max_level: 最大四叉树深度（P11-2：应与 tokenizer.max_depth 匹配）
            max_seq_len: 最大序列长度
            use_hilbert_encoding: 启用 Hilbert 路径编码
            use_spatial_encoding: 启用空间编码
            dropout: 位置编码的 dropout 率（I27-2）
        """
```

### 类：AreaEnhancedPositionEmbedding

```python
class AreaEnhancedPositionEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_level: int = 8,
        fourier_levels: int = 4,
        use_hilbert_encoding: bool = True,
        use_spatial_encoding: bool = True,
        dropout: float = 0.1,
    ):
        """
        参数:
            dim: 嵌入维度
            max_level: 最大四叉树深度
            fourier_levels: 傅里叶频率级别数
            use_hilbert_encoding: 启用 Hilbert 编码
            use_spatial_encoding: 启用空间编码
            dropout: dropout 率
        """
```

### 前向传播（基础）

```python
def forward(
    self,
    levels_info: torch.Tensor,
    regions: Optional[torch.Tensor] = None,
    image_size: Optional[int] = None,
) -> torch.Tensor:
    """
    参数:
        levels_info: (B, N, max_depth+1) - [depth, q_1, q_2, ..., q_d]
        regions: (B, N, 4) - 区域边界 [x1, y1, x2, y2]
        image_size: int 或 (W, H) - 图像维度

    返回:
        position_embedding: (B, N, D)
    """
    depths = levels_info[..., 0].clamp(0, self.max_level).long()
    paths = levels_info[..., 1:].long()

    # 深度嵌入
    depth_emb = self.depth_embedding(depths)

    # 带归一化的路径嵌入（STAB-4）
    level_offsets = torch.arange(paths.shape[-1], device=levels_info.device) * 4
    flat_indices = (paths + level_offsets).clamp(0, self.max_level * 4 - 1)
    path_embs = self.quadrant_embedding(flat_indices)

    # 掩码和归一化
    seq_indices = torch.arange(paths.shape[-1], device=levels_info.device)
    mask = seq_indices < depths.unsqueeze(-1)
    path_count = mask.sum(dim=-1, keepdim=True).clamp(min=1).float()
    path_final = (path_embs * mask.unsqueeze(-1)).sum(dim=-2) / torch.sqrt(path_count)

    # 融合
    combined = depth_emb + path_final
    return self.fusion_network(combined)
```

### 前向传播（面积增强）

```python
def forward(
    self,
    levels_info: torch.Tensor,
    regions: Optional[torch.Tensor] = None,
    image_size: Optional[int] = None,
) -> torch.Tensor:
    # 基础位置编码
    pos_emb = self.base_embedding(levels_info)

    # 面积编码（如果提供了区域）
    if regions is not None and image_size is not None:
        area_emb = self.area_encoder(regions, image_size)

        # 处理 CLS token 对齐
        if area_emb.shape[1] == pos_emb.shape[1] + 1:
            area_emb = area_emb[:, 1:, :]

        # 残差注入
        pos_emb = pos_emb + self.area_scale * area_emb

    return pos_emb
```

---

## 4.6 与标准位置编码的比较

| 方法 | 编码 | 层次化 | 自适应 | 面积感知 |
|:-------|:---------|:-------------|:---------|:-----------|
| 正弦 | $\sin(pos / 10000^{2i/d})$ | 否 | 否 | 否 |
| 可学习 1D | $W[pos]$ | 否 | 否 | 否 |
| 可学习 2D | $W_x[x] + W_y[y]$ | 否 | 否 | 否 |
| RoPE | 旋转矩阵 | 否 | 否 | 否 |
| **分形** | $E_{depth}(d) + E_{path}(p)$ | 是 | 是 | 否 |
| **分形+面积** | $E_{depth} + E_{path} + \lambda \cdot E_{area}$ | 是 | 是 | 是 |

---

## 4.7 位置编码中的 Dropout（I27-2）

位置编码充当**信息瓶颈**，因此 dropout 应保持保守：

**推荐配置**：

| 参数 | 值 | 原理 |
|:----------|:------|:----------|
| `dropout` | 0.1 | ~0.5 × 主 transformer dropout |
| 最大值 | 0.2 | 更高的值有位置信息丢失的风险 |

**分析**：
- $p_{pos} > 0.2$：位置信息退化 → 模型无法学习空间关系
- $p_{pos} < 0.05$：正则化不足 → 过拟合到特定位置

**经验公式**：

$$p_{pos} \approx 0.5 \times p_{transformer}$$

---

## 4.8 使用示例

### 基本配置

```python
from vit_pytorch import FractalPositionEmbedding

pos_embedding = FractalPositionEmbedding(
    dim=384,
    max_level=8,
    use_hilbert_encoding=True,
    use_spatial_encoding=True,
    dropout=0.1,
)

# levels_info: (B, N, max_depth+1)
# 格式: [depth, q_1, q_2, ..., q_{max_depth}]
levels_info = torch.zeros(2, 100, 9, dtype=torch.long)
levels_info[:, :, 0] = 2  # 所有 token 在深度 2

tokens = torch.randn(2, 100, 384)
tokens_with_pos = tokens + pos_embedding(levels_info)
```

### 带面积增强（I31-3）

```python
from vit_pytorch import AreaEnhancedPositionEmbedding

pos_embedding = AreaEnhancedPositionEmbedding(
    dim=384,
    max_level=8,
    fourier_levels=4,
    dropout=0.1,
)

# 带区域用于面积编码
regions = torch.rand(2, 100, 4)  # [B, N, 4] - [x1, y1, x2, y2]
image_size = 224

tokens_with_pos = tokens + pos_embedding(levels_info, regions=regions, image_size=image_size)
```

---

## 4.9 注意力偏置集成

位置编码和注意力偏置共享相同的 `AreaEncoder`：

```
┌─────────────────────────────────────────────────────────────┐
│                    位置编码                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐  │
│  │  深度嵌入    │ +  │  路径嵌入    │ +  │ 面积嵌入 (opt)    │  │
│  │  [0, D]     │    │  (norm)     │    │ [I31-3]         │  │
│  └─────────────┘    └─────────────┘    └─────────────────┘  │
│                         │                                   │
│                         ▼                                   │
│                   ┌───────────┐                             │
│                   │  融合     │                              │
│                   │  网络     │                              │
│                   └───────────┘                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  Token + Position   │
                    └─────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    注意力偏置                                │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐  │
│  │  LCA 嵌入    │ +  │  级别偏置    │ +  │ 面积调制 (opt)   │  │
│  │ [B,H,N,N]   │    │ [B,H,N,N]   │    │ [I31-3]         │  │
│  └─────────────┘    └─────────────┘    └─────────────────┘  │
│                         │                                   │
│                         ▼                                   │
│                   ┌───────────┐                             │
│                   │   缩放    │                             │
│                   │   因子    │                             │
│                   └───────────┘                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 4.10 常量参考

| 常量 | 值 | 用途 |
|:---------|:------|:--------|
| `EMBEDDING_INIT_STD` | 0.02 | 嵌入权重初始化标准差 |
| `HILBERT_BIAS_SCALE` | 1.0 | 注意力偏置缩放 |
| `LEVEL_BIAS_SCALE` | 0.1 | 级别偏置缩放 |

> **下一章**: [05_attention_mechanism.md](05_attention_mechanism.md) - Hilbert 感知注意力
