# 第五章：注意力机制

## 5.1 概述

`HilbertAwareMultiScaleAttention` 扩展了标准多头注意力，添加了**Hilbert 曲线派生的注意力偏置**，显式编码空间邻近性和层次关系。它支持**仿射调制**（I31-3）以实现面积感知的注意力偏置。

**复杂度说明**：注意力复杂度保持 $O(N^2 \cdot D)$。~40× 的效率增益来自于 token 数量减少（$N \approx 32$ vs 307K），而非渐近复杂度变化。

**温度选择**：默认 $\tau_h \approx 1.5$ 的选择是为了：
1. 提供有意义的偏置幅度（不会太小而被忽略）
2. 允许梯度通过 Softplus 参数化流动
3. 平衡空间局部性先验强度

---

## 5.2 数学形式化

### 5.2.1 标准多头注意力

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

### 5.2.2 Hilbert 感知注意力

$$\text{HilbertAttn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} \cdot \sigma_{scale} + \alpha_h \cdot B_{hilbert} + \alpha_l \cdot B_{level}\right) V$$

其中：
- $\sigma_{scale}$：可学习的深度相关缩放因子
- $\alpha_h$：Hilbert 偏置缩放常量（`HILBERT_BIAS_SCALE`）
- $\alpha_l$：级别偏置缩放常量（`LEVEL_BIAS_SCALE`）
- $B_{hilbert}$：基于 LCA 的 Hilbert 曲线偏置
- $B_{level}$：相对级别深度偏置

### 5.2.3 偏置缩放常量

为确保适当的梯度幅度，偏置项按常量缩放：

| 常量 | 值 | 用途 |
|:---------|:------|:--------|
| `HILBERT_BIAS_SCALE` | 1.0 | 空间偏置的 LCA 缩放 |
| `LEVEL_BIAS_SCALE` | 0.1 | 相对级别偏置的缩放 |

缩放的注意力公式：

$$\text{scores} = \frac{QK^T}{\sqrt{d_k}} \cdot \sigma_{scale} + HILBERT\_BIAS\_SCALE \cdot B_{hilbert} + LEVEL\_BIAS\_SCALE \cdot B_{level}$$

---

## 5.3 Hilbert 偏置模式

### 5.3.1 LCA Hilbert 偏置（标准）

该偏置使用四叉树结构中两个 token 的最近公共祖先（LCA）来编码树距离。

**数学定义**：

$$B[i,j] = \tau_h \cdot \text{LCAEmbed}(\text{LCA}(i, j))$$

其中：
- $\text{LCA}(i, j) \in \{0, \dots, d_{max}\}$：包含 $R_i$ 和 $R_j$ 的最小四叉树区域的深度
- $\text{LCAEmbed}: \mathbb{Z} \to \mathbb{R}^H$：可学习嵌入表
- $\tau_h \in \mathbb{R}^H$：每头温度参数

**温度参数（$\tau_h$）**：

为确保偏置强度为正且自适应，我们使用 Softplus 参数化：

$$\tau_h = \text{Softplus}(\gamma_h)$$

初始化使得 $\tau \approx 1.5$，增强空间局部性的先验。

**P11-3：基于区域的 LCA 计算**：

为准确的 LCA，直接从区域边界计算：

$$\text{Path}(R) = \text{bit}(cx, D-d) + 2 \cdot \text{bit}(cy, D-d)$$
$$\text{LCA}(i, j) = \text{Length}(\text{CommonPrefix}(\text{Path}(i), \text{Path}(j)))$$

> **注意**：这用直接的几何计算取代了脆弱的索引运算。

---

## 5.4 级别偏置

### 5.4.1 相对级别嵌入

编码不同尺度 token 之间的关系（例如，父子 vs 同级）。

$$B_{level}[i,j] = W_{rel}[\text{clamp}(d_i - d_j + L, 0, 2L)]$$

其中：
- $d_i, d_j$：token $i, j$ 的深度
- $L$：最大相对深度范围
- $W_{rel}$：大小为 $(2L+1) \times H$ 的嵌入表

### 5.4.2 级别缩放

根据查询的深度缩放注意力 logit，以稳定跨尺度的训练。

$$\sigma_{scale}(d) = \text{Softplus}(\text{LevelScaleEmb}(d))$$

较深的 token（更细的分辨率）通常学习较小的缩放因子以拓宽其注意力范围，反之亦然。

---

## 5.5 仿射调制偏置（I31-3）

**AffineModulatedBias** 通过面积感知调制增强空间注意力，使模型能够学习依赖于大小的注意力模式。

### 5.5.1 数学形式

**面积编码**（NeRF 风格傅里叶特征）：

$$f_{area} = \frac{\log(s_{patch} + 1)}{\log(S_{total} + 1)}$$
$$\gamma(f) = [\sin(2^k \pi f), \cos(2^k \pi f)]_{k=0}^{L-1}$$

其中：
- $s_{patch}$：patch 面积
- $S_{total}$：总图像面积
- $L$：傅里叶级别数

**仿射调制**：

$$B_{\text{final}} = \gamma(s_i, s_j) \odot B_{\text{spatial}} + \beta(s_i, s_j)$$

其中：
- $\gamma(s_i, s_j) = \sigma(\text{MLP}_\gamma(p_s))$：可学习缩放因子
- $\beta(s_i, s_j) = \text{MLP}_\beta(p_s)$：可学习偏置因子
- $p_s = \text{area\_emb}[i] \cdot \text{area\_emb}[j]$：面积相似性

**残差连接**：

$$B_{\text{combined}} = B_{\text{spatial}} + \alpha \cdot (B_{\text{final}} - B_{\text{spatial}})$$

其中 $\alpha$ 是一个可学习的零初始化参数，用于渐进激活。

### 5.5.2 AreaEncoder 架构

```
输入: regions [B, N, 4]
   │
   ▼
┌─────────────────────┐
│ 归一化面积           │
│ f = log(area+1) /   │
│     log(total+1)    │
└─────────────────────┘
   │
   ▼
┌─────────────────────┐
│ 傅里叶特征           │
│ [sin(2^kπf),        │
│  cos(2^kπf)]_k      │
│  → 2L 维度          │
└─────────────────────┘
   │
   ▼
┌─────────────────────┐
│ MLP 投影            │
│ Linear(2L) → hidden │
│ → Linear(hidden) →  │
│   Linear(hidden) →  │
│   dim               │
└─────────────────────┘
   │
   ▼
输出: area_emb [B, N, dim]
```

### 5.5.3 AffineModulatedBias 架构

```
regions [B, N, 4]
      │
      ├──────────────────┐
      ▼                  ▼
┌─────────────┐   ┌─────────────┐
│   LCA       │   │   面积      │
│   嵌入      │   │   编码器    │
└─────────────┘   └─────────────┘
      │                  │
      ▼                  ▼
┌─────────────┐   ┌─────────────┐
│   空间      │   │   面积      │
│   偏置      │   │   相似性    │
│ [B,dim,N,N] │   │   矩阵      │
└─────────────┘   └─────────────┘
      │                  │
      └────────┬─────────┘
               ▼
        ┌─────────────┐
        │   仿射      │
        │   调制     │
        │ γ·B + β     │
        └─────────────┘
               │
               ▼
        ┌─────────────┐
        │   残差      │
        │ B + α(·-B)  │
        └─────────────┘
               │
               ▼
输出: bias [B, dim, N, N]
```

---

## 5.6 形状-尺度偏置（I31）

**ShapeScaleEncoder** 捕获区域几何以增强注意力偏置。

### 5.6.1 数学形式

**长宽比**（对数变换以保证对称性）：

$$r = \log(w/h)$$

**归一化面积**：

$$s = \frac{w \cdot W_{patch}}{W_{total} \cdot H_{total}}$$

**门控组合**：

$$g = \sigma(\text{MLP}([r; s]))$$
$$E_{shape}(R) = \text{MLP}([r \cdot g; s \cdot (1-g)])$$

**形状-尺度相似性矩阵**：

$$B_{shape}[i,j] = E_{shape}(R_i) \cdot E_{shape}(R_j)^T$$

### 5.6.2 组合偏置

$$B_{final} = B_{LCA} + \tau \cdot B_{shape}$$

其中 $\tau$ 是可学习的零初始化权重。

---

## 5.7 偏置模式比较

| 模式 | 参数 | 复杂度 | 几何意义 |
|:-----|:-----------|:-----------|:------------------|
| `lca` | ~100 | $O(N^2)$ | 显式（LCA 深度） |
| `affine_modulation` | ~1K | $O(N^2)$ | 面积感知空间偏置 |
| `shape_scale` | ~1K | $O(N^2)$ | 几何感知偏置 |

**建议**：为效率使用 `lca` 模式，为在大小变化的数据集上提高性能使用 `affine_modulation`。

---

## 5.8 实现

### 类：HilbertAwareMultiScaleAttention

```python
class HilbertAwareMultiScaleAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        max_level: int = 8,
        use_hilbert_bias: bool = True,
        use_level_scaling: bool = True,
        lca_temperature: Optional[float] = 1.5,
        learnable_temperature: bool = True,
        use_affine_modulation: bool = False,
        fourier_levels: int = 4,
    ):
        """
        参数:
            dim: 输入维度
            heads: 注意力头数
            dim_head: 每头维度
            dropout: dropout 率
            max_level: 最大四叉树深度
            use_hilbert_bias: 启用基于 LCA 的 Hilbert 偏置
            use_level_scaling: 启用深度相关缩放
            lca_temperature: LCA 偏置的初始温度
            learnable_temperature: 温度是否可学习
            use_affine_modulation: 启用 I31-3 面积感知偏置
            fourier_levels: 傅里叶频率级别数
        """
```

### 类：LCAHilbertBias

```python
class LCAHilbertBias(HilbertBiasBase):
    def __init__(
        self,
        max_depth: int,
        heads: int,
        lca_temperature: Optional[float] = 1.5,
        learnable_temperature: bool = True,
    ):
        """
        使用 P6-2 可学习温度和 Softplus 参数化。
        支持 levels_info 和基于区域的 LCA 计算。
        """
```

### 类：AffineModulatedBias

```python
class AffineModulatedBias(nn.Module):
    def __init__(
        self,
        dim: int,
        max_depth: int,
        enable_area_modulation: bool = True,
        fourier_levels: int = 4,
    ):
        """
        参数:
            dim: 注意力维度
            max_depth: 最大四叉树深度
            enable_area_modulation: 启用面积感知调制
            fourier_levels: 傅里叶频率级别数
        """
```

---

## 5.9 使用示例

### 基本配置

```python
from vit_pytorch import HilbertAwareMultiScaleAttention

attn = HilbertAwareMultiScaleAttention(
    dim=384,
    heads=6,
    dim_head=64,
    max_level=8,
    use_hilbert_bias=True,
    use_level_scaling=True,
    lca_temperature=1.5,
    learnable_temperature=True,
)
```

### 带仿射调制（I31-3）

```python
attn = HilbertAwareMultiScaleAttention(
    dim=384,
    heads=6,
    max_level=8,
    use_affine_modulation=True,
    fourier_levels=4,
)

# 前向传播时，提供 regions 和 image_size
x = torch.randn(2, 100, 384)
regions = torch.zeros(2, 100, 4)  # 区域边界
image_size = 224

out = attn(x, regions=regions, image_size=image_size)
```

### 直接从区域计算 LCA

```python
from vit_pytorch.attn_hilbert_bias import LCAHilbertBias

lca_bias = LCAHilbertBias(
    max_depth=8,
    heads=6,
    lca_temperature=1.5,
)

# 直接从区域计算偏置（P11-3 推荐）
regions = torch.randn(2, 50, 4)  # [B, N, 4]
image_size = 224

bias = lca_bias.forward_from_regions(regions, image_size)
# 输出: [B, H, N, N]
```

---

## 5.10 缓存优化（I30-9）

为效率，LCA 深度计算在 transformer 层之间缓存：

**缓存键**：`data_ptr` + `torch_version`

**好处**：
- 消除每层的冗余 LCA 计算
- 6 层 transformer 约 6 倍加速
- 张量修改时自动失效

```python
# 缓存自动管理
lca_bias = LCAHilbertBias(max_depth=8, heads=6)

# 如需手动清除缓存
lca_bias.clear_cache()
```

---

## 5.11 数值稳定性

### 方差 epsilon

层归一化使用 $\epsilon = 10^{-5}$ 以保证方差稳定性：

$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

### Softplus 温度

温度参数化确保正值：

$$\tau_h = \text{Softplus}(\gamma_h) = \log(1 + e^{\gamma_h})$$

> **下一章**: [06_feedforward_network.md](06_feedforward_network.md) - 前馈网络
