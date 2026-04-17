# 第五章：注意力机制

## 5.1 概述

`ManifoldNativeAttention` (MNA) 是 Fractal Curve ViT 的核心注意力机制，旨在在变分辨率分形 tokens 的非欧几里得流形上运行。

**关键创新**：与将 tokens 视为平坦序列的标准 ViT 不同，该机制明确建模从 Hilbert 曲线和四叉树结构派生的**层次和空间关系**。

**效率**：通过对稀疏、非均匀 token 集（N ≈ 32-64 vs 标准 ViT 的 307K）进行操作，实现约 40× 的注意力矩阵复杂度 reduction。

### 5.1.1 架构

```
X_{l+1} = X_l + Attn(X_l) + FractalResidual(X_l)

Attn = BandedAttention(QK + B_manifold)
B_manifold = GeometricLatentDecoder(ξ)
```

### 5.1.2 复杂度分析

| 指标 | 标准 ViT | Fractal ViT |
|:-------|:-------------|:------------|
| 序列长度 (N) | ~196 | ~32-64 |
| 注意力矩阵 | $O(N^2)$ | $O(N \cdot W)$ |
| **有效减少** | - | **~40×** |

> **注意**：复杂度保持 $O(N^2 \cdot D)$。效率增益来自于 token 数量减少，而非渐近复杂度变化。

---

## 5.2 几何潜在解码器（ξ 空间）

`GeometricLatentDecoder` 将 tokens 从离散四叉树路径映射到连续的**5 维几何特征空间（ξ 空间）**。

### 5.2.1 坐标重建

Tokens 使用向量化位运算从其 `LevelsInfo` 路径重建为 2D 坐标：

$$x = \sum_{k=0}^{d-1} \text{bit}_k(q_k, 0) \cdot 2^{max\_level-k-1}$$

$$y = \sum_{k=0}^{d-1} \text{bit}_k(q_k, 1) \cdot 2^{max\_level-k-1}$$

这避免了传统四叉树遍历的 $O(N \cdot D)$ 串行开销。

### 5.2.2 统一几何特征向量 (ξ_ij)

Token $i$ 和 $j$ 之间的特征向量 $\xi_{ij}$ 由以下部分组成：

| 组件 | 公式 | 描述 |
|:----------|:--------|:------------|
| **归一化 Hilbert 距离** | $\Delta h_{ij} / N^2$ | Hilbert 索引差异 |
| **LCA 深度偏置** | $2^{d_{LCA}}$ | 最近公共祖先深度 |
| **欧几里得距离** | $\|pos_i - pos_j\|_2$ | 物理 2D 距离 |
| **面积比** | $area_i / area_j$ | 分块大小关系 |
| **旋转相似性** | $1$ 如果父节点共享象限，否则 $0$ | 空间方向 |

---

## 5.3 Poincaré 圆盘距离

为处理分形 tokens 的多尺度性质，模型在**Poincaré 圆盘**上计算距离，这是双曲几何的一个模型。

### 5.3.1 数学形式化

2D 坐标映射到单位圆盘：

$$u = \tanh(r/2) \cdot \frac{x - c}{\|x - c\|}$$

双曲距离为：

$$d_H(u, v) = \text{acosh}\left(1 + \frac{2\|u-v\|^2}{(1-\|u\|^2)(1-\|v\|^2)}\right)$$

### 5.3.2 数值稳定性 (I-NAN)

为防止梯度爆炸和 NaN 值：

| 保护 | 方法 |
|:-----------|:-------|
| **FP32 强制** | 内部计算使用 FP32 以防止 FP16 下溢 |
| **Epsilon 保护** | `acosh(x)` 需要 $x \geq 1 + \epsilon$ |
| **输出钳位** | 距离钳位到 $[0, 10.0]$ |

---

## 5.4 层次注意力偏置 (LCA)

### 5.4.1 LCA 矩阵计算

注意力分数通过从 token 对的**最近公共祖先（LCA）**派生的偏置 $B_{hilbert}$ 进行增强：

$$B[i,j] = \text{LCAEmbed}(\text{LCA}(i, j))$$

`get_lca_matrix` 函数计算每个 token 对的共享祖先深度。位于相同四叉树分支中的 tokens 接收更高的注意力偏置。

### 5.4.2 偏置缩放常量

| 常量 | 值 | 用途 |
|:---------|:------|:--------|
| `HILBERT_BIAS_SCALE` | 1.0 | 基于 LCA 的空间偏置缩放 |
| `LEVEL_BIAS_SCALE` | 1.0 | 相对级别偏置缩放 |

缩放的注意力公式：

$$\text{scores} = \frac{QK^T}{\sqrt{d_k}} \cdot \sigma_{scale} + HILBERT\_BIAS\_SCALE \cdot B_{hilbert} + LEVEL\_BIAS\_SCALE \cdot B_{level}$$

> **注意 (I122-2)**：原始的 τ_h 温度参数已被移除。偏置强度现在由 `HILBERT_BIAS_SCALE × √d_k` 控制。

---

## 5.5 级别偏置

### 5.5.1 相对级别嵌入

编码不同尺度 tokens 之间的关系：

$$B_{level}[i,j] = W_{rel}[\text{clamp}(d_i - d_j + L, 0, 2L)]$$

其中：
- $d_i, d_j$：token $i, j$ 的深度
- $L$：最大相对深度范围
- $W_{rel}$：大小为 $(2L+1) \times H$ 的嵌入表

### 5.5.2 级别缩放

根据查询深度缩放注意力 logit：

$$\sigma_{scale}(d) = \text{Softplus}(\text{LevelScaleEmb}(d))$$

---

## 5.6 层次软硬注意力 (I160-2)

> **推荐**：与 `DeterministicNeighborSplitter` 结合使用以获得硬局部性保证

### 5.6.1 三区域分区

| 区域 | LCA 深度条件 | 注意力行为 |
|:-------|:-------------------|:-------------------|
| **HARD_ZERO** | $\ell < \ell_{min}$ | 强制排除（mask = 0） |
| **SOFT_POSITIVE** | $\ell_{min} \leq \ell < \ell_{soft}$ | 软偏置鼓励 |
| **HARD_ONE** | $\ell \geq \ell_{soft}$ | 完全鼓励（mask = 1） |

默认值：$\ell_{min} = 1$（象限边界），$\ell_{soft} = 2$（子象限边界）

### 5.6.2 数学形式化

**层次掩码**：

$$M(\ell) = \begin{cases} 0 & \text{if } \ell < \ell_{min} \\ \sigma(\ell - \ell_{soft}/2) & \text{if } \ell_{min} \leq \ell < \ell_{soft} \\ 1 & \text{if } \ell \geq \ell_{soft} \end{cases}$$

**注意力公式**：

$$\tilde{A}_{ij} = \frac{QK^T}{\sqrt{d_k}}[i,j] + \alpha(\ell_{ij}) \cdot B_{hilbert}[i,j]$$

---

## 5.7 仿射调制偏置 (I31-3)

**AffineModulatedBias** 通过面积感知调制增强空间注意力。

### 5.7.1 面积编码（NeRF 风格傅里叶特征）

$$f_{area} = \frac{\log(s_{patch} + 1)}{\log(S_{total} + 1)}$$

$$\gamma(f) = [\sin(2^k \pi f), \cos(2^k \pi f)]_{k=0}^{L-1}$$

### 5.7.2 仿射调制

$$B_{final} = \gamma(s_i, s_j) \odot B_{spatial} + \beta(s_i, s_j)$$

---

## 5.8 形状-尺度偏置 (I31)

**ShapeScaleEncoder** 捕获区域几何以增强注意力偏置。

### 5.8.1 长宽比

$$r = \log(w/h)$$

### 5.8.2 门控组合

$$g = \sigma(\text{MLP}([r; s]))$$

$$E_{shape}(R) = \text{MLP}([r \cdot g; s \cdot (1-g)])$$

---

## 5.9 尺度感知残差（分形残差）

**ScaleAwareResidual** 通过学习的残差连接实现父到子的信息流。

### 5.9.1 数学形式化

$$X_{l+1}^{(child)} = X_l^{(parent)} \cdot \sigma(g_d) + \text{Attn}(X_l)$$

其中 $\sigma(g_d) = \text{sigmoid}(\text{Linear}(d))$ 是基于深度的可学习门。

### 5.9.2 关键属性

| 门值 | 行为 |
|:-----------|:---------|
| 接近 1 | 子特征由父投影主导（信息向上流动） |
| 接近 0 | 子特征是独立的 |

---

## 5.10 笛卡尔 2D RoPE (I167-4)

**Cartesian2DRoPE** 基于物理 2D 坐标 $(x, y)$ 实现旋转位置嵌入。

### 5.10.1 数学形式化

给定位置 $i$ 和物理坐标 $p_i = (x_i, y_i)$：

$$\theta_i = \text{atan2}(y_i, x_i)$$

**RoPE 应用**：

$$\text{RoPE}(q_i, k_i) = \begin{pmatrix} \cos(\theta_i / 2) & -\sin(\theta_i / 2) \\ \sin(\theta_i / 2) & \cos(\theta_i / 2) \end{pmatrix} \begin{pmatrix} q_i^{(0)} \\ q_i^{(1)} \end{pmatrix}$$

### 5.10.2 实现

```python
class Cartesian2DRoPE(nn.Module):
    def __init__(self, dim: int, max_level: int = 8):
        self.dim = dim
        self.max_level = max_level

    def forward(self, q: Tensor, paths: Tensor, depths: Tensor) -> Tensor:
        coords = coords_from_paths(paths, depths, self.max_level)
        angles = torch.atan2(coords[..., 1], coords[..., 0])
        cos_angle, sin_angle = torch.cos(angles), torch.sin(angles)
        # Apply rotation...
```

---

## 5.11 诊断和监控

### 5.11.1 CLSAttentionTracker

`CLSAttentionTracker` 监控从 `[CLS]` token 到不同四叉树深度的注意力流。

**链接断开检测**：当 `[CLS]` 失去与全局上下文的联系（Depth 0 注意力 < 20%）时，跟踪器发出信号以调整 `HILBERT_BIAS_SCALE`。

### 5.11.2 get_stats() 诊断

`get_stats()` 方法返回实时指标：

| 指标 | 描述 |
|:-------|:------------|
| `manifold_bias_std` | 几何偏置的离散度 |
| `poincare_dist_mean` | tokens 之间的平均双曲距离 |
| `effective_rank` | 注意力矩阵的 SVD 基于秩（检测模式崩溃） |

---

## 5.12 实现

### 类：ManifoldNativeAttention

```python
class ManifoldNativeAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        max_level: int = 8,
        use_cartesian_rope: bool = True,   # I167-4
        use_scale_aware_residual: bool = True,
        use_poincare_distance: bool = True,
        band_width: int = 32,
    ):
        """
        参数:
            dim: 输入维度
            heads: 注意力头数
            dim_head: 每头维度
            dropout: dropout 率
            max_level: 最大四叉树深度
            use_cartesian_rope: 启用笛卡尔 2D RoPE
            use_scale_aware_residual: 启用分形残差
            use_poincare_distance: 启用 Poincaré 圆盘距离
            band_width: Hilbert 带状注意力的带宽
        """
```

### 类：LCAHilbertBias

```python
class LCAHilbertBias(nn.Module):
    def __init__(
        self,
        max_depth: int,
        heads: int,
        lca_temperature: Optional[float] = None,  # I122-2 中已移除
        learnable_temperature: bool = False,      # I122-2 中已移除
    ):
        # 注意：lca_temperature 在 I122-2 中已移除
        # 偏置强度现在由 hilbert_bias_scale × √d_k 控制
```

---

## 5.13 偏置模式比较

| 模式 | 参数 | 复杂度 | 几何意义 |
|:-----|:-----------|:-----------|:------------------|
| `lca` | ~100 | $O(N^2)$ | 显式（LCA 深度） |
| `affine_modulation` | ~1K | $O(N^2)$ | 面积感知空间偏置 |
| `shape_scale` | ~1K | $O(N^2)$ | 几何感知偏置 |
| `hierarchical_soft_hard` | ~100 | $O(N^2)$ | 硬局部性保证 |

**建议**：为效率使用 `lca`，为最佳 Hilbert 局部性使用 `hierarchical_soft_hard` 配合 `DeterministicNeighborSplitter`。

---

## 5.14 缓存优化 (I30-9)

为效率，LCA 深度计算在 transformer 层之间缓存：

**缓存键**：`data_ptr` + `torch_version`

| 好处 | 描述 |
|:---------|:------------|
| LCA 计算 | 消除每层的冗余计算 |
| 加速 | 6 层 transformer 约 6× 加速 |
| 失效 | 张量修改时自动失效 |

```python
# 缓存自动管理
lca_bias = LCAHilbertBias(max_depth=8, heads=8)

# 如需手动清除缓存
lca_bias.clear_cache()
```

---

## 5.15 文档导航

| 章节 | 内容 |
|:--------|:--------|
| [05_attention_mechanism](05_attention_mechanism.md) | 流形本地注意力（本章） |
| [06_feedforward_network](06_feedforward_network.md) | 前馈网络 |
| [07_transformer_encoder](07_transformer_encoder.md) | Transformer 块 |

> **下一章**: [06_feedforward_network.md](06_feedforward_network.md) - 前馈网络
