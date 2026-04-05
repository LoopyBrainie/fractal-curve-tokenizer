# 第七章：Transformer 编码器

## 7.1 概述

`FractalTransformer` 堆叠多个 `FractalTransformerBlock` 层，具有**级别感知归一化**、**Hilbert 感知注意力**和**级别聚合**。它具有**深度感知残差门控**（STAB-5）以实现自适应信息流。

---

## 7.2 数学形式化

### 7.2.1 Transformer 块

$$x' = x + \text{DropPath}(\text{Attn}(\text{LN}_1(x)))$$
$$x'' = x' + \text{DropPath}(\text{FFN}(\text{LN}_2(x')))$$

### 7.2.2 层归一化

实现对注意力和 FFN 子层都使用标准 `nn.LayerNorm`（I106-2）：

$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sigma} + \beta$$

Hilbert 偏置处理跨不同 token 深度的尺度校准，消除了对深度相关归一化参数的需求。

### 7.2.3 残差门控（STAB-5, I34-10 / Task 3 重构）

残差门控使用 **sigmoid** 激活调制注意力和 FFN 输出的贡献：

$$g_{raw} \in \mathbb{R}, \quad g = \sigma(g_{raw}) \in [0, 1]$$
$$x' = x + g \cdot \text{DropPath}(\text{Attn}(\text{LN}(x)))$$
$$x'' = x' + g \cdot \text{DropPath}(\text{FFN}(\text{LN}(x')))$$

其中：
- 门控范围：$[0, 1]$（sigmoid 确保有界贡献）
- $g$ 直接缩放注意力/FFN 输出

**原理（Task 3 重构）**：直接使用 $\sigma$ 而非 $\tanh$ 提供更清晰的梯度流，值在 $[0, 1]$ 而非 $[-1, 1]$，且直接缩放 $g \cdot \text{Attn}$ 确保梯度始终为正且表现良好。

**V 形门控模式（I24-6）**：

基于训练观察：
- $d=0$（全局）：$g \approx 0.7$ - 抑制全局信息
- $d=1$：$g \approx 1.0$ - 保持原样
- $d=2$：$g \approx 0.85$ - 轻微抑制
- $d=3$（细粒度）：$g \approx 0.65$ - 抑制细节

**初始化**：

使用以零为中心的小随机初始化（std=0.01）：

$$g = \sigma(\mathcal{N}(0, 0.01)) \approx 0.5$$

这确保初始门控值约为 0.5，允许早期训练中的梯度流。

### 7.2.4 级别聚合

$$s_d = \sigma(\text{Embed}_{level}(d)) \in (0, 1)^D$$
$$r = W_2 \cdot \text{ReLU}(W_1 \cdot x)$$
$$x' = x + \lambda \cdot (r \odot s_d)$$

其中：
- $\lambda$：可学习缩放因子（初始化为 0.2）
- $\odot$：逐元素乘法

---

## 7.3 FractalTransformer

### 类定义

```python
class FractalTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int = 8,
        dim_head: int = 64,
        mlp_dim: int = None,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        max_level: int = 8,
        drop_path_rate: float = 0.1,
        ffn_type: str = 'swiglu_level',
        use_checkpoint: bool = False,
        lca_temperature: float = 1.5,
        learnable_temperature: bool = True,
        use_affine_modulation: bool = False,
        fourier_levels: int = 4,
    ):
        """
        参数:
            dim: 模型维度
            depth: Transformer 层数
            heads: 注意力头数
            dim_head: 每头维度
            mlp_dim: MLP 隐藏维度
            dropout: dropout 率
            drop_path: DropPath 率
            max_level: 最大四叉树深度（P11-2：应与 tokenizer.max_depth 匹配）
            drop_path_rate: 最大 DropPath 率（线性增加）
            ffn_type: FFN 类型（'gelu', 'swiglu', 'swiglu_level'）
            use_checkpoint: 梯度检查点以提高内存效率
            lca_temperature: LCA 偏置温度
            learnable_temperature: 温度是否可学习
            use_affine_modulation: 启用 I31-3 面积感知偏置
            fourier_levels: 面积编码的傅里叶频率级别数
        """
```

### 参数

| 参数 | 类型 | 默认值 | 描述 |
|:----------|:-----|:--------|:------------|
| `dim` | int | - | 模型维度 |
| `depth` | int | - | Transformer 层数 |
| `heads` | int | 8 | 注意力头数 |
| `dim_head` | int | 64 | 每头维度 |
| `mlp_dim` | int | dim × 4 | FFN 隐藏维度 |
| `dropout` | float | 0.0 | dropout 率 |
| `drop_path` | float | 0.0 | DropPath 率 |
| `max_level` | int | 8 | 最大四叉树深度 |
| `ffn_type` | str | 'swiglu_level' | FFN 类型 |
| `use_checkpoint` | bool | False | 梯度检查点 |
| `lca_temperature` | float | 1.5 | LCA 偏置温度 |
| `learnable_temperature` | bool | True | 可学习温度 |
| `use_affine_modulation` | bool | False | 面积感知偏置（I31-3） |
| `fourier_levels` | int | 4 | 傅里叶频率级别数 |

---

## 7.4 FractalTransformerBlock

### 块结构

```
输入 x [B, N, D]
      │
      ▼
┌─────────────────────────────────┐
│  级别感知 LayerNorm 1           │
│  γ_1(d), β_1(d) 每深度          │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│ HilbertAwareMultiScaleAttention │
│ + LCA 偏置                      │
│ + 级别偏置                      │
│ + 仿射调制（可选）              │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│   残差门控（STAB-5）             │
│   w_1(d) = 2·σ(g_1[d]) ∈ [0,2] │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│   DropPath                      │
│   x = x + w_1 · drop(attn)      │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│  级别感知 LayerNorm 2           │
│  γ_2(d), β_2(d) 每深度          │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│ AdaptiveFractalFeedForward      │
│ (SwiGLU + 级别自适应)           │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│   残差门控（STAB-5）             │
│   w_2(d) = 2·σ(g_2[d]) ∈ [0,2] │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│   DropPath                      │
│   x = x + w_2 · drop(ffn)       │
└─────────────────────────────────┘
      │
      ▼
输出 x [B, N, D]
```

### 实现

```python
class FractalTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        max_level: int = 8,
        drop_path: float = 0.0,
        ffn_type: FFNType = 'swiglu_level',
        lca_temperature: Optional[float] = 1.5,
        learnable_temperature: bool = True,
        use_affine_modulation: bool = False,
        fourier_levels: int = 4,
    ):
        super().__init__()
        self.dim = dim
        self.max_level = max_level

        # 注意力
        self.attention = HilbertAwareMultiScaleAttention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            max_level=max_level,
            lca_temperature=lca_temperature,
            learnable_temperature=learnable_temperature,
            use_affine_modulation=use_affine_modulation,
            fourier_levels=fourier_levels,
        )

        # FFN
        self.ff = AdaptiveFractalFeedForward(
            dim=dim,
            hidden_dim=mlp_dim,
            dropout=dropout,
            max_level=max_level,
            ffn_type=ffn_type,
        )

        # STAB-5：残差门控（级别感知）
        self._residual_gate = nn.Embedding(max_level + 1, 2)

        # 级别感知 LayerNorm
        self.norm1_gamma = nn.Embedding(max_level + 1, dim)
        self.norm1_beta = nn.Embedding(max_level + 1, dim)
        self.norm2_gamma = nn.Embedding(max_level + 1, dim)
        self.norm2_beta = nn.Embedding(max_level + 1, dim)

        # DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
```

### 前向传播

```python
def forward(
    self,
    x: torch.Tensor,
    levels_info: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    regions: Optional[torch.Tensor] = None,
    image_size: Optional[int] = None,
) -> torch.Tensor:
    """
    参数:
        x: (B, N, D) - 输入序列
        levels_info: (B, N, max_depth+1) - 级别信息
        attention_mask: (B, 1, 1, N) - 注意力掩码
        regions: (B, N, 4) - 用于 LCA 计算的区域边界
        image_size: int - 用于基于区域的 LCA 的图像大小

    返回:
        x: (B, N, D) - 编码序列
    """
    # 提取深度以进行级别感知操作
    if levels_info is not None and levels_info.numel() > 0:
        depths = extract_depths(levels_info, self.max_level)
        gate_raw = self._residual_gate(depths)
        gate = torch.sigmoid(gate_raw) * 2  # [B, N, 2] ∈ [0, 2]
        w1 = gate[:, :, 0].unsqueeze(-1)
        w2 = gate[:, :, 1].unsqueeze(-1)
    else:
        # CLS token 的默认门控
        default_gate = torch.sigmoid(self._residual_gate.weight[0]) * 2
        w1 = default_gate[0]
        w2 = default_gate[1]

    # 级别感知 LayerNorm 1
    norm1_x = self._apply_level_aware_norm(
        x, levels_info, self.norm1_gamma, self.norm1_beta, self.default_norm1
    )

    # 注意力
    attn_out = self.attention(
        norm1_x,
        levels_info=levels_info,
        attention_mask=attention_mask,
        regions=regions,
        image_size=image_size,
    )

    # 带门控的残差
    x = x + self.drop_path(attn_out * w1)

    # 级别感知 LayerNorm 2
    norm2_x = self._apply_level_aware_norm(
        x, levels_info, self.norm2_gamma, self.norm2_beta, self.default_norm2
    )

    # FFN
    ff_out = self.ff(norm2_x, levels_info)

    # 带门控的残差
    x = x + self.drop_path(ff_out * w2)

    return x
```

---

## 7.5 级别感知层归一化

### 实现

```python
def _apply_level_aware_norm(
    self,
    x: torch.Tensor,
    levels_info: Optional[torch.Tensor],
    gamma_emb: nn.Embedding,
    beta_emb: nn.Embedding,
    default_norm: nn.LayerNorm,
) -> torch.Tensor:
    """
    应用深度相关层归一化。

    每个深度级别有自己的缩放（γ）和移位（β）参数。

    P11-16：当 levels_info 为 None 时，使用深度 0 作为默认值。
    """
    if x.dim() != 3:
        raise ValueError(f"期望 x 为 3D [B, S, D]，得到 {x.dim()}D")

    # 处理 None levels_info（P11-16）
    if levels_info is None or levels_info.numel() == 0:
        depths = torch.zeros(x.shape[0], x.shape[1], dtype=torch.long, device=x.device)
    elif levels_info.dim() == 2:
        depths = extract_depths(levels_info, self.max_level)
        gamma = gamma_emb(depths).unsqueeze(0)
        beta = beta_emb(depths).unsqueeze(0)
    else:
        depths = extract_depths(levels_info, self.max_level)
        gamma = gamma_emb(depths)
        beta = beta_emb(depths)

    # 手动 LayerNorm
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (x - mean) / torch.sqrt(var + 1e-5)

    return x_norm * gamma + beta
```

### 目的

不同分辨率 token（深度）具有不同的统计特性：
- **浅层 token**（大区域）：更高的方差，更多全局信息
- **深层 token**（小区域）：更低的方差，更多局部细节

级别感知归一化允许模型学习深度特定的变换。

---

## 7.6 DropPath（随机深度）

### 数学定义

**训练**：

$$\text{DropPath}(x) = \begin{cases} 0 & \text{以概率 } p \\ \frac{x}{1-p} & \text{否则} \end{cases}$$

**推理**：

$$\text{DropPath}(x) = x$$

### 目的

1. 通过随机丢弃层进行正则化
2. 减少训练时的有效网络深度
3. 作为不同深度网络集合的隐式集成
4. 线性增加的丢弃率：$p_l = \frac{l}{L} \cdot p_{max}$

### 实现

```python
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)

        if self.scale_by_keep:
            random_tensor.div_(keep_prob)

        return x * random_tensor
```

---

## 7.7 级别聚合

在层堆叠之后，学习聚合器跨深度组合信息：

### 数学形式

$$s_d = \sigma(\text{Embed}_{level}(d)) \in (0, 1)^D$$
$$r = W_2 \cdot \text{ReLU}(W_1 \cdot x)$$
$$x' = x + \lambda \cdot (r \odot s_d)$$

### 实现

```python
class FractalTransformer(nn.Module):
    def _apply_level_aggregation(self, x: torch.Tensor, levels_info: torch.Tensor) -> torch.Tensor:
        """
        级别感知聚合（ARCH-R2）。

        公式:
            s_d = σ(Embed_level(d))
            r = W_2 · ReLU(W_1 · x)
            x' = x + λ · (r ⊙ s_d)
        """
        depths = extract_depths(levels_info, self.max_level)
        scale = torch.sigmoid(self._level_aggregator_scale(depths))

        # 调整形状以进行广播
        if scale.dim() == 2:
            scale = scale.unsqueeze(0)

        refined = self._level_aggregator_bottleneck(x)
        aggregated = refined * scale

        return x + aggregated * self._aggregator_scale
```

---

## 7.8 梯度检查点

对于长序列的内存高效训练：

```python
def forward(
    self,
    x: torch.Tensor,
    levels_info: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    regions: Optional[torch.Tensor] = None,
    image_size: Optional[int] = None,
) -> torch.Tensor:
    batch_size, seq_len, dim = x.shape

    for layer in self.layers:
        if self.use_checkpoint and self.training:
            # 重新计算激活以节省内存
            x = checkpoint(
                layer,
                x,
                levels_info,
                attention_mask,
                regions,
                image_size,
                use_reentrant=False,
            )
        else:
            x = layer(
                x,
                levels_info=levels_info,
                attention_mask=attention_mask,
                regions=regions,
                image_size=image_size,
            )

    # 级别聚合
    if levels_info is not None and levels_info.numel() > 0:
        x = self._apply_level_aggregation(x, levels_info)

    x = self.final_norm(x)
    return x
```

---

## 7.9 使用示例

```python
from vit_pytorch import FractalTransformer

transformer = FractalTransformer(
    dim=384,
    depth=6,
    heads=6,
    dim_head=64,
    mlp_dim=768,
    dropout=0.1,
    drop_path_rate=0.1,
    ffn_type='swiglu_level',
    use_checkpoint=False,
    lca_temperature=1.5,
    learnable_temperature=True,
)

x = torch.randn(2, 100, 384)
levels_info = torch.zeros(2, 100, 9, dtype=torch.long)
mask = torch.ones(2, 1, 1, 100, dtype=torch.bool)
regions = torch.rand(2, 100, 4)
image_size = 224

output = transformer(x, levels_info, mask, regions=regions, image_size=image_size)
# 输出: (2, 100, 384)
```

---

## 7.10 数值稳定性常量

所有常量都集中在 `constants.py` 中：

| 常量 | 值 | 用途 |
|:---------|:------|:--------|
| `GUMBEL_EPSILON` | 1e-8 | Gumbel 噪声稳定性 |
| `LOG_EPSILON` | 1e-8 | 对数稳定性 |
| `DIVISION_EPSILON` | 1e-8 | 除法稳定性 |
| `PROB_EPSILON` | 1e-5 | 概率钳位 |
| `LAYER_NORM_EPS` | 1e-5 | LayerNorm 方差 |
| `SPLITTER_TEMP_START` | 1.0 | 初始 Gumbel 温度 |
| `SPLITTER_TEMP_END` | 0.3 | 最终 Gumbel 温度 |
| `TEMPERATURE_MIN` | 0.3 | 最小温度（低于此梯度爆炸） |
| `DEPTH_KL_WEIGHT` | 0.5 | 深度平衡 KL 权重 |
| `DEPTH_QUOTA_TARGET` | - | 配额目标分布 |
| `LEARNABLE_QUOTA_ENABLED` | True | 方案 E 配额分配 |
| `QUOTA_MIN_PER_DEPTH` | 2 | 每深度最小配额 |
| `THRESHOLD_VAR_REG_WEIGHT` | 0.3 | 阈值方差正则化 |

---

## 7.11 复杂度分析

### 每块

| 操作 | 时间 | 空间 |
|:----------|:-----|:------|
| 注意力 | $O(B \cdot H \cdot N^2 \cdot d)$ | $O(B \cdot H \cdot N^2)$ |
| FFN | $O(B \cdot N \cdot D \cdot D_{ff})$ | $O(B \cdot N \cdot D_{ff})$ |
| 级别 Norm | $O(B \cdot N \cdot D)$ | $O(B \cdot N \cdot D)$ |
| 残差门控 | $O(B \cdot N)$ | $O(1)$ |

### 完整 Transformer（L 层）

| 指标 | 值 |
|:-------|:------|
| 时间 | $O(L \cdot B \cdot H \cdot N^2 \cdot d)$ |
| 空间（无检查点） | $O(L \cdot B \cdot H \cdot N^2)$ |
| 空间（带检查点） | $O(B \cdot H \cdot N^2 + L \cdot B \cdot N \cdot D)$ |

> **下一章**: [08_fractal_vit_model.md](08_fractal_vit_model.md) - 完整模型
