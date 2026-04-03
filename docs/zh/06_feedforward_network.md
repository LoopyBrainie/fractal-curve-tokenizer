# 第六章：前馈网络

## 6.1 概述

Fractal ViT 中的前馈网络（FFN）使用 **SwiGLU 激活**，带有可选的**级别自适应**，遵循 LLaMA/PaLM 架构。

---

## 6.2 数学形式化

### 6.2.1 SwiGLU FFN

$$\text{SwiGLU}(x) = W_{out} \cdot (\text{Swish}(W_{gate} \cdot x) \odot (W_{value} \cdot x))$$

其中：
- $\text{Swish}(x) = x \cdot \sigma(x)$（也称为 SiLU）
- $\sigma$：Sigmoid 函数
- $\odot$：逐元素乘法（门控）

### 6.2.2 级别自适应

$$\text{Output} = (1 - \alpha_d) \cdot \text{FFN}(x) + \alpha_d \cdot \text{Adapter}([x; E_{level}(d)])$$

其中：
- $\alpha_d = \text{softmax}(\text{MixingWeights})_d$：深度相关混合权重
- $[;]$：拼接运算符
- $E_{level}(d)$：级别嵌入

---

## 6.3 FFN 类型选项

| 类型 | 类 | 描述 | 用例 |
|:-----|:------|:------------|:---------|
| `gelu` | `AdaptiveFractalFeedForward` | 标准 GELU | 基线 |
| `swiglu` | `SwiGLUFFN` | 无自适应的 SwiGLU | 推理 |
| `swiglu_level` | `AdaptiveFractalFeedForward` | SwiGLU + 级别自适应 | **推荐** |

---

## 6.4 SwiGLU 实现

### 类：SwiGLUFFN

```python
class SwiGLUFFN(nn.Module):
    """
    SwiGLU 前馈网络（LLaMA/PaLM 风格）。

    数学形式:
        gate = Swish(W_gate · x)
        value = W_value · x
        hidden = gate ⊙ value
        output = W_out · hidden
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()

        self.w_gate = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_value = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_out = nn.Linear(hidden_dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        gate = F.silu(self.w_gate(x))  # Swish 激活
        value = self.w_value(x)
        hidden = gate * value  # 逐元素门控
        return self.dropout(self.w_out(hidden))
```

### 参数比较

对于 `dim=384`：

| 配置 | GELU (hidden=768) | SwiGLU (hidden=512) | 变化 |
|:--------------|:------------------|:--------------------|:-------|
| FFN 参数 | 590,592 | 589,824 | -0.1% |
| 总模型 | 1.27M | 1.15M | **-9.4%** |

**注意**：由于门控机制，SwiGLU 使用隐藏维度的 2/3。

---

## 6.5 级别自适应 FFN

### 类：AdaptiveFractalFeedForward

```python
class AdaptiveFractalFeedForward(nn.Module):
    """
    带级别自适应能力的前馈网络。

    扩展 SwiGLU 以支持深度感知处理。
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        max_level: int = 50,
        use_level_adaptation: bool = True,
        ffn_type: str = 'swiglu_level',
    ):
        super().__init__()

        # 主 FFN
        if 'swiglu' in ffn_type:
            self.main_ffn = SwiGLUFFN(dim, hidden_dim, dropout)
        else:
            self.main_ffn = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
            )

        # 级别自适应组件
        if use_level_adaptation:
            self.level_embedding = nn.Embedding(max_level + 1, dim)
            self.shared_level_adapter = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )
            self.level_mixing_weights = nn.Parameter(
                torch.zeros(max_level + 1)
            )
```

### 带级别自适应的前向传播

```python
def forward(self, x: Tensor, levels_info: Optional[Tensor] = None) -> Tensor:
    # 主 FFN 输出
    main_output = self.main_ffn(x)

    if not self.use_level_adaptation or levels_info is None:
        return main_output

    # 提取深度
    depths = levels_info[:, :, 0]  # (B, N)

    # 级别嵌入
    level_emb = self.level_embedding(depths)  # (B, N, D)

    # 适配器输入
    adapter_input = torch.cat([x, level_emb], dim=-1)  # (B, N, 2D)
    level_adapted = self.shared_level_adapter(adapter_input)  # (B, N, D)

    # 混合权重
    mixing = torch.sigmoid(self.level_mixing_weights[depths])  # (B, N)
    mixing = mixing.unsqueeze(-1)  # (B, N, 1)

    # 融合输出
    return (1 - mixing) * main_output + mixing * level_adapted
```

---

## 6.6 设计原理

### 为什么选择 SwiGLU？

1. **内置门控**：取代外部门控机制
2. **更平滑的梯度**：更好的训练动态
3. **已验证的有效性**：用于 LLaMA、PaLM、Gemma

### 为什么选择级别自适应？

1. **尺度感知处理**：不同深度代表不同分辨率
2. **轻量级**：添加最小参数开销（约 2%）
3. **残差设计**：当自适应不必要时保留主 FFN

---

## 6.7 已废弃特性

| 特性 | 原因 | 状态 |
|:--------|:-------|:-------|
| `use_feature_gating` | SwiGLU 有内置门控 | ⚠️ 已废弃 |
| `Dynamic Activation` | 熵 > 90%，无效 | ⚠️ 已废弃 |

---

## 6.8 使用示例

```python
from vit_pytorch import SwiGLUFFN, AdaptiveFractalFeedForward

# 简单 SwiGLU
ffn_simple = SwiGLUFFN(
    dim=384,
    hidden_dim=512,  # 典型 768 的 2/3
    dropout=0.1,
)

# 级别自适应 SwiGLU
ffn_adaptive = AdaptiveFractalFeedForward(
    dim=384,
    hidden_dim=768,
    dropout=0.1,
    use_level_adaptation=True,
    ffn_type='swiglu_level',
)

x = torch.randn(2, 100, 384)
levels_info = torch.zeros(2, 100, 5, dtype=torch.long)
levels_info[:, :, 0] = torch.randint(0, 5, (2, 100))  # 随机深度

out_simple = ffn_simple(x)  # (2, 100, 384)
out_adaptive = ffn_adaptive(x, levels_info)  # (2, 100, 384)
```

> **下一章**: [07_transformer_encoder.md](07_transformer_encoder.md) - Transformer 编码器
