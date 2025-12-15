# 第七章：Transformer 编码器 (transformer.py)

本章详细描述了 `EnhancedFractalTransformer` 及其构建块的数据处理逻辑。

## 7.1 数学形式化

### Transformer Block
$$x' = x + \text{DropPath}(\text{Attn}(\text{LN}_1(x)))$$
$$x'' = x' + \text{DropPath}(\text{FFN}(\text{LN}_2(x')))$$

### 层级感知归一化
$$\text{LevelNorm}(x, d) = \gamma_d \cdot \frac{x - \mu}{\sigma} + \beta_d$$

其中 $\gamma_d, \beta_d$ 是层级相关的可学习参数。

---

## 7.2 核心类：EnhancedFractalTransformer

### 初始化参数

| 参数 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `dim` | int | - | 模型维度 |
| `depth` | int | - | Transformer 层数 |
| `heads` | int | 8 | 注意力头数 |
| `dim_head` | int | 64 | 每头维度 |
| `mlp_dim` | int | - | FFN 隐藏层维度 |
| `dropout` | float | 0.0 | Dropout 比率 |
| `drop_path` | float | 0.0 | DropPath 比率 |
| `max_level` | int | 50 | 最大层级 |
| `use_hilbert_bias` | bool | True | 是否使用 Hilbert 偏置 |
| `bias_mode` | str | 'low_rank' | 偏置模式 |
| `ffn_type` | str | 'swiglu_level' | FFN 类型 |

### forward(x, levels_info, attention_mask)

**输入**:
- `x`: Token 序列 `(B, S, D)`
- `levels_info`: 层级信息 `(B, S, Info_Len)`
- `attention_mask`: 注意力掩码 `(B, 1, 1, S)`

**流程**:

**1. 层堆叠循环**
```python
for layer in self.layers:
    x = layer(x, levels_info, attention_mask)
```

**2. 全局上下文注意力**
```python
# 在所有层之后，执行一次标准 MultiheadAttention
key_padding_mask = ~attention_mask.squeeze(1).squeeze(1)  # 转换 mask
global_context, _ = self.global_attention(x, x, x, key_padding_mask)
x = x + global_context * 0.1
```

**3. 层级聚合**
```python
aggregated = self.level_aggregator(x)
x = x + aggregated * 0.2
```

**4. 最终归一化**
```python
x = self.final_norm(x)
```

**输出**: 编码后的序列 `(B, S, D)`

---

## 7.3 核心组件：EnhancedFractalTransformerBlock

这是单个 Transformer 层的实现。

### forward(x, levels_info, attention_mask)

**Step 1: 层级感知归一化 (Norm 1)**
```python
norm1_x = self._apply_level_aware_norm(
    x, levels_info, 
    self.norm1_gamma, self.norm1_beta, 
    self.default_norm1
)
```

**Step 2: 注意力机制**
```python
attn_out = self.attention(norm1_x, levels_info, attention_mask)
```
组件: `HilbertAwareMultiScaleAttention`

**Step 3: 残差连接 1**
```python
x = x + self.drop_path(attn_out * self.residual_weights[0])
```

**Step 4: 层级感知归一化 (Norm 2)**
```python
norm2_x = self._apply_level_aware_norm(
    x, levels_info,
    self.norm2_gamma, self.norm2_beta,
    self.default_norm2
)
```

**Step 5: 前馈网络 (FFN)**
```python
ff_out = self.ff(norm2_x, levels_info)
```
组件: `AdaptiveFractalFeedForward` 或 `SwiGLUFFN`

**Step 6: 残差连接 2**
```python
x = x + self.drop_path(ff_out * self.residual_weights[1])
```

---

## 7.4 辅助类：DropPath

实现随机深度 (Stochastic Depth) 正则化。

### 数学定义
训练时：
$$\text{DropPath}(x) = \begin{cases} 0 & \text{with prob } p \\ \frac{x}{1-p} & \text{otherwise} \end{cases}$$

推理时：
$$\text{DropPath}(x) = x$$

### 作用
- 相当于随机减少网络的有效深度
- 防止深层网络过拟合
- 可以视为一种 ensemble

---

## 7.5 层级感知归一化详解

### _apply_level_aware_norm()

```python
def _apply_level_aware_norm(self, x, levels_info, gamma, beta, default_norm):
    # 1. 提取深度
    depths = extract_depths(levels_info, self.max_level)
    
    # 2. 获取层级参数
    gamma_d = gamma[depths]  # (B, S, D)
    beta_d = beta[depths]    # (B, S, D)
    
    # 3. 标准归一化
    x_norm = default_norm(x)  # LayerNorm
    
    # 4. 应用层级参数
    return x_norm * gamma_d + beta_d
```

### 目的
让不同分辨率的 Token 拥有不同的分布特征，增强层级区分能力。

---

## 7.6 使用示例

```python
from vit_pytorch import EnhancedFractalTransformer

transformer = EnhancedFractalTransformer(
    dim=384,
    depth=6,
    heads=6,
    dim_head=64,
    mlp_dim=768,
    dropout=0.1,
    drop_path=0.1,
    bias_mode='low_rank',
    ffn_type='swiglu_level',
)

x = torch.randn(2, 100, 384)  # (B, S, D)
levels_info = torch.zeros(2, 100, 10, dtype=torch.long)
mask = torch.ones(2, 1, 1, 100, dtype=torch.bool)

out = transformer(x, levels_info, mask)  # (2, 100, 384)
```

---

## 7.7 架构图

```
Input (B, S, D)
    │
    ▼
┌───────────────────────────────────────┐
│  EnhancedFractalTransformerBlock × N  │
│  ┌─────────────────────────────────┐  │
│  │ Level-Aware LayerNorm          │  │
│  │        ↓                        │  │
│  │ HilbertAwareMultiScaleAttention │  │
│  │        ↓                        │  │
│  │ DropPath + Residual             │  │
│  │        ↓                        │  │
│  │ Level-Aware LayerNorm          │  │
│  │        ↓                        │  │
│  │ AdaptiveFractalFeedForward     │  │
│  │        ↓                        │  │
│  │ DropPath + Residual             │  │
│  └─────────────────────────────────┘  │
└───────────────────────────────────────┘
    │
    ▼
Global Context Attention
    │
    ▼
Level Aggregator
    │
    ▼
Final LayerNorm
    │
    ▼
Output (B, S, D)
```
