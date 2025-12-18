# 第六章：前馈网络 (feedforward.py)

本章解析 `SwiGLUFFN` 和 `AdaptiveFractalFeedForward`，它们是 Transformer Block 中的 FFN 部分。

## 6.1 数学形式化

### SwiGLU FFN (LLaMA/PaLM 风格)

$\text{SwiGLU}(x) = W_{out} \cdot (\text{Swish}(W_{gate} \cdot x) \odot (W_{value} \cdot x))$

其中：

- $\text{Swish}(x) = x \cdot \sigma(x)$，$\sigma$ 是 sigmoid
- $\odot$ 表示元素级乘法（门控）

### 层级自适应 (Level Adaptation)

$\text{Output} = (1 - \alpha_d) \cdot \text{FFN}(x) + \alpha_d \cdot \text{Adapter}([x; E_{level}(d)])$

其中：

- $\alpha_d = \text{softmax}(\text{MixingWeights})_d$
- $[;]$ 表示拼接

---

## 6.2 FFN 类型选项 (ffn_type)

| 类型             | 类                            | 特点                        | 推荐场景 |
|:-------------- |:---------------------------- |:------------------------- |:---- |
| `gelu`         | `AdaptiveFractalFeedForward` | 标准 GELU，向后兼容              | 对比实验 |
| `swiglu`       | `SwiGLUFFN`                  | 轻量级，无层级自适应                | 推理优化 |
| `swiglu_level` | `AdaptiveFractalFeedForward` | SwiGLU + Level Adaptation | ✅ 推荐 |

---

## 6.3 类：SwiGLUFFN

独立的 SwiGLU 实现（参考 LLaMA/PaLM）。

### 优势

- 内置门控机制，无需额外 `feature_gate`
- 梯度流动更平滑
- 参数量与 GELU FFN 相当（通过调整 hidden_dim）

### 初始化参数

| 参数           | 类型    | 默认值   | 说明                |
|:------------ |:----- |:----- |:----------------- |
| `dim`        | int   | -     | 输入/输出维度           |
| `hidden_dim` | int   | -     | 隐藏层维度（建议为原始的 2/3） |
| `dropout`    | float | 0.0   | Dropout 比率        |
| `bias`       | bool  | False | 是否使用偏置            |

### forward(x)

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    gate = F.silu(self.w_gate(x))  # Swish 激活
    value = self.w_value(x)
    hidden = gate * value  # 元素级门控
    return self.dropout(self.w_out(hidden))
```

### 参数量对比 (dim=384)

| 配置     | GELU (hidden=768) | SwiGLU (hidden=512) | 变化        |
|:------ |:----------------- |:------------------- |:--------- |
| FFN 参数 | 590,592           | 589,824             | -0.1%     |
| 总参数    | 1.27M             | 1.15M               | **-9.4%** |

---

## 6.4 类：AdaptiveFractalFeedForward

增强型前馈网络，支持层级自适应。

### 初始化参数

| 参数                     | 类型    | 默认值            | 说明         |
|:---------------------- |:----- |:-------------- |:---------- |
| `dim`                  | int   | -              | 输入维度       |
| `hidden_dim`           | int   | -              | 隐藏层维度      |
| `dropout`              | float | 0.0            | Dropout 比率 |
| `max_level`            | int   | 50             | 最大层级       |
| `use_level_adaptation` | bool  | True           | 是否使用层级自适应  |
| `ffn_type`             | str   | 'swiglu_level' | FFN 类型     |

### 层级自适应机制

```python
# 1. 获取层级嵌入
level_emb = self.level_embedding(depths)  # (B, S, D)

# 2. 拼接并通过 Adapter
adapter_input = torch.cat([x, level_emb], dim=-1)  # (B, S, 2D)
level_adapted = self.shared_level_adapter(adapter_input)  # (B, S, D)

# 3. 计算混合系数
mixing = F.softmax(self.level_mixing_weights[depths], dim=-1)

# 4. 融合
output = (1 - mixing) * main_output + mixing * level_adapted
```

### forward(x, levels_info)

**输入**:

- `x`: Token 序列 `(B, S, D)`
- `levels_info`: 层级信息 `(B, S, Info_Len)`（可选）

**输出**: 处理后的 Token 序列 `(B, S, D)`

---

## 6.5 废弃特性

以下特性在消融实验后被废弃：

| 特性                   | 原因           | 状态    |
|:-------------------- |:------------ |:----- |
| `use_feature_gating` | SwiGLU 已内置门控 | ⚠️ 废弃 |
| `Dynamic Activation` | 熵 > 90%，无效   | ⚠️ 废弃 |

---

## 6.6 使用示例

```python
from vit_pytorch import SwiGLUFFN, AdaptiveFractalFeedForward

# 简单版
ffn_simple = SwiGLUFFN(dim=384, hidden_dim=512, dropout=0.1)

# 完整版
ffn_full = AdaptiveFractalFeedForward(
    dim=384,
    hidden_dim=768,
    dropout=0.1,
    use_level_adaptation=True,
    ffn_type='swiglu_level',
)

x = torch.randn(2, 100, 384)
levels_info = torch.zeros(2, 100, 10, dtype=torch.long)

out = ffn_full(x, levels_info)  # (2, 100, 384)
```

---

## 6.7 性能对比

基于 CIFAR-10 的消融实验结果：

| FFN 类型         | 参数量   | 推理速度  | 准确率       |
|:-------------- |:----- |:----- |:--------- |
| GELU           | 1.27M | 1.0x  | 92.1%     |
| SwiGLU         | 1.15M | 1.1x  | 92.5%     |
| SwiGLU + Level | 1.18M | 1.05x | **93.2%** |
