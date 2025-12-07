# 第六章：前馈网络 (feedforward.py)

`AdaptiveFractalFeedForward` 是 Transformer Block 中的 FFN 部分，它被增强以适应分形层级结构。

## 6.1 AdaptiveFractalFeedForward 类

### 主网络 (main_net)
标准的 FFN 结构：
*   `Linear(dim, hidden_dim)` -> `GELU` -> `Dropout` -> `Linear(hidden_dim, dim)` -> `Dropout`。

### 层级自适应 (Level Adaptation)
为了让 FFN 能够根据 Token 的层级（分辨率）动态调整其行为：

1.  **Level Embedding**: `level_embedding(depth)` 获取层级特征。
2.  **Shared Adapter**: 一个共享的 MLP (`shared_level_adapter`)。
3.  **混合机制**：
    *   输入 `x` 与 `level_emb` 拼接。
    *   通过 Adapter 得到 `level_adapted` 特征。
    *   使用可学习的 `level_mixing_weights` 计算混合系数。
    *   最终输出是主网络输出与 Adapter 输出的加权融合。

### 特征门控 (Feature Gating)
*   `feature_gate`: 一个生成门控信号 Sigmoid(0~1) 的小网络。
*   用于对中间层的隐藏特征进行门控调节。

### 动态激活选择器 (_apply_dynamic_activation)
这是一个实验性特性，允许网络为不同 Token 选择不同的激活函数组合。
*   `activation_selector`: 输出 3 个权重。
*   计算 `GELU(x)`, `ReLU(x)`, `Swish(x)`。
*   根据权重进行加权求和。
*   *注：这增加了模型的非线性表达能力，但也增加了计算开销。*
