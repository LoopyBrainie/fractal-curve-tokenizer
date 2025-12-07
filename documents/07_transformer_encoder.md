# 第七章：Transformer 编码器 (transformer.py)

本章详细描述了 `EnhancedFractalTransformer` 及其构建块的数据处理逻辑。

## 7.1 核心类：EnhancedFractalTransformer

### `forward(x, levels_info, attention_mask)`
这是 Transformer 的主循环。

*   **输入**:
    *   `x`: Token 序列 `(B, S, Dim)`。
    *   `levels_info`: 层级信息 `(B, S, Info_Len)`。
    *   `attention_mask`: 注意力掩码 `(B, 1, 1, S)`。
*   **流程**:
    1.  **动态深度 (可选)**:
        *   如果启用 `use_dynamic_depth`，通过 `depth_selector` 计算每层的权重 `layer_weights`。
    2.  **层堆叠循环**:
        *   遍历 `self.layers` (List of `EnhancedFractalTransformerBlock`)。
        *   `x = layer(x, levels_info, attention_mask)`。
        *   如果启用动态深度，`x = x * weight`。
    3.  **全局上下文注意力 (Global Context Attention)**:
        *   在所有层之后，执行一次标准的 `MultiheadAttention`。
        *   **关键点**: 将 `attention_mask` (True=保留) 转换为 `key_padding_mask` (True=Mask) 以忽略 Padding Token。
        *   `x = x + global_context * 0.1` (残差连接)。
    4.  **层级聚合 (Level Aggregation)**:
        *   `aggregated = level_aggregator(x)`。
        *   `x = x + aggregated * 0.2`。
    5.  **最终归一化**: `final_norm(x)`。
*   **输出**: 编码后的序列 `(B, S, Dim)`。

---

## 7.2 核心组件：EnhancedFractalTransformerBlock

这是单个 Transformer 层的实现，包含 Attention 和 FFN。

### `forward(x, levels_info, attention_mask)`

**Step 1: 层级感知归一化 (Norm 1)**
*   **操作**: `_apply_level_aware_norm(x, levels_info, norm1_gamma, norm1_beta, default_norm1)`。
*   **逻辑**:
    *   根据 `levels_info` 中的深度，查表获取 `gamma` 和 `beta`。
    *   执行标准 LayerNorm: `(x - mean) / std`。
    *   应用层级参数: `* gamma + beta`。
    *   *目的*: 让不同分辨率的 Token 拥有不同的分布特征。

**Step 2: 注意力机制 (Attention)**
*   **操作**: `self.attention(norm1_x, levels_info, attention_mask)`。
*   **组件**: `HilbertAwareMultiScaleAttention`。
*   **输出**: `attn_out`。

**Step 3: 残差连接 1**
*   **操作**: `x = x + drop_path(attn_out * residual_weights[0])`。
*   **细节**: `DropPath` 随机丢弃整个残差分支（训练时）。

**Step 4: 层级感知归一化 (Norm 2)**
*   **操作**: 同 Step 1，但使用 `norm2` 参数。

**Step 5: 前馈网络 (FFN)**
*   **操作**: `self.ff(norm2_x, levels_info)`。
*   **组件**: `AdaptiveFractalFeedForward`。
*   **输出**: `ff_out`。

**Step 6: 残差连接 2**
*   **操作**: `x = x + drop_path(ff_out * residual_weights[1])`。

---

## 7.3 辅助类：DropPath

实现随机深度 (Stochastic Depth) 正则化。

*   **输入**: 张量 `x`。
*   **逻辑**:
    *   训练时: 以概率 `drop_prob` 将 `x` 置为 0。
    *   为了保持期望一致，未被丢弃的样本会除以 `(1 - drop_prob)`。
    *   推理时: 直接返回 `x` (Identity)。
*   **作用**: 相当于随机减少网络的有效深度，防止深层网络过拟合。
