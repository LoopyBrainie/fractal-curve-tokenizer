# 第五章：注意力机制 (attention.py)

本章详细解析 `HilbertAwareMultiScaleAttention`，这是模型理解分形结构和空间关系的核心组件。

## 5.1 类：HilbertAwareMultiScaleAttention

### `forward(x, levels_info, attention_mask)`

**Step 1: 投影 (QKV Projection)**
*   **输入**: `x` `(B, S, Dim)`。
*   **操作**:
    *   `self.norm(x)`。
    *   `self.to_qkv(x)` -> `(B, S, 3 * Inner_Dim)`。
    *   `chunk(3)` 分离出 Q, K, V。
    *   `rearrange` 重排为多头格式 `(B, Heads, S, Dim_Head)`。

**Step 2: 点积注意力 (Dot Product)**
*   **操作**: `matmul(q, k.transpose)`。
*   **输出**: `dots` `(B, Heads, S, S)` (Attention Logits)。
*   **缩放**: 乘以 `scale` ($\frac{1}{\sqrt{d_k}}$)。

**Step 3: 层级缩放 (Level Scaling)**
*   **条件**: `use_level_scaling=True`。
*   **逻辑**:
    *   从 `levels_info` 获取深度 `depths` `(B, S)`。
    *   查表 `level_scale_embedding` 获取缩放因子 `(B, S, Heads)`。
    *   调整形状并广播，乘以 `dots`。
*   **目的**: 调节不同层级 Token 的注意力分布锐度。

**Step 4: Hilbert 偏置注入 (Hilbert Bias)**
*   **条件**: `use_hilbert_bias=True`。
*   **调用**: `_compute_hilbert_bias(levels_info)`。
*   **逻辑**:
    1.  提取路径坐标 `paths` `(B, S, Path_Len)`。
    2.  计算所有 Token 对之间的 **欧氏距离** 和 **余弦相似度**。
    3.  得到特征图 `(B, S, S, 2)`。
    4.  通过 MLP `hilbert_bias_network` 映射为 `(B, S, S, Heads)`。
    5.  变换为 `(B, Heads, S, S)`。
*   **操作**: `dots = dots + hilbert_bias * 0.1`。
*   **意义**: 让模型显式感知 Token 在 Hilbert 曲线上的空间邻近关系。

**Step 5: 相对层级偏置 (Level Bias)**
*   **调用**: `_compute_level_bias(levels_info)`。
*   **逻辑**:
    1.  计算深度差 `diff = depth_i - depth_j`。
    2.  查表 `relative_pos_embedding`。
*   **操作**: `dots = dots + level_bias * 0.05`。
*   **意义**: 编码跨层级关系（如父节点关注子节点）。

**Step 6: 掩码 (Masking)**
*   **输入**: `attention_mask` `(B, 1, 1, S)` (True=保留, False=Mask)。
*   **操作**: `dots.masked_fill_(~attention_mask, -inf)`。
*   **作用**: 确保 Padding Token 的注意力权重为 0。

**Step 7: Softmax 与 输出**
*   **操作**:
    *   `attn = softmax(dots)`。
    *   `attn = dropout(attn)`。
    *   `out = matmul(attn, v)`。
    *   `rearrange` 合并多头 -> `(B, S, Dim)`。
    *   `to_out` 线性投影。
*   **输出**: `out` `(B, S, Dim)`。
