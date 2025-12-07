# 第四章：位置编码 (positional.py)

本章解析 `AdvancedFractalPositionEmbedding`，它解决了在非结构化分形网格中定义位置的问题。

## 4.1 类：AdvancedFractalPositionEmbedding

### `forward(levels_info, ...)`

*   **输入**: `levels_info` 张量。
    *   形状可以是 `(N, Info_Len)` (单序列) 或 `(B, N, Info_Len)` (Batch)。
    *   `Info_Len` 包含 `[depth, q_1, q_2, ..., q_depth, 0, ...]`.
*   **流程**:

    **1. 深度编码 (Depth Embedding)**
    *   提取第 0 列 `depths`。
    *   `depth_emb = self.depth_embedding(depths)`。
    *   输出: `(B, N, Dim)`。表示 Token 的分辨率层级。

    **2. 路径编码 (Path Embedding)**
    *   提取路径部分 `paths = levels_info[..., 1:]`。
    *   **坐标扁平化**:
        *   为了区分不同层级的同一象限（如第1层的左上和第2层的左上），需要加上偏移量。
        *   `offsets = arange(path_len) * 4`。
        *   `flat_indices = paths + offsets`。
    *   **查表**: `path_embs = self.quadrant_embedding(flat_indices)` -> `(B, N, Path_Len, Dim)`。
    *   **掩码聚合**:
        *   由于不同 Token 深度不同，路径长度也不同。
        *   生成掩码 `mask = index < depth`。
        *   `path_final = (path_embs * mask).sum(dim=-2)`。
    *   输出: `(B, N, Dim)`。表示 Token 在分形树中的绝对位置。

    **3. 特征融合**
    *   `combined = depth_emb + path_final`。
    *   `result = self.fusion_network(combined)`。
        *   `Linear` -> `LayerNorm` -> `GELU` -> `Dropout`。

*   **输出**: `result` `(B, N, Dim)`。

---

## 4.2 辅助方法

### `get_attention_bias(levels_info)`
*   **功能**: 计算基于层级差的注意力偏置矩阵。
*   **逻辑**:
    *   提取深度 `d`。
    *   计算 `bias[i, j] = table[d[i], d[j]]`。
*   **用途**: 可选地加到 Attention Logits 中（但在 `attention.py` 中已有类似实现，此处为备用或解耦实现）。
