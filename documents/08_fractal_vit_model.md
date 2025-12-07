# 第八章：完整 ViT 模型 (fractal_vit.py)

本章详尽描述了数据在 `NextGenerationFractalViT` 模型中的完整流动过程，从输入图像到最终分类结果。

## 8.1 核心类：NextGenerationFractalViT

### `forward(img, ...)` 数据流详解

此函数串联了整个模型的处理管线。

**Step 1: 分形分词 (Tokenization)**
*   **输入**: `img` 张量 `(B, C, H, W)`。
*   **操作**: 调用 `self.tokenizer.tokenize(img)`。
*   **输出**: `token_output` (`TokenizerOutput` 对象)。
    *   包含 $B$ 个 `TokenSequence`。
    *   第 $i$ 个序列包含 $N_i$ 个 Token，张量形状 `(N_i, Patch_Dim)`。
    *   $N_i$ 是动态的，取决于图像复杂度。

**Step 2: Token 处理与投影 (Token Processing)**
*   **操作**: 调用 `self.token_processor.process(token_output)`。
*   **内部流程** (在 `EnhancedFractalTokenProcessor` 中):
    1.  遍历每个序列。
    2.  `token_projection`: 线性映射 `Patch_Dim -> Dim`。
    3.  `level_type_embedding`: 根据层级加 Embedding。
    4.  `feature_enhancement`: 计算统计特征 -> MLP -> 加到 Token 上。
    5.  `shared_scale_adapter`: 多尺度适配。
*   **输出**: `processed_output` (`TokenizerOutput`)。Token 维度变为 `Dim`。

**Step 3: 批次对齐 (Batch Padding)**
*   **问题**: Transformer 需要固定形状的 Batch 输入，但 $N_i$ 各不相同。
*   **操作**:
    1.  提取所有 Token 张量。
    2.  使用 `torch.nn.utils.rnn.pad_sequence(..., batch_first=True)`。
    3.  记录 `lengths` 和 `valid_indices`。
*   **输出**:
    *   `padded_tokens`: `(B, S_max, Dim)`，其中 $S_{max} = \max(N_i)$。
    *   `padded_levels`: `(B, S_max, Info_Len)`。
    *   `key_padding_mask`: `(B, S_max)`，True 表示 Padding 位置。

**Step 4: 位置编码 (Positional Embedding)**
*   **操作**: 调用 `self.pos_embedding(padded_levels)`。
*   **细节**:
    *   输入层级信息 `(B, S_max, Info_Len)`。
    *   `AdvancedFractalPositionEmbedding` 计算深度编码和路径编码。
*   **输出**: `pos_emb` `(B, S_max, Dim)`。
*   **融合**: `x = padded_tokens + pos_emb`。

**Step 5: 添加 CLS Token**
*   **操作**:
    *   扩展 `self.cls_token` 为 `(B, 1, Dim)`。
    *   `torch.cat([cls_tokens, x], dim=1)`。
*   **输出**: `x` 形状变为 `(B, S_max + 1, Dim)`。
*   **Mask 更新**: `key_padding_mask` 扩展为 `(B, S_max + 1)`，第一位（CLS）设为 False（有效）。

**Step 6: Transformer 编码**
*   **操作**: 调用 `self.transformer(x, levels_info, attn_mask)`。
*   **输入**:
    *   `x`: `(B, S_max + 1, Dim)`。
    *   `levels_info`: 包含 CLS 的层级信息。
    *   `attn_mask`: 由 `key_padding_mask` 转换而来，用于屏蔽 Padding。
*   **输出**: 编码后的 `x` `(B, S_max + 1, Dim)`。

**Step 7: 池化 (Pooling)**
*   **策略**:
    *   `cls`: 取 `x[:, 0]`。
    *   `mean`: 取 `x[:, 1:]`，结合 Mask 计算加权平均（忽略 Padding）。
    *   `hybrid`: 学习一个权重，融合 CLS 和 Mean。
*   **输出**: `pooled` `(B, Dim)`。

**Step 8: 分类头 (Classification Head)**
*   **操作**: `self.mlp_head(pooled)`。
    *   `LayerNorm` -> `Linear` -> `GELU` -> `Dropout` -> `Linear`。
*   **输出**: `final_output` `(B, NumClasses)`。

---

### `get_tokenizer_loss(...)` 数据流详解

此函数用于训练阶段的策略更新。

*   **输入**: `reward` (通常是 `-CrossEntropyLoss`)。
*   **数据源**: `self.tokenizer.saved_log_probs` (List[Tensor])。
*   **计算流程**:
    1.  **堆叠**: 将所有 Log Probs 堆叠为张量。
    2.  **优势计算**: `advantage = reward - baseline`。
    3.  **策略梯度**: `policy_loss = -advantage * log_probs.mean()`。
        *   *注：此处存在 P0 级改进点，应按样本计算而非全局平均。*
    4.  **熵正则化**: 计算 `saved_entropies` 的均值，`loss -= coef * entropy`。
*   **输出**: `aux_loss` (Scalar Tensor)，用于加到总 Loss 中反向传播。

---

## 8.2 辅助类：EnhancedFractalTokenProcessor

### `process(batch)`
负责将原始像素数据映射到语义空间。

*   **输入**: `TokenizerOutput`。
*   **流程**:
    1.  **归一化**: `LayerNorm` 输入 Token。
    2.  **投影**: `Linear` 映射到 `output_dim`。
    3.  **类型注入**: 根据 Token 的 `level` 添加 `level_type_embedding`。
    4.  **特征增强 (Feature Enhancement)**:
        *   计算 Token 的统计特征（方差、边缘密度等）。
        *   通过 `stats_processor`, `edge_processor` 等 MLP 处理。
        *   融合并加回到 Token Embedding。
    5.  **动态加权**: `dynamic_weighting` 网络计算每个 Token 的重要性权重并缩放。
*   **输出**: 处理后的 `TokenizerOutput`。
