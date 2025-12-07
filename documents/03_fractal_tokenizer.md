# 第三章：分形 Tokenizer 核心 (fractal_curve_tokenizer.py)

本章详尽描述了图像数据如何通过分形分词器被转化为 Token 序列。这是整个模型的数据入口，决定了后续处理的粒度和质量。

## 3.1 数据流概览

1.  **输入**: 原始图像 Batch `(B, C, H, W)`。
2.  **处理**: 对每张图像独立进行递归分割。
    *   提取局部特征。
    *   通过策略网络决策是否分割。
    *   若分割，按 Hilbert 顺序递归处理子块。
    *   若停止，将当前 Patch 处理为固定尺寸并展平。
3.  **输出**: `TokenizerOutput` 对象，包含 $B$ 个 `TokenSequence`，每个序列长度 $N_i$ 不定。

---

## 3.2 核心类：FractalHilbertTokenizer

### `tokenize(images)`
这是分词器的入口函数。

*   **输入参数**: `images` (Tensor): 形状 `(B, C, H, W)`。
*   **执行流程**:
    1.  **初始化**: 计算 `estimated_max_level`，清空 `saved_log_probs`（用于 REINFORCE）。
    2.  **Batch 循环**: 遍历 Batch 中的每一张图片 `image`。
    3.  **递归调用**: 对每张图片调用 `fractal_partition(image, level=0, ...)`。
    4.  **结果收集**: 将 `fractal_partition` 返回的 `tokens` (List[Tensor]) 和 `levels` (List[List[int]]) 封装进 `TokenSequence`。
    5.  **封装**: 返回 `TokenizerOutput(sequences)`。

### `fractal_partition(patch, level, coord, ...)`
这是递归分割的核心引擎。

*   **输入参数**:
    *   `patch`: 当前图像块 `(C, H, W)`。
    *   `level`: 当前递归深度 (int)。
    *   `coord`: 当前路径坐标列表 (List[int])，记录了从根节点到当前的象限路径。
*   **执行流程**:
    1.  **停止条件检查**:
        *   物理限制: `H <= min_h` 或 `W <= min_w`。
        *   深度限制: `level >= max_level`。
        *   安全限制: 防止无限递归。
    2.  **分割决策 (Decision Phase)**:
        *   **可学习模式 (`learnable_split=True`)**:
            1.  **特征提取**: 调用 `_extract_enhanced_patch_features` 获取 6 维统计特征（方差、均值、边缘密度等）。
            2.  **CNN 特征**: 调用 `cnn_encoder(patch)` 获取 32 维视觉特征。
            3.  **特征融合**: 拼接得到 38 维特征向量。
            4.  **策略网络**: `split_decision(combined)` 输出 Logits `[not_split, split]`。
            5.  **采样**:
                *   训练时: `Categorical(logits).sample()`，并保存 `log_prob`。
                *   推理时: `argmax(logits)`。
        *   **启发式模式**: 计算 Patch 方差，若 `var < threshold` 则停止。
    3.  **动作执行 (Action Phase)**:
        *   **动作 0 (停止分割)**:
            1.  调用 `_process_patch_to_fixed_size(patch)` 将 Patch 统一为 `min_patch_size`。
            2.  展平为 Token 向量 `(C * min_h * min_w)`。
            3.  记录层级信息 `[level, q_1, q_2, ..., q_level, 0, ...]`.
            4.  返回 `([token], [level_info])`。
        *   **动作 1 (继续分割)**:
            1.  调用 `_adaptive_split(patch)` 将 Patch 切分为 4 个子块（左上、右上、左下、右下）。
            2.  调用 `_determine_traversal_order` (内部委托给 `hilbert` 模块) 获取 Hilbert 遍历顺序（如 `[2, 0, 1, 3]`）。
            3.  **递归**: 按顺序对每个子块调用 `fractal_partition`。
            4.  **聚合**: 将所有子块返回的 Token 和 Level 列表拼接并返回。

### `_adaptive_split(patch, H, W, ...)`
负责具体的张量切分操作。

*   **逻辑**:
    *   尝试 **智能四分法** (`_intelligent_quadrant_split`)：寻找最接近中心的整数分割点。
    *   如果无法四分（如长条形），则退化为 **二分法**。
    *   使用 `tensor[:, :split_h, :split_w]` 等切片操作生成子块。

### `_process_patch_to_fixed_size(patch, H, W)`
负责将任意尺寸的叶子节点 Patch 归一化。

*   **逻辑**:
    *   如果 Patch 尺寸等于 `min_patch_size`: 直接返回。
    *   如果 Patch 尺寸小于 `min_patch_size`: 使用 `F.pad` 进行填充。
    *   如果 Patch 尺寸大于 `min_patch_size`: 使用 `F.adaptive_avg_pool2d` 下采样。
*   **目的**: 确保输入 Transformer 的所有 Token 维度一致。

---

## 3.3 辅助神经网络类

### `MiniCNN`
轻量级特征提取器，为决策网络提供视觉信息。

*   **输入**: `(B, C, H, W)` 或 `(C, H, W)`。
*   **结构**:
    1.  `InstanceNorm2d`: 归一化，消除亮度差异。
    2.  `Conv2d` (3->16, k=3, s=2) + `ReLU`: 下采样。
    3.  `InstanceNorm2d`.
    4.  `Conv2d` (16->32, k=3, s=2) + `ReLU`: 下采样。
    5.  `AdaptiveAvgPool2d((1, 1))`: 全局池化。
    6.  `Flatten`: 展平。
*   **输出**: `(B, 32)` 特征向量。

### `LearnableSplitDecision`
策略网络 (Policy Network)。

*   **输入**: `(B, 38)` (6 统计特征 + 32 CNN 特征)。
*   **结构**:
    1.  `LayerNorm`: 归一化混合特征。
    2.  `Linear(38, 128)` -> `ReLU` -> `Dropout`.
    3.  `Linear(128, 64)` -> `ReLU` -> `Dropout`.
    4.  `Linear(64, 2)`.
*   **输出**: `(B, 2)` Logits，分别对应 `[停止, 分割]`。
*   **初始化**: 偏置 `bias[1] += 2.0`，初始阶段倾向于分割，鼓励探索。
