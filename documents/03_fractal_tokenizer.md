# 第三章：分形 Tokenizer 核心 (fractal_curve_tokenizer.py)

本章详尽描述了图像数据如何通过分形分词器被转化为 Token 序列。这是整个模型的数据入口，决定了后续处理的粒度和质量。

## 3.1 数据流概览

1.  **输入**: 原始图像 Batch `(B, C, H, W)`。
2.  **处理**: 对每张图像独立进行递归分割。
    *   提取局部特征（手工特征 + 可选 CNN 特征）。
    *   通过策略网络决策是否分割。
    *   若分割，按 Hilbert 顺序递归处理子块。
    *   若停止，将当前 Patch 处理为固定尺寸并展平。
3.  **输出**: `TokenizerOutput` 对象，包含 $B$ 个 `TokenSequence`，每个序列长度 $N_i$ 不定。

---

## 3.2 核心类：FractalHilbertTokenizer

### `tokenize(images)`
这是分词器的入口函数。

*   **输入参数**:
    *   `images` (Tensor): 形状 `(B, C, H, W)`。
    *   *(注: `use_cnn` 选项已移至 `__init__` 初始化参数，不再作为 `tokenize` 的参数)*
*   **执行流程**:
    1.  **初始化**: 计算 `estimated_max_level`，清空 `saved_log_probs`（用于 REINFORCE）。
    2.  **Batch 循环**: 遍历 Batch 中的每一张图片 `image`。
    3.  **分割调用**: 调用 `fractal_partition(image, level=0, ...)`。
        *   默认使用 **BFS 批处理模式** (`fractal_partition_batched`) 以提高 GPU 利用率。
        *   可通过配置回退到递归模式 (`fractal_partition_recursive`)。
    4.  **结果收集**: 将返回的 `tokens` (List[Tensor]) 和 `levels` (List[List[int]]) 封装进 `TokenSequence`。
    5.  **封装**: 返回 `TokenizerOutput(sequences)`。

### `fractal_partition_batched(image, ...)` (BFS 模式)
这是默认的高效分割引擎，采用广度优先搜索 (BFS) 和批处理策略。

*   **核心思想**: 将同一层级的所有 Patch 收集起来，组成 Batch 一次性通过 CNN 和策略网络，避免递归调用产生的大量微小 Kernel。
*   **数据结构**: `PatchInfo` dataclass
    *   `patch`: 图像块张量。
    *   `level`: 当前层级。
    *   `coord`: 象限坐标路径。
    *   `dfs_order`: **关键属性**，用于在 BFS 过程中追踪 Hilbert DFS 遍历顺序。
*   **执行流程**:
    1.  **初始化队列**: 将整图作为第一个 `PatchInfo` 加入队列 `current_level_patches`。
    2.  **层级循环 (BFS)**: 当队列不为空且 `level < max_level`：
        *   **批处理决策**:
            *   收集当前层所有 Patch。
            *   调用 `_batch_decide_splits` 统一计算分割概率。
            *   对于 `learnable_split`，调用 `_batch_learnable_decision` 一次性前向传播 CNN 和 MLP。
        *   **动作执行**:
            *   **停止分割**: 将 Patch 加入 `final_patches` 列表。
            *   **继续分割**:
                *   调用 `_adaptive_split` 切分 Patch。
                *   计算子节点的 `dfs_order` (父节点 order + 偏移量)。
                *   将子节点加入 `next_level_patches`。
        *   **推进**: `current_level_patches = next_level_patches`。
    3.  **处理剩余**: 将达到最大深度的 Patch 加入 `final_patches`。
    4.  **重排序**: 根据 `dfs_order` 对 `final_patches` 进行排序，恢复 Hilbert 遍历顺序。
    5.  **后处理**: 统一 Patch 尺寸并展平，返回 Token 列表。

### `fractal_partition_recursive(patch, ...)` (递归模式)
传统的深度优先 (DFS) 实现，逻辑直观但 GPU 效率较低。

*   **流程**:
    1.  **检查停止条件**: 尺寸过小或达到最大深度。
    2.  **单次决策**: 对当前 Patch 运行策略网络。
    3.  **递归**:
        *   若分割：切分 Patch，计算 Hilbert 顺序，递归调用子节点。
        *   若停止：处理并返回当前 Patch。
    4.  **聚合**: 拼接子节点的返回结果。

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
