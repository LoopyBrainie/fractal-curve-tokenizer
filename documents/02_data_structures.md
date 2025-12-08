# 第二章：数据流与抽象层

本章介绍了项目的基础数据结构和抽象层，它们定义了数据如何在各个模块之间流动。

## 2.1 tokenization.py - 基础数据结构

此文件定义了 Tokenizer 与模型其他部分交互的标准协议。

### TokenSequence
单样本 Token 序列容器。由于自适应分割导致每个样本的 Token 数量不同，我们需要一个灵活的容器。

*   **属性**：
    *   `tokens`: `torch.Tensor`，形状为 `[N, Dim]`，表示展平后的 Token 特征。
    *   `metadata`: `Dict[str, Any]`，存储元数据，最重要的是 `"levels"` 信息。
*   **方法**：
    *   `get_levels()`: 获取层级信息张量。
    *   `clone()`: 深拷贝序列。

### TokenizerOutput
批量输出容器。

*   **属性**：
    *   `sequences`: `List[TokenSequence]`，包含 Batch 中每个样本的 `TokenSequence`。
*   **方法**：
    *   `to_legacy()`: 转换为旧版格式（List[Tensor]），用于兼容旧代码。
    *   `tokens_list()` / `levels_list()`: 辅助提取方法。

### BaseTokenizer / BaseTokenProcessor
抽象基类，定义了组件的接口规范。

*   `BaseTokenizer`: 必须实现 `tokenize(images) -> TokenizerOutput`。
*   `BaseTokenProcessor`: 必须实现 `process(batch) -> TokenizerOutput`。

## 2.2 features.py - 特征计算

此模块用于计算 Token 的统计特征，这些特征后续会被用于特征增强网络。

### TokenFeatures dataclass
用于封装计算出的特征：
*   `stats`: 均值和方差。
*   `edge`: 边缘密度代理（基于差分）。
*   `spatial`: 原始 Patch 的空间尺寸 (H, W)。
*   `level`: 当前 Token 所处的递归层级。

### compute_token_features() 函数
*   **输入**：展平的 Token 张量、层级、Patch 尺寸。
*   **逻辑**：
    *   计算 Token 内部的方差和均值。
    *   计算 Token 内部的差分绝对值均值作为边缘密度。
    *   扩展空间和层级信息以匹配 Batch 维度。

## 2.3 utils.py - 工具函数

### 基础工具
*   `pair(t)`: 将输入转换为元组 (t, t)。
*   `exists(val)`: 检查变量是否不为 None。
*   `default(val, d)`: 如果 val 存在则返回 val，否则返回默认值 d。

### create_attention_mask()
**功能**：生成层级感知的注意力先验 Mask。
**逻辑**：
*   输入层级信息列表。
*   构建 `(B, S, S)` 的 Mask 矩阵。
*   **向量化实现**：使用 PyTorch 广播机制替代双重循环，大幅提升 CPU 性能。
*   **层级关系编码**：
    *   同一层级：权重 1.2 (强关联)
    *   相邻层级：权重 1.1 (中关联)
    *   其他：权重 1.0

### extract_depths()
**功能**：统一从 `levels_info` 提取深度索引。
**逻辑**：
*   自动处理 `(Seq, Info)` 和 `(Batch, Seq, Info)` 两种输入形状。
*   提取第 0 维（深度信息）。
*   执行 `clamp(0, max_level)` 确保索引安全。

### normalize_levels_info()
**功能**：规范化 `levels_info` 维度。
**逻辑**：
*   将 `(Seq, Info)` 自动升维为 `(1, Seq, Info)` 以统一批处理逻辑。

## 2.4 fractal_curve_tokenizer.py - 内部数据结构

### PatchInfo dataclass
用于 BFS 批处理过程中追踪 Patch 状态。

*   **属性**：
    *   `patch`: `torch.Tensor` `(C, H, W)`，图像块数据。
    *   `level`: `int`，当前递归深度。
    *   `coord`: `List[int]`，象限路径坐标。
    *   `dfs_order`: `float`，**关键属性**，用于在 BFS 过程中保持 Hilbert DFS 遍历顺序。

