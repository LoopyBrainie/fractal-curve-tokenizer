# 第二章：数据流与抽象层

本章介绍了项目的基础数据结构和抽象层，它们定义了数据如何在各个模块之间流动。

## 2.1 tokenization.py - 基础数据结构

此文件定义了 Tokenizer 与模型其他部分交互的标准协议。

### TokenSequence
单样本 Token 序列容器。

*   **属性**：
    *   `tokens`: `torch.Tensor`，形状为 `[N, D]`，Token 特征
    *   `metadata`: `Dict[str, Any]`，存储元数据，最重要的是 `"levels"` 信息
*   **方法**：
    *   `get_levels()`: 获取层级信息张量
    *   `clone()`: 深拷贝序列

### TokenizerOutput
批量输出容器。

*   **属性**：
    *   `sequences`: `List[TokenSequence]`，包含 Batch 中每个样本的 `TokenSequence`
*   **方法**：
    *   `to_legacy()`: 转换为旧版格式（兼容性）
    *   `tokens_list()` / `levels_list()`: 辅助提取方法

### BaseTokenizer / BaseTokenProcessor
抽象基类，定义了组件的接口规范。

*   `BaseTokenizer`: 必须实现 `tokenize(images) -> TokenizerOutput`
*   `BaseTokenProcessor`: 必须实现 `process(batch) -> TokenizerOutput`

## 2.2 features.py - 特征计算

此模块用于计算 Token 的统计特征，用于特征增强。

### TokenFeatures dataclass
封装计算出的 6 维特征：
*   `stats`: 均值 $\mu$ 和方差 $\sigma^2$
*   `edge`: 边缘密度代理（基于差分）
*   `spatial`: 原始 Patch 的空间尺寸 (H, W)
*   `level`: 当前 Token 所处的递归层级

**数学定义**:
$$f = [\sigma^2, \mu, \text{edge}, h, w, d] \in \mathbb{R}^6$$

### compute_token_features() 函数
*   **输入**：Token 张量、层级、Patch 尺寸
*   **输出**：`TokenFeatures` dataclass

## 2.3 utils.py - 工具函数

### 基础工具
*   `pair(t)`: 将输入转换为元组 (t, t)
*   `exists(val)`: 检查变量是否不为 None
*   `default(val, d)`: 如果 val 存在则返回 val，否则返回默认值 d

### extract_depths()
**功能**：统一从 `levels_info` 提取深度索引。

**数学定义**:
$$\text{depths} = \text{clamp}(L_{:,:,0}, 0, L_{max})$$

**逻辑**：
*   自动处理 `(Seq, Info)` 和 `(Batch, Seq, Info)` 两种输入形状
*   提取第 0 维（深度信息）
*   执行 `clamp(0, max_level)` 确保索引安全

### normalize_levels_info()
**功能**：规范化 `levels_info` 维度。
**逻辑**：将 `(Seq, Info)` 自动升维为 `(1, Seq, Info)` 以统一批处理逻辑。

### create_attention_mask()
**功能**：生成层级感知的注意力 Mask。
**逻辑**：
*   输入层级信息列表
*   构建 `(B, S, S)` 的 Mask 矩阵
*   **向量化实现**：使用 PyTorch 广播机制

## 2.4 streaming_tokenizer.py - 核心数据结构

### HilbertIndexer
用于预计算 Hilbert 曲线索引的工具类。

*   **方法**：
    *   `get_hilbert_order(grid_size)`: 返回从光栅顺序到 Hilbert 顺序的索引映射
*   **特点**：使用 `@lru_cache` 缓存避免重复计算

### MultiScalePatchEncoder
多尺度卷积金字塔，核心数据结构。

*   **数学定义**：
    $$F_s = \text{Conv}_s(I), \quad s \in \{1, \ldots, S\}$$
    每个尺度: `kernel_size = stride = patch_size_s`

## 2.5 constants.py - 超参数默认值

| 常量 | 值 | 说明 |
| :--- | :--- | :--- |
| `HILBERT_BIAS_SCALE` | 0.1 | Hilbert 偏置缩放因子 $\lambda_{hilbert}$ |
| `LEVEL_BIAS_SCALE` | 0.05 | 层级偏置缩放因子 $\lambda_{level}$ |
| `EMBEDDING_INIT_STD` | 0.02 | 嵌入初始化标准差 $\sigma_{emb}$ |
| `DEFAULT_MAX_LEVEL` | 50 | 默认最大层级 $L_{max}$ |
| `DEFAULT_DROPOUT` | 0.1 | 默认 Dropout 比率 |
