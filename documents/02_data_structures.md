# 第二章：数据流与抽象层

本章介绍了项目的基础数据结构和抽象层，它们定义了数据如何在各个模块之间流动。

## 2.1 tokenization.py - 基础数据结构

此文件定义了 Tokenizer 与模型其他部分交互的标准协议。

### TokenSequence

单样本 Token 序列容器。

* **属性**：
  * `tokens`: `torch.Tensor`，形状为 `[N, D]`，Token 特征
  * `metadata`: `Dict[str, Any]`，存储元数据，最重要的是 `"levels"` 信息
* **方法**：
  * `get_levels()`: 获取层级信息张量
  * `clone()`: 深拷贝序列

### TokenizerOutput

批量输出容器。

* **属性**：
  * `sequences`: `List[TokenSequence]`，包含 Batch 中每个样本的 `TokenSequence`
* **方法**：
  * `tokens_list()` / `levels_list()`: 辅助提取方法

### BaseTokenizer / BaseTokenProcessor

抽象基类，定义了组件的接口规范。

* `BaseTokenizer`: 必须实现 `tokenize(images) -> TokenizerOutput`
* `BaseTokenProcessor`: 必须实现 `process(batch) -> TokenizerOutput`

## 2.2 features.py - 特征计算

此模块用于计算 Token 的统计特征，用于特征增强。

### TokenFeatures dataclass

封装计算出的 6 维特征：

* `stats`: 均值 $\mu$ 和方差 $\sigma^2$
* `edge`: 边缘密度代理（基于差分）
* `spatial`: 原始 Patch 的空间尺寸 (H, W)
* `level`: 当前 Token 所处的递归层级

**数学定义**:
$f = [\sigma^2, \mu, \text{edge}, h, w, d] \in \mathbb{R}^6$

### compute_token_features() 函数

* **输入**：Token 张量、层级、Patch 尺寸
* **输出**：`TokenFeatures` dataclass

## 2.3 utils.py - 工具函数

### 基础工具

* `pair(t)`: 将输入转换为元组 (t, t)
* `exists(val)`: 检查变量是否不为 None
* `default(val, d)`: 如果 val 存在则返回 val，否则返回默认值 d

### extract_depths()

**功能**：统一从 `levels_info` 提取深度索引。

**数学定义**:
$\text{depths} = \text{clamp}(L_{:,:,0}, 0, L_{max})$

**逻辑**：

* 自动处理 `(Seq, Info)` 和 `(Batch, Seq, Info)` 两种输入形状
* 提取第 0 维（深度信息）
* 执行 `clamp(0, max_level)` 确保索引安全

### normalize_levels_info()

**功能**：规范化 `levels_info` 维度。
**逻辑**：将 `(Seq, Info)` 自动升维为 `(1, Seq, Info)` 以统一批处理逻辑。

### create_attention_mask()

**功能**：生成层级感知的注意力 Mask。
**逻辑**：

* 输入层级信息列表
* 构建 `(B, S, S)` 的 Mask 矩阵
* **向量化实现**：使用 PyTorch 广播机制

## 2.4 streaming_tokenizer.py - 核心数据结构

### HilbertIndexer

用于预计算 Hilbert/Pseudo-Hilbert 曲线索引的工具类。

* **方法**：
  * `get_hilbert_order(grid_size)`: 返回从光栅顺序到 Hilbert 顺序的索引映射
  * `get_hilbert_order_rect(grid_h, grid_w)`: 支持矩形网格
* **特点**：使用 `@lru_cache` 缓存避免重复计算

### HilbertPathCache

统一的 Hilbert 路径预计算缓存，存储两种映射：

* `hilbert_to_raster`: Hilbert 索引 → 光栅索引
* `quadtree_paths`: Hilbert 索引 → 四叉树路径

**数学定义**：
$$\text{quadtree\_path}[d, \ell] = q_\ell = \text{bit}(x, k-\ell) + 2 \times \text{bit}(y, k-\ell)$$

### MultiScalePatchEncoder

多尺度卷积金字塔，核心数据结构。

* **数学定义**：
  $F_s = \text{Conv}_s(I), \quad s \in \{1, \ldots, S\}$
  每个尺度: `kernel_size = stride = patch_size_s`

### Variable Depth Tokens 架构 (V3 核心)

Variable Depth Tokens (VDT) 是当前 V3 Tokenizer 的核心架构，取代了已废弃的 CrossScaleAttention。

* **数学定义**：
  $$C(R) = \alpha \cdot \frac{\text{Var}(R)}{\text{Var}(R) + \sigma_0^2} + (1-\alpha) \cdot \frac{G(R)}{G(R) + g_0^2}$$
  $$p_{split} = \sigma\left(\frac{C_\theta(R) - \tau_d}{T}\right), \quad z \sim \text{Gumbel-Softmax}(p)$$
  
* **关键组件**：
  * `LearnableSplitter`: 可学习的内容自适应分割器
  * `BalancedGreedySplitter`: 平衡贪心分割策略
  * `DPBudgetSplitter`: 动态规划预算分割
  
* **架构优势**：
  * 无尺度崩塌问题 (CrossScaleAttention 存在的数学缺陷)
  * 真正的内容自适应分辨率
  * STE + REINFORCE 混合梯度估计

## 2.5 constants.py - 超参数默认值

| 常量                   | 值    | 说明                                 |
|:-------------------- |:---- |:---------------------------------- |
| `HILBERT_BIAS_SCALE` | 0.1  | Hilbert 偏置缩放因子 $\lambda_{hilbert}$ |
| `LEVEL_BIAS_SCALE`   | 0.05 | 层级偏置缩放因子 $\lambda_{level}$         |
| `EMBEDDING_INIT_STD` | 0.02 | 嵌入初始化标准差 $\sigma_{emb}$            |
| `DEFAULT_MAX_LEVEL`  | 50   | 默认最大层级 $L_{max}$                   |
| `DEFAULT_DROPOUT`    | 0.1  | 默认 Dropout 比率                      |

## 2.6 fractal_config.py - 统一配置

### FractalConfig dataclass

项目核心配置类，集中管理所有超参数，提供类型安全和默认值。

```python
@dataclass
class FractalConfig:
    # 模型维度
    d_model: int = 384
    num_heads: int = 8

    # Hilbert 偏置模式 (核心)
    hilbert_bias_mode: BiasMode = 'lca'  # 推荐默认值

    # LowRank 偏置参数
    low_rank_dim: int = 16

    # LCA 偏置参数  
    max_depth: int = 10

    # 层级嵌入
    max_level: int = 10
    level_embedding_dim: int = 32

    # Gumbel-Softmax
    initial_temperature: float = 2.0
    min_temperature: float = 0.1

    # 正则化
    dropout: float = 0.1
```

**关键参数说明**:

| 参数                  | 类型       | 默认值     | 说明                                          |
|:------------------- |:-------- |:------- |:------------------------------------------- |
| `hilbert_bias_mode` | BiasMode | `'lca'` | 偏置模式: `'lca'`/`'low_rank'`/`'hierarchical'` |
| `low_rank_dim`      | int      | 16      | Low-Rank 投影维度 r                             |
| `max_depth`         | int      | 10      | LCA 最大深度 $D_{max}$                          |

**BiasMode 类型**:

```python
BiasMode = Literal['low_rank', 'lca', 'hierarchical', 'none']
```

### 使用示例

```python
from vit_pytorch import FractalConfig, FractalCurveViT

# 使用默认配置（LCA 模式）
config = FractalConfig()
model = FractalCurveViT(config=config)

# 自定义配置
config = FractalConfig(
    d_model=512,
    num_heads=8,
    hilbert_bias_mode='low_rank',  # 使用 Low-Rank 模式
    low_rank_dim=32,
)
```
