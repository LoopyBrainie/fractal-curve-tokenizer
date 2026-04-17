# 第二章：核心数学基础

## 2.1 概述

本章描述了支撑 Fractal Curve Tokenizer 系统的数学原语。架构依赖于四叉树结构和 Hilbert 空间填充曲线之间的同构，将 2D 空间信息转换为 1 维序列，同时保持最大局部性。

---

## 2.2 Hilbert 曲线算法

系统实现了几个 Hilbert 曲线变体来处理不同的硬件约束和图像几何。

### 2.2.1 FastBitwiseHilbert

使用 Gray 码和 Morton 编码（位交错）执行 2D 到 1D 映射的**向量化实现**，无需 Python 循环。

**关键特性**：
- O(1) 每元素转换的位运算
- 无递归开销
- 针对 GPU 张量运算优化

**源码**：`src/vit_pytorch/core/fast_bitwise_hilbert.py`

### 2.2.2 HilbertCurve（Butz 算法）

用于 $N = 2^k$ 平方网格的标准递归实现。

```python
class HilbertCurve:
    def __init__(self, level: int):
        self.level = level
        self.n = 2 ** level

    def d_to_xy(self, d: int) -> Tuple[int, int]:
        """将 Hilbert 索引转换为 2D 坐标。"""
        # Butz 算法实现
        ...

    def xy_to_d(self, x: int, y: int) -> int:
        """将 2D 坐标转换为 Hilbert 索引。"""
        ...
```

**源码**：`src/vit_pytorch/core/curve_hilbert.py`

### 2.2.3 PseudoHilbertCurve

将局部性优势扩展到**非方形或非 2 的幂次网格**。适用于任意图像尺寸。

### 2.2.4 HilbertTopologyCache

**延迟预计算模块**，存储 `coord_to_idx` 和 `idx_to_coord` 张量，以避免训练期间冗余计算。

```python
class HilbertTopologyCache:
    def __init__(self, max_level: int = 8):
        self.max_level = max_level
        self._coord_to_idx = {}  # 延迟初始化
        self._idx_to_coord = {}

    def get_coords(self, indices: Tensor) -> Tensor:
        """将 Hilbert 索引转换为 2D 坐标。"""
        if self.max_level not in self._idx_to_coord:
            self._precompute(self.max_level)
        return self._idx_to_coord[self.max_level][indices]
```

**缓存键**：`data_ptr` + `torch_version` 用于自动失效

**源码**：`src/vit_pytorch/core/hilbert_topology_cache.py`

---

## 2.3 四叉树和 LevelsInfo 数据结构

### 2.3.1 LevelsInfo 张量

分形分词的层次性质由 `LevelsInfo` 数据结构表示，将四叉树展平为适合 GPU 处理的张量格式。

**张量定义**：

$$L \in \mathbb{Z}^{B \times N \times (D+1)}$$

其中：

| 索引 | 内容 | 范围 | 描述 |
|:------|:--------|:------|:------------|
| `L[:, :, 0]` | 深度 | $[0, D_{max}]$ | Token 的四叉树深度 |
| `L[:, :, 1:]` | 路径 | $[0, 3]^{D_{max}}$ | 象限路径（象限索引） |

### 2.3.2 象限编码

```
象限索引（Hilbert 兼容）：
    ┌─────┬─────┐
    │  2  │  3  │
    ├─────┼─────┤
    │  0  │  1  │
    └─────┴─────┘
```

### 2.3.3 坐标重建

Token 使用向量化位运算从其 `LevelsInfo` 路径重建为 2D 坐标：

$$x = \sum_{k=0}^{d-1} \text{bit}_k(q_k, 0) \cdot 2^{max\_level-k-1}$$

$$y = \sum_{k=0}^{d-1} \text{bit}_k(q_k, 1) \cdot 2^{max\_level-k-1}$$

这避免了传统四叉树遍历的 $O(N \cdot D)$ 串行开销。

### 2.3.4 LCA 矩阵

系统计算任意两个 token 之间的**最近公共祖先（LCA）**，以在注意力机制中确定它们的几何偏置：

$$\text{LCA}(i, j) = \text{Length}(\text{CommonPrefix}(\text{Path}(i), \text{Path}(j)))$$

LCA 深度提供了四叉树层次中空间邻近性的度量。

### 2.3.5 预计算 LUT

为 `torch.compile` 兼容性，系统使用**急切初始化**预计算高达深度 8 的 Hilbert 查找表（LUT）。

```python
# 预计算 LUT 以便快速查找
_LUT_DEPTH_8 = self._build_lut(max_level=8)
```

**源码**：`src/vit_pytorch/core/levels_info.py`

---

## 2.4 配置和分割器协议

### 2.4.1 FractalConfig

自动推导图像几何，确定基于输入分辨率的 `max_level` 以及是否使用 `PseudoHilbertCurve`。

```python
@dataclass
class FractalConfig:
    image_size: int | Tuple[int, int]
    min_patch_size: int = 4
    max_level_limit: int = 8
    tokenizer_type: str = 'streaming_v3'

    def __post_init__(self):
        # 从图像大小自动推导 max_level
        if isinstance(self.image_size, int):
            self.max_level = int(np.log2(self.image_size // self.min_patch_size))
        else:
            self.max_level = min(
                int(np.log2(self.image_size[0] // self.min_patch_size)),
                int(np.log2(self.image_size[1] // self.min_patch_size))
            )
```

### 2.4.2 CoreSplitter 协议

不同分割策略的标准化接口：

```python
class CoreSplitter(Protocol):
    @property
    def num_candidates(self) -> int:
        """候选区域数量。"""
        ...

    def forward(
        self,
        features: Tensor,
        image_size: Tuple[int, int]
    ) -> SplitResult:
        """执行分割决策。"""
        ...

    def update_candidates(self, image_size: Tuple[int, int]) -> None:
        """为新图像大小更新候选区域。"""
        ...
```

### 2.4.3 树一致性公理（H1SS）

`HilbertOptimalSplitter` 遵循六个公理以确保有效的四叉树分割：

| 公理 | 描述 |
|:------|:------------|
| **A1** | 覆盖：根覆盖整个图像 |
| **A2** | 包含：每个子区域 ⊆ 父区域 |
| **A3** | 不相交：兄弟区域不重叠 |
| **A4** | 树深度：最大深度 ≤ $D_{max}$ |
| **A5** | 单调性：父区域被选中 ⇒ 至少一个子区域被选中 |
| **A6** | 连续性：选中区域形成连通子图 |

---

## 2.5 数学概念到代码实体的映射

```
┌─────────────────────────────────────────────────────────────────┐
│              Mathematical Concept Space                          │
├─────────────────────────────────────────────────────────────────┤
│  Hilbert Curve Mapping  │  Quadtree Decomposition  │  Spatial Indexing  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ uses
┌─────────────────────────────────────────────────────────────────┐
│                 Code Entity Space (L1 Foundation)               │
├─────────────────────────────────────────────────────────────────┤
│  FastBitwiseHilbert         │  HilbertCurve        │  HilbertTopologyCache  │
│  [core/fast_bitwise_hilbert.py]  [core/curve_hilbert.py]  [core/hilbert_topology_cache.py]  │
├─────────────────────────────────────────────────────────────────┤
│  LevelsInfo                      │  PatternEncoder  │  SplitterProtocol  │
│  [core/levels_info.py]              [core/pattern_encoder.py]  [core/splitter_protocol.py]  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2.6 SDS 指标（空间不连续性分数）

**SDS（结构扭曲分数）**验证 Hilbert 局部性保持：

$$\text{SDS} = \frac{1}{N^2} \sum_{i,j} \left| \|p_i - p_j\|_2 - \frac{|h_i - h_j|}{H_{\max}} \right|$$

其中：

- $p_i, p_j$：2D 物理坐标
- $h_i, h_j$：Hilbert 索引
- $H_{\max}$：网格的最大 Hilbert 索引

**解释**：

- SDS ≈ 0：完美的局部性保持（空间距离 ∝ Hilbert 距离）
- SDS >> 0：局部性扭曲（随机排序）

**源码**：`src/vit_pytorch/core/curve_hilbert.py` - `SDSMetric` 类

---

## 2.7 关键缩写

| 缩写 | 全称 | 上下文 |
|:------------|:---------|:--------|
| **SDS** | Spatial Discontinuity Score | Hilbert 局部性验证 |
| **LUT** | Look-Up Table | 预计算 Hilbert 映射 |
| **LCA** | Lowest Common Ancestor | 四叉树层次 |
| **H1SS** | Hilbert-Optimal Splitter (6 Axioms) | 默认分割器 |

---

## 2.8 文档导航

| 章节 | 内容 |
|:--------|:--------|
| [02_data_structures](02_data_structures.md) | 核心数学基础（本章） |
| [03_fractal_tokenizer](03_fractal_tokenizer.md) | 分词管道 |
| [04_positional_embedding](04_positional_embedding.md) | 位置编码 |

> **下一章**: [03_fractal_tokenizer.md](03_fractal_tokenizer.md) - 分形分词管道
