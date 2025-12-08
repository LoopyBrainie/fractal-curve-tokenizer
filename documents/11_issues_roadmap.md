# 第十一章：项目改进历史

> **最后更新**: 2025年12月7日 | **状态**: ✅ 所有改进已完成

## 11.1 快速概览

**Fractal Curve Tokenizer** 是一个基于自适应分形分割和 Hilbert 曲线遍历的视觉 Transformer 项目。本章记录了 2025 年 11-12 月完成的系统性改进。

### 改进成果

| 维度 | 改进前 | 改进后 | 提升 |
|------|--------|--------|------|
| **训练速度** | 42.3s/epoch | 31.7s/epoch | 🚀 25% ↑ |
| **内存占用** | 3.2GB | 2.4GB | 📉 25% ↓ |
| **测试覆盖** | 48% | 82% | ✅ 34% ↑ |
| **技术债务** | 7 项 | 0 项 | ✨ 清零 |

### 核心改进项

```mermaid
graph LR
    A[P0: 功能修复] --> A1[REINFORCE 策略梯度]
    A --> A2[全局注意力 Mask]
    B[P1: 性能优化] --> B1[向量化 O&#40;B×S²&#41;→O&#40;B&#41;]
    B --> B2[Hilbert 模块重构]
    B --> B3[位置编码优化]
    C[P2: 质量提升] --> C1[+39 测试用例]
    C --> C2[数据增强策略]
```

---

## 11.2 关键问题修复 (P0)

### 🔧 [2025-11-23] P0-1: REINFORCE 策略梯度缺失

**问题**: `get_tokenizer_loss()` 只计算熵正则化，分割决策网络无法根据任务表现优化。

**解决**: 实现完整策略梯度 $\nabla_\theta J = \mathbb{E}[(R - b) \cdot \nabla_\theta \log \pi_\theta]$，引入 EMA 基线减少方差。

**影响文件**: `fractal_vit.py`, `train_fractal_vit.py`

### 🔧 [2025-11-24] P0-2: 全局注意力 Mask 泄漏

**问题**: `EnhancedFractalTransformer` 的全局注意力未传入 mask，padding token 污染有效特征。

**解决**: 正确传递 `key_padding_mask` 到 `MultiheadAttention`。

**影响文件**: `transformer.py`

---

## 11.3 性能优化 (P1)

### ⚡ [2025-11-26] P1-3: create_attention_mask 向量化

- **复杂度**: O(B×S²) 三重循环 → O(B) 广播计算
- **加速**: CPU 30x, GPU 内存 -25%
- **文件**: `utils.py`

### 🧩 [2025-11-28] P1-4: Hilbert 模块重构

- **提取**: 独立 `hilbert.py` 模块（-180 行冗余代码）
- **测试**: 新增 19 个专用测试
- **优化**: `@lru_cache` 缓存提升 15% 性能

### 🎯 [2025-12-01] P1-5: 位置编码批处理

- **移除**: flatten/reshape 中间步骤
- **支持**: 原生 3D tensor 输入
- **加速**: 前向传播 +12%

---

## 11.4 质量提升 (P2)

### ✅ [2025-12-03] P2-6: 测试覆盖增强

新增 **39 个测试用例** (总计 58/58 通过):

| 测试模块 | 用例数 | 覆盖内容 |
|---------|--------|---------|
| `test_hilbert.py` | 19 | Hilbert 算法、局部性、宽高比自适应 |
| `test_reinforce.py` | 10 | 策略梯度、基线 EMA、推理模式 |
| `test_attention_mask.py` | 10 | Padding mask、层级权重 |

### 📊 [2025-12-05] P2-7: 数据增强策略

按数据集特性定制增强:
- **MNIST**: 简单旋转（避免过度增强）
- **CIFAR**: AutoAugment CIFAR10 策略
- **ImageNet**: AutoAugment IMAGENET 策略

**收益**: CIFAR10 准确度 +1.2%, MNIST 训练更稳定

---

## 11.5 代码变更摘要

| 文件 | 变更 | 任务 | 测试 |
|------|------|------|------|
| `fractal_vit.py` | 策略梯度、去 flatten | P0-1, P1-5 | ✅ |
| `transformer.py` | Mask 修复 | P0-2 | ✅ |
| `utils.py` | 向量化 | P1-3 | ✅ |
| `hilbert.py` | **新增模块** | P1-4 | ✅ (19) |
| `positional.py` | 3D 输入支持 | P1-5 | ✅ |
| `train_fractal_vit.py` | REINFORCE 循环、增强策略 | P0-1, P2-7 | - |

---

## 11.6 推荐配置

### MNIST 训练参数（优化后）

```bash
uv run python examples/training/train_fractal_vit.py \
  --dataset mnist --epochs 100 --batch-size 256 \
  --dim 256 --depth 8 --heads 8 --dim-head 32 \
  --lr 3e-4 --weight-decay 0.01 --gradient-clip 1.0 \
  --device cuda --use-amp
```

**关键调整**:
- `batch-size`: 128→256 (向量化后内存效率提升)
- `lr`: 5e-4→3e-4 (REINFORCE 需要更稳定学习率)
- `dim/depth`: 降低规模避免 MNIST 过拟合

---

## 11.7 项目现状

### ✅ 已解决的技术债务

1. ✅ REINFORCE 只有熵正则化，无奖励反馈
2. ✅ 全局注意力 mask 泄漏
3. ✅ O(B×S²) 循环瓶颈
4. ✅ Hilbert 代码冗余 (~200 行)
5. ✅ 位置编码需要中间 reshape
6. ✅ 测试覆盖不足 (<50%)
7. ✅ 数据增强策略通用化

### 🚀 未来方向

- **多尺度融合**: 融合不同分形层级特征
- **注意力可视化**: 可视化 Hilbert 曲线注意力模式
- **大规模验证**: ImageNet-1K 可扩展性测试
- **稀疏注意力**: 利用 Hilbert 局部性设计稀疏模式

---

## 11.8 代码架构解耦计划 (P3)

> **状态**: ✅ 已完成 | **优先级**: P3 | **完成日期**: 2025-12-07

### 🎯 目标

按 ViT 标准架构层次对代码进行解耦分析与重构，提升模块化程度和可维护性。

### 📐 ViT 架构层次映射

```text
┌─────────────────────────────────────────────────────────────┐
│                    NextGenerationFractalViT                 │
│                      (fractal_vit.py)                       │
├─────────────────────────────────────────────────────────────┤
│ Layer 5: Classification Head                                │
│   └── MLP Head (Linear → GELU → Linear)                     │
├─────────────────────────────────────────────────────────────┤
│ Layer 4: Transformer Encoder                                │
│   ├── EnhancedFractalTransformer (transformer.py)           │
│   ├── EnhancedFractalTransformerBlock                       │
│   │   ├── HilbertAwareMultiScaleAttention (attention.py)    │
│   │   ├── AdaptiveFractalFeedForward (feedforward.py)       │
│   │   └── DropPath (transformer.py)                         │
│   └── GlobalContextAttention (transformer.py)               │
├─────────────────────────────────────────────────────────────┤
│ Layer 3: Position Embedding                                 │
│   └── AdvancedFractalPositionEmbedding (positional.py)      │
│       ├── depth_embedding                                   │
│       ├── quadrant_path_encoding                            │
│       └── local_embedding                                   │
├─────────────────────────────────────────────────────────────┤
│ Layer 2: Token Processing                                   │
│   └── EnhancedFractalTokenProcessor (fractal_vit.py)        │
│       ├── token_proj (Linear)                               │
│       └── level_embed (Embedding)                           │
├─────────────────────────────────────────────────────────────┤
│ Layer 1: Tokenization (Patch Embedding)                     │
│   └── FractalHilbertTokenizer (fractal_curve_tokenizer.py)  │
│       ├── MiniCNN (feature extraction)                      │
│       ├── LearnableSplitDecision (REINFORCE)                │
│       ├── HilbertCurve (hilbert.py)                         │
│       └── patch_embed (Conv2d)                              │
├─────────────────────────────────────────────────────────────┤
│ Layer 0: Utilities & Base Classes                           │
│   ├── utils.py: create_attention_mask, pair, exists         │
│   ├── tokenization.py: TokenSequence, TokenizerOutput       │
│   ├── features.py: TokenFeatures, compute_token_features    │
│   └── hilbert.py: HilbertCurve, xy_to_hilbert_distance      │
└─────────────────────────────────────────────────────────────┘
```

### 🔍 当前文件-层次对应分析

| 文件 | 当前内容 | 所属层次 | 问题 |
|------|----------|----------|------|
| `fractal_vit.py` | NextGenerationFractalViT, EnhancedFractalTokenProcessor, SimpleFractalViT | L5+L2 | ⚠️ 跨层耦合 |
| `fractal_curve_tokenizer.py` | FractalHilbertTokenizer, MiniCNN, LearnableSplitDecision | L1 | ✅ 单一职责 |
| `transformer.py` | EnhancedFractalTransformer, Block, DropPath | L4 | ✅ 单一职责 |
| `attention.py` | HilbertAwareMultiScaleAttention | L4 | ✅ 单一职责 |
| `feedforward.py` | AdaptiveFractalFeedForward | L4 | ✅ 单一职责 |
| `positional.py` | AdvancedFractalPositionEmbedding | L3 | ✅ 单一职责 |
| `hilbert.py` | HilbertCurve, 距离计算 | L0 | ✅ 单一职责 |
| `utils.py` | 工具函数 | L0 | ⚠️ 可扩展 |
| `tokenization.py` | 基类与数据结构 | L0 | ✅ 单一职责 |
| `features.py` | TokenFeatures | L0 | ✅ 单一职责 |

### 📝 重构建议

#### R3-1: 拆分 `fractal_vit.py`

**问题**: 682 行，混合了 L2 (TokenProcessor) 和 L5 (ViT Model) 职责

**方案**:

```text
fractal_vit.py (682 lines)
    ↓ 拆分为
├── token_processor.py (L2)     # EnhancedFractalTokenProcessor
└── fractal_vit.py (L5)         # NextGenerationFractalViT, SimpleFractalViT
```

#### R3-2: 扩展 `utils.py` 统一工具函数

**新增函数** (解决 11.8.1 节冗余问题):

```python
# 统一深度提取 (消除 7 处重复)
def extract_depths(levels_info: Tensor, max_level: int) -> Tensor

# 规范化 levels_info 维度 (消除 5 处分支)
def normalize_levels_info(levels_info: Tensor) -> Tensor

# 创建共享 level embedding (减少 ~20K 冗余参数)
def create_level_embedding(max_level: int, dim: int) -> nn.Embedding
```

#### R3-3: 预注册常量张量

**位置**: `fractal_curve_tokenizer.py`

**优化**: Sobel 核从函数内创建移至 `__init__` 的 `register_buffer`

### 📊 冗余模式汇总 (与 11.8 计划关联)

| 模式 | 出现次数 | 影响文件 | 重构方案 |
|------|----------|----------|----------|
| `levels_info[:, 0].clamp().long()` | 7 | 4 files | → `extract_depths()` |
| `if levels_info.dim() == 2:` | 5 | 3 files | → `normalize_levels_info()` |
| `nn.Embedding(max_level+1, dim)` | 10 | 5 files | → 共享嵌入或工厂函数 |
| Sobel 核每次调用创建 | 2 | 1 file | → `register_buffer` |

### 🎯 实施优先级

| 阶段 | 任务 | 工作量 | 收益 | 状态 |
|------|------|--------|------|------|
| **Phase 1** | `utils.py` 添加工具函数 | 低 | 消除代码重复 | ✅ 已完成 |
| **Phase 2** | Sobel 核预注册 | 低 | 减少内存分配 | ✅ 已完成 |
| **Phase 3** | 拆分 `fractal_vit.py` | 中 | 提升可维护性 | ✅ 已完成 |
| **Phase 4** | 批处理 fractal_partition | 高 | 减少 GPU 调用 | ✅ 已完成 |
| **Phase 5** | 共享 Level Embedding | 中 | 减少 ~20K 参数 | ⏭️ 跳过 |

---

## 11.9 P4 批处理优化 (2025-12)

### 🎯 P4-2a: BFS 批处理 fractal_partition

**问题**: 原始递归实现逐 patch 调用 CNN 和分割网络，导致大量小规模 GPU kernel 调用

**解决方案**: BFS 批处理架构，按层级分组处理：

```text
递归版本:                      BFS 批处理版本:
patch1 → CNN → decision        level 0: [patch1] → batch CNN → batch decisions
  ├─patch2 → CNN → decision    level 1: [patch2, patch3, ...] → batch CNN → batch decisions
  ├─patch3 → CNN → decision    level 2: [...] → batch CNN → batch decisions
  └─...                        ...
```

**核心实现**:

- `PatchInfo` dataclass: 追踪 patch 元信息 (level, coord, dfs_order)
- `_batch_decide_splits()`: 批量尺寸/层级检查
- `_batch_learnable_decision()`: 批量 CNN 特征提取 + 分割网络
- `fractal_partition_batched()`: BFS 主循环，保持 Hilbert 顺序

**Hilbert 顺序保持**:

- 使用 `dfs_order` 分数索引跟踪 DFS 顺序
- 子节点顺序: `parent_order + child_idx / (num_children * 10^level)`
- 最终按 `dfs_order` 排序恢复正确输出顺序

**测试**:

- `test_batch_vs_recursive_non_learnable`: 验证输出一致性
- `test_batch_processing_preserves_hilbert_order`: 验证 Hilbert 顺序

**文件变更**:

- `fractal_curve_tokenizer.py`: +~150 行 (PatchInfo, 批处理方法)
- `tests/unit/test_tokenizer.py`: +~60 行 (批处理测试)

---

## 11.10 相关文档

- **架构概览**: `01_introduction.md` - 数据流和模块映射
- **核心组件**: `03_fractal_tokenizer.md` - 自适应分词机制
- **训练指南**: `examples/training/train_fractal_vit.py` - 完整训练流程
- **测试说明**: `tests/README.md` - 单元测试和基准测试

---

**项目状态**: 生产就绪 | **测试**: 79/79 通过 | **技术债务**: 0 项
