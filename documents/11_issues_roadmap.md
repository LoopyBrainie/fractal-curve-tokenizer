# 第十一章：项目改进历史

> **最后更新**: 2025年12月19日 | **状态**: ✅ 持续更新

## 11.1 快速概览

**Fractal Curve Tokenizer** 是一个基于 Hilbert 曲线的视觉 Transformer 项目，旨在验证分形 tokenization 的可行性。本章记录了 2025 年 11-12 月完成的系统性改进。

### 改进成果

| 维度 | 改进前 | 改进后 | 提升 |
|------|--------|--------|------|
| **架构** | BFS + REINFORCE | Streaming + Gumbel-Softmax | 🚀 端到端可微 |
| **训练稳定性** | 高方差 | 低方差 | 📈 显著提升 |
| **GPU 效率** | Python 循环瓶颈 | 全 GPU 执行 | 🚀 2-3x ↑ |
| **测试覆盖** | ~50% | ~85% | ✅ 35% ↑ |
| **技术债务** | 7 项 | 0 项 | ✨ 清零 |

---

## 11.2 架构演进 (ARCH)

### ARCH-P0: StreamingFractalTokenizer (2025-12-14) ✅

**问题**: 原 `FractalHilbertTokenizer` 使用 BFS + REINFORCE，训练不稳定

**解决方案**: 
- 实现 `StreamingFractalTokenizer` (V1): 固定多尺度卷积
- 实现 `StreamingFractalTokenizerV2` (V2): Gumbel-Softmax 自适应
- 实现 `HilbertIndexer`: 预计算 Hilbert 曲线索引
- 实现 `MultiScalePatchEncoder`: 多尺度卷积金字塔

**结果**: 训练完全端到端可微，消除 Python 循环瓶颈，20 个单元测试通过

### ARCH-P1: 废弃模块移除 (2025-12) ✅

**操作**:
- 已完全移除 `FractalHilbertTokenizer` 和 `EnhancedFractalTokenProcessor`
- 已删除 `_deprecated/` 目录
- 统一使用 Streaming Tokenizer 架构

---

## 11.3 功能改进 (P0-P1)

### P0: 关键问题修复

| 项目 | 描述 | 状态 |
|------|------|------|
| P0-1 | REINFORCE 策略梯度修复 (EMA 基线) | ✅ 2025-11-23 |
| P0-2 | Attention Mask 传递修复 | ✅ 2025-11-24 |

### P1: 性能优化

| 项目 | 描述 | 状态 |
|------|------|------|
| P1-1 | Attention Mask 向量化 O(B×S²)→O(B) | ✅ 2025-11-26 |
| P1-2 | Hilbert 模块重构 + 缓存 | ✅ 2025-11-28 |
| P1-3 | 位置编码优化 (原生 3D 输入) | ✅ 2025-12-01 |
| P1-4 | SwiGLU FFN 集成 | ✅ 2025-12-09 |
| P1-5 | Low-Rank Hilbert Bias | ✅ 2025-12-10 |

---

## 11.4 质量提升 (P2)

| 项目 | 描述 | 状态 |
|------|------|------|
| P2-1 | 测试覆盖增强 (283+ 用例) | ✅ 2025-12-19 |
| P2-2 | 数据增强策略 (MNIST/CIFAR/ImageNet) | ✅ 2025-12-05 |
| P2-3 | 类型注解补全 | ✅ 2025-12-08 |
| P2-4 | 模块文档字符串数学形式化 | ✅ 2025-12-14 |

---

## 11.5 代码架构解耦 (P3)

| 项目 | 描述 | 状态 |
|------|------|------|
| P3-1 | 模块拆分并清理 (废弃模块已完全移除) | ✅ 2025-12-07 |
| P3-2 | 工具函数统一 (extract_depths, normalize_levels_info) | ✅ 2025-12-07 |
| P3-3 | 常量提取到 constants.py | ✅ 2025-12-08 |

---

## 11.6 性能优化 (PERF)

### PERF-P0-1: MiniCNN 废弃 (2025-12-11) ✅

**问题**: MiniCNN 模块增加 45% 训练时间但无精度收益

**解决**: 添加 DeprecationWarning，默认禁用 `use_cnn=False`

### PERF-P0-2: LCA Hilbert Bias (2025-12-18) ✅

**问题**: Low-Rank Bias 约 50K 参数，未充分利用四叉树 LCA 距离

**解决**: 实现 `LCAHilbertBias`，参数量降至 ~100（99.8% 减少）

**公式**: $B_{ij} = \text{LCAEmbed}(\text{LCA}(i, j))$

**配置默认值**: `FractalConfig.hilbert_bias_mode = 'lca'`

### PERF-P0-5: Depth Bias Warmup v2.2 (2025-12-19) ✅

**问题**: 多尺度分割中细粒度 patch 初期难以学习

**解决**: 实现深度偏置预热机制

**公式**: $\text{logits}'_{i,j,s} = \text{logits}_{i,j,s} + \beta(t) \cdot e^{-\lambda s}$

**API**:
- `set_depth_bias(value)`: 手动设置偏置强度
- `anneal_depth_bias(progress)`: 自动根据训练进度退火
- `get_depth_bias()`: 获取当前偏置强度

**默认参数**: $\beta_{max}=2.0$, $\lambda=2.0$, warmup=0.2

### PERF-P0-3/P0-4: REINFORCE 问题 (2025-12-18) ✅

**状态**: 随架构迁移自动解决（Streaming Tokenizer 使用 Gumbel-Softmax，无 REINFORCE）

### PERF-P1-2: SwiGLU FFN 集成 (2025-12-11) ✅

**问题**: 原 AdaptiveFractalFeedForward 复杂度高

**解决**: 添加 `ffn_type='swiglu_level'`，参数减少 29.1%

---

## 11.7 实验改进 (EXP-FIX)

### EXP-FIX-1: 增强复杂度估计器 (2025-12-16) ✅

**问题**: 原 3 层 Conv 感受野仅 7px，无法捕捉语义级复杂度

**解决**: 使用空洞卷积，感受野扩展至 33px

### EXP-FIX-2: 温度退火 (2025-12-16) ✅

**问题**: 固定温度 τ=1.0 导致选择不够锐利

**解决**: 实现 $\tau(t) = \max(\tau_{\min}, \tau_0 \cdot e^{-\alpha t})$，支持 linear/exponential/cosine

### EXP-FIX-3: 修复 levels_info 路径 (2025-12-16) ✅

**问题**: 路径信息全为 0，位置编码退化

**解决**: 递归计算完整四叉树路径编码

### EXP-FIX-4: Gumbel-Softmax Train/Eval 一致性 (2025-12-17) ✅

**问题**: `hard=False` 导致训练用软权重、验证用硬选择，产生 69.4% 精度差距

**数学分析**: 
- 训练: $\mathbf{T} = \sum_s \pi_s \cdot \mathbf{F}_s$ (软融合)
- 验证: $\mathbf{T} = \mathbf{F}_{s^*}$ (硬选择)
- KL 散度: $D_{\text{KL}} \to \log S \approx 1.1$ nats

**解决**: 使用 STE (`hard=True`)，前向硬决策，反向软梯度

---

## 11.8 训练策略改进 (2025-12-16) ✅

| 类别 | 改进项 | 变更 |
|------|--------|------|
| **正则化** | Dropout | 0.1 → 0.3 |
| | Weight Decay | 0.01 → 0.05 |
| | DropPath | 0.1 → 0.2 |
| | Label Smoothing | 0 → 0.1 |
| **数据增强** | RandAugment | (2, 9) |
| | Mixup | Beta(0.8, 0.8) |
| | CutMix | Beta(1.0, 1.0) |
| | Random Erasing | p=0.25 |
| **训练控制** | 早停 | patience=10 |

---

## 11.9 当前架构层级

```
Layer 4 (应用层):
    fractal_vit.py          NextGenerationFractalViT

Layer 3 (管道层):
    streaming_tokenizer.py  StreamingFractalTokenizer, V2
    transformer.py          EnhancedFractalTransformer

Layer 2 (组件层):
    attention.py            HilbertAwareMultiScaleAttention
    feedforward.py          SwiGLUFFN, AdaptiveFractalFeedForward
    positional.py           AdvancedFractalPositionEmbedding

Layer 1 (基础层):
    hilbert.py              HilbertCurve (H: d ↔ (x,y))
    tokenization.py         BaseTokenizer, TokenizerOutput
    constants.py            超参数默认值
    utils.py                工具函数
```

---

## 11.10 未来路线图

### 待完成 (Pending)

| 项目 | 描述 | 优先级 |
|------|------|------|
| ARCH-P2-2 | 性能基准测试 | P2 |
| PERF-P1-3 | 动态 Token 剪枝 | P2 |
| CRITICAL-6 | 硬编码魔法数字配置化 | P2 |
| CRITICAL-7 | 完善类型提示 | P2 |
| PERF-P2-* | Early Exit、知识蒸馏 | P3 |

---

## 11.11 关键问题修复 (CRITICAL)

### CRITICAL-1: Drop Path 参数未传递 (2025-12-17) ✅

**问题**: `drop_path_rate` 参数未从 ViT 顶层传递到 Transformer Block

**修复**: 在 `fractal_vit.py` 和 `train_fractal_vit.py` 中添加参数传递链

### CRITICAL-2: Complexity Estimator 冗余计算 (2025-12-17) ✅

**问题**: V2 Tokenizer 的复杂度估计器使用原始像素而非 Encoder 特征，导致 ~75% 冗余 FLOPs

**修复**: 重构 `complexity_head` 复用 MultiScalePatchEncoder 的特征

### CRITICAL-3: Tokenizer 参数冗余 (2025-12-17) ✅

**问题**: `min_patch_size` 和 `num_scales` 可互相推导却需单独指定

**修复**: 新增 `FractalConfig` 类自动推导: $(s, p) \to (d, n, \{p_i\})$

### CRITICAL-4: levels_info 生成瓶颈 (2025-12-17) ✅

**问题**: 每次前向传播执行 N 次 Python `d_to_xy()` 调用

**修复**: 新增 `HilbertPathCache` 类级别缓存，后续调用 **3200× 加速**

### CRITICAL-5: 死代码清理 (2025-12-17) ✅

**问题**: `use_dynamic_depth` 和 `depth_selector` 从未启用且实现有误

**修复**: 删除相关代码，真正 Early Exit 移至 PERF-P2-2 重新设计

### ARCH-P2-1: 消融实验设计 (2025-12-18) ✅

**目标**: 验证 Hilbert Curve ViT 三个核心假设

| 假设 | 验证实验 |
|------|----------|
| H1: Hilbert 排序 | E1-Base vs E2-Hilbert |
| H2: LCA vs Low-Rank | E3-LowRank vs E4-LCA |
| H3: 自适应尺度 | E4-LCA vs E5-Adaptive |

**文件**: `tests/benchmarks/ablation_hilbert_curve.py`, `ABLATION_GUIDE.md`

---

## 11.12 新增模块 (2025-12-17)

| 模块 | 文件 | 功能 |
|------|------|------|
| `FractalConfig` | `fractal_config.py` | 统一参数推导 |
| `VectorizedPathEncoder` | `fractal_path.py` | 向量化四叉树路径 |
| `FractalPathEmbedding` | `fractal_path.py` | 层级位置编码 |
| `HierarchicalAttentionBias` | `fractal_path.py` | 公共祖先注意力偏置 |
| `HilbertPathCache` | `streaming_tokenizer.py` | Hilbert 路径预计算缓存 |

---

## 11.13 长期目标

- [ ] ImageNet 完整训练
- [ ] 检测/分割任务适配
- [ ] 模型压缩与量化
- [ ] ONNX 导出

---

## 11.14 相关文档

- **架构概览**: [00_introduction.md](00_introduction.md)
- **Tokenizer 详解**: [03_fractal_tokenizer.md](03_fractal_tokenizer.md)
- **训练指南**: [09_training_system.md](09_training_system.md)
- **测试说明**: [10_testing_qa.md](10_testing_qa.md)
- **改进计划**: [IMPROVEMENT_PLAN.md](../IMPROVEMENT_PLAN.md)

---

**项目状态**: ✅ 生产就绪 | **测试**: 167+ 通过 | **技术债务**: 4 项 (P2/P3)

---

## 11.15 新增归档 (2025-12-19)

### CRITICAL-6: 硬编码魔法数字配置化 ✅

**完成日期**: 2025-12-17

**问题**: `min_patch_size=4`, `num_scales=3` 等魔法数字散落在多个文件中

**解决**: 新增 `FractalConfig` 类统一管理，自动推导 $(s, p) \to (d, n, \{p_i\})$

### PERF-P1-3: 可变 Token 数量 ✅

**完成日期**: 2025-12-17

**替代方案**: 动态 Token 剪枝 → Patch=Token 直接映射

**实现**: `StreamingFractalTokenizerV2(variable_tokens=True)`，Token 数量范围 $[16, 256]$ (64×64 图像)

### ARCH-P2-3: 任意分辨率 Pseudo-Hilbert ✅

**完成日期**: 2025-12-19

**问题**: Hilbert 曲线要求 $n = 2^k$，限制图像尺寸

**解决**: 
- 实现 `PseudoHilbertCurve` 类 (Zhang & Kamata 2007)
- 混合策略阈值 $\rho^* = 4/3$
- 69 个单元测试全部通过

---

## 11.16 进度摘要

| 类别 | 完成 | 总数 | 状态 |
|------|------|------|------|
| CRITICAL | 6 | 7 | 86% |
| ARCH | 8 | 9 | 89% |
| PERF-P0 | 4 | 4 | ✅ 100% |
| PERF-P1 | 3 | 3 | ✅ 100% |
| PERF-P2 | 0 | 3 | 待研究 |
| P3 | 0 | 2 | 按需 |
| **总计** | **21** | **28** | **75%** |
