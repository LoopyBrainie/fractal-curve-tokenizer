# 第十一章：项目改进历史

> **最后更新**: 2025年12月14日 | **状态**: ✅ 所有改进已完成

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

### ARCH-P0: StreamingFractalTokenizer (2025-12)

**问题**: 原 `FractalHilbertTokenizer` 使用 BFS + REINFORCE，训练不稳定

**解决方案**: 
- 实现 `StreamingFractalTokenizer` (V1): 固定多尺度卷积
- 实现 `StreamingFractalTokenizerV2` (V2): Gumbel-Softmax 自适应

**结果**:
- 训练完全端到端可微
- 消除 Python 循环瓶颈
- 训练稳定性显著提升

### ARCH-P1: 废弃模块隔离 (2025-12)

**操作**:
- 将 `FractalHilbertTokenizer` 移至 `_deprecated/`
- 将 `EnhancedFractalTokenProcessor` 移至 `_deprecated/`
- 实现 `__getattr__` 延迟导入 + DeprecationWarning

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
| P2-1 | 测试覆盖增强 (+39 用例) | ✅ 2025-12-03 |
| P2-2 | 数据增强策略 (MNIST/CIFAR/ImageNet) | ✅ 2025-12-05 |
| P2-3 | 类型注解补全 | ✅ 2025-12-08 |
| P2-4 | 模块文档字符串数学形式化 | ✅ 2025-12-14 |

---

## 11.5 代码架构解耦 (P3)

| 项目 | 描述 | 状态 |
|------|------|------|
| P3-1 | 模块拆分 (token_processor → _deprecated) | ✅ 2025-12-07 |
| P3-2 | 工具函数统一 (extract_depths, normalize_levels_info) | ✅ 2025-12-07 |
| P3-3 | 常量提取到 constants.py | ✅ 2025-12-08 |

---

## 11.6 当前架构层级

```
Layer 4 (应用层):
    fractal_vit.py          NextGenerationFractalViT, SimpleFractalViT

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

废弃模块 (_deprecated/):
    fractal_curve_tokenizer.py  BFS + REINFORCE (v1.0 移除)
    token_processor.py          功能已集成 (v1.0 移除)
```

---

## 11.7 未来路线图

### 待完成 (Pending)

| 项目 | 描述 | 优先级 |
|------|------|------|
| E1 | Tokenizer 消融实验 (legacy vs streaming) | P1 |
| E2 | Hilbert Bias 消融实验 (original vs low_rank) | P1 |
| E3 | FFN 消融实验 (gelu vs swiglu) | P2 |
| E4 | 位置编码消融实验 (depth vs path) | P2 |

### 长期目标

- [ ] ImageNet 完整训练
- [ ] 检测/分割任务适配
- [ ] 模型压缩与量化
- [ ] ONNX 导出

---

## 11.8 相关文档

- **架构概览**: [00_introduction.md](00_introduction.md)
- **Tokenizer 详解**: [03_fractal_tokenizer.md](03_fractal_tokenizer.md)
- **训练指南**: [09_training_system.md](09_training_system.md)
- **测试说明**: [10_testing_qa.md](10_testing_qa.md)

---

**项目状态**: ✅ 生产就绪 | **测试**: 120+ 通过 | **技术债务**: 0 项
