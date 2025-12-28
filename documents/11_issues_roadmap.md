# Chapter 11: Development History

> **Last Updated**: December 2024 | **Status**: Active Development

## 11.1 Overview

This chapter documents the evolution of the Fractal Curve Tokenizer project, including architecture decisions, performance optimizations, and lessons learned.

---

## 11.2 Architecture Evolution

### Timeline

| Version | Date | Architecture | Status |
|:--------|:-----|:-------------|:-------|
| V1 | Nov 2024 | Fixed multi-scale convolution | ⚠️ Removed |
| V2 | Dec 2024 | Gumbel-Softmax scale selection | ⚠️ Removed |
| V3 | Dec 2024 | Variable Depth Tokens (quadtree) | ✓ Current |

### V1: Fixed Multi-Scale (Removed)

**Approach**: Fixed convolutional pyramid with predetermined scale ratios.

**Limitation**: No content adaptation—same tokenization for all images.

### V2: Gumbel-Softmax (Removed)

**Approach**: Learnable scale selection via Gumbel-Softmax.

**Problem**: Scale collapse. Mathematical proof showed softmax weights inevitably converge to finest scale:

$$\lim_{t \to \infty} \alpha_{s_0} = 1, \quad \text{where } s_0 = \arg\max_s H(F_s)$$

**Observation**: At epoch 17, scale_0 reached 99.99% weight.

### V3: Variable Depth Tokens (Current)

**Approach**: Content-adaptive quadtree splitting with explicit complexity thresholds.

**Key insight**: Avoid softmax competition by using independent per-region split decisions:

$$\text{Split}(R) \iff C(R) > \tau_d$$

**Result**: Stable depth distributions throughout training.

---

## 11.3 Key Improvements

### Architecture (ARCH)

| ID | Description | Status |
|:---|:------------|:-------|
| ARCH-P0 | Streaming tokenizer implementation | ✓ Done |
| ARCH-VDT | Variable Depth Tokens redesign | ✓ Done |
| ARCH-P1 | Deprecated module removal | ✓ Done |
| ARCH-R1 | Global context attention removed | ✓ Done |
| ARCH-R2 | Level aggregator added | ✓ Done |

### Performance (PERF)

| ID | Description | Impact | Status |
|:---|:------------|:-------|:-------|
| PERF-1 | LCA Hilbert Bias | 99.8% param reduction | ✓ Done |
| PERF-2 | SwiGLU FFN | 9.4% model size reduction | ✓ Done |
| PERF-3 | Attention mask vectorization | O(B×S²) → O(B) | ✓ Done |
| PERF-4 | Integral image caching | O(1) region stats | ✓ Done |
| PERF-5 | LCA computation caching | ~8× speedup | ✓ Done |

### Bug Fixes (P0)

| ID | Description | Status |
|:---|:------------|:-------|
| P0-1 | REINFORCE baseline (EMA) | ✓ Fixed |
| P0-2 | Attention mask propagation | ✓ Fixed |
| P0-3 | Hilbert bias mode config | ✓ Fixed |
| P0-4 | TokenizerOutput convenience properties | ✓ Fixed |
| P0-C1 | Cross-scale attention collapse | ✓ Replaced with VDT |

### Quality (P2)

| ID | Description | Status |
|:---|:------------|:-------|
| P2-1 | Test coverage (295+ tests) | ✓ Done |
| P2-2 | Data augmentation strategies | ✓ Done |
| P2-3 | Type annotations (98%) | ✓ Done |
| P2-4 | Mathematical docstrings | ✓ Done |

---

## 11.4 Lessons Learned

### 1. Softmax Scale Competition

**Problem**: When using softmax to weight multiple scales, the finest scale always wins because it retains the most information.

**Solution**: Use independent binary decisions per region instead of competitive softmax.

### 2. Gumbel-Softmax Gradient Sparsity

**Problem**: Straight-through estimator only propagates gradients to selected options.

**Solution**: Content-adaptive thresholding with differentiable complexity estimation.

### 3. LCA Distance vs Learned Embeddings

**Finding**: Simple LCA depth embedding (~100 params) outperforms complex learned path embeddings (~50K params).

**Reason**: LCA depth has explicit geometric meaning (spatial distance), requiring less learning.

### 4. FFN Capacity

**Problem**: Original mlp_dim = 2× dim was too small.

**Solution**: mlp_dim = 4× dim (standard ViT ratio) improved accuracy 3-5%.

### 5. Regularization Balance

**Problem**: dropout=0.3 + drop_path=0.2 caused under-fitting.

**Solution**: Reduced to 0.1 each, improving accuracy 3-5%.

---

## 11.5 Removed Components

The following components have been removed from the codebase:

| Component | Reason |
|:----------|:-------|
| `FractalHilbertTokenizer` | Replaced by streaming tokenizer |
| `EnhancedFractalTokenProcessor` | Merged into V3 |
| `StreamingFractalTokenizer` (V1) | Superseded by V3 |
| `StreamingFractalTokenizerV2` | Scale collapse issue |
| `CrossScaleAttention` | Mathematical collapse proven |
| `_deprecated/` directory | Cleanup complete |
| `original` bias mode | Replaced by LCA |

---

## 11.6 Current Recommendations

### Model Configuration

```python
model = FractalCurveViT(
    image_size=224,
    num_classes=1000,
    dim=384,
    depth=6,
    heads=6,
    mlp_dim=768,                    # 4× dim
    dropout=0.1,                    # Reduced from 0.3
    tokenizer_type='streaming_v3',  # V3 only
    hilbert_bias_mode='lca',        # ~100 params
    ffn_type='swiglu_level',        # SwiGLU + level adapt
)
```

### Training Configuration

```python
optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=0.03)
scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[5])
```

---

## 11.7 Future Directions

| Area | Description | Priority |
|:-----|:------------|:---------|
| Hierarchical attention | Multi-resolution attention patterns | Medium |
| 3D extension | Video and volumetric data | Low |
| Efficiency | FlashAttention integration | High |
| Scaling | Larger model variants | Medium |

---

## 11.8 P5: 性能优化 Issues (已结案)

> **状态**: ✅ 79% 已完成 (11/14), 3项延后  
> **结案日期**: 2025-12-27

| ID | 问题 | 状态 | 说明 |
|----|------|------|------|
| P5-1 | `_embed_batch` 向量化 | ✅ | ROI-Align 批量池化 |
| P5-2 | IntegralImageCache 批量 | ✅ | 支持 [B,C,H,W] 输入 |
| P5-3 | Hilbert 缓存扩容 | ✅ | 1024→4096, 64→256 |
| P5-4 | SwiGLU bias 参数化 | ✅ | bias=False 选项 |
| P5-5 | 深度调制范围扩展 | ✅ | beta=0.2, [1.0,1.2] |
| P5-6 | 全局上下文缩放 | ✅ | 随 ARCH-R1 删除 |
| P5-7 | HierarchicalBias 低秩 | ⏸️ | 非默认路径，延后 |
| P5-8 | levels_info 格式统一 | ✅ | HilbertBiasBase 基类 |
| P5-9 | 注意力掩码命名 | ⏸️ | 影响大收益小，延后 |
| P5-10 | 分割参数域验证 | ✅ | 域适应预设工厂 |
| P5-11 | torch.compile 兼容 | ⏸️ | 待 P4 后处理 |
| P5-12 | 复杂度文档补全 | ✅ | 添加时空复杂度 |
| P5-13 | level_attention_bias 示例 | ✅ | API 文档示例 |
| P5-14 | chunk_size 可配置化 | ✅ | 已有参数 |

---

## 11.9 P6: 尺度利用能力 Issues (已结案)

> **状态**: ✅ 80% 已完成 (4/5), 1项延后  
> **结案日期**: 2025-12-26

**核心发现**: 模型深度信号强度初始化过于保守

| ID | 问题 | 状态 | 说明 |
|----|------|------|------|
| P6-1 | depth_scale 范围扩展 | ✅ | sigmoid 参数化 [0.5,2.0] |
| P6-2 | LCA Bias 温度参数 | ✅ | 可学习温度 τ=1.5 |
| P6-3 | Depth Attention 可视化 | ✅ | DepthAttentionAnalyzer |
| P6-4 | Depth Embedding 分析 | ✅ | DepthEmbeddingAnalyzer |
| P6-5 | Depth Contrastive Loss | 🔵 | 研究性，延后 |

**关键公式**:
- depth_scale: $\sigma_d = \sigma_{min} + (\sigma_{max} - \sigma_{min}) \cdot \text{sigmoid}(\gamma_d)$
- LCA temperature: $B'_{h,i,j} = \tau_h \cdot B_{LCA(i,j)}$

---

## 11.10 P7: 可学习分割器 Issues (已结案)

> **状态**: ✅ 100% 已完成 (7/7)  
> **结案日期**: 2025-12-27

**核心问题**: 规则分割器存在复杂度饱和、阈值不匹配、梯度阻断三大缺陷

| ID | 问题 | 解决方案 |
|----|------|----------|
| P7-1 | 复杂度函数饱和 | 可学习 MLP 复杂度预测器 |
| P7-2 | 阈值-复杂度不匹配 | 可学习阈值向量 τ |
| P7-3 | 离散决策梯度阻断 | Gumbel-Softmax + STE |
| P7-4 | LearnableSplitter 实现 | 新增 ~400 行代码 |
| P7-5 | TokenizerV3 集成 | split_scheme='learnable' |
| P7-6 | 训练损失扩展 | 熵正则化 + 预算约束 |
| P7-7 | 温度退火调度 | 指数退火 T_start→T_end |

**数学形式化**:
- 复杂度: $C_\theta(R) = \sigma(\text{MLP}(\text{ROI-Pool}(F, R)))$
- 分割决策: $p_{split} = \sigma((C_\theta(R) - \tau_d) / T)$
- Gumbel-Softmax: $y = \text{softmax}((\log[1-p, p] + [g_0, g_1]) / \tau)$

---

## 11.11 P8: Splitter 深度优化 Issues (已结案)

> **状态**: ✅ 100% 已完成 (5/5)  
> **结案日期**: 2025-12-28

**核心发现**: P7 实现完成基本功能，存在5个可优化问题

| ID | 问题 | 解决方案 | 性能收益 |
|----|------|----------|----------|
| P8-1 | 多层可微分损失 | `get_multi_layer_depth_loss()` | 深层阈值梯度 |
| P8-2 | 并行化递归分割 | BFS层级批量评估 | O(D) 调用 |
| P8-3 | 空间索引加速 | `SpatialIndex` 类 | 8.4x 加速 |
| P8-4 | STE 梯度路径增强 | REINFORCE 策略梯度 | 梯度质量↑ |
| P8-5 | 动态温度调度 | `TemperatureScheduler` | 易用性↑ |

**验证结果**:
- BFS 与递归版本输出完全一致
- 邻居查询: 0.0303ms → 0.0036ms (8.4x)
- 阈值梯度: 0 → 非零

---

## 11.12 废弃组件存档

| 组件 | 移除原因 | 移除日期 |
|------|----------|----------|
| `StreamingFractalTokenizer` (V1) | 被 V3 取代 | 2025-12-26 |
| `StreamingFractalTokenizerV2` | 尺度崩塌问题 | 2025-12-25 |
| `CrossScaleAttention` | 数学崩塌证明 | 2025-12-25 |
| `FractalHilbertTokenizer` | 被流式替代 | 2025-12-24 |
| `EnhancedFractalTokenProcessor` | 合并至 V3 | 2025-12-24 |
| `global_context_attn` | ARCH-R1 删除 | 2025-12-26 |

> **Next**: [appendix.md](appendix.md) - Appendix
