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

## 11.12 P9: 训练性能瓶颈 Issues (已结案)

> **状态**: ✅ Phase 1-3 完成 (6/8), 2项延后  
> **结案日期**: 2025-12-28  
> **成果**: 训练速度从 24-25s/it 提升至 1.62s/it (14.8x 加速)

| ID | 问题 | 状态 | 说明 |
|----|------|------|------|
| P9-1 | Python 数据结构 GPU-CPU 同步阻塞 | ✅ | TensorSplitResult + 向量化 BFS |
| P9-2 | sort_tokens_vectorized CPU 回退 | ✅ | 与 P9-1 合并解决 |
| P9-3 | torch.tensor() 热路径反复创建 | ✅ | P9-1 隐式解决 |
| P9-4 | 重复 ROI-Align 计算 | ⚠️ | 需架构重构，暂不实施 |
| P9-5 | model._prepare_tokens 非向量化 | ✅ | 预填充缓存 |
| P9-6 | depth_distribution 重复计算 | ✅ | scatter_add 向量化 |
| P9-7 | get_multi_layer_depth_loss 每 batch 执行 | ⏸️ | 延后至训练优化阶段 |
| P9-8 | @torch._dynamo.disable 阻止编译 | ⏸️ | 延后 |

**核心重构**: `TensorSplitResult` 数据结构，完全消除 Python 循环

---

## 11.13 P10: 分割器训练失效 Issues (已结案)

> **状态**: ✅ P10-1~16 已完成, P10-17~20 待修复  
> **结案日期**: 2026-01-02  
> **待修复**: P10-17~20 (架构级改进)

### 已修复 Issues (P10-1~9)

| ID | 问题 | 解决方案 |
|----|------|----------|
| P10-1 | 主任务梯度无法回传 | STE 正确实现 (y_hard - y_soft.detach() + y_soft) |
| P10-2 | 初始化不稳定平衡点 | gain 从 0.1 → 1.0，扩大复杂度分布 |
| P10-3 | 极端分割振荡 | 与 P10-9 合并解决 |
| P10-4 | get_entropy_loss 无梯度 | 基于 BFS 路径的可微分软熵 |
| P10-5 | get_multi_layer_depth_loss 固定网格 | 与 P10-4 合并解决 |
| P10-6 | 阈值正则化仅约束范围 | 降为 P3 (软熵已提供隐式调整) |
| P10-7 | 温度退火过快 | 降为 P3 (P10-2 修复后临界区降至 22%) |
| P10-8 | 细分割捷径偏好 | P10-9 + P10-4/5 联合解决 |
| P10-9 | Elastic Budget 弹性预算 | 非对称 Dead Zone 惩罚机制 |

### 已修复 Issues (P10-10~16, 2026-01 新增)

| ID | 问题 | 解决方案 |
|----|------|----------|
| P10-10 | 软熵损失使用无梯度 EMA 概率 | 方案 A+D: forward() 始终缓存 p_split，阈值先验替代 EMA |
| P10-11 | 温度终点过低 (T_end=0.1) 导致梯度不稳定 | T_end 从 0.1 → 0.3，梯度放大从 10x 降至 3.3x |
| P10-12 | 训练初期 avg_tokens=1 的"鸡生蛋"问题 | 探索偏置 b=0.5 + 退火机制，P(split) 从 50% 提升到 70% |
| P10-13 | splitter_loss 负值导致语义混淆 | 使用 KL 散度替代负熵，确保损失始终非负 |
| P10-14 | 缺少 avg_tokens 异常检测机制 | `check_splitter_health()` + TensorBoard 监控 |
| P10-15 | 探索偏置退火与温度退火时序不协调 | 偏置退火延迟启动 + 与温度退火同步 |
| P10-16 | BFS 根节点单点失败导致全局崩塌 | 指数衰减深度偏置 β·γ^d (β=1.0, γ=0.5) |

**关键数学公式**:
- STE: $z_{ST} = z_{hard} - \text{sg}(y_{soft}) + y_{soft}$
- Elastic Budget: $L = \lambda_{over} \cdot \text{ReLU}(N-N_{max})^2 + \lambda_{under} \cdot \text{ReLU}(N_{min}-N)$
- 深度偏置: $\Delta b_d = \beta \cdot \gamma^d$ (根节点 +1.0，逐层衰减)
- KL 熵损失: $L = \log(D+1) - \tilde{H} + \text{anti\_collapse\_penalty}$

---

## 11.14 P11: 架构数学形式化审查 Issues (已结案)

> **状态**: ✅ 全部完成 (18/18)  
> **结案日期**: 2026-01-02

### P11 Issues 完整列表

| ID | 问题 | 解决方案 | 状态 |
|----|------|----------|------|
| P11-1 | LCA 缓存使用 data_ptr 存在碰撞风险 | WeakRef 缓存，使用 `is` 对象身份检查 | ✅ |
| P11-2 | Level-aware LayerNorm 参数爆炸 | 方案F: 动态深度 Embedding，从 tokenizer.max_depth 获取，参数节省 92%+ | ✅ |
| P11-3 | LowRankHilbertBias 路径维度截断 | 方案D: 从 regions 运行时计算路径，新增 `forward_from_regions()` | ✅ |
| P11-4 | level_scale_embedding 无正性约束 | Softplus 约束，初始化 softplus(0.54) ≈ 1.0 | ✅ |
| P11-5 | FractalPositionEmbedding 与 LCA 信息冗余 | 方案1: 删除死代码 `level_attention_bias` | ✅ |
| P11-6 | Elastic Budget EMA 冷启动问题 | 验证通过: 训练时使用缓存而非 EMA，无需修改 | ✅ |
| P11-7 | scale_weights 初始化偏离 1.0 | 初始化改为 softplus(0.54) ≈ 1.0 | ✅ |
| P11-8 | Hilbert Bias 多模式参数不一致 | 删除死代码: LowRankHilbertBias, HierarchicalHilbertBias | ✅ |
| P11-9 | 深度分布统计仍有 O(B) 循环 | scatter_add 向量化 | ✅ |
| P11-10 | ComplexityMLP 输出范围与阈值耦合 | 移除输出层 sigmoid，阈值初始化为 0，Barrier 边界扩展到 logit 空间 | ✅ |
| P11-11 | Gumbel-Softmax T→0.1 导致 STE 失效 | T_end 默认值从 0.1 → 0.3 + 安全下界保护 | ✅ |
| P11-12 | Hilbert 索引与 Quadtree 路径编码不一致 | 方案C: 设计决策确认，保持 Z-order LCA + Hilbert 排序，语义互补 | ✅ |
| P11-13 | depth_scale 语义歧义 | 方案A简化: 重命名 `_level_residual_embedding` → `_residual_gate` | ✅ |
| P11-14 | ROI-Align 对小区域采样退化 | 方案C: `min_region_size >= 2 * base_patch_size` | ✅ |
| P11-15 | level_mixing_weights 注释与实现不一致 | 更新注释: sigmoid 独立门控而非 softmax | ✅ |
| P11-16 | Transformer 双重 LayerNorm 不一致 | 方案B: levels_info=None 时默认处理为深度 0 | ✅ |
| P11-17 | quadrant_embedding 索引可能越界 | 已审查: 实际使用中不会触发，现有 clamp 保护合理 | ✅ |
| P11-18 | Soft Token Count 公式与 BFS 不匹配 | 已验证: 正确调用顺序下估计准确 | ✅ |

**核心修复成果**:
- 梯度提升: 4.05x (P11-10)
- 参数节省: 92%+ (P11-2)
- 缓存碰撞风险: 消除 (P11-1)
- 路径数据: 从 regions 正确计算 (P11-3)

---

## 11.15 废弃组件存档

| 组件 | 移除原因 | 移除日期 |
|------|----------|----------|
| `StreamingFractalTokenizer` (V1) | 被 V3 取代 | 2025-12-26 |
| `StreamingFractalTokenizerV2` | 尺度崩塌问题 | 2025-12-25 |
| `CrossScaleAttention` | 数学崩塌证明 | 2025-12-25 |
| `FractalHilbertTokenizer` | 被流式替代 | 2025-12-24 |
| `EnhancedFractalTokenProcessor` | 合并至 V3 | 2025-12-24 |
| `global_context_attn` | ARCH-R1 删除 | 2025-12-26 |
| `sort_tokens()` 系列 | P9-1 向量化替代 | 2025-12-28 |
| `forward()` 返回 List[SplitResult] | TensorSplitResult 替代 | 2025-12-28 |
| `LowRankHilbertBias` | P11-8 死代码删除 | 2025-12-30 |
| `HierarchicalHilbertBias` | P11-8 死代码删除 | 2025-12-30 |
| `level_attention_bias` | P11-5 死代码删除 | 2025-12-30 |
