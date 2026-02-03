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


## 11.16 Training Performance (I9)

| ID | Description | Result | Status |
|:---|:------------|:-------|:-------|
| I9 | Training speed bottleneck (24s/it) | **1.62s/it** (14.8x speedup) | ✓ Done |
| I9-1 | Vectorized BFS & GPU structs | Eliminated CPU blocking | ✓ Done |
| I9-2 | `TensorSplitResult` Implementation | Removed Python loops | ✓ Done |

---

## 11.17 Splitter Failure Analysis (I10)

| ID | Issue | Root Cause | Resolution |
|:---|:------|:-----------|:-----------|
| I10 | Learnable splitter collapse | BFS serial dependency | Replaced by **Scheme D** |
| I10-1 | No gradient flow | STE implementation error | Fixed in Scheme A |
| I10-18 | BFS serial dependency | Architecture flaw | Parallel Evaluation |
| I10-19 | Discrete-continuous mismatch | Quantization error | Continuous Relaxation |

---

## 11.18 Math Review (I11)

> Status: All 18 issues resolved.

| ID | Issue | Solution |
|:---|:------|:---------|
| I11-2 | LayerNorm Parameter Explosion | Dynamic Depth Embedding (92% reduction) |
| I11-10 | ComplexityMLP range coupling | Removed output sigmoid |
| I11-11 | Gumbel-Softmax low temp | Enforced T_end >= 0.3 |

---

## 11.19 Deep Math Review (I12)

| ID | Issue | Severity | Status |
|:---|:------|:---------|:-------|
| I12-1 | Gumbel-Softmax double sampling | High | Verified Correct |
| I12-2 | LCA irregular quadtree | Critical | Verified Correct |
| I12-3 | Empty path data source | Critical | Fixed |

---

## 11.20 Comprehensive Review (I13)

| ID | Component | Findings | Status |
|:---|:----------|:---------|:-------|
| I13-1 | FractalPositionEmbedding | Valid logic | Verified |
| I13-2 | AttnHilbertBias | O(N) memory achieved | Verified |
| I13-5 | FeedForwardNetwork | SwiGLU correctly implemented | Verified |

---

## 11.21 Experimental Verification (I14)

| Experiment | Configuration | Result | Note |
|:-----------|:--------------|:-------|:-----|
| E4070 | Tiny-ImageNet, 384d, 12L | 50.74% Acc | Baseline |
| E4071 | Scheme D Integration | Pending | **I14-5 Active** |

---

## 11.22 Training System Refactor (I15)

| Module | Change | Benefit |
|:-------|:-------|:--------|
| `examples/training` | Modular design | Separation of concerns |
| `Trainer` class | Hydra config integration | Reproducibility |
| Logging | TensorBoard + JSONL | Detailed analytics |

---

## 11.23 Collapse Root Cause (I16)

**Diagnosis**: The combination of **Rapid Temperature Annealing** (T < 0.1) and **Unconstrained Threshold Learning** allowed thresholds to grow unchecked, while the **Serial BFS** prevented deep gradients from correcting the behavior.

**Resolution**: Abandoned Scheme A (BFS) in favor of Scheme D (Gumbel Top-K).

---

## 11.24 End-to-End Refactor (I17)

All items in I17 (End-to-End Learnable Splitter) have been subsumed by **Scheme D** implementation.

---

## 11.25 Code Critique (I18)

| ID | Issue | Solution |
|:---|:------|:---------|
| I18-1 | Soft Entropy gradient block | Use cached MLP probabilities |
| I18-2 | Temperature unsafe lower bound | Force `TEMPERATURE_MIN = 0.1` |
| I18-5 | Temperature learning constraint | Added clamp to learner |

---

## 11.26 Scheme D: Gumbel-Top-K (I19)

**Concept**: Select exactly $K$ tokens from $N$ candidates using Gumbel-Top-K trick.
**Key Advantage**: 100% Gradient Coverage (all regions receive gradients via softmax-STE) + 100% Hilbert Locality (unlike continuous relaxation).

| Feature | Implementation | Status |
|:--------|:---------------|:-------|
| Logic | `GumbelTopKSplitter` | ✓ Done |
| Math | $mask = \mathbb{1}_{TopK} - \sigma(z).detach() + \sigma(z)$ | ✓ Verified |
| Tree | $parent 
otin selected$ | ✓ Enforced |

---

## 11.27 Final Analysis (I20)

**Conclusion**: Scheme D represents the mathematically correct solution to the adaptive tokenization problem, solving the "Serial Dependency" and "Gradient Sparsity" problems of Scheme A while maintaining the geometric properties that Scheme B lost.

## 11.28 Architecture Math Critique (I22)

> **结案日期**: 2026-01-15
> **状态**: 6/8 完成, 2项合并

### 已完成 Issues

| ID | 问题 | 解决方案 | 状态 |
|----|------|----------|------|
| I22-1 | 树一致性 gather 语义错误 | 删除冗余实现，保留 `hard_selected[:, safe_children]` | ✅ |
| I22-2 | LCA 偏置语义正确性验证 | P11-3 实现已验证正确性 | ✅ |
| I22-3 | Subset Softmax 梯度增强声称验证 | 理论验证 2.7x 增强 | ✅ |
| I22-4 | Log-Compensation Bias 公式推导 | 方案E替代实现 | ✅ |
| I22-6 | 温度参数理论最优值研究 | via I29-1 (τ=0.5 最优) | ✅ |

### 合并至其他 Issue

| ID | 合并目标 | 说明 |
|----|----------|------|
| I22-5 | → I25-2 | Hilbert 收益量化实验 |
| I22-8 | → I25-12 | PseudoHilbert 局部性衰减 |

---

## 11.29 Training Diagnosis (I23)

> **结案日期**: 2026-01-15
> **状态**: 4/9 完成, 5项待处理

### 已完成 Issues

| ID | 问题 | 解决方案 |
|----|------|----------|
| I23-1 | 深度分布崩塌 | 深度方差归一化 + KL权重 0.5 + 软配额正则化 |
| I23-2 | Token 数量下界失效 | K_max(K, K_min) 修复 + K_90 per-batch 计算 |
| I23-3 | 训练-测试泛化鸿沟 | 通过 I24-1 正则化增强解决 |
| I23-4-NaN | splitter_loss NaN/Inf | clamp 顺序修复 + 复杂度 logits 限幅 |
| I23-7 | 注意力分析模块失效 | 使用 `store_attn_weights` 条件修复 |

### 待处理 Issues

| ID | 优先级 | 问题 |
|----|--------|------|
| I23-8 | 🔵 P2 | Scale Distribution 熵过低 |
| I23-9 | 🔵 P2 | 类别-Token 数量相关性异常 |
| I23-10 | ⚪ P3 | 代码理论问题修复 |

---

## 11.30 Experiment 20260112 Analysis (I24)

> **结案日期**: 2026-01-15
> **状态**: 9/12 完成, 3项待处理

### 已完成 Issues

| ID | 问题 | 解决方案 |
|----|------|----------|
| I24-1 | 训练-验证泛化鸿沟 | dropout 0.15→0.20 + label_smoothing 0.1→0.15 |
| I24-2 | Log-Compensation 理论缺陷 | 方案E (可学习配额) 替代 |
| I24-3 | 类别 1 准确率 0% | 数据清洗 + label_smoothing |
| I24-9 | 位置编码边界舍入误差 | 现有 clamp 保护已足够 |
| I24-11 | 评估脚本注意力分析失效 | 与 I23-7 相同修复 |
| I24-12 | per_class_avg_tokens 数据异常 | 方案A Focal γ=2.5 |

### 合并/待处理

| ID | 状态 | 说明 |
|----|------|------|
| I24-10 | → I25-10 | DEPTH_QUOTA_TARGET 参数化 |
| I24-4 | 🟡 P1 | depth=1 完全缺失 |
| I24-5 | 🟡 P1 | 深度方差归一化批次统计 |

---

## 11.31 Experiment 20260114 Analysis (I25)

> **结案日期**: 2026-01-17
> **状态**: 8/13 完成, 5项待处理

### 已完成 Issues

| ID | 问题 | 解决方案/结论 |
|----|------|---------------|
| I25-4 | 温度敏感性消融 | I29-1 验证 τ=0.5 最优 |
| I25-5 | 阈值方差正则化 | threshold_var_loss 已实现 |
| I25-6 | 整数除法边界象限错误 | 现有逻辑正确 |
| I25-7 | Scale 多样性正则化 | Scheme E 改善 entropy 至 54.5% |
| I25-9 | 评估脚本注意力分析修复 | 已完成 |

### 待处理 Issues

| ID | 优先级 | 问题 |
|----|--------|------|
| I25-2 | 🔴 P0-Critical | Hilbert vs Raster 消融实验 |
| I25-8 | 🔵 P2 | 混合池化选择器收益评估 |
| I25-10 | 🔵 P2 | 配额参数暴露 (含 I24-10) |
| I25-12 | 🔵 P2 | PseudoHilbert 局部性量化 |

---

## 11.32 Tree Consistency Removal (I26)

> **结案日期**: 2026-01-15
> **状态**: 2/3 完成, 1项搁置

### 已完成 Issues

| ID | 问题 | 解决方案 |
|----|------|----------|
| I26-1 | 移除树一致性约束 | 删除 `_enforce_tree_consistency()` 调用 |
| I26-2 | threshold_var_loss 缺失 | 已实现 |

### 搁置

| ID | 问题 | 说明 |
|----|------|------|
| I26-3 | LCA 语义混合问题 | 需进一步分析 |

---

## 11.33 Class Accuracy Variance (I28)

> **结案日期**: 2026-01-15
> **状态**: 1/3 完成, 2项延后

### 已完成 Issues

| ID | 问题 | 解决方案 |
|----|------|----------|
| I28-1 | Worst Classes 准确率过低 | 方案A: Focal γ=2.5 + label_smoothing |

---

## 11.34 Math Consistency Fix (I29)

> **结案日期**: 2026-01-17
> **状态**: 4/5 完成, 1项延后

### 已完成 Issues

| ID | 问题 | 解决方案 |
|----|------|----------|
| I29-1 | SPLITTER_TEMP_END 常量不一致 | 统一为 0.5 |
| I29-2 | threshold_var_loss 实现 | 已完成 |
| I29-3 | LOG_COMPENSATION 条件冗余 | 方案E替代后删除 |
| I29-4 | 温度调度策略一致化 | 统一温度退火参数 |

### 延后

| ID | 问题 | 说明 |
|----|------|------|
| I29-5 | 硬编码 epsilon 常量化 | 低优先级 |

---

## 11.35 Comprehensive Math Critique (I30)

> **结案日期**: 2026-01-17
> **状态**: 3/17 完成, 14项待处理

### 已完成 Issues

| ID | 问题 | 解决方案 |
|----|------|----------|
| I30-3 | 过拟合严重 (Gap=12%) | dropout + label_smoothing 增强 |
| I30-4 | 类别准确率方差过大 | Focal Loss + 数据清洗 |
| I30-9 | 弱引用缓存失效 | 使用强引用 + WeakRef 双重保护 |

### 待处理 Issues (按优先级)

| ID | 优先级 | 问题 |
|----|--------|------|
| I30-1 | 🔴 P0-Critical | Hilbert vs Raster 核心收益验证 |
| I30-2 | 🔴 P0-Critical | Subset Softmax 梯度声明与实现矛盾 |
| I30-5 | 🔴 P0 | levels_info 值域验证 |
| I30-6 | 🟡 P1 | 小 batch 深度方差归一化 |
| I30-7 | 🟡 P1 | PseudoHilbert 跳跃界公式 |
| I30-8 | 🟡 P1 | GUMBEL_EPSILON 过小风险 |
| I30-10~13 | 🔵 P2 | 代码质量优化 |
| I30-14~16 | ⚪ P3 | 研究探索 |

---

## 11.36 LCA Pseudo Adaptation (I31)

> **状态**: 3/3 P0-Critical 待实施
> **结案日期**: 待定

### 待实施 Issues

| ID | 问题 | 优先级 |
|----|------|--------|
| I31-1 | 形状-尺度编码器实现 | 🔴 P0-Critical |
| I31-2 | LCAHilbertBias 扩展 | 🔴 P0-Critical |
| I31-3 | 面积位置编码补充 | 🔴 P0-Critical |

---

## 11.37 Issue 归档 (2026-01-26)

> **归档日期**: 2026-01-26
> **状态**: 归档已结案 Issue，保留活跃 Issue
> **归档范围**: CRIT-1~4, I78-2, I96-1~8, I97-1~13, I98-1~9, I99-1~11, I100-1~7, I102-1~11, I103-1~4, I104-1~3, I105-1, I106-1~2, I107-1~6

### 11.37.1 P0-Critical Issues (已归档)

| ID | 问题 | 解决方案 | 归档时间 |
|----|------|----------|----------|
| **CRIT-1** | Gumbel-Top-K梯度覆盖文档不准确 | 更新docstring，Scheme E使用全局Softmax | 2026-01-22 |
| **CRIT-2** | hilbert_bias_scale无上限约束 | 添加clamp(max=10.0) | 2026-01-22 |
| **CRIT-3** | 无界LCA缓存可导致OOM | 移除缓存，直接计算LCA | 2026-01-26 |
| **CRIT-4** | `.item()`同步点导致训练变慢 | GPU张量累积 + 单次同步 | 2026-01-26 |
| **I107-2** | Feature Cache保留引用 | 移除调试缓存变量 | 2026-01-26 |
| **I107-3** | 训练器梯度累积内存累积 | 使用`del`释放中间变量 | 2026-01-26 |

### 11.37.2 P0 Issues (已归档)

| ID | 问题 | 解决方案 | 归档时间 |
|----|------|----------|----------|
| **I78-2** | Tokenizer忽略K预算 | 直接使用selected_mask索引 | 2026-01-21 |
| **I96-1** | EMA clamp初始化不一致 | 统一DEPTH_VARIANCE_INIT_EPS | 2026-01-21 |
| **I96-2** | 温度退火未完成 | 训练配置问题，非代码bug | 2026-01-21 |
| **I97-1** | EMA评估模式未初始化 | 保守初始化 | 2026-01-22 |
| **I97-2** | `_compute_level_bias` 2D死代码 | 删除2D分支，添加断言 | 2026-01-22 |
| **I97-3** | `_precompute_candidates`方法重复 | 统一方法实现 | 2026-01-22 |
| **I99-1** | Batch独立性失败 | 显式排序逻辑 | 2026-01-22 |
| **I99-2** | Padding Token Masking失效 | 正确应用padding mask | 2026-01-22 |
| **I99-3** | Padding Sentinel实现不完整 | levels_info填充逻辑修复 | 2026-01-22 |
| **I99-4** | 路径编码验证失败 | 路径值范围约束 | 2026-01-22 |
| **I99-5** | Levels_info格式问题 | 格式一致性修复 | 2026-01-22 |
| **I100-1** | 树一致性训练/推理不对称 | I96-4验证通过，设计正确 | 2026-01-25 |
| **I102-1** | softplus逆变换数值不稳定 | log-space参数化 | 2026-01-24 |
| **I102-2** | omega除零保护不足 | clamp替代epsilon | 2026-01-24 |
| **I102-3** | total_prob clamp FP16下溢 | PROB_EPSILON_FP16 | 2026-01-24 |
| **I102-4** | shape_norm epsilon过小 | SHAPE_NORM_EPSILON | 2026-01-25 |
| **I102-5** | GPU-CPU同步阻塞 | 移除.item()同步点 | 2026-01-26 |
| **I103-1** | 层次注意力未向量化 | 批量Hilbert偏置计算 | 2026-01-25 |
| **I103-2** | Hilbert索引权重未缓存 | 类级权重缓存 | 2026-01-25 |
| **I103-3** | GumbelTopKSplitter设备传输 | 惰性设备端缓存 | 2026-01-25 |
| **I103-4** | Gumbel噪声CPU生成 | GPU生成噪声 | 2026-01-25 |
| **I107-1** | 预分配缓冲区256固定 | 动态调整缓冲区大小 | 2026-01-26 |
| **I107-4** | Python循环未向量化 | nonzero()替代列表推导 | 2026-01-26 |
| **I107-5** | 警告消息中.item()同步点 | 延迟同步已最优 | 2026-01-26 |
| **I107-6** | Hilbert曲线全局缓存清理 | 阶数阈值缓存 | 2026-01-26 |

### 11.37.3 P1 Issues (已归档)

| ID | 问题 | 解决方案 | 归档时间 |
|----|------|----------|----------|
| **I96-3** | 分层Top-K打破计算图 | detach()分离计算图 | 2026-01-22 |
| **I96-4** | 树一致性硬约束 | 软边距+模式区分 | 2026-01-22 |
| **I96-5** | ROI-Align重复计算 | 批量ROI-Align+去重 | 2026-01-22 |
| **I97-4** | 分层Top-K Python循环向量化 | 预计算深度索引 | 2026-01-22 |
| **I97-5** | 配置系统重复 | ModelConfig别名 | 2026-01-22 |
| **I97-6** | 移除未使用参数 | 删除冗余参数 | 2026-01-22 |
| **I100-3** | 缓存版本机制兼容性 | 内容哈希缓存 | 2026-01-25 |
| **I100-4** | ROI-Align与ROI-Pooling精度差异 | 强制依赖torchvision | 2026-01-25 |
| **I100-5** | 小batch EMA方差估计保守性 | 分层初始化B1/B2/B4 | 2026-01-25 |
| **I102-6** | 深度循环并行化 | 批量分块计算 | 2026-01-25 |
| **I102-7** | LCA矩阵向量化 | diagonal()方法 | 2026-01-25 |
| **I102-8** | Hilbert索引向量化 | 查找表方法 | 2026-01-25 |
| **I102-9** | points.index优化 | LRU缓存 | 2026-01-25 |
| **I104-1** | 未启用cuDNN benchmark | 添加cudnn.benchmark | 2026-01-25 |
| **I104-2** | DataLoader prefetch_factor过高 | 动态调整 | 2026-01-25 |
| **I104-3** | LCA偏置FP32存储 | FP16存储 | 2026-01-25 |

### 11.37.4 P2 Issues (已归档)

| ID | 问题 | 解决方案 | 归档时间 |
|----|------|----------|----------|
| **I96-6** | 梯度覆盖文档不准确 | 更新K/N梯度覆盖描述 | 2026-01-22 |
| **I96-7** | 深度下界约束 | 恢复固定下界 | 2026-01-22 |
| **I96-8** | 预算损失归一化 | 相对MSE损失 | 2026-01-22 |
| **I97-7** | 偏置缩放因子可学习化 | Softplus可学习参数 | 2026-01-22 |
| **I97-8** | 测试覆盖增强 | 新增测试文件 | 2026-01-22 |
| **I97-9** | Legacy代码清理 | 删除split_adaptive.py | 2026-01-22 |
| **I98-4** | levels_info契约规范化 | LevelsInfo dataclass | 2026-01-23 |
| **I98-5** | TokenizerOutput增强 | LevelsInfo强类型 | 2026-01-23 |
| **I98-6** | 配置系统统一 | ModelArchitectureConfig | 2026-01-25 |
| **I98-7** | Splitter接口协议 | Protocol层次定义 | 2026-01-25 |
| **I99-6** | 动态K边界常量测试过时 | 更新期望值 | 2026-01-22 |
| **I99-7** | Elastic Budget常量测试过时 | 更新期望值 | 2026-01-22 |
| **I99-8** | Quota Rounding测试约束 | 软正则化设计 | 2026-01-22 |
| **I99-9** | 测试验证清单更新 | 验证清单脚本 | 2026-01-22 |
| **I100-6** | 动态深度推理batch对齐 | 固定深度depth//2 | 2026-01-25 |
| **I102-10** | compute_max_depth注释错误 | 修正注释 | 2026-01-25 |
| **I102-11** | n power-of-2验证 | 运行时断言 | 2026-01-25 |
| **I105-1** | _compute_quota_loss向量化 | 向量化计算 | 2026-01-25 |

### 11.37.5 P3 Issues (已归档)

| ID | 问题 | 解决方案/结论 | 归档时间 |
|----|------|---------------|----------|
| **I97-10** | 层次化注意力 | 深度内独立Attention | 2026-01-22 |
| **I97-11** | 动态计算 | 推理专用动态深度 | 2026-01-22 |
| **I97-12** | 自定义CUDA kernel | 不推荐实施 | 2026-01-22 |
| **I97-13** | 形式化理论分析 | 探索性研究 | 2026-01-22 |
| **I98-8** | 事件驱动架构 | 探索完成 | 2026-01-23 |
| **I98-9** | Plugin系统 | 探索完成 | 2026-01-23 |
| **I99-10** | 测试策略重构 | conftest.py fixtures | 2026-01-22 |
| **I99-11** | 随机性隔离 | 随机性隔离测试套件 | 2026-01-22 |
| **I100-7** | 配额分配算法优化 | 重构为标准LRM | 2026-01-25 |
| **I100-8** | Pseudo-Hilbert局部性证明 | 文档补充 | 2026-01-25 |
| **I106-1** | Flash Attention集成 | 条件导入+偏置融合 | 2026-01-26 |
| **I106-2** | 双LayerNorm设计评估 | Attention标准LN+FFN层级感知LN | 2026-01-26 |

### 11.37.6 已跳过 P3 Issues (数学冲突)

| Issue | 跳过原因 | 数学分析 |
|-------|----------|----------|
| I25-13 | Depth Contrastive Loss | 与Hilbert局部性冲突 (~80%) |
| I30-14 | Depth Contrastive Loss | 同上，数学上不优 |
| I30-15 | 祖先关系独立偏置 | Hilbert曲线已编码路径 |
| I30-16 | 层级注意力路由 | LCA Embed已隐含深度关系 |
| I26-3 | 祖先关系独立偏置 | 同I30-15 |
| I32-9 | 层级偏置clamp | 普通clamp已足够 |
| I32-10 | DropPath余弦调度 | 线性调度已足够 |

---

## 11.38 活跃 Issue (待处理)

> **最后更新**: 2026-01-26
> **状态**: 保留待处理 Issue

### P0-Critical (紧急)

| ID | 问题 | 文件位置 | 状态 |
|----|------|----------|------|
| **CRIT-5** | LCA与Hilbert索引不一致(56%) | attn_hilbert_bias.py | ✅ 已完成 |
| **CRIT-6** | `round(softmax)`破坏配额梯度 | gumbel_topk_splitter.py:416 | ✅ 已完成 |

### P0 (紧急功能)

| ID | 问题 | 状态 |
|----|------|------|
| **I100-2** | 温度参数ablation study | 🔄 进行中 |
| **I108-1** | 多偏置融合无尺度归一化 | ⏳ 待处理 |
| **I108-2** | channels_last条件错误 | ⏳ 待处理 |

### P1 (性能优化)

| ID | 问题 | 状态 |
|----|------|------|
| **I108-3** | Python循环构建深度分布 | ⏳ 待处理 |
| **I108-4** | Hilbert缓存内存界计算不准 | ⏳ 待处理 |

### P3 (探索/按需)

| ID | 问题 | 状态 |
|----|------|------|
| **I108-5** | Hilbert局部性概率量化 | ⏳ 待探索 |
| **I108-6** | FP16 clamp边界优化 | ⏳ 待探索 |

---

## 11.36 Issue Archive (2026-01-20)

> **状态**: ✅ 全部结案 (40 Issues)
> **归档日期**: 2026-01-20

本章节记录所有已完成 Issue 的数学形式化分析、修复方案及验证结果。

---

### 11.36.1 P0-Critical Issues (4/4)

| Issue | 问题描述 | 数学形式化 | 修复方案 | 验证 |
|-------|----------|------------|----------|------|
| **I34-2** | 配额分配约束破坏 | $\sum_d K_d = K_{\text{total}}$ | 归一化重分配 | ✅ |
| **I34-5** | 深度计算 floor 偏差 | $d = \lceil \log_2(N/w) \rceil$ | ceil 替代 floor | ✅ |
| **I34-6** | feature_dim 不匹配 | $\text{feature\_dim} = d_{\text{model}}$ | 统一维度 | ✅ |
| **A1** | Token 容量 K_max | $K_{\max} = 0.05 \times N$ | 动态相对约束 | ✅ |

---

### 11.36.2 P0 Issues (7/7)

| Issue | 问题描述 | 数学形式化 | 修复方案 | 验证 |
|-------|----------|------------|----------|------|
| **I32-2** | 空 Token 语义 | $\text{levels\_info}[i] = -1$ | sentinel 值 | ✅ |
| **I32-3** | Quota 四舍五入 | $K_d = \text{round}(\pi_d \times K)$ | 59/59 PASSED | ✅ |
| **A16** | KL 权重过重 | $\lambda_{KL} = 0.1$ | 移除 KL 损失 | ✅ |
| **A17** | ShapeScaleEncoder | $\tau_c \cdot \text{ShapeScaleEncoder}$ | use_affine_modulation=True | ✅ |
| **A18** | Token 上限估计 | $K_{\max} \geq 256$ | A1 动态扩展 | ✅ |
| **I35-1** | 配额维度不匹配 | $\text{quota\_logits}[:D]$ | 切片修正 | ✅ |
| **I35-2** | 方差归一化 NaN | $\sqrt{v + \epsilon}$ | epsilon 保护 | ✅ |

---

### 11.36.3 P1 Issues (12/12)

| Issue | 问题描述 | 数学形式化 | 修复方案 | 验证 |
|-------|----------|------------|----------|------|
| **I30-6** | 小 batch 归一化 | EMA Running Statistics | 自适应归一化 | ✅ |
| **I30-7** | PseudoHilbert 跳跃 | 注释补充 | 边界说明 | ✅ |
| **I30-8** | GUMBEL_EPSILON | $\epsilon = 10^{-8}$ | 安全边界 | ✅ |
| **I30-9** | 缓存版本校验 | data_ptr + PyTorch 版本 | 版本校验 | ✅ |
| **I31-1** | 形状-尺度编码器 | $B^* = \tau_h \cdot \text{LCA} + \tau_c \cdot \text{SS}$ | 实现完成 | ✅ |
| **I31-2** | LCAHilbertBias 扩展 | Attention Bias | 形状注入 | ✅ |
| **I31-3** | 面积位置编码 | 深度位置编码 | 补充完成 | ✅ |
| **I32-5** | 方差缩放常数 | 自适应归一化 | 无固定公式 | ✅ |
| **I32-6** | 残差门控参数 | 单门控 + tanh | 零初始化 | ✅ |
| **I32-7** | 傅里叶频率 | Nyquist 约束 | 14/14 PASSED | ✅ |
| **I35-3** | 单深度方差 | $\sigma_d = 0$ when $N_d=1$ | 跳过测试 | ✅ |
| **I78** | 动态分辨率 | $\text{image\_size} = None$ | 支持完成 | ✅ |

---

### 11.36.4 P2 Issues (17/17)

| Issue | 问题描述 | 数学形式化 | 修复方案 | 验证 |
|-------|----------|------------|----------|------|
| **I25-8** | 混合池化收益 | pool="cls/mean/weighted" | 15/15 PASSED | ✅ |
| **I25-10** | 配额参数暴露 | config.quota | 参数可见 | ✅ |
| **I25-12** | PseudoHilbert 量化 | locality_preservation_rate | 指标工具类 | ✅ |
| **I30-10** | 配额参数暴露 | constants.py | 参数可见 | ✅ |
| **I30-11** | 混合池化评估 | test_pooling_strategies.py | 15/15 PASSED | ✅ |
| **I30-13** | SiLU/Swish 说明 | ffn_swiglu.py:11-24 | 文档完成 | ✅ |
| **I31-4** | 单元测试 | test_shape_scale_encoding | 100% PASSED | ✅ |
| **I32-8** | 有限记忆 EMA | Per-batch 统计量 | 替代 EMA | ✅ |
| **I32-11** | ARCH-R2 聚合器 | Xavier init | 正确初始化 | ✅ |
| **I32-12** | 方差 epsilon | $1e-6 \to 1e-5$ | 数值稳定 | ✅ |
| **A19** | Attention scale | $1/\sqrt{d_k}$ | 移除可学习权重 | ✅ |
| **A15** | Pseudo-Hilbert 局部性 | locality_preservation_rate | 实现完成 | ✅ |
| **I32-1** | ROI-Align 复杂度 | $O(B \times N \times C \times k^2)$ | 注释修正 | ✅ |
| **I36-7** | get_model_info | 统一诊断接口 | 实现完成 | ✅ |

---

### 11.36.5 P3 Issues Skipped (7/7)

| Issue | 跳过原因 | 数学分析 |
|-------|----------|----------|
| **I25-13** | Depth Contrastive Loss | 与 Hilbert 局部性冲突 (~80%) |
| **I30-14** | Depth Contrastive Loss | 同上，数学上不优 |
| **I30-15** | 祖先关系独立偏置 | Hilbert 曲线已编码路径，独立偏置冗余 |
| **I30-16** | 层级注意力路由 | LCA Embed 已隐含深度关系，梯度稀释 |
| **I26-3** | 祖先关系独立偏置 | 同 I30-15 |
| **I32-9** | 层级偏置 clamp | 普通 clamp 已足够 |
| **I32-10** | DropPath 余弦调度 | 线性调度已足够 |

---

### 11.36.6 验证统计

| 指标 | 数值 |
|------|------|
| 总测试数 | 665 |
| 通过 | 665 |
| 跳过 | 0 |
| 失败 | 0 |
| 覆盖率 | 100% |

---

## 11.37 Issue 归档 (2026-01-26)

> **归档日期**: 2026-01-26
> **状态**: 归档已结案 Issue，保留活跃 Issue
> **归档范围**: CRIT-1~4, I78-2, I96-1~8, I97-1~13, I98-1~9, I99-1~11, I100-1~7, I102-1~11, I103-1~4, I104-1~3, I105-1, I106-1~2, I107-1~6

### 11.37.1 P0-Critical Issues (已归档)

| ID | 问题 | 解决方案 | 归档时间 |
|----|------|----------|----------|
| **CRIT-1** | Gumbel-Top-K梯度覆盖文档不准确 | 更新docstring，Scheme E使用全局Softmax | 2026-01-22 |
| **CRIT-2** | hilbert_bias_scale无上限约束 | 添加clamp(max=10.0) | 2026-01-22 |
| **CRIT-3** | 无界LCA缓存可导致OOM | 移除缓存，直接计算LCA | 2026-01-26 |
| **CRIT-4** | `.item()`同步点导致训练变慢 | GPU张量累积 + 单次同步 | 2026-01-26 |
| **I107-2** | Feature Cache保留引用 | 移除调试缓存变量 | 2026-01-26 |
| **I107-3** | 训练器梯度累积内存累积 | 使用`del`释放中间变量 | 2026-01-26 |

### 11.37.2 P0 Issues (已归档)

| ID | 问题 | 解决方案 | 归档时间 |
|----|------|----------|----------|
| **I78-2** | Tokenizer忽略K预算 | 直接使用selected_mask索引 | 2026-01-21 |
| **I96-1** | EMA clamp初始化不一致 | 统一DEPTH_VARIANCE_INIT_EPS | 2026-01-21 |
| **I96-2** | 温度退火未完成 | 训练配置问题，非代码bug | 2026-01-21 |
| **I97-1** | EMA评估模式未初始化 | 保守初始化 | 2026-01-22 |
| **I97-2** | `_compute_level_bias` 2D死代码 | 删除2D分支，添加断言 | 2026-01-22 |
| **I97-3** | `_precompute_candidates`方法重复 | 统一方法实现 | 2026-01-22 |
| **I99-1** | Batch独立性失败 | 显式排序逻辑 | 2026-01-22 |
| **I99-2** | Padding Token Masking失效 | 正确应用padding mask | 2026-01-22 |
| **I99-3** | Padding Sentinel实现不完整 | levels_info填充逻辑修复 | 2026-01-22 |
| **I99-4** | 路径编码验证失败 | 路径值范围约束 | 2026-01-22 |
| **I99-5** | Levels_info格式问题 | 格式一致性修复 | 2026-01-22 |
| **I100-1** | 树一致性训练/推理不对称 | I96-4验证通过，设计正确 | 2026-01-25 |
| **I102-1** | softplus逆变换数值不稳定 | log-space参数化 | 2026-01-24 |
| **I102-2** | omega除零保护不足 | clamp替代epsilon | 2026-01-24 |
| **I102-3** | total_prob clamp FP16下溢 | PROB_EPSILON_FP16 | 2026-01-24 |
| **I102-4** | shape_norm epsilon过小 | SHAPE_NORM_EPSILON | 2026-01-25 |
| **I102-5** | GPU-CPU同步阻塞 | 移除.item()同步点 | 2026-01-26 |
| **I103-1** | 层次注意力未向量化 | 批量Hilbert偏置计算 | 2026-01-25 |
| **I103-2** | Hilbert索引权重未缓存 | 类级权重缓存 | 2026-01-25 |
| **I103-3** | GumbelTopKSplitter设备传输 | 惰性设备端缓存 | 2026-01-25 |
| **I103-4** | Gumbel噪声CPU生成 | GPU生成噪声 | 2026-01-25 |
| **I107-1** | 预分配缓冲区256固定 | 动态调整缓冲区大小 | 2026-01-26 |
| **I107-4** | Python循环未向量化 | nonzero()替代列表推导 | 2026-01-26 |
| **I107-5** | 警告消息中.item()同步点 | 延迟同步已最优 | 2026-01-26 |
| **I107-6** | Hilbert曲线全局缓存清理 | 阶数阈值缓存 | 2026-01-26 |

### 11.37.3 P1 Issues (已归档)

| ID | 问题 | 解决方案 | 归档时间 |
|----|------|----------|----------|
| **I96-3** | 分层Top-K打破计算图 | detach()分离计算图 | 2026-01-22 |
| **I96-4** | 树一致性硬约束 | 软边距+模式区分 | 2026-01-22 |
| **I96-5** | ROI-Align重复计算 | 批量ROI-Align+去重 | 2026-01-22 |
| **I97-4** | 分层Top-K Python循环向量化 | 预计算深度索引 | 2026-01-22 |
| **I97-5** | 配置系统重复 | ModelConfig别名 | 2026-01-22 |
| **I97-6** | 移除未使用参数 | 删除冗余参数 | 2026-01-22 |
| **I100-3** | 缓存版本机制兼容性 | 内容哈希缓存 | 2026-01-25 |
| **I100-4** | ROI-Align与ROI-Pooling精度差异 | 强制依赖torchvision | 2026-01-25 |
| **I100-5** | 小batch EMA方差估计保守性 | 分层初始化B1/B2/B4 | 2026-01-25 |
| **I102-6** | 深度循环并行化 | 批量分块计算 | 2026-01-25 |
| **I102-7** | LCA矩阵向量化 | diagonal()方法 | 2026-01-25 |
| **I102-8** | Hilbert索引向量化 | 查找表方法 | 2026-01-25 |
| **I102-9** | points.index优化 | LRU缓存 | 2026-01-25 |
| **I104-1** | 未启用cuDNN benchmark | 添加cudnn.benchmark | 2026-01-25 |
| **I104-2** | DataLoader prefetch_factor过高 | 动态调整 | 2026-01-25 |
| **I104-3** | LCA偏置FP32存储 | FP16存储 | 2026-01-25 |

### 11.37.4 P2 Issues (已归档)

| ID | 问题 | 解决方案 | 归档时间 |
|----|------|----------|----------|
| **I96-6** | 梯度覆盖文档不准确 | 更新K/N梯度覆盖描述 | 2026-01-22 |
| **I96-7** | 深度下界约束 | 恢复固定下界 | 2026-01-22 |
| **I96-8** | 预算损失归一化 | 相对MSE损失 | 2026-01-22 |
| **I97-7** | 偏置缩放因子可学习化 | Softplus可学习参数 | 2026-01-22 |
| **I97-8** | 测试覆盖增强 | 新增测试文件 | 2026-01-22 |
| **I97-9** | Legacy代码清理 | 删除split_adaptive.py | 2026-01-22 |
| **I98-4** | levels_info契约规范化 | LevelsInfo dataclass | 2026-01-23 |
| **I98-5** | TokenizerOutput增强 | LevelsInfo强类型 | 2026-01-23 |
| **I98-6** | 配置系统统一 | ModelArchitectureConfig | 2026-01-25 |
| **I98-7** | Splitter接口协议 | Protocol层次定义 | 2026-01-25 |
| **I99-6** | 动态K边界常量测试过时 | 更新期望值 | 2026-01-22 |
| **I99-7** | Elastic Budget常量测试过时 | 更新期望值 | 2026-01-22 |
| **I99-8** | Quota Rounding测试约束 | 软正则化设计 | 2026-01-22 |
| **I99-9** | 测试验证清单更新 | 验证清单脚本 | 2026-01-22 |
| **I100-6** | 动态深度推理batch对齐 | 固定深度depth//2 | 2026-01-25 |
| **I102-10** | compute_max_depth注释错误 | 修正注释 | 2026-01-25 |
| **I102-11** | n power-of-2验证 | 运行时断言 | 2026-01-25 |
| **I105-1** | _compute_quota_loss向量化 | 向量化计算 | 2026-01-25 |

### 11.37.5 P3 Issues (已归档)

| ID | 问题 | 解决方案/结论 | 归档时间 |
|----|------|---------------|----------|
| **I97-10** | 层次化注意力 | 深度内独立Attention | 2026-01-22 |
| **I97-11** | 动态计算 | 推理专用动态深度 | 2026-01-22 |
| **I97-12** | 自定义CUDA kernel | 不推荐实施 | 2026-01-22 |
| **I97-13** | 形式化理论分析 | 探索性研究 | 2026-01-22 |
| **I98-8** | 事件驱动架构 | 探索完成 | 2026-01-23 |
| **I98-9** | Plugin系统 | 探索完成 | 2026-01-23 |
| **I99-10** | 测试策略重构 | conftest.py fixtures | 2026-01-22 |
| **I99-11** | 随机性隔离 | 随机性隔离测试套件 | 2026-01-22 |
| **I100-7** | 配额分配算法优化 | 重构为标准LRM | 2026-01-25 |
| **I100-8** | Pseudo-Hilbert局部性证明 | 文档补充 | 2026-01-25 |
| **I106-1** | Flash Attention集成 | 条件导入+偏置融合 | 2026-01-26 |
| **I106-2** | 双LayerNorm设计评估 | Attention标准LN+FFN层级感知LN | 2026-01-26 |

### 11.37.6 已跳过 P3 Issues (数学冲突)

| Issue | 跳过原因 | 数学分析 |
|-------|----------|----------|
| I25-13 | Depth Contrastive Loss | 与Hilbert局部性冲突 (~80%) |
| I30-14 | Depth Contrastive Loss | 同上，数学上不优 |
| I30-15 | 祖先关系独立偏置 | Hilbert曲线已编码路径 |
| I30-16 | 层级注意力路由 | LCA Embed已隐含深度关系 |
| I26-3 | 祖先关系独立偏置 | 同I30-15 |
| I32-9 | 层级偏置clamp | 普通clamp已足够 |
| I32-10 | DropPath余弦调度 | 线性调度已足够 |

---

## 11.38 活跃 Issue (待处理)

> **最后更新**: 2026-01-26
> **状态**: 保留待处理 Issue

### P0-Critical (紧急)

| ID | 问题 | 文件位置 | 状态 |
|----|------|----------|------|
| **CRIT-5** | LCA与Hilbert索引不一致(56%) | attn_hilbert_bias.py | ✅ 已完成 |
| **CRIT-6** | `round(softmax)`破坏配额梯度 | gumbel_topk_splitter.py:416 | ✅ 已完成 |

### P0 (紧急功能)

| ID | 问题 | 状态 |
|----|------|------|
| **I100-2** | 温度参数ablation study | 🔄 进行中 |
| **I108-1** | 多偏置融合无尺度归一化 | ⏳ 待处理 |
| **I108-2** | channels_last条件错误 | ⏳ 待处理 |

### P1 (性能优化)

| ID | 问题 | 状态 |
|----|------|------|
| **I108-3** | Python循环构建深度分布 | ⏳ 待处理 |
| **I108-4** | Hilbert缓存内存界计算不准 | ⏳ 待处理 |

### P3 (探索/按需)

| ID | 问题 | 状态 |
|----|------|------|
| **I108-5** | Hilbert局部性概率量化 | ⏳ 待探索 |
| **I108-6** | FP16 clamp边界优化 | ⏳ 待探索 |

---

## 附录: Issue ID 索引

| ID | 日期 | 主题 | 归档位置 |
|----|------|------|----------|
| I36-I41 | 2026-01 | 显存优化与数值稳定性 | §11.37 |
| CRIT-1~6 | 2026-01 | Critical Issues | §11.37 |
| I78, I96-I99 | 2026-01 | 代码审查与测试修复 | §11.37 |
| I100-I108 | 2026-01 | 数学形式化与性能优化 | §11.37-§11.38 |

> **历史 Issue (已归档)**: I0-I35 的详细分析记录在 §11.3-§11.36

---

## 11.39 Issue 归档 (2026-02-03)

> **归档日期**: 2026-02-03
> **状态**: 归档 IMPROVEMENT_PLAN.md 中的已完成 Issue
> **归档范围**: I109, I110, I111 系列 (15个Issue)

### 11.39.1 P0-Critical Issues (已归档)

| ID | 问题 | 解决方案 | 归档时间 |
|----|------|----------|----------|
| **I109-2** | Z-Score除零风险 | 温度下界 + Z值约束 | 2026-01-29 |
| **I109-3** | Token Coverage参数与实现脱节 | 统一覆盖率参数 | 2026-01-29 |
| **I110-1** | SemanticRedundancySplitter - LookAheadHead | 双层MLP + GELU | 2026-02-02 |
| **I110-2** | SemanticRedundancySplitter - CorrelationGate | 余弦相似度计算 | 2026-02-02 |
| **I110-3** | SemanticRedundancySplitter - 核心分裂逻辑 | Gumbel-Softmax决策 | 2026-02-02 |
| **I111-1** | 温度参数传递失效 | 统一配置层 | 2026-02-02 |
| **I111-2** | SplitterConfig配置传递断裂 | 简化Config结构 | 2026-02-02 |
| **I111-3** | 深度熵正则化完全失效 | 硬选择计数 + 动态权重 | 2026-02-02 |

### 11.39.2 P0 Issues (已归档)

| ID | 问题 | 解决方案 | 归档时间 |
|----|------|----------|----------|
| **I109-4** | Elastic Budget惩罚权重不平衡 | 目标导向损失函数 | 2026-01-29 |
| **I110-4** | 语义冗余损失函数 | DiversityLoss + ReconstructionLoss | 2026-02-02 |
| **I111-4** | HilbertSplitterConfig配置层重构 | 统一数学结构 | 2026-02-02 |

### 11.39.3 P1 Issues (已归档)

| ID | 问题 | 解决方案 | 归档时间 |
|----|------|----------|----------|
| **I109-5** | 1x1网格边缘情况 | 恒等映射处理 | 2026-01-29 |
| **I109-6** | 梯度强度不平衡优化 | 覆盖率缩放STE | 2026-02-01 |
| **I110-5** | SemanticSplitterConfig配置类 | 数据类实现 | 2026-02-02 |
| **I110-6** | Tokenizer Streaming V3适配 | 条件初始化 | 2026-02-02 |
| **I110-7** | FractalViT模型集成 | 参数字段添加 | 2026-02-02 |
| **I111-5** | 相对预算公式完整实现 | 完整公式实现 | 2026-02-02 |

### 11.39.4 P2 Issues (已归档)

| ID | 问题 | 解决方案 | 归档时间 |
|----|------|----------|----------|
| **I108-3** | Python循环构建深度分布 | one-hot + einsum向量化 | 2026-01-26 |
| **I108-4** | Hilbert缓存内存界计算不准 | 修正内存注释 | 2026-01-26 |
| **I110-8** | 语义冗余分裂器单元测试 | 25个测试全部通过 | 2026-02-02 |
| **I111-6** | 深度分布实时监控 | Lazy Monitoring模式 | 2026-02-02 |

### 11.39.5 P3 Issues (已归档)

| ID | 问题 | 解决方案/结论 | 归档时间 |
|----|------|---------------|----------|
| **I108-5** | Hilbert局部性概率量化 | HilbertProbabilityMetrics类 | 2026-01-26 |
| **I108-6** | FP16 clamp边界优化 | 分层clamp常量 | 2026-01-26 |
| **I109-7** | log2精度问题修复 | bit_length()替代 | 2026-02-01 |
| **I110-9** | 语义冗余Tokenizer集成测试 | 13个测试全部通过 | 2026-02-02 |

### 11.39.6 P3-探索 Issues (已归档)

| ID | 问题 | 解决方案/结论 | 归档时间 |
|----|------|---------------|----------|
| **I109-8** | 熵正则化添加 | 深度熵 + 配额熵 | 2026-02-01 |
| **I109-9** | 温度Annealing调度实现 | 三种调度策略 | 2026-02-01 |
| **I109-10** | Elastic Budget与K_bounds死区匹配 | 目标导向重构 | 2026-01-29 |
| **I111-7** | 分层温度调度 | τ_d = τ_base × exp(-α×d) | 2026-02-02 |
| **I111-8** | 动态预算端到端学习 | STE bypass | 2026-02-02 |

### 11.39.7 归档统计

| 优先级 | 归档数量 |
|--------|----------|
| P0-Critical | 8 |
| P0 | 3 |
| P1 | 6 |
| P2 | 4 |
| P3 | 4 |
| P3-探索 | 5 |
| **总计** | **30** |

---

## 附录: Issue ID 索引 (更新)

| ID | 日期 | 主题 | 归档位置 |
|----|------|------|----------|
| I36-I41 | 2026-01 | 显存优化与数值稳定性 | §11.37 |
| CRIT-1~6 | 2026-01 | Critical Issues | §11.37 |
| I78, I96-I99 | 2026-01 | 代码审查与测试修复 | §11.37 |
| I100-I108 | 2026-01 | 数学形式化与性能优化 | §11.37-§11.38 |
| I109-I111 | 2026-02 | 语义分割器与配置重构 | §11.39 |

> **最后更新**: 2026-02-03
> **文档版本**: v2 (新增 §11.39 Issue归档)
