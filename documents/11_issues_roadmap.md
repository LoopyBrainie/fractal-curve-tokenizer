# 第十一章：已知问题与改进计划

本章总结了当前项目的状态、已知缺陷以及未来的改进路线图（基于 `IMPROVEMENT_PLAN.md`）。

## 11.1 P0 问题（已修复）

以下严重影响功能正确性的问题已在最新版本中得到修复：

1.  **REINFORCE 实现不完整**:
    *   *原问题*: `get_tokenizer_loss` 只计算了熵正则化，缺少策略梯度。
    *   *修复*: 实现了完整的 `reward * log_prob` 策略梯度计算，并引入了 Baseline EMA 机制减少方差。
2.  **全局注意力 Mask 泄漏**:
    *   *原问题*: `EnhancedFractalTransformer` 末尾的全局 Attention 未传入 Mask，导致 Padding Token 污染特征。
    *   *修复*: 正确构建并传入了 `key_padding_mask`。
3.  **MNIST 单通道支持**:
    *   *原问题*: 代码中部分硬编码假设输入为 3 通道。
    *   *修复*: `DatasetSpec` 和模型初始化已完全支持单通道输入。

## 11.2 P1 问题（已修复）

以下性能或可维护性问题已解决：

1.  **create_attention_mask 向量化**:
    *   *原问题*: 使用双重循环生成 Mask，复杂度 $O(B \times S^2)$，在 CPU 上极慢。
    *   *修复*: 使用 PyTorch 广播机制重写为向量化实现，性能大幅提升。
2.  **Learnable split 测试失败**:
    *   *原问题*: 部分测试用例在启用可学习分割时不稳定。
    *   *修复*: 修正了测试逻辑，使用方差阈值替代神经网络进行确定性测试。
3.  **位置编码 Batch 处理优化**:
    *   *原问题*: 需要 Flatten -> Forward -> Reshape。
    *   *修复*: `AdvancedFractalPositionEmbedding` 现已原生支持 3D 输入 `(B, S, Info)`。
4.  **Hilbert 代码重构**:
    *   *原问题*: `fractal_curve_tokenizer.py` 中存在冗余的 Hilbert 算法代码。
    *   *修复*: 提取了独立的 `hilbert.py` 模块，并增加了专用测试。

## 11.3 P2 改进项（进行中）

1.  **代码冗余清理**: 移除未使用的导入和遗留代码。
2.  **文档完善**: 持续更新本文档及 API 文档。
3.  **数据增强优化**: 为不同数据集定制更精细的增强策略（已实现 AutoAugment 策略分发）。
