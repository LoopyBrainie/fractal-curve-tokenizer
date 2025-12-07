# 第九章：训练策略与损失流 (train_fractal_vit.py)

本章描述了模型在训练过程中的数据流动，特别是损失函数的计算和梯度的反向传播。

## 9.1 训练循环 (Training Loop)

### `train_one_epoch`
这是单个 Epoch 的训练主控函数。

*   **输入**: `model`, `dataloader`, `optimizer`, `criterion`。
*   **流程**:
    1.  **数据加载**: 从 `dataloader` 获取 `images` `(B, C, H, W)` 和 `labels` `(B)`.
    2.  **前向传播**:
        *   `outputs = model(images)`。
        *   同时，模型内部会更新 `tokenizer.saved_log_probs`。
    3.  **主损失计算 (Classification Loss)**:
        *   `loss_cls = criterion(outputs, labels)`。
        *   通常使用 `CrossEntropyLoss`。
    4.  **辅助损失计算 (Auxiliary Loss)**:
        *   调用 `model.get_tokenizer_loss(-loss_cls.detach())`。
        *   **关键点**: 这里使用 `-loss_cls` 作为 Reward。Loss 越小，Reward 越大。
        *   `loss_aux` 包含策略梯度损失和熵正则化项。
    5.  **总损失**: `total_loss = loss_cls + loss_aux_weight * loss_aux`。
    6.  **反向传播**:
        *   `optimizer.zero_grad()`。
        *   `total_loss.backward()`。
        *   `optimizer.step()`。
    7.  **清理**: `model.tokenizer.clear_memory()` 清空保存的 Log Probs，防止内存泄漏。

---

## 9.2 损失函数详解

### 1. 分类损失 ($L_{cls}$)
衡量模型最终预测的准确性。
$$ L_{cls} = -\sum_{c=1}^M y_{o,c} \log(p_{o,c}) $$
其中 $y$ 是真实标签，$p$ 是模型输出的概率。

### 2. 决策网络损失 ($L_{aux}$)
用于训练 Tokenizer 中的策略网络。基于 REINFORCE 算法。

$$ L_{policy} = - \frac{1}{N} \sum_{i=1}^N (R_i - b) \cdot \log \pi(a_i|s_i) $$

*   $R_i$: 奖励信号 (Reward)。当前实现中使用 Batch 的平均分类 Loss 取反。
*   $b$: 基线 (Baseline)。当前实现中使用移动平均 Reward。
*   $\pi(a|s)$: 策略网络输出的动作概率 (分割 vs 不分割)。

$$ L_{entropy} = - \sum \pi(a|s) \log \pi(a|s) $$
熵正则化项，鼓励策略保持随机性，防止过早收敛到局部最优（如永远不分割）。

$$ L_{aux} = L_{policy} - \lambda \cdot L_{entropy} $$

---

## 9.3 梯度流向

1.  **主干网络 (ViT)**:
    *   $L_{cls}$ 的梯度通过 Transformer、Embedding、Token Processor 回传。
    *   **注意**: 梯度**不会**通过 Tokenizer 的离散采样步骤回传到策略网络（因为采样操作不可导）。

2.  **策略网络 (Tokenizer)**:
    *   $L_{aux}$ 专门用于更新策略网络的参数。
    *   它不依赖于计算图的连通性，而是直接最大化期望奖励。

3.  **CNN 特征提取器 (MiniCNN)**:
    *   作为策略网络的一部分，它接收来自 $L_{aux}$ 的梯度更新。
