# Fractal Curve Tokenizer 改进计划

基于对项目的深入代码审查，本文档详细规划了各项改进工作。

## 优先级定义

| 优先级 | 说明 | 时间预估 |
|--------|------|----------|
| **P0** | 阻塞性问题，影响核心功能正确性 | 立即处理 |
| **P1** | 重要问题，影响性能或可维护性 | 1-2周内 |
| **P2** | 改进项，提升代码质量 | 迭代完善 |

---

## P0: 关键问题修复

### 1. 完善 REINFORCE 实现

**问题描述**：
当前 `get_tokenizer_loss()` 只计算了熵正则化，没有真正实现策略梯度的奖励反馈机制。

**影响**：
可学习的分割决策网络无法根据分类任务的表现进行有效优化。

**涉及文件**：
- `src/vit_pytorch/fractal_vit.py` (NextGenerationFractalViT.get_tokenizer_loss)
- `examples/training/train_fractal_vit.py` (训练循环)

**修改方案**：

```python
# fractal_vit.py - 修改 get_tokenizer_loss 方法
def get_tokenizer_loss(self, reward: Optional[float] = None, baseline: float = 0.0) -> torch.Tensor:
    """
    获取 tokenizer 的策略梯度损失
    
    Args:
        reward: 外部提供的奖励信号（如 -classification_loss 或 accuracy）
        baseline: 基线值，用于减少方差
    """
    loss = torch.tensor(0.0, device=self.aux_loss_weight.device)
    
    if hasattr(self.tokenizer, "saved_log_probs") and len(self.tokenizer.saved_log_probs) > 0:
        log_probs = torch.stack(self.tokenizer.saved_log_probs)
        
        # 真正的 REINFORCE 策略梯度
        if reward is not None:
            advantage = reward - baseline
            policy_loss = -advantage * log_probs.mean()
            loss = loss + policy_loss
        
        # 熵正则化 (鼓励探索)
        if len(self.tokenizer.saved_entropies) > 0:
            entropies = torch.stack(self.tokenizer.saved_entropies)
            entropy_loss = -0.01 * entropies.mean()
            loss = loss + entropy_loss
    
    # Token 数量正则化 (防止过多或过少 token)
    if hasattr(self, '_last_token_count'):
        target_tokens = 64  # 可配置的目标 token 数
        token_penalty = 0.001 * (self._last_token_count - target_tokens) ** 2
        loss = loss + token_penalty
    
    return loss * self.aux_loss_weight
```

```python
# train_fractal_vit.py - 修改训练循环
def train_one_epoch(...):
    # ... 前向传播 ...
    loss = F.cross_entropy(output, target)
    
    # 计算 REINFORCE 奖励
    if hasattr(model, "get_tokenizer_loss"):
        with torch.no_grad():
            # 奖励 = 负损失（损失越低，奖励越高）
            reward = -loss.item()
            # 使用移动平均作为基线
            if not hasattr(train_one_epoch, 'baseline'):
                train_one_epoch.baseline = reward
            else:
                train_one_epoch.baseline = 0.99 * train_one_epoch.baseline + 0.01 * reward
        
        aux_loss = model.get_tokenizer_loss(reward=reward, baseline=train_one_epoch.baseline)
        loss = loss + aux_loss
```

**验收标准**：
- [x] 策略梯度正确计算
- [x] 基线机制有效减少方差
- [x] 分割决策网络参数有梯度更新

---

### 2. 修复全局注意力 Mask 泄漏

**问题描述**：
`EnhancedFractalTransformer.forward()` 中的 `global_context_attn` 没有传入 attention mask，导致 padding token 会参与计算。

**影响**：
Padding 信息泄漏到有效 token，可能影响模型性能。

**涉及文件**：
- `src/vit_pytorch/transformer.py`

**修改方案**：

```python
# transformer.py - 修改 EnhancedFractalTransformer.forward()
def forward(
    self,
    x: torch.Tensor,
    levels_info: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    use_dynamic_depth: bool = False,
) -> torch.Tensor:
    # ... 现有代码 ...
    
    if seq_len > 1:
        # 修复：传入 key_padding_mask
        key_padding_mask = None
        if attention_mask is not None:
            # attention_mask: (B, 1, 1, S) 其中 True=保留, False=mask
            # key_padding_mask: (B, S) 其中 True=mask, False=保留
            key_padding_mask = ~attention_mask.squeeze(1).squeeze(1)
        
        global_context, _ = self.global_context_attn(
            x, x, x, 
            key_padding_mask=key_padding_mask
        )
        x = x + global_context * 0.1
    
    # ... 后续代码 ...
```

**验收标准**：
- [x] 修改 padding 输入后，有效 token 的输出不变
- [x] 添加相应的单元测试验证

---

## P1: 性能与可维护性改进

### 3. 向量化 create_attention_mask

**问题描述**：
当前使用双重循环实现，复杂度 O(B × S²)，效率低下。

**涉及文件**：
- `src/vit_pytorch/utils.py`

**修改方案**：

```python
def create_attention_mask(levels_info: List[torch.Tensor], device: torch.device) -> torch.Tensor:
    """向量化实现的层级感知注意力偏置"""
    if len(levels_info) == 0:
        return torch.empty(0, 0, 0, device=device)
    
    batch_size = len(levels_info)
    max_len = max(info.shape[0] for info in levels_info if info.numel() > 0)
    
    if max_len == 0:
        return torch.ones(batch_size, 1, 1, device=device)
    
    # 批量提取 depths 并 pad
    depths_list = []
    for info in levels_info:
        if info.numel() > 0:
            d = info[:, 0].float()
        else:
            d = torch.zeros(0, device=device)
        # Pad to max_len
        padded = F.pad(d, (0, max_len - len(d)), value=-1)  # -1 表示 padding
        depths_list.append(padded)
    
    depths = torch.stack(depths_list)  # (B, S)
    
    # 向量化计算层级差异
    depth_i = depths.unsqueeze(2)  # (B, S, 1)
    depth_j = depths.unsqueeze(1)  # (B, 1, S)
    diff = torch.abs(depth_i - depth_j)  # (B, S, S)
    
    # 生成 mask 值
    mask = torch.where(diff == 0, 1.2, 
           torch.where(diff == 1, 1.1, 1.0))
    
    # 处理 padding 位置
    valid_mask = (depths >= 0).unsqueeze(2) & (depths >= 0).unsqueeze(1)
    mask = mask * valid_mask.float()
    
    return mask
```

**验收标准**：
- [x] 输出与原实现一致
- [x] 移除 O(B × S²) 循环，使用向量化操作

---

### 4. 简化 Hilbert 曲线代码

**问题描述**：
存在多个功能相似的 Hilbert 相关函数，代码冗余。

**涉及文件**：
- `src/vit_pytorch/fractal_curve_tokenizer.py`

**修改方案**：

创建独立的 `hilbert.py` 模块：

```python
# src/vit_pytorch/hilbert.py
"""统一的 Hilbert 曲线工具模块"""

from typing import List, Tuple
from functools import lru_cache

class HilbertCurve:
    """Hilbert 曲线核心实现"""
    
    @staticmethod
    @lru_cache(maxsize=128)
    def xy_to_d(n: int, x: int, y: int) -> int:
        """将 2D 坐标转换为 Hilbert 曲线距离"""
        d = 0
        s = n // 2
        while s > 0:
            rx = 1 if (x & s) > 0 else 0
            ry = 1 if (y & s) > 0 else 0
            d += s * s * ((3 * rx) ^ ry)
            # 旋转
            if ry == 0:
                if rx == 1:
                    x, y = s - 1 - x, s - 1 - y
                x, y = y, x
            s //= 2
        return d
    
    @staticmethod
    def get_quadrant_order(level: int, aspect_ratio: float = 1.0) -> List[int]:
        """获取四象限的 Hilbert 遍历顺序"""
        # 标准 Hilbert 顺序（U形）
        base_orders = [
            [2, 0, 1, 3],  # 向上开口
            [0, 2, 3, 1],  # 向右开口
            [1, 3, 2, 0],  # 向下开口
            [3, 1, 0, 2],  # 向左开口
        ]
        
        orientation = level % 4
        order = base_orders[orientation]
        
        # 根据宽高比调整
        if aspect_ratio > 1.6:  # 宽矩形
            return [2, 0, 1, 3] if level % 2 == 0 else [0, 2, 3, 1]
        elif aspect_ratio < 0.625:  # 高矩形
            return [0, 1, 3, 2] if level % 2 == 0 else [2, 3, 1, 0]
        
        return order
```

然后在 `fractal_curve_tokenizer.py` 中简化引用：

```python
from .hilbert import HilbertCurve

class FractalHilbertTokenizer(BaseTokenizer):
    def _determine_traversal_order(self, level, h, w, num_patches):
        if num_patches == 4:
            aspect_ratio = w / h if h > 0 else 1.0
            return HilbertCurve.get_quadrant_order(level, aspect_ratio)
        return list(range(num_patches))
```

**验收标准**：
- [x] 删除冗余函数（迁移至 hilbert.py）
- [x] Hilbert 模块独立可测试（19 个专用测试）
- [x] 现有测试通过

---

### 5. 优化位置编码 Batch 处理

**问题描述**：
当前先 flatten 再 reshape，增加了不必要的内存操作。

**涉及文件**：
- `src/vit_pytorch/positional.py`

**修改方案**：

让 `AdvancedFractalPositionEmbedding.forward` 原生支持 3D 输入：

```python
def forward(
    self,
    levels_info: torch.Tensor,
    sequence_positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Args:
        levels_info: (N, info_len) 或 (B, N, info_len) 的层级信息
    Returns:
        (N, dim) 或 (B, N, dim) 的位置嵌入
    """
    if levels_info.numel() == 0:
        return torch.zeros(0, self.dim, device=levels_info.device)
    
    # 支持 2D 和 3D 输入
    is_batched = levels_info.dim() == 3
    if not is_batched:
        levels_info = levels_info.unsqueeze(0)  # (1, N, info)
    
    batch_size, seq_len, info_len = levels_info.shape
    device = levels_info.device
    
    depths = levels_info[..., 0].clamp(0, self.max_level).long()  # (B, N)
    paths = levels_info[..., 1:].long()  # (B, N, path_len)
    
    # 深度编码
    depth_emb = self.depth_embedding(depths)  # (B, N, dim)
    
    # 路径编码 (向量化)
    path_len = paths.shape[-1]
    level_offsets = torch.arange(path_len, device=device) * 4  # (path_len,)
    flat_indices = (paths + level_offsets).clamp(0, self.max_level * 4 - 1)  # (B, N, path_len)
    path_embs = self.quadrant_embedding(flat_indices)  # (B, N, path_len, dim)
    
    # 掩码：只保留有效层级
    seq_indices = torch.arange(path_len, device=device)  # (path_len,)
    mask = seq_indices < depths.unsqueeze(-1)  # (B, N, path_len)
    path_final = (path_embs * mask.unsqueeze(-1)).sum(dim=-2)  # (B, N, dim)
    
    # 融合
    combined_emb = depth_emb + path_final
    result = self.fusion_network(combined_emb)
    
    if not is_batched:
        result = result.squeeze(0)
    
    return result
```

**验收标准**：
- [x] 支持 2D 和 3D 输入（原已支持 `...` 广播）
- [x] 减少 fractal_vit.py 中的 reshape 操作
- [x] 向量化 get_attention_bias 方法

---

## P2: 代码质量改进

### 6. 完善测试覆盖

**新增测试用例**：

```python
# tests/unit/test_hilbert.py
class TestHilbertCurve:
    def test_xy_to_d_known_values(self):
        """验证 Hilbert 距离计算的正确性"""
        # n=4 的情况下，已知的坐标-距离映射
        known_mappings = [
            (4, 0, 0, 0),   # 左下角
            (4, 1, 0, 1),
            (4, 1, 1, 2),
            (4, 0, 1, 3),
            # ... 更多测试点
        ]
        for n, x, y, expected_d in known_mappings:
            assert HilbertCurve.xy_to_d(n, x, y) == expected_d
    
    def test_quadrant_order_preserves_locality(self):
        """验证象限顺序保持局部性"""
        order = HilbertCurve.get_quadrant_order(0)
        # 相邻象限应该在顺序中也相邻
        assert abs(order.index(0) - order.index(2)) <= 2  # 左上和左下
        assert abs(order.index(1) - order.index(3)) <= 2  # 右上和右下


# tests/unit/test_attention_mask.py
class TestAttentionMaskEffectiveness:
    def test_padding_tokens_ignored(self):
        """验证 padding token 不影响有效 token 的输出"""
        # ... 测试实现 ...


# tests/unit/test_reinforce.py
class TestREINFORCE:
    def test_policy_gradient_computation(self):
        """验证策略梯度计算正确性"""
        # ... 测试实现 ...
    
    def test_baseline_reduces_variance(self):
        """验证基线机制有效降低方差"""
        # ... 测试实现 ...
```

**验收标准**：
- [x] 新增 Hilbert 模块测试 (19 个用例)
- [x] 新增 REINFORCE 策略梯度测试 (10 个用例)
- [x] 新增 Attention Mask 有效性测试 (10 个用例)
- [x] 关键算法有独立测试

---

### 7. 数据增强策略优化

**涉及文件**：
- `examples/training/train_fractal_vit.py`

**修改方案**：

```python
def build_transforms(spec: DatasetSpec) -> Tuple[transforms.Compose, transforms.Compose]:
    # 根据数据集选择合适的增强策略
    augment_policies = {
        "CIFAR10": transforms.AutoAugmentPolicy.CIFAR10,
        "CIFAR100": transforms.AutoAugmentPolicy.CIFAR10,  # CIFAR10 策略对 CIFAR100 也有效
        "MNIST": None,  # MNIST 不使用 AutoAugment
        "ImageNet": transforms.AutoAugmentPolicy.IMAGENET,
        "COCO": transforms.AutoAugmentPolicy.IMAGENET,
        "Caltech256": transforms.AutoAugmentPolicy.IMAGENET,
        "TinyImageNet": transforms.AutoAugmentPolicy.IMAGENET,
    }
    
    policy = augment_policies.get(spec.name)
    
    if spec.name.lower() == "mnist":
        train_ops = [
            transforms.Resize(32),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(spec.mean, spec.std),
        ]
    else:
        train_ops = [
            transforms.Resize(spec.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(spec.image_size, padding=4),
        ]
        
        if policy is not None:
            train_ops.append(transforms.AutoAugment(policy))
        
        train_ops.extend([
            transforms.ToTensor(),
            transforms.Normalize(spec.mean, spec.std),
            transforms.RandomErasing(p=0.25),
        ])
    
    test_ops = [
        transforms.Resize(spec.image_size),
        transforms.ToTensor(),
        transforms.Normalize(spec.mean, spec.std),
    ]
    
    return transforms.Compose(train_ops), transforms.Compose(test_ops)
```

**验收标准**：
- [x] 各数据集使用对应的增强策略
- [x] MNIST 使用简单增强（RandomRotation）
- [x] CIFAR 使用 CIFAR10 AutoAugment
- [x] ImageNet 类使用 IMAGENET AutoAugment

---

## 执行计划

### Phase 1: 关键修复 (Week 1) ✅
- [x] 完成 P0-1: REINFORCE 实现
- [x] 完成 P0-2: 全局注意力 mask 修复
- [x] 运行全部测试确保无回归

### Phase 2: 性能优化 (Week 2) ✅
- [x] 完成 P1-3: 向量化 attention mask
- [x] 完成 P1-4: Hilbert 代码重构
- [x] 完成 P1-5: 位置编码优化

### Phase 3: 质量提升 (Week 3) ✅
- [x] 完成 P2-6: 测试覆盖
- [x] 完成 P2-7: 数据增强优化

---

## 附录：代码变更跟踪

| 文件 | 变更类型 | 关联任务 | 状态 |
|------|----------|----------|------|
| `fractal_vit.py` | 修改 | P0-1, P1-5 | ✅ |
| `train_fractal_vit.py` | 修改 | P0-1, P2-7 | ✅ |
| `transformer.py` | 修改 | P0-2 | ✅ |
| `utils.py` | 修改 | P1-3 | ✅ |
| `fractal_curve_tokenizer.py` | 重构 | P1-4 | ✅ |
| `hilbert.py` | 新增 | P1-4 | ✅ |
| `positional.py` | 修改 | P1-5 | ✅ |
| `tests/unit/test_hilbert.py` | 新增 | P2-6 | ✅ |
| `tests/unit/test_reinforce.py` | 新增 | P2-6 | ✅ |
| `tests/unit/test_attention_mask.py` | 新增 | P2-6 | ✅ |
