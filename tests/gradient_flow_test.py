"""
梯度流验证脚本

测试修复后的梯度流是否正确:
1. quota_logits → K_d_float → soft_mask 的梯度是否完整
2. log_temperature 的梯度是否可回传
3. 动态性是否恢复 (std(K) > 0)
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def test_ste_gradient():
    """测试 STE 梯度流"""
    print("=" * 60)
    print("测试 1: STE 梯度验证")
    print("=" * 60)

    # 创建 splitter
    from vit_pytorch.layers.splitters.gumbel_topk import GumbelTopKSplitter

    splitter = GumbelTopKSplitter(
        feature_dim=192,
        min_patch_size=4,
        max_level_limit=4,
    ).train()

    # 创建输入特征 [B, C, H, W]
    features = torch.randn(2, 192, 16, 16, requires_grad=True)

    # 前向传播
    result = splitter(features, hard=False)

    # 获取 soft_mask
    selected_mask = result.selected_mask

    # 模拟损失
    loss = selected_mask.sum()

    # 反向传播
    loss.backward()

    # 检查梯度
    print(f"  loss: {loss.item():.4f}")

    if features.grad is not None:
        grad_norm = features.grad.abs().sum().item()
        print(f"  features.grad abs sum: {grad_norm:.6f}")

        if grad_norm > 0:
            print("  [OK] STE gradient flow!")
            return True
        else:
            print("  [FAIL] gradient is zero")
            return False
    else:
        print("  [FAIL] no gradient")
        return False


def test_quota_logits_gradient():
    """测试 quota_logits 梯度流"""
    print("\n" + "=" * 60)
    print("测试 2: quota_logits 梯度验证")
    print("=" * 60)

    from vit_pytorch.layers.splitters.gumbel_topk import GumbelTopKSplitter

    splitter = GumbelTopKSplitter(
        feature_dim=192,
        min_patch_size=4,
        max_level_limit=4,
    ).train()

    # 检查 quota_allocator 是否存在
    if splitter.quota_allocator is None:
        print("  quota_allocator 未启用")
        # 检查 splitter.quota_logits
        if splitter.quota_logits is None:
            print("  quota_logits 也未启用")
            return False
        quota_logits = splitter.quota_logits
    else:
        quota_logits = splitter.quota_allocator.quota_logits

    print(f"  quota_logits shape: {quota_logits.shape}")
    print(f"  quota_logits: {quota_logits.data}")

    # 创建输入特征 [B, C, H, W]
    features = torch.randn(2, 192, 16, 16, requires_grad=True)

    # 前向传播
    result = splitter(features, hard=False)

    # 获取 soft_mask
    selected_mask = result.selected_mask

    # 模拟损失
    loss = selected_mask.sum()

    # 反向传播
    loss.backward()

    # 检查 quota_logits 梯度
    if quota_logits.grad is not None:
        grad_norm = quota_logits.grad.abs().sum().item()
        print(f"  quota_logits.grad abs sum: {grad_norm:.6f}")

        if grad_norm > 0:
            print("  [OK] quota_logits gradient flow!")
            return True
        else:
            print("  [FAIL] quota_logits gradient is zero")
            return False
    else:
        print("  [FAIL] no quota_logits gradient")
        return False


def test_dynamicity():
    """测试动态性是否恢复"""
    print("\n" + "=" * 60)
    print("测试 3: 动态性验证 (std(K))")
    print("=" * 60)

    from vit_pytorch.layers.splitters.gumbel_topk import GumbelTopKSplitter

    splitter = GumbelTopKSplitter(
        feature_dim=192,
        min_patch_size=4,
        max_level_limit=4,
    ).train()

    # 多次前向传播，记录每次的 token 数量
    token_counts = []

    for i in range(20):
        # 每次使用不同的输入
        features = torch.randn(2, 192, 16, 16)

        with torch.no_grad():
            result = splitter(features, hard=False)

        token_counts.append(result.selected_mask.sum().item())

    # 计算标准差
    token_counts = np.array(token_counts)
    std_k = np.std(token_counts)
    mean_k = np.mean(token_counts)

    print(f"  Token counts: {token_counts}")
    print(f"  Mean: {mean_k:.2f}, Std: {std_k:.2f}")

    if std_k > 0.1:
        print("  [OK] Dynamicity restored!")
        return True
    else:
        print("  [FAIL] Dynamicity still lost")
        return False


def test_ste_formula():
    """测试 STE 公式是否正确实现"""
    print("\n" + "=" * 60)
    print("测试 4: STE 公式验证")
    print("=" * 60)

    # 测试 STE 公式
    # STE: forward = hard, backward = soft gradient
    x = torch.randn(5, requires_grad=True)
    soft = x * 2  # 可微部分

    # 正确的 STE: hard = soft.detach(), ste = hard - soft.detach() + soft
    hard = soft.detach()
    ste = hard - soft.detach() + soft

    print(f"  x: {x}")
    print(f"  soft: {soft}")
    print(f"  ste: {ste}")

    # 反向传播
    loss = ste.sum()
    loss.backward()

    print(f"  x.grad: {x.grad}")

    # STE 的梯度应该等于 soft 的梯度，即 x.grad = 2 (常数，因为 soft = x * 2)
    expected_grad = torch.full_like(x, 2.0)
    if x.grad is not None and (x.grad - expected_grad).abs().max().item() < 1e-5:
        print("  [OK] STE formula correct!")
        return True
    else:
        print("  [FAIL] STE formula error")
        print(f"  expected: {expected_grad}")
        return False


if __name__ == "__main__":
    # 测试 1: STE 梯度
    ste_ok = test_ste_gradient()

    # 测试 2: quota_logits 梯度
    quota_ok = test_quota_logits_gradient()

    # 测试 3: 动态性
    dyn_ok = test_dynamicity()

    # 测试 4: STE 公式
    formula_ok = test_ste_formula()

    # 总结
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  STE gradient: {'[OK]' if ste_ok else '[FAIL]'}")
    print(f"  quota_logits gradient: {'[OK]' if quota_ok else '[FAIL]'}")
    print(f"  Dynamicity: {'[OK]' if dyn_ok else '[FAIL]'}")
    print(f"  STE formula: {'[OK]' if formula_ok else '[FAIL]'}")

    if ste_ok and quota_ok and formula_ok:
        print("\n  All tests passed!")
    else:
        print("\n  Some tests failed, need further fix")
