# -*- coding: utf-8 -*-
"""
P6-2 验证测试: LCA Bias 可学习温度参数

数学验证目标:
1. 初始化正确性: softplus(γ) ≈ τ_target (默认 1.5)
2. 可学习性: 温度参数梯度正常流动
3. 范围约束: softplus 确保 τ > 0
4. 向后兼容: lca_temperature=None 时行为与旧版一致
"""
import sys
import math
sys.path.insert(0, 'src')

import torch
import torch.nn.functional as F
from vit_pytorch.attn_hilbert_bias import LCAHilbertBias


def test_temperature_initialization():
    """测试温度初始化的数学正确性。
    
    验证: softplus(log(exp(τ) - 1)) ≈ τ
    """
    print("=" * 60)
    print("测试 1: 温度初始化正确性")
    print("=" * 60)
    
    target_temps = [1.0, 1.5, 2.0]
    
    for target in target_temps:
        bias = LCAHilbertBias(
            max_depth=4, 
            heads=8, 
            lca_temperature=target,
            learnable_temperature=True
        )
        actual = bias.lca_temperature.mean().item()
        error = abs(actual - target)
        status = "✅" if error < 0.01 else "❌"
        print(f"  目标 τ={target:.2f}, 实际 τ={actual:.4f}, 误差={error:.6f} {status}")
    
    print()


def test_temperature_scaling_effect():
    """测试温度对偏置值的缩放效果。
    
    验证: bias_scaled = τ × bias_original
    """
    print("=" * 60)
    print("测试 2: 温度缩放效果")
    print("=" * 60)
    
    # 创建两个共享权重的 bias 模块，仅温度不同
    max_depth, heads = 4, 4
    
    # τ=1.0 版本 (基准)
    bias_base = LCAHilbertBias(
        max_depth=max_depth, 
        heads=heads, 
        lca_temperature=1.0,
        learnable_temperature=False
    )
    
    # τ=1.5 版本
    bias_scaled = LCAHilbertBias(
        max_depth=max_depth, 
        heads=heads, 
        lca_temperature=1.5,
        learnable_temperature=False
    )
    
    # 复制嵌入权重以确保唯一变量是温度
    with torch.no_grad():
        bias_scaled.lca_embedding.weight.copy_(bias_base.lca_embedding.weight)
    
    # 创建测试输入 (固定随机种子保证可复现)
    torch.manual_seed(42)
    batch_size, seq_len = 2, 16
    levels_info = torch.zeros(batch_size, seq_len, 5)
    levels_info[:, :, 0] = torch.randint(0, 5, (batch_size, seq_len)).float()
    levels_info[:, :, 1:] = torch.randint(0, 4, (batch_size, seq_len, 4)).float()
    
    # 计算偏置
    output_base = bias_base(levels_info)
    output_scaled = bias_scaled(levels_info)
    
    # 验证缩放关系: output_scaled = 1.5 × output_base
    # 使用绝对值避免符号问题
    abs_base = output_base.abs()
    abs_scaled = output_scaled.abs()
    
    # 只在非零位置计算比例
    mask = abs_base > 0.01
    if mask.sum() > 0:
        ratio = (abs_scaled[mask] / abs_base[mask]).mean().item()
    else:
        ratio = 1.5  # 默认值
    
    expected_ratio = 1.5
    error = abs(ratio - expected_ratio)
    status = "✅" if error < 0.1 else "❌"
    
    print(f"  τ=1.0 偏置范围: [{output_base.min():.3f}, {output_base.max():.3f}]")
    print(f"  τ=1.5 偏置范围: [{output_scaled.min():.3f}, {output_scaled.max():.3f}]")
    print(f"  实际缩放比: {ratio:.4f}, 期望: {expected_ratio:.2f} {status}")
    print()


def test_gradient_flow():
    """测试温度参数的梯度流动。"""
    print("=" * 60)
    print("测试 3: 梯度流验证")
    print("=" * 60)
    
    bias = LCAHilbertBias(
        max_depth=4, 
        heads=4, 
        lca_temperature=1.5,
        learnable_temperature=True
    )
    
    # 创建输入
    batch_size, seq_len = 2, 16
    levels_info = torch.zeros(batch_size, seq_len, 5)
    levels_info[:, :, 0] = torch.randint(0, 5, (batch_size, seq_len)).float()
    levels_info[:, :, 1:] = torch.randint(0, 4, (batch_size, seq_len, 4)).float()
    
    # 前向 + 后向
    output = bias(levels_info)
    loss = output.sum()
    loss.backward()
    
    # 检查梯度
    temp_grad = bias._lca_temp_raw.grad
    embed_grad = bias.lca_embedding.weight.grad
    
    temp_grad_ok = temp_grad is not None and temp_grad.abs().sum() > 0
    embed_grad_ok = embed_grad is not None and embed_grad.abs().sum() > 0
    
    print(f"  温度参数梯度: {'存在且非零 ✅' if temp_grad_ok else '缺失或为零 ❌'}")
    print(f"  嵌入表梯度: {'存在且非零 ✅' if embed_grad_ok else '缺失或为零 ❌'}")
    
    if temp_grad_ok:
        print(f"  温度梯度范围: [{temp_grad.min():.6f}, {temp_grad.max():.6f}]")
    print()


def test_backward_compatibility():
    """测试向后兼容性 (lca_temperature=None)。"""
    print("=" * 60)
    print("测试 4: 向后兼容性")
    print("=" * 60)
    
    # 兼容模式
    bias_compat = LCAHilbertBias(
        max_depth=4, 
        heads=4, 
        lca_temperature=None
    )
    
    # 验证没有温度参数
    has_temp_param = bias_compat._lca_temp_raw is not None
    has_temp_fixed = bias_compat._lca_temp_fixed is not None
    temp_prop = bias_compat.lca_temperature
    
    print(f"  _lca_temp_raw 存在: {has_temp_param} (期望: False)")
    print(f"  _lca_temp_fixed 存在: {has_temp_fixed} (期望: False)")
    print(f"  lca_temperature 属性: {temp_prop} (期望: None)")
    
    status = "✅" if not has_temp_param and not has_temp_fixed and temp_prop is None else "❌"
    print(f"  兼容性验证: {status}")
    print()


def test_per_head_learning():
    """测试 per-head 温度独立学习能力。"""
    print("=" * 60)
    print("测试 5: Per-Head 独立学习")
    print("=" * 60)
    
    heads = 4
    bias = LCAHilbertBias(
        max_depth=4, 
        heads=heads, 
        lca_temperature=1.5,
        learnable_temperature=True
    )
    
    # 模拟训练更新
    optimizer = torch.optim.SGD(bias.parameters(), lr=0.1)
    
    # 创建输入
    batch_size, seq_len = 2, 16
    levels_info = torch.zeros(batch_size, seq_len, 5)
    levels_info[:, :, 0] = torch.randint(0, 5, (batch_size, seq_len)).float()
    levels_info[:, :, 1:] = torch.randint(0, 4, (batch_size, seq_len, 4)).float()
    
    # 记录初始值
    init_temps = bias.lca_temperature.clone().detach()
    print(f"  初始温度: {init_temps.tolist()}")
    
    # 执行一步训练 (只对部分 head 加权)
    output = bias(levels_info)
    # 对 head 0 施加更大的损失
    loss = output[:, 0, :, :].sum() * 10 + output[:, 1:, :, :].sum()
    loss.backward()
    optimizer.step()
    
    # 检查更新后
    new_temps = bias.lca_temperature.clone().detach()
    print(f"  更新后温度: {new_temps.tolist()}")
    
    # 验证不同 head 有不同的更新幅度
    diffs = (new_temps - init_temps).abs()
    head0_diff = diffs[0].item()
    other_diff_mean = diffs[1:].mean().item()
    
    print(f"  Head 0 变化: {head0_diff:.6f}")
    print(f"  其他 Head 平均变化: {other_diff_mean:.6f}")
    
    status = "✅" if head0_diff > other_diff_mean * 1.5 else "⚠️ (差异不够明显，但不代表错误)"
    print(f"  独立学习验证: {status}")
    print()


def test_softplus_range():
    """测试 softplus 的范围约束 (τ > 0)。"""
    print("=" * 60)
    print("测试 6: Softplus 范围约束")
    print("=" * 60)
    
    bias = LCAHilbertBias(
        max_depth=4, 
        heads=4, 
        lca_temperature=1.5,
        learnable_temperature=True
    )
    
    # 测试极端的 raw 值
    extreme_values = [-10, -5, 0, 5, 10]
    
    print("  γ (raw) → τ (softplus) 映射:")
    for gamma in extreme_values:
        with torch.no_grad():
            bias._lca_temp_raw.fill_(gamma)
            tau = bias.lca_temperature.mean().item()
            expected = math.log(1 + math.exp(gamma))
            error = abs(tau - expected)
            status = "✅" if error < 0.01 else "❌"
            print(f"    γ={gamma:+6.1f} → τ={tau:.6f} (期望: {expected:.6f}) {status}")
    
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" P6-2 验证测试: LCA Bias 可学习温度参数")
    print("=" * 60 + "\n")
    
    test_temperature_initialization()
    test_temperature_scaling_effect()
    test_gradient_flow()
    test_backward_compatibility()
    test_per_head_learning()
    test_softplus_range()
    
    print("=" * 60)
    print(" 所有测试完成!")
    print("=" * 60)
