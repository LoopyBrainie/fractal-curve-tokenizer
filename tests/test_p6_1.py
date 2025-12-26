# -*- coding: utf-8 -*-
"""P6-1 测试: depth_scale 可学习化验证"""
import sys
sys.path.insert(0, 'src')

import torch
from vit_pytorch.embed_hilbert_patch import HilbertNativePatchEmbed

def test_p6_1():
    print('=== P6-1 测试: depth_scale 可学习化 ===\n')
    
    # 测试 1: 新版初始化 (默认)
    embed_new = HilbertNativePatchEmbed(
        channels=3, dim=256, max_depth=4,
        depth_scale_range=(0.5, 2.0), depth_scale_beta=0.2
    )
    
    print('新版 (sigmoid 参数化):')
    print(f'  原始参数 gamma: {embed_new._depth_scale_raw.data.tolist()}')
    print(f'  缩放因子 sigma: {embed_new.depth_scale.data.tolist()}')
    print(f'  可学习范围: [{embed_new.depth_scale_range[0]}, {embed_new.depth_scale_range[1]}]')
    
    # 测试 2: 旧版初始化 (向后兼容)
    embed_old = HilbertNativePatchEmbed(
        channels=3, dim=256, max_depth=4,
        depth_scale_range=None, depth_scale_beta=0.2
    )
    
    print('\n旧版 (固定线性初始化):')
    print(f'  缩放因子 sigma: {embed_old.depth_scale.data.tolist()}')
    
    # 测试 3: 验证初始值一致性
    print('\n初始值比较:')
    for d in range(5):
        new_val = embed_new.depth_scale[d].item()
        old_val = embed_old.depth_scale[d].item()
        diff = abs(new_val - old_val)
        print(f'  d={d}: 新版={new_val:.4f}, 旧版={old_val:.4f}, 差异={diff:.6f}')
        assert diff < 0.001, f"初始值差异过大: {diff}"
    
    # 测试 4: 验证学习能力
    print('\n学习能力验证 (模拟极端梯度):')
    embed_new._depth_scale_raw.data = torch.tensor([-10.0, -5.0, 0.0, 5.0, 10.0])
    scales = embed_new.depth_scale.data.tolist()
    print(f'  极端原始参数 gamma: [-10, -5, 0, 5, 10]')
    print(f'  对应缩放因子 sigma: {[f"{s:.4f}" for s in scales]}')
    
    dynamic_range = embed_new.depth_scale.max().item() / embed_new.depth_scale.min().item()
    print(f'  动态范围: {dynamic_range:.2f}x')
    
    # 验证范围约束
    assert embed_new.depth_scale.min().item() >= 0.5, "sigma_min 约束失败"
    assert embed_new.depth_scale.max().item() <= 2.0, "sigma_max 约束失败"
    assert dynamic_range >= 3.5, "动态范围不足"
    
    # 测试 5: 梯度流验证
    print('\n梯度流验证:')
    embed_test = HilbertNativePatchEmbed(channels=3, dim=256, max_depth=4)
    x = torch.randn(1, 3, 64, 64, requires_grad=True)
    
    # 创建模拟的 SplitResult
    from vit_pytorch.split_adaptive import Region, SplitToken, SplitResult
    tokens = [
        SplitToken(Region(0, 0, 32, 32), depth=1, path=[0], hilbert_idx=0, complexity=0.5),
        SplitToken(Region(32, 0, 64, 32), depth=2, path=[1, 0], hilbert_idx=1, complexity=0.5),
        SplitToken(Region(0, 32, 64, 64), depth=0, path=[], hilbert_idx=2, complexity=0.3),
    ]
    split_result = SplitResult(tokens)
    
    output, _ = embed_test(x, [split_result])
    loss = output.sum()
    loss.backward()
    
    print(f'  depth_scale_raw 梯度: {embed_test._depth_scale_raw.grad is not None}')
    if embed_test._depth_scale_raw.grad is not None:
        print(f'  梯度值: {embed_test._depth_scale_raw.grad.data.tolist()}')
    
    print('\n✅ P6-1 测试全部通过!')

if __name__ == '__main__':
    test_p6_1()
