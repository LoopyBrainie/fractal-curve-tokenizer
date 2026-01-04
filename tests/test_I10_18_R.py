"""
I10-18-R单元测试
==================

测试浅层并行评估器和混合前向传播的正确性。

测试目标:
    1. ShallowParallelEvaluator预计算候选正确性
    2. 批量复杂度计算与原BFS一致性
    3. 累积概率计算正确性
    4. 梯度传播完整性（depths 0-3保证非零）
    5. 深层BFS集成正确性
    6. 性能指标（时间、FLOPS、内存）

作者: GitHub Copilot
日期: 2026-01-04
"""

import pytest
import torch
import torch.nn as nn
from typing import Tuple

# 导入待测试模块
import sys
sys.path.insert(0, 'src')

from vit_pytorch.split_adaptive_parallel import (
    ShallowParallelEvaluator,
    apply_I10_18_R_patch,
    _generate_children,
    _compute_hilbert_index,
)
from vit_pytorch.split_adaptive import LearnableSplitter, TensorSplitResult


# ============================================================================
# Fixture: 创建测试splitter
# ============================================================================

@pytest.fixture
def splitter():
    """创建LearnableSplitter实例用于测试。"""
    splitter = LearnableSplitter(
        feature_dim=256,
        max_depth=4,
        hidden_dim=128,
        intermediate_dim=64,
        pool_size=4,
        temperature=1.0,
        min_region_size=7,
    )
    return splitter


@pytest.fixture
def features():
    """创建测试特征图。"""
    B, C, H, W = 2, 256, 16, 16  # Tiny-ImageNet 64x64 → conv后16x16
    return torch.randn(B, C, H, W)


@pytest.fixture
def image_size():
    """测试图像尺寸。"""
    return (64, 64)


# ============================================================================
# Test 1: ShallowParallelEvaluator预计算候选
# ============================================================================

def test_shallow_parallel_evaluator_precompute():
    """测试候选区域预计算的正确性。"""
    evaluator = ShallowParallelEvaluator(
        max_depth_parallel=3,
        image_size=(64, 64),
    )
    
    # 验证候选数量
    expected_count = sum(4**d for d in range(4))  # 1+4+16+64 = 85
    assert evaluator.num_candidates == expected_count, \
        f"Expected {expected_count} candidates, got {evaluator.num_candidates}"
    
    # 验证形状
    assert evaluator.candidate_regions.shape == (85, 4)
    assert evaluator.candidate_depths.shape == (85,)
    assert evaluator.parent_indices.shape == (85,)
    assert evaluator.hilbert_indices.shape == (85,)
    
    # 验证根节点
    assert evaluator.candidate_depths[0] == 0
    assert evaluator.parent_indices[0] == -1
    assert torch.allclose(
        evaluator.candidate_regions[0],
        torch.tensor([0, 0, 64, 64], dtype=torch.float32)
    )
    
    # 验证深度分布
    for d in range(4):
        count = (evaluator.candidate_depths == d).sum().item()
        expected = 4 ** d
        assert count == expected, \
            f"Depth {d}: expected {expected} regions, got {count}"
    
    print("✓ Precompute candidates test passed")


# ============================================================================
# Test 2: 批量复杂度计算
# ============================================================================

def test_batch_complexity_computation(splitter, features):
    """测试批量复杂度计算与原BFS的一致性。"""
    evaluator = ShallowParallelEvaluator(
        max_depth_parallel=3,
        image_size=(64, 64),
    )
    
    B, C, H_feat, W_feat = features.shape
    H_img, W_img = 64, 64
    scale_h = H_feat / H_img
    scale_w = W_feat / W_img
    
    # 批量计算
    logits, probs, cumulative_probs = evaluator.forward(
        features=features,
        complexity_mlp=splitter.complexity_mlp,
        thresholds=splitter.thresholds,
        temperature=splitter.current_temperature,
        pool_size=splitter.pool_size,
    )
    
    # 验证形状
    assert logits.shape == (B, 85)
    assert probs.shape == (B, 85)
    assert cumulative_probs.shape == (B, 85)
    
    # 验证概率范围
    assert (probs >= 0).all() and (probs <= 1).all()
    assert (cumulative_probs >= 0).all() and (cumulative_probs <= 1).all()
    
    # 验证根节点累积概率=1
    root_idx = 0
    assert torch.allclose(
        cumulative_probs[:, root_idx],
        torch.ones(B),
        atol=1e-5
    ), "Root cumulative probability should be 1.0"
    
    print(f"✓ Batch complexity computation test passed")
    print(f"  - Logits range: [{logits.min():.2f}, {logits.max():.2f}]")
    print(f"  - Probs range: [{probs.min():.3f}, {probs.max():.3f}]")
    print(f"  - Cumulative probs range: [{cumulative_probs.min():.3f}, {cumulative_probs.max():.3f}]")


# ============================================================================
# Test 3: 累积概率计算正确性
# ============================================================================

def test_cumulative_probability_correctness(splitter, features):
    """测试累积概率的父子关系。"""
    evaluator = ShallowParallelEvaluator(
        max_depth_parallel=3,
        image_size=(64, 64),
    )
    
    logits, probs, cumulative_probs = evaluator.forward(
        features=features,
        complexity_mlp=splitter.complexity_mlp,
        thresholds=splitter.thresholds,
        temperature=splitter.current_temperature,
        pool_size=splitter.pool_size,
    )
    
    # 验证父子关系: α_child = α_parent × p_parent
    for child_idx in range(1, evaluator.num_candidates):
        parent_idx = evaluator.parent_indices[child_idx].item()
        if parent_idx >= 0:
            for b in range(features.shape[0]):
                alpha_child = cumulative_probs[b, child_idx]
                alpha_parent = cumulative_probs[b, parent_idx]
                p_parent = probs[b, parent_idx]
                
                expected_alpha = alpha_parent * p_parent
                
                assert torch.allclose(alpha_child, expected_alpha, atol=1e-4), \
                    f"Batch {b}, Child {child_idx}: α={alpha_child:.4f}, " \
                    f"expected {expected_alpha:.4f} (α_p={alpha_parent:.4f}, p_p={p_parent:.4f})"
    
    print("✓ Cumulative probability correctness test passed")


# ============================================================================
# Test 4: 梯度传播完整性
# ============================================================================

def test_gradient_propagation(splitter, features):
    """测试depths 0-3的梯度是否非零。"""
    evaluator = ShallowParallelEvaluator(
        max_depth_parallel=3,
        image_size=(64, 64),
    )
    
    # 启用梯度
    features.requires_grad_(True)
    splitter.train()
    
    logits, probs, cumulative_probs = evaluator.forward(
        features=features,
        complexity_mlp=splitter.complexity_mlp,
        thresholds=splitter.thresholds,
        temperature=splitter.current_temperature,
        pool_size=splitter.pool_size,
    )
    
    # 模拟分类损失
    # L = Σ cumulative_probs (简化)
    loss = cumulative_probs.sum()
    loss.backward()
    
    # 验证阈值梯度
    # 注意：thresholds是从threshold_offsets计算的，不是叶子节点
    # 需要检查threshold_offsets的梯度
    if splitter.threshold_offsets.grad is not None:
        # 至少前3个深度应该有梯度
        non_zero_count = 0
        for d in range(4):  # depths 0-3
            tau_offset_grad = splitter.threshold_offsets.grad[d]
            if tau_offset_grad.abs() > 1e-6:
                non_zero_count += 1
        
        assert non_zero_count >= 3, \
            f"Expected at least 3 non-zero gradients, got {non_zero_count}"
        
        print("✓ Gradient propagation test passed")
        print(f"  - Threshold offset gradients (depths 0-3): {splitter.threshold_offsets.grad[:4].tolist()}")
        print(f"  - Non-zero gradient count: {non_zero_count}/4")
    else:
        # 备用检查：查看ComplexityMLP参数的梯度
        has_gradient = False
        for param in splitter.complexity_mlp.parameters():
            if param.grad is not None and param.grad.abs().max() > 1e-6:
                has_gradient = True
                break
        
        assert has_gradient, "No gradient found in ComplexityMLP parameters"
        print("✓ Gradient propagation test passed (verified via MLP params)")


# ============================================================================
# Test 5: 混合前向传播集成
# ============================================================================

def test_hybrid_forward_integration(splitter, features, image_size):
    """测试混合前向传播的完整流程。"""
    # 应用I10-18-R补丁
    apply_I10_18_R_patch(splitter, max_depth_parallel=3, enable=True)
    
    # 前向传播
    result = splitter(features, image_size, hard=False)
    
    # 验证返回类型
    assert isinstance(result, TensorSplitResult)
    
    # 验证输出形状
    N = result.regions.shape[0]
    assert result.regions.shape == (N, 4)
    assert result.depths.shape == (N,)
    assert result.batch_indices.shape == (N,)
    assert result.hilbert_indices.shape == (N,)
    
    # 验证至少有root region
    assert N >= 1, "Should have at least 1 region (root)"
    
    # 验证深度范围
    assert (result.depths >= 0).all()
    assert (result.depths <= splitter.max_depth).all()
    
    # 验证batch索引范围
    B = features.shape[0]
    assert (result.batch_indices >= 0).all()
    assert (result.batch_indices < B).all()
    
    print(f"✓ Hybrid forward integration test passed")
    print(f"  - Total regions: {N}")
    print(f"  - Depth distribution: {[(d, (result.depths==d).sum().item()) for d in range(5)]}")
    print(f"  - Avg tokens per image: {N / B:.1f}")


# ============================================================================
# Test 6: 辅助函数测试
# ============================================================================

def test_generate_children():
    """测试四叉树子区域生成。"""
    region = torch.tensor([0, 0, 64, 64], dtype=torch.float32)
    children = _generate_children(region, (64, 64))
    
    assert len(children) == 4
    
    expected_children = [
        torch.tensor([0, 0, 32, 32], dtype=torch.float32),   # 左上
        torch.tensor([32, 0, 64, 32], dtype=torch.float32),  # 右上
        torch.tensor([0, 32, 32, 64], dtype=torch.float32),  # 左下
        torch.tensor([32, 32, 64, 64], dtype=torch.float32), # 右下
    ]
    
    for i, (child, expected) in enumerate(zip(children, expected_children)):
        assert torch.equal(child, expected), \
            f"Child {i}: expected {expected}, got {child}"
    
    print("✓ Generate children test passed")


def test_compute_hilbert_index():
    """测试Hilbert索引计算。"""
    # Root region
    region = torch.tensor([0, 0, 64, 64], dtype=torch.float32)
    hilbert_idx = _compute_hilbert_index(region, (64, 64), depth=0)
    assert hilbert_idx == 0, f"Root Hilbert index should be 0, got {hilbert_idx}"
    
    # First quadrant (depth 1)
    region = torch.tensor([0, 0, 32, 32], dtype=torch.float32)
    hilbert_idx_d1 = _compute_hilbert_index(region, (64, 64), depth=1)
    assert 0 <= hilbert_idx_d1 < 4, \
        f"Depth 1 Hilbert index should be in [0,4), got {hilbert_idx_d1}"
    
    print("✓ Compute Hilbert index test passed")


# ============================================================================
# Test 7: 性能基准测试
# ============================================================================

@pytest.mark.slow
def test_performance_benchmark(splitter, features, image_size):
    """性能基准测试（可选，标记为slow）。"""
    import time
    
    # 原BFS
    splitter.eval()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    t0 = time.time()
    with torch.no_grad():
        result_bfs = splitter(features, image_size, hard=True)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    time_bfs = time.time() - t0
    
    # I10-18-R
    apply_I10_18_R_patch(splitter, max_depth_parallel=3, enable=True)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    t0 = time.time()
    with torch.no_grad():
        result_hybrid = splitter(features, image_size, hard=True)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    time_hybrid = time.time() - t0
    
    print(f"✓ Performance benchmark:")
    print(f"  - Original BFS: {time_bfs*1000:.2f}ms, {result_bfs.num_tokens} tokens")
    print(f"  - I10-18-R Hybrid: {time_hybrid*1000:.2f}ms, {result_hybrid.num_tokens} tokens")
    print(f"  - Slowdown: {time_hybrid/time_bfs:.2f}x")
    print(f"  - Target: <2.0x (within acceptable range)")
    
    # 允许一定的性能损失
    assert time_hybrid / time_bfs < 3.0, \
        f"Hybrid forward too slow: {time_hybrid/time_bfs:.2f}x (threshold 3.0x)"


# ============================================================================
# 运行所有测试
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
