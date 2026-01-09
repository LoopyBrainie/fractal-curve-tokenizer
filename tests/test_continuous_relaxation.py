"""
I10-19 连续松弛单元测试

验证架构完整性：
1. 参数正确传递
2. ShallowParallelEvaluator正确初始化
3. 补偿机制在连续模式下被跳过
4. 基本的forward pass可以运行

NOTE (I20): StreamingFractalTokenizerV3 现在使用 GumbelTopKSplitter (Scheme D)，
部分针对 Scheme B (连续松弛) 的测试已被跳过。
"""
import sys
from pathlib import Path

# 添加src目录到路径
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import pytest
import torch
import torch.nn as nn
from vit_pytorch.split_adaptive import LearnableSplitter, ShallowCandidateProbs
from vit_pytorch.split_adaptive_parallel import ShallowParallelEvaluator
from vit_pytorch.tokenizer_streaming import StreamingFractalTokenizerV3


def test_learnable_splitter_continuous_param():
    """测试LearnableSplitter接受use_continuous_relaxation参数."""
    splitter = LearnableSplitter(
        feature_dim=128,
        max_depth=3,
        hidden_dim=64,
        pool_size=4,
        temperature=1.0,
        use_gumbel=True,
        enforce_balance=True,
        min_region_size=8,
        init_tau_base=0.5,
        init_tau_gamma=0.85,
        use_continuous_relaxation=True,
    )
    
    assert hasattr(splitter, 'use_continuous_relaxation')
    assert splitter.use_continuous_relaxation == True
    assert hasattr(splitter, 'shallow_evaluator')
    assert splitter.shallow_evaluator is not None
    assert isinstance(splitter.shallow_evaluator, ShallowParallelEvaluator)
    print("✓ LearnableSplitter correctly accepts continuous_relaxation parameter")


def test_learnable_splitter_discrete_mode():
    """测试离散模式下不创建ShallowParallelEvaluator."""
    splitter = LearnableSplitter(
        feature_dim=128,
        max_depth=3,
        hidden_dim=64,
        pool_size=4,
        temperature=1.0,
        use_gumbel=True,
        enforce_balance=True,
        min_region_size=8,
        init_tau_base=0.5,
        init_tau_gamma=0.85,
        use_continuous_relaxation=False,
    )
    
    assert hasattr(splitter, 'use_continuous_relaxation')
    assert splitter.use_continuous_relaxation == False
    assert splitter.shallow_evaluator is None
    print("✓ Discrete mode does not create ShallowParallelEvaluator")


@pytest.mark.skip(reason="I20: StreamingFractalTokenizerV3 now uses GumbelTopKSplitter (Scheme D), not LearnableSplitter with continuous relaxation (Scheme B)")
def test_tokenizer_continuous_param():
    """测试StreamingFractalTokenizerV3接受连续松弛参数."""
    tokenizer = StreamingFractalTokenizerV3(
        image_size=64,
        channels=3,
        d_model=128,
        base_patch_size=4,
        max_depth=3,
        use_hilbert_order=True,
        target_tokens=64,
        enforce_balance=True,
        depth_scale_range=(0.5, 2.0),
        gamma=0.85,
        learnable_temperature=1.0,
        use_gumbel=True,
        use_continuous_relaxation=True,
        continuous_max_depth=3,
    )
    
    assert tokenizer.splitter.use_continuous_relaxation == True
    assert tokenizer.splitter.shallow_evaluator is not None
    print("✓ Tokenizer correctly passes continuous_relaxation to splitter")


def test_forward_pass_continuous_mode():
    """测试连续松弛模式下的基本forward pass."""
    torch.manual_seed(42)
    
    # 创建splitter
    splitter = LearnableSplitter(
        feature_dim=128,
        max_depth=3,
        hidden_dim=64,
        pool_size=4,
        temperature=1.0,
        use_gumbel=False,  # 关闭Gumbel以简化测试
        enforce_balance=True,
        min_region_size=8,
        init_tau_base=0.5,
        init_tau_gamma=0.85,
        use_continuous_relaxation=True,
    )
    
    # 创建测试输入
    B, C, H, W = 2, 128, 16, 16
    features = torch.randn(B, C, H, W)
    image_size = (64, 64)
    
    # Forward pass
    result = splitter(features, image_size, hard=False)
    
    # I10-19更新: 连续模式返回ShallowCandidateProbs而非TensorSplitResult
    from vit_pytorch.split_adaptive import ShallowCandidateProbs
    
    if isinstance(result, ShallowCandidateProbs):
        # 验证ShallowCandidateProbs结构
        assert hasattr(result, 'candidate_regions')
        assert hasattr(result, 'candidate_depths')
        assert hasattr(result, 'probs')
        assert hasattr(result, 'cumulative_probs')
        assert hasattr(result, 'parent_indices')
        
        # 验证形状
        assert result.probs.shape[0] == B
        N = result.num_candidates
        assert result.probs.shape[1] == N
        assert result.cumulative_probs.shape == (B, N)
        
        print(f"✓ Forward pass successful in continuous mode (ShallowCandidateProbs)")
        print(f"  - Batch size: {B}")
        print(f"  - Candidates: {N}")
        print(f"  - Probs shape: {result.probs.shape}")
    else:
        # 旧的TensorSplitResult结构验证
        assert hasattr(result, 'regions')
        assert hasattr(result, 'depths')
        assert hasattr(result, 'batch_indices')
        assert hasattr(result, 'hilbert_indices')
        assert hasattr(result, 'tokens_per_batch')
        
        # 验证batch数量
        assert result.tokens_per_batch.shape[0] == B
        assert result.tokens_per_batch.sum() == result.regions.shape[0]
        
        print(f"✓ Forward pass successful in continuous mode (TensorSplitResult)")
        print(f"  - Batch size: {B}")
        print(f"  - Total tokens: {result.regions.shape[0]}")
        print(f"  - Tokens per batch: {result.tokens_per_batch.tolist()}")


def test_forward_pass_discrete_vs_continuous_structure():
    """测试离散和连续模式输出结构对比.
    
    I10-19更新: 连续模式现在返回ShallowCandidateProbs而非TensorSplitResult。
    两种模式输出结构不同是设计决策，不再要求结构一致。
    """
    torch.manual_seed(42)
    
    from vit_pytorch.split_adaptive import ShallowCandidateProbs
    
    # 创建离散模式splitter
    splitter_discrete = LearnableSplitter(
        feature_dim=128,
        max_depth=2,
        hidden_dim=64,
        pool_size=4,
        temperature=1.0,
        use_gumbel=False,
        enforce_balance=False,
        min_region_size=8,
        init_tau_base=0.5,
        init_tau_gamma=0.85,
        use_continuous_relaxation=False,
    )
    
    # 创建连续模式splitter
    splitter_continuous = LearnableSplitter(
        feature_dim=128,
        max_depth=2,
        hidden_dim=64,
        pool_size=4,
        temperature=1.0,
        use_gumbel=False,
        enforce_balance=False,
        min_region_size=8,
        init_tau_base=0.5,
        init_tau_gamma=0.85,
        use_continuous_relaxation=True,
    )
    
    # 测试输入
    B, C, H, W = 2, 128, 16, 16
    features = torch.randn(B, C, H, W)
    image_size = (64, 64)
    
    # 两种模式的forward pass
    result_discrete = splitter_discrete(features, image_size, hard=True)
    result_continuous = splitter_continuous(features, image_size, hard=True)
    
    # 验证离散模式: TensorSplitResult
    assert hasattr(result_discrete, 'regions')
    assert hasattr(result_discrete, 'depths')
    assert hasattr(result_discrete, 'batch_indices')
    assert hasattr(result_discrete, 'tokens_per_batch')
    assert result_discrete.regions.shape[1] == 4
    
    # 验证连续模式: ShallowCandidateProbs
    assert isinstance(result_continuous, ShallowCandidateProbs)
    assert hasattr(result_continuous, 'candidate_regions')
    assert hasattr(result_continuous, 'candidate_depths')
    assert hasattr(result_continuous, 'probs')
    assert hasattr(result_continuous, 'cumulative_probs')
    
    print("✓ Discrete and continuous modes have distinct output structures (by design)")
    print(f"  - Discrete: TensorSplitResult with {result_discrete.regions.shape[0]} tokens")
    print(f"  - Continuous: ShallowCandidateProbs with {result_continuous.num_candidates} candidates")


def test_shallow_evaluator_candidate_count():
    """测试ShallowParallelEvaluator的候选数量计算."""
    for max_depth in [0, 1, 2, 3]:
        evaluator = ShallowParallelEvaluator(
            max_depth_parallel=max_depth,
            image_size=(64, 64),
        )
        
        # 预期候选数量: 1 + 4 + 16 + ... + 4^max_depth
        expected = sum(4**d for d in range(max_depth + 1))
        actual = evaluator.num_candidates
        
        assert actual == expected, \
            f"max_depth={max_depth}: expected {expected}, got {actual}"
        
        print(f"✓ max_depth={max_depth}: {actual} candidates (correct)")


def test_continuous_mode_depth_tracking():
    """测试连续松弛模式下的 depth 统计追踪 (I10-21 修复验证)."""
    torch.manual_seed(42)
    
    tokenizer = StreamingFractalTokenizerV3(
        image_size=64,
        channels=3,
        d_model=128,
        base_patch_size=4,
        max_depth=3,
        use_hilbert_order=True,
        target_tokens=64,
        enforce_balance=True,
        depth_scale_range=(0.5, 2.0),
        gamma=0.85,
        learnable_temperature=1.0,
        use_gumbel=False,
        use_continuous_relaxation=True,
        continuous_max_depth=3,
    )
    
    # Forward pass
    B = 2
    images = torch.randn(B, 3, 64, 64)
    output = tokenizer(images)  # TokenizerOutput object
    
    # 验证 forward pass 成功
    assert output is not None
    assert len(output) == B
    
    # 验证 depth count matrix 被正确设置 (I10-21 核心修复)
    assert tokenizer._last_depth_count_matrix is not None, \
        "_last_depth_count_matrix should NOT be None in continuous mode after I10-21 fix"
    
    # 验证 shape
    max_d = tokenizer.splitter.max_depth + 1
    assert tokenizer._last_depth_count_matrix.shape == (B, max_d), \
        f"Expected shape ({B}, {max_d}), got {tokenizer._last_depth_count_matrix.shape}"
    
    # 验证 depth counts 有效 (非全零)
    total_counts = tokenizer._last_depth_count_matrix.sum()
    assert total_counts > 0, "Depth count matrix should have non-zero entries"
    
    # 验证 get_scale_entropy 返回有效值
    entropy = tokenizer.get_scale_entropy()
    assert entropy is not None, "get_scale_entropy should return a value in continuous mode"
    assert not torch.isnan(torch.tensor(entropy)), "entropy should not be NaN"
    
    print(f"✓ Continuous mode depth tracking works correctly")
    print(f"  - Depth count matrix shape: {tokenizer._last_depth_count_matrix.shape}")
    print(f"  - Total token counts: {total_counts}")
    print(f"  - Scale entropy: {entropy:.4f}")


def test_continuous_mode_nan_protection():
    """测试连续松弛模式下的 NaN 保护 (I10-21 修复验证)."""
    torch.manual_seed(123)
    
    # 创建 splitter
    splitter = LearnableSplitter(
        feature_dim=128,
        max_depth=3,
        hidden_dim=64,
        pool_size=4,
        temperature=1.0,
        use_gumbel=False,
        enforce_balance=True,
        min_region_size=8,
        init_tau_base=0.5,
        init_tau_gamma=0.85,
        use_continuous_relaxation=True,
    )
    
    # 创建测试输入
    B, C, H, W = 2, 128, 16, 16
    features = torch.randn(B, C, H, W)
    image_size = (64, 64)
    
    # Forward pass 获取 shallow candidate probs
    result = splitter(features, image_size, hard=False)
    assert isinstance(result, ShallowCandidateProbs)
    
    # 验证 probs 和 cumulative_probs 没有 NaN
    assert not torch.isnan(result.probs).any(), "probs should not contain NaN"
    assert not torch.isnan(result.cumulative_probs).any(), "cumulative_probs should not contain NaN"
    
    # 验证 cumulative_probs 在有效范围内
    assert (result.cumulative_probs >= 0).all(), "cumulative_probs should be non-negative"
    assert (result.cumulative_probs <= 1).all(), "cumulative_probs should be <= 1"
    
    print(f"✓ NaN protection works correctly")
    print(f"  - Probs range: [{result.probs.min():.4f}, {result.probs.max():.4f}]")
    print(f"  - Cumulative probs range: [{result.cumulative_probs.min():.4f}, {result.cumulative_probs.max():.4f}]")


if __name__ == '__main__':
    print("="*70)
    print("I10-19 连续松弛单元测试")
    print("="*70)
    print()
    
    test_learnable_splitter_continuous_param()
    test_learnable_splitter_discrete_mode()
    test_tokenizer_continuous_param()
    test_shallow_evaluator_candidate_count()
    test_forward_pass_continuous_mode()
    test_forward_pass_discrete_vs_continuous_structure()
    test_continuous_mode_depth_tracking()
    test_continuous_mode_nan_protection()
    
    print()
    print("="*70)
    print("所有测试通过！")
    print("="*70)
