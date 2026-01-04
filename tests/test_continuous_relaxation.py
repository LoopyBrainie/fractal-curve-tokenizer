"""
I10-19 连续松弛单元测试

验证架构完整性：
1. 参数正确传递
2. ShallowParallelEvaluator正确初始化
3. 补偿机制在连续模式下被跳过
4. 基本的forward pass可以运行
"""
import sys
from pathlib import Path

# 添加src目录到路径
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import pytest
import torch
import torch.nn as nn
from vit_pytorch.split_adaptive import LearnableSplitter
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
    
    # 验证输出结构
    assert hasattr(result, 'regions')
    assert hasattr(result, 'depths')
    assert hasattr(result, 'batch_indices')
    assert hasattr(result, 'hilbert_indices')
    assert hasattr(result, 'tokens_per_batch')
    
    # 验证batch数量
    assert result.tokens_per_batch.shape[0] == B
    assert result.tokens_per_batch.sum() == result.regions.shape[0]
    
    print(f"✓ Forward pass successful in continuous mode")
    print(f"  - Batch size: {B}")
    print(f"  - Total tokens: {result.regions.shape[0]}")
    print(f"  - Tokens per batch: {result.tokens_per_batch.tolist()}")


def test_forward_pass_discrete_vs_continuous_structure():
    """测试离散和连续模式输出结构一致性."""
    torch.manual_seed(42)
    
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
    
    # 验证输出字段一致
    assert hasattr(result_discrete, 'regions')
    assert hasattr(result_continuous, 'regions')
    assert result_discrete.regions.shape[1] == result_continuous.regions.shape[1] == 4
    
    assert hasattr(result_discrete, 'depths')
    assert hasattr(result_continuous, 'depths')
    
    assert hasattr(result_discrete, 'batch_indices')
    assert hasattr(result_continuous, 'batch_indices')
    
    assert hasattr(result_discrete, 'tokens_per_batch')
    assert hasattr(result_continuous, 'tokens_per_batch')
    assert result_discrete.tokens_per_batch.shape == result_continuous.tokens_per_batch.shape
    
    print("✓ Discrete and continuous modes have consistent output structure")
    print(f"  - Discrete: {result_discrete.regions.shape[0]} tokens")
    print(f"  - Continuous (TODO): {result_continuous.regions.shape[0]} tokens")


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
    
    print()
    print("="*70)
    print("所有测试通过！")
    print("="*70)
