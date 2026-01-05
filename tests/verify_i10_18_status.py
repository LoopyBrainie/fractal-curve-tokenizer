#!/usr/bin/env python3
"""验证I10-18/I10-19连续松弛的实现状态"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from vit_pytorch.split_adaptive import LearnableSplitter, ShallowCandidateProbs
from vit_pytorch.split_adaptive_parallel import ShallowParallelEvaluator

print("=== I10-18/I10-19 Implementation Verification ===\n")

# 创建LearnableSplitter (连续松弛模式)
splitter = LearnableSplitter(
    feature_dim=256,
    max_depth=3,
    use_continuous_relaxation=True,
    min_region_size=8,
)

print("1. LearnableSplitter创建")
print(f"   use_continuous_relaxation: {splitter.use_continuous_relaxation}")
print(f"   shallow_evaluator类型: {type(splitter.shallow_evaluator).__name__}")

if splitter.shallow_evaluator is not None:
    info = splitter.shallow_evaluator.get_candidate_info()
    print(f"   候选区域数: {info['total_candidates']}")
    print(f"   深度分布: {info['depth_distribution']}")

# 测试前向传播
print("\n2. 前向传播测试")
x = torch.randn(2, 256, 8, 8)
result = splitter(x, image_size=(64, 64))

print(f"   返回类型: {type(result).__name__}")

if isinstance(result, ShallowCandidateProbs):
    print("   ✓ 返回ShallowCandidateProbs (连续松弛路径正确)")
    print(f"   候选数: {result.num_candidates}")
    print(f"   batch_size: {result.batch_size}")
    print(f"   probs形状: {result.probs.shape}")
    print(f"   cumulative_probs形状: {result.cumulative_probs.shape}")
    
    # 验证数学性质
    print("\n3. 数学性质验证")
    
    # 累积概率应该单调递减 (随深度增加)
    # 根节点累积概率应该为1
    root_cum_prob = result.cumulative_probs[0, 0].item()  # batch 0, idx 0 (root)
    print(f"   根节点累积概率: {root_cum_prob:.4f} (期望: 1.0)")
    
    # 概率应该在 [0, 1] 范围内
    probs_min = result.probs.min().item()
    probs_max = result.probs.max().item()
    print(f"   probs范围: [{probs_min:.4f}, {probs_max:.4f}] (期望: [0, 1])")
    
    cum_min = result.cumulative_probs.min().item()
    cum_max = result.cumulative_probs.max().item()
    print(f"   cumulative_probs范围: [{cum_min:.4f}, {cum_max:.4f}] (期望: [0, 1])")
    
    # 验证get_continuous_tokens是否可调用
    print("\n4. get_continuous_tokens验证 (不同阈值)")
    
    # 创建假的embeddings
    B = result.batch_size
    N = result.num_candidates
    D = 256
    embeddings = torch.randn(B, N, D)
    
    # 测试不同阈值
    thresholds = [0.01, 0.05, 0.1, 0.2, 0.5]
    
    for threshold in thresholds:
        try:
            tokens, weights = splitter.shallow_evaluator.get_continuous_tokens(
                features=x,
                embeddings=embeddings,
                probs=result.probs,
                cumulative_probs=result.cumulative_probs,
                embed_dim=D,
                threshold=threshold,
            )
            print(f"   threshold={threshold:.2f}: tokens形状={tokens.shape}, 有效token={tokens.shape[1]}")
        except Exception as e:
            print(f"   threshold={threshold:.2f}: 失败 - {e}")
    
    # 测试max_tokens限制
    print("\n5. max_tokens限制验证")
    for max_tokens in [10, 20, 50]:
        try:
            tokens, weights = splitter.shallow_evaluator.get_continuous_tokens(
                features=x,
                embeddings=embeddings,
                probs=result.probs,
                cumulative_probs=result.cumulative_probs,
                embed_dim=D,
                threshold=0.01,
                max_tokens=max_tokens,
            )
            print(f"   max_tokens={max_tokens}: tokens形状={tokens.shape}")
        except Exception as e:
            print(f"   max_tokens={max_tokens}: 失败 - {e}")
    
    # 测试深度跟踪
    print("\n6. 深度跟踪验证")
    tokens, weights = splitter.shallow_evaluator.get_continuous_tokens(
        features=x,
        embeddings=embeddings,
        probs=result.probs,
        cumulative_probs=result.cumulative_probs,
        embed_dim=D,
        threshold=0.1,
    )
    
    if hasattr(splitter.shallow_evaluator, '_last_token_depths'):
        depths = splitter.shallow_evaluator._last_token_depths
        print(f"   _last_token_depths形状: {depths.shape}")
        print(f"   深度范围: [{depths.min().item()}, {depths.max().item()}]")
        
        # 统计每个深度的token数量
        for d in range(4):
            count = (depths == d).sum().item()
            print(f"   深度{d}: {count}个token")
    else:
        print("   ✗ _last_token_depths未设置")
        
else:
    print("   ✗ 返回TensorSplitResult (应该是ShallowCandidateProbs)")

print("\n=== 验证完成 ===")

# 额外测试: FractalTokenizer端到端
print("\n=== 额外: FractalTokenizer端到端测试 ===")

try:
    from vit_pytorch.tokenizer_streaming import StreamingFractalTokenizerV3
    
    tokenizer = StreamingFractalTokenizerV3(
        image_size=64,
        base_patch_size=8,
        d_model=256,
        channels=3,
        max_depth=3,
        use_continuous_relaxation=True,
    )
    
    print("1. FractalTokenizer创建成功")
    print(f"   use_continuous_relaxation: {tokenizer.splitter.use_continuous_relaxation}")
    
    # 测试训练模式
    tokenizer.train()
    dummy_images = torch.randn(2, 3, 64, 64)
    
    output = tokenizer(dummy_images)
    
    print("\n2. 训练模式输出 (threshold=0.01)")
    print(f"   TokenizerOutput类型: {type(output).__name__}")
    print(f"   sequences数量: {len(output.sequences)}")
    
    # 获取第一个sequence的信息
    seq = output.sequences[0]
    print(f"   第一个sequence: tokens={seq.tokens.shape if hasattr(seq.tokens, 'shape') else len(seq.tokens)}")
    
    # 测试推理模式
    tokenizer.eval()
    with torch.no_grad():
        output_eval = tokenizer(dummy_images)
    
    print("\n3. 推理模式输出 (threshold=0.05)")
    seq_eval = output_eval.sequences[0]
    print(f"   第一个sequence: tokens={seq_eval.tokens.shape if hasattr(seq_eval.tokens, 'shape') else len(seq_eval.tokens)}")
    
    print("\n   ✓ FractalTokenizer端到端测试成功")
        
except Exception as e:
    import traceback
    print(f"✗ FractalTokenizer测试失败: {e}")
    traceback.print_exc()
