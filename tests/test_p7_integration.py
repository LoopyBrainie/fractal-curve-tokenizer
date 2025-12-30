"""P7-5/P7-6/P7-7 TokenizerV3 集成验证测试脚本."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

def main():
    print('='*60)
    print('P7-5/P7-6/P7-7 TokenizerV3 集成验证')
    print('='*60)

    # 1. 测试导入
    from src.vit_pytorch.tokenizer_streaming import StreamingFractalTokenizerV3
    print('[1] Import OK')

    # 2. 创建 Learnable Split 模式的 Tokenizer
    tokenizer = StreamingFractalTokenizerV3(
        image_size=64,
        base_patch_size=4,
        d_model=64,
        max_depth=3,
        split_scheme='learnable',
    )
    print(f'[2] Tokenizer created (learnable_split = {tokenizer._use_learnable_split})')

    # 3. 参数统计
    total_params = sum(p.numel() for p in tokenizer.parameters())
    trainable_params = sum(p.numel() for p in tokenizer.parameters() if p.requires_grad)
    print(f'[3] Parameters: total={total_params:,}, trainable={trainable_params:,}')

    # 4. 前向传播测试 (训练模式)
    tokenizer.train()  # 切换到训练模式以启用 soft sampling
    dummy = torch.randn(2, 3, 64, 64)
    output = tokenizer.tokenize(dummy)
    print(f'[4] Forward pass OK: output type = {type(output).__name__}')
    for i, seq in enumerate(output.sequences):
        print(f'    - Batch {i}: tokens shape = {seq.tokens.shape}')

    # 5. P7-6: 辅助损失测试
    aux_loss = tokenizer.get_learnable_split_loss(
        lambda_entropy=0.1,
        lambda_budget=0.01,
        target_tokens=64
    )
    print(f'[5] P7-6 Auxiliary loss: {aux_loss:.6f}, requires_grad={aux_loss.requires_grad}')
    
    # 反向传播验证
    tokenizer.zero_grad()
    aux_loss.backward()
    splitter_grads = sum(
        1 for n, p in tokenizer.named_parameters() 
        if 'splitter' in n and p.grad is not None
    )
    print(f'    Splitter parameters with gradients: {splitter_grads}')

    # 6. P7-7: 温度退火调度测试
    print(f'[6] P7-7 Temperature annealing:')
    
    # 模拟 warmup (epoch 1-5)
    # P11-11 修复: T_end 从 0.1 提高到 0.3 以避免梯度消失
    T_start, T_end, warmup = 1.0, 0.3, 5
    epochs = 20
    
    temps = []
    for epoch in range(1, epochs + 1):
        if epoch <= warmup:
            current_temp = T_start
        else:
            progress = (epoch - warmup) / max(1, epochs - warmup)
            ratio = T_end / T_start
            current_temp = T_start * (ratio ** progress)
        temps.append(current_temp)
        tokenizer.set_split_temperature(current_temp)
    
    print(f'    Epoch 1 (warmup): T = {temps[0]:.3f}')
    print(f'    Epoch 5 (warmup): T = {temps[4]:.3f}')
    print(f'    Epoch 10: T = {temps[9]:.3f}')
    print(f'    Epoch 20: T = {temps[19]:.3f}')
    
    # 验证最终温度
    stats = tokenizer.get_training_stats()
    assert abs(stats['learnable_temperature'] - temps[-1]) < 0.001, \
        f"Temperature mismatch: expected {temps[-1]}, got {stats['learnable_temperature']}"
    print(f'    Temperature verified: {stats["learnable_temperature"]:.3f}')

    # 7. Rule-based 对比
    tokenizer_rule = StreamingFractalTokenizerV3(
        image_size=64,
        base_patch_size=4,
        d_model=64,
        max_depth=3,
        split_scheme='adaptive',  # 默认 rule-based
    )
    output_rule = tokenizer_rule.tokenize(dummy)
    print(f'[7] Rule-based comparison: {[seq.tokens.shape[0] for seq in output_rule.sequences]} tokens')

    print()
    print('='*60)
    print('P7-5/P7-6/P7-7 TokenizerV3 集成验证通过!')
    print('='*60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
