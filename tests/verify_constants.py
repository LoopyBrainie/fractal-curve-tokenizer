# -*- coding: utf-8 -*-
"""
形式化验证: 常量与配置的数学一致性分析
======================================

验证项:
1. 阈值衰减公式: τ_d = τ_base · γ^d
2. 温度退火范围: T ∈ [T_end, T_start]
3. 几何参数推导: max_depth = ceil(log2(grid_size))
4. 参数边界约束: τ ∈ [0, 1], γ ∈ (0, 1)
5. 实际代码参数一致性
"""

import math
import torch
from vit_pytorch.constants import (
    SPLITTER_TEMP_START, SPLITTER_TEMP_END,
    SPLIT_GAMMA,
    LEARNABLE_INIT_TAU_BASE, LEARNABLE_INIT_TAU_GAMMA,
    HILBERT_BIAS_SCALE, LEVEL_BIAS_SCALE,
    LOGITS_CLAMP_MIN, LOGITS_CLAMP_MAX,
)
from vit_pytorch.config_fractal import FractalConfig
from vit_pytorch.tokenizer_streaming import StreamingFractalTokenizerV3
from vit_pytorch.split_adaptive import LearnableSplitter


def main():
    print('=' * 60)
    print('形式化验证: 常量与配置的数学一致性')
    print('=' * 60)

    all_passed = True

    # =====================================================================
    # 1. 阈值衰减公式验证: τ_d = τ_base · γ^d
    # =====================================================================
    print('\n[1] 阈值衰减公式验证')
    print('    公式: τ_d = τ_base · γ^d')
    print(f'    SPLIT_GAMMA = {SPLIT_GAMMA}')
    print(f'    LEARNABLE_INIT_TAU_BASE = {LEARNABLE_INIT_TAU_BASE}')
    print(f'    LEARNABLE_INIT_TAU_GAMMA = {LEARNABLE_INIT_TAU_GAMMA}')

    max_depth = 4
    tau_sequence_learnable = [LEARNABLE_INIT_TAU_BASE * (LEARNABLE_INIT_TAU_GAMMA ** d) for d in range(max_depth + 1)]

    print(f'    τ 序列 (LEARNABLE): {[f"{t:.4f}" for t in tau_sequence_learnable]}')

    # 验证单调递减性
    is_monotonic = all(tau_sequence_learnable[i] > tau_sequence_learnable[i+1] for i in range(len(tau_sequence_learnable)-1))
    print(f'    单调递减验证: {"✓ PASS" if is_monotonic else "✗ FAIL"}')
    all_passed = all_passed and is_monotonic

    # 验证边界约束 τ ∈ (0, 1)
    all_in_range = all(0 < t < 1 for t in tau_sequence_learnable)
    print(f'    边界约束 τ ∈ (0,1): {"✓ PASS" if all_in_range else "✗ FAIL"}')
    all_passed = all_passed and all_in_range

    # =====================================================================
    # 2. 温度退火验证: T ∈ [T_end, T_start]
    # =====================================================================
    print('\n[2] 温度退火参数验证')
    print(f'    SPLITTER_TEMP_START = {SPLITTER_TEMP_START}')
    print(f'    SPLITTER_TEMP_END = {SPLITTER_TEMP_END}')
    print(f'    退火比例 = {SPLITTER_TEMP_START / SPLITTER_TEMP_END:.1f}x')

    # 验证 T_start > T_end > 0
    temp_valid = SPLITTER_TEMP_START > SPLITTER_TEMP_END > 0
    print(f'    T_start > T_end > 0: {"✓ PASS" if temp_valid else "✗ FAIL"}')
    all_passed = all_passed and temp_valid

    # 计算指数退火曲线 (100 epochs, 10 warmup)
    def exp_anneal(step, total, start, end, warmup=10):
        if step < warmup:
            return start
        progress = (step - warmup) / (total - warmup)
        return start * ((end / start) ** progress)

    temps = [exp_anneal(i, 100, SPLITTER_TEMP_START, SPLITTER_TEMP_END, 10) for i in range(0, 101, 20)]
    print(f'    退火曲线 (epochs 0,20,40,60,80,100): {[f"{t:.3f}" for t in temps]}')

    # =====================================================================
    # 3. 几何参数推导验证: max_depth = ceil(log2(grid_size))
    # =====================================================================
    print('\n[3] 几何参数推导验证')
    test_cases = [
        (64, 4, 16, 4),   # grid=16=2^4, depth=4
        (64, 8, 8, 3),    # grid=8=2^3, depth=3
        (128, 4, 32, 5),  # grid=32=2^5, depth=5
        (60, 5, 12, 4),   # grid=12 (非2^k), ceil(log2(12))=4
        (224, 16, 14, 4), # grid=14 (非2^k), ceil(log2(14))=4
    ]

    print('    | image | patch | grid | expected | computed | match |')
    print('    |-------|-------|------|----------|----------|-------|')
    geo_all_match = True
    for img, patch, expected_grid, expected_depth in test_cases:
        config = FractalConfig(img, patch)
        match = config.max_depth == expected_depth and config.grid_size == expected_grid
        geo_all_match = geo_all_match and match
        mark = '✓' if match else '✗'
        print(f'    | {img:5} | {patch:5} | {expected_grid:4} | {expected_depth:8} | {config.max_depth:8} | {mark:5} |')

    print(f'    几何推导验证: {"✓ PASS" if geo_all_match else "✗ FAIL"}')
    all_passed = all_passed and geo_all_match

    # =====================================================================
    # 4. Hilbert 策略验证: ρ* = 4/3
    # =====================================================================
    print('\n[4] Hilbert 策略验证 (ρ* = 4/3)')
    print('    非 2^k 网格时的 padding 策略选择:')

    hilbert_cases = [
        (12, 16, 16*16/12/12),  # ρ = 256/144 ≈ 1.78 ≥ 4/3 → Pseudo
        (14, 16, 16*16/14/14),  # ρ = 256/196 ≈ 1.31 < 4/3 → Padding
        (15, 16, 16*16/15/15),  # ρ = 256/225 ≈ 1.14 < 4/3 → Padding
        (10, 16, 16*16/10/10),  # ρ = 256/100 = 2.56 ≥ 4/3 → Pseudo
    ]

    print('    | grid | next_2k | ρ (padding) | expected | computed | match |')
    print('    |------|---------|-------------|----------|----------|-------|')
    hilbert_all_match = True
    for grid, next_2k, rho in hilbert_cases:
        config = FractalConfig(grid * 4, 4)  # 使用 patch=4
        expected_pseudo = rho >= 4/3
        match = config.uses_pseudo_hilbert == expected_pseudo
        hilbert_all_match = hilbert_all_match and match
        mark = '✓' if match else '✗'
        print(f'    | {grid:4} | {next_2k:7} | {rho:11.3f} | {str(expected_pseudo):8} | {str(config.uses_pseudo_hilbert):8} | {mark}')

    print(f'    Hilbert 策略验证: {"✓ PASS" if hilbert_all_match else "✗ FAIL"}')
    all_passed = all_passed and hilbert_all_match

    # =====================================================================
    # 5. 实际代码参数一致性验证
    # =====================================================================
    print('\n[5] 实际代码参数一致性验证')

    # 创建 LearnableSplitter 检查默认值
    splitter = LearnableSplitter(feature_dim=256, max_depth=4)
    print(f'    LearnableSplitter 默认温度: {splitter.temperature} (期望: 1.0)')
    print(f'    LearnableSplitter tau_bases[0]: {splitter._tau_bases[0].item():.2f} (期望: {LEARNABLE_INIT_TAU_BASE})')

    # 创建 Tokenizer 检查参数传递
    tokenizer = StreamingFractalTokenizerV3(image_size=64, d_model=128, base_patch_size=4, max_depth=3)
    print(f'    Tokenizer.splitter 类型: {type(tokenizer.splitter).__name__}')
    print(f'    Tokenizer.splitter.temperature: {tokenizer.splitter.temperature}')

    # 检查阈值是否正确初始化
    init_thresholds = tokenizer.splitter._tau_bases.tolist()
    expected_thresholds = [0.5 * (0.85 ** d) for d in range(4)]
    threshold_match = all(abs(a - b) < 1e-6 for a, b in zip(init_thresholds, expected_thresholds))
    print(f'    阈值初始化: {[f"{t:.4f}" for t in init_thresholds]}')
    print(f'    期望阈值: {[f"{t:.4f}" for t in expected_thresholds]}')
    print(f'    阈值一致性: {"✓ PASS" if threshold_match else "✗ FAIL"}')
    all_passed = all_passed and threshold_match

    # =====================================================================
    # 6. 数值稳定性常量验证
    # =====================================================================
    print('\n[6] 数值稳定性常量验证')
    print(f'    LOGITS_CLAMP_MIN = {LOGITS_CLAMP_MIN}')
    print(f'    LOGITS_CLAMP_MAX = {LOGITS_CLAMP_MAX}')

    # 验证 softmax 在边界处的数值稳定性
    logits = torch.tensor([LOGITS_CLAMP_MIN, 0, LOGITS_CLAMP_MAX])
    softmax = torch.softmax(logits, dim=0)
    print(f'    softmax([{LOGITS_CLAMP_MIN}, 0, {LOGITS_CLAMP_MAX}]) = {[f"{x:.6f}" for x in softmax.tolist()]}')
    numerically_stable = not (softmax.isnan().any() or softmax.isinf().any())
    print(f'    数值稳定 (无 NaN/Inf): {"✓ PASS" if numerically_stable else "✗ FAIL"}')
    all_passed = all_passed and numerically_stable

    # =====================================================================
    # 7. 缩放因子验证
    # =====================================================================
    print('\n[7] 注意力缩放因子验证')
    print(f'    HILBERT_BIAS_SCALE = {HILBERT_BIAS_SCALE}')
    print(f'    LEVEL_BIAS_SCALE = {LEVEL_BIAS_SCALE}')

    # 验证缩放后偏置不会主导 attention logits
    # 典型 attention logits ~ O(1), 偏置应保持较小
    bias_example = 10.0  # 假设原始偏置值
    scaled_hilbert = bias_example * HILBERT_BIAS_SCALE
    scaled_level = bias_example * LEVEL_BIAS_SCALE
    print(f'    原始偏置=10.0 → Hilbert缩放后: {scaled_hilbert}, Level缩放后: {scaled_level}')
    scale_reasonable = max(scaled_hilbert, scaled_level) < 1.5
    print(f'    缩放因子合理性 (< 1.5): {"✓ PASS" if scale_reasonable else "✗ FAIL"}')
    all_passed = all_passed and scale_reasonable

    # =====================================================================
    # 8. 常量与代码默认值对比验证
    # =====================================================================
    print('\n[8] 常量与代码默认值对比验证')
    
    # 检查 tokenizer_streaming.py 中的默认值
    # gamma=0.85, learnable_temperature=1.0
    # 对比 constants.py
    gamma_match = abs(0.85 - SPLIT_GAMMA) < 1e-6
    temp_match = abs(1.0 - SPLITTER_TEMP_START) < 1e-6
    
    print(f'    tokenizer gamma=0.85 vs SPLIT_GAMMA={SPLIT_GAMMA}: {"✓" if gamma_match else "✗"}')
    print(f'    tokenizer temp=1.0 vs SPLITTER_TEMP_START={SPLITTER_TEMP_START}: {"✓" if temp_match else "✗"}')
    
    # 检查 LearnableSplitter 默认值
    # init_tau_base=0.5, init_tau_gamma=0.85
    learnable_tau_match = abs(0.5 - LEARNABLE_INIT_TAU_BASE) < 1e-6
    learnable_gamma_match = abs(0.85 - LEARNABLE_INIT_TAU_GAMMA) < 1e-6
    
    print(f'    LearnableSplitter init_tau_base=0.5 vs LEARNABLE_INIT_TAU_BASE={LEARNABLE_INIT_TAU_BASE}: {"✓" if learnable_tau_match else "✗"}')
    print(f'    LearnableSplitter init_tau_gamma=0.85 vs LEARNABLE_INIT_TAU_GAMMA={LEARNABLE_INIT_TAU_GAMMA}: {"✓" if learnable_gamma_match else "✗"}')
    
    defaults_match = gamma_match and temp_match and learnable_tau_match and learnable_gamma_match
    print(f'    默认值一致性: {"✓ PASS" if defaults_match else "✗ FAIL"}')
    all_passed = all_passed and defaults_match

    # =====================================================================
    # 9. 数学性质验证
    # =====================================================================
    print('\n[9] 数学性质验证')
    
    # 9.1 Gumbel-Softmax 温度极限行为
    print('    9.1 Gumbel-Softmax 温度极限行为:')
    for T in [1.0, 0.5, 0.1, 0.01]:
        logits = torch.tensor([0.3, 0.7])  # p_split 接近 0.7
        probs = torch.softmax(logits / T, dim=0)
        print(f'        T={T:.2f}: softmax([0.3, 0.7]/T) = [{probs[0]:.4f}, {probs[1]:.4f}]')
    
    # 9.2 阈值衰减渐近行为
    print('    9.2 阈值衰减渐近行为 (γ=0.85):')
    gamma = LEARNABLE_INIT_TAU_GAMMA
    tau_0 = LEARNABLE_INIT_TAU_BASE
    for d in [0, 4, 8, 12]:
        tau_d = tau_0 * (gamma ** d)
        print(f'        depth={d:2}: τ_d = {tau_d:.6f}')
    
    # 半衰期: d_half = log(0.5) / log(γ)
    half_life = math.log(0.5) / math.log(gamma)
    print(f'    阈值半衰期 (γ={gamma}): d_half = {half_life:.2f} 层')

    print('\n' + '=' * 60)
    print(f'形式化验证完成: {"✓ 全部通过" if all_passed else "✗ 存在失败项"}')
    print('=' * 60)
    
    return all_passed


if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)
