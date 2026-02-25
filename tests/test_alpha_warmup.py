"""
验证 Entmax Alpha 预热策略
"""

import sys
sys.path.insert(0, 'src')

from vit_pytorch.layers.splitters.hilbert_optimal_splitter import HilbertOptimalSplitter


def test_alpha_warmup():
    print("=" * 70)
    print("Entmax Alpha 预热策略验证")
    print("=" * 70)

    # 创建 Splitter
    splitter = HilbertOptimalSplitter(
        feature_dim=256,
        hidden_dim=64,
        min_patch_size=4,
        max_level_limit=6,
        K_min=8,
        K_max=64,
    )

    print("\n预热参数:")
    print(f"  entmax_alpha_init: {splitter.entmax_alpha_init}")
    print(f"  entmax_alpha_warmup: {splitter.entmax_alpha_warmup}")
    print(f"  entmax_alpha_max: {splitter.entmax_alpha_max}")
    print(f"  entmax_warmup_epochs: {splitter.entmax_warmup_epochs}")
    print(f"  entmax_schedule_epochs: {splitter.entmax_schedule_epochs}")

    print("\n" + "-" * 50)
    print("Epoch -> Alpha 值变化:")
    print("-" * 50)

    for epoch in range(25):
        splitter.set_epoch(epoch)
        print(f"  Epoch {epoch:2d}: α = {splitter.entmax_alpha:.4f}")

    print("\n" + "=" * 70)
    print("预期行为:")
    print("=" * 70)
    print("""
  Epoch 0-9:   α 从 1.2 线性增加到 1.5 (预热阶段)
  Epoch 10-19: α 从 1.5 线性增加到 2.0 (过渡阶段)
  Epoch 20+:   α = 2.0 (稳定阶段)

数学公式:
  α(t) = min(1.5, 1.2 + 0.3 × t / 10)        (t < 10)
  α(t) = min(2.0, 1.5 + 0.5 × (t-10) / 10)   (t >= 10)
""")
    print("=" * 70)


if __name__ == "__main__":
    test_alpha_warmup()
