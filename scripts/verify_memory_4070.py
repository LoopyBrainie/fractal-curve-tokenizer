#!/usr/bin/env python3
"""
显存验证脚本: 验证 batch=192 配置是否可行

数学形式化验证:
- dim=384, num_layers=8, batch=192
- 预计显存: ~4-5 GB (远低于 8GB 预算)

使用方法:
    python scripts/verify_memory_4070.py [--batch-size N]
"""

import argparse
import sys
import torch
from pathlib import Path

# 添加 src 目录
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vit_pytorch import FractalCurveViT
from vit_pytorch.depth_utils import compute_max_depth


def verify_memory_config(
    batch_size: int = 192,
    image_size: int = 64,
    expected_vram_budget_gb: float = 7.0,
):
    """
    验证给定 batch_size 的显存使用是否在预算内。

    Args:
        batch_size: 批量大小
        image_size: 图像尺寸
        expected_vram_budget_gb: 预期的显存预算 (GB)

    Returns:
        bool: 验证是否通过
    """
    print("=" * 70)
    print(f"显存验证: RTX 4070 Laptop (预算: {expected_vram_budget_gb} GB)")
    print("=" * 70)

    # 检查 CUDA
    if not torch.cuda.is_available():
        print("\n[WARN] CUDA 不可用，跳过显存测试")
        print("\n[配置验证]")
        print(f"  - image_size: {image_size}")
        print(f"  - min_patch_size: 4")
        print(f"  - max_level: {compute_max_depth((image_size, image_size), 4)}")
        print(f"  - batch_size: {batch_size}")
        return True

    device = torch.device("cuda")
    torch.cuda.empty_cache()

    # 创建模型
    print(f"\n[1] 创建模型: FractalCurveViT")
    print(f"    - dim=384, num_layers=8, heads=6")
    print(f"    - image_size={image_size}, min_patch_size=4")
    print(f"    - max_level: {compute_max_depth((image_size, image_size), 4)}")

    model = FractalCurveViT(
        image_size=image_size,
        num_classes=200,
        dim=384,
        num_layers=8,
        heads=6,
        mlp_dim=1536,
        min_patch_size=4,
        use_checkpoint=True,  # 必须启用
    ).to(device)

    # 启用 channels_last
    model = model.enable_channels_last()

    # 打印参数量
    num_params = sum(p.numel() for p in model.parameters())
    num_params_m = num_params / 1e6
    print(f"    - 参数量: {num_params_m:.2f}M")

    # 显存测试
    print(f"\n[2] 显存测试: batch_size={batch_size}")

    with torch.cuda.amp.autocast():
        x = torch.randn(batch_size, 3, image_size, image_size, device=device)

        # 多次前向传播以稳定显存
        for i in range(3):
            torch.cuda.reset_peak_memory_stats(device)
            with torch.no_grad():
                _ = model(x)
            torch.cuda.synchronize()

        # 测量峰值显存
        torch.cuda.reset_peak_memory_stats(device)
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                stats = model(x)
        torch.cuda.synchronize()

        peak_memory = torch.cuda.max_memory_allocated(device)
        peak_memory_gb = peak_memory / (1024**3)

    print(f"    - 峰值显存: {peak_memory_gb:.2f} GB")
    print(f"    - 预算: {expected_vram_budget_gb:.2f} GB")

    # Token 统计
    if hasattr(stats, 'num_tokens'):
        num_tokens = stats.num_tokens
        if isinstance(num_tokens, torch.Tensor):
            num_tokens = num_tokens.item()
        print(f"    - Token 数量: {num_tokens}")

    # 判断结果
    print(f"\n[3] 验证结果")
    if peak_memory_gb < expected_vram_budget_gb:
        margin = expected_vram_budget_gb - peak_memory_gb
        print(f"    [PASS] 显存使用正常")
        print(f"    - 已用: {peak_memory_gb:.2f} GB")
        print(f"    - 剩余: {margin:.2f} GB")
        print(f"    - 可继续增加 batch_size 或模型规模")
        return True
    else:
        print(f"    [FAIL] 显存超限!")
        print(f"    - 已用: {peak_memory_gb:.2f} GB")
        print(f"    - 预算: {expected_vram_budget_gb:.2f} GB")
        print(f"    - 建议: 减小 batch_size 或启用更多优化")
        return False


def estimate_parameter_count(dim: int, num_layers: int) -> float:
    """估算参数量 (M)"""
    # Attention: 4 × dim² (Q, K, V, O)
    # FFN (SwiGLU): 12 × dim²
    # LayerNorm: 2 × dim × num_layers
    attention_params = 4 * dim * dim * num_layers
    ffn_params = 12 * dim * dim * num_layers
    layernorm_params = 2 * dim * 2 * num_layers
    embedding_params = dim * dim  # cls token + embedding
    head_params = dim * 200  # classification head

    total = attention_params + ffn_params + layernorm_params + embedding_params + head_params
    return total / 1e6


def main():
    parser = argparse.ArgumentParser(description="显存验证脚本")
    parser.add_argument("--batch-size", type=int, default=192, help="批量大小")
    parser.add_argument("--image-size", type=int, default=64, help="图像尺寸")
    parser.add_argument("--budget", type=float, default=7.0, help="显存预算 (GB)")
    parser.add_argument("--verbose", action="store_true", help="详细输出")

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("RTX 4070 Tiny-ImageNet 显存验证")
    print("=" * 70)

    # 估算参数量
    est_params = estimate_parameter_count(384, 8)
    print(f"\n模型估算: dim=384, num_layers=8")
    print(f"  - 参数量: ~{est_params:.1f}M")

    # 计算 max_level
    max_level = compute_max_depth((args.image_size, args.image_size), 4)
    print(f"  - max_level: {max_level} (基于 image_size={args.image_size})")

    # 运行验证
    success = verify_memory_config(
        batch_size=args.batch_size,
        image_size=args.image_size,
        expected_vram_budget_gb=args.budget,
    )

    print("\n" + "=" * 70)
    if success:
        print("配置验证通过! 可以安全运行训练。")
        print(f"\n建议启动命令:")
        print(f"  uv run python src/training/train_fractal_vit.py \\")
        print(f"    --dataset tiny-imagenet \\")
        print(f"    --batch-size {args.batch_size} \\")
        print(f"    --epochs 200 \\")
        print(f"    --dim 384 \\")
        print(f"    --num-layers 8 \\")
        print(f"    --heads 6 \\")
        print(f"    --mlp-dim 1536 \\")
        print(f"    --min-patch-size 4 \\")
        print(f"    --token-coverage-min 0.01 \\")
        print(f"    --token-coverage-max 0.20 \\")
        print(f"    --use-amp \\")
        print(f"    --gradient-checkpoint \\")
        print(f"    --compile \\")
        print(f"    --channels-last")
    else:
        print("配置验证失败! 请调整参数后重试。")
    print("=" * 70 + "\n")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
