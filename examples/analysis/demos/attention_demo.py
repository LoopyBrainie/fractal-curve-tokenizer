# -*- coding: utf-8 -*-
"""
Attention Patterns Demo - 注意力模式可视化演示脚本

使用方法:
    python -m examples.analysis.demos.attention_demo
    python -m examples.analysis.demos.attention_demo --save-dir ./output
"""

import argparse
import sys
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from examples.analysis.visualization.attention_patterns import (
    plot_attention_heatmap,
    plot_depth_attention_matrix,
    plot_head_comparison,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Attention Patterns Demo")
    parser.add_argument("--save-dir", type=str, default=None, help="Directory to save figures")
    parser.add_argument("--no-show", action="store_true", help="Don't show figures")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--seq-len", type=int, default=64, help="Sequence length")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.save_dir:
        import os
        os.makedirs(args.save_dir, exist_ok=True)

    save_path = lambda name: str(Path(args.save_dir) / name) if args.save_dir else None

    print(f"\n{'=' * 70}")
    print(" Attention Patterns Visualization Demo")
    print(f"{'=' * 70}")

    print(f"\n[1] Generating attention heatmap (heads={args.num_heads}, seq_len={args.seq_len})...")
    plot_attention_heatmap(
        num_heads=args.num_heads,
        seq_len=args.seq_len,
        save_path=save_path("attention_heatmap.png")
    )

    print("\n[2] Plotting depth attention matrix...")
    plot_depth_attention_matrix(
        num_heads=args.num_heads,
        max_depth=5,
        save_path=save_path("depth_attention_matrix.png")
    )

    print("\n[3] Comparing attention heads...")
    plot_head_comparison(
        num_heads=args.num_heads,
        seq_len=args.seq_len,
        save_path=save_path("head_comparison.png")
    )

    print(f"\n[4] Demo Complete!")
    print(f"{'=' * 70}")

    if not args.no_show:
        import matplotlib.pyplot as plt
        print("\nShowing figures...")
        plt.show()


if __name__ == "__main__":
    main()
