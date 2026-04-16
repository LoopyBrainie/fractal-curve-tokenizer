# -*- coding: utf-8 -*-
"""
Hilbert Curve Demo - Hilbert 曲线演示脚本

使用方法:
    python -m examples.analysis.demos.hilbert_demo --order 4
    python -m examples.analysis.demos.hilbert_demo --order 4 --save-dir ./output
"""

import argparse
import sys
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from examples.analysis.visualization.hilbert_splitter import (
    plot_hilbert_curve,
    compare_orderings,
    visualize_mixed_depth_regions,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Hilbert Curve Demo")
    parser.add_argument("--order", type=int, default=4, help="Hilbert curve order")
    parser.add_argument("--save-dir", type=str, default=None, help="Directory to save figures")
    parser.add_argument("--no-show", action="store_true", help="Don't show figures")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.save_dir:
        import os
        os.makedirs(args.save_dir, exist_ok=True)

    def save_path(name):
        return str(Path(args.save_dir) / name) if args.save_dir else None

    print(f"\n{'=' * 70}")
    print(f" Hilbert Curve Demo (order={args.order})")
    print(f"{'=' * 70}")

    print("\n[1] Generating Hilbert curve visualization...")
    plot_hilbert_curve(order=args.order, save_path=save_path("hilbert_curve.png"))

    print("\n[2] Comparing orderings (Hilbert vs Raster)...")
    compare_orderings(order=args.order, save_path=save_path("order_comparison.png"))

    print("\n[3] Visualizing mixed depth regions...")
    visualize_mixed_depth_regions(save_path=save_path("mixed_depth_regions.png"))

    print("\n[4] Demo Complete!")
    print(f"{'=' * 70}")

    if not args.no_show:
        import matplotlib.pyplot as plt
        print("\nShowing figures...")
        plt.show()


if __name__ == "__main__":
    main()
