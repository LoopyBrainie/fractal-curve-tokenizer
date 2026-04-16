# -*- coding: utf-8 -*-
"""
Locality Analysis Demo - 空间局部性分析演示脚本

使用方法:
    python -m examples.analysis.demos.locality_demo --order 4
    python -m examples.analysis.demos.locality_demo --order 4 --save-dir ./output
"""

import argparse
import sys
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from examples.analysis.visualization.spatial_locality import (
    plot_hilbert_vs_raster,
    plot_locality_preservation,
    animate_hilbert_curve,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Locality Analysis Demo")
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
    print(f" Locality Analysis Demo (order={args.order})")
    print(f"{'=' * 70}")

    print("\n[1] Comparing Hilbert curve vs raster order...")
    plot_hilbert_vs_raster(order=args.order, save_path=save_path("hilbert_vs_raster.png"))

    print("\n[2] Measuring locality preservation...")
    plot_locality_preservation(order=args.order, save_path=save_path("locality_preservation.png"))

    print("\n[3] Animating Hilbert curve construction...")
    animate_hilbert_curve(order=args.order, save_path=save_path("hilbert_animation.gif"))

    print("\n[4] Demo Complete!")
    print(f"{'=' * 70}")

    if not args.no_show:
        import matplotlib.pyplot as plt
        print("\nShowing figures...")
        plt.show()


if __name__ == "__main__":
    main()
