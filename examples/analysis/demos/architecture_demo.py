# -*- coding: utf-8 -*-
"""
Architecture Demo - 架构可视化演示脚本

使用方法:
    python -m examples.analysis.demos.architecture_demo
    python -m examples.analysis.demos.architecture_demo --save-dir ./output
"""

import argparse
import sys
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

# 尝试导入架构模块，如果不存在则跳过
try:
    from examples.analysis.visualization.architecture_diagram import (
        plot_architecture_comparison,
        plot_model_components,
    )
    from examples.analysis.visualization.architecture_design import (
        plot_token_count_range,
        plot_position_encoding_comparison,
        plot_multi_scale_representation,
    )
    HAS_ARCHITECTURE = True
except ImportError:
    HAS_ARCHITECTURE = False
    print("Warning: Architecture modules not found. Skipping architecture demos.")


def parse_args():
    parser = argparse.ArgumentParser(description="Architecture Demo")
    parser.add_argument("--save-dir", type=str, default=None, help="Directory to save figures")
    parser.add_argument("--no-show", action="store_true", help="Don't show figures")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.save_dir:
        import os
        os.makedirs(args.save_dir, exist_ok=True)

    save_path = lambda name: str(Path(args.save_dir) / name) if args.save_dir else None

    print(f"\n{'=' * 70}")
    print(" Architecture Visualization Demo")
    print(f"{'=' * 70}")

    if not HAS_ARCHITECTURE:
        print("\nArchitecture modules not available. Please ensure they are implemented.")
        return

    print("\n[1] Comparing ViT architectures...")
    plot_architecture_comparison(save_path=save_path("architecture_comparison.png"))

    print("\n[2] Plotting model components...")
    plot_model_components(save_path=save_path("model_components.png"))

    print("\n[3] Visualizing token count range...")
    plot_token_count_range(save_path=save_path("token_count_range.png"))

    print("\n[4] Comparing position encoding...")
    plot_position_encoding_comparison(save_path=save_path("position_encoding.png"))

    print("\n[5] Visualizing multi-scale representation...")
    plot_multi_scale_representation(save_path=save_path("multi_scale.png"))

    print(f"\n[6] Demo Complete!")
    print(f"{'=' * 70}")

    if not args.no_show:
        import matplotlib.pyplot as plt
        print("\nShowing figures...")
        plt.show()


if __name__ == "__main__":
    main()
