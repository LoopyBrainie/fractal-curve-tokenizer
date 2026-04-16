# -*- coding: utf-8 -*-
"""
Depth Encoding Demo - 深度编码可视化演示脚本

使用方法:
    python -m examples.analysis.demos.depth_demo
    python -m examples.analysis.demos.depth_demo --save-dir ./output
"""

import argparse
import sys
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from examples.analysis.visualization.depth_encoding import (
    plot_depth_embedding_similarity,
    plot_depth_hierarchy,
    plot_pca_projection,
    plot_depth_clustermap,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Depth Encoding Demo")
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
    print(" Depth Encoding Visualization Demo")
    print(f"{'=' * 70}")

    print("\n[1] Plotting depth embedding similarity...")
    plot_depth_embedding_similarity(max_depth=5, save_path=save_path("depth_similarity.png"))

    print("\n[2] Visualizing depth hierarchy...")
    plot_depth_hierarchy(max_depth=5, save_path=save_path("depth_hierarchy.png"))

    print("\n[3] PCA projection of depth embeddings...")
    plot_pca_projection(max_depth=5, save_path=save_path("depth_pca.png"))

    print("\n[4] Cluster map of depth embeddings...")
    plot_depth_clustermap(max_depth=5, save_path=save_path("depth_clustermap.png"))

    print("\n[5] Demo Complete!")
    print(f"{'=' * 70}")

    if not args.no_show:
        import matplotlib.pyplot as plt
        print("\nShowing figures...")
        plt.show()


if __name__ == "__main__":
    main()
