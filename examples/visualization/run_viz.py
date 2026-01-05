"""统一入口：架构级可视化，无需训练结果。"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def run_seaborn(max_depth: int = 4):
    script_path = os.path.join(os.path.dirname(__file__), "seaborn_viz.py")
    env = os.environ.copy()
    env.setdefault("PYTHONWARNINGS", "ignore")
    cmd = [sys.executable, script_path, str(max_depth)]
    print(f"Running seaborn visualizations (max_depth={max_depth})…")
    subprocess.run(cmd, check=True, env=env)
    print("Static figures saved to workspace/visualizations")


def print_manim_help():
    print("Manim scenes (architecture demos, no training data):")
    print("  HilbertCurveScene")
    print("  AdaptiveQuadTreeScene")
    print("  HilbertTraversalScene")
    print("Example:")
    print("  manim -pql --media_dir workspace/visualizations/manim_media examples/visualization/manim_viz.py AdaptiveQuadTreeScene")


def main():
    parser = argparse.ArgumentParser(description="Run architecture-level visualizations")
    parser.add_argument("--max-depth", type=int, default=4, help="Quadtree max depth for static plots")
    parser.add_argument("--static", action="store_true", help="Run seaborn/matplotlib static plots")
    parser.add_argument("--manim-help", action="store_true", help="Print manim scene commands")
    args = parser.parse_args()

    ran = False
    if args.static or (not args.manim_help and not args.static):
        run_seaborn(max_depth=args.max_depth)
        ran = True
    if args.manim_help or not ran:
        print_manim_help()


if __name__ == "__main__":
    main()
