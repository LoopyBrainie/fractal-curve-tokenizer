"""
基于模型架构与源码的可视化（不依赖训练结果）。

生成内容：
- Hilbert 曲线路径示意
- 自适应四叉树分割 + Hilbert 序列顺序
- LCA 偏置热力图（由四叉树路径精确计算）
- 深度分布与分辨率-Token 标度分析（理论曲线）
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from vit_pytorch.curve_hilbert import HilbertCurve


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "workspace", "visualizations")
os.makedirs(OUTPUT_DIR, exist_ok=True)


@dataclass
class QuadLeaf:
    path: List[int]  # quadtree 路径，元素 ∈ {0,1,2,3}
    x0: int
    y0: int
    size: int
    hilbert_idx: int


def _hilbert_points(order: int = 4) -> np.ndarray:
    n = 2 ** order
    pts = np.zeros((n * n, 2), dtype=int)
    for d in range(n * n):
        x, y = HilbertCurve.d_to_xy(n, d)
        pts[d] = (x, y)
    return pts


def _synthetic_field(n: int) -> np.ndarray:
    """生成可复现的合成场，用于自适应拆分判据。"""
    xs = np.linspace(-1, 1, n)
    ys = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(xs, ys)
    field = (
        0.6 * np.sin(3 * np.pi * X) * np.cos(2 * np.pi * Y)
        + 0.3 * (X ** 2 + Y ** 2)
        + 0.1 * np.sin(5 * np.pi * (X + Y))
    )
    return field.astype(np.float32)


def _should_split(region: np.ndarray, depth: int, max_depth: int) -> bool:
    if depth >= max_depth:
        return False
    complexity = float(region.std())
    threshold = 0.12 * (1 + 0.18 * depth)
    return complexity > threshold


def _build_quadtree(field: np.ndarray, x0: int, y0: int, size: int, depth: int, max_depth: int, path: List[int]) -> List[QuadLeaf]:
    region = field[y0 : y0 + size, x0 : x0 + size]
    if not _should_split(region, depth, max_depth):
        n = field.shape[0]
        cx = x0 + size // 2
        cy = y0 + size // 2
        hilbert_idx = HilbertCurve.xy_to_d(n, cx, cy)
        return [QuadLeaf(path=list(path), x0=x0, y0=y0, size=size, hilbert_idx=hilbert_idx)]

    leaves: List[QuadLeaf] = []
    half = size // 2
    offsets = [(-half, -half), (half, -half), (-half, half), (half, half)]
    # 象限顺序：0=左上,1=右上,2=左下,3=右下
    coords = [
        (x0, y0),
        (x0 + half, y0),
        (x0, y0 + half),
        (x0 + half, y0 + half),
    ]
    for q, (nx, ny) in enumerate(coords):
        leaves.extend(
            _build_quadtree(field, nx, ny, half, depth + 1, max_depth, path + [q])
        )
    return leaves


def generate_quadtree(max_depth: int = 4) -> List[QuadLeaf]:
    n = 2 ** max_depth
    field = _synthetic_field(n)
    leaves = _build_quadtree(field, 0, 0, n, 0, max_depth, [])
    # 按 Hilbert 顺序排序，模拟 tokenizer 的序列化
    leaves.sort(key=lambda l: l.hilbert_idx)
    return leaves


def lca_depth(path_a: List[int], path_b: List[int]) -> int:
    depth = 0
    for a, b in zip(path_a, path_b):
        if a == b:
            depth += 1
        else:
            break
    return depth


def lca_matrix(leaves: List[QuadLeaf]) -> np.ndarray:
    n = len(leaves)
    mat = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i, n):
            d = lca_depth(leaves[i].path, leaves[j].path)
            mat[i, j] = mat[j, i] = d
    return mat


def plot_hilbert_curve(order: int = 4, save_name: str = "hilbert_curve.png"):
    pts = _hilbert_points(order)
    plt.figure(figsize=(6, 6))
    plt.plot(pts[:, 0], pts[:, 1], lw=1.5, color="#1f77b4")
    plt.scatter(pts[:, 0], pts[:, 1], s=6, color="#ff7f0e", alpha=0.8)
    plt.title(f"Hilbert Curve (order={order}, {len(pts)} points)")
    plt.axis("equal")
    plt.axis("off")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, save_name)
    plt.savefig(path, dpi=300)
    plt.close()


def plot_quadtree(leaves: List[QuadLeaf], max_depth: int, save_name: str = "quadtree.png"):
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    colors = sns.color_palette("viridis", max_depth + 1)
    for idx, leaf in enumerate(leaves):
        depth = len(leaf.path)
        rect = plt.Rectangle((leaf.x0, leaf.y0), leaf.size, leaf.size, fill=False, lw=1.2, color=colors[depth])
        ax.add_patch(rect)
        ax.text(
            leaf.x0 + leaf.size * 0.05,
            leaf.y0 + leaf.size * 0.05,
            f"{idx}",
            fontsize=7,
            color=colors[depth],
        )
    ax.set_xlim(0, 2 ** max_depth)
    ax.set_ylim(0, 2 ** max_depth)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    plt.title("Adaptive Quadtree (Hilbert order annotated)")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, save_name)
    plt.savefig(path, dpi=300)
    plt.close()


def plot_lca_heatmap(leaves: List[QuadLeaf], max_depth: int, save_name: str = "lca_bias.png"):
    mat = lca_matrix(leaves)
    mat_norm = mat / max_depth
    plt.figure(figsize=(7, 6))
    sns.heatmap(mat_norm, cmap="magma", square=True, cbar_kws={"label": "LCA depth / max_depth"})
    plt.title("LCA-derived Attention Bias (architecture only)")
    plt.xlabel("Key (Hilbert order)")
    plt.ylabel("Query (Hilbert order)")
    path = os.path.join(OUTPUT_DIR, save_name)
    plt.savefig(path, dpi=300)
    plt.close()


def plot_depth_histogram(leaves: List[QuadLeaf], save_name: str = "depth_hist.png"):
    depths = [len(l.path) for l in leaves]
    plt.figure(figsize=(6, 4))
    sns.histplot(depths, discrete=True, shrink=0.8, color="#2ca02c")
    plt.title("Token Depth Distribution (synthetic quadtree)")
    plt.xlabel("Depth")
    plt.ylabel("Count")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, save_name)
    plt.savefig(path, dpi=300)
    plt.close()


def plot_scaling_curve(save_name: str = "scaling.png"):
    resolutions = np.array([32, 64, 128, 256, 512, 1024])
    patch_size = 16
    n_standard = (resolutions / patch_size) ** 2
    # 理论自适应：D=1.5 代表介于线与面之间的有效维度
    n_fractal = 0.6 * (resolutions ** 1.5)
    plt.figure(figsize=(7, 5))
    plt.plot(resolutions, n_standard, "o--", label="Standard ViT O(H^2)", lw=2)
    plt.plot(resolutions, n_fractal, "s-", label="Fractal ViT ~O(H^1.5)", lw=2)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Resolution H")
    plt.ylabel("Token count N")
    plt.title("Token Scaling (architecture-level analysis)")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, save_name)
    plt.savefig(path, dpi=300)
    plt.close()


def run_all(max_depth: int = 4):
    sns.set_theme(style="white")
    print(f"Saving visualizations to: {OUTPUT_DIR}")
    plot_hilbert_curve(order=max_depth)
    leaves = generate_quadtree(max_depth=max_depth)
    plot_quadtree(leaves, max_depth=max_depth)
    plot_lca_heatmap(leaves, max_depth=max_depth)
    plot_depth_histogram(leaves)
    plot_scaling_curve()


if __name__ == "__main__":
    import sys

    max_depth_arg = 4
    if len(sys.argv) >= 2:
        try:
            max_depth_arg = int(sys.argv[1])
        except ValueError:
            pass
    run_all(max_depth=max_depth_arg)
