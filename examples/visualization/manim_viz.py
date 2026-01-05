"""Manim 场景：基于架构原理的可视化（不依赖训练结果）。"""

from __future__ import annotations

from manim import *

import numpy as np

from vit_pytorch.curve_hilbert import HilbertCurve


def synthetic_field(n: int) -> np.ndarray:
    xs = np.linspace(-1, 1, n)
    ys = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(xs, ys)
    field = (
        0.6 * np.sin(3 * np.pi * X) * np.cos(2 * np.pi * Y)
        + 0.3 * (X ** 2 + Y ** 2)
        + 0.1 * np.sin(5 * np.pi * (X + Y))
    )
    return field.astype(np.float32)


def should_split(region: np.ndarray, depth: int, max_depth: int) -> bool:
    if depth >= max_depth:
        return False
    threshold = 0.12 * (1 + 0.18 * depth)
    return float(region.std()) > threshold


def build_quadtree(field: np.ndarray, x0: int, y0: int, size: int, depth: int, max_depth: int, path: list[int]):
    region = field[y0 : y0 + size, x0 : x0 + size]
    if not should_split(region, depth, max_depth):
        n = field.shape[0]
        cx = x0 + size // 2
        cy = y0 + size // 2
        hilbert_idx = HilbertCurve.xy_to_d(n, cx, cy)
        return [
            {
                "path": list(path),
                "x0": x0,
                "y0": y0,
                "size": size,
                "hilbert": hilbert_idx,
            }
        ]
    leaves = []
    half = size // 2
    coords = [
        (x0, y0),
        (x0 + half, y0),
        (x0, y0 + half),
        (x0 + half, y0 + half),
    ]
    for q, (nx, ny) in enumerate(coords):
        leaves.extend(build_quadtree(field, nx, ny, half, depth + 1, max_depth, path + [q]))
    return leaves


class HilbertCurveScene(Scene):
    def construct(self):
        title = Text("Hilbert Space-Filling Curve", font_size=38)
        self.play(Write(title))
        self.play(title.animate.to_edge(UP))

        order = 5
        n = 2**order
        pts = [HilbertCurve.d_to_xy(n, d) for d in range(n * n)]
        scale = 6 / n
        mpts = [(x * scale - 3, y * scale - 3, 0) for x, y in pts]

        curve = VMobject(stroke_color=BLUE, stroke_width=2)
        curve.set_points_as_corners(mpts)

        self.play(Create(curve), run_time=6, rate_func=linear)
        caption = Text("1D sequence with 2D locality", font_size=26).next_to(curve, DOWN)
        self.play(FadeIn(caption))
        self.wait(2)


class AdaptiveQuadTreeScene(Scene):
    def construct(self):
        title = Text("Adaptive Quadtree Tokenization", font_size=36)
        self.play(Write(title))
        self.play(title.animate.to_edge(UP))

        max_depth = 4
        n = 2**max_depth
        field = synthetic_field(n)
        leaves = build_quadtree(field, 0, 0, n, 0, max_depth, [])
        leaves.sort(key=lambda x: x["hilbert"])

        # 绘制根框
        base_square = Square(side_length=6).move_to(ORIGIN)
        self.play(Create(base_square))

        # 色板按深度
        colors = color_gradient([BLUE, GREEN, YELLOW, ORANGE, RED], max_depth + 1)

        def map_to_screen(x, y):
            return np.array([ (x / n) * 6 - 3, (y / n) * 6 - 3, 0 ])

        # 按 Hilbert 顺序逐个高亮叶子
        for idx, leaf in enumerate(leaves[: min(len(leaves), 60)]):
            x0, y0, size = leaf["x0"], leaf["y0"], leaf["size"]
            depth = len(leaf["path"])
            p1 = map_to_screen(x0, y0)
            p2 = map_to_screen(x0 + size, y0 + size)
            rect = Rectangle(width=p2[0] - p1[0], height=p2[1] - p1[1]).move_to((p1 + p2) / 2)
            rect.set_stroke(color=colors[depth], width=2)
            rect.set_fill(color=colors[depth], opacity=0.12)
            label = Text(str(idx), font_size=18, color=colors[depth]).move_to(rect.get_center())
            self.play(Create(rect), FadeIn(label), run_time=0.15)

        legend = VGroup(
            *[
                VGroup(
                    Square(side_length=0.3, fill_color=colors[d], fill_opacity=0.8, stroke_width=0),
                    Text(f"depth {d}", font_size=18)
                ).arrange(RIGHT, buff=0.1)
                for d in range(max_depth + 1)
            ]
        ).arrange(RIGHT, buff=0.25).to_edge(DOWN)
        self.play(FadeIn(legend))
        self.wait(2)


class HilbertTraversalScene(Scene):
    def construct(self):
        title = Text("Hilbert Traversal → Token Sequence", font_size=32).to_edge(UP)
        self.add(title)
        order = 4
        n = 2**order
        pts = [HilbertCurve.d_to_xy(n, d) for d in range(n * n)]
        scale = 5.5 / n
        mpts = [(x * scale - 2.75, y * scale - 2.75, 0) for x, y in pts]
        dots = VGroup(*[Dot(point=p, radius=0.04, color=BLUE) for p in mpts])
        self.play(FadeIn(dots, lag_ratio=0.01, run_time=2))

        path = VMobject(stroke_color=YELLOW, stroke_width=2)
        path.set_points_as_corners(mpts)
        self.play(Create(path), run_time=5, rate_func=linear)

        seq = VGroup(*[Text(str(i), font_size=18) for i in range(16)])
        seq.arrange(RIGHT, buff=0.2).to_edge(DOWN)
        self.play(FadeIn(seq))
        self.wait(2)
