from manimlib.scene.scene import Scene
from manimlib.mobject.geometry import Square
from manimlib.mobject.geometry import VMobject
from manimlib.mobject.types import VGroup
from manimlib.mobject.geometry import Dot
from manimlib.constants import ORIGIN, YELLOW, GREEN, RED
from manimlib.animation.creation import ShowCreation
import numpy as np
from vit_pytorch.fractal_curve_tokenizer import zigzag_indices

class FractalZigzagPatch(Scene):
    def construct(self):
        # 设定patch大小
        height, width = 9, 16
        # 生成之字形索引
        idxs = zigzag_indices(height, width)
        # 画网格
        grid = VGroup()
        for r in range(height):
            for c in range(width):
                rect = Square(0.4).move_to(np.array([c, r, 0]))
                grid.add(rect)
        grid.move_to(ORIGIN)
        self.add(grid)
        # 画走线
        path = VMobject(color=YELLOW)
        path.set_points_as_corners([
            np.array([c, r, 0]) for r, c in idxs
        ])
        self.play(ShowCreation(path), run_time=3)
        # 标记入口和出口
        start = Dot(np.array([idxs[0][1], idxs[0][0], 0]), color=GREEN)
        end = Dot(np.array([idxs[-1][1], idxs[-1][0], 0]), color=RED)
        self.add(start, end)
        self.wait(2)
