import math
import numpy as np
import pytest

pytest.importorskip("manimlib")

from manimlib.animation.creation import ShowCreation
from manimlib.constants import GREEN, ORIGIN, RED, YELLOW
from manimlib.mobject.geometry import Dot, Square
from manimlib.mobject.types.vectorized_mobject import VMobject, VGroup
from manimlib.scene.scene import Scene

from vit_pytorch.fractal_curve_tokenizer import FractalHilbertTokenizer


class FractalHilbertPatch(Scene):
    def construct(self):
        grid_size = 8
        hilbert_order = int(math.log2(grid_size))
        tokenizer = FractalHilbertTokenizer(min_patch_size=(1, 1), max_level=hilbert_order)
        coords = tokenizer.generate_true_hilbert_curve(hilbert_order)

        grid = VGroup()
        for row in range(grid_size):
            for col in range(grid_size):
                rect = Square(0.35).move_to(np.array([col, row, 0]))
                grid.add(rect)

        grid.move_to(ORIGIN)
        self.add(grid)

        path = VMobject(color=YELLOW)
        path.set_points_as_corners([np.array([x, y, 0]) for x, y in coords])
        self.play(ShowCreation(path), run_time=3)

        start = Dot(np.array([coords[0][0], coords[0][1], 0]), color=GREEN)
        end = Dot(np.array([coords[-1][0], coords[-1][1], 0]), color=RED)
        self.add(start, end)
        self.wait(2)


def test_hilbert_curve_coordinates_are_unique() -> None:
    tokenizer = FractalHilbertTokenizer(min_patch_size=(1, 1))
    coords = tokenizer.generate_true_hilbert_curve(3)

    assert len(coords) == 64
    assert coords[0] == (0, 0)
    assert len({tuple(point) for point in coords}) == len(coords)
