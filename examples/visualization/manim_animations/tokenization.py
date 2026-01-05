from manim import *
import numpy as np
import sys
import os

# Add project root to path for imports if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.vit_pytorch.curve_hilbert import HilbertCurve

class AdaptiveTokenization(Scene):
    def construct(self):
        # 1. Setup Image and Grid
        title = Text("Fractal Adaptive Tokenization", font_size=36).to_edge(UP)
        self.play(Write(title))
        
        # Represent image as a square
        image_size = 6
        image_rect = Square(side_length=image_size).set_color(WHITE)
        self.play(Create(image_rect))
        
        # 2. Recursive Splitting Animation
        # We simulate the splitting of a complex region (e.g., top-left)
        
        patches = [] # Store (x, y, size, depth) relative to center
        
        # Initial state: 1 big patch
        self.wait(1)
        
        # Level 1 split
        # Split top-left and bottom-right (simulating complexity)
        # Coordinates: Center is (0,0). Top-Left is (-1.5, 1.5) for size 3 sub-squares
        
        # Helper to create grid lines
        def split_square(square, depth):
            # Visual split
            lines = VGroup()
            c = square.get_center()
            s = square.width
            v_line = Line(c + UP*s/2, c + DOWN*s/2)
            h_line = Line(c + LEFT*s/2, c + RIGHT*s/2)
            lines.add(v_line, h_line)
            return lines

        # Split Level 1
        l1_lines = split_square(image_rect, 1)
        self.play(Create(l1_lines))
        self.wait(0.5)
        
        # Define the 4 quadrants of Level 1
        # 0: TL, 1: TR, 2: BL, 3: BR (Manim coords)
        # Let's say TL (0) and BR (3) are complex and split again.
        
        s = image_size / 2
        tl_center = image_rect.get_center() + LEFT*s/2 + UP*s/2
        tr_center = image_rect.get_center() + RIGHT*s/2 + UP*s/2
        bl_center = image_rect.get_center() + LEFT*s/2 + DOWN*s/2
        br_center = image_rect.get_center() + RIGHT*s/2 + DOWN*s/2
        
        tl_rect = Square(side_length=s).move_to(tl_center)
        br_rect = Square(side_length=s).move_to(br_center)
        
        # Split Level 2 (TL and BR)
        l2_lines_tl = split_square(tl_rect, 2)
        l2_lines_br = split_square(br_rect, 2)
        
        self.play(Create(l2_lines_tl), Create(l2_lines_br))
        self.wait(0.5)
        
        # 3. Hilbert Curve Traversal
        # We define a recursive function to traverse the adaptive grid
        # matching the logic in src/vit_pytorch/curve_hilbert.py
        
        # Define the tree structure
        # 0: TL, 1: TR, 2: BL, 3: BR
        # Root children: [TL(Split), TR(Leaf), BL(Leaf), BR(Split)]
        
        class Node:
            def __init__(self, center, size, is_split=False, children=None):
                self.center = center
                self.size = size
                self.is_split = is_split
                self.children = children # List of 4 Nodes if split, else None

        def build_tree():
            # Root
            root_center = image_rect.get_center()
            root_size = image_size
            
            # Level 1 Centers
            s1 = root_size / 2
            c_tl = root_center + LEFT*s1/2 + UP*s1/2
            c_tr = root_center + RIGHT*s1/2 + UP*s1/2
            c_bl = root_center + LEFT*s1/2 + DOWN*s1/2
            c_br = root_center + RIGHT*s1/2 + DOWN*s1/2
            
            # Level 2 Centers (for TL)
            s2 = s1 / 2
            c_tl_tl = c_tl + LEFT*s2/2 + UP*s2/2
            c_tl_tr = c_tl + RIGHT*s2/2 + UP*s2/2
            c_tl_bl = c_tl + LEFT*s2/2 + DOWN*s2/2
            c_tl_br = c_tl + RIGHT*s2/2 + DOWN*s2/2
            
            # Level 2 Centers (for BR)
            c_br_tl = c_br + LEFT*s2/2 + UP*s2/2
            c_br_tr = c_br + RIGHT*s2/2 + UP*s2/2
            c_br_bl = c_br + LEFT*s2/2 + DOWN*s2/2
            c_br_br = c_br + RIGHT*s2/2 + DOWN*s2/2
            
            # Build Nodes
            # TL Children
            n_tl_tl = Node(c_tl_tl, s2)
            n_tl_tr = Node(c_tl_tr, s2)
            n_tl_bl = Node(c_tl_bl, s2)
            n_tl_br = Node(c_tl_br, s2)
            
            # BR Children
            n_br_tl = Node(c_br_tl, s2)
            n_br_tr = Node(c_br_tr, s2)
            n_br_bl = Node(c_br_bl, s2)
            n_br_br = Node(c_br_br, s2)
            
            # Level 1 Nodes
            n_tl = Node(c_tl, s1, is_split=True, children=[n_tl_tl, n_tl_tr, n_tl_bl, n_tl_br])
            n_tr = Node(c_tr, s1, is_split=False)
            n_bl = Node(c_bl, s1, is_split=False)
            n_br = Node(c_br, s1, is_split=True, children=[n_br_tl, n_br_tr, n_br_bl, n_br_br])
            
            # Root Node
            root = Node(root_center, root_size, is_split=True, children=[n_tl, n_tr, n_bl, n_br])
            return root

        root_node = build_tree()
        
        def get_hilbert_path(node, orientation):
            if not node.is_split:
                return [node.center]
            
            points = []
            # Get visit order for this orientation
            # BASE_ORDERS maps orientation to list of child indices (0=TL, 1=TR, 2=BL, 3=BR)
            visit_order = HilbertCurve.BASE_ORDERS[orientation]
            
            # Get orientation for each child
            # ORIENTATION_MAP maps orientation to list of child orientations
            child_orientations = HilbertCurve.ORIENTATION_MAP[orientation]
            
            for child_idx in visit_order:
                child = node.children[child_idx]
                child_orient = child_orientations[child_idx]
                points.extend(get_hilbert_path(child, child_orient))
                
            return points

        # Start traversal with 'up' orientation
        path_points = get_hilbert_path(root_node, 'up')
        
        # Draw the path
        path = VMobject()
        path.set_points_as_corners([p for p in path_points])
        path.set_color(YELLOW)
        
        self.play(Create(path), run_time=4)
        self.wait(1)
        
        # 4. Flatten to 1D Sequence
        # Show tokens lining up at the bottom
        tokens = VGroup()
        for i, p in enumerate(path_points):
            # Create a small token representation
            t = Square(side_length=0.4).set_fill(BLUE, opacity=0.5)
            tokens.add(t)
            
        tokens.arrange(RIGHT, buff=0.1)
        tokens.to_edge(DOWN)
        
        self.play(TransformFromCopy(path, tokens))
        
        seq_text = Text("1D Token Sequence", font_size=24).next_to(tokens, UP)
        self.play(Write(seq_text))
        
        self.wait(2)

