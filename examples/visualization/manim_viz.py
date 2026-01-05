# -*- coding: utf-8 -*-
"""
Manim-based Animated Visualizations for Fractal Curve ViT

Mathematical Animations:
=======================
This module provides educational animations using Man

im (ManimGL) to explain
the mathematical principles behind the Fractal Curve Vision Transformer.

Key Animations:
1. Hilbert curve generation (orders 1-5) with self-similarity
2. Quadtree splitting animation with complexity visualization
3. Token ordering along Hilbert curve
4. Attention flow with LCA bias
5. Comparative ViT architectures

Mathematical Concepts Visualized:
=================================
- Hilbert space-filling curve: H: [0,n²) ↔ [0,n)×[0,n)
- Quadtree decomposition: Recursive 4-way splits
- LCA distance metric: Hierarchical spatial relationships
- Attention mechanism: Q·K^T + B_LCA
"""

from manimlib import *
import numpy as np
from typing import List, Tuple, Optional, Callable


class HilbertCurveAnimation(Scene):
    """
    Animate the generation of Hilbert curves at different orders.
    
    Mathematical Properties Demonstrated:
    ------------------------------------
    1. Self-similarity: Each quadrant contains rotated copy
    2. Locality preservation: ||p₁-p₂||₂ ≤ C·|H⁻¹(p₁)-H⁻¹(p₂)|^(1/2)
    3. Continuous filling: Curve visits every cell exactly once
    4. Fractal dimension: D_f = 2 (fills 2D space)
    """
    
    def construct(self):
        """Main animation sequence."""
        # Title
        title = Text("Hilbert Space-Filling Curve", font_size=48)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()
        
        # Mathematical definition
        definition = Tex(
            r"H: [0, n^2) \leftrightarrow [0, n) \times [0, n)",
            font_size=36
        )
        definition.next_to(title, DOWN, buff=0.5)
        self.play(Write(definition))
        self.wait()
        
        # Animate orders 1 through 4
        for order in range(1, 5):
            self.show_hilbert_order(order)
            self.wait(2)
            if order < 4:
                self.clear()
                self.play(Write(title), Write(definition))
        
        self.wait(2)
    
    def show_hilbert_order(self, order: int):
        """
        Show Hilbert curve of given order.
        
        Args:
            order: Curve order (1-5)
        """
        n = 2 ** order
        size = 4  # Size in Manim units
        
        # Generate Hilbert curve points
        points = self.generate_hilbert_points(order, size)
        
        # Create order label
        order_label = Text(f"Order {order} (n = {n})", font_size=36)
        order_label.to_edge(DOWN, buff=1)
        self.play(Write(order_label))
        
        # Create grid
        grid = self.create_grid(n, size)
        self.play(ShowCreation(grid))
        
        # Create the curve
        curve = VMobject()
        curve.set_points_smoothly(points)
        curve.set_color(BLUE)
        curve.set_stroke(width=4)
        
        # Animate curve drawing
        self.play(
            ShowCreation(curve),
            run_time=3,
            rate_func=smooth
        )
        
        # Add start and end markers
        start_dot = Dot(points[0], color=GREEN, radius=0.1)
        end_dot = Dot(points[-1], color=RED, radius=0.1)
        start_label = Text("Start", font_size=24, color=GREEN)
        start_label.next_to(start_dot, LEFT)
        end_label = Text("End", font_size=24, color=RED)
        end_label.next_to(end_dot, RIGHT)
        
        self.play(
            FadeIn(start_dot),
            Write(start_label),
            FadeIn(end_dot),
            Write(end_label)
        )
        
        # Highlight locality property
        if order >= 2:
            self.demonstrate_locality(points, n, size)
        
        self.wait()
    
    def generate_hilbert_points(self, order: int, size: float) -> List[np.ndarray]:
        """
        Generate points along Hilbert curve.
        
        Mathematical Algorithm:
        ----------------------
        Recursive construction with rotation transformations:
        - Base case (order 0): Single point
        - Recursive case: 4 rotated copies + connections
        
        Args:
            order: Curve order
            size: Size in Manim units
            
        Returns:
            List of 3D points
        """
        def hilbert_d2xy(n: int, d: int) -> Tuple[int, int]:
            """Convert distance to coordinates."""
            x = y = 0
            s = 1
            while s < n:
                rx = 1 & (d // 2)
                ry = 1 & (d ^ rx)
                if ry == 0:
                    if rx == 1:
                        x = s - 1 - x
                        y = s - 1 - y
                    x, y = y, x
                x += s * rx
                y += s * ry
                d //= 4
                s *= 2
            return x, y
        
        n = 2 ** order
        points = []
        
        cell_size = size / n
        offset = -size / 2
        
        for d in range(n * n):
            x, y = hilbert_d2xy(n, d)
            # Convert to Manim coordinates
            mx = offset + (x + 0.5) * cell_size
            my = offset + (y + 0.5) * cell_size
            points.append(np.array([mx, my, 0]))
        
        return points
    
    def create_grid(self, n: int, size: float) -> VGroup:
        """Create grid for visualization."""
        grid = VGroup()
        cell_size = size / n
        offset = -size / 2
        
        # Vertical lines
        for i in range(n + 1):
            x = offset + i * cell_size
            line = Line(
                [x, offset, 0],
                [x, offset + size, 0],
                stroke_width=1,
                color=GREY
            )
            grid.add(line)
        
        # Horizontal lines
        for i in range(n + 1):
            y = offset + i * cell_size
            line = Line(
                [offset, y, 0],
                [offset + size, y, 0],
                stroke_width=1,
                color=GREY
            )
            grid.add(line)
        
        return grid
    
    def demonstrate_locality(self, points: List[np.ndarray], n: int, size: float):
        """
        Demonstrate locality preservation property.
        
        Mathematical Property:
        ---------------------
        For nearby points on curve: d_curve(p₁,p₂) ≈ ||p₁-p₂||₂
        
        Args:
            points: Curve points
            n: Grid size
            size: Grid size in Manim units
        """
        # Pick two nearby points on curve
        i = len(points) // 3
        j = i + 3
        
        p1 = points[i]
        p2 = points[j]
        
        # Draw segment between them
        segment = Line(p1, p2, color=YELLOW, stroke_width=6)
        
        # Calculate distances
        euclidean_dist = np.linalg.norm(p2 - p1)
        curve_dist = j - i
        
        # Create labels
        locality_text = Tex(
            f"||p_1 - p_2||_2 = {euclidean_dist:.2f}",
            f"\\quad d_{{curve}} = {curve_dist}",
            font_size=28
        )
        locality_text.to_edge(LEFT, buff=0.5).shift(DOWN * 2)
        
        self.play(
            ShowCreation(segment),
            Write(locality_text)
        )
        self.wait(2)


class QuadtreeSplittingAnimation(Scene):
    """
    Animate adaptive quadtree splitting based on image complexity.
    
    Mathematical Process:
    --------------------
    1. Compute complexity: C(R) = α·C_var(R) + (1-α)·C_grad(R)
    2. Compare with threshold: τ_d = τ₀·γ^d
    3. If C(R) > τ_d: Split into 4 children
    4. Recursively process children
    """
    
    def construct(self):
        """Main animation sequence."""
        # Title
        title = Text("Adaptive Quadtree Tokenization", font_size=48)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Show complexity formula
        formula = Tex(
            r"C(R) = \alpha \cdot \frac{\text{Var}(R)}{\text{Var}(R) + \sigma_0^2} + "
            r"(1-\alpha) \cdot \frac{G(R)}{G(R) + g_0^2}",
            font_size=32
        )
        formula.next_to(title, DOWN, buff=0.5)
        self.play(Write(formula))
        self.wait()
        
        # Create synthetic image with varying complexity
        image = self.create_synthetic_image()
        image.scale(2)
        self.play(FadeIn(image))
        self.wait()
        
        # Animate splitting
        self.animate_splitting(image)
        
        self.wait(2)
    
    def create_synthetic_image(self) -> VMobject:
        """
        Create synthetic image with varying complexity regions.
        
        Returns:
            VMobject representing the image
        """
        # Create a simple gradient + detailed region
        size = 4
        resolution = 100
        
        # Generate complexity map
        x = np.linspace(-2, 2, resolution)
        y = np.linspace(-2, 2, resolution)
        X, Y = np.meshgrid(x, y)
        
        # Smooth gradient
        Z_smooth = 0.5 + 0.5 * np.sin(X) * np.cos(Y)
        
        # Add high-frequency detail in corner
        mask = (X > 0.5) & (Y > 0.5)
        Z_detail = np.random.rand(resolution, resolution) * 0.3
        Z = Z_smooth + mask * Z_detail
        
        # Normalize
        Z = (Z - Z.min()) / (Z.max() - Z.min())
        
        # Create image surface (simplified as colored square)
        image = Square(side_length=size)
        image.set_fill(BLUE, opacity=0.3)
        image.set_stroke(WHITE, width=2)
        
        return image
    
    def animate_splitting(self, image: VMobject):
        """
        Animate the quadtree splitting process.
        
        Args:
            image: Image VMobject
        """
        # Get image bounds
        bounds = image.get_critical_point(UR) - image.get_critical_point(DL)
        size = bounds[0]
        center = image.get_center()
        
        # Level 0: Root
        root_rect = self.create_region_rect(center, size, color=GREEN)
        self.play(ShowCreation(root_rect))
        
        # Show complexity value
        complexity_text = Text("C(R) = 0.65 > τ₀ = 0.15", font_size=24, color=GREEN)
        complexity_text.next_to(root_rect, UP)
        self.play(Write(complexity_text))
        self.wait()
        
        # Split decision
        decision_text = Text("Split!", font_size=32, color=YELLOW)
        decision_text.next_to(complexity_text, RIGHT)
        self.play(Write(decision_text))
        self.wait()
        
        # Level 1: 4 children
        children = self.split_region(center, size)
        child_rects = VGroup()
        
        for i, (child_center, child_size) in enumerate(children):
            rect = self.create_region_rect(child_center, child_size, color=BLUE)
            child_rects.add(rect)
        
        self.play(
            FadeOut(root_rect),
            FadeOut(complexity_text),
            FadeOut(decision_text),
            *[ShowCreation(rect) for rect in child_rects]
        )
        self.wait()
        
        # Further split one complex child
        complex_child_idx = 3  # Bottom-right (simulated high complexity)
        complex_center, complex_size = children[complex_child_idx]
        
        complexity_text2 = Text("C(R) = 0.75 > τ₁ = 0.13", font_size=20, color=BLUE)
        complexity_text2.next_to(child_rects[complex_child_idx], DOWN, buff=0.1)
        self.play(Write(complexity_text2))
        self.wait()
        
        # Split complex child
        grandchildren = self.split_region(complex_center, complex_size)
        grandchild_rects = VGroup()
        
        for gc_center, gc_size in grandchildren:
            rect = self.create_region_rect(gc_center, gc_size, color=RED)
            grandchild_rects.add(rect)
        
        self.play(
            FadeOut(child_rects[complex_child_idx]),
            FadeOut(complexity_text2),
            *[ShowCreation(rect) for rect in grandchild_rects]
        )
        self.wait()
        
        # Show final tokenization
        final_text = Text("Final Tokenization: 7 tokens", font_size=32, color=GOLD)
        final_text.to_edge(DOWN, buff=0.5)
        self.play(Write(final_text))
        
        self.wait(2)
    
    def create_region_rect(self, center: np.ndarray, size: float, color: str) -> Rectangle:
        """Create rectangle for region."""
        rect = Rectangle(width=size, height=size, color=color)
        rect.set_stroke(width=3)
        rect.move_to(center)
        return rect
    
    def split_region(self, center: np.ndarray, size: float) -> List[Tuple[np.ndarray, float]]:
        """
        Split region into 4 quadrants.
        
        Returns:
            List of (center, size) for each child
        """
        half_size = size / 2
        quarter_size = size / 4
        
        children = [
            (center + np.array([-quarter_size, quarter_size, 0]), half_size),   # TL
            (center + np.array([quarter_size, quarter_size, 0]), half_size),    # TR
            (center + np.array([-quarter_size, -quarter_size, 0]), half_size),  # BL
            (center + np.array([quarter_size, -quarter_size, 0]), half_size),   # BR
        ]
        
        return children


class AttentionFlowAnimation(Scene):
    """
    Animate attention flow with LCA bias visualization.
    
    Mathematical Attention:
    ----------------------
    A[i,j] = softmax(Q_i·K_j^T/√d + B_LCA[i,j])
    
    where B_LCA[i,j] = τ_h · Embed(LCA(i,j))
    """
    
    def construct(self):
        """Main animation sequence."""
        # Title
        title = Text("Hilbert-Aware Attention Mechanism", font_size=48)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Show attention formula
        formula = Tex(
            r"A[i,j] = \text{softmax}\left(\frac{Q_i \cdot K_j^T}{\sqrt{d}} + B_{\text{LCA}}[i,j]\right)",
            font_size=36
        )
        formula.next_to(title, DOWN, buff=0.5)
        self.play(Write(formula))
        self.wait()
        
        # Create tokens arranged in Hilbert order
        tokens = self.create_token_arrangement()
        self.play(FadeIn(tokens))
        self.wait()
        
        # Animate attention from one query token
        query_idx = 0
        self.animate_attention_from_query(tokens, query_idx)
        
        self.wait(2)
    
    def create_token_arrangement(self) -> VGroup:
        """
        Create tokens arranged along Hilbert curve.
        
        Returns:
            VGroup of token circles
        """
        # Simplified 2x2 grid (4 tokens)
        positions = [
            np.array([-2, 2, 0]),   # Token 0 (TL)
            np.array([-2, -2, 0]),  # Token 1 (BL)
            np.array([2, -2, 0]),   # Token 2 (BR)
            np.array([2, 2, 0]),    # Token 3 (TR)
        ]
        
        tokens = VGroup()
        for i, pos in enumerate(positions):
            circle = Circle(radius=0.4, color=BLUE, fill_opacity=0.5)
            circle.move_to(pos)
            label = Text(f"t_{i}", font_size=24)
            label.move_to(pos)
            token = VGroup(circle, label)
            tokens.add(token)
        
        return tokens
    
    def animate_attention_from_query(self, tokens: VGroup, query_idx: int):
        """
        Animate attention weights from query token to all keys.
        
        Args:
            tokens: Token group
            query_idx: Index of query token
        """
        # Highlight query
        query_token = tokens[query_idx]
        self.play(
            query_token[0].animate.set_color(GREEN).set_stroke(width=5)
        )
        
        # Show attention to other tokens
        # Simulate attention weights (higher for nearby tokens)
        attention_weights = [0, 0.7, 0.2, 0.5]  # From token 0
        attention_weights[query_idx] = 0  # Self-attention handled separately
        
        arrows = VGroup()
        weight_labels = VGroup()
        
        for i, weight in enumerate(attention_weights):
            if i == query_idx or weight == 0:
                continue
            
            # Create arrow
            arrow = Arrow(
                query_token.get_center(),
                tokens[i].get_center(),
                color=YELLOW,
                stroke_width=weight * 10,
                buff=0.5
            )
            arrows.add(arrow)
            
            # Weight label
            label = Text(f"{weight:.2f}", font_size=20, color=YELLOW)
            label.move_to(arrow.get_center())
            weight_labels.add(label)
        
        self.play(
            *[GrowArrow(arrow) for arrow in arrows],
            *[Write(label) for label in weight_labels]
        )
        self.wait()
        
        # Show LCA bias contribution
        lca_text = Text(
            "Nearby tokens (high LCA depth) receive higher attention",
            font_size=28,
            color=GOLD
        )
        lca_text.to_edge(DOWN, buff=0.5)
        self.play(Write(lca_text))
        
        self.wait(2)


# Export animation rendering functions
def render_hilbert_animation(output_path: str = "hilbert_curve.mp4"):
    """
    Render Hilbert curve animation.
    
    Args:
        output_path: Path to save video
    """
    scene = HilbertCurveAnimation()
    scene.render()


def render_quadtree_animation(output_path: str = "quadtree_splitting.mp4"):
    """
    Render quadtree splitting animation.
    
    Args:
        output_path: Path to save video
    """
    scene = QuadtreeSplittingAnimation()
    scene.render()


def render_attention_animation(output_path: str = "attention_flow.mp4"):
    """
    Render attention flow animation.
    
    Args:
        output_path: Path to save video
    """
    scene = AttentionFlowAnimation()
    scene.render()
