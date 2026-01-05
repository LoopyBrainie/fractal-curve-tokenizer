from manim import *
import sys
import os

class LCAAttention(Scene):
    def construct(self):
        title = Text("LCA-Based Attention Bias", font_size=36).to_edge(UP)
        self.play(Write(title))
        
        # 1. Draw the Quadtree Structure
        # Root
        root = Circle(radius=0.3, color=WHITE).move_to(UP*2)
        
        # Level 1 Children (4 nodes)
        l1_nodes = VGroup()
        for i in range(4):
            node = Circle(radius=0.25, color=BLUE)
            node.move_to(UP*0.5 + LEFT*3 + RIGHT*2*i)
            l1_nodes.add(node)
            
        # Connect Root to L1
        l1_edges = VGroup()
        for node in l1_nodes:
            l1_edges.add(Line(root.get_bottom(), node.get_top()))
            
        self.play(Create(root), Create(l1_nodes), Create(l1_edges))
        
        # Level 2 Children (Split the 2nd node only for demo)
        parent = l1_nodes[1]
        l2_nodes = VGroup()
        for i in range(4):
            node = Circle(radius=0.2, color=GREEN)
            node.move_to(DOWN*1 + LEFT*1.5 + RIGHT*1*i) # Adjust positions
            l2_nodes.add(node)
            
        l2_edges = VGroup()
        for node in l2_nodes:
            l2_edges.add(Line(parent.get_bottom(), node.get_top()))
            
        self.play(Create(l2_nodes), Create(l2_edges))
        
        # 2. Select Two Tokens
        # Token A: A child of the split node (L2)
        # Token B: A sibling of the split node (L1)
        
        token_a = l2_nodes[0]
        token_b = l1_nodes[2]
        
        self.play(
            token_a.animate.set_fill(RED, opacity=0.5),
            token_b.animate.set_fill(RED, opacity=0.5)
        )
        
        label_a = Text("Token A", font_size=20).next_to(token_a, DOWN)
        label_b = Text("Token B", font_size=20).next_to(token_b, DOWN)
        self.play(Write(label_a), Write(label_b))
        
        # 3. Find LCA
        # Ancestors of A: parent (L1_1), root
        # Ancestors of B: root
        # LCA is root.
        
        # Highlight path to root
        path_a = VGroup(l2_edges[0], l1_edges[1])
        path_b = VGroup(l1_edges[2])
        
        self.play(
            path_a.animate.set_color(YELLOW).set_stroke(width=5),
            path_b.animate.set_color(YELLOW).set_stroke(width=5)
        )
        
        self.play(root.animate.set_fill(YELLOW, opacity=0.8))
        lca_label = Text("LCA", font_size=24).next_to(root, RIGHT)
        self.play(Write(lca_label))
        
        # 4. Show Attention Bias Formula
        # Bias ~ -Distance
        formula = MathTex(r"Bias_{i,j} = g(d(i, LCA) + d(j, LCA))").to_edge(DOWN)
        self.play(Write(formula))
        
        self.wait(2)
