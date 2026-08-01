from manim import *
import numpy as np

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section3Scene(TeachingScene):
    def construct(self):
        # Setup title and lecture lines
        title_text = "The Concept of Duality: Creating the 'Mirror' Graph"
        lecture_lines = [
            "Every planar graph has a hidden mirror dual.",
            "Place a new vertex inside every original face.",
            "Connect dual vertices if their faces share an edge.",
            "Each primal edge is crossed by one dual edge.",
            "The dual graph reveals the map's connectivity."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        GRAY = "#808080"
        YELLOW = "#FFFF00"
        ORANGE = "#FFA500"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(GRAY)
        
        # Create Primal Graph Vertices (Triangle with point inside)
        # v1 moved from A3 to B3 for clearance (Issue 37)
        v1 = self.place_at_grid(Dot(color=GRAY), "B3", scale_factor=1.0)
        v2 = self.place_at_grid(Dot(color=GRAY), "F1", scale_factor=1.0)
        v3 = self.place_at_grid(Dot(color=GRAY), "F5", scale_factor=1.0)
        v4 = self.place_at_grid(Dot(color=GRAY), "D3", scale_factor=1.0)
        
        # Create Primal Graph Edges
        edges = VGroup(
            Line(v1.get_center(), v2.get_center(), color=GRAY),
            Line(v2.get_center(), v3.get_center(), color=GRAY),
            Line(v3.get_center(), v1.get_center(), color=GRAY),
            Line(v1.get_center(), v4.get_center(), color=GRAY),
            Line(v2.get_center(), v4.get_center(), color=GRAY),
            Line(v3.get_center(), v4.get_center(), color=GRAY)
        )
        
        primal_graph = VGroup(v1, v2, v3, v4, edges)
        self.play(Create(primal_graph))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        
        # Define Interior Dual Vertices in face centers
        # d1/d2 moved from B2/B4 to C2/C4 to maintain center alignment (Issue 38)
        d1 = self.place_at_grid(Dot(color=YELLOW), "C2", scale_factor=1.2)
        d2 = self.place_at_grid(Dot(color=YELLOW), "C4", scale_factor=1.2)
        d3 = self.place_at_grid(Dot(color=YELLOW), "E3", scale_factor=1.2)
        
        dual_interior = VGroup(d1, d2, d3)
        self.play(Create(dual_interior))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        
        # Define Exterior Dual Vertex
        # d_ext moved from A6 to D6 and scaled down for better spacing (Issue 36)
        d_ext = self.place_at_grid(Dot(color=YELLOW), "D6", scale_factor=1.2)
        
        # Emphasize the exterior vertex
        pulse = Circle(radius=0.3, color=YELLOW).move_to(d_ext.get_center())
        self.play(Create(d_ext))
        self.play(pulse.animate.scale(2.5).set_stroke(opacity=0), run_time=0.8)
        self.remove(pulse)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(ORANGE)
        
        # Create Dual Edges (Orange) - ensuring they cross exactly one primal edge
        dual_edges = VGroup(
            Line(d1.get_center(), d2.get_center(), color=ORANGE), # crosses v1-v4
            Line(d2.get_center(), d3.get_center(), color=ORANGE), # crosses v3-v4
            Line(d3.get_center(), d1.get_center(), color=ORANGE), # crosses v2-v4
            Line(d1.get_center(), d_ext.get_center(), color=ORANGE), # crosses v1-v2
            Line(d2.get_center(), d_ext.get_center(), color=ORANGE), # crosses v1-v3
            Line(d3.get_center(), d_ext.get_center(), color=ORANGE)  # crosses v2-v3
        )
        
        self.play(Create(dual_edges))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(ORANGE)
        
        # Asset integration (Issue 28)
        mirror_asset = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/mirror.svg")
        self.place_at_grid(mirror_asset, "A6", scale_factor=0.6)
        mirror_asset.set_color(ORANGE)
        
        # Fade the gray primal graph to focus on the orange dual
        dual_graph = VGroup(dual_interior, d_ext, dual_edges)
        self.play(
            primal_graph.animate.set_stroke(opacity=0.2).set_fill(opacity=0.2),
            dual_graph.animate.scale(1.1),
            FadeIn(mirror_asset),
            run_time=1.5
        )
        self.wait(2)
