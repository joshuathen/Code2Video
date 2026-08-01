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
        # Setup the layout with section-specific content
        self.setup_layout(
            "The Concept of Duality: The Mirror World",
            [
                "Every planar graph possesses a twin called a dual.",
                "Original faces transform into vertices in the dual world.",
                "Dual edges connect vertices across every original boundary."
            ]
        )

        # === Animation for Lecture Line 1 ===
        # Display a planar graph G with 3 regions.
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        # Define vertices for a square with a diagonal to create 3 regions (2 internal + 1 external)
        # Using Grid Points: C2, C5, F5, F2 (Shifted down - Issue 35)
        v_tl = Dot(color=WHITE)
        self.place_at_grid(v_tl, "C2")
        v_tr = Dot(color=WHITE)
        self.place_at_grid(v_tr, "C5")
        v_br = Dot(color=WHITE)
        self.place_at_grid(v_br, "F5")
        v_bl = Dot(color=WHITE)
        self.place_at_grid(v_bl, "F2")
        
        # Original edges
        e_top = Line(v_tl.get_center(), v_tr.get_center(), color=WHITE)
        e_right = Line(v_tr.get_center(), v_br.get_center(), color=WHITE)
        e_bottom = Line(v_br.get_center(), v_bl.get_center(), color=WHITE)
        e_left = Line(v_bl.get_center(), v_tl.get_center(), color=WHITE)
        e_diag = Line(v_tr.get_center(), v_bl.get_center(), color=WHITE)
        
        graph_g = VGroup(v_tl, v_tr, v_br, v_bl, e_top, e_right, e_bottom, e_left, e_diag)
        self.play(Create(graph_g))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Place a magenta (#FF00FF) dot in every region (face) of G.
        self.play(self.lecture[1].animate.set_color("#FF00FF"))
        
        # Dual Vertices (Face Centers)
        # f1 (Interior Top-Left Triangle): Centroid at D3
        dv1 = Dot(color="#FF00FF", radius=0.15)
        self.place_at_grid(dv1, "D3")
        
        # f2 (Interior Bottom-Right Triangle): Centroid at E4
        dv2 = Dot(color="#FF00FF", radius=0.15)
        self.place_at_grid(dv2, "E4")
        
        # f3 (Exterior Region): Repositioned to B3 (Issue 36)
        dv3 = Dot(color="#FF00FF", radius=0.15)
        self.place_at_grid(dv3, "B3")
        
        self.play(FadeIn(dv1), FadeIn(dv2), FadeIn(dv3))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Draw cyan (#00FFFF) lines crossing original edges to connect the dots.
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        
        # Dual Edges (Cyan) - each crossing one original edge
        # Connection f1 <-> f2 crossing diagonal (e_diag)
        de_diag = Line(dv1.get_center(), dv2.get_center(), color="#00FFFF")
        
        # Connections for f1 crossing Top and Left boundaries
        de_top = ArcBetweenPoints(dv1.get_center(), dv3.get_center(), angle=-0.6, color="#00FFFF")
        de_left = ArcBetweenPoints(dv1.get_center(), dv3.get_center(), angle=0.8, color="#00FFFF")
        
        # Connections for f2 crossing Right and Bottom boundaries
        de_right = ArcBetweenPoints(dv2.get_center(), dv3.get_center(), angle=-1.1, color="#00FFFF")
        de_bottom = ArcBetweenPoints(dv2.get_center(), dv3.get_center(), angle=1.2, color="#00FFFF")
        
        dual_edges = VGroup(de_diag, de_top, de_left, de_right, de_bottom)
        self.play(Create(dual_edges))
        self.wait(2)
