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
        self.setup_layout("The Concept of Graph Duality", [
            "Construct the dual graph by placing nodes in faces.",
            "Connect dual nodes if faces share an edge.",
            "The dual graph acts like a structural mirror."
        ])
        
        # Original graph G components
        v1 = Dot(color=BLUE).move_to(self.grid["B3"])
        v2 = Dot(color=BLUE).move_to(self.grid["B5"])
        v3 = Dot(color=BLUE).move_to(self.grid["D5"])
        v4 = Dot(color=BLUE).move_to(self.grid["D3"])
        edges = VGroup(
            Line(v1.get_center(), v2.get_center(), color=BLUE),
            Line(v2.get_center(), v3.get_center(), color=BLUE),
            Line(v3.get_center(), v4.get_center(), color=BLUE),
            Line(v4.get_center(), v1.get_center(), color=BLUE),
            Line(v1.get_center(), v3.get_center(), color=BLUE)
        )
        graph_g = VGroup(v1, v2, v3, v4, edges)
        self.place_in_area(graph_g, 'B3', 'E5', scale_factor=0.9)
        
        # === Animation for Lecture Line 1 ===
        self.play(Create(graph_g))
        self.lecture[0].set_color(YELLOW)
        
        dual_n1 = Dot(color="#FFFFFF").move_to(self.grid["C4"])
        dual_n2 = Dot(color="#FFFFFF").move_to(self.grid["C3"])
        self.play(FadeIn(dual_n1), FadeIn(dual_n2))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        dual_edge = Line(dual_n1.get_center(), dual_n2.get_center(), color="#FFA500")
        self.play(Create(dual_edge))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        mirror_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/mirror.svg")
        self.place_at_grid(mirror_icon, 'C4', scale_factor=0.5)
        
        self.play(
            graph_g.animate.set_stroke(opacity=0.3).set_fill(opacity=0.3),
            dual_n1.animate.set_color("#808080"),
            dual_n2.animate.set_color("#808080"),
            dual_edge.animate.set_color("#808080"),
            FadeIn(mirror_icon)
        )
        self.wait(2)
