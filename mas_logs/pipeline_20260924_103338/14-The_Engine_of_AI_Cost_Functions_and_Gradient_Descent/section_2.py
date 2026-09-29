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

class Section2Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Imagine a landscape where height equals error.",
            "Parameters define our position on this terrain.",
            "Our goal is reaching the lowest valley."
        ]
        self.setup_layout("Visualizing the Cost Landscape", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Using SVG Assets
        surface = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/mountain.svg", color="#888888")
        surface_label = Text("Cost Landscape", font_size=20, color="#888888")
        self.place_in_area(surface, 'A4', 'B6', scale_factor=0.8)
        self.place_at_grid(surface_label, 'C5', scale_factor=0.7)
        self.play(FadeIn(surface), Write(surface_label))
        self.lecture[0].set_color("#888888")

        # === Animation for Lecture Line 2 ===
        # Parameters (dot)
        params_dot = Dot(color="#FFFF00")
        params_label = Text("Parameters", font_size=18, color="#FFFF00")
        self.place_at_grid(params_dot, 'D3', scale_factor=1.0)
        self.place_at_grid(params_label, 'D4', scale_factor=0.7)
        self.play(Create(params_dot), Write(params_label))
        self.lecture[1].set_color("#FFFF00")

        # === Animation for Lecture Line 3 ===
        # Terminal anchor (valley)
        anchor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/valley.svg", color="#00FF00")
        anchor_label = Text("Goal", font_size=18, color="#00FF00")
        self.place_at_grid(anchor, 'D5', scale_factor=0.6)
        self.place_at_grid(anchor_label, 'D6', scale_factor=0.7)
        self.play(FadeIn(anchor), Write(anchor_label))
        self.lecture[2].set_color("#00FF00")

        self.wait(2)
