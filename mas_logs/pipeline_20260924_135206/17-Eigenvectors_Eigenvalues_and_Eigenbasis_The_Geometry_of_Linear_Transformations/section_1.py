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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Most vectors rotate under a matrix transformation.",
            "Eigenvectors are special vectors that don't rotate.",
            "They only scale by a certain factor.",
            "Imagine a grid stretching like a rubber sheet.",
            "The line along the stretch remains perfectly fixed."
        ]
        self.setup_layout("The Intuition: When Transformations act as Scaling", lecture_lines)
        
        # Create elements
        # SVG path from Assets
        sheet_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/sheet.svg"
        
        # Vector and span line
        vec = Vector([1, 1], color=WHITE)
        span_line = Line(start=LEFT*3, end=RIGHT*3, color="#FF00FF", stroke_opacity=0.5)
        
        # Grid visual
        grid_visual = SVGMobject(sheet_asset)
        
        # Apply positioning fixes
        self.place_at_grid(vec, 'D3', scale_factor=0.6)
        self.place_in_area(span_line, 'D2', 'D4', scale_factor=0.7)
        self.place_in_area(grid_visual, 'B4', 'F6', scale_factor=0.85)
        
        # Add grid_visual to scene
        self.add(grid_visual)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        self.play(Create(vec))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        self.play(Create(span_line))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        self.play(vec.animate.scale(1.5))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FFA500"))
        # Simulating stretch
        self.play(vec.animate.stretch(2, dim=0))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FF00FF"))
        self.wait(2)
