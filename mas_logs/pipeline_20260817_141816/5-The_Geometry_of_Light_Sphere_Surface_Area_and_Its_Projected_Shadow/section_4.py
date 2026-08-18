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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Real-World Application & Summary", [
            "Satellites use this ratio for heat.",
            "Sunlight exposure depends on surface area.",
            "Knowing area prevents overheating in space."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Display a real-world object like an orange [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/orange.svg] in #FF8C00
        orange = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/orange.svg")
        orange.set_color("#FF8C00")
        self.place_at_grid(orange, 'C3', scale_factor=0.8)
        self.play(FadeIn(orange))
        self.lecture[0].set_color("#FF8C00")

        # === Animation for Lecture Line 2 ===
        # Label the radius of the orange as r
        radius_line = Line(orange.get_center(), orange.get_center() + RIGHT * 1, color=WHITE)
        label_r = MathTex("r", font_size=24).next_to(radius_line, UP, buff=0.1)
        self.play(Create(radius_line), Write(label_r))
        self.lecture[1].set_color("#FF8C00")

        # === Animation for Lecture Line 3 ===
        # Summarize the surface area formula 4 * pi * r^2
        formula = MathTex("S = 4 \\pi r^2", font_size=36, color="#FF8C00")
        self.place_at_grid(formula, 'E3', scale_factor=1.0)
        
        # Visual grouping
        combined_visual = VGroup(orange, radius_line, label_r, formula)
        
        self.play(Write(formula))
        self.lecture[2].set_color("#FF8C00")
        self.wait(2)
