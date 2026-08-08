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
        self.setup_layout("Prerequisite Review: The Distance Formula", [
            "Distance begins with the Pythagorean theorem.",
            "Euclidean distance extends to any dimension.",
            "Calculate the vector magnitude in N-space."
        ])
        
        # === Animation for Lecture Line 1 ===
        pythagoras = MathTex("a^2 + b^2 = c^2", color="#FFD700")
        self.place_in_area(pythagoras, 'A2', 'B5', scale_factor=0.9)
        self.play(Write(pythagoras))
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        euclidean = MathTex("d = \\sqrt{x_1^2 + x_2^2}", color="#00FFFF")
        self.place_in_area(euclidean, 'C2', 'D5', scale_factor=0.9)
        self.play(FadeIn(euclidean))
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        n_space = MathTex("d = \\sqrt{\\sum_{i=1}^{n} x_i^2}", color="#FF69B4")
        self.place_in_area(n_space, 'E2', 'F5', scale_factor=0.9)
        self.play(FadeIn(n_space))
        self.play(self.lecture[2].animate.set_color("#FF69B4"))
        self.wait(2)
