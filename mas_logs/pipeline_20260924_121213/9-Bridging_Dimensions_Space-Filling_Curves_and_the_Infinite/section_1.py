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
        self.setup_layout("The Dimensional Intuition", ["A line is inherently one-dimensional.", "A plane exists in two dimensions.", "Can a one-dimensional line fill space?"])
        
        # Elements
        line = Line(start=LEFT, end=RIGHT, color=WHITE)
        square = Square(side_length=2, color=WHITE)
        gaps = VGroup(
            Line(start=UP*1, end=DOWN*1, color=RED).shift(LEFT*0.5),
            Line(start=UP*1, end=DOWN*1, color=RED).shift(RIGHT*0.5)
        )
        label = Text("1D Line vs 2D Area", color="#00FFFF", font_size=24)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.place_at_grid(line, 'B2', scale_factor=0.8)
        self.play(Create(line))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFFFF"))
        self.place_at_grid(square, 'D2', scale_factor=0.9)
        self.play(Create(square))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        self.place_in_area(gaps, 'C3', 'E5', scale_factor=0.8)
        self.play(Create(gaps))
        self.place_in_area(label, 'A5', 'A6', scale_factor=0.7)
        self.play(Write(label))
        
        self.wait(2)
