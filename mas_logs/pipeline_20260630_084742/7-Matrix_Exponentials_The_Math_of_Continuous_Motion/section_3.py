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

class Section3Scene(Scene):
    def construct(self):
        # Setup the layout
        self.setup_layout(
            "Matrix Exponentials",
            [
                "- Continuous Motion",
                "- Linear ODEs",
                "- State Transition",
                "- Scaling & Squaring"
            ]
        )
        
        # Displaying a mathematical representation
        # Replaced MathTex with Text to avoid FileNotFoundError: 'latex'
        matrix_eq = Text(
            "e^At = Σ (At)^k / k!",
            font_size=36
        )
        self.place_at_grid(matrix_eq, "B4", scale_factor=1.0)
        
        # Displaying a vector transformation example
        vector_box = Rectangle(height=2, width=2, color=BLUE)
        self.place_at_grid(vector_box, "D4", scale_factor=0.8)
        
        vector = Arrow(start=ORIGIN, end=RIGHT, buff=0, color=YELLOW)
        vector.move_to(vector_box.get_center())
        
        self.play(Write(self.title), FadeIn(self.lecture))
        self.play(Write(matrix_eq))
        self.play(Create(vector_box), GrowArrow(vector))
        
        # Simple animation of rotation (representing continuous motion)
        self.play(Rotate(vector, angle=TAU, about_point=vector_box.get_center()), run_time=3)
        self.wait(2)

    def setup_layout(self, title_text, lecture_lines):
        # Background and Title
        self.title = Text(title_text, font_size=44).to_edge(UP, buff=0.5)
        self.lecture = VGroup(*[
            Text(line, font_size=24) for line in lecture_lines
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.3).to_edge(LEFT, buff=1).shift(UP * 0.5)

    def place_at_grid(self, mob, grid_pos, scale_factor=1.0):
        # Manual grid mapping for placement
        cols = {"A": -5.0, "B": -2.5, "C": 0.0, "D": 2.5, "E": 5.0}
        rows = {"1": 3.0, "2": 1.5, "3": 0.0, "4": -1.5, "5": -3.0}

        col_char = grid_pos[0].upper()
        row_char = grid_pos[1:]

        x = cols.get(col_char, 0)
        y = rows.get(row_char, 0)

        mob.scale(scale_factor)
        mob.move_to([x, y, 0])
