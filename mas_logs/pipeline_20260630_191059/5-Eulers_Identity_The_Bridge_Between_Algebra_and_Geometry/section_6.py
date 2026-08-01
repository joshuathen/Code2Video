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

class Section6Scene(Scene):
    def construct(self):
        # Initialize the layout and grid system
        self.setup_layout(
            "Euler's Identity: The Ultimate Bridge", 
            [
                "- Complex Exponentiation", 
                "- Circular Motion", 
                "- Geometry meets Algebra", 
                "- Conceptual Summary"
            ]
        )
        
        # Define objects for the scene
        # Replaced MathTex with Text to avoid the FileNotFoundError: 'latex'
        formula = Text("e^iπ + 1 = 0", font_size=42, color=YELLOW)
        self.place_at_grid(formula, "B3", scale_factor=1.5)
        
        circle = Circle(radius=1.2, color=BLUE)
        self.place_at_grid(circle, "D3")
        
        dot = Dot(color=RED)
        self.place_at_grid(dot, "D3")
        
        # Animations
        self.play(Write(self.title))
        self.play(FadeIn(self.lecture, shift=RIGHT))
        self.wait(1)
        
        self.play(Write(formula))
        self.play(Create(circle))
        self.play(dot.animate.move_to(circle.point_at_angle(PI)), run_time=2)
        
        self.play(Indicate(formula))
        self.wait(2)

    def setup_layout(self, title_text, lecture_lines):
        # Background and Title
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP, buff=0.5)

        # Left-side lecture content
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        self.lecture.to_edge(LEFT, buff=0.7).scale(0.9)

        # Define animation grid (Right side focus)
        # Rows A-F (Top to Bottom), Cols 1-6 (Left to Right)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]
        cols = ["1", "2", "3", "4", "5", "6"]

        for i, row_label in enumerate(rows):
            for j, col_label in enumerate(cols):
                # Map grid to the right side of the screen (x > 0)
                x_pos = 1.5 + (j * 1.0)
                y_pos = 2.0 - (i * 1.0)
                self.grid[f"{row_label}{col_label}"] = np.array([x_pos, y_pos, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        """Helper to position mobjects based on the defined grid."""
        mobject.scale(scale_factor)
        if grid_pos in self.grid:
            mobject.move_to(self.grid[grid_pos])
        else:
            # Default to center-right if grid key is invalid
            mobject.move_to(RIGHT * 3)
        return mobject
