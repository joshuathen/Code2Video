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
        # Initialization of layout with specific title and content
        self.setup_layout(
            "Matrix Exponentials: Continuous Motion", 
            [
                "- Discrete steps vs. Continuous flow", 
                "- Defining the matrix exponential e^At", 
                "- Solving linear differential equations",
                "- Mapping state trajectories"
            ]
        )
        
        # Define visual elements
        initial_state = Circle(color=BLUE, fill_opacity=0.6)
        transformed_state = Square(color=TEAL, fill_opacity=0.6)
        vector_indicator = Arrow(start=LEFT, end=RIGHT, color=YELLOW)
        
        # Placement using the grid system defined in setup_layout
        self.place_at_grid(initial_state, "B2", scale_factor=0.6)
        self.place_at_grid(transformed_state, "E5", scale_factor=0.6)
        self.place_at_grid(vector_indicator, "C4", scale_factor=0.8)
        
        # Animation sequence
        # Objects are already added in setup_layout, we play animations on them
        self.play(FadeIn(self.title))
        self.play(Write(self.lecture))
        self.wait(1)
        
        self.play(Create(initial_state))
        self.play(GrowArrow(vector_indicator))
        self.play(TransformFromCopy(initial_state, transformed_state))
        
        self.wait(3)
