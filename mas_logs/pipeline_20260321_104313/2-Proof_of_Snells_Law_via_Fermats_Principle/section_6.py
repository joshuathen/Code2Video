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

class Section6Scene(TeachingScene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        if lecture_texts:
            self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT, buff=0.3).scale(0.8)
            self.lecture.to_edge(LEFT, buff=0.5)
            self.add(self.lecture)

        # Define fine-grained animation grid (6x6 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]
        cols = ["1", "2", "3", "4", "5", "6"]

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 1.0 + j * 0.9
                y = 2.0 - i * 0.8
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        if hasattr(self, 'grid') and grid_pos in self.grid:
            mobject.move_to(self.grid[grid_pos])
        return mobject

    def construct(self):
        # Sample implementation of the scene
        title_text = "Section 6: Grid Placement Logic"
        lecture_lines = [
            "1. Define the grid coordinates",
            "2. Map keys to 3D vectors",
            "3. Position mobjects relative to keys",
            "4. Maintain layout consistency"
        ]
        
        self.setup_layout(title_text, lecture_lines)

        # Example demonstration of grid placement
        circle = Circle(radius=0.3, color=BLUE)
        self.place_at_grid(circle, "B2")
        
        square = Square(side_length=0.5, color=RED)
        self.place_at_grid(square, "D4")
        
        triangle = Triangle(color=GREEN).scale(0.3)
        self.place_at_grid(triangle, "A6")

        self.play(Create(circle))
        self.play(FadeIn(square))
        self.play(DrawBorderThenFill(triangle))
        
        self.wait(2)
