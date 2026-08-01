from manim import *
import numpy as np

class Section5Scene(Scene):
    def construct(self):
        # Sample content to demonstrate the functionality
        title_text = "Section 5: Advanced Layouts"
        lecture_lines = [
            "1. Coordinate Mapping",
            "2. Grid-based Placement",
            "3. Dynamic Transformations",
            "4. Content Alignment"
        ]
        
        # Initialize layout
        self.setup_layout(title_text, lecture_lines)

        # Example: Placing mobjects using the defined grid
        circle = Circle(radius=0.2, color=BLUE)
        self.place_at_grid(circle, "A1")
        
        square = Square(side_length=0.4, color=RED)
        self.place_at_grid(square, "C3")

        triangle = Triangle(color=GREEN).scale(0.2)
        self.place_at_grid(triangle, "F6")

        self.play(Create(circle))
        self.play(FadeIn(square))
        self.play(DrawBorderThenFill(triangle))
        self.wait(2)

    def setup_layout(self, title_text, lecture_lines):
        # Background and Title
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        self.lecture.scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.5)
        self.add(self.lecture)

        # Fine-grained animation grid (mapping for the right side of the screen)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                # Offset to the right half of the screen
                x = 1.0 + j * 0.9
                y = 2.0 - i * 0.8
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        """Places a mobject at a specific grid coordinate defined in setup_layout."""
        if grid_pos in self.grid:
            mobject.scale(scale_factor).move_to(self.grid[grid_pos])
        return mobject