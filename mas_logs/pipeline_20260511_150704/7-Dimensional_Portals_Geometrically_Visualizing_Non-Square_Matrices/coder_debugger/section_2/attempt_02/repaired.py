from manim import *
import numpy as np

class Section2Scene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        """Sets up the visual layout with a title and bullet points."""
        self.camera.background_color = "#000000"
        
        # Title at the top
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP, buff=0.5)
        self.add(self.title)

        # Lecture bullets on the left
        lecture_texts = [Text(line, font_size=20, color=WHITE) for line in lecture_lines]
        self.lecture_group = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        self.lecture_group.scale(0.8).to_edge(LEFT, buff=0.7)
        self.add(self.lecture_group)

        # Coordinate grid for positioning elements on the right half
        self.grid_coords = {}
        rows = ["A", "B", "C", "D", "E"]
        cols = ["1", "2", "3", "4"]
        
        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                # Offset to the right side of the screen
                pos_x = 1.5 + (j * 1.5)
                pos_y = 2.0 - (i * 1.2)
                self.grid_coords[f"{row}{col}"] = np.array([pos_x, pos_y, 0])

    def place_at_grid(self, mobject, grid_key, scale=1.0):
        """Utility to move mobjects to predefined grid positions."""
        mobject.scale(scale)
        mobject.move_to(self.grid_coords[grid_key])
        return mobject

    def construct(self):
        # Configuration for the specific lecture section
        title = "Geometrically Visualizing Non-Square Matrices"
        bullets = [
            "- Understanding Dimensional Portals",
            "- Transformation between R^n and R^m",
            "- Tall vs. Wide Matrices",
            "- Projection and Embedding"
        ]
        
        self.setup_layout(title, bullets)

        # Visual elements representing vector space transformations
        input_space = Circle(radius=0.7, color=BLUE, fill_opacity=0.2)
        arrow = Arrow(start=LEFT, end=RIGHT, color=YELLOW)
        output_space = Square(side_length=1.4, color=RED, fill_opacity=0.2)

        # Positioning using the grid system
        self.place_at_grid(input_space, "A1")
        self.place_at_grid(arrow, "A2")
        self.place_at_grid(output_space, "A3")

        # Animating the concepts
        self.play(
            Create(input_space),
            Write(Text("Input (R^n)", font_size=16).next_to(input_space, DOWN)),
            run_time=1.5
        )
        
        self.play(GrowArrow(arrow), run_time=1)
        
        self.play(
            Create(output_space),
            Write(Text("Output (R^m)", font_size=16).next_to(output_space, DOWN)),
            run_time=1.5
        )

        # Demonstration of a non-square matrix as a portal
        # Fix: Replaced MathTex with Text to avoid FileNotFoundError: 'latex'
        matrix_label = Text("A = [[a, b], [c, d], [e, f]]", font_size=24, color=GREEN)
        self.place_at_grid(matrix_label, "C2")
        
        self.play(FadeIn(matrix_label, shift=UP))
        
        # Highlight a point moving through the "portal"
        dot = Dot(point=self.grid_coords["A1"], color=WHITE)
        self.add(dot)
        # slow_into is a valid rate function in Manim CE
        self.play(dot.animate.move_to(self.grid_coords["A3"]), run_time=2, rate_func=slow_into)
        
        self.wait(3)