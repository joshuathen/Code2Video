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

class Section4Scene(Scene):
    """
    Fixed Manim Scene for Section 4 focusing on Euler's Formula.
    Uses Text and VGroup instead of MathTex to avoid the FileNotFoundError: 'latex' 
    caused by a missing LaTeX distribution in the environment.
    """
    def construct(self):
        # 1. Setup Title and Background
        self.camera.background_color = "#000000"
        title = Text("Euler's Identity: The Most Beautiful Equation", font_size=32, color=WHITE)
        title.to_edge(UP, buff=0.5)
        
        line = Line(LEFT, RIGHT, color=BLUE).scale(5).next_to(title, DOWN)
        
        # 2. Equation Definition (using Text objects to bypass LaTeX dependency)
        # Components: e, i, pi, +, 1, =, 0
        e_part = Text("e", font_size=72, color=YELLOW)
        i_part = Text("i", font_size=44, color=GREEN)
        pi_part = Text("π", font_size=44, color=RED)
        plus_part = Text("+", font_size=72, color=WHITE)
        one_part = Text("1", font_size=72, color=BLUE)
        equal_part = Text("=", font_size=72, color=WHITE)
        zero_part = Text("0", font_size=72, color=WHITE)

        # Layout for the "e^(i*pi)" part
        exponent = VGroup(i_part, pi_part).arrange(RIGHT, buff=0.05)
        
        # Main base layout arrangement
        # We include e_part here to set the horizontal spacing
        equation_base = VGroup(e_part, plus_part, one_part, equal_part, zero_part).arrange(RIGHT, buff=0.4)
        
        # Position the exponent specifically above/right of the "e"
        exponent.next_to(e_part, UR, buff=-0.12).shift(UP * 0.1)
        
        # Group everything into a single entity
        equation = VGroup(equation_base, exponent)
        equation.move_to(ORIGIN)

        # 3. Component Descriptions
        descriptions = VGroup(
            Text("- e: Base of Natural Logarithms", font_size=20, color=YELLOW),
            Text("- i: Imaginary Unit (sqrt(-1))", font_size=20, color=GREEN),
            Text("- π: Ratio of Circle Circumference to Diameter", font_size=20, color=RED),
            Text("- 1 & 0: Foundations of Arithmetic", font_size=20, color=BLUE)
        ).arrange(DOWN, aligned_edge=LEFT).to_edge(LEFT, buff=1)

        # 4. Animations
        self.play(Write(title))
        self.play(Create(line))
        self.wait(0.5)
        
        self.play(Write(equation))
        self.play(equation.animate.shift(UP * 0.5))
        
        self.play(
            LaggedStart(
                *[FadeIn(desc, shift=RIGHT) for desc in descriptions],
                lag_ratio=0.5
            )
        )
        self.wait(2)

        # 5. Highlight the result
        box = SurroundingRectangle(equation, color=GOLD, buff=0.3)
        self.play(Create(box))
        self.play(Indicate(equation))
        
        self.wait(3)

    def setup_layout(self, title_text, lecture_lines):
        """
        Helper method for structured layout.
        """
        self.title_obj = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title_obj)

        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture_group = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture_group.to_edge(LEFT, buff=0.2)
        self.add(self.lecture_group)
