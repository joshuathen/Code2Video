from manim import *
import numpy as np

class Section5Scene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # Use Tex instead of Text to avoid Pango SVG ParseError in restricted environments
        self.title = Tex(title_text, font_size=28, color=WHITE)
        self.add(self.title)
        # Use Text instead of Tex to avoid dependency on a LaTeX installation ('latex' command not found)
        self.title = Text(title_text, font_size=28, color=WHITE)
        self.add(self.title)
        # Use Text instead of MarkupText to avoid a known Pango SVG ParseError
        self.title = Text(title_text, font_size=28, color=WHITE)
        self.add(self.title)
        # Use MarkupText instead of Text to bypass a known Pango SVG parsing issue with remove_last_M
        self.title = MarkupText(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        self.bullets = VGroup(*[
            Text(line, font_size=24, color=WHITE)
            for line in lecture_lines
        ]).arrange(DOWN, aligned_edge=LEFT).next_to(self.title, DOWN, buff=0.5).to_edge(LEFT)
        self.add(self.bullets)

    def construct(self):
        # Implementation of Section 5 content: Fourier Transform concepts
        self.setup_layout(
            "The Fourier Transform",
            [
                "- Signal Decomposition",
                "- Frequency Analysis",
                "- Wave DNA Decoding"
            ]
        )
        self.wait(2)
        self.camera.background_color = "#000000"
        
        # Use Text instead of Tex to avoid dependency on a LaTeX installation ('latex' command not found)
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.5)
        self.add(self.lecture)

        # Define fine-grained animation grid (6x6 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                # Adjusted coordinates to center on the right side of the screen
                x = 1.0 + j * 0.9
                y = 2.2 - i * 0.9
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])

    def construct(self):
        # Implementation of Section 5 content: Fourier Transform concepts
        self.setup_layout(
            "The Fourier Transform", 
            [
                "- Signal Decomposition",
                "- Frequency Analysis",
                "- Time-Domain vs Frequency-Domain"
            ]
        )
        
        # Visualize a frequency component using a circle and a rotating radius
        frequency_circle = Circle(radius=1.2, color=BLUE)
        self.place_at_grid(frequency_circle, "C3", scale_factor=0.8)
        
        center_dot = Dot(color=WHITE)
        center_dot.move_to(frequency_circle.get_center())
        
        radius_line = Line(
            frequency_circle.get_center(), 
            frequency_circle.get_right(), 
            color=YELLOW
        )
        
        # Label for the visual - Use Text instead of MathTex to avoid LaTeX requirement
        label = Text("e^(iωt)", font_size=24).next_to(frequency_circle, UP)
        
        # Animation
        self.play(Create(frequency_circle), FadeIn(center_dot), Write(label))
        self.add(radius_line)
        self.play(
            Rotate(radius_line, angle=2 * PI, about_point=frequency_circle.get_center()),
            run_time=3,
            rate_func=linear
        )
        self.wait(2)