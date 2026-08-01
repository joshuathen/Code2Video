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
        # Initialize layout with specified title and lecture lines
        self.setup_layout(
            "The 'Divisibility' Challenge", 
            [
                'What if we only want specific subset sizes?', 
                'Imagine a sieve keeping only every k-th term.', 
                'How do we sum specific coefficients efficiently?'
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Use Text with Unicode subscripts to avoid FileNotFoundError: 'latex'
        subscripts = ["₀", "₁", "₂", "₃", "₄", "₅", "₆"]
        coeffs = VGroup(*[Text(f"a{subscripts[i]}", color=WHITE, font_size=36) for i in range(7)])
        coeffs.arrange(RIGHT, buff=0.4)
        # Improved vertical balance by moving to Row C and setting scale to 1.0 (Issue 43/33)
        self.place_in_area(coeffs, "C1", "C6", scale_factor=1.0)
        
        self.play(
            Write(coeffs),
            self.lecture[0].animate.set_color(WHITE)
        )
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # Dim out a_1, a_2, a_4, a_5 with a #555555 opacity filter.
        dim_indices = [1, 2, 4, 5]
        dim_color = "#555555"
        
        dim_anims = [coeffs[i].animate.set_color(dim_color) for i in dim_indices]
        
        self.play(
            *dim_anims,
            self.lecture[1].animate.set_color(dim_color)
        )
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # Brighten a_0, a_3, a_6 in #FFFF00 and group them with '+' signs.
        highlight_color = "#FFFF00"
        keep_indices = [0, 3, 6]
        
        bright_anims = [coeffs[i].animate.set_color(highlight_color) for i in keep_indices]
        
        # Creating the sum representation using Text to avoid 'latex' dependency
        plus_sign1 = Text("+", color=highlight_color, font_size=36)
        plus_sign2 = Text("+", color=highlight_color, font_size=36)
        
        sum_terms = VGroup(
            coeffs[0].copy().set_color(highlight_color),
            plus_sign1,
            coeffs[3].copy().set_color(highlight_color),
            plus_sign2,
            coeffs[6].copy().set_color(highlight_color)
        ).arrange(RIGHT, buff=0.3)
        
        # Fix scale mismatch and improve vertical placement (Issue 43/32)
        self.place_in_area(sum_terms, "E1", "E6", scale_factor=1.0)
        
        self.play(
            *bright_anims,
            FadeIn(sum_terms, shift=UP * 0.3),
            self.lecture[2].animate.set_color(highlight_color)
        )
        self.wait(2)
