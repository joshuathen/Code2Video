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
        self.setup_layout("The Wallis Formula Revealed", [
            "Pi over two is a product.", 
            "Numerators and denominators form patterns.", 
            "Each step brings more precision."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Pi/2 = (2/1) * (2/3) * (4/3) * (4/5) * (6/5) * (6/7)
        formula = MathTex(
            r"{\pi \over 2} = \left({2 \over 1}\right) \cdot \left({2 \over 3}\right) \cdot \left({4 \over 3}\right) \cdot \left({4 \over 5}\right) \cdot \left({6 \over 5}\right) \cdot \left({6 \over 7}\right) \cdots",
            font_size=28
        )
        self.place_in_area(formula, 'B2', 'C5', scale_factor=1.0)
        self.play(Write(formula))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        # Highlight pattern
        self.lecture[1].set_color("#00FFFF")
        self.play(Indicate(formula))

        # === Animation for Lecture Line 3 ===
        # Limit as n approaches infinity
        self.lecture[2].set_color("#FF00FF")
        limit_text = MathTex(
            r"\lim_{n \to \infty} \prod_{k=1}^{n} \left( \frac{2k}{2k-1} \cdot \frac{2k}{2k+1} \right) = \frac{\pi}{2}",
            font_size=30
        )
        self.place_in_area(limit_text, 'E2', 'F5', scale_factor=0.9)
        self.play(FadeIn(limit_text))
        self.wait(2)
