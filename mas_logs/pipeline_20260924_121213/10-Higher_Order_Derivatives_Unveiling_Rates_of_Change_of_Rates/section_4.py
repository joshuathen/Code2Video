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
        lecture_lines = [
            "Differentiate powers repeatedly using the power rule.",
            "Function f of x equals x to fourth.",
            "First derivative is four x cubed.",
            "Second derivative is twelve x squared.",
            "Third derivative is twenty-four x."
        ]
        self.setup_layout("Computational Step-by-Step", lecture_lines)

        # Colors
        c_main = "#4DB6AC"
        c_highlight = "#FFB74D"

        # Math objects
        f0 = MathTex("f(x) = x^4", color=c_main)
        f1 = MathTex("f'(x) = 4x^3", color=c_main)
        f2 = MathTex("f''(x) = 12x^2", color=c_main)
        f3 = MathTex("f'''(x) = 24x", color=c_main)
        
        # Grouping for area placement
        group_formulas = VGroup(f0, f1, f2, f3).arrange(DOWN, aligned_edge=LEFT)

        # Apply positioning constraints
        self.place_at_grid(f0, 'B6', scale_factor=0.5)
        self.place_in_area(group_formulas, 'B3', 'E5', scale_factor=0.6)
        self.place_at_grid(f3, 'E6', scale_factor=0.5)

        # Hide initially
        f0.set_opacity(0)
        f1.set_opacity(0)
        f2.set_opacity(0)
        f3.set_opacity(0)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(c_highlight))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(c_highlight), FadeIn(f0))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(c_highlight),
            FadeIn(f1)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(c_highlight),
            FadeIn(f2)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(c_highlight),
            FadeIn(f3)
        )
        self.wait(2)
