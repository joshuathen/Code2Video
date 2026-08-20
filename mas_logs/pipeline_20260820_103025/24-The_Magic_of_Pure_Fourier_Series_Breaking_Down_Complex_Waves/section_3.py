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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Fourier Series Formula", ["The formula sums sine and cosine terms.", "Coefficients define the weight of each.", "Think of a digital equalizer."])
        
        # === Animation for Lecture Line 1 ===
        # The formula sums sine and cosine terms.
        fourier_formula = MathTex(
            "f(x) = \\frac{a_0}{2} + \\sum_{n=1}^{\\infty} (a_n \\cos(nx) + b_n \\sin(nx))",
            color=WHITE
        )
        # Apply fix for issue 32
        self.place_in_area(fourier_formula, "A2", "B5", scale_factor=0.7)
        self.play(Write(fourier_formula))
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # === Animation for Lecture Line 2 ===
        # Coefficients define the weight of each.
        a_n = fourier_formula.get_part_by_tex("a_n")
        b_n = fourier_formula.get_part_by_tex("b_n")
        
        # Yellow: #FFFF00, Orange: #FF8C00
        self.play(
            a_n.animate.set_color("#FFFF00"),
            b_n.animate.set_color("#FF8C00"),
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # === Animation for Lecture Line 3 ===
        # Think of a digital equalizer.
        # Use SVGMobject for equalizer icon [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/equalizer.svg]
        equalizer_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/equalizer.svg")
        equalizer_icon.set_color("#00CED1")
        
        # Apply fix for issues 33 and 34 (34 refined positioning)
        self.place_in_area(equalizer_icon, "E2", "F5", scale_factor=0.8)
        
        self.play(FadeIn(equalizer_icon))
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        self.wait(1)
