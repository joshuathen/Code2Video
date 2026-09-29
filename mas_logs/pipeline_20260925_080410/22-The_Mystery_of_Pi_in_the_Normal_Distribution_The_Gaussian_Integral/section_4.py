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
        self.setup_layout("Closing the Loop: Solving the Integral", [
            "Calculate the new polar integral.",
            "Integration reveals a hidden π.",
            "So I squared equals π.",
            "Therefore I equals root π.",
            "The integral is solved."
        ])
        
        # Objects
        integral_expr = MathTex(r"\int_{0}^{2\pi} d\theta \int_{0}^{\infty} r e^{-r^2} dr")
        pi_const = MathTex(r"2\pi \cdot \frac{1}{2} = \pi")
        i_sq = MathTex(r"I^2 = \pi")
        i_result = MathTex(r"I = \sqrt{\pi}")
        final_text = Text("The integral is solved.", font_size=24)
        
        # Assets
        calc_icon1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg")
        calc_icon2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg")

        # === Animation for Lecture Line 1 ===
        self.place_in_area(integral_expr, 'A2', 'C5', scale_factor=0.9)
        self.place_at_grid(calc_icon1, 'B6', scale_factor=0.4)
        self.play(Write(integral_expr), FadeIn(calc_icon1))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        self.place_at_grid(pi_const, 'C3', scale_factor=0.8)
        self.play(Write(pi_const))
        pi_const.set_color("#FF4500")
        self.lecture[1].set_color("#FF4500")

        # === Animation for Lecture Line 3 ===
        self.place_in_area(i_sq, 'D2', 'D5', scale_factor=0.8)
        self.play(Write(i_sq))
        self.lecture[2].set_color("#FFD700")

        # === Animation for Lecture Line 4 ===
        self.place_in_area(i_result, 'E2', 'E5', scale_factor=0.8)
        self.place_at_grid(calc_icon2, 'E6', scale_factor=0.4)
        self.play(ReplacementTransform(i_sq.copy(), i_result), FadeIn(calc_icon2))
        i_result.set_color("#FFD700")
        self.lecture[3].set_color("#FFD700")

        # === Animation for Lecture Line 5 ===
        self.place_at_grid(final_text, 'F2', scale_factor=0.7)
        self.play(Write(final_text))
        self.lecture[4].set_color("#FFFFFF")
        self.wait(2)
