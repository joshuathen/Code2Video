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
            "Integrating over the radius yields the π factor.",
            "Final integral evaluation equals the square root of π.",
            "Link radius change to circumference."
        ]
        self.setup_layout("Solving the Integral: Integrating out the π", lecture_lines)
        
        # Assets
        scanner = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/scanner.svg")
        
        # Initial math objects
        integral_1 = MathTex(r"\int e^{-x^2} dx", color="#FFFF00")
        integral_sq = MathTex(r"I^2 = \iint e^{-(x^2+y^2)} dx dy", color="#00FFFF")
        integral_polar = MathTex(r"I^2 = \int_0^{2\pi} d\theta \int_0^\infty e^{-r^2} r dr", color="#FF00FF")
        result = MathTex(r"I = \sqrt{\pi}", color="#FF0000")

        # === Animation for Lecture Line 1 ===
        self.place_at_grid(scanner, 'A1', scale_factor=0.5)
        self.play(FadeIn(scanner), Write(self.title))
        self.play(FadeIn(integral_1))
        self.place_at_grid(integral_1, 'A1')
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(Transform(integral_1, integral_sq))
        self.place_in_area(integral_1, 'B2', 'C5', scale_factor=0.8)
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        self.wait(1)

        # Transform to polar
        self.play(Transform(integral_1, integral_polar))
        self.place_in_area(integral_1, 'B2', 'C5', scale_factor=0.8)
        self.play(self.lecture[2].animate.set_color("#FF00FF"))
        self.wait(1)

        # Final result
        self.play(FadeOut(integral_1))
        self.place_at_grid(result, 'D4', scale_factor=0.9)
        self.play(FadeIn(result))
        self.play(scanner.animate.move_to(result.get_center()))
        self.play(self.lecture[2].animate.set_color("#FF0000"))
        self.wait(2)
