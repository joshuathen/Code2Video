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
        self.setup_layout("The Rule of Summation", [
            "Adding two normal variables is a new normal.",
            "The mean of the sum is the sum of means.",
            "The variance of the sum is the sum of variances.",
            "The combined distribution is a new bell curve.",
            "This rule applies to all independent Gaussian distributions."
        ])

        # Colors
        COLOR_A = "#FF5733"
        COLOR_B = "#3357FF"
        COLOR_SUM = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_SUM)
        formula = MathTex(r"S = X + Y", color=COLOR_SUM)
        self.place_at_grid(formula, "B5", scale_factor=1.0)
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(COLOR_SUM)
        # Using SVG Assets
        icon_a = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/dots.svg", color=COLOR_A)
        group_a = VGroup(*[icon_a.copy().scale(0.2) for _ in range(5)])
        self.place_in_area(group_a, "D4", "E5", scale_factor=1.0)
        self.play(FadeIn(group_a))
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(COLOR_SUM)
        icon_b = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/squares.svg", color=COLOR_B)
        group_b = VGroup(*[icon_b.copy().scale(0.2) for _ in range(5)])
        self.place_in_area(group_b, "D5", "E6", scale_factor=1.0)
        self.play(FadeIn(group_b))
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(COLOR_SUM)
        combined = VGroup(group_a.copy(), group_b.copy())
        self.place_in_area(combined, "C4", "E6", scale_factor=0.8)
        self.play(FadeIn(combined))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(COLOR_SUM)
        self.play(FadeOut(formula), FadeOut(group_a), FadeOut(group_b), FadeOut(combined))
        conclusion = Text("Independent Gaussians Sum!", font_size=24, color=COLOR_SUM)
        self.place_at_grid(conclusion, "D3", scale_factor=0.9)
        self.play(Write(conclusion))
        self.wait(2)
