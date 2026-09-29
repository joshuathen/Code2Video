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
        lecture_lines = [
            "Fractal dimension bridges the gap between integers.",
            "We use logarithms to measure scaling behavior.",
            "The formula quantifies space-filling capacity.",
            "Koch snowflakes demonstrate this fractional dimension.",
            "Dimension is a measure of roughness."
        ]
        self.setup_layout("Defining the Fractal Dimension (Hausdorff)", lecture_lines)
        
        # Assets
        snowflake = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/snowflake.svg")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(GOLD)
        formula = MathTex(r"D = \frac{\log(N)}{\log(1/r)}", font_size=42, color=GOLD)
        self.place_in_area(formula, 'B4', 'C6', scale_factor=0.9)
        self.play(Write(formula))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(BLUE)
        n_label = MathTex("N", color=BLUE)
        self.place_at_grid(n_label, 'C3', scale_factor=0.7)
        self.play(FadeIn(n_label))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(RED)
        r_label = MathTex("r", color=RED)
        self.place_at_grid(r_label, 'E3', scale_factor=0.7)
        self.play(FadeIn(r_label))
        
        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(GREEN)
        self.place_at_grid(snowflake, 'E5', scale_factor=0.6)
        snowflake_val = MathTex("N=4, r=1/3", font_size=32, color=GOLD)
        self.place_at_grid(snowflake_val, 'F5', scale_factor=0.6)
        self.play(FadeIn(snowflake), Write(snowflake_val))
        
        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(PURPLE)
        d_val = MathTex(r"D \approx 1.26", font_size=40, color=PURPLE)
        self.place_at_grid(d_val, 'D5', scale_factor=0.8)
        target = Dot(color=GREEN)
        self.place_at_grid(target, 'D5', scale_factor=1.0)
        self.play(Write(d_val), FadeIn(target))
        self.wait(2)
