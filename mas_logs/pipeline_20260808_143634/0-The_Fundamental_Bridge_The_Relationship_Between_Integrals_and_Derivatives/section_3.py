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
            "Calculus relates these two concepts fundamentally.",
            "Differentiation and integration are inverse processes.",
            "Think of them as reverse operations.",
            "Slice functions into thin rectangles.",
            "Compress them back into slopes."
        ]
        self.setup_layout("The Fundamental Theorem of Calculus", lecture_lines)

        # Visuals
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 4, 1], axis_config={"include_tip": False}).scale(0.5)
        curve = axes.plot(lambda x: 0.2*x**2 + 1, x_range=[0, 4], color=WHITE)
        f_prime = MathTex("f'(x)", color="#FF5733")
        integral_sign = MathTex(r"\int", color="#33FF57")
        arrow = Arrow(start=LEFT, end=RIGHT, color=WHITE)
        knife = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/knife.svg")

        # Position elements
        self.place_in_area(axes, 'B3', 'E5', scale_factor=0.9)
        self.place_in_area(curve, 'B3', 'E5', scale_factor=0.9)
        self.place_at_grid(f_prime, 'C5', scale_factor=1.0)
        self.place_at_grid(integral_sign, 'E2', scale_factor=0.7)
        self.place_at_grid(arrow, 'E3', scale_factor=0.8)
        self.place_at_grid(knife, 'D3', scale_factor=0.2)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(axes), Create(curve))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(f_prime), FadeIn(integral_sign))
        self.lecture[1].set_color(YELLOW)

        # === Animation for Lecture Line 3 ===
        self.play(GrowArrow(arrow))
        self.lecture[2].set_color(YELLOW)

        # === Animation for Lecture Line 4 ===
        rects = VGroup(*[Rectangle(height=axes.c2p(0, 0.2*x**2+1)[1], width=0.1, color=BLUE, fill_opacity=0.5) for x in range(4)])
        self.play(FadeIn(rects), FadeIn(knife))
        self.lecture[3].set_color(YELLOW)

        # === Animation for Lecture Line 5 ===
        self.play(FadeOut(rects), FadeOut(knife))
        self.lecture[4].set_color(YELLOW)
