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
        lecture_lines = [
            "The formula uses the function and its derivative.",
            "Each step calculates the next, closer x value.",
            "Shallow slopes mean large, bold steps."
        ]
        self.setup_layout("The Mathematical Engine: Newton’s Iteration", lecture_lines)
        
        # Objects
        formula = MathTex("x_{n+1} = x_n - \\frac{f(x_n)}{f'(x_n)}", font_size=36)
        
        axes = Axes(x_range=[-2, 2, 1], y_range=[-1, 3, 1], x_length=4, y_length=3)
        func = axes.plot(lambda x: x**2 + 0.5, color=BLUE)
        ball = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ball.svg")
        
        formula_and_graph_group = VGroup(formula, axes, func, ball)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFF00")
        self.place_at_grid(formula, "B4", scale_factor=0.9)
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFF00")
        self.place_in_area(ball, "C2", "E5", scale_factor=0.8)
        self.play(Create(axes), Create(func))
        self.play(FadeIn(ball))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FF00")
        # Animate ball movement representing Newton iteration steps
        # Starting point
        start_pos = axes.c2p(1.5, 2.75)
        ball.move_to(start_pos)
        
        # Step updates
        target1 = axes.c2p(0.8, 1.14)
        target2 = axes.c2p(0.2, 0.54)
        
        self.play(ball.animate.move_to(target1), run_time=1.5)
        self.play(ball.animate.move_to(target2), run_time=1.5)
        self.wait(1)
