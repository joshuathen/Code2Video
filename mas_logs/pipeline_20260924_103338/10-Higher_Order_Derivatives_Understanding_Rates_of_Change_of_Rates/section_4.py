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
            "The second derivative tells us about curvature.",
            "Positive values indicate concave-up graphs.",
            "Negative values mean the graph is concave-down."
        ]
        self.setup_layout("Visualizing Curvature", lecture_lines)

        # Assets
        ball = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ball.svg")
        f_double_prime_pos = MathTex("f''(x) > 0", color="#00CED1")
        f_double_prime_neg = MathTex("f''(x) < 0", color="#FF6347")

        # Concave Up Curve (cyan)
        curve_up = FunctionGraph(lambda x: x**2, x_range=[-1.5, 1.5], color="#00CED1")
        self.place_in_area(curve_up, "A4", "C6", scale_factor=0.5)

        # Concave Down Curve (red)
        curve_down = FunctionGraph(lambda x: -x**2, x_range=[-1.5, 1.5], color="#FF6347")
        self.place_in_area(curve_down, "D4", "F6", scale_factor=0.5)

        # Labeling (positioned at B5 and E5)
        self.place_at_grid(f_double_prime_pos, "B5", scale_factor=0.7)
        self.place_at_grid(f_double_prime_neg, "E5", scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00CED1"))
        self.play(Create(curve_up))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF6347"))
        self.play(Create(curve_down))
        self.place_at_grid(ball, "D5", scale_factor=0.5) # Force object place
        self.play(FadeIn(ball))
        self.wait(1)
