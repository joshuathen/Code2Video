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
            "Brachistochrone means the shortest time path.",
            "Straight lines are short, but slow.",
            "Curves allow faster speeds early on.",
            "Gravity rewards a steep initial drop.",
            "The curve balances distance and speed."
        ]
        self.setup_layout("Defining the Brachistochrone", lecture_lines)

        # Assets
        marble = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/marble.svg")
        coaster = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rollercoaster.svg")

        # Define points A and B
        dot_a = Dot(color=BLUE)
        dot_b = Dot(color=GREEN)
        self.place_at_grid(dot_a, 'A4', scale_factor=0.7)
        self.place_at_grid(dot_b, 'F5', scale_factor=0.7)
        label_a = Text("A", font_size=20).next_to(dot_a, UP)
        label_b = Text("B", font_size=20).next_to(dot_b, RIGHT)
        pts = VGroup(dot_a, dot_b, label_a, label_b)
        
        marble.move_to(dot_a.get_center()).scale(0.3)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(pts), FadeIn(marble))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        line = Line(dot_a.get_center(), dot_b.get_center(), color=WHITE)
        self.play(Create(line))
        self.lecture[1].set_color(YELLOW)

        # === Animation for Lecture Line 3 ===
        curve = ArcBetweenPoints(dot_a.get_center(), dot_b.get_center(), angle=-TAU/6, color=RED)
        self.place_in_area(curve, 'A3', 'D6', scale_factor=0.9)
        self.play(Create(curve))
        self.lecture[2].set_color(YELLOW)

        # === Animation for Lecture Line 4 ===
        formula = MathTex(r"T = \int \frac{ds}{v}", color=WHITE)
        self.place_at_grid(formula, 'D3', scale_factor=1.0)
        self.play(Write(formula))
        self.lecture[3].set_color(YELLOW)

        # === Animation for Lecture Line 5 ===
        v_var = formula[0][6]
        coaster.move_to(dot_b.get_center()).scale(0.4)
        self.play(v_var.animate.set_color(RED), FadeIn(coaster))
        self.lecture[4].set_color(YELLOW)
        self.wait(2)
