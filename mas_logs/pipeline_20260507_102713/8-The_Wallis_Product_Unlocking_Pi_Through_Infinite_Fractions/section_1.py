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

class Section1Scene(TeachingScene):
    def construct(self):
        # Initialize layout
        lecture_lines_text = [
            "Pi relates a circle's circumference to its diameter.", 
            "It is usually viewed as a geometric constant.", 
            "Surprisingly, Pi also equals an infinite rational product.", 
            "John Wallis found this stunning link through calculus.", 
            "Let's derive this elegant product from sine integrals."
        ]
        self.setup_layout("Introduction: The Geometry-Algebra Bridge", lecture_lines_text)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        circle = Circle(radius=0.5, color=WHITE)
        self.place_in_area(circle, "B2", "C4", scale_factor=1.0)
        
        radius_line = Line(circle.get_center(), circle.get_right(), color=YELLOW)
        # Replaced MathTex with Text to avoid FileNotFoundError: 'latex'
        pi_def = Text("π = C / d", color=WHITE)
        self.place_at_grid(pi_def, "A3", scale_factor=0.8)

        self.play(Create(circle))
        self.play(Create(radius_line))
        self.play(FadeIn(pi_def))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Line segment of length pi (radius was 0.5, so C = 2 * pi * 0.5 = pi)
        straight_line = Line(LEFT * PI / 2, RIGHT * PI / 2, color=WHITE)
        self.place_in_area(straight_line, "B1", "B6", scale_factor=1.0)
        
        self.play(
            FadeOut(radius_line),
            FadeOut(pi_def),
            ReplacementTransform(circle, straight_line)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Replaced MathTex with VGroup of Text objects to maintain indexing while avoiding LaTeX dependency
        wallis_formula = VGroup(
            Text("π/2 =", color="#ADD8E6"),
            Text("( 2/1 · 2/3 )", color="#ADD8E6"),
            Text("·", color="#ADD8E6"),
            Text("( 4/3 · 4/5 )", color="#ADD8E6"),
            Text("·", color="#ADD8E6"),
            Text("( 6/5 · 6/7 )", color="#ADD8E6"),
            Text("...", color="#ADD8E6")
        ).arrange(RIGHT, buff=0.1)

        # Apply Issue 22 Fix: Positioning in area C1-D6
        self.place_in_area(wallis_formula, 'C1', 'D6', scale_factor=0.6)
        
        self.play(FadeIn(wallis_formula))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Flashing the first three pairs in yellow
        pairs_indices = [1, 3, 5]
        flash_anims = []
        for idx in pairs_indices:
            flash_anims.append(wallis_formula[idx].animate.set_color(YELLOW))
        
        self.play(*flash_anims)
        self.play(*[wallis_formula[idx].animate.set_color("#ADD8E6") for idx in pairs_indices])
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Align line and formula, draw dashed line to pi/2
        pi_half_point = straight_line.get_center()
        # The start of the formula is wallis_formula[0] which is "pi/2 ="
        target_point = wallis_formula[0].get_top()
        
        connecting_dash = DashedLine(
            start=pi_half_point,
            end=target_point,
            color="#ADD8E6"
        )
        
        # Replaced MathTex with Text
        label_pi_half = Text("π/2", color=WHITE, font_size=20)
        label_pi_half.next_to(pi_half_point, UP, buff=0.1)

        self.play(
            FadeIn(connecting_dash),
            FadeIn(label_pi_half)
        )
        self.wait(2)
