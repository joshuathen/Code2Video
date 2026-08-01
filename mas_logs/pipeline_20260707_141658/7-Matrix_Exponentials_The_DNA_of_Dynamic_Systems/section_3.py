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
            "Define the matrix exponential using the infinite Taylor sum.",
            "Each term adds a power of matrix A.",
            "Factorials ensure the infinite series always converges.",
            "The result is always a matrix of the same size.",
            "This matrix captures the system's entire growth logic."
        ]
        self.setup_layout("The Formal Definition of e^A", lecture_lines)

        # Colors
        COLOR_WHITE = "#FFFFFF"
        COLOR_GOLD = "#FFD700"
        COLOR_EMERALD = "#50C878"
        HIGHLIGHT_COLOR = "#FFFF00" # Bright yellow for current line highlight

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(HIGHLIGHT_COLOR))
        
        # Using Text for formula parts to avoid MathTex complexity
        # e^A = Σ (A^n / n!)
        formal_def = VGroup(
            Text("e", font_size=32),
            Text("A", font_size=20).shift(UP*0.2 + RIGHT*0.2),
            Text(" = Σ ", font_size=32),
            Text("A", font_size=26),
            Text("n", font_size=18).shift(UP*0.2 + RIGHT*0.1),
            Text(" / n!", font_size=26)
        ).set_color(COLOR_WHITE)
        # Fix alignment manually since they are just Text mobjects
        formal_def[1].next_to(formal_def[0], UR, buff=0.05)
        formal_def[2].next_to(formal_def[1], RIGHT, buff=0.1)
        formal_def[3].next_to(formal_def[2], RIGHT, buff=0.1)
        formal_def[4].next_to(formal_def[3], UR, buff=0.05)
        formal_def[5].next_to(formal_def[4], RIGHT, buff=0.05)
        
        self.place_in_area(formal_def, "A2", "A5", scale_factor=1.2)
        self.play(Write(formal_def))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(HIGHLIGHT_COLOR)
        )
        
        # Expanded sum: I + A + A^2/2! + ...
        # e^A = I + A + A^2/2! + ...
        expanded_sum = VGroup(
            Text("e^A =", font_size=28),
            Text("I", font_size=28),
            Text("+", font_size=28),
            Text("A", font_size=28),
            Text("+", font_size=28),
            Text("A^2/2!", font_size=28),
            Text("+ ...", font_size=28)
        ).arrange(RIGHT, buff=0.2).set_color(COLOR_WHITE)
        
        # Issue 28: Fix placement to avoid left-side occlusion
        self.place_in_area(expanded_sum, "B2", "B5", scale_factor=0.8)
        
        # Animate terms sequentially
        self.play(FadeIn(expanded_sum[0]))
        self.wait(0.3)
        self.play(FadeIn(expanded_sum[1])) # I
        self.wait(0.3)
        self.play(FadeIn(expanded_sum[2:4])) # + A
        self.wait(0.3)
        self.play(FadeIn(expanded_sum[4:6])) # + A^2/2!
        self.wait(0.3)
        self.play(FadeIn(expanded_sum[6])) # + ...
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(HIGHLIGHT_COLOR)
        )
        
        # Highlight factorial part of expanded_sum[5] ("A^2/2!")
        # Since it's a single Text object "A^2/2!", we can't easily color just "2!" without splitting.
        # Let's recreate it for the highlight.
        fact_part = Text("2!", font_size=28, color=COLOR_GOLD)
        fact_part.move_to(expanded_sum[5].get_right(), aligned_edge=RIGHT).shift(LEFT*0.1)
        
        self.play(expanded_sum[5][-2:].animate.set_color(COLOR_GOLD))
        
        converge_text = Text("Converges for any matrix A", font_size=22, color=COLOR_GOLD)
        # Issue 29: Fix centering/misalignment
        self.place_in_area(converge_text, "C2", "C5", scale_factor=0.8)
        self.play(Write(converge_text))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(HIGHLIGHT_COLOR)
        )
        
        # 2x2 Matrix Structure
        m_vals = VGroup(
            Text("m11", font_size=24), Text("m12", font_size=24),
            Text("m21", font_size=24), Text("m22", font_size=24)
        ).set_color(COLOR_WHITE)
        
        m_grid = VGroup(
            VGroup(m_vals[0], m_vals[1]).arrange(RIGHT, buff=0.6),
            VGroup(m_vals[2], m_vals[3]).arrange(RIGHT, buff=0.6)
        ).arrange(DOWN, buff=0.5)
        
        l_bracket = Text("[", font_size=60).next_to(m_grid, LEFT, buff=0.2)
        r_bracket = Text("]", font_size=60).next_to(m_grid, RIGHT, buff=0.2)
        
        matrix_2x2 = VGroup(l_bracket, m_grid, r_bracket)
        self.place_in_area(matrix_2x2, "D2", "E5", scale_factor=0.8)
        
        arrow = Arrow(start=expanded_sum.get_bottom(), end=matrix_2x2.get_top(), buff=0.2, color=COLOR_WHITE)
        
        self.play(Create(arrow), FadeIn(matrix_2x2))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(HIGHLIGHT_COLOR)
        )
        
        # Pulse the final matrix in Emerald Green
        self.play(matrix_2x2.animate.set_color(COLOR_EMERALD))
        self.play(
            matrix_2x2.animate.scale(1.15),
            rate_func=there_and_back,
            run_time=1.5
        )
        
        system_logic_label = Text("System Dynamics DNA", font_size=24, color=COLOR_EMERALD)
        # Issue 30: Fix centering/occlusion
        self.place_in_area(system_logic_label, "F2", "F5", scale_factor=0.8)
        self.play(Write(system_logic_label))
        
        self.wait(2)
