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
        # Lecture lines exactly as per snapshot script
        lecture_lines_text = [
            "Let's break down the binomial probability formula.",
            "n-choose-k counts all possible ways to arrange successes.",
            "p to the k represents the success probability.",
            "(1-p) to the (n-k) represents failure probability.",
            "Together, they calculate probability for exactly k successes."
        ]
        
        self.setup_layout("Deconstructing the Binomial Formula", lecture_lines_text)

        # Formula construction
        # We break the formula into parts to allow individual coloring
        # P(X=k) = nCk * p^k * (1-p)^(n-k)
        f_part1 = Text("P(X = k) = ", font_size=24, color=WHITE)
        f_part2 = Text("nCk", font_size=24, color=WHITE)
        f_part3 = Text(" * ", font_size=24, color=WHITE)
        f_part4 = Text("p^k", font_size=24, color=WHITE)
        f_part5 = Text(" * ", font_size=24, color=WHITE)
        f_part6 = Text("(1-p)^(n-k)", font_size=24, color=WHITE)
        
        formula = VGroup(f_part1, f_part2, f_part3, f_part4, f_part5, f_part6).arrange(RIGHT, buff=0.1)
        
        # Fix Issue 35: Formula in area B1-B6
        self.place_in_area(formula, "B1", "B6", scale_factor=1.2)
        
        # Labels for specific terms
        ways_label = Text("Ways to arrange", font_size=22, color=TEAL)
        prob_label = Text("Success probability", font_size=22, color=GREEN)
        fail_label = Text("Failure probability", font_size=22, color=RED)
        
        # Positioning according to Issue 36 (D2-D3) and Issue 37 (D4-D6)
        self.place_in_area(ways_label, "D2", "D3", scale_factor=0.8)
        self.place_in_area(prob_label, "D4", "D6", scale_factor=0.8)
        self.place_in_area(fail_label, "F1", "F6", scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        # "Let's break down the binomial probability formula."
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "n-choose-k counts all possible ways to arrange successes."
        # Term nCk turns teal labeled 'Ways to arrange'
        self.play(
            self.lecture[1].animate.set_color(TEAL),
            f_part2.animate.set_color(TEAL),
            FadeIn(ways_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "p to the k represents the success probability."
        # Term p^k turns green labeled 'Success probability'
        self.play(
            self.lecture[2].animate.set_color(GREEN),
            f_part4.animate.set_color(GREEN),
            FadeIn(prob_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "(1-p) to the (n-k) represents failure probability."
        # Term (1-p)^(n-k) turns red labeled 'Failure probability'
        self.play(
            self.lecture[3].animate.set_color(RED),
            f_part6.animate.set_color(RED),
            FadeIn(fail_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "Together, they calculate probability for exactly k successes."
        # Full formula pulses
        self.play(
            formula.animate.scale(1.1),
            self.lecture[4].animate.set_color(WHITE)
        )
        self.play(
            formula.animate.scale(1/1.1)
        )
        self.wait(3)
