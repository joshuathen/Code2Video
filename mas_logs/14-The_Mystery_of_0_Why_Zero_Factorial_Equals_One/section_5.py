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

class Section5Scene(TeachingScene):
    def construct(self):
        # Initial Setup
        title = "Conclusion: Why it Matters"
        lines = [
            "Zero factorial as one keeps mathematics consistent.",
            "It enables complex formulas in calculus and probability.",
            "Our mystery is solved, and Facto is satisfied."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        # Color line
        self.play(self.lecture[0].animate.set_color(WHITE))

        # Show algebraic consistency summary (Addressing VideoCritic issues)
        # formula_1: n! = n × (n - 1)!
        # formula_2: 1! = 1 × (1 - 0)!
        # formula_3: 1 = 1 × 0!
        # formula_4: 0! = 1
        formula_1 = Text("n! = n × (n - 1)!", color=WHITE, font_size=32)
        formula_2 = Text("1! = 1 × (1 - 0)!", color=WHITE, font_size=32)
        formula_3 = Text("1 = 1 × 0!", color=WHITE, font_size=32)
        formula_4 = Text("0! = 1", color=WHITE, font_size=32)

        # Placement based on Issue 37, 38, 39
        self.place_at_grid(formula_1, "B4", scale_factor=1.0)
        self.place_at_grid(formula_2, "C4", scale_factor=1.0)
        self.place_at_grid(formula_3, "D4", scale_factor=1.0)
        self.place_at_grid(formula_4, "E4", scale_factor=1.2)

        self.play(Write(formula_1))
        self.play(FadeIn(formula_2, shift=DOWN * 0.2))
        self.play(FadeIn(formula_3, shift=DOWN * 0.2))
        self.play(Indicate(formula_4), Create(SurroundingRectangle(formula_4, color=WHITE)))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Update colors
        self.play(
            self.lecture[0].animate.set_color(GRAY),
            self.lecture[1].animate.set_color("#ADD8E6")
        )

        # Complex formula (e^x series)
        taylor_text = "e^x = Σ x^n / n!"
        taylor = Text(taylor_text, color="#ADD8E6", font_size=36)
        # Place in column 5 to avoid horizontal crowding with the lecture text
        self.place_at_grid(taylor, "B6", scale_factor=1.0)
        self.play(Write(taylor))

        # Highlight n! denominator (indices for "n!")
        # 'e':0, '^':1, 'x':2, ' ':3, '=':4, ' ':5, 'Σ':6, ' ':7, 'x':8, '^':9, 'n':10, ' ':11, '/':12, ' ':13, 'n':14, '!':15
        highlight_box = SurroundingRectangle(taylor[14:16], color=YELLOW, buff=0.1)
        self.play(Create(highlight_box))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Update colors
        self.play(
            self.lecture[1].animate.set_color(GRAY),
            self.lecture[2].animate.set_color("#FFD700")
        )

        # Facto the Robot [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/robot.svg]
        facto = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/robot.svg")
        facto.set_color("#FFD700")
        self.place_at_grid(facto, "D6", scale_factor=1.0)
        self.play(DrawBorderThenFill(facto))

        # Final concluding text
        final_text = Text("0! = 1 completes the pattern.", color=WHITE, font_size=26)
        self.place_at_grid(final_text, "F5", scale_factor=1.0)
        self.play(FadeIn(final_text))

        self.wait(3)
