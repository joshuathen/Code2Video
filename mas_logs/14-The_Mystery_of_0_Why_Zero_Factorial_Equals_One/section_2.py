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
            "Let's look at factorials in a descending pattern.",
            "To find three factorial, divide twenty-four by four.",
            "Next, we divide by three to reach two factorial.",
            "Dividing by two leads us to one factorial.",
            "Dividing one by one reveals zero factorial equals one."
        ]
        self.setup_layout("The Pattern Approach (Division Logic)", lecture_lines)

        # 1. Prepare Factorial Equations (Using Text instead of MathTex to avoid LaTeX dependency)
        fact4 = Text("4! = 24", font_size=36)
        fact3 = Text("3! = 6", font_size=36)
        fact2 = Text("2! = 2", font_size=36)
        fact1 = Text("1! = 1", font_size=36)
        fact0 = Text("0! = 1", font_size=36, color=YELLOW)

        # Position equations at Column 4 (Addressing issues 31, 32, 33)
        self.place_at_grid(fact4, 'B4')
        self.place_at_grid(fact3, 'C4')
        self.place_at_grid(fact2, 'D4')
        self.place_at_grid(fact1, 'E4')
        self.place_at_grid(fact0, 'F4')

        # 2. Prepare Arrows and Labels
        arrow_color = "#00FF00"
        
        # Arrow 1: between fact4 (B4) and fact3 (C4), positioned at Col 5
        arrow1 = Arrow(start=self.grid['B5'], end=self.grid['C5'], color=arrow_color, buff=0.1)
        label1 = Text("divided by 4", font_size=16, color=arrow_color)
        self.place_in_area(label1, 'B6', 'C6')

        # Arrow 2: between fact3 (C4) and fact2 (D4), positioned at Col 5
        arrow2 = Arrow(start=self.grid['C5'], end=self.grid['D5'], color=arrow_color, buff=0.1)
        label2 = Text("divided by 3", font_size=16, color=arrow_color)
        self.place_in_area(label2, 'C6', 'D6')

        # Arrow 3: between fact2 (D4) and fact1 (E4), positioned at Col 5
        arrow3 = Arrow(start=self.grid['D5'], end=self.grid['E5'], color=arrow_color, buff=0.1)
        label3 = Text("divided by 2", font_size=16, color=arrow_color)
        self.place_in_area(label3, 'D6', 'E6')

        # Arrow 4: between fact1 (E4) and fact0 (F4), positioned at Col 5
        arrow4 = Arrow(start=self.grid['E5'], end=self.grid['F5'], color=arrow_color, buff=0.1)
        label4 = Text("divided by 1", font_size=16, color=arrow_color)
        self.place_in_area(label4, 'E6', 'F6')

        # === Animation for Lecture Line 1 ===
        # Let's look at factorials in a descending pattern.
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(
            AnimationGroup(
                FadeIn(fact4, shift=DOWN*0.2),
                FadeIn(fact3, shift=DOWN*0.2),
                FadeIn(fact2, shift=DOWN*0.2),
                FadeIn(fact1, shift=DOWN*0.2),
                lag_ratio=0.2
            )
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # To find three factorial, divide twenty-four by four.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(GREEN)
        )
        self.play(Create(arrow1), Write(label1))
        self.play(Indicate(fact3, color=GREEN))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Next, we divide by three to reach two factorial.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(GREEN)
        )
        self.play(Create(arrow2), Write(label2))
        self.play(Indicate(fact2, color=GREEN))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Dividing by two leads us to one factorial.
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(GREEN)
        )
        self.play(Create(arrow3), Write(label3))
        self.play(Indicate(fact1, color=GREEN))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Dividing one by one reveals zero factorial equals one.
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(YELLOW)
        )
        self.play(Create(arrow4), Write(label4))
        self.play(FadeIn(fact0, shift=DOWN*0.2))
        self.play(Indicate(fact0, color=YELLOW))
        self.wait(2)
