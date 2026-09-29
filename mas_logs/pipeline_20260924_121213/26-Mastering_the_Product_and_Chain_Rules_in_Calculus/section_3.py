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
            "Chain Rule handles composite, nested functions.",
            "Think of it as the Matryoshka doll principle.",
            "Derive the outer shell, then the inner mechanism.",
            "Multiply rates to find the total change.",
            "A mechanical claw demonstrates this perfectly."
        ]
        self.setup_layout("The Chain Rule: The Matryoshka Principle", lecture_lines)
        
        # Define elements
        matryoshka = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/matryoshka.svg").set_color("#FF66FF")
        formula = MathTex(r"f(g(x))", color="#00FFFF").scale(1.5)
        derivative_formula = MathTex(r"f'(g(x)) \cdot g'(x)", color="#00FFFF").scale(1.2)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF66FF")
        self.place_at_grid(formula, 'A3', scale_factor=0.9)
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF66FF")
        self.place_at_grid(matryoshka, 'D3', scale_factor=0.6)
        self.play(FadeIn(matryoshka))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FFFF")
        self.play(Indicate(matryoshka))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#00FFFF")
        group = VGroup(derivative_formula)
        self.place_in_area(group, 'B2', 'D4', scale_factor=0.8)
        self.play(Transform(formula, derivative_formula))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FF66FF")
        claw = VGroup(
            Line(UP, DOWN, color=WHITE).shift(LEFT*0.2),
            Line(UP, DOWN, color=WHITE).shift(RIGHT*0.2)
        )
        self.place_at_grid(claw, 'B5', scale_factor=0.9)
        self.play(Create(claw))
        self.play(claw.animate.rotate(0.2).rotate(-0.4).rotate(0.2))
        self.wait(2)
