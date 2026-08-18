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
        self.setup_layout("Infinite Sums: Convergence Reversal", [
            "Infinite sums behave differently here.",
            "The series 1+2+4+8 converges to -1.",
            "Terms must approach zero in the 2-adic sense."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Show partial sums of 1 + 2 + 4 +... in #FFFFFF using abacus.svg
        abacus = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/abacus.svg")
        self.place_at_grid(abacus, 'B4', scale_factor=0.6)
        
        sums = ["1", "3", "7", "15", "31"]
        seq = VGroup(*[Text(s, font_size=24, color=WHITE) for s in sums]).arrange(RIGHT, buff=0.3)
        self.place_at_grid(seq, 'C4', scale_factor=0.7)
        
        self.play(FadeIn(abacus), FadeIn(seq))
        self.lecture[0].set_color(WHITE)

        # === Animation for Lecture Line 2 ===
        # Show the 2-adic partial sums approaching -1 in #FFD700.
        res = MathTex(r"\sum_{n=0}^{\infty} 2^n = -1", color="#FFD700")
        self.place_at_grid(res, 'D4', scale_factor=1.1)
        self.play(Write(res))
        self.lecture[1].set_color("#FFD700")

        # === Animation for Lecture Line 3 ===
        # Transition from divergent in R to convergent in Q2 in #00FFFF using compass.svg
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        self.place_at_grid(compass, 'E4', scale_factor=0.6)
        
        explanation = Text("Terms: 2^n, |2^n|_2 = 1/2^n -> 0", font_size=18, color="#00FFFF")
        self.place_at_grid(explanation, 'E5', scale_factor=0.7)
        self.play(FadeIn(compass), FadeIn(explanation))
        self.lecture[2].set_color("#00FFFF")
        self.wait(2)
