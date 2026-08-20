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
        self.setup_layout("Defining Superposition", [
            "Superposition follows the linear combination principle.",
            "Alpha and beta scale the basic states.",
            "These act as complex probability amplitudes.",
            "Think of them as spinning circular phasors.",
            "Together they form a simultaneous quantum existence."
        ])
        
        # Elements
        psi = MathTex(r"|\psi\rangle").set_color("#0000FF")
        state0 = MathTex(r"|0\rangle").set_color("#FFFF00")
        state1 = MathTex(r"|1\rangle").set_color("#FF0000")
        lin_comb = MathTex(r"|\psi\rangle = \alpha|0\rangle + \beta|1\rangle").set_color(WHITE)

        # === Animation for Lecture Line 1 ===
        self.place_at_grid(psi, "B1", scale_factor=1.5)
        self.play(FadeIn(psi), FadeIn(self.lecture[0]))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(GRAY), FadeIn(self.lecture[1]))
        self.place_at_grid(state0, "B3", scale_factor=1.2)
        self.place_at_grid(state1, "B5", scale_factor=1.2)
        self.play(FadeIn(state0), FadeIn(state1))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(GRAY), FadeIn(self.lecture[2]))
        self.place_in_area(lin_comb, "C2", "D5", scale_factor=1.0)
        self.play(Write(lin_comb))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[2].animate.set_color(GRAY), FadeIn(self.lecture[3]))
        # Representation of phasors
        phasor_alpha = Circle(radius=0.5, color="#FFFF00").set_fill(opacity=0.3)
        phasor_beta = Circle(radius=0.5, color="#FF0000").set_fill(opacity=0.3)
        self.place_at_grid(phasor_alpha, "E2", scale_factor=0.8)
        self.place_at_grid(phasor_beta, "E5", scale_factor=0.8)
        self.play(Create(phasor_alpha), Create(phasor_beta))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[3].animate.set_color(GRAY), FadeIn(self.lecture[4]))
        self.play(FadeOut(psi), FadeOut(state0), FadeOut(state1), FadeOut(lin_comb), FadeOut(phasor_alpha), FadeOut(phasor_beta))
        self.wait(2)
