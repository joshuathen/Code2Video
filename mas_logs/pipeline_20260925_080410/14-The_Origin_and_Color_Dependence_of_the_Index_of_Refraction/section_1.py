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
        lecture_lines = [
            "Light travels slower in dielectric media.",
            "Incident light drives electron oscillations.",
            "Electrons act like simple harmonic oscillators."
        ]
        self.setup_layout("Hook & Prerequisite Review", lecture_lines)
        
        # Elements
        n_formula = MathTex(r"n = \frac{c}{v}").set_color(BLUE)
        electron = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/electron.svg").set_color(YELLOW)
        spring = VGroup(
            Line(UP*0.5, DOWN*0.5),
            Line(DOWN*0.5, DOWN*0.5 + RIGHT*0.2),
            Line(DOWN*0.5 + RIGHT*0.2, DOWN*0.5 + RIGHT*0.2 + UP*0.2)
        ).set_color(GRAY)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.place_at_grid(n_formula, "B5", scale_factor=1.2)
        self.play(FadeIn(n_formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.place_at_grid(electron, "D5", scale_factor=0.8)
        self.play(FadeIn(electron))
        
        # Visualizing driving force
        force_arrow = Arrow(start=LEFT*2, end=LEFT*0.5, color=RED)
        self.play(GrowArrow(force_arrow))
        self.play(electron.animate.shift(RIGHT*0.5), run_time=0.5)
        self.play(electron.animate.shift(LEFT*0.5), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GRAY))
        self.place_at_grid(spring, "E6", scale_factor=1.0)
        self.play(Create(spring))
        self.wait(2)
