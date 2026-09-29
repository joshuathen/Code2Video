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
        self.setup_layout("Application: Counting Constrained Sets", ["Evaluate (1+x)^n at cube roots.", "Simplify using polar forms.", "Sum constraints extract valid configurations."])
        
        # === Animation for Lecture Line 1 ===
        # Evaluate (1+x)^n at cube roots.
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg", color=WHITE)
        self.place_at_grid(compass, 'B2', scale_factor=0.5)
        
        expr = MathTex(r"(1+x)^n", color=BLUE)
        root1 = MathTex(r"\omega = e^{2\pi i / 3}", color=GREEN)
        self.place_at_grid(expr, 'A2', scale_factor=1.0)
        self.place_at_grid(root1, 'A5', scale_factor=1.0)
        
        self.play(FadeIn(compass), Write(expr), Write(root1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Simplify using polar forms.
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color("#FFD700"))
        
        protractor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/protractor.svg", color=WHITE)
        self.place_at_grid(protractor, 'B5', scale_factor=0.5)
        
        polar1 = MathTex(r"1+\omega = \sqrt{3}e^{i\pi/6}", color=YELLOW)
        self.place_in_area(polar1, 'C2', 'C5', scale_factor=0.85)
        
        self.play(FadeIn(protractor), FadeIn(polar1))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Sum constraints extract valid configurations.
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color("#FFD700"))
        sum_form = MathTex(r"\frac{1}{3} \sum_{k=0}^2 (1+\omega^k)^n", color=RED)
        self.place_in_area(sum_form, 'D2', 'D5', scale_factor=0.85)
        
        self.play(FadeIn(sum_form))
        self.wait(2)
