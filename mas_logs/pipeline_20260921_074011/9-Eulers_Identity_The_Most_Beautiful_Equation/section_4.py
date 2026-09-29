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
        lecture_lines = ["Euler's formula links exponentials to circles.", "e to the ix equals cos(x) plus i sin(x).", "A rotating vector maps these parts."]
        self.setup_layout("Deriving Euler’s Formula", lecture_lines)
        
        # Load asset
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        self.place_at_grid(compass, 'A4', scale_factor=0.3)
        
        # Define equations
        eq1 = MathTex(r"e^x = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \dots", font_size=32)
        eq2 = MathTex(r"e^{ix} = 1 + ix + \frac{(ix)^2}{2!} + \frac{(ix)^3}{3!} + \dots", font_size=32)
        eq3 = MathTex(r"e^{ix} = (1 - \frac{x^2}{2!} + \dots) + i(x - \frac{x^3}{3!} + \dots)", font_size=32)
        eq4 = MathTex(r"e^{ix} = \cos(x) + i\sin(x)", font_size=40)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.place_in_area(eq1, 'B2', 'B5', scale_factor=0.9)), FadeIn(compass))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(Transform(eq1, self.place_at_grid(eq2, 'C3')))
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(Transform(eq1, self.place_in_area(eq3, 'D2', 'D5', scale_factor=0.9)))
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        self.wait(1)
        
        self.play(Transform(eq1, self.place_in_area(eq4, 'F2', 'F5', scale_factor=1.0)))
        self.play(eq4.animate.set_color("#FFFFFF"))
        self.wait(2)
