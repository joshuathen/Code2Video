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
        lecture_lines = ["We use partial derivatives for PDEs.", "This symbol ∂ shows sensitivity.", "Variables change independently in space and time."]
        self.setup_layout("The Anatomy of a PDE", lecture_lines)
        
        # Define equation
        pde_eq = MathTex(
            r"\frac{\partial u}{\partial t}", "=", r"\alpha", r"\nabla^2", r"u"
        )
        self.place_in_area(pde_eq, 'B4', 'E5', scale_factor=1.0)
        
        # Asset: clock
        clock = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/clock.svg")
        
        # Symbol label for Issue 25
        partial_derivative_symbol = Text("∂ - Partial derivative", font_size=20)
        self.place_at_grid(partial_derivative_symbol, 'C4', scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.play(Write(pde_eq))
        self.play(pde_eq.animate.set_color("#FF00FF"))
        self.play(self.lecture[0].animate.set_color("#FF00FF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(Indicate(pde_eq[3]))
        self.play(pde_eq[3].animate.set_color("#FFFF00"))
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.place_at_grid(clock, 'E6', scale_factor=0.5)
        self.play(FadeIn(clock))
        self.play(Flash(pde_eq[0].get_center()))
        self.play(pde_eq[0].animate.set_color("#00FFFF"), clock.animate.set_color("#00FFFF"))
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        self.wait(2)
