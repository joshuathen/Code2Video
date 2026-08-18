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
        self.setup_layout("Mathematical Generalization (N-dimensions)", [
            "Defining the hypersphere equation.",
            "Generalizing to n-dimensions.",
            "Volume depends on dimension n."
        ])
        
        # Assets
        sphere_icon_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg"
        
        # === Animation for Lecture Line 1 ===
        formula = MathTex(r"x_1^2 + x_2^2 + \dots + x_n^2 = R^2", color=WHITE)
        sphere1 = SVGMobject(sphere_icon_path)
        self.place_in_area(formula, 'B2', 'B5', scale_factor=0.8)
        self.place_at_grid(sphere1, 'A3', scale_factor=0.5)
        self.play(Write(formula), FadeIn(sphere1))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        variables = MathTex(r"X_1, X_2, \dots, X_n", color="#FFD700")
        self.place_at_grid(variables, 'D4', scale_factor=0.8)
        self.play(FadeIn(variables))
        self.lecture[1].set_color("#FFD700")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        vol_formula = MathTex(r"V_n(R) = \frac{\pi^{n/2}}{\Gamma(n/2 + 1)} R^n", color="#00CED1")
        sphere2 = SVGMobject(sphere_icon_path)
        self.place_at_grid(vol_formula, 'E4', scale_factor=0.7)
        self.place_at_grid(sphere2, 'F4', scale_factor=0.8)
        self.play(FadeIn(vol_formula), sphere2.animate.scale(1.5))
        self.lecture[2].set_color("#00CED1")
        self.wait(2)
