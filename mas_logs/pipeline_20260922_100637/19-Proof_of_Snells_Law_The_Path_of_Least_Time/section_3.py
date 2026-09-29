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
            "Construct the total time equation.",
            "Distances relate to medium speeds.",
            "Minimize the time function T(x)."
        ]
        self.setup_layout("Formulating the Time Function", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Path asset, Color #FFFF33 (Yellow)
        path_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/path.svg")
        self.place_at_grid(path_icon, "A5", scale_factor=0.3)
        formula1 = MathTex("T = \\frac{d_1}{v_1} + \\frac{d_2}{v_2}", color="#FFFF33")
        self.place_at_grid(formula1, "B3", scale_factor=1.0)
        self.play(FadeIn(path_icon), Write(formula1))
        self.play(self.lecture[0].animate.set_color("#FFFF33"))
        
        # === Animation for Lecture Line 2 ===
        # Visualize distance variables d1 and d2, Color #FFFF33
        formula2 = MathTex("d_1 = \\sqrt{h_1^2 + x^2}, \\quad d_2 = \\sqrt{h_2^2 + (L-x)^2}", color="#FFFF33")
        self.place_at_grid(formula2, "C3", scale_factor=0.9)
        self.play(Write(formula2))
        self.play(self.lecture[1].animate.set_color("#FFFF33"))
        
        # === Animation for Lecture Line 3 ===
        # Boundary asset, T(x) function, Color #FFFFFF
        boundary_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/boundary.svg")
        self.place_at_grid(boundary_icon, "E1", scale_factor=0.3)
        formula_final = MathTex("T(x) = \\frac{\\sqrt{h_1^2 + x^2}}{v_1} + \\frac{\\sqrt{h_2^2 + (L-x)^2}}{v_2}", color="#FFFFFF")
        self.place_in_area(formula_final, "D2", "E6", scale_factor=0.6)
        self.play(FadeIn(boundary_icon), Write(formula_final))
        self.play(self.lecture[2].animate.set_color("#FFFF33"))
        
        self.wait(2)
