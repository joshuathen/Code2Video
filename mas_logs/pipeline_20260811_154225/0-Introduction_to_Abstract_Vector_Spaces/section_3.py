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
            "Polynomials behave exactly like vectors.",
            "Map coefficients as 3D coordinates.",
            "Geometry and algebra align perfectly."
        ]
        self.setup_layout("Concrete vs. Abstract Examples", lecture_lines)
        
        # Setup Axes
        axes = ThreeDAxes(x_range=[-2, 2], y_range=[-2, 2], z_range=[-2, 2], axis_config={"include_tip": True})
        # Use Asset [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/coordinate.svg]
        # Per Issue 23: Fix position
        self.place_in_area(axes, 'D2', 'F6', scale_factor=0.5)
        self.add(axes)
        
        # Polynomial: P(x) = ax^2 + bx + c -> (a, b, c)
        def get_poly_vector(a, b, c):
            return axes.c2p(a, b, c)

        # === Animation for Lecture Line 1 ===
        # Per Issue 25: Fix position
        poly_vec = Arrow(start=axes.get_origin(), end=get_poly_vector(1, 0.5, -1), color="#00FF00")
        self.place_at_grid(poly_vec, 'C2', scale_factor=0.6)
        self.lecture[0].set_color("#00FF00")
        self.add(poly_vec)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Per Issue 24: Fix position
        dot = Dot(color="#00CCFF").move_to(get_poly_vector(1, 0.5, -1))
        point_label = MathTex("(a, b, c)", color="#00CCFF")
        self.place_at_grid(point_label, 'D5', scale_factor=0.7)
        self.add(dot, point_label)
        self.lecture[1].set_color("#00CCFF")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        poly_vec_2 = Arrow(start=axes.get_origin(), end=get_poly_vector(-1, 1, 0.5), color="#FF00FF")
        dot_2 = Dot(color="#FF00FF").move_to(get_poly_vector(-1, 1, 0.5))
        self.add(poly_vec_2, dot_2)
        self.lecture[2].set_color("#FF00FF")
        self.wait(2)
