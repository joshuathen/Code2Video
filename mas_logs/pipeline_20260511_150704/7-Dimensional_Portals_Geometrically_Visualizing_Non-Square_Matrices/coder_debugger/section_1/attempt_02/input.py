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
        # 1. Setup Layout
        title = "Prerequisite: The Matrix as a Map Maker"
        lines = [
            'Matrix transformations move the basis vectors i and j.',
            'These vectors define how the entire grid transforms.',
            'In square matrices, the plane remains two-dimensional.'
        ]
        self.setup_layout(title, lines)

        # 2. Coordinate System Preparation
        area_center = (self.grid["B1"] + self.grid["F6"]) / 2
        unit_size = 0.8  

        # Standard Grid
        plane = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=6 * unit_size,
            y_length=6 * unit_size,
            background_line_style={"stroke_color": "#555555", "stroke_width": 2, "stroke_opacity": 0.6},
            axis_config={"stroke_color": WHITE, "stroke_width": 2}
        ).move_to(area_center)

        # Basis Vectors
        i_hat = Vector([unit_size, 0], color="#FF0000").shift(area_center)
        j_hat = Vector([0, unit_size], color="#00FF00").shift(area_center)
        
        # Use Text instead of MathTex to avoid LaTeX dependency error
        i_label = Text("i", color="#FF0000", slant=ITALIC).scale(0.6)
        j_label = Text("j", color="#00FF00", slant=ITALIC).scale(0.6)
        i_label.next_to(i_hat.get_end(), DOWN, buff=0.1)
        j_label.next_to(j_hat.get_end(), LEFT, buff=0.1)

        # Transformation Matrix built manually with Text objects
        mat_a_label = Text("A = ", color=WHITE, font_size=24)
        mat_l_bracket = Text("[", font_size=40)
        mat_r_bracket = Text("]", font_size=40)
        mat_elements = VGroup(
            VGroup(Text("2", font_size=24), Text("1", font_size=24)).arrange(RIGHT, buff=0.5),
            VGroup(Text("0", font_size=24), Text("1", font_size=24)).arrange(RIGHT, buff=0.5)
        ).arrange(DOWN, buff=0.3)
        matrix_tex = VGroup(mat_a_label, mat_l_bracket, mat_elements, mat_r_bracket).arrange(RIGHT, buff=0.1)
        
        self.place_at_grid(matrix_tex, "A5", scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.add(plane)
        self.play(
            GrowArrow(i_hat),
            GrowArrow(j_hat),
            Write(i_label),
            Write(j_label),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        self.lecture[0].set_color(WHITE)
        self.play(Write(matrix_tex))
        self.wait(0.5)

        # Define Transformation Logic for Matrix [[2, 1], [0, 1]]
        def transform_point(p):
            rel_p = p - area_center
            x, y = rel_p[0], rel_p[1]
            new_x = 2 * x + 1 * y
            new_y = 0 * x + 1 * y
            return area_center + np.array([new_x, new_y, 0])

        # Define target basis vectors
        target_i_end = area_center + np.array([2 * unit_size, 0, 0])
        target_j_end = area_center + np.array([1 * unit_size, 1 * unit_size, 0])
        
        target_i_hat = Vector(target_i_end - area_center, color="#FF0000").shift(area_center)
        target_j_hat = Vector(target_j_end - area_center, color="#00FF00").shift(area_center)

        # Animate Shear
        self.play(
            plane.animate.apply_function(transform_point),
            Transform(i_hat, target_i_hat),
            Transform(j_hat, target_j_hat),
            i_label.animate.next_to(target_i_end, DOWN, buff=0.1),
            j_label.animate.next_to(target_j_end, UP, buff=0.1),
            run_time=2.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        self.lecture[1].set_color(WHITE)
        self.wait(2)