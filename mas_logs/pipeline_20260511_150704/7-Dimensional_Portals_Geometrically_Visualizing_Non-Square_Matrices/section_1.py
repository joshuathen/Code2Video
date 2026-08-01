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
        # Standard Grid - Issue 29: Restricted area to avoid lecture notes
        plane = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            x_length=4,
            y_length=4,
            background_line_style={"stroke_color": "#555555", "stroke_width": 2, "stroke_opacity": 0.6},
            axis_config={"stroke_color": WHITE, "stroke_width": 2}
        )
        self.place_in_area(plane, 'B2', 'F6', scale_factor=0.8)

        # Basis Vectors relative to the coordinate system
        i_hat = Arrow(plane.c2p(0, 0), plane.c2p(1, 0), color="#FF0000", buff=0)
        j_hat = Arrow(plane.c2p(0, 0), plane.c2p(0, 1), color="#00FF00", buff=0)
        
        # Basis Labels - Issue 30: Fixed initial positions
        label_i = Text("i", color="#FF0000", slant=ITALIC)
        label_j = Text("j", color="#00FF00", slant=ITALIC)
        self.place_at_grid(label_i, 'D5', scale_factor=0.6)
        self.place_at_grid(label_j, 'B4', scale_factor=0.6)

        # Transformation Matrix - Issue 28: Repositioned to B6
        mat_a_label = Text("A = ", color=WHITE, font_size=24)
        mat_l_bracket = Text("[", font_size=40)
        mat_r_bracket = Text("]", font_size=40)
        mat_elements = VGroup(
            VGroup(Text("2", font_size=24), Text("1", font_size=24)).arrange(RIGHT, buff=0.5),
            VGroup(Text("0", font_size=24), Text("1", font_size=24)).arrange(RIGHT, buff=0.5)
        ).arrange(DOWN, buff=0.3)
        matrix_tex = VGroup(mat_a_label, mat_l_bracket, mat_elements, mat_r_bracket).arrange(RIGHT, buff=0.1)
        self.place_at_grid(matrix_tex, 'B6', scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.add(plane)
        self.play(
            GrowArrow(i_hat),
            GrowArrow(j_hat),
            Write(label_i),
            Write(label_j),
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
            coords = plane.p2c(p)
            x, y = coords[0], coords[1]
            new_x = 2 * x + 1 * y
            new_y = 0 * x + 1 * y
            return plane.c2p(new_x, new_y)

        # Target states for vectors
        target_i_end = plane.c2p(2, 0)
        target_j_end = plane.c2p(1, 1)
        target_i_hat = Arrow(plane.c2p(0, 0), target_i_end, color="#FF0000", buff=0)
        target_j_hat = Arrow(plane.c2p(0, 0), target_j_end, color="#00FF00", buff=0)

        # Animate Shear transformation
        self.play(
            plane.animate.apply_function(transform_point),
            Transform(i_hat, target_i_hat),
            Transform(j_hat, target_j_hat),
            label_i.animate.move_to(target_i_end + DOWN * 0.3),
            label_j.animate.move_to(target_j_end + UP * 0.3),
            run_time=2.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        self.lecture[1].set_color(WHITE)
        self.wait(2)
