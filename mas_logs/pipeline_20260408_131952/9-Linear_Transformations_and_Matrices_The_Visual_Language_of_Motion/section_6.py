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

class Section6Scene(TeachingScene):
    def construct(self):
        # Setup the layout with lecture bullet points
        self.setup_layout(
            "Linear Transformations and Matrices",
            [
                "- Matrices represent geometric transformations.",
                "- Each column vector shows where",
                "  basis vectors (i, j) land.",
                "- Transformation is a linear map",
                "  preserving the origin and lines."
            ]
        )

        # Create a coordinate plane for visual demonstration
        plane = NumberPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            x_length=4,
            y_length=4,
            background_line_style={"stroke_color": BLUE_D, "stroke_width": 2, "stroke_opacity": 0.4}
        )
        self.place_in_area(plane, "A3", "F6", scale_factor=0.9)

        # Basis vectors
        i_hat = Vector([1, 0], color=GREEN)
        j_hat = Vector([0, 1], color=RED)
        plane.add(i_hat, j_hat)

        # Matrix notation - Using Text instead of MathTex to avoid FileNotFoundError: 'latex'
        matrix_tex = Text(
            "A = [[0, -1], [1, 0]]",
            font_size=28,
            color=WHITE
        )
        self.place_at_grid(matrix_tex, "B1", scale_factor=1.2)

        # Animation sequence
        self.play(Create(plane), Write(matrix_tex))
        self.wait(1)

        # Matrix transformation application (90-degree rotation)
        # Transformation matrix: [[0, -1], [1, 0]]
        transformation_matrix = np.array([[0, -1], [1, 0]])
        
        self.play(
            plane.animate.apply_matrix(transformation_matrix),
            run_time=3,
            rate_func=slow_into
        )
        
        self.wait(2)
