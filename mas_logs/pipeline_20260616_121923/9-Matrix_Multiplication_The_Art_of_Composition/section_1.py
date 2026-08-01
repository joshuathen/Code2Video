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
        # Initial Setup
        title = "Prerequisite: The Matrix as a Movement"
        lines = [
            'Meet our Geometry Robot, starting at coordinate one zero.',
            'Matrix A acts as a rule for rotating space.',
            'As the grid transforms, the basis vectors change position.',
            'Our Robot follows this motion, rotating to zero one.',
            'Matrix columns track exactly where those basis vectors land.'
        ]
        self.setup_layout(title, lines)

        # Colors
        COLOR_GRID = "#555555"
        COLOR_ROBOT = "#00FF00"
        COLOR_MATRIX = "#FFFF00"
        COLOR_I = "#FF0000"
        COLOR_J = "#0000FF"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_ROBOT)
        
        # Define the coordinate system and area
        coord_sys = NumberPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            x_length=3.5,
            y_length=3.5,
            background_line_style={"stroke_color": COLOR_GRID, "stroke_opacity": 0.6},
            axis_config={"stroke_color": WHITE, "include_tip": True}
        )
        self.place_in_area(coord_sys, "B2", "E4")

        # Robot representation (Asset Integration)
        robot = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/robot.svg")
        robot.set_color(COLOR_ROBOT)
        robot.scale(0.2)
        # Initial position (1,0) in grid coordinates
        robot.move_to(coord_sys.c2p(1, 0))

        # Basis vectors
        vec_i = Arrow(coord_sys.c2p(0, 0), coord_sys.c2p(1, 0), buff=0, color=COLOR_I, stroke_width=4)
        vec_j = Arrow(coord_sys.c2p(0, 0), coord_sys.c2p(0, 1), buff=0, color=COLOR_J, stroke_width=4)
        label_i = Text("i", color=COLOR_I, font_size=20, slant=ITALIC).next_to(vec_i, DOWN, buff=0.1)
        label_j = Text("j", color=COLOR_J, font_size=20, slant=ITALIC).next_to(vec_j, LEFT, buff=0.1)

        self.play(Create(coord_sys), FadeIn(robot), FadeIn(vec_i, label_i), FadeIn(vec_j, label_j), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(COLOR_MATRIX)

        # Build Matrix A manually
        # Matrix A = [[0, -1], [1, 0]]
        a_eq = Text("A = ", color=COLOR_MATRIX, font_size=25)
        v0 = Text("0", color=COLOR_MATRIX, font_size=25)
        v1 = Text("-1", color=COLOR_MATRIX, font_size=25)
        v2 = Text("1", color=COLOR_MATRIX, font_size=25)
        v3 = Text("0", color=COLOR_MATRIX, font_size=25)
        
        mat_vals = VGroup(v0, v1, v2, v3).arrange_in_grid(rows=2, cols=2, buff=0.3)
        bracket_l = Text("[", font_size=40, color=COLOR_MATRIX).next_to(mat_vals, LEFT, buff=0.1)
        bracket_r = Text("]", font_size=40, color=COLOR_MATRIX).next_to(mat_vals, RIGHT, buff=0.1)
        
        matrix_a = VGroup(a_eq, bracket_l, mat_vals, bracket_r).arrange(RIGHT, buff=0.1)
        # Fix: Move matrix_a to A5 and scale per issues 27 and 28
        self.place_at_grid(matrix_a, "A5", scale_factor=1.2)
        
        self.play(Write(matrix_a))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(WHITE)
        
        # Transformation: A rotation of 90 degrees CCW
        # i(1,0) -> (0,1)
        # j(0,1) -> (-1,0)
        new_vec_i = Arrow(coord_sys.c2p(0, 0), coord_sys.c2p(0, 1), buff=0, color=COLOR_I, stroke_width=4)
        new_vec_j = Arrow(coord_sys.c2p(0, 0), coord_sys.c2p(-1, 0), buff=0, color=COLOR_J, stroke_width=4)
        
        self.play(
            coord_sys.animate.apply_matrix([[0, -1], [1, 0]]),
            Transform(vec_i, new_vec_i),
            Transform(vec_j, new_vec_j),
            label_i.animate.next_to(coord_sys.c2p(0, 1), RIGHT, buff=0.1),
            label_j.animate.next_to(coord_sys.c2p(-1, 0), UP, buff=0.1),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(COLOR_ROBOT)
        # Robot follows the motion to (0,1)
        self.play(
            robot.animate.move_to(coord_sys.c2p(0, 1)),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(COLOR_MATRIX)
        
        # Highlight columns of Matrix A
        # mat_vals are v0, v1, v2, v3. Column 1: v0, v2. Column 2: v1, v3.
        col1_rect = SurroundingRectangle(VGroup(v0, v2), color=COLOR_MATRIX, buff=0.1)
        col2_rect = SurroundingRectangle(VGroup(v1, v3), color=COLOR_MATRIX, buff=0.1)
        
        self.play(Create(col1_rect))
        self.play(vec_i.animate.set_stroke(width=8), run_time=0.5)
        self.wait(0.5)
        self.play(ReplacementTransform(col1_rect, col2_rect))
        self.play(vec_i.animate.set_stroke(width=4), vec_j.animate.set_stroke(width=8), run_time=0.5)
        self.wait(1)
        self.play(FadeOut(col2_rect), vec_j.animate.set_stroke(width=4))
        self.wait(2)
