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

class Section5Scene(TeachingScene):
    def create_matrix_mobject(self, matrix, color=WHITE):
        # matrix is a list of lists [[a, b], [c, d]]
        rows = []
        for r in matrix:
            row_vals = [Text(str(val), font_size=24, color=color) for val in r]
            rows.append(VGroup(*row_vals).arrange(RIGHT, buff=0.5))
        
        matrix_vgroup = VGroup(*rows).arrange(DOWN, buff=0.4)
        
        # Add brackets (using custom lines to look like brackets)
        bracket_h = 0.6
        bracket_w = 0.1
        left_bracket = VGroup(
            Line(UP*bracket_h/2, DOWN*bracket_h/2, stroke_width=2),
            Line(UP*bracket_h/2, RIGHT*bracket_w, stroke_width=2),
            Line(DOWN*bracket_h/2, RIGHT*bracket_w, stroke_width=2)
        ).next_to(matrix_vgroup, LEFT, buff=0.15).set_color(color)
        
        right_bracket = VGroup(
            Line(UP*bracket_h/2, DOWN*bracket_h/2, stroke_width=2),
            Line(UP*bracket_h/2, LEFT*bracket_w, stroke_width=2),
            Line(DOWN*bracket_h/2, LEFT*bracket_w, stroke_width=2)
        ).next_to(matrix_vgroup, RIGHT, buff=0.15).set_color(color)
        
        return VGroup(left_bracket, matrix_vgroup, right_bracket)

    def construct(self):
        title = "Application: Transforming 'Leo the Lion'"
        lines = [
            "Let’s apply these transformations to Leo the Lion.",
            "A shear matrix makes Leo lean to the side.",
            "Notice how his feet stay fixed at the origin.",
            "A scaling matrix changes his height and width.",
            "The identity matrix returns Leo to his original shape."
        ]
        self.setup_layout(title, lines)

        # Color constants
        GOLD_COLOR = "#FFD700"
        SHEAR_COLOR = "#00FFFF"
        SCALE_COLOR = "#ADFF2F"
        ORIGIN_COLOR = "#FF0000"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(GOLD_COLOR)
        
        # Create Leo the Lion from Asset [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/lion.svg]
        leo = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/lion.svg")
        leo.set_color(GOLD_COLOR)
        leo.set_stroke(GOLD_COLOR, width=2)
        leo.set_fill(GOLD_COLOR, opacity=0.3)
        
        # Background Grid for Leo to stand on
        plane = NumberPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            x_length=4,
            y_length=4,
            background_line_style={"stroke_opacity": 0.4}
        ).set_color(GRAY)
        
        visual_area = VGroup(plane, leo)
        self.place_in_area(visual_area, "A1", "F6", scale_factor=0.8)
        # Position Leo so his feet are near origin (c2p(0,0))
        leo.move_to(plane.c2p(0, 0, 0), aligned_edge=DOWN)
        
        self.play(Create(plane), run_time=1)
        self.play(FadeIn(leo), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(SHEAR_COLOR)
        
        # Matrix Label for Shear
        shear_matrix_tex = self.create_matrix_mobject([[1, 1], [0, 1]], color=SHEAR_COLOR)
        # Apply Issue 35: Spatial balance fix
        self.place_in_area(shear_matrix_tex, 'A1', 'B2', scale_factor=0.8)
        
        self.play(Write(shear_matrix_tex))
        
        # Apply shear matrix to the lion silhouette [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/lion.svg]
        shear_matrix = [[1, 1], [0, 1]]
        self.play(
            leo.animate.apply_matrix(shear_matrix),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(ORIGIN_COLOR)
        
        origin_dot = Dot(plane.c2p(0, 0, 0), color=ORIGIN_COLOR)
        origin_label = Text("(0,0)", font_size=16, color=ORIGIN_COLOR).next_to(origin_dot, DOWN, buff=0.1)
        
        self.play(FadeIn(origin_dot), FadeIn(origin_label))
        self.play(Indicate(origin_dot, color=ORIGIN_COLOR, scale_factor=2))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(SCALE_COLOR)
        
        # Scaling matrix label
        scale_matrix_tex = self.create_matrix_mobject([[0.5, 0], [0, 2]], color=SCALE_COLOR)
        # Apply Issue 36: Spatial balance fix
        self.place_in_area(scale_matrix_tex, 'A1', 'B2', scale_factor=0.8)
        
        # Revert shear and apply scaling to the lion silhouette [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/lion.svg]
        self.play(
            FadeOut(shear_matrix_tex),
            leo.animate.apply_matrix([[1, -1], [0, 1]]), 
            run_time=1
        )
        
        self.play(Write(scale_matrix_tex))
        scale_matrix = [[0.5, 0], [0, 2]]
        self.play(
            leo.animate.apply_matrix(scale_matrix),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(WHITE)
        
        identity_label = Text("Identity", font_size=20, color=WHITE)
        # Apply Issue 37: Proximity fix
        self.place_at_grid(identity_label, 'B2', scale_factor=0.8)
        
        # Revert scaling to return to original shape
        self.play(
            FadeOut(scale_matrix_tex),
            FadeIn(identity_label),
            leo.animate.apply_matrix([[2, 0], [0, 0.5]]), 
            run_time=1.5
        )
        
        self.wait(2)
        self.play(FadeOut(origin_dot), FadeOut(origin_label), FadeOut(identity_label))
        self.wait(1)
