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
        # Initial layout setup
        lecture_lines = [
            'Track where the basis vectors land after transformation.',
            'These new landing spots form the matrix columns.',
            'Each column represents a transformed basis vector.'
        ]
        self.setup_layout("The Matrix as a Coordinate Map", lecture_lines)

        # Plane setup
        plane = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            x_length=5,
            y_length=5,
            axis_config={"stroke_width": 2},
            background_line_style={
                "stroke_color": BLUE_D,
                "stroke_width": 1,
                "stroke_opacity": 0.5
            }
        )
        # Issue 31 Fix: Shifted top corner from A2 to B2 to avoid lecture note obstruction
        self.place_in_area(plane, 'B2', 'F6')
        self.add(plane)

        # Original basis vectors
        i_vec = Vector(plane.c2p(1, 0) - plane.get_center(), color="#FF0000").shift(plane.get_center())
        j_vec = Vector(plane.c2p(0, 1) - plane.get_center(), color="#00FF00").shift(plane.get_center())
        
        # Transformation matrix components
        matrix_vals = np.array([[1, 3], [-2, 0]])

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(Create(i_vec), Create(j_vec))
        
        self.play(
            plane.animate.apply_matrix(matrix_vals),
            i_vec.animate.put_start_and_end_on(plane.get_center(), plane.c2p(1, -2)),
            j_vec.animate.put_start_and_end_on(plane.get_center(), plane.c2p(3, 0)),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        
        # Using Text and VGroup to avoid LaTeX dependency (FileNotFoundError: 'latex')
        l_bracket = Text("[", font_size=60).scale(1.8)
        r_bracket = Text("]", font_size=60).scale(1.8)
        matrix_brackets = VGroup(l_bracket, r_bracket).arrange(RIGHT, buff=1.4)
        
        # Issue 32 Fix: Moved matrix position and adjusted scale factor
        self.place_at_grid(matrix_brackets, 'B3', scale_factor=1.1)
        self.play(FadeIn(matrix_brackets))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        
        # Construct matrix columns using Text and VGroup
        col1 = VGroup(Text("1", font_size=36), Text("-2", font_size=36)).arrange(DOWN, buff=0.4).set_color("#FF0000")
        col2 = VGroup(Text("3", font_size=36), Text("0", font_size=36)).arrange(DOWN, buff=0.4).set_color("#00FF00")
        
        # Scale and position columns relative to brackets
        col1.scale(0.9).move_to(matrix_brackets.get_center() + LEFT * 0.35)
        col2.scale(0.9).move_to(matrix_brackets.get_center() + RIGHT * 0.35)
        
        self.play(Write(col1))
        self.play(Write(col2))
        
        # Pulse matrix and remorph grid
        matrix_full = VGroup(matrix_brackets, col1, col2)
        inv_matrix = np.linalg.inv(matrix_vals)
        
        self.play(
            plane.animate.apply_matrix(inv_matrix),
            i_vec.animate.put_start_and_end_on(plane.get_center(), plane.c2p(1, 0)),
            j_vec.animate.put_start_and_end_on(plane.get_center(), plane.c2p(0, 1)),
            matrix_full.animate.scale(1.1).set_color(WHITE),
            run_time=2,
            rate_func=there_and_back
        )
        
        self.wait(2)
