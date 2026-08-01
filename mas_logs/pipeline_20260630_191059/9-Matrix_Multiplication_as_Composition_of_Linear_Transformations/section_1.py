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
        # Section 1 Title and Lecture Lines
        title = "Prerequisite: The Geometry of a Single Transformation"
        lines = [
            "A matrix transformation moves vectors in 2D space.",
            "Meet Momo the Robot, standing on our grid.",
            "We apply this matrix to the entire space.",
            "Watch Momo stretch as the grid morphs.",
            "Notice where the basis vectors land."
        ]
        self.setup_layout(title, lines)

        # Colors
        i_color = "#FF0000"
        j_color = "#00FF00"
        matrix_color = "#FFFF00"
        grid_color = "#FFFFFF"
        momo_color = "#88C0D0"

        # === Animation for Lecture Line 1 ===
        # A matrix transformation moves vectors in 2D space.
        self.lecture[0].set_color(matrix_color)
        
        # Display Matrix A at the top
        # Fix Issue 30: Adjust position to A3-A5 and scale to 0.8
        matrix_a = Text("A = [[2, 0], [0, 0.5]]", color=matrix_color)
        self.place_in_area(matrix_a, 'A3', 'A5', scale_factor=0.8)
        self.play(Write(matrix_a))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Meet Momo the Robot, standing on our grid.
        self.lecture[1].set_color(i_color)
        
        # Coordinate system (White grid)
        # Fix Issue 29: Adjust position to B2-F6 and scale to 0.7
        plane = NumberPlane(
            x_range=[-2.5, 2.5, 1],
            y_range=[-2.5, 2.5, 1],
            background_line_style={"stroke_color": grid_color, "stroke_width": 1, "stroke_opacity": 0.5}
        )
        self.place_in_area(plane, 'B2', 'F6', scale_factor=0.7)
        
        # Basis vectors
        i_hat = Arrow(plane.c2p(0,0), plane.c2p(1,0), buff=0, color=i_color, stroke_width=5)
        j_hat = Arrow(plane.c2p(0,0), plane.c2p(0,1), buff=0, color=j_color, stroke_width=5)
        
        i_label = Text("i", color=i_color, font_size=20, slant=ITALIC).next_to(i_hat.get_end(), RIGHT, buff=0.1)
        j_label = Text("j", color=j_color, font_size=20, slant=ITALIC).next_to(j_hat.get_end(), UP, buff=0.1)

        # Fix Issue 27: Use SVG Asset for Momo the Robot
        momo = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/robot.svg")
        momo.set_color(momo_color)
        momo.scale(0.3)
        # Position Momo at origin
        momo.move_to(plane.c2p(0, 0), aligned_edge=DOWN)

        self.play(Create(plane))
        self.play(Create(i_hat), Create(j_hat), Write(i_label), Write(j_label))
        self.play(FadeIn(momo))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # We apply this matrix to the entire space.
        self.lecture[2].set_color(matrix_color)
        self.play(Indicate(matrix_a))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Watch Momo stretch as the grid morphs.
        self.lecture[3].set_color(j_color)
        
        matrix_array = np.array([[2, 0], [0, 0.5]])
        
        i_final_pos = plane.c2p(2, 0)
        j_final_pos = plane.c2p(0, 0.5)

        self.play(
            plane.animate.apply_matrix(matrix_array),
            momo.animate.apply_matrix(matrix_array),
            i_hat.animate.apply_matrix(matrix_array),
            j_hat.animate.apply_matrix(matrix_array),
            i_label.animate.move_to(i_final_pos + [0.3, 0, 0]),
            j_label.animate.move_to(j_final_pos + [0, 0.3, 0]),
            run_time=3
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Notice where the basis vectors land.
        self.lecture[4].set_color(grid_color)
        
        self.play(
            Indicate(i_hat, color=i_color),
            Indicate(j_hat, color=j_color)
        )
        
        i_coord = Text("(2, 0)", color=i_color, font_size=16).next_to(i_label, DOWN, buff=0.1)
        j_coord = Text("(0, 0.5)", color=j_color, font_size=16).next_to(j_label, RIGHT, buff=0.1)
        
        self.play(Write(i_coord), Write(j_coord))
        self.wait(2)
