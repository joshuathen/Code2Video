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
        self.setup_layout("The Linear System as a Transformation", [
            "A matrix maps a vector space to another.", 
            "Consider a robot arm moving to target.", 
            "Matrix A defines these transformation rules.", 
            "The grid deforms as A acts.", 
            "This visualizes the linear system Ax=b."
        ])
        
        # Basis vectors
        grid = NumberPlane(x_range=[-3, 3], y_range=[-3, 3], background_line_style={"stroke_opacity": 0.5})
        self.place_in_area(grid, 'A1', 'F6', scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        i_vec = Vector(RIGHT, color="#FF5733")
        j_vec = Vector(UP, color="#FF5733")
        i_vec.move_to(grid.c2p(1,0))
        j_vec.move_to(grid.c2p(0,1))
        self.play(FadeIn(grid), Create(i_vec), Create(j_vec))
        self.lecture[0].set_color("#FF5733")

        # === Animation for Lecture Line 2 ===
        # Using asset robot.svg
        robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg", color="#33FF57")
        self.place_at_grid(robot, 'C2', scale_factor=0.3)
        self.play(FadeIn(robot))
        self.lecture[1].set_color("#33FF57")

        # === Animation for Lecture Line 3 ===
        matrix_a = MathTex("A = \\begin{pmatrix} 1 & 1 \\\\ 0 & 1 \\end{pmatrix}", color="#3357FF")
        self.place_at_grid(matrix_a, 'A4', scale_factor=0.8)
        self.play(Write(matrix_a))
        self.lecture[2].set_color("#3357FF")

        # === Animation for Lecture Line 4 ===
        self.play(grid.animate.apply_matrix([[1, 1], [0, 1]]), run_time=2)
        self.lecture[3].set_color("#FFFF33")

        # === Animation for Lecture Line 5 ===
        ax_b = MathTex("Ax = b", color="#FF33FF", font_size=40)
        self.place_at_grid(ax_b, 'E6', scale_factor=0.9)
        self.play(FadeIn(ax_b))
        self.lecture[4].set_color("#FF33FF")
        
        self.wait(2)
