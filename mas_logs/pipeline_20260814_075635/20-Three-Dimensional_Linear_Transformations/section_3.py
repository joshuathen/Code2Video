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
            "The 3x3 matrix acts as an instruction manual.",
            "It maps an arbitrary vector to a new position.",
            "Columns define the transformation's effect on space.",
            "Watch the cube transform into a parallelepiped.",
            "Base vectors remain fixed during the shear."
        ]
        self.setup_layout("The Mechanism: 3x3 Matrix Multiplication", lecture_lines)
        
        # Elements
        matrix = MathTex(
            r"\\begin{pmatrix} 1 & 1 & 0 \\\\ 0 & 1 & 0 \\\\ 0 & 0 & 1 \\end{pmatrix}",
            font_size=36
        )
        # Issue 27/42 Fix
        self.place_at_grid(matrix, 'B2', scale_factor=0.9)
        
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/cube.svg
        cube = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cube.svg")
        # Issue 28/43 Fix
        self.place_at_grid(cube, 'D4', scale_factor=1.0)
        
        # Animations
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(matrix))
        self.lecture[0].set_color("#FFFFFF")
        
        # === Animation for Lecture Line 2 ===
        vector = Arrow(ORIGIN, RIGHT, color=BLUE)
        self.play(FadeIn(vector))
        self.lecture[1].set_color("#FF00FF")
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00")
        self.wait(1)
        
        # === Animation for Lecture Line 4 ===
        # Shear transformation logic
        matrix_val = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
        # Note: Since cube is an SVGMobject, we use scale/transform logic if apply_matrix is tricky,
        # but apply_matrix should work for general mobjects.
        transformed_cube = cube.copy().apply_matrix(matrix_val)
        self.play(Transform(cube, transformed_cube))
        self.lecture[3].set_color("#00FFFF")
        
        # === Animation for Lecture Line 5 ===
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/cube.svg
        self.lecture[4].set_color("#00FF00")
        self.wait(2)
