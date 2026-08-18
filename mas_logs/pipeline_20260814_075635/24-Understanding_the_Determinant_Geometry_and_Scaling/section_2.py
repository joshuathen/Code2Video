from manim import *
import numpy as np
import os

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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisite: The 2x2 Case", [
            "Basis vectors [1,0] and [0,1] form unit area.",
            "Matrices map basis to new parallelogram edges.",
            "Determinant is the signed area of that parallelogram."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Show a 2x2 matrix
        # Fix for issue 23: Positioning matrix
        matrix = MathTex(r"M = \\begin{bmatrix} a & b \\\\ c & d \\end{bmatrix}", color="#FFFFFF")
        self.place_in_area(matrix, 'A2', 'B3', scale_factor=1.0)
        self.play(Write(matrix))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Visualize the transformation of unit square using Assets
        # Fix for issue 17 & 24: Use asset icons, fix positioning
        square_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/square.svg"
        para_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/parallelogram.svg"
        
        square = SVGMobject(square_path, color=BLUE, fill_opacity=0.3)
        parallelogram = SVGMobject(para_path, color="#FF00FF", fill_opacity=0.3)
        
        # Position grouping
        group = VGroup(square)
        self.place_in_area(group, 'D3', 'F6', scale_factor=0.5)
        self.add(group)
        
        self.play(Transform(square, parallelogram))
        self.play(self.lecture[1].animate.set_color("#FF00FF"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Calculate determinant as area
        # Fix for issue 23: Positioning det text
        det_text = MathTex(r"\\det(M) = |ad - bc|", color="#FFFF00")
        self.place_in_area(det_text, 'A4', 'B6', scale_factor=0.9)
        self.play(FadeIn(det_text))
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        self.wait(2)
