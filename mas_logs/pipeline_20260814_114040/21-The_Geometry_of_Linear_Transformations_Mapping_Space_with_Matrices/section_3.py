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
        self.setup_layout("The Matrix as a Transformation Recipe", [
            "A 2x2 matrix encodes these basis vector movements.",
            "Each column represents the new landing basis vector.",
            "Matrix multiplication tracks where every vector lands."
        ])
        
        # Define visual elements
        matrix = MathTex(r"M = \\begin{bmatrix} 2 & 0 \\\\ 0 & 1.5 \\end{bmatrix}", color=WHITE)
        col_1 = MathTex(r"v_1 = \\begin{bmatrix} 2 \\\\ 0 \\end{bmatrix}", color="#00FF00")
        col_2 = MathTex(r"v_2 = \\begin{bmatrix} 0 \\\\ 1.5 \\end{bmatrix}", color="#00FF00")
        transformation = MathTex(r"M \\cdot \\begin{bmatrix} 1 \\\\ 1 \\end{bmatrix} = \\begin{bmatrix} 2 \\\\ 1.5 \\end{bmatrix}", color="#FFFF00")
        
        # Load Assets
        grid_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg", color=WHITE)
        col_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/column.svg", color="#00FF00")
        path_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/path.svg", color="#FFFF00")

        # === Animation for Lecture Line 1 ===
        self.place_at_grid(grid_asset, 'B4', scale_factor=0.5)
        self.place_at_grid(matrix, 'B3', scale_factor=1.0)
        self.play(FadeIn(grid_asset), Write(matrix))
        self.lecture[0].set_color(WHITE)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.place_at_grid(col_asset, 'D4', scale_factor=0.5)
        self.place_at_grid(col_1, 'D3', scale_factor=0.9)
        self.place_at_grid(col_2, 'D4', scale_factor=0.9)
        self.play(FadeIn(col_asset), Write(col_1), Write(col_2))
        self.lecture[1].set_color("#00FF00")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.place_at_grid(path_asset, 'E4', scale_factor=0.5)
        self.place_at_grid(transformation, 'E3', scale_factor=1.0)
        self.play(FadeIn(path_asset), Write(transformation))
        self.lecture[2].set_color("#FFFF00")
        self.wait(1)
