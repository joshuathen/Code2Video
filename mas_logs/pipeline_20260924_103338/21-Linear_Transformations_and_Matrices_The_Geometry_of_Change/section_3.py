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
            "Matrix multiplication applies a transformation to a vector.",
            "The resulting vector shows the transformed location.",
            "Geometric warping maps input points to new coordinates."
        ]
        self.setup_layout("Visualizing Operations: Matrix-Vector Multiplication", lecture_lines)
        
        # Load Assets
        bg_grid = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        bg_graph = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/graph.svg")
        
        # Define objects
        matrix = MathTex(r"A = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}").set_color(BLUE)
        vec_v = MathTex(r"\vec{v} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}").set_color(GREEN)
        result_vec = MathTex(r"A\vec{v} = \begin{bmatrix} 2 \\ 1 \end{bmatrix}").set_color(YELLOW)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.place_at_grid(bg_grid, 'B5', scale_factor=0.3)
        self.add(bg_grid)
        self.place_at_grid(matrix, 'A2', scale_factor=0.7)
        self.place_at_grid(vec_v, 'A4', scale_factor=0.7)
        self.play(Write(matrix), Write(vec_v))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.place_at_grid(bg_graph, 'E5', scale_factor=0.3)
        self.add(bg_graph)
        self.place_at_grid(result_vec, 'C3', scale_factor=0.8)
        self.play(Write(result_vec))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(RED))
        
        # Simple geometric representation of "warping"
        square = Square(side_length=1.5).set_stroke(WHITE, width=2)
        sheared_square = Square(side_length=1.5).set_fill(RED, opacity=0.3).apply_matrix([[1, 1], [0, 1]])
        self.place_in_area(square, 'D2', 'D3', scale_factor=0.5)
        self.place_in_area(sheared_square, 'D4', 'D5', scale_factor=0.5)
        
        self.play(Create(square))
        self.play(Transform(square, sheared_square))
        self.wait(2)
