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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisite: The 2x2 Basic Calculation", [
            "Determinants for two by two matrices use ad-bc.", 
            "Cross multiply vectors to find the area change.", 
            "Example: a matrix with result six scales area."
        ])

        # Define matrix
        matrix_tex = MathTex(
            r"A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}",
            font_size=48
        )
        self.place_at_grid(matrix_tex, 'B2', scale_factor=1.0)
        
        # Asset: calculator
        calculator = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg")
        self.place_at_grid(calculator, 'B5', scale_factor=0.3)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FFFF")
        self.play(Write(matrix_tex), FadeIn(calculator))
        
        formula = MathTex(r"\det(A) = ad - bc", font_size=40)
        # Fix 66: positioned correctly using place_in_area
        self.place_in_area(formula, 'C1', 'C6', scale_factor=0.85)
        self.play(FadeIn(formula))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#00FFFF")
        
        # Highlight cross multiplication
        a = matrix_tex[0][2]
        d = matrix_tex[0][7]
        b = matrix_tex[0][4]
        c = matrix_tex[0][6]
        
        self.play(
            Indicate(VGroup(a, d), color=YELLOW),
            Indicate(VGroup(b, c), color=RED)
        )
        
        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#00FFFF")
        
        example_mat = MathTex(
            r"A = \begin{pmatrix} 3 & 1 \\ 0 & 2 \end{pmatrix}",
            font_size=40
        )
        # Fix 39: positioned correctly using place_in_area
        self.place_in_area(example_mat, 'E1', 'F3', scale_factor=0.9)
        
        res = MathTex(r"(3 \times 2) - (1 \times 0) = 6", font_size=40)
        # Fix 40: positioned correctly using place_at_grid
        self.place_at_grid(res, 'F4', scale_factor=0.9)
        
        self.play(FadeIn(example_mat), FadeIn(res))
        self.wait(2)
