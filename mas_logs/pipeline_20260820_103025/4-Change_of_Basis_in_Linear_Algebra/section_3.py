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
        self.setup_layout("Deriving the Calculation Process", [
            "Equation: the vector in A equals P times B.",
            "Matrix P columns are basis vectors in A.",
            "Basis vectors serve as the transformation's building blocks."
        ])
        
        # Load assets
        block_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg")
        brick_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/brick.svg")

        # === Animation for Lecture Line 1 ===
        # Equation: the vector in A equals P times B.
        eq = MathTex(r"[v]_A = P \cdot [v]_B", font_size=40)
        self.place_in_area(eq, 'A2', 'B5', scale_factor=0.6)
        self.play(Write(eq))
        self.lecture[0].set_color("#3498DB")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Matrix P columns are basis vectors in A.
        p_matrix = MathTex(r"P = \begin{bmatrix} | & | \\ b_1 & b_2 \\ | & | \end{bmatrix}", font_size=36)
        self.place_in_area(p_matrix, 'C3', 'E6', scale_factor=0.6)
        
        # Place block icon as operator
        self.place_at_grid(block_icon, 'C2', scale_factor=0.7)
        block_icon.set_color("#F1C40F")
        
        self.play(Write(p_matrix), FadeIn(block_icon))
        self.lecture[1].set_color("#F1C40F")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Basis vectors serve as the transformation's building blocks.
        
        # Place brick icons
        brick1 = brick_icon.copy()
        brick2 = brick_icon.copy()
        
        self.place_at_grid(brick1, 'E2', scale_factor=0.5)
        self.place_at_grid(brick2, 'E4', scale_factor=0.5)
        
        self.play(FadeIn(brick1), FadeIn(brick2))
        
        highlight = SurroundingRectangle(p_matrix, color="#2ECC71", buff=0.1)
        self.play(Create(highlight))
        self.lecture[2].set_color("#2ECC71")
        self.wait(2)
