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
        self.setup_layout("The Geometry of Cramer's Rule", [
            "Cramer's Rule relates areas.",
            "Replace columns to define new parallelograms.",
            "Ratio of areas equals the coordinate."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Display equations for x and y variables
        eq_x = MathTex("x = \\frac{\\det(A_1)}{\\det(A)}", color=WHITE)
        eq_y = MathTex("y = \\frac{\\det(A_2)}{\\det(A)}", color=WHITE)
        equations = VGroup(eq_x, eq_y).arrange(DOWN, buff=0.5)
        self.place_in_area(equations, 'A1', 'C3', scale_factor=0.8)
        self.play(Write(equations))
        self.lecture[0].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Replace the column of the system matrix
        # Create a visual matrix representation
        matrix = MathTex("A = \\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix}", color=WHITE)
        self.place_at_grid(matrix, 'D2', scale_factor=0.6) # Applied fix from Issue 26
        self.play(FadeIn(matrix))
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/parallelogram.svg]
        # Use SVG for parallelogram replacement
        para_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/parallelogram.svg")
        self.place_at_grid(para_icon, 'D4', scale_factor=0.5) # Applied fix from Issue 27
        self.play(FadeIn(para_icon))
        
        self.lecture[1].set_color(BLUE)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Show the determinant ratio on screen
        ratio = MathTex("\\frac{\\text{Area}(A_i)}{\\text{Area}(A)} = x_i", color=WHITE)
        self.place_at_grid(ratio, 'E3', scale_factor=0.7) # Applied fix from Issue 28
        self.play(FadeIn(ratio))
        ratio.set_color("#FF00FF")
        self.lecture[2].set_color("#FF00FF")
        self.wait(2)
