from manim import *
import numpy as np

# Apply configuration to prevent LaTeX cleanup race conditions
config.no_latex_cleanup = True

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
        self.setup_layout("Finding Eigenbasis: A New Perspective", [
            "An eigenbasis uses eigenvectors as coordinate axes.",
            "This simplifies matrix operations to diagonal form.",
            "Complex systems become simple, independent movements."
        ])
        
        # Load asset
        matrix_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/matrix.svg")
        
        # === Animation for Lecture Line 1 ===
        # Visualize basis change into eigenvector space.
        axes = Axes(x_length=3, y_length=3, x_range=[-2, 2], y_range=[-2, 2]).add_coordinates()
        self.place_at_grid(axes, 'C2', scale_factor=0.6)
        
        v1 = Vector([1, 1], color=BLUE)
        v2 = Vector([-1, 1], color=RED)
        
        self.play(Create(axes), Create(v1), Create(v2), FadeIn(matrix_icon.copy().scale(0.3).next_to(axes, UP)))
        self.lecture[0].set_color(BLUE)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show matrix 'A' becoming diagonal matrix 'D'.
        matrix_A = MathTex(r"A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}").scale(0.8)
        matrix_D = MathTex(r"D = \begin{pmatrix} \lambda_1 & 0 \\ 0 & \lambda_2 \end{pmatrix}").scale(0.8)
        
        self.place_at_grid(matrix_A, 'B4', scale_factor=0.7)
        self.play(Write(matrix_A), FadeIn(matrix_icon.copy().scale(0.3).next_to(matrix_A, UP)))
        self.lecture[1].set_color(GREEN)
        
        self.place_at_grid(matrix_D, 'D4', scale_factor=0.7)
        self.play(ReplacementTransform(matrix_A, matrix_D))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Color diagonal elements #00FF00 for emphasis.
        lambda_1 = matrix_D.get_parts_by_tex(r"\lambda_1")
        lambda_2 = matrix_D.get_parts_by_tex(r"\lambda_2")
        
        self.play(
            lambda_1.animate.set_color("#00FF00"),
            lambda_2.animate.set_color("#00FF00"),
            self.lecture[2].animate.set_color(YELLOW)
        )
        self.wait(2)
