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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["How do we reverse the translation?", "P-inverse takes us back to A.", "Grid stretches and shears together."]
        self.setup_layout("Visualizing the Inverse", lecture_lines)
        
        # Define base matrix visualization elements
        matrix_p = MathTex(r"P = \begin{pmatrix} 2 & 1 \\ 0 & 1 \end{pmatrix}", color="#00FFFF")
        matrix_inv = MathTex(r"P^{-1} = \begin{pmatrix} 0.5 & -0.5 \\ 0 & 1 \end{pmatrix}", color="#FF00FF")
        
        # Asset: SVG grid
        grid_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FFFF")
        self.place_at_grid(matrix_p, "B2", scale_factor=0.9)
        self.place_at_grid(grid_svg, "B5", scale_factor=0.4)
        self.play(Write(matrix_p), FadeIn(grid_svg))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF00FF")
        self.place_at_grid(matrix_inv, "D2", scale_factor=0.9)
        self.play(FadeIn(matrix_inv))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        self.place_at_grid(grid_svg, "B5", scale_factor=0.4) # Ensure persistent
        
        # Animate \"stretch and shear\"
        self.play(
            grid_svg.animate.apply_matrix([[0.5, -0.5], [0, 1]]),
            run_time=2
        )
        self.wait(2)
