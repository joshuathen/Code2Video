from manim import *
import numpy as np

class Section4Scene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # Background and Title
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)

        # Left-side lecture content
        lecture_texts = [Text(line, font_size=20, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        self.lecture.scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.5)

        # Define animation grid (6x6 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]
        cols = ["1", "2", "3", "4", "5", "6"]

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                # Positions shifted to the right half of the screen
                x = 1.5 + j * 0.8
                y = 2.0 - i * 0.8
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        if grid_pos in self.grid:
            mobject.move_to(self.grid[grid_pos])
        return mobject

    def construct(self):
        # 1. Setup Layout
        lecture_content = [
            "- Non-Square Matrices (m x n)",
            "- Mapping between dimensions",
            "- Tall matrices: R^2 -> R^3",
            "- Wide matrices: R^3 -> R^2",
            "- Loss of info vs Embedding"
        ]
        self.setup_layout("Geometrically Visualizing Non-Square Matrices", lecture_content)

        # Show Layout
        self.add(self.title)
        self.play(FadeIn(self.lecture))
        self.wait(1)

        # 2. Visualizing a 3x2 Matrix (Embedding 2D into 3D)
        # REPLACEMENT: Switched MathTex to Text to avoid FileNotFoundError: [Errno 2] No such file or directory: 'latex'
        matrix_tex = Text(
            "A = [[1, 0], [0, 1], [1, 1]]",
            font_size=24
        )
        self.place_at_grid(matrix_tex, "B2")
        
        # Add a descriptive label
        matrix_label = Text("Tall Matrix (3x2)", font_size=18, color=BLUE)
        matrix_label.next_to(matrix_tex, UP)

        self.play(Write(matrix_tex), FadeIn(matrix_label))
        self.wait(1)

        # 3. Visualizing a Wide Matrix
        wide_matrix_tex = Text(
            "B = [[1, 0, 1], [0, 1, 1]]",
            font_size=24
        )
        self.place_at_grid(wide_matrix_tex, "E2")
        
        wide_label = Text("Wide Matrix (2x3)", font_size=18, color=GREEN)
        wide_label.next_to(wide_matrix_tex, UP)

        self.play(Write(wide_matrix_tex), FadeIn(wide_label))
        self.wait(2)