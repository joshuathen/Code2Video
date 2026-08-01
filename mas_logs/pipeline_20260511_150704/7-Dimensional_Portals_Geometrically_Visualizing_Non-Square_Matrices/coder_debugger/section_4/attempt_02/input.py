from manim import *
import numpy as np

class Section4Scene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # Background and Title
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content
        lecture_texts = [Text(line, font_size=20, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        self.lecture.scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.5)
        self.add(self.lecture)

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

        # 2. Visualizing a 3x2 Matrix (Embedding 2D into 3D)
        matrix_tex = MathTex(
            "A = \\begin{bmatrix} 1 & 0 \\\\ 0 & 1 \\\\ 1 & 1 \\end{bmatrix}",
            font_size=32
        )
        self.place_at_grid(matrix_tex, "B2")

        input_label = Text("Input: R^2", font_size=18, color=BLUE)
        self.place_at_grid(input_label, "D1")
        
        output_label = Text("Output: R^3", font_size=18, color=GREEN)
        self.place_at_grid(output_label, "D3")

        # Representing dimensions with simple shapes
        plane_2d = Square(side_length=1.0, color=BLUE, fill_opacity=0.3)
        self.place_at_grid(plane_2d, "E1")

        space_3d = Cube(side_length=1.0, color=GREEN, fill_opacity=0.3)
        self.place_at_grid(space_3d, "E3")
        space_3d.rotate(30 * DEGREES, axis=OUT).rotate(30 * DEGREES, axis=RIGHT)

        arrow = Arrow(start=plane_2d.get_right(), end=space_3d.get_left(), buff=0.2)

        # 3. Animations
        self.play(
            Write(self.title),
            FadeIn(self.lecture, shift=RIGHT),
            run_time=1.5
        )
        self.wait(0.5)

        self.play(
            Write(matrix_tex),
            FadeIn(input_label),
            FadeIn(output_label)
        )
        
        self.play(
            Create(plane_2d),
            GrowArrow(arrow)
        )
        
        self.play(
            ReplacementTransform(plane_2d.copy(), space_3d),
            run_time=2
        )

        # 4. Highlight Column Vectors
        col_v1 = MathTex("\\vec{v}_1", color=YELLOW, font_size=24)
        col_v2 = MathTex("\\vec{v}_2", color=ORANGE, font_size=24)
        self.place_at_grid(col_v1, "B4")
        self.place_at_grid(col_v2, "C4")

        self.play(
            FadeIn(col_v1),
            FadeIn(col_v2)
        )

        self.wait(2)