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
            "Transition matrix P translates between bases.",
            "Columns of P are new basis vectors.",
            "Apply [v]_old = P * [v]_new for conversion.",
            "Visualize the mapping process.",
            "Example: P = [[1, -1], [1, 1]] maps coordinates."
        ]
        self.setup_layout("The Mechanism: The Transition Matrix (P)", lecture_lines)
        
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg
        grid_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        self.place_at_grid(grid_icon, "A5", scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        # Show matrix P
        matrix_p = MathTex("P = \\begin{pmatrix} 1 & -1 \\\\ 1 & 1 \\end{pmatrix}")
        self.place_at_grid(matrix_p, "B3", scale_factor=0.7)
        self.play(FadeIn(matrix_p), FadeIn(grid_icon))
        self.lecture[0].set_color("#FFFFFF")
        
        # === Animation for Lecture Line 2 ===
        # Columns of P are new basis vectors
        # Note: MathTex indexing is fragile, selecting specific parts based on structure
        col1 = matrix_p[0][6:8] 
        col2 = matrix_p[0][8:10]
        self.play(col1.animate.set_color("#00FF00"), col2.animate.set_color("#00FF00"))
        self.lecture[1].set_color("#00FF00")
        
        # === Animation for Lecture Line 3 ===
        # Apply [v]_old = P * [v]_new
        equation = MathTex("[v]_{old} = P \\cdot [v]_{new}")
        self.place_at_grid(equation, "C3", scale_factor=0.7)
        self.play(Write(equation))
        self.lecture[2].set_color("#FFFF00")
        
        # === Animation for Lecture Line 4 ===
        # Visualize the mapping process
        axes = Axes(x_range=[-2, 2], y_range=[-2, 2], axis_config={"include_tip": False})
        self.place_in_area(axes, "B4", "D6", scale_factor=0.6)
        dot = Dot(color=RED).move_to(axes.c2p(1, 1))
        self.play(Create(axes), FadeIn(dot))
        self.lecture[3].set_color("#00FFFF")
        
        # === Animation for Lecture Line 5 ===
        # Example P
        example = Text("P maps basis vectors.", font_size=20, color=ORANGE)
        self.place_at_grid(example, "E2", scale_factor=0.8)
        self.play(Write(example))
        self.lecture[4].set_color(ORANGE)
        
        self.wait(2)
