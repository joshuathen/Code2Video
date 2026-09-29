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
        lecture_lines = [
            "Tall matrices have more rows than columns.",
            "They embed lower dimensions into higher ones.",
            "Think of a 2D plane in 3D.",
            "A simple stick figure in 3D space.",
            "Represented by a 3x2 matrix transformation."
        ]
        self.setup_layout("Case 1: Dimensional Expansion (Tall Matrices)", lecture_lines)
        
        # Matrix visualization
        matrix = MathTex(r"\begin{pmatrix} a & b \\ c & d \\ e & f \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} x' \\ y' \\ z' \end{pmatrix}")
        self.place_at_grid(matrix, 'B2', scale_factor=0.6)
        
        # Setup 3D space for subspace visualization
        axes = ThreeDAxes(x_range=[-2, 2], y_range=[-2, 2], z_range=[-2, 2], axis_config={"include_tip": True})
        axes.set_color(GRAY)
        self.place_at_grid(axes, 'E3', scale_factor=0.5)
        
        # Grid manifold (embedded)
        grid_plane = NumberPlane(x_range=[-1, 1], y_range=[-1, 1], background_line_style={"stroke_color": BLUE, "stroke_opacity": 0.5})
        self.place_in_area(grid_plane, 'D4', 'F6', scale_factor=0.4)
        grid_plane.rotate(PI/4, axis=RIGHT)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(Write(matrix))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.play(FadeIn(axes))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.play(Create(grid_plane))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(YELLOW))
        # Simple stick figure as line
        stick_fig = VGroup(Line(UP*0.5, DOWN*0.5), Line(DOWN*0.5, DOWN*0.8+LEFT*0.3), Line(DOWN*0.5, DOWN*0.8+RIGHT*0.3)).set_color(RED)
        self.place_at_grid(stick_fig, 'E5', scale_factor=0.4)
        self.play(FadeIn(stick_fig))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(YELLOW))
        self.play(matrix.animate.set_color(GREEN))
        self.wait(1)
