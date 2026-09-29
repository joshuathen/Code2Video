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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisite: The Mapping Concept", 
                          ["Linear transformations map between vector spaces.", 
                           "An m by n matrix defines this mapping.", 
                           "We move from n dimensions to m dimensions."])
        
        # Objects
        axes = Axes(x_range=[-3, 3], y_range=[-3, 3], axis_config={"include_tip": True})
        vec_v = Vector([1, 2], color="#FF5733")
        label_v = MathTex(r"\\vec{v}", color="#FF5733").next_to(vec_v.get_end(), UP)
        
        matrix_m = MathTex(r"M = \\begin{pmatrix} 1 & 0 \\\\ 0 & 1 \\end{pmatrix}", color="#33FF57")
        
        vec_mv = Vector([1.5, 1], color="#3357FF")
        label_mv = MathTex(r"M\\vec{v}", color="#3357FF")
        
        # === Animation for Lecture Line 1 ===
        self.place_in_area(axes, 'B3', 'E5', scale_factor=0.6)
        self.add(axes)
        self.play(Create(vec_v), Write(label_v))
        self.lecture[0].set_color("#FF5733")

        # === Animation for Lecture Line 2 ===
        self.place_at_grid(matrix_m, 'A5', scale_factor=0.6)
        self.play(FadeIn(matrix_m))
        self.lecture[1].set_color("#33FF57")

        # === Animation for Lecture Line 3 ===
        self.place_at_grid(label_mv, 'C4', scale_factor=0.7)
        self.play(Create(vec_mv), Write(label_mv))
        self.lecture[2].set_color("#3357FF")
        self.wait(2)
