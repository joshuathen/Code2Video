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
            "A basis B defines our coordinate system.",
            "Any vector is a linear combination of B.",
            "The Change of Basis matrix P is a dictionary.",
            "P maps B-basis coordinates to standard ones.",
            "P uses basis vectors as columns."
        ]
        self.setup_layout("Defining the Basis Matrix", lecture_lines)
        
        # Define objects
        vec_i = Vector([1, 0.5], color=YELLOW)
        vec_j = Vector([-0.5, 1], color=YELLOW)
        matrix_m = MathTex(r"P = \begin{bmatrix} 1 & -0.5 \\ 0.5 & 1 \end{bmatrix}", color=WHITE)
        dictionary_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/dictionary.svg")
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.place_at_grid(vec_i, 'C2', scale_factor=0.8)
        self.place_at_grid(vec_j, 'C4', scale_factor=0.8)
        self.play(Create(vec_i), Create(vec_j))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.place_in_area(matrix_m, 'E3', 'E5', scale_factor=0.9)
        self.place_at_grid(dictionary_icon, 'E1', scale_factor=0.5)
        self.play(FadeIn(matrix_m), FadeIn(dictionary_icon))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(WHITE))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(GREEN))
        new_vec_i = Vector([0.8, 0.8], color=GREEN)
        new_vec_j = Vector([-0.8, 0.8], color=GREEN)
        self.play(
            ReplacementTransform(vec_i, new_vec_i),
            ReplacementTransform(vec_j, new_vec_j)
        )
        self.wait(2)
