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
        self.setup_layout("The Transformation Matrix (The 'Recipe')", [
            "Matrix columns show where basis lands.",
            "This defines the entire space transformation.",
            "Think of it as a movement recipe.",
            "The robot arm tracks its axes.",
            "Final positions form the matrix columns."
        ])

        matrix = Matrix([[r"a", r"b"], [r"c", r"d"]], element_alignment_corner=ORIGIN)
        matrix.set_color("#FFFFFF")
        # Applying requested alignment: balanced placement at C4
        self.place_at_grid(matrix, 'C4', scale_factor=0.9)
        
        # === Animation for Lecture Line 1 ===
        self.play(Write(matrix))
        self.lecture[0].set_color("#FFD700")

        # === Animation for Lecture Line 2 ===
        i_hat_dest = Dot(self.grid['B5'], color="#FFD700")
        j_hat_dest = Dot(self.grid['D5'], color="#FFD700")
        i_label = Text("i-hat", font_size=18).next_to(i_hat_dest, UP)
        j_label = Text("j-hat", font_size=18).next_to(j_hat_dest, UP)
        self.play(FadeIn(i_hat_dest), FadeIn(j_hat_dest), Write(i_label), Write(j_label))
        self.lecture[1].set_color("#FFD700")

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFD700")
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        col1 = matrix.get_columns()[0]
        col2 = matrix.get_columns()[1]
        self.play(col1.animate.set_color("#00FF00"), col2.animate.set_color("#00FF00"))
        
        robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        self.place_at_grid(robot, 'F3', scale_factor=0.5)
        self.play(FadeIn(robot))
        
        self.lecture[3].set_color("#00FF00")

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFFFFF")
        self.wait(2)
