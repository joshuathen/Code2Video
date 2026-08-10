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
        self.setup_layout("The Mechanism of Coordinate Transformation", 
                          ["We map old to new coordinates.", 
                           "The change-of-basis matrix is P.", 
                           "P columns are new basis vectors.", 
                           "Robot dogs use P to navigate.", 
                           "Matrix P translates local steps globally."])
        
        # === Animation for Lecture Line 1 ===
        # v_new = P v_old
        equation = MathTex(r"v_{new} = P v_{old}", color=WHITE)
        self.place_at_grid(equation, 'B2', scale_factor=0.8)
        self.play(Write(equation))
        self.lecture[0].set_color(WHITE)

        # === Animation for Lecture Line 2 ===
        # P in yellow
        matrix_p = MathTex(r"P", color="#FFFF00")
        self.place_at_grid(matrix_p, 'C3', scale_factor=1.2)
        self.play(Indicate(matrix_p))
        self.lecture[1].set_color("#FFFF00")

        # === Animation for Lecture Line 3 ===
        # P columns are new basis vectors
        matrix_p_cols = MathTex(r"P = \begin{bmatrix} \vec{i}' & \vec{j}' \end{bmatrix}", color="#FFFF00")
        self.place_at_grid(matrix_p_cols, 'B4', scale_factor=0.7)
        self.play(ReplacementTransform(matrix_p, matrix_p_cols))
        self.lecture[2].set_color("#FFFF00")

        # === Animation for Lecture Line 4 ===
        # Robot Navigator + Asset
        robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        nav_text = Text("Robot Navigator", color="#87CEFA", font_size=20)
        group = VGroup(robot, nav_text).arrange(DOWN)
        self.place_at_grid(group, 'D5', scale_factor=0.6)
        self.play(FadeIn(group))
        self.lecture[3].set_color("#87CEFA")

        # === Animation for Lecture Line 5 ===
        # Robot navigating with matrix P
        vec = Vector(UP * 1 + RIGHT * 1, color="#00FF00")
        self.place_at_grid(vec, 'C5', scale_factor=0.9)
        grid = NumberPlane(x_range=[-3, 3], y_range=[-3, 3], background_line_style={"stroke_color": "#4682B4", "stroke_opacity": 0.5})
        self.place_in_area(grid, 'A3', 'F6', scale_factor=0.4)
        
        self.play(Create(grid), Create(vec))
        self.lecture[4].set_color("#00FF00")
        self.wait(1)
