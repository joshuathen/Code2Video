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
        # Retrieve lecture lines and title
        title_text = "Matrices as Space Morphing"
        lecture_lines = [
            "A matrix is a function that transforms space.",
            "It moves the entire grid in a linear way.",
            "The columns tell us where basis vectors land.",
            "See i-hat and j-hat move to their new positions.",
            "The rest of the grid follows their lead perfectly."
        ]
        
        self.setup_layout(title_text, lecture_lines)

        # Color constants
        I_HAT_COLOR = "#FF0000"
        J_HAT_COLOR = "#00FF00"
        HIGHLIGHT_COLOR = "#FFFF00"
        MATRIX_COLOR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Line: "A matrix is a function that transforms space."
        self.lecture[0].set_color(WHITE)
        
        grid = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            background_line_style={"stroke_color": WHITE, "stroke_width": 1, "stroke_opacity": 0.4}
        )
        # Position grid in the lower-right area (C2 to F6) - Fix for Issue 25, 26, 35
        self.place_in_area(grid, "C2", "F6", scale_factor=0.65)
        
        # Origin and unit vectors based on grid mapping
        origin = grid.c2p(0, 0)
        i_tip = grid.c2p(1, 0)
        j_tip = grid.c2p(0, 1)
        
        i_hat = Arrow(origin, i_tip, color=I_HAT_COLOR, buff=0, stroke_width=4)
        j_hat = Arrow(origin, j_tip, color=J_HAT_COLOR, buff=0, stroke_width=4)
        
        i_label = MathTex(r"\hat{i}", color=I_HAT_COLOR, font_size=24)
        j_label = MathTex(r"\hat{j}", color=J_HAT_COLOR, font_size=24)
        
        i_label.next_to(i_tip, RIGHT, buff=0.1)
        j_label.next_to(j_tip, UP, buff=0.1)
        
        self.play(Create(grid), run_time=1)
        self.play(Create(i_hat), Create(j_hat), Write(i_label), Write(j_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line: "It moves the entire grid in a linear way."
        self.lecture[1].set_color(WHITE)
        
        # Matrix [[0, -1], [1, 0]] (Column representation: col1=[0,1], col2=[-1,0])
        matrix_mobject = Matrix([[0, -1], [1, 0]], 
                                left_bracket="[", right_bracket="]",
                                element_to_mobject_config={"color": MATRIX_COLOR})
        # Place matrix at the top of the right side area (A3 to A5) - Fix for Issue 24, 35
        self.place_in_area(matrix_mobject, "A3", "A5", scale_factor=0.6)
        
        self.play(Write(matrix_mobject))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line: "The columns tell us where basis vectors land."
        self.lecture[2].set_color(HIGHLIGHT_COLOR)
        
        # Highlight first column [0, 1]
        col1 = matrix_mobject.get_columns()[0]
        rect1 = SurroundingRectangle(col1, color=HIGHLIGHT_COLOR, buff=0.1)
        
        self.play(Create(rect1))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Line: "See i-hat and j-hat move to their new positions."
        self.lecture[3].set_color(HIGHLIGHT_COLOR)
        
        # i-hat moves to (0,1) according to column 1
        new_i_tip = grid.c2p(0, 1)
        self.play(
            i_hat.animate.put_start_and_end_on(origin, new_i_tip),
            i_label.animate.next_to(new_i_tip, UP, buff=0.1),
            run_time=1.5
        )
        
        # Transition highlight to second column [-1, 0]
        col2 = matrix_mobject.get_columns()[1]
        rect2 = SurroundingRectangle(col2, color=HIGHLIGHT_COLOR, buff=0.1)
        
        self.play(ReplacementTransform(rect1, rect2))
        
        # j-hat moves to (-1,0) according to column 2
        new_j_tip = grid.c2p(-1, 0)
        self.play(
            j_hat.animate.put_start_and_end_on(origin, new_j_tip),
            j_label.animate.next_to(new_j_tip, LEFT, buff=0.1),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Line: "The rest of the grid follows their lead perfectly."
        self.lecture[4].set_color(HIGHLIGHT_COLOR)
        
        # Morph the background grid to match the transformation
        # Transformation matrix: [[0, -1], [1, 0]]
        matrix_np = np.array([[0, -1], [1, 0]])
        
        self.play(FadeOut(rect2))
        self.play(
            grid.animate.apply_matrix(matrix_np),
            run_time=2
        )
        self.wait(2)
