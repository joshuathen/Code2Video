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
        # Setup the layout with lecture lines and title
        title = "The Mechanics: Basis Vector Tracking"
        lines = [
            "To find BA, track the basis vectors.",
            "Watch where i-hat lands after both steps.",
            "Then, track where j-hat ends up.",
            "These final positions form the columns of BA.",
            "This explains the 'row by column' rule."
        ]
        self.setup_layout(title, lines)

        # Color definitions
        I_COLOR = "#FF0000" # Red
        J_COLOR = "#00FF00" # Green
        HIGHLIGHT_COLOR = "#FFFF00" # Yellow
        
        # Matrices for demonstration
        # A: Rotation 90 degrees CCW
        matrix_a = [[0, -1], [1, 0]]
        # B: Scale x by 2
        matrix_b = [[2, 0], [0, 1]]
        
        # === Animation for Lecture Line 1 ===
        # Intro: Set up plane and basis vectors
        plane = NumberPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            x_length=3,
            y_length=3,
            background_line_style={"stroke_opacity": 0.4}
        )
        self.place_in_area(plane, 'B2', 'D4', scale_factor=0.8)
        
        i_hat = Vector([1, 0], color=I_COLOR)
        j_hat = Vector([0, 1], color=J_COLOR)
        
        # Shift vectors to plane origin
        plane_origin = plane.get_origin()
        i_hat.shift(plane_origin)
        j_hat.shift(plane_origin)
        
        # Avoid MathTex by using Text (Fixes FileNotFoundError: latex)
        i_label = Text("i", slant=ITALIC, color=I_COLOR, font_size=24).next_to(i_hat, DOWN, buff=0.1)
        j_label = Text("j", slant=ITALIC, color=J_COLOR, font_size=24).next_to(j_hat, LEFT, buff=0.1)

        self.play(
            FadeIn(plane),
            Create(i_hat), Create(j_hat),
            Write(i_label), Write(j_label),
            self.lecture[0].animate.set_color(HIGHLIGHT_COLOR)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Watch i-hat: A then B
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(I_COLOR)
        )
        
        # Transform A: Rotation
        self.play(
            i_hat.animate.apply_matrix(matrix_a, about_point=plane_origin),
            i_label.animate.next_to(plane_origin + UP, RIGHT, buff=0.1),
            run_time=1.5
        )
        # Transform B: Scale
        self.play(
            i_hat.animate.apply_matrix(matrix_b, about_point=plane_origin),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Watch j-hat: A then B
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(J_COLOR)
        )
        
        # Transform A: Rotation
        self.play(
            j_hat.animate.apply_matrix(matrix_a, about_point=plane_origin),
            j_label.animate.next_to(plane_origin + LEFT, DOWN, buff=0.1),
            run_time=1.5
        )
        # Transform B: Scale
        self.play(
            j_hat.animate.apply_matrix(matrix_b, about_point=plane_origin),
            j_label.animate.next_to(plane_origin + LEFT * 2, DOWN, buff=0.1),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Columns of BA
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(HIGHLIGHT_COLOR)
        )
        
        # Build Matrix result manually using Text to avoid LaTeX
        m00 = Text("0", font_size=24, color=I_COLOR)
        m10 = Text("1", font_size=24, color=I_COLOR)
        m01 = Text("-2", font_size=24, color=J_COLOR)
        m11 = Text("0", font_size=24, color=J_COLOR)
        
        col1 = VGroup(m00, m10).arrange(DOWN, buff=0.4)
        col2 = VGroup(m01, m11).arrange(DOWN, buff=0.4)
        matrix_grid = VGroup(col1, col2).arrange(RIGHT, buff=0.6)
        
        l_bracket = Text("[", font_size=40).next_to(matrix_grid, LEFT, buff=0.1)
        r_bracket = Text("]", font_size=40).next_to(matrix_grid, RIGHT, buff=0.1)
        res_matrix = VGroup(matrix_grid, l_bracket, r_bracket)
        
        res_label = Text("BA =", font_size=28)
        full_res = VGroup(res_label, res_matrix).arrange(RIGHT, buff=0.2)
        
        self.place_in_area(full_res, 'E2', 'F5', scale_factor=0.9)
        
        self.play(Write(full_res))
        
        # Highlight columns matching vector colors
        self.play(Indicate(col1, color=I_COLOR))
        self.play(Indicate(col2, color=J_COLOR))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Conclusion
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(HIGHLIGHT_COLOR)
        )
        self.wait(3)
