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

class Section6Scene(TeachingScene):
    def create_text_matrix(self, data, color=WHITE, font_size=24):
        """
        Creates a matrix representation using Text mobjects to bypass TeX requirements.
        """
        rows = []
        for r in data:
            row_items = VGroup(*[
                Text(str(item), font_size=font_size, color=color) for item in r
            ]).arrange(RIGHT, buff=0.6)
            rows.append(row_items)
        
        matrix_content = VGroup(*rows).arrange(DOWN, buff=0.5)
        bracket_l = Text("[", font_size=font_size * 2.5, color=color).next_to(matrix_content, LEFT, buff=0.2)
        bracket_r = Text("]", font_size=font_size * 2.5, color=color).next_to(matrix_content, RIGHT, buff=0.2)
        return VGroup(bracket_l, matrix_content, bracket_r)

    def construct(self):
        # Initial Setup
        title = "Summary: The DNA of Transformation"
        lines = [
            "Matrix multiplication is just the composition of movements.",
            "One matrix captures a sequence of complex transformations.",
            "Think of products as instructions for a final destination."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # 3 matrices sequence: A * B * C
        # A: Shear, B: Scale, C: Rotation
        m_a_data = [[1, 1], [0, 1]]
        m_b_data = [[2, 0], [0, 2]]
        m_c_data = [[0, -1], [1, 0]]
        
        m1 = self.create_text_matrix(m_a_data, color=BLUE_B)
        m2 = self.create_text_matrix(m_b_data, color=BLUE_C)
        m3 = self.create_text_matrix(m_c_data, color=BLUE_D)
        
        dot1 = Text("×", font_size=24).next_to(m1, RIGHT, buff=0.2)
        dot2 = Text("×", font_size=24).next_to(m2, RIGHT, buff=0.2)
        
        matrix_group = VGroup(m1, dot1, m2, dot2, m3).arrange(RIGHT, buff=0.2)
        self.place_in_area(matrix_group, 'B2', 'E6', scale_factor=0.9)
        
        matrix_label = Text("Composition of Actions", font_size=28, color=YELLOW)
        self.place_in_area(matrix_label, 'B2', 'B5', scale_factor=0.8)
        
        self.play(FadeIn(matrix_label))
        self.play(Write(matrix_group))
        self.wait(1)
        
        # Multiply them into a Master Matrix
        master_data = [[2, -2], [2, 0]]
        matrix_a = self.create_text_matrix(master_data, color=GREEN_B)
        self.place_in_area(matrix_a, 'C2', 'E5', scale_factor=1.0)
        
        master_label = Text("Master Matrix (M)", font_size=28, color=GREEN_B)
        self.place_in_area(master_label, 'B2', 'B5', scale_factor=0.8)

        self.play(
            ReplacementTransform(matrix_group, matrix_a),
            ReplacementTransform(matrix_label, master_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(TEAL)
        
        # Complex geometric pattern (kaleidoscope-like)
        pattern = VGroup()
        for i in range(8):
            poly = RegularPolygon(n=4, radius=0.5, color=TEAL, stroke_width=2)
            poly.rotate(i * PI/4)
            pattern.add(poly)
        
        self.place_at_grid(pattern, "D4", scale_factor=1.0)
        self.play(FadeIn(pattern))
        self.wait(0.5)
        
        # One-step transformation using the Master Matrix
        self.play(
            pattern.animate.apply_matrix([[2, -2], [2, 0]]),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(WHITE) 
        
        # End with final summary text
        conclusion = Text("Matrix multiplication is composition", font_size=32, color=WHITE)
        self.place_in_area(conclusion, 'F1', 'F6', scale_factor=1.0)
        
        self.play(Write(conclusion))
        self.play(Indicate(conclusion, color=YELLOW))
        self.wait(2)
