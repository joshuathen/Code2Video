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
        # Title and Lecture Lines from Storyboard
        title_text = "The Solution: The Transition Matrix"
        lecture_lines = [
            "- Describe the new basis using our standard coordinates.",
            "- Place these vectors into the columns of matrix P.",
            "- This formula defines the conversion between coordinate systems.",
            "- Matrix P transforms new coordinates back to old ones.",
            "- The columns are the new basis in old terms."
        ]
        
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Describe the new basis using our standard coordinates.
        self.lecture[0].set_color(YELLOW)
        
        plane = NumberPlane(
            x_range=[-3, 3, 1], y_range=[-3, 3, 1],
            x_length=4, y_length=4,
            background_line_style={"stroke_opacity": 0.4}
        )
        
        b1 = Arrow(plane.c2p(0,0), plane.c2p(1,1), buff=0, color=GREEN)
        b2 = Arrow(plane.c2p(0,0), plane.c2p(-1,1), buff=0, color=RED)
        b1_label = Text("b1=(1,1)", font_size=18, color=GREEN)
        b2_label = Text("b2=(-1,1)", font_size=18, color=RED)
        
        b1_label.next_to(b1.get_end(), UR, buff=0.1)
        b2_label.next_to(b2.get_end(), UL, buff=0.1)
        
        standard_basis_group = VGroup(plane, b1, b2, b1_label, b2_label)
        # Issue 32: self.place_in_area(standard_basis_group, 'A2', 'D5', scale_factor=0.8)
        self.place_in_area(standard_basis_group, 'A2', 'D5', scale_factor=0.8)
        
        self.play(Create(plane))
        self.play(GrowArrow(b1), GrowArrow(b2))
        self.play(Write(b1_label), Write(b2_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Place these vectors into the columns of matrix P.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        p_label = Text("P = ", font_size=24, color="#FFFF00")
        v11 = Text("1", font_size=24, color=GREEN)
        v12 = Text("-1", font_size=24, color=RED)
        v21 = Text("1", font_size=24, color=GREEN)
        v22 = Text("1", font_size=24, color=RED)
        p_matrix_vals = VGroup(v11, v12, v21, v22).arrange_in_grid(rows=2, cols=2, buff=0.4)
        
        bracket_l = Text("[", font_size=40, color="#FFFF00").next_to(p_matrix_vals, LEFT, buff=0.1)
        bracket_r = Text("]", font_size=40, color="#FFFF00").next_to(p_matrix_vals, RIGHT, buff=0.1)
        p_matrix_group = VGroup(p_label, bracket_l, p_matrix_vals, bracket_r).arrange(RIGHT, buff=0.2)
        
        formula = Text("[v]s = P [v]b", font_size=24, color="#00FFFF")
        matrix_formula_group = VGroup(formula, p_matrix_group).arrange(DOWN, buff=0.5)
        # Issue 34: self.place_in_area(matrix_formula_group, 'E2', 'F5', scale_factor=0.75)
        self.place_in_area(matrix_formula_group, 'E2', 'F5', scale_factor=0.75)
        
        self.play(Write(p_label), Write(bracket_l), Write(bracket_r))
        self.play(
            TransformFromCopy(b1_label, v11),
            TransformFromCopy(b1_label, v21),
            TransformFromCopy(b2_label, v12),
            TransformFromCopy(b2_label, v22)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # This formula defines the conversion between coordinate systems.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Matrix P transforms new coordinates back to old ones.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        v_target = plane.c2p(2,2)
        v_vec = Arrow(plane.c2p(0,0), v_target, buff=0, color=ORANGE)
        v_b_label = Text("[2,0]b", font_size=18, color=ORANGE).next_to(v_target, DR, buff=0.1)
        v_s_label = Text("[2,2]s", font_size=18, color=ORANGE).next_to(v_target, UR, buff=0.1)
        
        self.play(GrowArrow(v_vec), Write(v_b_label))
        self.wait(1)
        self.play(Transform(v_b_label, v_s_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # The columns are the new basis in old terms.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        col1_rect = SurroundingRectangle(VGroup(v11, v21), color=GREEN, buff=0.1)
        col2_rect = SurroundingRectangle(VGroup(v12, v22), color=RED, buff=0.1)
        
        self.play(Create(col1_rect))
        self.play(Indicate(col1_rect), Indicate(b1))
        self.play(ReplacementTransform(col1_rect, col2_rect))
        self.play(Indicate(col2_rect), Indicate(b2))
        self.play(FadeOut(col2_rect))
        self.wait(2)
