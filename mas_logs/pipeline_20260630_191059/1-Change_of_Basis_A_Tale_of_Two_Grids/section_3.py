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

class TextMatrix(VGroup):
    """
    A custom Matrix replacement using Text brackets to avoid LaTeX dependencies.
    """
    def __init__(self, mob_matrix, left_bracket="[", right_bracket="]"):
        super().__init__()
        self.mob_matrix = mob_matrix
        # Flatten entries for the elements VGroup
        self.elements = VGroup(*[mob for row in mob_matrix for mob in row])
        # Arrange in grid (rows, cols)
        self.elements.arrange_in_grid(rows=len(mob_matrix), cols=len(mob_matrix[0]), buff=(0.6, 0.4))
        self.add(self.elements)
        
        # Create brackets using Text instead of MathTex
        self.l_bracket = Text(left_bracket).scale(2)
        self.r_bracket = Text(right_bracket).scale(2)
        
        # Adjust height to fit the entries
        self.l_bracket.stretch_to_fit_height(self.elements.height + 0.3)
        self.r_bracket.stretch_to_fit_height(self.elements.height + 0.3)
        
        self.l_bracket.next_to(self.elements, LEFT, buff=0.15)
        self.r_bracket.next_to(self.elements, RIGHT, buff=0.15)
        
        self.add(self.l_bracket, self.r_bracket)

    def get_columns(self):
        """Returns a VGroup of VGroups, where each internal VGroup is a column of entries."""
        return VGroup(*[
            VGroup(*[self.mob_matrix[i][j] for i in range(len(self.mob_matrix))])
            for j in range(len(self.mob_matrix[0]))
        ])

class Section3Scene(TeachingScene):
    def construct(self):
        # 1. Fetch info
        title = "Introducing the Transition Matrix"
        lines = [
            "We describe Bob's basis vectors using Alice's grid.",
            "These descriptions become the columns of transition matrix P.",
            "This matrix P translates between Bob and Alice."
        ]
        
        self.setup_layout(title, lines)
        
        # Define colors
        b1_color = "#FFFF00"  # Yellow
        b2_color = "#FF00FF"  # Magenta
        alice_color = BLUE_D
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Create Alice's grid (NumberPlane)
        alice_grid = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            background_line_style={"stroke_color": alice_color, "stroke_opacity": 0.4},
            axis_config={"stroke_color": alice_color},
            x_length=4,
            y_length=4
        )
        # Fix Issue 36: Scaling and positioning the grid
        self.place_in_area(alice_grid, 'B1', 'E4', scale_factor=0.85)
        self.play(Create(alice_grid))

        # Integration of Asset (Issue 27)
        bob_icon = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/bob.svg")
        self.place_at_grid(bob_icon, "A1", scale_factor=0.5)
        bob_label = Text("Bob", font_size=18).next_to(bob_icon, DOWN, buff=0.1)
        self.play(FadeIn(bob_icon), Write(bob_label))
        
        # Create Bob's basis vectors
        b1_vec = Arrow(alice_grid.c2p(0, 0), alice_grid.c2p(2, 1), buff=0, color=b1_color, stroke_width=4)
        b2_vec = Arrow(alice_grid.c2p(0, 0), alice_grid.c2p(-1, 1), buff=0, color=b2_color, stroke_width=4)
        
        b1_label = Text("b1", color=b1_color, font_size=24).next_to(b1_vec.get_end(), UR, buff=0.1)
        b2_label = Text("b2", color=b2_color, font_size=24).next_to(b2_vec.get_end(), UL, buff=0.1)
        
        self.play(GrowArrow(b1_vec), Write(b1_label))
        self.play(GrowArrow(b2_vec), Write(b2_label))
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Create matrix P using custom TextMatrix
        m_entries = [
            [Text("2", color=b1_color, font_size=32), Text("-1", color=b2_color, font_size=32)],
            [Text("1", color=b1_color, font_size=32), Text("1", color=b2_color, font_size=32)]
        ]
        matrix_p = TextMatrix(m_entries)
        p_label = Text("P =", font_size=32, color=WHITE).next_to(matrix_p, LEFT)
        matrix_group = VGroup(p_label, matrix_p)
        # Fix Issue 35: Positioning matrix_group
        self.place_at_grid(matrix_group, 'C6', scale_factor=0.7)
        
        # Show coordinate vectors momentarily near vectors
        b1_coords = TextMatrix(
            [[Text("2", font_size=24)], [Text("1", font_size=24)]],
        ).set_color(b1_color).scale(0.8).next_to(b1_vec.get_end(), DR, buff=0.2)
        
        b2_coords = TextMatrix(
            [[Text("-1", font_size=24)], [Text("1", font_size=24)]],
        ).set_color(b2_color).scale(0.8).next_to(b2_vec.get_end(), DL, buff=0.2)
        
        self.play(Write(b1_coords), Write(b2_coords))
        self.wait(0.5)
        
        # Reveal Matrix skeleton
        self.play(FadeIn(p_label), FadeIn(matrix_p.l_bracket), FadeIn(matrix_p.r_bracket))
        
        # Animate coordinates into matrix columns
        self.play(
            ReplacementTransform(b1_coords, matrix_p.get_columns()[0]),
            ReplacementTransform(b2_coords, matrix_p.get_columns()[1]),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        col1 = matrix_p.get_columns()[0]
        col2 = matrix_p.get_columns()[1]
        
        self.play(
            Flash(b1_vec, color=b1_color, line_length=0.3),
            Flash(col1, color=b1_color, line_length=0.3)
        )
        self.wait(0.5)
        self.play(
            Flash(b2_vec, color=b2_color, line_length=0.3),
            Flash(col2, color=b2_color, line_length=0.3)
        )
        
        self.wait(2)
