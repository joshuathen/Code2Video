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

class Section5Scene(TeachingScene):
    def construct(self):
        # Setup content
        lecture_lines = [
            'Matrix P stores the new basis vectors as columns.',
            'We represent the new coordinates as a column vector.',
            'Multiplying by P translates between these two languages.',
            'The calculation returns our familiar standard coordinates.',
            'Both coordinate systems describe the exact same point.'
        ]
        self.setup_layout("The Transition Matrix (The Translator)", lecture_lines)

        # Colors
        COLOR_P = "#FFFF00"  # Yellow
        COLOR_B1 = "#FF5555" # Light Red
        COLOR_B2 = "#5555FF" # Light Blue
        COLOR_COORD = "#55FF55" # Light Green

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Basis vector labels (A2, B2 as per Issue 48)
        b1_vec_text = Text("b1 = [2, 1]", color=COLOR_B1, font_size=24)
        b2_vec_text = Text("b2 = [-1, 1]", color=COLOR_B2, font_size=24)
        self.place_at_grid(b1_vec_text, "A2", scale_factor=0.7)
        self.place_at_grid(b2_vec_text, "B2", scale_factor=0.7)
        
        # Matrix P Shell (A3 to B5 as per Issue 48)
        p_label = Text("P =", color=COLOR_P, font_size=32)
        matrix_brackets = Text("[       ]", color=WHITE, font_size=40)
        p_matrix_group = VGroup(p_label, matrix_brackets).arrange(RIGHT, buff=0.2)
        self.place_in_area(p_matrix_group, "A3", "B5", scale_factor=1.0)
        
        # Matrix Entries
        col1 = Text("2\n1", color=COLOR_B1, font_size=24, line_spacing=0.5)
        col2 = Text("-1\n1", color=COLOR_B2, font_size=24, line_spacing=0.5)
        col1.move_to(matrix_brackets.get_center() + LEFT * 0.4)
        col2.move_to(matrix_brackets.get_center() + RIGHT * 0.4)
        
        self.play(FadeIn(p_matrix_group))
        self.play(Write(b1_vec_text), Write(b2_vec_text))
        self.wait(0.5)
        
        # Assembly animation
        self.play(
            ReplacementTransform(b1_vec_text.copy(), col1),
            ReplacementTransform(b2_vec_text.copy(), col2),
            b1_vec_text.animate.set_opacity(0.3),
            b2_vec_text.animate.set_opacity(0.3)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        # Vector [1, 2]_B as a column (B6 as per Issue 48)
        lb = Text("[", color=WHITE, font_size=40).scale(1.2)
        rb = Text("]", color=WHITE, font_size=40).scale(1.2)
        vals = Text("1\n2", color=COLOR_COORD, font_size=24, line_spacing=0.5)
        coord_b_vgroup = VGroup(lb, vals, rb).arrange(RIGHT, buff=0.1)
        coord_b_sub = Text("B", color=COLOR_COORD, font_size=16)
        coord_b_label = VGroup(coord_b_vgroup, coord_b_sub).arrange(RIGHT, aligned_edge=DOWN, buff=0.05)
        
        self.place_at_grid(coord_b_label, "B6", scale_factor=0.9)
        
        self.play(Write(coord_b_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        # Move vector next to P to show P * [1, 2]
        multi_p = VGroup(p_matrix_group, col1, col2)
        self.play(
            coord_b_label.animate.next_to(multi_p, RIGHT, buff=0.1)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)

        # Step-by-step arithmetic (Row C as per Issue 48)
        calc = Text("= 1 * [2, 1] + 2 * [-1, 1]", font_size=20, color=WHITE)
        result = Text("= [0, 3] Std", color=WHITE, font_size=24)
        
        self.place_in_area(calc, "C3", "C5", scale_factor=0.8)
        self.place_at_grid(result, "C6", scale_factor=0.8)

        self.play(Write(calc))
        self.wait(1)
        self.play(Write(result))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)

        # Visual alignment on grid
        # Group everything to move it up
        equation_all = VGroup(
            multi_p, coord_b_label, calc, result, b1_vec_text, b2_vec_text
        )

        plane = NumberPlane(
            x_range=[-3, 3, 1], y_range=[-1, 5, 1],
            background_line_style={"stroke_opacity": 0.2},
            axis_config={"stroke_width": 1}
        )
        # Using Area D2-F6 as per Issue 48
        self.place_in_area(plane, "D2", "F6", scale_factor=0.6)

        # Basis and components
        b1_vec = Arrow(plane.c2p(0,0), plane.c2p(2,1), buff=0, color=COLOR_B1, stroke_width=3)
        b2_vec = Arrow(plane.c2p(0,0), plane.c2p(-1,1), buff=0, color=COLOR_B2, stroke_width=3)
        comp1 = Arrow(plane.c2p(0,0), plane.c2p(2,1), buff=0, color=COLOR_B1, stroke_width=2, stroke_opacity=0.5)
        comp2 = Arrow(plane.c2p(2,1), plane.c2p(0,3), buff=0, color=COLOR_B2, stroke_width=2, stroke_opacity=0.5)
        
        target_point = Dot(plane.c2p(0, 3), color=COLOR_COORD)
        point_label = Text("(1, 2)B = (0, 3)Std", font_size=16, color=COLOR_COORD)
        point_label.next_to(target_point, UR, buff=0.1)

        # Reposition equation to top area to clear space for the plane
        self.play(
            equation_all.animate.scale(0.7).move_to(self.grid["A4"]),
            FadeIn(plane)
        )
        self.play(Create(b1_vec), Create(b2_vec))
        self.wait(0.5)
        self.play(Create(comp1))
        self.play(Create(comp2))
        self.play(Create(target_point), Write(point_label))
        
        self.wait(2)
        self.lecture[4].set_color(WHITE)
        self.wait(1)
