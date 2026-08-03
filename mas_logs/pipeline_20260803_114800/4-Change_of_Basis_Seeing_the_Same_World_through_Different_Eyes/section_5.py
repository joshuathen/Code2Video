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
        # Initialize Layout
        title_text = "Going Backwards: The Inverse Matrix"
        lecture_lines = [
            "To speak Bob's language, use the inverse matrix.",
            "The inverse P reverses the coordinate translation process.",
            "Now Alice can describe points for Bob's map."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        ALICE_COLOR = WHITE
        BOB_COLOR = BLUE
        INVERSE_COLOR = "#00FFFF" # Cyan
        HIGHLIGHT_COLOR = YELLOW

        # Matrix P maps Bob to Alice: [1, -0.5], [0.5, 1]
        # P^-1 maps Alice to Bob: [0.8, 0.4], [-0.4, 0.8]
        # (4, 2)_Alice -> (4, 0)_Bob
        matrix_P = [[1, -0.5], [0.5, 1]]

        # === Animation for Lecture Line 1 ===
        # Show Alice's point (4, 2) on her standard white grid.
        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        
        alice_grid = NumberPlane(
            x_range=[-1, 5, 1],
            y_range=[-1, 4, 1],
            background_line_style={"stroke_color": ALICE_COLOR, "stroke_opacity": 0.3},
            axis_config={"stroke_color": ALICE_COLOR}
        )
        self.place_in_area(alice_grid, "B1", "F6", scale_factor=0.5)
        
        dot = Dot(alice_grid.c2p(4, 2), color=ALICE_COLOR)
        dot_label = MathTex("(4, 2)_{Alice}", font_size=24, color=ALICE_COLOR)
        dot_label.next_to(dot, UR, buff=0.1)
        
        self.play(Create(alice_grid), FadeIn(dot), Write(dot_label))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Introduce the inverse matrix P^-1 in cyan (#00FFFF) next to point.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(HIGHLIGHT_COLOR)
        
        # P^-1 matrix and calculation
        inv_tex = r"P^{-1} = \begin{pmatrix} 0.8 & 0.4 \\ -0.4 & 0.8 \end{pmatrix}"
        inverse_matrix = MathTex(inv_tex, color=INVERSE_COLOR, font_size=32)
        # Fix for Issue 40: Horizontally cramped when restricted to a single grid cell (A2).
        self.place_in_area(inverse_matrix, 'A1', 'A3', scale_factor=0.8)
        
        calc_tex = r"P^{-1} \begin{pmatrix} 4 \\ 2 \end{pmatrix} = \begin{pmatrix} 4 \\ 0 \end{pmatrix}"
        calculation = MathTex(calc_tex, color=INVERSE_COLOR, font_size=32)
        # Fix for Issue 41: Matrix 'calculation' is wide and appears cluttered when forced into A5.
        self.place_in_area(calculation, 'A4', 'A6', scale_factor=0.8)
        
        self.play(Write(inverse_matrix))
        self.wait(1)
        self.play(Write(calculation))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Apply P^-1 to (4, 2) and show point on Bob's blue grid.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HIGHLIGHT_COLOR)
        
        bob_grid = NumberPlane(
            x_range=[-1, 5, 1],
            y_range=[-1, 4, 1],
            background_line_style={"stroke_color": BOB_COLOR, "stroke_opacity": 0.3},
            axis_config={"stroke_color": BOB_COLOR}
        )
        bob_grid.apply_matrix(matrix_P)
        # Fix for Issue 39: bob_grid extends beyond the intended right-side area.
        self.place_in_area(bob_grid, "B1", "F6", scale_factor=0.5)
        
        dot_bob_label = MathTex("(4, 0)_{Bob}", font_size=24, color=BOB_COLOR)
        dot_bob_label.next_to(dot, DR, buff=0.1)
        
        self.play(
            ReplacementTransform(alice_grid, bob_grid),
            ReplacementTransform(dot_label, dot_bob_label),
            dot.animate.set_color(BOB_COLOR),
            FadeOut(inverse_matrix),
            FadeOut(calculation),
            run_time=2
        )
        self.wait(3)
