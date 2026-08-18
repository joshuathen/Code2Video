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
        # Setup the layout with the lecture lines and title
        lecture_lines = [
            "Matrix P translates coordinates between different basis systems.",
            "It converts Basis B coordinates into the standard system.",
            "Columns of P are Basis B vectors in standard form.",
            "Multiplying Bob's coordinates by P yields Alice's coordinates.",
            "This bridge connects two different ways of seeing space."
        ]
        self.setup_layout("The Bridge: The Change of Basis Matrix", lecture_lines)

        # Colors
        color_col1 = "#FFA500" # Orange
        color_col2 = "#800080" # Purple

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        matrix_p_initial = MathTex(
            "P", "=", r"\begin{bmatrix} 1 & -1 \\ 1 & 1 \end{bmatrix}",
            font_size=42, color=WHITE
        )
        # Resolved Issue 32: scale_factor=0.8
        self.place_at_grid(matrix_p_initial, "B2", scale_factor=0.8)
        self.play(Write(matrix_p_initial))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        bridge_label = MathTex(r"P: [\vec{v}]_B \mapsto [\vec{v}]_{\text{Standard}}", font_size=32, color=BLUE)
        self.place_at_grid(bridge_label, "A2", scale_factor=0.8)
        self.play(FadeIn(bridge_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Matrix P elements with coloring
        matrix_elements = [["1", "-1"], ["1", "1"]]
        matrix_p_colored_m = Matrix(
            matrix_elements,
            element_to_mobject_config={"font_size": 42},
            left_bracket="[",
            right_bracket="]"
        )
        # Apply colors to columns
        matrix_p_colored_m.get_columns()[0].set_color(color_col1)
        matrix_p_colored_m.get_columns()[1].set_color(color_col2)
        
        p_label = MathTex("P =", font_size=42).next_to(matrix_p_colored_m, LEFT)
        matrix_p_colored = VGroup(p_label, matrix_p_colored_m)
        
        # Resolved Issue 31: scale_factor=0.8
        self.place_at_grid(matrix_p_colored, "B2", scale_factor=0.8)
        
        self.play(Transform(matrix_p_initial, matrix_p_colored))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(YELLOW)
        )
        
        # Calculation: P * [1, 1]_B = [0, 2]_Standard
        calc = MathTex(
            r"P \cdot \begin{bmatrix} 1 \\ 1 \end{bmatrix}_B",
            "=",
            r"\begin{bmatrix} 0 \\ 2 \end{bmatrix}_S",
            font_size=36
        )
        # Resolved Issue 30: grid position C2 and scale_factor=0.8
        self.place_at_grid(calc, "C2", scale_factor=0.8)
        self.play(Write(calc))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(YELLOW)
        )
        
        # Center of visual area (roughly D5)
        viz_center = self.grid["D5"]
        
        # Alice's Standard Grid
        std_grid = NumberPlane(
            x_range=[-3, 3, 1], y_range=[-3, 3, 1],
            background_line_style={"stroke_opacity": 0.2},
            axis_config={"stroke_opacity": 0.4}
        ).scale(0.5).move_to(viz_center)
        
        # Bob's Basis B Grid (Transformed by P)
        bob_grid = NumberPlane(
            x_range=[-3, 3, 1], y_range=[-3, 3, 1],
            background_line_style={"stroke_color": BLUE, "stroke_opacity": 0.3},
            axis_config={"stroke_color": BLUE, "stroke_opacity": 0.5}
        ).scale(0.5).apply_matrix([[1, -1], [1, 1]]).move_to(viz_center)
        
        self.play(Create(std_grid))
        self.play(Create(bob_grid))
        
        # Plot the point: [1, 1] in Basis B = [0, 2] in Standard
        point_coords = viz_center + np.array([0, 2, 0]) * 0.5
        point = Dot(point_coords, color=RED, radius=0.1)
        point_lbl = MathTex(r"(1,1)_B = (0,2)_S", color=RED, font_size=20).next_to(point, UR, buff=0.1)
        
        self.play(FadeIn(point), FadeIn(point_lbl))
        
        # Pulse animation to show position invariant property
        self.play(point.animate.scale(2), run_time=0.4, rate_func=there_and_back)
        self.play(point.animate.scale(2), run_time=0.4, rate_func=there_and_back)
        
        self.wait(2)
        self.play(self.lecture[4].animate.set_color(WHITE))
