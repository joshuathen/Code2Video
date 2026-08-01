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
        # Data from storyboard and outline
        title_text = "The Inverse: Returning to the Alternate Basis"
        lecture_lines = [
            "- Let's go backwards using Alice's point at one-two.",
            "- We use the inverse of the transition matrix P.",
            "- Multiplying by P-inverse transforms Alice's coordinates back.",
            "- Alice's standard grid squishes to match Bob's perspective.",
            "- The point arrives at its coordinate one-one for Bob."
        ]

        self.setup_layout(title_text, lecture_lines)

        # Matrix P for transformation (from Section 3/4 context)
        # b1 = [2, 1], b2 = [-1, 1]
        matrix_p = [[2, -1], [1, 1]]

        # === Animation for Lecture Line 1 ===
        # Show the standard white grid (#FFFFFF) with a point at (1, 2).
        self.lecture[0].set_color(WHITE)
        alice_grid = NumberPlane(
            x_range=[-4, 4],
            y_range=[-4, 4],
            background_line_style={"stroke_color": WHITE, "stroke_opacity": 0.3},
            axis_config={"stroke_color": WHITE}
        )
        # Ensure it fits in the designated area to avoid overlapping lecture text
        # Addresses Issue 40 and 41 by using a defined right-side area
        self.place_in_area(alice_grid, "B3", "F6", scale_factor=0.6)
        
        # Point at (1, 2) in Alice's grid
        point_alice_pos = alice_grid.c2p(1, 2)
        dot = Dot(point_alice_pos, color=RED)
        dot_label = Text("(1, 2)", font_size=18, color=RED).next_to(dot, UR, buff=0.1)

        self.play(Create(alice_grid), FadeIn(dot), Write(dot_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Write the formula "v_alt = P_inv * v_std" in white (#FFFFFF).
        self.lecture[1].set_color(WHITE)
        
        # Creating a formula using Text components for robustness (avoiding LaTeX issues)
        # Formula: v_alt = P⁻¹ v_std
        formula_parts = VGroup(
            Text("v", slant=ITALIC, font_size=24), 
            Text("_alt = P", slant=ITALIC, font_size=24), 
            Text("-1", font_size=14).shift(UP*0.1), 
            Text(" * v", slant=ITALIC, font_size=24), 
            Text("_std", slant=ITALIC, font_size=24)
        ).arrange(RIGHT, buff=0.05).set_color(WHITE)
        
        # Position formula at top of animation area (A4)
        self.place_at_grid(formula_parts, "A4", scale_factor=1.0)
        
        self.play(Write(formula_parts))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight the "v_std" part of the formula in white (#FFFFFF).
        self.lecture[2].set_color(WHITE)
        
        # Pulsing the color of v_std part
        v_std_part = formula_parts[-2:]
        self.play(v_std_part.animate.set_color(YELLOW).scale(1.2))
        self.play(v_std_part.animate.set_color(WHITE).scale(1/1.2))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Transform the white grid into the yellow tilted grid (#FFFF00).
        self.lecture[3].set_color(WHITE)
        
        # Bob's grid is Alice's grid transformed by P
        bob_grid = alice_grid.copy().apply_matrix(matrix_p).set_color(YELLOW)
        bob_grid.set_style(stroke_opacity=0.5)

        self.play(
            Transform(alice_grid, bob_grid),
            dot_label.animate.set_opacity(0),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # The point arrives at its coordinate one-one for Bob.
        self.lecture[4].set_color(WHITE)
        
        # New label for Bob's coordinate system
        bob_dot_label = Text("(1, 1)", font_size=18, color=YELLOW).next_to(dot, UR, buff=0.1)
        
        self.play(Write(bob_dot_label))
        self.wait(2)
