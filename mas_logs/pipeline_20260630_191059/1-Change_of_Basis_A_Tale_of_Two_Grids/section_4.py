from manim import *
import numpy as np

# Use the provided TeachingScene base class without modification.
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
        # Define lecture lines with bullets as per instruction
        lecture_lines = [
            "- Bob's coordinates are transformed using this matrix formula.",
            "- We start with Bob's coordinate point at one-one.",
            "- Multiplying by P warps Bob's grid into Alice's.",
            "- The calculation results in Alice's coordinates one-two.",
            "- The point now matches Alice's standard grid perspective."
        ]
        
        self.setup_layout("The Transformation Formula", lecture_lines)

        # Assets
        bob_icon_path = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/bob.svg"

        # === Animation for Lecture Line 1 ===
        # Bob's coordinates are transformed using this matrix formula.
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Formula: v_std = P * v_alt
        # Using Text blocks to avoid LaTeX dependencies
        formula = VGroup(
            Text("v_std", font_size=32, color=WHITE),
            Text("=", font_size=32, color=WHITE),
            Text("P", font_size=32, color=WHITE),
            Text("*", font_size=32, color=WHITE),
            Text("v_alt", font_size=32, color=WHITE)
        ).arrange(RIGHT, buff=0.15)
        
        # [Issue 37] Fix: Position formula at ('A2', 'B6') with scale 1.0
        self.place_in_area(formula, "A2", "B6", scale_factor=1.0)
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # We start with Bob's coordinate point at one-one.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Bob's yellow grid (tilted by basis vectors b1=[2,1], b2=[-1,1])
        matrix_p = [[2, -1], [1, 1]]
        bob_grid = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            background_line_style={"stroke_color": YELLOW, "stroke_opacity": 0.4},
            axis_config={"stroke_color": YELLOW, "stroke_width": 2}
        )
        bob_grid.apply_matrix(matrix_p)
        
        # [Issue 38] Fix: Position bob_grid at ('C2', 'F6') with scale 0.5
        self.place_in_area(bob_grid, "C2", "F6", scale_factor=0.5)
        
        # [Issue 28] Asset Integration: Bob icon
        bob_icon = SVGMobject(bob_icon_path).scale(0.3).set_color(YELLOW)
        self.place_at_grid(bob_icon, "C2")
        
        # Point at (1,1) in Bob's basis
        point_pos = bob_grid.c2p(1, 1)
        dot = Dot(point_pos, color=RED, radius=0.08)
        
        # Label for Bob's coordinates
        label_alt = Text("(1, 1) alt", color=YELLOW, font_size=18)
        label_alt.next_to(dot, UR, buff=0.1)
        
        self.play(
            Create(bob_grid),
            FadeIn(bob_icon),
            FadeIn(dot),
            Write(label_alt)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Multiplying by P warps Bob's grid into Alice's.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Highlight v_alt part of the formula in yellow
        self.play(formula[4].animate.set_color(YELLOW))
        self.wait(0.5)
        
        # Alice's white standard grid
        alice_grid = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            background_line_style={"stroke_color": WHITE, "stroke_opacity": 0.4},
            axis_config={"stroke_color": WHITE, "stroke_width": 2}
        )
        # [Issue 39] Fix: Position alice_grid at ('C2', 'F6') with scale 0.5
        self.place_in_area(alice_grid, "C2", "F6", scale_factor=0.5)
        
        # Transform yellow grid to white grid
        self.play(
            ReplacementTransform(bob_grid, alice_grid),
            formula[4].animate.set_color(WHITE),
            FadeOut(bob_icon)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # The calculation results in Alice's coordinates one-two.
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(YELLOW)
        )
        
        # Highlight v_std part of the formula in yellow
        self.play(formula[0].animate.set_color(YELLOW))
        
        # Update point label to Alice's standard coordinates (1, 2)
        label_std = Text("(1, 2) std", color=WHITE, font_size=18)
        label_std.next_to(dot, UR, buff=0.1)
        
        self.play(
            ReplacementTransform(label_alt, label_std)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # The point now matches Alice's standard grid perspective.
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(YELLOW)
        )
        
        # Final emphasis on the resulting point
        self.play(
            Indicate(dot, color=RED, scale_factor=1.5),
            formula[0].animate.set_color(WHITE)
        )
        self.wait(2)
