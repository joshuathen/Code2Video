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
    def construct(self):
        # Setup the layout with section title and lecture lines
        self.setup_layout(
            "Operation 3: Solving for the Exponent (Logarithms)",
            [
                "Finding the Exponent replaces the \"log\" notation.",
                "A rising balloon seeks the high top vertex.",
                "With Base 5 and Result 25, Exponent is 2."
            ]
        )

        # Pre-define grid positions for the Triangle of Power
        # Middle of the grid horizontally is between cols 3 and 4
        top_pos = (self.grid["B3"] + self.grid["B4"]) / 2
        bl_pos = self.grid["E2"]
        br_pos = self.grid["E5"]
        
        # Define the Triangle
        triangle_lines = VGroup(
            Line(bl_pos, top_pos, color=WHITE),
            Line(br_pos, top_pos, color=WHITE),
            Line(bl_pos, br_pos, color=WHITE)
        )

        # Define elements
        # Resolving Issue 35: scale_factor changed to 0.8
        base_5 = MathTex("5", color=WHITE)
        self.place_at_grid(base_5, "E2", scale_factor=0.8)
        
        # Resolving Issue 36: scale_factor changed to 0.8
        result_25 = MathTex("25", color=WHITE)
        self.place_at_grid(result_25, "E5", scale_factor=0.8)
        
        question_mark = MathTex("?", color="#FF4500")
        question_mark.move_to(top_pos).scale(1.2)
        
        exponent_2 = MathTex("2", color="#1E90FF")
        exponent_2.move_to(top_pos).scale(1.2)

        # Define the Balloon Asset (SVG version)
        # Resolving Issue 25: Use SVGMobject for the balloon asset
        balloon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/balloon.svg")
        balloon.set_color("#1E90FF")
        balloon.scale(0.4)
        
        # Initial position for balloon (at the bottom of the triangle)
        balloon.move_to((bl_pos + br_pos) / 2)

        # === Animation for Lecture Line 1 ===
        # Show the triangle with 5 at bottom-left and 25 at bottom-right; top is '?' (#FF4500).
        self.lecture[0].set_color("#FF4500")
        self.play(
            Create(triangle_lines),
            Write(base_5),
            Write(result_25),
            Write(question_mark),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # A blue balloon icon (#1E90FF) floats from the bottom up to the top vertex.
        self.lecture[1].set_color("#1E90FF")
        self.play(
            FadeIn(balloon, shift=UP * 0.5),
            run_time=1
        )
        self.play(
            balloon.animate.move_to(top_pos + UP * 0.3), # Floats just above the top vertex
            run_time=2,
            rate_func=slow_into
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The '?' at the top transforms into the number 2 in dodger blue (#1E90FF).
        self.lecture[2].set_color("#1E90FF")
        self.play(
            Transform(question_mark, exponent_2),
            balloon.animate.shift(UP * 2).set_opacity(0), # Balloon floats away
            run_time=1.5
        )
        self.wait(2)
