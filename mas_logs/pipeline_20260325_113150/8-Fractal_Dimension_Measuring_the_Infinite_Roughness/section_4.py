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
        # 1. Setup layout
        title_text = "The Math: Solving for the 'In-between'"
        lecture_lines = [
            "We use logarithms to find the dimension.",
            "Formula: dimension equals log N over log S.",
            "For our gasket, it is log 3 over log 2.",
            "This results in a dimension of 1.58.",
            "Fractals occupy space differently than smooth shapes."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        cyan_hex = "#00FFFF"
        white_hex = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(white_hex))
        eq1 = Text("N = S^D", color=white_hex)
        # Resolved Issue 54: scale 1.0, area A2-B5
        self.place_in_area(eq1, "A2", "B5", scale_factor=1.0)
        self.play(Write(eq1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(white_hex))
        eq2 = Text("D = log(N) / log(S)", color=white_hex)
        self.place_in_area(eq2, "A2", "B5", scale_factor=1.0)
        self.play(Transform(eq1, eq2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(white_hex))
        eq3 = Text("D = log(3) / log(2)", color=white_hex)
        self.place_in_area(eq3, "C1", "C6", scale_factor=1.2)
        self.play(Write(eq3))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(cyan_hex))
        res = Text("D ≈ 1.585", color=cyan_hex)
        # Resolved Issue 53: scale 0.9, area D2-D5
        self.place_in_area(res, "D2", "D5", scale_factor=0.9)
        self.play(Write(res))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(white_hex))
        
        # Create NumberLine - Pass label_constructor=Text to avoid LaTeX dependency
        nl = NumberLine(
            x_range=[1, 2, 0.5],
            length=4,
            include_numbers=True,
            color=white_hex,
            label_direction=DOWN,
            label_constructor=Text
        )
        
        # Marker and Label
        marker_pos = nl.n2p(1.585)
        marker = Arrow(
            start=marker_pos + UP * 0.6,
            end=marker_pos,
            color=cyan_hex,
            buff=0,
            stroke_width=5
        )
        marker_label = Text("1.585", font_size=18, color=cyan_hex)
        marker_label.next_to(marker, UP, buff=0.1)
        
        # Group and place
        nl_group = VGroup(nl, marker, marker_label)
        # Resolved Issue 55: scale 0.8, area E1-F6
        self.place_in_area(nl_group, "E1", "F6", scale_factor=0.8)
        
        self.play(Create(nl))
        self.play(FadeIn(marker), Write(marker_label))
        self.wait(2)
