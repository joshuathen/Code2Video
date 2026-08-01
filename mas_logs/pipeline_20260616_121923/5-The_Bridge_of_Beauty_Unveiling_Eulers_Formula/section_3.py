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

class Section3Scene(TeachingScene):
    def construct(self):
        # Initial Setup
        title = "Prerequisite 2: The Identity of 'e'"
        lines = [
            'The constant e is the base of natural growth.',
            'Usually, it drives scaling in a straight line.',
            'What happens if growth is pushed in imaginary directions?'
        ]
        self.setup_layout(title, lines)

        # Colors
        COLOR_E = "#FFA500"  # Orange
        COLOR_TEXT = "#FFFFFF"  # White

        # === Animation for Lecture Line 1 ===
        # Highlight first line
        self.play(self.lecture[0].animate.set_color(COLOR_E))
        
        # Static Axes for reference
        real_axis = Line(self.grid["D1"], self.grid["D6"], color=GREY_C, stroke_width=2)
        imag_axis = Line(self.grid["F3"], self.grid["A3"], color=GREY_C, stroke_width=2)
        origin = self.grid["D3"]
        self.add(real_axis, imag_axis)

        # Constant 'e' label - Issue 38 Fixed: Moved to C5, scale 1.2
        e_label = Text("e", color=COLOR_E, slant=ITALIC)
        self.place_at_grid(e_label, "C5", scale_factor=1.2)
        
        # Growth vector along real axis
        growth_vec = Arrow(origin, self.grid["D4"], color=COLOR_E, buff=0)
        
        self.play(
            FadeIn(e_label),
            GrowArrow(growth_vec)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second line
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_E)
        )
        
        # Scaling growth vector (straight line growth)
        new_vec_target = Arrow(origin, self.grid["D6"], color=COLOR_E, buff=0)
        self.play(Transform(growth_vec, new_vec_target))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight third line
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_TEXT)
        )

        # Update label to e^{ix} - Issue 39 Fixed: Moved to B4, scale 1.2
        e_ix_label = Text("e^ix", color=COLOR_E, slant=ITALIC)
        self.place_at_grid(e_ix_label, "B4", scale_factor=1.2)
        
        # Vector bends upwards into imaginary plane
        # Reset vector to standard length first for clearer rotation visualization
        base_vec = Arrow(origin, self.grid["D4"], color=COLOR_E, buff=0)
        
        # Text "Pushing growth into the imaginary direction"
        # Issue 40 Fixed: Moved to E1-E6 area
        push_text = Text("Pushing growth into the imaginary direction", font_size=24, color=COLOR_TEXT)
        self.place_in_area(push_text, "E1", "E6", scale_factor=0.7)

        self.play(
            Transform(e_label, e_ix_label),
            Transform(growth_vec, base_vec)
        )
        
        # Rotation animation
        self.play(
            Rotate(growth_vec, angle=PI/2, about_point=origin),
            Write(push_text),
            run_time=2
        )
        self.wait(2)
