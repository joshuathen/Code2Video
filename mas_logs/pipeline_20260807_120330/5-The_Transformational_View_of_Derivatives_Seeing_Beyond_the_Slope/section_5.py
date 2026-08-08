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
        # Title and Lecture Lines
        title = "Density and Compression"
        lecture_lines = [
            "Derivatives also reveal how point density changes during mapping.",
            "High derivatives spread points out, creating lower density.",
            "Low derivatives crowd points together, increasing the local density.",
            "At critical points, the derivative is zero, collapsing space.",
            "This 'crushing' effect happens momentarily before space expands again."
        ]
        self.setup_layout(title, lecture_lines)

        # Colors
        C_INTRO = "#88C0D0" # Frost Blue
        C_SPREAD = "#A3BE8C" # Green
        C_CROWD = "#EBCB8B" # Yellow-ish
        C_CRUSH = "#FFFF00" # Pure Yellow
        C_EXPAND = "#D08770" # Orange

        # 1. Setup Static Visuals (Number Lines and Labels)
        # Using label_constructor=Text for stability and consistency with previous section code
        input_line = NumberLine(
            x_range=[-1.2, 1.2, 0.5], 
            length=4, 
            include_numbers=True, 
            font_size=18, 
            color=GRAY_C,
            label_constructor=Text
        )
        output_line = NumberLine(
            x_range=[-1.2, 1.2, 0.5], 
            length=4, 
            include_numbers=True, 
            font_size=18, 
            color=GRAY_C,
            label_constructor=Text
        )
        
        # Positions
        # Input line spans Row B, Output line spans Row E
        self.place_in_area(input_line, "B2", "B5")
        self.place_in_area(output_line, "E2", "E5")
        
        # Labels - Addressing Issues 38 and 39
        input_label = Text("Input x", font_size=20, color=WHITE)
        output_label = Text("Output f(x) = x³", font_size=20, color=WHITE)
        self.place_in_area(input_label, "A3", "A4") # Fix for Issue 38
        self.place_in_area(output_label, "D3", "D4") # Fix for Issue 39
        
        self.add(input_line, output_line, input_label, output_label)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(C_INTRO))
        
        x_vals = np.linspace(-1, 1, 10)
        input_dots = VGroup(*[
            Dot(input_line.n2p(x), radius=0.08, color=C_INTRO) 
            for x in x_vals
        ])
        
        self.play(Create(input_dots))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(C_SPREAD))
        
        output_dots = VGroup(*[
            Dot(output_line.n2p(x**3), radius=0.08, color=C_SPREAD) 
            for x in x_vals
        ])
        
        mapping_dots = input_dots.copy()
        # Transform represents the mapping effect
        self.play(Transform(mapping_dots, output_dots), run_time=2)
        self.add(output_dots)
        self.remove(mapping_dots)
        
        # Contrast with sparse distribution far from origin
        outer_rects = VGroup(
            SurroundingRectangle(output_dots[0:2], color=C_SPREAD, buff=0.1),
            SurroundingRectangle(output_dots[8:10], color=C_SPREAD, buff=0.1)
        )
        self.play(Create(outer_rects))
        self.wait(1)
        self.play(FadeOut(outer_rects))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(C_CROWD))
        
        # Highlight crowding near the origin where derivative is low
        self.play(output_dots[3:7].animate.set_color(C_CROWD))
        inner_rect = SurroundingRectangle(output_dots[4:6], color=C_CROWD, buff=0.1)
        self.play(Create(inner_rect))
        self.wait(1)
        self.play(FadeOut(inner_rect))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(C_CRUSH))
        
        # Glow effect at x=0 to indicate high density/crush
        glow = VGroup(*[
            Circle(radius=r, color=C_CRUSH, fill_opacity=op, stroke_width=0)
            for r, op in zip([0.15, 0.25, 0.35], [0.6, 0.3, 0.1])
        ])
        glow.move_to(output_line.n2p(0))
        
        deriv_label = Text("f'(0) = 0", font_size=24, color=C_CRUSH)
        self.place_at_grid(deriv_label, "D6") # Fix for Issue 40
        
        self.play(FadeIn(glow, scale=0.2), Write(deriv_label))
        self.play(glow.animate.scale(1.5), run_time=1, rate_func=there_and_back)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(C_EXPAND))
        
        # Show expansion after the momentary crush at the origin
        arrow_l = Arrow(output_line.n2p(-0.2), output_line.n2p(-0.9), color=C_EXPAND, buff=0, tip_length=0.15)
        arrow_r = Arrow(output_line.n2p(0.2), output_line.n2p(0.9), color=C_EXPAND, buff=0, tip_length=0.15)
        
        self.play(GrowArrow(arrow_l), GrowArrow(arrow_r))
        self.play(glow.animate.scale(0.8), run_time=0.5)
        self.play(glow.animate.scale(1.2), run_time=0.5)
        
        self.wait(2)
