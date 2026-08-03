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
        self.setup_layout(
            "The Inverse Relationship (The Big Reveal)",
            [
                "Differentiation and integration are actually inverse operations.",
                "They function like addition and subtraction.",
                "Integrating then differentiating returns the original function.",
                "This link is the Fundamental Theorem of Calculus.",
                "It connects the slope and the area."
            ]
        )

        # Colors
        GREY = "#A9A9A9"
        PURPLE = "#DA70D6"
        ORANGE = "#FFA500"
        WHITE_COLOR = "#FFFFFF"
        HIGHLIGHT = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        # Line 1: Differentiation and integration are actually inverse operations.
        self.play(self.lecture[0].animate.set_color(GREY))
        
        # Grey 'Function Machine' (#A9A9A9) appears in the center.
        machine = RoundedRectangle(corner_radius=0.2, width=5.2, height=3.2, color=GREY, fill_opacity=0.1)
        self.place_in_area(machine, "B1", "E6")
        
        # Issue 28: Fix machine_label positioning
        machine_label = Text("FUNCTION MACHINE", font_size=18, color=GREY)
        self.place_in_area(machine_label, "B2", "B5", scale_factor=0.8)
        
        self.play(Create(machine), Write(machine_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line 2: They function like addition and subtraction.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(WHITE_COLOR)
        )
        
        plus_minus = MathTex("+ \\leftrightarrow -", color=WHITE_COLOR)
        self.place_at_grid(plus_minus, "A2", scale_factor=1.2)
        
        # Issue 30: Fix mult_div positioning to A4
        mult_div = MathTex("\\times \\leftrightarrow \\div", color=WHITE_COLOR)
        self.place_at_grid(mult_div, "A4", scale_factor=1.2)

        self.play(Write(plus_minus), Write(mult_div))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line 3: Integrating then differentiating returns the original function.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(PURPLE)
        )
        
        # Issue 29: Fix fx_in scale factor to 0.6
        # Input 'f(x)' (#FFFFFF) enters
        fx_in = MathTex("f(x)", color=WHITE_COLOR)
        self.place_at_grid(fx_in, "C1", scale_factor=0.6)
        
        # Purple 'Integral Gear' (#DA70D6)
        int_gear = Star(n=12, inner_radius=0.4, outer_radius=0.55, color=PURPLE, fill_opacity=0.3)
        int_sym = MathTex("\\int", color=PURPLE)
        int_group = VGroup(int_gear, int_sym)
        self.place_at_grid(int_group, "C2", scale_factor=1.0)
        
        # Orange 'Derivative Gear' (#FFA500)
        der_gear = Star(n=12, inner_radius=0.4, outer_radius=0.55, color=ORANGE, fill_opacity=0.3)
        der_sym = MathTex("\\frac{d}{dx}", color=ORANGE)
        der_group = VGroup(der_gear, der_sym)
        self.place_at_grid(der_group, "C4", scale_factor=1.0)
        
        self.play(FadeIn(fx_in))
        self.play(Create(int_group), Create(der_group))
        
        # f(x) enters purple gear
        self.play(
            fx_in.animate.move_to(self.grid["C2"]),
            Rotate(int_gear, angle=2*PI),
            run_time=1.5
        )
        
        # f(x) enters orange gear
        self.play(
            fx_in.animate.move_to(self.grid["C4"]),
            Rotate(der_gear, angle=-2*PI),
            run_time=1.5
        )
        
        # Original 'f(x)' (#FFFFFF) exits the machine
        self.play(
            fx_in.animate.move_to(self.grid["C6"]),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Line 4: This link is the Fundamental Theorem of Calculus.
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(ORANGE)
        )
        
        ftc_text = Text("Fundamental Theorem of Calculus", font_size=20, color=WHITE)
        self.place_in_area(ftc_text, "F2", "F5", scale_factor=1.0)
        
        self.play(Write(ftc_text))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Line 5: It connects the slope and the area.
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(HIGHLIGHT)
        )
        
        area_label = Text("Area", font_size=18, color=PURPLE)
        slope_label = Text("Slope", font_size=18, color=ORANGE)
        self.place_at_grid(area_label, "D2", scale_factor=1.0)
        self.place_at_grid(slope_label, "D4", scale_factor=1.0)
        
        self.play(FadeIn(area_label), FadeIn(slope_label))
        
        # Symbols '∫' and 'd/dx' appear and cancel
        int_cancel = MathTex("\\int", color=PURPLE).scale(1.5)
        der_cancel = MathTex("\\frac{d}{dx}", color=ORANGE).scale(1.5)
        self.place_at_grid(int_cancel, "E2", scale_factor=0.8)
        self.place_at_grid(der_cancel, "E4", scale_factor=0.8)
        
        cancel_group = VGroup(int_cancel, der_cancel)
        self.play(FadeIn(cancel_group))
        
        cross_line = Line(
            start=int_cancel.get_left() + LEFT*0.1,
            end=der_cancel.get_right() + RIGHT*0.1,
            color=RED,
            stroke_width=6
        )
        
        self.play(Create(cross_line))
        self.wait(2)
