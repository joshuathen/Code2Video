import os
import numpy as np
from pathlib import Path
from manim import *

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
        lecture_lines = [
            'We define the quantum state with this equation.',
            "Alpha and beta tell us the state's composition.",
            'Think of mixing blue and red paint together.',
            'Purple paint represents the superposition state psi.',
            'These coefficients determine the strength of each component.'
        ]
        self.setup_layout("Defining the Superposition State", lecture_lines)
        
        # Colors
        COLOR_0 = "#00FFFF"  # Blue/Cyan
        COLOR_1 = "#FF00FF"  # Red/Magenta
        COLOR_PSI = "#A020F0" # Purple

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Formula: |ψ⟩ = α|0⟩ + β|1⟩ using Text/Unicode symbols
        f_psi = Text("|ψ⟩")
        f_eq = Text(" = ")
        f_alpha = Text("α", color=COLOR_0)
        f_ket0 = Text("|0⟩")
        f_plus = Text(" + ")
        f_beta = Text("β", color=COLOR_1)
        f_ket1 = Text("|1⟩")
        
        formula = VGroup(f_psi, f_eq, f_alpha, f_ket0, f_plus, f_beta, f_ket1).arrange(RIGHT, buff=0.1)
        # Resolved Issue 41: Positioning formula to avoid crowding notes
        self.place_in_area(formula, 'A3', 'A5', scale_factor=0.8)
        
        self.play(FadeIn(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Highlight alpha and beta
        self.play(
            f_alpha.animate.scale(1.2),
            f_beta.animate.scale(1.2),
            run_time=0.5
        )
        self.play(
            f_alpha.animate.scale(1/1.2),
            f_beta.animate.scale(1/1.2),
            run_time=0.5
        )

        # Create Beaker using Asset (Issue 34)
        # Resolved Issue 34: Integrated [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/beaker.svg]
        beaker_svg = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/beaker.svg").set_color(WHITE)
        beaker_label = Text("|ψ⟩", font_size=24)
        beaker_group = VGroup(beaker_svg, beaker_label)
        
        # Resolved Issue 42: Move beaker lower for more pouring space
        self.place_in_area(beaker_group, 'D4', 'F5', scale_factor=1.0)
        beaker_label.next_to(beaker_svg, DOWN, buff=0.1)
        
        self.play(Create(beaker_svg), FadeIn(beaker_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Blue pour stream
        blue_stream = Rectangle(width=0.08, height=1.0, color=COLOR_0, fill_opacity=0.8, stroke_width=0)
        # Resolved Issue 43: Align blue stream to beaker
        self.place_at_grid(blue_stream, 'B4', scale_factor=0.8)
        
        # Blue liquid filling - positioned at the bottom of the beaker
        blue_liquid = Rectangle(width=0.8, height=0.01, color=COLOR_0, fill_opacity=0.8, stroke_width=0)
        blue_liquid.move_to(beaker_svg.get_bottom() + UP * 0.1)
        
        self.play(FadeIn(blue_stream, shift=DOWN))
        self.play(
            blue_liquid.animate.stretch_to_fit_height(0.4).shift(UP * 0.2),
            run_time=1.5
        )
        self.play(FadeOut(blue_stream))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Red pour stream
        red_stream = Rectangle(width=0.08, height=1.0, color=COLOR_1, fill_opacity=0.8, stroke_width=0)
        # Resolved Issue 43: Align red stream to beaker
        self.place_at_grid(red_stream, 'B5', scale_factor=0.8)
        
        # Red liquid filling
        red_liquid = Rectangle(width=0.8, height=0.01, color=COLOR_1, fill_opacity=0.8, stroke_width=0)
        red_liquid.move_to(blue_liquid.get_top() + UP * 0.01)
        
        self.play(FadeIn(red_stream, shift=DOWN))
        self.play(
            red_liquid.animate.stretch_to_fit_height(0.4).shift(UP * 0.2),
            run_time=1.5
        )
        self.play(FadeOut(red_stream))
        
        # Mix to Purple
        mixed_liquid = Rectangle(width=0.8, height=0.8, color=COLOR_PSI, fill_opacity=0.8, stroke_width=0)
        mixed_liquid.move_to(blue_liquid.get_bottom() + UP * 0.4)
        
        self.play(
            ReplacementTransform(VGroup(blue_liquid, red_liquid), mixed_liquid),
            beaker_label.animate.set_color(COLOR_PSI),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Pulse coefficients for emphasis
        self.play(
            f_alpha.animate.scale(1.4).set_color(WHITE),
            f_beta.animate.scale(1.4).set_color(WHITE),
            run_time=0.4,
            rate_func=there_and_back
        )
        self.play(
            f_alpha.animate.scale(1.2).set_color(COLOR_0),
            f_beta.animate.scale(1.2).set_color(COLOR_1),
            run_time=0.4,
            rate_func=there_and_back
        )
        
        self.wait(2)
        self.lecture[4].set_color(WHITE)
