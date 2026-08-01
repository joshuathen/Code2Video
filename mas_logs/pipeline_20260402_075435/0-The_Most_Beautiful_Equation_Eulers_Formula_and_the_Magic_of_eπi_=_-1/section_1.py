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

class Section1Scene(TeachingScene):
    def construct(self):
        # Setup the layout for Section 1: Euler's Formula
        # Note: Changed LaTeX syntax to standard text to avoid dependency on system 'latex' executable
        self.setup_layout(
            "Euler's Formula: The Most Beautiful Equation",
            [
                "- Formula: e^iπ + 1 = 0",
                "- Combines 5 fundamental constants",
                "- e, i, π, 1, and 0",
                "- Connects calculus and trig"
            ]
        )

        # Create the central mathematical expression using Text to bypass FileNotFoundError: 'latex'
        euler_eq = Text("e^iπ + 1 = 0", font_size=60)
        self.place_at_grid(euler_eq, "B3", scale_factor=1.5)

        # Animation Sequence
        self.play(Write(euler_eq))
        self.play(euler_eq.animate.set_color(YELLOW))
        
        # Highlight components using Text instead of MathTex
        e_term = Text("e", color=BLUE).scale(1.2).move_to(self.grid["D2"])
        pi_term = Text("π", color=RED).scale(1.2).move_to(self.grid["D3"])
        i_term = Text("i", color=GREEN).scale(1.2).move_to(self.grid["D4"])
        
        self.play(
            FadeIn(e_term, shift=UP),
            FadeIn(pi_term, shift=UP),
            FadeIn(i_term, shift=UP)
        )
        
        self.wait(2)
