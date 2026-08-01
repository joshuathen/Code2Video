from manim import *
import os
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
        # Setup basic layout with exact lecture lines from prompt
        title_str = "The Naming of the Constant"
        lecture_lines = [
            "Long Latin phrases once described this ratio.",
            "William Jones introduced the symbol pi in 1706.",
            "Leonhard Euler later made the symbol world-famous."
        ]
        self.setup_layout(title_str, lecture_lines)

        # Colors
        GRAY_COLOR = "#808080"
        CYAN_COLOR = "#00FFFF"

        # === Animation for Lecture Line 1 ===
        # Highlight lecture line
        self.play(self.lecture[0].animate.set_color(GRAY_COLOR))
        
        # Wordy Latin description
        latin_text = Text(
            "quantitas, in quam cum multiflicetur\ndiameter, provenit circumferentia",
            color=GRAY_COLOR,
            font_size=22,
            line_spacing=1.2
        )
        self.place_in_area(latin_text, "B1", "D6")
        
        self.play(Write(latin_text))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # Highlight lecture line
        self.play(self.lecture[1].animate.set_color(CYAN_COLOR))
        
        # Cyan pi symbol
        pi_symbol = Text("π", color=CYAN_COLOR, font_size=160)
        # Fix for Issue 32: Narrower area and scale factor
        self.place_in_area(pi_symbol, "B2", "D5", scale_factor=0.8)
        
        # William Jones Label
        jones_label = Text("William Jones (1706)", color=CYAN_COLOR, font_size=24)
        # Fix for Issue 33: Narrower area and scale factor
        self.place_in_area(jones_label, "E2", "E5", scale_factor=0.9)
        
        # Transform Latin text to pi symbol and show Jones label
        self.play(
            ReplacementTransform(latin_text, pi_symbol),
            FadeIn(jones_label, shift=UP)
        )
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # Highlight lecture line
        self.play(self.lecture[2].animate.set_color(CYAN_COLOR))
        
        # Leonhard Euler Label
        euler_label = Text("Leonhard Euler (1737)", color=CYAN_COLOR, font_size=24)
        # Fix for Issue 34: Narrower area and scale factor
        self.place_in_area(euler_label, "F2", "F5", scale_factor=0.9)
        
        # Show Euler label
        self.play(FadeIn(euler_label, shift=UP))
        self.wait(3)
