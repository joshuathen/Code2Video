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

class Section6Scene(Scene):
    def construct(self):
        # Initialize layout with title and bullet points
        title_text = "Euler's Formula and the Magic of e^iπ = -1"
        lecture_notes = [
            "- Relationship between e, i, and pi",
            "- Rotation in the complex plane",
            "- The identity e^{iπ} + 1 = 0",
            "- Beauty through mathematical unity"
        ]
        
        self.setup_layout(title_text, lecture_notes)

        # Create mathematical visualization using the grid system
        # Replaced MathTex with Text to avoid FileNotFoundError: 'latex'
        euler_identity = Text("e^iπ = -1", font_size=42, color=YELLOW)
        
        # Grid positions: A-F (vertical), 1-6 (horizontal)
        # We place the formula in the center-right area (C4)
        euler_identity.scale(1.5).move_to(2 * RIGHT + 0.5 * UP)

        # Animation sequence
        self.play(FadeIn(self.title))
        self.play(Write(self.lecture), run_time=2)
        self.play(Write(euler_identity))
        self.wait(3)

    def setup_layout(self, title_text, lecture_lines):
        # BASE BACKGROUND
        self.camera.background_color = "#000000"
        
        # Title Setup
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP, buff=0.5)
        
        # Left side lecture notes initialization
        self.lecture = VGroup(*[Text(line, font_size=24) for line in lecture_lines]).arrange(DOWN, aligned_edge=LEFT).to_edge(LEFT, buff=1)
