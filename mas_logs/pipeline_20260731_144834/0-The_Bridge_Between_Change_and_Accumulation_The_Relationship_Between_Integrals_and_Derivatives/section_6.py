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
        # Initialize the layout with title and lecture lines
        self.setup_layout("Summary: The Symmetry of Calculus", [
            "Calculus reveals a beautiful symmetry in change.",
            "Differentiation and integration are perfect inverses.",
            "Together, they bridge the gap between change and accumulation."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Highlight first line and show central function f(x)
        self.lecture[0].set_color(YELLOW)
        
        # Using Text instead of MathTex (Belief B008)
        fx = Text("f(x)", color="#FFFFFF")
        # Position f(x) at top-center of the diagram loop
        self.place_in_area(fx, "B3", "B4", scale_factor=1.2)
        
        self.play(Write(fx))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second line
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Derivative appears at the bottom
        f_prime = Text("f'(x)", color="#FFFFFF")
        self.place_in_area(f_prime, "E3", "E4", scale_factor=1.2)
        
        # Red clockwise arrow for Differentiation - Fixed position for label per Issue 33
        diff_arrow = CurvedArrow(self.grid["B5"], self.grid["E5"], angle=-TAU/4, color="#FF0000")
        diff_label = Text("Differentiate", font_size=18, color="#FF0000")
        self.place_at_grid(diff_label, "D5", scale_factor=0.8) # Issue 33 Fix
        
        # Blue counter-clockwise arrow for Integration - Fixed position for label per Issue 34
        int_arrow = CurvedArrow(self.grid["E2"], self.grid["B2"], angle=-TAU/4, color="#0000FF")
        int_label = Text("Integrate", font_size=18, color="#0000FF")
        self.place_at_grid(int_label, "C2", scale_factor=0.8) # Issue 34 Fix

        # Animate the logic loop
        self.play(
            Create(diff_arrow),
            Write(diff_label),
            Write(f_prime)
        )
        self.wait(0.5)
        
        self.play(
            Create(int_arrow),
            Write(int_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight final line and pulse the whole symmetry loop
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        loop_group = VGroup(fx, f_prime, diff_label, int_label)
        
        self.play(
            loop_group.animate.set_color(YELLOW),
            run_time=1
        )
        self.play(
            fx.animate.set_color(WHITE),
            f_prime.animate.set_color(WHITE),
            diff_label.animate.set_color("#FF0000"),
            int_label.animate.set_color("#0000FF"),
            run_time=1
        )
        
        self.wait(2)
