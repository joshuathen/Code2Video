from manim import *
import numpy as np
import os

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

class Section2Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Differentiate the outer function, leaving the inner part alone.",
            "Then, multiply by the derivative of the inner function.",
            "Treat y as f(x) and use dy/dx for f'(x)."
        ]
        self.setup_layout("Prerequisite: The Chain Rule Reminder", lecture_lines)
        
        # Colors
        COL_WHITE = "#FFFFFF"
        COL_CYAN = "#00FFFF"
        COL_YELLOW = "#FFFF00"
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COL_WHITE))
        
        # Using VGroup of Text to simulate MathTex parts for indexing
        formula_1 = VGroup(
            Text("[", color=COL_WHITE),
            Text("f(x)", color=COL_WHITE),
            Text("]³", color=COL_WHITE)
        ).arrange(RIGHT, buff=0.05)
        # Fix Issue 34: Move from B1-B4 to B2-B5
        self.place_in_area(formula_1, "B2", "B5", scale_factor=1.0)
        
        # Label/Notation
        prefix = Text("d/dx", color=COL_WHITE, font_size=24)
        # Fix Issue 35: Move from B1 to B2
        self.place_at_grid(prefix, "B2", scale_factor=1.0)
        formula_1.next_to(prefix, RIGHT, buff=0.2)
        
        self.play(Write(prefix), Write(formula_1))
        self.wait(1)
        
        # Show outer derivative 3[f(x)]²
        outer_deriv = VGroup(
            Text("=", color=COL_CYAN),
            Text("3", color=COL_CYAN),
            Text("[", color=COL_CYAN),
            Text("f(x)", color=COL_CYAN),
            Text("]", color=COL_CYAN),
            Text("²", color=COL_CYAN)
        ).arrange(RIGHT, buff=0.05)
        self.place_in_area(outer_deriv, "C2", "C5", scale_factor=1.0)
        self.play(FadeIn(outer_deriv))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(COL_CYAN))
        
        # Highlight inner function f(x) (index 3 in our VGroup)
        inner_rect = SurroundingRectangle(outer_deriv[3], color=COL_CYAN, buff=0.1)
        self.play(Create(inner_rect))
        
        # show f'(x) being multiplied
        chain_part = VGroup(
            Text("·", color=COL_CYAN),
            Text("f'(x)", color=COL_CYAN)
        ).arrange(RIGHT, buff=0.1)
        chain_part.next_to(outer_deriv, RIGHT, buff=0.2)
        self.play(Write(chain_part))
        self.wait(1)
        self.play(FadeOut(inner_rect))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(COL_YELLOW))
        
        # Replace f(x) with 'y' and f'(x) with 'dy/dx'
        # Final term: 3y² · (dy/dx)
        final_formula = VGroup(
            Text("3", color=COL_YELLOW),
            Text("y", color=COL_YELLOW),
            Text("²", color=COL_YELLOW),
            Text("·", color=COL_YELLOW),
            Text("dy/dx", color=COL_YELLOW)
        ).arrange(RIGHT, buff=0.08)
        # Fix Issue 36: Move from E2-E5 to D2-D5
        self.place_in_area(final_formula, "D2", "D5", scale_factor=1.2)
        
        self.play(TransformFromCopy(VGroup(outer_deriv, chain_part), final_formula))
        self.wait(2)
