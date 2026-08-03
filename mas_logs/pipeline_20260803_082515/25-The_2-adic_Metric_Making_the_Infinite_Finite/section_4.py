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
        title_text = "Convergence Reimagined"
        lecture_lines = [
            "Consider the series: 1 plus 2 plus 4 plus 8.",
            "In real space, this sum clearly diverges to infinity.",
            "But 2-adic terms get smaller as powers of 2 increase.",
            "The series now satisfies the main condition for convergence.",
            "Watch the 2-adic tower stabilize while the real tower grows."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        COLOR_WHITE = "#FFFFFF"
        COLOR_RED = "#FF0000"
        COLOR_GREEN = "#00FF00"

        # === Animation for Lecture Line 1 ===
        # Show '1 + 2 + 4 + 8 +...' #FFFFFF at top.
        self.lecture[0].set_color(COLOR_WHITE)
        formula = MathTex("1 + 2 + 4 + 8 + \\dots", color=COLOR_WHITE)
        # Resolved Issue 28: Reduced scale_factor to 0.9 to avoid title occlusion
        self.place_in_area(formula, "A1", "A6", scale_factor=0.9)
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Create a 'Real World' box #FF0000 on the left. Inside, stack squares vertically.
        self.lecture[1].set_color(COLOR_RED)
        
        real_box = Rectangle(width=2.5, height=4.5, color=COLOR_RED).set_stroke(opacity=0.5)
        self.place_in_area(real_box, "B1", "F3")
        
        real_label = Text("Real World", color=COLOR_RED, font_size=24)
        # Resolved Issue 27: Centered label in area B1-B3
        self.place_in_area(real_label, "B1", "B3")
        
        # Terms: 1, 2, 4, 8 (scaled relative to each other)
        real_squares = VGroup(*[
            Square(side_length=s, color=COLOR_RED, fill_opacity=0.4, stroke_width=2) 
            for s in [0.2, 0.4, 0.7, 1.1]
        ]).arrange(UP, buff=0.1)
        self.place_in_area(real_squares, "C1", "F3", scale_factor=1.0)
        
        self.play(Create(real_box), FadeIn(real_label))
        self.play(LaggedStart(*[FadeIn(s) for s in real_squares], lag_ratio=0.4))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Create a '2-adic World' box #00FF00 on the right. Inside, stack squares vertically.
        self.lecture[2].set_color(COLOR_GREEN)
        
        adic_box = Rectangle(width=2.5, height=4.5, color=COLOR_GREEN).set_stroke(opacity=0.5)
        self.place_in_area(adic_box, "B4", "F6")
        
        adic_label = Text("2-adic World", color=COLOR_GREEN, font_size=24)
        # Resolved Issue 27: Centered label in area B4-B6
        self.place_in_area(adic_label, "B4", "B6")
        
        # Terms: 1, 1/2, 1/4, 1/8
        adic_squares = VGroup(*[
            Square(side_length=s, color=COLOR_GREEN, fill_opacity=0.4, stroke_width=2) 
            for s in [1.2, 0.6, 0.3, 0.15]
        ]).arrange(UP, buff=0.1)
        self.place_in_area(adic_squares, "C4", "F6", scale_factor=1.0)
        
        self.play(Create(adic_box), FadeIn(adic_label))
        self.play(LaggedStart(*[FadeIn(s) for s in adic_squares], lag_ratio=0.4))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Highlight convergence property in 2-adic space.
        self.lecture[3].set_color(COLOR_GREEN)
        
        indicator = Text("Terms approach 0", color=COLOR_GREEN, font_size=20)
        # Resolved Issue 26: Moved indicator to B6 to avoid overlap with squares
        self.place_at_grid(indicator, "B6", scale_factor=0.5)
        
        self.play(FadeIn(indicator))
        self.play(Indicate(adic_squares, color=COLOR_GREEN))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Watch the 2-adic tower stabilize while the real tower grows.
        self.lecture[4].set_color(COLOR_WHITE)
        
        # New towers including more terms
        real_squares_new = VGroup(*[
            Square(side_length=s, color=COLOR_RED, fill_opacity=0.4, stroke_width=2) 
            for s in [0.2, 0.4, 0.7, 1.1, 1.6]
        ]).arrange(UP, buff=0.1)
        self.place_in_area(real_squares_new, "C1", "F3", scale_factor=1.0)
        
        adic_squares_new = VGroup(*[
            Square(side_length=s, color=COLOR_GREEN, fill_opacity=0.4, stroke_width=2) 
            for s in [1.2, 0.6, 0.3, 0.15, 0.075]
        ]).arrange(UP, buff=0.1)
        self.place_in_area(adic_squares_new, "C4", "F6", scale_factor=1.0)
        
        # Transform current towers to updated ones to show growth vs stabilization
        self.play(
            ReplacementTransform(real_squares, real_squares_new),
            ReplacementTransform(adic_squares, adic_squares_new),
            run_time=3
        )
        self.wait(3)
