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
        lines = [
            'First, differentiate both sides with respect to x.', 
            'Apply the Chain Rule to every y term.', 
            'Move all dy dx terms to one side.', 
            'Factor out dy dx from those terms.', 
            'Finally, solve for dy dx to find the slope.'
        ]
        self.setup_layout("The Step-by-Step Recipe", lines)
        
        # Colors
        COLOR_BLUE = "#0077FF"
        COLOR_RED = "#FF4444"
        COLOR_HIGHLIGHT = "#FFFF00"
        COLOR_ACTIVE = "#00FF00"
        
        # === Animation for Lecture Line 1 ===
        # Step: Differentiate
        self.lecture[0].set_color(COLOR_ACTIVE)
        
        # Visualizing terms before differentiation - Using Text instead of MathTex to avoid LaTeX dependency
        eq_initial = Text("x² + y² = 1", font_size=36)
        self.place_at_grid(eq_initial, "B3", scale_factor=1.2)
        self.play(Write(eq_initial))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Step: Chain Rule
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_ACTIVE)
        
        # Showing terms after differentiation: 2x + 2y (dy/dx) = 0
        blue_block_1 = VGroup(
            Square(side_length=0.8, fill_opacity=0.8, color=COLOR_BLUE),
            Text("2x", color=WHITE, font_size=24)
        )
        red_block_1 = VGroup(
            Square(side_length=1.2, fill_opacity=0.8, color=COLOR_RED),
            Text("2y dy/dx", color=WHITE, font_size=24)
        )
        equals_sign = Text("=", font_size=36)
        zero_val = Text("0", font_size=36)
        plus_sign = Text("+", font_size=30)
        
        eq_after_diff = VGroup(blue_block_1, plus_sign, red_block_1, equals_sign, zero_val).arrange(RIGHT, buff=0.3)
        self.place_in_area(eq_after_diff, "C1", "C6", scale_factor=0.9)
        
        self.play(FadeOut(eq_initial))
        self.play(FadeIn(eq_after_diff))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Step: Move dy/dx terms to one side
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_ACTIVE)
        
        # Sorting blocks: 2y(dy/dx) = -2x
        blue_block_moved = VGroup(
            Square(side_length=0.8, fill_opacity=0.8, color=COLOR_BLUE),
            Text("-2x", color=WHITE, font_size=24)
        )
        
        target_red_pos = self.grid["D2"]
        target_eq_pos = self.grid["D3"]
        target_blue_pos = self.grid["D4"]
        
        self.play(
            red_block_1.animate.move_to(target_red_pos),
            equals_sign.animate.move_to(target_eq_pos),
            ReplacementTransform(blue_block_1, blue_block_moved),
            blue_block_moved.animate.move_to(target_blue_pos),
            FadeOut(plus_sign),
            FadeOut(zero_val)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Step: Factor out dy/dx
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(COLOR_ACTIVE)
        
        # Factoring visualization: (2y) * dy/dx = -2x
        red_factor = Text("(2y)", color=WHITE, font_size=36)
        dydx_isolated = Text("dy/dx", color=COLOR_HIGHLIGHT, font_size=36)
        factored_red_group = VGroup(red_factor, dydx_isolated).arrange(RIGHT, buff=0.1)
        self.place_at_grid(factored_red_group, "D2", scale_factor=1.0)
        
        self.play(
            ReplacementTransform(red_block_1, factored_red_group)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Step: Solve for dy/dx
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(COLOR_ACTIVE)
        
        # Final isolation: dy/dx = -2x / 2y
        # Using a VGroup of Text to maintain array-like indexing for coloring and highlighting
        final_eq = VGroup(
            Text("dy/dx", font_size=40, color=COLOR_HIGHLIGHT),
            Text("=", font_size=40),
            Text("-2x / 2y", font_size=40)
        ).arrange(RIGHT, buff=0.3)
        
        self.place_in