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
            'Follow these four steps to solve implicit equations.', 
            'Sort dy dx terms to the left side.', 
            'Factor out dy dx and solve for the slope.'
        ]
        self.setup_layout("The Step-by-Step Recipe", lines)
        
        # Colors
        COLOR_BLUE = "#0077FF"
        COLOR_RED = "#FF4444"
        COLOR_HIGHLIGHT = "#FFFF00"
        COLOR_ACTIVE = "#00FF00"
        
        # === Animation for Lecture Line 1 ===
        # Step: Follow 4 steps
        self.lecture[0].set_color(COLOR_ACTIVE)
        
        # Asset: recipe icon and 4-step list
        recipe_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/recipe.svg").scale(0.6)
        steps_list = VGroup(
            Text("1. Diff", font_size=24, color=WHITE),
            Text("2. Chain", font_size=24, color=WHITE),
            Text("3. Collect", font_size=24, color=WHITE),
            Text("4. Factor", font_size=24, color=WHITE)
        ).arrange(DOWN, aligned_edge=LEFT)
        
        header_group = VGroup(recipe_icon, steps_list).arrange(RIGHT, buff=0.5)
        self.place_in_area(header_group, "A1", "A6", scale_factor=0.8)
        
        # Initial Equation
        eq_initial = Text("x² + y² = 1", font_size=36)
        self.place_in_area(eq_initial, "B2", "B5", scale_factor=1.0)
        
        self.play(FadeIn(header_group))
        self.play(Write(eq_initial))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Step: Sort dy/dx terms to the left side
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_ACTIVE)
        
        # Symbolic blocks representing 2x + 2y(dy/dx) = 0
        blue_block = VGroup(
            Square(side_length=0.8, fill_opacity=0.8, color=COLOR_BLUE),
            Text("2x", color=WHITE, font_size=22)
        )
        red_block = VGroup(
            Square(side_length=1.1, fill_opacity=0.8, color=COLOR_RED),
            Text("2y dy/dx", color=WHITE, font_size=22)
        )
        plus = Text("+", font_size=30)
        equals = Text("=", font_size=36)
        zero = Text("0", font_size=36)
        
        eq_blocks = VGroup(blue_block, plus, red_block, equals, zero).arrange(RIGHT, buff=0.3)
        self.place_in_area(eq_blocks, "C1", "C6", scale_factor=0.9)
        
        self.play(FadeOut(eq_initial))
        self.play(FadeIn(eq_blocks))
        self.wait(1)
        
        # Sorting: 2y dy/dx = -2x
        blue_block_negative = VGroup(
            Square(side_length=0.8, fill_opacity=0.8, color=COLOR_BLUE),
            Text("-2x", color=WHITE, font_size=22)
        )
        
        # Targets for sorting
        target_red = self.grid["C2"]
        target_eq = self.grid["C3"]
        target_blue = self.grid["C4"]
        
        self.play(
            red_block.animate.move_to(target_red),
            equals.animate.move_to(target_eq),
            ReplacementTransform(blue_block, blue_block_negative),
            blue_block_negative.animate.move_to(target_blue),
            FadeOut(plus),
            FadeOut(zero)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Step: Factor out dy/dx and solve
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_ACTIVE)
        
        # Factoring visualization: (2y) * dy/dx = -2x
        red_factor = Text("(2y)", color=WHITE, font_size=32)
        dydx_isolated = Text("dy/dx", color=COLOR_HIGHLIGHT, font_size=32)
        factored_red_group = VGroup(red_factor, dydx_isolated).arrange(RIGHT, buff=0.1)
        self.place_in_area(factored_red_group, "D1", "D6", scale_factor=1.0)
        
        # Position adjustments for the rest of the equation on line D
        equals_d = equals.copy()
        blue_target_d = blue_block_negative.copy()
        eq_on_d = VGroup(factored_red_group, equals_d, blue_target_d).arrange(RIGHT, buff=0.3)
        self.place_in_area(eq_on_d, "D1", "D6", scale_factor=1.0)
        
        self.play(
            ReplacementTransform(red_block, factored_red_group),
            equals.animate.move_to(equals_d.get_center()),
            blue_block_negative.animate.move_to(blue_target_d.get_center())
        )
        self.wait(1)
        
        # Final isolation: dy/dx = -2x / 2y
        final_eq = VGroup(
            Text("dy/dx", font_size=40, color=COLOR_HIGHLIGHT),
            Text("=", font_size=40),
            Text("-2x / 2y", font_size=40)
        ).arrange(RIGHT, buff=0.3)
        self.place_in_area(final_eq, "E1", "E6", scale_factor=1.1)
        
        self.play(Write(final_eq))
        self.wait(2)
