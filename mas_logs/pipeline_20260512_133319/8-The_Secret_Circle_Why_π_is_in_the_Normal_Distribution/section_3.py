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
            "The Dimensional Leap (The Squaring Trick)", 
            [
                'We bypass this by squaring the total area.', 
                'This expands our problem into two dimensions.', 
                'The curve becomes a symmetric three-dimensional hill.', 
                'Dots represent independent horizontal and vertical errors.', 
                'This exponent reveals the distance from the center.'
            ]
        )
        
        # Colors
        GAUSSIAN_BLUE = "#58C4DD"
        HIGHLIGHT_YELLOW = "#FFFF00"
        WHITE_COLOR = "#FFFFFF"
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE_COLOR)
        
        # I² = ∫ e^(-x²) dx * ∫ e^(-y²) dy
        formula1 = VGroup(
            Text("I² = ", font_size=36),
            Text("(", font_size=36),
            Text("∫", font_size=42),
            Text("e", slant=ITALIC, font_size=36),
            Text("-x²", font_size=20).shift(UP*0.2 + RIGHT*0.1),
            Text("dx", font_size=36),
            Text(")", font_size=36),
            Text("·", font_size=36),
            Text("(", font_size=36),
            Text("∫", font_size=42),
            Text("e", slant=ITALIC, font_size=36),
            Text("-y²", font_size=20).shift(UP*0.2 + RIGHT*0.1),
            Text("dy", font_size=36),
            Text(")", font_size=36)
        ).arrange(RIGHT, buff=0.1)
        
        # Fix: Issue 33 - position at A1-A6, scale 0.8
        self.place_in_area(formula1, "A1", "A6", scale_factor=0.8)
        self.play(Write(formula1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(WHITE_COLOR)
        
        # I² = ∫ ∫ e^{-(x² + y²)} dx dy
        # Constructing double integral
        formula2 = VGroup(
            Text("I² = ", font_size=36),
            Text("∫", font_size=42),
            Text("∫", font_size=42),
            Text("e", slant=ITALIC, font_size=36),
            Text("-(x²+y²)", font_size=20).shift(UP*0.2 + RIGHT*0.1),
            Text("dx dy", font_size=36)
        ).arrange(RIGHT, buff=0.1)
        
        # Fix: Issue 34 - position at A1-A6, scale 0.8
        self.place_in_area(formula2, "A1", "A6", scale_factor=0.8)
        
        self.play(ReplacementTransform(formula1, formula2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(GAUSSIAN_BLUE)
        
        # Asset: hill.svg
        hill_asset = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/hill.svg")
        hill_asset.set_color(GAUSSIAN_BLUE)
        self.place_in_area(hill_asset, "C1", "F6", scale_factor=2.0)
        
        self.play(DrawBorderThenFill(hill_asset))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(WHITE_COLOR)
        
        # Asset: dots.svg
        dots_asset = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/dots.svg")
        dots_asset.set_color(WHITE_COLOR)
        # Place over the hill asset
        dots_asset.move_to(hill_asset.get_center())
        dots_asset.scale_to_fit_width(hill_asset.width * 0.8)
        
        self.play(FadeIn(dots_asset))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(HIGHLIGHT_YELLOW)
        
        # Highlight exponent in formula2: index 4 is "-(x²+y²)"
        exponent_part = formula2[4]
        
        self.play(
            exponent_part.animate.set_color(HIGHLIGHT_YELLOW),
            Flash(exponent_part, color=HIGHLIGHT_YELLOW, flash_radius=0.5)
        )
        self.wait(2)
