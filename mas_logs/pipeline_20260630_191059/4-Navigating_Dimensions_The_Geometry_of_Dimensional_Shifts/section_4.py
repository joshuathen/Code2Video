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
        # Color definitions
        pink_color = "#FF69B4"
        
        # Section Content
        title_text = "The Projection Trick: Solving the Shape-Hole Puzzle"
        lecture_lines = [
            "One object can cast many different two-dimensional shadows.",
            "A cylinder looks like a circle from the top.",
            "Rotated sideways, that same cylinder appears as a square."
        ]
        
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Lecture: "One object can cast many different two-dimensional shadows."
        # Highlight the current lecture line in pink to match the visual
        self.play(self.lecture[0].animate.set_color(pink_color))
        
        # Load the 3D Cylinder asset
        # [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/cylinder.svg]
        cylinder_asset = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/cylinder.svg")
        cylinder_asset.set_color(pink_color)
        
        # Position in a central area of the right grid (Issue 35: Shifted to C3-D5)
        self.place_in_area(cylinder_asset, "C3", "D5", scale_factor=1.5)
        
        self.play(DrawBorderThenFill(cylinder_asset))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Lecture: "A cylinder looks like a circle from the top."
        # Transition lecture line colors: current pink, previous white
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(pink_color)
        )
        
        # Top View Projection: Circle shadow (Issue 35: Positioned at B4)
        circle_shadow = Circle(radius=0.5, color=pink_color, fill_opacity=0.6)
        self.place_at_grid(circle_shadow, "B4", scale_factor=1.0)
        
        # Fade in the projection to illustrate the "view"
        self.play(FadeIn(circle_shadow))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Lecture: "Rotated sideways, that same cylinder appears as a square."
        # Transition lecture line colors
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(pink_color)
        )
        
        # Side View Projection: Square shadow (Issue 35: Positioned at E4)
        # Using a Square to represent the side-profile (rectangle with h=2r)
        square_shadow = Square(side_length=1.0, color=pink_color, fill_opacity=0.6)
        self.place_at_grid(square_shadow, "E4", scale_factor=1.0)
        
        # Fade in the side projection
        self.play(FadeIn(square_shadow))
        self.wait(2)
