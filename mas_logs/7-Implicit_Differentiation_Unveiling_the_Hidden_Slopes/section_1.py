from manim import *
import numpy as np
import os

# Pre-emptively create the text directory to avoid race conditions
os.makedirs(os.path.join("media", "texts"), exist_ok=True)

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
        self.setup_layout("Prerequisite Check: The Chain Rule Tool", ["Before implicit differentiation, we must master the Chain Rule.", "Think of y as a hidden function of x.", "Differentiating y squared is not just two y.", "We treat y as a box containing x.", "Multiply by dy/dx for the inner layer."])
        # 1. Setup Layout
        title = Text("Implicit Differentiation", font_size=36, color=BLUE).to_edge(UP)
        
        bullets = VGroup(
            Text("• Explicit vs Implicit Functions", font_size=24),
            Text("• The Chain Rule Application", font_size=24),
            Text("• Finding dy/dx", font_size=24),
            Text("• Tangent Lines to Curves", font_size=24)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4).scale(0.8).to_edge(LEFT, buff=1.0)

        # 2. Mathematical Visualization
        # Use a coordinate system on the right
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=5,
            y_length=5,
            axis_config={"include_tip": True}
        ).to_edge(RIGHT, buff=0.5)

        # Circle: x^2 + y^2 = 4 (Radius 2)
        circle = Circle(radius=2, color=WHITE).move_to(axes.c2p(0, 0))
        
        # Point on the circle at 45 degrees
        point_coords = axes.c2p(np.sqrt(2), np.sqrt(2))
