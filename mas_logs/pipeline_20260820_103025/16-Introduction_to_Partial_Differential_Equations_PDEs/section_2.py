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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Visualizing Multidimensional Change", [
            "Partial derivatives measure change along specific axes.",
            "Imagine a terrain; slope varies in x and y.",
            "Heat maps visualize these gradients on surfaces."
        ])
        
        # Use SVGMobject for terrain asset
        terrain_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/terrain.svg")
        terrain_asset.set_color("#FF4444")
        
        # Position terrain_group (VideoCritic fix)
        self.place_in_area(terrain_asset, "A4", "C6", scale_factor=0.45)
        
        # Labels for colors
        lbl1 = Text("Temperature Field", color="#FF4444", font_size=20)
        self.place_at_grid(lbl1, "E4", scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(terrain_asset), Write(lbl1))
        self.lecture[0].set_color("#FF4444")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Slice X representation
        slice_x = Line(start=self.grid["B3"], end=self.grid["E3"], color="#FFFF00")
        self.play(Create(slice_x))
        self.lecture[1].set_color("#FFFF00")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Slice Y representation
        slice_y = Line(start=self.grid["D2"], end=self.grid["D5"], color="#00FF00")
        self.play(Create(slice_y))
        self.lecture[2].set_color("#00FF00")
        self.wait(1)
