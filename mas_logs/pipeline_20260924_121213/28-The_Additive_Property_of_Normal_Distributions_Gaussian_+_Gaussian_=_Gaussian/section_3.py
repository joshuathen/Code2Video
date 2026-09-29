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
        lecture_lines = [
            "Two narrow distributions represent specific uncertainties.",
            "Adding them shifts the mean to the right.",
            "The final curve is wider due to added variance.",
            "This visual superposition confirms the rule of summation.",
            "Uncertainty grows as we combine independent processes."
        ]
        self.setup_layout("Visualizing Uncertainty Accumulation", lecture_lines)
        
        # Grid setup (using SVG as requested by asset integration requirements)
        # Assuming SVG objects would be loaded via SVGMobject. 
        # Using placeholder since assets aren't directly available to me.
        grid_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        row_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rows.svg")
        col_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/columns.svg")
        border_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/border.svg")
        
        # Fix: using self.place_in_area as requested by Issue 27 & 29
        self.place_in_area(grid_asset, 'B3', 'F5', scale_factor=0.9)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF5733"))
        self.add(grid_asset) # Display grid
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF5733"))
        # Animate moving row asset to reveal structure
        self.place_at_grid(row_asset, 'C4')
        self.play(row_asset.animate.shift(RIGHT * 1.5), row_asset.animate.set_color("#FF5733"))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#33FFBD"))
        # Change color of grid border
        self.place_at_grid(border_asset, 'C4')
        self.play(border_asset.animate.set_color("#33FFBD"))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#33FFBD"))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FF5733"))
        # Fade out grid and show the multiplication result '6'
        result = Text("6", font_size=48, color=WHITE)
        # Fix: using self.place_at_grid with E5 as requested by Issue 28
        self.place_at_grid(result, 'E5', scale_factor=1.0)
        self.play(FadeOut(grid_asset), FadeOut(row_asset), FadeOut(border_asset), FadeIn(result))
        self.wait(1)
