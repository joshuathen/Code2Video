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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Application and Synthesis", ["Evaluate the limit of (e^x - 1) / x.", "Substitution leads to an indeterminate zero over zero.", "L'Hôpital's rule uses derivatives to find the ratio.", "Evaluating at zero gives a final limit of one.", "The epsilon-delta box confirms this value's precision."])
        # Set background color
        self.camera.background_color = BLACK

        # Replacing MathTex with Text to resolve FileNotFoundError: 'latex'
        # This allows the scene to render even when a TeX distribution is not installed.
        lhopital_expr = Text(
            "lim_{x -> 0} (e^x - 1) / x = lim_{x -> 0} e^x / 1", 
            font_size=36
        )

        # Positioning using the grid logic
        self.place_at_grid(lhopital_expr, "B3", scale_factor=1.5)

        # Animation sequence
        self.play(Write(lhopital_expr))
        self.wait(2)

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        """
        Placing mobjects on a coordinate grid.
        'B3' corresponds to a specific point in the scene.
        """
        # Mapping for grid positions
        row_map = {"A": 2, "B": 0, "C": -2}
        col_map = {"1": -4, "2": 0, "3": 4}
        
        # Determine coordinates
        row_key = grid_pos[0] if len(grid_pos) > 0 else "B"
        col_key = grid_pos[1] if len(grid_pos) > 1 else "2"
        
        y = row_map.get(row_key, 0)
        x = col_map.get(col_key, 0)
        
        # Apply transformations
        mobject.scale(scale_factor)
        mobject.move_to(np.array([x, y, 0]))
