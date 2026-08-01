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
        # Data from storyboard
        title_text = "Transformation: Morphing the Grid"
        lecture_lines = [
            "Linear transformations warp the entire grid of space.",
            "The origin must always stay fixed in place.",
            "All grid lines must remain parallel and evenly spaced."
        ]
        
        # Initialize the layout
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        GRID_COLOR = "#FFFFFF"
        ORIGIN_COLOR = "#FFFF00"
        HIGHLIGHT_COLOR = "#FFFF00"
        
        # Load Assets
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg]
        grid_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        grid_asset.set_color(GRID_COLOR)
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/origin.svg]
        origin_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/origin.svg")
        origin_asset.set_color(ORIGIN_COLOR)
        
        # Positioning (Issue 38: B1 to F6, scale 0.7)
        self.place_in_area(grid_asset, 'B1', 'F6', scale_factor=0.7)
        local_origin = grid_asset.get_center()
        origin_asset.scale(0.3).move_to(local_origin)
        
        # Transformation matrix (Shear)
        matrix = [[1.2, 0.5], [0, 1]]
        
        # === Animation for Lecture Line 1 ===
        # "Linear transformations warp the entire grid of space."
        self.play(self.lecture[0].animate.set_color(HIGHLIGHT_COLOR))
        self.play(Create(grid_asset), run_time=1.5)
        self.play(
            grid_asset.animate.apply_matrix(matrix, about_point=local_origin),
            run_time=2.5,
            rate_func=slow_into
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "The origin must always stay fixed in place."
        self.play(self.lecture[1].animate.set_color(HIGHLIGHT_COLOR))
        self.play(FadeIn(origin_asset, scale=0.5))
        self.play(Flash(origin_asset, color=ORIGIN_COLOR, line_length=0.3))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "All grid lines must remain parallel and evenly spaced."
        self.play(self.lecture[2].animate.set_color(HIGHLIGHT_COLOR))
        # Pulse the grid lines to show they are parallel and equidistant
        self.play(
            Indicate(grid_asset, color=GRID_COLOR, scale_factor=1.05),
            run_time=2
        )
        self.wait(2)
