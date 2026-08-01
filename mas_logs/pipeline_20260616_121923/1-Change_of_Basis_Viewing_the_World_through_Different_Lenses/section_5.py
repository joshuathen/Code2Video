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
        # Colors
        SKY_BLUE = "#87CEEB"
        GOLD = "#FFD700"
        RED = "#FF0000"
        
        lecture_lines = [
            'Matrix multiplication physically warps the entire grid.',
            'The standard grid morphs into the new basis.',
            'Watch how the coordinate lines bend and stretch.',
            'The vector arrow stays still while the grid shifts.',
            'Coordinates change inversely to keep the vector fixed.'
        ]
        
        self.setup_layout("The Visual Transformation: Warping the Grid", lecture_lines)
        
        # Transformation matrix: Shear M = [[1, 1], [0, 1]]
        matrix = [[1, 1], [0, 1]]
        
        # 1. Initialize Assets (Issue 39)
        # Standard Grid Asset
        grid = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/grid.svg")
        grid.set_color(SKY_BLUE)
        self.place_in_area(grid, "A1", "F6")
        
        # Target Vector Asset (Fixed in space)
        # Positioned to point from roughly D3 (origin) to C4 (1,1)
        vector_asset = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/arrow.svg")
        vector_asset.set_color(RED)
        self.place_in_area(vector_asset, "C4", "D3", scale_factor=0.3)
        vector_asset.rotate(45 * DEGREES) # Rotate to look like a (1,1) vector
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(SKY_BLUE)
        self.play(FadeIn(grid), run_time=1.5)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(GOLD)
        # "The standard grid morphs into the new basis."
        # We ensure the vector is present so the viewer sees it "stays still while the grid shifts" (Issue 39)
        self.add(vector_asset) 
        self.play(
            grid.animate.apply_matrix(matrix).set_color(GOLD),
            run_time=2.5,
            rate_func=smooth
        )
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(WHITE)
        # Emphasize the bending and stretching
        self.play(
            grid.animate.scale(1.05),
            run_time=0.4
        )
        self.play(
            grid.animate.scale(1/1.05),
            run_time=0.4
        )
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(RED)
        
        # Issue 50: Use 'B5' for coordinate label proximity
        coords_label = Text("(1, 1)", color=WHITE)
        self.place_at_grid(coords_label, "B5", scale_factor=0.8)
        
        self.play(Write(coords_label))
        self.play(Flash(vector_asset, color=RED, line_length=0.4))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(GOLD)
        
        # Issue 51: Use 'B5' for new coordinate label proximity
        new_coords_label = Text("(0, 1)", color=GOLD)
        self.place_at_grid(new_coords_label, "B5", scale_factor=0.8)
        
        self.play(
            Transform(coords_label, new_coords_label),
            Flash(grid, color=GOLD, flash_radius=1.5, num_lines=12),
            run_time=2
        )
        self.wait(3)
