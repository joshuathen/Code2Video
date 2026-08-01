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
        # 1. Setup Layout with the specified teaching script
        lecture_lines = [
            'This red vector exists independently in space.',
            'In the standard grid, its coordinates are three, two.',
            'Now, we switch to a different basis grid.',
            'Relative to this new grid, its coordinates change.',
            'The vector stays fixed; only the coordinates change.'
        ]
        self.setup_layout("The Core Concept: One Vector, Two Names", lecture_lines)

        # Colors
        COLOR_VECTOR = "#FF0000"
        COLOR_STD_GRID = "#87CEEB"
        COLOR_SLANTED_GRID = "#FFD700"

        # Assets - Issue 37 integration
        asset_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/the.svg"
        grid_asset = SVGMobject(asset_path).set_color(COLOR_STD_GRID)
        self.place_at_grid(grid_asset, "A6", scale_factor=0.5)
        
        slanted_asset = SVGMobject(asset_path).set_color(COLOR_SLANTED_GRID)
        self.place_at_grid(slanted_asset, "A6", scale_factor=0.5)

        # 2. Standard Grid - Resolving Issue 47
        standard_grid = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            x_length=3,
            y_length=3,
            background_line_style={"stroke_color": COLOR_STD_GRID, "stroke_opacity": 0.5},
            axis_config={"stroke_color": COLOR_STD_GRID}
        )
        self.place_in_area(standard_grid, 'B2', 'E5', scale_factor=1.0)
        
        # 3. Vector
        # The vector should stay physically fixed. We calculate its position based on standard coordinates.
        v_start = standard_grid.coords_to_point(0, 0)
        v_end = standard_grid.coords_to_point(3, 2)
        vector = Arrow(v_start, v_end, buff=0, color=COLOR_VECTOR, stroke_width=6)
        
        # 4. Coordinate Labels - Resolving Issue 48
        v_label_std = Text("[3, 2]", font_size=24, color=WHITE)
        self.place_at_grid(v_label_std, 'B5', scale_factor=0.8)
        
        v_label_slanted = Text("[1, 1]", font_size=24, color=WHITE)
        self.place_at_grid(v_label_slanted, 'B5', scale_factor=0.8)

        # 5. Slanted Grid - Resolving Issue 49
        # Transformation matrix: [[2, 1], [1, 1]]
        # This basis means 1*v1 + 1*v2 = (3,2) in standard coordinates.
        basis_grid = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            x_length=3,
            y_length=3,
            background_line_style={"stroke_color": COLOR_SLANTED_GRID, "stroke_opacity": 0.6},
            axis_config={"stroke_color": COLOR_SLANTED_GRID}
        )
        self.place_in_area(basis_grid, 'B2', 'E5', scale_factor=1.0)
        basis_grid.apply_matrix([[2, 1], [1, 1]])

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_VECTOR)
        self.play(GrowArrow(vector))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(COLOR_STD_GRID)
        self.play(
            FadeIn(standard_grid),
            FadeIn(grid_asset),
            Write(v_label_std)
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(WHITE)
        self.wait(1.5)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(COLOR_SLANTED_GRID)
        # The grid morphs and the asset updates its color to match the new basis context
        self.play(
            ReplacementTransform(standard_grid, basis_grid),
            ReplacementTransform(grid_asset, slanted_asset),
            ReplacementTransform(v_label_std, v_label_slanted),
            run_time=2.5
        )
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(WHITE)
        # Highlight that the physical vector remains fixed despite the ruler (grid) changing
        self.play(Indicate(vector, color=COLOR_VECTOR, scale_factor=1.2))
        self.wait(3)
