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

class Section6Scene(TeachingScene):
    def construct(self):
        # 1. Setup layout with updated lecture lines
        lecture_lines = [
            "To find new coordinates from standard ones...", 
            "...we multiply by the inverse of matrix P.", 
            "The grid morphs, revealing the vector's new name."
        ]
        self.setup_layout("Reverse Engineering: Changing to the New Basis", lecture_lines)
        
        # Colors
        DEER_COLOR = "#98FB98"
        MATRIX_COLOR = "#FFA07A"
        BASIS_COLOR = "#87CEFA"
        
        # Assets (Issue 30)
        DEER_ASSET = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/deer.svg"

        # === Animation for Lecture Line 1 ===
        # Standard Grid - Repositioned for Issue 42
        plane = NumberPlane(
            x_range=[-1, 6, 1],
            y_range=[-1, 6, 1],
            x_length=4.5,
            y_length=4.5,
            background_line_style={"stroke_opacity": 0.4},
            axis_config={"include_tip": True}
        )
        self.place_in_area(plane, 'C2', 'F6', scale_factor=0.9)
        
        # Deer point using SVG asset (Issue 30)
        deer_icon = SVGMobject(DEER_ASSET, color=DEER_COLOR)
        deer_icon.scale(0.2)
        deer_icon.move_to(plane.c2p(4, 5))
        
        # Deer label positioned per Issue 43
        deer_label = Text("Deer: (4, 5)", font_size=24, color=DEER_COLOR)
        self.place_at_grid(deer_label, 'C6', scale_factor=0.7)
        
        self.lecture[0].set_color(DEER_COLOR)
        self.play(
            Create(plane),
            FadeIn(deer_icon),
            Write(deer_label),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show P^-1 Matrix - Positioned per Issue 41
        p_inv_mat = Text(
            "P⁻¹ = [[2, -1], [-2, 2]]",
            color=MATRIX_COLOR,
            font_size=28
        )
        self.place_in_area(p_inv_mat, 'A2', 'A4', scale_factor=0.8)
        
        self.lecture[1].set_color(MATRIX_COLOR)
        self.play(Write(p_inv_mat))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Morph grid to Basis B (Issue 30)
        # Coordinate calculation: P^-1 * [4, 5]^T = [3, 2]^T
        new_deer_label = Text("Deer: (3, 2)_B", font_size=24, color=BASIS_COLOR)
        # Prepare target for Transform using same grid position
        new_deer_label.scale(0.7)
        new_deer_label.move_to(self.grid['C6'])
        
        self.lecture[2].set_color(BASIS_COLOR)
        
        # Basis transformation matrix P = [[1, 0.5], [1, 1]]
        # (Transforms standard grid lines to basis B vectors)
        matrix_P = [[1, 0.5], [1, 1]]
        
        self.play(
            plane.animate.apply_matrix(matrix_P),
            Transform(deer_label, new_deer_label),
            run_time=2
        )
        self.wait(2)
