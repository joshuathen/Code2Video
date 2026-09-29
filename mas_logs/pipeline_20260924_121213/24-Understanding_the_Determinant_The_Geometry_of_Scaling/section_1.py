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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisite: The Concept of Linear Transformation", [
            "Matrices transform the 2D plane.",
            "Vectors i and j define a square.",
            "The square deforms into a parallelogram."
        ])

        # Assets paths
        grid_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg"
        origin_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/origin.svg"
        vector_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/vector.svg"
        parallelogram_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/parallelogram.svg"

        # Load SVG assets
        grid_svg = SVGMobject(grid_asset).set_color(WHITE)
        origin_svg = SVGMobject(origin_asset).set_color("#FF0000")
        i_vec_svg = SVGMobject(vector_asset).set_color("#00FF00")
        j_vec_svg = SVGMobject(vector_asset).set_color("#0000FF")
        para_svg = SVGMobject(parallelogram_asset).set_color(WHITE).set_fill(opacity=0.2)

        # Place grid (Issue 21 fix: Area A4-F6, scale 0.7)
        self.place_in_area(grid_svg, 'A4', 'F6', scale_factor=0.7)
        self.add(grid_svg)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFD700")
        # Origin and vectors (Issue 23 fix: i_vec to D6, j_vec to B5)
        self.place_at_grid(origin_svg, 'D3', scale_factor=0.3)
        self.place_at_grid(i_vec_svg, 'D6', scale_factor=0.25)
        self.place_at_grid(j_vec_svg, 'B5', scale_factor=0.25)
        self.play(FadeIn(origin_svg), FadeIn(i_vec_svg), FadeIn(j_vec_svg))

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFD700")
        # Square definition (Issue 22 fix: C5, scale 0.4)
        square = Square(side_length=1, color=WHITE, fill_opacity=0.2)
        self.place_at_grid(square, 'C5', scale_factor=0.4)
        self.play(Create(square))

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FFD700")
        # Transform into parallelogram
        self.play(
            ReplacementTransform(square, para_svg),
            i_vec_svg.animate.shift(RIGHT * 0.5 + UP * 0.5),
            j_vec_svg.animate.shift(LEFT * 0.5)
        )
        self.wait(1)
