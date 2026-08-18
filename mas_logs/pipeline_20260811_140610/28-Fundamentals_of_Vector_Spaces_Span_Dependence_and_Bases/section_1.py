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
        lecture_lines = [
            "Vectors are directed segments from the origin.",
            "Visualize 2D vectors as simple arrows.",
            "Scaling and adding vectors creates linear combinations."
        ]
        self.setup_layout("Prerequisite Review: The Vector Concept", lecture_lines)
        
        # Assets
        grid_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        protractor_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/protractor.svg")

        # Layout Assets
        self.place_in_area(grid_asset, "A1", "F6", scale_factor=1.0)
        self.add(grid_asset)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF5733")
        v1 = Vector(RIGHT + UP, color="#FF5733")
        self.place_in_area(v1, "C3", "E5", scale_factor=0.9)
        self.play(Create(v1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#33FF57")
        # Place protractor
        self.place_at_grid(protractor_asset, "B2", scale_factor=0.5)
        self.add(protractor_asset)
        
        v_scaled = Vector(2 * (RIGHT + UP), color="#FF5733")
        self.place_at_grid(v_scaled, "D4", scale_factor=0.75)
        
        v_rotated = v_scaled.copy().rotate(PI/4)
        
        self.play(Transform(v1, v_scaled))
        self.play(Transform(v1, v_rotated))
        self.remove(protractor_asset)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#5733FF")
        v2 = Vector(RIGHT*2 + DOWN*0.5, color="#5733FF")
        self.place_at_grid(v2, "D3", scale_factor=0.8)
        
        res = Vector(v_rotated.get_end() + v2.get_end(), color="#33FF57")
        self.place_at_grid(res, "E4", scale_factor=0.85)
        
        self.play(Create(v2))
        self.play(Create(res))
        self.wait(1)
