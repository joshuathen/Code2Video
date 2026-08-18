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
            "Radius, area, and volume scale as powers.",
            "Dimensions define our spatial boundaries.",
            "Unit spheres exist in any dimension.",
            "Visualizing hypercubes shows growth.",
            "Scaling intuition guides our journey."
        ]
        self.setup_layout("Prerequisite Review: Scaling Intuition", lecture_lines)
        
        # Animation Elements using Assets
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/square.svg]
        square = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/square.svg", color=WHITE)
        self.place_in_area(square, 'D4', 'E5', scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(square))
        self.lecture[0].set_color("#FFD700")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Increase side length to 2, square scales up to area 4
        # Updated per issue 23/36
        new_square = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/square.svg", color="#FFFF00")
        self.place_in_area(new_square, 'D2', 'E3', scale_factor=0.6)
        
        self.play(Transform(square, new_square))
        self.lecture[1].set_color("#00FFFF")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Contrast volume increase in 3D: cube expands to volume 8
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/cube.svg]
        # Updated per issue 22/36
        cube = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cube.svg", color="#00FFFF")
        self.place_in_area(cube, 'F2', 'F5', scale_factor=0.5)
        
        self.play(FadeIn(cube))
        self.lecture[2].set_color("#FF69B4")
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#7FFF00")
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFA500")
        self.wait(1)
